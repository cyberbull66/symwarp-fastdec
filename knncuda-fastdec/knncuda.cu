// Adapted from kNN-CUDA (https://github.com/vincentfpgarcia/kNN-CUDA) by Vincent Garcia,
// Eric Debreuve and Michel Barlaud. Licensed CC BY-NC-SA 3.0 (not MIT); see knncuda-fastdec/LICENSE.
#include <stdio.h>
#include <cuda.h>
#include <float.h>
#include <limits.h>
#include "knncuda.h"
#include "dataset.h"
#include "dcreductioncuda.h"

// Block dimension for CUDA kernels. A common choice is 16 or 32 for 2D blocks. In this implementation,
// This constant is used to define the size of shared memory tiles and the block size for launching kernels.
// It's maximum value is constrained by two factors: (1) the maximum number of threads per block (1024 for many GPUs) 
// and (2) the amount of shared memory available per block (48KB or 96KB on many GPUs). 
// For the former constraint, BLOCK_DIM must satisfy BLOCK_DIM^2 ≤ 1024, which implies BLOCK_DIM ≤ 32. 
// For the latter constraint, each block uses 2 * BLOCK_DIM^2 * sizeof(float) bytes of shared memory for 
// the two tiles (shared_A and shared_B). This means 2 * BLOCK_DIM^2 * 4 bytes ≤ 48KB (or 96KB), 
// which implies BLOCK_DIM ≤ 78 (or 110). The value of 16 provides a safe margin and good occupancy.
#define BLOCK_DIM 16


/**
 * Computes the squared Euclidean distance matrix between the query points and the reference points.
 *
 * @param ref          refence points stored in the global memory
 * @param ref_width    number of reference points
 * @param ref_pitch    pitch of the reference points array in number of column
 * @param query        query points stored in the global memory
 * @param query_width  number of query points
 * @param query_pitch  pitch of the query points array in number of columns
 * @param height       dimension of points = height of texture `ref` and of the array `query`
 * @param dist         array containing the query_width x ref_width computed distances
 * @param dist_pitch   pitch of the distance array in number of columns
 */
__global__ void compute_distances(const float * ref,
                                  int           ref_width,
                                  int           ref_pitch,
                                  const float * query,
                                  int           query_width,
                                  int           query_pitch,
                                  int           height,
                                  float *       dist,
                                  int           dist_pitch) {

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ size_t begin_A;
    __shared__ size_t begin_B;
    __shared__ size_t step_A;
    __shared__ size_t step_B;
    __shared__ size_t end_A;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Initializarion of the SSD for the current thread
    float ssd = 0.f;

    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y;
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = (size_t)BLOCK_DIM * ref_pitch;
    step_B  = (size_t)BLOCK_DIM * query_pitch;
    end_A   = begin_A + (size_t)(height-1) * ref_pitch;   // size_t: (height-1)*ref_pitch exceeds INT_MAX when (D-1)*N > 2^31

    // Conditions
    int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
    int cond1 = (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array 
    int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (size_t a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/ref_pitch + ty < height) {
            shared_A[ty][tx] = (cond0)? ref[a + (size_t)ref_pitch * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? query[b + (size_t)query_pitch * ty + tx] : 0;
        }
        else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k){
                float tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1) {
        dist[ (long long)(begin_A + ty) * dist_pitch + begin_B + tx ] = ssd;
    }
}


/**
 * For each reference point (i.e. each column) finds the k-th smallest distances
 * of the distance matrix and their respective indexes and gathers them at the top
 * of the 2 arrays.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist         distance matrix
 * @param dist_pitch   pitch of the distance matrix given in number of columns
 * @param index        index matrix
 * @param index_pitch  pitch of the index matrix given in number of columns
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 * @param k            number of values to find
 * @param exclude_self whether to exclude each query point from its own neighbors (self k-NN, introduced for FastDEC)
 * @param batch_start  starting index of the current batch of points (introduced to support big datasets)
 */
__global__ void modified_insertion_sort(float * dist,
                                        int     dist_pitch,
                                        int *   index,
                                        int     index_pitch,
                                        int     width,
                                        int     height,
                                        int     k,
                                        bool    exclude_self,
                                        int     batch_start){

    // Column position
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Do nothing if we are out of bounds
    if (xIndex < width) {

        // Pointer shift
        float * p_dist  = dist  + xIndex;
        int *   p_index = index + xIndex;

        // Initialise the first index
        p_index[0] = 0;
        // Exclude the query point itself in self k-NN (row i is global
        // point batch_start + i; column xIndex is the query point). Exact
        // duplicates (zero distance, different index) remain eligible.
        if (exclude_self && batch_start == (int)xIndex)
            p_dist[0] = FLT_MAX;

        // Go through all points
        for (int i=1; i<height; ++i) {
            // Exclude the query point itself in self k-NN
            if (exclude_self && batch_start + i == (int)xIndex)
                p_dist[(long long)i * dist_pitch] = FLT_MAX;

            // Store current distance and associated index
            float curr_dist = p_dist[(long long)i*dist_pitch];
            int   curr_index  = i;

            // Skip the current value if its index is >= k and if it's higher the k-th already sorted smallest value
            if (i >= k && curr_dist >= p_dist[(k-1)*dist_pitch]) {
                continue;
            }

            // Shift values (and indexes) higher that the current distance to the right
            int j = min(i, k-1);
            while (j > 0 && p_dist[(j-1)*dist_pitch] > curr_dist) {
                p_dist[j*dist_pitch]   = p_dist[(j-1)*dist_pitch];
                p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j*dist_pitch]   = curr_dist;
            p_index[j*index_pitch] = curr_index; 
        }
    }
}

/**
 * Computes the square root of the first k lines of the distance matrix.
 *
 * @param dist   distance matrix
 * @param width  width of the distance matrix
 * @param pitch  pitch of the distance matrix given in number of columns
 * @param k      number of values to consider
 */
__global__ void compute_sqrt(float * dist, int width, int pitch, int k){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
}


__device__ __forceinline__ int knn_index_compare_value(int index_value) {
    return (index_value < 0) ? INT_MAX : index_value;
}

__device__ __forceinline__ bool knn_pair_better(float lhs_dist,
                                                int   lhs_index,
                                                float rhs_dist,
                                                int   rhs_index) {
    if (lhs_dist < rhs_dist) {
        return true;
    }
    if (lhs_dist > rhs_dist) {
        return false;
    }
    return knn_index_compare_value(lhs_index) < knn_index_compare_value(rhs_index);
}


# if (KNN_SELECT_METHOD != 0) // Heap-based top-k selection method (opt in for KNN_SELECT_METHOD 1 and 2)
__device__ __forceinline__ bool knn_pair_worse(float lhs_dist,
                                               int   lhs_index,
                                               float rhs_dist,
                                               int   rhs_index) {
    if (lhs_dist > rhs_dist) {
        return true;
    }
    if (lhs_dist < rhs_dist) {
        return false;
    }
    return knn_index_compare_value(lhs_index) > knn_index_compare_value(rhs_index);
}

__device__ __forceinline__ void knn_swap_entries(float * p_dist,
                                                 int     dist_pitch,
                                                 int *   p_index,
                                                 int     index_pitch,
                                                 int     lhs,
                                                 int     rhs) {
    float dist_tmp = p_dist[(long long)lhs * dist_pitch];
    int   index_tmp = p_index[(long long)lhs * index_pitch];

    p_dist[(long long)lhs * dist_pitch] = p_dist[(long long)rhs * dist_pitch];
    p_index[(long long)lhs * index_pitch] = p_index[(long long)rhs * index_pitch];
    p_dist[(long long)rhs * dist_pitch] = dist_tmp;
    p_index[(long long)rhs * index_pitch] = index_tmp;
}

__device__ __forceinline__ void knn_max_heap_sift_down(float * p_dist,
                                                       int     dist_pitch,
                                                       int *   p_index,
                                                       int     index_pitch,
                                                       int     root,
                                                       int     heap_size) {
    while (true) {
        int left = root * 2 + 1;
        if (left >= heap_size) {
            break;
        }

        int largest = left;
        int right = left + 1;
        if (right < heap_size &&
            knn_pair_worse(p_dist[(long long)right * dist_pitch],
                           p_index[(long long)right * index_pitch],
                           p_dist[(long long)left * dist_pitch],
                           p_index[(long long)left * index_pitch])) {
            largest = right;
        }

        if (!knn_pair_worse(p_dist[(long long)largest * dist_pitch],
                            p_index[(long long)largest * index_pitch],
                            p_dist[(long long)root * dist_pitch],
                            p_index[(long long)root * index_pitch])) {
            break;
        }

        knn_swap_entries(p_dist, dist_pitch, p_index, index_pitch, root, largest);
        root = largest;
    }
}

__device__ __forceinline__ void knn_build_max_heap(float * p_dist,
                                                   int     dist_pitch,
                                                   int *   p_index,
                                                   int     index_pitch,
                                                   int     heap_size) {
    for (int root = heap_size / 2 - 1; root >= 0; --root) {
        knn_max_heap_sift_down(p_dist, dist_pitch, p_index, index_pitch, root, heap_size);
    }
}

__global__ void modified_heap_select(float * dist,
                                     int     dist_pitch,
                                     int *   index,
                                     int     index_pitch,
                                     int     width,
                                     int     height,
                                     int     k,
                                     bool    exclude_self,
                                     int     batch_start) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (xIndex >= (unsigned int)width || k <= 0) {
        return;
    }

    float * p_dist  = dist  + xIndex;
    int *   p_index = index + xIndex;

    int initial_count = (height < k) ? height : k;
    for (int i = 0; i < initial_count; ++i) {
        float curr_dist = p_dist[(long long)i * dist_pitch];
        if (exclude_self && batch_start + i == (int)xIndex) {
            curr_dist = FLT_MAX;
        }

        p_dist[(long long)i * dist_pitch] = curr_dist;
        p_index[(long long)i * index_pitch] = i;
    }

    for (int i = initial_count; i < k; ++i) {
        p_dist[(long long)i * dist_pitch] = FLT_MAX;
        p_index[(long long)i * index_pitch] = -1;
    }

    if (k > 1) {
        knn_build_max_heap(p_dist, dist_pitch, p_index, index_pitch, k);
    }

    for (int i = initial_count; i < height; ++i) {
        float curr_dist = p_dist[(long long)i * dist_pitch];
        if (exclude_self && batch_start + i == (int)xIndex) {
            curr_dist = FLT_MAX;
        }

        if (!knn_pair_better(curr_dist,
                             i,
                             p_dist[0],
                             p_index[0])) {
            continue;
        }

        p_dist[0] = curr_dist;
        p_index[0] = i;
        if (k > 1) {
            knn_max_heap_sift_down(p_dist, dist_pitch, p_index, index_pitch, 0, k);
        }
    }

    for (int end = k - 1; end > 0; --end) {
        knn_swap_entries(p_dist, dist_pitch, p_index, index_pitch, 0, end);
        knn_max_heap_sift_down(p_dist, dist_pitch, p_index, index_pitch, 0, end);
    }
}
#endif // KNN_SELECT_METHOD != 0


#if (KNN_MERGE_METHOD == 0) // Insertion-based merge method
// Extend the KnnCuda implementation to support large datasets with batched processing - Option 0: Insertion-based merge method.

/**
 * Merges per-batch kNN results into the accumulated kNN results.
 * Both the accumulated and batch result lists are sorted in ascending order per column.
 * For each query point, inserts batch candidates into the accumulated list if they
 * improve the current k-th best distance.
 *
 * @param knn_dist          accumulated k best distances (k rows x width columns)
 * @param knn_dist_pitch    pitch of knn_dist in number of columns
 * @param knn_index         accumulated k best indices (k rows x width columns)
 * @param knn_index_pitch   pitch of knn_index in number of columns
 * @param batch_dist        batch k best distances (k rows x width columns), sorted ascending
 * @param batch_dist_pitch  pitch of batch_dist in number of columns
 * @param batch_index       batch k best indices (k rows x width columns)
 * @param batch_index_pitch pitch of batch_index in number of columns
 * @param width             number of query points
 * @param k                 number of nearest neighbors
 * @param batch_offset      offset added to batch indices to obtain global reference indices
 */
__global__ void merge_knn_results_insertion(float * knn_dist,
                                  int     knn_dist_pitch,
                                  int *   knn_index,
                                  int     knn_index_pitch,
                                  float * batch_dist,
                                  int     batch_dist_pitch,
                                  int *   batch_index,
                                  int     batch_index_pitch,
                                  int     width,
                                  int     k,
                                  int     batch_offset) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex < width) {
        for (int b = 0; b < k; ++b) {
            float bd = batch_dist[b * batch_dist_pitch + xIndex];

            // Batch results are sorted ascending; if this one doesn't fit, none after it will
            if (bd >= knn_dist[(k - 1) * knn_dist_pitch + xIndex]) {
                break;
            }

            int bi = batch_index[b * batch_index_pitch + xIndex] + batch_offset;

            // Insertion sort into the accumulated knn list
            int pos = k - 1;
            while (pos > 0 && knn_dist[(pos - 1) * knn_dist_pitch + xIndex] > bd) {
                knn_dist[pos * knn_dist_pitch + xIndex]   = knn_dist[(pos - 1) * knn_dist_pitch + xIndex];
                knn_index[pos * knn_index_pitch + xIndex] = knn_index[(pos - 1) * knn_index_pitch + xIndex];
                --pos;
            }
            knn_dist[pos * knn_dist_pitch + xIndex]   = bd;
            knn_index[pos * knn_index_pitch + xIndex] = bi;
        }
    }
}
#endif // KNN_MERGE_METHOD == 0


#if (KNN_MERGE_METHOD == 1) // Linear merge method
// Extend the KnnCuda implementation to support large datasets with batched processing - Option 1: Linear merge method.

/**
 * Merges the KNN results from the accumulated knn list and the current batch into the output arrays.
 *
 * @param knn_dist           accumulated KNN distances
 * @param knn_dist_pitch     pitch of the accumulated KNN distances
 * @param knn_index          accumulated KNN indices
 * @param knn_index_pitch    pitch of the accumulated KNN indices
 * @param batch_dist         current batch KNN distances
 * @param batch_dist_pitch   pitch of the current batch KNN distances
 * @param batch_index        current batch KNN indices
 * @param batch_index_pitch  pitch of the current batch KNN indices
 * @param out_dist           output KNN distances
 * @param out_dist_pitch     pitch of the output KNN distances
 * @param out_index          output KNN indices
 * @param out_index_pitch    pitch of the output KNN indices
 * @param width              width of the KNN arrays
 * @param k                  number of neighbors
 * @param batch_offset       offset to add to batch indices
 */
__global__ void merge_knn_results_linear(float * knn_dist,
                                         int     knn_dist_pitch,
                                         int *   knn_index,
                                         int     knn_index_pitch,
                                         float * batch_dist,
                                         int     batch_dist_pitch,
                                         int *   batch_index,
                                         int     batch_index_pitch,
                                         float * out_dist,
                                         int     out_dist_pitch,
                                         int *   out_index,
                                         int     out_index_pitch,
                                         int     width,
                                         int     k,
                                         int     batch_offset) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (xIndex < (unsigned int)width) {
        int acc_pos = 0;
        int batch_pos = 0;

        for (int out_pos = 0; out_pos < k; ++out_pos) {
            float acc_dist = (acc_pos < k)
                           ? knn_dist[(long long)acc_pos * knn_dist_pitch + xIndex]
                           : FLT_MAX;
            int acc_idx = (acc_pos < k)
                        ? knn_index[(long long)acc_pos * knn_index_pitch + xIndex]
                        : -1;

            float batch_dist_value = (batch_pos < k)
                                   ? batch_dist[(long long)batch_pos * batch_dist_pitch + xIndex]
                                   : FLT_MAX;
            int batch_raw_index = (batch_pos < k)
                                ? batch_index[(long long)batch_pos * batch_index_pitch + xIndex]
                                : -1;
            int batch_idx = (batch_raw_index < 0) ? -1 : (batch_raw_index + batch_offset);

            bool take_batch = (batch_pos < k) &&
                              ((acc_pos >= k) ||
                               knn_pair_better(batch_dist_value,
                                               batch_idx,
                                               acc_dist,
                                               acc_idx));

            if (take_batch) {
                out_dist[(long long)out_pos * out_dist_pitch + xIndex] = batch_dist_value;
                out_index[(long long)out_pos * out_index_pitch + xIndex] = batch_idx;
                ++batch_pos;
            }
            else {
                out_dist[(long long)out_pos * out_dist_pitch + xIndex] = acc_dist;
                out_index[(long long)out_pos * out_index_pitch + xIndex] = acc_idx;
                ++acc_pos;
            }
        }
    }
}
#endif // KNN_MERGE_METHOD == 1

__global__ void init_float_array(float * arr, int pitch, int width, int height, float value) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex < width && yIndex < height) {
        arr[yIndex * pitch + xIndex] = value;
    }
}

__global__ void init_int_array(int * arr, int pitch, int width, int height, int value) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex < width && yIndex < height) {
        arr[yIndex * pitch + xIndex] = value;
    }
}

/**
 * For each input query point, locates the k-NN (indexes and distances) among the reference points.
 * This implementation uses global memory to store reference and query points.
 *
 * @param ref                  reference points on the GPU (dim rows x ref_nb columns, pitched); read-only, owned by the caller
 * @param ref_pitch_in_bytes   row pitch of ref in bytes
 * @param ref_nb               number of reference points
 * @param query                query points on the GPU (dim rows x query_nb columns, pitched); ignored when self_knn is true
 * @param query_pitch_in_bytes row pitch of query in bytes; ignored when self_knn is true
 * @param query_nb             number of query points; ignored when self_knn is true
 * @param dim                  dimension of points
 * @param k                    number of neighbors to consider
 * @param cuda_device_id       ID of the CUDA device to use for computation
 * @param knn_dist             output array containing the query_nb x k distances
 * @param knn_index            output array containing the query_nb x k indexes
 * @param self_knn             true for a self k-NN: the reference points are the queries and each point is
 *                             excluded from its own neighbors (introduced for FastDEC); false for a normal k-NN
 * @return true if successful, false otherwise.
 */
static bool knn_cuda_global(const float * ref,
                     size_t        ref_pitch_in_bytes,
                     int           ref_nb,
                     const float * query,
                     size_t        query_pitch_in_bytes,
                     int           query_nb,
                     int           dim,
                     int           k,
                     int           cuda_device_id,
                     float *       knn_dist,
                     int *         knn_index,
                     bool          self_knn) {

    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int   = sizeof(int);

    // Self k-NN: the reference points are also the query points.
    if (self_knn) {
        query = ref;
        query_pitch_in_bytes = ref_pitch_in_bytes;
        query_nb = ref_nb;
    }

    if (ref == NULL || ref_pitch_in_bytes % size_of_float != 0 ||
        query == NULL || query_pitch_in_bytes % size_of_float != 0) {
        printf("ERROR: knn_cuda_global received an unusable GPU buffer (ref %s, ref pitch %zu bytes, "
               "query %s, query pitch %zu bytes); kNN was not computed. Pass device buffers whose "
               "pitches are multiples of sizeof(float).\n",
               ref == NULL ? "null" : "set", ref_pitch_in_bytes,
               query == NULL ? "null" : "set", query_pitch_in_bytes);
        return false;
    }

    cudaError_t err0, err1, err2, err3;

    if (!select_cuda_device(cuda_device_id)) {
        printf("ERROR: kNN-CUDA could not initialize CUDA device %d; "
               "kNN was not computed. Verify the --gpu index and CUDA runtime state.\n",
               cuda_device_id);
        return false;
    }

    size_t ref_pitch   = ref_pitch_in_bytes   / size_of_float;
    size_t query_pitch = query_pitch_in_bytes / size_of_float;

    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    // The batch distance matrix and the k-row result arrays are all query_nb wide.
    size_t reserved = (size_t)k * query_pitch_in_bytes
                    + (size_t)k * (query_pitch * size_of_int)
                    + (size_t)k * (query_pitch * size_of_int)
#if (KNN_MERGE_METHOD == 1)
                    + (size_t)k * query_pitch_in_bytes
                    + (size_t)k * (query_pitch * size_of_int)
#endif
                    + 64ULL * 1024 * 1024;
    size_t available = (free_bytes > reserved) ? (free_bytes - reserved) : 0;
    int batch_size = (int)(available / query_pitch_in_bytes);
    if (batch_size > ref_nb) batch_size = ref_nb;
    if (batch_size < 1) batch_size = 1;

    float * dist_dev        = NULL;
    int *   batch_index_dev = NULL;
    float * knn_dist_dev    = NULL;
    int *   knn_index_dev   = NULL;
    float * merge_dist_dev  = NULL;
    int *   merge_index_dev = NULL;

    // Frees the work buffers on every return path. It holds references because the
    // result and merge buffers are swapped after each batch.
    struct WorkBufferGuard {
        float *& dist;
        int *&   batch_index;
        float *& knn_dist;
        int *&   knn_index;
        float *& merge_dist;
        int *&   merge_index;
        ~WorkBufferGuard() {
            cudaFree(dist);
            cudaFree(batch_index);
            cudaFree(knn_dist);
            cudaFree(knn_index);
            cudaFree(merge_dist);
            cudaFree(merge_index);
        }
    } work_buffers{dist_dev, batch_index_dev, knn_dist_dev, knn_index_dev, merge_dist_dev, merge_index_dev};

    size_t  dist_pitch_in_bytes;
    err0 = cudaMallocPitch((void**)&dist_dev, &dist_pitch_in_bytes, query_nb * size_of_float, batch_size);

    size_t batch_index_pitch_in_bytes;
    err1 = cudaMallocPitch((void**)&batch_index_dev, &batch_index_pitch_in_bytes, query_nb * size_of_int, k);

    size_t  knn_dist_pitch_in_bytes;
    err2 = cudaMallocPitch((void**)&knn_dist_dev, &knn_dist_pitch_in_bytes, query_nb * size_of_float, k);

    size_t knn_index_pitch_in_bytes;
    err3 = cudaMallocPitch((void**)&knn_index_dev, &knn_index_pitch_in_bytes, query_nb * size_of_int, k);

    cudaError_t err4 = cudaSuccess;
    cudaError_t err5 = cudaSuccess;

#if (KNN_MERGE_METHOD == 1)
    size_t  merge_dist_pitch_in_bytes = 0;
    size_t  merge_index_pitch_in_bytes = 0;
    err4 = cudaMallocPitch((void**)&merge_dist_dev, &merge_dist_pitch_in_bytes, query_nb * size_of_float, k);
    err5 = cudaMallocPitch((void**)&merge_index_dev, &merge_index_pitch_in_bytes, query_nb * size_of_int, k);
#endif

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess ||
        err4 != cudaSuccess || err5 != cudaSuccess) {
        printf("ERROR: Memory allocation error for batched kNN arrays\n");
        return false;
    }

    size_t dist_pitch        = dist_pitch_in_bytes        / size_of_float;
    size_t batch_index_pitch = batch_index_pitch_in_bytes / size_of_int;
    size_t knn_dist_pitch    = knn_dist_pitch_in_bytes    / size_of_float;
    size_t knn_index_pitch   = knn_index_pitch_in_bytes   / size_of_int;

#if (KNN_MERGE_METHOD == 1)
    size_t merge_dist_pitch  = merge_dist_pitch_in_bytes  / size_of_float;
    size_t merge_index_pitch = merge_index_pitch_in_bytes / size_of_int;
#endif

    {
        dim3 block_init(BLOCK_DIM, BLOCK_DIM, 1);
        dim3 grid_init((query_nb + BLOCK_DIM - 1) / BLOCK_DIM,
                       (k + BLOCK_DIM - 1) / BLOCK_DIM, 1);
        init_float_array<<<grid_init, block_init>>>(knn_dist_dev, knn_dist_pitch, query_nb, k, FLT_MAX);
        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel - init_float_array\n");
            return false;
        }
        init_int_array<<<grid_init, block_init>>>(knn_index_dev, knn_index_pitch, query_nb, k, -1);
        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel - init_int_array\n");
            return false;
        }
    }

    for (int batch_start = 0; batch_start < ref_nb; batch_start += batch_size) {
        int current_batch = (batch_start + batch_size <= ref_nb)
                          ? batch_size
                          : (ref_nb - batch_start);

        dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
        dim3 grid0((query_nb + BLOCK_DIM - 1) / BLOCK_DIM,
                   (current_batch + BLOCK_DIM - 1) / BLOCK_DIM, 1);
        compute_distances<<<grid0, block0>>>(ref + batch_start, current_batch, ref_pitch,
                                             query, query_nb, query_pitch, dim,
                                             dist_dev, dist_pitch);
        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel - compute_distances (batch at %d)\n", batch_start);
            return false;
        }

        dim3 block1(256, 1, 1);
        dim3 grid1((query_nb + 255) / 256, 1, 1);
#if (KNN_SELECT_METHOD == 0)
        modified_insertion_sort<<<grid1, block1>>>(dist_dev, dist_pitch,
                                                   batch_index_dev, batch_index_pitch,
                                                   query_nb, current_batch, k, self_knn, batch_start);
#elif (KNN_SELECT_METHOD == 1)
        modified_heap_select<<<grid1, block1>>>(dist_dev, dist_pitch,
                                                batch_index_dev, batch_index_pitch,
                                                query_nb, current_batch, k, self_knn, batch_start);
#elif (KNN_SELECT_METHOD == 2)
        if (k >= KNN_SELECT_HEAP_THRESHOLD) {
            modified_heap_select<<<grid1, block1>>>(dist_dev, dist_pitch,
                                                    batch_index_dev, batch_index_pitch,
                                                    query_nb, current_batch, k, self_knn, batch_start);
        }
        else {
            modified_insertion_sort<<<grid1, block1>>>(dist_dev, dist_pitch,
                                                       batch_index_dev, batch_index_pitch,
                                                       query_nb, current_batch, k, self_knn, batch_start);
        }
#else
#error "Unsupported KNN_SELECT_METHOD"
#endif
        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel - top-k selection (batch at %d)\n", batch_start);
            return false;
        }

        dim3 block_merge(256, 1, 1);
        dim3 grid_merge((query_nb + 255) / 256, 1, 1);
#if (KNN_MERGE_METHOD == 0)
        merge_knn_results_insertion<<<grid_merge, block_merge>>>(knn_dist_dev, knn_dist_pitch,
                                                       knn_index_dev, knn_index_pitch,
                                                       dist_dev, dist_pitch,
                                                       batch_index_dev, batch_index_pitch,
                                                       query_nb, k, batch_start);
#elif (KNN_MERGE_METHOD == 1)
        merge_knn_results_linear<<<grid_merge, block_merge>>>(knn_dist_dev, knn_dist_pitch,
                                                              knn_index_dev, knn_index_pitch,
                                                              dist_dev, dist_pitch,
                                                              batch_index_dev, batch_index_pitch,
                                                              merge_dist_dev, merge_dist_pitch,
                                                              merge_index_dev, merge_index_pitch,
                                                              query_nb, k, batch_start);
#else
#error "Unsupported KNN_MERGE_METHOD"
#endif
        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel - merge_knn_results (batch at %d)\n", batch_start);
            return false;
        }

#if (KNN_MERGE_METHOD == 1)
        {
            float * dist_swap_ptr = knn_dist_dev;
            knn_dist_dev = merge_dist_dev;
            merge_dist_dev = dist_swap_ptr;

            int * index_swap_ptr = knn_index_dev;
            knn_index_dev = merge_index_dev;
            merge_index_dev = index_swap_ptr;

            size_t dist_swap_pitch = knn_dist_pitch;
            knn_dist_pitch = merge_dist_pitch;
            merge_dist_pitch = dist_swap_pitch;

            size_t index_swap_pitch = knn_index_pitch;
            knn_index_pitch = merge_index_pitch;
            merge_index_pitch = index_swap_pitch;

            size_t dist_swap_pitch_bytes = knn_dist_pitch_in_bytes;
            knn_dist_pitch_in_bytes = merge_dist_pitch_in_bytes;
            merge_dist_pitch_in_bytes = dist_swap_pitch_bytes;

            size_t index_swap_pitch_bytes = knn_index_pitch_in_bytes;
            knn_index_pitch_in_bytes = merge_index_pitch_in_bytes;
            merge_index_pitch_in_bytes = index_swap_pitch_bytes;
        }
#endif
    }

    {
        dim3 block2(BLOCK_DIM, BLOCK_DIM, 1);
        dim3 grid2((query_nb + BLOCK_DIM - 1) / BLOCK_DIM,
                   (k + BLOCK_DIM - 1) / BLOCK_DIM, 1);
        compute_sqrt<<<grid2, block2>>>(knn_dist_dev, query_nb, knn_dist_pitch, k);
        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel - compute_sqrt\n");
            return false;
        }
    }


    err0 = cudaMemcpy2D(knn_dist,  query_nb * size_of_float, knn_dist_dev,  knn_dist_pitch_in_bytes,
                        query_nb * size_of_float, k, cudaMemcpyDeviceToHost);
    err1 = cudaMemcpy2D(knn_index, query_nb * size_of_int,   knn_index_dev, knn_index_pitch_in_bytes,
                        query_nb * size_of_int,   k, cudaMemcpyDeviceToHost);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy kNN results from device to host\n");
        return false;
    }

    return true;
}

bool knncuda_self_knn(Dataset &dataset,
                      int k,
                      float *knn_dist,
                      int *knn_index)
{
    if (!dataset.EnsureUploaded())
        return false;

    return knn_cuda_global(dataset.device_data(),
                           dataset.device_pitch_bytes(),
                           dataset.num_points(),
                           dataset.device_data(),
                           dataset.device_pitch_bytes(),
                           dataset.num_points(),
                           dataset.dim(),
                           k,
                           dataset.cuda_device_id(),
                           knn_dist,
                           knn_index,
                           true);
}
