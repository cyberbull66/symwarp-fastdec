#include "symwarpfastdeccuda.h"
#include <float.h>
#include <stdio.h>
#include <string>
#include "appexception.h"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#define FDG_CUDA_CHECK(call)                                                          \
    do {                                                                              \
        cudaError_t _err = (call);                                                    \
        if (_err != cudaSuccess) {                                                    \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                   \
                   cudaGetErrorString(_err));                                         \
            throw AppException(std::string("CUDA error in ") + __func__ + ": " +      \
                               cudaGetErrorString(_err));                             \
        }                                                                             \
    } while (0)

#define FDG_BLOCK_SIZE 256
#define FDG_SUM_BLOCKS 1024
#define FDG_MAX_JUMP_ROUNDS 64

static int num_blocks(size_t count)
{
    return (int)((count + FDG_BLOCK_SIZE - 1) / FDG_BLOCK_SIZE);
}

void gpu_copy_to_host(void* dst_host, const void* src_dev, size_t bytes)
{
    if (bytes > 0)
        FDG_CUDA_CHECK(cudaMemcpy(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost));
}

void gpu_copy_to_device(void* dst_dev, const void* src_host, size_t bytes)
{
    if (bytes > 0)
        FDG_CUDA_CHECK(cudaMemcpy(dst_dev, src_host, bytes, cudaMemcpyHostToDevice));
}

void gpu_copy_on_device(void* dst_dev, const void* src_dev, size_t bytes)
{
    if (bytes > 0)
        FDG_CUDA_CHECK(cudaMemcpy(dst_dev, src_dev, bytes, cudaMemcpyDeviceToDevice));
}

__device__ double block_sum(double v)
{
    __shared__ double partial[FDG_BLOCK_SIZE];
    partial[threadIdx.x] = v;
    __syncthreads();
    for (int half = FDG_BLOCK_SIZE / 2; half > 0; half >>= 1)
    {
        if (threadIdx.x < half)
            partial[threadIdx.x] += partial[threadIdx.x + half];
        __syncthreads();
    }
    return partial[0];
}

__global__ void sum_partials_kernel(const float* values, size_t count, double* block_sums)
{
    double sum = 0.0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x)
    {
        sum += values[i];
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0)
        block_sums[blockIdx.x] = sum;
}

__global__ void sum_final_kernel(const double* block_sums, int num_block_sums, double* total)
{
    double sum = 0.0;
    for (int i = threadIdx.x; i < num_block_sums; i += blockDim.x)
        sum += block_sums[i];
    sum = block_sum(sum);
    if (threadIdx.x == 0)
        *total = sum;
}

__global__ void density_laplacian_kernel(const float* knn_dists, int num_points, int k,
                                         const double* dist_sum, float* densities)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    const double mean = *dist_sum / ((double)num_points * k);
    double density = 0.0;
    for (int q = 0; q < k; ++q)
    {
        density += exp(-knn_dists[(size_t)q * num_points + i] / 2 / mean);
    }
    densities[i] = (float)density;
}

__global__ void density_gaussian_kernel(const float* knn_dists, int num_points, int k,
                                        const double* dist_sum, float* densities)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    const double mean = *dist_sum / ((double)num_points * k);
    const double two_mean_sq = 2.0 * mean * mean;
    double density = 0.0;
    for (int q = 0; q < k; ++q)
    {
        const double d = knn_dists[(size_t)q * num_points + i];
        density += exp(-d * d / two_mean_sq);
    }
    densities[i] = (float)density;
}

template <typename DensityKernel>
static void gpu_density_estimation(DensityKernel density_kernel, const float* knn_dists_dev,
                                   int num_points, int k, float* densities_dev)
{
    if (num_points <= 0)
        return;

    const size_t count = (size_t)num_points * k;
    int sum_blocks = num_blocks(count);
    if (sum_blocks > FDG_SUM_BLOCKS)
        sum_blocks = FDG_SUM_BLOCKS;
    if (sum_blocks < 1)
        sum_blocks = 1;

    DeviceBuffer<double> sums;
    sums.allocate(sum_blocks + 1);
    double* total = sums.get() + sum_blocks;

    sum_partials_kernel<<<sum_blocks, FDG_BLOCK_SIZE>>>(knn_dists_dev, count, sums.get());
    FDG_CUDA_CHECK(cudaGetLastError());
    sum_final_kernel<<<1, FDG_BLOCK_SIZE>>>(sums.get(), sum_blocks, total);
    FDG_CUDA_CHECK(cudaGetLastError());

    density_kernel<<<num_blocks(num_points), FDG_BLOCK_SIZE>>>(knn_dists_dev, num_points, k,
                                                               total, densities_dev);
    FDG_CUDA_CHECK(cudaGetLastError());
    FDG_CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_density_estimation_laplacian(const float* knn_dists_dev, int num_points, int k,
                                      float* densities_dev)
{
    gpu_density_estimation(density_laplacian_kernel, knn_dists_dev, num_points, k, densities_dev);
}

void gpu_density_estimation_gaussian(const float* knn_dists_dev, int num_points, int k,
                                     float* densities_dev)
{
    gpu_density_estimation(density_gaussian_kernel, knn_dists_dev, num_points, k, densities_dev);
}

__global__ void dc_detection_kernel(const int* knn_indices, const float* densities,
                                    int num_points, int k, int* cdh_ids)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    const float density = densities[i];
    int cdh = -1;
    for (int q = 0; q < k; ++q)
    {
        int j = knn_indices[(size_t)q * num_points + i];
        if (density < densities[j])
        {
            cdh = j;
            break;
        }
    }
    cdh_ids[i] = cdh;
}

struct IsCore
{
    __host__ __device__ bool operator()(int cdh_id) const { return cdh_id == -1; }
};

void gpu_dc_detection(const int* knn_indices_dev, const float* densities_dev, int num_points, int k,
                      int* cdh_ids_dev, int* core_indices_dev, int* num_cores)
{
    *num_cores = 0;
    if (num_points <= 0)
        return;

    dc_detection_kernel<<<num_blocks(num_points), FDG_BLOCK_SIZE>>>(knn_indices_dev, densities_dev,
                                                                    num_points, k, cdh_ids_dev);
    FDG_CUDA_CHECK(cudaGetLastError());

    int* end = thrust::copy_if(thrust::device,
                               thrust::counting_iterator<int>(0),
                               thrust::counting_iterator<int>(num_points),
                               cdh_ids_dev,
                               core_indices_dev,
                               IsCore());
    FDG_CUDA_CHECK(cudaDeviceSynchronize());
    *num_cores = (int)(end - core_indices_dev);
}

__global__ void gather_kernel(const float* values, const int* indices, int count, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        out[i] = values[indices[i]];
}

__global__ void scatter_kernel(const int* values, const int* indices, int count, int* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        out[indices[i]] = values[i];
}

__global__ void gather_points_kernel(const float* data, int data_ld, int dim,
                                     const int* indices, int count, float* out)
{
    size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= (size_t)count * dim)
        return;

    int d = (int)(t / count);
    int c = (int)(t % count);
    out[t] = data[(size_t)d * data_ld + indices[c]];
}

void gpu_gather(const float* values_dev, const int* indices_dev, int count, float* out_dev)
{
    if (count <= 0)
        return;

    gather_kernel<<<num_blocks(count), FDG_BLOCK_SIZE>>>(values_dev, indices_dev, count, out_dev);
    FDG_CUDA_CHECK(cudaGetLastError());
}

void gpu_scatter(const int* values_dev, const int* indices_dev, int count, int* out_dev)
{
    if (count <= 0)
        return;

    scatter_kernel<<<num_blocks(count), FDG_BLOCK_SIZE>>>(values_dev, indices_dev, count, out_dev);
    FDG_CUDA_CHECK(cudaGetLastError());
}

void gpu_gather_points(const float* data_dev, int data_ld, int dim,
                       const int* indices_dev, int count, float* out_dev)
{
    if (count <= 0 || dim <= 0)
        return;

    gather_points_kernel<<<num_blocks((size_t)count * dim), FDG_BLOCK_SIZE>>>(
        data_dev, data_ld, dim, indices_dev, count, out_dev);
    FDG_CUDA_CHECK(cudaGetLastError());
}

__global__ void set_center_labels_kernel(const int* centers, int num_centers, int* center_labels)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < num_centers)
        center_labels[centers[c]] = c;
}

__global__ void init_parents_kernel(const int* cdh_ids, const int* center_labels, int num_points,
                                    int* parent)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    const bool chain_end = center_labels[i] >= 0 || cdh_ids[i] == -1;
    parent[i] = chain_end ? i : cdh_ids[i];
}

__global__ void pointer_jumping_kernel(int* parent, int num_points, int* changed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    int p = parent[i];
    int pp = parent[p];
    if (pp != p)
    {
        parent[i] = pp;
        *changed = 1;
    }
}

__global__ void assign_labels_kernel(const int* parent, const int* center_labels, int num_points,
                                     int* labels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    int label = center_labels[parent[i]];
    labels[i] = (label >= 0) ? label : 0;
}

void gpu_final_cluster(const int* cdh_ids_dev, int num_points,
                       const int* center_indices_dev, int num_centers,
                       int* labels_host)
{
    if (num_points <= 0)
        return;

    DeviceBuffer<int> center_labels;
    DeviceBuffer<int> parent;
    DeviceBuffer<int> labels;
    DeviceBuffer<int> changed;
    center_labels.allocate(num_points);
    parent.allocate(num_points);
    labels.allocate(num_points);
    changed.allocate(1);

    FDG_CUDA_CHECK(cudaMemset(center_labels.get(), 0xFF, (size_t)num_points * sizeof(int)));
    if (num_centers > 0)
    {
        set_center_labels_kernel<<<num_blocks(num_centers), FDG_BLOCK_SIZE>>>(
            center_indices_dev, num_centers, center_labels.get());
        FDG_CUDA_CHECK(cudaGetLastError());
    }

    const int blocks = num_blocks(num_points);
    init_parents_kernel<<<blocks, FDG_BLOCK_SIZE>>>(cdh_ids_dev, center_labels.get(), num_points,
                                                    parent.get());
    FDG_CUDA_CHECK(cudaGetLastError());

    int rounds = 0;
    for (;; ++rounds)
    {
        if (rounds == FDG_MAX_JUMP_ROUNDS)
            throw AppException("gpu_final_cluster: pointer jumping did not converge (cycle in CDH links?)");

        int h_changed = 0;
        FDG_CUDA_CHECK(cudaMemset(changed.get(), 0, sizeof(int)));
        pointer_jumping_kernel<<<blocks, FDG_BLOCK_SIZE>>>(parent.get(), num_points, changed.get());
        FDG_CUDA_CHECK(cudaGetLastError());
        gpu_copy_to_host(&h_changed, changed.get(), sizeof(int));
        if (!h_changed)
            break;
    }

    assign_labels_kernel<<<blocks, FDG_BLOCK_SIZE>>>(parent.get(), center_labels.get(), num_points,
                                                     labels.get());
    FDG_CUDA_CHECK(cudaGetLastError());
    gpu_copy_to_host(labels_host, labels.get(), (size_t)num_points * sizeof(int));

    printf("Final clustering: %d pointer-jumping round(s)\n", rounds + 1);
}
