#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include "symtriknncuda.h"
#include "dataset.h"
#include "knndcshared.h"

#ifndef SKN_FORCE_TILE
#define SKN_FORCE_TILE 0
#endif
#if SKN_FORCE_TILE < 0
#error "SKN_FORCE_TILE must be 0 (auto) or a positive tile size"
#endif

#define SKN_WARP 32
#ifndef SKN_SELECTOR_WARPS_PER_BLOCK
#define SKN_SELECTOR_WARPS_PER_BLOCK 8
#endif
#if SKN_SELECTOR_WARPS_PER_BLOCK != 2 && SKN_SELECTOR_WARPS_PER_BLOCK != 4 && SKN_SELECTOR_WARPS_PER_BLOCK != 8
#error "SKN_SELECTOR_WARPS_PER_BLOCK must be 2, 4, or 8"
#endif
#define SKN_BLOCK (SKN_WARP * SKN_SELECTOR_WARPS_PER_BLOCK)

#ifndef SKN_HEAP_HOLE_SIFT
#define SKN_HEAP_HOLE_SIFT 1
#endif
#if SKN_HEAP_HOLE_SIFT != 0 && SKN_HEAP_HOLE_SIFT != 1
#error "SKN_HEAP_HOLE_SIFT must be 0 or 1"
#endif

static inline int skn_min_int(int a, int b) { return (a < b) ? a : b; }

static inline size_t skn_selector_shared_bytes(int warps, int neighbors)
{
    return (size_t)warps * neighbors * (sizeof(float) + sizeof(int));
}

#define SKN_CUDA_CHECK_GOTO(call, label) KNNDC_CUDA_CHECK_GOTO((call), label)
#define SKN_CUBLAS_CHECK_GOTO(call, label) KNNDC_CUBLAS_CHECK_GOTO((call), label)

__global__ void skn_init_topk(int N, int K, float* out_dist, int* out_idx)
{
    
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t total = (size_t)N * K;
    for (size_t i = tid; i < total; i += stride) {
        out_dist[i] = FLT_MAX;
        out_idx[i] = -1;
    }
}

__global__ void skn_compute_norms(const float* __restrict__ data,
                                  int N,
                                  int D,
                                  int pitch,
                                  float* __restrict__ norms_sq)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            float v = data[(size_t)d * pitch + i];
            sum = fmaf(v, v, sum);
        }
        norms_sq[i] = sum;
    }
}

__device__ __forceinline__ void skn_heapify_down(float* h_dist,
                                                  int* h_idx,
                                                  int K,
                                                  int root
                                                  )
{
    int parent = root;
    while (true) {
        int left = 2 * parent + 1;
        int right = 2 * parent + 2;
        int largest = parent;
        
        if (left < K && (h_dist[left] > h_dist[largest] ||
                         (h_dist[left] == h_dist[largest] && h_idx[left] > h_idx[largest]))) largest = left;
        if (right < K && (h_dist[right] > h_dist[largest] ||
                          (h_dist[right] == h_dist[largest] && h_idx[right] > h_idx[largest]))) largest = right;
        if (largest == parent) break;


        float td = h_dist[parent];
        h_dist[parent] = h_dist[largest];
        h_dist[largest] = td;

        int ti = h_idx[parent];
        h_idx[parent] = h_idx[largest];
        h_idx[largest] = ti;
        parent = largest;
    }
}

__device__ __forceinline__ void skn_heap_insert(float* h_dist,
                                                 int* h_idx,
                                                 int K,
                                                 float dist,
                                                 int idx
                                                 )
{
    if (dist >= h_dist[0]) return;
#if SKN_HEAP_HOLE_SIFT == 1
    int parent = 0;
    while (2 * parent + 1 < K) {
        int child = 2 * parent + 1;
        const int right = child + 1;
        if (right < K && (h_dist[right] > h_dist[child] ||
                         (h_dist[right] == h_dist[child] && h_idx[right] > h_idx[child]))) {
            child = right;
        }
        const float child_dist = h_dist[child];
        const int child_idx = h_idx[child];
        if (!(child_dist > dist || (child_dist == dist && child_idx > idx))) break;
        h_dist[parent] = child_dist;
        h_idx[parent] = child_idx;
        parent = child;
    }
    h_dist[parent] = dist;
    h_idx[parent] = idx;
#else
    h_dist[0] = dist;
    h_idx[0] = idx;
    skn_heapify_down(h_dist, h_idx, K, 0
                     );
#endif
}

#include "symtriknnfds.h"

__global__ void skn_update_topk_from_dot_block(const float* __restrict__ dot_block,
                                               int rb,
                                               int qb,
                                               int q_start,
                                               int r_start,
                                               int K,
                                               const float* __restrict__ norms_sq,
                                               float* __restrict__ out_dist,
                                               int* __restrict__ out_idx
                                               )
{
    const int num_warps = blockDim.x / SKN_WARP;
    const int warp_id = threadIdx.x / SKN_WARP;
    const int lane_id = threadIdx.x % SKN_WARP;
    const int q_local = blockIdx.x * num_warps + warp_id;
    if (q_local >= qb) return;

    const int q_global = q_start + q_local;
    float* hd = out_dist + (size_t)q_global * K;
    int* hi = out_idx + (size_t)q_global * K;

    extern __shared__ char smem[];
    float* s_d = (float*)smem;
    int* s_i = (int*)&s_d[(size_t)num_warps * K];
    float* heap_d = s_d + (size_t)warp_id * K;
    int* heap_i = s_i + (size_t)warp_id * K;

    for (int i = lane_id; i < K; i += SKN_WARP) {
        heap_d[i] = hd[i];
        heap_i[i] = hi[i];
    }
    __syncwarp();

    float norm_q = norms_sq[q_global];
    float worst = (lane_id == 0) ? heap_d[0] : FLT_MAX;
    worst = __shfl_sync(0xffffffff, worst, 0);

    for (int r_base = 0; r_base < rb; r_base += SKN_WARP) {
        int r_local = r_base + lane_id;
        int r_global = r_start + r_local;

        float dist = FLT_MAX;
        if (r_local < rb) {
            float neg2dot = dot_block[(size_t)r_local + (size_t)q_local * rb];
            dist = norm_q + norms_sq[r_global] + neg2dot;
            dist = fmaxf(dist, 0.0f);
        }

        bool valid = (r_local < rb) && (r_global != q_global) && (dist < worst);
        unsigned cand = __ballot_sync(0xffffffff, valid);

        while (cand != 0) {
            int l = __ffs(cand) - 1;
            float ld = __shfl_sync(0xffffffff, dist, l);
            int li = __shfl_sync(0xffffffff, r_global, l);
            if (lane_id == 0) {
                skn_heap_insert(heap_d, heap_i, K, ld, li
                                );
                worst = heap_d[0];
            }
            worst = __shfl_sync(0xffffffff, worst, 0);
            cand &= (cand - 1);
        }
    }

    __syncwarp();
    for (int i = lane_id; i < K; i += SKN_WARP) {
        hd[i] = heap_d[i];
        hi[i] = heap_i[i];
    }
}

__global__ void skn_gather_heap_roots(const float* __restrict__ out_dist,
                                      int K,
                                      int r_start,
                                      int rb,
                                      float* __restrict__ roots)
{
    int r_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (r_local < rb) roots[r_local] = out_dist[(size_t)(r_start + r_local) * K];
}

#include "symtriknnreverse.h"
#include "symtriknnsort.h"

__global__ void skn_sqrt_topk(int N, int K, float* out_dist)
{
    
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t total = (size_t)N * K;
    for (size_t i = tid; i < total; i += stride) {
        out_dist[i] = sqrtf(fmaxf(out_dist[i], 0.0f));
    }
}

__global__ void skn_transpose_to_rank_major(const float* src_dist,
                                            const int* src_idx,
                                            float* dst_dist,
                                            int* dst_idx,
                                            int N,
                                            int K,
                                            int dst_dist_pitch,
                                            int dst_idx_pitch)
{
    int pt = blockIdx.x * blockDim.x + threadIdx.x;
    int rank = blockIdx.y * blockDim.y + threadIdx.y;
    if (pt < N && rank < K) {
        dst_dist[(size_t)rank * dst_dist_pitch + pt] = src_dist[(size_t)pt * K + rank];
        dst_idx[(size_t)rank * dst_idx_pitch + pt] = src_idx[(size_t)pt * K + rank];
    }
}

static int skn_choose_block_auto(int N, size_t free_bytes)
{
    const size_t mem_budget = free_bytes / 3;
    const int large_candidates[] = {9216, 8192, 6144, 4096, 3072, 2048, 1024};
    const int small_candidates[] = {3072, 2048, 1024, 512, 256, 128, 64, 32, 16, 8};

    const int* candidates = (N < 32768) ? small_candidates : large_candidates;
    int count = (N < 32768)
        ? (int)(sizeof(small_candidates) / sizeof(small_candidates[0]))
        : (int)(sizeof(large_candidates) / sizeof(large_candidates[0]));

    for (int i = 0; i < count; ++i) {
        int b = candidates[i];
        if (b > N) continue;
        size_t ws = (size_t)b * b * sizeof(float);
        if (ws <= mem_budget) return b;
    }

    return (N < 1024) ? N : 1024;
}

static bool skn_try_alloc_workspace(int N,
                                    int preferred_block,
                                    float** d_dot_ws,
                                    int* selected_qb,
                                    int* selected_rb)
{
    const int fallback_candidates[] = {16384, 14336, 12288, 10240, 9216, 8192,
                                       6144, 4096, 3072, 2048, 1024, 512,
                                       256, 128, 64, 32, 16, 8};
    
    if (preferred_block > 0 && preferred_block <= N) {
        size_t ws_bytes = (size_t)preferred_block * preferred_block * sizeof(float);
        if (cudaMalloc((void**)d_dot_ws, ws_bytes) == cudaSuccess) {
            *selected_qb = preferred_block;
            *selected_rb = preferred_block;
            return true;
        }
        cudaGetLastError();
    }
 
    int count = (int)(sizeof(fallback_candidates) / sizeof(fallback_candidates[0]));
    for (int i = 0; i < count; ++i) {
        int b = fallback_candidates[i];
        if (b == preferred_block || b <= 0 || b > N) continue;
        size_t ws_bytes = (size_t)b * b * sizeof(float);
        if (cudaMalloc((void**)d_dot_ws, ws_bytes) == cudaSuccess) {
            *selected_qb = b;
            *selected_rb = b;
            return true;
        }
        cudaGetLastError();
    }

    return false;
}

static bool symtri_knn_cuda_impl(const float * data_dev,
                                 size_t        data_pitch_bytes,
                                 int           num_points,
                                 int           dim,
                                 int           k,
                                 int           cuda_device_id,
                                 float *       knn_dist,
                                 int *         knn_index)
{
    if (num_points <= 1 || k <= 0) {
        return true;
    }
    if (k >= num_points) {
        return false;
    }
    if (data_dev == NULL || data_pitch_bytes % sizeof(float) != 0) {
        return false;
    }

    if (!select_cuda_device(cuda_device_id)) {
        return false;
    }
    cudaError_t err;

    const unsigned int szf = sizeof(float);
    const unsigned int szi = sizeof(int);

    const int data_pitch = (int)(data_pitch_bytes / szf);

    float* norms_dev = NULL;
    float* out_dist_point = NULL;      
    int* out_idx_point = NULL;         
    float* dot_ws = NULL;              
    float* dot_transposed_ws = NULL;
    float* rev_roots = NULL;           
    unsigned* rev_bitmap = NULL;       
    int rev_stride = 0;                
    float* fds_roots_q = NULL;         
    float* fds_roots_r = NULL;         
    unsigned* fds_fwd_bitmap = NULL;   
    unsigned* fds_rev_bitmap = NULL;   
    int fds_stride = 0;
    bool use_fds = false;              

    float* out_dist_dev = NULL;        
    int* out_idx_dev = NULL;           
    size_t out_dist_pitch_bytes = 0;
    size_t out_idx_pitch_bytes = 0;
    int out_dist_pitch = 0;
    int out_idx_pitch = 0;

    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    int cublas_qb = 0;
    int cublas_rb = 0;

    cublasHandle_t cublas_handle = NULL;
    
    SKN_CUDA_CHECK_GOTO(cudaMalloc((void**)&norms_dev, (size_t)num_points * szf), cleanup_fail);
    SKN_CUDA_CHECK_GOTO(cudaMalloc((void**)&out_dist_point, (size_t)num_points * k * szf), cleanup_fail);
    SKN_CUDA_CHECK_GOTO(cudaMalloc((void**)&out_idx_point, (size_t)num_points * k * szi), cleanup_fail);

    SKN_CUDA_CHECK_GOTO(cudaMallocPitch((void**)&out_dist_dev, &out_dist_pitch_bytes,
                                        (size_t)num_points * szf, k), cleanup_fail);
    SKN_CUDA_CHECK_GOTO(cudaMallocPitch((void**)&out_idx_dev, &out_idx_pitch_bytes,
                                        (size_t)num_points * szi, k), cleanup_fail);
    out_dist_pitch = (int)(out_dist_pitch_bytes / szf);
    out_idx_pitch = (int)(out_idx_pitch_bytes / szi);
    
    SKN_CUBLAS_CHECK_GOTO(cublasCreate(&cublas_handle), cleanup_fail);
    SKN_CUBLAS_CHECK_GOTO(cublasSetStream(cublas_handle, 0), cleanup_fail);
    
#if (USE_CUBLAS_TF32 == 1)
    if (dim >= TF32_MIN_DIM) {
        math_mode = CUBLAS_TF32_TENSOR_OP_MATH;
    }
#endif
    SKN_CUBLAS_CHECK_GOTO(cublasSetMathMode(cublas_handle, math_mode), cleanup_fail);

    {
        int block = 256;
        int grid = skn_min_int(4096, (num_points + block - 1) / block);
        skn_compute_norms<<<grid, block>>>(data_dev, num_points, dim, data_pitch, norms_dev);
        SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);
    }

    {
        size_t free_bytes = 0, total_bytes = 0;
        SKN_CUDA_CHECK_GOTO(cudaMemGetInfo(&free_bytes, &total_bytes), cleanup_fail);

        if (SKN_FORCE_TILE > 0) {
            int forced = skn_min_int(SKN_FORCE_TILE, num_points);
            err = cudaMalloc((void**)&dot_ws, (size_t)forced * forced * szf);
            if (err != cudaSuccess) {
                cudaGetLastError();
                goto cleanup_fail;
            }
            cublas_qb = cublas_rb = forced;
        } else {
            int preferred_block = skn_choose_block_auto(num_points, free_bytes);
            if (!skn_try_alloc_workspace(num_points, preferred_block,
                                         &dot_ws, &cublas_qb, &cublas_rb)) {
                goto cleanup_fail;
            }
        }
    }

    if (k >= SKN_COALESCED_TRANSPOSE_MIN_K && cublas_qb == cublas_rb &&
        num_points > cublas_qb) {
        err = cudaMalloc((void**)&dot_transposed_ws, (size_t)cublas_qb * cublas_rb * szf);
        if (err != cudaSuccess) {
            cudaGetLastError();
        }
    }
    
#if SKN_FUSED_REVERSE
    if (cublas_qb == cublas_rb && num_points > cublas_qb) {
        rev_stride = (cublas_qb + 31) / 32;
        const size_t bitmap_bytes = (size_t)cublas_rb * rev_stride * sizeof(unsigned);
        err = cudaMalloc((void**)&rev_roots, (size_t)cublas_rb * szf);
        if (err == cudaSuccess) err = cudaMalloc((void**)&rev_bitmap, bitmap_bytes);
        if (err == cudaSuccess) err = cudaMemset(rev_bitmap, 0, bitmap_bytes);
        if (err != cudaSuccess) {
            cudaGetLastError();
            if (rev_roots) cudaFree(rev_roots);
            if (rev_bitmap) cudaFree(rev_bitmap);
            rev_roots = NULL;
            rev_bitmap = NULL;
        }
    }
#endif
    
#if SKN_FDS_MAX_DIM > 0 && USE_CUBLAS_TF32 == 0
    use_fds = (cublas_qb == cublas_rb) && dim <= SKN_FDS_MAX_DIM;
    if (use_fds && num_points > cublas_qb && !SKN_FDS_REFERENCE) {
        fds_stride = (cublas_qb + 31) / 32;
        const size_t bitmap_bytes = (size_t)cublas_qb * fds_stride * sizeof(unsigned);
        err = cudaMalloc((void**)&fds_roots_q, (size_t)cublas_qb * szf);
        if (err == cudaSuccess) err = cudaMalloc((void**)&fds_roots_r, (size_t)cublas_qb * szf);
        if (err == cudaSuccess) err = cudaMalloc((void**)&fds_fwd_bitmap, bitmap_bytes);
        if (err == cudaSuccess) err = cudaMalloc((void**)&fds_rev_bitmap, bitmap_bytes);
        if (err != cudaSuccess) {
            cudaGetLastError();
            use_fds = false;
        }
    }
#endif

    {
        const float alpha = -2.0f;
        const float beta = 0.0f;

        int init_block = 256;
        int init_grid = (int)((((size_t)num_points * k) + init_block - 1) / init_block);
        if (init_grid > 4096) init_grid = 4096;
        skn_init_topk<<<init_grid, init_block>>>(num_points, k, out_dist_point, out_idx_point);
        SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);

        int block = SKN_BLOCK;
        int warps = block / SKN_WARP;
         size_t smem = skn_selector_shared_bytes(warps, k);
 
        const bool use_upper_triangle = (cublas_qb == cublas_rb);
        
        const bool use_fused_reverse = use_upper_triangle && rev_bitmap != NULL && k <= cublas_qb;
        for (int q_start = 0; q_start < num_points; q_start += cublas_qb) {
            int qb = skn_min_int(cublas_qb, num_points - q_start);
            const float* q_ptr = data_dev + q_start;

            int r_begin = use_upper_triangle ? q_start : 0;
            for (int r_start = r_begin; r_start < num_points; r_start += cublas_rb) {
                int rb = skn_min_int(cublas_rb, num_points - r_start);
                const float* r_ptr = data_dev + r_start;

                if (use_fds && !SKN_FDS_REFERENCE && q_start > 0 && r_start != q_start) {
                    skn_gather_heap_roots<<<(qb + 255) / 256, 256>>>(out_dist_point, k, q_start, qb, fds_roots_q);
                    skn_gather_heap_roots<<<(rb + 255) / 256, 256>>>(out_dist_point, k, r_start, rb, fds_roots_r);
                    dim3 fds_grid((rb + SKN_FDS_TILE - 1) / SKN_FDS_TILE, (qb + SKN_FDS_TILE - 1) / SKN_FDS_TILE);
                    skn_fds_tile_kernel<false><<<fds_grid, SKN_FDS_THREADS>>>(
                        data_dev, data_pitch, dim, norms_dev, q_start, qb, r_start, rb,
                        fds_roots_q, fds_roots_r, fds_fwd_bitmap, fds_rev_bitmap, fds_stride, dot_ws);
                    SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);
                    const int q_blocks = (qb + warps - 1) / warps, r_blocks = (rb + warps - 1) / warps;
                    skn_fds_replay_both<<<q_blocks + r_blocks, block, smem>>>(
                        q_start, qb, r_start, rb, k, out_dist_point, out_idx_point,
                        fds_fwd_bitmap, fds_rev_bitmap, fds_stride, dot_ws, q_blocks);
                    SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);
                    continue;
                }

                if (use_fds) {
                    dim3 fds_grid((rb + SKN_FDS_TILE - 1) / SKN_FDS_TILE, (qb + SKN_FDS_TILE - 1) / SKN_FDS_TILE);
                    skn_fds_tile_kernel<true><<<fds_grid, SKN_FDS_THREADS>>>(
                        data_dev, data_pitch, dim, norms_dev, q_start, qb, r_start, rb,
                        NULL, NULL, NULL, NULL, 0, dot_ws);
                    SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);
                } else
                SKN_CUBLAS_CHECK_GOTO(cublasSgemm(cublas_handle,
                                                  CUBLAS_OP_N,
                                                  CUBLAS_OP_T,
                                                  rb,
                                                  qb,
                                                  dim,
                                                  &alpha,
                                                  r_ptr,
                                                  data_pitch,
                                                  q_ptr,
                                                  data_pitch,
                                                  &beta,
                                                  dot_ws,
                                                  rb), cleanup_fail);

                int grid = (qb + warps - 1) / warps;
                const bool fused_tile = use_fused_reverse && q_start > 0 && r_start != q_start;
                if (fused_tile) {
                    skn_gather_heap_roots<<<(rb + 255) / 256, 256>>>(
                        out_dist_point, k, r_start, rb, rev_roots);
                    SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);
                    skn_update_topk_fused_forward<<<grid, block, smem>>>(
                        dot_ws, rb, qb, q_start, r_start, k,
                        norms_dev, out_dist_point, out_idx_point,
                        rev_roots, rev_bitmap, rev_stride
                        );
                } else {
                    skn_update_topk_from_dot_block<<<grid, block, smem>>>(
                        dot_ws, rb, qb, q_start, r_start, k,
                        norms_dev, out_dist_point, out_idx_point
                        );
                }
                SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);

                if (use_upper_triangle && r_start != q_start) {
                    int grid_t = (rb + warps - 1) / warps;
                    if (fused_tile) {
                        skn_update_topk_from_reverse_bitmap<<<grid_t, block, smem>>>(
                            dot_ws, rb, qb, r_start, q_start, k,
                            norms_dev, out_dist_point, out_idx_point,
                            rev_bitmap, rev_stride
                            );
                    } else if (dot_transposed_ws) {
                        dim3 transpose_block(32, 8);
                        dim3 transpose_grid((rb + 31) / 32, (qb + 31) / 32);
                        skn_transpose_dot_tile<<<transpose_grid, transpose_block>>>(
                            dot_ws, dot_transposed_ws, rb, qb);
                        SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);
                        skn_update_topk_from_dot_block<<<grid_t, block, smem>>>(
                            dot_transposed_ws, qb, rb, r_start, q_start, k,
                            norms_dev, out_dist_point, out_idx_point
                            );
                    } else {
                        skn_update_topk_from_dot_block_transpose<<<grid_t, block, smem>>>(
                            dot_ws, rb, qb, r_start, q_start, k,
                            norms_dev, out_dist_point, out_idx_point
                            );
                    }
                    SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);
                }
            }
        }

        skn_launch_sort_topk(num_points, k, out_dist_point, out_idx_point);
        SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);

        int sqrt_block = 256;
        int sqrt_grid = (int)((((size_t)num_points * k) + sqrt_block - 1) / sqrt_block);
        if (sqrt_grid > 4096) sqrt_grid = 4096;
        skn_sqrt_topk<<<sqrt_grid, sqrt_block>>>(num_points, k, out_dist_point);
        SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);
    }
    
    {
        dim3 block(16, 16);
        dim3 grid((num_points + 15) / 16, (k + 15) / 16);
        skn_transpose_to_rank_major<<<grid, block>>>(
            out_dist_point, out_idx_point,
            out_dist_dev, out_idx_dev,
            num_points, k, out_dist_pitch, out_idx_pitch);
        SKN_CUDA_CHECK_GOTO(cudaGetLastError(), cleanup_fail);
    }

    SKN_CUDA_CHECK_GOTO(cudaDeviceSynchronize(), cleanup_fail);

    SKN_CUDA_CHECK_GOTO(cudaMemcpy2D(knn_dist, (size_t)num_points * szf,
                                     out_dist_dev, out_dist_pitch_bytes,
                                     (size_t)num_points * szf, k,
                                     cudaMemcpyDeviceToHost), cleanup_fail);

    SKN_CUDA_CHECK_GOTO(cudaMemcpy2D(knn_index, (size_t)num_points * szi,
                                     out_idx_dev, out_idx_pitch_bytes,
                                     (size_t)num_points * szi, k,
                                     cudaMemcpyDeviceToHost), cleanup_fail);

    cudaFree(norms_dev);
    cudaFree(out_dist_point);
    cudaFree(out_idx_point);
    cudaFree(dot_ws);
    cudaFree(dot_transposed_ws);
    cudaFree(rev_roots);
    cudaFree(rev_bitmap);
    cudaFree(fds_roots_q);
    cudaFree(fds_roots_r);
    cudaFree(fds_fwd_bitmap);
    cudaFree(fds_rev_bitmap);
    cudaFree(out_dist_dev);
    cudaFree(out_idx_dev);
    if (cublas_handle) cublasDestroy(cublas_handle);
    return true;

cleanup_fail:
    if (norms_dev) cudaFree(norms_dev);
    if (out_dist_point) cudaFree(out_dist_point);
    if (out_idx_point) cudaFree(out_idx_point);
    if (dot_ws) cudaFree(dot_ws);
    if (dot_transposed_ws) cudaFree(dot_transposed_ws);
    if (rev_roots) cudaFree(rev_roots);
    if (rev_bitmap) cudaFree(rev_bitmap);
    if (fds_roots_q) cudaFree(fds_roots_q);
    if (fds_roots_r) cudaFree(fds_roots_r);
    if (fds_fwd_bitmap) cudaFree(fds_fwd_bitmap);
    if (fds_rev_bitmap) cudaFree(fds_rev_bitmap);
    if (out_dist_dev) cudaFree(out_dist_dev);
    if (out_idx_dev) cudaFree(out_idx_dev);
    if (cublas_handle) cublasDestroy(cublas_handle);
    return false;
}

bool symtri_knn_cuda(Dataset &dataset,
                     int      k,
                     float *  knn_dist,
                     int *    knn_index)
{
    if (!dataset.EnsureUploaded())
        return false;

    return symtri_knn_cuda_impl(dataset.device_data(),
                                dataset.device_pitch_bytes(),
                                dataset.num_points(),
                                dataset.dim(),
                                k,
                                dataset.cuda_device_id(),
                                knn_dist,
                                knn_index);
}
