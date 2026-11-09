#include "dcreductioncuda.h"
#include "dataset.h"
#include "knndcshared.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#define DC_CB_CHECK_CUDA(expr, label) KNNDC_CUDA_CHECK_GOTO((expr), cleanup)
#define DC_CB_CHECK_LAST(label) KNNDC_CUDA_CHECK_LAST_GOTO(cleanup)
#define DC_CB_CHECK_CUBLAS(expr, label) KNNDC_CUBLAS_CHECK_GOTO((expr), cleanup)

__device__ __forceinline__ unsigned long long pack_dist_idx(float d2, int idx)
{
    unsigned int d_bits = __float_as_uint(d2);
    unsigned int i_bits = (unsigned int)idx;  
    return ((unsigned long long)d_bits << 32) | (unsigned long long)i_bits;
}

__device__ __forceinline__ float unpack_dist(unsigned long long p)
{
    return __uint_as_float((unsigned int)(p >> 32));
}

__device__ __forceinline__ int unpack_idx(unsigned long long p)
{
    return (int)(unsigned int)(p & 0xFFFFFFFFu);
}

__global__ void dc_cb_col_norms(const float *matrix,
                                int cols,
                                int dim,
                                int row_stride,
                                float *norms)
{
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (unsigned int)cols) return;
    float s = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float v = matrix[(size_t)d * row_stride + c];
        s = fmaf(v, v, s);
    }
    norms[c] = s;
}

__global__ void dc_cb_init_packed(unsigned long long *packed, int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)n) return;
    packed[i] = pack_dist_idx(FLT_MAX, -1);
}

__global__ void dc_cb_init_int(int *arr, int n, int val)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)n) return;
    arr[i] = val;
}

__global__ void dc_cb_stageA_reducer(const float *dot,
                                     const float *ds_den_full,
                                     const float *ds_norm_full,
                                     int ds_start,
                                     int ds_count,
                                     const float *core_den_tile,
                                     const float *core_norm_tile,
                                     int current_tile,
                                     unsigned long long *packed_tile)
{
    int lane = threadIdx.x;             
    int warp = threadIdx.y;             
    int core_local = blockIdx.x * blockDim.y + warp;
    if (core_local >= current_tile) return;

    float cden  = core_den_tile[core_local];
    float cnorm = core_norm_tile[core_local];

    float best_d2 = FLT_MAX;
    int   best_gi = -1;

    for (int li = lane; li < ds_count; li += 32) {
        int gi = ds_start + li;
        if (cden < ds_den_full[gi]) {
            float dv  = dot[(size_t)li * current_tile + core_local];
            float d2  = ds_norm_full[gi] + cnorm - 2.0f * dv;
            if (d2 < 0.0f) d2 = 0.0f;
            if (d2 < best_d2 || (d2 == best_d2 && (best_gi < 0 || gi < best_gi))) {
                best_d2 = d2;
                best_gi = gi;
            }
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float od2 = __shfl_down_sync(0xFFFFFFFFu, best_d2, off);
        int   ogi = __shfl_down_sync(0xFFFFFFFFu, best_gi, off);
        if (od2 < best_d2 || (od2 == best_d2 && ogi >= 0 && (best_gi < 0 || ogi < best_gi))) {
            best_d2 = od2;
            best_gi = ogi;
        }
    }

    if (lane == 0 && best_gi >= 0) {
        unsigned long long cand = pack_dist_idx(best_d2, best_gi);
        atomicMin(&packed_tile[core_local], cand);
    }
}

__global__ void dc_cb_split_packed(const unsigned long long *packed,
                                   int n,
                                   float *cdh_d2,
                                   int *cdh_index) 
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)n) return;
    unsigned long long p = packed[i];
    int idx = unpack_idx(p);
    if (idx < 0) {
        cdh_d2[i] = -1.0f;
        cdh_index[i] = -1;
    } else {
        cdh_d2[i] = unpack_dist(p);
        cdh_index[i] = idx;
    }
}

__global__ void dc_cb_stageB_reducer(const float *dot,
                                     const float *ds_norm_full,
                                     int ds_start,
                                     int ds_count,
                                     const float *core_norm_tile,
                                     const float *cdh_d2_tile,
                                     int current_tile,
                                     int *rank_tile)
{
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int core_local = blockIdx.x * blockDim.y + warp;
    if (core_local >= current_tile) return;

    float d2_cdh = cdh_d2_tile[core_local];
    if (d2_cdh < 0.0f) {
        
        return;
    }

    float cnorm = core_norm_tile[core_local];
    int local_cnt = 0;

    for (int li = lane; li < ds_count; li += 32) {
        int gi = ds_start + li;
        float dv = dot[(size_t)li * current_tile + core_local];
        float d2 = ds_norm_full[gi] + cnorm - 2.0f * dv;
        if (d2 < 0.0f) d2 = 0.0f;
        if (d2 < d2_cdh) ++local_cnt;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_cnt += __shfl_down_sync(0xFFFFFFFFu, local_cnt, off);
    }

    if (lane == 0 && local_cnt > 0) {
        atomicAdd(&rank_tile[core_local], local_cnt);
    }
}

__global__ void dc_cb_finalize(const float *cdh_d2,
                               int n,
                               float *cdh_dist,
                               int *cdh_rank)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)n) return;
    float d2 = cdh_d2[i];
    if (d2 < 0.0f) {
        cdh_dist[i] = -1.0f;
        cdh_rank[i] = -1;
    } else {
        cdh_dist[i] = sqrtf(d2);
        if (cdh_rank[i] == 0) cdh_rank[i] = -1;
    }
}

static bool dc_reduction_cuda(Dataset     &ds,
                              const float *ds_den_host,
                              const float *core_host,
                              const float *core_den_host,
                              int          core_nb,
                              int         *cdh_index_host,
                              float       *cdh_dist_dev_out,
                              int         *cdh_rank_dev_out)
{
    if (cdh_dist_dev_out == NULL || cdh_rank_dev_out == NULL)
        return false;

    if (!select_cuda_device(ds.cuda_device_id()))
        return false;

    if (!ds.EnsureUploaded())
        return false;

    const int    ds_nb        = ds.num_points();
    const int    dim          = ds.dim();
    const float *ds_dev       = ds.device_data();
    const int    ds_ld_floats = (int)(ds.device_pitch_bytes() / sizeof(float));

    bool ok = false;
    
    int core_tile = DC_CUBLAS_CORE_TILE;
    if (core_tile > core_nb) core_tile = core_nb;

    int ds_block = DC_CUBLAS_DS_BLOCK;
    if (ds_block > ds_nb) ds_block = ds_nb;

    const int warps_per_block = DC_CUBLAS_WARPS_PER_BLOCK;
    const dim3 reducer_block(32, warps_per_block, 1);
    const dim3 reducer_grid((core_tile + warps_per_block - 1) / warps_per_block, 1, 1);
    const int  finalize_bdim = 256;
    
    float *ds_den_dev      = NULL;
    float *core_dev        = NULL;
    float *core_den_dev    = NULL;
    float *ds_norm_dev     = NULL;
    float *core_norm_dev   = NULL;
    float *cdh_d2_dev      = cdh_dist_dev_out;
    int   *cdh_index_dev   = NULL;
    int   *cdh_rank_dev    = cdh_rank_dev_out;
    unsigned long long *cdh_packed_dev = NULL;
    float *dot_buf[2]      = { NULL, NULL };
    
    cudaStream_t streams[2] = { 0, 0 };
    cublasHandle_t handles[2] = { NULL, NULL };
    bool streams_created[2] = { false, false };
    bool handles_created[2] = { false, false };

    const size_t f = sizeof(float);
    const size_t ds_den_bytes   = (size_t)ds_nb * f;
    const size_t core_bytes     = (size_t)core_nb * dim * f;
    const size_t core_den_bytes = (size_t)core_nb * f;
    const size_t core_i_bytes   = (size_t)core_nb * sizeof(int);
    const size_t ds_norm_bytes  = (size_t)ds_nb * f;
    const size_t core_norm_bytes = (size_t)core_nb * f;
    
    const float one  = 1.0f;
    const float zero = 0.0f;
    
    const bool dc_use_tf32 = (USE_CUBLAS_TF32 == 1) && (dim >= TF32_MIN_DIM);
    const cublasMath_t dc_math_mode = dc_use_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH
                                                  : CUBLAS_DEFAULT_MATH;
    int slab_counter = 0;
    int num_core_tiles = (core_nb + core_tile - 1) / core_tile;
    int num_ds_blocks  = (ds_nb  + ds_block  - 1) / ds_block;
    const bool interleave_streams = (DC_CUBLAS_INTERLEAVE_STREAMS == 1);
    
    bool single_slab_fast_path = false;
    if (DC_CUBLAS_SINGLE_SLAB_MAX_MB > 0) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);

        size_t fixed_b = ds_den_bytes + core_bytes + core_den_bytes
                       + ds_norm_bytes + core_norm_bytes
                       + core_i_bytes
                       + (size_t)core_nb * sizeof(unsigned long long);

        const size_t reserve_b = (size_t)DC_CUBLAS_MEM_RESERVE_MB * 1024ull * 1024ull;
        size_t avail_b = (free_b > fixed_b + reserve_b)
                           ? (free_b - fixed_b - reserve_b) : 0;
        avail_b = (avail_b / 100) * 85;             
        size_t per_buf_budget = avail_b / 2;        
        const size_t hard_cap = (size_t)DC_CUBLAS_SINGLE_SLAB_MAX_MB
                                * 1024ull * 1024ull;
        if (per_buf_budget > hard_cap) per_buf_budget = hard_cap;

        const size_t per_core_b = (size_t)ds_nb * f;
        if (per_core_b > 0 && per_buf_budget >= per_core_b) {
            int max_ct = (int)(per_buf_budget / per_core_b);
            if (max_ct > core_nb) max_ct = core_nb;

            int min_ct = DC_CUBLAS_MIN_CORE_TILE;
            if (min_ct > core_nb) min_ct = core_nb;

            int planned_ct = (max_ct < core_tile) ? max_ct : core_tile;
            if (planned_ct >= min_ct) {
                core_tile = planned_ct;
                ds_block = ds_nb;
                num_core_tiles = (core_nb + core_tile - 1) / core_tile;
                num_ds_blocks  = 1;
                single_slab_fast_path = true;
            }
        }
    }
    
    if (!single_slab_fast_path && num_ds_blocks == 1) {
        single_slab_fast_path = true;
    }

    const size_t dot_buf_bytes  = (size_t)core_tile * (size_t)ds_block * f;

    DC_CB_CHECK_CUDA(cudaMalloc((void**)&ds_den_dev,    ds_den_bytes),    "alloc ds_den_dev");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&core_dev,      core_bytes),      "alloc core_dev");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&core_den_dev,  core_den_bytes),  "alloc core_den_dev");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&ds_norm_dev,   ds_norm_bytes),   "alloc ds_norm_dev");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&core_norm_dev, core_norm_bytes), "alloc core_norm_dev");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&cdh_index_dev, core_i_bytes),    "alloc cdh_index_dev");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&cdh_packed_dev, (size_t)core_nb * sizeof(unsigned long long)), "alloc cdh_packed_dev");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&dot_buf[0], dot_buf_bytes), "alloc dot_buf[0]");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&dot_buf[1], dot_buf_bytes), "alloc dot_buf[1]");

    DC_CB_CHECK_CUDA(cudaMemcpy(ds_den_dev,   ds_den_host,   ds_den_bytes,   cudaMemcpyHostToDevice), "copy ds_den H2D");
    DC_CB_CHECK_CUDA(cudaMemcpy(core_dev,     core_host,     core_bytes,     cudaMemcpyHostToDevice), "copy core H2D");
    DC_CB_CHECK_CUDA(cudaMemcpy(core_den_dev, core_den_host, core_den_bytes, cudaMemcpyHostToDevice), "copy core_den H2D");


    DC_CB_CHECK_CUDA(cudaStreamCreate(&streams[0]), "create stream0");
    streams_created[0] = true;
    DC_CB_CHECK_CUDA(cudaStreamCreate(&streams[1]), "create stream1");
    streams_created[1] = true;
    DC_CB_CHECK_CUBLAS(cublasCreate(&handles[0]), "create handle0");
    handles_created[0] = true;
    DC_CB_CHECK_CUBLAS(cublasCreate(&handles[1]), "create handle1");
    handles_created[1] = true;
    DC_CB_CHECK_CUBLAS(cublasSetStream(handles[0], streams[0]), "set stream0");
    DC_CB_CHECK_CUBLAS(cublasSetStream(handles[1], streams[1]), "set stream1");
    
    DC_CB_CHECK_CUBLAS(cublasSetMathMode(handles[0], dc_math_mode), "set math0");
    DC_CB_CHECK_CUBLAS(cublasSetMathMode(handles[1], dc_math_mode), "set math1");

    {
        int g = (ds_nb + finalize_bdim - 1) / finalize_bdim;
        dc_cb_col_norms<<<g, finalize_bdim, 0, streams[0]>>>(
            ds_dev, ds_nb, dim, ds_ld_floats, ds_norm_dev);
        DC_CB_CHECK_LAST("dc_cb_col_norms(ds)");
    }
    
    {
        int g = (core_nb + finalize_bdim - 1) / finalize_bdim;
        dc_cb_col_norms<<<g, finalize_bdim, 0, streams[0]>>>(
            core_dev, core_nb, dim, core_nb, core_norm_dev);
        DC_CB_CHECK_LAST("dc_cb_col_norms(core)");
    }
    
    DC_CB_CHECK_CUDA(cudaStreamSynchronize(streams[0]), "sync after norms");

    {
        int g_init = (core_nb + finalize_bdim - 1) / finalize_bdim;
        dc_cb_init_packed<<<g_init, finalize_bdim, 0, streams[0]>>>(cdh_packed_dev, core_nb);
        DC_CB_CHECK_LAST("dc_cb_init_packed");
        DC_CB_CHECK_CUDA(cudaStreamSynchronize(streams[0]), "sync after init_packed");
    }

    if (single_slab_fast_path) 
    {
        {
            int g = (core_nb + finalize_bdim - 1) / finalize_bdim;
            dc_cb_init_int<<<g, finalize_bdim, 0, streams[0]>>>(cdh_rank_dev, core_nb, 0);
            DC_CB_CHECK_LAST("dc_cb_init_int(rank)");
            DC_CB_CHECK_CUDA(cudaStreamSynchronize(streams[0]), "sync after init_rank");
        }

        for (int t = 0; t < num_core_tiles; ++t) {
            int core_start = t * core_tile;
            int cur_tile = core_nb - core_start;
            if (cur_tile > core_tile) cur_tile = core_tile;

            int s = interleave_streams ? (t & 1) : 0;
            dim3 r_grid((cur_tile + warps_per_block - 1) / warps_per_block, 1, 1);

            DC_CB_CHECK_CUBLAS(cublasSgemm(handles[s],
                                           CUBLAS_OP_N, CUBLAS_OP_T,
                                           cur_tile, ds_nb, dim,
                                           &one,
                                           core_dev + core_start, core_nb,
                                           ds_dev, ds_ld_floats,
                                           &zero,
                                           dot_buf[s], cur_tile), "fastpath sgemm");

            dc_cb_stageA_reducer<<<r_grid, reducer_block, 0, streams[s]>>>(
                dot_buf[s],
                ds_den_dev,
                ds_norm_dev,
                0, ds_nb,
                core_den_dev + core_start,
                core_norm_dev + core_start,
                cur_tile,
                cdh_packed_dev + core_start);
            DC_CB_CHECK_LAST("fastpath stageA_reducer");

            {
                int g = (cur_tile + finalize_bdim - 1) / finalize_bdim;
                dc_cb_split_packed<<<g, finalize_bdim, 0, streams[s]>>>(
                    cdh_packed_dev + core_start,
                    cur_tile,
                    cdh_d2_dev + core_start,
                    cdh_index_dev + core_start);
                DC_CB_CHECK_LAST("fastpath split_packed");
            }

            dc_cb_stageB_reducer<<<r_grid, reducer_block, 0, streams[s]>>>(
                dot_buf[s],
                ds_norm_dev,
                0, ds_nb,
                core_norm_dev + core_start,
                cdh_d2_dev + core_start,
                cur_tile,
                cdh_rank_dev + core_start);
            DC_CB_CHECK_LAST("fastpath stageB_reducer");
        }

        DC_CB_CHECK_CUDA(cudaStreamSynchronize(streams[0]), "sync stream0 fast path");
        DC_CB_CHECK_CUDA(cudaStreamSynchronize(streams[1]), "sync stream1 fast path");
    } 
    else 
    {
        for (int t = 0; t < num_core_tiles; ++t) 
        {
            int core_start = t * core_tile;
            int cur_tile = core_nb - core_start;
            if (cur_tile > core_tile) cur_tile = core_tile;

            dim3 r_grid((cur_tile + warps_per_block - 1) / warps_per_block, 1, 1);

            for (int b = 0; b < num_ds_blocks; ++b, ++slab_counter) {
                int ds_start = b * ds_block;
                int cur_ds = ds_nb - ds_start;
                if (cur_ds > ds_block) cur_ds = ds_block;

                int s = interleave_streams ? (slab_counter & 1) : 0;

                DC_CB_CHECK_CUBLAS(cublasSgemm(handles[s],
                                            CUBLAS_OP_N, CUBLAS_OP_T,
                                            cur_tile, cur_ds, dim,
                                            &one,
                                            core_dev + core_start, core_nb,
                                            ds_dev + ds_start, ds_ld_floats,
                                            &zero,
                                            dot_buf[s], cur_tile), "stageA sgemm");

                dc_cb_stageA_reducer<<<r_grid, reducer_block, 0, streams[s]>>>(
                    dot_buf[s],
                    ds_den_dev,
                    ds_norm_dev,
                    ds_start, cur_ds,
                    core_den_dev + core_start,
                    core_norm_dev + core_start,
                    cur_tile,
                    cdh_packed_dev + core_start);
                DC_CB_CHECK_LAST("dc_cb_stageA_reducer");
            }
        }

        DC_CB_CHECK_CUDA(cudaStreamSynchronize(streams[0]), "sync stream0 end stageA");
        DC_CB_CHECK_CUDA(cudaStreamSynchronize(streams[1]), "sync stream1 end stageA");

        {
            int g = (core_nb + finalize_bdim - 1) / finalize_bdim;
            dc_cb_split_packed<<<g, finalize_bdim, 0, streams[0]>>>(
                cdh_packed_dev, core_nb, cdh_d2_dev, cdh_index_dev);
            DC_CB_CHECK_LAST("dc_cb_split_packed");
        }

        {
            int g = (core_nb + finalize_bdim - 1) / finalize_bdim;
            dc_cb_init_int<<<g, finalize_bdim, 0, streams[0]>>>(cdh_rank_dev, core_nb, 0);
            DC_CB_CHECK_LAST("dc_cb_init_int(rank)");
            DC_CB_CHECK_CUDA(cudaStreamSynchronize(streams[0]), "sync after init_rank");
        }

        slab_counter = 0;
        for (int t = 0; t < num_core_tiles; ++t) 
        {
            int core_start = t * core_tile;
            int cur_tile = core_nb - core_start;
            if (cur_tile > core_tile) cur_tile = core_tile;

            dim3 r_grid((cur_tile + warps_per_block - 1) / warps_per_block, 1, 1);

            for (int b = 0; b < num_ds_blocks; ++b, ++slab_counter) 
            {
                int ds_start = b * ds_block;
                int cur_ds = ds_nb - ds_start;
                if (cur_ds > ds_block) cur_ds = ds_block;

                int s = interleave_streams ? (slab_counter & 1) : 0;

                DC_CB_CHECK_CUBLAS(cublasSgemm(handles[s],
                                            CUBLAS_OP_N, CUBLAS_OP_T,
                                            cur_tile, cur_ds, dim,
                                            &one,
                                            core_dev + core_start, core_nb,
                                            ds_dev + ds_start, ds_ld_floats,
                                            &zero,
                                            dot_buf[s], cur_tile), "stageB sgemm");

                dc_cb_stageB_reducer<<<r_grid, reducer_block, 0, streams[s]>>>(
                    dot_buf[s],
                    ds_norm_dev,
                    ds_start, cur_ds,
                    core_norm_dev + core_start,
                    cdh_d2_dev + core_start,
                    cur_tile,
                    cdh_rank_dev + core_start);
                DC_CB_CHECK_LAST("dc_cb_stageB_reducer");
            }
        }

        DC_CB_CHECK_CUDA(cudaStreamSynchronize(streams[0]), "sync stream0 end stageB");
        DC_CB_CHECK_CUDA(cudaStreamSynchronize(streams[1]), "sync stream1 end stageB");
    }  

    {
        int g = (core_nb + finalize_bdim - 1) / finalize_bdim;
        
        dc_cb_finalize<<<g, finalize_bdim, 0, streams[0]>>>(
            cdh_d2_dev, core_nb, cdh_d2_dev, cdh_rank_dev);
        DC_CB_CHECK_LAST("dc_cb_finalize");
    }
    DC_CB_CHECK_CUDA(cudaStreamSynchronize(streams[0]), "sync finalize");

    DC_CB_CHECK_CUDA(cudaMemcpy(cdh_index_host, cdh_index_dev, core_i_bytes, cudaMemcpyDeviceToHost), "copy cdh_index D2H");

    ok = true;

cleanup:
    if (handles_created[0]) cublasDestroy(handles[0]);
    if (handles_created[1]) cublasDestroy(handles[1]);
    if (streams_created[0]) cudaStreamDestroy(streams[0]);
    if (streams_created[1]) cudaStreamDestroy(streams[1]);

    if (dot_buf[0])      cudaFree(dot_buf[0]);
    if (dot_buf[1])      cudaFree(dot_buf[1]);
    if (cdh_packed_dev)  cudaFree(cdh_packed_dev);
    if (cdh_index_dev)   cudaFree(cdh_index_dev);
    if (core_norm_dev)   cudaFree(core_norm_dev);
    if (ds_norm_dev)     cudaFree(ds_norm_dev);
    if (core_den_dev)    cudaFree(core_den_dev);
    if (core_dev)        cudaFree(core_dev);
    if (ds_den_dev)      cudaFree(ds_den_dev);
    
    return ok;
}


__global__ void dc_select_centers_kernel(const float *core_den,
                                         const float *g_in,
                                         const int   *k_th_in,
                                         const int   *core_indices,
                                         int          core_nb,
                                         int          num_centers,
                                         int         *center_indices)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (core_nb <= 0 || num_centers <= 0) return;

    float g_fill = -1.0f;
    int k_fill = -1;
    for (int i = 0; i < core_nb; ++i) {
        if (g_in[i] > g_fill) g_fill = g_in[i];
        if (k_th_in[i] > k_fill) k_fill = k_th_in[i];
    }

    float min_den = FLT_MAX, max_den = -FLT_MAX;
    float min_g = FLT_MAX, max_g = -FLT_MAX;
    float min_k = FLT_MAX, max_k = -FLT_MAX;
    for (int i = 0; i < core_nb; ++i) {
        float den = core_den[i];
        float g = (g_in[i] < 0.0f) ? g_fill : g_in[i];
        float k = (float)((k_th_in[i] < 0) ? k_fill : k_th_in[i]);

        if (den < min_den) min_den = den;
        if (den > max_den) max_den = den;
        if (g < min_g) min_g = g;
        if (g > max_g) max_g = g;
        if (k < min_k) min_k = k;
        if (k > max_k) max_k = k;
    }

    const float den_range = max_den - min_den;
    const float g_range = max_g - min_g;
    const float k_range = max_k - min_k;

    const int select_n = (num_centers < core_nb) ? num_centers : core_nb;
    for (int sel = 0; sel < select_n; ++sel) {
        float best_sd = -FLT_MAX;
        int best_idx = -1;
        for (int i = 0; i < core_nb; ++i) {
            bool chosen = false;
            for (int p = 0; p < sel; ++p) {
                if (center_indices[p] == core_indices[i]) {
                    chosen = true;
                    break;
                }
            }
            if (chosen) continue;

            float den = core_den[i];
            float g = (g_in[i] < 0.0f) ? g_fill : g_in[i];
            float k = (float)((k_th_in[i] < 0) ? k_fill : k_th_in[i]);
            float den_n = (den_range == 0.0f) ? 1.0f : (den - min_den) / den_range;
            float g_n = (g_range == 0.0f) ? 1.0f : (g - min_g) / g_range;
            float k_n = (k_range == 0.0f) ? 1.0f : (k - min_k) / k_range;
            float sd = den_n * g_n * k_n;

            if (sd > best_sd || (sd == best_sd && (best_idx < 0 || i < best_idx))) {
                best_sd = sd;
                best_idx = i;
            }
        }
        center_indices[sel] = (best_idx >= 0) ? core_indices[best_idx] : -1;
    }

    for (int sel = select_n; sel < num_centers; ++sel) {
        center_indices[sel] = -1;
    }
}

bool warp_parallel_dc_reduction_cuda(Dataset     &ds,
                                     const float *ds_den_host,
                                     const int   *core_indices_host,
                                     int          core_nb,
                                     int          num_centers,
                                     int         *core_cdh_index_host,
                                     int         *center_indices_host)
{
    const float* data_host = ds.host_data();
    const int dim = ds.dim();
    const int num_points = ds.num_points();
    if (core_nb <= 0 || num_centers <= 0 || data_host == NULL || ds_den_host == NULL ||
        core_indices_host == NULL || core_cdh_index_host == NULL || center_indices_host == NULL ||
        dim <= 0 || num_points <= 0) {
        return false;
    }

    std::vector<float> core_host((size_t)core_nb * dim);
    std::vector<float> core_den_host(core_nb);
    for (int i = 0; i < core_nb; ++i) {
        const int core_idx = core_indices_host[i];
        if (core_idx < 0 || core_idx >= num_points) {
            return false;
        }
        for (int d = 0; d < dim; ++d) {
            core_host[(size_t)d * core_nb + i] = data_host[(size_t)d * num_points + core_idx];
        }
        core_den_host[i] = ds_den_host[core_idx];
    }

    float *core_den_dev = NULL;
    float *g_dev = NULL;
    int *k_dev = NULL;
    int *core_idx_dev = NULL;
    int *center_idx_dev = NULL;
    bool ok = false;

    const size_t core_f_bytes = (size_t)core_nb * sizeof(float);
    const size_t core_i_bytes = (size_t)core_nb * sizeof(int);
    const size_t center_i_bytes = (size_t)num_centers * sizeof(int);

    DC_CB_CHECK_CUDA(cudaMalloc((void**)&core_den_dev, core_f_bytes), "alloc core_den_dev");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&g_dev, core_f_bytes), "alloc g_dev");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&k_dev, core_i_bytes), "alloc k_dev");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&core_idx_dev, core_i_bytes), "alloc core_idx_dev");
    DC_CB_CHECK_CUDA(cudaMalloc((void**)&center_idx_dev, center_i_bytes), "alloc center_idx_dev");

    if (!dc_reduction_cuda(ds,
                           ds_den_host,
                           core_host.data(),
                           core_den_host.data(),
                           core_nb,
                           core_cdh_index_host,
                           g_dev,
                           k_dev)) {
        goto cleanup;
    }

    DC_CB_CHECK_CUDA(cudaMemcpy(core_den_dev, core_den_host.data(), core_f_bytes, cudaMemcpyHostToDevice),
                     "copy core_den H2D");
    DC_CB_CHECK_CUDA(cudaMemcpy(core_idx_dev, core_indices_host, core_i_bytes, cudaMemcpyHostToDevice),
                     "copy core_idx H2D");

    dc_select_centers_kernel<<<1, 1>>>(core_den_dev, g_dev, k_dev, core_idx_dev,
                                       core_nb, num_centers, center_idx_dev);
    DC_CB_CHECK_LAST("dc_select_centers_kernel");
    DC_CB_CHECK_CUDA(cudaDeviceSynchronize(), "sync dc_select_centers_kernel");
    DC_CB_CHECK_CUDA(cudaMemcpy(center_indices_host, center_idx_dev, center_i_bytes, cudaMemcpyDeviceToHost),
                     "copy center_idx D2H");

    ok = true;

cleanup:
    if (core_den_dev) cudaFree(core_den_dev);
    if (g_dev) cudaFree(g_dev);
    if (k_dev) cudaFree(k_dev);
    if (core_idx_dev) cudaFree(core_idx_dev);
    if (center_idx_dev) cudaFree(center_idx_dev);
    return ok;
}
