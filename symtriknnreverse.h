#pragma once

#ifndef SKN_COALESCED_TRANSPOSE_MIN_K
#define SKN_COALESCED_TRANSPOSE_MIN_K 128
#endif

#ifndef SKN_FUSED_REVERSE
#define SKN_FUSED_REVERSE 1
#endif
#if SKN_FUSED_REVERSE != 0 && SKN_FUSED_REVERSE != 1
#error "SKN_FUSED_REVERSE must be 0 or 1"
#endif

__global__ void skn_update_topk_fused_forward(const float* __restrict__ dot_block,
                                              int rb,
                                              int qb,
                                              int q_start,
                                              int r_start,
                                              int K,
                                              const float* __restrict__ norms_sq,
                                              float* __restrict__ out_dist,
                                              int* __restrict__ out_idx,
                                              const float* __restrict__ rev_roots,
                                              unsigned* __restrict__ rev_bitmap,
                                              int rev_stride
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
    const unsigned q_bit = 1u << (q_local & 31);
    unsigned* bitmap_column = rev_bitmap + (q_local >> 5);

    for (int r_base = 0; r_base < rb; r_base += SKN_WARP) {
        int r_local = r_base + lane_id;
        int r_global = r_start + r_local;

        float dist = FLT_MAX;
        if (r_local < rb) {
            float neg2dot = dot_block[(size_t)r_local + (size_t)q_local * rb];
            dist = norm_q + norms_sq[r_global] + neg2dot;
            dist = fmaxf(dist, 0.0f);
            if (dist < __ldg(&rev_roots[r_local])) {
                atomicOr(bitmap_column + (size_t)r_local * rev_stride, q_bit);
            }
        }

        bool valid = (r_local < rb) && (dist < worst);
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

__global__ void skn_update_topk_from_reverse_bitmap(const float* __restrict__ dot_block,
                                                    int rb,
                                                    int qb,
                                                    int r_start,
                                                    int q_start,
                                                    int K,
                                                    const float* __restrict__ norms_sq,
                                                    float* __restrict__ out_dist,
                                                    int* __restrict__ out_idx,
                                                    unsigned* __restrict__ rev_bitmap,
                                                    int rev_stride
                                                    )
{
    const int num_warps = blockDim.x / SKN_WARP;
    const int warp_id = threadIdx.x / SKN_WARP;
    const int lane_id = threadIdx.x % SKN_WARP;
    const int r_local = blockIdx.x * num_warps + warp_id;
    if (r_local >= rb) return;

    const int r_global = r_start + r_local;
    float* hd = out_dist + (size_t)r_global * K;
    int* hi = out_idx + (size_t)r_global * K;

    extern __shared__ char smem[];
    float* s_d = (float*)smem;
    int* s_i = (int*)&s_d[(size_t)num_warps * K];
    float* heap_d = s_d + (size_t)warp_id * K;
    int* heap_i = s_i + (size_t)warp_id * K;

    unsigned* bitmap_row = rev_bitmap + (size_t)r_local * rev_stride;
    const int words = (qb + 31) / 32;
    const float norm_r = norms_sq[r_global];
    bool loaded = false;
    float worst = FLT_MAX;

    for (int word_base = 0; word_base < words; word_base += SKN_WARP) {
        const int word = word_base + lane_id;
        unsigned bits = (word < words) ? bitmap_row[word] : 0u;
        if (bits != 0u) bitmap_row[word] = 0u;
        unsigned words_with_bits = __ballot_sync(0xffffffff, bits != 0u);

        while (words_with_bits != 0) {
            const int source_lane = __ffs(words_with_bits) - 1;
            const unsigned group_bits = __shfl_sync(0xffffffff, bits, source_lane);
            if (!loaded) {
                for (int i = lane_id; i < K; i += SKN_WARP) {
                    heap_d[i] = hd[i];
                    heap_i[i] = hi[i];
                }
                __syncwarp();
                worst = (lane_id == 0) ? heap_d[0] : FLT_MAX;
                worst = __shfl_sync(0xffffffff, worst, 0);
                loaded = true;
            }

            const int q_local = (word_base + source_lane) * SKN_WARP + lane_id;
            const int q_global = q_start + q_local;
            const bool marked = (group_bits >> lane_id) & 1u;
            float dist = FLT_MAX;
            if (marked) {
                float neg2dot = dot_block[(size_t)r_local + (size_t)q_local * rb];
                dist = norm_r + norms_sq[q_global] + neg2dot;
                dist = fmaxf(dist, 0.0f);
            }

            bool valid = marked && (dist < worst);
            unsigned cand = __ballot_sync(0xffffffff, valid);

            while (cand != 0) {
                int l = __ffs(cand) - 1;
                float ld = __shfl_sync(0xffffffff, dist, l);
                int li = __shfl_sync(0xffffffff, q_global, l);
                if (lane_id == 0) {
                    skn_heap_insert(heap_d, heap_i, K, ld, li
                                    );
                    worst = heap_d[0];
                }
                worst = __shfl_sync(0xffffffff, worst, 0);
                cand &= (cand - 1);
            }
            words_with_bits &= (words_with_bits - 1);
        }
    }

    if (loaded) {
        __syncwarp();
        for (int i = lane_id; i < K; i += SKN_WARP) {
            hd[i] = heap_d[i];
            hi[i] = heap_i[i];
        }
    }
}

__global__ void skn_transpose_dot_tile(const float* source, float* destination,
                                       int rows, int columns)
{
    __shared__ float tile[32][33];
    const int source_row = blockIdx.x * 32 + threadIdx.x;
    const int source_column = blockIdx.y * 32 + threadIdx.y;
    for (int offset = 0; offset < 32; offset += 8) {
        if (source_row < rows && source_column + offset < columns) {
            tile[threadIdx.y + offset][threadIdx.x] =
                source[(size_t)(source_column + offset) * rows + source_row];
        }
    }
    __syncthreads();
    const int destination_row = blockIdx.y * 32 + threadIdx.x;
    const int destination_column = blockIdx.x * 32 + threadIdx.y;
    for (int offset = 0; offset < 32; offset += 8) {
        if (destination_row < columns && destination_column + offset < rows) {
            destination[(size_t)(destination_column + offset) * columns + destination_row] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

__global__ void skn_update_topk_from_dot_block_transpose(const float* __restrict__ dot_block,
                                                         int rb,
                                                         int qb,
                                                         int r_start,
                                                         int q_start,
                                                         int K,
                                                         const float* __restrict__ norms_sq,
                                                         float* __restrict__ out_dist,
                                                         int* __restrict__ out_idx
                                                         )
{
    const int num_warps = blockDim.x / SKN_WARP;
    const int warp_id = threadIdx.x / SKN_WARP;
    const int lane_id = threadIdx.x % SKN_WARP;
    const int r_local = blockIdx.x * num_warps + warp_id;
    if (r_local >= rb) return;

    const int r_global = r_start + r_local;
    float* hd = out_dist + (size_t)r_global * K;
    int* hi = out_idx + (size_t)r_global * K;

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

    float norm_r = norms_sq[r_global];
    float worst = (lane_id == 0) ? heap_d[0] : FLT_MAX;
    worst = __shfl_sync(0xffffffff, worst, 0);

    for (int q_base = 0; q_base < qb; q_base += SKN_WARP) {
        int q_local = q_base + lane_id;
        int q_global = q_start + q_local;

        float dist = FLT_MAX;
        if (q_local < qb) {
            float neg2dot = dot_block[(size_t)r_local + (size_t)q_local * rb];
            dist = norm_r + norms_sq[q_global] + neg2dot;
            dist = fmaxf(dist, 0.0f);
        }

        bool valid = (q_local < qb) && (q_global != r_global) && (dist < worst);
        unsigned cand = __ballot_sync(0xffffffff, valid);

        while (cand != 0) {
            int l = __ffs(cand) - 1;
            float ld = __shfl_sync(0xffffffff, dist, l);
            int li = __shfl_sync(0xffffffff, q_global, l);
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
