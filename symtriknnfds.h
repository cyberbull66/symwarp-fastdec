#pragma once

#ifndef SKN_FDS_MAX_DIM
#define SKN_FDS_MAX_DIM 128
#endif
#if SKN_FDS_MAX_DIM < 0
#error "SKN_FDS_MAX_DIM must be >= 0 (0 disables the fused path)"
#endif

#ifndef SKN_FDS_REFERENCE
#define SKN_FDS_REFERENCE 0
#endif
#if SKN_FDS_REFERENCE != 0 && SKN_FDS_REFERENCE != 1
#error "SKN_FDS_REFERENCE must be 0 or 1"
#endif

#define SKN_FDS_TILE 128
#define SKN_FDS_DK 8
#define SKN_FDS_THREADS 256
#define SKN_FDS_MICRO 8
#define SKN_FDS_WORDS (SKN_FDS_TILE / 32)

__device__ __forceinline__ float skn_fds_finish(float norm_a, float norm_b, float dot)
{
    return fmaxf(__fadd_rn(__fadd_rn(norm_a, norm_b), __fmul_rn(-2.0f, dot)), 0.0f);
}

__device__ __forceinline__ int skn_fds_row(int t, int i)
{
    return (i >> 2) * (SKN_FDS_TILE / 2) + t * 4 + (i & 3);
}

template <bool kWriteTile>
__global__ void __launch_bounds__(SKN_FDS_THREADS)
skn_fds_tile_kernel(const float* __restrict__ data, int pitch, int dim,
                    const float* __restrict__ norms,
                    int q_start, int qb, int r_start, int rb,
                    const float* __restrict__ roots_q, const float* __restrict__ roots_r,
                    unsigned* __restrict__ fwd_bitmap, unsigned* __restrict__ rev_bitmap, int stride,
                    float* __restrict__ tile_out)
{
    __shared__ __align__(16) float s_q[2][SKN_FDS_DK][SKN_FDS_TILE];
    __shared__ __align__(16) float s_r[2][SKN_FDS_DK][SKN_FDS_TILE];
    __shared__ float s_norm_q[SKN_FDS_TILE], s_norm_r[SKN_FDS_TILE];
    __shared__ float s_root_q[SKN_FDS_TILE], s_root_r[SKN_FDS_TILE];
    __shared__ unsigned s_fwd[SKN_FDS_TILE][SKN_FDS_WORDS];
    __shared__ unsigned s_rev[SKN_FDS_TILE][SKN_FDS_WORDS];

    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;
    const int r_off = blockIdx.x * SKN_FDS_TILE;
    const int q_off = blockIdx.y * SKN_FDS_TILE;

    for (int p = tid; p < SKN_FDS_TILE; p += SKN_FDS_THREADS) {
        const int q = q_off + p, r = r_off + p;
        s_norm_q[p] = (q < qb) ? norms[q_start + q] : 0.0f;
        s_norm_r[p] = (r < rb) ? norms[r_start + r] : 0.0f;
        if constexpr (!kWriteTile) {
            s_root_q[p] = (q < qb) ? roots_q[q] : 0.0f;
            s_root_r[p] = (r < rb) ? roots_r[r] : 0.0f;
        }
    }
    if constexpr (!kWriteTile) {
        for (int w = tid; w < SKN_FDS_TILE * SKN_FDS_WORDS; w += SKN_FDS_THREADS) {
            (&s_fwd[0][0])[w] = 0u;
            (&s_rev[0][0])[w] = 0u;
        }
    }

    float acc[SKN_FDS_MICRO][SKN_FDS_MICRO];
#pragma unroll
    for (int i = 0; i < SKN_FDS_MICRO; ++i)
#pragma unroll
        for (int j = 0; j < SKN_FDS_MICRO; ++j) acc[i][j] = 0.0f;

    constexpr int kLoads = SKN_FDS_DK * SKN_FDS_TILE / SKN_FDS_THREADS;
    float pre_q[kLoads], pre_r[kLoads];
    auto fetch = [&](int d0) {
#pragma unroll
        for (int t = 0; t < kLoads; ++t) {
            const int e = tid + t * SKN_FDS_THREADS;
            const int kk = e / SKN_FDS_TILE, p = e % SKN_FDS_TILE, d = d0 + kk;
            const int q = q_off + p, r = r_off + p;
            pre_q[t] = (d < dim && q < qb) ? data[(size_t)d * pitch + q_start + q] : 0.0f;
            pre_r[t] = (d < dim && r < rb) ? data[(size_t)d * pitch + r_start + r] : 0.0f;
        }
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int t = 0; t < kLoads; ++t) {
            const int e = tid + t * SKN_FDS_THREADS;
            s_q[buf][e / SKN_FDS_TILE][e % SKN_FDS_TILE] = pre_q[t];
            s_r[buf][e / SKN_FDS_TILE][e % SKN_FDS_TILE] = pre_r[t];
        }
    };
    const int chunks = (dim + SKN_FDS_DK - 1) / SKN_FDS_DK;
    fetch(0);
    stash(0);
    __syncthreads();
    for (int c = 0; c < chunks; ++c) {
        const int buf = c & 1;
        if (c + 1 < chunks) fetch((c + 1) * SKN_FDS_DK);
        const int steps = min(SKN_FDS_DK, dim - c * SKN_FDS_DK);
#pragma unroll
        for (int kk = 0; kk < SKN_FDS_DK; ++kk) {
            if (kk < steps) {
                const float4 a0 = *reinterpret_cast<const float4*>(&s_q[buf][kk][ty * 4]);
                const float4 a1 = *reinterpret_cast<const float4*>(&s_q[buf][kk][SKN_FDS_TILE / 2 + ty * 4]);
                const float4 b0 = *reinterpret_cast<const float4*>(&s_r[buf][kk][tx * 4]);
                const float4 b1 = *reinterpret_cast<const float4*>(&s_r[buf][kk][SKN_FDS_TILE / 2 + tx * 4]);
                const float a[SKN_FDS_MICRO] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const float b[SKN_FDS_MICRO] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < SKN_FDS_MICRO; ++i)
#pragma unroll
                    for (int j = 0; j < SKN_FDS_MICRO; ++j) acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
            }
        }
        if (c + 1 < chunks) stash(buf ^ 1);
        __syncthreads();
    }

    if constexpr (kWriteTile) {
#pragma unroll
        for (int i = 0; i < SKN_FDS_MICRO; ++i) {
            const int q = q_off + skn_fds_row(ty, i);
#pragma unroll
            for (int j = 0; j < SKN_FDS_MICRO; ++j) {
                const int r = r_off + skn_fds_row(tx, j);
                if (q < qb && r < rb) tile_out[(size_t)r + (size_t)q * rb] = __fmul_rn(-2.0f, acc[i][j]);
            }
        }
    } else {
        float norm_r[SKN_FDS_MICRO], root_r[SKN_FDS_MICRO];
#pragma unroll
        for (int j = 0; j < SKN_FDS_MICRO; ++j) {
            norm_r[j] = s_norm_r[skn_fds_row(tx, j)];
            root_r[j] = s_root_r[skn_fds_row(tx, j)];
        }
#pragma unroll
        for (int i = 0; i < SKN_FDS_MICRO; ++i) {
            const int qi = skn_fds_row(ty, i);
            if (q_off + qi >= qb) continue;
            const float norm_q = s_norm_q[qi], root_q = s_root_q[qi];
#pragma unroll
            for (int j = 0; j < SKN_FDS_MICRO; ++j) {
                const int rj = skn_fds_row(tx, j);
                if (r_off + rj >= rb) continue;
                const float dist = skn_fds_finish(norm_q, norm_r[j], acc[i][j]);
                const bool fwd = dist < root_q, rev = dist < root_r[j];
                if (fwd) atomicOr(&s_fwd[qi][rj >> 5], 1u << (rj & 31));
                if (rev) atomicOr(&s_rev[rj][qi >> 5], 1u << (qi & 31));
                if (fwd || rev) tile_out[(size_t)(r_off + rj) + (size_t)(q_off + qi) * rb] = dist;
            }
        }
        __syncthreads();
        const int words_r = (rb + 31) / 32, words_q = (qb + 31) / 32;
        for (int w = tid; w < SKN_FDS_TILE * SKN_FDS_WORDS; w += SKN_FDS_THREADS) {
            const int row = w / SKN_FDS_WORDS, col = w % SKN_FDS_WORDS;
            const int q = q_off + row, word_r = r_off / 32 + col;
            if (q < qb && word_r < words_r) fwd_bitmap[(size_t)q * stride + word_r] = s_fwd[row][col];
            const int r = r_off + row, word_q = q_off / 32 + col;
            if (r < rb && word_q < words_q) rev_bitmap[(size_t)r * stride + word_q] = s_rev[row][col];
        }
    }
}

__device__ __forceinline__ void skn_fds_replay_point(int p_local, int self_start, int n_self,
                                                     int other_start, int n_other, int K,
                                                     float* __restrict__ out_dist, int* __restrict__ out_idx,
                                                     const unsigned* __restrict__ bitmap, int stride,
                                                     const float* __restrict__ survivors, int rb,
                                                     bool self_is_query
                                                     )
{
    const int num_warps = blockDim.x / SKN_WARP;
    const int warp_id = threadIdx.x / SKN_WARP;
    const int lane_id = threadIdx.x % SKN_WARP;
    if (p_local >= n_self) return;
    const int p = self_start + p_local;
    float* hd = out_dist + (size_t)p * K;
    int* hi = out_idx + (size_t)p * K;
    extern __shared__ char smem[];
    float* s_d = (float*)smem;
    int* s_i = (int*)&s_d[(size_t)num_warps * K];
    float* heap_d = s_d + (size_t)warp_id * K;
    int* heap_i = s_i + (size_t)warp_id * K;

    const unsigned* row = bitmap + (size_t)p_local * stride;
    const int words = (n_other + 31) / 32;
    bool loaded = false;
    float worst = FLT_MAX;

    for (int word_base = 0; word_base < words; word_base += SKN_WARP) {
        const int word = word_base + lane_id;
        const unsigned bits = (word < words) ? row[word] : 0u;
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
            const int o_local = (word_base + source_lane) * SKN_WARP + lane_id;
            const int o = other_start + o_local;
            const bool marked = (group_bits >> lane_id) & 1u;
            float dist = FLT_MAX;
            if (marked) {
                dist = self_is_query ? survivors[(size_t)o_local + (size_t)p_local * rb]
                                     : survivors[(size_t)p_local + (size_t)o_local * rb];
            }
            const bool valid = marked && (dist < worst);
            unsigned cand = __ballot_sync(0xffffffff, valid);
            while (cand != 0) {
                const int l = __ffs(cand) - 1;
                const float ld = __shfl_sync(0xffffffff, dist, l);
                const int li = __shfl_sync(0xffffffff, o, l);
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

__global__ void skn_fds_replay_both(int q_start, int qb, int r_start, int rb, int K,
                                    float* __restrict__ out_dist, int* __restrict__ out_idx,
                                    const unsigned* __restrict__ fwd_bitmap,
                                    const unsigned* __restrict__ rev_bitmap, int stride,
                                    const float* __restrict__ survivors, int q_blocks)
{
    const int warps = blockDim.x / SKN_WARP;
    const int warp_id = threadIdx.x / SKN_WARP;
    if ((int)blockIdx.x < q_blocks) {
        skn_fds_replay_point(blockIdx.x * warps + warp_id, q_start, qb, r_start, rb, K,
                             out_dist, out_idx, fwd_bitmap, stride, survivors, rb, true
                             );
    } else {
        skn_fds_replay_point((blockIdx.x - q_blocks) * warps + warp_id, r_start, rb, q_start, qb, K,
                             out_dist, out_idx, rev_bitmap, stride, survivors, rb, false
                             );
    }
}
