#pragma once

#include <cub/block/block_radix_sort.cuh>

__global__ void skn_sort_topk(int N, int K, float* out_dist, int* out_idx)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= N) return;

    float* d = out_dist + (size_t)q * K;
    int* i = out_idx + (size_t)q * K;

    for (int a = 1; a < K; ++a) {
        float kd = d[a];
        int ki = i[a];
        int j = a - 1;

        while (j >= 0 && (d[j] > kd || (d[j] == kd && i[j] > ki))) {
            d[j + 1] = d[j];
            i[j + 1] = i[j];
            --j;
        }
        d[j + 1] = kd;
        i[j + 1] = ki;
    }
}

template<int ItemsPerThread>
__global__ void skn_radix_sort_topk(int neighbors, float* out_dist, int* out_idx)
{
    typedef cub::BlockRadixSort<uint64_t, 128, ItemsPerThread> BlockSort;
    __shared__ typename BlockSort::TempStorage storage;
    uint64_t keys[ItemsPerThread];
    const size_t base = (size_t)blockIdx.x * neighbors;

    for (int item = 0; item < ItemsPerThread; ++item) {
        const int rank = threadIdx.x * ItemsPerThread + item;
        keys[item] = UINT64_MAX;
        if (rank < neighbors) {
            const uint32_t distance_bits = __float_as_uint(out_dist[base + rank]);
            const uint32_t index_key = (uint32_t)out_idx[base + rank] ^ 0x80000000u;
            keys[item] = ((uint64_t)distance_bits << 32) | index_key;
        }
    }

    BlockSort(storage).Sort(keys);

    for (int item = 0; item < ItemsPerThread; ++item) {
        const int rank = threadIdx.x * ItemsPerThread + item;
        if (rank < neighbors) {
            out_dist[base + rank] = __uint_as_float((uint32_t)(keys[item] >> 32));
            out_idx[base + rank] = (int)((uint32_t)keys[item] ^ 0x80000000u);
        }
    }
}

static void skn_launch_sort_topk(int num_points, int neighbors, float* out_dist, int* out_idx)
{
    if (neighbors <= 32 || neighbors > 4096) {
        skn_sort_topk<<<(num_points + 127) / 128, 128>>>(
            num_points, neighbors, out_dist, out_idx);
    } else if (neighbors <= 128) {
        skn_radix_sort_topk<1><<<num_points, 128>>>(neighbors, out_dist, out_idx);
    } else if (neighbors <= 256) {
        skn_radix_sort_topk<2><<<num_points, 128>>>(neighbors, out_dist, out_idx);
    } else if (neighbors <= 512) {
        skn_radix_sort_topk<4><<<num_points, 128>>>(neighbors, out_dist, out_idx);
    } else if (neighbors <= 1024) {
        skn_radix_sort_topk<8><<<num_points, 128>>>(neighbors, out_dist, out_idx);
    } else if (neighbors <= 2048) {
        skn_radix_sort_topk<16><<<num_points, 128>>>(neighbors, out_dist, out_idx);
    } else {
        skn_radix_sort_topk<32><<<num_points, 128>>>(neighbors, out_dist, out_idx);
    }
}
