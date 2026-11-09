#include "symtriknncuda.cu"
#include "dataset.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        fprintf(stderr, "Sort test failed during %s: %s (CUDA code %d). "
                        "Check GPU availability and launch limits before rerunning.\n",
                operation, cudaGetErrorString(status), (int)status);
        exit(1);
    }
}

static bool check_sort(int neighbors)
{
    const int num_points = 13;
    const size_t count = (size_t)num_points * neighbors;
    std::vector<float> distances(count);
    std::vector<int> indices(count);
    std::vector<std::pair<float, int>> expected(count);
    std::vector<std::pair<float, int>> row(neighbors);
    std::mt19937 generator(42);

    for (int point = 0; point < num_points; ++point) {
        for (int rank = 0; rank < neighbors; ++rank) {
            const float distance = point == 0 ? 0.0f : (float)(generator() % 17);
            row[rank] = std::make_pair(distance, rank);
        }
        if (point == 1) row.back() = std::make_pair(FLT_MAX, -1);
        if (point == 2) {
            for (int rank = 0; rank < neighbors; ++rank) {
                row[rank] = std::make_pair(FLT_MAX, rank);
            }
            row.back().second = -1;
        }
        std::make_heap(row.begin(), row.end());
        for (int rank = 0; rank < neighbors; ++rank) {
            const size_t offset = (size_t)point * neighbors + rank;
            distances[offset] = row[rank].first;
            indices[offset] = row[rank].second;
        }
        std::sort(row.begin(), row.end());
        std::copy(row.begin(), row.end(), expected.begin() + (size_t)point * neighbors);
    }

    float* device_distances = NULL;
    int* device_indices = NULL;
    check_cuda(cudaMalloc((void**)&device_distances, count * sizeof(float)), "distance allocation");
    check_cuda(cudaMalloc((void**)&device_indices, count * sizeof(int)), "index allocation");
    check_cuda(cudaMemcpy(device_distances, distances.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice), "distance upload");
    check_cuda(cudaMemcpy(device_indices, indices.data(), count * sizeof(int),
                          cudaMemcpyHostToDevice), "index upload");
    skn_launch_sort_topk(num_points, neighbors, device_distances, device_indices);
    check_cuda(cudaGetLastError(), "sort launch");
    check_cuda(cudaDeviceSynchronize(), "sort execution");
    check_cuda(cudaMemcpy(distances.data(), device_distances, count * sizeof(float),
                          cudaMemcpyDeviceToHost), "distance download");
    check_cuda(cudaMemcpy(indices.data(), device_indices, count * sizeof(int),
                          cudaMemcpyDeviceToHost), "index download");
    check_cuda(cudaFree(device_distances), "distance cleanup");
    check_cuda(cudaFree(device_indices), "index cleanup");

    for (size_t offset = 0; offset < count; ++offset) {
        if (distances[offset] != expected[offset].first || indices[offset] != expected[offset].second) {
            fprintf(stderr, "Sort output mismatch at k=%d, offset=%zu: "
                            "expected (%g, %d), got (%g, %d). "
                            "Inspect radix keys and dispatch before using this build.\n",
                    neighbors, offset, expected[offset].first, expected[offset].second,
                    distances[offset], indices[offset]);
            return false;
        }
    }
    printf("PASS k=%d: CPU-reference ordering, ties, padding, and index pairing\n", neighbors);
    return true;
}

static bool check_reverse_selection(int rows, int columns, int neighbors, int warps, bool diagonal)
{
    const int num_points = 2 * columns + rows + 3;
    const int reference_start = diagonal ? 0 : 2 * columns + 3;
    const size_t tile_count = (size_t)rows * columns;
    const size_t heap_count = (size_t)num_points * neighbors;
    std::vector<float> source(tile_count), transposed(tile_count), norms(num_points);
    std::mt19937 generator(123);
    for (float& value : source) value = (float)((int)(generator() % 127) - 8);
    for (float& value : norms) value = (float)(generator() % 31);
    float *device_source = NULL, *device_transposed = NULL, *device_norms = NULL;
    std::vector<float> expected_distances(heap_count, FLT_MAX);
    std::vector<int> expected_indices(heap_count, -1);
#if SKN_HEAP_WORK_COUNTERS
    SknHeapWork expected_work{};
    expected_work.query_updates = 2ULL * rows;
    SknHeapWork measured_work[4]{};
    check_cuda(cudaMemcpyToSymbol(skn_heap_work, measured_work, sizeof(measured_work)), "counter reset");
#endif
    for (int pass = 0; pass < 2; ++pass) {
        for (int row = 0; row < rows; ++row) {
            const int point = reference_start + row;
            const size_t base = (size_t)point * neighbors;
#if SKN_HEAP_WORK_COUNTERS
            float filter_threshold = FLT_MAX;
#endif
            for (int column = 0; column < columns; ++column) {
                const int candidate = pass * columns + column;
                const float distance = std::max(0.0f, norms[point] + norms[candidate] +
                                                     source[(size_t)column * rows + row]);
#if SKN_HEAP_WORK_COUNTERS
                if (column % 32 == 0) filter_threshold = expected_distances[base];
                if (candidate != point && distance < filter_threshold) ++expected_work.filtered;
#endif
                if (candidate == point || distance >= expected_distances[base]) continue;
#if SKN_HEAP_WORK_COUNTERS
                ++expected_work.admitted;
                const bool filling = expected_indices[base] == -1;
                expected_work.filling_admitted += filling;
#endif
                expected_distances[base] = distance;
                expected_indices[base] = candidate;
                int parent = 0;
                while (true) {
                    int largest = parent;
                    for (int child : {2 * parent + 1, 2 * parent + 2}) {
                        if (child < neighbors &&
                            std::make_pair(expected_distances[base + child], expected_indices[base + child]) >
                            std::make_pair(expected_distances[base + largest], expected_indices[base + largest])) {
                            largest = child;
                        }
                    }
                    if (largest == parent) break;
#if SKN_HEAP_WORK_COUNTERS
                    ++expected_work.sift_levels;
                    expected_work.filling_sift_levels += filling;
#endif
                    std::swap(expected_distances[base + parent], expected_distances[base + largest]);
                    std::swap(expected_indices[base + parent], expected_indices[base + largest]);
                    parent = largest;
                }
            }
        }
    }
    float* device_distances[2] = {NULL, NULL};
    int* device_indices[2] = {NULL, NULL};
    check_cuda(cudaMalloc((void**)&device_source, tile_count * sizeof(float)), "tile allocation");
    check_cuda(cudaMalloc((void**)&device_transposed, tile_count * sizeof(float)), "transpose allocation");
    check_cuda(cudaMalloc((void**)&device_norms, num_points * sizeof(float)), "norm allocation");
    check_cuda(cudaMemcpy(device_source, source.data(), tile_count * sizeof(float), cudaMemcpyHostToDevice), "tile upload");
    check_cuda(cudaMemcpy(device_norms, norms.data(), num_points * sizeof(float), cudaMemcpyHostToDevice), "norm upload");
    skn_transpose_dot_tile<<<dim3((rows + 31) / 32, (columns + 31) / 32), dim3(32, 8)>>>(
        device_source, device_transposed, rows, columns);
    check_cuda(cudaGetLastError(), "tile transpose launch");
    check_cuda(cudaMemcpy(transposed.data(), device_transposed, tile_count * sizeof(float), cudaMemcpyDeviceToHost), "transpose download");
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < rows; ++row) {
            if (source[(size_t)column * rows + row] != transposed[(size_t)row * columns + column]) {
                fprintf(stderr, "Tile transpose mismatch at row=%d column=%d. Inspect edge predicates before rerunning.\n", row, column);
                return false;
            }
        }
    }
    const size_t shared_bytes = skn_selector_shared_bytes(warps, neighbors);
    for (int method = 0; method < 2; ++method) {
        check_cuda(cudaMalloc((void**)&device_distances[method], heap_count * sizeof(float)), "heap distance allocation");
        check_cuda(cudaMalloc((void**)&device_indices[method], heap_count * sizeof(int)), "heap index allocation");
        skn_init_topk<<<128, 256>>>(num_points, neighbors, device_distances[method], device_indices[method]);
        check_cuda(cudaGetLastError(), "heap initialization");
        for (int pass = 0; pass < 2; ++pass) {
            if (method == 0) {
                skn_update_topk_from_dot_block_transpose<<<(rows + warps - 1) / warps, warps * 32, shared_bytes>>>(
                    device_source, rows, columns, reference_start, pass * columns, neighbors,
                    device_norms, device_distances[method], device_indices[method]
#if SKN_HEAP_WORK_COUNTERS
                    , method
#endif
                    );
            } else {
                skn_update_topk_from_dot_block<<<(rows + warps - 1) / warps, warps * 32, shared_bytes>>>(
                    device_transposed, columns, rows, reference_start, pass * columns, neighbors,
                    device_norms, device_distances[method], device_indices[method]
#if SKN_HEAP_WORK_COUNTERS
                    , method
#endif
                    );
            }
            check_cuda(cudaGetLastError(), "reverse selection launch");
        }
    }
    bool matches = true;
#if SKN_HEAP_WORK_COUNTERS
    check_cuda(cudaMemcpyFromSymbol(measured_work, skn_heap_work, sizeof(measured_work)), "counter download");
    for (int method = 0; method < 2; ++method) {
        const SknHeapWork& actual = measured_work[method];
        if (actual.query_updates != expected_work.query_updates ||
            actual.filtered != expected_work.filtered || actual.admitted != expected_work.admitted ||
            actual.sift_levels != expected_work.sift_levels ||
            actual.filling_admitted != expected_work.filling_admitted ||
            actual.filling_sift_levels != expected_work.filling_sift_levels) {
            fprintf(stderr, "Heap counters differ from the CPU reference at k=%d method=%d. "
                            "Work measurements are invalid; inspect ballot thresholds and sift counting.\n",
                    neighbors, method);
            matches = false;
        }
    }
#endif
    std::vector<float> distances(heap_count);
    std::vector<int> indices(heap_count);
    for (int method = 0; method < 2; ++method) {
        check_cuda(cudaMemcpy(distances.data(), device_distances[method], heap_count * sizeof(float), cudaMemcpyDeviceToHost), "heap distance download");
        check_cuda(cudaMemcpy(indices.data(), device_indices[method], heap_count * sizeof(int), cudaMemcpyDeviceToHost), "heap index download");
        if (distances != expected_distances || indices != expected_indices) {
            fprintf(stderr, "Selector differs from serial heap reference: method=%d diagonal=%d "
                            "rows=%d columns=%d k=%d warps=%d. "
                            "Do not use this build; inspect admission and heap update order.\n",
                    method, diagonal, rows, columns, neighbors, warps);
            matches = false;
        }
        check_cuda(cudaFree(device_distances[method]), "heap distance cleanup");
        check_cuda(cudaFree(device_indices[method]), "heap index cleanup");
    }
    check_cuda(cudaFree(device_source), "tile cleanup");
    check_cuda(cudaFree(device_transposed), "transpose cleanup");
    check_cuda(cudaFree(device_norms), "norm cleanup");
    if (matches) {
         printf("PASS selector rows=%d columns=%d k=%d warps=%d diagonal=%d: "
             "both orientations match serial heap reference\n",
             rows, columns, neighbors, warps, diagonal);
    }
    return matches;
}

// Single-read reverse selection: for two off-diagonal tiles in sequence, compares the original
// forward + strided reverse pass (method 0) with the fused forward + bitmap reverse pass, used
// for the second tile only (method 1, heaps filled by the first tile) or for both tiles
// (method 2, starting from empty heaps). All heap buffers must match byte for byte, and the
// bitmap must be all zero afterwards.
static bool check_fused_reverse(int rows, int columns, int neighbors, int warps)
{
    const int query_start = 0;
    const int reference_start = columns + 3;
    const int num_points = reference_start + rows;
    const size_t tile_count = (size_t)rows * columns;
    const size_t heap_count = (size_t)num_points * neighbors;
    const int bitmap_stride = (columns + 31) / 32 + 1;
    const size_t bitmap_count = (size_t)rows * bitmap_stride;
    std::mt19937 generator(321 + rows + 7 * columns + 13 * neighbors);
    std::vector<float> tiles[2] = {std::vector<float>(tile_count), std::vector<float>(tile_count)};
    std::vector<float> norms(num_points);
    for (auto& tile : tiles) {
        for (float& value : tile) value = (float)((int)(generator() % 127) - 8);
    }
    for (float& value : norms) value = (float)(generator() % 31);

    float *device_tiles[2] = {NULL, NULL}, *device_norms = NULL, *device_roots = NULL;
    unsigned* device_bitmap = NULL;
    for (int tile = 0; tile < 2; ++tile) {
        check_cuda(cudaMalloc((void**)&device_tiles[tile], tile_count * sizeof(float)), "fused tile allocation");
        check_cuda(cudaMemcpy(device_tiles[tile], tiles[tile].data(), tile_count * sizeof(float),
                              cudaMemcpyHostToDevice), "fused tile upload");
    }
    check_cuda(cudaMalloc((void**)&device_norms, num_points * sizeof(float)), "fused norm allocation");
    check_cuda(cudaMemcpy(device_norms, norms.data(), num_points * sizeof(float), cudaMemcpyHostToDevice),
               "fused norm upload");
    check_cuda(cudaMalloc((void**)&device_roots, rows * sizeof(float)), "fused root allocation");
    check_cuda(cudaMalloc((void**)&device_bitmap, bitmap_count * sizeof(unsigned)), "fused bitmap allocation");
    check_cuda(cudaMemset(device_bitmap, 0, bitmap_count * sizeof(unsigned)), "fused bitmap reset");

    const size_t shared_bytes = skn_selector_shared_bytes(warps, neighbors);
    const int threads = warps * 32;
    std::vector<float> distances[3];
    std::vector<int> indices[3];
    for (int method = 0; method < 3; ++method) {
        float* device_distances = NULL;
        int* device_indices = NULL;
        check_cuda(cudaMalloc((void**)&device_distances, heap_count * sizeof(float)), "fused heap distance allocation");
        check_cuda(cudaMalloc((void**)&device_indices, heap_count * sizeof(int)), "fused heap index allocation");
        skn_init_topk<<<128, 256>>>(num_points, neighbors, device_distances, device_indices);
        check_cuda(cudaGetLastError(), "fused heap initialization");
        for (int tile = 0; tile < 2; ++tile) {
            const bool fused = (method == 2) || (method == 1 && tile == 1);
            if (fused) {
                skn_gather_heap_roots<<<(rows + 255) / 256, 256>>>(
                    device_distances, neighbors, reference_start, rows, device_roots);
                skn_update_topk_fused_forward<<<(columns + warps - 1) / warps, threads, shared_bytes>>>(
                    device_tiles[tile], rows, columns, query_start, reference_start, neighbors,
                    device_norms, device_distances, device_indices,
                    device_roots, device_bitmap, bitmap_stride);
                skn_update_topk_from_reverse_bitmap<<<(rows + warps - 1) / warps, threads, shared_bytes>>>(
                    device_tiles[tile], rows, columns, reference_start, query_start, neighbors,
                    device_norms, device_distances, device_indices,
                    device_bitmap, bitmap_stride);
            } else {
                skn_update_topk_from_dot_block<<<(columns + warps - 1) / warps, threads, shared_bytes>>>(
                    device_tiles[tile], rows, columns, query_start, reference_start, neighbors,
                    device_norms, device_distances, device_indices);
                skn_update_topk_from_dot_block_transpose<<<(rows + warps - 1) / warps, threads, shared_bytes>>>(
                    device_tiles[tile], rows, columns, reference_start, query_start, neighbors,
                    device_norms, device_distances, device_indices);
            }
            check_cuda(cudaGetLastError(), "fused selection launch");
        }
        distances[method].resize(heap_count);
        indices[method].resize(heap_count);
        check_cuda(cudaMemcpy(distances[method].data(), device_distances, heap_count * sizeof(float),
                              cudaMemcpyDeviceToHost), "fused heap distance download");
        check_cuda(cudaMemcpy(indices[method].data(), device_indices, heap_count * sizeof(int),
                              cudaMemcpyDeviceToHost), "fused heap index download");
        check_cuda(cudaFree(device_distances), "fused heap distance cleanup");
        check_cuda(cudaFree(device_indices), "fused heap index cleanup");
    }
    std::vector<unsigned> bitmap(bitmap_count);
    check_cuda(cudaMemcpy(bitmap.data(), device_bitmap, bitmap_count * sizeof(unsigned), cudaMemcpyDeviceToHost),
               "fused bitmap download");
    for (int tile = 0; tile < 2; ++tile) check_cuda(cudaFree(device_tiles[tile]), "fused tile cleanup");
    check_cuda(cudaFree(device_norms), "fused norm cleanup");
    check_cuda(cudaFree(device_roots), "fused root cleanup");
    check_cuda(cudaFree(device_bitmap), "fused bitmap cleanup");

    bool matches = true;
    for (int method = 1; method < 3; ++method) {
        if (memcmp(distances[method].data(), distances[0].data(), heap_count * sizeof(float)) != 0 ||
            indices[method] != indices[0]) {
            fprintf(stderr, "Single-read reverse selection differs from the original passes: method=%d "
                            "rows=%d columns=%d k=%d warps=%d. Do not use this build; inspect the "
                            "bitmap filter and candidate order.\n",
                    method, rows, columns, neighbors, warps);
            matches = false;
        }
    }
    for (unsigned word : bitmap) {
        if (word != 0u) {
            fprintf(stderr, "Reverse bitmap not cleared after use: rows=%d columns=%d k=%d warps=%d. "
                            "Later tiles would read stale survivors.\n", rows, columns, neighbors, warps);
            matches = false;
            break;
        }
    }
    if (matches) {
        printf("PASS fused reverse rows=%d columns=%d k=%d warps=%d: "
               "prefilled and empty-heap runs match the original passes\n",
               rows, columns, neighbors, warps);
    }
    return matches;
}

#if !SKN_HEAP_WORK_COUNTERS
// Fused distance selection. Points: Q = [0, qb), R1 (77 points) and R2 (rb points), dim-major with
// small-integer coordinates (many exact distance ties, including zero distances between distinct
// points). The test tile (Q, R2) runs either through the reference path (dense fused-distance tile
// + the -ors forward and strided reverse selectors) or through the fused path (root gather, filter,
// replay of both directions). With prefill, the tiles (Q, R1) and (R1, R2) run through the reference
// path first, so both sides of the test tile start with partly filled heaps; without it the heaps
// start empty. All heap buffers must match byte for byte. For one k per shape the dense tile is also
// checked against a host std::fma reference, element by element.
static bool check_fds_tile(int qb, int rb, int dim, int neighbors, int warps, bool prefill, bool check_tile)
{
    const int r1 = 77;
    const int q_start = 0, r1_start = qb + 3, r2_start = r1_start + r1 + 2;
    const int num_points = r2_start + rb;
    const size_t heap_count = (size_t)num_points * neighbors;
    std::mt19937 generator(1000 + qb + 7 * rb + 13 * dim + 31 * neighbors + (prefill ? 1 : 0));
    std::vector<float> data((size_t)dim * num_points);   // data[d * num_points + i]
    for (float& v : data) v = (float)(generator() % 6);

    float *device_data = NULL, *device_norms = NULL, *device_tile = NULL, *device_roots_q = NULL, *device_roots_r = NULL;
    unsigned *device_fwd = NULL, *device_rev = NULL;
    const int stride = (std::max(qb, rb) + 31) / 32 + 1;   // deliberately wider than the used words
    const size_t tile_count = (size_t)std::max(qb, r1) * std::max(rb, r1);
    check_cuda(cudaMalloc((void**)&device_data, data.size() * sizeof(float)), "fds data allocation");
    check_cuda(cudaMemcpy(device_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice), "fds data upload");
    check_cuda(cudaMalloc((void**)&device_norms, num_points * sizeof(float)), "fds norm allocation");
    check_cuda(cudaMalloc((void**)&device_tile, tile_count * sizeof(float)), "fds tile allocation");
    check_cuda(cudaMalloc((void**)&device_roots_q, num_points * sizeof(float)), "fds root allocation");
    check_cuda(cudaMalloc((void**)&device_roots_r, num_points * sizeof(float)), "fds root allocation");
    check_cuda(cudaMalloc((void**)&device_fwd, (size_t)std::max(qb, rb) * stride * sizeof(unsigned)), "fds bitmap allocation");
    check_cuda(cudaMalloc((void**)&device_rev, (size_t)std::max(qb, rb) * stride * sizeof(unsigned)), "fds bitmap allocation");
    skn_compute_norms<<<64, 256>>>(device_data, num_points, dim, num_points, device_norms);
    check_cuda(cudaGetLastError(), "fds norms");

    const size_t shared_bytes = skn_selector_shared_bytes(warps, neighbors);
    const int threads = warps * 32;
    float* device_distances = NULL;
    int* device_indices = NULL;
    check_cuda(cudaMalloc((void**)&device_distances, heap_count * sizeof(float)), "fds heap allocation");
    check_cuda(cudaMalloc((void**)&device_indices, heap_count * sizeof(int)), "fds heap allocation");

    auto reference_tile = [&](int qs, int qn, int rs, int rn) {
        dim3 grid((rn + SKN_FDS_TILE - 1) / SKN_FDS_TILE, (qn + SKN_FDS_TILE - 1) / SKN_FDS_TILE);
        skn_fds_tile_kernel<true><<<grid, SKN_FDS_THREADS>>>(device_data, num_points, dim, device_norms,
            qs, qn, rs, rn, NULL, NULL, NULL, NULL, 0, device_tile);
        skn_update_topk_from_dot_block<<<(qn + warps - 1) / warps, threads, shared_bytes>>>(
            device_tile, rn, qn, qs, rs, neighbors, device_norms, device_distances, device_indices);
        skn_update_topk_from_dot_block_transpose<<<(rn + warps - 1) / warps, threads, shared_bytes>>>(
            device_tile, rn, qn, rs, qs, neighbors, device_norms, device_distances, device_indices);
        check_cuda(cudaGetLastError(), "fds reference tile");
    };
    auto fused_tile = [&](int qs, int qn, int rs, int rn) {
        skn_gather_heap_roots<<<(qn + 255) / 256, 256>>>(device_distances, neighbors, qs, qn, device_roots_q);
        skn_gather_heap_roots<<<(rn + 255) / 256, 256>>>(device_distances, neighbors, rs, rn, device_roots_r);
        dim3 grid((rn + SKN_FDS_TILE - 1) / SKN_FDS_TILE, (qn + SKN_FDS_TILE - 1) / SKN_FDS_TILE);
        skn_fds_tile_kernel<false><<<grid, SKN_FDS_THREADS>>>(device_data, num_points, dim, device_norms,
            qs, qn, rs, rn, device_roots_q, device_roots_r, device_fwd, device_rev, stride, device_tile);
        const int q_blocks = (qn + warps - 1) / warps, r_blocks = (rn + warps - 1) / warps;
        skn_fds_replay_both<<<q_blocks + r_blocks, threads, shared_bytes>>>(qs, qn, rs, rn, neighbors,
            device_distances, device_indices, device_fwd, device_rev, stride, device_tile, q_blocks);
        check_cuda(cudaGetLastError(), "fds fused tile");
    };

    std::vector<float> distances[2];
    std::vector<int> indices[2];
    long long tile_mismatch = 0;
    for (int method = 0; method < 2; ++method) {
        skn_init_topk<<<128, 256>>>(num_points, neighbors, device_distances, device_indices);
        check_cuda(cudaGetLastError(), "fds heap initialization");
        if (prefill) {
            reference_tile(q_start, qb, r1_start, r1);
            reference_tile(r1_start, r1, r2_start, rb);
        }
        if (method == 0) reference_tile(q_start, qb, r2_start, rb);
        else fused_tile(q_start, qb, r2_start, rb);
        distances[method].resize(heap_count);
        indices[method].resize(heap_count);
        check_cuda(cudaMemcpy(distances[method].data(), device_distances, heap_count * sizeof(float),
                              cudaMemcpyDeviceToHost), "fds heap download");
        check_cuda(cudaMemcpy(indices[method].data(), device_indices, heap_count * sizeof(int),
                              cudaMemcpyDeviceToHost), "fds heap download");
        if (method == 0 && check_tile) {
            std::vector<float> tile((size_t)qb * rb);
            check_cuda(cudaMemcpy(tile.data(), device_tile, tile.size() * sizeof(float), cudaMemcpyDeviceToHost),
                       "fds tile download");
            for (int q = 0; q < qb; ++q) {
                for (int r = 0; r < rb; ++r) {
                    float dot = 0.0f;
                    for (int d = 0; d < dim; ++d)
                        dot = std::fma(data[(size_t)d * num_points + q_start + q], data[(size_t)d * num_points + r2_start + r], dot);
                    const float expected = -2.0f * dot;
                    if (memcmp(&expected, &tile[(size_t)r + (size_t)q * rb], sizeof(float)) != 0) ++tile_mismatch;
                }
            }
        }
    }
    for (void* p : {(void*)device_data, (void*)device_norms, (void*)device_tile, (void*)device_roots_q,
                    (void*)device_roots_r, (void*)device_fwd, (void*)device_rev, (void*)device_distances,
                    (void*)device_indices}) {
        check_cuda(cudaFree(p), "fds cleanup");
    }

    bool ok = true;
    if (memcmp(distances[0].data(), distances[1].data(), heap_count * sizeof(float)) != 0 || indices[0] != indices[1]) {
        fprintf(stderr, "Fused distance selection differs from its dense-tile reference: qb=%d rb=%d dim=%d k=%d "
                        "warps=%d prefill=%d. Do not use this build; inspect the filter thresholds, the "
                        "survivor values and the replay order.\n", qb, rb, dim, neighbors, warps, prefill);
        ok = false;
    }
    if (tile_mismatch != 0) {
        fprintf(stderr, "Fused distance tile differs from the host std::fma reference in %lld of %lld elements: "
                        "qb=%d rb=%d dim=%d. The FMA order or rounding of skn_fds_tile_kernel changed.\n",
                tile_mismatch, (long long)qb * rb, qb, rb, dim);
        ok = false;
    }
    if (ok) {
        printf("PASS fds qb=%d rb=%d dim=%d k=%d warps=%d prefill=%d%s: fused path matches its dense-tile reference\n",
               qb, rb, dim, neighbors, warps, prefill, check_tile ? " (tile checked against std::fma)" : "");
    }
    return ok;
}
#endif

static int dump_knn(const char* input_path, const char* output_path, int num_points, int dimensions)
{
    if (num_points <= 768 || num_points > 100000 || dimensions <= 0 || dimensions > 4096) {
        fprintf(stderr, "kNN export rejected the sample shape; no comparison was produced. "
                        "Use 769..100000 points and 1..4096 dimensions.\n");
        return 1;
    }
    const size_t count = (size_t)num_points * dimensions;
    std::vector<float> row_major(count), data(count);
    std::ifstream input(input_path, std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(row_major.data()), count * sizeof(float)) ||
        input.peek() != std::char_traits<char>::eof()) {
        fprintf(stderr, "kNN export could not read the exact input matrix; comparison aborted. "
                        "Regenerate the row-major float32 sample for the requested shape.\n");
        return 1;
    }
    for (int point = 0; point < num_points; ++point) {
        for (int dimension = 0; dimension < dimensions; ++dimension) {
            data[(size_t)dimension * num_points + point] = row_major[(size_t)point * dimensions + dimension];
        }
    }
    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        fprintf(stderr, "kNN export could not open its output; comparison aborted. "
                        "Check output-directory permissions and rerun.\n");
        return 1;
    }
    Dataset dataset;
    dataset.Bind(data.data(), num_points, dimensions, 0);
    for (int neighbors : {8, 32, 64, 128, 256, 512, 768}) {
        const size_t output_count = (size_t)num_points * neighbors;
        std::vector<float> distances(output_count);
        std::vector<int> indices(output_count);
        if (!symtri_knn_cuda(dataset, neighbors, distances.data(), indices.data())) {
            fprintf(stderr, "kNN export failed at k=%d; output is incomplete. "
                            "Inspect the preceding CUDA diagnostics and rerun.\n", neighbors);
            return 1;
        }
        output.write(reinterpret_cast<const char*>(distances.data()), output_count * sizeof(float));
        output.write(reinterpret_cast<const char*>(indices.data()), output_count * sizeof(int));
        if (!output) {
            fprintf(stderr, "kNN export write failed at k=%d; output is incomplete. "
                            "Check free disk space and rerun.\n", neighbors);
            return 1;
        }
        printf("EXPORTED N=%d D=%d k=%d: complete rank-major distance/index buffers\n",
               num_points, dimensions, neighbors);
    }
    output.close();
    if (!output) {
        fprintf(stderr, "kNN export could not finish writing; output is incomplete. "
                        "Check free disk space and rerun.\n");
        return 1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 1) {
        if (argc == 6 && std::string(argv[1]) == "--dump-knn") {
            try {
                size_t points_end = 0, dimensions_end = 0;
                const int num_points = std::stoi(argv[4], &points_end);
                const int dimensions = std::stoi(argv[5], &dimensions_end);
                if (argv[4][points_end] != '\0' || argv[5][dimensions_end] != '\0') {
                    throw std::invalid_argument("shape");
                }
                return dump_knn(argv[2], argv[3], num_points, dimensions);
            } catch (const std::exception&) {
                fprintf(stderr, "kNN export could not prepare the sample; comparison aborted. "
                                "Check integer shape arguments and available host memory.\n");
                return 1;
            }
        }
        fprintf(stderr, "Invalid test arguments; no tests ran. "
                        "Run without arguments, or use --dump-knn INPUT OUTPUT N D.\n");
        return 1;
    }
    const int neighbor_counts[] = {
        1, 8, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
        511, 512, 513, 767, 768, 769, 1023, 1024, 1025,
        2047, 2048, 2049, 4095, 4096, 4097
    };
    for (int neighbors : neighbor_counts) {
        if (!check_sort(neighbors)) return 1;
    }
    for (int warps : {2, 4, 8}) {
        for (int neighbors : {1, 8, 64, 256, 512, 736, 737, 768}) {
            for (const auto& shape : {std::make_pair(1, 1), std::make_pair(31, 33),
                                      std::make_pair(33, 31), std::make_pair(65, 513)}) {
                for (bool diagonal : {false, true}) {
                    if (!check_reverse_selection(shape.first, shape.second, neighbors, warps, diagonal)) return 1;
                }
            }
        }
    }
#if !SKN_HEAP_WORK_COUNTERS
    for (int warps : {2, 4, 8}) {
        for (int neighbors : {1, 8, 64, 256, 512, 768}) {
            for (const auto& shape : {std::make_pair(1, 1), std::make_pair(31, 33), std::make_pair(33, 31),
                                      std::make_pair(65, 513), std::make_pair(300, 257),
                                      std::make_pair(1000, 1100)}) {
                if (!check_fused_reverse(shape.first, shape.second, neighbors, warps)) return 1;
            }
        }
    }
    for (const auto& shape : {std::make_pair(1, 1), std::make_pair(33, 31), std::make_pair(129, 65),
                              std::make_pair(300, 257), std::make_pair(1000, 1100)}) {
        for (int dim : {1, 8, 33}) {
            for (int neighbors : {1, 8, 64, 256, 768}) {
                for (int warps : {2, 4, 8}) {
                    for (bool prefill : {false, true}) {
                        const bool check_tile = (neighbors == 8 && warps == 8 && !prefill);
                        if (!check_fds_tile(shape.first, shape.second, dim, neighbors, warps, prefill, check_tile)) return 1;
                    }
                }
            }
        }
    }
#endif
    return 0;
}