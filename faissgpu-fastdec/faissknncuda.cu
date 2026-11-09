#include "faissknncuda.h"
#include "dataset.h"
#include <cuda_runtime.h>
#include <faiss/Index.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

#if (FASTDEC_FAISS_KNN_MODE == FASTDEC_FAISS_KNN_MODE_APPROX)
int select_ivf_nlist(int num_points) {
#if (FASTDEC_FAISS_IVF_NLIST > 0)
    return FASTDEC_FAISS_IVF_NLIST;
#else
    int nlist = (int)std::sqrt((double)num_points);
    if (nlist < 1) {
        nlist = 1;
    }
    return nlist;
#endif
}

int select_ivf_nprobe(int nlist) {
#if (FASTDEC_FAISS_IVF_NPROBE > 0)
    int nprobe = FASTDEC_FAISS_IVF_NPROBE;
#else
    int nprobe = 10;
#endif
    if (nprobe < 1) {
        nprobe = 1;
    }
    if (nprobe > nlist) {
        nprobe = nlist;
    }
    return nprobe;
}
#endif

void transposed_to_row_major(const float *data_t,
                             int num_points,
                             int dim,
                             std::vector<float> *data_rm) {
    data_rm->resize((size_t)num_points * dim);

    for (int i = 0; i < num_points; ++i) {
        for (int d = 0; d < dim; ++d) {
            (*data_rm)[(size_t)i * dim + d] = data_t[(size_t)d * num_points + i];
        }
    }
}

void fill_output_with_invalid(int num_points,
                              int k,
                              float *knn_dist,
                              int *knn_index) {
    const float inf = std::numeric_limits<float>::infinity();
    for (int r = 0; r < k; ++r) {
        for (int i = 0; i < num_points; ++i) {
            size_t out_pos = (size_t)r * num_points + i;
            knn_dist[out_pos] = inf;
            knn_index[out_pos] = -1;
        }
    }
}

void write_filtered_knn(const std::vector<float> &search_dist_sq,
                        const std::vector<faiss::idx_t> &search_idx,
                        int num_points,
                        int k,
                        int search_k,
                        float *knn_dist,
                        int *knn_index) {
    const float inf = std::numeric_limits<float>::infinity();

    for (int q = 0; q < num_points; ++q) {
        int out_rank = 0;
        size_t base = (size_t)q * search_k;

        for (int c = 0; c < search_k && out_rank < k; ++c) {
            faiss::idx_t idx = search_idx[base + c];
            if (idx < 0 || idx == q) {
                continue;
            }

            float d2 = search_dist_sq[base + c];
            if (d2 < 0.0f) {
                d2 = 0.0f;
            }

            size_t out_pos = (size_t)out_rank * num_points + q;
            knn_dist[out_pos] = std::sqrt(d2);
            knn_index[out_pos] = (int)idx;
            ++out_rank;
        }

        while (out_rank < k) {
            size_t out_pos = (size_t)out_rank * num_points + q;
            knn_dist[out_pos] = inf;
            knn_index[out_pos] = -1;
            ++out_rank;
        }
    }
}

} // namespace


bool faiss_self_knn_cuda(Dataset &dataset,
                         int      k,
                         float *  knn_dist,
                         int *    knn_index) {
    const float* data = dataset.host_data();
    const int num_points = dataset.num_points();
    const int dim = dataset.dim();
    const int cuda_device_id = dataset.cuda_device_id();

    if (data == NULL || knn_dist == NULL || knn_index == NULL) {
        printf("ERROR: Null input/output pointer in faiss_self_knn_cuda\n");
        return false;
    }

    if (num_points <= 0 || dim <= 0 || k <= 0) {
        printf("SelfKNN(FAISS): nothing to compute (N=%d, D=%d, K=%d)\n",
               num_points,
               dim,
               k);
        return true;
    }

    if (k >= num_points) {
        printf("ERROR: K must be smaller than N for self-kNN (K=%d, N=%d)\n",
               k,
               num_points);
        fill_output_with_invalid(num_points, k, knn_dist, knn_index);
        return false;
    }

    if (!select_cuda_device(cuda_device_id)) {
        printf("ERROR: FAISS self-kNN could not initialize CUDA device %d; "
               "the search did not run. Verify the --gpu index and CUDA runtime state.\n",
               cuda_device_id);
        fill_output_with_invalid(num_points, k, knn_dist, knn_index);
        return false;
    }

    const int search_k = k + 1;

    std::vector<float> data_row_major;
    std::vector<float> search_dist_sq((size_t)num_points * search_k, 0.0f);
    std::vector<faiss::idx_t> search_idx((size_t)num_points * search_k, -1);

    try {
        transposed_to_row_major(data, num_points, dim, &data_row_major);

        faiss::gpu::StandardGpuResources gpu_res;

#if (FASTDEC_FAISS_KNN_MODE == FASTDEC_FAISS_KNN_MODE_EXACT)
        printf("FAISS GPU self-kNN: exact mode (GpuIndexFlatL2)\n");

        faiss::gpu::GpuIndexFlatConfig flat_cfg;
        flat_cfg.device = cuda_device_id;

        faiss::gpu::GpuIndexFlatL2 gpu_index(&gpu_res, dim, flat_cfg);
        gpu_index.add(num_points, data_row_major.data());
        gpu_index.search(num_points,
                         data_row_major.data(),
                         search_k,
                         search_dist_sq.data(),
                         search_idx.data());
#else
        const int nlist = select_ivf_nlist(num_points);
        const int nprobe = select_ivf_nprobe(nlist);

        printf("FAISS GPU self-kNN: approximate mode (GpuIndexIVFFlat), nlist=%d, nprobe=%d\n",
               nlist,
               nprobe);

        faiss::gpu::GpuIndexIVFFlatConfig ivf_cfg;
        ivf_cfg.device = cuda_device_id;

        faiss::gpu::GpuIndexIVFFlat gpu_index(&gpu_res,
                                              dim,
                                              nlist,
                                              faiss::METRIC_L2,
                                              ivf_cfg);
        gpu_index.train(num_points, data_row_major.data());
        gpu_index.add(num_points, data_row_major.data());
        gpu_index.nprobe = nprobe;
        gpu_index.search(num_points,
                         data_row_major.data(),
                         search_k,
                         search_dist_sq.data(),
                         search_idx.data());
#endif
    } catch (const std::exception &ex) {
        printf("ERROR: FAISS GPU kNN failed: %s\n", ex.what());
        fill_output_with_invalid(num_points, k, knn_dist, knn_index);
        return false;
    } catch (...) {
        printf("ERROR: FAISS GPU kNN failed with unknown exception\n");
        fill_output_with_invalid(num_points, k, knn_dist, knn_index);
        return false;
    }

    write_filtered_knn(search_dist_sq,
                       search_idx,
                       num_points,
                       k,
                       search_k,
                       knn_dist,
                       knn_index);

    return true;
}
