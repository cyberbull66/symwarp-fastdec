#include "symwarpfullgpufastdec.h"
#include <stdio.h>
#include <vector>
#include "appexception.h"
#include "dcreductioncuda.h"
#include "elapsedtime.h"
#include "symtriknncuda.h"

#ifndef FASTDEC_STAGE_TIMING
#define FASTDEC_STAGE_TIMING 0
#endif

#ifndef USE_GAUSSIAN_KERNEL
    #define USE_GAUSSIAN_KERNEL 0
#endif

#if (USE_GAUSSIAN_KERNEL)
    #define DENSITY_ESTIMATION density_estimation_gaussian
#else
    #define DENSITY_ESTIMATION density_estimation_laplacian
#endif

SymWarpFullGpuFastDEC::SymWarpFullGpuFastDEC(int num_k_points, int num_clusters, const char* name, int cuda_device_id)
{
    printf("- %-17s : \n", name);
    m_num_k_points = num_k_points;
    m_num_clusters = num_clusters;
    m_cuda_device_id = cuda_device_id;
}

void SymWarpFullGpuFastDEC::Fit(const float* data, int num_points, int num_dim)
{
    m_num_points = num_points;
    m_num_dims = num_dim;
    m_dataset.Bind(data, num_points, num_dim, m_cuda_device_id);

#if FASTDEC_STAGE_TIMING
    SystemTimer timer;
    timer.Start();
#endif

    build_knn_graph();

#if FASTDEC_STAGE_TIMING
    timer.Stop("Finished querying KNN for all data points");
    timer.Resume();
#endif

    m_densities_dev.allocate(m_num_points);
    m_cdh_ids_dev.allocate(m_num_points);
    m_core_indices_dev.allocate(m_num_points);

    DENSITY_ESTIMATION();

#if FASTDEC_STAGE_TIMING
    timer.Stop("Finished density estimation");
    timer.Resume();
#endif

    dc_detection();

    while (m_num_cores < m_num_clusters)
    {
        if (m_num_k_points <= 1)
        {
            throw AppException("Fit: not enough core points");
        }

        m_num_k_points = m_num_k_points / 2;
        DENSITY_ESTIMATION();
        dc_detection();
    }

#if FASTDEC_STAGE_TIMING
    timer.Stop("Finished DC detection");
    timer.Resume();
#endif

    m_knn_dists_dev.reset();
    m_knn_indices_dev.reset();

    if (m_num_cores == m_num_clusters)
    {
        m_center_indices_dev.allocate(m_num_clusters);
        gpu_copy_on_device(m_center_indices_dev.get(), m_core_indices_dev.get(), (size_t)m_num_clusters * sizeof(int));
    }
    else
    {
        dc_reduction();
    }

#if FASTDEC_STAGE_TIMING
    timer.Stop("Finished DC reduction");
    timer.Resume();
#endif

    m_dataset.Release();
    final_cluster();

#if FASTDEC_STAGE_TIMING
    timer.Stop("Finished final clustering");
    timer.Print();
#endif
}

void SymWarpFullGpuFastDEC::build_knn_graph()
{
    std::vector<float> knn_dist((size_t)m_num_points * m_num_k_points);
    std::vector<int> knn_index((size_t)m_num_points * m_num_k_points);

    if (!symtri_knn_cuda(m_dataset, m_num_k_points, knn_dist.data(), knn_index.data()))
    {
        throw AppException("build_knn_graph: failed to compute kNN on GPU");
    }

    m_knn_dists_dev.allocate(knn_dist.size());
    m_knn_indices_dev.allocate(knn_index.size());
    gpu_copy_to_device(m_knn_dists_dev.get(), knn_dist.data(), knn_dist.size() * sizeof(float));
    gpu_copy_to_device(m_knn_indices_dev.get(), knn_index.data(), knn_index.size() * sizeof(int));
}

void SymWarpFullGpuFastDEC::density_estimation_laplacian()
{
    gpu_density_estimation_laplacian(m_knn_dists_dev.get(), m_num_points, m_num_k_points,
                                     m_densities_dev.get());
}

void SymWarpFullGpuFastDEC::density_estimation_gaussian()
{
    gpu_density_estimation_gaussian(m_knn_dists_dev.get(), m_num_points, m_num_k_points,
                                    m_densities_dev.get());
}

void SymWarpFullGpuFastDEC::dc_detection()
{
    gpu_dc_detection(m_knn_indices_dev.get(), m_densities_dev.get(), m_num_points, m_num_k_points,
                     m_cdh_ids_dev.get(), m_core_indices_dev.get(), &m_num_cores);
}

const integer_1d_array& SymWarpFullGpuFastDEC::get_core_indices()
{
    m_core_indices.setlength(m_num_cores);
    gpu_copy_to_host(m_core_indices.getcontent(), m_core_indices_dev.get(), (size_t)m_num_cores * sizeof(int));
    return m_core_indices;
}

const integer_1d_array& SymWarpFullGpuFastDEC::get_center_indices()
{
    m_center_indices.setlength((ae_int_t)m_center_indices_dev.size());
    gpu_copy_to_host(m_center_indices.getcontent(), m_center_indices_dev.get(), m_center_indices_dev.size() * sizeof(int));
    return m_center_indices;
}

std::vector<float> SymWarpFullGpuFastDEC::get_densities() const
{
    std::vector<float> densities(m_densities_dev.size());
    gpu_copy_to_host(densities.data(), m_densities_dev.get(), densities.size() * sizeof(float));
    return densities;
}

void SymWarpFullGpuFastDEC::dc_reduction()
{
    const int n_cores = m_num_cores;
    std::vector<float> densities_host((size_t)m_num_points);
    std::vector<int> core_indices_host((size_t)n_cores);
    std::vector<int> core_cdh_index_host((size_t)n_cores, -1);
    std::vector<int> center_indices_host((size_t)m_num_clusters, -1);

    gpu_copy_to_host(densities_host.data(), m_densities_dev.get(), (size_t)m_num_points * sizeof(float));
    gpu_copy_to_host(core_indices_host.data(), m_core_indices_dev.get(), (size_t)n_cores * sizeof(int));

    if (!warp_parallel_dc_reduction_cuda(m_dataset,
                                         densities_host.data(),
                                         core_indices_host.data(),
                                         n_cores,
                                         m_num_clusters,
                                         core_cdh_index_host.data(),
                                         center_indices_host.data()))
    {
        throw AppException("dc_reduction: warp_parallel_dc_reduction_cuda failed");
    }

    DeviceBuffer<int> core_cdh_index_dev;
    core_cdh_index_dev.allocate((size_t)n_cores);
    gpu_copy_to_device(core_cdh_index_dev.get(), core_cdh_index_host.data(), (size_t)n_cores * sizeof(int));
    gpu_scatter(core_cdh_index_dev.get(), m_core_indices_dev.get(), n_cores, m_cdh_ids_dev.get());

    m_center_indices_dev.allocate(m_num_clusters);
    gpu_copy_to_device(m_center_indices_dev.get(), center_indices_host.data(), (size_t)m_num_clusters * sizeof(int));
}

void SymWarpFullGpuFastDEC::final_cluster()
{
    m_labels.setlength(m_num_points, -1);
    gpu_final_cluster(m_cdh_ids_dev.get(), m_num_points,
                      m_center_indices_dev.get(), (int)m_center_indices_dev.size(),
                      m_labels.getcontent());
}
