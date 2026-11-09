#pragma once

#include <vector>
#include "arrays.h"
#include "dataset.h"
#include "symwarpfastdeccuda.h"

class SymWarpFullGpuFastDEC
{
private:
    int m_num_k_points = 0;
    int m_num_clusters = 0;
    int m_num_points = 0;
    int m_num_dims = 0;
    int m_cuda_device_id = 0;

    Dataset m_dataset;

    DeviceBuffer<float> m_knn_dists_dev;
    DeviceBuffer<int> m_knn_indices_dev;
    DeviceBuffer<float> m_densities_dev;
    DeviceBuffer<int> m_cdh_ids_dev;
    DeviceBuffer<int> m_core_indices_dev;
    int m_num_cores = 0;
    DeviceBuffer<int> m_center_indices_dev;

    integer_1d_array m_core_indices;
    integer_1d_array m_center_indices;
    integer_1d_array m_labels;

public:
    SymWarpFullGpuFastDEC(int num_k_points, int num_clusters, const char* name, int cuda_device_id = 0);

    const integer_1d_array& get_labels() { return m_labels; }
    const integer_1d_array& get_core_indices();
    const integer_1d_array& get_center_indices();
    std::vector<float> get_densities() const;

    void Fit(const float* data, int num_points, int num_dim);

private:
    void build_knn_graph();
    void density_estimation_laplacian();
    void density_estimation_gaussian();
    void dc_detection();
    void dc_reduction();
    void final_cluster();
};
