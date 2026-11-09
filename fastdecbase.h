#pragma once

#include <stdio.h>
#include <vector>
#include <algorithm>
#include "arrays.h"
#include "dataset.h"

class FastDECBase
{
protected:
    int m_num_k_points = 0;
    int m_num_clusters = 0;
    int m_num_points = 0;
    int m_num_dims = 0;
    int m_cuda_device_id = 0;

    real_2d_array m_knn_dists;
    integer_2d_array m_knn_indices;
    real_1d_array m_densities;

    integer_1d_array m_core_indices;
    integer_1d_array m_cdh_ids;
    real_2d_array m_knn_dists_cores;
    integer_2d_array m_knn_indices_cores;

    integer_1d_array m_center_indices;
    integer_1d_array m_labels;

    Dataset m_dataset;

public:
    FastDECBase(int num_k_points, int num_clusters, const char* name, int cuda_device_id = 0);
    virtual ~FastDECBase() = default;

    const integer_1d_array& get_labels() const { return m_labels; }
    const integer_1d_array& get_core_indices() const { return m_core_indices; }
    const integer_1d_array& get_center_indices() const { return m_center_indices; }

    void Fit(const float* data, int num_points, int num_dim);

protected:
    virtual bool compute_knn(Dataset& dataset,
                             int num_k_points,
                             float* out_knn_dists,
                             int* out_knn_indices) = 0;

private:
    void build_knn_graph();
    double mean_knn_dist() const;
    void density_estimation_laplacian();
    void density_estimation_gaussian();
    void dc_detection();
    void dc_reduction();
    void final_cluster();

    struct value_and_index
    {
        float value;
        int index;
    };

    static int compare(const void* a, const void* b)
    {
        auto fa = *(const value_and_index*)a;
        auto fb = *(const value_and_index*)b;
        if (fa.value != fb.value)
            return (fa.value < fb.value) - (fa.value > fb.value);
        return (fa.index > fb.index) - (fa.index < fb.index);
    }

    std::vector<int> qsort_value_index(const float* values, int size) const;
};
