#pragma once

#include "fastdecbase.h"

class KnnCudaFastDEC : public FastDECBase
{
public:
    KnnCudaFastDEC(int num_k_points, int num_clusters, const char* name, int cuda_device_id = 0)
        : FastDECBase(num_k_points, num_clusters, name, cuda_device_id) {}

protected:
    bool compute_knn(Dataset& dataset,
                     int num_k_points,
                     float* out_knn_dists,
                     int* out_knn_indices) override;
};
