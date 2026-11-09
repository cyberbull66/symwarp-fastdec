#include "knncudafastdec.h"
#include "knncuda.h"

bool KnnCudaFastDEC::compute_knn(Dataset& dataset,
                                 int num_k_points,
                                 float* out_knn_dists,
                                 int* out_knn_indices)
{
    return knncuda_self_knn(dataset, num_k_points, out_knn_dists, out_knn_indices);
}
