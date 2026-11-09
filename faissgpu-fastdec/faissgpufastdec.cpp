#include "faissgpufastdec.h"
#include "faissknncuda.h"

bool FaissGpuFastDEC::compute_knn(Dataset& dataset,
                                  int num_k_points,
                                  float* out_knn_dists,
                                  int* out_knn_indices)
{
    return faiss_self_knn_cuda(dataset, num_k_points, out_knn_dists, out_knn_indices);
}
