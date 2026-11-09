#pragma once

#include "knndcshared.h"

class Dataset;

/**
 * Implements the SymTri-kNN algorithm for a self k-nearest neighbors search,
 * combining symmetric upper triangle traversal and warp-cooperative top-k selection.
 *
 * @param dataset the dataset containing the points on the GPU.
 * @param k Number of nearest neighbors to find.
 * @param knn_dist Output array for kNN distances (k x num_points, row-major, Euclidean distances).
 * @param knn_index Output array for kNN indices (k x num_points, row-major).
 * @return true if successful, false otherwise.
 */
bool symtri_knn_cuda(Dataset &dataset,
                     int      k,
                     float *  knn_dist,
                     int *    knn_index);
