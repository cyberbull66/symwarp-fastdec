#pragma once

#include "knndcshared.h"

class Dataset;

// kNN-CUDA implementation (the global memory version) from https://github.com/vincentfpgarcia/kNN-CUDA,
// extended to support large datasets with batched processing.
// Original work by Vincent Garcia, Eric Debreuve and Michel Barlaud. This adaptation is licensed
// CC BY-NC-SA 3.0 (not MIT); see knncuda-fastdec/LICENSE.

// KNN_SELECT_METHOD controls the exact top-k selection kernel used by the
// shared-memory batched path after the distance tile has been computed:
//   0 = original modified_insertion_sort
//   1 = modified_heap_select
//   2 = hybrid: insertion sort below KNN_SELECT_HEAP_THRESHOLD, heap above it
#define KNN_SELECT_METHOD 0

// Threshold for hybrid top-k selection, used when KNN_SELECT_METHOD == 2.
#define KNN_SELECT_HEAP_THRESHOLD 8

// KNN_MERGE_METHOD controls how per-batch top-k results are merged into the
// accumulated top-k lists for the shared-memory batched path:
//   0 = insertion-based merge similar to the original modified_insertion_sort
//   1 = linear merge of two sorted top-k lists
#define KNN_MERGE_METHOD 1

/**
 * Computes the self k-NN for a given dataset using the kNN-CUDA github implementation.
 *
 * @param dataset    the dataset containing the points on the GPU
 * @param k          number of neighbors to consider
 * @param knn_dist   output array containing the distances to the k nearest neighbors
 * @param knn_index  output array containing the indexes of the k nearest neighbors
 * @return true if successful, false otherwise.
 */
bool knncuda_self_knn(Dataset &dataset,
                      int k,
                      float *knn_dist,
                      int *knn_index);
