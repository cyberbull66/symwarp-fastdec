#pragma once

#include "knndcshared.h"

class Dataset;

// Search mode control.
// 0: exact kNN with GpuIndexFlatL2
// 1: approximate kNN with GpuIndexIVFFlat
#define FASTDEC_FAISS_KNN_MODE_EXACT  0
#define FASTDEC_FAISS_KNN_MODE_APPROX 1

#ifndef FASTDEC_FAISS_KNN_MODE
#define FASTDEC_FAISS_KNN_MODE FASTDEC_FAISS_KNN_MODE_EXACT
#endif

#if (FASTDEC_FAISS_KNN_MODE != FASTDEC_FAISS_KNN_MODE_EXACT) && \
    (FASTDEC_FAISS_KNN_MODE != FASTDEC_FAISS_KNN_MODE_APPROX)
#error "FASTDEC_FAISS_KNN_MODE must be FASTDEC_FAISS_KNN_MODE_EXACT or FASTDEC_FAISS_KNN_MODE_APPROX"
#endif

// Approximate IVF parameters (used only when FASTDEC_FAISS_KNN_MODE is APPROX).
// If FASTDEC_FAISS_IVF_NLIST <= 0, nlist is auto-selected as sqrt(num_points).
#ifndef FASTDEC_FAISS_IVF_NLIST
#define FASTDEC_FAISS_IVF_NLIST 0
#endif

// If FASTDEC_FAISS_IVF_NPROBE <= 0, nprobe is auto-selected as min(nlist, 10).
#ifndef FASTDEC_FAISS_IVF_NPROBE
#define FASTDEC_FAISS_IVF_NPROBE 10
#endif

/**
 * Self k-NN via FAISS GPU exact kNN indexing backend.
 *
 * @param dataset Dataset used by the FAISS backend.
 * @param k Number of nearest neighbors to query.
 * @param knn_dist Output array for kNN distances (k x num_points, row-major, Euclidean distances).
 * @param knn_index Output array for kNN indices (k x num_points, row-major).
 * @return true if the kNN search was successful, false otherwise.
 */
bool faiss_self_knn_cuda(Dataset &dataset,
                         int      k,
                         float *  knn_dist,
                         int *    knn_index);
