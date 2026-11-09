#pragma once

#ifndef DC_CUBLAS_CORE_TILE
#define DC_CUBLAS_CORE_TILE 1024
#endif

#ifndef DC_CUBLAS_DS_BLOCK
#define DC_CUBLAS_DS_BLOCK 32768
#endif

#ifndef DC_CUBLAS_WARPS_PER_BLOCK
#define DC_CUBLAS_WARPS_PER_BLOCK 8
#endif

#ifndef DC_CUBLAS_INTERLEAVE_STREAMS
#define DC_CUBLAS_INTERLEAVE_STREAMS 0
#endif

#ifndef DC_CUBLAS_SINGLE_SLAB_MAX_MB
#define DC_CUBLAS_SINGLE_SLAB_MAX_MB 2560
#endif

#ifndef DC_CUBLAS_MIN_CORE_TILE
#define DC_CUBLAS_MIN_CORE_TILE 256
#endif

#ifndef DC_CUBLAS_MEM_RESERVE_MB
#define DC_CUBLAS_MEM_RESERVE_MB 1024
#endif

class Dataset;

/**
 * Perform Warp-Parallel DC Reduction and center selection based on each DC's significance score on the GPU.
 *
 * @param ds Dataset wrapper.
 * @param ds_den_host Dataset densities (length ds.num_points()).
 * @param core_indices_host Global dataset indices of the cores/modes (length core_nb).
 * @param core_nb Number of core/mode points.
 * @param num_centers Number of centers to select.
 * @param core_cdh_index_host Output CDH indices for each core/mode (length core_nb).
 * @param center_indices_host Output selected center indices in global indexing (length num_centers).
 * @return true if successful, false otherwise.
 */
bool warp_parallel_dc_reduction_cuda(Dataset     &ds,
                                     const float *ds_den_host,
                                     const int   *core_indices_host,
                                     int          core_nb,
                                     int          num_centers,
                                     int         *core_cdh_index_host,
                                     int         *center_indices_host);
