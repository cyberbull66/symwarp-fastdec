#include "fastdecbase.h"
#include <math.h>
#include <stdlib.h>
#include <climits>
#include <thread>
#include "helper.h"
#include "dcreductioncuda.h"
#include "appexception.h"
#include "elapsedtime.h"

#ifndef FASTDEC_STAGE_TIMING
#define FASTDEC_STAGE_TIMING 0
#endif

#define OMP_MIN_POINTS 100000

// Default thread count for density estimation; 0 means hardware_concurrency().
// Overridden at build time by `make FASTDEC_DENSITY_THREADS=N` and at run time
// by the FASTDEC_DENSITY_THREADS environment variable.
#ifndef FASTDEC_DENSITY_THREADS
#define FASTDEC_DENSITY_THREADS 1
#endif

namespace {

// Returns -1 when text is unset or not a non-negative integer.
int parse_non_negative_int(const char* text)
{
    if (text == NULL || *text == '\0')
        return -1;

    char* end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || (end != NULL && *end != '\0') || parsed < 0)
        return -1;
    return (parsed > INT_MAX) ? INT_MAX : (int)parsed;
}

int choose_density_threads(int num_rows)
{
    if (num_rows < OMP_MIN_POINTS)
        return 1;

    int threads = FASTDEC_DENSITY_THREADS;
    int env_threads = parse_non_negative_int(getenv("FASTDEC_DENSITY_THREADS"));
    if (env_threads >= 0)
        threads = env_threads;

    if (threads == 0)
        threads = (int)std::thread::hardware_concurrency();

    if (threads > num_rows)
        threads = num_rows;
    return std::max(threads, 1);
}

void density_estimation_laplacian_range(const real_2d_array& knn_dists,
                                        int num_cols,
                                        double mean_value,
                                        int begin_row,
                                        int end_row,
                                        real_1d_array& out_densities)
{
    for (int row_index = begin_row; row_index < end_row; ++row_index)
    {
        const float* row = knn_dists[row_index];
        double density = 0.0;
        for (int column_index = 0; column_index < num_cols; ++column_index)
        {
            density += exp(-row[column_index] / 2 / mean_value);
        }
        out_densities[row_index] = (float)density;
    }
}

void density_estimation_gaussian_range(const real_2d_array& knn_dists,
                                       int num_cols,
                                       double two_mean_sq,
                                       int begin_row,
                                       int end_row,
                                       real_1d_array& out_densities)
{
    for (int row_index = begin_row; row_index < end_row; ++row_index)
    {
        const float* row = knn_dists[row_index];
        double density = 0.0;
        for (int column_index = 0; column_index < num_cols; ++column_index)
        {
            const double d = row[column_index];
            density += exp(-d * d / two_mean_sq);
        }
        out_densities[row_index] = (float)density;
    }
}

template <typename RangeFn>
void for_each_density_row_range(int num_rows, RangeFn range_fn)
{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (num_rows >= OMP_MIN_POINTS)
    for (int row_index = 0; row_index < num_rows; ++row_index)
    {
        range_fn(row_index, row_index + 1);
    }
#else
    int num_threads = choose_density_threads(num_rows);
    if (num_threads <= 1)
    {
        range_fn(0, num_rows);
        return;
    }

    int chunk = (num_rows + num_threads - 1) / num_threads;
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (int thread_index = 0; thread_index < num_threads; ++thread_index)
    {
        int begin_row = thread_index * chunk;
        if (begin_row >= num_rows)
            break;
        int end_row = std::min(begin_row + chunk, num_rows);
        workers.emplace_back([&range_fn, begin_row, end_row]() {
            range_fn(begin_row, end_row);
        });
    }
    for (std::thread& worker : workers)
        worker.join();
#endif
}

}

#if (USE_CPP11)
    #define MIN_ELEMENT min_element
    #define MAX_ELEMENT max_element
#else
    #define MIN_ELEMENT std::min_element
    #define MAX_ELEMENT std::max_element
#endif

#ifndef USE_GAUSSIAN_KERNEL
    #define USE_GAUSSIAN_KERNEL 0
#endif

#if (USE_GAUSSIAN_KERNEL)
    #define DENSITY_ESTIMATION density_estimation_gaussian
#else
    #define DENSITY_ESTIMATION density_estimation_laplacian
#endif

FastDECBase::FastDECBase(int num_k_points, int num_clusters, const char* name, int cuda_device_id)
{
    printf("- %-17s : \n", name);
    m_num_k_points = num_k_points;
    m_num_clusters = num_clusters;
    m_cuda_device_id = cuda_device_id;
}

void FastDECBase::Fit(const float* data, int num_points, int num_dim)
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
    timer.Stop("Finished building kNN graph");
    timer.Resume();
#endif

    DENSITY_ESTIMATION();

#if FASTDEC_STAGE_TIMING
    timer.Stop("Finished density estimation");
    timer.Resume();
#endif

    dc_detection();

    while (m_core_indices.length() < m_num_clusters)
    {
        if (m_num_k_points <= 1)
        {
            throw AppException("Fit: not enough core points");
        }

        m_num_k_points = m_num_k_points / 2;
        m_knn_dists.setlength(m_knn_dists.rows(), m_num_k_points);
        m_knn_indices.setlength(m_knn_indices.rows(), m_num_k_points);

        DENSITY_ESTIMATION();
        dc_detection();
    }

#if FASTDEC_STAGE_TIMING
    timer.Stop("Finished DC detection");
    timer.Resume();
#endif

    if (m_core_indices.length() == m_num_clusters)
    {
        m_center_indices = m_core_indices;
    }
    else
    {
        dc_reduction();
    }

    m_dataset.Release();

#if FASTDEC_STAGE_TIMING
    timer.Stop("Finished DC reduction");
    timer.Resume();
#endif

    final_cluster();

#if FASTDEC_STAGE_TIMING
    timer.Stop("Finished final clustering");
    timer.Print();
#endif
}

void FastDECBase::build_knn_graph()
{
    m_knn_dists.setlength(m_num_points, m_num_k_points);
    m_knn_indices.setlength(m_num_points, m_num_k_points);

    float* test_knn_dist = (float*)malloc((size_t)m_num_points * m_num_k_points * sizeof(float));
    int* test_knn_index = (int*)malloc((size_t)m_num_points * m_num_k_points * sizeof(int));

    if (!test_knn_dist || !test_knn_index) {
        free(test_knn_dist);
        free(test_knn_index);
        throw AppException("build_knn_graph: failed to allocate memory for kNN results");
    }

    if (!compute_knn(m_dataset, m_num_k_points, test_knn_dist, test_knn_index)) {
        free(test_knn_dist);
        free(test_knn_index);
        throw AppException("build_knn_graph: failed to compute kNN on GPU");
    }

    knncuda_idx_to_2d_array(test_knn_index, m_num_points, m_num_k_points, m_knn_indices);
    knncuda_dist_to_2d_array(test_knn_dist, m_num_points, m_num_k_points, m_knn_dists);

    free(test_knn_dist);
    free(test_knn_index);
}

double FastDECBase::mean_knn_dist() const
{
    double mean_value = 0.0;
    const int num_rows = m_knn_dists.rows();
    const int num_cols = m_knn_dists.cols();

    for (int row_index = 0; row_index < num_rows; ++row_index)
    {
        const float* row = m_knn_dists[row_index];
        for (int column_index = 0; column_index < num_cols; ++column_index)
        {
            mean_value += row[column_index];
        }
    }
    mean_value = mean_value / ((double)num_rows * num_cols);

    return mean_value;
}

void FastDECBase::density_estimation_laplacian()
{
    const double mean_value = mean_knn_dist();
    const int num_rows = m_knn_dists.rows();
    const int num_cols = m_knn_dists.cols();

    m_densities.setlength(num_rows);
    for_each_density_row_range(num_rows, [&](int begin_row, int end_row) {
        density_estimation_laplacian_range(m_knn_dists, num_cols, mean_value,
                                           begin_row, end_row, m_densities);
    });
}

void FastDECBase::density_estimation_gaussian()
{
    const double mean_value = mean_knn_dist();
    const double two_mean_sq = 2.0 * mean_value * mean_value;
    const int num_rows = m_knn_dists.rows();
    const int num_cols = m_knn_dists.cols();

    m_densities.setlength(num_rows);
    for_each_density_row_range(num_rows, [&](int begin_row, int end_row) {
        density_estimation_gaussian_range(m_knn_dists, num_cols, two_mean_sq,
                                          begin_row, end_row, m_densities);
    });
}

void FastDECBase::dc_detection()
{
    std::vector<ae_int_t> cdh_ids(m_num_points, -1);
    std::vector<ae_int_t> k_int(m_num_points, m_num_points);

    int j;
    for (int i = 0; i < m_num_points; ++i)
    {
        for (int q = 0; q < m_knn_indices.cols(); ++q)
        {
            j = m_knn_indices(i, q);
            if (m_densities[i] < m_densities[j])
            {
                cdh_ids[i] = j;
                k_int[i] = q + 1;
                break;
            }
        }
    }

    std::vector<ae_int_t> core_idx_v;
    core_idx_v.reserve(m_num_k_points);
    for (int i = 0; i < m_num_points; ++i)
    {
        if (cdh_ids[i] == -1)
            core_idx_v.push_back(i);
    }

    m_core_indices.setcontent(core_idx_v.size(), core_idx_v.data());
    m_cdh_ids.setcontent(m_num_points, cdh_ids.data());
}

void FastDECBase::dc_reduction()
{
    int n_cores = m_core_indices.length();
    std::vector<int> g_idx(n_cores, -1);

    m_center_indices.setlength(m_num_clusters);
    if (!warp_parallel_dc_reduction_cuda(m_dataset,
                                         m_densities.getcontent(),
                                         m_core_indices.getcontent(),
                                         n_cores,
                                         m_num_clusters,
                                         g_idx.data(),
                                         m_center_indices.getcontent()))
    {
        throw AppException("dc_reduction: gpu warp_parallel_dc_reduction failed");
    }
    
    for (int i = 0; i < n_cores; ++i)
    {
        m_cdh_ids[m_core_indices[i]] = g_idx[i];
    }
}

void FastDECBase::final_cluster()
{
    m_labels.setlength(m_num_points, -1);

    auto sorted_density_indices = qsort_value_index(m_densities.getcontent(), m_densities.length());

    int count = 0;
    for (int i = 0; i < m_center_indices.length(); i++)
    {
        m_labels[m_center_indices[i]] = count;
        count++;
    }
    for (int i : sorted_density_indices)
    {
        if (m_labels[i] == -1)
        {
            int cdh = m_cdh_ids[i];
            if (cdh == -1)
            {
                m_labels[i] = 0;
            }
            else
            {
                m_labels[i] = m_labels[cdh];
            }
        }
    }
}

std::vector<int> FastDECBase::qsort_value_index(const float* values, int size) const
{
    std::vector<value_and_index> values_indexed(size);
    for (int i = 0; i < size; ++i)
    {
        values_indexed[i].value = values[i];
        values_indexed[i].index = i;
    }
    qsort(values_indexed.data(), size, sizeof(value_and_index), compare);
    std::vector<int> sorted_values_indices(size);
    for (int i = 0; i < size; ++i)
    {
        sorted_values_indices[i] = values_indexed[i].index;
    }
    return sorted_values_indices;
}
