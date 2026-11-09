#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <string>
#include "appexception.h"

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() {}
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(size_t count)
    {
        reset();
        if (count == 0)
            return;
        cudaError_t err = cudaMalloc((void**)&m_ptr, count * sizeof(T));
        if (err != cudaSuccess)
        {
            m_ptr = nullptr;
            throw AppException(std::string("DeviceBuffer: cudaMalloc failed: ") + cudaGetErrorString(err));
        }
        m_size = count;
    }

    void reset()
    {
        if (m_ptr)
            cudaFree(m_ptr);
        m_ptr = nullptr;
        m_size = 0;
    }

    T* get() { return m_ptr; }
    const T* get() const { return m_ptr; }
    size_t size() const { return m_size; }

private:
    T* m_ptr = nullptr;
    size_t m_size = 0;
};

void gpu_copy_to_host(void* dst_host, const void* src_dev, size_t bytes);

void gpu_copy_to_device(void* dst_dev, const void* src_host, size_t bytes);

void gpu_copy_on_device(void* dst_dev, const void* src_dev, size_t bytes);

void gpu_density_estimation_laplacian(const float* knn_dists_dev, int num_points, int k,
                                      float* densities_dev);

void gpu_density_estimation_gaussian(const float* knn_dists_dev, int num_points, int k,
                                     float* densities_dev);

void gpu_dc_detection(const int* knn_indices_dev, const float* densities_dev, int num_points, int k,
                      int* cdh_ids_dev, int* core_indices_dev, int* num_cores);

void gpu_gather(const float* values_dev, const int* indices_dev, int count, float* out_dev);

void gpu_scatter(const int* values_dev, const int* indices_dev, int count, int* out_dev);

void gpu_gather_points(const float* data_dev, int data_ld, int dim,
                       const int* indices_dev, int count, float* out_dev);
                       
void gpu_final_cluster(const int* cdh_ids_dev, int num_points,
                       const int* center_indices_dev, int num_centers,
                       int* labels_host);
