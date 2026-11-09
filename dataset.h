#pragma once

#include <cstddef>

// The dataset (dim rows x num_points columns) used by one FastDEC fit:
// a non-owning pointer to the host copy and, once uploaded, an owned pitched GPU copy
// shared by the kNN backend and the DC reduction.
class Dataset
{
public:
    Dataset() = default;
    ~Dataset();

    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    void Bind(const float* host_data, int num_points, int dim, int cuda_device_id = 0);

    bool EnsureUploaded();

    bool IsUploaded() const { return m_device_data != nullptr; }

    void Release();

    const float* host_data() const { return m_host_data; }
    const float* device_data() const { return m_device_data; }
    size_t device_pitch_bytes() const { return m_device_pitch_bytes; }
    int num_points() const { return m_num_points; }
    int dim() const { return m_dim; }
    int cuda_device_id() const { return m_cuda_device_id; }

private:
    const float* m_host_data = nullptr;
    float* m_device_data = nullptr;
    size_t m_device_pitch_bytes = 0;
    int m_num_points = 0;
    int m_dim = 0;
    int m_cuda_device_id = 0;
};
