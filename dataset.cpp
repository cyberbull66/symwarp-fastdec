#include "dataset.h"

#include <stdio.h>
#include <cuda_runtime.h>

#include "knndcshared.h"

Dataset::~Dataset()
{
    Release();
}

void Dataset::Bind(const float* host_data, int num_points, int dim, int cuda_device_id)
{
    Release();
    m_host_data = host_data;
    m_num_points = num_points;
    m_dim = dim;
    m_cuda_device_id = cuda_device_id;
}

bool Dataset::EnsureUploaded()
{
    if (IsUploaded())
        return true;

    if (m_host_data == nullptr || m_num_points <= 0 || m_dim <= 0) {
        printf("ERROR: Dataset upload rejected (host pointer %s, N=%d, D=%d); "
               "the dataset is not on the GPU. Bind a non-empty dataset first.\n",
               m_host_data == nullptr ? "null" : "set", m_num_points, m_dim);
        return false;
    }

    cudaError_t err = cudaSetDevice(m_cuda_device_id);
    if (err != cudaSuccess) {
        printf("ERROR: Dataset upload could not select GPU %d: %s (code %d); "
               "the dataset is not on the GPU. Check the --gpu index.\n",
               m_cuda_device_id, cudaGetErrorString(err), (int)err);
        return false;
    }

    const size_t row_bytes = (size_t)m_num_points * sizeof(float);
    float* device_data = nullptr;
    size_t pitch_bytes = 0;
    err = cudaMallocPitch((void**)&device_data, &pitch_bytes, row_bytes, (size_t)m_dim);
    if (err != cudaSuccess) {
        printf("ERROR: Dataset upload could not allocate %.2f MB on GPU %d: %s (code %d); "
               "the dataset is not on the GPU. Free GPU memory or use a smaller dataset.\n",
               (double)row_bytes * m_dim / (1024.0 * 1024.0), m_cuda_device_id,
               cudaGetErrorString(err), (int)err);
        return false;
    }

    err = cudaMemcpy2D(device_data, pitch_bytes, m_host_data, row_bytes, row_bytes, (size_t)m_dim,
                       cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("ERROR: Dataset upload could not copy the dataset to GPU %d: %s (code %d); "
               "the dataset is not on the GPU. Check the CUDA device state and rerun.\n",
               m_cuda_device_id, cudaGetErrorString(err), (int)err);
        cudaFree(device_data);
        return false;
    }

    m_device_data = device_data;
    m_device_pitch_bytes = pitch_bytes;
    return true;
}

void Dataset::Release()
{
    if (m_device_data != nullptr) {
        cudaFree(m_device_data);
    }
    m_device_data = nullptr;
    m_device_pitch_bytes = 0;
}
