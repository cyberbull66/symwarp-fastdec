#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef USE_CUBLAS_TF32
#define USE_CUBLAS_TF32 0
#endif

#ifndef TF32_MIN_DIM
#define TF32_MIN_DIM 8
#endif

#define KNNDC_CUDA_CHECK_GOTO(expr, label)                                      \
    do {                                                                         \
        cudaError_t _knndc_cuda_err = (expr);                                    \
        if (_knndc_cuda_err != cudaSuccess) {                                    \
            goto label;                                                          \
        }                                                                        \
    } while (0)

#define KNNDC_CUDA_CHECK_LAST_GOTO(label)                                        \
    do {                                                                         \
        cudaError_t _knndc_cuda_err = cudaGetLastError();                        \
        if (_knndc_cuda_err != cudaSuccess) {                                    \
            goto label;                                                          \
        }                                                                        \
    } while (0)

#define KNNDC_CUBLAS_CHECK_GOTO(expr, label)                                     \
    do {                                                                         \
        cublasStatus_t _knndc_cublas_status = (expr);                            \
        if (_knndc_cublas_status != CUBLAS_STATUS_SUCCESS) {                     \
            goto label;                                                          \
        }                                                                        \
    } while (0)

static inline bool select_cuda_device(int cuda_device_id)
{
    int nb_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&nb_devices);
    if (err != cudaSuccess || nb_devices == 0) {
        return false;
    }

    if (cuda_device_id < 0 || cuda_device_id >= nb_devices) {
        return false;
    }

    err = cudaSetDevice(cuda_device_id);
    if (err != cudaSuccess) {
        return false;
    }

    return true;
}
