#include "openmx_common.h"
#include "utility.h"
#include <cuda_runtime.h>
#include <openacc.h>
#include <stdio.h>

void my_cublasDgemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double const * A,
                    double const * B, double * C)
{
    cublasHandle_t handle;
    wait_cudafunc(cublasCreate(&handle));

    int64_t memsize = get_gpu_total_memory_in_bytes();
    int64_t datasize = (int64_t)(sizeof(double) * (int64_t)(m * k) + sizeof(double) * (int64_t)(n * k) +
                                 sizeof(double) * (int64_t)(m * n));
    if (memsize < datasize) {
        fprintf(stderr, "There's not enough memory on the device (GPU) to continue processing!");
        exit(1);
    }

    double *d_A, *d_B, *d_C;
    wait_cudafunc(cudaMalloc((void **)&d_A, m * k * sizeof(double)));
    wait_cudafunc(cudaMalloc((void **)&d_B, n * k * sizeof(double)));
    wait_cudafunc(cudaMalloc((void **)&d_C, m * n * sizeof(double)));

    wait_cudafunc(cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
    wait_cudafunc(cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice));

    double const alpha = 1.0;
    double const beta  = 0.0;

    wait_cudafunc(cublasDgemm(handle, transa, transb, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));

    wait_cudafunc(cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    wait_cudafunc(cudaFree(d_A));
    wait_cudafunc(cudaFree(d_B));
    wait_cudafunc(cudaFree(d_C));
    wait_cudafunc(cublasDestroy(handle));
}

void my_cublasDgemm_openacc(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double const * A,
                            double const * B, double * C)
{
    cublasHandle_t handle;
    wait_cudafunc(cublasCreate(&handle));

#pragma acc data      present(A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        double const alpha = 1.0;
        double const beta  = 0.0;

        wait_cudafunc(cublasDgemm(handle, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m));

        wait_cudafunc(cublasDestroy(handle));
    }
}