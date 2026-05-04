#include "openmx_common.h"
#include <cuda_runtime.h>
#include <openacc.h>
#include <stdio.h>

void my_cublasZgemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, dcomplex const * A,
    dcomplex const * B, dcomplex * C)
{
    cublasHandle_t handle;
    wait_cudafunc(cublasCreate(&handle));

    cuDoubleComplex *d_A, *d_B, *d_C;
    wait_cudafunc(cudaMalloc((void **)&d_A, m * k * sizeof(cuDoubleComplex)));
    wait_cudafunc(cudaMalloc((void **)&d_B, n * k * sizeof(cuDoubleComplex)));
    wait_cudafunc(cudaMalloc((void **)&d_C, m * n * sizeof(cuDoubleComplex)));

    wait_cudafunc(cudaMemcpy(d_A, A, m * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    wait_cudafunc(cudaMemcpy(d_B, B, n * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cuDoubleComplex const alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex const beta  = make_cuDoubleComplex(0.0, 0.0);

    wait_cudafunc(openmx_gemmul8Zgemm(handle, transa, transb, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));

    wait_cudafunc(cudaMemcpy(C, d_C, m * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    wait_cudafunc(cudaFree(d_A));
    wait_cudafunc(cudaFree(d_B));
    wait_cudafunc(cudaFree(d_C));
    wait_cudafunc(cublasDestroy(handle));
}

void my_cublasZgemm_openacc(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, dcomplex const * A,
    dcomplex const * B, dcomplex * C)
{
    cublasHandle_t handle;
    wait_cudafunc(cublasCreate(&handle));

#pragma acc data      present(A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        // cuDoubleComplex *d_A, *d_B, *d_C;
        // wait_cudafunc(cudaMalloc((void**)&d_A, m * k * sizeof(cuDoubleComplex)));
        // wait_cudafunc(cudaMalloc((void**)&d_B, n * k * sizeof(cuDoubleComplex)));
        // wait_cudafunc(cudaMalloc((void**)&d_C, m * n * sizeof(cuDoubleComplex)));

        // wait_cudafunc(cudaMemcpy(d_A, A, m * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        // wait_cudafunc(cudaMemcpy(d_B, B, n * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        cuDoubleComplex const alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex const beta  = make_cuDoubleComplex(0.0, 0.0);

        wait_cudafunc(openmx_gemmul8Zgemm(handle, transa, transb, m, n, k, &alpha, (cuDoubleComplex const *)A, m,
                                  (cuDoubleComplex const *)B, k, &beta, (cuDoubleComplex *)C, m));

        // wait_cudafunc(cudaMemcpy(C, d_C, m * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        // cudaFree(d_A);
        // cudaFree(d_B);
        // cudaFree(d_C);
        wait_cudafunc(cublasDestroy(handle));
    }
}
