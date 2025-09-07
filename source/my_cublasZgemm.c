#include <cuda_runtime.h>
#include <mpi.h>
#include <openacc.h>
#include <stdlib.h>
#include "openmx_common.h"

void my_cublasZgemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, dcomplex const * A, dcomplex const * B, dcomplex * C)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudacall(cudaSetDevice(rank % getDeviceCount()));

    cublasHandle_t handle;
    cudacall(cublasCreate(&handle));

    cuDoubleComplex *d_A, *d_B, *d_C;
    cudacall(cudaMalloc((void**)&d_A, m * k * sizeof(cuDoubleComplex)));
    cudacall(cudaMalloc((void**)&d_B, n * k * sizeof(cuDoubleComplex)));
    cudacall(cudaMalloc((void**)&d_C, m * n * sizeof(cuDoubleComplex)));

    cudacall(cudaMemcpy(d_A, A, m * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(d_B, B, n * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cuDoubleComplex const alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex const beta = make_cuDoubleComplex(0.0, 0.0);

    cublasZgemm(handle, transa, transb, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

    cudacall(cudaMemcpy(C, d_C, m * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

void my_cublasZgemm_openacc(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, dcomplex const * A, dcomplex const * B, dcomplex* C)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudacall(cudaSetDevice(rank % getDeviceCount()));

    // OpenACC
    int local_numdevices = acc_get_num_devices(acc_device_nvidia);
    acc_set_device_num(rank % local_numdevices, acc_device_nvidia);

    cublasHandle_t handle;
    cudacall(cublasCreate(&handle));

#pragma acc data present(A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        // cuDoubleComplex *d_A, *d_B, *d_C;
        // cudacall(cudaMalloc((void**)&d_A, m * k * sizeof(cuDoubleComplex)));
        // cudacall(cudaMalloc((void**)&d_B, n * k * sizeof(cuDoubleComplex)));
        // cudacall(cudaMalloc((void**)&d_C, m * n * sizeof(cuDoubleComplex)));

        // cudacall(cudaMemcpy(d_A, A, m * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        // cudacall(cudaMemcpy(d_B, B, n * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        cuDoubleComplex const alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex const beta = make_cuDoubleComplex(0.0, 0.0);

        cublasZgemm(handle, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m);

        // cudacall(cudaMemcpy(C, d_C, m * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        // cudaFree(d_A);
        // cudaFree(d_B);
        // cudaFree(d_C);
        cublasDestroy(handle);
    }
}
