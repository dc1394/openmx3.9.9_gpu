#include "openmx_common.h"
#include <cuda_runtime.h>
#include <mpi.h>
#include <openacc.h>
#include <stdlib.h>
#include <time.h>

void my_cublasDgemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double const * A,
                    double const * B, double * C)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudacall(cudaSetDevice(rank % getDeviceCount()));

    cublasHandle_t handle;
    wait_cudafunc(cublasCreate(&handle));

    double *d_A, *d_B, *d_C;
    wait_cudafunc(cudaMalloc((void **)&d_A, m * k * sizeof(double)));
    wait_cudafunc(cudaMalloc((void **)&d_B, n * k * sizeof(double)));
    wait_cudafunc(cudaMalloc((void **)&d_C, m * n * sizeof(double)));

    cudacall(cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice));

    double const alpha = 1.0;
    double const beta  = 0.0;

    cublasDgemm(handle, transa, transb, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

    cudacall(cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

void my_cublasDgemm_openacc(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double const * A,
                            double const * B, double * C)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudacall(cudaSetDevice(rank % getDeviceCount()));

    // OpenACC
    int local_numdevices = acc_get_num_devices(acc_device_nvidia);
    acc_set_device_num(rank % local_numdevices, acc_device_nvidia);

    cublasHandle_t handle;
    wait_cudafunc(cublasCreate(&handle));

#pragma acc data      present(A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        double const alpha = 1.0;
        double const beta  = 0.0;

        cublasDgemm(handle, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m);

        cublasDestroy(handle);
    }
}