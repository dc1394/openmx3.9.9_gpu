/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include "openmx_common.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <openacc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static long long get_gpu_total_memory_in_bytes();

int cusolver_Syevdx(double * A, double * W, int m, int MaxN)
{
    const int lda = m;

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t       stream    = NULL;
    int                deviceCount;
    wait_cudafunc(cudaGetDeviceCount(&deviceCount));

    int rank;
    MPI_Comm_rank(mpi_comm_level1, &rank);
    wait_cudafunc(cudaSetDevice(rank % deviceCount));

    double * d_A = NULL;
    double * d_W = NULL;
    double   vl;
    double   vu;
    int64_t  h_meig = 0;
    int *    d_info = NULL;

    int info = 0;

    size_t workspaceInBytesOnDevice = 0;    /* size of workspace */
    void * d_work                   = NULL; /* device workspace */
    size_t workspaceInBytesOnHost   = 0;    /* size of workspace */
    void * h_work                   = NULL; /* host workspace for */

    /* step 1: create cusolver handle, bind a stream */
    wait_cudafunc(cusolverDnCreate(&cusolverH));

    wait_cudafunc(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    wait_cudafunc(cusolverDnSetStream(cusolverH, stream));

    wait_cudafunc(cudaMalloc((void **)(&d_A), sizeof(double) * lda * m));
    wait_cudafunc(cudaMalloc((void **)(&d_W), sizeof(double) * m));
    wait_cudafunc(cudaMalloc((void **)(&d_info), sizeof(int)));

    wait_cudafunc(cudaMemcpyAsync(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice, stream));

    // step 3: query working space of syevd
    cusolverEigMode_t  jobz = CUSOLVER_EIG_MODE_VECTOR;  // compute eigenvalues and eigenvectors.
    cublasFillMode_t   uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range;
    if (m == MaxN) {
        range = CUSOLVER_EIG_RANGE_ALL;
    } else {
        range = CUSOLVER_EIG_RANGE_I;
    }

    wait_cudafunc(cusolverDnXsyevdx_bufferSize(cusolverH, NULL, jobz, range, uplo, m, CUDA_R_64F, d_A, lda, &vl, &vu,
                                               1L, (long int)(MaxN), &h_meig, CUDA_R_64F, d_W, CUDA_R_64F,
                                               &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    wait_cudafunc(cudaMalloc((void **)(&d_work), workspaceInBytesOnDevice));
    h_work = malloc(workspaceInBytesOnHost);
    if (!h_work) {
        fprintf(stderr, "Could not allocate host memory.\n");
        exit(1);
    }

    // step 4: compute spectrum
    wait_cudafunc(cusolverDnXsyevdx(cusolverH, NULL, jobz, range, uplo, m, CUDA_R_64F, d_A, lda, &vl, &vu, 1L,
                                    (long int)(MaxN), &h_meig, CUDA_R_64F, d_W, CUDA_R_64F, d_work,
                                    workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));

    wait_cudafunc(cudaMemcpyAsync(A, d_A, sizeof(double) * lda * m, cudaMemcpyDeviceToHost, stream));
    wait_cudafunc(cudaMemcpyAsync(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost, stream));
    wait_cudafunc(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    wait_cudafunc(cudaStreamSynchronize(stream));
    /* free resources */
    wait_cudafunc(cudaFree(d_A));
    wait_cudafunc(cudaFree(d_W));
    wait_cudafunc(cudaFree(d_info));
    wait_cudafunc(cudaFree(d_work));
    free(h_work);

    wait_cudafunc(cusolverDnDestroy(cusolverH));

    wait_cudafunc(cudaStreamDestroy(stream));

    return info;
}

int cusolver_Syevdx_openacc(double * A, double * W, int32_t m, int32_t MaxN)
{
    int32_t deviceCount;
    wait_cudafunc(cudaGetDeviceCount(&deviceCount));

    int32_t rank;
    MPI_Comm_rank(mpi_comm_level1, &rank);
    wait_cudafunc(cudaSetDevice(rank % deviceCount));

    // OpenACC
    int local_numdevices = acc_get_num_devices(acc_device_nvidia);
    acc_set_device_num(rank % local_numdevices, acc_device_nvidia);

    // printf("OK %s:%d\n", __FILE__, __LINE__);
    // double *d_A = NULL;
    // double *d_W = NULL;
    // int* d_info = NULL;

    int32_t * pinfo = (int32_t *)malloc(sizeof(int32_t));

    // printf("OK %s:%d\n", __FILE__, __LINE__);

    /* step 1: create cusolver handle, bind a stream */
#pragma acc data      copy(pinfo[0 : 1])
#pragma acc data      present(A[0 : m * m])
#pragma acc data      deviceptr(W)
#pragma acc host_data use_device(A, W, pinfo)
    {
        cusolverDnHandle_t cusolverH = NULL;
        // printf("OK %s:%d\n", __FILE__, __LINE__);
        wait_cudafunc(cusolverDnCreate(&cusolverH));

        cudaStream_t stream = NULL;
        wait_cudafunc(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        wait_cudafunc(cusolverDnSetStream(cusolverH, stream));

        // printf("OK %s:%d\n", __FILE__, __LINE__);
        //  wait_cudafunc(cudaMalloc((void **)(&d_A), sizeof(double) * lda * m));
        //  wait_cudafunc(cudaMalloc((void **)(&d_W), sizeof(double) * m));
        // wait_cudafunc(cudaMalloc((void **)(&d_info), sizeof(int)));

        // printf("OK %s:%d\n", __FILE__, __LINE__);
        //  wait_cudafunc(cudaMemcpyAsync(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice,
        //                             stream));

        // step 3: query working space of syevd
        cusolverEigMode_t const  jobz  = CUSOLVER_EIG_MODE_VECTOR;  // compute eigenvalues and eigenvectors.
        cublasFillMode_t const   uplo  = CUBLAS_FILL_MODE_LOWER;
        cusolverEigRange_t const range = CUSOLVER_EIG_RANGE_I;

        int32_t const lda = m;
        double        vl;
        double        vu;
        int64_t       h_meig;

        size_t workspaceInBytesOnDevice = 0; /* size of workspace */
        size_t workspaceInBytesOnHost   = 0; /* size of workspace */

        // printf("OK %s:%d\n", __FILE__, __LINE__);
        wait_cudafunc(cusolverDnXsyevdx_bufferSize(cusolverH, NULL, jobz, range, uplo, m, CUDA_R_64F, A, lda, &vl, &vu,
                                                   1L, (long int)(MaxN), &h_meig, CUDA_R_64F, W, CUDA_R_64F,
                                                   &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

        long long memorysize = get_gpu_total_memory_in_bytes();
        long long datasize   = (long long)sizeof(double) * (long long)m * (long long)m +
                             (long long)sizeof(double) * (long long)(m + 1) + (long long)workspaceInBytesOnDevice;

        if (memorysize < datasize) {
            printf("There's not enough memory on the device (GPU) to continue processing!");
            exit(1);
        }

        void * d_work = NULL; /* device workspace */

        // printf("OK %s:%d\n", __FILE__, __LINE__);
        wait_cudafunc(cudaMalloc((void **)(&d_work), workspaceInBytesOnDevice));

        void * h_work = NULL; /* host workspace for */

        h_work = malloc(workspaceInBytesOnHost);
        if (!h_work) {
            fprintf(stderr, "Could not allocate host memory.\n");
            exit(1);
        }

        // printf("OK %s:%d\n", __FILE__, __LINE__);
        //  step 4: compute spectrum
        wait_cudafunc(cusolverDnXsyevdx(cusolverH, NULL, jobz, range, uplo, m, CUDA_R_64F, A, lda, &vl, &vu, 1L,
                                        (long int)(MaxN), &h_meig, CUDA_R_64F, W, CUDA_R_64F, d_work,
                                        workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, pinfo));

        // wait_cudafunc(cudaMemcpyAsync(A, d_A, sizeof(double) * lda * m, cudaMemcpyDeviceToHost,
        //                            stream));
        // wait_cudafunc(cudaMemcpyAsync(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost,
        //                            stream));
        // wait_cudafunc(cudaMemcpyAsync(&pinfo, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
        ////printf("OK %s:%d\n", __FILE__, __LINE__);
        wait_cudafunc(cudaStreamSynchronize(stream));
        /* free resources */
        // wait_cudafunc(cudaFree(d_A));
        // wait_cudafunc(cudaFree(d_W));
        // wait_cudafunc(cudaFree(d_info));
        wait_cudafunc(cudaFree(d_work));
        free(h_work);
        // printf("OK %s:%d\n", __FILE__, __LINE__);
        wait_cudafunc(cusolverDnDestroy(cusolverH));
        // printf("OK %s:%d\n", __FILE__, __LINE__);
        wait_cudafunc(cudaStreamDestroy(stream));
    }

    int32_t const info = *pinfo;
    free(pinfo);

    return info;
}

int cusolver_Syevdx_Complex(dcomplex * A, double * W, int m, int MaxN)
{
    const int lda = m;

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t       stream    = NULL;
    int                deviceCount;
    wait_cudafunc(cudaGetDeviceCount(&deviceCount));

    int rank;
    MPI_Comm_rank(mpi_comm_level1, &rank);
    wait_cudafunc(cudaSetDevice(rank % deviceCount));

    cuDoubleComplex * d_A    = NULL;
    double *          d_W    = NULL;
    double            vl     = 0.0;
    double            vu     = 0.0;
    int64_t           h_meig = 0;
    int *             d_info = NULL;

    int info = 0;

    size_t workspaceInBytesOnDevice = 0;    /* size of workspace */
    void * d_work                   = NULL; /* device workspace */
    size_t workspaceInBytesOnHost   = 0;    /* size of workspace */
    void * h_work                   = NULL; /* host workspace for */

    /* step 1: create cusolver handle, bind a stream */
    wait_cudafunc(cusolverDnCreate(&cusolverH));

    wait_cudafunc(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    wait_cudafunc(cusolverDnSetStream(cusolverH, stream));

    wait_cudafunc(cudaMalloc((void **)(&d_A), sizeof(cuDoubleComplex) * lda * m));
    wait_cudafunc(cudaMalloc((void **)(&d_W), sizeof(double) * m));
    wait_cudafunc(cudaMalloc((void **)(&d_info), sizeof(int)));

    wait_cudafunc(cudaMemcpyAsync(d_A, A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice, stream));

    // step 3: query working space of syevd
    cusolverEigMode_t  jobz  = CUSOLVER_EIG_MODE_VECTOR;  // compute eigenvalues and eigenvectors.
    cublasFillMode_t   uplo  = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;

    wait_cudafunc(cusolverDnXsyevdx_bufferSize(cusolverH, NULL, jobz, range, uplo, m, CUDA_C_64F, d_A, lda, &vl, &vu,
                                               1L, (long int)(MaxN), &h_meig, CUDA_R_64F, d_W, CUDA_C_64F,
                                               &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    wait_cudafunc(cudaMalloc((void **)(&d_work), workspaceInBytesOnDevice));
    h_work = malloc(workspaceInBytesOnHost);
    if (!h_work) {
        fprintf(stderr, "Could not allocate host memory.\n");
        exit(1);
    }

    // step 4: compute spectrum
    wait_cudafunc(cusolverDnXsyevdx(cusolverH, NULL, jobz, range, uplo, m, CUDA_C_64F, d_A, lda, &vl, &vu, 1L,
                                    (long int)(MaxN), &h_meig, CUDA_R_64F, d_W, CUDA_C_64F, d_work,
                                    workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));

    wait_cudafunc(cudaMemcpyAsync(A, d_A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyDeviceToHost, stream));
    wait_cudafunc(cudaMemcpyAsync(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost, stream));
    wait_cudafunc(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    wait_cudafunc(cudaStreamSynchronize(stream));
    /* free resources */
    wait_cudafunc(cudaFree(d_A));
    wait_cudafunc(cudaFree(d_W));
    wait_cudafunc(cudaFree(d_info));
    wait_cudafunc(cudaFree(d_work));
    free(h_work);

    wait_cudafunc(cusolverDnDestroy(cusolverH));

    wait_cudafunc(cudaStreamDestroy(stream));

    return info;
}

int cusolver_Syevdx_Complex_openacc(dcomplex * A, double * W, int m, int MaxN)
{
    const int lda = m;

    cusolverDnHandle_t * cusolverH = NULL;
    cudaStream_t *       stream    = NULL;
    int                  deviceCount;
    wait_cudafunc(cudaGetDeviceCount(&deviceCount));

    int rank;
    MPI_Comm_rank(mpi_comm_level1, &rank);
    wait_cudafunc(cudaSetDevice((rank % deviceCount)));

    // OpenACC
    int local_numdevices = acc_get_num_devices(acc_device_nvidia);
    acc_set_device_num(rank % local_numdevices, acc_device_nvidia);

    // cuDoubleComplex* d_A = NULL;
    // double* d_W = NULL;
    double  vl     = 0.0;
    double  vu     = 0.0;
    int64_t h_meig = 0;
    // int* d_info = NULL;

    int *pinfo, info;
    pinfo = (int *)malloc(sizeof(int));

    size_t workspaceInBytesOnDevice = 0;    /* size of workspace */
    void * d_work                   = NULL; /* device workspace */
    size_t workspaceInBytesOnHost   = 0;    /* size of workspace */
    void * h_work                   = NULL; /* host workspace for */

    /* step 1: create cusolver handle, bind a stream */
#pragma acc data      copy(pinfo[0 : 1])
#pragma acc data      present(A[0 : m * m])
#pragma acc data      deviceptr(W)
#pragma acc host_data use_device(A, W, pinfo)
    {
        wait_cudafunc(cusolverDnCreate(&cusolverH));
        wait_cudafunc(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        wait_cudafunc(cusolverDnSetStream(cusolverH, stream));

        // wait_cudafunc(cudaMallocAsync((void**)(&d_A), sizeof(cuDoubleComplex) * lda * m, stream));
        // wait_cudafunc(cudaMallocAsync((void**)(&d_W), sizeof(double) * m, stream));
        // wait_cudafunc(cudaMallocAsync((void**)(&d_info), sizeof(int), stream));

        // wait_cudafunc(cudaMemcpyAsync(d_A, A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice,
        // stream));

        // step 3: query working space of syevd
        cusolverEigMode_t  jobz = CUSOLVER_EIG_MODE_VECTOR;  // compute eigenvalues and eigenvectors.
        cublasFillMode_t   uplo = CUBLAS_FILL_MODE_LOWER;
        cusolverEigRange_t range;
        if (m == MaxN) {
            range = CUSOLVER_EIG_RANGE_ALL;
        } else {
            range = CUSOLVER_EIG_RANGE_I;
        }

        wait_cudafunc(cusolverDnXsyevdx_bufferSize(cusolverH, NULL, jobz, range, uplo, m, CUDA_C_64F, A, lda, &vl, &vu,
                                                   1L, (long int)(MaxN), &h_meig, CUDA_R_64F, W, CUDA_C_64F,
                                                   &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

        long long memorysize = get_gpu_total_memory_in_bytes();
        long long datasize   = (long long)sizeof(dcomplex) * (long long)m * (long long)m +
                             (long long)sizeof(double) * (long long)(m + 1) + (long long)workspaceInBytesOnDevice;

        if (memorysize < datasize) {
            printf("There's not enough memory on the device (GPU) to continue processing!");
            exit(1);
        }

        wait_cudafunc(cudaMallocAsync((void **)(&d_work), workspaceInBytesOnDevice, stream));

        h_work = malloc(workspaceInBytesOnHost);
        if (!h_work) {
            fprintf(stderr, "Could not allocate host memory.\n");
            exit(1);
        }

        // step 4: compute spectrum
        wait_cudafunc(cusolverDnXsyevdx(cusolverH, NULL, jobz, range, uplo, m, CUDA_C_64F, A, lda, &vl, &vu, 1L,
                                        (long int)(MaxN), &h_meig, CUDA_R_64F, W, CUDA_C_64F, d_work,
                                        workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, pinfo));

        // wait_cudafunc(cudaMemcpyAsync(A, d_A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyDeviceToHost,
        //     stream));

        // wait_cudafunc(cudaMemcpyAsync(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost,
        //     stream));

        // wait_cudafunc(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

        /* free resources */
        // wait_cudafunc(cudaFreeAsync(d_A, stream));
        // wait_cudafunc(cudaFreeAsync(d_W, stream));
        // wait_cudafunc(cudaFreeAsync(d_info, stream));
        wait_cudafunc(cudaFreeAsync(d_work, stream));

        if (h_work != NULL) {
            free(h_work);
        }

        wait_cudafunc(cudaStreamSynchronize(stream));
        wait_cudafunc(cusolverDnDestroy(cusolverH));
        wait_cudafunc(cudaStreamDestroy(stream));
    }

    info = *pinfo;
    free(pinfo);

    return info;
}

/*
 * This function retrieves the total memory of "GPU 0" from nvidia-smi output
 * in an MPI environment. It uses popen("nvidia-smi") to capture the output
 * and writes it to a temporary file named with the MPI rank.
 *
 * Then it scans each line to find a pattern "xxxMiB / yyyMiB". Specifically,
 * it looks for the substring "MiB /", and from that point, it parses the
 * total memory (yyy) with sscanf.
 *
 * Returns:
 *   - A positive long long value representing the total memory (in bytes).
 *   - -1 if no memory info could be found or if an error occurs.
 */
long long get_gpu_total_memory_in_bytes()
{
    int32_t deviceCount;
    wait_cudafunc(cudaGetDeviceCount(&deviceCount));

    int32_t rank;
    MPI_Comm_rank(mpi_comm_level1, &rank);
    wait_cudafunc(cudaSetDevice(rank % deviceCount));

    // メモリ情報を取得
    size_t freeMem  = 0;
    size_t totalMem = 0;
    wait_cudafunc(cudaMemGetInfo(&freeMem, &totalMem));

    return (long long)totalMem;
}