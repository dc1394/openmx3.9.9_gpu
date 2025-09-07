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
#include <stdint.h>
#include <stdlib.h>

int32_t cusolver_Syevd(double * A, double * W, int32_t m)
{
    int32_t deviceCount;
    wait_cudafunc(cudaGetDeviceCount(&deviceCount));

    int32_t rank;
    MPI_Comm_rank(mpi_comm_level1, &rank);
    wait_cudafunc(cudaSetDevice(rank % deviceCount));

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t       stream    = NULL;

    /* step 1: create cusolver handle, bind a stream */
    wait_cudafunc(cusolverDnCreate(&cusolverH));

    wait_cudafunc(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    wait_cudafunc(cusolverDnSetStream(cusolverH, stream));

    int32_t const lda    = m;
    double *      d_A    = NULL;
    double *      d_W    = NULL;
    int32_t *     d_info = NULL;

    wait_cudafunc(cudaMalloc((void **)(&d_A), sizeof(double) * lda * m));
    wait_cudafunc(cudaMalloc((void **)(&d_W), sizeof(double) * m));
    wait_cudafunc(cudaMalloc((void **)(&d_info), sizeof(int32_t)));

    wait_cudafunc(cudaMemcpyAsync(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice, stream));

    // step 3: query working space of syevd
    cusolverEigMode_t const jobz = CUSOLVER_EIG_MODE_VECTOR;  // compute eigenvalues and eigenvectors.
    cublasFillMode_t  const uplo = CUBLAS_FILL_MODE_LOWER;

    size_t d_lwork = 0; /* size of workspace */
    size_t h_lwork = 0; /* size of workspace */

    wait_cudafunc(cusolverDnXsyevd_bufferSize(cusolverH, NULL, jobz, uplo, m, CUDA_R_64F, d_A, lda, CUDA_R_64F, d_W,
                                              CUDA_R_64F, &d_lwork, &h_lwork));

    void * d_work = NULL; /* device workspace */

    wait_cudafunc(cudaMalloc((void **)(&d_work), d_lwork));

    // host workspace for
    void * h_work = malloc(h_lwork);
    if (!h_work) {
        fprintf(stderr, "Could not allocate host memory.\n");
        exit(1);
    }

    // step 4: compute spectrum
    wait_cudafunc(cusolverDnXsyevd(cusolverH, NULL, jobz, uplo, m, CUDA_R_64F, d_A, lda, CUDA_R_64F, d_W, CUDA_R_64F,
                                   d_work, d_lwork, h_work, h_lwork, d_info));

    wait_cudafunc(cudaMemcpyAsync(A, d_A, sizeof(double) * lda * m, cudaMemcpyDeviceToHost, stream));
    wait_cudafunc(cudaMemcpyAsync(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost, stream));

    int32_t info;

    wait_cudafunc(cudaMemcpyAsync(&info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));

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
