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
#include "set_cuda_default_device_from_local_rank.h"
#include "set_openacc_device_from_local_rank.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <openacc.h>
#include <stdint.h>
#include <stdio.h>

int32_t gpusolver_Syevdx(double * A, double * W, int32_t m, int32_t MaxN)
{
    int32_t const lda = m;

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t       stream    = NULL;

    double *  d_A = NULL;
    double *  d_W = NULL;
    double    vl     = 0.0;
    double    vu     = 0.0;
    int64_t   h_meig = 0;
    int32_t * d_info = NULL;

    int32_t info = 0;

    size_t workspaceInBytesOnDevice = 0;    /* size of workspace */
    void * d_work                   = NULL; /* device workspace */
    size_t workspaceInBytesOnHost   = 0;    /* size of workspace */
    void * h_work                   = NULL; /* host workspace for */

    /* step 1: create gpusolver handle, bind a stream */
    wait_cudafunc(cusolverDnCreate(&cusolverH));

    wait_cudafunc(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    wait_cudafunc(cusolverDnSetStream(cusolverH, stream));

    wait_cudafunc(cudaMalloc((void **)(&d_A), sizeof(double) * lda * m));
    wait_cudafunc(cudaMalloc((void **)(&d_W), sizeof(double) * m));
    wait_cudafunc(cudaMalloc((void **)(&d_info), sizeof(int32_t)));

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
                                               1L, MaxN, &h_meig, CUDA_R_64F, d_W, CUDA_R_64F,
                                               &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    wait_cudafunc(cudaMalloc((void **)(&d_work), workspaceInBytesOnDevice));
    h_work = (workspaceInBytesOnHost == 0) ? NULL : malloc(workspaceInBytesOnHost);
    if (workspaceInBytesOnHost != 0 && !h_work) {
        fprintf(stderr, "Could not allocate host memory.\n");
        exit(1);
    }

    // step 4: compute spectrum
    wait_cudafunc(cusolverDnXsyevdx(cusolverH, NULL, jobz, range, uplo, m, CUDA_R_64F, d_A, lda, &vl, &vu, 1L, MaxN,
                                    &h_meig, CUDA_R_64F, d_W, CUDA_R_64F, d_work, workspaceInBytesOnDevice, h_work,
                                    workspaceInBytesOnHost, d_info));

    wait_cudafunc(cudaMemcpyAsync(A, d_A, sizeof(double) * lda * m, cudaMemcpyDeviceToHost, stream));
    wait_cudafunc(cudaMemcpyAsync(W, d_W, sizeof(double) * MaxN, cudaMemcpyDeviceToHost, stream));
    wait_cudafunc(cudaMemcpyAsync(&info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));

    wait_cudafunc(cudaStreamSynchronize(stream));
    /* free resources */
    wait_cudafunc(cudaFree(d_A));
    wait_cudafunc(cudaFree(d_W));
    wait_cudafunc(cudaFree(d_info));
    wait_cudafunc(cudaFree(d_work));
    if (h_work != NULL) free(h_work);

    wait_cudafunc(cusolverDnDestroy(cusolverH));

    wait_cudafunc(cudaStreamDestroy(stream));

    return info;
}

int32_t gpusolver_Syevdx_openacc(double * A, double * W, int32_t m, int32_t MaxN)
{
    int32_t info = 0;

    /* step 1: create gpusolver handle, bind a stream */
#pragma acc data      present(A[0 : m * m])
#pragma acc data      present(W[0 : MaxN])
#pragma acc host_data use_device(A, W)
    {
        cusolverDnHandle_t cusolverH = NULL;

        wait_cudafunc(cusolverDnCreate(&cusolverH));

        cudaStream_t stream = NULL;
        wait_cudafunc(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        wait_cudafunc(cusolverDnSetStream(cusolverH, stream));

        //  wait_cudafunc(cudaMalloc((void **)(&d_A), sizeof(double) * lda * m));
        //  wait_cudafunc(cudaMalloc((void **)(&d_W), sizeof(double) * m));
        // wait_cudafunc(cudaMalloc((void **)(&d_info), sizeof(int32_t)));

        //  wait_cudafunc(cudaMemcpyAsync(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice,
        //                             stream));

        // step 3: query working space of syevd
        cusolverEigMode_t const  jobz  = CUSOLVER_EIG_MODE_VECTOR;  // compute eigenvalues and eigenvectors.
        cublasFillMode_t const   uplo  = CUBLAS_FILL_MODE_LOWER;
        cusolverEigRange_t const range = CUSOLVER_EIG_RANGE_I;

        int32_t const lda = m;
        double        vl     = 0.0;
        double        vu     = 0.0;
        int64_t       h_meig = 0;

        size_t workspaceInBytesOnDevice = 0; /* size of workspace */
        size_t workspaceInBytesOnHost   = 0; /* size of workspace */
        int32_t *d_info = NULL;

        wait_cudafunc(cusolverDnXsyevdx_bufferSize(cusolverH, NULL, jobz, range, uplo, m, CUDA_R_64F, A, lda, &vl, &vu,
                                                   1L, MaxN, &h_meig, CUDA_R_64F, W, CUDA_R_64F,
                                                   &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

        void * d_work = NULL; /* device workspace */

        wait_cudafunc(cudaMalloc((void **)(&d_work), workspaceInBytesOnDevice));
        wait_cudafunc(cudaMalloc((void **)(&d_info), sizeof(int32_t)));

        void * h_work = NULL; /* host workspace for */

        h_work = (workspaceInBytesOnHost == 0) ? NULL : malloc(workspaceInBytesOnHost);
        if (workspaceInBytesOnHost != 0 && !h_work) {
            fprintf(stderr, "Could not allocate host memory.\n");
            exit(1);
        }

        //  step 4: compute spectrum
        wait_cudafunc(cusolverDnXsyevdx(cusolverH, NULL, jobz, range, uplo, m, CUDA_R_64F, A, lda, &vl, &vu, 1L, MaxN,
                                        &h_meig, CUDA_R_64F, W, CUDA_R_64F, d_work, workspaceInBytesOnDevice, h_work,
                                        workspaceInBytesOnHost, d_info));

        // wait_cudafunc(cudaMemcpyAsync(A, d_A, sizeof(double) * lda * m, cudaMemcpyDeviceToHost,
        //                            stream));
        // wait_cudafunc(cudaMemcpyAsync(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost,
        //                            stream));
        wait_cudafunc(cudaMemcpyAsync(&info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        wait_cudafunc(cudaStreamSynchronize(stream));
        /* free resources */
        // wait_cudafunc(cudaFree(d_A));
        // wait_cudafunc(cudaFree(d_W));
        wait_cudafunc(cudaFree(d_info));
        wait_cudafunc(cudaFree(d_work));
        if (h_work != NULL) free(h_work);

        wait_cudafunc(cusolverDnDestroy(cusolverH));
        wait_cudafunc(cudaStreamDestroy(stream));
    }

    return info;
}

int32_t gpusolver_Syevdx_Complex(dcomplex * A, double * W, int32_t m, int32_t MaxN)
{
    int32_t const lda = m;

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t       stream    = NULL;

    cuDoubleComplex * d_A    = NULL;
    double *          d_W    = NULL;
    double            vl     = 0.0;
    double            vu     = 0.0;
    int64_t           h_meig = 0;
    int32_t *         d_info = NULL;

    int32_t info = 0;

    size_t workspaceInBytesOnDevice = 0;    /* size of workspace */
    void * d_work                   = NULL; /* device workspace */
    size_t workspaceInBytesOnHost   = 0;    /* size of workspace */
    void * h_work                   = NULL; /* host workspace for */

    /* step 1: create gpusolver handle, bind a stream */
    wait_cudafunc(cusolverDnCreate(&cusolverH));

    wait_cudafunc(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    wait_cudafunc(cusolverDnSetStream(cusolverH, stream));

    wait_cudafunc(cudaMalloc((void **)(&d_A), sizeof(cuDoubleComplex) * lda * m));
    wait_cudafunc(cudaMalloc((void **)(&d_W), sizeof(double) * m));
    wait_cudafunc(cudaMalloc((void **)(&d_info), sizeof(int32_t)));

    wait_cudafunc(cudaMemcpyAsync(d_A, A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice, stream));

    // step 3: query working space of syevd
    cusolverEigMode_t  jobz  = CUSOLVER_EIG_MODE_VECTOR;  // compute eigenvalues and eigenvectors.
    cublasFillMode_t   uplo  = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;

    wait_cudafunc(cusolverDnXsyevdx_bufferSize(cusolverH, NULL, jobz, range, uplo, m, CUDA_C_64F, d_A, lda, &vl, &vu,
                                               1L, MaxN, &h_meig, CUDA_R_64F, d_W, CUDA_C_64F,
                                               &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    wait_cudafunc(cudaMalloc((void **)(&d_work), workspaceInBytesOnDevice));
    h_work = (workspaceInBytesOnHost == 0) ? NULL : malloc(workspaceInBytesOnHost);
    if (workspaceInBytesOnHost != 0 && !h_work) {
        fprintf(stderr, "Could not allocate host memory.\n");
        exit(1);
    }

    // step 4: compute spectrum
    wait_cudafunc(cusolverDnXsyevdx(cusolverH, NULL, jobz, range, uplo, m, CUDA_C_64F, d_A, lda, &vl, &vu, 1L, MaxN,
                                    &h_meig, CUDA_R_64F, d_W, CUDA_C_64F, d_work, workspaceInBytesOnDevice, h_work,
                                    workspaceInBytesOnHost, d_info));

    wait_cudafunc(cudaMemcpyAsync(A, d_A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyDeviceToHost, stream));
    wait_cudafunc(cudaMemcpyAsync(W, d_W, sizeof(double) * MaxN, cudaMemcpyDeviceToHost, stream));
    wait_cudafunc(cudaMemcpyAsync(&info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));

    wait_cudafunc(cudaStreamSynchronize(stream));
    /* free resources */
    wait_cudafunc(cudaFree(d_A));
    wait_cudafunc(cudaFree(d_W));
    wait_cudafunc(cudaFree(d_info));
    wait_cudafunc(cudaFree(d_work));
    if (h_work != NULL) free(h_work);

    wait_cudafunc(cusolverDnDestroy(cusolverH));

    wait_cudafunc(cudaStreamDestroy(stream));

    return info;
}

int32_t gpusolver_Syevdx_Complex_openacc(dcomplex * A, double * W, int32_t m, int32_t MaxN)
{
    int32_t const lda = m;

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t       stream    = NULL;
    //set_cuda_default_device_from_local_rank();

    // OpenACC
    //set_openacc_nvidia_device_from_local_rank();

    // cuDoubleComplex* d_A = NULL;
    // double* d_W = NULL;
    double  vl     = 0.0;
    double  vu     = 0.0;
    int64_t h_meig = 0;
    // int32_t* d_info = NULL;

    int32_t info = 0;
    int32_t *d_info = NULL;

    size_t workspaceInBytesOnDevice = 0;    /* size of workspace */
    void * d_work                   = NULL; /* device workspace */
    size_t workspaceInBytesOnHost   = 0;    /* size of workspace */
    void * h_work                   = NULL; /* host workspace for */

    /* step 1: create gpusolver handle, bind a stream */
#pragma acc data      present(A[0 : m * m])
#pragma acc data      present(W[0 : MaxN])
#pragma acc host_data use_device(A, W)
    {
        wait_cudafunc(cusolverDnCreate(&cusolverH));
        wait_cudafunc(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        wait_cudafunc(cusolverDnSetStream(cusolverH, stream));

        // wait_cudafunc(cudaMallocAsync((void**)(&d_A), sizeof(cuDoubleComplex) * lda * m, stream));
        // wait_cudafunc(cudaMallocAsync((void**)(&d_W), sizeof(double) * m, stream));
        // wait_cudafunc(cudaMallocAsync((void**)(&d_info), sizeof(int32_t), stream));

        // wait_cudafunc(cudaMemcpyAsync(d_A, A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice,
        // stream));

        // step 3: query working space of syevd
        cusolverEigMode_t const jobz = CUSOLVER_EIG_MODE_VECTOR;  // compute eigenvalues and eigenvectors.
        cublasFillMode_t const  uplo = CUBLAS_FILL_MODE_LOWER;
        cusolverEigRange_t      range;
        if (m == MaxN) {
            range = CUSOLVER_EIG_RANGE_ALL;
        } else {
            range = CUSOLVER_EIG_RANGE_I;
        }

        wait_cudafunc(cusolverDnXsyevdx_bufferSize(cusolverH, NULL, jobz, range, uplo, m, CUDA_C_64F, A, lda, &vl, &vu,
                                                   1L, MaxN, &h_meig, CUDA_R_64F, W, CUDA_C_64F,
                                                   &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

        wait_cudafunc(cudaMallocAsync((void **)(&d_work), workspaceInBytesOnDevice, stream));
        wait_cudafunc(cudaMallocAsync((void **)(&d_info), sizeof(int32_t), stream));

        h_work = (workspaceInBytesOnHost == 0) ? NULL : malloc(workspaceInBytesOnHost);
        if (workspaceInBytesOnHost != 0 && !h_work) {
            fprintf(stderr, "Could not allocate host memory.\n");
            exit(1);
        }

        // step 4: compute spectrum
        wait_cudafunc(cusolverDnXsyevdx(cusolverH, NULL, jobz, range, uplo, m, CUDA_C_64F, A, lda, &vl, &vu, 1L, MaxN,
                                        &h_meig, CUDA_R_64F, W, CUDA_C_64F, d_work, workspaceInBytesOnDevice, h_work,
                                        workspaceInBytesOnHost, d_info));

        // wait_cudafunc(cudaMemcpyAsync(A, d_A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyDeviceToHost,
        //     stream));

        // wait_cudafunc(cudaMemcpyAsync(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost,
        //     stream));

        wait_cudafunc(cudaMemcpyAsync(&info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));

        /* free resources */
        // wait_cudafunc(cudaFreeAsync(d_A, stream));
        // wait_cudafunc(cudaFreeAsync(d_W, stream));
        // wait_cudafunc(cudaFreeAsync(d_info, stream));
        wait_cudafunc(cudaStreamSynchronize(stream));

        wait_cudafunc(cudaFree(d_work));
        wait_cudafunc(cudaFree(d_info));

        if (h_work != NULL) {
            free(h_work);
        }

        wait_cudafunc(cusolverDnDestroy(cusolverH));
        wait_cudafunc(cudaStreamDestroy(stream));
    }

    return info;
}

/* Same contract as gpusolver_Syevdx_Complex_openacc (A and W must already be
   present on the device), but the cusolver handle, stream and workspaces are
   cached across calls.  The divide-conquer solver calls this once per atom,
   so recreating the handle each time dominated small solves. */
typedef struct {
    int                initialized;
    int                device_id;
    size_t             d_work_bytes;
    size_t             h_work_bytes;
    cusolverDnHandle_t handle;
    cudaStream_t       stream;
    void *             d_work;
    void *             h_work;
    int32_t *          d_info;
} GpusolverSyevdxCachedCtx;

static GpusolverSyevdxCachedCtx gpusolver_syevdx_cached_ctx = {0};

static void gpusolver_Syevdx_Cached_Destroy(void)
{
    GpusolverSyevdxCachedCtx *ctx = &gpusolver_syevdx_cached_ctx;

    if (ctx->d_work != NULL)
        wait_cudafunc(cudaFree(ctx->d_work));
    if (ctx->d_info != NULL)
        wait_cudafunc(cudaFree(ctx->d_info));
    if (ctx->h_work != NULL)
        free(ctx->h_work);
    if (ctx->handle != NULL)
        wait_cudafunc(cusolverDnDestroy(ctx->handle));
    if (ctx->stream != NULL)
        wait_cudafunc(cudaStreamDestroy(ctx->stream));

    ctx->initialized  = 0;
    ctx->device_id    = -1;
    ctx->d_work_bytes = 0;
    ctx->h_work_bytes = 0;
    ctx->handle       = NULL;
    ctx->stream       = NULL;
    ctx->d_work       = NULL;
    ctx->h_work       = NULL;
    ctx->d_info       = NULL;
}

static void gpusolver_Syevdx_Cached_Init(void)
{
    GpusolverSyevdxCachedCtx *ctx = &gpusolver_syevdx_cached_ctx;
    int                       current_device;

    wait_cudafunc(cudaGetDevice(&current_device));

    if (ctx->initialized && ctx->device_id == current_device) {
        return;
    }

    if (ctx->initialized) {
        gpusolver_Syevdx_Cached_Destroy();
    }

    wait_cudafunc(cusolverDnCreate(&ctx->handle));
    wait_cudafunc(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));
    wait_cudafunc(cusolverDnSetStream(ctx->handle, ctx->stream));
    wait_cudafunc(cudaMalloc((void **)&ctx->d_info, sizeof(int32_t)));

    ctx->initialized = 1;
    ctx->device_id   = current_device;
}

int32_t gpusolver_Syevdx_Complex_openacc_cached(dcomplex * A, double * W, int32_t m, int32_t MaxN)
{
    GpusolverSyevdxCachedCtx *ctx = &gpusolver_syevdx_cached_ctx;
    int32_t const             lda = m;

    double  vl     = 0.0;
    double  vu     = 0.0;
    int64_t h_meig = 0;
    int32_t info   = 0;

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost   = 0;

    gpusolver_Syevdx_Cached_Init();

#pragma acc data      present(A[0 : m * m])
#pragma acc data      present(W[0 : MaxN])
#pragma acc host_data use_device(A, W)
    {
        cusolverEigMode_t const jobz = CUSOLVER_EIG_MODE_VECTOR;
        cublasFillMode_t const  uplo = CUBLAS_FILL_MODE_LOWER;
        cusolverEigRange_t      range;
        if (m == MaxN) {
            range = CUSOLVER_EIG_RANGE_ALL;
        } else {
            range = CUSOLVER_EIG_RANGE_I;
        }

        wait_cudafunc(cusolverDnXsyevdx_bufferSize(ctx->handle, NULL, jobz, range, uplo, m, CUDA_C_64F, A, lda, &vl,
                                                   &vu, 1L, MaxN, &h_meig, CUDA_R_64F, W, CUDA_C_64F,
                                                   &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

        if (workspaceInBytesOnDevice > ctx->d_work_bytes) {
            if (ctx->d_work != NULL)
                wait_cudafunc(cudaFree(ctx->d_work));
            ctx->d_work = NULL;
            wait_cudafunc(cudaMalloc((void **)&ctx->d_work, workspaceInBytesOnDevice));
            ctx->d_work_bytes = workspaceInBytesOnDevice;
        }
        if (workspaceInBytesOnHost > ctx->h_work_bytes) {
            if (ctx->h_work != NULL)
                free(ctx->h_work);
            ctx->h_work = malloc(workspaceInBytesOnHost);
            if (!ctx->h_work) {
                fprintf(stderr, "Could not allocate host memory.\n");
                exit(1);
            }
            ctx->h_work_bytes = workspaceInBytesOnHost;
        }

        wait_cudafunc(cusolverDnXsyevdx(ctx->handle, NULL, jobz, range, uplo, m, CUDA_C_64F, A, lda, &vl, &vu, 1L,
                                        MaxN, &h_meig, CUDA_R_64F, W, CUDA_C_64F, ctx->d_work,
                                        workspaceInBytesOnDevice, ctx->h_work, workspaceInBytesOnHost, ctx->d_info));

        wait_cudafunc(cudaMemcpyAsync(&info, ctx->d_info, sizeof(int32_t), cudaMemcpyDeviceToHost, ctx->stream));
        wait_cudafunc(cudaStreamSynchronize(ctx->stream));
    }

    return info;
}
