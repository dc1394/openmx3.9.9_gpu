/**********************************************************************
  Divide_Conquer_LNO.c

  Collinear-focused version with node-local GPU-owner proxy for DC-LNO.

  Policy:
    - Keep global OpenMX MPI decomposition unchanged.
    - Inside DC-LNO only, do not split one eigenproblem across many MPI ranks.
    - For large local cluster matrices:
        * ranks sharing the same GPU form a node-local GPU-group
        * rank 0 in that GPU-group is the GPU owner
        * non-owner ranks send the dense eigen task to the owner
    - For small matrices: solve locally on CPU
    - Collinear only is implemented here.
    - Noncollinear entry points are kept as compile stubs.

***********************************************************************/

#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "set_cuda_default_device_from_local_rank.h"
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define measure_time 0
#define GPU_CPU_SWITCH_NUM2 800

/* task-level threshold for proxy-to-GPU */
#define DCLNO_GPU_PROXY_EIGEN_THRESHOLD_COL  (GPU_CPU_SWITCH_NUM2)

static int DCLNO_GpuEigenThreshold(void)
{
    static int threshold = -1;
    if (threshold < 0) {
        const char *value = getenv("OPENMX_DCLNO_GPU_THRESHOLD");
        threshold = DCLNO_GPU_PROXY_EIGEN_THRESHOLD_COL;
        if (value != NULL && value[0] != '\0') {
            int parsed = atoi(value);
            if (0 < parsed) threshold = parsed;
        }
    }
    return threshold;
}

#define DCLNO_NONCOL_GPU_CPU_SWITCH_NUM 400

static int DCLNO_NonColGpuEigenThreshold(void)
{
    static int threshold = -1;
    if (threshold < 0) {
        const char *value = getenv("OPENMX_DCLNO_NONCOL_GPU_THRESHOLD");
        threshold = DCLNO_NONCOL_GPU_CPU_SWITCH_NUM;
        if (value != NULL && value[0] != '\0') {
            int parsed = atoi(value);
            if (0 < parsed) threshold = parsed;
        }
    }
    return threshold;
}

/* MPI tags for node-local proxy traffic */
#define DCLNO_PROXY_TAG_COL_S      41001
#define DCLNO_PROXY_TAG_COL_H      41002
#define DCLNO_PROXY_TAG_COL_EVAL   41003
#define DCLNO_PROXY_TAG_COL_CVEC   41004

/* ------------------------------------------------------------------ */
/* forward declarations                                               */
/* ------------------------------------------------------------------ */

static double DC_Col(char * mode, int MD_iter, int SCF_iter, int SucceedReadingDMfile,
                     double ***** Hks, double **** OLP0, double ***** CDM, double ***** EDM,
                     double Eele0[2], double Eele1[2]);

static double DC_NonCol(char * mode, int MD_iter, int SCF_iter, int SucceedReadingDMfile,
                        double ***** Hks, double ***** ImNL, double **** OLP0,
                        double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2]);

static void Save_DOS_Col(double ****** Residues, double **** OLP0, double *** EVal,
                         int ** LO_TC, int ** HO_TC);

static void Save_DOS_NonCol(dcomplex ****** Residues, double **** OLP0, double ** EVal,
                            int * LO_TC, int * HO_TC);

/* ------------------------------------------------------------------ */
/* node-local GPU proxy context                                       */
/* ------------------------------------------------------------------ */

static int      DCLNO_gpu_proxy_initialized = 0;
static MPI_Comm DCLNO_node_comm      = MPI_COMM_NULL;
static MPI_Comm DCLNO_gpu_group_comm = MPI_COMM_NULL;
static int      DCLNO_node_rank      = 0;
static int      DCLNO_node_size      = 1;
static int      DCLNO_gpu_group_rank = 0;
static int      DCLNO_gpu_group_size = 1;
static int      DCLNO_ngpu           = 0;
static int      DCLNO_gpu_id         = 0;
static int      DCLNO_is_gpu_owner   = 0;

#define DCLNO_ENABLE_SHARED_GPU_PROXY 0

/* ------------------------------------------------------------------ */
/* communicator selection                                             */
/* ------------------------------------------------------------------ */

static MPI_Comm DCLNO_GPUProxy_BaseComm(void)
{
    return (mpi_comm_level1 != MPI_COMM_NULL) ? mpi_comm_level1 : MPI_COMM_WORLD;
}

static void DCLNO_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    MPI_Abort(DCLNO_GPUProxy_BaseComm(), 1);
}

static size_t DCLNO_CheckedArrayBytes(size_t count, size_t elem_size, const char *label)
{
    if (count != 0 && elem_size > SIZE_MAX / count) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Allocation size overflow in Divide_Conquer_LNO.c: %s", label);
        DCLNO_AbortWithMessage(msg);
    }

    return count * elem_size;
}

static size_t DCLNO_CheckedMulCount(size_t a, size_t b, const char *label)
{
    if (a != 0 && b > SIZE_MAX / a) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Dimension overflow in Divide_Conquer_LNO.c: %s", label);
        DCLNO_AbortWithMessage(msg);
    }

    return a * b;
}

static void *DCLNO_MallocArray(size_t count, size_t elem_size, const char *label)
{
    size_t bytes = DCLNO_CheckedArrayBytes(count, elem_size, label);
    void * ptr = malloc((bytes == 0) ? 1 : bytes);

    if (ptr == NULL) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Out of memory in Divide_Conquer_LNO.c: %s (%zu bytes)", label, bytes);
        DCLNO_AbortWithMessage(msg);
    }

    return ptr;
}

static void *DCLNO_CallocArray(size_t count, size_t elem_size, const char *label)
{
    void * ptr;

    (void)DCLNO_CheckedArrayBytes(count, elem_size, label);
    ptr = calloc((count == 0) ? 1 : count, (elem_size == 0) ? 1 : elem_size);

    if (ptr == NULL) {
        char msg[512];
        size_t bytes = DCLNO_CheckedArrayBytes(count, elem_size, label);
        snprintf(msg, sizeof(msg), "Out of memory in Divide_Conquer_LNO.c: %s (%zu bytes)", label, bytes);
        DCLNO_AbortWithMessage(msg);
    }

    return ptr;
}

static void DCLNO_GPUProxy_Init(void)
{
    MPI_Comm base_comm;
    int color;
    cudaError_t cuda_err;

    if (DCLNO_gpu_proxy_initialized) return;

    base_comm = DCLNO_GPUProxy_BaseComm();

    MPI_Comm_split_type(base_comm, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &DCLNO_node_comm);
    MPI_Comm_rank(DCLNO_node_comm, &DCLNO_node_rank);
    MPI_Comm_size(DCLNO_node_comm, &DCLNO_node_size);

    cuda_err = cudaGetDeviceCount(&DCLNO_ngpu);
    if (cuda_err != cudaSuccess || DCLNO_ngpu <= 0) {
        fprintf(stderr, "DC-LNO GPU proxy: failed to detect CUDA devices (err=%d, count=%d).\n",
                (int)cuda_err, DCLNO_ngpu);
        fflush(stderr);
        MPI_Abort(base_comm, 1);
    }

    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < DCLNO_ngpu) {
        DCLNO_ngpu = SCF_Gpu_Num;
    }

    DCLNO_gpu_id = DCLNO_node_rank % DCLNO_ngpu;
    color = DCLNO_gpu_id;

    MPI_Comm_split(DCLNO_node_comm, color, DCLNO_node_rank, &DCLNO_gpu_group_comm);
    MPI_Comm_rank(DCLNO_gpu_group_comm, &DCLNO_gpu_group_rank);
    MPI_Comm_size(DCLNO_gpu_group_comm, &DCLNO_gpu_group_size);

    DCLNO_is_gpu_owner = (DCLNO_gpu_group_rank == 0);

    wait_cudafunc(cudaSetDevice(DCLNO_gpu_id));

    DCLNO_gpu_proxy_initialized = 1;
}

/* ------------------------------------------------------------------ */
/* utilities                                                          */
/* ------------------------------------------------------------------ */

static double **DCLNO_AllocRealMatrix1B(int n)
{
    int i;
    double **a = (double**)DCLNO_MallocArray((size_t)(n + 1), sizeof(double*), "real matrix row pointers");
    for (i=0; i<=n; i++) {
        a[i] = (double*)DCLNO_MallocArray((size_t)(n + 1), sizeof(double), "real matrix row");
    }
    return a;
}

static void DCLNO_FreeRealMatrix1B(double **a, int n)
{
    int i;
    if (a==NULL) return;
    for (i=0; i<=n; i++) free(a[i]);
    free(a);
}

static void DCLNO_PackRealC(double **C, int num, double *buf)
{
    int i, j;
    for (i=1; i<=num; i++) {
        for (j=1; j<=num; j++) {
            buf[(i-1)*num + (j-1)] = C[i][j];
        }
    }
}

static void DCLNO_UnpackRealC(const double *buf, int num, double **C)
{
    int i, j;
    for (i=1; i<=num; i++) {
        for (j=1; j<=num; j++) {
            C[i][j] = buf[(i-1)*num + (j-1)];
        }
    }
}

static void DCLNO_CopyPackedEigvecsToC(const double *buf, int num, int num2, double **C)
{
    int i, j;

    for (j = 1; j <= num2; j++) {
        for (i = 1; i <= num; i++) {
            C[j][i] = buf[(j - 1) * num + (i - 1)];
        }
    }
}

static void DCLNO_Eigen_lapack_d_reuse(double **a, double *ko, int n0, int EVmax,
                                       double *A, double *WORK, INTEGER *IWORK)
{
    char JOBZ[] = "V";
    char UPLO[] = "L";
    INTEGER n = n0;
    INTEGER LDA = n;
    INTEGER LWORK = 1 + 6 * n + 2 * n * n;
    INTEGER LIWORK = 3 + 5 * n;
    INTEGER INFO;
    int i, j;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n0; j++)
            A[i * n0 + j] = a[i + 1][j + 1];

    F77_NAME(dsyevd, DSYEVD)(JOBZ, UPLO, &n, A, &LDA, ko,
                             WORK, &LWORK, IWORK, &LIWORK, &INFO);
    if (INFO < 0) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Eigen_lapack_d workspace solve failed in Divide_Conquer_LNO.c: info=%d",
                 (int)INFO);
        DCLNO_AbortWithMessage(msg);
    }

    for (i = 0; i < EVmax; i++)
        for (j = 0; j < n0; j++)
            a[j + 1][i + 1] = A[i * n0 + j];

    for (i = EVmax; i >= 1; i--) ko[i] = ko[i - 1];
}

typedef struct {
    int                initialized;
    int                max_num;
    size_t             d_work_bytes;
    size_t             h_work_bytes;
    cudaStream_t       stream;
    cublasHandle_t     cublas;
    cusolverDnHandle_t gpusolver;
    double *           d_S;
    double *           d_H;
    double *           d_tmp;
    double *           d_W;
    int32_t *          d_info;
    void *             d_work;
    void *             h_work;
} DCLNO_GpuSolverCtx;

static DCLNO_GpuSolverCtx DCLNO_gpusolver_ctx = {0};

static void DCLNO_GpuSolver_Destroy(void)
{
    DCLNO_GpuSolverCtx *ctx = &DCLNO_gpusolver_ctx;

    if (ctx->stream != NULL) wait_cudafunc(cudaStreamSynchronize(ctx->stream));
    if (ctx->d_S != NULL) wait_cudafunc(cudaFree(ctx->d_S));
    if (ctx->d_H != NULL) wait_cudafunc(cudaFree(ctx->d_H));
    if (ctx->d_tmp != NULL) wait_cudafunc(cudaFree(ctx->d_tmp));
    if (ctx->d_W != NULL) wait_cudafunc(cudaFree(ctx->d_W));
    if (ctx->d_info != NULL) wait_cudafunc(cudaFree(ctx->d_info));
    if (ctx->d_work != NULL) wait_cudafunc(cudaFree(ctx->d_work));
    if (ctx->h_work != NULL) free(ctx->h_work);
    if (ctx->gpusolver != NULL) wait_cudafunc(cusolverDnDestroy(ctx->gpusolver));
    if (ctx->cublas != NULL) wait_cudafunc(cublasDestroy(ctx->cublas));
    if (ctx->stream != NULL) wait_cudafunc(cudaStreamDestroy(ctx->stream));
    memset(ctx, 0, sizeof(*ctx));
}

static void DCLNO_GpuSolver_Init(void)
{
    DCLNO_GpuSolverCtx *ctx = &DCLNO_gpusolver_ctx;

    if (ctx->initialized) return;

    wait_cudafunc(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));
    wait_cudafunc(cublasCreate(&ctx->cublas));
    wait_cudafunc(cusolverDnCreate(&ctx->gpusolver));
    wait_cudafunc(cublasSetStream(ctx->cublas, ctx->stream));
    wait_cudafunc(cusolverDnSetStream(ctx->gpusolver, ctx->stream));

    ctx->initialized = 1;
}

static void DCLNO_GpuSolver_EnsureMatrixCapacity(int num)
{
    DCLNO_GpuSolverCtx *ctx = &DCLNO_gpusolver_ctx;
    size_t             matrix_bytes;

    if (num <= 0) {
        DCLNO_AbortWithMessage("Invalid matrix size in DCLNO_GpuSolver_EnsureMatrixCapacity.");
    }

    DCLNO_GpuSolver_Init();

    if (num <= ctx->max_num) return;

    if (ctx->d_S    != NULL) wait_cudafunc(cudaFree(ctx->d_S));
    if (ctx->d_H    != NULL) wait_cudafunc(cudaFree(ctx->d_H));
    if (ctx->d_tmp  != NULL) wait_cudafunc(cudaFree(ctx->d_tmp));
    if (ctx->d_W    != NULL) wait_cudafunc(cudaFree(ctx->d_W));
    if (ctx->d_info != NULL) wait_cudafunc(cudaFree(ctx->d_info));

    matrix_bytes = DCLNO_CheckedArrayBytes(DCLNO_CheckedMulCount((size_t)num, (size_t)num,
                                                                 "GPUSOLVER dense matrix dimensions"),
                                           sizeof(double),
                                           "GPUSOLVER dense matrix buffer");

    wait_cudafunc(cudaMalloc((void**)&ctx->d_S, matrix_bytes));
    wait_cudafunc(cudaMalloc((void**)&ctx->d_H, matrix_bytes));
    wait_cudafunc(cudaMalloc((void**)&ctx->d_tmp, matrix_bytes));
    wait_cudafunc(cudaMalloc((void**)&ctx->d_W,
                             DCLNO_CheckedArrayBytes((size_t)num, sizeof(double), "GPUSOLVER eigenvalue buffer")));
    wait_cudafunc(cudaMalloc((void**)&ctx->d_info, sizeof(int32_t)));

    ctx->max_num = num;
}

static void DCLNO_GpuSolver_EnsureWorkspace(int m, int maxn)
{
    DCLNO_GpuSolverCtx *ctx = &DCLNO_gpusolver_ctx;
    cusolverEigMode_t  jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t   uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range;
    double             vl = 0.0;
    double             vu = 0.0;
    int64_t            h_meig = 0;
    size_t             d_bytes = 0;
    size_t             h_bytes = 0;

    if (m <= 0 || maxn <= 0 || maxn > m) {
        DCLNO_AbortWithMessage("Invalid eigensolver dimensions in DCLNO_GpuSolver_EnsureWorkspace.");
    }

    DCLNO_GpuSolver_EnsureMatrixCapacity(m);

    range = (m == maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;

    wait_cudafunc(cusolverDnXsyevdx_bufferSize(ctx->gpusolver, NULL, jobz, range, uplo, m,
                                               CUDA_R_64F, ctx->d_S, m, &vl, &vu, 1L, maxn,
                                               &h_meig, CUDA_R_64F, ctx->d_W, CUDA_R_64F,
                                               &d_bytes, &h_bytes));

    if (d_bytes > ctx->d_work_bytes) {
        if (ctx->d_work != NULL) wait_cudafunc(cudaFree(ctx->d_work));
        ctx->d_work = NULL;
        if (d_bytes > 0) {
            wait_cudafunc(cudaMalloc((void**)&ctx->d_work, d_bytes));
        }
        ctx->d_work_bytes = d_bytes;
    }

    if (h_bytes == 0) {
        if (ctx->h_work != NULL) free(ctx->h_work);
        ctx->h_work = NULL;
        ctx->h_work_bytes = 0;
    }
    else if (h_bytes > ctx->h_work_bytes) {
        if (ctx->h_work != NULL) free(ctx->h_work);
        ctx->h_work = DCLNO_MallocArray(h_bytes, 1, "GPUSOLVER host workspace");
        ctx->h_work_bytes = h_bytes;
    }
}

static void DCLNO_GpuSolver_Eigen(double *d_A, int m, int maxn, double *W)
{
    DCLNO_GpuSolverCtx *ctx = &DCLNO_gpusolver_ctx;
    cusolverEigMode_t  jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t   uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range;
    double             vl = 0.0;
    double             vu = 0.0;
    int64_t            h_meig = 0;
    int32_t            info = 0;

    DCLNO_GpuSolver_EnsureWorkspace(m, maxn);

    range = (m == maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;

    wait_cudafunc(cusolverDnXsyevdx(ctx->gpusolver, NULL, jobz, range, uplo, m, CUDA_R_64F,
                                    d_A, m, &vl, &vu, 1L, maxn, &h_meig, CUDA_R_64F,
                                    ctx->d_W, CUDA_R_64F, ctx->d_work, ctx->d_work_bytes,
                                    ctx->h_work, ctx->h_work_bytes, ctx->d_info));

    wait_cudafunc(cudaMemcpyAsync(W, ctx->d_W, sizeof(double) * (size_t)maxn,
                                  cudaMemcpyDeviceToHost, ctx->stream));
    wait_cudafunc(cudaMemcpyAsync(&info, ctx->d_info, sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    if (info != 0) {
        fprintf(stderr, "cusolverDnXsyevdx failed in DC-LNO: info=%d\n", (int)info);
        exit(10);
    }
}

static void DCLNO_Solve_Col_GpuSolver(int NUM, int NUM2, double *Smat, double *Hmat, double *ko)
{
    DCLNO_GpuSolverCtx *ctx = &DCLNO_gpusolver_ctx;
    double             alpha = 1.0;
    double             beta = 0.0;
    size_t             full_bytes;
    size_t             partial_bytes;
    int                l;

    if (NUM <= 0 || NUM2 <= 0 || NUM2 > NUM) {
        DCLNO_AbortWithMessage("Invalid matrix dimensions in DCLNO_Solve_Col_GpuSolver.");
    }

    full_bytes = DCLNO_CheckedArrayBytes(DCLNO_CheckedMulCount((size_t)NUM, (size_t)NUM,
                                                               "GPUSOLVER full matrix dimensions"),
                                         sizeof(double),
                                         "GPUSOLVER full matrix copy");
    partial_bytes = DCLNO_CheckedArrayBytes(DCLNO_CheckedMulCount((size_t)NUM, (size_t)NUM2,
                                                                  "GPUSOLVER eigenvector dimensions"),
                                            sizeof(double),
                                            "GPUSOLVER partial eigenvector copy");

    DCLNO_GpuSolver_EnsureMatrixCapacity(NUM);

    wait_cudafunc(cudaMemcpyAsync(ctx->d_S, Smat, full_bytes, cudaMemcpyHostToDevice, ctx->stream));
    wait_cudafunc(cudaMemcpyAsync(ctx->d_H, Hmat, full_bytes, cudaMemcpyHostToDevice, ctx->stream));

    DCLNO_GpuSolver_Eigen(ctx->d_S, NUM, NUM, ko + 1);

    for (l = 1; l <= NUM; l++) {
        ko[l] = 1.0 / sqrt(fabs(ko[l]));
    }

    wait_cudafunc(cudaMemcpyAsync(ctx->d_W, ko + 1,
                                  DCLNO_CheckedArrayBytes((size_t)NUM, sizeof(double),
                                                          "GPUSOLVER host-to-device eigenvalue scale"),
                                  cudaMemcpyHostToDevice, ctx->stream));

    wait_cudafunc(cublasDdgmm(ctx->cublas, CUBLAS_SIDE_RIGHT, NUM, NUM,
                              ctx->d_S, NUM, ctx->d_W, 1, ctx->d_tmp, NUM));

    wait_cudafunc(openmx_gemmul8Dgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N, NUM, NUM, NUM, &alpha, ctx->d_H, NUM,
                                     ctx->d_tmp, NUM, &beta, ctx->d_S, NUM));

    wait_cudafunc(openmx_gemmul8Dgemm(ctx->cublas, CUBLAS_OP_C, CUBLAS_OP_N, NUM, NUM, NUM, &alpha, ctx->d_tmp, NUM,
                                     ctx->d_S, NUM, &beta, ctx->d_H, NUM));

    DCLNO_GpuSolver_Eigen(ctx->d_H, NUM, NUM2, ko + 1);

    wait_cudafunc(openmx_gemmul8Dgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N, NUM, NUM2, NUM, &alpha, ctx->d_tmp,
                                     NUM, ctx->d_H, NUM, &beta, ctx->d_S, NUM));

    wait_cudafunc(cudaMemcpyAsync(Hmat, ctx->d_S, partial_bytes,
                                  cudaMemcpyDeviceToHost, ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));
}

/* ------------------------------------------------------------------ */
/* local dense solve kernel (collinear)                               */
/* ------------------------------------------------------------------ */

static void DCLNO_Solve_Col_Local(int NUM,
                                  int NUM2,
                                  double *Smat,
                                  double *Hmat,
                                  double **C,
                                  double *ko,
                                  double *Tmp,
                                  double *Eig_A,
                                  double *Eig_Work,
                                  INTEGER *Eig_IWork)
{
    int i1, j1, l;
    int owns_tmp = 0;
    double alpha, beta;
    int BM, BN, BK;

    if (Tmp == NULL) {
        Tmp = (double*)DCLNO_MallocArray(DCLNO_CheckedMulCount((size_t)NUM, (size_t)NUM,
                                                               "local DGEMM scratch dimensions"),
                                         sizeof(double), "local DGEMM scratch");
        owns_tmp = 1;
    }

    /* diagonalize S */
    for (i1=1; i1<=NUM; i1++) {
        for (j1=1; j1<=NUM; j1++) {
            C[i1][j1] = Smat[(j1-1)*NUM + (i1-1)];
        }
    }

    DCLNO_Eigen_lapack_d_reuse(C, ko, NUM, NUM, Eig_A, Eig_Work, Eig_IWork);

    for (l=1; l<=NUM; l++) ko[l] = 1.0/sqrt(fabs(ko[l]));

    /* Smat <- U * s^{-1/2} */
    for (i1=1; i1<=NUM; i1++) {
        for (j1=1; j1<=NUM; j1++) {
            Smat[(j1-1)*NUM + (i1-1)] = C[i1][j1]*ko[j1];
        }
    }

    /* Tmp = H * Smat */
    BM = NUM; BN = NUM; BK = NUM;
    alpha = 1.0; beta = 0.0;
    F77_NAME(dgemm,DGEMM)
    ("N","N",&BM,&BN,&BK,&alpha,
     Hmat,&BM,Smat,&BK,&beta,Tmp,&BM);

    /* Hmat = Smat^+ * Tmp */
    alpha = 1.0; beta = 0.0;
    F77_NAME(dgemm,DGEMM)
    ("C","N",&BM,&BN,&BK,&alpha,
     Smat,&BM,Tmp,&BK,&beta,Hmat,&BM);

    /* diagonalize transformed H */
    for (i1=1; i1<=NUM; i1++) {
        for (j1=1; j1<=NUM; j1++) {
            C[i1][j1] = Hmat[(j1-1)*NUM + (i1-1)];
        }
    }

    DCLNO_Eigen_lapack_d_reuse(C, ko, NUM, NUM2, Eig_A, Eig_Work, Eig_IWork);

    /* Hmat (first NUM2 cols) <- transformed eigenvectors */
    for (i1=1; i1<=NUM; i1++) {
        for (j1=1; j1<=NUM2; j1++) {
            Hmat[(j1-1)*NUM + (i1-1)] = C[i1][j1];
        }
    }

    /* Tmp(:,1:NUM2) = Smat * Hmat(:,1:NUM2) */
    BM = NUM; BN = NUM2; BK = NUM;
    alpha = 1.0; beta = 0.0;
    F77_NAME(dgemm,DGEMM)
    ("N","N",&BM,&BN,&BK,&alpha,
     Smat,&BM,Hmat,&BK,&beta,Tmp,&BM);

    /* return C[eig_index][basis_index] */
    for (j1=1; j1<=NUM2; j1++) {
        for (i1=1; i1<=NUM; i1++) {
            C[j1][i1] = Tmp[(j1-1)*NUM + (i1-1)];
        }
    }

    if (owns_tmp) free(Tmp);
}

/* GPU dense solve kernel (noncollinear)                              */
/* ------------------------------------------------------------------ */

typedef struct {
    int                initialized;
    int                max_num;
    size_t             d_work_bytes;
    size_t             h_work_bytes;
    cudaStream_t       stream;
    cublasHandle_t     cublas;
    cusolverDnHandle_t gpusolver;
    dcomplex *         d_S;
    dcomplex *         d_H;
    dcomplex *         d_tmp;
    dcomplex *         d_scale;
    dcomplex *         h_scale;
    double *           d_W;
    int32_t *          d_info;
    void *             d_work;
    void *             h_work;
} DCLNO_GpuSolverZCtx;

static DCLNO_GpuSolverZCtx DCLNO_gpusolverz_ctx = {0};

static void DCLNO_GpuSolverZ_Destroy(void)
{
    DCLNO_GpuSolverZCtx *ctx = &DCLNO_gpusolverz_ctx;

    if (!ctx->initialized) return;

    if (ctx->stream != NULL) wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    if (ctx->d_S != NULL)     wait_cudafunc(cudaFree(ctx->d_S));
    if (ctx->d_H != NULL)     wait_cudafunc(cudaFree(ctx->d_H));
    if (ctx->d_tmp != NULL)   wait_cudafunc(cudaFree(ctx->d_tmp));
    if (ctx->d_scale != NULL) wait_cudafunc(cudaFree(ctx->d_scale));
    if (ctx->d_W != NULL)     wait_cudafunc(cudaFree(ctx->d_W));
    if (ctx->d_info != NULL)  wait_cudafunc(cudaFree(ctx->d_info));
    if (ctx->d_work != NULL)  wait_cudafunc(cudaFree(ctx->d_work));
    free(ctx->h_work);
    free(ctx->h_scale);
    if (ctx->gpusolver != NULL) wait_cudafunc(cusolverDnDestroy(ctx->gpusolver));
    if (ctx->cublas != NULL)    wait_cudafunc(cublasDestroy(ctx->cublas));
    if (ctx->stream != NULL)    wait_cudafunc(cudaStreamDestroy(ctx->stream));

    memset(ctx, 0, sizeof(*ctx));
}

static void DCLNO_GpuSolverZ_Init(void)
{
    DCLNO_GpuSolverZCtx *ctx = &DCLNO_gpusolverz_ctx;

    if (ctx->initialized) return;

    wait_cudafunc(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));
    wait_cudafunc(cublasCreate(&ctx->cublas));
    wait_cudafunc(cusolverDnCreate(&ctx->gpusolver));
    wait_cudafunc(cublasSetStream(ctx->cublas, ctx->stream));
    wait_cudafunc(cusolverDnSetStream(ctx->gpusolver, ctx->stream));

    ctx->initialized = 1;
}

static void DCLNO_GpuSolverZ_EnsureMatrixCapacity(int num)
{
    DCLNO_GpuSolverZCtx *ctx = &DCLNO_gpusolverz_ctx;
    size_t matrix_bytes;

    if (num <= 0) {
        DCLNO_AbortWithMessage("Invalid matrix size in DCLNO_GpuSolverZ_EnsureMatrixCapacity.");
    }

    DCLNO_GpuSolverZ_Init();

    if (num <= ctx->max_num) return;

    if (ctx->d_S != NULL)     wait_cudafunc(cudaFree(ctx->d_S));
    if (ctx->d_H != NULL)     wait_cudafunc(cudaFree(ctx->d_H));
    if (ctx->d_tmp != NULL)   wait_cudafunc(cudaFree(ctx->d_tmp));
    if (ctx->d_scale != NULL) wait_cudafunc(cudaFree(ctx->d_scale));
    if (ctx->d_W != NULL)     wait_cudafunc(cudaFree(ctx->d_W));
    if (ctx->d_info != NULL)  wait_cudafunc(cudaFree(ctx->d_info));
    free(ctx->h_scale);

    matrix_bytes = DCLNO_CheckedArrayBytes(DCLNO_CheckedMulCount((size_t)num, (size_t)num,
                                                                 "GPUSOLVER NC dense matrix dimensions"),
                                           sizeof(dcomplex),
                                           "GPUSOLVER NC dense matrix buffer");

    wait_cudafunc(cudaMalloc((void**)&ctx->d_S, matrix_bytes));
    wait_cudafunc(cudaMalloc((void**)&ctx->d_H, matrix_bytes));
    wait_cudafunc(cudaMalloc((void**)&ctx->d_tmp, matrix_bytes));
    wait_cudafunc(cudaMalloc((void**)&ctx->d_scale,
                             DCLNO_CheckedArrayBytes((size_t)num, sizeof(dcomplex),
                                                     "GPUSOLVER NC scale buffer")));
    wait_cudafunc(cudaMalloc((void**)&ctx->d_W,
                             DCLNO_CheckedArrayBytes((size_t)num, sizeof(double),
                                                     "GPUSOLVER NC eigenvalue buffer")));
    wait_cudafunc(cudaMalloc((void**)&ctx->d_info, sizeof(int32_t)));
    ctx->h_scale = (dcomplex*)DCLNO_MallocArray((size_t)num, sizeof(dcomplex),
                                                "GPUSOLVER NC host scale buffer");

    ctx->max_num = num;
}

static void DCLNO_GpuSolverZ_EnsureWorkspace(int m, int maxn)
{
    DCLNO_GpuSolverZCtx *ctx = &DCLNO_gpusolverz_ctx;
    cusolverEigMode_t  jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t   uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range;
    double vl = 0.0, vu = 0.0;
    int64_t h_meig = 0;
    size_t d_bytes = 0, h_bytes = 0;

    if (m <= 0 || maxn <= 0 || maxn > m) {
        DCLNO_AbortWithMessage("Invalid eigensolver dimensions in DCLNO_GpuSolverZ_EnsureWorkspace.");
    }

    range = (m == maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;

    wait_cudafunc(cusolverDnXsyevdx_bufferSize(ctx->gpusolver, NULL, jobz, range, uplo, m,
                                               CUDA_C_64F, ctx->d_tmp, m, &vl, &vu, 1L, maxn, &h_meig,
                                               CUDA_R_64F, ctx->d_W, CUDA_C_64F, &d_bytes, &h_bytes));

    if (ctx->d_work_bytes < d_bytes) {
        if (ctx->d_work != NULL) wait_cudafunc(cudaFree(ctx->d_work));
        ctx->d_work = NULL;
        if (0 < d_bytes) {
            wait_cudafunc(cudaMalloc((void**)&ctx->d_work, d_bytes));
        }
        ctx->d_work_bytes = d_bytes;
    }

    if (h_bytes == 0) {
        free(ctx->h_work);
        ctx->h_work = NULL;
        ctx->h_work_bytes = 0;
    }
    else if (ctx->h_work_bytes < h_bytes) {
        free(ctx->h_work);
        ctx->h_work = DCLNO_MallocArray(h_bytes, 1, "GPUSOLVER NC host workspace");
        ctx->h_work_bytes = h_bytes;
    }
}

static void DCLNO_GpuSolverZ_Eigen(dcomplex *d_A, int m, int maxn, double *W)
{
    DCLNO_GpuSolverZCtx *ctx = &DCLNO_gpusolverz_ctx;
    cusolverEigMode_t  jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t   uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range;
    double vl = 0.0, vu = 0.0;
    int64_t h_meig = 0;
    int32_t info = 0;

    DCLNO_GpuSolverZ_EnsureWorkspace(m, maxn);

    range = (m == maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;

    wait_cudafunc(cusolverDnXsyevdx(ctx->gpusolver, NULL, jobz, range, uplo, m, CUDA_C_64F,
                                    d_A, m, &vl, &vu, 1L, maxn, &h_meig, CUDA_R_64F, ctx->d_W,
                                    CUDA_C_64F, ctx->d_work, ctx->d_work_bytes,
                                    ctx->h_work, ctx->h_work_bytes, ctx->d_info));
    wait_cudafunc(cudaMemcpyAsync(&info, ctx->d_info, sizeof(int32_t), cudaMemcpyDeviceToHost, ctx->stream));
    wait_cudafunc(cudaMemcpyAsync(W, ctx->d_W,
                                  DCLNO_CheckedArrayBytes((size_t)maxn, sizeof(double),
                                                          "GPUSOLVER NC eigenvalue copy"),
                                  cudaMemcpyDeviceToHost, ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    if (info != 0) {
        fprintf(stderr, "cusolverDnXsyevdx failed in DC-LNO (NC): info=%d\n", (int)info);
        DCLNO_AbortWithMessage("cusolverDnXsyevdx (complex) failed in Divide_Conquer_LNO.c.");
    }
}

/*
   GPU counterpart of the noncollinear per-cluster solve for
   Eigen_MPI_flag==0.  Reproduces the CPU flow, which diagonalizes the
   TRANSPOSE (= conjugate for a Hermitian matrix) of the transformed
   Hamiltonian and stores the back-transformed eigenvectors transposed
   (state-major rows with leading dimension NUM) into Hmat.
*/
static void DCLNO_Solve_NonCol_GpuSolver(int NUM, int NUMH, int NUM2, int rebuild_s,
                                         const dcomplex *Smat, dcomplex *Hmat, double *ko)
{
    DCLNO_GpuSolverZCtx *ctx = &DCLNO_gpusolverz_ctx;
    cuDoubleComplex zone  = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zzero = make_cuDoubleComplex(0.0, 0.0);
    size_t full_bytes;
    int l;

    if (NUM <= 0 || NUM2 <= 0 || NUM2 > NUM || 2*NUMH != NUM) {
        DCLNO_AbortWithMessage("Invalid matrix dimensions in DCLNO_Solve_NonCol_GpuSolver.");
    }
    if (!rebuild_s) {
        DCLNO_AbortWithMessage("DCLNO_Solve_NonCol_GpuSolver requires the overlap stage (LNO_recalc_flag==1).");
    }

    full_bytes = DCLNO_CheckedArrayBytes(DCLNO_CheckedMulCount((size_t)NUM, (size_t)NUM,
                                                               "GPUSOLVER NC full matrix dimensions"),
                                         sizeof(dcomplex),
                                         "GPUSOLVER NC full matrix copy");

    DCLNO_GpuSolverZ_EnsureMatrixCapacity(NUM);

    wait_cudafunc(cudaMemcpyAsync(ctx->d_H, Hmat, full_bytes, cudaMemcpyHostToDevice, ctx->stream));

    /* diagonalize the NUMH x NUMH overlap and build the spin-doubled
       transform T = diag(U s^{-1/2}, U s^{-1/2}) in d_S */

    wait_cudafunc(cudaMemcpyAsync(ctx->d_tmp, Smat,
                                  DCLNO_CheckedArrayBytes(DCLNO_CheckedMulCount((size_t)NUMH, (size_t)NUMH,
                                                                                "GPUSOLVER NC overlap dimensions"),
                                                          sizeof(dcomplex),
                                                          "GPUSOLVER NC overlap copy"),
                                  cudaMemcpyHostToDevice, ctx->stream));

    DCLNO_GpuSolverZ_Eigen(ctx->d_tmp, NUMH, NUMH, ko + 1);

    for (l = 1; l <= NUMH; l++) {
        ko[l] = 1.0 / sqrt(fabs(ko[l]));
        ctx->h_scale[l-1].r = ko[l];
        ctx->h_scale[l-1].i = 0.0;
    }

    wait_cudafunc(cudaMemcpyAsync(ctx->d_scale, ctx->h_scale,
                                  DCLNO_CheckedArrayBytes((size_t)NUMH, sizeof(dcomplex),
                                                          "GPUSOLVER NC scale copy"),
                                  cudaMemcpyHostToDevice, ctx->stream));

    wait_cudafunc(cudaMemsetAsync(ctx->d_S, 0, full_bytes, ctx->stream));

    /* top-left block: U * s^{-1/2}, written with leading dimension NUM */
    wait_cudafunc(cublasZdgmm(ctx->cublas, CUBLAS_SIDE_RIGHT, NUMH, NUMH,
                              (cuDoubleComplex*)ctx->d_tmp, NUMH,
                              (cuDoubleComplex*)ctx->d_scale, 1,
                              (cuDoubleComplex*)ctx->d_S, NUM));

    /* bottom-right block: copy of the top-left block */
    wait_cudafunc(cudaMemcpy2DAsync(ctx->d_S + (size_t)NUMH * (size_t)NUM + (size_t)NUMH,
                                    (size_t)NUM * sizeof(dcomplex),
                                    ctx->d_S,
                                    (size_t)NUM * sizeof(dcomplex),
                                    (size_t)NUMH * sizeof(dcomplex),
                                    (size_t)NUMH,
                                    cudaMemcpyDeviceToDevice, ctx->stream));

    /* H' = T^H * H * T */

    wait_cudafunc(openmx_gemmul8Zgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N, NUM, NUM, NUM, &zone,
                                      (cuDoubleComplex*)ctx->d_H, NUM,
                                      (cuDoubleComplex*)ctx->d_S, NUM, &zzero,
                                      (cuDoubleComplex*)ctx->d_tmp, NUM));

    wait_cudafunc(openmx_gemmul8Zgemm(ctx->cublas, CUBLAS_OP_C, CUBLAS_OP_N, NUM, NUM, NUM, &zone,
                                      (cuDoubleComplex*)ctx->d_S, NUM,
                                      (cuDoubleComplex*)ctx->d_tmp, NUM, &zzero,
                                      (cuDoubleComplex*)ctx->d_H, NUM));

    /* d_tmp = (H')^T = conj(H'), matching the CPU packing into C */
    wait_cudafunc(cublasZgeam(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N, NUM, NUM,
                              &zone, (cuDoubleComplex*)ctx->d_H, NUM,
                              &zzero, (cuDoubleComplex*)ctx->d_H, NUM,
                              (cuDoubleComplex*)ctx->d_tmp, NUM));

    DCLNO_GpuSolverZ_Eigen(ctx->d_tmp, NUM, NUM2, ko + 1);

    /* Hmat(state, basis) = (T * V)^T = V^T * T^T, stored with ld NUM */
    wait_cudafunc(openmx_gemmul8Zgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_T, NUM2, NUM, NUM, &zone,
                                      (cuDoubleComplex*)ctx->d_tmp, NUM,
                                      (cuDoubleComplex*)ctx->d_S, NUM, &zzero,
                                      (cuDoubleComplex*)ctx->d_H, NUM));

    wait_cudafunc(cudaMemcpyAsync(Hmat, ctx->d_H, full_bytes, cudaMemcpyDeviceToHost, ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));
}

/* ------------------------------------------------------------------ */
/* per-iteration GPU memory hygiene                                   */
/* ------------------------------------------------------------------ */

static int DCLNO_GpuTurnRelease(void)
{
    /* Releasing the GEMMul8 workspace and the solver buffers once the DC-LNO
       phase is done keeps them from occupying the shared device through the
       grid/force phases of the SCF step; opt out with
       OPENMX_DCLNO_GPU_TURN_RELEASE=0 (falls back to the band-path knob). */
    const char *value = getenv("OPENMX_DCLNO_GPU_TURN_RELEASE");

    if (value == NULL) {
        value = getenv("OPENMX_BAND_GEMMUL8_TURN_RELEASE");
    }
    if (value == NULL) {
        return 1;
    }
    return (atoi(value) != 0);
}

static void DCLNO_GpuSolver_ReleaseDeviceMemory(void)
{
    DCLNO_GpuSolverCtx *ctx = &DCLNO_gpusolver_ctx;

    if (!ctx->initialized) return;

    if (ctx->stream != NULL) wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    if (ctx->d_S != NULL)    wait_cudafunc(cudaFree(ctx->d_S));
    if (ctx->d_H != NULL)    wait_cudafunc(cudaFree(ctx->d_H));
    if (ctx->d_tmp != NULL)  wait_cudafunc(cudaFree(ctx->d_tmp));
    if (ctx->d_W != NULL)    wait_cudafunc(cudaFree(ctx->d_W));
    if (ctx->d_info != NULL) wait_cudafunc(cudaFree(ctx->d_info));
    if (ctx->d_work != NULL) wait_cudafunc(cudaFree(ctx->d_work));

    ctx->d_S = NULL;
    ctx->d_H = NULL;
    ctx->d_tmp = NULL;
    ctx->d_W = NULL;
    ctx->d_info = NULL;
    ctx->d_work = NULL;
    ctx->d_work_bytes = 0;
    ctx->max_num = 0;
}

static void DCLNO_GpuSolverZ_ReleaseDeviceMemory(void)
{
    DCLNO_GpuSolverZCtx *ctx = &DCLNO_gpusolverz_ctx;

    if (!ctx->initialized) return;

    if (ctx->stream != NULL) wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    if (ctx->d_S != NULL)     wait_cudafunc(cudaFree(ctx->d_S));
    if (ctx->d_H != NULL)     wait_cudafunc(cudaFree(ctx->d_H));
    if (ctx->d_tmp != NULL)   wait_cudafunc(cudaFree(ctx->d_tmp));
    if (ctx->d_scale != NULL) wait_cudafunc(cudaFree(ctx->d_scale));
    if (ctx->d_W != NULL)     wait_cudafunc(cudaFree(ctx->d_W));
    if (ctx->d_info != NULL)  wait_cudafunc(cudaFree(ctx->d_info));
    if (ctx->d_work != NULL)  wait_cudafunc(cudaFree(ctx->d_work));
    free(ctx->h_scale);

    ctx->d_S = NULL;
    ctx->d_H = NULL;
    ctx->d_tmp = NULL;
    ctx->d_scale = NULL;
    ctx->h_scale = NULL;
    ctx->d_W = NULL;
    ctx->d_info = NULL;
    ctx->d_work = NULL;
    ctx->d_work_bytes = 0;
    ctx->max_num = 0;
}

/*
   All ranks sharing a device keep their solver buffers resident through the
   whole atom loop, so the direct per-rank dispatch only pays off when the
   combined footprint fits; otherwise ranks spin in allocation retries and
   the GPU path ends up slower than the CPU eigensolver.  Collective over
   DCLNO_gpu_group_comm.
*/
static int DCLNO_GpuGroupMemoryFits(int max_msize, int is_complex)
{
    const size_t elem = is_complex ? sizeof(dcomplex) : sizeof(double);
    const size_t reserve = 1536ULL*1024ULL*1024ULL;
    const size_t overhead = 256ULL*1024ULL*1024ULL;
    size_t mm, need;
    size_t free_bytes = 0, total_bytes = 0;
    unsigned long long my_need, group_need = 0ULL;
    unsigned long long my_free, group_free = 0ULL;
    static int warned = 0;

    if (max_msize <= 0) max_msize = 1;

    mm = DCLNO_CheckedMulCount((size_t)max_msize, (size_t)max_msize, "GPU fit-check matrix count");
    /* three dense matrices plus roughly four matrix-sized eigensolver work
       areas, the GEMMul8 workspace, and per-rank CUDA context overhead */
    need = DCLNO_CheckedMulCount(mm, 7u*elem, "GPU fit-check per-rank bytes");
    need += is_complex ? openmx_gemmul8ZWorkspaceSize(max_msize, max_msize, max_msize)
                       : openmx_gemmul8DWorkspaceSize(max_msize, max_msize, max_msize);
    need += overhead;

    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        free_bytes = 0;
    }

    my_need = (unsigned long long)need;
    my_free = (unsigned long long)free_bytes;
    MPI_Allreduce(&my_need, &group_need, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, DCLNO_gpu_group_comm);
    MPI_Allreduce(&my_free, &group_free, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, DCLNO_gpu_group_comm);

    if (group_free < (unsigned long long)reserve ||
        group_need > group_free - (unsigned long long)reserve) {
        if (!warned && DCLNO_gpu_group_rank == 0) {
            printf("<DC-LNO> GPU device %d: %d rank(s) would need %.3f GiB "
                   "(free %.3f GiB, reserve %.3f GiB); using the CPU eigensolver instead.\n",
                   DCLNO_gpu_id, DCLNO_gpu_group_size,
                   (double)group_need/(1024.0*1024.0*1024.0),
                   (double)group_free/(1024.0*1024.0*1024.0),
                   (double)reserve/(1024.0*1024.0*1024.0));
            fflush(stdout);
            warned = 1;
        }
        return 0;
    }

    return 1;
}

/* ------------------------------------------------------------------ */

static void DCLNO_GPUProxy_Finalize(void)
{
    if (!DCLNO_gpu_proxy_initialized) return;

    DCLNO_GpuSolver_Destroy();
    DCLNO_GpuSolverZ_Destroy();
    if (DCLNO_gpu_group_comm != MPI_COMM_NULL) MPI_Comm_free(&DCLNO_gpu_group_comm);
    if (DCLNO_node_comm != MPI_COMM_NULL) MPI_Comm_free(&DCLNO_node_comm);

    DCLNO_gpu_proxy_initialized = 0;
    DCLNO_node_comm = MPI_COMM_NULL;
    DCLNO_gpu_group_comm = MPI_COMM_NULL;
    DCLNO_node_rank = 0;
    DCLNO_node_size = 1;
    DCLNO_gpu_group_rank = 0;
    DCLNO_gpu_group_size = 1;
    DCLNO_ngpu = 0;
    DCLNO_gpu_id = 0;
    DCLNO_is_gpu_owner = 0;
}

/* ------------------------------------------------------------------ */
/* node-local proxy service                                           */
/* ------------------------------------------------------------------ */

static void DCLNO_GPUProxy_Col_Service(int active,
                                       int NUM,
                                       int NUM2,
                                       double *Smat,
                                       double *Hmat,
                                       double **C,
                                       double *ko)
{
    int rank, size, src;
    int myinfo[3], *allinfo;
    int any_active = 0;
    double **recv_Sbuf = NULL;
    double **recv_Hbuf = NULL;
    MPI_Request *recv_Sreq = NULL;
    MPI_Request *recv_Hreq = NULL;

    if (DCLNO_gpu_group_comm == MPI_COMM_NULL) return;

    MPI_Comm_rank(DCLNO_gpu_group_comm, &rank);
    MPI_Comm_size(DCLNO_gpu_group_comm, &size);

    myinfo[0] = active;
    myinfo[1] = NUM;
    myinfo[2] = NUM2;

    allinfo = (int*)DCLNO_MallocArray(DCLNO_CheckedMulCount(3u, (size_t)size, "GPU proxy allgather metadata count"),
                                      sizeof(int), "GPU proxy allgather metadata");
    MPI_Allgather(myinfo, 3, MPI_INT, allinfo, 3, MPI_INT, DCLNO_gpu_group_comm);

    for (src=0; src<size; src++) {
        if (allinfo[3*src + 0]) {
            any_active = 1;
            break;
        }
    }

    if (!any_active) {
        free(allinfo);
        return;
    }

    if (rank == 0) {

        recv_Sbuf = (double**)DCLNO_CallocArray((size_t)size, sizeof(double*), "GPU proxy S receive buffers");
        recv_Hbuf = (double**)DCLNO_CallocArray((size_t)size, sizeof(double*), "GPU proxy H receive buffers");
        recv_Sreq = (MPI_Request*)DCLNO_MallocArray((size_t)size, sizeof(MPI_Request), "GPU proxy S requests");
        recv_Hreq = (MPI_Request*)DCLNO_MallocArray((size_t)size, sizeof(MPI_Request), "GPU proxy H requests");

        for (src = 0; src < size; src++) {
            recv_Sreq[src] = MPI_REQUEST_NULL;
            recv_Hreq[src] = MPI_REQUEST_NULL;
        }

        for (src = 1; src < size; src++) {
            int s_active = allinfo[3*src + 0];
            int s_num    = allinfo[3*src + 1];
            int s_num2   = allinfo[3*src + 2];

            if (!s_active) continue;
            if (s_num <= 0 || s_num2 <= 0 || s_num2 > s_num) {
                DCLNO_AbortWithMessage("Invalid GPU proxy task metadata in DCLNO_GPUProxy_Col_Service.");
            }

            recv_Sbuf[src] = (double*)DCLNO_MallocArray(DCLNO_CheckedMulCount((size_t)s_num, (size_t)s_num,
                                                                              "GPU proxy received overlap matrix dimensions"),
                                                        sizeof(double),
                                                        "GPU proxy received overlap matrix");
            recv_Hbuf[src] = (double*)DCLNO_MallocArray(DCLNO_CheckedMulCount((size_t)s_num, (size_t)s_num,
                                                                              "GPU proxy received Hamiltonian matrix dimensions"),
                                                        sizeof(double),
                                                        "GPU proxy received Hamiltonian matrix");

            MPI_Irecv(recv_Sbuf[src], s_num*s_num, MPI_DOUBLE, src,
                      DCLNO_PROXY_TAG_COL_S, DCLNO_gpu_group_comm, &recv_Sreq[src]);
            MPI_Irecv(recv_Hbuf[src], s_num*s_num, MPI_DOUBLE, src,
                      DCLNO_PROXY_TAG_COL_H, DCLNO_gpu_group_comm, &recv_Hreq[src]);
        }

        /* owner handles its own task first */
        if (active) {
            DCLNO_Solve_Col_GpuSolver(NUM, NUM2, Smat, Hmat, ko);
            DCLNO_CopyPackedEigvecsToC(Hmat, NUM, NUM2, C);
        }

        /* then serve other ranks in the group */
        for (src=1; src<size; src++) {
            int s_active = allinfo[3*src + 0];
            int s_num    = allinfo[3*src + 1];
            int s_num2   = allinfo[3*src + 2];

            if (!s_active) continue;

            {
                double *Sbuf, *Hbuf;
                double *kbuf;

                MPI_Wait(&recv_Sreq[src], MPI_STATUS_IGNORE);
                MPI_Wait(&recv_Hreq[src], MPI_STATUS_IGNORE);

                Sbuf = recv_Sbuf[src];
                Hbuf = recv_Hbuf[src];
                kbuf = (double*)DCLNO_MallocArray((size_t)(s_num + 2), sizeof(double),
                                                 "GPU proxy eigenvalue buffer");

                DCLNO_Solve_Col_GpuSolver(s_num, s_num2, Sbuf, Hbuf, kbuf);

                MPI_Send(&kbuf[1], s_num2, MPI_DOUBLE, src,
                         DCLNO_PROXY_TAG_COL_EVAL, DCLNO_gpu_group_comm);
                MPI_Send(Hbuf, s_num*s_num2, MPI_DOUBLE, src,
                         DCLNO_PROXY_TAG_COL_CVEC, DCLNO_gpu_group_comm);

                free(Sbuf);
                free(Hbuf);
                free(kbuf);
            }
        }

        free(recv_Sbuf);
        free(recv_Hbuf);
        free(recv_Sreq);
        free(recv_Hreq);
    }
    else if (active) {
        double *Cbuf = (double*)DCLNO_MallocArray(DCLNO_CheckedMulCount((size_t)NUM, (size_t)NUM2,
                                                                        "GPU proxy packed eigenvector dimensions"),
                                                  sizeof(double),
                                                  "GPU proxy packed eigenvectors");

        MPI_Send(Smat, NUM*NUM, MPI_DOUBLE, 0,
                 DCLNO_PROXY_TAG_COL_S, DCLNO_gpu_group_comm);
        MPI_Send(Hmat, NUM*NUM, MPI_DOUBLE, 0,
                 DCLNO_PROXY_TAG_COL_H, DCLNO_gpu_group_comm);

        MPI_Recv(&ko[1], NUM2, MPI_DOUBLE, 0,
                 DCLNO_PROXY_TAG_COL_EVAL, DCLNO_gpu_group_comm, MPI_STATUS_IGNORE);
        MPI_Recv(Cbuf, NUM*NUM2, MPI_DOUBLE, 0,
                 DCLNO_PROXY_TAG_COL_CVEC, DCLNO_gpu_group_comm, MPI_STATUS_IGNORE);

        DCLNO_CopyPackedEigvecsToC(Cbuf, NUM, NUM2, C);

        free(Cbuf);
    }

    free(allinfo);
}

/* ------------------------------------------------------------------ */
/* public wrapper                                                     */
/* ------------------------------------------------------------------ */

double Divide_Conquer_LNO(char * mode, int MD_iter, int SCF_iter, int SucceedReadingDMfile,
                          double ***** Hks, double ***** ImNL, double **** OLP0,
                          double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2])
{
    double time0;

    if (SpinP_switch == 0 || SpinP_switch == 1) {
        time0 = DC_Col(mode, MD_iter, SCF_iter, SucceedReadingDMfile,
                       Hks, OLP0, CDM, EDM, Eele0, Eele1);
    }
    else if (SpinP_switch == 3) {
        time0 = DC_NonCol(mode, MD_iter, SCF_iter, SucceedReadingDMfile,
                          Hks, ImNL, OLP0, CDM, EDM, Eele0, Eele1);
    }
    else {
        time0 = 0.0;
    }

    return time0;
}

/* ------------------------------------------------------------------ */
/* collinear main                                                     */
/* ------------------------------------------------------------------ */

static double DC_Col(char * mode, int MD_iter, int SCF_iter, int SucceedReadingDMfile,
                     double ***** Hks, double **** OLP0, double ***** CDM, double ***** EDM,
                     double Eele0[2], double Eele1[2])
{
    static int      firsttime = 1;
    static int      BLAS_allocate_flag = 0;
    static double * BLAS_OLP = NULL;
    static int      LO_HO_allocate_flag = 0;
    static int **   LO_TC = NULL;
    static int **   HO_TC = NULL;

    int             Mc_AN, Gc_AN, i, Gi, wan, wanA, wanB, Anum;
    int             size1, size2, num, NUM, NUM2, n2, Cwan, Hwan;
    int             LNO_recalc_flag;
    int             Mi, Mj, ig, ian, j, kl, jg, jan, Bnum, m, n, spin;
    int             l, i1, j1, i2, ip, po1, k;
    int             po, loopN, tno1, tno2, h_AN, Gh_AN, iwan, jwan;
    int             MA_AN, GA_AN, GB_AN, tnoA, tnoB, ino, jno;
    long int        m_size;
    long int        size_Residues;
    double          My_TZ, TZ, sumS, sumH, FermiF;
    double          tmp1, x;
    double          My_Num_State, Num_State;
    double          Dnum, ex, ex1, coe;
    double          TStime, TEtime;
    double          My_Eele0[2], My_Eele1[2];
    double          max_x = 30.0, Erange;
    double          ChemP_MAX, ChemP_MIN, spin_degeneracy;
    double *        Sc, *Hc, *Stmp, *Htmp;
    double *        ko, **C;
    double ***      EVal;
    double ******   Residues;
    double ***      PDOS_DC;
    int *           MP, *Msize, Max_Msize;
    double *        BLAS_H, *BLAS_C, *BLAS_Tmp, *BLAS_Eig_A, *BLAS_Eig_Work;
    INTEGER *       BLAS_Eig_IWork;
    double *        tmp_array;
    double *        tmp_array2;
    int *           Snd_H_Size, *Rcv_H_Size;
    int *           Snd_S_Size, *Rcv_S_Size;
    int             numprocs0, myid0, ID, IDS, IDR, tag = 999;
    double          Stime_atom, Etime_atom;
    double          stime, etime;
    double          time0, time1, time2, time3, time6, time7, time8, time9;
    MPI_Status      stat;
    MPI_Request     request;
    int             use_gpu_accel, use_gpu_proxy;
    int             gpu_group_max_msize;
    int             group_max_atoms, atom_slot;
    int             have_local_atom, use_gpu_task;
    size_t          eig_matrix_count, eig_work_count, eig_iwork_count;

    MPI_Comm_size(mpi_comm_level1, &numprocs0);
    MPI_Comm_rank(mpi_comm_level1, &myid0);

    dtime(&TStime);

    if (strcasecmp(mode, "free") == 0) {
        if (LO_HO_allocate_flag == 1) {
            if (LO_TC != NULL) {
                for (spin = 0; spin < 2; spin++) free(LO_TC[spin]);
                free(LO_TC);
                LO_TC = NULL;
            }
            if (HO_TC != NULL) {
                for (spin = 0; spin < 2; spin++) free(HO_TC[spin]);
                free(HO_TC);
                HO_TC = NULL;
            }
            LO_HO_allocate_flag = 0;
        }
        free(BLAS_OLP);
        BLAS_OLP = NULL;
        BLAS_allocate_flag = 0;
        firsttime = 1;
        DCLNO_GPUProxy_Finalize();
        openmx_gemmul8ReleaseWorkspaces();
        return 0.0;
    }

    time0 = 0.0;
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time6 = 0.0;
    time7 = 0.0;
    time8 = 0.0;
    time9 = 0.0;

    use_gpu_accel = (scf_eigen_lib_flag == GPUSOLVER);
    if (use_gpu_accel) {
        DCLNO_GPUProxy_Init();
    }
    use_gpu_proxy = (use_gpu_accel && DCLNO_ENABLE_SHARED_GPU_PROXY &&
                     DCLNO_gpu_group_size > 1);

    if (firsttime && myid0 == Host_ID && use_gpu_accel && 0 < level_stdout) {
        printf("<DC-LNO> GPUSOLVER direct per-rank dispatch is enabled for local matrices with dimension >= %d on %d CUDA device(s).\n",
               DCLNO_GpuEigenThreshold(), DCLNO_ngpu);
        fflush(stdout);
    }

    size_Residues = 0;

    /****************************************************
              find the total number of electrons
    ****************************************************/

    My_TZ = 0.0;
    for (i = 1; i <= Matomnum; i++) {
        Gc_AN = M2G[i];
        wan   = WhatSpecies[Gc_AN];
        My_TZ += Spe_Core_Charge[wan];
    }

    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_TZ, &TZ, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /****************************************************
                    calculation of LNOs
    ****************************************************/

    LNO_recalc_flag = 1;
    if (LNO_recalc_flag == 1) {
        time0 = LNO(mode, SCF_iter, OLP0, Hks, CDM);
    }

    /****************************************************
      allocation of LO/HO trackers
    ****************************************************/

    if (LO_HO_allocate_flag == 1 && SCF_iter == 1) {
        for (spin = 0; spin < 2; spin++) free(LO_TC[spin]);
        free(LO_TC);
        for (spin = 0; spin < 2; spin++) free(HO_TC[spin]);
        free(HO_TC);
        LO_HO_allocate_flag = 0;
    }

    if (LO_HO_allocate_flag == 0) {
        LO_TC = (int**)DCLNO_MallocArray(2, sizeof(int*), "LO tracker pointers");
        HO_TC = (int**)DCLNO_MallocArray(2, sizeof(int*), "HO tracker pointers");

        for (spin = 0; spin < 2; spin++) {
            LO_TC[spin] = (int*)DCLNO_MallocArray((size_t)(Matomnum + 1), sizeof(int), "LO tracker");
            HO_TC[spin] = (int*)DCLNO_MallocArray((size_t)(Matomnum + 1), sizeof(int), "HO tracker");
            for (i = 0; i <= Matomnum; i++) {
                LO_TC[spin][i] = 1;
                HO_TC[spin][i] = 1;
            }
        }

        LO_HO_allocate_flag = 1;
    }

    /****************************************************
      sizing
    ****************************************************/

    Msize = (int*)DCLNO_MallocArray((size_t)(Matomnum + 1), sizeof(int), "cluster matrix sizes");
    Max_Msize = 0;
    m_size = 0;

    EVal = (double***)DCLNO_MallocArray((size_t)(SpinP_switch + 1), sizeof(double**), "eigenvalue spin pointers");
    for (spin = 0; spin <= SpinP_switch; spin++) {
        EVal[spin] = (double**)DCLNO_MallocArray((size_t)(Matomnum + 1), sizeof(double*),
                                                 "eigenvalue atom pointers");

        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Msize[Mc_AN] = 1;
                n2 = 1;
            }
            else {
                Gc_AN = M2G[Mc_AN];
                Anum  = 1;

                for (i = 0; i <= FNAN_DCLNO[Gc_AN]; i++) {
                    Gi   = natn[Gc_AN][i];
                    wanA = WhatSpecies[Gi];
                    Anum += Spe_Total_CNO[wanA];
                }

                for (i = (FNAN_DCLNO[Gc_AN] + 1);
                     i <= (FNAN_DCLNO[Gc_AN] + SNAN_DCLNO[Gc_AN]); i++) {
                    Gi = natn[Gc_AN][i];
                    Anum += LNO_Num[Gi];
                }

                Msize[Mc_AN] = Anum - 1;
                n2           = Msize[Mc_AN] + 3;
            }

            if (Max_Msize < Msize[Mc_AN]) Max_Msize = Msize[Mc_AN];
            m_size += n2;
            EVal[spin][Mc_AN] = (double*)DCLNO_MallocArray((size_t)n2, sizeof(double), "eigenvalue row");
        }
    }

    /****************************************************
      static work array
    ****************************************************/

    if (BLAS_allocate_flag == 1 && LNO_recalc_flag == 1) {
        free(BLAS_OLP);
        BLAS_allocate_flag = 0;
    }

    if (BLAS_allocate_flag == 0) {
        BLAS_OLP = (double*)DCLNO_MallocArray(
            DCLNO_CheckedMulCount(DCLNO_CheckedMulCount((size_t)Max_Msize, (size_t)Max_Msize,
                                                        "overlap work buffer matrix dimensions"),
                                  (size_t)(SpinP_switch + 1),
                                  "overlap work buffer spin dimensions"),
            sizeof(double), "overlap work buffer");
        BLAS_allocate_flag = 1;
    }

    BLAS_H = (double*)DCLNO_MallocArray(
        DCLNO_CheckedMulCount(DCLNO_CheckedMulCount((size_t)Max_Msize, (size_t)Max_Msize,
                                                    "Hamiltonian work buffer matrix dimensions"),
                              (size_t)(SpinP_switch + 1),
                              "Hamiltonian work buffer spin dimensions"),
        sizeof(double), "Hamiltonian work buffer");
    BLAS_Tmp = (double*)DCLNO_MallocArray(
        DCLNO_CheckedMulCount((size_t)Max_Msize, (size_t)Max_Msize,
                              "local eigensolver scratch matrix dimensions"),
        sizeof(double), "local eigensolver scratch matrix");
    eig_matrix_count = DCLNO_CheckedMulCount((size_t)Max_Msize, (size_t)Max_Msize,
                                             "local eigensolver LAPACK matrix dimensions");
    eig_work_count = 1 + 6 * (size_t)Max_Msize + 2 * eig_matrix_count;
    eig_iwork_count = 3 + 5 * (size_t)Max_Msize;
    BLAS_Eig_A = (double*)DCLNO_MallocArray(eig_matrix_count, sizeof(double),
                                            "local eigensolver LAPACK matrix");
    BLAS_Eig_Work = (double*)DCLNO_MallocArray(eig_work_count, sizeof(double),
                                               "local eigensolver LAPACK work");
    BLAS_Eig_IWork = (INTEGER*)DCLNO_MallocArray(eig_iwork_count, sizeof(INTEGER),
                                                 "local eigensolver LAPACK iwork");
    BLAS_C = (double*)DCLNO_MallocArray((size_t)(Max_Msize + 1), sizeof(double), "occupation buffer");

    ko = (double*)DCLNO_MallocArray((size_t)(Max_Msize + 2), sizeof(double), "eigenvalue scratch");
    C  = DCLNO_AllocRealMatrix1B(Max_Msize + 1);

    Sc   = (double*)DCLNO_MallocArray(DCLNO_CheckedMulCount((size_t)List_YOUSO[7], (size_t)List_YOUSO[7],
                                                            "Sc scratch dimensions"),
                                      sizeof(double), "Sc scratch");
    Hc   = (double*)DCLNO_MallocArray(DCLNO_CheckedMulCount((size_t)List_YOUSO[7], (size_t)List_YOUSO[7],
                                                            "Hc scratch dimensions"),
                                      sizeof(double), "Hc scratch");
    Stmp = (double*)DCLNO_MallocArray(DCLNO_CheckedMulCount((size_t)List_YOUSO[7], (size_t)List_YOUSO[7],
                                                            "Stmp scratch dimensions"),
                                      sizeof(double), "Stmp scratch");
    Htmp = (double*)DCLNO_MallocArray(DCLNO_CheckedMulCount((size_t)List_YOUSO[7], (size_t)List_YOUSO[7],
                                                            "Htmp scratch dimensions"),
                                      sizeof(double), "Htmp scratch");

    Snd_H_Size = (int*)DCLNO_MallocArray((size_t)numprocs0, sizeof(int), "send Hamiltonian sizes");
    Rcv_H_Size = (int*)DCLNO_MallocArray((size_t)numprocs0, sizeof(int), "recv Hamiltonian sizes");
    Snd_S_Size = (int*)DCLNO_MallocArray((size_t)numprocs0, sizeof(int), "send overlap sizes");
    Rcv_S_Size = (int*)DCLNO_MallocArray((size_t)numprocs0, sizeof(int), "recv overlap sizes");

    MP = (int*)DCLNO_MallocArray((size_t)List_YOUSO[2], sizeof(int), "MP map");

    if (use_gpu_accel && !use_gpu_proxy &&
        !DCLNO_GpuGroupMemoryFits(Max_Msize >= DCLNO_GpuEigenThreshold() ? Max_Msize : 0, 0)) {
        use_gpu_accel = 0;
    }

    gpu_group_max_msize = Max_Msize;
    if (use_gpu_proxy) {
        MPI_Allreduce(MPI_IN_PLACE, &gpu_group_max_msize, 1, MPI_INT, MPI_MAX, DCLNO_gpu_group_comm);
        if (DCLNO_is_gpu_owner && gpu_group_max_msize >= DCLNO_GpuEigenThreshold()) {
            DCLNO_GpuSolver_EnsureMatrixCapacity(gpu_group_max_msize);
            DCLNO_GpuSolver_EnsureWorkspace(gpu_group_max_msize, gpu_group_max_msize);
        }
    }
    else if (use_gpu_accel && Max_Msize >= DCLNO_GpuEigenThreshold()) {
        DCLNO_GpuSolver_EnsureMatrixCapacity(Max_Msize);
        DCLNO_GpuSolver_EnsureWorkspace(Max_Msize, Max_Msize);
    }

    /****************************************************
      allocation of Residues
    ****************************************************/

    Residues = (double******)DCLNO_MallocArray((size_t)(SpinP_switch + 1), sizeof(double*****),
                                               "residue spin pointers");
    for (spin = 0; spin <= SpinP_switch; spin++) {
        Residues[spin] = (double*****)DCLNO_MallocArray((size_t)Matomnum, sizeof(double****),
                                                        "residue atom pointers");
        for (Mc_AN = 0; Mc_AN < Matomnum; Mc_AN++) {
            Gc_AN = M2G[Mc_AN + 1];
            wanA  = WhatSpecies[Gc_AN];
            tno1  = Spe_Total_CNO[wanA];

            Residues[spin][Mc_AN] = (double****)DCLNO_MallocArray((size_t)(FNAN[Gc_AN] + 1), sizeof(double***),
                                                                  "residue neighbor pointers");

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                Gh_AN = natn[Gc_AN][h_AN];
                wanB  = WhatSpecies[Gh_AN];
                tno2  = Spe_Total_CNO[wanB];

                Residues[spin][Mc_AN][h_AN] = (double***)DCLNO_MallocArray((size_t)tno1, sizeof(double**),
                                                                           "residue orbital-row pointers");
                for (i = 0; i < tno1; i++) {
                    Residues[spin][Mc_AN][h_AN][i] = (double**)DCLNO_MallocArray((size_t)tno2, sizeof(double*),
                                                                                  "residue orbital-column pointers");
                }
            }
        }
    }

    /****************************************************
      PDOS
    ****************************************************/

    PDOS_DC = (double***)DCLNO_MallocArray((size_t)(SpinP_switch + 1), sizeof(double**), "PDOS spin pointers");
    for (spin = 0; spin <= SpinP_switch; spin++) {
        PDOS_DC[spin] = (double**)DCLNO_MallocArray((size_t)Matomnum, sizeof(double*), "PDOS atom pointers");
        for (Mc_AN = 0; Mc_AN < Matomnum; Mc_AN++) {
            PDOS_DC[spin][Mc_AN] = (double*)DCLNO_MallocArray((size_t)(Msize[Mc_AN + 1] + 3), sizeof(double),
                                                              "PDOS row");
        }
    }

    if (firsttime) {
        PrintMemory("Divide_Conquer_LNO(col): EVal", sizeof(double) * m_size, NULL);
        PrintMemory("Divide_Conquer_LNO(col): BLAS_OLP",
                    sizeof(double) * Max_Msize * Max_Msize * (SpinP_switch + 1), NULL);
        PrintMemory("Divide_Conquer_LNO(col): BLAS_H",
                    sizeof(double) * Max_Msize * Max_Msize * (SpinP_switch + 1), NULL);
        PrintMemory("Divide_Conquer_LNO(col): BLAS_C",
                    sizeof(double) * (Max_Msize + 1), NULL);
    }

    /****************************************************
     MPI of Hks
    ****************************************************/

    for (ID = 0; ID < numprocs0; ID++) {

        IDS = (myid0 + ID) % numprocs0;
        IDR = (myid0 - ID + numprocs0) % numprocs0;

        if (ID != 0) {

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                size1 = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            size1 += tno1 * tno2;
                        }
                    }
                }

                Snd_H_Size[IDS] = size1;
                MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
            }
            else {
                Snd_H_Size[IDS] = 0;
            }

            if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {
                MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                Rcv_H_Size[IDR] = size2;
            }
            else {
                Rcv_H_Size[IDR] = 0;
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                MPI_Wait(&request, &stat);
            }
        }
        else {
            Snd_H_Size[IDS] = 0;
            Rcv_H_Size[IDR] = 0;
        }
    }

    for (ID = 0; ID < numprocs0; ID++) {

        IDS = (myid0 + ID) % numprocs0;
        IDR = (myid0 - ID + numprocs0) % numprocs0;

        if (ID != 0) {

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                size1 = Snd_H_Size[IDS];
                tmp_array = (double*)DCLNO_MallocArray((size_t)size1, sizeof(double), "MPI send Hamiltonian buffer");

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    tmp_array[num] = Hks[spin][Mc_AN][h_AN][i][j];
                                    num++;
                                }
                            }
                        }
                    }
                }

                MPI_Isend(tmp_array, size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
            }

            if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                size2 = Rcv_H_Size[IDR];
                tmp_array2 = (double*)DCLNO_MallocArray((size_t)size2, sizeof(double), "MPI recv Hamiltonian buffer");
                MPI_Recv(tmp_array2, size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    Mc_AN = S_TopMAN[IDR] - 1;
                    for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++) {
                        Mc_AN++;
                        Gc_AN = Rcv_GAN[IDR][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];

                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    Hks[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }
                }

                free(tmp_array2);
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                MPI_Wait(&request, &stat);
                free(tmp_array);
            }
        }
    }

    /****************************************************
     MPI of OLP0 (SCF_iter==1)
    ****************************************************/

    if (SCF_iter == 1) {

        for (ID = 0; ID < numprocs0; ID++) {

            IDS = (myid0 + ID) % numprocs0;
            IDR = (myid0 - ID + numprocs0) % numprocs0;

            if (ID != 0) {

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    size1 = 0;
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            size1 += tno1 * tno2;
                        }
                    }

                    Snd_S_Size[IDS] = size1;
                    MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                }
                else {
                    Snd_S_Size[IDS] = 0;
                }

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {
                    MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                    Rcv_S_Size[IDR] = size2;
                }
                else {
                    Rcv_S_Size[IDR] = 0;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
                }
            }
            else {
                Snd_S_Size[IDS] = 0;
                Rcv_S_Size[IDR] = 0;
            }
        }

        for (ID = 0; ID < numprocs0; ID++) {

            IDS = (myid0 + ID) % numprocs0;
            IDR = (myid0 - ID + numprocs0) % numprocs0;

            if (ID != 0) {

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                    size1 = Snd_S_Size[IDS];
                    tmp_array = (double*)DCLNO_MallocArray((size_t)size1, sizeof(double), "MPI send overlap buffer");

                    num = 0;
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    tmp_array[num] = OLP0[Mc_AN][h_AN][i][j];
                                    num++;
                                }
                            }
                        }
                    }

                    MPI_Isend(tmp_array, size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
                }

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                    size2 = Rcv_S_Size[IDR];
                    tmp_array2 = (double*)DCLNO_MallocArray((size_t)size2, sizeof(double), "MPI recv overlap buffer");
                    MPI_Recv(tmp_array2, size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                    num   = 0;
                    Mc_AN = S_TopMAN[IDR] - 1;
                    for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++) {
                        Mc_AN++;
                        Gc_AN = Rcv_GAN[IDR][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];

                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    OLP0[Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }

                    free(tmp_array2);
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
                    free(tmp_array);
                }
            }
        }
    }

    /****************************************************
      group max atom count for lock-step proxy service
    ****************************************************/

    group_max_atoms = Matomnum;
    if (use_gpu_proxy) {
        MPI_Allreduce(MPI_IN_PLACE, &group_max_atoms, 1, MPI_INT, MPI_MAX, DCLNO_gpu_group_comm);
    }

    /****************************************************
      per-atom loop
    ****************************************************/

    for (atom_slot = 1; atom_slot <= group_max_atoms; atom_slot++) {

        have_local_atom = (atom_slot <= Matomnum);

        if (have_local_atom) {
            dtime(&Stime_atom);
            if (measure_time) dtime(&stime);

            Mc_AN = atom_slot;
            Gc_AN = M2G[Mc_AN];
            wan   = WhatSpecies[Gc_AN];

            Anum = 1;
            for (i = 0; i <= FNAN_DCLNO[Gc_AN]; i++) {
                MP[i] = Anum;
                Gi    = natn[Gc_AN][i];
                wanA  = WhatSpecies[Gi];
                Anum += Spe_Total_CNO[wanA];
            }

            for (i = (FNAN_DCLNO[Gc_AN] + 1);
                 i <= (FNAN_DCLNO[Gc_AN] + SNAN_DCLNO[Gc_AN]); i++) {
                MP[i] = Anum;
                Gi    = natn[Gc_AN][i];
                Anum += LNO_Num[Gi];
            }

            NUM = Anum - 1;
            n2  = NUM + 3;

#define S_ref(k, i, j) BLAS_OLP[(k) * NUM * NUM + ((j) - 1) * NUM + ((i) - 1)]
#define H_ref(k, i, j) BLAS_H[(k) * NUM * NUM + ((j) - 1) * NUM + ((i) - 1)]

            for (spin = 0; spin <= SpinP_switch; spin++) {

                for (i = 0; i <= (FNAN_DCLNO[Gc_AN] + SNAN_DCLNO[Gc_AN]); i++) {

                    ig   = natn[Gc_AN][i];
                    iwan = WhatSpecies[ig];
                    ian  = Spe_Total_CNO[iwan];
                    ino  = LNO_Num[ig];

                    Anum = MP[i];
                    Mi   = S_G2M[ig];

                    for (j = 0; j <= (FNAN_DCLNO[Gc_AN] + SNAN_DCLNO[Gc_AN]); j++) {

                        kl   = RMI1[Mc_AN][i][j];
                        jg   = natn[Gc_AN][j];
                        jwan = WhatSpecies[jg];
                        jan  = Spe_Total_CNO[jwan];
                        jno  = LNO_Num[jg];

                        Bnum = MP[j];
                        Mj   = S_G2M[jg];

                        if (0 <= kl) {

                            if (i <= FNAN_DCLNO[Gc_AN] && j <= FNAN_DCLNO[Gc_AN]) {

                                for (m = 0; m < ian; m++) {
                                    for (n = 0; n < jan; n++) {
                                        H_ref(spin, Anum + m, Bnum + n) = Hks[spin][Mi][kl][m][n];
                                    }
                                }

                                if (LNO_recalc_flag == 1) {
                                    for (m = 0; m < ian; m++) {
                                        for (n = 0; n < jan; n++) {
                                            S_ref(spin, Anum + m, Bnum + n) = OLP0[Mi][kl][m][n];
                                        }
                                    }
                                }
                            }

                            else if (i <= FNAN_DCLNO[Gc_AN] && FNAN_DCLNO[Gc_AN] < j) {

                                for (m = 0; m < ian; m++) {
                                    for (n = 0; n < jno; n++) {
                                        sumH = 0.0;
                                        for (k = 0; k < jan; k++) {
                                            sumH += Hks[spin][Mi][kl][m][k] *
                                                    LNO_coes[spin][Mj][n * jan + k];
                                        }
                                        H_ref(spin, Anum + m, Bnum + n) = sumH;
                                    }
                                }

                                if (LNO_recalc_flag == 1) {
                                    for (m = 0; m < ian; m++) {
                                        for (n = 0; n < jno; n++) {
                                            sumS = 0.0;
                                            for (k = 0; k < jan; k++) {
                                                sumS += OLP0[Mi][kl][m][k] *
                                                        LNO_coes[spin][Mj][n * jan + k];
                                            }
                                            S_ref(spin, Anum + m, Bnum + n) = sumS;
                                        }
                                    }
                                }
                            }

                            else if (FNAN_DCLNO[Gc_AN] < i && j <= FNAN_DCLNO[Gc_AN]) {

                                for (m = 0; m < ian; m++) {
                                    for (n = 0; n < jan; n++) {
                                        Hc[n * ian + m] = Hks[spin][Mi][kl][m][n];
                                    }
                                }

                                for (m = 0; m < ino; m++) {
                                    for (n = 0; n < jan; n++) {
                                        sumH = 0.0;
                                        for (k = 0; k < ian; k++) {
                                            sumH += LNO_coes[spin][Mi][m * ian + k] *
                                                    Hc[n * ian + k];
                                        }
                                        H_ref(spin, Anum + m, Bnum + n) = sumH;
                                    }
                                }

                                if (LNO_recalc_flag == 1) {
                                    for (m = 0; m < ian; m++) {
                                        for (n = 0; n < jan; n++) {
                                            Sc[n * ian + m] = OLP0[Mi][kl][m][n];
                                        }
                                    }

                                    for (m = 0; m < ino; m++) {
                                        for (n = 0; n < jan; n++) {
                                            sumS = 0.0;
                                            for (k = 0; k < ian; k++) {
                                                sumS += LNO_coes[spin][Mi][m * ian + k] *
                                                        Sc[n * ian + k];
                                            }
                                            S_ref(spin, Anum + m, Bnum + n) = sumS;
                                        }
                                    }
                                }
                            }

                            else if (FNAN_DCLNO[Gc_AN] < i && FNAN_DCLNO[Gc_AN] < j) {

                                for (m = 0; m < ian; m++) {
                                    for (n = 0; n < jan; n++) {
                                        Hc[n * ian + m] = Hks[spin][Mi][kl][m][n];
                                    }
                                }

                                for (m = 0; m < ino; m++) {
                                    for (n = 0; n < jan; n++) {
                                        sumH = 0.0;
                                        for (k = 0; k < ian; k++) {
                                            sumH += LNO_coes[spin][Mi][m * ian + k] *
                                                    Hc[n * ian + k];
                                        }
                                        Htmp[m * jan + n] = sumH;
                                    }
                                }

                                for (m = 0; m < ino; m++) {
                                    for (n = 0; n < jno; n++) {
                                        sumH = 0.0;
                                        for (k = 0; k < jan; k++) {
                                            sumH += Htmp[m * jan + k] *
                                                    LNO_coes[spin][Mj][n * jan + k];
                                        }
                                        H_ref(spin, Anum + m, Bnum + n) = sumH;
                                    }
                                }

                                if (LNO_recalc_flag == 1) {

                                    for (m = 0; m < ian; m++) {
                                        for (n = 0; n < jan; n++) {
                                            Sc[n * ian + m] = OLP0[Mi][kl][m][n];
                                        }
                                    }

                                    for (m = 0; m < ino; m++) {
                                        for (n = 0; n < jan; n++) {
                                            sumS = 0.0;
                                            for (k = 0; k < ian; k++) {
                                                sumS += LNO_coes[spin][Mi][m * ian + k] *
                                                        Sc[n * ian + k];
                                            }
                                            Stmp[m * jan + n] = sumS;
                                        }
                                    }

                                    for (m = 0; m < ino; m++) {
                                        for (n = 0; n < jno; n++) {
                                            sumS = 0.0;
                                            for (k = 0; k < jan; k++) {
                                                sumS += Stmp[m * jan + k] *
                                                        LNO_coes[spin][Mj][n * jan + k];
                                            }
                                            S_ref(spin, Anum + m, Bnum + n) = sumS;
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            int ni, nj;

                            if (i <= FNAN_DCLNO[Gc_AN] && j <= FNAN_DCLNO[Gc_AN]) {
                                ni = ian; nj = jan;
                            }
                            else if (i <= FNAN_DCLNO[Gc_AN] && FNAN_DCLNO[Gc_AN] < j) {
                                ni = ian; nj = jno;
                            }
                            else if (FNAN_DCLNO[Gc_AN] < i && j <= FNAN_DCLNO[Gc_AN]) {
                                ni = ino; nj = jan;
                            }
                            else {
                                ni = ino; nj = jno;
                            }

                            for (m = 0; m < ni; m++) {
                                for (n = 0; n < nj; n++) {
                                    H_ref(spin, Anum + m, Bnum + n) = 0.0;
                                }
                            }

                            if (LNO_recalc_flag == 1) {
                                for (m = 0; m < ni; m++) {
                                    for (n = 0; n < nj; n++) {
                                        S_ref(spin, Anum + m, Bnum + n) = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (measure_time) {
                dtime(&etime);
                time1 += etime - stime;
            }
        }

        /****************************************************
          spin loop: solve local dense generalized EVP
        ****************************************************/

        for (spin = 0; spin <= SpinP_switch; spin++) {

            if (have_local_atom) {
                Mc_AN = atom_slot;
                Gc_AN = M2G[Mc_AN];
                NUM   = Msize[Mc_AN];

                if (SCF_iter <= 2) {
                    NUM2 = NUM;
                }
                else {
                    NUM2 = HO_TC[spin][Mc_AN] + 100;
                    if (NUM < NUM2) NUM2 = NUM;
                    if (NUM2 < 1)   NUM2 = NUM;
                }

                use_gpu_task = (use_gpu_accel && NUM >= DCLNO_GpuEigenThreshold());

                if (measure_time) dtime(&stime);

                if (use_gpu_task && !use_gpu_proxy) {
                    DCLNO_Solve_Col_GpuSolver(NUM, NUM2,
                                             &BLAS_OLP[spin * NUM * NUM],
                                             &BLAS_H[spin * NUM * NUM],
                                             ko);
                    DCLNO_CopyPackedEigvecsToC(&BLAS_H[spin * NUM * NUM], NUM, NUM2, C);
                }
                else if (!use_gpu_task) {
                    DCLNO_Solve_Col_Local(NUM, NUM2,
                                          &BLAS_OLP[spin * NUM * NUM],
                                          &BLAS_H[spin * NUM * NUM],
                                          C, ko, BLAS_Tmp,
                                          BLAS_Eig_A, BLAS_Eig_Work, BLAS_Eig_IWork);
                }
            }
            else {
                NUM = 0;
                NUM2 = 0;
                use_gpu_task = 0;
            }

            if (use_gpu_proxy && DCLNO_gpu_group_size > 1) {
                DCLNO_GPUProxy_Col_Service(use_gpu_task,
                                           NUM,
                                           NUM2,
                                           (have_local_atom ? &BLAS_OLP[spin * NUM * NUM] : NULL),
                                           (have_local_atom ? &BLAS_H[spin * NUM * NUM]   : NULL),
                                           C, ko);
            }

            if (!have_local_atom) continue;

            if (measure_time) {
                dtime(&etime);
                time2 += etime - stime;
            }

            if (measure_time) dtime(&stime);

            Mc_AN = atom_slot;
            Gc_AN = M2G[Mc_AN];
            NUM   = Msize[Mc_AN];

            for (i1 = 1; i1 <= NUM2; i1++) {
                EVal[spin][Mc_AN][i1] = ko[i1];
            }

            /* store eigenvectors in H_ref(spin,i_eig,basis) layout */
            for (i1 = 1; i1 <= NUM2; i1++) {
                for (j1 = 1; j1 <= NUM; j1++) {
                    H_ref(spin, i1, j1) = C[i1][j1];
                }
            }

            /******************************************************
              set energy window for residues
            ******************************************************/

            Erange = 10.0 / 27.2113845;

            i   = 1;
            ip  = 1;
            po1 = 0;
            do {
                if ((ChemP - Erange) < EVal[spin][Mc_AN][i]) {
                    ip  = i;
                    po1 = 1;
                }
                i++;
            } while (po1 == 0 && i <= NUM2);

            LO_TC[spin][Mc_AN] = ip;

            if ((TZ / 2 - 5) < LO_TC[spin][Mc_AN]) LO_TC[spin][Mc_AN] -= 30;
            if (LO_TC[spin][Mc_AN] < 1) LO_TC[spin][Mc_AN] = 1;

            i   = 1;
            ip  = NUM2;
            po1 = 0;
            do {
                if ((ChemP + Erange) < EVal[spin][Mc_AN][i]) {
                    ip  = i;
                    po1 = 1;
                }
                i++;
            } while (po1 == 0 && i <= NUM2);

            HO_TC[spin][Mc_AN] = ip;

            if (HO_TC[spin][Mc_AN] < (TZ / 2 + 5)) HO_TC[spin][Mc_AN] += 30;
            if (NUM2 < HO_TC[spin][Mc_AN]) HO_TC[spin][Mc_AN] = NUM2;

            n2 = HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 3;
            if (n2 < 1) n2 = 1;

            /******************************************************
              store residues
            ******************************************************/

            wanA = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[wanA];

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                Gh_AN = natn[Gc_AN][h_AN];
                wanB  = WhatSpecies[Gh_AN];
                tno2  = Spe_Total_CNO[wanB];
                Bnum  = MP[h_AN];

                for (i = 0; i < tno1; i++) {
                    for (j = 0; j < tno2; j++) {

                        size_Residues += n2;
                        Residues[spin][Mc_AN - 1][h_AN][i][j] =
                            (double*)DCLNO_MallocArray((size_t)n2, sizeof(double), "residue value row");
                        Residues[spin][Mc_AN - 1][h_AN][i][j][0] = 0.0;
                        Residues[spin][Mc_AN - 1][h_AN][i][j][1] = 0.0;

                        for (i1 = 1; i1 < LO_TC[spin][Mc_AN]; i1++) {
                            tmp1 = H_ref(spin, i1, 1 + i) * H_ref(spin, i1, Bnum + j);
                            Residues[spin][Mc_AN - 1][h_AN][i][j][0] += tmp1;
                            Residues[spin][Mc_AN - 1][h_AN][i][j][1] += tmp1 * EVal[spin][Mc_AN][i1];
                        }

                        for (i1 = LO_TC[spin][Mc_AN]; i1 <= HO_TC[spin][Mc_AN]; i1++) {
                            i2 = i1 - LO_TC[spin][Mc_AN] + 2;
                            Residues[spin][Mc_AN - 1][h_AN][i][j][i2] =
                                H_ref(spin, i1, 1 + i) * H_ref(spin, i1, Bnum + j);
                        }
                    }
                }
            }

            if (measure_time) {
                dtime(&etime);
                time3 += etime - stime;
            }
        }

        if (have_local_atom) {
            dtime(&Etime_atom);
            time_per_atom[M2G[atom_slot]] += Etime_atom - Stime_atom;
        }
    }

    /****************************************************
      projected DOS, ChemP, CDM/EDM
    ****************************************************/

    if (strcasecmp(mode, "scf") == 0 || strcasecmp(mode, "full") == 0) {

        if (measure_time) dtime(&stime);

        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];

                for (i1 = 0; i1 < (Msize[Mc_AN] + 3); i1++) {
                    PDOS_DC[spin][Mc_AN - 1][i1] = 0.0;
                }

                for (i = 0; i < tno1; i++) {
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB  = WhatSpecies[Gh_AN];
                        tno2  = Spe_Total_CNO[wanB];

                        for (j = 0; j < tno2; j++) {
                            tmp1 = OLP0[Mc_AN][h_AN][i][j];
                            PDOS_DC[spin][Mc_AN - 1][0] += Residues[spin][Mc_AN - 1][h_AN][i][j][0] * tmp1;
                            PDOS_DC[spin][Mc_AN - 1][1] += Residues[spin][Mc_AN - 1][h_AN][i][j][1] * tmp1;

                            for (i1 = LO_TC[spin][Mc_AN]; i1 <= HO_TC[spin][Mc_AN]; i1++) {
                                i2 = i1 - LO_TC[spin][Mc_AN] + 2;
                                PDOS_DC[spin][Mc_AN - 1][i2] +=
                                    Residues[spin][Mc_AN - 1][h_AN][i][j][i2] * tmp1;
                            }
                        }
                    }
                }
            }
        }

        if (measure_time) {
            dtime(&etime);
            time6 += etime - stime;
        }

        /****************************************************
          chemical potential
        ****************************************************/

        MPI_Barrier(mpi_comm_level1);
        if (measure_time) dtime(&stime);

        po    = 0;
        loopN = 0;
        Dnum  = 100.0;

        if (SCF_iter <= 2) {
            ChemP_MIN = -10.0;
            ChemP_MAX =  10.0;
        }
        else {
            ChemP_MIN = ChemP - 5.0;
            ChemP_MAX = ChemP + 5.0;
        }

        if      (SpinP_switch == 0) spin_degeneracy = 2.0;
        else if (SpinP_switch == 1) spin_degeneracy = 1.0;
        else                        spin_degeneracy = 1.0;

        do {
            ChemP = 0.50 * (ChemP_MAX + ChemP_MIN);
            My_Num_State = 0.0;

            for (spin = 0; spin <= SpinP_switch; spin++) {
                for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
                    My_Num_State += spin_degeneracy * PDOS_DC[spin][Mc_AN - 1][0];

                    for (i = LO_TC[spin][Mc_AN]; i <= HO_TC[spin][Mc_AN]; i++) {
                        i1 = i - LO_TC[spin][Mc_AN] + 2;
                        x  = (EVal[spin][Mc_AN][i] - ChemP) * Beta;
                        if (x <= -max_x) x = -max_x;
                        if ( max_x <= x) x =  max_x;

                        ex  = exp(x);
                        ex1 = 1.0 + ex;
                        coe = spin_degeneracy * PDOS_DC[spin][Mc_AN - 1][i1];

                        My_Num_State += coe / ex1;
                    }
                }
            }

            MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

            Dnum = (TZ - Num_State) - system_charge;

            if (0.0 <= Dnum) ChemP_MIN = ChemP;
            else             ChemP_MAX = ChemP;

            if (fabs(Dnum) < 1.0e-12) po = 1;

            loopN++;
        } while (po == 0 && loopN < 1000);

        if (measure_time) {
            dtime(&etime);
            time7 += etime - stime;
        }

        /****************************************************
          eigenenergy
        ****************************************************/

        if (measure_time) dtime(&stime);

        My_Eele0[0] = 0.0;
        My_Eele0[1] = 0.0;

        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
                My_Eele0[spin] += PDOS_DC[spin][Mc_AN - 1][1];

                for (i = LO_TC[spin][Mc_AN]; i <= HO_TC[spin][Mc_AN]; i++) {
                    i1 = i - LO_TC[spin][Mc_AN] + 2;
                    x  = (EVal[spin][Mc_AN][i] - ChemP) * Beta;
                    if (x <= -max_x) x = -max_x;
                    if ( max_x <= x) x =  max_x;
                    FermiF = 1.0 / (1.0 + exp(x));

                    My_Eele0[spin] += FermiF * EVal[spin][Mc_AN][i] *
                                      PDOS_DC[spin][Mc_AN - 1][i1];
                }
            }
        }

        MPI_Barrier(mpi_comm_level1);
        for (spin = 0; spin <= SpinP_switch; spin++) {
            MPI_Allreduce(&My_Eele0[spin], &Eele0[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        }

        if (SpinP_switch == 0) Eele0[1] = Eele0[0];

        if (measure_time) {
            dtime(&etime);
            time8 += etime - stime;
        }

        /****************************************************
          CDM / EDM
        ****************************************************/

        if (measure_time) dtime(&stime);

        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];

                for (i1 = LO_TC[spin][Mc_AN]; i1 <= HO_TC[spin][Mc_AN]; i1++) {
                    x = (EVal[spin][Mc_AN][i1] - ChemP) * Beta;
                    if (x <= -max_x) x = -max_x;
                    if ( max_x <= x) x =  max_x;
                    BLAS_C[i1] = 1.0 / (1.0 + exp(x));
                }

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];

                    for (i = 0; i < tno1; i++) {
                        for (j = 0; j < tno2; j++) {

                            double sum1 = Residues[spin][Mc_AN - 1][h_AN][i][j][0];
                            double sum2 = Residues[spin][Mc_AN - 1][h_AN][i][j][1];

                            for (i1 = LO_TC[spin][Mc_AN]; i1 <= HO_TC[spin][Mc_AN]; i1++) {
                                i2   = i1 - LO_TC[spin][Mc_AN] + 2;
                                tmp1 = BLAS_C[i1] * Residues[spin][Mc_AN - 1][h_AN][i][j][i2];
                                sum1 += tmp1;
                                sum2 += tmp1 * EVal[spin][Mc_AN][i1];
                            }

                            CDM[spin][Mc_AN][h_AN][i][j] = sum1;
                            EDM[spin][Mc_AN][h_AN][i][j] = sum2;
                        }
                    }
                }
            }
        }

        /****************************************************
          bond energies
        ****************************************************/

        My_Eele1[0] = 0.0;
        My_Eele1[1] = 0.0;

        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
                GA_AN = M2G[MA_AN];
                wanA  = WhatSpecies[GA_AN];
                tnoA  = Spe_Total_CNO[wanA];

                for (j = 0; j <= FNAN[GA_AN]; j++) {
                    GB_AN = natn[GA_AN][j];
                    wanB  = WhatSpecies[GB_AN];
                    tnoB  = Spe_Total_CNO[wanB];

                    for (k = 0; k < tnoA; k++) {
                        for (l = 0; l < tnoB; l++) {
                            My_Eele1[spin] += CDM[spin][MA_AN][j][k][l] *
                                              Hks[spin][MA_AN][j][k][l];
                        }
                    }
                }
            }
        }

        MPI_Barrier(mpi_comm_level1);
        for (spin = 0; spin <= SpinP_switch; spin++) {
            MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        }

        if (SpinP_switch == 0) Eele1[1] = Eele1[0];

        if (measure_time) {
            dtime(&etime);
            time9 += etime - stime;
        }
    }

    else if (strcasecmp(mode, "dos") == 0) {
        Save_DOS_Col(Residues, OLP0, EVal, LO_TC, HO_TC);
    }

    if (measure_time) {
        printf("Divide_Conquer_LNO(col) myid0=%2d time0=%7.3f time1=%7.3f time2=%7.3f time3=%7.3f time6=%7.3f time7=%7.3f time8=%7.3f time9=%7.3f\n",
               myid0, time0, time1, time2, time3, time6, time7, time8, time9);
        fflush(stdout);
    }

    if (firsttime) {
        PrintMemory("Divide_Conquer_LNO(col): Residues", sizeof(double) * size_Residues, NULL);
    }

    /****************************************************
      free
    ****************************************************/

    free(MP);
    free(Sc);
    free(Hc);
    free(Stmp);
    free(Htmp);
    free(Snd_H_Size);
    free(Rcv_H_Size);
    free(Snd_S_Size);
    free(Rcv_S_Size);
    free(Msize);
    free(BLAS_H);
    free(BLAS_Tmp);
    free(BLAS_Eig_A);
    free(BLAS_Eig_Work);
    free(BLAS_Eig_IWork);
    free(BLAS_C);
    free(ko);
    DCLNO_FreeRealMatrix1B(C, Max_Msize + 1);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) free(EVal[spin][Mc_AN]);
        free(EVal[spin]);
    }
    free(EVal);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 0; Mc_AN < Matomnum; Mc_AN++) {
            Gc_AN = M2G[Mc_AN + 1];
            wanA  = WhatSpecies[Gc_AN];
            tno1  = Spe_Total_CNO[wanA];

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                Gh_AN = natn[Gc_AN][h_AN];
                wanB  = WhatSpecies[Gh_AN];
                tno2  = Spe_Total_CNO[wanB];

                for (i = 0; i < tno1; i++) {
                    for (j = 0; j < tno2; j++) {
                        free(Residues[spin][Mc_AN][h_AN][i][j]);
                    }
                    free(Residues[spin][Mc_AN][h_AN][i]);
                }
                free(Residues[spin][Mc_AN][h_AN]);
            }
            free(Residues[spin][Mc_AN]);
        }
        free(Residues[spin]);
    }
    free(Residues);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 0; Mc_AN < Matomnum; Mc_AN++) free(PDOS_DC[spin][Mc_AN]);
        free(PDOS_DC[spin]);
    }
    free(PDOS_DC);

    if (SCF_iter == 2) firsttime = 0;

    if (use_gpu_accel && DCLNO_GpuTurnRelease()) {
        DCLNO_GpuSolver_ReleaseDeviceMemory();
        openmx_gemmul8ReleaseWorkspaces();
    }

    dtime(&TEtime);
    return (TEtime - TStime);
}

/* ------------------------------------------------------------------ */
/* noncollinear stub                                                  */
/* ------------------------------------------------------------------ */

static double DC_NonCol(char *mode,
			int MD_iter,
			int SCF_iter,
			int SucceedReadingDMfile,
			double *****Hks,
			double *****ImNL,
                        double ****OLP0,
			double *****CDM,
			double *****EDM,
			double Eele0[2], double Eele1[2])
{
  static int firsttime=1;
  int Mc_AN,Gc_AN,i,Gi,wan,wanA,wanB,Anum;
  int size1,size2,num,NUM,NUMH,NUM2,n2,Cwan,Hwan;
  static int BLAS_allocate_flag=0;
  static dcomplex *BLAS_OLP;
  dcomplex *BLAS_H,*BLAS_C;
  dcomplex **C;
  int LNO_recalc_flag,so;
  int Mi,Mj,ig,ian,j,kl,jg,jan,Bnum,m,n,spin;
  int l,i1,j1,i2,ip,po1,m_size,ns,ne,BM,BN,BK,ni,nj;
  int po,loopN,tno1,tno2,h_AN,Gh_AN,iwan,jwan;
  int MA_AN,GA_AN,GB_AN,tnoA,tnoB,k,ino,jno,size_Residues;
  double My_TZ,TZ,sum,sumS,sumH,FermiF;
  double tmp1,tmp2,tmp3,x;
  dcomplex ctmp1,ctmp2;
  double My_Num_State,Num_State;
  double My_Num_State_1d,Num_State_1d;
  double My_Num_State_2d,Num_State_2d;
  double Dnum,g1,g2,ex,ex1,coe;
  double TStime,TEtime;
  double My_Eele0[2],My_Eele1[2];
  double max_x=30.0,Erange;
  double ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double *Sc,*Stmp;
  double *Hc00r,*Hc11r,*Hc01r;
  double *Hc00i,*Hc11i,*Hc01i;
  double *Hc00r_tmp,*Hc11r_tmp,*Hc01r_tmp;
  double *Hc00i_tmp,*Hc11i_tmp,*Hc01i_tmp;
  double *ko;
  double **EVal;
  double sumH00r,sumH00i,sumH11r,sumH11i,sumH01r,sumH01i;
  dcomplex ******Residues;
  double **PDOS_DC;
  int *MP,*Msize,Max_Msize;
  int *is1,*ie1,*is2,*ie2;
  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;
  double *tmp_array;
  double *tmp_array2;
  int *Snd_H_Size,*Rcv_H_Size;
  int *Snd_iHNL_Size,*Rcv_iHNL_Size;
  int *Snd_S_Size,*Rcv_S_Size;
  static int LO_HO_allocate_flag=0;
  static int *LO_TC;
  static int *HO_TC;
  int numprocs0,myid0,ID1,ID,IDS,IDR,tag=999;
  int numprocs1,myid1,numprocs2,myid2;
  double Stime_atom, Etime_atom;
  double OLP_eigen_cut=Threshold_OLP_Eigen;
  double stime,etime;
  double time0,time1,time2,time3,time4;
  double time5,time6,time7,time8;
  int Eigen_MPI_flag,bcast_flag;
  int use_gpu_accel_nc,use_gpu_task_nc;
  double sum1,sum2,sum4;
  int i1s, j1s;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  size_Residues = 0;

  /* for time check */

  dtime(&TStime);

  if (measure_time){
    time0 = 0.0;
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
    time5 = 0.0;
    time6 = 0.0;
    time7 = 0.0;
    time8 = 0.0;
  }

  /****************************************************
                      if mode==free
  ****************************************************/

  if ( strcasecmp(mode,"free")==0 ){

    free(LO_TC);
    free(HO_TC);
    LO_HO_allocate_flag = 0;
    free(BLAS_OLP);
    BLAS_allocate_flag = 0;
    return 0.0;
  }

  /****************************************************
            find the total number of electrons
  ****************************************************/

  My_TZ = 0.0;
  for (i=1; i<=Matomnum; i++){
    Gc_AN = M2G[i];
    wan = WhatSpecies[Gc_AN];
    My_TZ += Spe_Core_Charge[wan];
  }

  /* MPI, My_TZ */

  MPI_Barrier(mpi_comm_level1);
  MPI_Allreduce(&My_TZ, &TZ, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /****************************************************
                 calculaiton of LNOs
  ****************************************************/

  /*
  if (MD_iter==1){
    LNO_recalc_flag = 1;
  }
  else if (SCF_iter==1 && SucceedReadingDMfile==1){
    LNO_recalc_flag = 1;
  }
  else if (SCF_iter==2 && SucceedReadingDMfile==0){
    LNO_recalc_flag = 1;
  }
  else{
    LNO_recalc_flag = 0;
  }
  */

  LNO_recalc_flag = 1;

  if (LNO_recalc_flag==1){
    time0 = LNO(mode,SCF_iter,OLP0,Hks,CDM);
  }

  /****************************************************
     get information of MPI for eigenvalue problems
  ****************************************************/

  use_gpu_accel_nc = (scf_eigen_lib_flag == GPUSOLVER);
  if (use_gpu_accel_nc) {
    DCLNO_GPUProxy_Init();
  }

  if (firsttime && myid0 == Host_ID && use_gpu_accel_nc && 0 < level_stdout) {
    printf("<DC-LNO> GPUSOLVER direct per-rank dispatch (noncollinear) is enabled for local matrices with dimension >= %d on %d CUDA device(s).\n",
           DCLNO_NonColGpuEigenThreshold(), DCLNO_ngpu);
    fflush(stdout);
  }

  if ( atomnum<=numprocs0 ){

    MPI_Comm_size(MPI_CommWD1_DCLNO[myworld1_DCLNO],&numprocs1);
    MPI_Comm_rank(MPI_CommWD1_DCLNO[myworld1_DCLNO],&myid1);

    MPI_Comm_size(MPI_CommWD2_DCLNO[myworld2_DCLNO],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2_DCLNO[myworld2_DCLNO],&myid2);

    Eigen_MPI_flag = 1;
  }
  else {

    numprocs1 = 1;
    myid1 = 0;

    numprocs2 = 1;
    myid2 = 0;

    Eigen_MPI_flag = 0;
  }

  if (Eigen_MPI_flag==1){

    Matomnum = 1;
    MPI_Bcast(&M2G[0], Matomnum+1, MPI_INT, 0, MPI_CommWD1_DCLNO[myworld1_DCLNO]);
  }

  if (LO_HO_allocate_flag==1 && SCF_iter==1){
    free(LO_TC);
    free(HO_TC);
    LO_HO_allocate_flag = 0;
  }

  if (LO_HO_allocate_flag==0){
    LO_TC = (int*)malloc(sizeof(int)*(Matomnum+1));
    HO_TC = (int*)malloc(sizeof(int)*(Matomnum+1));
    LO_HO_allocate_flag = 1;
  }

  /****************************************************
    allocation of arrays:

    int MP[List_YOUSO[2]];
    int Msize[Matomnum+1];
    double EVal[Matomnum+1][n2];
  ****************************************************/

  Sc = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);

  Hc00r = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Hc11r = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Hc01r = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Hc00i = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Hc11i = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Hc01i = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);

  Hc00r_tmp = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Hc11r_tmp = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Hc01r_tmp = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Hc00i_tmp = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Hc11i_tmp = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Hc01i_tmp = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);

  Stmp = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);

  Snd_H_Size = (int*)malloc(sizeof(int)*numprocs0);
  Rcv_H_Size = (int*)malloc(sizeof(int)*numprocs0);
  Snd_iHNL_Size = (int*)malloc(sizeof(int)*numprocs0);
  Rcv_iHNL_Size = (int*)malloc(sizeof(int)*numprocs0);
  Snd_S_Size = (int*)malloc(sizeof(int)*numprocs0);
  Rcv_S_Size = (int*)malloc(sizeof(int)*numprocs0);

  m_size = 0;
  Msize = (int*)malloc(sizeof(int)*(Matomnum+1));
  Max_Msize = 0;

  EVal = (double**)malloc(sizeof(double*)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Gc_AN = 0;
      FNAN_DCLNO[0] = 1;
      SNAN_DCLNO[0] = 0;
      n2 = 1;
      Msize[Mc_AN] = 1;
    }
    else{

      Gc_AN = M2G[Mc_AN];
      Anum = 1;

      for (i=0; i<=FNAN_DCLNO[Gc_AN]; i++){
	Gi = natn[Gc_AN][i];
	wanA = WhatSpecies[Gi];
	Anum += Spe_Total_CNO[wanA];
      }

      for (i=(FNAN_DCLNO[Gc_AN]+1); i<=(FNAN_DCLNO[Gc_AN]+SNAN_DCLNO[Gc_AN]); i++){
	Gi = natn[Gc_AN][i];
	Anum += LNO_Num[Gi];
      }

      NUM = 2*(Anum-1);
      Msize[Mc_AN] = NUM;
      n2 = NUM + 3;
    }

    if ( Max_Msize<Msize[Mc_AN] ) Max_Msize = Msize[Mc_AN];

    m_size += n2;
    EVal[Mc_AN] = (double*)malloc(sizeof(double)*n2);
  }

  if (Eigen_MPI_flag==1){
    MPI_Bcast(&Max_Msize, 1, MPI_INT, 0, MPI_CommWD1_DCLNO[myworld1_DCLNO]);
  }

  if (use_gpu_accel_nc &&
      !DCLNO_GpuGroupMemoryFits((Eigen_MPI_flag==0 && Max_Msize >= DCLNO_NonColGpuEigenThreshold()) ? Max_Msize : 0, 1)) {
    use_gpu_accel_nc = 0;
  }

  if (BLAS_allocate_flag==1 && LNO_recalc_flag==1){
    free(BLAS_OLP);
    BLAS_allocate_flag = 0;
  }

  if (BLAS_allocate_flag==0){
    BLAS_OLP = (dcomplex*)malloc(sizeof(dcomplex)*Max_Msize*Max_Msize);
    BLAS_allocate_flag = 1;
  }

  BLAS_H = (dcomplex*)malloc(sizeof(dcomplex)*Max_Msize*Max_Msize);
  BLAS_C = (dcomplex*)malloc(sizeof(dcomplex)*Max_Msize*Max_Msize);

  ko = (double*)malloc(sizeof(double)*(Max_Msize+2));

  C = (dcomplex**)malloc(sizeof(dcomplex*)*(Max_Msize+2));
  for (i=0; i<(Max_Msize+2); i++){
    C[i] = (dcomplex*)malloc(sizeof(dcomplex)*(Max_Msize+2));
  }

  if (firsttime){
  PrintMemory("Divide_Conquer_LNO: EVal",sizeof(double)*m_size,NULL);
  PrintMemory("Divide_Conquer_LNO: ko",  sizeof(double)*(Max_Msize+2),NULL);
  PrintMemory("Divide_Conquer_LNO: C",   sizeof(double)*(Max_Msize+2)*(Max_Msize+2),NULL);
  PrintMemory("Divide_Conquer_LNO: BLAS_OLP", sizeof(double)*Max_Msize*Max_Msize,NULL);
  PrintMemory("Divide_Conquer_LNO: BLAS_H",   sizeof(double)*Max_Msize*Max_Msize,NULL);
  PrintMemory("Divide_Conquer_LNO: BLAS_C",   sizeof(double)*Max_Msize*Max_Msize,NULL);
  }

  if (2<=level_stdout){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        printf("<DC> myid0=%i Mc_AN=%2d Gc_AN=%2d Msize=%3d\n",
        myid0,Mc_AN,M2G[Mc_AN],Msize[Mc_AN]);
    }
  }

  /****************************************************
    allocation of arrays:

    dcomplex Residues[3]
                     [Matomnum]
                     [FNAN[Gc_AN]+1]
                     [Spe_Total_CNO[Gc_AN]]
                     [Spe_Total_CNO[Gh_AN]]
                     [HO_TC-LO_TC+3]
  ****************************************************/

  if ( Eigen_MPI_flag==0 || (Eigen_MPI_flag==1 && myid2==0) ){

    Residues = (dcomplex******)malloc(sizeof(dcomplex*****)*3);
    for (spin=0; spin<3; spin++){
      Residues[spin] = (dcomplex*****)malloc(sizeof(dcomplex****)*Matomnum);
      for (Mc_AN=0; Mc_AN<Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN+1];
	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];
	n2 = Msize[Mc_AN+1] + 2;

	Residues[spin][Mc_AN] = (dcomplex****)malloc(sizeof(dcomplex***)*(FNAN[Gc_AN]+1));

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  wanB = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wanB];

	  Residues[spin][Mc_AN][h_AN] = (dcomplex***)malloc(sizeof(dcomplex**)*tno1);
	  for (i=0; i<tno1; i++){
	    Residues[spin][Mc_AN][h_AN][i] = (dcomplex**)malloc(sizeof(dcomplex*)*tno2);
	    /* note that the array is allocated once more in the loop */
	  }
	}
      }
    }
  }

  MP = (int*)malloc(sizeof(int)*List_YOUSO[2]);

  /****************************************************
    allocation of arrays:

    double PDOS_DC[2][Matomnum+1][NUM]
  ****************************************************/

  m_size = 0;
  PDOS_DC = (double**)malloc(sizeof(double*)*Matomnum);
  for (Mc_AN=0; Mc_AN<Matomnum; Mc_AN++){

    n2 = Msize[Mc_AN+1] + 3;
    m_size += n2;
    PDOS_DC[Mc_AN] = (double*)malloc(sizeof(double)*n2);
  }

  if (firsttime){
  PrintMemory("Divide_Conquer_LNO: PDOS_DC",sizeof(double)*m_size,NULL);
  }

  /****************************************************
   MPI

   Hks
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs0; ID++){

    IDS = (myid0 + ID) % numprocs0;
    IDR = (myid0 - ID + numprocs0) % numprocs0;

    if (ID!=0){
      tag = 999;

      /* find data size to send block data */
      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

        size1 = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              size1 += tno1*tno2;
	    }
          }
	}

        Snd_H_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_H_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_H_Size[IDR] = size2;
      }
      else{
        Rcv_H_Size[IDR] = 0;
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
    }
    else{
      Snd_H_Size[IDS] = 0;
      Rcv_H_Size[IDR] = 0;
    }
  }

  /***********************************
             data transfer
  ************************************/

  tag = 999;
  for (ID=0; ID<numprocs0; ID++){

    IDS = (myid0 + ID) % numprocs0;
    IDR = (myid0 - ID + numprocs0) % numprocs0;

    if (ID!=0){

      /*****************************
              sending of data
      *****************************/

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

        size1 = Snd_H_Size[IDS];

        /* allocation of array */

        tmp_array = (double*)malloc(sizeof(double)*size1);

	if (firsttime){
	  PrintMemory("Divide_Conquer_LNO: tmp_array",sizeof(double)*size1,NULL);
	}

        /* multidimentional array to vector array */

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  tmp_array[num] = Hks[spin][Mc_AN][h_AN][i][j];
                  num++;
                }
              }
	    }
          }
	}

        MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);

      }

      /*****************************
         receiving of block data
      *****************************/

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

        size2 = Rcv_H_Size[IDR];

        /* allocation of array */
        tmp_array2 = (double*)malloc(sizeof(double)*size2);

	if (firsttime){
	  PrintMemory("Divide_Conquer_LNO: tmp_array2",sizeof(double)*size2,NULL);
	}

        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          Mc_AN = S_TopMAN[IDR] - 1;  /* S_TopMAN should be used. */
          for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[Cwan];

            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  Hks[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
		}
	      }
	    }
	  }
	}

        /* freeing of array */
        free(tmp_array2);
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
        MPI_Wait(&request,&stat);
        free(tmp_array); /* freeing of array */
      }
    }
  }

  /****************************************************
   MPI

   ImNL
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  /* spin-orbit coupling or LDA+U */

  if (SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
      || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

    for (ID=0; ID<numprocs0; ID++){

      IDS = (myid0 + ID) % numprocs0;
      IDR = (myid0 - ID + numprocs0) % numprocs0;

      if (ID!=0){
	tag = 999;

	/* find data size to send block data */
	if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	  size1 = 0;
	  for (so=0; so<List_YOUSO[5]; so++){
	    for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
	      Mc_AN = Snd_MAN[IDS][n];
	      Gc_AN = Snd_GAN[IDS][n];
	      Cwan = WhatSpecies[Gc_AN];
	      tno1 = Spe_Total_CNO[Cwan];
	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno2 = Spe_Total_CNO[Hwan];
		for (i=0; i<tno1; i++){
		  for (j=0; j<tno2; j++){
		    size1++;
		  }
		}
	      }
	    }
	  }

	  Snd_iHNL_Size[IDS] = size1;
	  MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
	}
	else{
	  Snd_iHNL_Size[IDS] = 0;
	}

	/* receiving of size of data */

	if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
	  MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	  Rcv_iHNL_Size[IDR] = size2;
	}
	else{
	  Rcv_iHNL_Size[IDR] = 0;
	}

	if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);

      }
      else{
	Snd_iHNL_Size[IDS] = 0;
	Rcv_iHNL_Size[IDR] = 0;
      }
    }

    /***********************************
               data transfer
    ************************************/

    tag = 999;
    for (ID=0; ID<numprocs0; ID++){

      IDS = (myid0 + ID) % numprocs0;
      IDR = (myid0 - ID + numprocs0) % numprocs0;

      if (ID!=0){

	/*****************************
               sending of data
	*****************************/

	if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	  size1 = Snd_iHNL_Size[IDS];

	  /* allocation of array */

	  tmp_array = (double*)malloc(sizeof(double)*size1);

	  /* multidimentional array to vector array */

	  num = 0;
	  for (so=0; so<List_YOUSO[5]; so++){
	    for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
	      Mc_AN = Snd_MAN[IDS][n];
	      Gc_AN = Snd_GAN[IDS][n];
	      Cwan = WhatSpecies[Gc_AN];
	      tno1 = Spe_Total_CNO[Cwan];
	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno2 = Spe_Total_CNO[Hwan];
		for (i=0; i<tno1; i++){
		  for (j=0; j<tno2; j++){
		    tmp_array[num] = ImNL[so][Mc_AN][h_AN][i][j];
		    num++;
		  }
		}
	      }
	    }
	  }

	  MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);

	}

	/*****************************
	   receiving of block data
	*****************************/

	if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

	  size2 = Rcv_iHNL_Size[IDR];

	  /* allocation of array */
	  tmp_array2 = (double*)malloc(sizeof(double)*size2);

	  MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	  num = 0;
	  for (so=0; so<List_YOUSO[5]; so++){
	    Mc_AN = S_TopMAN[IDR] - 1;  /* S_TopMAN should be used. */
	    for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){
	      Mc_AN++;
	      Gc_AN = Rcv_GAN[IDR][n];
	      Cwan = WhatSpecies[Gc_AN];
	      tno1 = Spe_Total_CNO[Cwan];

	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno2 = Spe_Total_CNO[Hwan];
		for (i=0; i<tno1; i++){
		  for (j=0; j<tno2; j++){
		    ImNL[so][Mc_AN][h_AN][i][j] = tmp_array2[num];
		    num++;
		  }
		}
	      }
	    }
	  }

	  /* freeing of array */
	  free(tmp_array2);
	}

	if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
	  MPI_Wait(&request,&stat);
	  free(tmp_array); /* freeing of array */
	}
      }
    }
  }

  /****************************************************
   MPI

   OLP0
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  if (SCF_iter==1){

    for (ID=0; ID<numprocs0; ID++){

      IDS = (myid0 + ID) % numprocs0;
      IDR = (myid0 - ID + numprocs0) % numprocs0;

      if (ID!=0){
        tag = 999;

        /* find data size to send block data */

        if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
          size1 = 0;
          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  size1++;
                }
              }
	    }
          }

          Snd_S_Size[IDS] = size1;
          MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
        }
        else{
          Snd_S_Size[IDS] = 0;
        }

        /* receiving of size of data */

        if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
          MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
          Rcv_S_Size[IDR] = size2;
        }
        else{
          Rcv_S_Size[IDR] = 0;
        }

        if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
      }
      else{
        Snd_S_Size[IDS] = 0;
        Rcv_S_Size[IDR] = 0;
      }
    }

    /***********************************
               data transfer
    ************************************/

    tag = 999;
    for (ID=0; ID<numprocs0; ID++){

      IDS = (myid0 + ID) % numprocs0;
      IDR = (myid0 - ID + numprocs0) % numprocs0;

      if (ID!=0){

        /*****************************
                sending of data
        *****************************/

        if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

          size1 = Snd_S_Size[IDS];

          /* allocation of array */

          tmp_array = (double*)malloc(sizeof(double)*size1);

          /* multidimentional array to vector array */

          num = 0;

          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  tmp_array[num] = OLP0[Mc_AN][h_AN][i][j];
                  num++;
                }
              }
	    }
          }

          MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
        }

        /*****************************
           receiving of block data
        *****************************/

        if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

          size2 = Rcv_S_Size[IDR];

         /* allocation of array */
          tmp_array2 = (double*)malloc(sizeof(double)*size2);

          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

          num = 0;
          Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
          for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[Cwan];

            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  OLP0[Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
	        }
	      }
	    }
	  }

          /* freeing of array */
          free(tmp_array2);

        }

        if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
          MPI_Wait(&request,&stat);
          free(tmp_array); /* freeing of array */
	}
      }
    }
  }

  /****************************************************
      Setting of Hamiltonian and overlap matrices

         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  is1 = (int*)malloc(sizeof(int)*numprocs2);
  ie1 = (int*)malloc(sizeof(int)*numprocs2);

  is2 = (int*)malloc(sizeof(int)*numprocs2);
  ie2 = (int*)malloc(sizeof(int)*numprocs2);

  stat_send = malloc(sizeof(MPI_Status)*numprocs2);
  request_send = malloc(sizeof(MPI_Request)*numprocs2);
  request_recv = malloc(sizeof(MPI_Request)*numprocs2);

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);
    if (measure_time) dtime(&stime);

    /***********************************************
       find the size of matrix for the atom Mc_AN
                 and set the MP vector

       Note:
         MP indicates the starting position of
              atom i in arraies H and S
    ***********************************************/

    Gc_AN = M2G[Mc_AN];
    wan = WhatSpecies[Gc_AN];

    Anum = 1;
    for (i=0; i<=FNAN_DCLNO[Gc_AN]; i++){
      MP[i] = Anum;
      Gi = natn[Gc_AN][i];
      wanA = WhatSpecies[Gi];
      Anum += Spe_Total_CNO[wanA];
    }

    for (i=(FNAN_DCLNO[Gc_AN]+1); i<=(FNAN_DCLNO[Gc_AN]+SNAN_DCLNO[Gc_AN]); i++){
      MP[i] = Anum;
      Gi = natn[Gc_AN][i];
      Anum += LNO_Num[Gi];
    }

    NUM = 2*(Anum-1);
    NUMH = Anum-1;

    n2 = NUM + 3;

    /*
    if (myid0==0){
      printf("QQQ0 %2d %2d\n",FNAN[Gc_AN],SNAN[Gc_AN]);
      printf("QQQ1 %2d %2d\n",FNAN_DCLNO[Gc_AN],SNAN_DCLNO[Gc_AN]);
      printf("QQQ2 %2d\n",NUM);
    }
    */

    /* define S_ref(i,j) and H_ref(i,j) */

#define SNC_ref(i,j)  BLAS_OLP[(j-1)*NUMH+(i-1)]
#define SNC2_ref(i,j) BLAS_OLP[(j-1)*NUM+(i-1)]
#define HNC_ref(i,j)  BLAS_H[(j-1)*NUM+(i-1)]

    /* making is1 and ie1 */

    if ( numprocs2<=NUM ){

      for (ID=0; ID<numprocs2; ID++){
	is1[ID] = (int)((NUM*ID)/numprocs2) + 1;
	ie1[ID] = (int)((NUM*(ID+1))/numprocs2);
      }
    }

    else{

      for (ID=0; ID<NUM; ID++){
	is1[ID] = ID + 1;
	ie1[ID] = ID + 1;
      }
      for (ID=NUM; ID<numprocs2; ID++){
	is1[ID] =  1;
	ie1[ID] = -2;
      }
    }

    /*****************************************************
        construct cluster full matrices of Hamiltonian
              and overlap for the atom Mc_AN
    *****************************************************/

    if ( Eigen_MPI_flag==0 || (Eigen_MPI_flag==1 && myid1==0) ){

      for (i=0; i<=(FNAN_DCLNO[Gc_AN]+SNAN_DCLNO[Gc_AN]); i++){

	ig = natn[Gc_AN][i];
	iwan = WhatSpecies[ig];
	ian = Spe_Total_CNO[iwan];
	ino = LNO_Num[ig];

	Anum = MP[i];
	Mi = S_G2M[ig];

	for (j=0; j<=(FNAN_DCLNO[Gc_AN]+SNAN_DCLNO[Gc_AN]); j++){

	  kl = RMI1[Mc_AN][i][j];
	  jg = natn[Gc_AN][j];
	  jwan = WhatSpecies[jg];
	  jan = Spe_Total_CNO[jwan];
	  jno = LNO_Num[jg];

	  Bnum = MP[j];
	  Mj = S_G2M[jg];

	  if (0<=kl){

            /* i<=FNAN_DCLNO[Gc_AN] && j<=FNAN_DCLNO[Gc_AN] */

	    if ( i<=FNAN_DCLNO[Gc_AN] && j<=FNAN_DCLNO[Gc_AN] ){

	      /* non-spin-orbit coupling and non-LDA+U */
	      if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0
		  && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){

		for (m=0; m<ian; m++){
		  for (n=0; n<jan; n++){
		    HNC_ref(Anum+m,     Bnum+n     ).r = Hks[0][Mi][kl][m][n];
		    HNC_ref(Anum+m,     Bnum+n     ).i = 0.0;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).r = Hks[1][Mi][kl][m][n];
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).i = 0.0;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).r = Hks[2][Mi][kl][m][n];
		    HNC_ref(Anum+m,     Bnum+n+NUMH).i = Hks[3][Mi][kl][m][n];
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).r = HNC_ref(Anum+m, Bnum+n+NUMH).r;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).i =-HNC_ref(Anum+m, Bnum+n+NUMH).i;
		  }
		}
	      }

	      /* spin-orbit coupling or LDA+U */
	      else {
		for (m=0; m<ian; m++){
		  for (n=0; n<jan; n++){

		    /*
                    if ( Gc_AN==1 && (Anum+m)==1 && (Bnum+n)==5 ){
                      printf("GGG1 Gc_AN=%2d i=%2d j=%2d Mi=%2d ig=%2d kl=%2d %15.12f\n",
                              Gc_AN,i,j,Mi,ig,kl,Hks[0][Mi][kl][m][n]);
		    }
		    */

		    HNC_ref(Anum+m,     Bnum+n     ).r =  Hks[0][Mi][kl][m][n];
		    HNC_ref(Anum+m,     Bnum+n     ).i = ImNL[0][Mi][kl][m][n];
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).r =  Hks[1][Mi][kl][m][n];
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).i = ImNL[1][Mi][kl][m][n];
		    HNC_ref(Anum+m,     Bnum+n+NUMH).r =  Hks[2][Mi][kl][m][n];
		    HNC_ref(Anum+m,     Bnum+n+NUMH).i =  Hks[3][Mi][kl][m][n] + ImNL[2][Mi][kl][m][n];
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).r = HNC_ref(Anum+m, Bnum+n+NUMH).r;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).i =-HNC_ref(Anum+m, Bnum+n+NUMH).i;
		  }
		}
	      }

	      if (LNO_recalc_flag==1){
		for (m=0; m<ian; m++){
		  for (n=0; n<jan; n++){
		    SNC_ref(Anum+m, Bnum+n).r = OLP0[Mi][kl][m][n];
		    SNC_ref(Anum+m, Bnum+n).i = 0.0;
		  }
		}
	      }
	    }

            /* i<=FNAN_DCLNO[Gc_AN] && FNAN_DCLNO[Gc_AN]<j */

	    else if ( i<=FNAN_DCLNO[Gc_AN] && FNAN_DCLNO[Gc_AN]<j ){

	      /* non-spin-orbit coupling and non-LDA+U */
	      if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0
		  && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){

		for (m=0; m<ian; m++){
		  for (n=0; n<jno; n++){

		    sumH00r = 0.0;
		    sumH11r = 0.0;
		    sumH01r = 0.0;
		    sumH01i = 0.0;

		    for (k=0; k<jan; k++){
		      sumH00r += Hks[0][Mi][kl][m][k]*LNO_coes[0][Mj][n*jan+k];
		      sumH11r += Hks[1][Mi][kl][m][k]*LNO_coes[0][Mj][n*jan+k];
		      sumH01r += Hks[2][Mi][kl][m][k]*LNO_coes[0][Mj][n*jan+k];
		      sumH01i += Hks[3][Mi][kl][m][k]*LNO_coes[0][Mj][n*jan+k];
		    }

		    HNC_ref(Anum+m,     Bnum+n     ).r = sumH00r;
		    HNC_ref(Anum+m,     Bnum+n     ).i = 0.0;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).r = sumH11r;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).i = 0.0;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).r = sumH01r;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).i = sumH01i;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).r = HNC_ref(Anum+m, Bnum+n+NUMH).r;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).i =-HNC_ref(Anum+m, Bnum+n+NUMH).i;
		  }
		}
	      }

	      /* spin-orbit coupling or LDA+U */
	      else {
		for (m=0; m<ian; m++){
		  for (n=0; n<jno; n++){

		    sumH00r = 0.0;
		    sumH00i = 0.0;
		    sumH11r = 0.0;
		    sumH11i = 0.0;
		    sumH01r = 0.0;
		    sumH01i = 0.0;

		    for (k=0; k<jan; k++){
		      sumH00r +=  Hks[0][Mi][kl][m][k]*LNO_coes[0][Mj][n*jan+k];
		      sumH00i += ImNL[0][Mi][kl][m][k]*LNO_coes[0][Mj][n*jan+k];
		      sumH11r +=  Hks[1][Mi][kl][m][k]*LNO_coes[0][Mj][n*jan+k];
		      sumH11i += ImNL[1][Mi][kl][m][k]*LNO_coes[0][Mj][n*jan+k];
		      sumH01r +=  Hks[2][Mi][kl][m][k]*LNO_coes[0][Mj][n*jan+k];
		      sumH01i += (Hks[3][Mi][kl][m][k]+ImNL[2][Mi][kl][m][k])*LNO_coes[0][Mj][n*jan+k];
		    }

		    HNC_ref(Anum+m,     Bnum+n     ).r = sumH00r;
		    HNC_ref(Anum+m,     Bnum+n     ).i = sumH00i;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).r = sumH11r;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).i = sumH11i;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).r = sumH01r;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).i = sumH01i;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).r = HNC_ref(Anum+m, Bnum+n+NUMH).r;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).i =-HNC_ref(Anum+m, Bnum+n+NUMH).i;
		  }
		}
	      }

	      if (LNO_recalc_flag==1){
		for (m=0; m<ian; m++){
		  for (n=0; n<jno; n++){
		    sumS = 0.0;
		    for (k=0; k<jan; k++){
		      sumS += OLP0[Mi][kl][m][k]*LNO_coes[0][Mj][n*jan+k];
		    }

		    SNC_ref(Anum+m, Bnum+n).r = sumS;
		    SNC_ref(Anum+m, Bnum+n).i = 0.0;
		  }
		}
	      }
	    }

            /* FNAN_DCLNO[Gc_AN]<i && j<=FNAN_DCLNO[Gc_AN] */

	    else if ( FNAN_DCLNO[Gc_AN]<i && j<=FNAN_DCLNO[Gc_AN] ){

	      /* non-spin-orbit coupling and non-LDA+U */
	      if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0
		  && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){

		for (m=0; m<ian; m++){
		  for (n=0; n<jan; n++){
		    Hc00r[n*ian+m] = Hks[0][Mi][kl][m][n];
		    Hc11r[n*ian+m] = Hks[1][Mi][kl][m][n];
		    Hc01r[n*ian+m] = Hks[2][Mi][kl][m][n];
		    Hc01i[n*ian+m] = Hks[3][Mi][kl][m][n];
		  }
		}

		for (m=0; m<ino; m++){
		  for (n=0; n<jan; n++){

		    sumH00r = 0.0;
		    sumH11r = 0.0;
		    sumH01r = 0.0;
		    sumH01i = 0.0;

		    for (k=0; k<ian; k++){
		      sumH00r += LNO_coes[0][Mi][m*ian+k]*Hc00r[n*ian+k];
		      sumH11r += LNO_coes[0][Mi][m*ian+k]*Hc11r[n*ian+k];
		      sumH01r += LNO_coes[0][Mi][m*ian+k]*Hc01r[n*ian+k];
		      sumH01i += LNO_coes[0][Mi][m*ian+k]*Hc01i[n*ian+k];
		    }

		    HNC_ref(Anum+m,     Bnum+n     ).r = sumH00r;
		    HNC_ref(Anum+m,     Bnum+n     ).i = 0.0;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).r = sumH11r;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).i = 0.0;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).r = sumH01r;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).i = sumH01i;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).r = HNC_ref(Anum+m, Bnum+n+NUMH).r;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).i =-HNC_ref(Anum+m, Bnum+n+NUMH).i;
		  }
		}
	      }

	      /* spin-orbit coupling or LDA+U */
	      else {

		for (m=0; m<ian; m++){
		  for (n=0; n<jan; n++){
		    Hc00r[n*ian+m] =  Hks[0][Mi][kl][m][n];
		    Hc00i[n*ian+m] = ImNL[0][Mi][kl][m][n];
		    Hc11r[n*ian+m] =  Hks[1][Mi][kl][m][n];
		    Hc11i[n*ian+m] = ImNL[1][Mi][kl][m][n];
		    Hc01r[n*ian+m] =  Hks[2][Mi][kl][m][n];
		    Hc01i[n*ian+m] =  Hks[3][Mi][kl][m][n] + ImNL[2][Mi][kl][m][n];
		  }
		}

		for (m=0; m<ino; m++){
		  for (n=0; n<jan; n++){

		    sumH00r = 0.0;
		    sumH00i = 0.0;
		    sumH11r = 0.0;
		    sumH11i = 0.0;
		    sumH01r = 0.0;
		    sumH01i = 0.0;

		    for (k=0; k<ian; k++){
		      sumH00r += LNO_coes[0][Mi][m*ian+k]*Hc00r[n*ian+k];
		      sumH00i += LNO_coes[0][Mi][m*ian+k]*Hc00i[n*ian+k];
		      sumH11r += LNO_coes[0][Mi][m*ian+k]*Hc11r[n*ian+k];
		      sumH11i += LNO_coes[0][Mi][m*ian+k]*Hc11i[n*ian+k];
		      sumH01r += LNO_coes[0][Mi][m*ian+k]*Hc01r[n*ian+k];
		      sumH01i += LNO_coes[0][Mi][m*ian+k]*Hc01i[n*ian+k];
		    }

		    HNC_ref(Anum+m,     Bnum+n     ).r = sumH00r;
		    HNC_ref(Anum+m,     Bnum+n     ).i = sumH00i;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).r = sumH11r;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).i = sumH11i;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).r = sumH01r;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).i = sumH01i;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).r = HNC_ref(Anum+m, Bnum+n+NUMH).r;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).i =-HNC_ref(Anum+m, Bnum+n+NUMH).i;
		  }
		}
	      }

	      if (LNO_recalc_flag==1){
		for (m=0; m<ian; m++){
		  for (n=0; n<jan; n++){
		    Sc[n*ian+m] = OLP0[Mi][kl][m][n];
		  }
		}

		for (m=0; m<ino; m++){
		  for (n=0; n<jan; n++){
		    sumS = 0.0;
		    for (k=0; k<ian; k++){
		      sumS += LNO_coes[0][Mi][m*ian+k]*Sc[n*ian+k];
		    }

		    SNC_ref(Anum+m, Bnum+n).r = sumS;
		    SNC_ref(Anum+m, Bnum+n).i = 0.0;
		  }
		}
	      }
	    }

            /* FNAN_DCLNO[Gc_AN]<i && FNAN_DCLNO[Gc_AN]<j */

	    else if ( FNAN_DCLNO[Gc_AN]<i && FNAN_DCLNO[Gc_AN]<j ){

	      /* non-spin-orbit coupling and non-LDA+U */
	      if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0
		  && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){

		for (m=0; m<ian; m++){
		  for (n=0; n<jan; n++){
		    Hc00r[n*ian+m] = Hks[0][Mi][kl][m][n];
		    Hc11r[n*ian+m] = Hks[1][Mi][kl][m][n];
		    Hc01r[n*ian+m] = Hks[2][Mi][kl][m][n];
		    Hc01i[n*ian+m] = Hks[3][Mi][kl][m][n];
		  }
		}

		for (m=0; m<ino; m++){
		  for (n=0; n<jan; n++){

		    sumH00r = 0.0;
		    sumH11r = 0.0;
		    sumH01r = 0.0;
		    sumH01i = 0.0;

		    for (k=0; k<ian; k++){
		      sumH00r += LNO_coes[0][Mi][m*ian+k]*Hc00r[n*ian+k];
		      sumH11r += LNO_coes[0][Mi][m*ian+k]*Hc11r[n*ian+k];
		      sumH01r += LNO_coes[0][Mi][m*ian+k]*Hc01r[n*ian+k];
		      sumH01i += LNO_coes[0][Mi][m*ian+k]*Hc01i[n*ian+k];
		    }

		    Hc00r_tmp[m*jan+n] = sumH00r;
		    Hc11r_tmp[m*jan+n] = sumH11r;
		    Hc01r_tmp[m*jan+n] = sumH01r;
		    Hc01i_tmp[m*jan+n] = sumH01i;
		  }
		}

		for (m=0; m<ino; m++){
		  for (n=0; n<jno; n++){

		    sumH00r = 0.0;
		    sumH11r = 0.0;
		    sumH01r = 0.0;
		    sumH01i = 0.0;

		    for (k=0; k<jan; k++){
		      sumH00r += Hc00r_tmp[m*jan+k]*LNO_coes[0][Mj][n*jan+k];
		      sumH11r += Hc11r_tmp[m*jan+k]*LNO_coes[0][Mj][n*jan+k];
		      sumH01r += Hc01r_tmp[m*jan+k]*LNO_coes[0][Mj][n*jan+k];
		      sumH01i += Hc01i_tmp[m*jan+k]*LNO_coes[0][Mj][n*jan+k];
		    }

		    HNC_ref(Anum+m,     Bnum+n     ).r = sumH00r;
		    HNC_ref(Anum+m,     Bnum+n     ).i = 0.0;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).r = sumH11r;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).i = 0.0;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).r = sumH01r;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).i = sumH01i;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).r = HNC_ref(Anum+m, Bnum+n+NUMH).r;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).i =-HNC_ref(Anum+m, Bnum+n+NUMH).i;
		  }
		}
	      }

	      /* spin-orbit coupling or LDA+U */
	      else {

		for (m=0; m<ian; m++){
		  for (n=0; n<jan; n++){
		    Hc00r[n*ian+m] =  Hks[0][Mi][kl][m][n];
		    Hc00i[n*ian+m] = ImNL[0][Mi][kl][m][n];
		    Hc11r[n*ian+m] =  Hks[1][Mi][kl][m][n];
		    Hc11i[n*ian+m] = ImNL[1][Mi][kl][m][n];
		    Hc01r[n*ian+m] =  Hks[2][Mi][kl][m][n];
		    Hc01i[n*ian+m] =  Hks[3][Mi][kl][m][n] + ImNL[2][Mi][kl][m][n];
		  }
		}

		for (m=0; m<ino; m++){
		  for (n=0; n<jan; n++){

		    sumH00r = 0.0;
		    sumH00i = 0.0;
		    sumH11r = 0.0;
		    sumH11i = 0.0;
		    sumH01r = 0.0;
		    sumH01i = 0.0;

		    for (k=0; k<ian; k++){
		      sumH00r += LNO_coes[0][Mi][m*ian+k]*Hc00r[n*ian+k];
		      sumH00i += LNO_coes[0][Mi][m*ian+k]*Hc00i[n*ian+k];
		      sumH11r += LNO_coes[0][Mi][m*ian+k]*Hc11r[n*ian+k];
		      sumH11i += LNO_coes[0][Mi][m*ian+k]*Hc11i[n*ian+k];
		      sumH01r += LNO_coes[0][Mi][m*ian+k]*Hc01r[n*ian+k];
		      sumH01i += LNO_coes[0][Mi][m*ian+k]*Hc01i[n*ian+k];
		    }

		    Hc00r_tmp[m*jan+n] = sumH00r;
		    Hc00i_tmp[m*jan+n] = sumH00i;
		    Hc11r_tmp[m*jan+n] = sumH11r;
		    Hc11i_tmp[m*jan+n] = sumH11i;
		    Hc01r_tmp[m*jan+n] = sumH01r;
		    Hc01i_tmp[m*jan+n] = sumH01i;
		  }
		}

		for (m=0; m<ino; m++){
		  for (n=0; n<jno; n++){

		    sumH00r = 0.0;
		    sumH00i = 0.0;
		    sumH11r = 0.0;
		    sumH11i = 0.0;
		    sumH01r = 0.0;
		    sumH01i = 0.0;

		    for (k=0; k<jan; k++){
		      sumH00r += Hc00r_tmp[m*jan+k]*LNO_coes[0][Mj][n*jan+k];
		      sumH00i += Hc00i_tmp[m*jan+k]*LNO_coes[0][Mj][n*jan+k];
		      sumH11r += Hc11r_tmp[m*jan+k]*LNO_coes[0][Mj][n*jan+k];
		      sumH11i += Hc11i_tmp[m*jan+k]*LNO_coes[0][Mj][n*jan+k];
		      sumH01r += Hc01r_tmp[m*jan+k]*LNO_coes[0][Mj][n*jan+k];
		      sumH01i += Hc01i_tmp[m*jan+k]*LNO_coes[0][Mj][n*jan+k];
		    }

		    HNC_ref(Anum+m,     Bnum+n     ).r = sumH00r;
		    HNC_ref(Anum+m,     Bnum+n     ).i = sumH00i;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).r = sumH11r;
		    HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).i = sumH11i;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).r = sumH01r;
		    HNC_ref(Anum+m,     Bnum+n+NUMH).i = sumH01i;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).r = HNC_ref(Anum+m, Bnum+n+NUMH).r;
		    HNC_ref(Bnum+n+NUMH,Anum+m     ).i =-HNC_ref(Anum+m, Bnum+n+NUMH).i;
		  }
		}
	      }

	      if (LNO_recalc_flag==1){

		for (m=0; m<ian; m++){
		  for (n=0; n<jan; n++){
		    Sc[n*ian+m] = OLP0[Mi][kl][m][n];
		  }
		}

		for (m=0; m<ino; m++){
		  for (n=0; n<jan; n++){
		    sumS = 0.0;
		    for (k=0; k<ian; k++){
		      sumS += LNO_coes[0][Mi][m*ian+k]*Sc[n*ian+k];
		    }
		    Stmp[m*jan+n] = sumS;
		  }
		}

		for (m=0; m<ino; m++){
		  for (n=0; n<jno; n++){
		    sumS = 0.0;
		    for (k=0; k<jan; k++){
		      sumS += Stmp[m*jan+k]*LNO_coes[0][Mj][n*jan+k];
		    }

		    SNC_ref(Anum+m, Bnum+n).r = sumS;
		    SNC_ref(Anum+m, Bnum+n).i = 0.0;
		  }
		}
	      }
	    }

	  }

          /* else */

	  else{

	    if      ( i<=FNAN_DCLNO[Gc_AN] && j<=FNAN_DCLNO[Gc_AN] ) { ni = ian; nj = jan; }
	    else if ( i<=FNAN_DCLNO[Gc_AN] && FNAN_DCLNO[Gc_AN]<j )  { ni = ian; nj = jno; }
	    else if ( FNAN_DCLNO[Gc_AN]<i && j<=FNAN_DCLNO[Gc_AN] )  { ni = ino; nj = jan; }
	    else if ( FNAN_DCLNO[Gc_AN]<i && FNAN_DCLNO[Gc_AN]<j )   { ni = ino; nj = jno; }

	    for (m=0; m<ni; m++){
	      for (n=0; n<nj; n++){
		HNC_ref(Anum+m,     Bnum+n     ).r = 0.0;
		HNC_ref(Anum+m,     Bnum+n     ).i = 0.0;
		HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).r = 0.0;
		HNC_ref(Anum+m+NUMH,Bnum+n+NUMH).i = 0.0;
		HNC_ref(Anum+m,     Bnum+n+NUMH).r = 0.0;
		HNC_ref(Anum+m,     Bnum+n+NUMH).i = 0.0;
		HNC_ref(Bnum+n+NUMH,Anum+m     ).r = 0.0;
		HNC_ref(Bnum+n+NUMH,Anum+m     ).i = 0.0;
	      }
	    }

	    if (LNO_recalc_flag==1){
	      for (m=0; m<ni; m++){
		for (n=0; n<nj; n++){
		  SNC_ref(Anum+m, Bnum+n).r = 0.0;
		  SNC_ref(Anum+m, Bnum+n).i = 0.0;
		}
	      }
	    }
	  }
	}
      }

    } /* if ( Eigen_MPI_flag==0 || (Eigen_MPI_flag==1 && myid1==0) ) */

    /* MPI: BLAS_OLP and BLAS_H */

    if ( Eigen_MPI_flag==1 ){

      if (myworld2_DCLNO==0){
        MPI_Bcast(&BLAS_H[0], NUM*NUM*sizeof(dcomplex), MPI_BYTE, 0, MPI_CommWD2_DCLNO[myworld2_DCLNO]);
        if (LNO_recalc_flag==1){
          MPI_Bcast(&BLAS_OLP[0], NUM*NUM*sizeof(dcomplex), MPI_BYTE, 0, MPI_CommWD2_DCLNO[myworld2_DCLNO]);
	}
      }

    } /* if ( Eigen_MPI_flag==1 ) */

    if (measure_time){
      dtime(&etime);
      time1 += etime - stime;
    }

    /*
    printf("ABC0 %2d %2d %2d %2d\n",sizeof(BLAS_OLP),sizeof(tmp1),sizeof(l),sizeof(dcomplex));
    */


    if (Gc_AN==1){

      /*
    printf("HNC_ref.r\n");
    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){
        printf("HNC_ref.r i=%2d j=%2d %7.4f\n",i,j,HNC_ref(i,j).r);
      }
    }

    printf("HNC_ref.i\n");
    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){
        printf("HNC_ref.i i=%2d j=%2d %7.4f\n",i,j,HNC_ref(i,j).i);
      }
    }
      */

    }

    /*
      for (i1=1; i1<=NUM; i1++){
	for (j1=1; j1<=NUM; j1++){
	  C[i1][j1] = HNC_ref(i1,j1);
	}
      }

    EigenBand_lapack(C, ko, NUM, NUM, 1);

    if (Gc_AN==1){
      for (l=1; l<=NUM; l++){
	printf("ABC8 myid0=%2d l=%2d ko=%15.12f\n",myid0,l,ko[l]);fflush(stdout);
      }
    }
    */

    /****************************************************
       Solve the generalized eigenvalue problem
       HC = SCE
    ****************************************************/

    if (measure_time) dtime(&stime);

    use_gpu_task_nc = (use_gpu_accel_nc && Eigen_MPI_flag==0 &&
                       DCLNO_NonColGpuEigenThreshold() <= NUM);

    if (use_gpu_task_nc){

      /* same rule as the CPU flow below */
      if (SCF_iter<=2){
        NUM2 = NUM;
      }
      else{
        NUM2 = HO_TC[Mc_AN] + 100;
        if (NUM<NUM2) NUM2 = NUM;
      }

      DCLNO_Solve_NonCol_GpuSolver(NUM, NUMH, NUM2, LNO_recalc_flag,
                                   BLAS_OLP, BLAS_H, ko);

      if (measure_time){
        dtime(&etime);
        time3 += etime - stime;
      }
    }
    else{

    /* if (LNO_recalc_flag==1) */

    if (LNO_recalc_flag==1){

      /****************************************************
                          diagonalize S
      ****************************************************/

      for (i1=1; i1<=NUMH; i1++){
	for (j1=1; j1<=NUMH; j1++){
	  C[i1][j1] = SNC_ref(i1,j1);
	}
      }

      bcast_flag = 1;
      Eigen_PHH(MPI_CommWD2_DCLNO[myworld2_DCLNO],C,ko,NUMH,NUMH,bcast_flag);

      /***********************************************
                calculation of S^{-1/2}
      ************************************************/

      for (l=1; l<=NUMH; l++)  ko[l] = 1.0/sqrt(fabs(ko[l]));

       for (i=0; i<NUM*NUM; i++){
         BLAS_OLP[i].r = 0.0; BLAS_OLP[i].i = 0.0;
       }

      for (i1=1; i1<=NUMH; i1++){
	for (j1=1; j1<=NUMH; j1++){
	  SNC2_ref(i1,     j1     ).r = C[i1][j1].r*ko[j1];
	  SNC2_ref(i1,     j1     ).i = C[i1][j1].i*ko[j1];
	  SNC2_ref(i1+NUMH,j1+NUMH).r = C[i1][j1].r*ko[j1];
	  SNC2_ref(i1+NUMH,j1+NUMH).i = C[i1][j1].i*ko[j1];
	}
      }
    }

    if (measure_time){
      dtime(&etime);
      time2 += etime - stime;
    }

    /***********************************************
              transform Hamiltonian matrix
    ************************************************/

    if (Gc_AN==1){

      /*
    printf("SNC2_ref.r\n");
    for (i=1; i<=NUMH; i++){
      for (j=1; j<=NUMH; j++){
        printf("SNC2_ref.r i=%2d j=%2d %7.4f\n",i,j,SNC2_ref(i,j).r);
      }
    }
      */

    /*
    printf("SNC2_ref.i\n");
    for (i=1; i<=40; i++){
      for (j=1; j<=40; j++){
        printf("SNC2_ref.i i=%2d j=%2d %7.4f\n",i,j,SNC2_ref(i,j).i);
      }
    }
    */
    }

    if (measure_time) dtime(&stime);

    /* H * U * ko^{-1/2} */
    /* C is distributed by row in each processor */

    /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */

    ns = is1[myid2];
    ne = ie1[myid2];

    BM = NUM;
    BN = ne - ns + 1;
    BK = NUM;

    ctmp1.r = 1.0; ctmp1.i = 0.0;
    ctmp2.r = 0.0; ctmp2.i = 0.0;

    if (0<BN){

      F77_NAME(zgemm,ZGEMM)("N","N", &BM,&BN,&BK,
			    &ctmp1,
			    &BLAS_H[0], &BM,
			    &BLAS_OLP[(ns-1)*NUM], &BK,
			    &ctmp2,
			    &BLAS_C[(ns-1)*NUM], &BM);
    }

    /* ko^{-1/2} * U^+ H * U * ko^{-1/2} */
    /* H is distributed by row in each processor */

    ns = is1[myid2];
    ne = ie1[myid2];

    BM = NUM;
    BN = ne - ns + 1;
    BK = NUM;

    ctmp1.r = 1.0; ctmp1.i = 0.0;
    ctmp2.r = 0.0; ctmp2.i = 0.0;

    if (0<BN){
      F77_NAME(zgemm,ZGEMM)("C","N", &BM,&BN,&BK,
			    &ctmp1,
			    &BLAS_OLP[0], &BM,
			    &BLAS_C[(ns-1)*NUM], &BK,
			    &ctmp2,
			    &BLAS_H[(ns-1)*NUM], &BM);
    }

    for (j1=ns; j1<=ne; j1++){
      for (i1=1; i1<=NUM; i1++){
	C[j1][i1] = BLAS_H[(j1-1)*NUM+i1-1];
      }
    }

    /*
    if (Gc_AN==1){

      printf("WWW1 NUM=%2d ns=%2d ne=%2d\n",NUM,ns,ne);

    printf("BLAS_C.r\n");
    for (i=1; i<=40; i++){
      for (j=1; j<=40; j++){
        printf("BLAS_C.r i=%2d j=%2d %7.4f\n",i,j,BLAS_C[j*NUM+i].r);
      }
    }

    printf("BLAS_C.i\n");
    for (i=1; i<=40; i++){
      for (j=1; j<=40; j++){
        printf("BLAS_C.i i=%2d j=%2d %7.4f\n",i,j,BLAS_C[j*NUM+i].i);
      }
    }

    }
    */

    /* broadcast C */

    if (Eigen_MPI_flag==1){

      BroadCast_ComplexMatrix( MPI_CommWD2_DCLNO[myworld2_DCLNO], C,
			       NUM, is1, ie1, myid2, numprocs2,
			       stat_send, request_send, request_recv );
    }

    if (Gc_AN==1){

      /*
    printf("C.r\n");
    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){
        printf("ABC2 C.r i=%2d j=%2d %7.4f\n",i,j,C[i][j].r);
      }
    }

    printf("C.i\n");
    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){
        printf("ABC2 C.i i=%2d j=%2d %7.4f\n",i,j,C[i][j].i);
      }
    }
      */
      /*
    printf("C.r\n");
    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){
        printf("%7.4f",C[i][j].r);
      }
      printf("\n");
    }

    printf("C.i\n");
    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){
        printf("%7.4f",C[i][j].i);
      }
      printf("\n");
    }
      */

    }
    /*
    MPI_Finalize();
    exit(0);
    */

    /***************************************************
      estimate the number of orbitals to be calculated
    ***************************************************/

    if (Eigen_MPI_flag==1){
      MPI_Bcast(&HO_TC[Mc_AN], 1, MPI_INT, 0, MPI_CommWD2_DCLNO[myworld2_DCLNO]);
    }

    if (SCF_iter<=2){
      NUM2 = NUM;
    }
    else{
      NUM2 = HO_TC[Mc_AN] + 100;
    }

    if (NUM<NUM2) NUM2 = NUM;

    /*
    if (myid0==0){
      printf("ABC0 NUM2=%2d\n",NUM2);
    }
    */

    /* printf("ABC0 myid0=%2d NUM=%2d NUM2=%2d\n",myid0,NUM,NUM2); */

    /* making is2 and ie2 */

    if ( numprocs2<=NUM2 ){

      for (ID=0; ID<numprocs2; ID++){
	is2[ID] = (int)((NUM2*ID)/numprocs2) + 1;
	ie2[ID] = (int)((NUM2*(ID+1))/numprocs2);
      }
    }

    else{

      for (ID=0; ID<NUM2; ID++){
	is2[ID] = ID + 1;
	ie2[ID] = ID + 1;
      }
      for (ID=NUM2; ID<numprocs2; ID++){
	is2[ID] =  1;
	ie2[ID] = -2;
      }
    }

    /***********************************************
     diagonalize the transformed Hamiltonian matrix
    ************************************************/

    bcast_flag = 0;
    Eigen_PHH(MPI_CommWD2_DCLNO[myworld2_DCLNO],C,ko,NUM,NUM2,bcast_flag);

    /* The H matrix is distributed by column */
    /* C to H */

    for (i1=1; i1<=NUM; i1++){
      for (j1=is2[myid2]; j1<=ie2[myid2]; j1++){
	BLAS_H[(j1-1)*NUM+i1-1] = C[i1][j1];
      }
    }

    /***********************************************
      transformation to the original eigenvectors.
                     NOTE 244P
    ***********************************************/

    /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */

    ns = is2[myid2];
    ne = ie2[myid2];

    BM = NUM;
    BN = ne - ns + 1;
    BK = NUM;

    ctmp1.r = 1.0; ctmp1.i = 0.0;
    ctmp2.r = 0.0; ctmp2.i = 0.0;

    if (0<BN){
      F77_NAME(zgemm,ZGEMM)("N","N", &BM,&BN,&BK,
			    &ctmp1,
			    &BLAS_OLP[0], &BM,
			    &BLAS_H[(ns-1)*NUM], &BK,
			    &ctmp2,
			    &BLAS_C[(ns-1)*NUM], &BM);
    }

    for (j1=ns; j1<=ne; j1++){
      for (i1=1; i1<=NUM; i1++){
	C[j1][i1] = BLAS_C[(j1-1)*NUM+i1-1];
      }
    }

    /* broadcast C:
       C is distributed by row in each processor
    */

    if (Eigen_MPI_flag==1){

      BroadCast_ComplexMatrix( MPI_CommWD2_DCLNO[myworld2_DCLNO], C,
			       NUM, is2, ie2, myid2, numprocs2,
			       stat_send, request_send, request_recv );
    }

    /* C to H
       H consists of column vectors
    */

    for (i1=1; i1<=NUM2; i1++){
      for (j1=1; j1<=NUM; j1++){
	HNC_ref(i1,j1) = C[i1][j1];
      }
    }

    if (measure_time){
      dtime(&etime);
      time3 += etime - stime;
    }

    } /* else: CPU eigensolver fallback */

    /***********************************************
        store eigenvalues and residues of poles
    ***********************************************/

    if (measure_time) dtime(&stime);

    if ( Eigen_MPI_flag==0 || (Eigen_MPI_flag==1 && myid2==0) ){

      for (i1=1; i1<=NUM2; i1++){
	EVal[Mc_AN][i1] = ko[i1];
      }

      /******************************************************
        set an energy range (-Erange+ChemP to Erange+ChemP)
        of eigenvalues used to store the Residues.
      ******************************************************/

      Erange = 10.0/27.2113845;  /* in hartree, corresponds to 8.0 eV */

      /***********************************************
          find LO_TC and HO_TC
      ***********************************************/

      /* LO_TC */
      i = 1;
      ip = 1;
      po1 = 0;
      do{
	if ( (ChemP-Erange)<EVal[Mc_AN][i]){
	  ip = i;
	  po1 = 1;
	}
	i++;
      } while (po1==0 && i<=NUM2);

      LO_TC[Mc_AN] = ip;

      /* HO_TC */
      i = 1;
      ip = NUM2;
      po1 = 0;
      do{
	if ( (ChemP+Erange)<EVal[Mc_AN][i]){
	  ip = i;
	  po1 = 1;
	}
	i++;
      } while (po1==0 && i<=NUM2);

      HO_TC[Mc_AN] = ip;

      n2 = HO_TC[Mc_AN] - LO_TC[Mc_AN] + 3;
      if (n2<1) n2 = 1;

      /***********************************************
          store residues of poles
      ***********************************************/

      wanA = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wanA];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	wanB = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wanB];
	Bnum = MP[h_AN];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){

	    /* <allocation of Residues */
            size_Residues += n2*3;
            for (spin=0; spin<=2; spin++){
              Residues[spin][Mc_AN-1][h_AN][i][j] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
	    }
	    /* allocation of Residues> */

            for (spin=0; spin<=2; spin++){
              Residues[spin][Mc_AN-1][h_AN][i][j][0].r = 0.0;
              Residues[spin][Mc_AN-1][h_AN][i][j][1].r = 0.0;
              Residues[spin][Mc_AN-1][h_AN][i][j][0].i = 0.0;
              Residues[spin][Mc_AN-1][h_AN][i][j][1].i = 0.0;
	    }

	    for (i1=1; i1<LO_TC[Mc_AN]; i1++){

	      /* Re11 */
	      tmp1 = HNC_ref(i1,1+i).r*HNC_ref(i1,Bnum+j).r + HNC_ref(i1,1+i).i*HNC_ref(i1,Bnum+j).i;
	      Residues[0][Mc_AN-1][h_AN][i][j][0].r += tmp1;
	      Residues[0][Mc_AN-1][h_AN][i][j][1].r += tmp1*EVal[Mc_AN][i1];

	      /* Re22 */
	      tmp1 = HNC_ref(i1,1+i+NUMH).r*HNC_ref(i1,Bnum+j+NUMH).r + HNC_ref(i1,1+i+NUMH).i*HNC_ref(i1,Bnum+j+NUMH).i;
	      Residues[1][Mc_AN-1][h_AN][i][j][0].r += tmp1;
	      Residues[1][Mc_AN-1][h_AN][i][j][1].r += tmp1*EVal[Mc_AN][i1];

	      /* Re12 */
	      tmp1 = HNC_ref(i1,1+i).r*HNC_ref(i1,Bnum+j+NUMH).r + HNC_ref(i1,1+i).i*HNC_ref(i1,Bnum+j+NUMH).i;
	      Residues[2][Mc_AN-1][h_AN][i][j][0].r += tmp1;
	      Residues[2][Mc_AN-1][h_AN][i][j][1].r += tmp1*EVal[Mc_AN][i1];

	      /* Im12 */
	      tmp1 = -HNC_ref(i1,1+i).r*HNC_ref(i1,Bnum+j+NUMH).i + HNC_ref(i1,1+i).i*HNC_ref(i1,Bnum+j+NUMH).r;
	      Residues[2][Mc_AN-1][h_AN][i][j][0].i += tmp1;
	      Residues[2][Mc_AN-1][h_AN][i][j][1].i += tmp1*EVal[Mc_AN][i1];
	    }

	    /* spin-orbit coupling or LDA+U */
	    if ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
		 || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

	      for (i1=1; i1<LO_TC[Mc_AN]; i1++){

		/* Im11 */
		tmp1 = -HNC_ref(i1,1+i).r*HNC_ref(i1,Bnum+j).i + HNC_ref(i1,1+i).i*HNC_ref(i1,Bnum+j).r;
		Residues[0][Mc_AN-1][h_AN][i][j][0].i += tmp1;
		Residues[0][Mc_AN-1][h_AN][i][j][1].i += tmp1*EVal[Mc_AN][i1];

		/* Im22 */
		tmp1 = -HNC_ref(i1,1+i+NUMH).r*HNC_ref(i1,Bnum+j+NUMH).i + HNC_ref(i1,1+i+NUMH).i*HNC_ref(i1,Bnum+j+NUMH).r;
		Residues[1][Mc_AN-1][h_AN][i][j][0].i += tmp1;
		Residues[1][Mc_AN-1][h_AN][i][j][1].i += tmp1*EVal[Mc_AN][i1];
	      }
	    }

	    for (i1=LO_TC[Mc_AN]; i1<=HO_TC[Mc_AN]; i1++){

              i2 = i1 - LO_TC[Mc_AN] + 2;

	      /* Re11 */
	      Residues[0][Mc_AN-1][h_AN][i][j][i2].r = HNC_ref(i1,1+i).r*HNC_ref(i1,Bnum+j).r
                                                     + HNC_ref(i1,1+i).i*HNC_ref(i1,Bnum+j).i;

	      /* Re22 */
	      Residues[1][Mc_AN-1][h_AN][i][j][i2].r = HNC_ref(i1,1+i+NUMH).r*HNC_ref(i1,Bnum+j+NUMH).r
                                                     + HNC_ref(i1,1+i+NUMH).i*HNC_ref(i1,Bnum+j+NUMH).i;

	      /* Re12 */
	      Residues[2][Mc_AN-1][h_AN][i][j][i2].r = HNC_ref(i1,1+i).r*HNC_ref(i1,Bnum+j+NUMH).r
                                                     + HNC_ref(i1,1+i).i*HNC_ref(i1,Bnum+j+NUMH).i;

	      /* Im12 */
	      Residues[2][Mc_AN-1][h_AN][i][j][i2].i = -HNC_ref(i1,1+i).r*HNC_ref(i1,Bnum+j+NUMH).i
                                                       +HNC_ref(i1,1+i).i*HNC_ref(i1,Bnum+j+NUMH).r;

	      /* spin-orbit coupling or LDA+U */
	      if ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
		   || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

		/* Im11 */
		Residues[0][Mc_AN-1][h_AN][i][j][i2].i =-HNC_ref(i1,1+i).r*HNC_ref(i1,Bnum+j).i
                                                        +HNC_ref(i1,1+i).i*HNC_ref(i1,Bnum+j).r;

		/*
                printf("VVV1 Mc_AN=%2d h_AN=%2d i=%2d j=%2d i1=%2d %15.10f\n",
                        Mc_AN,h_AN,i,j,i1,Residues[0][Mc_AN-1][h_AN][i][j][i2].i);
		*/

		/* Im22 */
		Residues[1][Mc_AN-1][h_AN][i][j][i2].i =-HNC_ref(i1,1+i+NUMH).r*HNC_ref(i1,Bnum+j+NUMH).i
                                                        +HNC_ref(i1,1+i+NUMH).i*HNC_ref(i1,Bnum+j+NUMH).r;
	      }
	    }
	  }
	}
      }

    } /* end of if ( Eigen_MPI_flag==0 || (Eigen_MPI_flag==1 && myid2==0) ) */

    if (measure_time){
      dtime(&etime);
      time4 += etime - stime;
    }

    if ( Eigen_MPI_flag==0 || (Eigen_MPI_flag==1 && myid1==0) ){
      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

  } /* end of Mc_AN */

  /* reset Matomnum */

  if (Eigen_MPI_flag==1 && myid1!=0) Matomnum = 0;
  if (Eigen_MPI_flag==1 && myid2==0) Matomnum = 1;

  /* projected DOS, find chemical potential, eigenenergy, density and energy density matrices */

  if ( strcasecmp(mode,"scf")==0 || strcasecmp(mode,"full")==0 ){

    /****************************************************
                  calculate projected DOS
    ****************************************************/

    if (measure_time) dtime(&stime);

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wanA = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wanA];

      for (i1=0; i1<(Msize[Mc_AN]+3); i1++){
	PDOS_DC[Mc_AN-1][i1] = 0.0;
      }

      for (i=0; i<tno1; i++){
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wanB = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wanB];
	  for (j=0; j<tno2; j++){

	    tmp1 = OLP0[Mc_AN][h_AN][i][j];
            PDOS_DC[Mc_AN-1][0] += (Residues[0][Mc_AN-1][h_AN][i][j][0].r+Residues[1][Mc_AN-1][h_AN][i][j][0].r)*tmp1;
            PDOS_DC[Mc_AN-1][1] += (Residues[0][Mc_AN-1][h_AN][i][j][1].r+Residues[1][Mc_AN-1][h_AN][i][j][1].r)*tmp1;

	    for (i1=LO_TC[Mc_AN]; i1<=HO_TC[Mc_AN]; i1++){
              i2 = i1 - LO_TC[Mc_AN] + 2;
	      PDOS_DC[Mc_AN-1][i2] += (Residues[0][Mc_AN-1][h_AN][i][j][i2].r+Residues[1][Mc_AN-1][h_AN][i][j][i2].r)*tmp1;
	    }
	  }
	}
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    if (measure_time){
      dtime(&etime);
      time5 += etime - stime;
    }

    /****************************************************
                   find chemical potential
    ****************************************************/

    MPI_Barrier(mpi_comm_level1);
    if (measure_time) dtime(&stime);

    po = 0;
    loopN = 0;
    Dnum = 100.0;

    if (SCF_iter<=2){
      ChemP_MIN =-10.0;
      ChemP_MAX = 10.0;
    }
    else{
      ChemP_MIN = ChemP - 5.0;
      ChemP_MAX = ChemP + 5.0;
    }

    do {

      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
      My_Num_State = 0.0;

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	dtime(&Stime_atom);

	Gc_AN = M2G[Mc_AN];
        My_Num_State += PDOS_DC[Mc_AN-1][0];

        for (i=LO_TC[Mc_AN]; i<=HO_TC[Mc_AN]; i++){

          i1 = i - LO_TC[Mc_AN] + 2;
	  x = (EVal[Mc_AN][i] - ChemP)*Beta;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;

	  ex = exp(x);
	  ex1 = 1.0 + ex;
	  coe = PDOS_DC[Mc_AN-1][i1];

	  My_Num_State += coe/ex1;
	}

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }

      /* MPI, My_Num_State */

      MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      Dnum = (TZ - Num_State) - system_charge;

      if (0.0<=Dnum) ChemP_MIN = ChemP;
      else           ChemP_MAX = ChemP;

      if (fabs(Dnum)<1.0e-12) po = 1;

      if (myid0==Host_ID && 2<=level_stdout){
	printf("loopN=%2d ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",loopN,ChemP,TZ,Num_State);
      }

      loopN++;
    }
    while (po==0 && loopN<1000);

    if (measure_time){
      dtime(&etime);
      time6 += etime - stime;
    }

    /****************************************************
        eigenenergy by summing up eigenvalues
    ****************************************************/

    if (measure_time) dtime(&stime);

    My_Eele0[0] = 0.0;
    My_Eele0[1] = 0.0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      My_Eele0[0] += PDOS_DC[Mc_AN-1][1];

      for (i=LO_TC[Mc_AN]; i<=HO_TC[Mc_AN]; i++){

        i1 = i - LO_TC[Mc_AN] + 2;
	x = (EVal[Mc_AN][i] - ChemP)*Beta;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = 1.0/(1.0 + exp(x));

	My_Eele0[0] += FermiF*EVal[Mc_AN][i]*PDOS_DC[Mc_AN-1][i1];
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    /* MPI, My_Eele0 */

    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_Eele0[0], &Eele0[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    if (measure_time){
      dtime(&etime);
      time7 += etime - stime;
    }

    /****************************************************
       calculate density and energy density matrices

        CDM[0]  Re alpha alpha density matrix
        CDM[1]  Re beta  beta  density matrix
        CDM[2]  Re alpha beta  density matrix
        CDM[3]  Im alpha beta  density matrix
        iDM[0][0]  Im alpha alpha density matrix
        iDM[0][1]  Im beta  beta  density matrix

        EDM[0]  Re alpha alpha energy density matrix
        EDM[1]  Re beta  beta  energy density matrix
        EDM[2]  Re alpha beta  energy density matrix
        EDM[3]  Im alpha beta  energy density matrix
    ****************************************************/

    if (measure_time) dtime(&stime);

    /* initialize CDM, iDM, and EDM */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wanB = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wanB];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      CDM[spin][Mc_AN][h_AN][i][j] = 0.0;
	      EDM[spin][Mc_AN][h_AN][i][j] = 0.0;
	    }
	  }

	  /* spin-orbit coupling or LDA+U */
	  if ( (SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
		|| Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) && spin==0 ){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		iDM[0][0][Mc_AN][h_AN][i][j] = 0.0;
		iDM[0][1][Mc_AN][h_AN][i][j] = 0.0;
	      }
	    }
	  }
	}
      }
    }

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wanA = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wanA];

      for (i1=LO_TC[Mc_AN]; i1<=HO_TC[Mc_AN]; i1++){
	x   = (EVal[Mc_AN][i1] - ChemP)*Beta;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	ko[i1] = 1.0/(1.0 + exp(x));  /* temporal use of ko */
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	wanB = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wanB];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){

	    CDM[0][Mc_AN][h_AN][i][j] = Residues[0][Mc_AN-1][h_AN][i][j][0].r;
	    EDM[0][Mc_AN][h_AN][i][j] = Residues[0][Mc_AN-1][h_AN][i][j][1].r;

	    CDM[1][Mc_AN][h_AN][i][j] = Residues[1][Mc_AN-1][h_AN][i][j][0].r;
	    EDM[1][Mc_AN][h_AN][i][j] = Residues[1][Mc_AN-1][h_AN][i][j][1].r;

	    CDM[2][Mc_AN][h_AN][i][j] = Residues[2][Mc_AN-1][h_AN][i][j][0].r;
	    EDM[2][Mc_AN][h_AN][i][j] = Residues[2][Mc_AN-1][h_AN][i][j][1].r;

	    CDM[3][Mc_AN][h_AN][i][j] = Residues[2][Mc_AN-1][h_AN][i][j][0].i;
	    EDM[3][Mc_AN][h_AN][i][j] = Residues[2][Mc_AN-1][h_AN][i][j][1].i;

	    /* spin-orbit coupling or LDA+U */
	    if (SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
		|| Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

	      iDM[0][0][Mc_AN][h_AN][i][j] = Residues[0][Mc_AN-1][h_AN][i][j][0].i;
	      iDM[0][1][Mc_AN][h_AN][i][j] = Residues[1][Mc_AN-1][h_AN][i][j][0].i;
	    }

            for (i1=LO_TC[Mc_AN]; i1<=HO_TC[Mc_AN]; i1++){

              i2 = i1 - LO_TC[Mc_AN] + 2;

	      tmp1 = ko[i1]*Residues[0][Mc_AN-1][h_AN][i][j][i2].r;
	      sum1 = tmp1;
	      sum2 = tmp1*EVal[Mc_AN][i1];

	      CDM[0][Mc_AN][h_AN][i][j] += sum1;
	      EDM[0][Mc_AN][h_AN][i][j] += sum2;

	      tmp1 = ko[i1]*Residues[1][Mc_AN-1][h_AN][i][j][i2].r;
	      sum1 = tmp1;
	      sum2 = tmp1*EVal[Mc_AN][i1];

	      CDM[1][Mc_AN][h_AN][i][j] += sum1;
	      EDM[1][Mc_AN][h_AN][i][j] += sum2;

	      tmp1 = ko[i1]*Residues[2][Mc_AN-1][h_AN][i][j][i2].r;
	      sum1 = tmp1;
	      sum2 = tmp1*EVal[Mc_AN][i1];

	      CDM[2][Mc_AN][h_AN][i][j] += sum1;
	      EDM[2][Mc_AN][h_AN][i][j] += sum2;

	      tmp1 = ko[i1]*Residues[2][Mc_AN-1][h_AN][i][j][i2].i;
	      sum1 = tmp1;
	      sum2 = tmp1*EVal[Mc_AN][i1];

	      CDM[3][Mc_AN][h_AN][i][j] += sum1;
	      EDM[3][Mc_AN][h_AN][i][j] += sum2;

	      /* spin-orbit coupling or LDA+U */
	      if (SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
		  || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

		iDM[0][0][Mc_AN][h_AN][i][j] += ko[i1]*Residues[0][Mc_AN-1][h_AN][i][j][i2].i;
		iDM[0][1][Mc_AN][h_AN][i][j] += ko[i1]*Residues[1][Mc_AN-1][h_AN][i][j][i2].i;
	      }
	    }
	  }
	}
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    /*
      if (myid0==0){
	for (l=1; l<=NUMH; l++){
	  printf("ABC9 myid0=%2d l=%2d ko=%15.12f\n",myid0,l,ko[l]);fflush(stdout);
	}
      }
      MPI_Finalize();
      exit(0);
    */

    /****************************************************
                       reset Matomnum
    ****************************************************/

    if (Eigen_MPI_flag==1 && myid1!=0) Matomnum = 0;

    /****************************************************
                        bond energies
    ****************************************************/

    My_Eele1[0] = 0.0;

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (j=0; j<=FNAN[GA_AN]; j++){

        GB_AN = natn[GA_AN][j];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];

	/* non-spin-orbit coupling and non-LDA+U */
	if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0
	    && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){

	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){

	      My_Eele1[0] +=
		+ CDM[0][MA_AN][j][k][l]*Hks[0][MA_AN][j][k][l]
		+ CDM[1][MA_AN][j][k][l]*Hks[1][MA_AN][j][k][l]
		+ 2.0*CDM[2][MA_AN][j][k][l]*Hks[2][MA_AN][j][k][l]
		- 2.0*CDM[3][MA_AN][j][k][l]*Hks[3][MA_AN][j][k][l];
	    }
	  }
	}

	/* spin-orbit coupling or LDA+U */
	else {
	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){

	      My_Eele1[0] +=
		+ CDM[0][MA_AN][j][k][l]*Hks[0][MA_AN][j][k][l]
		- iDM[0][0][MA_AN][j][k][l]*ImNL[0][MA_AN][j][k][l]
		+ CDM[1][MA_AN][j][k][l]*Hks[1][MA_AN][j][k][l]
		- iDM[0][1][MA_AN][j][k][l]*ImNL[1][MA_AN][j][k][l]
		+ 2.0*CDM[2][MA_AN][j][k][l]*Hks[2][MA_AN][j][k][l]
		- 2.0*CDM[3][MA_AN][j][k][l]*(Hks[3][MA_AN][j][k][l]
				            +ImNL[2][MA_AN][j][k][l]);
	    }
	  }
	}

      }
    }

    /* MPI, My_Eele1 */
    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_Eele1[0], &Eele1[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    if (3<=level_stdout && myid0==Host_ID){
      printf("Eele0=%15.12f\n",Eele0[0]);
      printf("Eele1=%15.12f\n",Eele1[0]);
    }

    if (measure_time){
      dtime(&etime);
      time8 += etime - stime;
    }

  } /* if ( strcasecmp(mode,"scf")==0 || strcasecmp(mode,"full")==0 ) */

  else if ( strcasecmp(mode,"dos")==0 ){

    if ( Eigen_MPI_flag==0 || (Eigen_MPI_flag==1 && myid2==0) ){
      Save_DOS_NonCol(Residues,OLP0,EVal,LO_TC,HO_TC);
    }
  }

  if (measure_time){
    printf("Divide_Conquer_LNO myid0=%2d NUM=%2d time0=%7.3f time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f time8=%7.3f\n",
	   myid0,NUM,time0,time1,time2,time3,time4,time5,time6,time7,time8);fflush(stdout);
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/

  if (firsttime){
    PrintMemory("Divide_Conquer_LNO: Residues",sizeof(double)*size_Residues,NULL);
  }

  if (Eigen_MPI_flag==1) Matomnum = 1;

  free(MP);

  free(is1);
  free(ie1);

  free(is2);
  free(ie2);

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(Sc);
  free(Stmp);

  free(Hc00r);
  free(Hc11r);
  free(Hc01r);
  free(Hc00i);
  free(Hc11i);
  free(Hc01i);

  free(Hc00r_tmp);
  free(Hc11r_tmp);
  free(Hc01r_tmp);
  free(Hc00i_tmp);
  free(Hc11i_tmp);
  free(Hc01i_tmp);

  free(Snd_H_Size);
  free(Rcv_H_Size);

  free(Snd_iHNL_Size);
  free(Rcv_iHNL_Size);

  free(Snd_S_Size);
  free(Rcv_S_Size);

  free(Msize);

  free(BLAS_H);
  free(BLAS_C);

  free(ko);

  for (i=0; i<(Max_Msize+2); i++){
    free(C[i]);
  }
  free(C);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    free(EVal[Mc_AN]);
  }
  free(EVal);

  if ( Eigen_MPI_flag==0 || (Eigen_MPI_flag==1 && myid2==0) ){

    for (spin=0; spin<3; spin++){
      for (Mc_AN=0; Mc_AN<Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN+1];
	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  wanB = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wanB];

	  for (i=0; i<tno1; i++){

	    for (j=0; j<tno2; j++){
	      free(Residues[spin][Mc_AN][h_AN][i][j]);
	    }
	    free(Residues[spin][Mc_AN][h_AN][i]);
	  }
	  free(Residues[spin][Mc_AN][h_AN]);
	}
	free(Residues[spin][Mc_AN]);
      }
      free(Residues[spin]);
    }
    free(Residues);
  }

  for (Mc_AN=0; Mc_AN<Matomnum; Mc_AN++){
    free(PDOS_DC[Mc_AN]);
  }
  free(PDOS_DC);

  /* reset Matomnum */

  if (Eigen_MPI_flag==1 && myid1!=0) Matomnum = 0;

  /* for PrintMemory */
  if (SCF_iter==2) firsttime=0;

  /* for time */
  /* free the GEMMul8 workspace and solver buffers so the grid/force phases
     of this SCF step see the memory; handles stay alive. */
  if (use_gpu_accel_nc && DCLNO_GpuTurnRelease()) {
    DCLNO_GpuSolverZ_ReleaseDeviceMemory();
    openmx_gemmul8ReleaseWorkspaces();
  }

  dtime(&TEtime);

  return (TEtime - TStime);
}

static void Save_DOS_NonCol(dcomplex ******Residues, double ****OLP0, double **EVal, int *LO_TC, int *HO_TC)
{
  int spin,Mc_AN,wanA,Gc_AN,tno1;
  int i1,i2,i,j,MaxL,l,h_AN,Gh_AN,wanB,tno2;
  double Stime_atom,Etime_atom;
  double tmp1,tmp2,tmp3,sum,SDup,SDdn;
  double Re11,Re22,Re12,Im12;
  double theta,phi,sit,cot,sip,cop;
  int i_vec[10];
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;
  int numprocs,myid,ID,tag;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* open file pointers */

  if (myid==Host_ID){

    sprintf(file_eig,"%s%s.Dos.val",filepath,filename);
    if ( (fp_eig=fopen(file_eig,"w"))==NULL ) {
      printf("cannot open a file %s\n",file_eig);
    }
  }

  sprintf(file_ev, "%s%s.Dos.vec%i",filepath,filename,myid);
  if ( (fp_ev=fopen(file_ev,"w"))==NULL ) {
    printf("cannot open a file %s\n",file_ev);
  }

  /****************************************************
                   save *.Dos.vec
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];
    wanA = WhatSpecies[Gc_AN];
    tno1 = Spe_Total_CNO[wanA];

    theta = Angle0_Spin[Gc_AN];
    phi   = Angle1_Spin[Gc_AN];

    sit = sin(theta);
    cot = cos(theta);
    sip = sin(phi);
    cop = cos(phi);

    fprintf(fp_ev,"<AN%d\n",Gc_AN);
    fprintf(fp_ev,"%d %d\n",HO_TC[Mc_AN]-LO_TC[Mc_AN]+1,HO_TC[Mc_AN]-LO_TC[Mc_AN]+1);

    for (i1=LO_TC[Mc_AN]; i1<=HO_TC[Mc_AN]; i1++){

      i2 = i1 - LO_TC[Mc_AN] + 2;

      fprintf(fp_ev,"%4d  %10.6f %10.6f ",i1,EVal[Mc_AN][i1],EVal[Mc_AN][i1]);

      for (i=0; i<tno1; i++){

	Re11 = 0.0;
	Re22 = 0.0;
	Re12 = 0.0;
	Im12 = 0.0;

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  wanB = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wanB];

	  for (j=0; j<tno2; j++){

	    Re11 += Residues[0][Mc_AN-1][h_AN][i][j][i2].r*
	                   OLP0[Mc_AN][h_AN][i][j];

	    Re22 += Residues[1][Mc_AN-1][h_AN][i][j][i2].r*
	                   OLP0[Mc_AN][h_AN][i][j];

	    Re12 += Residues[2][Mc_AN-1][h_AN][i][j][i2].r*
	                   OLP0[Mc_AN][h_AN][i][j];

	    Im12 += Residues[2][Mc_AN-1][h_AN][i][j][i2].i*
	                   OLP0[Mc_AN][h_AN][i][j];

	  }
	}

	tmp1 = 0.5*(Re11 + Re22);
	tmp2 = 0.5*cot*(Re11 - Re22);
	tmp3 = (Re12*cop - Im12*sip)*sit;

	SDup = tmp1 + tmp2 + tmp3;
	SDdn = tmp1 - tmp2 - tmp3;

	fprintf(fp_ev,"%8.5f %8.5f ",SDup,SDdn);
      }
      fprintf(fp_ev,"\n");
    }

    fprintf(fp_ev,"AN%d>\n",Gc_AN);

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  }

  /****************************************************
                   save *.Dos.val
  ****************************************************/

  if (myid==Host_ID){

    fprintf(fp_eig,"mode        5\n");
    fprintf(fp_eig,"NonCol      1\n");
    /*      fprintf(fp_eig,"N           %d\n",n); */
    fprintf(fp_eig,"Nspin       %d\n",1);  /* switch to 1 */
    fprintf(fp_eig,"Erange      %lf %lf\n",Dos_Erange[0],Dos_Erange[1]);
    fprintf(fp_eig,"atomnum     %d\n",atomnum);

    fprintf(fp_eig,"<WhatSpecies\n");
    for (i=1;i<=atomnum;i++) {
      fprintf(fp_eig,"%d ",WhatSpecies[i]);
    }
    fprintf(fp_eig,"\nWhatSpecies>\n");

    fprintf(fp_eig,"SpeciesNum     %d\n",SpeciesNum);
    fprintf(fp_eig,"<Spe_Total_CNO\n");
    for (i=0;i<SpeciesNum;i++) {
      fprintf(fp_eig,"%d ",Spe_Total_CNO[i]);
    }
    fprintf(fp_eig,"\nSpe_Total_CNO>\n");

    MaxL=Supported_MaxL;
    fprintf(fp_eig,"MaxL           %d\n",Supported_MaxL);
    fprintf(fp_eig,"<Spe_Num_CBasis\n");
    for (i=0;i<SpeciesNum;i++) {
      for (l=0;l<=MaxL;l++) {
	fprintf(fp_eig,"%d ",Spe_Num_CBasis[i][l]);
      }
      fprintf(fp_eig,"\n");
    }
    fprintf(fp_eig,"Spe_Num_CBasis>\n");
    fprintf(fp_eig,"ChemP       %lf\n",ChemP);

    fprintf(fp_eig,"<SpinAngle\n");
    for (i=1; i<=atomnum; i++) {
      fprintf(fp_eig,"%lf %lf\n",Angle0_Spin[i],Angle1_Spin[i]);
    }
    fprintf(fp_eig,"SpinAngle>\n");
  }

  /* close file pointers */

  if (myid==Host_ID){
    if (fp_eig) fclose(fp_eig);
  }

  if (fp_ev)  fclose(fp_ev);

}

void Save_DOS_Col(double ****** Residues, double **** OLP0, double *** EVal, int ** LO_TC, int ** HO_TC)
{
    int    spin, Mc_AN, wanA, Gc_AN, tno1;
    int    i1, i, j, MaxL, l, h_AN, Gh_AN, wanB, tno2;
    double Stime_atom, Etime_atom;
    double sum;
    char   file_eig[YOUSO10], file_ev[YOUSO10];
    FILE * fp_eig = NULL, *fp_ev = NULL;
    int    numprocs, myid;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    if (myid == Host_ID) {
        printf("The DOS is supported for a range from -8 to 8 eV for the O(N) DC-LNO method.\n");
        if (snprintf(file_eig, sizeof(file_eig), "%s%s.Dos.val", filepath, filename) >= (int)sizeof(file_eig)) {
            DCLNO_AbortWithMessage("Output path is too long for *.Dos.val in Save_DOS_Col.");
        }
        fp_eig = fopen(file_eig, "w");
        if (fp_eig == NULL) {
            printf("cannot open a file %s\n", file_eig);
        }
    }

    if (snprintf(file_ev, sizeof(file_ev), "%s%s.Dos.vec%i", filepath, filename, myid) >= (int)sizeof(file_ev)) {
        DCLNO_AbortWithMessage("Output path is too long for *.Dos.vec in Save_DOS_Col.");
    }
    fp_ev = fopen(file_ev, "w");
    if (fp_ev == NULL) {
        printf("cannot open a file %s\n", file_ev);
        if (fp_eig != NULL) fclose(fp_eig);
        return;
    }

    /****************************************************
                     save *.Dos.vec
    ****************************************************/

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            wanA  = WhatSpecies[Gc_AN];
            tno1  = Spe_Total_CNO[wanA];

            fprintf(fp_ev, "<AN%dAN%d\n", Gc_AN, spin);
            fprintf(fp_ev, "%d\n", (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1));

            for (i1 = 0; i1 < (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1); i1++) {

                fprintf(fp_ev, "%4d  %10.6f  ",
                        i1, EVal[spin][Mc_AN][i1 + LO_TC[spin][Mc_AN]]);

                for (i = 0; i < tno1; i++) {

                    sum = 0.0;
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB  = WhatSpecies[Gh_AN];
                        tno2  = Spe_Total_CNO[wanB];
                        for (j = 0; j < tno2; j++) {
                            sum += Residues[spin][Mc_AN - 1][h_AN][i][j][i1 + 2] *
                                   OLP0[Mc_AN][h_AN][i][j];
                        }
                    }

                    fprintf(fp_ev, "%8.5f", sum);
                }
                fprintf(fp_ev, "\n");
            }

            fprintf(fp_ev, "AN%dAN%d>\n", Gc_AN, spin);

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
        }
    }

    /****************************************************
                     save *.Dos.val
    ****************************************************/

    if (myid == Host_ID && fp_eig != NULL) {

        fprintf(fp_eig, "mode        5\n");
        fprintf(fp_eig, "NonCol      0\n");
        fprintf(fp_eig, "Nspin       %d\n", SpinP_switch);
        fprintf(fp_eig, "Erange      %lf %lf\n", Dos_Erange[0], Dos_Erange[1]);
        fprintf(fp_eig, "atomnum     %d\n", atomnum);

        fprintf(fp_eig, "<WhatSpecies\n");
        for (i = 1; i <= atomnum; i++) {
            fprintf(fp_eig, "%d ", WhatSpecies[i]);
        }
        fprintf(fp_eig, "\nWhatSpecies>\n");

        fprintf(fp_eig, "SpeciesNum     %d\n", SpeciesNum);
        fprintf(fp_eig, "<Spe_Total_CNO\n");
        for (i = 0; i < SpeciesNum; i++) {
            fprintf(fp_eig, "%d ", Spe_Total_CNO[i]);
        }
        fprintf(fp_eig, "\nSpe_Total_CNO>\n");

        MaxL = Supported_MaxL;
        fprintf(fp_eig, "MaxL           %d\n", Supported_MaxL);
        fprintf(fp_eig, "<Spe_Num_CBasis\n");
        for (i = 0; i < SpeciesNum; i++) {
            for (l = 0; l <= MaxL; l++) {
                fprintf(fp_eig, "%d ", Spe_Num_CBasis[i][l]);
            }
            fprintf(fp_eig, "\n");
        }
        fprintf(fp_eig, "Spe_Num_CBasis>\n");
        fprintf(fp_eig, "ChemP       %lf\n", ChemP);

        fclose(fp_eig);
    }

    fclose(fp_ev);
}
