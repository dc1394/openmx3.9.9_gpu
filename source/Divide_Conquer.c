/**********************************************************************
  Divide_Conquer.c:

     Divide_Conquer.c is a subroutine to perform a divide and conquer
     method for eigenvalue problem

  Log of Divide_Conquer.c:

     10/Dec/2003  Released by T.Ozaki

***********************************************************************/

#include "openmx_common.h"
#include "set_cuda_default_device_from_local_rank.h"
#include "set_openacc_device_from_local_rank.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <openacc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define measure_time 0
// #define _BENCHMARK   1

static double DC_Col(char * mode, int SCF_iter, double ***** Hks, double **** OLP0, double ***** CDM, double ***** EDM,
                     double Eele0[2], double Eele1[2]);

static double DC_NonCol(char * mode, int SCF_iter, double ***** Hks, double ***** ImNL, double **** OLP0,
                        double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2]);

static void Save_DOS_Col(double ****** Residues, double **** OLP0, double *** EVal, int * Msize);
static void Save_DOS_NonCol(dcomplex ****** Residues, double **** OLP0, double ** EVal, int * Msize);

static int DC_GPU_Threshold(void)
{
    static int initialized = 0;
    static int threshold   = GPU_CPU_SWITCH_NUM;

    if (!initialized) {
        char *env = getenv("OPENMX_DC_GPU_THRESHOLD");
        if (env != NULL && env[0] != '\0') {
            int env_threshold = atoi(env);
            if (0 < env_threshold) {
                threshold = env_threshold;
            }
        }
        initialized = 1;
    }

    return threshold;
}

static inline double DC_FermiWeight(double eval, double chemP, double beta, double max_x)
{
    double x = (eval - chemP) * beta;
    if (x <= -max_x)
        x = -max_x;
    if (max_x <= x)
        x = max_x;
    return 1.0 / (1.0 + exp(x));
}

static inline void DC_DotDensityResidueReal(int n, const double * restrict residue,
                                            const double * restrict weight,
                                            const double * restrict eweight,
                                            double * restrict cdm,
                                            double * restrict edm)
{
    int    i;
    double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
    double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;

    for (i = 0; i + 3 < n; i += 4) {
        const double r0 = residue[i];
        const double r1 = residue[i + 1];
        const double r2 = residue[i + 2];
        const double r3 = residue[i + 3];

        c0 += weight[i] * r0;
        e0 += eweight[i] * r0;
        c1 += weight[i + 1] * r1;
        e1 += eweight[i + 1] * r1;
        c2 += weight[i + 2] * r2;
        e2 += eweight[i + 2] * r2;
        c3 += weight[i + 3] * r3;
        e3 += eweight[i + 3] * r3;
    }

    for (; i < n; i++) {
        const double r = residue[i];
        c0 += weight[i] * r;
        e0 += eweight[i] * r;
    }

    *cdm = (c0 + c1) + (c2 + c3);
    *edm = (e0 + e1) + (e2 + e3);
}

static inline void DC_DotDensityResidueNonCol(int n, const dcomplex * restrict res0,
                                              const dcomplex * restrict res1,
                                              const dcomplex * restrict res2,
                                              const double * restrict weight,
                                              const double * restrict eweight, int use_iDM,
                                              double * restrict cdm0, double * restrict edm0,
                                              double * restrict cdm1, double * restrict edm1,
                                              double * restrict cdm2, double * restrict edm2,
                                              double * restrict cdm3, double * restrict edm3,
                                              double * restrict idm00, double * restrict idm01)
{
    int    i;
    double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
    double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;
    double im00 = 0.0, im01 = 0.0;

    if (use_iDM) {
        for (i = 0; i < n; i++) {
            const double f  = weight[i];
            const double ef = eweight[i];
            const double r0 = res0[i].r;
            const double r1 = res1[i].r;
            const double r2 = res2[i].r;
            const double r3 = res2[i].i;

            c0 += f * r0;
            e0 += ef * r0;
            c1 += f * r1;
            e1 += ef * r1;
            c2 += f * r2;
            e2 += ef * r2;
            c3 += f * r3;
            e3 += ef * r3;
            im00 += f * res0[i].i;
            im01 += f * res1[i].i;
        }
    } else {
        for (i = 0; i < n; i++) {
            const double f  = weight[i];
            const double ef = eweight[i];
            const double r0 = res0[i].r;
            const double r1 = res1[i].r;
            const double r2 = res2[i].r;
            const double r3 = res2[i].i;

            c0 += f * r0;
            e0 += ef * r0;
            c1 += f * r1;
            e1 += ef * r1;
            c2 += f * r2;
            e2 += ef * r2;
            c3 += f * r3;
            e3 += ef * r3;
        }
    }

    *cdm0 = c0;
    *edm0 = e0;
    *cdm1 = c1;
    *edm1 = e1;
    *cdm2 = c2;
    *edm2 = e2;
    *cdm3 = c3;
    *edm3 = e3;
    *idm00 = im00;
    *idm01 = im01;
}

static void DC_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1, 1);
}

static size_t DC_CheckedMulCount(size_t a, size_t b, const char *label)
{
    if (a != 0 && b > SIZE_MAX / a) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Dimension overflow in Divide_Conquer.c: %s", label);
        DC_AbortWithMessage(msg);
    }

    return a * b;
}

static size_t DC_CheckedArrayBytes(size_t count, size_t elem_size, const char *label)
{
    if (count != 0 && elem_size > SIZE_MAX / count) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Allocation size overflow in Divide_Conquer.c: %s", label);
        DC_AbortWithMessage(msg);
    }

    return count * elem_size;
}

static void *DC_MallocArray(size_t count, size_t elem_size, const char *label)
{
    size_t bytes = DC_CheckedArrayBytes(count, elem_size, label);
    void * ptr   = malloc(bytes == 0 ? 1 : bytes);

    if (ptr == NULL) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Out of memory in Divide_Conquer.c: %s (%zu bytes)", label, bytes);
        DC_AbortWithMessage(msg);
    }

    return ptr;
}

typedef struct {
    int                initialized;
    int                device_id;
    int                matrix_dim;
    int                loaded_s_dim;
    size_t             d_work_bytes;
    size_t             h_work_bytes;
    cudaStream_t       stream;
    cublasHandle_t     cublas;
    cusolverDnHandle_t cusolver;
    double *           d_S;
    double *           d_H;
    double *           d_tmp;
    double *           d_A;
    double *           d_C;
    double *           d_W;
    int32_t *          d_info;
    void *             d_work;
    void *             h_work;
    double *           h_matrix;
} DCCuSolverCtx;

static DCCuSolverCtx DC_cusolver_ctx = {0};

static void DC_CuSolver_Destroy(void)
{
    DCCuSolverCtx *ctx = &DC_cusolver_ctx;

    if (ctx->d_S != NULL)
        wait_cudafunc(cudaFree(ctx->d_S));
    if (ctx->d_H != NULL)
        wait_cudafunc(cudaFree(ctx->d_H));
    if (ctx->d_tmp != NULL)
        wait_cudafunc(cudaFree(ctx->d_tmp));
    if (ctx->d_A != NULL)
        wait_cudafunc(cudaFree(ctx->d_A));
    if (ctx->d_C != NULL)
        wait_cudafunc(cudaFree(ctx->d_C));
    if (ctx->d_W != NULL)
        wait_cudafunc(cudaFree(ctx->d_W));
    if (ctx->d_info != NULL)
        wait_cudafunc(cudaFree(ctx->d_info));
    if (ctx->d_work != NULL)
        wait_cudafunc(cudaFree(ctx->d_work));
    if (ctx->h_work != NULL)
        free(ctx->h_work);
    if (ctx->h_matrix != NULL)
        wait_cudafunc(cudaFreeHost(ctx->h_matrix));
    if (ctx->cusolver != NULL)
        wait_cudafunc(cusolverDnDestroy(ctx->cusolver));
    if (ctx->cublas != NULL)
        wait_cudafunc(cublasDestroy(ctx->cublas));
    if (ctx->stream != NULL)
        wait_cudafunc(cudaStreamDestroy(ctx->stream));

    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id    = -1;
    ctx->loaded_s_dim = 0;
}

static void DC_CuSolver_Init(void)
{
    DCCuSolverCtx *ctx = &DC_cusolver_ctx;
    int            current_device;

    wait_cudafunc(cudaGetDevice(&current_device));

    if (ctx->initialized && ctx->device_id == current_device) {
        return;
    }

    if (ctx->initialized) {
        DC_CuSolver_Destroy();
    }

    wait_cudafunc(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));
    wait_cudafunc(cublasCreate(&ctx->cublas));
    wait_cudafunc(cusolverDnCreate(&ctx->cusolver));
    wait_cudafunc(cublasSetStream(ctx->cublas, ctx->stream));
    wait_cudafunc(cusolverDnSetStream(ctx->cusolver, ctx->stream));

    ctx->initialized  = 1;
    ctx->device_id    = current_device;
    ctx->loaded_s_dim = 0;
}

static void DC_CuSolver_EnsureMatrixCapacity(int num)
{
    DCCuSolverCtx *ctx = &DC_cusolver_ctx;
    size_t         matrix_bytes;

    if (num <= 0) {
        DC_AbortWithMessage("Invalid matrix size in DC_CuSolver_EnsureMatrixCapacity.");
    }

    DC_CuSolver_Init();

    if (num <= ctx->matrix_dim) {
        return;
    }

    if (ctx->d_S != NULL)
        wait_cudafunc(cudaFree(ctx->d_S));
    if (ctx->d_H != NULL)
        wait_cudafunc(cudaFree(ctx->d_H));
    if (ctx->d_tmp != NULL)
        wait_cudafunc(cudaFree(ctx->d_tmp));
    if (ctx->d_A != NULL)
        wait_cudafunc(cudaFree(ctx->d_A));
    if (ctx->d_C != NULL)
        wait_cudafunc(cudaFree(ctx->d_C));
    if (ctx->d_W != NULL)
        wait_cudafunc(cudaFree(ctx->d_W));
    if (ctx->d_info != NULL)
        wait_cudafunc(cudaFree(ctx->d_info));
    if (ctx->h_matrix != NULL)
        wait_cudafunc(cudaFreeHost(ctx->h_matrix));

    matrix_bytes = DC_CheckedArrayBytes(DC_CheckedMulCount((size_t)num, (size_t)num, "CuSOLVER dense matrix"),
                                        sizeof(double), "CuSOLVER dense matrix");

    wait_cudafunc(cudaMalloc((void **)&ctx->d_S, matrix_bytes));
    wait_cudafunc(cudaMalloc((void **)&ctx->d_H, matrix_bytes));
    wait_cudafunc(cudaMalloc((void **)&ctx->d_tmp, matrix_bytes));
    wait_cudafunc(cudaMalloc((void **)&ctx->d_A, matrix_bytes));
    wait_cudafunc(cudaMalloc((void **)&ctx->d_C, matrix_bytes));
    wait_cudafunc(cudaMalloc((void **)&ctx->d_W,
                             DC_CheckedArrayBytes((size_t)num, sizeof(double), "CuSOLVER eigenvalue buffer")));
    wait_cudafunc(cudaMalloc((void **)&ctx->d_info, sizeof(int32_t)));
    wait_cudafunc(cudaMallocHost((void **)&ctx->h_matrix, matrix_bytes));

    ctx->matrix_dim   = num;
    ctx->loaded_s_dim = 0;
}

static void DC_CuSolver_EnsureWorkspace(int m, int maxn, double *d_A)
{
    DCCuSolverCtx *    ctx   = &DC_cusolver_ctx;
    cusolverEigMode_t  jobz  = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t   uplo  = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range = (m == maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;
    double             vl    = 0.0;
    double             vu    = 0.0;
    int64_t            h_meig;
    size_t             d_bytes = 0;
    size_t             h_bytes = 0;

    if (m <= 0 || maxn <= 0 || maxn > m) {
        DC_AbortWithMessage("Invalid eigensolver dimensions in DC_CuSolver_EnsureWorkspace.");
    }

    DC_CuSolver_EnsureMatrixCapacity(m);

    wait_cudafunc(cusolverDnXsyevdx_bufferSize(ctx->cusolver, NULL, jobz, range, uplo, m, CUDA_R_64F, d_A, m, &vl,
                                               &vu, 1L, maxn, &h_meig, CUDA_R_64F, ctx->d_W, CUDA_R_64F, &d_bytes,
                                               &h_bytes));

    if (d_bytes > ctx->d_work_bytes) {
        if (ctx->d_work != NULL)
            wait_cudafunc(cudaFree(ctx->d_work));
        ctx->d_work = NULL;
        if (0 < d_bytes) {
            wait_cudafunc(cudaMalloc((void **)&ctx->d_work, d_bytes));
        }
        ctx->d_work_bytes = d_bytes;
    }

    if (h_bytes == 0) {
        if (ctx->h_work != NULL)
            free(ctx->h_work);
        ctx->h_work       = NULL;
        ctx->h_work_bytes = 0;
    } else if (h_bytes > ctx->h_work_bytes) {
        if (ctx->h_work != NULL)
            free(ctx->h_work);
        ctx->h_work       = DC_MallocArray(h_bytes, 1, "CuSOLVER host workspace");
        ctx->h_work_bytes = h_bytes;
    }
}

static void DC_CuSolver_Eigen(double *d_A, int m, int maxn, double *W)
{
    DCCuSolverCtx *    ctx   = &DC_cusolver_ctx;
    cusolverEigMode_t  jobz  = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t   uplo  = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range = (m == maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;
    double             vl    = 0.0;
    double             vu    = 0.0;
    int64_t            h_meig = 0;
    int32_t            info   = 0;
    char               msg[256];

    DC_CuSolver_EnsureWorkspace(m, maxn, d_A);

    wait_cudafunc(cusolverDnXsyevdx(ctx->cusolver, NULL, jobz, range, uplo, m, CUDA_R_64F, d_A, m, &vl, &vu, 1L, maxn,
                                    &h_meig, CUDA_R_64F, ctx->d_W, CUDA_R_64F, ctx->d_work, ctx->d_work_bytes,
                                    ctx->h_work, ctx->h_work_bytes, ctx->d_info));

    wait_cudafunc(cudaMemcpyAsync(W, ctx->d_W, sizeof(double) * (size_t)maxn, cudaMemcpyDeviceToHost, ctx->stream));
    wait_cudafunc(cudaMemcpyAsync(&info, ctx->d_info, sizeof(int32_t), cudaMemcpyDeviceToHost, ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    if (info != 0) {
        snprintf(msg, sizeof(msg), "cusolverDnXsyevdx failed in Divide_Conquer.c: info=%d", (int)info);
        DC_AbortWithMessage(msg);
    }
    if (h_meig != (int64_t)maxn) {
        snprintf(msg, sizeof(msg), "cusolverDnXsyevdx returned %lld eigenpairs, expected %d in Divide_Conquer.c.",
                 (long long)h_meig, maxn);
        DC_AbortWithMessage(msg);
    }
}

static void DCCol_CuSolver_PackMatrix(int n, double **src, double *dst)
{
    int i, j;

    for (j = 1; j <= n; j++) {
        for (i = 1; i <= n; i++) {
            dst[(size_t)(j - 1) * (size_t)n + (size_t)(i - 1)] = src[i][j];
        }
    }
}

static void DCCol_CuSolver_UnpackMatrix(int n, const double *src, double **dst)
{
    int i, j;

    for (j = 1; j <= n; j++) {
        for (i = 1; i <= n; i++) {
            dst[i][j] = src[(size_t)(j - 1) * (size_t)n + (size_t)(i - 1)];
        }
    }
}

static void DCCol_CuSolver_LoadTransformedOverlap(int n, double **S_DC)
{
    DCCuSolverCtx *ctx = &DC_cusolver_ctx;
    size_t         matrix_bytes;

    DC_CuSolver_EnsureMatrixCapacity(n);

    matrix_bytes = DC_CheckedArrayBytes(DC_CheckedMulCount((size_t)n, (size_t)n, "DC transformed overlap"),
                                        sizeof(double), "DC transformed overlap");

    DCCol_CuSolver_PackMatrix(n, S_DC, ctx->h_matrix);
    wait_cudafunc(cudaMemcpyAsync(ctx->d_S, ctx->h_matrix, matrix_bytes, cudaMemcpyHostToDevice, ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    ctx->loaded_s_dim = n;
}

static void DCCol_CuSolver_DiagonalizeOverlap(int n, double **S_DC, double *ko)
{
    DCCuSolverCtx *ctx = &DC_cusolver_ctx;
    size_t         matrix_bytes;

    DC_CuSolver_EnsureMatrixCapacity(n);

    matrix_bytes = DC_CheckedArrayBytes(DC_CheckedMulCount((size_t)n, (size_t)n, "DC overlap matrix"),
                                        sizeof(double), "DC overlap matrix");

    DCCol_CuSolver_PackMatrix(n, S_DC, ctx->h_matrix);
    wait_cudafunc(cudaMemcpyAsync(ctx->d_S, ctx->h_matrix, matrix_bytes, cudaMemcpyHostToDevice, ctx->stream));

    DC_CuSolver_Eigen(ctx->d_S, n, n, ko + 1);

    wait_cudafunc(cudaMemcpyAsync(ctx->h_matrix, ctx->d_S, matrix_bytes, cudaMemcpyDeviceToHost, ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    DCCol_CuSolver_UnpackMatrix(n, ctx->h_matrix, S_DC);
    ctx->loaded_s_dim = 0;
}

static void DCCol_CuSolver_SolveHamiltonian(int n, int p_min, double **H_DC_spin, double *ko, double **C)
{
    DCCuSolverCtx *ctx = &DC_cusolver_ctx;
    double         alpha = 1.0;
    double         beta  = 0.0;
    int            num1  = n - (p_min - 1);
    int            i, j;
    size_t         matrix_bytes;

    if (ctx->loaded_s_dim != n) {
        DC_AbortWithMessage("Transformed overlap is not loaded in DCCol_CuSolver_SolveHamiltonian.");
    }
    if (num1 <= 0 || num1 > n) {
        DC_AbortWithMessage("Invalid active subspace size in DCCol_CuSolver_SolveHamiltonian.");
    }

    matrix_bytes = DC_CheckedArrayBytes(DC_CheckedMulCount((size_t)n, (size_t)n, "DC Hamiltonian matrix"),
                                        sizeof(double), "DC Hamiltonian matrix");

    DCCol_CuSolver_PackMatrix(n, H_DC_spin, ctx->h_matrix);
    wait_cudafunc(cudaMemcpyAsync(ctx->d_H, ctx->h_matrix, matrix_bytes, cudaMemcpyHostToDevice, ctx->stream));

    wait_cudafunc(openmx_gemmul8Dgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, ctx->d_H, n, ctx->d_S,
                                     n, &beta, ctx->d_tmp, n));
    wait_cudafunc(openmx_gemmul8Dgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, ctx->d_S, n,
                                     ctx->d_tmp, n, &beta, ctx->d_H, n));

    wait_cudafunc(cudaMemcpy2DAsync(ctx->d_A, sizeof(double) * (size_t)num1,
                                    ctx->d_H + (size_t)(p_min - 1) * (size_t)n + (size_t)(p_min - 1),
                                    sizeof(double) * (size_t)n, sizeof(double) * (size_t)num1, (size_t)num1,
                                    cudaMemcpyDeviceToDevice, ctx->stream));

    DC_CuSolver_Eigen(ctx->d_A, num1, num1, ko + 1);

    wait_cudafunc(openmx_gemmul8Dgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, num1, num1, &alpha,
                                     ctx->d_S + (size_t)(p_min - 1) * (size_t)n, n, ctx->d_A, num1, &beta,
                                     ctx->d_C, n));

    wait_cudafunc(cudaMemcpyAsync(ctx->h_matrix, ctx->d_C, sizeof(double) * (size_t)n * (size_t)num1,
                                  cudaMemcpyDeviceToHost, ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    for (j = 1; j <= num1; j++) {
        for (i = 1; i <= n; i++) {
            C[i][j] = ctx->h_matrix[(size_t)(j - 1) * (size_t)n + (size_t)(i - 1)];
        }
    }
}

double Divide_Conquer(char * mode, int SCF_iter, double ***** Hks, double ***** ImNL, double **** OLP0,
                      double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2])
{
    double time0;

    /****************************************************
           collinear without spin-orbit coupling
    ****************************************************/

    if ((SpinP_switch == 0 || SpinP_switch == 1) && SO_switch == 0) {
        time0 = DC_Col(mode, SCF_iter, Hks, OLP0, CDM, EDM, Eele0, Eele1);
    }

    /****************************************************
     non-collinear with and without spin-orbit coupling
    ****************************************************/

    else if (SpinP_switch == 3) {
        time0 = DC_NonCol(mode, SCF_iter, Hks, ImNL, OLP0, CDM, EDM, Eele0, Eele1);
    }

    return time0;
}

#pragma optimization_level 2
static double DC_Col(char * mode, int SCF_iter, double ***** Hks, double **** OLP0, double ***** CDM, double ***** EDM,
                     double Eele0[2], double Eele1[2])
{
    static int    firsttime = 1;
    int           Mc_AN, Gc_AN, i, Gi, wan, wanA, wanB, Anum;
    int           size1, size2, num, NUM, NUM1, n2, Cwan, Hwan;
    int           ih, ig, ian, j, kl, jg, jan, Bnum, m, n, spin;
    int           l, i1, j1, P_min, m_size;
    int           po, loopN, tno1, tno2, h_AN, Gh_AN;
    int           MA_AN, GA_AN, GB_AN, tnoA, tnoB, k;
    double        My_TZ, TZ, sum, FermiF, time0;
    double        tmp1, tmp2;
    double        My_Num_State, Num_State, x, Dnum;
    double        TStime, TEtime;
    double        My_Eele0[2], My_Eele1[2];
    double        max_x = 30.0;
    double        ChemP_MAX, ChemP_MIN, spin_degeneracy;
    double **     S_DC, ***H_DC, *ko, *M1;
    double **     C;
    double ***    EVal;
    double ****** Residues;
    double ***    PDOS_DC;
    int *         MP, *Msize;
    double *      tmp_array;
    double *      tmp_array2;
    int *         Snd_H_Size, *Rcv_H_Size;
    int *         Snd_S_Size, *Rcv_S_Size;
    int           numprocs, myid, ID, IDS, IDR, tag = 999;
    double        Stime_atom, Etime_atom;
    double        OLP_eigen_cut = Threshold_OLP_Eigen;
    double        stime, etime;
    double        time1, time2, time3, time4, time5, time6;

    double sum1, sum2, sum3, sum4;
    int    i1s, j1s;

    MPI_Status  stat;
    MPI_Request request;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    int is_scf_mode = (strcasecmp(mode, "scf") == 0);

    // Set the device to be used by CUDA and OpenACC
    if (scf_eigen_lib_flag == CuSOLVER) {
        // CUDA
        set_cuda_default_device_from_local_rank();

        // OpenACC
        set_openacc_nvidia_device_from_local_rank();
    }

    dtime(&TStime);

    if (measure_time) {
        time1 = 0.0;
        time2 = 0.0;
        time3 = 0.0;
        time4 = 0.0;
        time5 = 0.0;
        time6 = 0.0;
    }

    /****************************************************
      allocation of arrays:

      int MP[List_YOUSO[2]];
      int Msize[Matomnum+1];
      double EVal[SpinP_switch+1][Matomnum+1][n2];
    ****************************************************/

    Snd_H_Size = (int *)malloc(sizeof(int) * numprocs);
    Rcv_H_Size = (int *)malloc(sizeof(int) * numprocs);
    Snd_S_Size = (int *)malloc(sizeof(int) * numprocs);
    Rcv_S_Size = (int *)malloc(sizeof(int) * numprocs);

    m_size = 0;
    Msize  = (int *)malloc(sizeof(int) * (Matomnum + 1));

    EVal = (double ***)malloc(sizeof(double **) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        EVal[spin] = (double **)malloc(sizeof(double *) * (Matomnum + 1));

        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN        = 0;
                FNAN[0]      = 1;
                SNAN[0]      = 0;
                n2           = 1;
                Msize[Mc_AN] = 1;
            } else {

                Gc_AN = M2G[Mc_AN];
                Anum  = 1;
                for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                    Gi   = natn[Gc_AN][i];
                    wanA = WhatSpecies[Gi];
                    Anum += Spe_Total_CNO[wanA];
                }
                NUM          = Anum - 1;
                Msize[Mc_AN] = NUM;
                n2           = NUM + 3;
            }

            m_size += n2;
            EVal[spin][Mc_AN] = (double *)malloc(sizeof(double) * n2);
        }
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: EVal", sizeof(double) * m_size, NULL);

    if (2 <= level_stdout) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            printf("<DC> myid=%i Mc_AN=%2d Gc_AN=%2d Msize=%3d\n", myid, Mc_AN, M2G[Mc_AN], Msize[Mc_AN]);
        }
    }

    /****************************************************
      allocation of arrays:

      double Residues[SpinP_switch+1]
                     [Matomnum+1]
                     [FNAN[Gc_AN]+1]
                     [Spe_Total_CNO[Gc_AN]]
                     [Spe_Total_CNO[Gh_AN]]
                     [NUM2]
       To reduce the memory size, the size of NUM2 is
       needed to be found in the loop.
    ****************************************************/

    m_size   = 0;
    Residues = (double ******)malloc(sizeof(double *****) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        Residues[spin] = (double *****)malloc(sizeof(double ****) * (Matomnum + 1));
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN   = 0;
                FNAN[0] = 1;
                tno1    = 1;
                n2      = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];
                n2    = Msize[Mc_AN] + 2;
            }

            Residues[spin][Mc_AN] = (double ****)malloc(sizeof(double ***) * (FNAN[Gc_AN] + 1));

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                }

                Residues[spin][Mc_AN][h_AN] = (double ***)malloc(sizeof(double **) * tno1);
                for (i = 0; i < tno1; i++) {
                    Residues[spin][Mc_AN][h_AN][i] = (double **)malloc(sizeof(double *) * tno2);
                    for (j = 0; j < tno2; j++) {
                        Residues[spin][Mc_AN][h_AN][i][j] = (double *)malloc(sizeof(double) * n2);
                    }
                }

                m_size += tno1 * tno2 * n2;
            }
        }
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: Residues", sizeof(double) * m_size, NULL);

    /****************************************************
      allocation of arrays:

      double PDOS[SpinP_switch+1]
                 [Matomnum+1]
                 [NUM]
    ****************************************************/

    m_size  = 0;
    PDOS_DC = (double ***)malloc(sizeof(double **) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        PDOS_DC[spin] = (double **)malloc(sizeof(double *) * (Matomnum + 1));
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0)
                n2 = 1;
            else
                n2 = Msize[Mc_AN] + 2;

            m_size += n2;
            PDOS_DC[spin][Mc_AN] = (double *)malloc(sizeof(double) * n2);
        }
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: PDOS_DC", sizeof(double) * m_size, NULL);

    /****************************************************
     MPI

     Hks
    ****************************************************/

    /***********************************
               set data size
    ************************************/

    for (ID = 0; ID < numprocs; ID++) {

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (ID != 0) {
            tag = 999;

            /* find data size to send block data */
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
            } else {
                Snd_H_Size[IDS] = 0;
            }

            /* receiving of size of data */

            if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                Rcv_H_Size[IDR] = size2;
            } else {
                Rcv_H_Size[IDR] = 0;
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                MPI_Wait(&request, &stat);
        } else {
            Snd_H_Size[IDS] = 0;
            Rcv_H_Size[IDR] = 0;
        }
    }

    /***********************************
               data transfer
    ************************************/

    tag = 999;
    for (ID = 0; ID < numprocs; ID++) {

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (ID != 0) {

            /*****************************
                    sending of data
            *****************************/

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                size1 = Snd_H_Size[IDS];

                /* allocation of array */

                tmp_array = (double *)malloc(sizeof(double) * size1);

                /* multidimentional array to vector array */

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

                MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
            }

            /*****************************
               receiving of block data
            *****************************/

            if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                size2 = Rcv_H_Size[IDR];

                /* allocation of array */
                tmp_array2 = (double *)malloc(sizeof(double) * size2);

                MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
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

                /* freeing of array */
                free(tmp_array2);
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                MPI_Wait(&request, &stat);
                free(tmp_array); /* freeing of array */
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

    if (SCF_iter <= 2) {

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {
                tag = 999;

                /* find data size to send block data */
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
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    size1++;
                                }
                            }
                        }
                    }

                    Snd_S_Size[IDS] = size1;
                    MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                } else {
                    Snd_S_Size[IDS] = 0;
                }

                /* receiving of size of data */

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {
                    MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                    Rcv_S_Size[IDR] = size2;
                } else {
                    Rcv_S_Size[IDR] = 0;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                    MPI_Wait(&request, &stat);
            } else {
                Snd_S_Size[IDS] = 0;
                Rcv_S_Size[IDR] = 0;
            }
        }

        /***********************************
                   data transfer
        ************************************/

        tag = 999;
        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {

                /*****************************
                        sending of data
                *****************************/

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                    size1 = Snd_S_Size[IDS];

                    /* allocation of array */

                    tmp_array = (double *)malloc(sizeof(double) * size1);

                    /* multidimentional array to vector array */

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

                    MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
                }

                /*****************************
                   receiving of block data
                *****************************/

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                    size2 = Rcv_S_Size[IDR];

                    /* allocation of array */
                    tmp_array2 = (double *)malloc(sizeof(double) * size2);

                    MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                    num   = 0;
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
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

                    /* freeing of array */
                    free(tmp_array2);
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
                    free(tmp_array); /* freeing of array */
                }
            }
        }
    }

    /****************************************************
              find the total number of electrons
    ****************************************************/

    My_TZ = 0.0;
    for (i = 1; i <= Matomnum; i++) {
        Gc_AN = M2G[i];
        wan   = WhatSpecies[Gc_AN];
        My_TZ = My_TZ + Spe_Core_Charge[wan];
    }

    /* MPI, My_TZ */

    MPI_Allreduce(&My_TZ, &TZ, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    if (scf_eigen_lib_flag == CuSOLVER) {
        // CUDA
        set_cuda_default_device_from_local_rank();

        // OpenACC
        set_openacc_nvidia_device_from_local_rank();
    }

    /****************************************************
        Setting of Hamiltonian and overlap matrices

           MP indicates the starting position of
                atom i in arraies H and S
    ****************************************************/

    int mysize = Matomnum > 0 ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &mysize, 1, MPI_INT, MPI_SUM, mpi_comm_level1);

    // #pragma omp parallel shared(OLP_eigen_cut, List_YOUSO, Etime_atom, time_per_atom, time3, Residues, EVal, time2, time1, \
    //                                 S12, level_stdout, SpinP_switch, Hks, OLP0, SCF_iter, RMI1, S_G2M, Spe_Total_CNO,      \
    //                                 natn, FNAN, SNAN, WhatSpecies, M2G, Matomnum)                                          \
    //     private(OMPID, Nthrds, Nprocs, Mc_AN, Stime_atom, Gc_AN, wan, Anum, i, j, MP, Gi, wanA, NUM, NUM1, n2, spin, S_DC, \
    //                 H_DC, ko, M1, C, ig, ian, ih, kl, jg, jan, Bnum, m, n, stime, P_min, l, i1, j1, etime, tmp1, tmp2,     \
    //                 sum1, sum2, sum3, sum4, j1s, sum, tno1, h_AN, Gh_AN, wanB, tno2)
    {

        /* get info. on OpenMP */

        OMPID  = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        /* allocation of arrays */

        MP = (int *)malloc(sizeof(int) * List_YOUSO[2]);

        /* start of the Mc_AN loop which is parallelized by OpenMP */

        for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            wan   = WhatSpecies[Gc_AN];

            /***********************************************
             find the size of matrix for the atom Mc_AN
                       and set the MP vector

              Note:
               MP indicates the starting position of
                    atom i in arraies H and S
            ***********************************************/

            Anum = 1;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                MP[i] = Anum;
                Gi    = natn[Gc_AN][i];
                wanA  = WhatSpecies[Gi];
                Anum += Spe_Total_CNO[wanA];
            }
            NUM = Anum - 1;
            n2  = NUM + 3;

            /***********************************************
             allocation of arrays:

            double S_DC[n2][n2];
            double H_DC[SpinP_switch+1][n2][n2];
            double ko[n2];
            ***********************************************/

            double *S_DC_store =
                (double *)DC_MallocArray((size_t)n2 * (size_t)n2, sizeof(double), "DC overlap matrix");
            S_DC = (double **)DC_MallocArray((size_t)n2, sizeof(double *), "DC overlap row pointers");
            for (i = 0; i < n2; i++) {
                S_DC[i] = S_DC_store + (size_t)i * (size_t)n2;
            }

            double *H_DC_store =
                (double *)DC_MallocArray((size_t)(SpinP_switch + 1) * (size_t)n2 * (size_t)n2, sizeof(double),
                                         "DC Hamiltonian matrices");
            H_DC = (double ***)DC_MallocArray((size_t)(SpinP_switch + 1), sizeof(double **),
                                              "DC Hamiltonian spin pointers");
            for (spin = 0; spin <= SpinP_switch; spin++) {
                H_DC[spin] = (double **)DC_MallocArray((size_t)n2, sizeof(double *), "DC Hamiltonian row pointers");
                for (i = 0; i < n2; i++) {
                    H_DC[spin][i] = H_DC_store + ((size_t)spin * (size_t)n2 + (size_t)i) * (size_t)n2;
                }
            }

            ko = (double *)DC_MallocArray((size_t)n2, sizeof(double), "DC eigenvalues");
            M1 = (double *)DC_MallocArray((size_t)n2, sizeof(double), "DC overlap scales");

            double *C_store =
                (double *)DC_MallocArray((size_t)n2 * (size_t)n2, sizeof(double), "DC eigenvector matrix");
            C = (double **)DC_MallocArray((size_t)n2, sizeof(double *), "DC eigenvector row pointers");
            for (i = 0; i < n2; i++) {
                C[i] = C_store + (size_t)i * (size_t)n2;
            }

            int use_dc_gpu = (scf_eigen_lib_flag == CuSOLVER && DC_GPU_Threshold() <= NUM);
            if (SCF_iter <= 2) {
                memset(S_DC_store, 0, sizeof(double) * (size_t)n2 * (size_t)n2);
            }
            memset(H_DC_store, 0, sizeof(double) * (size_t)(SpinP_switch + 1) * (size_t)n2 * (size_t)n2);

            /***********************************************
             construct cluster full matrices of Hamiltonian
                   and overlap for the atom Mc_AN
            ***********************************************/

            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                ig   = natn[Gc_AN][i];
                ian  = Spe_Total_CNO[WhatSpecies[ig]];
                Anum = MP[i];
                ih   = S_G2M[ig];

                for (j = 0; j <= (FNAN[Gc_AN] + SNAN[Gc_AN]); j++) {

                    kl   = RMI1[Mc_AN][i][j];
                    jg   = natn[Gc_AN][j];
                    jan  = Spe_Total_CNO[WhatSpecies[jg]];
                    Bnum = MP[j];

                    if (0 <= kl) {

                        if (SCF_iter <= 2) {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    S_DC[Anum + m][Bnum + n] = OLP0[ih][kl][m][n];
                                }
                            }
                        }

                        for (spin = 0; spin <= SpinP_switch; spin++) {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    H_DC[spin][Anum + m][Bnum + n] = Hks[spin][ih][kl][m][n];
                                }
                            }
                        }
                    }
                }
            }

            /****************************************************
             Solve the generalized eigenvalue problem
             HC = SCE

             1) diagonalize S
             2) search negative eigenvalues of S
            ****************************************************/

            if (SCF_iter <= 2) {

                if (measure_time)
                    dtime(&stime);

                if (use_dc_gpu) {
                    DCCol_CuSolver_DiagonalizeOverlap(NUM, S_DC, ko);
                } else {
                    Eigen_lapack(S_DC, ko, NUM, NUM);
                }

                /***********************************************
                      Searching of negative eigenvalues
                ************************************************/

                P_min = 1;
                for (l = 1; l <= NUM; l++) {

                    if (ko[l] < OLP_eigen_cut) {
                        P_min = l + 1;
                        if (3 <= level_stdout) {
                            printf("<DC>  Negative EV of OLP %2d %15.12f\n", l, ko[l]);
                        }
                    }
                }

                S12[Mc_AN][0][0] = P_min;

                for (l = 1; l < P_min; l++)
                    M1[l] = 0.0;
                for (l = P_min; l <= NUM; l++)
                    M1[l] = 1.0 / sqrt(ko[l]);

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = 1; j1 <= NUM; j1++) {
                        S_DC[i1][j1]       = S_DC[i1][j1] * M1[j1];
                        S12[Mc_AN][i1][j1] = S_DC[i1][j1];
                    }
                }

                if (measure_time) {
                    dtime(&etime);
                    time1 += etime - stime;
                }

            }

            else {

                P_min = (int)S12[Mc_AN][0][0];

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = 1; j1 <= NUM; j1++) {
                        S_DC[i1][j1] = S12[Mc_AN][i1][j1];
                    }
                }
            }

            /***********************************************
              transform Hamiltonian matrix
            ************************************************/

            // compiler's bug
            Anum = 1;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                MP[i] = Anum;
                Gi    = natn[Gc_AN][i];
                wanA  = WhatSpecies[Gi];
                Anum += Spe_Total_CNO[wanA];
            }
            NUM = Anum - 1;

            if (use_dc_gpu) {
                DCCol_CuSolver_LoadTransformedOverlap(NUM, S_DC);
            }

            int use_dc_openacc = 0;

            if (use_dc_openacc) {
                size_t free_memory, total_memory;
                size_t needmemsize = 0;
                needmemsize += sizeof(double) * (SpinP_switch + 1) * (NUM + 1) * (NUM + 1);
                needmemsize += sizeof(double) * (NUM + 1) * (NUM + 1);
                needmemsize += sizeof(double) * (NUM + 1);
                needmemsize += sizeof(double) * (NUM + 1) * (NUM + 1);

                while (true) {
                    cudaMemGetInfo(&free_memory, &total_memory);
                    if (free_memory > needmemsize) {
                        break;
                    }

                    double wait_time    = (double)rand() / (double)RAND_MAX * WAITTIME;
                    double start_time   = MPI_Wtime();
                    double current_time = start_time;
                    while ((current_time - start_time) < wait_time) {
                        current_time = MPI_Wtime();
                    }
                }
            }

#pragma acc data if(use_dc_openacc) copyin(H_DC[0 : SpinP_switch + 1][0 : NUM + 1][0 : NUM + 1], S_DC[0 : NUM + 1][0 : NUM + 1])
#pragma acc data if(use_dc_openacc) create(ko[0 : NUM + 1], C[0 : NUM + 1][0 : NUM + 1])
            for (spin = 0; spin <= SpinP_switch; spin++) {

                if (measure_time)
                    dtime(&stime);

                if (use_dc_gpu) {
                    NUM1 = NUM - (P_min - 1);
                    DCCol_CuSolver_SolveHamiltonian(NUM, P_min, H_DC[spin], ko, C);

                } else if (use_dc_openacc) {
                    // OpenACC

                    /* transpose S */
#pragma acc kernels
#pragma acc loop independent
                    for (i1 = 1; i1 <= NUM; i1++) {
#pragma acc loop independent
                        for (j1 = i1 + 1; j1 <= NUM; j1++) {
                            tmp1         = S_DC[i1][j1];
                            tmp2         = S_DC[j1][i1];
                            S_DC[i1][j1] = tmp2;
                            S_DC[j1][i1] = tmp1;
                        }
                    }

                    /* H * U * M1 */

                    /***********************************************
                             find the size of matrix for the atom Mc_AN
                                    and set the MP vector

                            Note:
                            MP indicates the starting position of
                                    atom i in arraies H and S
                            ***********************************************/

#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (j1 = 1; j1 <= NUM - 3; j1 = j1 + 4) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            double sum1 = 0.0;
                            double sum2 = 0.0;
                            double sum3 = 0.0;
                            double sum4 = 0.0;
#pragma acc loop independent reduction(+ : sum1) reduction(+ : sum2) reduction(+ : sum3) reduction(+ : sum4)
                            for (l = 1; l <= NUM; l++) {
                                sum1 += H_DC[spin][i1][l] * S_DC[j1][l];
                                sum2 += H_DC[spin][i1][l] * S_DC[j1 + 1][l];
                                sum3 += H_DC[spin][i1][l] * S_DC[j1 + 2][l];
                                sum4 += H_DC[spin][i1][l] * S_DC[j1 + 3][l];
                            }
                            C[j1][i1]     = sum1;
                            C[j1 + 1][i1] = sum2;
                            C[j1 + 2][i1] = sum3;
                            C[j1 + 3][i1] = sum4;
                        }
                    }

                    j1s = NUM - NUM % 4 + 1;
#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (j1 = j1s; j1 <= NUM; j1++) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum = 0.0;
#pragma acc loop independent reduction(+ : sum)
                            for (l = 1; l <= NUM; l++) {
                                sum += H_DC[spin][i1][l] * S_DC[j1][l];
                            }
                            C[j1][i1] = sum;
                        }
                    }

                    /* M1 * U^+ H * U * M1 */

#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (j1 = 1; j1 <= NUM - 3; j1 = j1 + 4) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            double sum1 = 0.0;
                            double sum2 = 0.0;
                            double sum3 = 0.0;
                            double sum4 = 0.0;
#pragma acc loop independent reduction(+ : sum1) reduction(+ : sum2) reduction(+ : sum3) reduction(+ : sum4)
                            for (l = 1; l <= NUM; l++) {
                                sum1 += S_DC[i1][l] * C[j1][l];
                                sum2 += S_DC[i1][l] * C[j1 + 1][l];
                                sum3 += S_DC[i1][l] * C[j1 + 2][l];
                                sum4 += S_DC[i1][l] * C[j1 + 3][l];
                            }
                            H_DC[spin][j1][i1]     = sum1;
                            H_DC[spin][j1 + 1][i1] = sum2;
                            H_DC[spin][j1 + 2][i1] = sum3;
                            H_DC[spin][j1 + 3][i1] = sum4;
                        }
                    }

                    j1s = NUM - NUM % 4 + 1;
#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (j1 = j1s; j1 <= NUM; j1++) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum1 = 0.0;
#pragma acc loop independent reduction(+ : sum1)
                            for (l = 1; l <= NUM; l++) {
                                sum1 += S_DC[i1][l] * C[j1][l];
                            }
                            H_DC[spin][j1][i1] = sum1;
                        }
                    }

                    /* H_DC to C (transposition) */

#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (i1 = P_min; i1 <= NUM; i1++) {
                        for (j1 = P_min; j1 <= NUM; j1++) {
                            C[j1 - (P_min - 1)][i1 - (P_min - 1)] = H_DC[spin][i1][j1];
                        }
                    }

                    /***********************************************
                             diagonalize the trasformed Hamiltonian matrix
                            ************************************************/

                    NUM1 = NUM - (P_min - 1);

                    Eigen_lapack_openacc(C, ko, NUM1, NUM1);

                    /* C to H (transposition) */

#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (i1 = 1; i1 <= NUM; i1++) {
                        for (j1 = 1; j1 <= NUM1; j1++) {
                            H_DC[spin][j1][i1] = C[i1][j1];
                        }
                    }

                    /***********************************************
                             transformation to the original eigen vectors.
                                            NOTE 244P
                            ***********************************************/

                    /* transpose */

#pragma acc kernels
#pragma acc loop independent
                    for (i1 = 1; i1 <= NUM; i1++) {
#pragma acc loop independent
                        for (j1 = i1 + 1; j1 <= NUM; j1++) {
                            tmp1         = S_DC[i1][j1];
                            tmp2         = S_DC[j1][i1];
                            S_DC[i1][j1] = tmp2;
                            S_DC[j1][i1] = tmp1;
                        }
                    }

#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (j1 = 1; j1 <= NUM1; j1++) {
                        for (l = NUM; P_min <= l; l--) {
                            H_DC[spin][j1][l] = H_DC[spin][j1][l - (P_min - 1)];
                        }
                    }

#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (j1 = 1; j1 <= NUM - 3; j1 = j1 + 4) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            double sum1 = 0.0;
                            double sum2 = 0.0;
                            double sum3 = 0.0;
                            double sum4 = 0.0;
#pragma acc loop independent reduction(+ : sum1) reduction(+ : sum2) reduction(+ : sum3) reduction(+ : sum4)
                            for (l = P_min; l <= NUM; l++) {
                                sum1 += S_DC[i1][l] * H_DC[spin][j1][l];
                                sum2 += S_DC[i1][l] * H_DC[spin][j1 + 1][l];
                                sum3 += S_DC[i1][l] * H_DC[spin][j1 + 2][l];
                                sum4 += S_DC[i1][l] * H_DC[spin][j1 + 3][l];
                            }
                            C[i1][j1]     = sum1;
                            C[i1][j1 + 1] = sum2;
                            C[i1][j1 + 2] = sum3;
                            C[i1][j1 + 3] = sum4;
                        }
                    }

                    j1s = NUM - NUM % 4 + 1;
#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (j1 = j1s; j1 <= NUM; j1++) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum = 0.0;
#pragma acc loop independent reduction(+ : sum)
                            for (l = P_min; l <= NUM; l++) {
                                sum += S_DC[i1][l] * H_DC[spin][j1][l];
                            }
                            C[i1][j1] = sum;
                        }
                    }

                    // compiler's bug
                    Anum = 1;
                    for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                        MP[i] = Anum;
                        Gi    = natn[Gc_AN][i];
                        wanA  = WhatSpecies[Gi];
                        Anum += Spe_Total_CNO[wanA];
                    }
                    NUM = Anum - 1;

#pragma acc update self(ko[0 : NUM + 1], C[0 : NUM + 1][0 : NUM + 1])
                } else {
                    /* transpose S */
                    for (i1 = 1; i1 <= NUM; i1++) {
                        for (j1 = i1 + 1; j1 <= NUM; j1++) {
                            tmp1         = S_DC[i1][j1];
                            tmp2         = S_DC[j1][i1];
                            S_DC[i1][j1] = tmp2;
                            S_DC[j1][i1] = tmp1;
                        }
                    }

                    /* H * U * M1 */

                    for (j1 = 1; j1 <= NUM - 3; j1 = j1 + 4) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum1 = 0.0;
                            sum2 = 0.0;
                            sum3 = 0.0;
                            sum4 = 0.0;
                            for (l = 1; l <= NUM; l++) {
                                sum1 += H_DC[spin][i1][l] * S_DC[j1][l];
                                sum2 += H_DC[spin][i1][l] * S_DC[j1 + 1][l];
                                sum3 += H_DC[spin][i1][l] * S_DC[j1 + 2][l];
                                sum4 += H_DC[spin][i1][l] * S_DC[j1 + 3][l];
                            }
                            C[j1][i1]     = sum1;
                            C[j1 + 1][i1] = sum2;
                            C[j1 + 2][i1] = sum3;
                            C[j1 + 3][i1] = sum4;
                        }
                    }

                    j1s = NUM - NUM % 4 + 1;
                    for (j1 = j1s; j1 <= NUM; j1++) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum = 0.0;
                            for (l = 1; l <= NUM; l++) {
                                sum += H_DC[spin][i1][l] * S_DC[j1][l];
                            }
                            C[j1][i1] = sum;
                        }
                    }

                    /* M1 * U^+ H * U * M1 */

                    for (j1 = 1; j1 <= NUM - 3; j1 = j1 + 4) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum1 = 0.0;
                            sum2 = 0.0;
                            sum3 = 0.0;
                            sum4 = 0.0;
                            for (l = 1; l <= NUM; l++) {
                                sum1 += S_DC[i1][l] * C[j1][l];
                                sum2 += S_DC[i1][l] * C[j1 + 1][l];
                                sum3 += S_DC[i1][l] * C[j1 + 2][l];
                                sum4 += S_DC[i1][l] * C[j1 + 3][l];
                            }
                            H_DC[spin][j1][i1]     = sum1;
                            H_DC[spin][j1 + 1][i1] = sum2;
                            H_DC[spin][j1 + 2][i1] = sum3;
                            H_DC[spin][j1 + 3][i1] = sum4;
                        }
                    }
                    j1s = NUM - NUM % 4 + 1;
                    for (j1 = j1s; j1 <= NUM; j1++) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum1 = 0.0;
                            for (l = 1; l <= NUM; l++) {
                                sum1 += S_DC[i1][l] * C[j1][l];
                            }
                            H_DC[spin][j1][i1] = sum1;
                        }
                    }

                    /* H_DC to C (transposition) */

                    for (i1 = P_min; i1 <= NUM; i1++) {
                        for (j1 = P_min; j1 <= NUM; j1++) {
                            C[j1 - (P_min - 1)][i1 - (P_min - 1)] = H_DC[spin][i1][j1];
                        }
                    }

                    /***********************************************
                     diagonalize the trasformed Hamiltonian matrix
                    ************************************************/

                    NUM1 = NUM - (P_min - 1);
                    Eigen_lapack(C, ko, NUM1, NUM1);

                    /* C to H (transposition) */

                    for (i1 = 1; i1 <= NUM; i1++) {
                        for (j1 = 1; j1 <= NUM1; j1++) {
                            H_DC[spin][j1][i1] = C[i1][j1];
                        }
                    }

                    /***********************************************
                     transformation to the original eigen vectors.
                                    NOTE 244P
                    ***********************************************/

                    /* transpose */

                    for (i1 = 1; i1 <= NUM; i1++) {
                        for (j1 = i1 + 1; j1 <= NUM; j1++) {
                            tmp1         = S_DC[i1][j1];
                            tmp2         = S_DC[j1][i1];
                            S_DC[i1][j1] = tmp2;
                            S_DC[j1][i1] = tmp1;
                        }
                    }

                    for (j1 = 1; j1 <= NUM1; j1++) {
                        for (l = NUM; P_min <= l; l--) {
                            H_DC[spin][j1][l] = H_DC[spin][j1][l - (P_min - 1)];
                        }
                    }

                    for (j1 = 1; j1 <= NUM - 3; j1 = j1 + 4) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum1 = 0.0;
                            sum2 = 0.0;
                            sum3 = 0.0;
                            sum4 = 0.0;
                            for (l = P_min; l <= NUM; l++) {
                                sum1 += S_DC[i1][l] * H_DC[spin][j1][l];
                                sum2 += S_DC[i1][l] * H_DC[spin][j1 + 1][l];
                                sum3 += S_DC[i1][l] * H_DC[spin][j1 + 2][l];
                                sum4 += S_DC[i1][l] * H_DC[spin][j1 + 3][l];
                            }
                            C[i1][j1]     = sum1;
                            C[i1][j1 + 1] = sum2;
                            C[i1][j1 + 2] = sum3;
                            C[i1][j1 + 3] = sum4;
                        }
                    }
                    j1s = NUM - NUM % 4 + 1;
                    for (j1 = j1s; j1 <= NUM; j1++) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum = 0.0;
                            for (l = P_min; l <= NUM; l++) {
                                sum += S_DC[i1][l] * H_DC[spin][j1][l];
                            }
                            C[i1][j1] = sum;
                        }
                    }
                }

                if (measure_time) {
                    dtime(&etime);
                    time2 += etime - stime;
                }

                /*
                  for (i=1; i<=NUM1; i++){
                  printf("ko %15.12f 1.0\n",ko[i]);
                  }

                  for (i=1; i<=NUM; i++){
                  for (j=1; j<=NUM; j++){
                  printf("%15.12f ",C[i][j]);
                  }
                  printf("\n");
                  }
                */

                /*
                  MPI_Finalize();
                  exit(0);
                */

                /***********************************************
                   store eigenvalues and residues of poles
                ***********************************************/

                if (measure_time)
                    dtime(&stime);

                for (i1 = 1; i1 <= NUM; i1++) {
                    EVal[spin][Mc_AN][i1 - 1] = 1000.0;
                }
                for (i1 = 1; i1 <= NUM1; i1++) {
                    EVal[spin][Mc_AN][i1 - 1] = ko[i1];
                }

                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];

                if (is_scf_mode) {
                    for (i1 = 0; i1 < Msize[Mc_AN]; i1++) {
                        PDOS_DC[spin][Mc_AN][i1] = 0.0;
                    }
                }

                for (i = 0; i < tno1; i++) {
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB  = WhatSpecies[Gh_AN];
                        tno2  = Spe_Total_CNO[wanB];
                        Bnum  = MP[h_AN];
                        for (j = 0; j < tno2; j++) {
                            tmp1 = OLP0[Mc_AN][h_AN][i][j];
                            for (i1 = 1; i1 <= NUM1; i1++) {
                                tmp2 = C[1 + i][i1] * C[Bnum + j][i1];
                                Residues[spin][Mc_AN][h_AN][i][j][i1 - 1] = tmp2;
                                if (is_scf_mode) {
                                    PDOS_DC[spin][Mc_AN][i1 - 1] += tmp2 * tmp1;
                                }
                            }
                        }
                    }
                }

                if (measure_time) {
                    dtime(&etime);
                    time3 += etime - stime;
                }

            } /* end of spin */

            /****************************************************
                              free arrays
            ****************************************************/

            free(S_DC_store);
            free(S_DC);

            for (spin = 0; spin <= SpinP_switch; spin++) {
                free(H_DC[spin]);
            }
            free(H_DC_store);
            free(H_DC);

            free(ko);
            free(M1);

            free(C_store);
            free(C);

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
        } /* end of Mc_AN */

        /* freeing of arrays */

        free(MP);
    } /* //#pragma omp parallel */

    if (is_scf_mode) {

        /****************************************************
                  calculate projected DOS
        ****************************************************/

        if (measure_time)
            dtime(&stime);

        /* PDOS_DC was accumulated while storing Residues, avoiding a second pass here. */

        if (measure_time) {
            dtime(&etime);
            time4 += etime - stime;
        }

        /****************************************************
                       find chemical potential
        ****************************************************/

        if (measure_time)
            dtime(&stime);

        po    = 0;
        loopN = 0;

        ChemP_MAX = 30.0;
        ChemP_MIN = -30.0;
        if (SpinP_switch == 0)
            spin_degeneracy = 2.0;
        else if (SpinP_switch == 1)
            spin_degeneracy = 1.0;

        do {
            ChemP = 0.50 * (ChemP_MAX + ChemP_MIN);

            My_Num_State = 0.0;
            for (spin = 0; spin <= SpinP_switch; spin++) {
                for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                    dtime(&Stime_atom);

                    Gc_AN = M2G[Mc_AN];

                    for (i = 0; i < Msize[Mc_AN]; i++) {
                        FermiF = DC_FermiWeight(EVal[spin][Mc_AN][i], ChemP, Beta, max_x);
                        My_Num_State += spin_degeneracy * FermiF * PDOS_DC[spin][Mc_AN][i];
                    }

                    dtime(&Etime_atom);
                    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
                }
            }

            /* MPI, My_Num_State */

            MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

            Dnum = (TZ - Num_State) - system_charge;
            if (0.0 <= Dnum)
                ChemP_MIN = ChemP;
            else
                ChemP_MAX = ChemP;
            if (fabs(Dnum) < 1.0e-13)
                po = 1;

            if (myid == Host_ID && 2 <= level_stdout) {
                printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n", ChemP, TZ, Num_State);
            }

            loopN++;
        } while (po == 0 && loopN < 1000);

        /****************************************************
            eigenenergy by summing up eigenvalues
        ****************************************************/

        double ***FermiW =
            (double ***)DC_MallocArray((size_t)(SpinP_switch + 1), sizeof(double **), "DC col Fermi weights");
        double ***EFermiW =
            (double ***)DC_MallocArray((size_t)(SpinP_switch + 1), sizeof(double **), "DC col energy Fermi weights");

        My_Eele0[0] = 0.0;
        My_Eele0[1] = 0.0;
        for (spin = 0; spin <= SpinP_switch; spin++) {
            FermiW[spin] =
                (double **)DC_MallocArray((size_t)(Matomnum + 1), sizeof(double *), "DC col Fermi weight atoms");
            EFermiW[spin] = (double **)DC_MallocArray((size_t)(Matomnum + 1), sizeof(double *),
                                                      "DC col energy Fermi weight atoms");
            FermiW[spin][0]  = NULL;
            EFermiW[spin][0] = NULL;

            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                dtime(&Stime_atom);

                Gc_AN = M2G[Mc_AN];
                {
                    const int     n       = Msize[Mc_AN];
                    const double *eval    = EVal[spin][Mc_AN];
                    double *      weights = (double *)DC_MallocArray((size_t)(2 * n), sizeof(double),
                                                                     "DC col Fermi weight buffer");
                    double *      fweight = weights;
                    double *      eweight = weights + n;

                    FermiW[spin][Mc_AN]  = fweight;
                    EFermiW[spin][Mc_AN] = eweight;

                    for (i = 0; i < n; i++) {
                        FermiF     = DC_FermiWeight(eval[i], ChemP, Beta, max_x);
                        fweight[i] = FermiF;
                        eweight[i] = FermiF * eval[i];
                        My_Eele0[spin] += eweight[i] * PDOS_DC[spin][Mc_AN][i];
                    }
                }

                dtime(&Etime_atom);
                time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
            }
        }

        /* MPI, My_Eele0 */
        for (spin = 0; spin <= SpinP_switch; spin++) {
            MPI_Allreduce(&My_Eele0[spin], &Eele0[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        }

        if (SpinP_switch == 0) {
            Eele0[1] = Eele0[0];
        }

        if (measure_time) {
            dtime(&etime);
            time5 += etime - stime;
        }

        /****************************************************
           calculate density and energy density matrices
        ****************************************************/

        if (measure_time)
            dtime(&stime);

        const int nspins_eele1 = SpinP_switch + 1;
        const int nthreads_eele1 = omp_get_max_threads();
        double *My_Eele1_threads = (double *)DC_MallocArray((size_t)(nthreads_eele1 * nspins_eele1),
                                                            sizeof(double), "DC col Eele1 threads");
        for (i = 0; i < nthreads_eele1 * nspins_eele1; i++) {
            My_Eele1_threads[i] = 0.0;
        }

        //#pragma omp parallel shared(FNAN, time_per_atom, EDM, CDM, Residues, natn, max_x, Beta, ChemP, EVal, Msize, Spe_Total_CNO, WhatSpecies, M2G, SpinP_switch, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, spin, Stime_atom, Gc_AN, wanA, tno1, i1, x, FermiF, h_AN, Gh_AN, wanB, tno2, i, j, tmp1, Etime_atom)
        {
            /* get info. on OpenMP */

            OMPID  = omp_get_thread_num();
            Nthrds = omp_get_num_threads();
            Nprocs = omp_get_num_procs();

            for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {
                for (spin = 0; spin <= SpinP_switch; spin++) {

                    dtime(&Stime_atom);

                    Gc_AN = M2G[Mc_AN];
                    wanA  = WhatSpecies[Gc_AN];
                    tno1  = Spe_Total_CNO[wanA];

                    const int     n       = Msize[Mc_AN];
                    const double *fweight = FermiW[spin][Mc_AN];
                    const double *eweight = EFermiW[spin][Mc_AN];

                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB  = WhatSpecies[Gh_AN];
                        tno2  = Spe_Total_CNO[wanB];
                        for (i = 0; i < tno1; i++) {
                            for (j = 0; j < tno2; j++) {
                                double cdm_sum = 0.0;
                                double edm_sum = 0.0;
                                double *res    = Residues[spin][Mc_AN][h_AN][i][j];

                                DC_DotDensityResidueReal(n, res, fweight, eweight, &cdm_sum, &edm_sum);
                                CDM[spin][Mc_AN][h_AN][i][j] = cdm_sum;
                                EDM[spin][Mc_AN][h_AN][i][j] = edm_sum;
                                My_Eele1_threads[OMPID * nspins_eele1 + spin] +=
                                    cdm_sum * Hks[spin][Mc_AN][h_AN][i][j];
                            }
                        }
                    }

                    dtime(&Etime_atom);
                    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
                }
            }

        } /* //#pragma omp parallel */

        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
                free(FermiW[spin][Mc_AN]);
            }
            free(EFermiW[spin]);
            free(FermiW[spin]);
        }
        free(EFermiW);
        free(FermiW);

        /****************************************************
                         bond energies
        ****************************************************/

        My_Eele1[0] = 0.0;
        My_Eele1[1] = 0.0;
        for (k = 0; k < nthreads_eele1; k++) {
            for (spin = 0; spin <= SpinP_switch; spin++) {
                My_Eele1[spin] += My_Eele1_threads[k * nspins_eele1 + spin];
            }
        }
        free(My_Eele1_threads);

        /* MPI, My_Eele1 */
        for (spin = 0; spin <= SpinP_switch; spin++) {
            MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        }

        if (SpinP_switch == 0) {
            Eele1[1] = Eele1[0];
        }

        if (3 <= level_stdout && myid == Host_ID) {
            printf("Eele00=%15.12f Eele01=%15.12f\n", Eele0[0], Eele0[1]);
            printf("Eele10=%15.12f Eele11=%15.12f\n", Eele1[0], Eele1[1]);
        }

        if (measure_time) {
            dtime(&etime);
            time6 += etime - stime;
        }

    } /* if ( strcasecmp(mode,"scf")==0 ) */

    else if (strcasecmp(mode, "dos") == 0) {
        Save_DOS_Col(Residues, OLP0, EVal, Msize);
    }

    if (measure_time) {
        printf("Divide_Conquer myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f\n",
               myid, time1, time2, time3, time4, time5, time6);
        fflush(stdout);
    }

    /****************************************************
      freeing of arrays:

    ****************************************************/

    free(Snd_H_Size);
    free(Rcv_H_Size);

    free(Snd_S_Size);
    free(Rcv_S_Size);

    free(Msize);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
            free(EVal[spin][Mc_AN]);
        }
        free(EVal[spin]);
    }
    free(EVal);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN   = 0;
                FNAN[0] = 1;
                tno1    = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];
            }

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                }

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
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
            free(PDOS_DC[spin][Mc_AN]);
        }
        free(PDOS_DC[spin]);
    }
    free(PDOS_DC);

    /* for time */
    dtime(&TEtime);
    time0 = TEtime - TStime;

    /* for PrintMemory */
    firsttime = 0;

    return time0;
}

#pragma optimization_level 2
static double DC_NonCol(char * mode, int SCF_iter, double ***** Hks, double ***** ImNL, double **** OLP0,
                        double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2])
{
    static int      firsttime = 1;
    int             Mc_AN, Gc_AN, i, Gi, wan, wanA, wanB, Anum;
    int             size1, size2, num, NUM, NUM1, n2, Cwan, Hwan;
    int             ih, ig, ian, j, kl, jg, jan, Bnum, m, n, spin, so;
    int             l, i1, j1, P_min, m_size;
    int             po, loopN, tno1, tno2, h_AN, Gh_AN;
    int             k, ii1, jj1, k1, l1, ompi;
    int             MA_AN, GA_AN, GB_AN, tnoA, tnoB;
    double          My_TZ, TZ, sum, FermiF, time0;
    double          My_Num_State, Num_State, x, Dnum;
    double          sum_r, sum_i, tmp1, tmp2;
    double          sum1_r, sum1_i, sum2_r, sum2_i;
    double          TStime, TEtime;
    double          My_Eele0[2], My_Eele1[2];
    double          max_x = 50.0;
    double          ChemP_MAX, ChemP_MIN, spin_degeneracy;
    double **       S_DC, *ko, *M1;
    dcomplex **     C, **H_DC;
    double **       EVal;
    dcomplex ****** Residues;
    double **       PDOS_DC;
    int *           MP, *Msize;
    double *        tmp_array;
    double *        tmp_array2;
    double *        omp_tmp;
    int *           Snd_H_Size, *Rcv_H_Size;
    int *           Snd_iHNL_Size, *Rcv_iHNL_Size;
    int *           Snd_S_Size, *Rcv_S_Size;
    int             numprocs, myid, ID, IDS, IDR, tag = 999;
    double          Stime_atom, Etime_atom;
    double          OLP_eigen_cut = Threshold_OLP_Eigen;

    MPI_Status  stat;
    MPI_Request request;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs, Nthrds0;

    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    int is_scf_mode = (strcasecmp(mode, "scf") == 0);

    // Set the device to be used by CUDA and OpenACC
    if (scf_eigen_lib_flag == CuSOLVER) {
        // CUDA
        set_cuda_default_device_from_local_rank();

        // OpenACC
        set_openacc_nvidia_device_from_local_rank();
    }

    dtime(&TStime);

    /****************************************************
      allocation of arrays:

      int MP[List_YOUSO[2]];
      int Msize[Matomnum+1];
      double EVal[Matomnum+1][n2];
    ****************************************************/

    Snd_H_Size    = (int *)malloc(sizeof(int) * numprocs);
    Rcv_H_Size    = (int *)malloc(sizeof(int) * numprocs);
    Snd_iHNL_Size = (int *)malloc(sizeof(int) * numprocs);
    Rcv_iHNL_Size = (int *)malloc(sizeof(int) * numprocs);
    Snd_S_Size    = (int *)malloc(sizeof(int) * numprocs);
    Rcv_S_Size    = (int *)malloc(sizeof(int) * numprocs);

    m_size = 0;
    Msize  = (int *)malloc(sizeof(int) * (Matomnum + 1));

    EVal = (double **)malloc(sizeof(double *) * (Matomnum + 1));

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

        if (Mc_AN == 0) {
            Gc_AN        = 0;
            FNAN[0]      = 1;
            SNAN[0]      = 0;
            n2           = 1;
            Msize[Mc_AN] = 1;
        } else {
            Gc_AN = M2G[Mc_AN];
            Anum  = 1;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                Gi   = natn[Gc_AN][i];
                wanA = WhatSpecies[Gi];
                Anum = Anum + Spe_Total_CNO[wanA];
            }
            NUM          = Anum - 1;
            Msize[Mc_AN] = NUM;
            n2           = 2 * NUM + 3;
        }

        m_size += n2;

        EVal[Mc_AN] = (double *)malloc(sizeof(double) * n2);
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: EVal", sizeof(double) * m_size, NULL);

    if (2 <= level_stdout) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            printf("<DC> myid=%i Mc_AN=%2d Gc_AN=%2d Msize=%3d\n", myid, Mc_AN, M2G[Mc_AN], Msize[Mc_AN]);
        }
    }

    /****************************************************
      allocation of arrays:

      dcomplex Residues[3]
                       [Matomnum+1]
                       [FNAN[Gc_AN]+1]
                       [Spe_Total_CNO[Gc_AN]]
                       [Spe_Total_CNO[Gh_AN]]
                       [n2]
       To reduce the memory size, the size of NUM2
       should be found in the loop.
    ****************************************************/

    m_size = 0;

    Residues = (dcomplex ******)malloc(sizeof(dcomplex *****) * 3);
    for (spin = 0; spin < 3; spin++) {
        Residues[spin] = (dcomplex *****)malloc(sizeof(dcomplex ****) * (Matomnum + 1));
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN   = 0;
                FNAN[0] = 1;
                tno1    = 1;
                n2      = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];
                n2    = 2 * Msize[Mc_AN] + 2;
            }

            Residues[spin][Mc_AN] = (dcomplex ****)malloc(sizeof(dcomplex ***) * (FNAN[Gc_AN] + 1));

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                }

                Residues[spin][Mc_AN][h_AN] = (dcomplex ***)malloc(sizeof(dcomplex **) * tno1);
                for (i = 0; i < tno1; i++) {
                    Residues[spin][Mc_AN][h_AN][i] = (dcomplex **)malloc(sizeof(dcomplex *) * tno2);
                    for (j = 0; j < tno2; j++) {
                        Residues[spin][Mc_AN][h_AN][i][j] = (dcomplex *)malloc(sizeof(dcomplex) * n2);
                    }
                }

                m_size += tno1 * tno2 * n2;
            }
        }
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: Residues", sizeof(dcomplex) * m_size, NULL);

    /****************************************************
      allocation of arrays:

      double PDOS_DC[Matomnum+1]
                    [n2]
    ****************************************************/

    m_size = 0;

    PDOS_DC = (double **)malloc(sizeof(double *) * (Matomnum + 1));
    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

        if (Mc_AN == 0)
            n2 = 1;
        else
            n2 = 2 * Msize[Mc_AN] + 2;

        m_size += n2;
        PDOS_DC[Mc_AN] = (double *)malloc(sizeof(double) * n2);
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: PDOS_DC", sizeof(double) * m_size, NULL);

    /* allocation of a temporal array for OpenMP */

    // #pragma omp parallel shared(Nthrds0)
    {
        Nthrds0 = omp_get_num_threads();
    }

    omp_tmp = (double *)malloc(sizeof(double) * Nthrds0);

    /****************************************************
     MPI

     Hks
    ****************************************************/

    /***********************************
              set data size
    ************************************/

    for (ID = 0; ID < numprocs; ID++) {

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (ID != 0) {
            tag = 999;

            /* find data size to send block data */
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
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    size1++;
                                }
                            }
                        }
                    }
                }

                Snd_H_Size[IDS] = size1;
                MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
            } else {
                Snd_H_Size[IDS] = 0;
            }

            /* receiving of size of data */

            if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {
                MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                Rcv_H_Size[IDR] = size2;
            } else {
                Rcv_H_Size[IDR] = 0;
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                MPI_Wait(&request, &stat);

        } else {
            Snd_H_Size[IDS] = 0;
            Rcv_H_Size[IDR] = 0;
        }
    }

    /***********************************
               data transfer
    ************************************/

    tag = 999;
    for (ID = 0; ID < numprocs; ID++) {

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (ID != 0) {

            /*****************************
                    sending of data
            *****************************/

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                size1 = Snd_H_Size[IDS];

                /* allocation of array */

                tmp_array = (double *)malloc(sizeof(double) * size1);

                /* multidimentional array to vector array */

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

                MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
            }

            /*****************************
               receiving of block data
            *****************************/

            if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                size2 = Rcv_H_Size[IDR];

                /* allocation of array */
                tmp_array2 = (double *)malloc(sizeof(double) * size2);

                MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
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

                /* freeing of array */
                free(tmp_array2);
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                MPI_Wait(&request, &stat);
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

    if (SO_switch == 1 || Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 ||
        Zeeman_NCO_switch == 1) {

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {
                tag = 999;

                /* find data size to send block data */
                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                    size1 = 0;
                    for (so = 0; so < List_YOUSO[5]; so++) {
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
                                        size1++;
                                    }
                                }
                            }
                        }
                    }

                    Snd_iHNL_Size[IDS] = size1;
                    MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                } else {
                    Snd_iHNL_Size[IDS] = 0;
                }

                /* receiving of size of data */

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {
                    MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                    Rcv_iHNL_Size[IDR] = size2;
                } else {
                    Rcv_iHNL_Size[IDR] = 0;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                    MPI_Wait(&request, &stat);

            } else {
                Snd_iHNL_Size[IDS] = 0;
                Rcv_iHNL_Size[IDR] = 0;
            }
        }

        /***********************************
                   data transfer
        ************************************/

        tag = 999;
        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {

                /*****************************
                       sending of data
                *****************************/

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                    size1 = Snd_iHNL_Size[IDS];

                    /* allocation of array */

                    tmp_array = (double *)malloc(sizeof(double) * size1);

                    /* multidimentional array to vector array */

                    num = 0;
                    for (so = 0; so < List_YOUSO[5]; so++) {
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

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                    size2 = Rcv_iHNL_Size[IDR];

                    /* allocation of array */
                    tmp_array2 = (double *)malloc(sizeof(double) * size2);

                    MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                    num = 0;
                    for (so = 0; so < List_YOUSO[5]; so++) {
                        Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
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

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
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

    if (SCF_iter <= 2) {

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {
                tag = 999;

                /* find data size to send block data */
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
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    size1++;
                                }
                            }
                        }
                    }

                    Snd_S_Size[IDS] = size1;
                    MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                } else {
                    Snd_S_Size[IDS] = 0;
                }

                /* receiving of size of data */

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {
                    MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                    Rcv_S_Size[IDR] = size2;
                } else {
                    Rcv_S_Size[IDR] = 0;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                    MPI_Wait(&request, &stat);
            } else {
                Snd_S_Size[IDS] = 0;
                Rcv_S_Size[IDR] = 0;
            }
        }

        /***********************************
                   data transfer
        ************************************/

        tag = 999;
        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {

                /*****************************
                        sending of data
                *****************************/

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                    size1 = Snd_S_Size[IDS];

                    /* allocation of array */

                    tmp_array = (double *)malloc(sizeof(double) * size1);

                    /* multidimentional array to vector array */

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

                    MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
                }

                /*****************************
                   receiving of block data
                *****************************/

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                    size2 = Rcv_S_Size[IDR];

                    /* allocation of array */
                    tmp_array2 = (double *)malloc(sizeof(double) * size2);

                    MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                    num   = 0;
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
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

                    /* freeing of array */
                    free(tmp_array2);
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
                    free(tmp_array); /* freeing of array */
                }
            }
        }
    }

    /****************************************************
              find the total number of electrons
    ****************************************************/

    My_TZ = 0.0;
    for (i = 1; i <= Matomnum; i++) {
        Gc_AN = M2G[i];
        wan   = WhatSpecies[Gc_AN];
        My_TZ = My_TZ + Spe_Core_Charge[wan];
    }

    /* MPI, My_TZ */

    MPI_Allreduce(&My_TZ, &TZ, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    // static double timeA = 0.0, timeB = 0.0, timeC = 0.0, timeD = 0.0, timeE = 0.0, timeF = 0.0, timeG = 0.0, timeH = 0.0;

    /****************************************************
        Setting of Hamiltonian and overlap matrices

           MP indicates the starting position of
                atom i in arraies H and S
    ****************************************************/

    // #pragma omp parallel shared(List_YOUSO, time_per_atom, Residues, EVal, S12, OLP_eigen_cut, ImNL, Hks, Zeeman_NCO_switch, Zeeman_NCS_switch, Constraint_NCS_switch, Hub_U_switch, SO_switch, OLP0, SCF_iter, RMI1, S_G2M, Spe_Total_CNO, natn, SNAN, FNAN, WhatSpecies, M2G, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, Etime_atom, Stime_atom, Gc_AN, wan, Anum, i, MP, Gi, wanA, NUM, n2, S_DC, H_DC, ko, M1, C, ig, ian, ih, j, kl, jg, jan, Bnum, m, n, P_min, l, i1, j1, tmp1, tmp2, k, jj1, k1, sum_r, sum_i, l1, sum1_r, sum1_i, sum2_r, sum2_i, ii1, NUM1, tno1, h_AN, Gh_AN, wanB, tno2)
    {

        /* get info. on OpenMP */

        OMPID  = 0;  // omp_get_thread_num();
        Nthrds = 1;  // omp_get_num_threads();
        Nprocs = 1;  // omp_get_num_procs();

        /* allocation of arrays */

        MP = (int *)malloc(sizeof(int) * List_YOUSO[2]);

        /* start of the Mc_AN loop which is parallelized by OpenMP */

        for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            wan   = WhatSpecies[Gc_AN];

            /***********************************************
            find the size of matrix for the atom Mc_AN
                      and set the MP vector

           Note:
               MP indicates the starting position of
                    atom i in arraies H and S
            ***********************************************/

            Anum = 1;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                MP[i] = Anum;
                Gi    = natn[Gc_AN][i];
                wanA  = WhatSpecies[Gi];
                Anum  = Anum + Spe_Total_CNO[wanA];
            }
            NUM = Anum - 1;

            n2 = 2 * NUM + 3;

            /***********************************************
             allocation of arrays:

             double   S_DC[NUM+2][NUM+2];
             dcomplex H_DC[n2][n2];
             double   ko[n2];
             double   M1[n2];
             dcomplex C[n2][n2];
            ***********************************************/

            S_DC = (double **)malloc(sizeof(double *) * (NUM + 2));
            for (i = 0; i < (NUM + 2); i++) {
                S_DC[i] = (double *)malloc(sizeof(double) * (NUM + 2));
            }

            H_DC = (dcomplex **)malloc(sizeof(dcomplex *) * n2);
            for (i = 0; i < n2; i++) {
                H_DC[i] = (dcomplex *)malloc(sizeof(dcomplex) * n2);
            }

            ko = (double *)malloc(sizeof(double) * n2);
            M1 = (double *)malloc(sizeof(double) * n2);

            C = (dcomplex **)malloc(sizeof(dcomplex *) * n2);
            for (i = 0; i < n2; i++) {
                C[i] = (dcomplex *)malloc(sizeof(dcomplex) * n2);
            }

            if (SCF_iter <= 2) {
                for (i = 0; i < (NUM + 2); i++) {
                    memset(S_DC[i], 0, sizeof(double) * (size_t)(NUM + 2));
                }
            }
            for (i = 0; i < n2; i++) {
                memset(H_DC[i], 0, sizeof(dcomplex) * (size_t)n2);
            }

            /***********************************************
             construct cluster full matrices of Hamiltonian
                     and overlap for the atom Mc_AN
            ***********************************************/

            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                ig   = natn[Gc_AN][i];
                ian  = Spe_Total_CNO[WhatSpecies[ig]];
                Anum = MP[i];
                ih   = S_G2M[ig];

                for (j = 0; j <= (FNAN[Gc_AN] + SNAN[Gc_AN]); j++) {

                    kl   = RMI1[Mc_AN][i][j];
                    jg   = natn[Gc_AN][j];
                    jan  = Spe_Total_CNO[WhatSpecies[jg]];
                    Bnum = MP[j];

                    if (0 <= kl) {

                        if (SCF_iter <= 2) {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    S_DC[Anum + m][Bnum + n] = OLP0[ih][kl][m][n];
                                }
                            }
                        }

                        /* non-spin-orbit coupling and non-LDA+U */
                        if (SO_switch == 0 && Hub_U_switch == 0 && Constraint_NCS_switch == 0 &&
                            Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0) {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    H_DC[Anum + m][Bnum + n].r             = Hks[0][ih][kl][m][n];
                                    H_DC[Anum + m][Bnum + n].i             = 0.0;
                                    H_DC[Anum + m + NUM][Bnum + n + NUM].r = Hks[1][ih][kl][m][n];
                                    H_DC[Anum + m + NUM][Bnum + n + NUM].i = 0.0;
                                    H_DC[Anum + m][Bnum + n + NUM].r       = Hks[2][ih][kl][m][n];
                                    H_DC[Anum + m][Bnum + n + NUM].i       = Hks[3][ih][kl][m][n];
                                    H_DC[Bnum + n + NUM][Anum + m].r       = H_DC[Anum + m][Bnum + n + NUM].r;
                                    H_DC[Bnum + n + NUM][Anum + m].i       = -H_DC[Anum + m][Bnum + n + NUM].i;
                                }
                            }
                        }

                        /* spin-orbit coupling or LDA+U */
                        else {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    H_DC[Anum + m][Bnum + n].r             = Hks[0][ih][kl][m][n];
                                    H_DC[Anum + m][Bnum + n].i             = ImNL[0][ih][kl][m][n];
                                    H_DC[Anum + m + NUM][Bnum + n + NUM].r = Hks[1][ih][kl][m][n];
                                    H_DC[Anum + m + NUM][Bnum + n + NUM].i = ImNL[1][ih][kl][m][n];
                                    H_DC[Anum + m][Bnum + n + NUM].r       = Hks[2][ih][kl][m][n];
                                    H_DC[Anum + m][Bnum + n + NUM].i = Hks[3][ih][kl][m][n] + ImNL[2][ih][kl][m][n];
                                    H_DC[Bnum + n + NUM][Anum + m].r = H_DC[Anum + m][Bnum + n + NUM].r;
                                    H_DC[Bnum + n + NUM][Anum + m].i = -H_DC[Anum + m][Bnum + n + NUM].i;
                                }
                            }
                        }

                    }
                }
            }

            /*
          if (Gc_AN==1){

          printf("H_DC.r\n");
          for (i=1; i<=NUM*2; i++){
            for (j=1; j<=NUM*2; j++){
              printf("H_DC.r i=%2d j=%2d %7.4f\n",i,j,H_DC[i][j].r);
            }
          }

          printf("H_DC.i\n");
          for (i=1; i<=NUM*2; i++){
            for (j=1; j<=NUM*2; j++){
              printf("H_DC.i i=%2d j=%2d %7.4f\n",i,j,H_DC[i][j].i);
            }
          }
          }
            */

            /*
            EigenBand_lapack(H_DC, ko, NUM*2, NUM*2, 1);

            if (Gc_AN==1){
              for (l=1; l<=NUM*2; l++){
                printf("ABC0 myid=%2d l=%2d ko=%15.12f\n",myid,l,ko[l]);fflush(stdout);
              }
            }
            */

            /****************************************************
             Solve the generalized eigenvalue problem
             HC = SCE

             1) diagonalize S
             2) search negative eigenvalues of S
            ****************************************************/

            if (SCF_iter <= 2) {
                // dtime(&timeA);

                Eigen_lapack(S_DC, ko, NUM, NUM);

                // dtime(&timeB);
                // printf("timeA = %.3f\n", timeB - timeA);

                /***********************************************
                     Searching of negative eigenvalues
                ************************************************/

                P_min = 1;
                for (l = 1; l <= NUM; l++) {
                    if (ko[l] < OLP_eigen_cut) {
                        P_min = l + 1;
                        if (3 <= level_stdout) {
                            printf("<DC>  Negative EV of OLP %2d %15.12f\n", l, ko[l]);
                        }
                    }
                }

                S12[Mc_AN][0][0] = P_min;

                for (l = 1; l < P_min; l++)
                    M1[l] = 0.0;
                for (l = P_min; l <= NUM; l++)
                    M1[l] = 1.0 / sqrt(ko[l]);

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = 1; j1 <= NUM; j1++) {
                        S_DC[i1][j1]       = S_DC[i1][j1] * M1[j1];
                        S12[Mc_AN][i1][j1] = S_DC[i1][j1];
                    }
                }
                // dtime(&timeC);
                // printf("timeB = %.3f\n", timeC - timeB);
            }

            else {

                P_min = (int)S12[Mc_AN][0][0];

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = 1; j1 <= NUM; j1++) {
                        S_DC[i1][j1] = S12[Mc_AN][i1][j1];
                    }
                }
                // dtime(&timeC);
            }

            /*
            printf("S_DC Gc_AN=%2d\n",Gc_AN);

          if (Gc_AN==1){

          printf("S_DC\n");


          for (i=1; i<=NUM; i++){
            for (j=1; j<=NUM; j++){
              printf("S_DC i=%2d j=%2d %7.4f\n",i,j,S_DC[i][j]);
            }
          }

          }
            */

            /***********************************************
                   transform Hamiltonian matrix
            ************************************************/

            int use_dc_openacc = (scf_eigen_lib_flag == CuSOLVER && GPU_CPU_SWITCH_NUM <= 2*NUM);

            if (use_dc_openacc) {
                // compiler's bug

                Anum = 1;
                for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                    MP[i] = Anum;
                    Gi    = natn[Gc_AN][i];
                    wanA  = WhatSpecies[Gi];
                    Anum  = Anum + Spe_Total_CNO[wanA];
                }
                NUM = Anum - 1;

                n2 = 2 * NUM + 3;

                /* transpose S */

#pragma acc data copyin(H_DC[ : n2][ : n2])
#pragma acc data copyin(S_DC[ : NUM + 1][ : NUM + 1])
#pragma acc data copy(C[ : n2][ : n2], ko[ : n2])
                {
#pragma acc kernels
#pragma acc loop independent
                    for (i1 = 1; i1 <= NUM; i1++) {
#pragma acc loop independent
                        for (j1 = i1 + 1; j1 <= NUM; j1++) {
                            double tmp1  = S_DC[i1][j1];
                            double tmp2  = S_DC[j1][i1];
                            S_DC[i1][j1] = tmp2;
                            S_DC[j1][i1] = tmp1;
                        }
                    }

                    /* H * U * M1 */

#pragma acc kernels
#pragma acc loop independent collapse(3) private(jj1, k1, l1, sum_r, sum_i) gang vector
                    for (j1 = P_min; j1 <= NUM; j1++) {
                        for (k = 0; k <= 1; k++) {
                            for (i1 = 1; i1 <= 2 * NUM; i1++) {
                                jj1 = 2 * j1 - P_min + k;
                                k1  = k * NUM;

                                double sum_r = 0.0;
                                double sum_i = 0.0;
#pragma acc loop independent reduction(+ : sum_r) reduction(+ : sum_i)
                                for (l = 1; l <= NUM; l++) {
                                    l1 = k1 + l;
                                    sum_r += H_DC[i1][l1].r * S_DC[j1][l];
                                    sum_i += H_DC[i1][l1].i * S_DC[j1][l];
                                }

                                C[jj1][i1].r = sum_r;
                                C[jj1][i1].i = sum_i;
                            }
                        }
                    }

                    /*
                if (Gc_AN==1){
                printf("C.r\n");
                for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                    printf("ABC1 C.r i=%2d j=%2d %7.4f\n",i,j,C[i][j].r);
                    }
                }

                printf("C.i\n");
                for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                    printf("ABC1 C.i i=%2d j=%2d %7.4f\n",i,j,C[i][j].i);
                    }
                }
                }
                    */

                    /* M1 * U^+ H * U * M1 */

#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (j1 = 1; j1 <= 2 * NUM; j1++) {
                        for (i1 = P_min; i1 <= NUM; i1++) {

                            double sum1_r = 0.0;
                            double sum1_i = 0.0;
                            double sum2_r = 0.0;
                            double sum2_i = 0.0;

#pragma acc loop independent reduction(+ : sum1_r) reduction(+ : sum1_i)
                            for (l = 1; l <= NUM; l++) {
                                sum1_r += S_DC[i1][l] * C[j1][l].r;
                                sum1_i += S_DC[i1][l] * C[j1][l].i;
                            }

#pragma acc loop independent reduction(+ : sum2_r) reduction(+ : sum2_i)
                            for (l = NUM + 1; l <= 2 * NUM; l++) {
                                l1 = l - NUM;
                                sum2_r += S_DC[i1][l1] * C[j1][l].r;
                                sum2_i += S_DC[i1][l1] * C[j1][l].i;
                            }

                            int ii1         = 2 * i1 - P_min;
                            H_DC[j1][ii1].r = sum1_r;
                            H_DC[j1][ii1].i = sum1_i;

                            ii1             = 2 * i1 - P_min + 1;
                            H_DC[j1][ii1].r = sum2_r;
                            H_DC[j1][ii1].i = sum2_i;
                        }
                    }

                    /* H to C (transposition) */

#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (i1 = P_min; i1 <= 2 * NUM; i1++) {
                        for (j1 = P_min; j1 <= 2 * NUM; j1++) {
                            C[j1 - (P_min - 1)][i1 - (P_min - 1)].r = H_DC[i1][j1].r;
                            C[j1 - (P_min - 1)][i1 - (P_min - 1)].i = H_DC[i1][j1].i;
                        }
                    }

                    if (Gc_AN == 1) {

                        /*
                printf("C.r\n");
                for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                    printf("ABC2 C.r i=%2d j=%2d %7.4f\n",i,j,C[i][j].r);
                    }
                }

                printf("C.i\n");
                for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                    printf("ABC2 C.i i=%2d j=%2d %7.4f\n",i,j,C[i][j].i);
                    }
                }
                    */

                        /*
                printf("C.r\n");
                for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                    printf("%7.4f",C[i][j].r);
                    }
                    printf("\n");
                }

                printf("C.i\n");
                for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                    printf("%7.4f",C[i][j].i);
                    }
                    printf("\n");
                }
                    */
                    }

                    /***********************************************
                 diagonalize the trasformed Hamiltonian matrix
                ************************************************/

                    // dtime(&timeD);
                    // printf("timeC = %.3f\n", timeD - timeC);

                    NUM1 = 2 * NUM - (P_min - 1);

                    Eigen_cusolver_x_complex_openacc(C, ko, NUM1, NUM1);

                    // dtime(&timeE);
                    // printf("timeD = %.3f\n", timeE - timeD);

                    /*
            if (Gc_AN==1){
                for (l=1; l<=NUM1; l++){
                printf("ABC10 myid=%2d l=%2d ko=%15.12f\n",myid,l,ko[l]);fflush(stdout);
                }
            }

            MPI_Finalize();
            exit(0);
                */

                    /* C to H (transposition) */

#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (i1 = 1; i1 <= NUM1; i1++) {
                        for (j1 = 1; j1 <= NUM1; j1++) {
                            H_DC[j1][i1].r = C[i1][j1].r;
                            H_DC[j1][i1].i = C[i1][j1].i;
                        }
                    }

                    /***********************************************
                    transformation to the original eigen vectors.
                    NOTE 244P    C = U * lambda^{-1/2} * D
                    ***********************************************/

                    /* transpose S */
#pragma acc kernels
#pragma acc loop independent
                    for (i1 = 1; i1 <= NUM; i1++) {
#pragma acc loop independent
                        for (j1 = i1 + 1; j1 <= NUM; j1++) {
                            double tmp1  = S_DC[i1][j1];
                            double tmp2  = S_DC[j1][i1];
                            S_DC[i1][j1] = tmp2;
                            S_DC[j1][i1] = tmp1;
                        }
                    }

#pragma acc kernels
#pragma acc loop independent collapse(3)
                    for (j1 = 1; j1 <= NUM1; j1++) {
                        for (k = 0; k <= 1; k++) {
                            for (i1 = 1; i1 <= NUM; i1++) {
                                double sum_r = 0.0;
                                double sum_i = 0.0;

#pragma acc loop independent reduction(+ : sum_r) reduction(+ : sum_i)
                                for (int l = P_min; l <= NUM; l++) {
                                    sum_r += S_DC[i1][l] * H_DC[j1][2 * (l - P_min) + 1 + k].r;
                                    sum_i += S_DC[i1][l] * H_DC[j1][2 * (l - P_min) + 1 + k].i;
                                }
                                C[i1 + k * NUM][j1].r = sum_r;
                                C[i1 + k * NUM][j1].i = sum_i;
                            }
                        }
                    }
                }
            } else {
                /* transpose S */

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = i1 + 1; j1 <= NUM; j1++) {
                        tmp1         = S_DC[i1][j1];
                        tmp2         = S_DC[j1][i1];
                        S_DC[i1][j1] = tmp2;
                        S_DC[j1][i1] = tmp1;
                    }
                }

                /* H * U * M1 */

                for (j1 = P_min; j1 <= NUM; j1++) {
                    for (k = 0; k <= 1; k++) {
                        jj1 = 2 * j1 - P_min + k;
                        k1  = k * NUM;

                        for (i1 = 1; i1 <= 2 * NUM; i1++) {

                            sum_r = 0.0;
                            sum_i = 0.0;

                            for (l = 1; l <= NUM; l++) {
                                l1 = k1 + l;
                                sum_r += H_DC[i1][l1].r * S_DC[j1][l];
                                sum_i += H_DC[i1][l1].i * S_DC[j1][l];
                            }

                            C[jj1][i1].r = sum_r;
                            C[jj1][i1].i = sum_i;
                        }
                    }
                }

                /*
              if (Gc_AN==1){
              printf("C.r\n");
              for (i=1; i<=NUM*2; i++){
                for (j=1; j<=NUM*2; j++){
                  printf("ABC1 C.r i=%2d j=%2d %7.4f\n",i,j,C[i][j].r);
                }
              }

              printf("C.i\n");
              for (i=1; i<=NUM*2; i++){
                for (j=1; j<=NUM*2; j++){
                  printf("ABC1 C.i i=%2d j=%2d %7.4f\n",i,j,C[i][j].i);
                }
              }
              }
                */

                /* M1 * U^+ H * U * M1 */

                for (j1 = 1; j1 <= 2 * NUM; j1++) {
                    for (i1 = P_min; i1 <= NUM; i1++) {

                        sum1_r = 0.0;
                        sum1_i = 0.0;
                        sum2_r = 0.0;
                        sum2_i = 0.0;

                        for (l = 1; l <= NUM; l++) {
                            sum1_r += S_DC[i1][l] * C[j1][l].r;
                            sum1_i += S_DC[i1][l] * C[j1][l].i;
                        }

                        for (l = NUM + 1; l <= 2 * NUM; l++) {
                            l1 = l - NUM;
                            sum2_r += S_DC[i1][l1] * C[j1][l].r;
                            sum2_i += S_DC[i1][l1] * C[j1][l].i;
                        }

                        ii1             = 2 * i1 - P_min;
                        H_DC[j1][ii1].r = sum1_r;
                        H_DC[j1][ii1].i = sum1_i;

                        ii1             = 2 * i1 - P_min + 1;
                        H_DC[j1][ii1].r = sum2_r;
                        H_DC[j1][ii1].i = sum2_i;
                    }
                }

                /* H to C (transposition) */

                for (i1 = P_min; i1 <= 2 * NUM; i1++) {
                    for (j1 = P_min; j1 <= 2 * NUM; j1++) {
                        C[j1 - (P_min - 1)][i1 - (P_min - 1)].r = H_DC[i1][j1].r;
                        C[j1 - (P_min - 1)][i1 - (P_min - 1)].i = H_DC[i1][j1].i;
                    }
                }

                if (Gc_AN == 1) {

                    /*
                  printf("C.r\n");
                  for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                      printf("ABC2 C.r i=%2d j=%2d %7.4f\n",i,j,C[i][j].r);
                    }
                  }

                  printf("C.i\n");
                  for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                      printf("ABC2 C.i i=%2d j=%2d %7.4f\n",i,j,C[i][j].i);
                    }
                  }
                    */

                    /*
                  printf("C.r\n");
                  for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                      printf("%7.4f",C[i][j].r);
                    }
                    printf("\n");
                  }

                  printf("C.i\n");
                  for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                      printf("%7.4f",C[i][j].i);
                    }
                    printf("\n");
                  }
                    */
                }

                /***********************************************
                 diagonalize the trasformed Hamiltonian matrix
                ************************************************/

                // dtime(&timeD);
                // printf("timeC = %.3f\n", timeD - timeC);

                NUM1 = 2 * NUM - (P_min - 1);
                EigenBand_lapack(C, ko, NUM1, NUM1, 1);

                // dtime(&timeE);
                // printf("timeD = %.3f\n", timeE - timeD);

                /*
              if (Gc_AN==1){
                for (l=1; l<=NUM1; l++){
                  printf("ABC10 myid=%2d l=%2d ko=%15.12f\n",myid,l,ko[l]);fflush(stdout);
                }
              }

              MPI_Finalize();
              exit(0);
                */

                /* C to H (transposition) */

                for (i1 = 1; i1 <= NUM1; i1++) {
                    for (j1 = 1; j1 <= NUM1; j1++) {
                        H_DC[j1][i1].r = C[i1][j1].r;
                        H_DC[j1][i1].i = C[i1][j1].i;
                    }
                }

                /***********************************************
                transformation to the original eigen vectors.
                NOTE 244P    C = U * lambda^{-1/2} * D
                ***********************************************/

                /* transpose S */

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = i1 + 1; j1 <= NUM; j1++) {
                        tmp1         = S_DC[i1][j1];
                        tmp2         = S_DC[j1][i1];
                        S_DC[i1][j1] = tmp2;
                        S_DC[j1][i1] = tmp1;
                    }
                }

                for (j1 = 1; j1 <= NUM1; j1++) {
                    for (k = 0; k <= 1; k++) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum_r = 0.0;
                            sum_i = 0.0;
                            for (l = P_min; l <= NUM; l++) {
                                sum_r += S_DC[i1][l] * H_DC[j1][2 * (l - P_min) + 1 + k].r;
                                sum_i += S_DC[i1][l] * H_DC[j1][2 * (l - P_min) + 1 + k].i;
                            }
                            C[i1 + k * NUM][j1].r = sum_r;
                            C[i1 + k * NUM][j1].i = sum_i;
                        }
                    }
                }
            }

            // dtime(&timeF);
            // printf("timeE = %.3f\n", timeF - timeE);

            /***********************************************
              store eigenvalues and residues of poles
            ***********************************************/

            for (i1 = 1; i1 <= 2 * NUM; i1++) {
                EVal[Mc_AN][i1 - 1] = 1000.0;
            }
            for (i1 = 1; i1 <= NUM1; i1++) {
                EVal[Mc_AN][i1 - 1] = ko[i1];
            }

            wanA = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[wanA];

            if (is_scf_mode) {
                for (i1 = 0; i1 < 2 * Msize[Mc_AN]; i1++) {
                    PDOS_DC[Mc_AN][i1] = 0.0;
                }
            }

            for (i = 0; i < tno1; i++) {
                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                    Bnum  = MP[h_AN];
                    for (j = 0; j < tno2; j++) {
                        tmp1 = OLP0[Mc_AN][h_AN][i][j];
                        for (i1 = 1; i1 <= NUM1; i1++) {

                            /* Re11 */
                            const double re11 =
                                C[1 + i][i1].r * C[Bnum + j][i1].r + C[1 + i][i1].i * C[Bnum + j][i1].i;
                            Residues[0][Mc_AN][h_AN][i][j][i1 - 1].r = re11;

                            /* Re22 */
                            const double re22 = C[1 + i + NUM][i1].r * C[Bnum + j + NUM][i1].r +
                                                C[1 + i + NUM][i1].i * C[Bnum + j + NUM][i1].i;
                            Residues[1][Mc_AN][h_AN][i][j][i1 - 1].r = re22;

                            if (is_scf_mode) {
                                PDOS_DC[Mc_AN][i1 - 1] += (re11 + re22) * tmp1;
                            }

                            /* Re12 */
                            Residues[2][Mc_AN][h_AN][i][j][i1 - 1].r =
                                C[1 + i][i1].r * C[Bnum + j + NUM][i1].r + C[1 + i][i1].i * C[Bnum + j + NUM][i1].i;

                            /* Im12 */
                            Residues[2][Mc_AN][h_AN][i][j][i1 - 1].i =
                                C[1 + i][i1].r * C[Bnum + j + NUM][i1].i - C[1 + i][i1].i * C[Bnum + j + NUM][i1].r;

                            /* spin-orbit coupling or LDA+U */
                            if (SO_switch == 1 || Hub_U_switch == 1 || 1 <= Constraint_NCS_switch ||
                                Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1) {

                                /* Im11 */
                                Residues[0][Mc_AN][h_AN][i][j][i1 - 1].i =
                                    C[1 + i][i1].r * C[Bnum + j][i1].i - C[1 + i][i1].i * C[Bnum + j][i1].r;

                                /*
                                printf("VVV1 Mc_AN=%2d h_AN=%2d i=%2d j=%2d i1=%2d %15.10f\n",
                                        Mc_AN,h_AN,i,j,i1,Residues[0][Mc_AN][h_AN][i][j][i1-1].i);
                                */

                                /* Im22 */
                                Residues[1][Mc_AN][h_AN][i][j][i1 - 1].i =
                                    C[1 + i + NUM][i1].r * C[Bnum + j + NUM][i1].i -
                                    C[1 + i + NUM][i1].i * C[Bnum + j + NUM][i1].r;
                            }
                        }
                    }
                }
            }

            // dtime(&timeG);
            // printf("timeF = %.3f\n", timeG - timeF);

            /****************************************************
                              free arrays
            ****************************************************/

            for (i = 0; i < (NUM + 2); i++) {
                free(S_DC[i]);
            }
            free(S_DC);

            for (i = 0; i < n2; i++) {
                free(H_DC[i]);
            }
            free(H_DC);

            free(ko);
            free(M1);

            for (i = 0; i < n2; i++) {
                free(C[i]);
            }
            free(C);

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

            /*
              for (i1=1; i1<=NUM1; i1++){
              printf("Mc_AN=%2d i1=%2d EVal=%15.12f\n",Mc_AN,i1,EVal[Mc_AN][i1-1]);
              }
            */

        } /* end of Mc_AN */

        /* freeing of arrays */

        free(MP);

    } /* //#pragma omp parallel */

    if (is_scf_mode) {

        /****************************************************
                  calculate projected DOS
        ****************************************************/

        /* PDOS_DC was accumulated while storing Residues, avoiding a second pass here. */

        /****************************************************
        find chemical potential
        ****************************************************/

        po    = 0;
        loopN = 0;

        ChemP_MAX = 15.0;
        ChemP_MIN = -15.0;

        do {
            ChemP = 0.50 * (ChemP_MAX + ChemP_MIN);

            // #pragma omp parallel shared(omp_tmp, max_x, ChemP, Beta, EVal, Msize, Matomnum, M2G) private(OMPID, Nthrds, Nprocs, Mc_AN, Gc_AN, i, x, FermiF)
            {

                /* get info. on OpenMP */

                OMPID  = omp_get_thread_num();
                Nthrds = omp_get_num_threads();
                Nprocs = omp_get_num_procs();

                omp_tmp[OMPID] = 0.0;

                for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

                    Gc_AN = M2G[Mc_AN];

                    for (i = 0; i < 2 * Msize[Mc_AN]; i++) {
                        FermiF = DC_FermiWeight(EVal[Mc_AN][i], ChemP, Beta, max_x);
                        omp_tmp[OMPID] += FermiF * PDOS_DC[Mc_AN][i];
                    }
                }

            } /* //#pragma omp parallel */

            My_Num_State = 0.0;
            for (ompi = 0; ompi < Nthrds0; ompi++) {
                My_Num_State += omp_tmp[ompi];
            }

            /* MPI, My_Num_State */

            MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

            Dnum = (TZ - Num_State) - system_charge;
            if (0.0 <= Dnum)
                ChemP_MIN = ChemP;
            else
                ChemP_MAX = ChemP;
            if (fabs(Dnum) < 1.0e-11)
                po = 1;

            if (myid == Host_ID && 2 <= level_stdout) {
                printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n", ChemP, TZ, Num_State);
            }

            loopN++;
        } while (po == 0 && loopN < 1000);

        /****************************************************
            eigenenergy by summing up eigenvalues
        ****************************************************/

        double **FermiW =
            (double **)DC_MallocArray((size_t)(Matomnum + 1), sizeof(double *), "DC noncol Fermi weights");
        double **EFermiW = (double **)DC_MallocArray((size_t)(Matomnum + 1), sizeof(double *),
                                                    "DC noncol energy Fermi weights");
        FermiW[0]  = NULL;
        EFermiW[0] = NULL;

        My_Eele0[0] = 0.0;
        My_Eele0[1] = 0.0;

        // #pragma omp parallel shared(PDOS_DC, Beta, ChemP, max_x, EVal, Msize, M2G, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, Gc_AN, i, x, FermiF)
        {

            /* get info. on OpenMP */

            OMPID  = omp_get_thread_num();
            Nthrds = omp_get_num_threads();
            Nprocs = omp_get_num_procs();

            omp_tmp[OMPID] = 0.0;

            for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

                Gc_AN = M2G[Mc_AN];
                {
                    const int     n       = 2 * Msize[Mc_AN];
                    const double *eval    = EVal[Mc_AN];
                    double *      weights = (double *)DC_MallocArray((size_t)(2 * n), sizeof(double),
                                                                     "DC noncol Fermi weight buffer");
                    double *      fweight = weights;
                    double *      eweight = weights + n;

                    FermiW[Mc_AN]  = fweight;
                    EFermiW[Mc_AN] = eweight;

                    for (i = 0; i < n; i++) {
                        FermiF     = DC_FermiWeight(eval[i], ChemP, Beta, max_x);
                        fweight[i] = FermiF;
                        eweight[i] = FermiF * eval[i];
                        omp_tmp[OMPID] += eweight[i] * PDOS_DC[Mc_AN][i];
                    }
                }
            }

        } /* //#pragma omp parallel */

        for (ompi = 0; ompi < Nthrds0; ompi++) {
            My_Eele0[0] += omp_tmp[ompi];
        }

        /* MPI, My_Eele0 */

        MPI_Allreduce(&My_Eele0[0], &Eele0[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        MPI_Allreduce(&My_Eele0[1], &Eele0[1], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

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

        for (ompi = 0; ompi < Nthrds0; ompi++) {
            omp_tmp[ompi] = 0.0;
        }

        // #pragma omp parallel shared(iDM, Zeeman_NCO_switch, Zeeman_NCS_switch, Constraint_NCS_switch, Hub_U_switch, SO_switch, time_per_atom, EDM, CDM, Residues, natn, max_x, Beta, ChemP, EVal, Msize, Spe_Total_CNO, WhatSpecies, M2G, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, Stime_atom, Gc_AN, wanA, tno1, i1, x, FermiF, h_AN, Gh_AN, wanB, tno2, i, j, tmp1, Etime_atom)
        {

            /* get info. on OpenMP */

            OMPID  = omp_get_thread_num();
            Nthrds = omp_get_num_threads();
            Nprocs = omp_get_num_procs();

            for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

                dtime(&Stime_atom);

                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];
                int use_iDM = (SO_switch == 1 || Hub_U_switch == 1 || 1 <= Constraint_NCS_switch ||
                               Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1);

                const int     n       = 2 * Msize[Mc_AN];
                const double *fweight = FermiW[Mc_AN];
                const double *eweight = EFermiW[Mc_AN];

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                    for (i = 0; i < tno1; i++) {
                        for (j = 0; j < tno2; j++) {
                            double cdm0 = 0.0, cdm1 = 0.0, cdm2 = 0.0, cdm3 = 0.0;
                            double edm0 = 0.0, edm1 = 0.0, edm2 = 0.0, edm3 = 0.0;
                            double idm00 = 0.0, idm01 = 0.0;
                            dcomplex *res0 = Residues[0][Mc_AN][h_AN][i][j];
                            dcomplex *res1 = Residues[1][Mc_AN][h_AN][i][j];
                            dcomplex *res2 = Residues[2][Mc_AN][h_AN][i][j];

                            DC_DotDensityResidueNonCol(n, res0, res1, res2, fweight, eweight, use_iDM,
                                                       &cdm0, &edm0, &cdm1, &edm1, &cdm2, &edm2, &cdm3, &edm3,
                                                       &idm00, &idm01);

                            CDM[0][Mc_AN][h_AN][i][j] = cdm0;
                            EDM[0][Mc_AN][h_AN][i][j] = edm0;
                            CDM[1][Mc_AN][h_AN][i][j] = cdm1;
                            EDM[1][Mc_AN][h_AN][i][j] = edm1;
                            CDM[2][Mc_AN][h_AN][i][j] = cdm2;
                            EDM[2][Mc_AN][h_AN][i][j] = edm2;
                            CDM[3][Mc_AN][h_AN][i][j] = cdm3;
                            EDM[3][Mc_AN][h_AN][i][j] = edm3;

                            if (use_iDM) {
                                iDM[0][0][Mc_AN][h_AN][i][j] = idm00;
                                iDM[0][1][Mc_AN][h_AN][i][j] = idm01;
                                omp_tmp[OMPID] +=
                                    +cdm0 * Hks[0][Mc_AN][h_AN][i][j] -
                                    idm00 * ImNL[0][Mc_AN][h_AN][i][j] +
                                    cdm1 * Hks[1][Mc_AN][h_AN][i][j] -
                                    idm01 * ImNL[1][Mc_AN][h_AN][i][j] +
                                    2.0 * cdm2 * Hks[2][Mc_AN][h_AN][i][j] -
                                    2.0 * cdm3 * (Hks[3][Mc_AN][h_AN][i][j] + ImNL[2][Mc_AN][h_AN][i][j]);
                            } else {
                                omp_tmp[OMPID] +=
                                    +cdm0 * Hks[0][Mc_AN][h_AN][i][j] +
                                    cdm1 * Hks[1][Mc_AN][h_AN][i][j] +
                                    2.0 * cdm2 * Hks[2][Mc_AN][h_AN][i][j] -
                                    2.0 * cdm3 * Hks[3][Mc_AN][h_AN][i][j];
                            }
                        }
                    }
                }

                dtime(&Etime_atom);
                time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
            }
        }

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            free(FermiW[Mc_AN]);
        }
        free(EFermiW);
        free(FermiW);

        /****************************************************
                         bond energies
        ****************************************************/

        My_Eele1[0] = 0.0;
        for (ompi = 0; ompi < Nthrds0; ompi++) {
            My_Eele1[0] += omp_tmp[ompi];
        }

        /* MPI, My_Eele1 */
        MPI_Allreduce(&My_Eele1[0], &Eele1[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

        if (3 <= level_stdout && myid == Host_ID) {
            printf("Eele0=%15.12f\n", Eele0[0]);
            printf("Eele1=%15.12f\n", Eele1[0]);
        }

    } /* if ( strcasecmp(mode,"scf")==0 ) */

    else if (strcasecmp(mode, "dos") == 0) {
        Save_DOS_NonCol(Residues, OLP0, EVal, Msize);
    }

    // dtime(&timeH);
    // printf("timeG = %.3f\n", timeH - timeG);
    /****************************************************
      freeing of arrays:

    ****************************************************/

    free(Snd_H_Size);
    free(Rcv_H_Size);

    free(Snd_iHNL_Size);
    free(Rcv_iHNL_Size);

    free(Snd_S_Size);
    free(Rcv_S_Size);

    free(Msize);

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
        free(EVal[Mc_AN]);
    }
    free(EVal);

    for (spin = 0; spin < 3; spin++) {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN   = 0;
                FNAN[0] = 1;
                tno1    = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];
            }

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                }

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

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
        free(PDOS_DC[Mc_AN]);
    }
    free(PDOS_DC);

    free(omp_tmp);

    /* for time */
    dtime(&TEtime);
    time0 = TEtime - TStime;

    /* for PrintMemory */
    firsttime = 0;

    return time0;
}

void Save_DOS_Col(double ****** Residues, double **** OLP0, double *** EVal, int * Msize)
{
    int    spin, Mc_AN, wanA, Gc_AN, tno1;
    int    i1, i, j, MaxL, l, h_AN, Gh_AN, wanB, tno2;
    double Stime_atom, Etime_atom;
    double sum;
    int    i_vec[10];
    char   file_eig[YOUSO10], file_ev[YOUSO10];
    FILE * fp_eig, *fp_ev;
    int    numprocs, myid, ID, tag;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /* open file pointers */

    if (myid == Host_ID) {

        sprintf(file_eig, "%s%s.Dos.val", filepath, filename);
        if ((fp_eig = fopen(file_eig, "w")) == NULL) {
            printf("cannot open a file %s\n", file_eig);
        }
    }

    sprintf(file_ev, "%s%s.Dos.vec%i", filepath, filename, myid);
    if ((fp_ev = fopen(file_ev, "w")) == NULL) {
        printf("cannot open a file %s\n", file_ev);
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
            fprintf(fp_ev, "%d\n", Msize[Mc_AN]);

            for (i1 = 0; i1 < Msize[Mc_AN]; i1++) {

                fprintf(fp_ev, "%4d  %10.6f  ", i1, EVal[spin][Mc_AN][i1]);

                for (i = 0; i < tno1; i++) {

                    sum = 0.0;
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB  = WhatSpecies[Gh_AN];
                        tno2  = Spe_Total_CNO[wanB];
                        for (j = 0; j < tno2; j++) {
                            sum += Residues[spin][Mc_AN][h_AN][i][j][i1] * OLP0[Mc_AN][h_AN][i][j];
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

    if (myid == Host_ID) {

        fprintf(fp_eig, "mode        5\n");
        fprintf(fp_eig, "NonCol      0\n");
        /*      fprintf(fp_eig,"N           %d\n",n); */
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
    }

    /* close file pointers */

    if (myid == Host_ID) {
        if (fp_eig)
            fclose(fp_eig);
    }

    if (fp_ev)
        fclose(fp_ev);
}

void Save_DOS_NonCol(dcomplex ****** Residues, double **** OLP0, double ** EVal, int * Msize)
{
    int    spin, Mc_AN, wanA, Gc_AN, tno1;
    int    i1, i, j, MaxL, l, h_AN, Gh_AN, wanB, tno2;
    double Stime_atom, Etime_atom;
    double tmp1, tmp2, tmp3, sum, SDup, SDdn;
    double Re11, Re22, Re12, Im12;
    double theta, phi, sit, cot, sip, cop;
    int    i_vec[10];
    char   file_eig[YOUSO10], file_ev[YOUSO10];
    FILE * fp_eig, *fp_ev;
    int    numprocs, myid, ID, tag;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /* open file pointers */

    if (myid == Host_ID) {

        sprintf(file_eig, "%s%s.Dos.val", filepath, filename);
        if ((fp_eig = fopen(file_eig, "w")) == NULL) {
            printf("cannot open a file %s\n", file_eig);
        }
    }

    sprintf(file_ev, "%s%s.Dos.vec%i", filepath, filename, myid);
    if ((fp_ev = fopen(file_ev, "w")) == NULL) {
        printf("cannot open a file %s\n", file_ev);
    }

    /****************************************************
                     save *.Dos.vec
    ****************************************************/

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

        dtime(&Stime_atom);

        Gc_AN = M2G[Mc_AN];
        wanA  = WhatSpecies[Gc_AN];
        tno1  = Spe_Total_CNO[wanA];

        theta = Angle0_Spin[Gc_AN];
        phi   = Angle1_Spin[Gc_AN];

        sit = sin(theta);
        cot = cos(theta);
        sip = sin(phi);
        cop = cos(phi);

        fprintf(fp_ev, "<AN%d\n", Gc_AN);
        fprintf(fp_ev, "%d %d\n", 2 * Msize[Mc_AN], 2 * Msize[Mc_AN]);

        for (i1 = 0; i1 < 2 * Msize[Mc_AN]; i1++) {

            fprintf(fp_ev, "%4d  %10.6f %10.6f ", i1, EVal[Mc_AN][i1], EVal[Mc_AN][i1]);

            for (i = 0; i < tno1; i++) {

                Re11 = 0.0;
                Re22 = 0.0;
                Re12 = 0.0;
                Im12 = 0.0;

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];

                    for (j = 0; j < tno2; j++) {

                        Re11 += Residues[0][Mc_AN][h_AN][i][j][i1].r * OLP0[Mc_AN][h_AN][i][j];

                        Re22 += Residues[1][Mc_AN][h_AN][i][j][i1].r * OLP0[Mc_AN][h_AN][i][j];

                        Re12 += Residues[2][Mc_AN][h_AN][i][j][i1].r * OLP0[Mc_AN][h_AN][i][j];

                        Im12 += Residues[2][Mc_AN][h_AN][i][j][i1].i * OLP0[Mc_AN][h_AN][i][j];
                    }
                }

                tmp1 = 0.5 * (Re11 + Re22);
                tmp2 = 0.5 * cot * (Re11 - Re22);
                tmp3 = (Re12 * cop - Im12 * sip) * sit;

                SDup = tmp1 + tmp2 + tmp3;
                SDdn = tmp1 - tmp2 - tmp3;

                fprintf(fp_ev, "%8.5f %8.5f ", SDup, SDdn);
            }
            fprintf(fp_ev, "\n");
        }

        fprintf(fp_ev, "AN%d>\n", Gc_AN);

        dtime(&Etime_atom);
        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    /****************************************************
                     save *.Dos.val
    ****************************************************/

    if (myid == Host_ID) {

        fprintf(fp_eig, "mode        5\n");
        fprintf(fp_eig, "NonCol      1\n");
        /*      fprintf(fp_eig,"N           %d\n",n); */
        fprintf(fp_eig, "Nspin       %d\n", 1); /* switch to 1 */
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

        fprintf(fp_eig, "<SpinAngle\n");
        for (i = 1; i <= atomnum; i++) {
            fprintf(fp_eig, "%lf %lf\n", Angle0_Spin[i], Angle1_Spin[i]);
        }
        fprintf(fp_eig, "SpinAngle>\n");
    }

    /* close file pointers */

    if (myid == Host_ID) {
        if (fp_eig)
            fclose(fp_eig);
    }

    if (fp_ev)
        fclose(fp_ev);
}
