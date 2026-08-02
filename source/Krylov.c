/**********************************************************************
  Krylov.c:

     Krylov.c is a subroutine to perform a Krylov subspace
     method developed by T.Ozaki.

  Log of Krylov.c

     10/June/2005  Released by T.Ozaki

***********************************************************************/

#include "lapack_prototypes.h"
#include "mpi.h"
#include "openmx_common.h"
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#define measure_time 0
#define error_check 0
#define cutoff_value Threshold_OLP_Eigen
#define KRYLOV_ENABLE_GPU 1
/* GPU/CPU crossover for the projected eigensolve, compared against Msize3. */
#define KRYLOV_GPU_EIGEN_MIN 300
#define KRYLOV_GPU_DGEMM_MIN_FLOPS 50000000.0

#include "tran_prototypes.h"

#ifndef __x86_64__
#include "mimic_sse.h"
#else
#include <emmintrin.h>
#endif

#ifdef kcomp
#define _mm_loadu_pd _mm_load_pd
#define _mm_storeu_pd _mm_store_pd
#endif

static void Generate_pMatrix(int myid, int spin, int Mc_AN, double***** Hks, double**** OLP0, double** invS,
    double*** Krylov_U, double*** Krylov_U_OLP, double** inv_RS, int* MP,
    int* Msize, int* Msize2, int* Msize3, int* Msize4, int Msize2_max,
    double** tmpvec0, double** tmpvec1, double** tmpvec2);

static void Generate_pMatrix2(int myid, int spin, int Mc_AN, double***** Hks, double**** OLP0, double*** Krylov_U,
    int* MP, int* Msize, int* Msize2, int* Msize3,
    double** tmpvec1);
static void Krylov_IOLP(int Mc_AN, double**** OLP0, double*** Krylov_U_OLP, double** inv_RS,
    int* MP, int* Msize2, int* Msize4, int Msize2_max,
    double** tmpvec0, double** tmpvec1);
static void S_orthonormalize_vec(int Mc_AN, int ct_on, double** vec,
    double** workvec, double**** OLP0,
    double** tmpmat0, double* ko, double* iko, int* MP, int* Msize2);
static void Embedding_Matrix(int spin, int Mc_AN, double***** Hks,
    double*** Krylov_U, double**** EC_matrix,
    int* MP, int* Msize, int* Msize2, int* Msize3,
    double** tmpvec1);

static void Inverse_S_by_Cholesky(int Mc_AN, double**** OLP0, double** invS, int* MP, int NUM, double* LoS);

static void Save_DOS_Col(double****** Residues, double**** OLP0, double*** EVal, int** LO_TC, int** HO_TC);

static double Krylov_Col(char* mode,
    int SCF_iter,
    double***** Hks,
    double**** OLP0,
    double***** CDM,
    double***** EDM,
    double Eele0[2],
    double Eele1[2]);

/*******************************************************
  The following subroutines are called only when
  Matomnum==1 && openmp_threads_num>1
*******************************************************/

static double Krylov_Col_trd(char* mode,
    int SCF_iter,
    double***** Hks,
    double**** OLP0,
    double***** CDM,
    double***** EDM,
    double Eele0[2],
    double Eele1[2]);

static void Generate_pMatrix_trd(int myid, int spin, int Mc_AN, double***** Hks, double**** OLP0, double** invS,
    double*** Krylov_U, double*** Krylov_U_OLP, double** inv_RS, int* MP,
    int* Msize, int* Msize2, int* Msize3, int* Msize4, int Msize2_max,
    double** tmpvec0, double** tmpvec1, double** tmpvec2);

static void Generate_pMatrix2_trd(int myid, int spin, int Mc_AN, double***** Hks, double**** OLP0, double*** Krylov_U,
    int* MP, int* Msize, int* Msize2, int* Msize3,
    double** tmpvec1);
static void Krylov_IOLP_trd(int Mc_AN, double**** OLP0, double*** Krylov_U_OLP, double** inv_RS,
    int* MP, int* Msize2, int* Msize4, int Msize2_max,
    double** tmpvec0, double** tmpvec1);
static void S_orthonormalize_vec_trd(int Mc_AN, int ct_on, double** vec,
    double** workvec, double**** OLP0,
    double** tmpmat0, double* ko, double* iko, int* MP, int* Msize2);
static void Embedding_Matrix_trd(int spin, int Mc_AN, double***** Hks,
    double*** Krylov_U, double**** EC_matrix,
    int* MP, int* Msize, int* Msize2, int* Msize3,
    double** tmpvec1, int EKC_core_size_max, int Msize2_max);

static int Eigen_lapack_x(double** a, double* ko, int n, int EVmax);
static int Eigen_lapack_d(double** a, double* ko, int n, int EVmax);
static int Eigen_lapack_r(double** a, double* ko, int n, int EVmax);

static int Krylov_GPU_Enabled(void)
{
  /*
    Keep Krylov's local dense GPU path behind one switch so it is easy to
    fall back to the CPU BLAS/LAPACK path when needed.
  */
  return KRYLOV_ENABLE_GPU && scf_eigen_lib_flag == GPUSOLVER;
}

typedef struct {
  int active;
  cublasHandle_t cublas;
  cusolverDnHandle_t gpusolver;
  cudaStream_t stream;
  double *d_A;
  double *d_B;
  double *d_C;
  double *d_W;
  int32_t *d_info;
  void *d_work;
  void *h_work;
  double *h_A;
  size_t d_A_count;
  size_t d_B_count;
  size_t d_C_count;
  size_t d_W_count;
  size_t d_work_bytes;
  size_t h_work_bytes;
  size_t h_A_count;
} Krylov_GPU_Workspace;

static void Krylov_GPU_InitOnce(void);
static void Krylov_GPU_Workspace_Init(Krylov_GPU_Workspace *ws);
static void Krylov_GPU_Workspace_Free(Krylov_GPU_Workspace *ws);
static void Krylov_GPU_EnsureCublas(Krylov_GPU_Workspace *ws);
static void Krylov_GPU_EnsureGpusolver(Krylov_GPU_Workspace *ws);
static void Krylov_GPU_PreparePool(int nthrds);
static Krylov_GPU_Workspace *Krylov_GPU_GetWorkspace(int thread_id);
static void Krylov_Dgemm(Krylov_GPU_Workspace *ws,
                         cublasOperation_t transa, cublasOperation_t transb,
                         int m, int n, int k,
                         const double *A, int lda,
                         const double *B, int ldb,
                         double *C, int ldc);
static void Krylov_Eigen2(Krylov_GPU_Workspace *ws, double *a, int csize, double *ko, int n, int EVmax);


/* Dispatch thresholds; the env overrides let small systems exercise the GPU
   path and let production runs tune the crossover.  Initialized once from
   Krylov_GPU_InitOnce (single threaded), read-only afterwards. */
static double Krylov_gpu_dgemm_min_flops = KRYLOV_GPU_DGEMM_MIN_FLOPS;
static int    Krylov_gpu_eigen_min       = KRYLOV_GPU_EIGEN_MIN;
static int    Krylov_gpu_fused           = 1;

static void Krylov_GPU_InitThresholds(void)
{
  const char *env;

  env = getenv("OPENMX_KRYLOV_GPU_DGEMM_MIN_FLOPS");
  if (env != NULL && env[0] != '\0'){
    double p = atof(env);
    if (0.0 <= p) Krylov_gpu_dgemm_min_flops = p;
  }

  env = getenv("OPENMX_KRYLOV_GPU_EIGEN_MIN");
  if (env != NULL && env[0] != '\0'){
    int p = atoi(env);
    if (0 < p) Krylov_gpu_eigen_min = p;
  }

  env = getenv("OPENMX_KRYLOV_GPU_FUSED");
  if (env != NULL && env[0] != '\0'){
    Krylov_gpu_fused = (atoi(env) != 0);
  }
}

static void Krylov_GPU_InitOnce(void)
{
  static int initialized = 0;

  if (initialized) return;
  initialized = 1;

  Krylov_GPU_InitThresholds();

  if (Krylov_GPU_Enabled()){
    wait_cudafunc(cudaFree(0));
  }
}

static Krylov_GPU_Workspace *Krylov_gpu_ws_pool = NULL;
static int Krylov_gpu_ws_pool_size = 0;

static void Krylov_GPU_PreparePool(int nthrds)
{
  int i;

  if (!Krylov_GPU_Enabled() || nthrds <= Krylov_gpu_ws_pool_size) return;

  Krylov_GPU_Workspace *new_pool =
    (Krylov_GPU_Workspace*)realloc(Krylov_gpu_ws_pool, sizeof(Krylov_GPU_Workspace)*(size_t)nthrds);
  if (new_pool == NULL){
    fprintf(stderr,"Krylov: could not allocate GPU workspace pool.\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  Krylov_gpu_ws_pool = new_pool;
  for (i=Krylov_gpu_ws_pool_size; i<nthrds; i++){
    memset(&Krylov_gpu_ws_pool[i],0,sizeof(Krylov_GPU_Workspace));
  }
  Krylov_gpu_ws_pool_size = nthrds;
}

static Krylov_GPU_Workspace *Krylov_GPU_GetWorkspace(int thread_id)
{
  Krylov_GPU_Workspace *ws;

  if (!Krylov_GPU_Enabled() ||
      thread_id < 0 || Krylov_gpu_ws_pool_size <= thread_id) return NULL;

  ws = &Krylov_gpu_ws_pool[thread_id];
  if (!ws->active && ws->stream == NULL){
    Krylov_GPU_Workspace_Init(ws);
  }

  return ws;
}

static void Krylov_GPU_EnsureDouble(double **ptr, size_t *capacity, size_t count)
{
  if (*capacity < count){
    if (*ptr != NULL) wait_cudafunc(cudaFree(*ptr));
    wait_cudafunc(cudaMalloc((void**)ptr, sizeof(double)*count));
    *capacity = count;
  }
}

static void Krylov_GPU_EnsureInfo(int32_t **ptr)
{
  if (*ptr == NULL){
    wait_cudafunc(cudaMalloc((void**)ptr, sizeof(int32_t)));
  }
}

static void Krylov_GPU_EnsureDeviceWork(Krylov_GPU_Workspace *ws, size_t bytes)
{
  if (ws->d_work_bytes < bytes){
    if (ws->d_work != NULL) wait_cudafunc(cudaFree(ws->d_work));
    if (bytes != 0){
      wait_cudafunc(cudaMalloc((void**)&ws->d_work, bytes));
    }
    else{
      ws->d_work = NULL;
    }
    ws->d_work_bytes = bytes;
  }
}

static void Krylov_GPU_EnsureHostWork(Krylov_GPU_Workspace *ws, size_t bytes)
{
  if (ws->h_work_bytes < bytes){
    if (ws->h_work != NULL) free(ws->h_work);
    ws->h_work = (bytes == 0) ? NULL : malloc(bytes);
    if (bytes != 0 && ws->h_work == NULL){
      fprintf(stderr,"Krylov: could not allocate GPUSOLVER host workspace.\n");
      MPI_Abort(MPI_COMM_WORLD,1);
    }
    ws->h_work_bytes = bytes;
  }
}

static void Krylov_GPU_EnsureHostMatrix(Krylov_GPU_Workspace *ws, size_t count)
{
  if (ws->h_A_count < count){
    double *p = (double*)realloc(ws->h_A, sizeof(double)*count);
    if (p == NULL){
      fprintf(stderr,"Krylov: could not allocate host staging matrix.\n");
      MPI_Abort(MPI_COMM_WORLD,1);
    }
    ws->h_A = p;
    ws->h_A_count = count;
  }
}

static void Krylov_GPU_Workspace_Init(Krylov_GPU_Workspace *ws)
{
  memset(ws,0,sizeof(Krylov_GPU_Workspace));

  if (!Krylov_GPU_Enabled()) return;

  ws->active = 1;
  wait_cudafunc(cudaStreamCreateWithFlags(&ws->stream, cudaStreamNonBlocking));
}

static void Krylov_GPU_Workspace_Free(Krylov_GPU_Workspace *ws)
{
  if (ws->stream != NULL) wait_cudafunc(cudaStreamSynchronize(ws->stream));
  if (ws->d_A    != NULL) wait_cudafunc(cudaFree(ws->d_A));
  if (ws->d_B    != NULL) wait_cudafunc(cudaFree(ws->d_B));
  if (ws->d_C    != NULL) wait_cudafunc(cudaFree(ws->d_C));
  if (ws->d_W    != NULL) wait_cudafunc(cudaFree(ws->d_W));
  if (ws->d_info != NULL) wait_cudafunc(cudaFree(ws->d_info));
  if (ws->d_work != NULL) wait_cudafunc(cudaFree(ws->d_work));
  if (ws->h_work != NULL) free(ws->h_work);
  if (ws->h_A    != NULL) free(ws->h_A);
  if (ws->gpusolver != NULL) wait_cudafunc(cusolverDnDestroy(ws->gpusolver));
  if (ws->cublas   != NULL) wait_cudafunc(cublasDestroy(ws->cublas));
  if (ws->stream   != NULL) wait_cudafunc(cudaStreamDestroy(ws->stream));
  memset(ws,0,sizeof(Krylov_GPU_Workspace));
}

static void Krylov_GPU_EnsureCublas(Krylov_GPU_Workspace *ws)
{
  if (ws->cublas == NULL){
    wait_cudafunc(cublasCreate(&ws->cublas));
    wait_cudafunc(cublasSetStream(ws->cublas, ws->stream));
  }
}

static void Krylov_GPU_EnsureGpusolver(Krylov_GPU_Workspace *ws)
{
  if (ws->gpusolver == NULL){
    wait_cudafunc(cusolverDnCreate(&ws->gpusolver));
    wait_cudafunc(cusolverDnSetStream(ws->gpusolver, ws->stream));
  }
}

static void Krylov_Dgemm(Krylov_GPU_Workspace *ws,
                         cublasOperation_t transa, cublasOperation_t transb,
                         int m, int n, int k,
                         const double *A, int lda,
                         const double *B, int ldb,
                         double *C, int ldc)
{
  const double alpha = 1.0;
  const double beta  = 0.0;

  if (ws == NULL || !ws->active || ((double)m*(double)n*(double)k < Krylov_gpu_dgemm_min_flops)){
    char ta = (transa == CUBLAS_OP_N) ? 'N' : 'T';
    char tb = (transb == CUBLAS_OP_N) ? 'N' : 'T';
    F77_NAME(dgemm,DGEMM)(&ta, &tb, &m, &n, &k, (double*)&alpha,
                          (double*)A, &lda, (double*)B, &ldb, (double*)&beta, C, &ldc);
    return;
  }

  Krylov_GPU_EnsureCublas(ws);

  size_t cols_A = (transa == CUBLAS_OP_N) ? (size_t)k : (size_t)m;
  size_t cols_B = (transb == CUBLAS_OP_N) ? (size_t)n : (size_t)k;
  size_t count_A = (size_t)lda*cols_A;
  size_t count_B = (size_t)ldb*cols_B;
  size_t count_C = (size_t)ldc*(size_t)n;

  Krylov_GPU_EnsureDouble(&ws->d_A, &ws->d_A_count, count_A);
  Krylov_GPU_EnsureDouble(&ws->d_B, &ws->d_B_count, count_B);
  Krylov_GPU_EnsureDouble(&ws->d_C, &ws->d_C_count, count_C);

  wait_cudafunc(cudaMemcpyAsync(ws->d_A, A, sizeof(double)*count_A,
                                cudaMemcpyHostToDevice, ws->stream));
  wait_cudafunc(cudaMemcpyAsync(ws->d_B, B, sizeof(double)*count_B,
                                cudaMemcpyHostToDevice, ws->stream));

  wait_cudafunc(openmx_gemmul8Dgemm(ws->cublas, transa, transb, m, n, k,
                                    &alpha, ws->d_A, lda, ws->d_B, ldb, &beta, ws->d_C, ldc));

  wait_cudafunc(cudaMemcpyAsync(C, ws->d_C, sizeof(double)*count_C,
                                cudaMemcpyDeviceToHost, ws->stream));
  wait_cudafunc(cudaStreamSynchronize(ws->stream));
}

static void Krylov_Eigen2(Krylov_GPU_Workspace *ws, double *a, int csize, double *ko, int n, int EVmax)
{
  int i,j;

  if (ws == NULL || !ws->active || n < Krylov_gpu_eigen_min){
    Eigen_lapack2(a,csize,ko,n,EVmax);
    return;
  }

  Krylov_GPU_EnsureGpusolver(ws);

  Krylov_GPU_EnsureHostMatrix(ws,(size_t)n*(size_t)n);
  Krylov_GPU_EnsureDouble(&ws->d_A, &ws->d_A_count, (size_t)n*(size_t)n);
  Krylov_GPU_EnsureDouble(&ws->d_W, &ws->d_W_count, (size_t)n);
  Krylov_GPU_EnsureInfo(&ws->d_info);

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      ws->h_A[i*n+j] = a[(i+1)*csize+j+1];
    }
  }

  wait_cudafunc(cudaMemcpyAsync(ws->d_A, ws->h_A, sizeof(double)*(size_t)n*(size_t)n,
                                cudaMemcpyHostToDevice, ws->stream));

  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  double vl = 0.0;
  double vu = 0.0;
  int64_t h_meig = 0;
  size_t d_work_bytes = 0;
  size_t h_work_bytes = 0;

  if (n == EVmax){
    wait_cudafunc(cusolverDnXsyevd_bufferSize(ws->gpusolver, NULL, jobz, uplo,
                                              n, CUDA_R_64F, ws->d_A, n,
                                              CUDA_R_64F, ws->d_W, CUDA_R_64F,
                                              &d_work_bytes, &h_work_bytes));
    h_meig = EVmax;
  }
  else{
    cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;

    wait_cudafunc(cusolverDnXsyevdx_bufferSize(ws->gpusolver, NULL, jobz, range, uplo,
                                               n, CUDA_R_64F, ws->d_A, n,
                                               &vl, &vu, 1L, EVmax, &h_meig,
                                               CUDA_R_64F, ws->d_W, CUDA_R_64F,
                                               &d_work_bytes, &h_work_bytes));
  }

  Krylov_GPU_EnsureDeviceWork(ws,d_work_bytes);
  Krylov_GPU_EnsureHostWork(ws,h_work_bytes);

  if (n == EVmax){
    wait_cudafunc(cusolverDnXsyevd(ws->gpusolver, NULL, jobz, uplo,
                                   n, CUDA_R_64F, ws->d_A, n,
                                   CUDA_R_64F, ws->d_W, CUDA_R_64F,
                                   ws->d_work, d_work_bytes,
                                   ws->h_work, h_work_bytes, ws->d_info));
  }
  else{
    cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;

    wait_cudafunc(cusolverDnXsyevdx(ws->gpusolver, NULL, jobz, range, uplo,
                                    n, CUDA_R_64F, ws->d_A, n,
                                    &vl, &vu, 1L, EVmax, &h_meig,
                                    CUDA_R_64F, ws->d_W, CUDA_R_64F,
                                    ws->d_work, d_work_bytes,
                                    ws->h_work, h_work_bytes, ws->d_info));
  }

  int32_t info = 0;
  wait_cudafunc(cudaMemcpyAsync(ws->h_A, ws->d_A, sizeof(double)*(size_t)n*(size_t)n,
                                cudaMemcpyDeviceToHost, ws->stream));
  wait_cudafunc(cudaMemcpyAsync(ko+1, ws->d_W, sizeof(double)*(size_t)EVmax,
                                cudaMemcpyDeviceToHost, ws->stream));
  wait_cudafunc(cudaMemcpyAsync(&info, ws->d_info, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, ws->stream));
  wait_cudafunc(cudaStreamSynchronize(ws->stream));

  if (info != 0 || h_meig != (int64_t)EVmax){
    if (info != 0){
      fprintf(stderr,"Krylov: %s failed, info=%d; falling back to LAPACK.\n",
              (n == EVmax) ? "cusolverDnXsyevd" : "cusolverDnXsyevdx",(int)info);
    }
    else{
      fprintf(stderr,"Krylov: cusolverDnXsyevdx returned %lld eigenpairs, expected %d; falling back to LAPACK.\n",
              (long long)h_meig,EVmax);
    }
    Eigen_lapack2(a,csize,ko,n,EVmax);
    return;
  }

  for (i=0; i<EVmax; i++){
    for (j=0; j<n; j++){
      a[i*n+j] = ws->h_A[i*n+j];
    }
  }
}

/*******************************************************
  Device-resident cache of the repacked Krylov bases

  The projection basis Krylov_U is generated only at SCF_iter==1, but the
  per-atom solve used to repack and re-upload it three times per atom in
  every SCF iteration.  The arena keeps the repacked (Msize x Msize3)
  panels on the device across iterations; admission is greedy within
  (free - reserve)/node_ranks so other GPU phases are never evicted.
*******************************************************/

typedef struct {
  int state;      /* 0 = not built, 1 = active, -1 = disabled */
  int matomnum;
  int nspin;
  int *dim_n;     /* [slot] Msize;  0 = not admitted */
  int *dim_m;     /* [slot] Msize3 */
  int *valid;     /* [slot] panel holds this SCF cycle's basis */
  size_t *off;    /* [slot] element offset into the arena */
  double *d_arena;
  size_t arena_elems;
} Krylov_KUCache;

static Krylov_KUCache Krylov_kucache = {0};

static int Krylov_KUCache_Enabled(void)
{
  const char *env = getenv("OPENMX_KRYLOV_GPU_KU_CACHE");
  if (env == NULL || env[0] == '\0') return 1;
  return (atoi(env) != 0);
}

static size_t Krylov_KUCache_ReserveBytes(void)
{
  const char *env = getenv("OPENMX_KRYLOV_GPU_KU_RESERVE_MB");
  unsigned long mib = 4096ul;
  if (env != NULL && env[0] != '\0'){
    char *end = NULL;
    unsigned long parsed = strtoul(env,&end,10);
    if (end != env && *end == '\0') mib = parsed;
  }
  return (size_t)mib*(size_t)1024*(size_t)1024;
}

/* Ranks sharing this node's device(s); conservative on multi-GPU nodes.
   The split is collective, so the first call must be reached uniformly. */
static int Krylov_NodeRanks(void)
{
  static int ranks = 0;

  if (ranks == 0){
    MPI_Comm node_comm = MPI_COMM_NULL;

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_size(node_comm,&ranks);
    MPI_Comm_free(&node_comm);
    if (ranks < 1) ranks = 1;
  }

  return ranks;
}

void Krylov_Release_GPU_KUCache(void)
{
  if (Krylov_kucache.d_arena != NULL){
    wait_cudafunc(cudaFree(Krylov_kucache.d_arena));
  }
  free(Krylov_kucache.dim_n);
  free(Krylov_kucache.dim_m);
  free(Krylov_kucache.valid);
  free(Krylov_kucache.off);
  Krylov_kucache.dim_n = NULL;
  Krylov_kucache.dim_m = NULL;
  Krylov_kucache.valid = NULL;
  Krylov_kucache.off = NULL;
  Krylov_kucache.d_arena = NULL;
  Krylov_kucache.arena_elems = 0;
  Krylov_kucache.matomnum = 0;
  Krylov_kucache.nspin = 0;
  if (Krylov_kucache.state == 1) Krylov_kucache.state = 0;
}

/* Build (or re-validate) the arena.  Collective only through
   Krylov_NodeRanks(); every other decision is rank-local.  Must be called
   before the OpenMP atom loop. */
static void Krylov_KUCache_Prepare(int matomnum, int nspin, const int *Msize, const int *Msize3, int scf_iter)
{
  int rebuild = 0;
  int admitted = 0;
  int spin,Mc_AN;
  size_t nslot,slot,used_elems = 0;
  size_t free_bytes = 0,total_bytes = 0,budget_bytes;
  cudaError_t cuda_status;
  int node_ranks = Krylov_NodeRanks();

  if (Krylov_kucache.state == -1) return;

  if (!Krylov_KUCache_Enabled()){
    Krylov_kucache.state = -1;
    return;
  }

  nslot = (size_t)nspin*(size_t)(matomnum+1);

  if (Krylov_kucache.state == 1){
    if (Krylov_kucache.matomnum != matomnum || Krylov_kucache.nspin != nspin){
      rebuild = 1;
    }
    else{
      for (spin=0; spin<nspin && !rebuild; spin++){
        for (Mc_AN=1; Mc_AN<=matomnum; Mc_AN++){
          slot = (size_t)spin*(size_t)(matomnum+1)+(size_t)Mc_AN;
          if (Krylov_kucache.dim_n[slot] != 0 &&
              (Krylov_kucache.dim_n[slot] != Msize[Mc_AN] ||
               Krylov_kucache.dim_m[slot] != Msize3[Mc_AN])){
            rebuild = 1;
            break;
          }
        }
      }
    }

    if (!rebuild){
      /* the bases are regenerated while SCF_iter==1; drop stale panels */
      if (scf_iter <= 1){
        memset(Krylov_kucache.valid,0,sizeof(int)*nslot);
      }
      return;
    }

    Krylov_Release_GPU_KUCache();
  }

  Krylov_kucache.matomnum = matomnum;
  Krylov_kucache.nspin = nspin;
  Krylov_kucache.dim_n = (int*)malloc(sizeof(int)*nslot);
  Krylov_kucache.dim_m = (int*)malloc(sizeof(int)*nslot);
  Krylov_kucache.valid = (int*)malloc(sizeof(int)*nslot);
  Krylov_kucache.off = (size_t*)malloc(sizeof(size_t)*nslot);
  if (Krylov_kucache.dim_n == NULL || Krylov_kucache.dim_m == NULL ||
      Krylov_kucache.valid == NULL || Krylov_kucache.off == NULL){
    fprintf(stderr,"Krylov: could not allocate the KU cache tables.\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  memset(Krylov_kucache.dim_n,0,sizeof(int)*nslot);
  memset(Krylov_kucache.dim_m,0,sizeof(int)*nslot);
  memset(Krylov_kucache.valid,0,sizeof(int)*nslot);
  memset(Krylov_kucache.off,0,sizeof(size_t)*nslot);

  cuda_status = cudaMemGetInfo(&free_bytes,&total_bytes);
  if (cuda_status != cudaSuccess){
    Krylov_Release_GPU_KUCache();
    Krylov_kucache.state = -1;
    return;
  }

  budget_bytes = Krylov_KUCache_ReserveBytes();
  budget_bytes = (free_bytes > budget_bytes) ? (free_bytes - budget_bytes)/(size_t)node_ranks : 0;

  for (spin=0; spin<nspin; spin++){
    for (Mc_AN=1; Mc_AN<=matomnum; Mc_AN++){
      size_t elems = (size_t)Msize[Mc_AN]*(size_t)Msize3[Mc_AN];
      size_t bytes = elems*sizeof(double);

      slot = (size_t)spin*(size_t)(matomnum+1)+(size_t)Mc_AN;
      if (bytes > budget_bytes || used_elems*sizeof(double) > budget_bytes - bytes) continue;

      Krylov_kucache.dim_n[slot] = Msize[Mc_AN];
      Krylov_kucache.dim_m[slot] = Msize3[Mc_AN];
      Krylov_kucache.off[slot] = used_elems;
      used_elems += elems;
      admitted++;
    }
  }

  if (0 < used_elems){
    cuda_status = cudaMalloc((void**)&Krylov_kucache.d_arena,used_elems*sizeof(double));
    if (cuda_status != cudaSuccess){
      Krylov_kucache.d_arena = NULL;
      Krylov_Release_GPU_KUCache();
      Krylov_kucache.state = -1;
      return;
    }
  }

  Krylov_kucache.arena_elems = used_elems;
  Krylov_kucache.state = 1;

  if (0 < level_stdout && 0 < admitted){
    int myid = -1;

    MPI_Comm_rank(mpi_comm_level1,&myid);
    if (myid == Host_ID){
      printf("<Krylov> GPU KU-cache: rank %d keeps %d of %d basis panels on the device (%.1f MiB)\n",
             myid,admitted,(int)(nslot-(size_t)nspin),
             (double)(used_elems*sizeof(double))/(1024.0*1024.0));
      fflush(stdout);
    }
  }
}

/* Return the device pointer of this atom's repacked basis, uploading it
   through the caller's KU buffer on the first use of the SCF cycle.  Only
   the thread that owns Mc_AN touches its slots, so no locking is needed. */
static const double *Krylov_KUCache_Get(Krylov_GPU_Workspace *ws, int spin, int Mc_AN,
                                        int n, int m3, int msize2, double ***Krylov_U, double *KU)
{
  Krylov_KUCache *kc = &Krylov_kucache;
  size_t slot;
  double *dst;
  int i,j;

  if (kc->state != 1) return NULL;
  if (kc->matomnum < Mc_AN || kc->nspin <= spin) return NULL;

  slot = (size_t)spin*(size_t)(kc->matomnum+1)+(size_t)Mc_AN;
  if (kc->dim_n[slot] != n || kc->dim_m[slot] != m3) return NULL;

  dst = kc->d_arena + kc->off[slot];

  if (!kc->valid[slot]){
    for (i=0; i<n; i++){
      for (j=0; j<m3; j++){
        KU[j*n+i] = Krylov_U[spin][Mc_AN][j*msize2+i+1];
      }
    }
    wait_cudafunc(cudaMemcpyAsync(dst,KU,sizeof(double)*(size_t)n*(size_t)m3,
                                  cudaMemcpyHostToDevice,ws->stream));
    kc->valid[slot] = 1;
  }

  return dst;
}

/* Device-resident eigen solve of the corrected projected Hamiltonian; on
   success the eigenvectors are left in ws->d_A in exactly the layout the
   back-transform GEMM reads.  Returns 0 so the caller can run the LAPACK
   fallback (matching Krylov_Eigen2's behavior). */
static int Krylov_Eigen2_Device(Krylov_GPU_Workspace *ws, double *a, int csize, double *ko, int n)
{
  int i,j;
  int32_t info = 0;
  int64_t h_meig = n;
  size_t d_work_bytes = 0;
  size_t h_work_bytes = 0;
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  Krylov_GPU_EnsureGpusolver(ws);

  Krylov_GPU_EnsureHostMatrix(ws,(size_t)n*(size_t)n);
  Krylov_GPU_EnsureDouble(&ws->d_A, &ws->d_A_count, (size_t)n*(size_t)n);
  Krylov_GPU_EnsureDouble(&ws->d_W, &ws->d_W_count, (size_t)n);
  Krylov_GPU_EnsureInfo(&ws->d_info);

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      ws->h_A[i*n+j] = a[(i+1)*csize+j+1];
    }
  }

  wait_cudafunc(cudaMemcpyAsync(ws->d_A, ws->h_A, sizeof(double)*(size_t)n*(size_t)n,
                                cudaMemcpyHostToDevice, ws->stream));

  wait_cudafunc(cusolverDnXsyevd_bufferSize(ws->gpusolver, NULL, jobz, uplo,
                                            n, CUDA_R_64F, ws->d_A, n,
                                            CUDA_R_64F, ws->d_W, CUDA_R_64F,
                                            &d_work_bytes, &h_work_bytes));

  Krylov_GPU_EnsureDeviceWork(ws,d_work_bytes);
  Krylov_GPU_EnsureHostWork(ws,h_work_bytes);

  wait_cudafunc(cusolverDnXsyevd(ws->gpusolver, NULL, jobz, uplo,
                                 n, CUDA_R_64F, ws->d_A, n,
                                 CUDA_R_64F, ws->d_W, CUDA_R_64F,
                                 ws->d_work, d_work_bytes,
                                 ws->h_work, h_work_bytes, ws->d_info));

  wait_cudafunc(cudaMemcpyAsync(ko+1, ws->d_W, sizeof(double)*(size_t)n,
                                cudaMemcpyDeviceToHost, ws->stream));
  wait_cudafunc(cudaMemcpyAsync(&info, ws->d_info, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, ws->stream));
  wait_cudafunc(cudaStreamSynchronize(ws->stream));

  if (info != 0 || h_meig != (int64_t)n){
    fprintf(stderr,"Krylov: cusolverDnXsyevd failed, info=%d; falling back to LAPACK.\n",(int)info);
    return 0;
  }

  return 1;
}

/* Fused per-atom projected solve: H_DC and the basis are shipped to the
   device once, the intermediates stay resident, and only the small
   projected Hamiltonian and the final eigenvector panel travel back.
   Every GEMM and the eigen dispatch match the per-call path bit for bit.
   Returns 0 when the caller must run the original per-call sequence. */
static int Krylov_ProjectedSolve_GPU(Krylov_GPU_Workspace *ws, int spin, int Mc_AN,
                                     const int *Msize, const int *Msize2, const int *Msize3,
                                     double ***Krylov_U, double ****EC_matrix,
                                     double *H_DC, double *KU, double *C, double *ko)
{
  const double alpha = 1.0;
  const double beta  = 0.0;
  const double *d_KU;
  int n  = Msize[Mc_AN];
  int m3 = Msize3[Mc_AN];
  size_t evec_count = (size_t)m3*(size_t)m3;
  size_t big_count;
  int i,j,m;

  static int profile = -1;
  double t0=0.0,t1=0.0,t2=0.0,t3=0.0,t4=0.0,t5=0.0,t6=0.0,t7=0.0;

  if (ws == NULL || !ws->active || !Krylov_gpu_fused) return 0;

  /* only fuse when every GEMM would have gone to the GPU anyway */
  if ((double)n*(double)m3*(double)n   < Krylov_gpu_dgemm_min_flops) return 0;
  if ((double)m3*(double)m3*(double)n < Krylov_gpu_dgemm_min_flops) return 0;
  if ((double)m3*(double)n*(double)m3 < Krylov_gpu_dgemm_min_flops) return 0;

  if (profile < 0){
    const char *env = getenv("OPENMX_KRYLOV_GPU_PROFILE");
    profile = (env != NULL && env[0] != '\0' && atoi(env) != 0);
  }
  if (profile) dtime(&t0);

  Krylov_GPU_EnsureCublas(ws);

  big_count = (size_t)n*(size_t)n;
  if (big_count < evec_count) big_count = evec_count;
  Krylov_GPU_EnsureDouble(&ws->d_A, &ws->d_A_count, big_count);
  Krylov_GPU_EnsureDouble(&ws->d_C, &ws->d_C_count, (size_t)n*(size_t)m3);

  d_KU = Krylov_KUCache_Get(ws,spin,Mc_AN,n,m3,Msize2[Mc_AN],Krylov_U,KU);
  if (d_KU == NULL){
    for (i=0; i<n; i++){
      for (j=0; j<m3; j++){
        KU[j*n+i] = Krylov_U[spin][Mc_AN][j*Msize2[Mc_AN]+i+1];
      }
    }
    Krylov_GPU_EnsureDouble(&ws->d_B, &ws->d_B_count, (size_t)n*(size_t)m3);
    wait_cudafunc(cudaMemcpyAsync(ws->d_B, KU, sizeof(double)*(size_t)n*(size_t)m3,
                                  cudaMemcpyHostToDevice, ws->stream));
    d_KU = ws->d_B;
  }

  if (profile) dtime(&t1);

  wait_cudafunc(cudaMemcpyAsync(ws->d_A, H_DC, sizeof(double)*(size_t)n*(size_t)n,
                                cudaMemcpyHostToDevice, ws->stream));

  if (profile) dtime(&t2);

  /* C = H_DC * u1 */
  wait_cudafunc(openmx_gemmul8Dgemm(ws->cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m3, n,
                                    &alpha, ws->d_A, n, d_KU, n, &beta, ws->d_C, n));

  if (profile) dtime(&t3);

  /* u1^+ * H_DC * u1 (the projected Hamiltonian replaces H_DC on the host) */
  wait_cudafunc(openmx_gemmul8Dgemm(ws->cublas, CUBLAS_OP_T, CUBLAS_OP_N, m3, m3, n,
                                    &alpha, d_KU, n, ws->d_C, n, &beta, ws->d_A, m3));

  if (profile) dtime(&t4);

  wait_cudafunc(cudaMemcpyAsync(H_DC, ws->d_A, sizeof(double)*evec_count,
                                cudaMemcpyDeviceToHost, ws->stream));
  wait_cudafunc(cudaStreamSynchronize(ws->stream));

  /* correction for ZeroNum */

  m = (int)Krylov_U[spin][Mc_AN][0];
  for (i=0; i<m; i++){
    H_DC[i*m3+i] = 1.0e+3;
  }

  /* H0 = u1^+ * H_DC * u1 + D */

  for (i=(m3-1); 0<=i; i--){
    for (j=0; j<m3; j++){
      H_DC[(i+1)*(m3+1)+(j+1)] = H_DC[i*m3+j] + EC_matrix[spin][Mc_AN][i+1][j+1];
    }
  }

  if (profile) dtime(&t5);

  /* diagonalize; the eigenvector panel ends up in ws->d_A either way */

  if (m3 < Krylov_gpu_eigen_min || !Krylov_Eigen2_Device(ws,H_DC,m3+1,ko,m3)){
    Eigen_lapack2(H_DC,m3+1,ko,m3,m3);
    wait_cudafunc(cudaMemcpyAsync(ws->d_A, H_DC, sizeof(double)*evec_count,
                                  cudaMemcpyHostToDevice, ws->stream));
  }

  if (profile) dtime(&t6);

  /* back transformation of eigenvectors: c = u1 * b */

  wait_cudafunc(openmx_gemmul8Dgemm(ws->cublas, CUBLAS_OP_T, CUBLAS_OP_T, m3, n, m3,
                                    &alpha, ws->d_A, m3, d_KU, n, &beta, ws->d_C, m3));

  wait_cudafunc(cudaMemcpyAsync(C, ws->d_C, sizeof(double)*(size_t)m3*(size_t)n,
                                cudaMemcpyDeviceToHost, ws->stream));
  wait_cudafunc(cudaStreamSynchronize(ws->stream));

  if (profile){
    dtime(&t7);
    if (Mc_AN == 1 && spin == 0){
      printf("KRYFUSE n=%d m3=%d ku=%.4f hup=%.4f g1=%.4f g2=%.4f h33=%.4f eig=%.4f g3=%.4f tot=%.4f\n",
             n,m3,t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6,t7-t0);
      fflush(stdout);
    }
  }

  return 1;
}

double Krylov(char* mode,
    int SCF_iter,
    double***** Hks,
    double***** ImNL,
    double**** OLP0,
    double***** CDM,
    double***** EDM,
    double Eele0[2], double Eele1[2])
{
    double time0;

    /****************************************************
           collinear without spin-orbit coupling
    ****************************************************/

    if ((SpinP_switch == 0 || SpinP_switch == 1) && SO_switch == 0) {

        if (atomnum <= NUMPROCS_MPI_COMM_WORLD && 1 < openmp_threads_num)
            time0 = Krylov_Col_trd(mode, SCF_iter, Hks, OLP0, CDM, EDM, Eele0, Eele1);
        else
            time0 = Krylov_Col(mode, SCF_iter, Hks, OLP0, CDM, EDM, Eele0, Eele1);
    }

    /****************************************************
           collinear with spin-orbit coupling
    ****************************************************/

    else if ((SpinP_switch == 0 || SpinP_switch == 1) && SO_switch == 1) {
        printf("Spin-orbit coupling is not supported for collinear DFT calculations.\n");
        MPI_Finalize();
        exit(1);
    }

    /****************************************************
     non-collinear with and without spin-orbit coupling
    ****************************************************/

    else if (SpinP_switch == 3) {
        printf("The O(N) Krylov subspace method is not supported for non-collinear DFT calculations.\n");
        MPI_Finalize();
        exit(1);
    }

    return time0;
}

static double Krylov_Col(char* mode,
    int SCF_iter,
    double***** Hks,
    double**** OLP0,
    double***** CDM,
    double***** EDM,
    double Eele0[2],
    double Eele1[2])
{
    static int firsttime = 1, recalc_firsttime = 1, recalc_flag;
    int Mc_AN, Gc_AN, i, is, js, Gi, wan, wanA, wanB, Anum;
    int num, NUM0, NUM, NUM1, n2, Cwan, Hwan, Rn2;
    int size1, size2, max_size1, max_size2;
    int ih, ig, ian, j, kl, jg, jan, Bnum, m, n, spin, i2, ip;
    int k, l, i1, j1, P_min, m_size, q1, q2, csize, Residues_size;
    int h_AN1, Mh_AN1, h_AN2, Gh_AN1, Gh_AN2, wan1, wan2;
    int po, po1, loopN, tno1, tno2, h_AN, Gh_AN, rl1, rl2, rl;
    int MA_AN, GA_AN, tnoA, GB_AN, tnoB, ct_on;
    int Msize2_max;
    static double TZ;
    double My_TZ, sum, FermiF, time0, srt;
    double sum00, sum10, sum20, sum30;
    double sum01, sum11, sum21, sum31;
    double sum02, sum12, sum22, sum32;
    double sum03, sum13, sum23, sum33;
    double tmp0, tmp1, tmp2, tmp3, b2, co, x0, Sx, Dx, xmin, xmax;
    double My_Num_State, Num_State, x, Dnum;
    double emin, emax, de;
    double TStime, TEtime, Stime1, Etime1;
    double Stime2, Etime2;
    double time1, time2, time3, time4, time5;
    double time6, time7, time8, time9, time10;
    double time11, time12, time13, time14, time15, time16;
    double Erange;
    double My_Eele0[2], My_Eele1[2];
    double max_x = 30.0;
    double ChemP_MAX, ChemP_MIN, spin_degeneracy;
    double spetrum_radius;
    double*** EVal;
    double****** Residues;
    double*** PDOS_DC;
    double* tmp_array;
    double* tmp_array2;

    int* MP;

    /*****************************************************
    Msize:   \sum_FNAN Spe_Total_CNO
    Msize2:  \sum_FNAN+SNAN Spe_Total_CNO
    Msize3:  rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]
    Msize4:  rlmax_EC2[Mc_AN]*EKC_core_size[Mc_AN]
    Msize5:  dimension for the last column of Residues
    *****************************************************/

    int* Msize;
    int* Msize2;
    int* Msize3;
    int* Msize4;
    int* Msize5;
    int** LO_TC;
    int** HO_TC;
    int numprocs, myid, ID, IDS, IDR, tag = 999;
    double Stime_atom, Etime_atom;

    MPI_Status stat;
    MPI_Request request;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    /* MPI */
    MPI_Barrier(mpi_comm_level1);
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    Krylov_GPU_InitOnce();
    Krylov_GPU_PreparePool(omp_get_max_threads());

    dtime(&TStime);

    if (measure_time == 1) {
        time1 = 0.0;
        time2 = 0.0;
        time3 = 0.0;
        time4 = 0.0;
        time5 = 0.0;
        time6 = 0.0;
        time7 = 0.0;
        time8 = 0.0;
        time9 = 0.0;
        time10 = 0.0;
        time11 = 0.0;
        time12 = 0.0;
        time13 = 0.0;
        time14 = 0.0;
        time15 = 0.0;
    }

    /****************************************************
                     allocation of arrays:
    ****************************************************/

    Msize = (int*)malloc(sizeof(int) * (Matomnum + 1));
    Msize2 = (int*)malloc(sizeof(int) * (Matomnum + 1));
    Msize3 = (int*)malloc(sizeof(int) * (Matomnum + 1));
    Msize4 = (int*)malloc(sizeof(int) * (Matomnum + 1));
    Msize2_max = 0;

    /* find Msize */

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

        if (Mc_AN == 0) {
            Msize[Mc_AN] = 1;
        } else {

            Gc_AN = M2G[Mc_AN];

            NUM = 0;
            for (i = 0; i <= FNAN[Gc_AN]; i++) {
                Gi = natn[Gc_AN][i];
                wanA = WhatSpecies[Gi];
                NUM += Spe_Total_CNO[wanA];
            }
            Msize[Mc_AN] = NUM;
        }
    }

    /* find Msize2 and Msize2_max */

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

        if (Mc_AN == 0) {
            Msize2[Mc_AN] = 1;
        } else {

            Gc_AN = M2G[Mc_AN];

            NUM = 0;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                Gi = natn[Gc_AN][i];
                wanA = WhatSpecies[Gi];
                NUM += Spe_Total_CNO[wanA];
            }
            Msize2[Mc_AN] = NUM;
        }

        if (Msize2_max < Msize2[Mc_AN])
            Msize2_max = Msize2[Mc_AN] + 4;
    }

    /* find Msize3 and Msize4 */

    Msize3[0] = 1;
    Msize4[0] = 1;

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        wan = WhatSpecies[Gc_AN];
        ct_on = Spe_Total_CNO[wan];
        Msize3[Mc_AN] = rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN];
        Msize4[Mc_AN] = rlmax_EC2[Mc_AN] * EKC_core_size[Mc_AN];
    }

    if (Krylov_GPU_Enabled()) {
        Krylov_KUCache_Prepare(Matomnum, SpinP_switch + 1, Msize, Msize3, SCF_iter);
    }

    m_size = 0;

    EVal = (double***)malloc(sizeof(double**) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        EVal[spin] = (double**)malloc(sizeof(double*) * (Matomnum + 1));

        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
            n2 = Msize3[Mc_AN] + 2;
            m_size += n2;
            EVal[spin][Mc_AN] = (double*)malloc(sizeof(double) * n2);
        }
    }

    if (firsttime) {
        PrintMemory("Krylov: EVal", sizeof(double) * m_size, NULL);
    }

    if (2 <= level_stdout) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            printf("<Krylov> myid=%4d Mc_AN=%4d Gc_AN=%4d Msize=%4d\n",
                myid, Mc_AN, M2G[Mc_AN], Msize[Mc_AN]);
        }
    }

    /****************************************************
      allocation of arrays:

      double PDOS[SpinP_switch+1]
                 [Matomnum+1]
                 [n2]
    ****************************************************/

    m_size = 0;
    PDOS_DC = (double***)malloc(sizeof(double**) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        PDOS_DC[spin] = (double**)malloc(sizeof(double*) * (Matomnum + 1));
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
            n2 = Msize3[Mc_AN] + 4;
            m_size += n2;
            PDOS_DC[spin][Mc_AN] = (double*)malloc(sizeof(double) * n2);
        }
    }

    if (firsttime) {
        PrintMemory("Krylov: PDOS_DC", sizeof(double) * m_size, NULL);
    }

    /****************************************************
      allocation of arrays:

      int LO_TC[SpinP_switch+1][Matomnum+1]
      int HO_TC[SpinP_switch+1][Matomnum+1]
    ****************************************************/

    LO_TC = (int**)malloc(sizeof(int*) * (SpinP_switch + 1));
    for (spin = 0; spin < (SpinP_switch + 1); spin++) {
        LO_TC[spin] = (int*)malloc(sizeof(int) * (Matomnum + 1));
    }

    HO_TC = (int**)malloc(sizeof(int*) * (SpinP_switch + 1));
    for (spin = 0; spin < (SpinP_switch + 1); spin++) {
        HO_TC[spin] = (int*)malloc(sizeof(int) * (Matomnum + 1));
    }

    /****************************************************
      allocation of array:

      double Residues[SpinP_switch+1]
                     [Matomnum+1]
                     [FNAN[Gc_AN]+1]
                     [Spe_Total_CNO[Gc_AN]]
                     [Spe_Total_CNO[Gh_AN]]
                     [HO_TC-LO_TC+3]
    ****************************************************/

    Residues = (double******)malloc(sizeof(double*****) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        Residues[spin] = (double*****)malloc(sizeof(double****) * (Matomnum + 1));
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN = 0;
                FNAN[0] = 0;
                tno1 = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];
            }

            Residues[spin][Mc_AN] = (double****)malloc(sizeof(double***) * (FNAN[Gc_AN] + 1));

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB = WhatSpecies[Gh_AN];
                    tno2 = Spe_Total_CNO[wanB];
                }

                Residues[spin][Mc_AN][h_AN] = (double***)malloc(sizeof(double**) * tno1);
                for (i = 0; i < tno1; i++) {
                    Residues[spin][Mc_AN][h_AN][i] = (double**)malloc(sizeof(double*) * tno2);
                    /* note that the array is allocated once more in the loop */

                    /*
                    for (j=0; j<tno2; j++){
                      Residues[spin][Mc_AN][h_AN][i][j] = (double*)malloc(sizeof(double)*4100);
                    }
                    */
                }
            }
        }
    }

    for (spin = 0; spin <= SpinP_switch; spin++) {
        Residues[spin][0][0][0][0] = (double*)malloc(sizeof(double) * 1);
    }

    /****************************************************
       initialize density and energy density matrices
    ****************************************************/

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            Gc_AN = M2G[Mc_AN];
            wanA = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[wanA];
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                Gh_AN = natn[Gc_AN][h_AN];
                wanB = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_CNO[wanB];
                for (i = 0; i < tno1; i++) {
                    for (j = 0; j < tno2; j++) {
                        CDM[spin][Mc_AN][h_AN][i][j] = 0.0;
                        EDM[spin][Mc_AN][h_AN][i][j] = 0.0;
                    }
                }
            }
        }
    }

    /****************************************************
     MPI

     Hks
    ****************************************************/

    if (measure_time == 1)
        dtime(&Stime1);

    if (SCF_iter == 1) {

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
                            Cwan = WhatSpecies[Gc_AN];
                            tno1 = Spe_Total_CNO[Cwan];
                            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                                Gh_AN = natn[Gc_AN][h_AN];
                                Hwan = WhatSpecies[Gh_AN];
                                tno2 = Spe_Total_CNO[Hwan];
                                size1 += tno1 * tno2;
                            }
                        }
                    }

                    Snd_HFS_Size[IDS] = size1;
                    MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                } else {
                    Snd_HFS_Size[IDS] = 0;
                }

                /* receiving of size of data */

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                    MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                    Rcv_HFS_Size[IDR] = size2;
                } else {
                    Rcv_HFS_Size[IDR] = 0;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                    MPI_Wait(&request, &stat);
            } else {
                Snd_HFS_Size[IDS] = 0;
                Rcv_HFS_Size[IDR] = 0;
            }
        }
    }

    /***********************************
               data transfer
    ************************************/

    /* find maximum size of size1 */

    max_size1 = 0;
    max_size2 = 0;
    for (ID = 0; ID < numprocs; ID++) {
        size1 = Snd_HFS_Size[ID];
        if (max_size1 < size1)
            max_size1 = size1;
        size2 = Rcv_HFS_Size[ID];
        if (max_size2 < size2)
            max_size2 = size2;
    }

    /* allocation of arrays */

    tmp_array = (double*)malloc(sizeof(double) * max_size1);
    tmp_array2 = (double*)malloc(sizeof(double) * max_size2);

    /* MPI communication */

    tag = 999;
    for (ID = 0; ID < numprocs; ID++) {

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (ID != 0) {

            /*****************************
                    sending of data
            *****************************/

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                size1 = Snd_HFS_Size[IDS];

                /* multidimentional array to vector array */

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan = WhatSpecies[Gc_AN];
                        tno1 = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan = WhatSpecies[Gh_AN];
                            tno2 = Spe_Total_CNO[Hwan];
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

                size2 = Rcv_HFS_Size[IDR];

                MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
                    for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++) {
                        Mc_AN++;
                        Gc_AN = Rcv_GAN[IDR][n];
                        Cwan = WhatSpecies[Gc_AN];
                        tno1 = Spe_Total_CNO[Cwan];

                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan = WhatSpecies[Gh_AN];
                            tno2 = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    Hks[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }
                }
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                MPI_Wait(&request, &stat);
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

    if (SCF_iter == 1) {

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

                    size1 = Snd_HFS_Size[IDS] / (1 + SpinP_switch);

                    /* multidimentional array to vector array */

                    num = 0;

                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan = WhatSpecies[Gc_AN];
                        tno1 = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan = WhatSpecies[Gh_AN];
                            tno2 = Spe_Total_CNO[Hwan];
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

                    size2 = Rcv_HFS_Size[IDR] / (1 + SpinP_switch);

                    MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                    num = 0;
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
                    for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++) {
                        Mc_AN++;
                        Gc_AN = Rcv_GAN[IDR][n];
                        Cwan = WhatSpecies[Gc_AN];
                        tno1 = Spe_Total_CNO[Cwan];

                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan = WhatSpecies[Gh_AN];
                            tno2 = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    OLP0[Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
                }
            }
        }
    }

    /* freeing of arrays */
    free(tmp_array);
    free(tmp_array2);

    if (measure_time == 1) {
        dtime(&Etime1);
        time1 = Etime1 - Stime1;
    }

    /***********************************************
      for regeneration of the buffer matrix
    ***********************************************/

    if (sqrt(fabs(NormRD[0])) < (0.2 + 0.10 * (double)atomnum) && recalc_firsttime) {
        recalc_flag = 1;
        recalc_firsttime = 0;
    } else {
        recalc_flag = 0;
    }

    if (SCF_iter == 1)
        recalc_firsttime = 1;

    if (error_check == 1) {
        printf("SCF_iter=%2d recalc_firsttime=%2d\n", SCF_iter, recalc_firsttime);
    }

    if (measure_time == 1)
        dtime(&Stime2);

#pragma omp parallel shared(EKC_invS_flag, List_YOUSO, Residues, EDM, CDM, HO_TC, LO_TC, ChemP, EVal, RMI1, S_G2M, EC_matrix, recalc_flag, recalc_EM, Krylov_U, SpinP_switch, EKC_core_size, EKC_core_size_max, rlmax_EC, rlmax_EC2, time11, time10, time9, time8, time7, time6, time5, time4, time3, time2, Hks, OLP0, EKC_Exact_invS_flag, SCF_iter, Msize4, Msize3, Msize2, Msize, Msize2_max, natn, FNAN, SNAN, Spe_Total_CNO, WhatSpecies, M2G, Matomnum, myid, time_per_atom, firsttime)
    {
        int OMPID, Nthrds, Nprocs;
        int Mc_AN, Gc_AN, wan, spin;
        int ig, ian, ih, kl, jg, jan, Bnum, m, n, rl;
        int Anum, i, j, k, Gi, wanA, NUM, n2, csize, is, i2;
        int i1, rl1, js, ip, po1, tno1, h_AN, Gh_AN, wanB, tno2;
        int* MP;
        int KU_d1, KU_d2, lda, ldb, ldc, M, N, K;
        double alpha, beta;
        double** invS;
        double* LoS;
        double *C, *KU;
        double *H_DC, *ko;
        double*** Krylov_U_OLP;
        double** inv_RS;
        double sum00, sum10, sum20, sum30, sum;
        double tmp0, tmp1, tmp2, tmp3;
        double Erange;
        double Stime_atom, Etime_atom;
        double Stime1, Etime1;
        double** tmpvec0;
        double** tmpvec1;
        double** tmpvec2;
        Krylov_GPU_Workspace* gpu_ws;

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();
        gpu_ws = Krylov_GPU_GetWorkspace(OMPID);

        /* allocation of arrays */

        MP = (int*)malloc(sizeof(int) * List_YOUSO[2]);

        tmpvec0 = (double**)malloc(sizeof(double*) * EKC_core_size_max);
        for (i = 0; i < EKC_core_size_max; i++) {
            tmpvec0[i] = (double*)malloc(sizeof(double) * Msize2_max);
        }

        tmpvec1 = (double**)malloc(sizeof(double*) * EKC_core_size_max);
        for (i = 0; i < EKC_core_size_max; i++) {
            tmpvec1[i] = (double*)malloc(sizeof(double) * Msize2_max);
        }

        tmpvec2 = (double**)malloc(sizeof(double*) * EKC_core_size_max);
        for (i = 0; i < EKC_core_size_max; i++) {
            tmpvec2[i] = (double*)malloc(sizeof(double) * Msize2_max);
        }

        if (firsttime && OMPID == 0) {
            PrintMemory("Krylov: tmpvec0", sizeof(double) * EKC_core_size_max * Msize2_max, NULL);
            PrintMemory("Krylov: tmpvec1", sizeof(double) * EKC_core_size_max * Msize2_max, NULL);
            PrintMemory("Krylov: tmpvec2", sizeof(double) * EKC_core_size_max * Msize2_max, NULL);
        }

        /***********************************************
         main loop of calculation
        ***********************************************/

        for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            wan = WhatSpecies[Gc_AN];

            /* MP array */

            Anum = 1;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                MP[i] = Anum;
                Gi = natn[Gc_AN][i];
                wanA = WhatSpecies[Gi];
                Anum += Spe_Total_CNO[wanA];
            }
            NUM = Anum - 1;
            n2 = NUM + 40;

            /***********************************************
                       allocation of arrays:
            ***********************************************/

            if (Msize[Mc_AN] < Msize3[Mc_AN])
                csize = Msize3[Mc_AN] + 40;
            else
                csize = Msize[Mc_AN] + 40;

            KU_d1 = EKC_core_size[Mc_AN] * Msize2[Mc_AN];
            KU_d2 = Msize2[Mc_AN];

            H_DC = (double*)malloc(sizeof(double) * csize * csize);
            ko = (double*)malloc(sizeof(double) * csize);
            C = (double*)malloc(sizeof(double) * csize * csize);
            KU = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 2) * Msize3[Mc_AN]);

            /***********************************************
                calculate the inverse of overlap matrix
            ***********************************************/

            if (SCF_iter == 1 && Msize3[Mc_AN] < Msize2[Mc_AN] && EKC_Exact_invS_flag == 1) {

                LoS = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 3) * (Msize2[Mc_AN] + 3));

                invS = (double**)malloc(sizeof(double*) * (Msize2[Mc_AN] + 3));
                for (i = 0; i < (Msize2[Mc_AN] + 3); i++) {
                    invS[i] = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 3));
                }

                if (measure_time == 1)
                    dtime(&Stime1);

                Inverse_S_by_Cholesky(Mc_AN, OLP0, invS, MP, NUM, LoS);

                if (measure_time == 1 && OMPID == 0) {
                    dtime(&Etime1);
                    time2 += Etime1 - Stime1;
                }
            }

            else if (SCF_iter == 1 && Msize3[Mc_AN] < Msize2[Mc_AN] && EKC_invS_flag == 1) {

                Krylov_U_OLP = (double***)malloc(sizeof(double**) * rlmax_EC2[Mc_AN]);
                for (i = 0; i < rlmax_EC2[Mc_AN]; i++) {
                    Krylov_U_OLP[i] = (double**)malloc(sizeof(double*) * EKC_core_size[Mc_AN]);
                    for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
                        Krylov_U_OLP[i][j] = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 3));
                        for (k = 0; k < (Msize2[Mc_AN] + 3); k++)
                            Krylov_U_OLP[i][j][k] = 0.0;
                    }
                }

                inv_RS = (double**)malloc(sizeof(double*) * (rlmax_EC2[Mc_AN] + 1) * EKC_core_size[Mc_AN]);
                for (i = 0; i < (rlmax_EC2[Mc_AN] + 1) * EKC_core_size[Mc_AN]; i++) {
                    inv_RS[i] = (double*)malloc(sizeof(double) * (rlmax_EC2[Mc_AN] + 1) * EKC_core_size[Mc_AN]);
                }

                if (measure_time == 1)
                    dtime(&Stime1);

                Krylov_IOLP(Mc_AN, OLP0, Krylov_U_OLP, inv_RS, MP, Msize2, Msize4, Msize2_max, tmpvec0, tmpvec1);

                if (measure_time == 1 && OMPID == 0) {
                    dtime(&Etime1);
                    time2 += Etime1 - Stime1;
                }
            }

            for (spin = 0; spin <= SpinP_switch; spin++) {

                /****************************************************
                        generate a preconditioning matrix
                ****************************************************/

                if (measure_time == 1)
                    dtime(&Stime1);

                if (SCF_iter == 1 && Msize3[Mc_AN] < Msize2[Mc_AN]) {

                    Generate_pMatrix(myid, spin, Mc_AN, Hks, OLP0, invS, Krylov_U, Krylov_U_OLP, inv_RS, MP,
                        Msize, Msize2, Msize3, Msize4, Msize2_max, tmpvec0, tmpvec1, tmpvec2);

                } else if (SCF_iter == 1) {
                    Generate_pMatrix2(myid, spin, Mc_AN, Hks, OLP0, Krylov_U, MP, Msize, Msize2, Msize3, tmpvec1);
                }

                if (measure_time == 1 && OMPID == 0) {
                    dtime(&Etime1);
                    time3 += Etime1 - Stime1;
                }

                if (measure_time == 1)
                    dtime(&Stime1);

                if (recalc_EM == 1 || SCF_iter == 1 || recalc_flag == 1) {

                    Embedding_Matrix(spin, Mc_AN, Hks, Krylov_U, EC_matrix, MP, Msize, Msize2, Msize3, tmpvec1);
                }

                if (measure_time == 1 && OMPID == 0) {
                    dtime(&Etime1);
                    time4 += Etime1 - Stime1;
                }

                /****************************************************
                        construct the Hamiltonian matrix
                ****************************************************/

                if (measure_time == 1)
                    dtime(&Stime1);

                for (i = 0; i <= FNAN[Gc_AN]; i++) {
                    ig = natn[Gc_AN][i];
                    ian = Spe_Total_CNO[WhatSpecies[ig]];
                    Anum = MP[i] - 1;
                    ih = S_G2M[ig]; /* S_G2M should be used */

                    for (j = 0; j <= FNAN[Gc_AN]; j++) {

                        kl = RMI1[Mc_AN][i][j];
                        jg = natn[Gc_AN][j];
                        jan = Spe_Total_CNO[WhatSpecies[jg]];
                        Bnum = MP[j] - 1;

                        if (0 <= kl) {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    H_DC[(Anum + m) * Msize[Mc_AN] + Bnum + n] = Hks[spin][ih][kl][m][n];
                                }
                            }
                        }

                        else {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    H_DC[(Anum + m) * Msize[Mc_AN] + Bnum + n] = 0.0;
                                }
                            }
                        }
                    }
                }

                if (measure_time == 1 && OMPID == 0) {
                    dtime(&Etime1);
                    time5 += Etime1 - Stime1;
                }

                /****************************************************
                           transform u1^+ * H_DC * u1
                ****************************************************/

                /* H_DC * u1 */

                if (measure_time == 1)
                    dtime(&Stime1);

                if (Krylov_ProjectedSolve_GPU(gpu_ws, spin, Mc_AN, Msize, Msize2, Msize3,
                                              Krylov_U, EC_matrix, H_DC, KU, C, ko) == 0) {

                /* original version

                for (i=1; i<=Msize[Mc_AN]; i++){
                  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
                    for (n=0; n<EKC_core_size[Mc_AN]; n++){

                      sum = 0.0;
                      for (j=1; j<=Msize[Mc_AN]; j++){
                        sum += H_DC[i][j]*Krylov_U[spin][Mc_AN][rl][n][j];
                      }

                      C[rl*EKC_core_size[Mc_AN]+n+1][i] = sum;
                    }
                  }
                }
                */

                /* BLAS3 version */

                for (i = 0; i < Msize[Mc_AN]; i++) {
                    for (j = 0; j < Msize3[Mc_AN]; j++) {
                        KU[j * Msize[Mc_AN] + i] = Krylov_U[spin][Mc_AN][j * Msize2[Mc_AN] + i + 1];
                    }
                }

                alpha = 1.0;
                beta = 0.0;
                M = Msize[Mc_AN];
                N = Msize3[Mc_AN];
                K = Msize[Mc_AN];
                lda = M;
                ldb = K;
                ldc = Msize[Mc_AN];

                Krylov_Dgemm(gpu_ws, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                             H_DC, lda, KU, ldb, C, ldc);

                if (measure_time == 1 && OMPID == 0) {
                    dtime(&Etime1);
                    time6 += Etime1 - Stime1;
                }

                /* u1^+ * H_DC * u1 */

                if (measure_time == 1)
                    dtime(&Stime1);

                /* original version */

                /*
                for (rl1=0; rl1<rlmax_EC[Mc_AN]; rl1++){
                  for (m=0; m<EKC_core_size[Mc_AN]; m++){
                    for (rl2=rl1; rl2<rlmax_EC[Mc_AN]; rl2++){
                      for (n=0; n<EKC_core_size[Mc_AN]; n++){
                        sum = 0.0;
                        i2 = rl2*EKC_core_size[Mc_AN] + n + 1;
                        for (i=1; i<=Msize[Mc_AN]; i++){
                          sum += Krylov_U[spin][Mc_AN][rl1][m][i]*C[i2][i];
                        }

                        H_DC[rl1*EKC_core_size[Mc_AN]+m+1][rl2*EKC_core_size[Mc_AN]+n+1] = sum;
                        H_DC[rl2*EKC_core_size[Mc_AN]+n+1][rl1*EKC_core_size[Mc_AN]+m+1] = sum;
                      }
                    }
                  }
                }
                */

                /* BLAS3 version */

                alpha = 1.0;
                beta = 0.0;
                M = Msize3[Mc_AN];
                N = Msize3[Mc_AN];
                K = Msize[Mc_AN];
                lda = K;
                ldb = K;
                ldc = Msize3[Mc_AN];

                Krylov_Dgemm(gpu_ws, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                             KU, lda, C, ldb, H_DC, ldc);

                if (measure_time == 1 && OMPID == 0) {
                    dtime(&Etime1);
                    time7 += Etime1 - Stime1;
                }

                /* correction for ZeroNum */

                m = (int)Krylov_U[spin][Mc_AN][0];
                for (i = 0; i < m; i++) {
                    H_DC[i * Msize3[Mc_AN] + i] = 1.0e+3;
                }

                /****************************************************
                    H0 = u1^+ * H_DC * u1 + D
                ****************************************************/

                if (measure_time == 1)
                    dtime(&Stime1);

                for (i = (Msize3[Mc_AN] - 1); 0 <= i; i--) {
                    for (j = 0; j < Msize3[Mc_AN]; j++) {
                        H_DC[(i + 1) * (Msize3[Mc_AN] + 1) + (j + 1)] = H_DC[i * Msize3[Mc_AN] + j] + EC_matrix[spin][Mc_AN][i + 1][j + 1];
                    }
                }

                if (measure_time == 1 && OMPID == 0) {
                    dtime(&Etime1);
                    time8 += Etime1 - Stime1;
                }

                /****************************************************
                   diagonalize
                ****************************************************/

                if (measure_time == 1)
                    dtime(&Stime1);

                Krylov_Eigen2(gpu_ws, H_DC, Msize3[Mc_AN] + 1, ko,
                              Msize3[Mc_AN], Msize3[Mc_AN]);

                if (measure_time == 1 && OMPID == 0) {
                    dtime(&Etime1);
                    time9 += Etime1 - Stime1;
                }

                /********************************************
                   back transformation of eigenvectors
                              c = u1 * b
                *********************************************/

                if (measure_time == 1)
                    dtime(&Stime1);

                /* original version */

                /*
                for (i=1; i<=Msize[Mc_AN]; i++){
                  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
                    for (n=0; n<EKC_core_size[Mc_AN]; n++){

                      tmp1 = Krylov_U[spin][Mc_AN][rl][n][i];
                      i1 = rl*EKC_core_size[Mc_AN] + n + 1;

                      for (j=1; j<=Msize3[Mc_AN]; j++){
                        C[i][j] += tmp1*H_DC[i1][j];
                      }
                    }
                  }
                }
                */

                /* BLAS3 version */

                alpha = 1.0;
                beta = 0.0;
                M = Msize3[Mc_AN];
                N = Msize[Mc_AN];
                K = Msize3[Mc_AN];
                lda = K;
                ldb = N;
                ldc = Msize3[Mc_AN];

                Krylov_Dgemm(gpu_ws, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
                             H_DC, lda, KU, ldb, C, ldc);

                if (measure_time == 1 && OMPID == 0) {
                    dtime(&Etime1);
                    time10 += Etime1 - Stime1;
                }

                } /* fallback of Krylov_ProjectedSolve_GPU */

                if (measure_time == 1)
                    dtime(&Stime1);

                /***********************************************
                  store eigenvalues and residues of poles
                ***********************************************/

                for (i = 1; i <= Msize3[Mc_AN]; i++) {
                    EVal[spin][Mc_AN][i - 1] = ko[i];
                }

                /******************************************************
                set an energy range (-Erange+ChemP to Erange+ChemP)
                of eigenvalues used to store the Residues.
                ******************************************************/

                Erange = 0.367493245; /* in hartree, corresponds to 10 eV */

                /***********************************************
                  find LO_TC and HO_TC
                ***********************************************/

                /* LO_TC */
                i = 0;
                ip = 0;
                po1 = 0;
                do {
                    if ((ChemP - Erange) < EVal[spin][Mc_AN][i]) {
                        ip = i;
                        po1 = 1;
                    }
                    i++;
                } while (po1 == 0 && i < Msize3[Mc_AN]);

                LO_TC[spin][Mc_AN] = ip;

                /* HO_TC */
                i = 0;
                ip = Msize3[Mc_AN] - 1;
                po1 = 0;
                do {
                    if ((ChemP + Erange) < EVal[spin][Mc_AN][i]) {
                        ip = i;
                        po1 = 1;
                    }
                    i++;
                } while (po1 == 0 && i < Msize3[Mc_AN]);

                HO_TC[spin][Mc_AN] = ip;

                /***********************************************
                  store residues of poles
                ***********************************************/

                n2 = HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 3;
                if (n2 < 1)
                    n2 = 1;

                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB = WhatSpecies[Gh_AN];
                    tno2 = Spe_Total_CNO[wanB];
                    Bnum = MP[h_AN] - 1;

                    for (i = 0; i < tno1; i++) {
                        for (j = 0; j < tno2; j++) {

                            for (i1 = 0; i1 < LO_TC[spin][Mc_AN]; i1++) {
                                tmp1 = C[i * Msize3[Mc_AN] + i1] * C[(Bnum + j) * Msize3[Mc_AN] + i1];
                                CDM[spin][Mc_AN][h_AN][i][j] += tmp1;
                                EDM[spin][Mc_AN][h_AN][i][j] += tmp1 * EVal[spin][Mc_AN][i1];
                            }

                            /* <allocation of Residues */
                            Residues[spin][Mc_AN][h_AN][i][j] = (double*)malloc(sizeof(double) * n2);
                            /* allocation of Residues> */

                            for (i1 = LO_TC[spin][Mc_AN]; i1 <= HO_TC[spin][Mc_AN]; i1++) {
                                Residues[spin][Mc_AN][h_AN][i][j][i1 - LO_TC[spin][Mc_AN]]
                                    = C[i * Msize3[Mc_AN] + i1] * C[(Bnum + j) * Msize3[Mc_AN] + i1];
                            }
                        }
                    }
                }

                if (measure_time == 1 && OMPID == 0) {
                    dtime(&Etime1);
                    time11 += Etime1 - Stime1;
                }

            } /* spin */

            /***********************************************
                          freeing of arrays:
            ***********************************************/

            free(H_DC);
            free(ko);
            free(C);
            free(KU);

            if (SCF_iter == 1 && Msize3[Mc_AN] < Msize2[Mc_AN] && EKC_Exact_invS_flag == 1) {

                free(LoS);

                for (i = 0; i < (Msize2[Mc_AN] + 3); i++) {
                    free(invS[i]);
                }
                free(invS);
            }

            else if (SCF_iter == 1 && Msize3[Mc_AN] < Msize2[Mc_AN] && EKC_invS_flag == 1) {

                for (i = 0; i < rlmax_EC2[Mc_AN]; i++) {
                    for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
                        free(Krylov_U_OLP[i][j]);
                    }
                    free(Krylov_U_OLP[i]);
                }
                free(Krylov_U_OLP);

                for (i = 0; i < (rlmax_EC2[Mc_AN] + 1) * EKC_core_size[Mc_AN]; i++) {
                    free(inv_RS[i]);
                }
                free(inv_RS);
            }

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

        } /* Mc_AN */

        /* freeing of array */

        free(MP);

        for (i = 0; i < EKC_core_size_max; i++) {
            free(tmpvec0[i]);
        }
        free(tmpvec0);

        for (i = 0; i < EKC_core_size_max; i++) {
            free(tmpvec1[i]);
        }
        free(tmpvec1);

        for (i = 0; i < EKC_core_size_max; i++) {
            free(tmpvec2[i]);
        }
        free(tmpvec2);

    } /* #pragma omp parallel */

    if (measure_time == 1) {
        dtime(&Etime2);
        time16 = Etime2 - Stime2;
    }

    if (firsttime) {

        Residues_size = 1;

        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
                Gc_AN = M2G[Mc_AN];
                wan = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wan];
                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB = WhatSpecies[Gh_AN];
                    tno2 = Spe_Total_CNO[wanB];
                    n2 = HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 3;
                    Residues_size += tno1 * tno2 * n2;
                }
            }
        }

        PrintMemory("Krylov: Residues", sizeof(double) * Residues_size, NULL);
    }

    /****************************************************
                  calculate the projected DOS
    ****************************************************/

    if (measure_time == 1)
        dtime(&Stime1);

#pragma omp parallel shared(time_per_atom, Residues, LO_TC, HO_TC, EDM, CDM, OLP0, natn, FNAN, PDOS_DC, Msize3, Spe_Total_CNO, WhatSpecies, M2G, SpinP_switch, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, Stime_atom, spin, Gc_AN, wanA, tno1, i1, i, h_AN, Gh_AN, wanB, tno2, j, tmp1, Etime_atom)
    {

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

            dtime(&Stime_atom);

            for (spin = 0; spin <= SpinP_switch; spin++) {

                Gc_AN = M2G[Mc_AN];
                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];

                for (i1 = 0; i1 <= (Msize3[Mc_AN] + 1); i1++) {
                    PDOS_DC[spin][Mc_AN][i1] = 0.0;
                }

                for (i = 0; i < tno1; i++) {
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB = WhatSpecies[Gh_AN];
                        tno2 = Spe_Total_CNO[wanB];
                        for (j = 0; j < tno2; j++) {

                            tmp1 = OLP0[Mc_AN][h_AN][i][j];

                            PDOS_DC[spin][Mc_AN][0] += tmp1 * CDM[spin][Mc_AN][h_AN][i][j];
                            PDOS_DC[spin][Mc_AN][1] += tmp1 * EDM[spin][Mc_AN][h_AN][i][j];

                            for (i1 = 0; i1 < (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1); i1++) {
                                PDOS_DC[spin][Mc_AN][i1 + 2] += Residues[spin][Mc_AN][h_AN][i][j][i1] * tmp1;
                            }
                        }
                    }
                }

            } /* spin */

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

        } /* Mc_AN */

    } /* #pragma omp parallel */

    if (measure_time == 1) {
        dtime(&Etime1);
        time12 += Etime1 - Stime1;
    }

    /****************************************************
             find the total number of electrons
    ****************************************************/

    if (measure_time == 1)
        dtime(&Stime1);

    My_TZ = 0.0;
    for (i = 1; i <= Matomnum; i++) {
        Gc_AN = M2G[i];
        wan = WhatSpecies[Gc_AN];
        My_TZ += Spe_Core_Charge[wan];
    }

    /* MPI, My_TZ */

    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_TZ, &TZ, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /****************************************************
                  find the chemical potential
    ****************************************************/

    po = 0;
    loopN = 0;

    ChemP_MAX = 10.0;
    ChemP_MIN = -10.0;
    if (SpinP_switch == 0)
        spin_degeneracy = 2.0;
    else if (SpinP_switch == 1)
        spin_degeneracy = 1.0;

    do {
        ChemP = 0.50 * (ChemP_MAX + ChemP_MIN);

        My_Num_State = 0.0;

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            for (spin = 0; spin <= SpinP_switch; spin++) {

                dtime(&Stime_atom);

                Gc_AN = M2G[Mc_AN];

                My_Num_State += spin_degeneracy * PDOS_DC[spin][Mc_AN][0];

                for (i = 0; i < (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1); i++) {

                    x = (EVal[spin][Mc_AN][i + LO_TC[spin][Mc_AN]] - ChemP) * Beta;
                    if (x <= -max_x)
                        x = -max_x;
                    if (max_x <= x)
                        x = max_x;
                    FermiF = 1.0 / (1.0 + exp(x));
                    My_Num_State += spin_degeneracy * FermiF * PDOS_DC[spin][Mc_AN][i + 2];
                }

                dtime(&Etime_atom);
                time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
            }
        }

        /* MPI, My_Num_State */

        MPI_Barrier(mpi_comm_level1);
        MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

        Dnum = (TZ - Num_State) - system_charge;
        if (0.0 <= Dnum)
            ChemP_MIN = ChemP;
        else
            ChemP_MAX = ChemP;
        if (fabs(Dnum) < 1.0e-11)
            po = 1;

        if (myid == Host_ID && 3 <= level_stdout) {
            printf("  ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n", ChemP, TZ, Num_State);
        }

        loopN++;
    } while (po == 0 && loopN < 1000);

    if (measure_time == 1) {
        dtime(&Etime1);
        time13 += Etime1 - Stime1;
    }

    /****************************************************
          eigenenergy by summing up eigenvalues
    ****************************************************/

    if (measure_time == 1)
        dtime(&Stime1);

    My_Eele0[0] = 0.0;
    My_Eele0[1] = 0.0;
    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            My_Eele0[spin] += PDOS_DC[spin][Mc_AN][1];

            for (i = 0; i < (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1); i++) {

                x = (EVal[spin][Mc_AN][i + LO_TC[spin][Mc_AN]] - ChemP) * Beta;

                if (x <= -max_x)
                    x = -max_x;
                if (max_x <= x)
                    x = max_x;
                FermiF = 1.0 / (1.0 + exp(x));
                My_Eele0[spin] += FermiF * EVal[spin][Mc_AN][i + LO_TC[spin][Mc_AN]] * PDOS_DC[spin][Mc_AN][i + 2];
            }

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
        }
    }

    /* MPI, My_Eele0 */
    for (spin = 0; spin <= SpinP_switch; spin++) {
        MPI_Barrier(mpi_comm_level1);
        MPI_Allreduce(&My_Eele0[spin], &Eele0[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }

    if (SpinP_switch == 0) {
        Eele0[1] = Eele0[0];
    }

    if (measure_time == 1) {
        dtime(&Etime1);
        time14 += Etime1 - Stime1;
    }

    if (measure_time == 1)
        dtime(&Stime1);

#pragma omp parallel shared(FNAN, time_per_atom, EDM, CDM, Residues, natn, max_x, Beta, ChemP, EVal, LO_TC, HO_TC, Spe_Total_CNO, WhatSpecies, M2G, SpinP_switch, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, spin, Stime_atom, Gc_AN, wanA, tno1, i1, x, FermiF, h_AN, Gh_AN, wanB, tno2, i, j, tmp1, Etime_atom)
    {

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {
            for (spin = 0; spin <= SpinP_switch; spin++) {

                dtime(&Stime_atom);

                Gc_AN = M2G[Mc_AN];
                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];

                for (i1 = 0; i1 < (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1); i1++) {

                    x = (EVal[spin][Mc_AN][i1 + LO_TC[spin][Mc_AN]] - ChemP) * Beta;
                    if (x <= -max_x)
                        x = -max_x;
                    if (max_x <= x)
                        x = max_x;
                    FermiF = 1.0 / (1.0 + exp(x));

                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB = WhatSpecies[Gh_AN];
                        tno2 = Spe_Total_CNO[wanB];
                        for (i = 0; i < tno1; i++) {
                            for (j = 0; j < tno2; j++) {
                                tmp1 = FermiF * Residues[spin][Mc_AN][h_AN][i][j][i1];
                                CDM[spin][Mc_AN][h_AN][i][j] += tmp1;
                                EDM[spin][Mc_AN][h_AN][i][j] += tmp1 * EVal[spin][Mc_AN][i1 + LO_TC[spin][Mc_AN]];
                            }
                        }
                    }
                }

                dtime(&Etime_atom);
                time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
            }
        }

    } /* #pragma omp parallel */

    /****************************************************
                        bond energies
    ****************************************************/

    My_Eele1[0] = 0.0;
    My_Eele1[1] = 0.0;
    for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        GA_AN = M2G[MA_AN];
        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];

        for (j = 0; j <= FNAN[GA_AN]; j++) {
            GB_AN = natn[GA_AN][j];
            wanB = WhatSpecies[GB_AN];
            tnoB = Spe_Total_CNO[wanB];

            for (k = 0; k < tnoA; k++) {
                for (l = 0; l < tnoB; l++) {
                    for (spin = 0; spin <= SpinP_switch; spin++) {
                        My_Eele1[spin] += CDM[spin][MA_AN][j][k][l] * Hks[spin][MA_AN][j][k][l];
                    }
                }
            }
        }
    }

    /* MPI, My_Eele1 */
    MPI_Barrier(mpi_comm_level1);
    for (spin = 0; spin <= SpinP_switch; spin++) {
        MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE,
            MPI_SUM, mpi_comm_level1);
    }

    if (SpinP_switch == 0) {
        Eele1[1] = Eele1[0];
    }

    if (3 <= level_stdout && myid == Host_ID) {
        printf("  Eele00=%15.12f Eele01=%15.12f\n", Eele0[0], Eele0[1]);
        printf("  Eele10=%15.12f Eele11=%15.12f\n", Eele1[0], Eele1[1]);
    }

    if (measure_time == 1) {
        dtime(&Etime1);
        time15 += Etime1 - Stime1;
    }

    if (strcasecmp(mode, "dos") == 0) {
        Save_DOS_Col(Residues, OLP0, EVal, LO_TC, HO_TC);
    }

    if (measure_time == 1) {
        printf("myid=%2d time1 =%5.3f time2 =%5.3f time3 =%5.3f time4 =%5.3f time5 =%5.3f\n",
            myid, time1, time2, time3, time4, time5);
        printf("myid=%2d time6 =%5.3f time7 =%5.3f time8 =%5.3f time9 =%5.3f time10=%5.3f\n",
            myid, time6, time7, time8, time9, time10);
        printf("myid=%2d time11=%5.3f time12=%5.3f time13=%5.3f time14=%5.3f time15=%5.3f\n",
            myid, time11, time12, time13, time14, time15);
        printf("myid=%2d time16=%5.3f\n", myid, time16);
    }

    /****************************************************
      freeing of arrays:

    ****************************************************/

    free(Msize);
    free(Msize2);
    free(Msize3);
    free(Msize4);

    for (spin = 0; spin < (SpinP_switch + 1); spin++) {
        free(LO_TC[spin]);
    }
    free(LO_TC);

    for (spin = 0; spin < (SpinP_switch + 1); spin++) {
        free(HO_TC[spin]);
    }
    free(HO_TC);

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
                Gc_AN = 0;
                FNAN[0] = 0;
                tno1 = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];
            }

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB = WhatSpecies[Gh_AN];
                    tno2 = Spe_Total_CNO[wanB];
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

    if (measure_time == 1) {
        printf("total time=%15.12f\n", time0);
    }

    /* for PrintMemory */
    firsttime = 0;

    return time0;
}

void Generate_pMatrix(int myid, int spin, int Mc_AN, double***** Hks,
    double**** OLP0, double** invS,
    double*** Krylov_U, double*** Krylov_U_OLP, double** inv_RS, int* MP,
    int* Msize, int* Msize2, int* Msize3, int* Msize4, int Msize2_max,
    double** tmpvec0, double** tmpvec1, double** tmpvec2)
{
    int rl, rl0, rl1, ct_AN, fan, san, can, wan, ct_on, i, j;
    int n, Anum, Bnum, k, ian, ih, kl, jg, ig, jan, m, m1, n1;
    int ZeroNum, Gh_AN, wanB, m1s, is;
    int rl00, rl01, rl02, rl03, rl04, rl05, rl06, rl07;
    int mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7;

    int KU_d1, KU_d2, csize;
    __m128d mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5;

    double mmArr[8];
    double time1, time2, time3, time4, time5;
    double time6, time7, time8, time9, time10;
    double Stime1, Etime1;
    double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
    double sum, dum, tmp0, tmp1, tmp2, tmp3, rcutA, r0;
    double **Utmp, **matRS0, **matRS1;
    double** tmpmat0;
    double *ko, *iko;
    double** FS;
    double*** U0;

    ct_AN = M2G[Mc_AN];
    fan = FNAN[ct_AN];
    san = SNAN[ct_AN];
    can = fan + san;
    wan = WhatSpecies[ct_AN];
    ct_on = Spe_Total_CNO[wan];
    rcutA = Spe_Atom_Cut1[wan];

    if (Msize[Mc_AN] < Msize3[Mc_AN])
        csize = Msize3[Mc_AN] + 40;
    else
        csize = Msize[Mc_AN] + 40;

    KU_d1 = EKC_core_size[Mc_AN] * Msize2[Mc_AN];
    KU_d2 = Msize2[Mc_AN];

    /* allocation of arrays */

    Utmp = (double**)malloc(sizeof(double*) * rlmax_EC[Mc_AN]);
    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        Utmp[i] = (double*)malloc(sizeof(double) * EKC_core_size[Mc_AN]);
    }

    U0 = (double***)malloc(sizeof(double**) * rlmax_EC[Mc_AN]);
    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        U0[i] = (double**)malloc(sizeof(double*) * EKC_core_size[Mc_AN]);
        for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
            U0[i][j] = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 3));
            for (k = 0; k < (Msize2[Mc_AN] + 3); k++)
                U0[i][j][k] = 0.0;
        }
    }

    tmpmat0 = (double**)malloc(sizeof(double*) * (EKC_core_size[Mc_AN] + 4));
    for (i = 0; i < (EKC_core_size[Mc_AN] + 4); i++) {
        tmpmat0[i] = (double*)malloc(sizeof(double) * (EKC_core_size[Mc_AN] + 4));
    }

    FS = (double**)malloc(sizeof(double*) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    for (i = 0; i < (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        FS[i] = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    }

    ko = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    iko = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);

    matRS0 = (double**)malloc(sizeof(double*) * (EKC_core_size[Mc_AN] + 2));
    for (i = 0; i < (EKC_core_size[Mc_AN] + 2); i++) {
        matRS0[i] = (double*)malloc(sizeof(double) * (Msize4[Mc_AN] + 3));
    }

    matRS1 = (double**)malloc(sizeof(double*) * (Msize4[Mc_AN] + 3));
    for (i = 0; i < (Msize4[Mc_AN] + 3); i++) {
        matRS1[i] = (double*)malloc(sizeof(double) * (EKC_core_size[Mc_AN] + 2));
    }

    /****************************************************
        initialize
    ****************************************************/

    if (measure_time == 1) {
        time1 = 0.0;
        time2 = 0.0;
        time3 = 0.0;
        time4 = 0.0;
        time5 = 0.0;
        time6 = 0.0;
        time7 = 0.0;
        time8 = 0.0;
        time9 = 0.0;
        time10 = 0.0;
    }

    if (measure_time == 1)
        dtime(&Stime1);

    for (i = 0; i < EKC_core_size_max; i++) {
        for (j = 0; j < Msize2_max; j++) {
            tmpvec0[i][j] = 0.0;
            tmpvec1[i][j] = 0.0;
        }
    }

    /* find the nearest atom with distance of r0 */

    r0 = 1.0e+10;
    for (k = 1; k <= FNAN[ct_AN]; k++) {
        Gh_AN = natn[ct_AN][k];
        wanB = WhatSpecies[Gh_AN];
        if (Dis[ct_AN][k] < r0)
            r0 = Dis[ct_AN][k];
    }

    /* starting vector */

    m = 0;
    for (k = 0; k <= FNAN[ct_AN]; k++) {

        Gh_AN = natn[ct_AN][k];
        wanB = WhatSpecies[Gh_AN];

        if (Dis[ct_AN][k] < (scale_rc_EKC[Mc_AN] * r0)) {

            Anum = MP[k] - 1;

            for (i = 0; i < Spe_Total_CNO[wanB]; i++) {

                tmpvec0[m][Anum + i] = 1.0;

                m++;
            }
        }
    }

    S_orthonormalize_vec(Mc_AN, ct_on, tmpvec0, tmpvec1, OLP0, tmpmat0, ko, iko, MP, Msize2);

    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
        for (i = 0; i < Msize2[Mc_AN]; i++) {
            U0[0][n][i] = tmpvec0[n][i];
        }
    }

    if (measure_time == 1) {
        dtime(&Etime1);
        time1 = Etime1 - Stime1;
    }

    /****************************************************
             generate Krylov subspace vectors
    ****************************************************/

    for (rl = 0; rl < (rlmax_EC[Mc_AN] - 1); rl++) {

        if (measure_time == 1)
            dtime(&Stime1);

        /*******************************************************
                                H * |Wn)
        *******************************************************/

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec1[n][i] = 0.0;
            }
        }

        for (i = 0; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = 0; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j] - 1;

                if (0 <= kl) {

#ifndef __x86_64__

                    /* original version */

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += Hks[spin][ih][kl][m][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec1[n][Anum + m] += sum;
                        }
                    }

#else
                    /* Loop Unrolling + SSE version */

                    for (m = 0; m < (ian - 3); m += 4) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            mmSum00 = _mm_setzero_pd();
                            mmSum01 = _mm_setzero_pd();
                            mmSum10 = _mm_setzero_pd();
                            mmSum11 = _mm_setzero_pd();
                            mmSum20 = _mm_setzero_pd();
                            mmSum21 = _mm_setzero_pd();
                            mmSum30 = _mm_setzero_pd();
                            mmSum31 = _mm_setzero_pd();

                            for (k = 0; k < (jan - 3); k += 4) {
                                mmTmp0 = _mm_loadu_pd(&tmpvec0[n][Bnum + k + 0]);
                                mmTmp1 = _mm_loadu_pd(&tmpvec0[n][Bnum + k + 2]);

                                mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 0]), mmTmp0));
                                mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 2]), mmTmp1));

                                mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 0]), mmTmp0));
                                mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 2]), mmTmp1));

                                mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 0]), mmTmp0));
                                mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 2]), mmTmp1));

                                mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 0]), mmTmp0));
                                mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 2]), mmTmp1));
                            }

                            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                            _mm_storeu_pd(&mmArr[0], mmSum00);
                            _mm_storeu_pd(&mmArr[2], mmSum10);
                            _mm_storeu_pd(&mmArr[4], mmSum20);
                            _mm_storeu_pd(&mmArr[6], mmSum30);

                            sum0 = mmArr[0] + mmArr[1];
                            sum1 = mmArr[2] + mmArr[3];
                            sum2 = mmArr[4] + mmArr[5];
                            sum3 = mmArr[6] + mmArr[7];

                            for (; k < jan; k++) {
                                sum0 += Hks[spin][ih][kl][m + 0][k] * tmpvec0[n][Bnum + k];
                                sum1 += Hks[spin][ih][kl][m + 1][k] * tmpvec0[n][Bnum + k];
                                sum2 += Hks[spin][ih][kl][m + 2][k] * tmpvec0[n][Bnum + k];
                                sum3 += Hks[spin][ih][kl][m + 3][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec1[n][Anum + m + 0] += sum0;
                            tmpvec1[n][Anum + m + 1] += sum1;
                            tmpvec1[n][Anum + m + 2] += sum2;
                            tmpvec1[n][Anum + m + 3] += sum3;
                        }
                    }

                    for (; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += Hks[spin][ih][kl][m][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
#endif
                }
            }
        }

        if (measure_time == 1) {
            dtime(&Etime1);
            time2 += Etime1 - Stime1;
        }

        /*******************************************************
               S^{-1} * H * |Wn)
        *******************************************************/

        if (EKC_Exact_invS_flag == 1) {

            if (measure_time == 1)
                dtime(&Stime1);

#ifndef __x86_64__

            /* original version */
            /*
            for (n=0; n<EKC_core_size[Mc_AN]; n++){
              for (i=0; i<Msize2[Mc_AN]; i++){
                sum = 0.0;
                for (j=0; j<Msize2[Mc_AN]; j++){
                  sum += invS[i][j]*tmpvec1[n][j];
                }
                tmpvec0[n][i] = sum;
              }
            }
            */
            /* unrolling version */

            for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    sum0 = 0.0;
                    sum1 = 0.0;
                    sum2 = 0.0;
                    sum3 = 0.0;

                    for (j = 0; j < Msize2[Mc_AN]; j++) {
                        sum0 += invS[i + 0][j] * tmpvec1[n][j];
                        sum1 += invS[i + 1][j] * tmpvec1[n][j];
                        sum2 += invS[i + 2][j] * tmpvec1[n][j];
                        sum3 += invS[i + 3][j] * tmpvec1[n][j];
                    }

                    tmpvec0[n][i + 0] = sum0;
                    tmpvec0[n][i + 1] = sum1;
                    tmpvec0[n][i + 2] = sum2;
                    tmpvec0[n][i + 3] = sum3;
                }
            }

            is = Msize2[Mc_AN] - Msize2[Mc_AN] % 4;

            for (i = is; i < Msize2[Mc_AN]; i++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (j = 0; j < Msize2[Mc_AN]; j++) {
                        sum += invS[i][j] * tmpvec1[n][j];
                    }
                    tmpvec0[n][i] = sum;
                }
            }

#else
            /* unrolling + SSE version */

            for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    mmSum00 = _mm_setzero_pd();
                    mmSum01 = _mm_setzero_pd();
                    mmSum10 = _mm_setzero_pd();
                    mmSum11 = _mm_setzero_pd();
                    mmSum20 = _mm_setzero_pd();
                    mmSum21 = _mm_setzero_pd();
                    mmSum30 = _mm_setzero_pd();
                    mmSum31 = _mm_setzero_pd();

                    for (j = 0; j < (Msize2[Mc_AN] - 3); j += 4) {
                        mmTmp0 = _mm_loadu_pd(&tmpvec1[n][j + 0]);
                        mmTmp1 = _mm_loadu_pd(&tmpvec1[n][j + 2]);

                        mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&invS[i + 0][j + 0]), mmTmp0));
                        mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&invS[i + 0][j + 2]), mmTmp1));

                        mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&invS[i + 1][j + 0]), mmTmp0));
                        mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&invS[i + 1][j + 2]), mmTmp1));

                        mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&invS[i + 2][j + 0]), mmTmp0));
                        mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&invS[i + 2][j + 2]), mmTmp1));

                        mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&invS[i + 3][j + 0]), mmTmp0));
                        mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&invS[i + 3][j + 2]), mmTmp1));
                    }

                    mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                    mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                    mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                    mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                    _mm_storeu_pd(&mmArr[0], mmSum00);
                    _mm_storeu_pd(&mmArr[2], mmSum10);
                    _mm_storeu_pd(&mmArr[4], mmSum20);
                    _mm_storeu_pd(&mmArr[6], mmSum30);

                    sum0 = mmArr[0] + mmArr[1];
                    sum1 = mmArr[2] + mmArr[3];
                    sum2 = mmArr[4] + mmArr[5];
                    sum3 = mmArr[6] + mmArr[7];

                    for (; j < Msize2[Mc_AN]; j++) {
                        sum0 += invS[i + 0][j] * tmpvec1[n][j];
                        sum1 += invS[i + 1][j] * tmpvec1[n][j];
                        sum2 += invS[i + 2][j] * tmpvec1[n][j];
                        sum3 += invS[i + 3][j] * tmpvec1[n][j];
                    }

                    tmpvec0[n][i + 0] = sum0;
                    tmpvec0[n][i + 1] = sum1;
                    tmpvec0[n][i + 2] = sum2;
                    tmpvec0[n][i + 3] = sum3;
                }
            }

            is = Msize2[Mc_AN] - Msize2[Mc_AN] % 4;

            for (i = is; i < Msize2[Mc_AN]; i++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (j = 0; j < Msize2[Mc_AN]; j++) {
                        sum += invS[i][j] * tmpvec1[n][j];
                    }
                    tmpvec0[n][i] = sum;
                }
            }

#endif

            if (measure_time == 1) {
                dtime(&Etime1);
                time3 += Etime1 - Stime1;
            }
        }

        /*******************************************************
              U * RS^-1 * U^+ * H * |Wn)
        *******************************************************/

        else if (EKC_invS_flag == 1) {

            /*  U^+ * H * |Wn) */

            for (rl0 = 0; rl0 < rlmax_EC2[Mc_AN]; rl0++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    for (m = 0; m < EKC_core_size[Mc_AN]; m++) {

                        sum = 0.0;
                        for (i = 0; i < Msize2[Mc_AN]; i++) {
                            sum += Krylov_U_OLP[rl0][n][i] * tmpvec1[m][i];
                        }

                        /* transpose the later calcualtion */
                        matRS0[m][rl0 * EKC_core_size[Mc_AN] + n] = sum;
                    }
                }
            }

            /*  RS^-1 * U^+ * H * |Wn) */

            for (rl0 = 0; rl0 < rlmax_EC2[Mc_AN]; rl0++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    for (m = 0; m < EKC_core_size[Mc_AN]; m++) {

                        sum = 0.0;
                        for (i = 0; i < Msize4[Mc_AN]; i++) {
                            sum += inv_RS[rl0 * EKC_core_size[Mc_AN] + n][i] * matRS0[m][i];
                        }

                        matRS1[rl0 * EKC_core_size[Mc_AN] + n][m] = sum;
                    }
                }
            }

            /*  U * RS^-1 * U^+ * H * |Wn) */

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    tmpvec0[n][i] = 0.0;
                }
            }

            for (rl0 = 0; rl0 < rlmax_EC2[Mc_AN]; rl0++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                        tmp0 = matRS1[rl0 * EKC_core_size[Mc_AN] + n][m];
                        for (i = 0; i < Msize2[Mc_AN]; i++) {
                            tmpvec0[m][i] += Krylov_U_OLP[rl0][n][i] * tmp0;
                        }
                    }
                }
            }
        }

        else {
            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    tmpvec0[n][i] = tmpvec1[n][i];
                }
            }
        }

        if (measure_time == 1)
            dtime(&Stime1);

        /*************************************************************
         S-orthogonalization by a classical block Gram-Schmidt method
        *************************************************************/

        /* |tmpvec2) = S * |tmpvec0) */

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec2[n][i] = 0.0;
            }
        }

        for (i = 0; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = 0; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j] - 1;

                if (0 <= kl) {

#ifndef __x86_64__

                    /* Original version */

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec2[n][Anum + m] += sum;
                        }
                    }

#else

                    /* Unrolling + SSE version */

                    for (m = 0; m < (ian - 3); m += 4) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            mmSum00 = _mm_setzero_pd();
                            mmSum01 = _mm_setzero_pd();
                            mmSum10 = _mm_setzero_pd();
                            mmSum11 = _mm_setzero_pd();
                            mmSum20 = _mm_setzero_pd();
                            mmSum21 = _mm_setzero_pd();
                            mmSum30 = _mm_setzero_pd();
                            mmSum31 = _mm_setzero_pd();

                            for (k = 0; k < (jan - 3); k += 4) {
                                mmTmp0 = _mm_loadu_pd(&tmpvec0[n][Bnum + k + 0]);
                                mmTmp1 = _mm_loadu_pd(&tmpvec0[n][Bnum + k + 2]);

                                mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 0]), mmTmp0));
                                mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 2]), mmTmp1));

                                mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 0]), mmTmp0));
                                mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 2]), mmTmp1));

                                mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 0]), mmTmp0));
                                mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 2]), mmTmp1));

                                mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 0]), mmTmp0));
                                mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 2]), mmTmp1));
                            }

                            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                            _mm_storeu_pd(&mmArr[0], mmSum00);
                            _mm_storeu_pd(&mmArr[2], mmSum10);
                            _mm_storeu_pd(&mmArr[4], mmSum20);
                            _mm_storeu_pd(&mmArr[6], mmSum30);

                            sum0 = mmArr[0] + mmArr[1];
                            sum1 = mmArr[2] + mmArr[3];
                            sum2 = mmArr[4] + mmArr[5];
                            sum3 = mmArr[6] + mmArr[7];

                            for (; k < jan; k++) {
                                sum0 += OLP0[ih][kl][m + 0][k] * tmpvec0[n][Bnum + k];
                                sum1 += OLP0[ih][kl][m + 1][k] * tmpvec0[n][Bnum + k];
                                sum2 += OLP0[ih][kl][m + 2][k] * tmpvec0[n][Bnum + k];
                                sum3 += OLP0[ih][kl][m + 3][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec2[n][Anum + m + 0] += sum0;
                            tmpvec2[n][Anum + m + 1] += sum1;
                            tmpvec2[n][Anum + m + 2] += sum2;
                            tmpvec2[n][Anum + m + 3] += sum3;
                        }
                    }

                    for (; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec2[n][Anum + m] += sum;
                        }
                    }

#endif
                }
            }
        }

        if (measure_time == 1) {
            dtime(&Etime1);
            time4 += Etime1 - Stime1;
        }

        if (measure_time == 1)
            dtime(&Stime1);

#ifndef __x86_64__

        /* Original version */

        for (rl0 = 0; rl0 <= rl; rl0++) {

            /* (U_rl0|tmpvec2) */

            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += U0[rl0][m][i] * tmpvec2[n][i];
                    }
                    tmpmat0[m][n] = sum;
                }
            }

            /* |tmpvec0) - |U_rl0) * (U_rl0|tmpvec2) */

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (k = 0; k < EKC_core_size[Mc_AN]; k++) {
                    dum = tmpmat0[k][n];
                    for (i = 0; i < Msize2[Mc_AN]; i++)
                        tmpvec0[n][i] -= U0[rl0][k][i] * dum;
                }
            }
        }

#else

        /* Unrolling + SSE version */

        for (rl0 = 0; rl0 <= rl; rl0++) {

            /* (U_rl0|tmpvec2) */

            for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    mmSum00 = _mm_setzero_pd();
                    mmSum01 = _mm_setzero_pd();
                    mmSum10 = _mm_setzero_pd();
                    mmSum11 = _mm_setzero_pd();
                    mmSum20 = _mm_setzero_pd();
                    mmSum21 = _mm_setzero_pd();
                    mmSum30 = _mm_setzero_pd();
                    mmSum31 = _mm_setzero_pd();

                    for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                        mmTmp0 = _mm_loadu_pd(&tmpvec2[n][i + 0]);
                        mmTmp1 = _mm_loadu_pd(&tmpvec2[n][i + 2]);

                        mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 0]), mmTmp0));
                        mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 2]), mmTmp1));

                        mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 0]), mmTmp0));
                        mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 2]), mmTmp1));

                        mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 0]), mmTmp0));
                        mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 2]), mmTmp1));

                        mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 0]), mmTmp0));
                        mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 2]), mmTmp1));
                    }

                    mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                    mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                    mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                    mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                    _mm_storeu_pd(&mmArr[0], mmSum00);
                    _mm_storeu_pd(&mmArr[2], mmSum10);
                    _mm_storeu_pd(&mmArr[4], mmSum20);
                    _mm_storeu_pd(&mmArr[6], mmSum30);

                    sum0 = mmArr[0] + mmArr[1];
                    sum1 = mmArr[2] + mmArr[3];
                    sum2 = mmArr[4] + mmArr[5];
                    sum3 = mmArr[6] + mmArr[7];

                    for (; i < Msize2[Mc_AN]; i++) {
                        sum0 += U0[rl0][m + 0][i] * tmpvec2[n][i];
                        sum1 += U0[rl0][m + 1][i] * tmpvec2[n][i];
                        sum2 += U0[rl0][m + 2][i] * tmpvec2[n][i];
                        sum3 += U0[rl0][m + 3][i] * tmpvec2[n][i];
                    }

                    tmpmat0[m + 0][n] = sum0;
                    tmpmat0[m + 1][n] = sum1;
                    tmpmat0[m + 2][n] = sum2;
                    tmpmat0[m + 3][n] = sum3;
                }
            }

            for (; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += U0[rl0][m][i] * tmpvec2[n][i];
                    }
                    tmpmat0[m][n] = sum;
                }
            }

            /* |tmpvec0) - |U_rl0) * (U_rl0|tmpvec2) */

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (k = 0; k < EKC_core_size[Mc_AN]; k++) {
                    dum = tmpmat0[k][n];
                    for (i = 0; i < Msize2[Mc_AN]; i++)
                        tmpvec0[n][i] -= U0[rl0][k][i] * dum;
                }
            }
        }

#endif

        if (measure_time == 1) {
            dtime(&Etime1);
            time5 += Etime1 - Stime1;
        }

        /*************************************************************
                       S-orthonormalization of tmpvec0
        *************************************************************/

        if (measure_time == 1)
            dtime(&Stime1);

        S_orthonormalize_vec(Mc_AN, ct_on, tmpvec0, tmpvec1, OLP0, tmpmat0, ko, iko, MP, Msize2);

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                U0[rl + 1][n][i] = tmpvec0[n][i];
            }
        }

        if (measure_time == 1) {
            dtime(&Etime1);
            time6 += Etime1 - Stime1;
        }

    } /* rl */

    /************************************************************
                orthogonalization by diagonalization
    ************************************************************/

    if (measure_time == 1)
        dtime(&Stime1);

    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

        /*  S * |Vn) */

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec1[n][i] = 0.0;
            }
        }

        for (i = 0; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = 0; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j] - 1;

                if (0 <= kl) {

#ifndef __x86_64__

                    /* Original version */
                    /**/
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                            }
                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
                    /**/

#else

                    /* Unrolling + SSE version */
                    /**/
                    for (m = 0; m < (ian - 3); m += 4) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            mmSum00 = _mm_setzero_pd();
                            mmSum01 = _mm_setzero_pd();
                            mmSum10 = _mm_setzero_pd();
                            mmSum11 = _mm_setzero_pd();
                            mmSum20 = _mm_setzero_pd();
                            mmSum21 = _mm_setzero_pd();
                            mmSum30 = _mm_setzero_pd();
                            mmSum31 = _mm_setzero_pd();

                            for (k = 0; k < (jan - 3); k += 4) {
                                mmTmp0 = _mm_loadu_pd(&U0[rl][n][Bnum + k + 0]);
                                mmTmp1 = _mm_loadu_pd(&U0[rl][n][Bnum + k + 2]);

                                mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 0]), mmTmp0));
                                mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 2]), mmTmp1));

                                mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 0]), mmTmp0));
                                mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 2]), mmTmp1));

                                mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 0]), mmTmp0));
                                mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 2]), mmTmp1));

                                mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 0]), mmTmp0));
                                mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 2]), mmTmp1));
                            }

                            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                            _mm_storeu_pd(&mmArr[0], mmSum00);
                            _mm_storeu_pd(&mmArr[2], mmSum10);
                            _mm_storeu_pd(&mmArr[4], mmSum20);
                            _mm_storeu_pd(&mmArr[6], mmSum30);

                            sum0 = mmArr[0] + mmArr[1];
                            sum1 = mmArr[2] + mmArr[3];
                            sum2 = mmArr[4] + mmArr[5];
                            sum3 = mmArr[6] + mmArr[7];

                            for (; k < jan; k++) {
                                sum0 += OLP0[ih][kl][m + 0][k] * U0[rl][n][Bnum + k];
                                sum1 += OLP0[ih][kl][m + 1][k] * U0[rl][n][Bnum + k];
                                sum2 += OLP0[ih][kl][m + 2][k] * U0[rl][n][Bnum + k];
                                sum3 += OLP0[ih][kl][m + 3][k] * U0[rl][n][Bnum + k];
                            }

                            tmpvec1[n][Anum + m + 0] += sum0;
                            tmpvec1[n][Anum + m + 1] += sum1;
                            tmpvec1[n][Anum + m + 2] += sum2;
                            tmpvec1[n][Anum + m + 3] += sum3;
                        }
                    }

                    for (; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                            }
                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
                    /**/
#endif
                }
            }
        }

#ifndef __x86_64__

        /* Original version */
        /**/
        for (rl0 = rl; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += U0[rl0][m][i] * tmpvec1[n][i];
                    }
                    FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum;
                }
            }
        }
        /**/

#else

        /* Unrolling + SSE version */
        /**/
        for (rl0 = rl; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    mmSum00 = _mm_setzero_pd();
                    mmSum01 = _mm_setzero_pd();
                    mmSum10 = _mm_setzero_pd();
                    mmSum11 = _mm_setzero_pd();
                    mmSum20 = _mm_setzero_pd();
                    mmSum21 = _mm_setzero_pd();
                    mmSum30 = _mm_setzero_pd();
                    mmSum31 = _mm_setzero_pd();

                    for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                        mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i + 0]);
                        mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i + 2]);

                        mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 0]), mmTmp0));
                        mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 2]), mmTmp1));

                        mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 0]), mmTmp0));
                        mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 2]), mmTmp1));

                        mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 0]), mmTmp0));
                        mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 2]), mmTmp1));

                        mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 0]), mmTmp0));
                        mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 2]), mmTmp1));
                    }

                    mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                    mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                    mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                    mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                    _mm_storeu_pd(&mmArr[0], mmSum00);
                    _mm_storeu_pd(&mmArr[2], mmSum10);
                    _mm_storeu_pd(&mmArr[4], mmSum20);
                    _mm_storeu_pd(&mmArr[6], mmSum30);

                    sum0 = mmArr[0] + mmArr[1];
                    sum1 = mmArr[2] + mmArr[3];
                    sum2 = mmArr[4] + mmArr[5];
                    sum3 = mmArr[6] + mmArr[7];

                    for (; i < Msize2[Mc_AN]; i++) {
                        sum0 += U0[rl0][m + 0][i] * tmpvec1[n][i];
                        sum1 += U0[rl0][m + 1][i] * tmpvec1[n][i];
                        sum2 += U0[rl0][m + 2][i] * tmpvec1[n][i];
                        sum3 += U0[rl0][m + 3][i] * tmpvec1[n][i];
                    }

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum0;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum0;

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 2][rl * EKC_core_size[Mc_AN] + n + 1] = sum1;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 2] = sum1;

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 3][rl * EKC_core_size[Mc_AN] + n + 1] = sum2;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 3] = sum2;

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 4][rl * EKC_core_size[Mc_AN] + n + 1] = sum3;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 4] = sum3;
                }
            }

            for (; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += U0[rl0][m][i] * tmpvec1[n][i];
                    }
                    FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum;
                }
            }
        }
        /**/

#endif
    }

    if (measure_time == 1) {
        dtime(&Etime1);
        time7 += Etime1 - Stime1;
    }

    if (measure_time == 1)
        dtime(&Stime1);

    Eigen_lapack(FS, ko, Msize3[Mc_AN], Msize3[Mc_AN]);

    if (measure_time == 1) {
        dtime(&Etime1);
        time8 += Etime1 - Stime1;
    }

    ZeroNum = 0;

    for (i = 1; i <= Msize3[Mc_AN]; i++) {

        if (error_check == 1) {
            printf("spin=%2d Mc_AN=%2d i=%3d ko[i]=%18.15f\n", spin, Mc_AN, i, ko[i]);
        }

        if (cutoff_value < ko[i]) {
            ko[i] = sqrt(fabs(ko[i]));
            iko[i] = 1.0 / ko[i];
        } else {
            ZeroNum++;
            ko[i] = 0.0;
            iko[i] = 0.0;
        }
    }

    if (error_check == 1) {
        printf("spin=%2d Mc_AN=%2d ZeroNum=%2d\n", spin, Mc_AN, ZeroNum);
    }

    for (i = 1; i <= Msize3[Mc_AN]; i++) {
        for (j = 1; j <= Msize3[Mc_AN]; j++) {
            FS[i][j] = FS[i][j] * iko[j];
        }
    }

    /* transpose for later calculation */
    for (i = 1; i <= Msize3[Mc_AN]; i++) {
        for (j = i + 1; j <= Msize3[Mc_AN]; j++) {
            tmp1 = FS[i][j];
            tmp2 = FS[j][i];
            FS[i][j] = tmp2;
            FS[j][i] = tmp1;
        }
    }

    if (measure_time == 1)
        dtime(&Stime1);

        /* U0 * U * lamda^{-1/2} */

#ifndef __x86_64__

    /* original version */
    /*
    for (i=0; i<Msize2[Mc_AN]; i++){
      for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
        for (m=0; m<EKC_core_size[Mc_AN]; m++){

          m1 = rl0*EKC_core_size[Mc_AN] + m + 1;

          sum = 0.0;
          for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

            n1 = rl*EKC_core_size[Mc_AN] + 1;

            for (n=0; n<EKC_core_size[Mc_AN]; n++){
              sum += U0[rl][n][i]*FS[m1][n1+n];
            }
          }

          Utmp[rl0][m] = sum;
        }
      }

      for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
        for (m=0; m<EKC_core_size[Mc_AN]; m++){
          U0[rl0][m][i] = Utmp[rl0][m];
        }
      }
    }
    */

    /* unrolling version */

    for (i = 0; i < Msize2[Mc_AN]; i++) {

        for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                Utmp[rl][n] = U0[rl][n][i];
            }
        }

        for (m1 = 1; m1 <= (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - 3); m1 += 4) {

            rl00 = (m1 + 0 - 1) / EKC_core_size[Mc_AN];
            rl01 = (m1 + 1 - 1) / EKC_core_size[Mc_AN];
            rl02 = (m1 + 2 - 1) / EKC_core_size[Mc_AN];
            rl03 = (m1 + 3 - 1) / EKC_core_size[Mc_AN];

            mm0 = (m1 + 0 - 1) % EKC_core_size[Mc_AN];
            mm1 = (m1 + 1 - 1) % EKC_core_size[Mc_AN];
            mm2 = (m1 + 2 - 1) % EKC_core_size[Mc_AN];
            mm3 = (m1 + 3 - 1) % EKC_core_size[Mc_AN];

            sum0 = 0.0;
            sum1 = 0.0;
            sum2 = 0.0;
            sum3 = 0.0;

            for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                n1 = rl * EKC_core_size[Mc_AN] + 1;

                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum0 += Utmp[rl][n] * FS[m1 + 0][n1 + n];
                    sum1 += Utmp[rl][n] * FS[m1 + 1][n1 + n];
                    sum2 += Utmp[rl][n] * FS[m1 + 2][n1 + n];
                    sum3 += Utmp[rl][n] * FS[m1 + 3][n1 + n];
                }
            }

            U0[rl00][mm0][i] = sum0;
            U0[rl01][mm1][i] = sum1;
            U0[rl02][mm2][i] = sum2;
            U0[rl03][mm3][i] = sum3;
        }

        m1s = rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]) % 4 + 1;

        for (m1 = m1s; m1 <= rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]; m1++) {

            rl0 = (m1 - 1) / EKC_core_size[Mc_AN];
            m = (m1 - 1) % EKC_core_size[Mc_AN];

            sum = 0.0;

            for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                n1 = rl * EKC_core_size[Mc_AN] + 1;

                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum += Utmp[rl][n] * FS[m1][n1 + n];
                }
            }

            U0[rl0][m][i] = sum;
        }
    } /* i */

#else

    /* Unrolling + SSE version */
    /**/

    for (i = 0; i < Msize2[Mc_AN]; i++) {

        for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                Utmp[rl][n] = U0[rl][n][i];
            }
        }

        for (m1 = 1; m1 <= (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - 3); m1 += 4) {

            rl00 = (m1 + 0 - 1) / EKC_core_size[Mc_AN];
            rl01 = (m1 + 1 - 1) / EKC_core_size[Mc_AN];
            rl02 = (m1 + 2 - 1) / EKC_core_size[Mc_AN];
            rl03 = (m1 + 3 - 1) / EKC_core_size[Mc_AN];

            mm0 = (m1 + 0 - 1) % EKC_core_size[Mc_AN];
            mm1 = (m1 + 1 - 1) % EKC_core_size[Mc_AN];
            mm2 = (m1 + 2 - 1) % EKC_core_size[Mc_AN];
            mm3 = (m1 + 3 - 1) % EKC_core_size[Mc_AN];

            sum0 = 0.0;
            sum1 = 0.0;
            sum2 = 0.0;
            sum3 = 0.0;

            mmSum00 = _mm_setzero_pd();
            mmSum01 = _mm_setzero_pd();
            mmSum10 = _mm_setzero_pd();
            mmSum11 = _mm_setzero_pd();
            mmSum20 = _mm_setzero_pd();
            mmSum21 = _mm_setzero_pd();
            mmSum30 = _mm_setzero_pd();
            mmSum31 = _mm_setzero_pd();

            for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                n1 = rl * EKC_core_size[Mc_AN] + 1;

                if (0) {
                    mmTmp0 = _mm_loadu_pd(&Utmp[rl][n + 0]);
                    mmTmp1 = _mm_loadu_pd(&Utmp[rl][n + 2]);
                }

                for (n = 0; n < (EKC_core_size[Mc_AN] - 3); n += 4) {
                    mmTmp0 = _mm_loadu_pd(&Utmp[rl][n + 0]);
                    mmTmp1 = _mm_loadu_pd(&Utmp[rl][n + 2]);

                    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 0][n1 + n + 0]), mmTmp0));
                    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 0][n1 + n + 2]), mmTmp1));

                    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 1][n1 + n + 0]), mmTmp0));
                    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 1][n1 + n + 2]), mmTmp1));

                    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 2][n1 + n + 0]), mmTmp0));
                    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 2][n1 + n + 2]), mmTmp1));

                    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 3][n1 + n + 0]), mmTmp0));
                    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 3][n1 + n + 2]), mmTmp1));
                }

                for (; n < EKC_core_size[Mc_AN]; n++) {
                    sum0 += Utmp[rl][n] * FS[m1 + 0][n1 + n];
                    sum1 += Utmp[rl][n] * FS[m1 + 1][n1 + n];
                    sum2 += Utmp[rl][n] * FS[m1 + 2][n1 + n];
                    sum3 += Utmp[rl][n] * FS[m1 + 3][n1 + n];
                }
            }

            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

            _mm_storeu_pd(&mmArr[0], mmSum00);
            _mm_storeu_pd(&mmArr[2], mmSum10);
            _mm_storeu_pd(&mmArr[4], mmSum20);
            _mm_storeu_pd(&mmArr[6], mmSum30);

            sum0 += mmArr[0] + mmArr[1];
            sum1 += mmArr[2] + mmArr[3];
            sum2 += mmArr[4] + mmArr[5];
            sum3 += mmArr[6] + mmArr[7];

            U0[rl00][mm0][i] = sum0;
            U0[rl01][mm1][i] = sum1;
            U0[rl02][mm2][i] = sum2;
            U0[rl03][mm3][i] = sum3;
        }

        m1s = rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]) % 4 + 1;

        for (m1 = m1s; m1 <= rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]; m1++) {

            rl0 = (m1 - 1) / EKC_core_size[Mc_AN];
            m = (m1 - 1) % EKC_core_size[Mc_AN];

            sum = 0.0;

            for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                n1 = rl * EKC_core_size[Mc_AN] + 1;

                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum += Utmp[rl][n] * FS[m1][n1 + n];
                }
            }

            U0[rl0][m][i] = sum;
        }
    } /* i */

#endif

    Krylov_U[spin][Mc_AN][0] = ZeroNum;

    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + i + 1] = U0[rl][n][i];
            }
        }
    }

    if (measure_time == 1) {
        dtime(&Etime1);
        time9 += Etime1 - Stime1;
    }

    /************************************************************
           check the orthonormality of Krylov vectors
    ************************************************************/

    if (error_check == 1) {

        for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    tmpvec1[n][i] = 0.0;
                }
            }

            for (i = 0; i <= can; i++) {

                ig = natn[ct_AN][i];
                ian = Spe_Total_CNO[WhatSpecies[ig]];
                Anum = MP[i] - 1;
                ih = S_G2M[ig];

                for (j = 0; j <= can; j++) {

                    kl = RMI1[Mc_AN][i][j];
                    jg = natn[ct_AN][j];
                    jan = Spe_Total_CNO[WhatSpecies[jg]];
                    Bnum = MP[j] - 1;

                    if (0 <= kl) {

                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                sum = 0.0;
                                for (k = 0; k < jan; k++) {
                                    sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                                }

                                tmpvec1[n][Anum + m] += sum;
                            }
                        }
                    }
                }
            }

            for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum = 0.0;
                        for (i = 0; i < Msize2[Mc_AN]; i++) {
                            sum += U0[rl0][m][i] * tmpvec1[n][i];
                        }

                        if (rl == rl0 && m == n) {
                            if (1.0e-10 < fabs(sum - 1.0)) {
                                printf("A spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
                                    spin, Mc_AN, rl, rl0, m, n, sum);
                            }
                        } else {
                            if (1.0e-10 < fabs(sum)) {
                                printf("B spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
                                    spin, Mc_AN, rl, rl0, m, n, sum);
                            }
                        }
                    }
                }
            }
        }
    }

    if (measure_time == 1) {
        printf("pMatrix myid=%2d time1 =%5.3f time2 =%5.3f time3 =%5.3f time4 =%5.3f\n",
            myid, time1, time2, time3, time4);
        printf("pMatrix myid=%2d time5 =%5.3f time6 =%5.3f time7 =%5.3f time8 =%5.3f\n",
            myid, time5, time6, time7, time8);
        printf("pMatrix myid=%2d time9 =%5.3f\n", myid, time9);
    }

    /* freeing of arrays */

    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        free(Utmp[i]);
    }
    free(Utmp);

    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
            free(U0[i][j]);
        }
        free(U0[i]);
    }
    free(U0);

    for (i = 0; i < (EKC_core_size[Mc_AN] + 4); i++) {
        free(tmpmat0[i]);
    }
    free(tmpmat0);

    for (i = 0; i < (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        free(FS[i]);
    }
    free(FS);

    free(ko);
    free(iko);

    for (i = 0; i < (EKC_core_size[Mc_AN] + 2); i++) {
        free(matRS0[i]);
    }
    free(matRS0);

    for (i = 0; i < (Msize4[Mc_AN] + 3); i++) {
        free(matRS1[i]);
    }
    free(matRS1);
}

void Generate_pMatrix2(int myid, int spin, int Mc_AN, double***** Hks, double**** OLP0,
    double*** Krylov_U, int* MP, int* Msize, int* Msize2, int* Msize3,
    double** tmpvec1)
{
    int rl, rl0, rl1, ct_AN, fan, san, can, wan, ct_on, i, j;
    int n, Anum, Bnum, k, ian, ih, kl, jg, ig, jan, m, m1, n1;
    int ZeroNum, rl_half;
    int KU_d1, KU_d2, csize;
    double sum, dum, tmp0, tmp1, tmp2, tmp3;
    double** Utmp;
    double *ko, *iko;
    double** FS;
    double*** U0;

    int rl00, rl01, rl02, rl03, rl04, rl05, rl06, rl07;
    int mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7, m1s;
    __m128d mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5;

    double mmArr[8];
    double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;

    ct_AN = M2G[Mc_AN];
    fan = FNAN[ct_AN];
    san = SNAN[ct_AN];
    can = fan + san;
    wan = WhatSpecies[ct_AN];
    ct_on = Spe_Total_CNO[wan];

    if (Msize[Mc_AN] < Msize3[Mc_AN])
        csize = Msize3[Mc_AN] + 40;
    else
        csize = Msize[Mc_AN] + 40;

    KU_d1 = EKC_core_size[Mc_AN] * Msize2[Mc_AN];
    KU_d2 = Msize2[Mc_AN];

    /* allocation of arrays */

    Utmp = (double**)malloc(sizeof(double*) * rlmax_EC[Mc_AN]);
    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        Utmp[i] = (double*)malloc(sizeof(double) * EKC_core_size[Mc_AN]);
    }

    U0 = (double***)malloc(sizeof(double**) * rlmax_EC[Mc_AN]);
    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        U0[i] = (double**)malloc(sizeof(double*) * EKC_core_size[Mc_AN]);
        for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
            U0[i][j] = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 3));
            for (k = 0; k < (Msize2[Mc_AN] + 3); k++)
                U0[i][j][k] = 0.0;
        }
    }

    FS = (double**)malloc(sizeof(double*) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    for (i = 0; i < (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        FS[i] = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    }

    ko = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    iko = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);

    /****************************************************
     initialize
    ****************************************************/

    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                U0[rl][n][i] = 0.0;
            }
        }
    }

    i = 0;
    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            if (i < Msize2[Mc_AN])
                U0[rl][n][i] = 1.0;
            i++;
        }
    }

    /************************************************************
                orthogonalization by diagonalization
    ************************************************************/

    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

        /*  S * |Vn) */

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec1[n][i] = 0.0;
            }
        }

        for (i = 0; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = 0; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j] - 1;

                if (0 <= kl) {

#ifndef __x86_64__

                    /* Original version */
                    /**/
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                            }
                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
                    /**/

#else

                    /* Unrolling + SSE version */

                    for (m = 0; m < (ian - 3); m += 4) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            mmSum00 = _mm_setzero_pd();
                            mmSum01 = _mm_setzero_pd();
                            mmSum10 = _mm_setzero_pd();
                            mmSum11 = _mm_setzero_pd();
                            mmSum20 = _mm_setzero_pd();
                            mmSum21 = _mm_setzero_pd();
                            mmSum30 = _mm_setzero_pd();
                            mmSum31 = _mm_setzero_pd();

                            for (k = 0; k < (jan - 3); k += 4) {
                                mmTmp0 = _mm_loadu_pd(&U0[rl][n][Bnum + k + 0]);
                                mmTmp1 = _mm_loadu_pd(&U0[rl][n][Bnum + k + 2]);

                                mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 0]), mmTmp0));
                                mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 2]), mmTmp1));

                                mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 0]), mmTmp0));
                                mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 2]), mmTmp1));

                                mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 0]), mmTmp0));
                                mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 2]), mmTmp1));

                                mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 0]), mmTmp0));
                                mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 2]), mmTmp1));
                            }

                            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                            _mm_storeu_pd(&mmArr[0], mmSum00);
                            _mm_storeu_pd(&mmArr[2], mmSum10);
                            _mm_storeu_pd(&mmArr[4], mmSum20);
                            _mm_storeu_pd(&mmArr[6], mmSum30);

                            sum0 = mmArr[0] + mmArr[1];
                            sum1 = mmArr[2] + mmArr[3];
                            sum2 = mmArr[4] + mmArr[5];
                            sum3 = mmArr[6] + mmArr[7];

                            for (; k < jan; k++) {
                                sum0 += OLP0[ih][kl][m + 0][k] * U0[rl][n][Bnum + k];
                                sum1 += OLP0[ih][kl][m + 1][k] * U0[rl][n][Bnum + k];
                                sum2 += OLP0[ih][kl][m + 2][k] * U0[rl][n][Bnum + k];
                                sum3 += OLP0[ih][kl][m + 3][k] * U0[rl][n][Bnum + k];
                            }

                            tmpvec1[n][Anum + m + 0] += sum0;
                            tmpvec1[n][Anum + m + 1] += sum1;
                            tmpvec1[n][Anum + m + 2] += sum2;
                            tmpvec1[n][Anum + m + 3] += sum3;
                        }
                    }

                    for (; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                            }
                            tmpvec1[n][Anum + m] += sum;
                        }
                    }

#endif
                }
            }
        }

#ifndef __x86_64__

        /* Original version */
        /**/
        for (rl0 = rl; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += U0[rl0][m][i] * tmpvec1[n][i];
                    }
                    FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum;
                }
            }
        }
        /**/

#else

        /* Unrolling + SSE version */

        for (rl0 = rl; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    mmSum00 = _mm_setzero_pd();
                    mmSum01 = _mm_setzero_pd();
                    mmSum10 = _mm_setzero_pd();
                    mmSum11 = _mm_setzero_pd();
                    mmSum20 = _mm_setzero_pd();
                    mmSum21 = _mm_setzero_pd();
                    mmSum30 = _mm_setzero_pd();
                    mmSum31 = _mm_setzero_pd();

                    for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                        mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i + 0]);
                        mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i + 2]);

                        mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 0]), mmTmp0));
                        mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 2]), mmTmp1));

                        mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 0]), mmTmp0));
                        mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 2]), mmTmp1));

                        mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 0]), mmTmp0));
                        mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 2]), mmTmp1));

                        mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 0]), mmTmp0));
                        mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 2]), mmTmp1));
                    }

                    mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                    mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                    mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                    mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                    _mm_storeu_pd(&mmArr[0], mmSum00);
                    _mm_storeu_pd(&mmArr[2], mmSum10);
                    _mm_storeu_pd(&mmArr[4], mmSum20);
                    _mm_storeu_pd(&mmArr[6], mmSum30);

                    sum0 = mmArr[0] + mmArr[1];
                    sum1 = mmArr[2] + mmArr[3];
                    sum2 = mmArr[4] + mmArr[5];
                    sum3 = mmArr[6] + mmArr[7];

                    for (; i < Msize2[Mc_AN]; i++) {
                        sum0 += U0[rl0][m + 0][i] * tmpvec1[n][i];
                        sum1 += U0[rl0][m + 1][i] * tmpvec1[n][i];
                        sum2 += U0[rl0][m + 2][i] * tmpvec1[n][i];
                        sum3 += U0[rl0][m + 3][i] * tmpvec1[n][i];
                    }

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum0;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum0;

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 2][rl * EKC_core_size[Mc_AN] + n + 1] = sum1;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 2] = sum1;

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 3][rl * EKC_core_size[Mc_AN] + n + 1] = sum2;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 3] = sum2;

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 4][rl * EKC_core_size[Mc_AN] + n + 1] = sum3;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 4] = sum3;
                }
            }

            for (; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += U0[rl0][m][i] * tmpvec1[n][i];
                    }
                    FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum;
                }
            }
        }

#endif
    }

    Eigen_lapack(FS, ko, Msize3[Mc_AN], Msize3[Mc_AN]);
    ZeroNum = 0;

    for (i = 1; i <= Msize3[Mc_AN]; i++) {

        if (error_check == 1) {
            printf("spin=%2d Mc_AN=%2d i=%3d ko[i]=%18.15f\n", spin, Mc_AN, i, ko[i]);
        }

        if (cutoff_value < ko[i]) {
            ko[i] = sqrt(fabs(ko[i]));
            iko[i] = 1.0 / ko[i];
        } else {
            ZeroNum++;
            ko[i] = 0.0;
            iko[i] = 0.0;
        }
    }

    if (error_check == 1) {
        printf("spin=%2d Mc_AN=%2d ZeroNum=%2d\n", spin, Mc_AN, ZeroNum);
    }

    for (i = 1; i <= Msize3[Mc_AN]; i++) {
        for (j = 1; j <= Msize3[Mc_AN]; j++) {
            FS[i][j] = FS[i][j] * iko[j];
        }
    }

    /* transpose for later calculation */
    for (i = 1; i <= Msize3[Mc_AN]; i++) {
        for (j = i + 1; j <= Msize3[Mc_AN]; j++) {
            tmp1 = FS[i][j];
            tmp2 = FS[j][i];
            FS[i][j] = tmp2;
            FS[j][i] = tmp1;
        }
    }

    /* U0 * U * lamda^{-1/2} */

    for (i = 0; i < Msize2[Mc_AN]; i++) {
        for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {

                m1 = rl0 * EKC_core_size[Mc_AN] + m + 1;

                sum = 0.0;
                for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                    n1 = rl * EKC_core_size[Mc_AN] + 1;

                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum += U0[rl][n][i] * FS[m1][n1 + n];
                    }
                }

                Utmp[rl0][m] = sum;
            }
        }

        for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                U0[rl0][m][i] = Utmp[rl0][m];
            }
        }
    }

    if (0) {

#ifndef __x86_64__

        /* original version */

        /**/
        for (i = 0; i < Msize2[Mc_AN]; i++) {
            for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < EKC_core_size[Mc_AN]; m++) {

                    m1 = rl0 * EKC_core_size[Mc_AN] + m + 1;

                    sum = 0.0;
                    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                        n1 = rl * EKC_core_size[Mc_AN] + 1;

                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                            sum += U0[rl][n][i] * FS[m1][n1 + n];
                        }
                    }

                    Utmp[rl0][m] = sum;
                }
            }

            for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                    U0[rl0][m][i] = Utmp[rl0][m];
                }
            }
        }
        /**/

#else

        /* Unrolling + SSE version */
        /**/
        for (i = 0; i < Msize2[Mc_AN]; i++) {

            for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    Utmp[rl][n] = U0[rl][n][i];
                }
            }

            for (m1 = 1; m1 <= (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - 3); m1 += 4) {

                rl00 = (m1 + 0 - 1) / EKC_core_size[Mc_AN];
                rl01 = (m1 + 1 - 1) / EKC_core_size[Mc_AN];
                rl02 = (m1 + 2 - 1) / EKC_core_size[Mc_AN];
                rl03 = (m1 + 3 - 1) / EKC_core_size[Mc_AN];

                mm0 = (m1 + 0 - 1) % EKC_core_size[Mc_AN];
                mm1 = (m1 + 1 - 1) % EKC_core_size[Mc_AN];
                mm2 = (m1 + 2 - 1) % EKC_core_size[Mc_AN];
                mm3 = (m1 + 3 - 1) % EKC_core_size[Mc_AN];

                sum0 = 0.0;
                sum1 = 0.0;
                sum2 = 0.0;
                sum3 = 0.0;

                mmSum00 = _mm_setzero_pd();
                mmSum01 = _mm_setzero_pd();
                mmSum10 = _mm_setzero_pd();
                mmSum11 = _mm_setzero_pd();
                mmSum20 = _mm_setzero_pd();
                mmSum21 = _mm_setzero_pd();
                mmSum30 = _mm_setzero_pd();
                mmSum31 = _mm_setzero_pd();

                for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                    n1 = rl * EKC_core_size[Mc_AN] + 1;

                    for (n = 0; n < (EKC_core_size[Mc_AN] - 3); n += 4) {

                        mmTmp0 = _mm_loadu_pd(&Utmp[rl][n + 0]);
                        mmTmp1 = _mm_loadu_pd(&Utmp[rl][n + 2]);

                        mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 0][n1 + n + 0]), mmTmp0));
                        mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 0][n1 + n + 2]), mmTmp1));

                        mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 1][n1 + n + 0]), mmTmp0));
                        mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 1][n1 + n + 2]), mmTmp1));

                        mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 2][n1 + n + 0]), mmTmp0));
                        mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 2][n1 + n + 2]), mmTmp1));

                        mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 3][n1 + n + 0]), mmTmp0));
                        mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 3][n1 + n + 2]), mmTmp1));
                    }

                    for (; n < EKC_core_size[Mc_AN]; n++) {
                        sum0 += Utmp[rl][n] * FS[m1 + 0][n1 + n];
                        sum1 += Utmp[rl][n] * FS[m1 + 1][n1 + n];
                        sum2 += Utmp[rl][n] * FS[m1 + 2][n1 + n];
                        sum3 += Utmp[rl][n] * FS[m1 + 3][n1 + n];
                    }
                }

                mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                _mm_storeu_pd(&mmArr[0], mmSum00);
                _mm_storeu_pd(&mmArr[2], mmSum10);
                _mm_storeu_pd(&mmArr[4], mmSum20);
                _mm_storeu_pd(&mmArr[6], mmSum30);

                sum0 += mmArr[0] + mmArr[1];
                sum1 += mmArr[2] + mmArr[3];
                sum2 += mmArr[4] + mmArr[5];
                sum3 += mmArr[6] + mmArr[7];

                U0[rl00][mm0][i] = sum0;
                U0[rl01][mm1][i] = sum1;
                U0[rl02][mm2][i] = sum2;
                U0[rl03][mm3][i] = sum3;
            }

            m1s = rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]) % 4 + 1;

            for (m1 = m1s; m1 <= rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]; m1++) {

                rl0 = (m1 - 1) / EKC_core_size[Mc_AN];
                m = (m1 - 1) % EKC_core_size[Mc_AN];

                sum = 0.0;

                for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                    n1 = rl * EKC_core_size[Mc_AN] + 1;

                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum += Utmp[rl][n] * FS[m1][n1 + n];
                    }
                }

                U0[rl0][m][i] = sum;
            }
        } /* i */

#endif
    }

    Krylov_U[spin][Mc_AN][0] = ZeroNum;

    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + i + 1] = U0[rl][n][i];
            }
        }
    }

    /************************************************************
           check the orthonormality of Krylov vectors
    ************************************************************/

    if (error_check == 1) {

        for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    tmpvec1[n][i] = 0.0;
                }
            }

            for (i = 0; i <= can; i++) {

                ig = natn[ct_AN][i];
                ian = Spe_Total_CNO[WhatSpecies[ig]];
                Anum = MP[i] - 1;
                ih = S_G2M[ig];

                for (j = 0; j <= can; j++) {

                    kl = RMI1[Mc_AN][i][j];
                    jg = natn[ct_AN][j];
                    jan = Spe_Total_CNO[WhatSpecies[jg]];
                    Bnum = MP[j] - 1;

                    if (0 <= kl) {

                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                sum = 0.0;
                                for (k = 0; k < jan; k++) {
                                    sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                                }

                                tmpvec1[n][Anum + m] += sum;
                            }
                        }
                    }
                }
            }

            for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum = 0.0;
                        for (i = 0; i < Msize2[Mc_AN]; i++) {
                            sum += U0[rl0][m][i] * tmpvec1[n][i];
                        }

                        if (rl == rl0 && m == n) {
                            if (1.0e-10 < fabs(sum - 1.0)) {
                                printf("A spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
                                    spin, Mc_AN, rl, rl0, m, n, sum);
                            }
                        } else {
                            if (1.0e-10 < fabs(sum)) {
                                printf("B spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
                                    spin, Mc_AN, rl, rl0, m, n, sum);
                            }
                        }
                    }
                }
            }
        }
    }

    /* freeing of arrays */

    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        free(Utmp[i]);
    }
    free(Utmp);

    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
            free(U0[i][j]);
        }
        free(U0[i]);
    }
    free(U0);

    for (i = 0; i < (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        free(FS[i]);
    }
    free(FS);

    free(ko);
    free(iko);
}

void Embedding_Matrix(int spin, int Mc_AN, double***** Hks,
    double*** Krylov_U, double**** EC_matrix,
    int* MP, int* Msize, int* Msize2, int* Msize3,
    double** tmpvec1)
{
    int ct_AN, fan, san, can, wan, ct_on;
    int rl, rl0, m, n, i, j, k, kl, jg, jan, ih, ian;
    int Anum, Bnum, ig;
    int KU_d1, KU_d2, csize;
    double sum, tmp1, tmp2, tmp3;
    __m128d mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5;

    double mmArr[8];
    double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
    double stime, etime;
    double time1, time2, time3, time4, time5, time6, time7, time8;

    ct_AN = M2G[Mc_AN];
    fan = FNAN[ct_AN];
    san = SNAN[ct_AN];
    can = fan + san;
    wan = WhatSpecies[ct_AN];
    ct_on = Spe_Total_CNO[wan];

    if (Msize[Mc_AN] < Msize3[Mc_AN])
        csize = Msize3[Mc_AN] + 40;
    else
        csize = Msize[Mc_AN] + 40;

    KU_d1 = EKC_core_size[Mc_AN] * Msize2[Mc_AN];
    KU_d2 = Msize2[Mc_AN];

    if (measure_time == 1) {
        time1 = 0.0;
        time2 = 0.0;
        time3 = 0.0;
        time4 = 0.0;
        time5 = 0.0;
        time6 = 0.0;
        time7 = 0.0;
        time8 = 0.0;
    }

    /*******************************
              u1^+ C^+ u2
    *******************************/

    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

        /* C^+ u2 */

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec1[n][i] = 0.0;
            }
        }

        for (i = 0; i <= fan; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = fan + 1; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j];

                if (0 <= kl) {

#ifndef __x86_64__

                    if (measure_time == 1)
                        dtime(&stime);

                    /* Original version */
                    /**/
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += Hks[spin][ih][kl][m][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                            }

                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
                    /**/

                    if (measure_time == 1) {
                        dtime(&etime);
                        time1 += etime - stime;
                    }

#else
                    if (measure_time == 1)
                        dtime(&stime);

                    /* Unrolling + SSE version */
                    /**/
                    for (m = 0; m < (ian - 3); m += 4) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            mmSum00 = _mm_setzero_pd();
                            mmSum01 = _mm_setzero_pd();
                            mmSum10 = _mm_setzero_pd();
                            mmSum11 = _mm_setzero_pd();
                            mmSum20 = _mm_setzero_pd();
                            mmSum21 = _mm_setzero_pd();
                            mmSum30 = _mm_setzero_pd();
                            mmSum31 = _mm_setzero_pd();

                            for (k = 0; k < (jan - 3); k += 4) {
                                mmTmp0 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0]);
                                mmTmp1 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 2]);

                                mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 0]), mmTmp0));
                                mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 2]), mmTmp1));

                                mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 0]), mmTmp0));
                                mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 2]), mmTmp1));

                                mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 0]), mmTmp0));
                                mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 2]), mmTmp1));

                                mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 0]), mmTmp0));
                                mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 2]), mmTmp1));
                            }

                            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                            _mm_storeu_pd(&mmArr[0], mmSum00);
                            _mm_storeu_pd(&mmArr[2], mmSum10);
                            _mm_storeu_pd(&mmArr[4], mmSum20);
                            _mm_storeu_pd(&mmArr[6], mmSum30);

                            sum0 = mmArr[0] + mmArr[1];
                            sum1 = mmArr[2] + mmArr[3];
                            sum2 = mmArr[4] + mmArr[5];
                            sum3 = mmArr[6] + mmArr[7];

                            for (; k < jan; k++) {
                                sum0 += Hks[spin][ih][kl][m + 0][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                                sum1 += Hks[spin][ih][kl][m + 1][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                                sum2 += Hks[spin][ih][kl][m + 2][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                                sum3 += Hks[spin][ih][kl][m + 3][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                            }

                            tmpvec1[n][Anum + m + 0] += sum0;
                            tmpvec1[n][Anum + m + 1] += sum1;
                            tmpvec1[n][Anum + m + 2] += sum2;
                            tmpvec1[n][Anum + m + 3] += sum3;
                        }
                    }

                    for (; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += Hks[spin][ih][kl][m][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                            }

                            tmpvec1[n][Anum + m] += sum;
                        }
                    }

                    if (measure_time == 1) {
                        dtime(&etime);
                        time1 += etime - stime;
                    }
#endif
                }
            }
        }

        /* u1^+ C^+ u2 */

#ifndef __x86_64__

        if (measure_time == 1)
            dtime(&stime);

        /* Original version */
        /**/
        for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize[Mc_AN]; i++) {
                        sum += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + m * KU_d2 + i + 1] * tmpvec1[n][i];
                    }
                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                }
            }
        }
        /**/

        if (measure_time == 1) {
            dtime(&etime);
            time2 += etime - stime;
        }

#else

        /*
        printf("ABC1 Mc_AN=%2d %2d %2d %2d\n",Mc_AN,rlmax_EC[Mc_AN],EKC_core_size[Mc_AN],Msize[Mc_AN]);
        */

        if (measure_time == 1)
            dtime(&stime);

        /* Unrolling + SSE version */
        /**/
        for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    mmSum00 = _mm_setzero_pd();
                    mmSum01 = _mm_setzero_pd();
                    mmSum10 = _mm_setzero_pd();
                    mmSum11 = _mm_setzero_pd();
                    mmSum20 = _mm_setzero_pd();
                    mmSum21 = _mm_setzero_pd();
                    mmSum30 = _mm_setzero_pd();
                    mmSum31 = _mm_setzero_pd();

                    for (i = 0; i < (Msize[Mc_AN] - 3); i += 4) {
                        mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i + 0]);
                        mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i + 2]);

                        mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1]), mmTmp0));
                        mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 3]), mmTmp1));

                        mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 1]), mmTmp0));
                        mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 3]), mmTmp1));

                        mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 1]), mmTmp0));
                        mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 3]), mmTmp1));

                        mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 1]), mmTmp0));
                        mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 3]), mmTmp1));
                    }

                    mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                    mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                    mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                    mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                    _mm_storeu_pd(&mmArr[0], mmSum00);
                    _mm_storeu_pd(&mmArr[2], mmSum10);
                    _mm_storeu_pd(&mmArr[4], mmSum20);
                    _mm_storeu_pd(&mmArr[6], mmSum30);

                    sum0 = mmArr[0] + mmArr[1];
                    sum1 = mmArr[2] + mmArr[3];
                    sum2 = mmArr[4] + mmArr[5];
                    sum3 = mmArr[6] + mmArr[7];

                    for (; i < Msize[Mc_AN]; i++) {
                        sum0 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1] * tmpvec1[n][i];
                        sum1 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 1] * tmpvec1[n][i];
                        sum2 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 1] * tmpvec1[n][i];
                        sum3 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 1] * tmpvec1[n][i];
                    }

                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum0;
                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 2][rl * EKC_core_size[Mc_AN] + n + 1] = sum1;
                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 3][rl * EKC_core_size[Mc_AN] + n + 1] = sum2;
                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 4][rl * EKC_core_size[Mc_AN] + n + 1] = sum3;
                }
            }

            for (; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize[Mc_AN]; i++) {
                        sum += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1] * tmpvec1[n][i];
                    }
                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                }
            }
        }
        /**/

        if (measure_time == 1) {
            dtime(&etime);
            time2 += etime - stime;
        }

#endif

    } /* rl */

    /*******************************
              u2^+ C u1
    *******************************/

    for (i = 1; i <= Msize3[Mc_AN]; i++) {
        for (j = i; j <= Msize3[Mc_AN]; j++) {

            tmp1 = EC_matrix[spin][Mc_AN][i][j];
            tmp2 = EC_matrix[spin][Mc_AN][j][i];
            tmp3 = tmp1 + tmp2;

            EC_matrix[spin][Mc_AN][i][j] = tmp3;
            EC_matrix[spin][Mc_AN][j][i] = tmp3;
        }
    }

    /*******************************
               u2^+ B u2
    *******************************/

    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

        /* B u2 */

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec1[n][i] = 0.0;
            }
        }

        for (i = fan + 1; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = fan + 1; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j];

                if (0 <= kl) {

#ifndef __x86_64__

                    if (measure_time == 1)
                        dtime(&stime);

                    /* Original version */
                    /**/
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += Hks[spin][ih][kl][m][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                            }

                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
                    /**/

                    if (measure_time == 1) {
                        dtime(&etime);
                        time3 += etime - stime;
                    }

#else

                    if (measure_time == 1)
                        dtime(&stime);

                    /* Unrolling + SSE version */
                    /**/
                    for (m = 0; m < (ian - 3); m += 4) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            mmSum00 = _mm_setzero_pd();
                            mmSum01 = _mm_setzero_pd();
                            mmSum10 = _mm_setzero_pd();
                            mmSum11 = _mm_setzero_pd();
                            mmSum20 = _mm_setzero_pd();
                            mmSum21 = _mm_setzero_pd();
                            mmSum30 = _mm_setzero_pd();
                            mmSum31 = _mm_setzero_pd();

                            for (k = 0; k < (jan - 3); k += 4) {
                                mmTmp0 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0]);
                                mmTmp1 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 2]);

                                mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 0]), mmTmp0));
                                mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 2]), mmTmp1));

                                mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 0]), mmTmp0));
                                mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 2]), mmTmp1));

                                mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 0]), mmTmp0));
                                mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 2]), mmTmp1));

                                mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 0]), mmTmp0));
                                mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 2]), mmTmp1));
                            }

                            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                            _mm_storeu_pd(&mmArr[0], mmSum00);
                            _mm_storeu_pd(&mmArr[2], mmSum10);
                            _mm_storeu_pd(&mmArr[4], mmSum20);
                            _mm_storeu_pd(&mmArr[6], mmSum30);

                            sum0 = mmArr[0] + mmArr[1];
                            sum1 = mmArr[2] + mmArr[3];
                            sum2 = mmArr[4] + mmArr[5];
                            sum3 = mmArr[6] + mmArr[7];

                            for (; k < jan; k++) {
                                sum0 += Hks[spin][ih][kl][m + 0][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0];
                                sum1 += Hks[spin][ih][kl][m + 1][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0];
                                sum2 += Hks[spin][ih][kl][m + 2][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0];
                                sum3 += Hks[spin][ih][kl][m + 3][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0];
                            }

                            tmpvec1[n][Anum + m + 0] += sum0;
                            tmpvec1[n][Anum + m + 1] += sum1;
                            tmpvec1[n][Anum + m + 2] += sum2;
                            tmpvec1[n][Anum + m + 3] += sum3;
                        }
                    }

                    for (; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += Hks[spin][ih][kl][m][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                            }

                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
                    /**/

                    if (measure_time == 1) {
                        dtime(&etime);
                        time3 += etime - stime;
                    }

#endif
                }
            }
        }

        /* u2^+ B u2 */

#ifndef __x86_64__

        if (measure_time == 1)
            dtime(&stime);

        /* Original version */
        /**/
        for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = Msize[Mc_AN]; i < Msize2[Mc_AN]; i++) {
                        sum += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + m * KU_d2 + i + 1] * tmpvec1[n][i];
                    }
                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] += sum;
                }
            }
        }
        /**/

        if (measure_time == 1) {
            dtime(&etime);
            time4 += etime - stime;
        }

#else

        if (measure_time == 1)
            dtime(&stime);

        /* Unrolling + SSE version */
        /**/
        for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    mmSum00 = _mm_setzero_pd();
                    mmSum01 = _mm_setzero_pd();
                    mmSum10 = _mm_setzero_pd();
                    mmSum11 = _mm_setzero_pd();
                    mmSum20 = _mm_setzero_pd();
                    mmSum21 = _mm_setzero_pd();
                    mmSum30 = _mm_setzero_pd();
                    mmSum31 = _mm_setzero_pd();

                    for (i = Msize[Mc_AN]; i < (Msize2[Mc_AN] - 3); i += 4) {
                        mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i + 0]);
                        mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i + 2]);

                        mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1]), mmTmp0));
                        mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 3]), mmTmp1));

                        mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 1]), mmTmp0));
                        mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 3]), mmTmp1));

                        mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 1]), mmTmp0));
                        mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 3]), mmTmp1));

                        mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 1]), mmTmp0));
                        mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 3]), mmTmp1));
                    }

                    mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                    mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                    mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                    mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                    _mm_storeu_pd(&mmArr[0], mmSum00);
                    _mm_storeu_pd(&mmArr[2], mmSum10);
                    _mm_storeu_pd(&mmArr[4], mmSum20);
                    _mm_storeu_pd(&mmArr[6], mmSum30);

                    sum0 = mmArr[0] + mmArr[1];
                    sum1 = mmArr[2] + mmArr[3];
                    sum2 = mmArr[4] + mmArr[5];
                    sum3 = mmArr[6] + mmArr[7];

                    for (; i < Msize2[Mc_AN]; i++) {
                        sum0 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1] * tmpvec1[n][i];
                        sum1 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 1] * tmpvec1[n][i];
                        sum2 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 1] * tmpvec1[n][i];
                        sum3 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 1] * tmpvec1[n][i];
                    }

                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] += sum0;
                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 2][rl * EKC_core_size[Mc_AN] + n + 1] += sum1;
                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 3][rl * EKC_core_size[Mc_AN] + n + 1] += sum2;
                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 4][rl * EKC_core_size[Mc_AN] + n + 1] += sum3;
                }
            }

            for (; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = Msize[Mc_AN]; i < Msize2[Mc_AN]; i++) {
                        sum += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1] * tmpvec1[n][i];
                    }
                    EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] += sum;
                }
            }
        }
        /**/

        if (measure_time == 1) {
            dtime(&etime);
            time4 += etime - stime;
        }

#endif

    } /* rl */

    if (measure_time == 1) {
        printf("Embedding_Matrix time1=%5.3f time2=%5.3f time3=%5.3f time4=%5.3f\n",
            time1, time2, time3, time4);
    }
}

void Krylov_IOLP(int Mc_AN, double**** OLP0, double*** Krylov_U_OLP, double** inv_RS, int* MP,
    int* Msize2, int* Msize4, int Msize2_max, double** tmpvec0, double** tmpvec1)
{
    int rl, ct_AN, fan, san, can, wan, ct_on, i, j;
    int n, Anum, Bnum, k, ian, ih, kl, jg, ig, jan, m, m1, n1;
    int rl0, ZeroNum, Neumann_series, ns, Gh_AN, wanB;
    double sum, dum, tmp0, tmp1, tmp2, tmp3, rcutA, r0;
    double** tmpmat0;
    double *ko, *iko;
    double** FS;

    ct_AN = M2G[Mc_AN];
    fan = FNAN[ct_AN];
    san = SNAN[ct_AN];
    can = fan + san;
    wan = WhatSpecies[ct_AN];
    ct_on = Spe_Total_CNO[wan];
    rcutA = Spe_Atom_Cut1[wan];

    /* allocation of arrays */

    tmpmat0 = (double**)malloc(sizeof(double*) * (EKC_core_size[Mc_AN] + 4));
    for (i = 0; i < (EKC_core_size[Mc_AN] + 4); i++) {
        tmpmat0[i] = (double*)malloc(sizeof(double) * (EKC_core_size[Mc_AN] + 4));
    }

    FS = (double**)malloc(sizeof(double*) * (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    for (i = 0; i < (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        FS[i] = (double*)malloc(sizeof(double) * (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    }

    ko = (double*)malloc(sizeof(double) * (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    iko = (double*)malloc(sizeof(double) * (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]);

    /****************************************************
        initialize
    ****************************************************/

    for (i = 0; i < EKC_core_size_max; i++) {
        for (j = 0; j < Msize2_max; j++) {
            tmpvec0[i][j] = 0.0;
            tmpvec1[i][j] = 0.0;
        }
    }

    /* find the nearest atom with distance of r0 */

    r0 = 1.0e+10;
    for (k = 1; k <= FNAN[ct_AN]; k++) {
        Gh_AN = natn[ct_AN][k];
        wanB = WhatSpecies[Gh_AN];
        if (Dis[ct_AN][k] < r0)
            r0 = Dis[ct_AN][k];
    }

    /* starting vector */

    m = 0;
    for (k = 0; k <= FNAN[ct_AN]; k++) {

        Gh_AN = natn[ct_AN][k];
        wanB = WhatSpecies[Gh_AN];

        if (Dis[ct_AN][k] < (scale_rc_EKC[Mc_AN] * r0)) {

            Anum = MP[k] - 1;

            for (i = 0; i < Spe_Total_CNO[wanB]; i++) {

                tmpvec0[m][Anum + i] = 1.0;
                Krylov_U_OLP[0][m][Anum + i] = 1.0;

                m++;
            }
        }
    }

    /*
    for (i=0; i<EKC_core_size[Mc_AN]; i++){
      tmpvec0[i][i]         = 1.0;
      Krylov_U_OLP[0][i][i] = 1.0;
    }
    */

    /****************************************************
             generate Krylov subspace vectors
    ****************************************************/

    for (rl = 0; rl < (rlmax_EC2[Mc_AN] - 1); rl++) {

        /*******************************************************
                                S * |Wn)
        *******************************************************/

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec1[n][i] = 0.0;
            }
        }

        for (i = 0; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = 0; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j] - 1;

                if (0 <= kl) {

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
                }
            }
        }

        /*************************************************************
          orthogonalization by a modified block Gram-Schmidt method
        *************************************************************/

        /* |tmpvec1) := (I - \sum_{rl0} |U_rl0)(U_rl0|)|tmpvec1) */

        for (rl0 = 0; rl0 <= rl; rl0++) {

            /* (U_rl0|tmpvec1) */

            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += Krylov_U_OLP[rl0][m][i] * tmpvec1[n][i];
                    }
                    tmpmat0[m][n] = sum;
                }
            }

            /* |tmpvec1) := |tmpvec1) - |U_rl0)(U_rl0|tmpvec1) */

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (k = 0; k < EKC_core_size[Mc_AN]; k++) {
                    dum = tmpmat0[k][n];
                    for (i = 0; i < Msize2[Mc_AN]; i++)
                        tmpvec1[n][i] -= Krylov_U_OLP[rl0][k][i] * dum;
                }
            }
        }

        /*************************************************************
                           normalization of tmpvec1
        *************************************************************/

        for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                sum = 0.0;
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    sum += tmpvec1[m][i] * tmpvec1[n][i];
                }

                tmpmat0[m + 1][n + 1] = sum;
            }
        }

        /* diagonalize tmpmat0 */

        if (EKC_core_size[Mc_AN] == 1) {
            ko[1] = tmpmat0[1][1];
            tmpmat0[1][1] = 1.0;
        } else {
            Eigen_lapack(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
        }

        ZeroNum = 0;

        for (n = 1; n <= EKC_core_size[Mc_AN]; n++) {
            if (cutoff_value < ko[n]) {
                ko[n] = sqrt(fabs(ko[n]));
                iko[n] = 1.0 / ko[n];
            } else {
                ZeroNum++;
                ko[n] = 0.0;
                iko[n] = 0.0;
            }

            if (error_check == 1) {
                printf("rl=%3d ko=%18.15f\n", rl, ko[n]);
            }
        }

        if (error_check == 1) {
            printf("rl=%3d ZeroNum=%3d\n", rl, ZeroNum);
        }

        /* tmpvec0 = tmpvec1 * tmpmat0^{-1/2} */

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec0[n][i] = 0.0;
            }
        }

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (k = 0; k < EKC_core_size[Mc_AN]; k++) {
                dum = tmpmat0[k + 1][n + 1] * iko[n + 1];
                for (i = 0; i < Msize2[Mc_AN]; i++)
                    tmpvec0[n][i] += tmpvec1[k][i] * dum;
            }
        }

        /*************************************************************
                             store Krylov vectors
        *************************************************************/

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                Krylov_U_OLP[rl + 1][n][i] = tmpvec0[n][i];
            }
        }

    } /* rl */

    /************************************************************
         calculate the inverse of the reduced overlap matrix
    ************************************************************/

    /*  construct the reduced overlap matrix */

    for (rl = 0; rl < rlmax_EC2[Mc_AN]; rl++) {

        /*  S * |Vn) */

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec1[n][i] = 0.0;
            }
        }

        for (i = 0; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = 0; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j] - 1;

                if (0 <= kl) {

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * Krylov_U_OLP[rl][n][Bnum + k];
                            }
                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
                }
            }
        }

        for (rl0 = rl; rl0 < rlmax_EC2[Mc_AN]; rl0++) {
            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += Krylov_U_OLP[rl0][m][i] * tmpvec1[n][i];
                    }
                    FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum;
                }
            }
        }
    }

    /* diagonalize FS */

    Eigen_lapack(FS, ko, Msize4[Mc_AN], Msize4[Mc_AN]);

    /* find ill-conditioned eigenvalues */

    ZeroNum = 0;
    for (i = 1; i <= Msize4[Mc_AN]; i++) {

        if (error_check == 1) {
            printf("Mc_AN=%2d i=%3d ko[i]=%18.15f\n", Mc_AN, i, ko[i]);
        }

        if (cutoff_value < ko[i]) {
            iko[i] = 1.0 / ko[i];
        } else {
            ZeroNum++;
            ko[i] = 0.0;
            iko[i] = 0.0;
        }
    }

    if (error_check == 1) {
        printf("Mc_AN=%2d ZeroNum=%2d\n", Mc_AN, ZeroNum);
    }

    /* construct the inverse */

    for (i = 1; i <= Msize4[Mc_AN]; i++) {
        for (j = 1; j <= Msize4[Mc_AN]; j++) {
            sum = 0.0;
            for (k = 1; k <= Msize4[Mc_AN]; k++) {
                sum += FS[i][k] * iko[k] * FS[j][k];
            }
            inv_RS[i - 1][j - 1] = sum;
        }
    }

    /* symmetrization of inv_RS */

    for (i = 1; i <= Msize4[Mc_AN]; i++) {
        for (j = i + 1; j <= Msize4[Mc_AN]; j++) {
            tmp0 = inv_RS[i - 1][j - 1];
            tmp1 = inv_RS[j - 1][i - 1];
            tmp2 = 0.5 * (tmp0 + tmp1);
            inv_RS[i - 1][j - 1] = tmp2;
            inv_RS[j - 1][i - 1] = tmp2;
        }
    }

    /*
    {


      double mat1[1000][1000];
      double tsum;
      int myid;

      MPI_Comm_rank(mpi_comm_level1,&myid);

      if (myid==0){

        printf("check normalization\n");

        for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
          for (m=0; m<EKC_core_size[Mc_AN]; m++){
            for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
              for (n=0; n<EKC_core_size[Mc_AN]; n++){

                sum = 0.0;
                for (i=0; i<Msize2[Mc_AN]; i++){
                  sum += Krylov_U_OLP[rl][m][i]*Krylov_U_OLP[rl0][n][i];
                }
                printf("rl=%3d rl0=%3d m=%3d n=%3d  <|>=%18.15f\n",rl,rl0,m,n,sum);
              }
            }
          }
        }


        printf("\n\ninvS\n");


        for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
          for (m=0; m<EKC_core_size[Mc_AN]; m++){

            for (i=0; i<Msize2[Mc_AN]; i++){

              sum = 0.0;

              for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
                for (n=0; n<EKC_core_size[Mc_AN]; n++){
                  sum += inv_RS[rl*EKC_core_size[Mc_AN]+m][rl0*EKC_core_size[Mc_AN]+n]*Krylov_U_OLP[rl0][n][i];
                }
              }

              mat1[rl*EKC_core_size[Mc_AN]+m][i] = sum;
            }
          }
        }


        tsum = 0.0;

        for (i=0; i<Msize2[Mc_AN]; i++){
          for (j=0; j<Msize2[Mc_AN]; j++){

            sum = 0.0;
            for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
              for (m=0; m<EKC_core_size[Mc_AN]; m++){
                sum += Krylov_U_OLP[rl][m][i]*mat1[rl*EKC_core_size[Mc_AN]+m][j];
              }
            }

            printf("i=%4d j=%4d %18.15f\n",i,j,sum);

            tsum += fabs(sum);

          }
        }


        printf("\n\ntsum=%18.15f\n",tsum);

      }


      MPI_Finalize();
      exit(0);
    }
    */

    /* freeing of arrays */

    for (i = 0; i < (EKC_core_size[Mc_AN] + 4); i++) {
        free(tmpmat0[i]);
    }
    free(tmpmat0);

    for (i = 0; i < (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        free(FS[i]);
    }
    free(FS);

    free(ko);
    free(iko);
}

void S_orthonormalize_vec(int Mc_AN, int ct_on, double** vec,
    double** workvec, double**** OLP0,
    double** tmpmat0, double* ko, double* iko,
    int* MP, int* Msize2)
{
    int n, i, j, can, san, fan, ct_AN;
    int k, m, ZeroNum, Anum, Bnum, ih;
    int kl, jg, jan, ig, ian;
    double dum, sum;
    double tmp0, tmp1, tmp2, tmp3;
    __m128d mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5;
    double mmArr[8];
    double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;

    ct_AN = M2G[Mc_AN];
    fan = FNAN[ct_AN];
    san = SNAN[ct_AN];
    can = fan + san;

    /* S|Vn) */

    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
        for (i = 0; i < Msize2[Mc_AN]; i++) {
            workvec[n][i] = 0.0;
        }
    }

    for (i = 0; i <= can; i++) {

        ig = natn[ct_AN][i];
        ian = Spe_Total_CNO[WhatSpecies[ig]];
        Anum = MP[i] - 1;
        ih = S_G2M[ig];

        for (j = 0; j <= can; j++) {

            kl = RMI1[Mc_AN][i][j];
            jg = natn[ct_AN][j];
            jan = Spe_Total_CNO[WhatSpecies[jg]];
            Bnum = MP[j] - 1;

            if (0 <= kl) {

#ifndef __x86_64__

                /* Original version */
                /**/
                for (m = 0; m < ian; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                        sum = 0.0;
                        for (k = 0; k < jan; k++) {
                            sum += OLP0[ih][kl][m][k] * vec[n][Bnum + k];
                        }

                        workvec[n][Anum + m] += sum;
                    }
                }
                /**/

#else

                /* Unrolling + SSE version */
                /**/
                for (m = 0; m < (ian - 3); m += 4) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                        mmSum00 = _mm_setzero_pd();
                        mmSum01 = _mm_setzero_pd();
                        mmSum10 = _mm_setzero_pd();
                        mmSum11 = _mm_setzero_pd();
                        mmSum20 = _mm_setzero_pd();
                        mmSum21 = _mm_setzero_pd();
                        mmSum30 = _mm_setzero_pd();
                        mmSum31 = _mm_setzero_pd();

                        for (k = 0; k < (jan - 3); k += 4) {
                            mmTmp0 = _mm_loadu_pd(&vec[n][Bnum + k + 0]);
                            mmTmp1 = _mm_loadu_pd(&vec[n][Bnum + k + 2]);

                            mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 0]), mmTmp0));
                            mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 2]), mmTmp1));

                            mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 0]), mmTmp0));
                            mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 2]), mmTmp1));

                            mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 0]), mmTmp0));
                            mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 2]), mmTmp1));

                            mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 0]), mmTmp0));
                            mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 2]), mmTmp1));
                        }

                        mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                        mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                        mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                        mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                        _mm_storeu_pd(&mmArr[0], mmSum00);
                        _mm_storeu_pd(&mmArr[2], mmSum10);
                        _mm_storeu_pd(&mmArr[4], mmSum20);
                        _mm_storeu_pd(&mmArr[6], mmSum30);

                        sum0 = mmArr[0] + mmArr[1];
                        sum1 = mmArr[2] + mmArr[3];
                        sum2 = mmArr[4] + mmArr[5];
                        sum3 = mmArr[6] + mmArr[7];

                        for (; k < jan; k++) {
                            sum0 += OLP0[ih][kl][m + 0][k] * vec[n][Bnum + k];
                            sum1 += OLP0[ih][kl][m + 1][k] * vec[n][Bnum + k];
                            sum2 += OLP0[ih][kl][m + 2][k] * vec[n][Bnum + k];
                            sum3 += OLP0[ih][kl][m + 3][k] * vec[n][Bnum + k];
                        }

                        workvec[n][Anum + m + 0] += sum0;
                        workvec[n][Anum + m + 1] += sum1;
                        workvec[n][Anum + m + 2] += sum2;
                        workvec[n][Anum + m + 3] += sum3;
                    }
                }

                for (; m < ian; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                        sum = 0.0;
                        for (k = 0; k < jan; k++) {
                            sum += OLP0[ih][kl][m][k] * vec[n][Bnum + k];
                        }

                        workvec[n][Anum + m] += sum;
                    }
                }
                /**/

#endif
            }
        }
    }

    /*  (Vn|S|Vn) */

#ifndef __x86_64__

    /* Original version */
    /**/
    for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
        for (n = m; n < EKC_core_size[Mc_AN]; n++) {
            sum = 0.0;
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                sum += vec[m][i] * workvec[n][i];
            }
            tmpmat0[m + 1][n + 1] = sum;
            tmpmat0[n + 1][m + 1] = sum;
        }
    }
    /**/

#else

    /* Unrolling + SSE version */
    /**/
    for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
        for (n = m; n < EKC_core_size[Mc_AN]; n++) {
            mmSum00 = _mm_setzero_pd();
            mmSum01 = _mm_setzero_pd();
            mmSum10 = _mm_setzero_pd();
            mmSum11 = _mm_setzero_pd();
            mmSum20 = _mm_setzero_pd();
            mmSum21 = _mm_setzero_pd();
            mmSum30 = _mm_setzero_pd();
            mmSum31 = _mm_setzero_pd();

            for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                mmTmp0 = _mm_loadu_pd(&workvec[n][i + 0]);
                mmTmp1 = _mm_loadu_pd(&workvec[n][i + 2]);

                mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&vec[m + 0][i + 0]), mmTmp0));
                mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&vec[m + 0][i + 2]), mmTmp1));

                mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&vec[m + 1][i + 0]), mmTmp0));
                mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&vec[m + 1][i + 2]), mmTmp1));

                mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&vec[m + 2][i + 0]), mmTmp0));
                mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&vec[m + 2][i + 2]), mmTmp1));

                mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&vec[m + 3][i + 0]), mmTmp0));
                mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&vec[m + 3][i + 2]), mmTmp1));
            }

            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

            _mm_storeu_pd(&mmArr[0], mmSum00);
            _mm_storeu_pd(&mmArr[2], mmSum10);
            _mm_storeu_pd(&mmArr[4], mmSum20);
            _mm_storeu_pd(&mmArr[6], mmSum30);

            sum0 = mmArr[0] + mmArr[1];
            sum1 = mmArr[2] + mmArr[3];
            sum2 = mmArr[4] + mmArr[5];
            sum3 = mmArr[6] + mmArr[7];

            for (; i < Msize2[Mc_AN]; i++) {
                sum0 += vec[m + 0][i] * workvec[n][i];
                sum1 += vec[m + 1][i] * workvec[n][i];
                sum2 += vec[m + 2][i] * workvec[n][i];
                sum3 += vec[m + 3][i] * workvec[n][i];
            }

            tmpmat0[m + 1][n + 1] = sum0;
            tmpmat0[n + 1][m + 1] = sum0;

            tmpmat0[m + 2][n + 1] = sum1;
            tmpmat0[n + 1][m + 2] = sum1;

            tmpmat0[m + 3][n + 1] = sum2;
            tmpmat0[n + 1][m + 3] = sum2;

            tmpmat0[m + 4][n + 1] = sum3;
            tmpmat0[n + 1][m + 4] = sum3;
        }
    }

    for (; m < EKC_core_size[Mc_AN]; m++) {
        for (n = m; n < EKC_core_size[Mc_AN]; n++) {
            sum = 0.0;
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                sum += vec[m][i] * workvec[n][i];
            }
            tmpmat0[m + 1][n + 1] = sum;
            tmpmat0[n + 1][m + 1] = sum;
        }
    }
    /**/

#endif

    /* diagonalize tmpmat0 */

    if (EKC_core_size[Mc_AN] == 1) {
        ko[1] = tmpmat0[1][1];
        tmpmat0[1][1] = 1.0;
    } else {
        Eigen_lapack(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
    }

    ZeroNum = 0;

    for (n = 1; n <= EKC_core_size[Mc_AN]; n++) {
        if (cutoff_value < ko[n]) {
            ko[n] = sqrt(fabs(ko[n]));
            iko[n] = 1.0 / ko[n];
        } else {
            ZeroNum++;
            ko[n] = 0.0;
            iko[n] = 0.0;
        }
    }

    /* U0 = vec * tmpmat0^{-1/2} */

    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
        for (i = 0; i < Msize2[Mc_AN]; i++) {
            workvec[n][i] = 0.0;
        }
    }

    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
        for (k = 0; k < EKC_core_size[Mc_AN]; k++) {
            dum = tmpmat0[k + 1][n + 1] * iko[n + 1];
            for (i = 0; i < Msize2[Mc_AN]; i++)
                workvec[n][i] += vec[k][i] * dum;
        }
    }

    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
        for (i = 0; i < Msize2[Mc_AN]; i++) {
            vec[n][i] = workvec[n][i];
        }
    }
}

void Inverse_S_by_Cholesky(int Mc_AN, double**** OLP0, double** invS, int* MP, int NUM, double* LoS)
{
    char* UPLO = "U";
    INTEGER N, lda, info, lwork;
    int Gc_AN, i, j, k, Anum, Gi, wanA, time1, time2;
    int ig, ian, ih, kl, jg, jan, Bnum, m, n;
    double tmp0, tmp1;

    N = NUM;
    Gc_AN = M2G[Mc_AN];

    /* OLP0 to invS */

    for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {

        ig = natn[Gc_AN][i];
        ian = Spe_Total_CNO[WhatSpecies[ig]];
        Anum = MP[i] - 1;
        ih = S_G2M[ig];

        for (j = 0; j <= (FNAN[Gc_AN] + SNAN[Gc_AN]); j++) {

            kl = RMI1[Mc_AN][i][j];
            jg = natn[Gc_AN][j];
            jan = Spe_Total_CNO[WhatSpecies[jg]];
            Bnum = MP[j] - 1;

            if (0 <= kl) {
                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {
                        invS[Anum + m][Bnum + n] = OLP0[ih][kl][m][n];
                    }
                }
            }

            else {
                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {
                        invS[Anum + m][Bnum + n] = 0.0;
                    }
                }
            }
        }
    }

    /* invS -> LoS */

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            LoS[j * N + i] = invS[i][j];
        }
    }

    /* call dpotrf_() in clapack */

    lda = N;
    F77_NAME(dpotrf, DPOTRF)
    (UPLO, &N, LoS, &lda, &info);

    if (info != 0) {
        printf("Error in dpotrf_() which is called from Embedding_Cluster.c  info=%2d\n", info);
    }

    /* call dpotri_() in clapack */

    lwork = N;
    F77_NAME(dpotri, DPOTRI)
    (UPLO, &N, LoS, &lda, &info);

    if (info != 0) {
        printf("Error in dpotri_() which is called from Embedding_Cluster.c  info=%2d\n", info);
    }

    /* LoS -> invS */

    for (j = 0; j < N; j++) {
        m = j * N;
        for (i = 0; i <= j; i++) {
            invS[i][j] = LoS[m + i];
            invS[j][i] = LoS[m + i];
        }
    }
}

void Save_DOS_Col(double****** Residues, double**** OLP0, double*** EVal, int** LO_TC, int** HO_TC)
{
    int spin, Mc_AN, wanA, Gc_AN, tno1;
    int i1, i, j, MaxL, l, h_AN, Gh_AN, wanB, tno2;
    double Stime_atom, Etime_atom;
    double sum;
    int i_vec[10];
    char file_eig[YOUSO10], file_ev[YOUSO10];
    FILE *fp_eig, *fp_ev;
    int numprocs, myid, ID, tag;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    if (myid == Host_ID) {
        printf("The DOS is supported for a range from -10 to 10 eV for the O(N) Krylov subspace method.\n");
    }

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
            wanA = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[wanA];

            fprintf(fp_ev, "<AN%dAN%d\n", Gc_AN, spin);
            fprintf(fp_ev, "%d\n", (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1));

            for (i1 = 0; i1 < (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1); i1++) {

                fprintf(fp_ev, "%4d  %10.6f  ", i1, EVal[spin][Mc_AN][i1 - 1 + LO_TC[spin][Mc_AN]]);

                for (i = 0; i < tno1; i++) {

                    sum = 0.0;
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB = WhatSpecies[Gh_AN];
                        tno2 = Spe_Total_CNO[wanB];
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

    if ((fp_eig = fopen(file_eig, "w")) != NULL) {

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

static double Krylov_Col_trd(char* mode,
    int SCF_iter,
    double***** Hks,
    double**** OLP0,
    double***** CDM,
    double***** EDM,
    double Eele0[2],
    double Eele1[2])

/**********************************************************************************
  The structure of this subroutine is exactly the same as the original Krylov_Col,
  except for the main loop where OMP parallelization at the atom level is removed.
**********************************************************************************/

{
    static int firsttime = 1, recalc_firsttime = 1, recalc_flag;
    int Mc_AN, Gc_AN, i, is, js, Gi, wan, wanA, wanB, Anum;
    int num, NUM0, NUM, NUM1, n2, Cwan, Hwan, Rn2;
    int size1, size2, max_size1, max_size2;
    int ih, ig, ian, j, kl, jg, jan, Bnum, m, n, spin, i2, ip;
    int k, l, i1, j1, P_min, m_size, q1, q2, csize, Residues_size;
    int h_AN1, Mh_AN1, h_AN2, Gh_AN1, Gh_AN2, wan1, wan2;
    int po, po1, loopN, tno1, tno2, h_AN, Gh_AN, rl1, rl2, rl;
    int MA_AN, GA_AN, tnoA, GB_AN, tnoB, ct_on;
    int Msize2_max;
    static double TZ;
    double My_TZ, sum, FermiF, time0, srt;
    double sum00, sum10, sum20, sum30;
    double sum01, sum11, sum21, sum31;
    double sum02, sum12, sum22, sum32;
    double sum03, sum13, sum23, sum33;
    double tmp0, tmp1, tmp2, tmp3, b2, co, x0, Sx, Dx, xmin, xmax;
    double My_Num_State, Num_State, x, Dnum;
    double emin, emax, de;
    double TStime, TEtime, Stime1, Etime1;
    double Stime2, Etime2;
    double time1, time2, time3, time4, time5;
    double time6, time7, time8, time9, time10;
    double time11, time12, time13, time14, time15, time16;
    double Erange;
    double My_Eele0[2], My_Eele1[2];
    double max_x = 30.0;
    double ChemP_MAX, ChemP_MIN, spin_degeneracy;
    double spetrum_radius;
    double*** EVal;
    double****** Residues;
    double*** PDOS_DC;
    double* tmp_array;
    double* tmp_array2;
    int* MP;

    /*****************************************************
    Msize:   \sum_FNAN Spe_Total_CNO
    Msize2:  \sum_FNAN+SNAN Spe_Total_CNO
    Msize3:  rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]
    Msize4:  rlmax_EC2[Mc_AN]*EKC_core_size[Mc_AN]
    Msize5:  dimension for the last column of Residues
    *****************************************************/

    int* Msize;
    int* Msize2;
    int* Msize3;
    int* Msize4;
    int* Msize5;
    int** LO_TC;
    int** HO_TC;
    int numprocs, myid, ID, IDS, IDR, tag = 999;
    double Stime_atom, Etime_atom;

    MPI_Status stat;
    MPI_Request request;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    /* MPI */
    MPI_Barrier(mpi_comm_level1);
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    Krylov_GPU_InitOnce();
    Krylov_GPU_PreparePool(omp_get_max_threads());

    printf("ABC1 myid=%2d\n", myid);

    dtime(&TStime);

    if (measure_time == 1) {
        time1 = 0.0;
        time2 = 0.0;
        time3 = 0.0;
        time4 = 0.0;
        time5 = 0.0;
        time6 = 0.0;
        time7 = 0.0;
        time8 = 0.0;
        time9 = 0.0;
        time10 = 0.0;
        time11 = 0.0;
        time12 = 0.0;
        time13 = 0.0;
        time14 = 0.0;
        time15 = 0.0;
    }

    /****************************************************
                     allocation of arrays:
    ****************************************************/

    Msize = (int*)malloc(sizeof(int) * (Matomnum + 1));
    Msize2 = (int*)malloc(sizeof(int) * (Matomnum + 1));
    Msize3 = (int*)malloc(sizeof(int) * (Matomnum + 1));
    Msize4 = (int*)malloc(sizeof(int) * (Matomnum + 1));
    Msize2_max = 0;

    /* find Msize */

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

        if (Mc_AN == 0) {
            Msize[Mc_AN] = 1;
        } else {

            Gc_AN = M2G[Mc_AN];

            NUM = 0;
            for (i = 0; i <= FNAN[Gc_AN]; i++) {
                Gi = natn[Gc_AN][i];
                wanA = WhatSpecies[Gi];
                NUM += Spe_Total_CNO[wanA];
            }
            Msize[Mc_AN] = NUM;
        }
    }

    /* find Msize2 and Msize2_max */

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

        if (Mc_AN == 0) {
            Msize2[Mc_AN] = 1;
        } else {

            Gc_AN = M2G[Mc_AN];

            NUM = 0;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                Gi = natn[Gc_AN][i];
                wanA = WhatSpecies[Gi];
                NUM += Spe_Total_CNO[wanA];
            }
            Msize2[Mc_AN] = NUM;
        }

        if (Msize2_max < Msize2[Mc_AN])
            Msize2_max = Msize2[Mc_AN] + 4;
    }

    /* find Msize3 and Msize4 */

    Msize3[0] = 1;
    Msize4[0] = 1;

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        wan = WhatSpecies[Gc_AN];
        ct_on = Spe_Total_CNO[wan];
        Msize3[Mc_AN] = rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN];
        Msize4[Mc_AN] = rlmax_EC2[Mc_AN] * EKC_core_size[Mc_AN];
    }

    if (Krylov_GPU_Enabled()) {
        Krylov_KUCache_Prepare(Matomnum, SpinP_switch + 1, Msize, Msize3, SCF_iter);
    }

    m_size = 0;

    EVal = (double***)malloc(sizeof(double**) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        EVal[spin] = (double**)malloc(sizeof(double*) * (Matomnum + 1));

        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
            n2 = Msize3[Mc_AN] + 2;
            m_size += n2;
            EVal[spin][Mc_AN] = (double*)malloc(sizeof(double) * n2);
        }
    }

    if (firsttime) {
        PrintMemory("Krylov: EVal", sizeof(double) * m_size, NULL);
    }

    if (2 <= level_stdout) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            printf("<Krylov> myid=%4d Mc_AN=%4d Gc_AN=%4d Msize=%4d\n",
                myid, Mc_AN, M2G[Mc_AN], Msize[Mc_AN]);
        }
    }

    /****************************************************
      allocation of arrays:

      double PDOS[SpinP_switch+1]
                 [Matomnum+1]
                 [n2]
    ****************************************************/

    m_size = 0;
    PDOS_DC = (double***)malloc(sizeof(double**) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        PDOS_DC[spin] = (double**)malloc(sizeof(double*) * (Matomnum + 1));
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
            n2 = Msize3[Mc_AN] + 4;
            m_size += n2;
            PDOS_DC[spin][Mc_AN] = (double*)malloc(sizeof(double) * n2);
        }
    }

    if (firsttime) {
        PrintMemory("Krylov: PDOS_DC", sizeof(double) * m_size, NULL);
    }

    /****************************************************
      allocation of arrays:

      int LO_TC[SpinP_switch+1][Matomnum+1]
      int HO_TC[SpinP_switch+1][Matomnum+1]
    ****************************************************/

    LO_TC = (int**)malloc(sizeof(int*) * (SpinP_switch + 1));
    for (spin = 0; spin < (SpinP_switch + 1); spin++) {
        LO_TC[spin] = (int*)malloc(sizeof(int) * (Matomnum + 1));
    }

    HO_TC = (int**)malloc(sizeof(int*) * (SpinP_switch + 1));
    for (spin = 0; spin < (SpinP_switch + 1); spin++) {
        HO_TC[spin] = (int*)malloc(sizeof(int) * (Matomnum + 1));
    }

    /****************************************************
      allocation of array:

      double Residues[SpinP_switch+1]
                     [Matomnum+1]
                     [FNAN[Gc_AN]+1]
                     [Spe_Total_CNO[Gc_AN]]
                     [Spe_Total_CNO[Gh_AN]]
                     [HO_TC-LO_TC+3]
    ****************************************************/

    Residues = (double******)malloc(sizeof(double*****) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        Residues[spin] = (double*****)malloc(sizeof(double****) * (Matomnum + 1));
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN = 0;
                FNAN[0] = 0;
                tno1 = 1;
            } else {

                Gc_AN = M2G[Mc_AN];
                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];
            }

            Residues[spin][Mc_AN] = (double****)malloc(sizeof(double***) * (FNAN[Gc_AN] + 1));

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB = WhatSpecies[Gh_AN];
                    tno2 = Spe_Total_CNO[wanB];
                }

                Residues[spin][Mc_AN][h_AN] = (double***)malloc(sizeof(double**) * tno1);
                for (i = 0; i < tno1; i++) {
                    Residues[spin][Mc_AN][h_AN][i] = (double**)malloc(sizeof(double*) * tno2);
                    /* note that the array is allocated once more in the loop */
                }
            }
        }
    }

    for (spin = 0; spin <= SpinP_switch; spin++) {
        Residues[spin][0][0][0][0] = (double*)malloc(sizeof(double) * 1);
    }

    /****************************************************
       initialize density and energy density matrices
    ****************************************************/

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            Gc_AN = M2G[Mc_AN];
            wanA = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[wanA];
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                Gh_AN = natn[Gc_AN][h_AN];
                wanB = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_CNO[wanB];
                for (i = 0; i < tno1; i++) {
                    for (j = 0; j < tno2; j++) {
                        CDM[spin][Mc_AN][h_AN][i][j] = 0.0;
                        EDM[spin][Mc_AN][h_AN][i][j] = 0.0;
                    }
                }
            }
        }
    }

    /****************************************************
     MPI

     Hks
    ****************************************************/

    if (measure_time == 1)
        dtime(&Stime1);

    if (SCF_iter == 1) {

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
                            Cwan = WhatSpecies[Gc_AN];
                            tno1 = Spe_Total_CNO[Cwan];
                            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                                Gh_AN = natn[Gc_AN][h_AN];
                                Hwan = WhatSpecies[Gh_AN];
                                tno2 = Spe_Total_CNO[Hwan];
                                size1 += tno1 * tno2;
                            }
                        }
                    }

                    Snd_HFS_Size[IDS] = size1;
                    MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                } else {
                    Snd_HFS_Size[IDS] = 0;
                }

                /* receiving of size of data */

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                    MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                    Rcv_HFS_Size[IDR] = size2;
                } else {
                    Rcv_HFS_Size[IDR] = 0;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                    MPI_Wait(&request, &stat);
            } else {
                Snd_HFS_Size[IDS] = 0;
                Rcv_HFS_Size[IDR] = 0;
            }
        }
    }

    /***********************************
               data transfer
    ************************************/

    /* find maximum size of size1 */

    max_size1 = 0;
    max_size2 = 0;
    for (ID = 0; ID < numprocs; ID++) {
        size1 = Snd_HFS_Size[ID];
        if (max_size1 < size1)
            max_size1 = size1;
        size2 = Rcv_HFS_Size[ID];
        if (max_size2 < size2)
            max_size2 = size2;
    }

    /* allocation of arrays */

    tmp_array = (double*)malloc(sizeof(double) * max_size1);
    tmp_array2 = (double*)malloc(sizeof(double) * max_size2);

    /* MPI communication */

    tag = 999;
    for (ID = 0; ID < numprocs; ID++) {

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (ID != 0) {

            /*****************************
                    sending of data
            *****************************/

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                size1 = Snd_HFS_Size[IDS];

                /* multidimentional array to vector array */

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan = WhatSpecies[Gc_AN];
                        tno1 = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan = WhatSpecies[Gh_AN];
                            tno2 = Spe_Total_CNO[Hwan];
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

                size2 = Rcv_HFS_Size[IDR];

                MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
                    for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++) {
                        Mc_AN++;
                        Gc_AN = Rcv_GAN[IDR][n];
                        Cwan = WhatSpecies[Gc_AN];
                        tno1 = Spe_Total_CNO[Cwan];

                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan = WhatSpecies[Gh_AN];
                            tno2 = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    Hks[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }
                }
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                MPI_Wait(&request, &stat);
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

    if (SCF_iter == 1) {

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

                    size1 = Snd_HFS_Size[IDS] / (1 + SpinP_switch);

                    /* multidimentional array to vector array */

                    num = 0;

                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan = WhatSpecies[Gc_AN];
                        tno1 = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan = WhatSpecies[Gh_AN];
                            tno2 = Spe_Total_CNO[Hwan];
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

                    size2 = Rcv_HFS_Size[IDR] / (1 + SpinP_switch);

                    MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                    num = 0;
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
                    for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++) {
                        Mc_AN++;
                        Gc_AN = Rcv_GAN[IDR][n];
                        Cwan = WhatSpecies[Gc_AN];
                        tno1 = Spe_Total_CNO[Cwan];

                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan = WhatSpecies[Gh_AN];
                            tno2 = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    OLP0[Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
                }
            }
        }
    }

    /* freeing of arrays */
    free(tmp_array);
    free(tmp_array2);

    if (measure_time == 1) {
        dtime(&Etime1);
        time1 = Etime1 - Stime1;
    }

    /***********************************************
      for regeneration of the buffer matrix
    ***********************************************/

    if (sqrt(fabs(NormRD[0])) < (0.2 + 0.10 * (double)atomnum) && recalc_firsttime) {
        recalc_flag = 1;
        recalc_firsttime = 0;
    } else {
        recalc_flag = 0;
    }

    if (SCF_iter == 1)
        recalc_firsttime = 1;

    if (error_check == 1) {
        printf("SCF_iter=%2d recalc_firsttime=%2d\n", SCF_iter, recalc_firsttime);
    }

    if (measure_time == 1)
        dtime(&Stime2);

    /*
  #pragma omp parallel shared(List_YOUSO,Residues,EDM,CDM,HO_TC,LO_TC,ChemP,EVal,RMI1,S_G2M,EC_matrix,recalc_flag,recalc_EM,Krylov_U,SpinP_switch,EKC_core_size,rlmax_EC,rlmax_EC2,time11,time10,time9,time8,time7,time6,time5,time4,time3,time2,Hks,OLP0,EKC_Exact_invS_flag,SCF_iter,Msize3,Msize2,Msize,natn,FNAN,SNAN,Spe_Total_CNO,WhatSpecies,M2G,Matomnum,myid,time_per_atom,firsttime)
    */

    {
        int OMPID, Nthrds, Nprocs;
        int Mc_AN, Gc_AN, wan, spin;
        int ig, ian, ih, kl, jg, jan, Bnum, m, n, rl;
        int Anum, i, j, k, Gi, wanA, NUM, n2, csize, is, i2;
        int i1, rl1, js, ip, po1, tno1, h_AN, Gh_AN, wanB, tno2;
        int* MP;
        int KU_d1, KU_d2, lda, ldb, ldc, M, N, K;
        double alpha, beta;
        double** invS;
        double* LoS;
        double *C, *KU;
        double *H_DC, *ko;
        double*** Krylov_U_OLP;
        double** inv_RS;
        double sum00, sum10, sum20, sum30, sum;
        double tmp0, tmp1, tmp2, tmp3;
        double Erange;
        double Stime_atom, Etime_atom;
        double Stime1, Etime1;
        double** tmpvec0;
        double** tmpvec1;
        double** tmpvec2;
        Krylov_GPU_Workspace* gpu_ws;

        __m128d mmSum00, mmSum01, mmSum10, mmSum11;
        __m128d mmSum20, mmSum21, mmSum30, mmSum31;
        __m128d mmTmp0, mmTmp1, mmTmp2, mmTmp3;
        __m128d mmTmp4, mmTmp5;

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        gpu_ws = Krylov_GPU_GetWorkspace(OMPID);

        /*
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();
        */

        /* allocation of arrays */

        MP = (int*)malloc(sizeof(int) * List_YOUSO[2]);

        tmpvec0 = (double**)malloc(sizeof(double*) * EKC_core_size_max);
        for (i = 0; i < EKC_core_size_max; i++) {
            tmpvec0[i] = (double*)malloc(sizeof(double) * Msize2_max);
        }

        tmpvec1 = (double**)malloc(sizeof(double*) * EKC_core_size_max);
        for (i = 0; i < EKC_core_size_max; i++) {
            tmpvec1[i] = (double*)malloc(sizeof(double) * Msize2_max);
        }

        tmpvec2 = (double**)malloc(sizeof(double*) * EKC_core_size_max);
        for (i = 0; i < EKC_core_size_max; i++) {
            tmpvec2[i] = (double*)malloc(sizeof(double) * Msize2_max);
        }

        if (firsttime && OMPID == 0) {
            PrintMemory("Krylov: tmpvec0", sizeof(double) * EKC_core_size_max * Msize2_max, NULL);
            PrintMemory("Krylov: tmpvec1", sizeof(double) * EKC_core_size_max * Msize2_max, NULL);
            PrintMemory("Krylov: tmpvec2", sizeof(double) * EKC_core_size_max * Msize2_max, NULL);
        }

        /***********************************************
         main loop of calculation
        ***********************************************/

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            wan = WhatSpecies[Gc_AN];

            /* MP array */

            Anum = 1;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                MP[i] = Anum;
                Gi = natn[Gc_AN][i];
                wanA = WhatSpecies[Gi];
                Anum += Spe_Total_CNO[wanA];
            }
            NUM = Anum - 1;
            n2 = NUM + 40;

            /***********************************************
                       allocation of arrays:
            ***********************************************/

            if (Msize[Mc_AN] < Msize3[Mc_AN])
                csize = Msize3[Mc_AN] + 40;
            else
                csize = Msize[Mc_AN] + 40;

            KU_d1 = EKC_core_size[Mc_AN] * Msize2[Mc_AN];
            KU_d2 = Msize2[Mc_AN];

            H_DC = (double*)malloc(sizeof(double) * csize * csize);
            ko = (double*)malloc(sizeof(double) * csize);
            C = (double*)malloc(sizeof(double) * csize * csize);
            KU = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 2) * Msize3[Mc_AN]);

            /***********************************************
              calculate the inverse of overlap matrix
            ***********************************************/

            if (SCF_iter == 1 && Msize3[Mc_AN] < Msize2[Mc_AN] && EKC_Exact_invS_flag == 1) {

                LoS = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 3) * (Msize2[Mc_AN] + 3));

                invS = (double**)malloc(sizeof(double*) * (Msize2[Mc_AN] + 3));
                for (i = 0; i < (Msize2[Mc_AN] + 3); i++) {
                    invS[i] = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 3));
                }

                if (measure_time == 1)
                    dtime(&Stime1);

                Inverse_S_by_Cholesky(Mc_AN, OLP0, invS, MP, NUM, LoS);

                if (measure_time == 1) {
                    dtime(&Etime1);
                    time2 += Etime1 - Stime1;
                }
            }

            else if (SCF_iter == 1 && Msize3[Mc_AN] < Msize2[Mc_AN] && EKC_invS_flag == 1) {

                Krylov_U_OLP = (double***)malloc(sizeof(double**) * rlmax_EC2[Mc_AN]);
                for (i = 0; i < rlmax_EC2[Mc_AN]; i++) {
                    Krylov_U_OLP[i] = (double**)malloc(sizeof(double*) * EKC_core_size[Mc_AN]);
                    for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
                        Krylov_U_OLP[i][j] = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 3));
                        for (k = 0; k < (Msize2[Mc_AN] + 3); k++)
                            Krylov_U_OLP[i][j][k] = 0.0;
                    }
                }

                inv_RS = (double**)malloc(sizeof(double*) * (rlmax_EC2[Mc_AN] + 1) * EKC_core_size[Mc_AN]);
                for (i = 0; i < (rlmax_EC2[Mc_AN] + 1) * EKC_core_size[Mc_AN]; i++) {
                    inv_RS[i] = (double*)malloc(sizeof(double) * (rlmax_EC2[Mc_AN] + 1) * EKC_core_size[Mc_AN]);
                }

                if (measure_time == 1)
                    dtime(&Stime1);

                Krylov_IOLP_trd(Mc_AN, OLP0, Krylov_U_OLP, inv_RS, MP, Msize2, Msize4, Msize2_max, tmpvec0, tmpvec1);

                if (measure_time == 1) {
                    dtime(&Etime1);
                    time2 += Etime1 - Stime1;
                }
            }

            for (spin = 0; spin <= SpinP_switch; spin++) {

                /****************************************************
                        generate a preconditioning matrix
                ****************************************************/

                if (measure_time == 1)
                    dtime(&Stime1);

                if (SCF_iter == 1 && Msize3[Mc_AN] < Msize2[Mc_AN]) {

                    Generate_pMatrix_trd(myid, spin, Mc_AN, Hks, OLP0, invS, Krylov_U, Krylov_U_OLP, inv_RS, MP,
                        Msize, Msize2, Msize3, Msize4, Msize2_max, tmpvec0, tmpvec1, tmpvec2);
                } else if (SCF_iter == 1) {

                    Generate_pMatrix2_trd(myid, spin, Mc_AN, Hks, OLP0, Krylov_U, MP, Msize, Msize2, Msize3, tmpvec1);
                }

                if (measure_time == 1) {
                    dtime(&Etime1);
                    time3 += Etime1 - Stime1;
                }

                if (measure_time == 1)
                    dtime(&Stime1);

                if (recalc_EM == 1 || SCF_iter <= 3 || recalc_flag == 1) {

                    Embedding_Matrix_trd(spin, Mc_AN, Hks, Krylov_U, EC_matrix, MP, Msize, Msize2, Msize3, tmpvec1, EKC_core_size_max, Msize2_max);
                }

                if (measure_time == 1) {
                    dtime(&Etime1);
                    time4 += Etime1 - Stime1;
                }

                /****************************************************
                        construct the Hamiltonian matrix
                ****************************************************/

                if (measure_time == 1)
                    dtime(&Stime1);

                for (i = 0; i <= FNAN[Gc_AN]; i++) {
                    ig = natn[Gc_AN][i];
                    ian = Spe_Total_CNO[WhatSpecies[ig]];
                    Anum = MP[i] - 1;
                    ih = S_G2M[ig]; /* S_G2M should be used */

                    for (j = 0; j <= FNAN[Gc_AN]; j++) {

                        kl = RMI1[Mc_AN][i][j];
                        jg = natn[Gc_AN][j];
                        jan = Spe_Total_CNO[WhatSpecies[jg]];
                        Bnum = MP[j] - 1;

                        if (0 <= kl) {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    H_DC[(Anum + m) * Msize[Mc_AN] + Bnum + n] = Hks[spin][ih][kl][m][n];
                                }
                            }
                        }

                        else {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    H_DC[(Anum + m) * Msize[Mc_AN] + Bnum + n] = 0.0;
                                }
                            }
                        }
                    }
                }

                if (measure_time == 1) {
                    dtime(&Etime1);
                    time5 += Etime1 - Stime1;
                }

                /****************************************************
                           transform u1^+ * H_DC * u1
                ****************************************************/

                /* H_DC * u1 */

                if (measure_time == 1)
                    dtime(&Stime1);

                if (Krylov_ProjectedSolve_GPU(gpu_ws, spin, Mc_AN, Msize, Msize2, Msize3,
                                              Krylov_U, EC_matrix, H_DC, KU, C, ko) == 0) {

                /* original version

                for (i=1; i<=Msize[Mc_AN]; i++){
                  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
                    for (n=0; n<EKC_core_size[Mc_AN]; n++){

                      sum = 0.0;
                      for (j=1; j<=Msize[Mc_AN]; j++){
                        sum += H_DC[i][j]*Krylov_U[spin][Mc_AN][rl][n][j];
                      }

                      C[rl*EKC_core_size[Mc_AN]+n+1][i] = sum;
                    }
                  }
                }
                */

                /* BLAS3 version */

                for (i = 0; i < Msize[Mc_AN]; i++) {
                    for (j = 0; j < Msize3[Mc_AN]; j++) {
                        KU[j * Msize[Mc_AN] + i] = Krylov_U[spin][Mc_AN][j * Msize2[Mc_AN] + i + 1];
                    }
                }

                alpha = 1.0;
                beta = 0.0;
                M = Msize[Mc_AN];
                N = Msize3[Mc_AN];
                K = Msize[Mc_AN];
                lda = M;
                ldb = K;
                ldc = Msize[Mc_AN];

                Krylov_Dgemm(gpu_ws, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                             H_DC, lda, KU, ldb, C, ldc);

                if (measure_time == 1) {
                    dtime(&Etime1);
                    time6 += Etime1 - Stime1;
                }

                /* u1^+ * H_DC * u1 */

                if (measure_time == 1)
                    dtime(&Stime1);

                /* original version */

                /*
                for (rl1=0; rl1<rlmax_EC[Mc_AN]; rl1++){
                  for (m=0; m<EKC_core_size[Mc_AN]; m++){
                    for (rl2=rl1; rl2<rlmax_EC[Mc_AN]; rl2++){
                      for (n=0; n<EKC_core_size[Mc_AN]; n++){
                        sum = 0.0;
                        i2 = rl2*EKC_core_size[Mc_AN] + n + 1;
                        for (i=1; i<=Msize[Mc_AN]; i++){
                          sum += Krylov_U[spin][Mc_AN][rl1][m][i]*C[i2][i];
                        }

                        H_DC[rl1*EKC_core_size[Mc_AN]+m+1][rl2*EKC_core_size[Mc_AN]+n+1] = sum;
                        H_DC[rl2*EKC_core_size[Mc_AN]+n+1][rl1*EKC_core_size[Mc_AN]+m+1] = sum;
                      }
                    }
                  }
                }
                */

                /* BLAS3 version */

                alpha = 1.0;
                beta = 0.0;
                M = Msize3[Mc_AN];
                N = Msize3[Mc_AN];
                K = Msize[Mc_AN];
                lda = K;
                ldb = K;
                ldc = Msize3[Mc_AN];

                Krylov_Dgemm(gpu_ws, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                             KU, lda, C, ldb, H_DC, ldc);

                if (measure_time == 1) {
                    dtime(&Etime1);
                    time7 += Etime1 - Stime1;
                }

                /* correction for ZeroNum */

                m = (int)Krylov_U[spin][Mc_AN][0];
                for (i = 0; i < m; i++) {
                    H_DC[i * Msize3[Mc_AN] + i] = 1.0e+3;
                }

                /****************************************************
                    H0 = u1^+ * H_DC * u1 + D
                ****************************************************/

                if (measure_time == 1)
                    dtime(&Stime1);

                for (i = (Msize3[Mc_AN] - 1); 0 <= i; i--) {
                    for (j = 0; j < Msize3[Mc_AN]; j++) {
                        H_DC[(i + 1) * (Msize3[Mc_AN] + 1) + (j + 1)] = H_DC[i * Msize3[Mc_AN] + j] + EC_matrix[spin][Mc_AN][i + 1][j + 1];
                    }
                }

                if (measure_time == 1) {
                    dtime(&Etime1);
                    time8 += Etime1 - Stime1;
                }

                /****************************************************
                   diagonalize
                ****************************************************/

                if (measure_time == 1)
                    dtime(&Stime1);

                Krylov_Eigen2(gpu_ws, H_DC, Msize3[Mc_AN] + 1, ko,
                              Msize3[Mc_AN], Msize3[Mc_AN]);

                if (measure_time == 1) {
                    dtime(&Etime1);
                    time9 += Etime1 - Stime1;
                }

                /********************************************
                   back transformation of eigenvectors
                              c = u1 * b
                *********************************************/

                if (measure_time == 1)
                    dtime(&Stime1);

                /* original version */

                /*
                for (i=1; i<=Msize[Mc_AN]; i++){
                  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
                    for (n=0; n<EKC_core_size[Mc_AN]; n++){

                      tmp1 = Krylov_U[spin][Mc_AN][rl][n][i];
                      i1 = rl*EKC_core_size[Mc_AN] + n + 1;

                      for (j=1; j<=Msize3[Mc_AN]; j++){
                        C[i][j] += tmp1*H_DC[i1][j];
                      }
                    }
                  }
                }
                */

                /* BLAS3 version */

                alpha = 1.0;
                beta = 0.0;
                M = Msize3[Mc_AN];
                N = Msize[Mc_AN];
                K = Msize3[Mc_AN];
                lda = K;
                ldb = N;
                ldc = Msize3[Mc_AN];

                Krylov_Dgemm(gpu_ws, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
                             H_DC, lda, KU, ldb, C, ldc);

                if (measure_time == 1) {
                    dtime(&Etime1);
                    time10 += Etime1 - Stime1;
                }

                } /* fallback of Krylov_ProjectedSolve_GPU */

                if (measure_time == 1)
                    dtime(&Stime1);

                /***********************************************
                  store eigenvalues and residues of poles
                ***********************************************/

                for (i = 1; i <= Msize3[Mc_AN]; i++) {
                    EVal[spin][Mc_AN][i - 1] = ko[i];
                }

                /******************************************************
                set an energy range (-Erange+ChemP to Erange+ChemP)
                of eigenvalues used to store the Residues.
                ******************************************************/

                Erange = 0.367493245; /* in hartree, corresponds to 10 eV */

                /***********************************************
                  find LO_TC and HO_TC
                ***********************************************/

                /* LO_TC */
                i = 0;
                ip = 0;
                po1 = 0;
                do {
                    if ((ChemP - Erange) < EVal[spin][Mc_AN][i]) {
                        ip = i;
                        po1 = 1;
                    }
                    i++;
                } while (po1 == 0 && i < Msize3[Mc_AN]);

                LO_TC[spin][Mc_AN] = ip;

                /* HO_TC */
                i = 0;
                ip = Msize3[Mc_AN] - 1;
                po1 = 0;
                do {
                    if ((ChemP + Erange) < EVal[spin][Mc_AN][i]) {
                        ip = i;
                        po1 = 1;
                    }
                    i++;
                } while (po1 == 0 && i < Msize3[Mc_AN]);

                HO_TC[spin][Mc_AN] = ip;

                /***********************************************
                  store residues of poles
                ***********************************************/

                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];

                for (i = 0; i < tno1; i++) {
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB = WhatSpecies[Gh_AN];
                        tno2 = Spe_Total_CNO[wanB];
                        Bnum = MP[h_AN] - 1;
                        for (j = 0; j < tno2; j++) {

                            for (i1 = 0; i1 < LO_TC[spin][Mc_AN]; i1++) {
                                tmp1 = C[i * Msize3[Mc_AN] + i1] * C[(Bnum + j) * Msize3[Mc_AN] + i1];
                                CDM[spin][Mc_AN][h_AN][i][j] += tmp1;
                                EDM[spin][Mc_AN][h_AN][i][j] += tmp1 * EVal[spin][Mc_AN][i1];
                            }

                            /* <allocation of Residues */
                            n2 = HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 3;
                            Residues[spin][Mc_AN][h_AN][i][j] = (double*)malloc(sizeof(double) * n2);
                            /* allocation of Residues> */
                            for (i1 = LO_TC[spin][Mc_AN]; i1 <= HO_TC[spin][Mc_AN]; i1++) {
                                Residues[spin][Mc_AN][h_AN][i][j][i1 - LO_TC[spin][Mc_AN]]
                                    = C[i * Msize3[Mc_AN] + i1] * C[(Bnum + j) * Msize3[Mc_AN] + i1];
                            }
                        }
                    }
                }

                if (measure_time == 1) {
                    dtime(&Etime1);
                    time11 += Etime1 - Stime1;
                }

            } /* spin */

            /***********************************************
                          freeing of arrays:
            ***********************************************/

            free(H_DC);
            free(ko);
            free(C);
            free(KU);

            if (SCF_iter == 1 && Msize3[Mc_AN] < Msize2[Mc_AN] && EKC_Exact_invS_flag == 1) {

                free(LoS);

                for (i = 0; i < (Msize2[Mc_AN] + 3); i++) {
                    free(invS[i]);
                }
                free(invS);
            }

            else if (SCF_iter == 1 && Msize3[Mc_AN] < Msize2[Mc_AN] && EKC_invS_flag == 1) {

                for (i = 0; i < rlmax_EC2[Mc_AN]; i++) {
                    for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
                        free(Krylov_U_OLP[i][j]);
                    }
                    free(Krylov_U_OLP[i]);
                }
                free(Krylov_U_OLP);

                for (i = 0; i < (rlmax_EC2[Mc_AN] + 1) * EKC_core_size[Mc_AN]; i++) {
                    free(inv_RS[i]);
                }
                free(inv_RS);
            }

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

        } /* Mc_AN */

        /* freeing of array */

        free(MP);

        for (i = 0; i < EKC_core_size_max; i++) {
            free(tmpvec0[i]);
        }
        free(tmpvec0);

        for (i = 0; i < EKC_core_size_max; i++) {
            free(tmpvec1[i]);
        }
        free(tmpvec1);

        for (i = 0; i < EKC_core_size_max; i++) {
            free(tmpvec2[i]);
        }
        free(tmpvec2);

    } /* #pragma omp parallel */

    if (measure_time == 1) {
        dtime(&Etime2);
        time16 = Etime2 - Stime2;
    }

    if (firsttime) {

        Residues_size = 1;

        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
                Gc_AN = M2G[Mc_AN];
                wan = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wan];
                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB = WhatSpecies[Gh_AN];
                    tno2 = Spe_Total_CNO[wanB];
                    n2 = HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 3;
                    Residues_size += tno1 * tno2 * n2;
                }
            }
        }

        PrintMemory("Krylov: Residues", sizeof(double) * Residues_size, NULL);
    }

    /****************************************************
                  calculate the projected DOS
    ****************************************************/

    if (measure_time == 1)
        dtime(&Stime1);

    /*
  #pragma omp parallel shared(time_per_atom,Residues,LO_TC,HO_TC,EDM,CDM,OLP0,natn,FNAN,PDOS_DC,Msize3,Spe_Total_CNO,WhatSpecies,M2G,SpinP_switch,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,spin,Gc_AN,wanA,tno1,i1,i,h_AN,Gh_AN,wanB,tno2,j,tmp1,Etime_atom)
    */

    {

        /* get info. on OpenMP */

        /*
        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();
        */

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

            dtime(&Stime_atom);

            for (spin = 0; spin <= SpinP_switch; spin++) {

                Gc_AN = M2G[Mc_AN];
                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];

                for (i1 = 0; i1 <= (Msize3[Mc_AN] + 1); i1++) {
                    PDOS_DC[spin][Mc_AN][i1] = 0.0;
                }

                for (i = 0; i < tno1; i++) {

                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB = WhatSpecies[Gh_AN];
                        tno2 = Spe_Total_CNO[wanB];
                        for (j = 0; j < tno2; j++) {

                            tmp1 = OLP0[Mc_AN][h_AN][i][j];

                            PDOS_DC[spin][Mc_AN][0] += tmp1 * CDM[spin][Mc_AN][h_AN][i][j];
                            PDOS_DC[spin][Mc_AN][1] += tmp1 * EDM[spin][Mc_AN][h_AN][i][j];

                            for (i1 = 0; i1 < (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1); i1++) {
                                PDOS_DC[spin][Mc_AN][i1 + 2] += Residues[spin][Mc_AN][h_AN][i][j][i1] * tmp1;
                            }
                        }
                    }
                }

            } /* spin */

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

        } /* Mc_AN */

    } /* #pragma omp parallel */

    if (measure_time == 1) {
        dtime(&Etime1);
        time12 += Etime1 - Stime1;
    }

    /****************************************************
             find the total number of electrons
    ****************************************************/

    if (measure_time == 1)
        dtime(&Stime1);

    My_TZ = 0.0;
    for (i = 1; i <= Matomnum; i++) {
        Gc_AN = M2G[i];
        wan = WhatSpecies[Gc_AN];
        My_TZ += Spe_Core_Charge[wan];
    }

    /* MPI, My_TZ */

    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_TZ, &TZ, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /****************************************************
                  find the chemical potential
    ****************************************************/

    po = 0;
    loopN = 0;

    ChemP_MAX = 10.0;
    ChemP_MIN = -10.0;
    if (SpinP_switch == 0)
        spin_degeneracy = 2.0;
    else if (SpinP_switch == 1)
        spin_degeneracy = 1.0;

    do {
        ChemP = 0.50 * (ChemP_MAX + ChemP_MIN);

        My_Num_State = 0.0;

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            for (spin = 0; spin <= SpinP_switch; spin++) {

                dtime(&Stime_atom);

                Gc_AN = M2G[Mc_AN];

                My_Num_State += spin_degeneracy * PDOS_DC[spin][Mc_AN][0];

                for (i = 0; i < (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1); i++) {

                    x = (EVal[spin][Mc_AN][i + LO_TC[spin][Mc_AN]] - ChemP) * Beta;
                    if (x <= -max_x)
                        x = -max_x;
                    if (max_x <= x)
                        x = max_x;
                    FermiF = 1.0 / (1.0 + exp(x));
                    My_Num_State += spin_degeneracy * FermiF * PDOS_DC[spin][Mc_AN][i + 2];
                }

                dtime(&Etime_atom);
                time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
            }
        }

        /* MPI, My_Num_State */

        MPI_Barrier(mpi_comm_level1);
        MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

        Dnum = (TZ - Num_State) - system_charge;
        if (0.0 <= Dnum)
            ChemP_MIN = ChemP;
        else
            ChemP_MAX = ChemP;
        if (fabs(Dnum) < 1.0e-11)
            po = 1;

        if (myid == Host_ID && 3 <= level_stdout) {
            printf("  ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n", ChemP, TZ, Num_State);
        }

        loopN++;
    } while (po == 0 && loopN < 1000);

    if (measure_time == 1) {
        dtime(&Etime1);
        time13 += Etime1 - Stime1;
    }

    /****************************************************
          eigenenergy by summing up eigenvalues
    ****************************************************/

    if (measure_time == 1)
        dtime(&Stime1);

    My_Eele0[0] = 0.0;
    My_Eele0[1] = 0.0;
    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            My_Eele0[spin] += PDOS_DC[spin][Mc_AN][1];

            for (i = 0; i < (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1); i++) {

                x = (EVal[spin][Mc_AN][i + LO_TC[spin][Mc_AN]] - ChemP) * Beta;

                if (x <= -max_x)
                    x = -max_x;
                if (max_x <= x)
                    x = max_x;
                FermiF = 1.0 / (1.0 + exp(x));
                My_Eele0[spin] += FermiF * EVal[spin][Mc_AN][i + LO_TC[spin][Mc_AN]] * PDOS_DC[spin][Mc_AN][i + 2];
            }

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
        }
    }

    /* MPI, My_Eele0 */
    for (spin = 0; spin <= SpinP_switch; spin++) {
        MPI_Barrier(mpi_comm_level1);
        MPI_Allreduce(&My_Eele0[spin], &Eele0[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }

    if (SpinP_switch == 0) {
        Eele0[1] = Eele0[0];
    }

    if (measure_time == 1) {
        dtime(&Etime1);
        time14 += Etime1 - Stime1;
    }

    if (measure_time == 1)
        dtime(&Stime1);

    /*
  #pragma omp parallel shared(time_per_atom,EDM,CDM,Residues,natn,max_x,Beta,ChemP,EVal,LO_TC,HO_TC,Spe_Total_CNO,WhatSpecies,M2G,SpinP_switch,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,spin,Stime_atom,Gc_AN,wanA,tno1,i1,x,FermiF,h_AN,Gh_AN,wanB,tno2,i,j,tmp1,Etime_atom)
    */

    {

        /* get info. on OpenMP */

        /*
        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();
        */

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            for (spin = 0; spin <= SpinP_switch; spin++) {

                dtime(&Stime_atom);

                Gc_AN = M2G[Mc_AN];
                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];

                for (i1 = 0; i1 < (HO_TC[spin][Mc_AN] - LO_TC[spin][Mc_AN] + 1); i1++) {

                    x = (EVal[spin][Mc_AN][i1 + LO_TC[spin][Mc_AN]] - ChemP) * Beta;
                    if (x <= -max_x)
                        x = -max_x;
                    if (max_x <= x)
                        x = max_x;
                    FermiF = 1.0 / (1.0 + exp(x));

                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB = WhatSpecies[Gh_AN];
                        tno2 = Spe_Total_CNO[wanB];
                        for (i = 0; i < tno1; i++) {
                            for (j = 0; j < tno2; j++) {
                                tmp1 = FermiF * Residues[spin][Mc_AN][h_AN][i][j][i1];
                                CDM[spin][Mc_AN][h_AN][i][j] += tmp1;
                                EDM[spin][Mc_AN][h_AN][i][j] += tmp1 * EVal[spin][Mc_AN][i1 + LO_TC[spin][Mc_AN]];
                            }
                        }
                    }
                }

                dtime(&Etime_atom);
                time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
            }
        }

    } /* #pragma omp parallel */

    /****************************************************
                        bond energies
    ****************************************************/

    My_Eele1[0] = 0.0;
    My_Eele1[1] = 0.0;
    for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        GA_AN = M2G[MA_AN];
        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];

        for (j = 0; j <= FNAN[GA_AN]; j++) {
            GB_AN = natn[GA_AN][j];
            wanB = WhatSpecies[GB_AN];
            tnoB = Spe_Total_CNO[wanB];

            for (k = 0; k < tnoA; k++) {
                for (l = 0; l < tnoB; l++) {
                    for (spin = 0; spin <= SpinP_switch; spin++) {
                        My_Eele1[spin] += CDM[spin][MA_AN][j][k][l] * Hks[spin][MA_AN][j][k][l];
                    }
                }
            }
        }
    }

    /* MPI, My_Eele1 */
    MPI_Barrier(mpi_comm_level1);
    for (spin = 0; spin <= SpinP_switch; spin++) {
        MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE,
            MPI_SUM, mpi_comm_level1);
    }

    if (SpinP_switch == 0) {
        Eele1[1] = Eele1[0];
    }

    if (3 <= level_stdout && myid == Host_ID) {
        printf("  Eele00=%15.12f Eele01=%15.12f\n", Eele0[0], Eele0[1]);
        printf("  Eele10=%15.12f Eele11=%15.12f\n", Eele1[0], Eele1[1]);
    }

    if (measure_time == 1) {
        dtime(&Etime1);
        time15 += Etime1 - Stime1;
    }

    if (strcasecmp(mode, "dos") == 0) {
        Save_DOS_Col(Residues, OLP0, EVal, LO_TC, HO_TC);
    }

    if (measure_time == 1) {
        printf("myid=%2d time1 =%5.3f time2 =%5.3f time3 =%5.3f time4 =%5.3f time5 =%5.3f\n",
            myid, time1, time2, time3, time4, time5);
        printf("myid=%2d time6 =%5.3f time7 =%5.3f time8 =%5.3f time9 =%5.3f time10=%5.3f\n",
            myid, time6, time7, time8, time9, time10);
        printf("myid=%2d time11=%5.3f time12=%5.3f time13=%5.3f time14=%5.3f time15=%5.3f\n",
            myid, time11, time12, time13, time14, time15);
        printf("myid=%2d time16=%5.3f\n", myid, time16);
    }

    /****************************************************
      freeing of arrays:

    ****************************************************/

    free(Msize);
    free(Msize2);
    free(Msize3);
    free(Msize4);

    for (spin = 0; spin < (SpinP_switch + 1); spin++) {
        free(LO_TC[spin]);
    }
    free(LO_TC);

    for (spin = 0; spin < (SpinP_switch + 1); spin++) {
        free(HO_TC[spin]);
    }
    free(HO_TC);

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
                Gc_AN = 0;
                FNAN[0] = 0;
                tno1 = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];
            }

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB = WhatSpecies[Gh_AN];
                    tno2 = Spe_Total_CNO[wanB];
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

    if (measure_time == 1) {
        printf("total time=%15.12f\n", time0);
    }

    /* for PrintMemory */
    firsttime = 0;

    return time0;
}

void Generate_pMatrix_trd(int myid, int spin, int Mc_AN, double***** Hks,
    double**** OLP0, double** invS,
    double*** Krylov_U, double*** Krylov_U_OLP, double** inv_RS, int* MP,
    int* Msize, int* Msize2, int* Msize3, int* Msize4, int Msize2_max,
    double** tmpvec0, double** tmpvec1, double** tmpvec2)

/* This subroutine is exactly the same as the original Generate_pMatrix,
except for the OMP parallelized loops in orthogonalization by diagonalization,
and Eigen_lapack is replaced with Eigen_lapack_d, _x, or _r
*/

{
    int rl, rl0, rl1, ct_AN, fan, san, can, wan, ct_on, i, j;
    int n, Anum, Bnum, k, ian, ih, kl, jg, ig, jan, m, m1, n1;
    int ZeroNum, Gh_AN, wanB, m1s, is, info;
    int rl00, rl01, rl02, rl03, rl04, rl05, rl06, rl07;
    int mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7;

    int KU_d1, KU_d2, csize;
    __m128d mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5;

    double mmArr[8];
    double time1, time2, time3, time4, time5;
    double time6, time7, time8, time9, time10;
    double Stime1, Etime1;
    double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
    double sum, dum, tmp0, tmp1, tmp2, tmp3, rcutA, r0;
    double **Utmp, **matRS0, **matRS1;
    double** tmpmat0;
    double *ko, *iko;
    double** FS;
    double*** U0;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    ct_AN = M2G[Mc_AN];
    fan = FNAN[ct_AN];
    san = SNAN[ct_AN];
    can = fan + san;
    wan = WhatSpecies[ct_AN];
    ct_on = Spe_Total_CNO[wan];
    rcutA = Spe_Atom_Cut1[wan];

    if (Msize[Mc_AN] < Msize3[Mc_AN])
        csize = Msize3[Mc_AN] + 40;
    else
        csize = Msize[Mc_AN] + 40;

    KU_d1 = EKC_core_size[Mc_AN] * Msize2[Mc_AN];
    KU_d2 = Msize2[Mc_AN];

    /* allocation of arrays */

    Utmp = (double**)malloc(sizeof(double*) * rlmax_EC[Mc_AN]);
    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        Utmp[i] = (double*)malloc(sizeof(double) * EKC_core_size[Mc_AN]);
    }

    U0 = (double***)malloc(sizeof(double**) * rlmax_EC[Mc_AN]);
    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        U0[i] = (double**)malloc(sizeof(double*) * EKC_core_size[Mc_AN]);
        for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
            U0[i][j] = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 3));
            for (k = 0; k < (Msize2[Mc_AN] + 3); k++)
                U0[i][j][k] = 0.0;
        }
    }

    tmpmat0 = (double**)malloc(sizeof(double*) * (EKC_core_size[Mc_AN] + 4));
    for (i = 0; i < (EKC_core_size[Mc_AN] + 4); i++) {
        tmpmat0[i] = (double*)malloc(sizeof(double) * (EKC_core_size[Mc_AN] + 4));
    }

    FS = (double**)malloc(sizeof(double*) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    for (i = 0; i < (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        FS[i] = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    }

    ko = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    iko = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);

    matRS0 = (double**)malloc(sizeof(double*) * (EKC_core_size[Mc_AN] + 2));
    for (i = 0; i < (EKC_core_size[Mc_AN] + 2); i++) {
        matRS0[i] = (double*)malloc(sizeof(double) * (Msize4[Mc_AN] + 3));
    }

    matRS1 = (double**)malloc(sizeof(double*) * (Msize4[Mc_AN] + 3));
    for (i = 0; i < (Msize4[Mc_AN] + 3); i++) {
        matRS1[i] = (double*)malloc(sizeof(double) * (EKC_core_size[Mc_AN] + 2));
    }

    /****************************************************
        initialize
    ****************************************************/

    if (measure_time == 1) {
        time1 = 0.0;
        time2 = 0.0;
        time3 = 0.0;
        time4 = 0.0;
        time5 = 0.0;
        time6 = 0.0;
        time7 = 0.0;
        time8 = 0.0;
        time9 = 0.0;
        time10 = 0.0;
    }

    if (measure_time == 1)
        dtime(&Stime1);

    for (i = 0; i < EKC_core_size_max; i++) {
        for (j = 0; j < Msize2_max; j++) {
            tmpvec0[i][j] = 0.0;
            tmpvec1[i][j] = 0.0;
        }
    }

    /* find the nearest atom with distance of r0 */

    r0 = 1.0e+10;
    for (k = 1; k <= FNAN[ct_AN]; k++) {
        Gh_AN = natn[ct_AN][k];
        wanB = WhatSpecies[Gh_AN];
        if (Dis[ct_AN][k] < r0)
            r0 = Dis[ct_AN][k];
    }

    /* starting vector */

    m = 0;
    for (k = 0; k <= FNAN[ct_AN]; k++) {

        Gh_AN = natn[ct_AN][k];
        wanB = WhatSpecies[Gh_AN];

        if (Dis[ct_AN][k] < (scale_rc_EKC[Mc_AN] * r0)) {

            Anum = MP[k] - 1;

            for (i = 0; i < Spe_Total_CNO[wanB]; i++) {

                tmpvec0[m][Anum + i] = 1.0;

                m++;
            }
        }
    }

    S_orthonormalize_vec_trd(Mc_AN, ct_on, tmpvec0, tmpvec1, OLP0, tmpmat0, ko, iko, MP, Msize2);

    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
        for (i = 0; i < Msize2[Mc_AN]; i++) {
            U0[0][n][i] = tmpvec0[n][i];
        }
    }

    if (measure_time == 1) {
        dtime(&Etime1);
        time1 = Etime1 - Stime1;
    }

    /****************************************************
             generate Krylov subspace vectors
    ****************************************************/

    for (rl = 0; rl < (rlmax_EC[Mc_AN] - 1); rl++) {

        if (measure_time == 1)
            dtime(&Stime1);

        /*******************************************************
                                H * |Wn)
        *******************************************************/

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec1[n][i] = 0.0;
            }
        }

        for (i = 0; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = 0; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j] - 1;

                if (0 <= kl) {

#ifndef __x86_64__

                    /* original version */

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += Hks[spin][ih][kl][m][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec1[n][Anum + m] += sum;
                        }
                    }

#else
                    /* Loop Unrolling + SSE version */

                    for (m = 0; m < (ian - 3); m += 4) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            mmSum00 = _mm_setzero_pd();
                            mmSum01 = _mm_setzero_pd();
                            mmSum10 = _mm_setzero_pd();
                            mmSum11 = _mm_setzero_pd();
                            mmSum20 = _mm_setzero_pd();
                            mmSum21 = _mm_setzero_pd();
                            mmSum30 = _mm_setzero_pd();
                            mmSum31 = _mm_setzero_pd();

                            for (k = 0; k < (jan - 3); k += 4) {
                                mmTmp0 = _mm_loadu_pd(&tmpvec0[n][Bnum + k + 0]);
                                mmTmp1 = _mm_loadu_pd(&tmpvec0[n][Bnum + k + 2]);

                                mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 0]), mmTmp0));
                                mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 2]), mmTmp1));

                                mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 0]), mmTmp0));
                                mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 2]), mmTmp1));

                                mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 0]), mmTmp0));
                                mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 2]), mmTmp1));

                                mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 0]), mmTmp0));
                                mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 2]), mmTmp1));
                            }

                            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                            _mm_storeu_pd(&mmArr[0], mmSum00);
                            _mm_storeu_pd(&mmArr[2], mmSum10);
                            _mm_storeu_pd(&mmArr[4], mmSum20);
                            _mm_storeu_pd(&mmArr[6], mmSum30);

                            sum0 = mmArr[0] + mmArr[1];
                            sum1 = mmArr[2] + mmArr[3];
                            sum2 = mmArr[4] + mmArr[5];
                            sum3 = mmArr[6] + mmArr[7];

                            for (; k < jan; k++) {
                                sum0 += Hks[spin][ih][kl][m + 0][k] * tmpvec0[n][Bnum + k];
                                sum1 += Hks[spin][ih][kl][m + 1][k] * tmpvec0[n][Bnum + k];
                                sum2 += Hks[spin][ih][kl][m + 2][k] * tmpvec0[n][Bnum + k];
                                sum3 += Hks[spin][ih][kl][m + 3][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec1[n][Anum + m + 0] += sum0;
                            tmpvec1[n][Anum + m + 1] += sum1;
                            tmpvec1[n][Anum + m + 2] += sum2;
                            tmpvec1[n][Anum + m + 3] += sum3;
                        }
                    }

                    for (; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += Hks[spin][ih][kl][m][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
#endif
                }
            }
        }

        if (measure_time == 1) {
            dtime(&Etime1);
            time2 += Etime1 - Stime1;
        }

        /*******************************************************
               S^{-1} * H * |Wn)
        *******************************************************/

        if (EKC_Exact_invS_flag == 1) {

            if (measure_time == 1)
                dtime(&Stime1);

#ifndef __x86_64__

            /* original version */
            /*
            for (n=0; n<EKC_core_size[Mc_AN]; n++){
              for (i=0; i<Msize2[Mc_AN]; i++){
                sum = 0.0;
                for (j=0; j<Msize2[Mc_AN]; j++){
                  sum += invS[i][j]*tmpvec1[n][j];
                }
                tmpvec0[n][i] = sum;
              }
            }
            */
            /* unrolling version */

            for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    sum0 = 0.0;
                    sum1 = 0.0;
                    sum2 = 0.0;
                    sum3 = 0.0;

                    for (j = 0; j < Msize2[Mc_AN]; j++) {
                        sum0 += invS[i + 0][j] * tmpvec1[n][j];
                        sum1 += invS[i + 1][j] * tmpvec1[n][j];
                        sum2 += invS[i + 2][j] * tmpvec1[n][j];
                        sum3 += invS[i + 3][j] * tmpvec1[n][j];
                    }

                    tmpvec0[n][i + 0] = sum0;
                    tmpvec0[n][i + 1] = sum1;
                    tmpvec0[n][i + 2] = sum2;
                    tmpvec0[n][i + 3] = sum3;
                }
            }

            is = Msize2[Mc_AN] - Msize2[Mc_AN] % 4;

            for (i = is; i < Msize2[Mc_AN]; i++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (j = 0; j < Msize2[Mc_AN]; j++) {
                        sum += invS[i][j] * tmpvec1[n][j];
                    }
                    tmpvec0[n][i] = sum;
                }
            }

#else
            /* unrolling + SSE version */

            for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    mmSum00 = _mm_setzero_pd();
                    mmSum01 = _mm_setzero_pd();
                    mmSum10 = _mm_setzero_pd();
                    mmSum11 = _mm_setzero_pd();
                    mmSum20 = _mm_setzero_pd();
                    mmSum21 = _mm_setzero_pd();
                    mmSum30 = _mm_setzero_pd();
                    mmSum31 = _mm_setzero_pd();

                    for (j = 0; j < (Msize2[Mc_AN] - 3); j += 4) {
                        mmTmp0 = _mm_loadu_pd(&tmpvec1[n][j + 0]);
                        mmTmp1 = _mm_loadu_pd(&tmpvec1[n][j + 2]);

                        mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&invS[i + 0][j + 0]), mmTmp0));
                        mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&invS[i + 0][j + 2]), mmTmp1));

                        mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&invS[i + 1][j + 0]), mmTmp0));
                        mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&invS[i + 1][j + 2]), mmTmp1));

                        mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&invS[i + 2][j + 0]), mmTmp0));
                        mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&invS[i + 2][j + 2]), mmTmp1));

                        mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&invS[i + 3][j + 0]), mmTmp0));
                        mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&invS[i + 3][j + 2]), mmTmp1));
                    }

                    mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                    mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                    mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                    mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                    _mm_storeu_pd(&mmArr[0], mmSum00);
                    _mm_storeu_pd(&mmArr[2], mmSum10);
                    _mm_storeu_pd(&mmArr[4], mmSum20);
                    _mm_storeu_pd(&mmArr[6], mmSum30);

                    sum0 = mmArr[0] + mmArr[1];
                    sum1 = mmArr[2] + mmArr[3];
                    sum2 = mmArr[4] + mmArr[5];
                    sum3 = mmArr[6] + mmArr[7];

                    for (; j < Msize2[Mc_AN]; j++) {
                        sum0 += invS[i + 0][j] * tmpvec1[n][j];
                        sum1 += invS[i + 1][j] * tmpvec1[n][j];
                        sum2 += invS[i + 2][j] * tmpvec1[n][j];
                        sum3 += invS[i + 3][j] * tmpvec1[n][j];
                    }

                    tmpvec0[n][i + 0] = sum0;
                    tmpvec0[n][i + 1] = sum1;
                    tmpvec0[n][i + 2] = sum2;
                    tmpvec0[n][i + 3] = sum3;
                }
            }

            is = Msize2[Mc_AN] - Msize2[Mc_AN] % 4;

            for (i = is; i < Msize2[Mc_AN]; i++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (j = 0; j < Msize2[Mc_AN]; j++) {
                        sum += invS[i][j] * tmpvec1[n][j];
                    }
                    tmpvec0[n][i] = sum;
                }
            }

#endif

            if (measure_time == 1) {
                dtime(&Etime1);
                time3 += Etime1 - Stime1;
            }
        }

        /*******************************************************
              U * RS^-1 * U^+ * H * |Wn)
        *******************************************************/

        else if (EKC_invS_flag == 1) {

            /*  U^+ * H * |Wn) */

            for (rl0 = 0; rl0 < rlmax_EC2[Mc_AN]; rl0++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    for (m = 0; m < EKC_core_size[Mc_AN]; m++) {

                        sum = 0.0;
                        for (i = 0; i < Msize2[Mc_AN]; i++) {
                            sum += Krylov_U_OLP[rl0][n][i] * tmpvec1[m][i];
                        }

                        /* transpose the later calcualtion */
                        matRS0[m][rl0 * EKC_core_size[Mc_AN] + n] = sum;
                    }
                }
            }

            /*  RS^-1 * U^+ * H * |Wn) */

            for (rl0 = 0; rl0 < rlmax_EC2[Mc_AN]; rl0++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    for (m = 0; m < EKC_core_size[Mc_AN]; m++) {

                        sum = 0.0;
                        for (i = 0; i < Msize4[Mc_AN]; i++) {
                            sum += inv_RS[rl0 * EKC_core_size[Mc_AN] + n][i] * matRS0[m][i];
                        }

                        matRS1[rl0 * EKC_core_size[Mc_AN] + n][m] = sum;
                    }
                }
            }

            /*  U * RS^-1 * U^+ * H * |Wn) */

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    tmpvec0[n][i] = 0.0;
                }
            }

            for (rl0 = 0; rl0 < rlmax_EC2[Mc_AN]; rl0++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                        tmp0 = matRS1[rl0 * EKC_core_size[Mc_AN] + n][m];
                        for (i = 0; i < Msize2[Mc_AN]; i++) {
                            tmpvec0[m][i] += Krylov_U_OLP[rl0][n][i] * tmp0;
                        }
                    }
                }
            }
        }

        else {
            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    tmpvec0[n][i] = tmpvec1[n][i];
                }
            }
        }

        if (measure_time == 1)
            dtime(&Stime1);

        /*************************************************************
         S-orthogonalization by a classical block Gram-Schmidt method
        *************************************************************/

        /* |tmpvec2) = S * |tmpvec0) */

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec2[n][i] = 0.0;
            }
        }

        for (i = 0; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = 0; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j] - 1;

                if (0 <= kl) {

#ifndef __x86_64__

                    /* Original version */

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec2[n][Anum + m] += sum;
                        }
                    }

#else

                    /* Unrolling + SSE version */

                    for (m = 0; m < (ian - 3); m += 4) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            mmSum00 = _mm_setzero_pd();
                            mmSum01 = _mm_setzero_pd();
                            mmSum10 = _mm_setzero_pd();
                            mmSum11 = _mm_setzero_pd();
                            mmSum20 = _mm_setzero_pd();
                            mmSum21 = _mm_setzero_pd();
                            mmSum30 = _mm_setzero_pd();
                            mmSum31 = _mm_setzero_pd();

                            for (k = 0; k < (jan - 3); k += 4) {
                                mmTmp0 = _mm_loadu_pd(&tmpvec0[n][Bnum + k + 0]);
                                mmTmp1 = _mm_loadu_pd(&tmpvec0[n][Bnum + k + 2]);

                                mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 0]), mmTmp0));
                                mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 2]), mmTmp1));

                                mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 0]), mmTmp0));
                                mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 2]), mmTmp1));

                                mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 0]), mmTmp0));
                                mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 2]), mmTmp1));

                                mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 0]), mmTmp0));
                                mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 2]), mmTmp1));
                            }

                            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                            _mm_storeu_pd(&mmArr[0], mmSum00);
                            _mm_storeu_pd(&mmArr[2], mmSum10);
                            _mm_storeu_pd(&mmArr[4], mmSum20);
                            _mm_storeu_pd(&mmArr[6], mmSum30);

                            sum0 = mmArr[0] + mmArr[1];
                            sum1 = mmArr[2] + mmArr[3];
                            sum2 = mmArr[4] + mmArr[5];
                            sum3 = mmArr[6] + mmArr[7];

                            for (; k < jan; k++) {
                                sum0 += OLP0[ih][kl][m + 0][k] * tmpvec0[n][Bnum + k];
                                sum1 += OLP0[ih][kl][m + 1][k] * tmpvec0[n][Bnum + k];
                                sum2 += OLP0[ih][kl][m + 2][k] * tmpvec0[n][Bnum + k];
                                sum3 += OLP0[ih][kl][m + 3][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec2[n][Anum + m + 0] += sum0;
                            tmpvec2[n][Anum + m + 1] += sum1;
                            tmpvec2[n][Anum + m + 2] += sum2;
                            tmpvec2[n][Anum + m + 3] += sum3;
                        }
                    }

                    for (; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec2[n][Anum + m] += sum;
                        }
                    }

#endif
                }
            }
        }

        if (measure_time == 1) {
            dtime(&Etime1);
            time4 += Etime1 - Stime1;
        }

        if (measure_time == 1)
            dtime(&Stime1);

#ifndef __x86_64__

        /* Original version */

        for (rl0 = 0; rl0 <= rl; rl0++) {

            /* (U_rl0|tmpvec2) */

            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += U0[rl0][m][i] * tmpvec2[n][i];
                    }
                    tmpmat0[m][n] = sum;
                }
            }

            /* |tmpvec0) - |U_rl0) * (U_rl0|tmpvec2) */

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (k = 0; k < EKC_core_size[Mc_AN]; k++) {
                    dum = tmpmat0[k][n];
                    for (i = 0; i < Msize2[Mc_AN]; i++)
                        tmpvec0[n][i] -= U0[rl0][k][i] * dum;
                }
            }
        }

#else

        /* Unrolling + SSE version */

        for (rl0 = 0; rl0 <= rl; rl0++) {

            /* (U_rl0|tmpvec2) */

            for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    mmSum00 = _mm_setzero_pd();
                    mmSum01 = _mm_setzero_pd();
                    mmSum10 = _mm_setzero_pd();
                    mmSum11 = _mm_setzero_pd();
                    mmSum20 = _mm_setzero_pd();
                    mmSum21 = _mm_setzero_pd();
                    mmSum30 = _mm_setzero_pd();
                    mmSum31 = _mm_setzero_pd();

                    for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                        mmTmp0 = _mm_loadu_pd(&tmpvec2[n][i + 0]);
                        mmTmp1 = _mm_loadu_pd(&tmpvec2[n][i + 2]);

                        mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 0]), mmTmp0));
                        mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 2]), mmTmp1));

                        mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 0]), mmTmp0));
                        mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 2]), mmTmp1));

                        mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 0]), mmTmp0));
                        mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 2]), mmTmp1));

                        mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 0]), mmTmp0));
                        mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 2]), mmTmp1));
                    }

                    mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                    mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                    mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                    mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                    _mm_storeu_pd(&mmArr[0], mmSum00);
                    _mm_storeu_pd(&mmArr[2], mmSum10);
                    _mm_storeu_pd(&mmArr[4], mmSum20);
                    _mm_storeu_pd(&mmArr[6], mmSum30);

                    sum0 = mmArr[0] + mmArr[1];
                    sum1 = mmArr[2] + mmArr[3];
                    sum2 = mmArr[4] + mmArr[5];
                    sum3 = mmArr[6] + mmArr[7];

                    for (; i < Msize2[Mc_AN]; i++) {
                        sum0 += U0[rl0][m + 0][i] * tmpvec2[n][i];
                        sum1 += U0[rl0][m + 1][i] * tmpvec2[n][i];
                        sum2 += U0[rl0][m + 2][i] * tmpvec2[n][i];
                        sum3 += U0[rl0][m + 3][i] * tmpvec2[n][i];
                    }

                    tmpmat0[m + 0][n] = sum0;
                    tmpmat0[m + 1][n] = sum1;
                    tmpmat0[m + 2][n] = sum2;
                    tmpmat0[m + 3][n] = sum3;
                }
            }

            for (; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += U0[rl0][m][i] * tmpvec2[n][i];
                    }
                    tmpmat0[m][n] = sum;
                }
            }

            /* |tmpvec0) - |U_rl0) * (U_rl0|tmpvec2) */

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (k = 0; k < EKC_core_size[Mc_AN]; k++) {
                    dum = tmpmat0[k][n];
                    for (i = 0; i < Msize2[Mc_AN]; i++)
                        tmpvec0[n][i] -= U0[rl0][k][i] * dum;
                }
            }
        }

#endif

        if (measure_time == 1) {
            dtime(&Etime1);
            time5 += Etime1 - Stime1;
        }

        /*************************************************************
                       S-orthonormalization of tmpvec0
        *************************************************************/

        if (measure_time == 1)
            dtime(&Stime1);

        S_orthonormalize_vec_trd(Mc_AN, ct_on, tmpvec0, tmpvec1, OLP0, tmpmat0, ko, iko, MP, Msize2);

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                U0[rl + 1][n][i] = tmpvec0[n][i];
            }
        }

        if (measure_time == 1) {
            dtime(&Etime1);
            time6 += Etime1 - Stime1;
        }

    } /* rl */

    /************************************************************
                orthogonalization by diagonalization
    ************************************************************/

    if (measure_time == 1)
        dtime(&Stime1);

#pragma omp parallel shared(EKC_core_size_max, Msize2_max, rlmax_EC, Mc_AN, EKC_core_size, Msize2, can, natn, ct_AN, ct_on, Spe_Total_CNO, WhatSpecies, MP, S_G2M, RMI1, Hks, spin, Msize, Msize4, OLP0, U0, rlmax_EC2, FS) private(OMPID, Nthrds, Nprocs, rl, n, i, ig, ian, Anum, ih, j, kl, jg, jan, Bnum, m, sum, k, mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1, mmArr, sum0, sum1, sum2, sum3, rl0)

    {

        double** tmpvec1;

        tmpvec1 = (double**)malloc(sizeof(double*) * EKC_core_size_max);
        for (i = 0; i < EKC_core_size_max; i++) {
            tmpvec1[i] = (double*)malloc(sizeof(double) * Msize2_max);
        }

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (rl = 0 + OMPID; rl < rlmax_EC[Mc_AN]; rl += Nthrds) {

            /*  S * |Vn) */

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    tmpvec1[n][i] = 0.0;
                }
            }

            for (i = 0; i <= can; i++) {

                ig = natn[ct_AN][i];
                ian = Spe_Total_CNO[WhatSpecies[ig]];
                Anum = MP[i] - 1;
                ih = S_G2M[ig];

                for (j = 0; j <= can; j++) {

                    kl = RMI1[Mc_AN][i][j];
                    jg = natn[ct_AN][j];
                    jan = Spe_Total_CNO[WhatSpecies[jg]];
                    Bnum = MP[j] - 1;

                    if (0 <= kl) {

#ifndef __x86_64__

                        /* Original version */
                        /**/
                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                sum = 0.0;
                                for (k = 0; k < jan; k++) {
                                    sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                                }
                                tmpvec1[n][Anum + m] += sum;
                            }
                        }
                        /**/

#else

                        /* Unrolling + SSE version */
                        /**/
                        for (m = 0; m < (ian - 3); m += 4) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                mmSum00 = _mm_setzero_pd();
                                mmSum01 = _mm_setzero_pd();
                                mmSum10 = _mm_setzero_pd();
                                mmSum11 = _mm_setzero_pd();
                                mmSum20 = _mm_setzero_pd();
                                mmSum21 = _mm_setzero_pd();
                                mmSum30 = _mm_setzero_pd();
                                mmSum31 = _mm_setzero_pd();

                                for (k = 0; k < (jan - 3); k += 4) {
                                    mmTmp0 = _mm_loadu_pd(&U0[rl][n][Bnum + k + 0]);
                                    mmTmp1 = _mm_loadu_pd(&U0[rl][n][Bnum + k + 2]);

                                    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 0]), mmTmp0));
                                    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 2]), mmTmp1));

                                    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 0]), mmTmp0));
                                    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 2]), mmTmp1));

                                    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 0]), mmTmp0));
                                    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 2]), mmTmp1));

                                    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 0]), mmTmp0));
                                    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 2]), mmTmp1));
                                }

                                mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                                mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                                mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                                mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                                _mm_storeu_pd(&mmArr[0], mmSum00);
                                _mm_storeu_pd(&mmArr[2], mmSum10);
                                _mm_storeu_pd(&mmArr[4], mmSum20);
                                _mm_storeu_pd(&mmArr[6], mmSum30);

                                sum0 = mmArr[0] + mmArr[1];
                                sum1 = mmArr[2] + mmArr[3];
                                sum2 = mmArr[4] + mmArr[5];
                                sum3 = mmArr[6] + mmArr[7];

                                for (; k < jan; k++) {
                                    sum0 += OLP0[ih][kl][m + 0][k] * U0[rl][n][Bnum + k];
                                    sum1 += OLP0[ih][kl][m + 1][k] * U0[rl][n][Bnum + k];
                                    sum2 += OLP0[ih][kl][m + 2][k] * U0[rl][n][Bnum + k];
                                    sum3 += OLP0[ih][kl][m + 3][k] * U0[rl][n][Bnum + k];
                                }

                                tmpvec1[n][Anum + m + 0] += sum0;
                                tmpvec1[n][Anum + m + 1] += sum1;
                                tmpvec1[n][Anum + m + 2] += sum2;
                                tmpvec1[n][Anum + m + 3] += sum3;
                            }
                        }

                        for (; m < ian; m++) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                sum = 0.0;
                                for (k = 0; k < jan; k++) {
                                    sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                                }
                                tmpvec1[n][Anum + m] += sum;
                            }
                        }
                        /**/
#endif
                    }
                }
            }

#ifndef __x86_64__

            /* Original version */
            /**/
            for (rl0 = rl; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum = 0.0;
                        for (i = 0; i < Msize2[Mc_AN]; i++) {
                            sum += U0[rl0][m][i] * tmpvec1[n][i];
                        }
                        FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                        FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum;
                    }
                }
            }
            /**/

#else

            /* Unrolling + SSE version */
            /**/
            for (rl0 = rl; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                        mmSum00 = _mm_setzero_pd();
                        mmSum01 = _mm_setzero_pd();
                        mmSum10 = _mm_setzero_pd();
                        mmSum11 = _mm_setzero_pd();
                        mmSum20 = _mm_setzero_pd();
                        mmSum21 = _mm_setzero_pd();
                        mmSum30 = _mm_setzero_pd();
                        mmSum31 = _mm_setzero_pd();

                        for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                            mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i + 0]);
                            mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i + 2]);

                            mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 0]), mmTmp0));
                            mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 2]), mmTmp1));

                            mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 0]), mmTmp0));
                            mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 2]), mmTmp1));

                            mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 0]), mmTmp0));
                            mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 2]), mmTmp1));

                            mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 0]), mmTmp0));
                            mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 2]), mmTmp1));
                        }

                        mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                        mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                        mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                        mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                        _mm_storeu_pd(&mmArr[0], mmSum00);
                        _mm_storeu_pd(&mmArr[2], mmSum10);
                        _mm_storeu_pd(&mmArr[4], mmSum20);
                        _mm_storeu_pd(&mmArr[6], mmSum30);

                        sum0 = mmArr[0] + mmArr[1];
                        sum1 = mmArr[2] + mmArr[3];
                        sum2 = mmArr[4] + mmArr[5];
                        sum3 = mmArr[6] + mmArr[7];

                        for (; i < Msize2[Mc_AN]; i++) {
                            sum0 += U0[rl0][m + 0][i] * tmpvec1[n][i];
                            sum1 += U0[rl0][m + 1][i] * tmpvec1[n][i];
                            sum2 += U0[rl0][m + 2][i] * tmpvec1[n][i];
                            sum3 += U0[rl0][m + 3][i] * tmpvec1[n][i];
                        }

                        FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum0;
                        FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum0;

                        FS[rl0 * EKC_core_size[Mc_AN] + m + 2][rl * EKC_core_size[Mc_AN] + n + 1] = sum1;
                        FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 2] = sum1;

                        FS[rl0 * EKC_core_size[Mc_AN] + m + 3][rl * EKC_core_size[Mc_AN] + n + 1] = sum2;
                        FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 3] = sum2;

                        FS[rl0 * EKC_core_size[Mc_AN] + m + 4][rl * EKC_core_size[Mc_AN] + n + 1] = sum3;
                        FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 4] = sum3;
                    }
                }

                for (; m < EKC_core_size[Mc_AN]; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum = 0.0;
                        for (i = 0; i < Msize2[Mc_AN]; i++) {
                            sum += U0[rl0][m][i] * tmpvec1[n][i];
                        }
                        FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                        FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum;
                    }
                }
            }
            /**/

#endif
        }

        for (i = 0; i < EKC_core_size_max; i++) {
            free(tmpvec1[i]);
        }
        free(tmpvec1);

    } /* #pragma omp parallel */

    if (measure_time == 1) {
        dtime(&Etime1);
        time7 += Etime1 - Stime1;
    }

    if (measure_time == 1)
        dtime(&Stime1);

    info = Eigen_lapack_d(FS, ko, Msize3[Mc_AN], Msize3[Mc_AN]);
    if (info != 0) {
        info = Eigen_lapack_x(FS, ko, Msize3[Mc_AN], Msize3[Mc_AN]);
        if (info != 0) {
            Eigen_lapack_r(FS, ko, Msize3[Mc_AN], Msize3[Mc_AN]);
        }
    }

    if (measure_time == 1) {
        dtime(&Etime1);
        time8 += Etime1 - Stime1;
    }

    ZeroNum = 0;

    for (i = 1; i <= Msize3[Mc_AN]; i++) {

        if (error_check == 1) {
            printf("spin=%2d Mc_AN=%2d i=%3d ko[i]=%18.15f\n", spin, Mc_AN, i, ko[i]);
        }

        if (cutoff_value < ko[i]) {
            ko[i] = sqrt(fabs(ko[i]));
            iko[i] = 1.0 / ko[i];
        } else {
            ZeroNum++;
            ko[i] = 0.0;
            iko[i] = 0.0;
        }
    }

    if (error_check == 1) {
        printf("spin=%2d Mc_AN=%2d ZeroNum=%2d\n", spin, Mc_AN, ZeroNum);
    }

    for (i = 1; i <= Msize3[Mc_AN]; i++) {
        for (j = 1; j <= Msize3[Mc_AN]; j++) {
            FS[i][j] = FS[i][j] * iko[j];
        }
    }

    /* transpose for later calculation */
    for (i = 1; i <= Msize3[Mc_AN]; i++) {
        for (j = i + 1; j <= Msize3[Mc_AN]; j++) {
            tmp1 = FS[i][j];
            tmp2 = FS[j][i];
            FS[i][j] = tmp2;
            FS[j][i] = tmp1;
        }
    }

    if (measure_time == 1)
        dtime(&Stime1);

        /* U0 * U * lamda^{-1/2} */

#ifndef __x86_64__

    /* original version */
    /*
    for (i=0; i<Msize2[Mc_AN]; i++){
      for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
        for (m=0; m<EKC_core_size[Mc_AN]; m++){

          m1 = rl0*EKC_core_size[Mc_AN] + m + 1;

          sum = 0.0;
          for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

            n1 = rl*EKC_core_size[Mc_AN] + 1;

            for (n=0; n<EKC_core_size[Mc_AN]; n++){
              sum += U0[rl][n][i]*FS[m1][n1+n];
            }
          }

          Utmp[rl0][m] = sum;
        }
      }

      for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
        for (m=0; m<EKC_core_size[Mc_AN]; m++){
          U0[rl0][m][i] = Utmp[rl0][m];
        }
      }
    }
    */

    /* unrolling version */

    for (i = 0; i < Msize2[Mc_AN]; i++) {

        for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                Utmp[rl][n] = U0[rl][n][i];
            }
        }

        for (m1 = 1; m1 <= (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - 3); m1 += 4) {

            rl00 = (m1 + 0 - 1) / EKC_core_size[Mc_AN];
            rl01 = (m1 + 1 - 1) / EKC_core_size[Mc_AN];
            rl02 = (m1 + 2 - 1) / EKC_core_size[Mc_AN];
            rl03 = (m1 + 3 - 1) / EKC_core_size[Mc_AN];

            mm0 = (m1 + 0 - 1) % EKC_core_size[Mc_AN];
            mm1 = (m1 + 1 - 1) % EKC_core_size[Mc_AN];
            mm2 = (m1 + 2 - 1) % EKC_core_size[Mc_AN];
            mm3 = (m1 + 3 - 1) % EKC_core_size[Mc_AN];

            sum0 = 0.0;
            sum1 = 0.0;
            sum2 = 0.0;
            sum3 = 0.0;

            for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                n1 = rl * EKC_core_size[Mc_AN] + 1;

                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum0 += Utmp[rl][n] * FS[m1 + 0][n1 + n];
                    sum1 += Utmp[rl][n] * FS[m1 + 1][n1 + n];
                    sum2 += Utmp[rl][n] * FS[m1 + 2][n1 + n];
                    sum3 += Utmp[rl][n] * FS[m1 + 3][n1 + n];
                }
            }

            U0[rl00][mm0][i] = sum0;
            U0[rl01][mm1][i] = sum1;
            U0[rl02][mm2][i] = sum2;
            U0[rl03][mm3][i] = sum3;
        }

        m1s = rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]) % 4 + 1;

        for (m1 = m1s; m1 <= rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]; m1++) {

            rl0 = (m1 - 1) / EKC_core_size[Mc_AN];
            m = (m1 - 1) % EKC_core_size[Mc_AN];

            sum = 0.0;

            for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                n1 = rl * EKC_core_size[Mc_AN] + 1;

                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum += Utmp[rl][n] * FS[m1][n1 + n];
                }
            }

            U0[rl0][m][i] = sum;
        }
    } /* i */

#else

        /* Unrolling + SSE version */
        /**/

#pragma omp parallel shared(rlmax_EC, Mc_AN, EKC_core_size, Msize2, U0, FS) private(OMPID, Nthrds, Nprocs, rl, n, i, j, m, sum, k, mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1, mmArr, sum0, sum1, sum2, sum3, rl0, rl00, rl01, rl02, rl03, mm0, mm1, mm2, mm3, n1, m1, m1s, Utmp)

    {

        double** Utmp;
        Utmp = (double**)malloc(sizeof(double*) * rlmax_EC[Mc_AN]);
        for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
            Utmp[i] = (double*)malloc(sizeof(double) * EKC_core_size[Mc_AN]);
        }

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (i = 0 + OMPID; i < Msize2[Mc_AN]; i += Nthrds) {

            for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    Utmp[rl][n] = U0[rl][n][i];
                }
            }

            for (m1 = 1; m1 <= (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - 3); m1 += 4) {

                rl00 = (m1 + 0 - 1) / EKC_core_size[Mc_AN];
                rl01 = (m1 + 1 - 1) / EKC_core_size[Mc_AN];
                rl02 = (m1 + 2 - 1) / EKC_core_size[Mc_AN];
                rl03 = (m1 + 3 - 1) / EKC_core_size[Mc_AN];

                mm0 = (m1 + 0 - 1) % EKC_core_size[Mc_AN];
                mm1 = (m1 + 1 - 1) % EKC_core_size[Mc_AN];
                mm2 = (m1 + 2 - 1) % EKC_core_size[Mc_AN];
                mm3 = (m1 + 3 - 1) % EKC_core_size[Mc_AN];

                sum0 = 0.0;
                sum1 = 0.0;
                sum2 = 0.0;
                sum3 = 0.0;

                mmSum00 = _mm_setzero_pd();
                mmSum01 = _mm_setzero_pd();
                mmSum10 = _mm_setzero_pd();
                mmSum11 = _mm_setzero_pd();
                mmSum20 = _mm_setzero_pd();
                mmSum21 = _mm_setzero_pd();
                mmSum30 = _mm_setzero_pd();
                mmSum31 = _mm_setzero_pd();

                for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                    n1 = rl * EKC_core_size[Mc_AN] + 1;

                    if (0) {
                        mmTmp0 = _mm_loadu_pd(&Utmp[rl][n + 0]);
                        mmTmp1 = _mm_loadu_pd(&Utmp[rl][n + 2]);
                    }

                    for (n = 0; n < (EKC_core_size[Mc_AN] - 3); n += 4) {
                        mmTmp0 = _mm_loadu_pd(&Utmp[rl][n + 0]);
                        mmTmp1 = _mm_loadu_pd(&Utmp[rl][n + 2]);

                        mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 0][n1 + n + 0]), mmTmp0));
                        mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 0][n1 + n + 2]), mmTmp1));

                        mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 1][n1 + n + 0]), mmTmp0));
                        mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 1][n1 + n + 2]), mmTmp1));

                        mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 2][n1 + n + 0]), mmTmp0));
                        mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 2][n1 + n + 2]), mmTmp1));

                        mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 3][n1 + n + 0]), mmTmp0));
                        mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 3][n1 + n + 2]), mmTmp1));
                    }

                    for (; n < EKC_core_size[Mc_AN]; n++) {
                        sum0 += Utmp[rl][n] * FS[m1 + 0][n1 + n];
                        sum1 += Utmp[rl][n] * FS[m1 + 1][n1 + n];
                        sum2 += Utmp[rl][n] * FS[m1 + 2][n1 + n];
                        sum3 += Utmp[rl][n] * FS[m1 + 3][n1 + n];
                    }
                }

                mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                _mm_storeu_pd(&mmArr[0], mmSum00);
                _mm_storeu_pd(&mmArr[2], mmSum10);
                _mm_storeu_pd(&mmArr[4], mmSum20);
                _mm_storeu_pd(&mmArr[6], mmSum30);

                sum0 += mmArr[0] + mmArr[1];
                sum1 += mmArr[2] + mmArr[3];
                sum2 += mmArr[4] + mmArr[5];
                sum3 += mmArr[6] + mmArr[7];

                U0[rl00][mm0][i] = sum0;
                U0[rl01][mm1][i] = sum1;
                U0[rl02][mm2][i] = sum2;
                U0[rl03][mm3][i] = sum3;
            }

            m1s = rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]) % 4 + 1;

            for (m1 = m1s; m1 <= rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]; m1++) {

                rl0 = (m1 - 1) / EKC_core_size[Mc_AN];
                m = (m1 - 1) % EKC_core_size[Mc_AN];

                sum = 0.0;

                for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                    n1 = rl * EKC_core_size[Mc_AN] + 1;

                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum += Utmp[rl][n] * FS[m1][n1 + n];
                    }
                }

                U0[rl0][m][i] = sum;
            }
        } /* i */

        for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
            free(Utmp[i]);
        }
        free(Utmp);

    } /* #pragma omp parallel */

#endif

    Krylov_U[spin][Mc_AN][0] = ZeroNum;

    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + i + 1] = U0[rl][n][i];
            }
        }
    }

    if (measure_time == 1) {
        dtime(&Etime1);
        time9 += Etime1 - Stime1;
    }

    /************************************************************
           check the orthonormality of Krylov vectors
    ************************************************************/

    if (error_check == 1) {

        for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    tmpvec1[n][i] = 0.0;
                }
            }

            for (i = 0; i <= can; i++) {

                ig = natn[ct_AN][i];
                ian = Spe_Total_CNO[WhatSpecies[ig]];
                Anum = MP[i] - 1;
                ih = S_G2M[ig];

                for (j = 0; j <= can; j++) {

                    kl = RMI1[Mc_AN][i][j];
                    jg = natn[ct_AN][j];
                    jan = Spe_Total_CNO[WhatSpecies[jg]];
                    Bnum = MP[j] - 1;

                    if (0 <= kl) {

                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                sum = 0.0;
                                for (k = 0; k < jan; k++) {
                                    sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                                }

                                tmpvec1[n][Anum + m] += sum;
                            }
                        }
                    }
                }
            }

            for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum = 0.0;
                        for (i = 0; i < Msize2[Mc_AN]; i++) {
                            sum += U0[rl0][m][i] * tmpvec1[n][i];
                        }

                        if (rl == rl0 && m == n) {
                            if (1.0e-10 < fabs(sum - 1.0)) {
                                printf("A spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
                                    spin, Mc_AN, rl, rl0, m, n, sum);
                            }
                        } else {
                            if (1.0e-10 < fabs(sum)) {
                                printf("B spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
                                    spin, Mc_AN, rl, rl0, m, n, sum);
                            }
                        }
                    }
                }
            }
        }
    }

    if (measure_time == 1) {
        printf("pMatrix myid=%2d time1 =%5.3f time2 =%5.3f time3 =%5.3f time4 =%5.3f\n",
            myid, time1, time2, time3, time4);
        printf("pMatrix myid=%2d time5 =%5.3f time6 =%5.3f time7 =%5.3f time8 =%5.3f\n",
            myid, time5, time6, time7, time8);
        printf("pMatrix myid=%2d time9 =%5.3f\n", myid, time9);
    }

    /* freeing of arrays */

    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        free(Utmp[i]);
    }
    free(Utmp);

    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
            free(U0[i][j]);
        }
        free(U0[i]);
    }
    free(U0);

    for (i = 0; i < (EKC_core_size[Mc_AN] + 4); i++) {
        free(tmpmat0[i]);
    }
    free(tmpmat0);

    for (i = 0; i < (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        free(FS[i]);
    }
    free(FS);

    free(ko);
    free(iko);

    for (i = 0; i < (EKC_core_size[Mc_AN] + 2); i++) {
        free(matRS0[i]);
    }
    free(matRS0);

    for (i = 0; i < (Msize4[Mc_AN] + 3); i++) {
        free(matRS1[i]);
    }
    free(matRS1);
}

void Generate_pMatrix2_trd(int myid, int spin, int Mc_AN, double***** Hks, double**** OLP0,
    double*** Krylov_U, int* MP, int* Msize, int* Msize2, int* Msize3,
    double** tmpvec1)

/* This subroutine is exactly the same as the original Generate_pMatrix2,
except that Eigen_lapack is replaced with Eigen_lapack_d, _x, or _r
*/

{
    int rl, rl0, rl1, ct_AN, fan, san, can, wan, ct_on, i, j;
    int n, Anum, Bnum, k, ian, ih, kl, jg, ig, jan, m, m1, n1, info;
    int ZeroNum, rl_half;
    int KU_d1, KU_d2, csize;
    double sum, dum, tmp0, tmp1, tmp2, tmp3;
    double** Utmp;
    double *ko, *iko;
    double** FS;
    double*** U0;

    int rl00, rl01, rl02, rl03, rl04, rl05, rl06, rl07;
    int mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7, m1s;
    __m128d mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5;

    double mmArr[8];
    double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;

    ct_AN = M2G[Mc_AN];
    fan = FNAN[ct_AN];
    san = SNAN[ct_AN];
    can = fan + san;
    wan = WhatSpecies[ct_AN];
    ct_on = Spe_Total_CNO[wan];

    if (Msize[Mc_AN] < Msize3[Mc_AN])
        csize = Msize3[Mc_AN] + 40;
    else
        csize = Msize[Mc_AN] + 40;

    KU_d1 = EKC_core_size[Mc_AN] * Msize2[Mc_AN];
    KU_d2 = Msize2[Mc_AN];

    /* allocation of arrays */

    Utmp = (double**)malloc(sizeof(double*) * rlmax_EC[Mc_AN]);
    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        Utmp[i] = (double*)malloc(sizeof(double) * EKC_core_size[Mc_AN]);
    }

    U0 = (double***)malloc(sizeof(double**) * rlmax_EC[Mc_AN]);
    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        U0[i] = (double**)malloc(sizeof(double*) * EKC_core_size[Mc_AN]);
        for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
            U0[i][j] = (double*)malloc(sizeof(double) * (Msize2[Mc_AN] + 3));
            for (k = 0; k < (Msize2[Mc_AN] + 3); k++)
                U0[i][j][k] = 0.0;
        }
    }

    FS = (double**)malloc(sizeof(double*) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    for (i = 0; i < (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        FS[i] = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    }

    ko = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    iko = (double*)malloc(sizeof(double) * (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]);

    /****************************************************
     initialize
    ****************************************************/

    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                U0[rl][n][i] = 0.0;
            }
        }
    }

    i = 0;
    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            if (i < Msize2[Mc_AN])
                U0[rl][n][i] = 1.0;
            i++;
        }
    }

    /************************************************************
                orthogonalization by diagonalization
    ************************************************************/

    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

        /*  S * |Vn) */

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec1[n][i] = 0.0;
            }
        }

        for (i = 0; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = 0; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j] - 1;

                if (0 <= kl) {

#ifndef __x86_64__

                    /* Original version */
                    /**/
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                            }
                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
                    /**/

#else

                    /* Unrolling + SSE version */

                    for (m = 0; m < (ian - 3); m += 4) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            mmSum00 = _mm_setzero_pd();
                            mmSum01 = _mm_setzero_pd();
                            mmSum10 = _mm_setzero_pd();
                            mmSum11 = _mm_setzero_pd();
                            mmSum20 = _mm_setzero_pd();
                            mmSum21 = _mm_setzero_pd();
                            mmSum30 = _mm_setzero_pd();
                            mmSum31 = _mm_setzero_pd();

                            for (k = 0; k < (jan - 3); k += 4) {
                                mmTmp0 = _mm_loadu_pd(&U0[rl][n][Bnum + k + 0]);
                                mmTmp1 = _mm_loadu_pd(&U0[rl][n][Bnum + k + 2]);

                                mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 0]), mmTmp0));
                                mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 2]), mmTmp1));

                                mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 0]), mmTmp0));
                                mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 2]), mmTmp1));

                                mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 0]), mmTmp0));
                                mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 2]), mmTmp1));

                                mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 0]), mmTmp0));
                                mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 2]), mmTmp1));
                            }

                            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                            _mm_storeu_pd(&mmArr[0], mmSum00);
                            _mm_storeu_pd(&mmArr[2], mmSum10);
                            _mm_storeu_pd(&mmArr[4], mmSum20);
                            _mm_storeu_pd(&mmArr[6], mmSum30);

                            sum0 = mmArr[0] + mmArr[1];
                            sum1 = mmArr[2] + mmArr[3];
                            sum2 = mmArr[4] + mmArr[5];
                            sum3 = mmArr[6] + mmArr[7];

                            for (; k < jan; k++) {
                                sum0 += OLP0[ih][kl][m + 0][k] * U0[rl][n][Bnum + k];
                                sum1 += OLP0[ih][kl][m + 1][k] * U0[rl][n][Bnum + k];
                                sum2 += OLP0[ih][kl][m + 2][k] * U0[rl][n][Bnum + k];
                                sum3 += OLP0[ih][kl][m + 3][k] * U0[rl][n][Bnum + k];
                            }

                            tmpvec1[n][Anum + m + 0] += sum0;
                            tmpvec1[n][Anum + m + 1] += sum1;
                            tmpvec1[n][Anum + m + 2] += sum2;
                            tmpvec1[n][Anum + m + 3] += sum3;
                        }
                    }

                    for (; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                            }
                            tmpvec1[n][Anum + m] += sum;
                        }
                    }

#endif
                }
            }
        }

#ifndef __x86_64__

        /* Original version */
        /**/
        for (rl0 = rl; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += U0[rl0][m][i] * tmpvec1[n][i];
                    }
                    FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum;
                }
            }
        }
        /**/

#else

        /* Unrolling + SSE version */

        for (rl0 = rl; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                    mmSum00 = _mm_setzero_pd();
                    mmSum01 = _mm_setzero_pd();
                    mmSum10 = _mm_setzero_pd();
                    mmSum11 = _mm_setzero_pd();
                    mmSum20 = _mm_setzero_pd();
                    mmSum21 = _mm_setzero_pd();
                    mmSum30 = _mm_setzero_pd();
                    mmSum31 = _mm_setzero_pd();

                    for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                        mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i + 0]);
                        mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i + 2]);

                        mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 0]), mmTmp0));
                        mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 0][i + 2]), mmTmp1));

                        mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 0]), mmTmp0));
                        mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 1][i + 2]), mmTmp1));

                        mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 0]), mmTmp0));
                        mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 2][i + 2]), mmTmp1));

                        mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 0]), mmTmp0));
                        mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m + 3][i + 2]), mmTmp1));
                    }

                    mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                    mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                    mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                    mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                    _mm_storeu_pd(&mmArr[0], mmSum00);
                    _mm_storeu_pd(&mmArr[2], mmSum10);
                    _mm_storeu_pd(&mmArr[4], mmSum20);
                    _mm_storeu_pd(&mmArr[6], mmSum30);

                    sum0 = mmArr[0] + mmArr[1];
                    sum1 = mmArr[2] + mmArr[3];
                    sum2 = mmArr[4] + mmArr[5];
                    sum3 = mmArr[6] + mmArr[7];

                    for (; i < Msize2[Mc_AN]; i++) {
                        sum0 += U0[rl0][m + 0][i] * tmpvec1[n][i];
                        sum1 += U0[rl0][m + 1][i] * tmpvec1[n][i];
                        sum2 += U0[rl0][m + 2][i] * tmpvec1[n][i];
                        sum3 += U0[rl0][m + 3][i] * tmpvec1[n][i];
                    }

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum0;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum0;

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 2][rl * EKC_core_size[Mc_AN] + n + 1] = sum1;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 2] = sum1;

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 3][rl * EKC_core_size[Mc_AN] + n + 1] = sum2;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 3] = sum2;

                    FS[rl0 * EKC_core_size[Mc_AN] + m + 4][rl * EKC_core_size[Mc_AN] + n + 1] = sum3;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 4] = sum3;
                }
            }

            for (; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += U0[rl0][m][i] * tmpvec1[n][i];
                    }
                    FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum;
                }
            }
        }

#endif
    }

    info = Eigen_lapack_d(FS, ko, Msize3[Mc_AN], Msize3[Mc_AN]);
    if (info != 0) {
        info = Eigen_lapack_x(FS, ko, Msize3[Mc_AN], Msize3[Mc_AN]);
        if (info != 0) {
            Eigen_lapack_r(FS, ko, Msize3[Mc_AN], Msize3[Mc_AN]);
        }
    }

    ZeroNum = 0;

    for (i = 1; i <= Msize3[Mc_AN]; i++) {

        if (error_check == 1) {
            printf("spin=%2d Mc_AN=%2d i=%3d ko[i]=%18.15f\n", spin, Mc_AN, i, ko[i]);
        }

        if (cutoff_value < ko[i]) {
            ko[i] = sqrt(fabs(ko[i]));
            iko[i] = 1.0 / ko[i];
        } else {
            ZeroNum++;
            ko[i] = 0.0;
            iko[i] = 0.0;
        }
    }

    if (error_check == 1) {
        printf("spin=%2d Mc_AN=%2d ZeroNum=%2d\n", spin, Mc_AN, ZeroNum);
    }

    for (i = 1; i <= Msize3[Mc_AN]; i++) {
        for (j = 1; j <= Msize3[Mc_AN]; j++) {
            FS[i][j] = FS[i][j] * iko[j];
        }
    }

    /* transpose for later calculation */
    for (i = 1; i <= Msize3[Mc_AN]; i++) {
        for (j = i + 1; j <= Msize3[Mc_AN]; j++) {
            tmp1 = FS[i][j];
            tmp2 = FS[j][i];
            FS[i][j] = tmp2;
            FS[j][i] = tmp1;
        }
    }

    /* U0 * U * lamda^{-1/2} */

#ifndef __x86_64__

    /* original version */

    /**/
    for (i = 0; i < Msize2[Mc_AN]; i++) {
        for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {

                m1 = rl0 * EKC_core_size[Mc_AN] + m + 1;

                sum = 0.0;
                for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                    n1 = rl * EKC_core_size[Mc_AN] + 1;

                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum += U0[rl][n][i] * FS[m1][n1 + n];
                    }
                }

                Utmp[rl0][m] = sum;
            }
        }

        for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                U0[rl0][m][i] = Utmp[rl0][m];
            }
        }
    }
    /**/

#else

    /* Unrolling + SSE version */
    /**/
    for (i = 0; i < Msize2[Mc_AN]; i++) {

        for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                Utmp[rl][n] = U0[rl][n][i];
            }
        }

        for (m1 = 1; m1 <= (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - 3); m1 += 4) {

            rl00 = (m1 + 0 - 1) / EKC_core_size[Mc_AN];
            rl01 = (m1 + 1 - 1) / EKC_core_size[Mc_AN];
            rl02 = (m1 + 2 - 1) / EKC_core_size[Mc_AN];
            rl03 = (m1 + 3 - 1) / EKC_core_size[Mc_AN];

            mm0 = (m1 + 0 - 1) % EKC_core_size[Mc_AN];
            mm1 = (m1 + 1 - 1) % EKC_core_size[Mc_AN];
            mm2 = (m1 + 2 - 1) % EKC_core_size[Mc_AN];
            mm3 = (m1 + 3 - 1) % EKC_core_size[Mc_AN];

            sum0 = 0.0;
            sum1 = 0.0;
            sum2 = 0.0;
            sum3 = 0.0;

            mmSum00 = _mm_setzero_pd();
            mmSum01 = _mm_setzero_pd();
            mmSum10 = _mm_setzero_pd();
            mmSum11 = _mm_setzero_pd();
            mmSum20 = _mm_setzero_pd();
            mmSum21 = _mm_setzero_pd();
            mmSum30 = _mm_setzero_pd();
            mmSum31 = _mm_setzero_pd();

            for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                n1 = rl * EKC_core_size[Mc_AN] + 1;

                if (0) {
                    mmTmp0 = _mm_loadu_pd(&Utmp[rl][n + 0]); /*???????????*/
                    mmTmp1 = _mm_loadu_pd(&Utmp[rl][n + 2]); /*???????????*/
                }

                for (n = 0; n < (EKC_core_size[Mc_AN] - 3); n += 4) {
                    mmTmp0 = _mm_loadu_pd(&Utmp[rl][n + 0]);
                    mmTmp1 = _mm_loadu_pd(&Utmp[rl][n + 2]);
                    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 0][n1 + n + 0]), mmTmp0));
                    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 0][n1 + n + 2]), mmTmp1));

                    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 1][n1 + n + 0]), mmTmp0));
                    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 1][n1 + n + 2]), mmTmp1));

                    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 2][n1 + n + 0]), mmTmp0));
                    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 2][n1 + n + 2]), mmTmp1));

                    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 3][n1 + n + 0]), mmTmp0));
                    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&FS[m1 + 3][n1 + n + 2]), mmTmp1));
                }

                for (; n < EKC_core_size[Mc_AN]; n++) {
                    sum0 += Utmp[rl][n] * FS[m1 + 0][n1 + n];
                    sum1 += Utmp[rl][n] * FS[m1 + 1][n1 + n];
                    sum2 += Utmp[rl][n] * FS[m1 + 2][n1 + n];
                    sum3 += Utmp[rl][n] * FS[m1 + 3][n1 + n];
                }
            }

            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

            _mm_storeu_pd(&mmArr[0], mmSum00);
            _mm_storeu_pd(&mmArr[2], mmSum10);
            _mm_storeu_pd(&mmArr[4], mmSum20);
            _mm_storeu_pd(&mmArr[6], mmSum30);

            sum0 += mmArr[0] + mmArr[1];
            sum1 += mmArr[2] + mmArr[3];
            sum2 += mmArr[4] + mmArr[5];
            sum3 += mmArr[6] + mmArr[7];

            U0[rl00][mm0][i] = sum0;
            U0[rl01][mm1][i] = sum1;
            U0[rl02][mm2][i] = sum2;
            U0[rl03][mm3][i] = sum3;
        }

        m1s = rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]) % 4 + 1;

        for (m1 = m1s; m1 <= rlmax_EC[Mc_AN] * EKC_core_size[Mc_AN]; m1++) {

            rl0 = (m1 - 1) / EKC_core_size[Mc_AN];
            m = (m1 - 1) % EKC_core_size[Mc_AN];

            sum = 0.0;

            for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

                n1 = rl * EKC_core_size[Mc_AN] + 1;

                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum += Utmp[rl][n] * FS[m1][n1 + n];
                }
            }

            U0[rl0][m][i] = sum;
        }
    } /* i */

#endif

    Krylov_U[spin][Mc_AN][0] = ZeroNum;

    for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {
        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + i + 1] = U0[rl][n][i];
            }
        }
    }

    /************************************************************
           check the orthonormality of Krylov vectors
    ************************************************************/

    if (error_check == 1) {

        for (rl = 0; rl < rlmax_EC[Mc_AN]; rl++) {

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    tmpvec1[n][i] = 0.0;
                }
            }

            for (i = 0; i <= can; i++) {

                ig = natn[ct_AN][i];
                ian = Spe_Total_CNO[WhatSpecies[ig]];
                Anum = MP[i] - 1;
                ih = S_G2M[ig];

                for (j = 0; j <= can; j++) {

                    kl = RMI1[Mc_AN][i][j];
                    jg = natn[ct_AN][j];
                    jan = Spe_Total_CNO[WhatSpecies[jg]];
                    Bnum = MP[j] - 1;

                    if (0 <= kl) {

                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                sum = 0.0;
                                for (k = 0; k < jan; k++) {
                                    sum += OLP0[ih][kl][m][k] * U0[rl][n][Bnum + k];
                                }

                                tmpvec1[n][Anum + m] += sum;
                            }
                        }
                    }
                }
            }

            for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum = 0.0;
                        for (i = 0; i < Msize2[Mc_AN]; i++) {
                            sum += U0[rl0][m][i] * tmpvec1[n][i];
                        }

                        if (rl == rl0 && m == n) {
                            if (1.0e-10 < fabs(sum - 1.0)) {
                                printf("A spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
                                    spin, Mc_AN, rl, rl0, m, n, sum);
                            }
                        } else {
                            if (1.0e-10 < fabs(sum)) {
                                printf("B spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
                                    spin, Mc_AN, rl, rl0, m, n, sum);
                            }
                        }
                    }
                }
            }
        }
    }

    /* freeing of arrays */

    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        free(Utmp[i]);
    }
    free(Utmp);

    for (i = 0; i < rlmax_EC[Mc_AN]; i++) {
        for (j = 0; j < EKC_core_size[Mc_AN]; j++) {
            free(U0[i][j]);
        }
        free(U0[i]);
    }
    free(U0);

    for (i = 0; i < (rlmax_EC[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        free(FS[i]);
    }
    free(FS);

    free(ko);
    free(iko);
}

void Embedding_Matrix_trd(int spin, int Mc_AN, double***** Hks,
    double*** Krylov_U, double**** EC_matrix,
    int* MP, int* Msize, int* Msize2, int* Msize3,
    double** tmpvec, int EKC_core_size_max, int Msize2_max)

/* This subroutine is exactly the same as the original Embedding_Matrix,
except for the OMP parallelized loops
*/

{
    int ct_AN, fan, san, can, wan, ct_on;
    int rl, rl0, m, n, i, j, k, kl, jg, jan, ih, ian;
    int Anum, Bnum, ig;
    int KU_d1, KU_d2, csize;
    double sum, tmp1, tmp2, tmp3;

    __m128d mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1;
    double mmArr[8];
    double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    ct_AN = M2G[Mc_AN];
    fan = FNAN[ct_AN];
    san = SNAN[ct_AN];
    can = fan + san;
    wan = WhatSpecies[ct_AN];
    ct_on = Spe_Total_CNO[wan];

    if (Msize[Mc_AN] < Msize3[Mc_AN])
        csize = Msize3[Mc_AN] + 40;
    else
        csize = Msize[Mc_AN] + 40;

    KU_d1 = EKC_core_size[Mc_AN] * Msize2[Mc_AN];
    KU_d2 = Msize2[Mc_AN];

    /*******************************
              u1^+ C^+ u2
    *******************************/

#pragma omp parallel shared(EKC_core_size_max, Msize2_max, rlmax_EC, Mc_AN, EKC_core_size, Msize2, fan, can, natn, ct_AN, Spe_Total_CNO, WhatSpecies, MP, S_G2M, RMI1, Hks, Krylov_U, spin, KU_d1, KU_d2, Msize, EC_matrix) private(OMPID, Nthrds, Nprocs, rl, n, i, ig, ian, Anum, ih, j, kl, jg, jan, Bnum, m, sum, k, mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1, mmArr, sum0, sum1, sum2, sum3, rl0)

    {

        double** tmpvec1;
        tmpvec1 = (double**)malloc(sizeof(double*) * EKC_core_size_max);
        for (i = 0; i < EKC_core_size_max; i++) {
            tmpvec1[i] = (double*)malloc(sizeof(double) * Msize2_max);
        }

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (rl = 0 + OMPID; rl < rlmax_EC[Mc_AN]; rl += Nthrds) {

            /* C^+ u2 */

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    tmpvec1[n][i] = 0.0;
                }
            }

            for (i = 0; i <= fan; i++) {

                ig = natn[ct_AN][i];
                ian = Spe_Total_CNO[WhatSpecies[ig]];
                Anum = MP[i] - 1;
                ih = S_G2M[ig];

                for (j = fan + 1; j <= can; j++) {

                    kl = RMI1[Mc_AN][i][j];
                    jg = natn[ct_AN][j];
                    jan = Spe_Total_CNO[WhatSpecies[jg]];
                    Bnum = MP[j];

                    if (0 <= kl) {

#ifndef __x86_64__

                        /* Original version */
                        /**/
                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                sum = 0.0;
                                for (k = 0; k < jan; k++) {
                                    sum += Hks[spin][ih][kl][m][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                                }

                                tmpvec1[n][Anum + m] += sum;
                            }
                        }
                        /**/

#else

                        /* Unrolling + SSE version */
                        /**/
                        for (m = 0; m < (ian - 3); m += 4) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                mmSum00 = _mm_setzero_pd();
                                mmSum01 = _mm_setzero_pd();
                                mmSum10 = _mm_setzero_pd();
                                mmSum11 = _mm_setzero_pd();
                                mmSum20 = _mm_setzero_pd();
                                mmSum21 = _mm_setzero_pd();
                                mmSum30 = _mm_setzero_pd();
                                mmSum31 = _mm_setzero_pd();

                                for (k = 0; k < (jan - 3); k += 4) {
                                    mmTmp0 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0]);
                                    mmTmp1 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 2]);

                                    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 0]), mmTmp0));
                                    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 2]), mmTmp1));

                                    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 0]), mmTmp0));
                                    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 2]), mmTmp1));

                                    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 0]), mmTmp0));
                                    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 2]), mmTmp1));

                                    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 0]), mmTmp0));
                                    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 2]), mmTmp1));
                                }

                                mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                                mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                                mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                                mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                                _mm_storeu_pd(&mmArr[0], mmSum00);
                                _mm_storeu_pd(&mmArr[2], mmSum10);
                                _mm_storeu_pd(&mmArr[4], mmSum20);
                                _mm_storeu_pd(&mmArr[6], mmSum30);

                                sum0 = mmArr[0] + mmArr[1];
                                sum1 = mmArr[2] + mmArr[3];
                                sum2 = mmArr[4] + mmArr[5];
                                sum3 = mmArr[6] + mmArr[7];

                                for (; k < jan; k++) {
                                    sum0 += Hks[spin][ih][kl][m + 0][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                                    sum1 += Hks[spin][ih][kl][m + 1][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                                    sum2 += Hks[spin][ih][kl][m + 2][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                                    sum3 += Hks[spin][ih][kl][m + 3][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                                }

                                tmpvec1[n][Anum + m + 0] += sum0;
                                tmpvec1[n][Anum + m + 1] += sum1;
                                tmpvec1[n][Anum + m + 2] += sum2;
                                tmpvec1[n][Anum + m + 3] += sum3;
                            }
                        }

                        for (; m < ian; m++) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                sum = 0.0;
                                for (k = 0; k < jan; k++) {
                                    sum += Hks[spin][ih][kl][m][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                                }

                                tmpvec1[n][Anum + m] += sum;
                            }
                        }
                        /**/
#endif
                    }
                }
            }

            /* u1^+ C^+ u2 */

#ifndef __x86_64__

            /* Original version */
            /**/
            for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum = 0.0;
                        for (i = 0; i < Msize[Mc_AN]; i++) {
                            sum += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + m * KU_d2 + i + 1] * tmpvec1[n][i];
                        }
                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                    }
                }
            }
            /**/

#else

            /* Unrolling + SSE version */
            /**/
            for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                        mmSum00 = _mm_setzero_pd();
                        mmSum01 = _mm_setzero_pd();
                        mmSum10 = _mm_setzero_pd();
                        mmSum11 = _mm_setzero_pd();
                        mmSum20 = _mm_setzero_pd();
                        mmSum21 = _mm_setzero_pd();
                        mmSum30 = _mm_setzero_pd();
                        mmSum31 = _mm_setzero_pd();

                        for (i = 0; i < (Msize[Mc_AN] - 3); i += 4) {
                            mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i + 0]);
                            mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i + 2]);

                            mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1]), mmTmp0));
                            mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 3]), mmTmp1));

                            mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 1]), mmTmp0));
                            mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 3]), mmTmp1));

                            mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 1]), mmTmp0));
                            mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 3]), mmTmp1));

                            mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 1]), mmTmp0));
                            mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 3]), mmTmp1));
                        }

                        mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                        mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                        mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                        mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                        _mm_storeu_pd(&mmArr[0], mmSum00);
                        _mm_storeu_pd(&mmArr[2], mmSum10);
                        _mm_storeu_pd(&mmArr[4], mmSum20);
                        _mm_storeu_pd(&mmArr[6], mmSum30);

                        sum0 = mmArr[0] + mmArr[1];
                        sum1 = mmArr[2] + mmArr[3];
                        sum2 = mmArr[4] + mmArr[5];
                        sum3 = mmArr[6] + mmArr[7];

                        for (; i < Msize[Mc_AN]; i++) {
                            sum0 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1] * tmpvec1[n][i];
                            sum1 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 1] * tmpvec1[n][i];
                            sum2 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 1] * tmpvec1[n][i];
                            sum3 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 1] * tmpvec1[n][i];
                        }

                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum0;
                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 2][rl * EKC_core_size[Mc_AN] + n + 1] = sum1;
                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 3][rl * EKC_core_size[Mc_AN] + n + 1] = sum2;
                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 4][rl * EKC_core_size[Mc_AN] + n + 1] = sum3;
                    }
                }

                for (; m < EKC_core_size[Mc_AN]; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum = 0.0;
                        for (i = 0; i < Msize[Mc_AN]; i++) {
                            sum += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1] * tmpvec1[n][i];
                        }
                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                    }
                }
            }
            /**/

#endif

        } /* rl */

        for (i = 0; i < EKC_core_size_max; i++) {
            free(tmpvec1[i]);
        }
        free(tmpvec1);

    } /* #pragma omp parallel */

    /*******************************
              u2^+ C u1
    *******************************/

    for (i = 1; i <= Msize3[Mc_AN]; i++) {
        for (j = i; j <= Msize3[Mc_AN]; j++) {

            tmp1 = EC_matrix[spin][Mc_AN][i][j];
            tmp2 = EC_matrix[spin][Mc_AN][j][i];
            tmp3 = tmp1 + tmp2;

            EC_matrix[spin][Mc_AN][i][j] = tmp3;
            EC_matrix[spin][Mc_AN][j][i] = tmp3;
        }
    }

    /*******************************
               u2^+ B u2
    *******************************/

#pragma omp parallel shared(EKC_core_size_max, Msize2_max, rlmax_EC, Mc_AN, EKC_core_size, Msize2, fan, can, natn, ct_AN, Spe_Total_CNO, WhatSpecies, MP, S_G2M, RMI1, Hks, Krylov_U, spin, KU_d1, KU_d2, Msize, EC_matrix) private(OMPID, Nthrds, Nprocs, rl, n, i, ig, ian, Anum, ih, j, kl, jg, jan, Bnum, m, sum, k, mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1, mmArr, sum0, sum1, sum2, sum3, rl0)

    {

        double** tmpvec1;
        tmpvec1 = (double**)malloc(sizeof(double*) * EKC_core_size_max);
        for (i = 0; i < EKC_core_size_max; i++) {
            tmpvec1[i] = (double*)malloc(sizeof(double) * Msize2_max);
        }

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (rl = 0 + OMPID; rl < rlmax_EC[Mc_AN]; rl += Nthrds) {

            /* B u2 */

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    tmpvec1[n][i] = 0.0;
                }
            }

            for (i = fan + 1; i <= can; i++) {

                ig = natn[ct_AN][i];
                ian = Spe_Total_CNO[WhatSpecies[ig]];
                Anum = MP[i] - 1;
                ih = S_G2M[ig];

                for (j = fan + 1; j <= can; j++) {

                    kl = RMI1[Mc_AN][i][j];
                    jg = natn[ct_AN][j];
                    jan = Spe_Total_CNO[WhatSpecies[jg]];
                    Bnum = MP[j];

                    if (0 <= kl) {

#ifndef __x86_64__

                        /* Original version */
                        /**/
                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                sum = 0.0;
                                for (k = 0; k < jan; k++) {
                                    sum += Hks[spin][ih][kl][m][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                                }

                                tmpvec1[n][Anum + m] += sum;
                            }
                        }
                        /**/

#else

                        /* Unrolling + SSE version */
                        /**/
                        for (m = 0; m < (ian - 3); m += 4) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                mmSum00 = _mm_setzero_pd();
                                mmSum01 = _mm_setzero_pd();
                                mmSum10 = _mm_setzero_pd();
                                mmSum11 = _mm_setzero_pd();
                                mmSum20 = _mm_setzero_pd();
                                mmSum21 = _mm_setzero_pd();
                                mmSum30 = _mm_setzero_pd();
                                mmSum31 = _mm_setzero_pd();

                                for (k = 0; k < (jan - 3); k += 4) {
                                    mmTmp0 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0]);
                                    mmTmp1 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 2]);

                                    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 0]), mmTmp0));
                                    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 0][k + 2]), mmTmp1));

                                    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 0]), mmTmp0));
                                    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 1][k + 2]), mmTmp1));

                                    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 0]), mmTmp0));
                                    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 2][k + 2]), mmTmp1));

                                    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 0]), mmTmp0));
                                    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m + 3][k + 2]), mmTmp1));
                                }

                                mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                                mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                                mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                                mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                                _mm_storeu_pd(&mmArr[0], mmSum00);
                                _mm_storeu_pd(&mmArr[2], mmSum10);
                                _mm_storeu_pd(&mmArr[4], mmSum20);
                                _mm_storeu_pd(&mmArr[6], mmSum30);

                                sum0 = mmArr[0] + mmArr[1];
                                sum1 = mmArr[2] + mmArr[3];
                                sum2 = mmArr[4] + mmArr[5];
                                sum3 = mmArr[6] + mmArr[7];

                                for (; k < jan; k++) {
                                    sum0 += Hks[spin][ih][kl][m + 0][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0];
                                    sum1 += Hks[spin][ih][kl][m + 1][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0];
                                    sum2 += Hks[spin][ih][kl][m + 2][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0];
                                    sum3 += Hks[spin][ih][kl][m + 3][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k + 0];
                                }

                                tmpvec1[n][Anum + m + 0] += sum0;
                                tmpvec1[n][Anum + m + 1] += sum1;
                                tmpvec1[n][Anum + m + 2] += sum2;
                                tmpvec1[n][Anum + m + 3] += sum3;
                            }
                        }

                        for (; m < ian; m++) {
                            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                                sum = 0.0;
                                for (k = 0; k < jan; k++) {
                                    sum += Hks[spin][ih][kl][m][k] * Krylov_U[spin][Mc_AN][rl * KU_d1 + n * KU_d2 + Bnum + k];
                                }

                                tmpvec1[n][Anum + m] += sum;
                            }
                        }
                        /**/

#endif
                    }
                }
            }

            /* u2^+ B u2 */

#ifndef __x86_64__

            /* Original version */
            /**/
            for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum = 0.0;
                        for (i = Msize[Mc_AN]; i < Msize2[Mc_AN]; i++) {
                            sum += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + m * KU_d2 + i + 1] * tmpvec1[n][i];
                        }
                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] += sum;
                    }
                }
            }
            /**/

#else

            /* Unrolling + SSE version */
            /**/
            for (rl0 = 0; rl0 < rlmax_EC[Mc_AN]; rl0++) {
                for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                        mmSum00 = _mm_setzero_pd();
                        mmSum01 = _mm_setzero_pd();
                        mmSum10 = _mm_setzero_pd();
                        mmSum11 = _mm_setzero_pd();
                        mmSum20 = _mm_setzero_pd();
                        mmSum21 = _mm_setzero_pd();
                        mmSum30 = _mm_setzero_pd();
                        mmSum31 = _mm_setzero_pd();

                        for (i = Msize[Mc_AN]; i < (Msize2[Mc_AN] - 3); i += 4) {
                            mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i + 0]);
                            mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i + 2]);

                            mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1]), mmTmp0));
                            mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 3]), mmTmp1));

                            mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 1]), mmTmp0));
                            mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 3]), mmTmp1));

                            mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 1]), mmTmp0));
                            mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 3]), mmTmp1));

                            mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 1]), mmTmp0));
                            mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 3]), mmTmp1));
                        }

                        mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                        mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                        mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                        mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                        _mm_storeu_pd(&mmArr[0], mmSum00);
                        _mm_storeu_pd(&mmArr[2], mmSum10);
                        _mm_storeu_pd(&mmArr[4], mmSum20);
                        _mm_storeu_pd(&mmArr[6], mmSum30);

                        sum0 = mmArr[0] + mmArr[1];
                        sum1 = mmArr[2] + mmArr[3];
                        sum2 = mmArr[4] + mmArr[5];
                        sum3 = mmArr[6] + mmArr[7];

                        for (; i < Msize2[Mc_AN]; i++) {
                            sum0 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1] * tmpvec1[n][i];
                            sum1 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 1) * KU_d2 + i + 1] * tmpvec1[n][i];
                            sum2 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 2) * KU_d2 + i + 1] * tmpvec1[n][i];
                            sum3 += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 3) * KU_d2 + i + 1] * tmpvec1[n][i];
                        }

                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] += sum0;
                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 2][rl * EKC_core_size[Mc_AN] + n + 1] += sum1;
                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 3][rl * EKC_core_size[Mc_AN] + n + 1] += sum2;
                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 4][rl * EKC_core_size[Mc_AN] + n + 1] += sum3;
                    }
                }

                for (; m < EKC_core_size[Mc_AN]; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                        sum = 0.0;
                        for (i = Msize[Mc_AN]; i < Msize2[Mc_AN]; i++) {
                            sum += Krylov_U[spin][Mc_AN][rl0 * KU_d1 + (m + 0) * KU_d2 + i + 1] * tmpvec1[n][i];
                        }
                        EC_matrix[spin][Mc_AN][rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] += sum;
                    }
                }
            }
            /**/

#endif

        } /* rl */

        for (i = 0; i < EKC_core_size_max; i++) {
            free(tmpvec1[i]);
        }
        free(tmpvec1);

    } /* #pragma omp parallel */
}

void Krylov_IOLP_trd(int Mc_AN, double**** OLP0, double*** Krylov_U_OLP, double** inv_RS, int* MP,
    int* Msize2, int* Msize4, int Msize2_max, double** tmpvec0, double** tmpvec1)

/* This subroutine is exactly the same as the original Krylov_IOLP,
except that Eigen_lapack is replaced with Eigen_lapack_d, _x, or _r
*/

{
    int rl, ct_AN, fan, san, can, wan, ct_on, i, j;
    int n, Anum, Bnum, k, ian, ih, kl, jg, ig, jan, m, m1, n1, info;
    int rl0, ZeroNum, Neumann_series, ns, Gh_AN, wanB;
    double sum, dum, tmp0, tmp1, tmp2, tmp3, rcutA, r0;
    double** tmpmat0;
    double *ko, *iko;
    double** FS;

    ct_AN = M2G[Mc_AN];
    fan = FNAN[ct_AN];
    san = SNAN[ct_AN];
    can = fan + san;
    wan = WhatSpecies[ct_AN];
    ct_on = Spe_Total_CNO[wan];
    rcutA = Spe_Atom_Cut1[wan];

    /* allocation of arrays */

    tmpmat0 = (double**)malloc(sizeof(double*) * (EKC_core_size[Mc_AN] + 4));
    for (i = 0; i < (EKC_core_size[Mc_AN] + 4); i++) {
        tmpmat0[i] = (double*)malloc(sizeof(double) * (EKC_core_size[Mc_AN] + 4));
    }

    FS = (double**)malloc(sizeof(double*) * (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    for (i = 0; i < (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        FS[i] = (double*)malloc(sizeof(double) * (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    }

    ko = (double*)malloc(sizeof(double) * (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]);
    iko = (double*)malloc(sizeof(double) * (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]);

    /****************************************************
        initialize
    ****************************************************/

    for (i = 0; i < EKC_core_size_max; i++) {
        for (j = 0; j < Msize2_max; j++) {
            tmpvec0[i][j] = 0.0;
            tmpvec1[i][j] = 0.0;
        }
    }

    /* find the nearest atom with distance of r0 */

    r0 = 1.0e+10;
    for (k = 1; k <= FNAN[ct_AN]; k++) {
        Gh_AN = natn[ct_AN][k];
        wanB = WhatSpecies[Gh_AN];
        if (Dis[ct_AN][k] < r0)
            r0 = Dis[ct_AN][k];
    }

    /* starting vector */

    m = 0;
    for (k = 0; k <= FNAN[ct_AN]; k++) {

        Gh_AN = natn[ct_AN][k];
        wanB = WhatSpecies[Gh_AN];

        if (Dis[ct_AN][k] < (scale_rc_EKC[Mc_AN] * r0)) {

            Anum = MP[k] - 1;

            for (i = 0; i < Spe_Total_CNO[wanB]; i++) {

                tmpvec0[m][Anum + i] = 1.0;
                Krylov_U_OLP[0][m][Anum + i] = 1.0;

                m++;
            }
        }
    }

    /*
    for (i=0; i<EKC_core_size[Mc_AN]; i++){
      tmpvec0[i][i]         = 1.0;
      Krylov_U_OLP[0][i][i] = 1.0;
    }
    */

    /****************************************************
             generate Krylov subspace vectors
    ****************************************************/

    for (rl = 0; rl < (rlmax_EC2[Mc_AN] - 1); rl++) {

        /*******************************************************
                                S * |Wn)
        *******************************************************/

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec1[n][i] = 0.0;
            }
        }

        for (i = 0; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = 0; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j] - 1;

                if (0 <= kl) {

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * tmpvec0[n][Bnum + k];
                            }

                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
                }
            }
        }

        /*************************************************************
          orthogonalization by a modified block Gram-Schmidt method
        *************************************************************/

        /* |tmpvec1) := (I - \sum_{rl0} |U_rl0)(U_rl0|)|tmpvec1) */

        for (rl0 = 0; rl0 <= rl; rl0++) {

            /* (U_rl0|tmpvec1) */

            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += Krylov_U_OLP[rl0][m][i] * tmpvec1[n][i];
                    }
                    tmpmat0[m][n] = sum;
                }
            }

            /* |tmpvec1) := |tmpvec1) - |U_rl0)(U_rl0|tmpvec1) */

            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                for (k = 0; k < EKC_core_size[Mc_AN]; k++) {
                    dum = tmpmat0[k][n];
                    for (i = 0; i < Msize2[Mc_AN]; i++)
                        tmpvec1[n][i] -= Krylov_U_OLP[rl0][k][i] * dum;
                }
            }
        }

        /*************************************************************
                           normalization of tmpvec1
        *************************************************************/

        for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
            for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                sum = 0.0;
                for (i = 0; i < Msize2[Mc_AN]; i++) {
                    sum += tmpvec1[m][i] * tmpvec1[n][i];
                }

                tmpmat0[m + 1][n + 1] = sum;
            }
        }

        /* diagonalize tmpmat0 */

        if (EKC_core_size[Mc_AN] == 1) {
            ko[1] = tmpmat0[1][1];
            tmpmat0[1][1] = 1.0;
        } else {

            info = Eigen_lapack_d(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
            if (info != 0) {
                info = Eigen_lapack_x(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
                if (info != 0) {
                    Eigen_lapack_r(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
                }
            }
        }

        ZeroNum = 0;

        for (n = 1; n <= EKC_core_size[Mc_AN]; n++) {
            if (cutoff_value < ko[n]) {
                ko[n] = sqrt(fabs(ko[n]));
                iko[n] = 1.0 / ko[n];
            } else {
                ZeroNum++;
                ko[n] = 0.0;
                iko[n] = 0.0;
            }

            if (error_check == 1) {
                printf("rl=%3d ko=%18.15f\n", rl, ko[n]);
            }
        }

        if (error_check == 1) {
            printf("rl=%3d ZeroNum=%3d\n", rl, ZeroNum);
        }

        /* tmpvec0 = tmpvec1 * tmpmat0^{-1/2} */

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec0[n][i] = 0.0;
            }
        }

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (k = 0; k < EKC_core_size[Mc_AN]; k++) {
                dum = tmpmat0[k + 1][n + 1] * iko[n + 1];
                for (i = 0; i < Msize2[Mc_AN]; i++)
                    tmpvec0[n][i] += tmpvec1[k][i] * dum;
            }
        }

        /*************************************************************
                             store Krylov vectors
        *************************************************************/

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                Krylov_U_OLP[rl + 1][n][i] = tmpvec0[n][i];
            }
        }

    } /* rl */

    /************************************************************
         calculate the inverse of the reduced overlap matrix
    ************************************************************/

    /*  construct the reduced overlap matrix */

    for (rl = 0; rl < rlmax_EC2[Mc_AN]; rl++) {

        /*  S * |Vn) */

        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                tmpvec1[n][i] = 0.0;
            }
        }

        for (i = 0; i <= can; i++) {

            ig = natn[ct_AN][i];
            ian = Spe_Total_CNO[WhatSpecies[ig]];
            Anum = MP[i] - 1;
            ih = S_G2M[ig];

            for (j = 0; j <= can; j++) {

                kl = RMI1[Mc_AN][i][j];
                jg = natn[ct_AN][j];
                jan = Spe_Total_CNO[WhatSpecies[jg]];
                Bnum = MP[j] - 1;

                if (0 <= kl) {

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                            sum = 0.0;
                            for (k = 0; k < jan; k++) {
                                sum += OLP0[ih][kl][m][k] * Krylov_U_OLP[rl][n][Bnum + k];
                            }
                            tmpvec1[n][Anum + m] += sum;
                        }
                    }
                }
            }
        }

        for (rl0 = rl; rl0 < rlmax_EC2[Mc_AN]; rl0++) {
            for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
                for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
                    sum = 0.0;
                    for (i = 0; i < Msize2[Mc_AN]; i++) {
                        sum += Krylov_U_OLP[rl0][m][i] * tmpvec1[n][i];
                    }
                    FS[rl0 * EKC_core_size[Mc_AN] + m + 1][rl * EKC_core_size[Mc_AN] + n + 1] = sum;
                    FS[rl * EKC_core_size[Mc_AN] + n + 1][rl0 * EKC_core_size[Mc_AN] + m + 1] = sum;
                }
            }
        }
    }

    /* diagonalize FS */

    info = Eigen_lapack_d(FS, ko, Msize4[Mc_AN], Msize4[Mc_AN]);
    if (info != 0) {
        info = Eigen_lapack_x(FS, ko, Msize4[Mc_AN], Msize4[Mc_AN]);
        if (info != 0) {
            Eigen_lapack_r(FS, ko, Msize4[Mc_AN], Msize4[Mc_AN]);
        }
    }

    /* find ill-conditioned eigenvalues */

    ZeroNum = 0;
    for (i = 1; i <= Msize4[Mc_AN]; i++) {

        if (error_check == 1) {
            printf("Mc_AN=%2d i=%3d ko[i]=%18.15f\n", Mc_AN, i, ko[i]);
        }

        if (cutoff_value < ko[i]) {
            iko[i] = 1.0 / ko[i];
        } else {
            ZeroNum++;
            ko[i] = 0.0;
            iko[i] = 0.0;
        }
    }

    if (error_check == 1) {
        printf("Mc_AN=%2d ZeroNum=%2d\n", Mc_AN, ZeroNum);
    }

    /* construct the inverse */

    for (i = 1; i <= Msize4[Mc_AN]; i++) {
        for (j = 1; j <= Msize4[Mc_AN]; j++) {
            sum = 0.0;
            for (k = 1; k <= Msize4[Mc_AN]; k++) {
                sum += FS[i][k] * iko[k] * FS[j][k];
            }
            inv_RS[i - 1][j - 1] = sum;
        }
    }

    /* symmetrization of inv_RS */

    for (i = 1; i <= Msize4[Mc_AN]; i++) {
        for (j = i + 1; j <= Msize4[Mc_AN]; j++) {
            tmp0 = inv_RS[i - 1][j - 1];
            tmp1 = inv_RS[j - 1][i - 1];
            tmp2 = 0.5 * (tmp0 + tmp1);
            inv_RS[i - 1][j - 1] = tmp2;
            inv_RS[j - 1][i - 1] = tmp2;
        }
    }

    /*
    {


      double mat1[1000][1000];
      double tsum;
      int myid;

      MPI_Comm_rank(mpi_comm_level1,&myid);

      if (myid==0){

        printf("check normalization\n");

        for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
          for (m=0; m<EKC_core_size[Mc_AN]; m++){
            for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
              for (n=0; n<EKC_core_size[Mc_AN]; n++){

                sum = 0.0;
                for (i=0; i<Msize2[Mc_AN]; i++){
                  sum += Krylov_U_OLP[rl][m][i]*Krylov_U_OLP[rl0][n][i];
                }
                printf("rl=%3d rl0=%3d m=%3d n=%3d  <|>=%18.15f\n",rl,rl0,m,n,sum);
              }
            }
          }
        }


        printf("\n\ninvS\n");


        for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
          for (m=0; m<EKC_core_size[Mc_AN]; m++){

            for (i=0; i<Msize2[Mc_AN]; i++){

              sum = 0.0;

              for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
                for (n=0; n<EKC_core_size[Mc_AN]; n++){
                  sum += inv_RS[rl*EKC_core_size[Mc_AN]+m][rl0*EKC_core_size[Mc_AN]+n]*Krylov_U_OLP[rl0][n][i];
                }
              }

              mat1[rl*EKC_core_size[Mc_AN]+m][i] = sum;
            }
          }
        }


        tsum = 0.0;

        for (i=0; i<Msize2[Mc_AN]; i++){
          for (j=0; j<Msize2[Mc_AN]; j++){

            sum = 0.0;
            for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
              for (m=0; m<EKC_core_size[Mc_AN]; m++){
                sum += Krylov_U_OLP[rl][m][i]*mat1[rl*EKC_core_size[Mc_AN]+m][j];
              }
            }

            printf("i=%4d j=%4d %18.15f\n",i,j,sum);

            tsum += fabs(sum);

          }
        }


        printf("\n\ntsum=%18.15f\n",tsum);

      }


      MPI_Finalize();
      exit(0);
    }
    */

    /* freeing of arrays */

    for (i = 0; i < (EKC_core_size[Mc_AN] + 4); i++) {
        free(tmpmat0[i]);
    }
    free(tmpmat0);

    for (i = 0; i < (rlmax_EC2[Mc_AN] + 2) * EKC_core_size[Mc_AN]; i++) {
        free(FS[i]);
    }
    free(FS);

    free(ko);
    free(iko);
}

void S_orthonormalize_vec_trd(int Mc_AN, int ct_on, double** vec,
    double** workvec, double**** OLP0,
    double** tmpmat0, double* ko, double* iko,
    int* MP, int* Msize2)

/* This subroutine is exactly the same as the original S_orthonormalize_vec,
except that Eigen_lapack is replaced with Eigen_lapack_d, _x, or _r
*/

{
    int n, i, j, can, san, fan, ct_AN;
    int k, m, ZeroNum, Anum, Bnum, ih;
    int kl, jg, jan, ig, ian, info;
    double dum, sum;
    double tmp0, tmp1, tmp2, tmp3;
    __m128d mmSum00, mmSum01, mmSum10, mmSum11, mmSum20, mmSum21, mmSum30, mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5;
    double mmArr[8];
    double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;

    ct_AN = M2G[Mc_AN];
    fan = FNAN[ct_AN];
    san = SNAN[ct_AN];
    can = fan + san;

    /* S|Vn) */

    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
        for (i = 0; i < Msize2[Mc_AN]; i++) {
            workvec[n][i] = 0.0;
        }
    }

    for (i = 0; i <= can; i++) {

        ig = natn[ct_AN][i];
        ian = Spe_Total_CNO[WhatSpecies[ig]];
        Anum = MP[i] - 1;
        ih = S_G2M[ig];

        for (j = 0; j <= can; j++) {

            kl = RMI1[Mc_AN][i][j];
            jg = natn[ct_AN][j];
            jan = Spe_Total_CNO[WhatSpecies[jg]];
            Bnum = MP[j] - 1;

            if (0 <= kl) {

#ifndef __x86_64__

                /* Original version */
                /**/
                for (m = 0; m < ian; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                        sum = 0.0;
                        for (k = 0; k < jan; k++) {
                            sum += OLP0[ih][kl][m][k] * vec[n][Bnum + k];
                        }

                        workvec[n][Anum + m] += sum;
                    }
                }
                /**/

#else

                /* Unrolling + SSE version */
                /**/
                for (m = 0; m < (ian - 3); m += 4) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                        mmSum00 = _mm_setzero_pd();
                        mmSum01 = _mm_setzero_pd();
                        mmSum10 = _mm_setzero_pd();
                        mmSum11 = _mm_setzero_pd();
                        mmSum20 = _mm_setzero_pd();
                        mmSum21 = _mm_setzero_pd();
                        mmSum30 = _mm_setzero_pd();
                        mmSum31 = _mm_setzero_pd();

                        for (k = 0; k < (jan - 3); k += 4) {
                            mmTmp0 = _mm_loadu_pd(&vec[n][Bnum + k + 0]);
                            mmTmp1 = _mm_loadu_pd(&vec[n][Bnum + k + 2]);

                            mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 0]), mmTmp0));
                            mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 0][k + 2]), mmTmp1));

                            mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 0]), mmTmp0));
                            mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 1][k + 2]), mmTmp1));

                            mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 0]), mmTmp0));
                            mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 2][k + 2]), mmTmp1));

                            mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 0]), mmTmp0));
                            mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m + 3][k + 2]), mmTmp1));
                        }

                        mmSum00 = _mm_add_pd(mmSum00, mmSum01);
                        mmSum10 = _mm_add_pd(mmSum10, mmSum11);
                        mmSum20 = _mm_add_pd(mmSum20, mmSum21);
                        mmSum30 = _mm_add_pd(mmSum30, mmSum31);

                        _mm_storeu_pd(&mmArr[0], mmSum00);
                        _mm_storeu_pd(&mmArr[2], mmSum10);
                        _mm_storeu_pd(&mmArr[4], mmSum20);
                        _mm_storeu_pd(&mmArr[6], mmSum30);

                        sum0 = mmArr[0] + mmArr[1];
                        sum1 = mmArr[2] + mmArr[3];
                        sum2 = mmArr[4] + mmArr[5];
                        sum3 = mmArr[6] + mmArr[7];

                        for (; k < jan; k++) {
                            sum0 += OLP0[ih][kl][m + 0][k] * vec[n][Bnum + k];
                            sum1 += OLP0[ih][kl][m + 1][k] * vec[n][Bnum + k];
                            sum2 += OLP0[ih][kl][m + 2][k] * vec[n][Bnum + k];
                            sum3 += OLP0[ih][kl][m + 3][k] * vec[n][Bnum + k];
                        }

                        workvec[n][Anum + m + 0] += sum0;
                        workvec[n][Anum + m + 1] += sum1;
                        workvec[n][Anum + m + 2] += sum2;
                        workvec[n][Anum + m + 3] += sum3;
                    }
                }

                for (; m < ian; m++) {
                    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {

                        sum = 0.0;
                        for (k = 0; k < jan; k++) {
                            sum += OLP0[ih][kl][m][k] * vec[n][Bnum + k];
                        }

                        workvec[n][Anum + m] += sum;
                    }
                }
                /**/

#endif
            }
        }
    }

    /*  (Vn|S|Vn) */

#ifndef __x86_64__

    /* Original version */
    /**/
    for (m = 0; m < EKC_core_size[Mc_AN]; m++) {
        for (n = m; n < EKC_core_size[Mc_AN]; n++) {
            sum = 0.0;
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                sum += vec[m][i] * workvec[n][i];
            }
            tmpmat0[m + 1][n + 1] = sum;
            tmpmat0[n + 1][m + 1] = sum;
        }
    }
    /**/

#else

    /* Unrolling + SSE version */
    /**/
    for (m = 0; m < (EKC_core_size[Mc_AN] - 3); m += 4) {
        for (n = m; n < EKC_core_size[Mc_AN]; n++) {
            mmSum00 = _mm_setzero_pd();
            mmSum01 = _mm_setzero_pd();
            mmSum10 = _mm_setzero_pd();
            mmSum11 = _mm_setzero_pd();
            mmSum20 = _mm_setzero_pd();
            mmSum21 = _mm_setzero_pd();
            mmSum30 = _mm_setzero_pd();
            mmSum31 = _mm_setzero_pd();

            for (i = 0; i < (Msize2[Mc_AN] - 3); i += 4) {
                mmTmp0 = _mm_loadu_pd(&workvec[n][i + 0]);
                mmTmp1 = _mm_loadu_pd(&workvec[n][i + 2]);

                mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&vec[m + 0][i + 0]), mmTmp0));
                mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&vec[m + 0][i + 2]), mmTmp1));

                mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&vec[m + 1][i + 0]), mmTmp0));
                mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&vec[m + 1][i + 2]), mmTmp1));

                mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&vec[m + 2][i + 0]), mmTmp0));
                mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&vec[m + 2][i + 2]), mmTmp1));

                mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&vec[m + 3][i + 0]), mmTmp0));
                mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&vec[m + 3][i + 2]), mmTmp1));
            }

            mmSum00 = _mm_add_pd(mmSum00, mmSum01);
            mmSum10 = _mm_add_pd(mmSum10, mmSum11);
            mmSum20 = _mm_add_pd(mmSum20, mmSum21);
            mmSum30 = _mm_add_pd(mmSum30, mmSum31);

            _mm_storeu_pd(&mmArr[0], mmSum00);
            _mm_storeu_pd(&mmArr[2], mmSum10);
            _mm_storeu_pd(&mmArr[4], mmSum20);
            _mm_storeu_pd(&mmArr[6], mmSum30);

            sum0 = mmArr[0] + mmArr[1];
            sum1 = mmArr[2] + mmArr[3];
            sum2 = mmArr[4] + mmArr[5];
            sum3 = mmArr[6] + mmArr[7];

            for (; i < Msize2[Mc_AN]; i++) {
                sum0 += vec[m + 0][i] * workvec[n][i];
                sum1 += vec[m + 1][i] * workvec[n][i];
                sum2 += vec[m + 2][i] * workvec[n][i];
                sum3 += vec[m + 3][i] * workvec[n][i];
            }

            tmpmat0[m + 1][n + 1] = sum0;
            tmpmat0[n + 1][m + 1] = sum0;

            tmpmat0[m + 2][n + 1] = sum1;
            tmpmat0[n + 1][m + 2] = sum1;

            tmpmat0[m + 3][n + 1] = sum2;
            tmpmat0[n + 1][m + 3] = sum2;

            tmpmat0[m + 4][n + 1] = sum3;
            tmpmat0[n + 1][m + 4] = sum3;
        }
    }

    for (; m < EKC_core_size[Mc_AN]; m++) {
        for (n = m; n < EKC_core_size[Mc_AN]; n++) {
            sum = 0.0;
            for (i = 0; i < Msize2[Mc_AN]; i++) {
                sum += vec[m][i] * workvec[n][i];
            }
            tmpmat0[m + 1][n + 1] = sum;
            tmpmat0[n + 1][m + 1] = sum;
        }
    }
    /**/

#endif

    /* diagonalize tmpmat0 */

    if (EKC_core_size[Mc_AN] == 1) {
        ko[1] = tmpmat0[1][1];
        tmpmat0[1][1] = 1.0;
    } else {

        info = Eigen_lapack_d(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
        if (info != 0) {
            info = Eigen_lapack_x(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
            if (info != 0) {
                Eigen_lapack_r(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
            }
        }
    }

    ZeroNum = 0;

    for (n = 1; n <= EKC_core_size[Mc_AN]; n++) {
        if (cutoff_value < ko[n]) {
            ko[n] = sqrt(fabs(ko[n]));
            iko[n] = 1.0 / ko[n];
        } else {
            ZeroNum++;
            ko[n] = 0.0;
            iko[n] = 0.0;
        }
    }

    /* U0 = vec * tmpmat0^{-1/2} */

    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
        for (i = 0; i < Msize2[Mc_AN]; i++) {
            workvec[n][i] = 0.0;
        }
    }

    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
        for (k = 0; k < EKC_core_size[Mc_AN]; k++) {
            dum = tmpmat0[k + 1][n + 1] * iko[n + 1];
            for (i = 0; i < Msize2[Mc_AN]; i++)
                workvec[n][i] += vec[k][i] * dum;
        }
    }

    for (n = 0; n < EKC_core_size[Mc_AN]; n++) {
        for (i = 0; i < Msize2[Mc_AN]; i++) {
            vec[n][i] = workvec[n][i];
        }
    }
}

int Eigen_lapack_x(double** a, double* ko, int n0, int EVmax)
{

    /*
      F77_NAME(dsyevx,DSYEVX)()

      input:  n;
      input:  a[n][n];  matrix A
      output: a[n][n];  eigevectors
      output: ko[n];    eigenvalues
    */

    char* name = "Eigen_lapack";

    char* JOBZ = "V";
    char* RANGE = "I";
    char* UPLO = "L";

    INTEGER n = n0;
    INTEGER LDA = n0;
    double VL, VU; /* dummy */
    INTEGER IL, IU;
    double ABSTOL = 1.0e-13;
    INTEGER M;

    double *A, *Z;
    INTEGER LDZ = n;
    INTEGER LWORK;
    double* WORK;
    INTEGER* IWORK;
    INTEGER *IFAIL, INFO;

    int i, j;

    A = (double*)malloc(sizeof(double) * n * n);
    Z = (double*)malloc(sizeof(double) * n * n);

    LWORK = n * 8;
    WORK = (double*)malloc(sizeof(double) * LWORK);
    IWORK = (INTEGER*)malloc(sizeof(INTEGER) * n * 5);
    IFAIL = (INTEGER*)malloc(sizeof(INTEGER) * n);

    IL = 1;
    IU = EVmax;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i * n + j] = a[i + 1][j + 1];
        }
    }

#if 0
  printf("A=\n");
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
       printf("%f ",A[i*n+j]);
    }
    printf("\n");
  }
  fflush(stdout);
#endif

    F77_NAME(dsyevx, DSYEVX)
    (JOBZ, RANGE, UPLO, &n, A, &LDA, &VL, &VU, &IL, &IU,
        &ABSTOL, &M, ko, Z, &LDZ, WORK, &LWORK, IWORK,
        IFAIL, &INFO);

    if (INFO == 0) {

        /* store eigenvectors */
        for (i = 0; i < EVmax; i++) {
            for (j = 0; j < n; j++) {
                /*  a[i+1][j+1]= Z[i*n+j]; */
                a[j + 1][i + 1] = Z[i * n + j];
            }
        }

        /* shift ko by 1 */
        for (i = EVmax; i >= 1; i--) {
            ko[i] = ko[i - 1];
        }
    }

    if (INFO != 0) {
        /* printf("\n%s: error in dsyevx_, info=%d\n\n",name,INFO); */
    }

    free(IFAIL);
    free(IWORK);
    free(WORK);
    free(Z);
    free(A);
    return INFO;
}

int Eigen_lapack_d(double** a, double* ko, int n0, int EVmax)
{

    /*
      F77_NAME(dsyevd,DSYEVD)()

      input:  n;
      input:  a[n][n];  matrix A
      output: a[n][n];  eigevectors
      output: ko[n];    eigenvalues
    */

    static char* name = "Eigen_lapack";

    char* JOBZ = "V";
    char* UPLO = "L";

    INTEGER n = n0;
    INTEGER LDA = n;
    double VL, VU; /* dummy */
    INTEGER IL, IU;
    double ABSTOL = 1.0e-13;
    INTEGER M;

    double* A;
    INTEGER LDZ = n;
    INTEGER LWORK, LIWORK;
    double* WORK;
    INTEGER* IWORK;
    INTEGER INFO;

    int i, j;

    A = (double*)malloc(sizeof(double) * n * n);

    LWORK = 1 + 6 * n + 2 * n * n;
    WORK = (double*)malloc(sizeof(double) * LWORK);

    LIWORK = 3 + 5 * n;
    IWORK = (INTEGER*)malloc(sizeof(INTEGER) * LIWORK);

    IL = 1;
    IU = EVmax;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i * n + j] = a[i + 1][j + 1];
        }
    }

#if 0
  printf("A=\n");
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
       printf("%f ",A[i*n+j]);
    }
    printf("\n");
  }
  fflush(stdout);
#endif

    F77_NAME(dsyevd, DSYEVD)
    (JOBZ, UPLO, &n, A, &LDA, ko, WORK, &LWORK, IWORK, &LIWORK, &INFO);

    if (INFO == 0) {

        /* store eigenvectors */
        for (i = 0; i < EVmax; i++) {
            for (j = 0; j < n; j++) {
                /* a[i+1][j+1]= Z[i*n+j]; */
                a[j + 1][i + 1] = A[i * n + j];
            }
        }

        /* shift ko by 1 */
        for (i = EVmax; i >= 1; i--) {
            ko[i] = ko[i - 1];
        }
    }

    if (INFO != 0) {
        /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
    }

    free(IWORK);
    free(WORK);
    free(A);
    return INFO;
}

int Eigen_lapack_r(double** a, double* ko, int n0, int EVmax)
{

    /*
      F77_NAME(dsyevr,DSYEVR)()

      input:  n;
      input:  a[n][n];  matrix A
      output: a[n][n];  eigevectors
      output: ko[n];    eigenvalues
    */

    static char* name = "Eigen_lapack";

    char* JOBZ = "V";
    char* RANGE = "I";
    char* UPLO = "L";

    INTEGER n = n0;
    INTEGER LDA = n;
    double VL, VU; /* dummy */
    INTEGER IL, IU;
    double ABSTOL = 1.0e-13;
    INTEGER M;

    double *A, *Z;
    INTEGER LDZ = n;
    INTEGER LWORK, LIWORK;
    double* WORK;
    INTEGER* IWORK;
    INTEGER* ISUPPZ;
    INTEGER INFO;

    int i, j;

    A = (double*)malloc(sizeof(double) * n * n);
    Z = (double*)malloc(sizeof(double) * n * n);

    LWORK = (n + 16) * n; /*  n*26 of (n+6)*n */
    WORK = (double*)malloc(sizeof(double) * LWORK);

    LIWORK = (n + 1) * n; /*  n*10 or ??? */
    IWORK = (INTEGER*)malloc(sizeof(INTEGER) * LIWORK);

    ISUPPZ = (INTEGER*)malloc(sizeof(INTEGER) * n * 2);

    IL = 1;
    IU = EVmax;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i * n + j] = a[i + 1][j + 1];
        }
    }

#if 0
  printf("A=\n");
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
       printf("%f ",A[i*n+j]);
    }
    printf("\n");
  }
  fflush(stdout);
#endif

    F77_NAME(dsyevr, DSYEVR)
    (JOBZ, RANGE, UPLO, &n, A, &LDA, &VL, &VU, &IL, &IU,
        &ABSTOL, &M, ko, Z, &LDZ, ISUPPZ, WORK, &LWORK,
        IWORK, &LIWORK, &INFO);

    if (INFO == 0) {

        /* store eigenvectors */
        for (i = 0; i < EVmax; i++) {
            for (j = 0; j < n; j++) {
                /*  a[i+1][j+1]= Z[i*n+j]; */
                a[j + 1][i + 1] = Z[i * n + j];
            }
        }

        /* shift ko by 1 */
        for (i = EVmax; i >= 1; i--) {
            ko[i] = ko[i - 1];
        }
    }

    if (INFO != 0) {
        printf("\n%s: error in dsyevr_, info=%d\n\n", name, INFO);
        MPI_Finalize();
        exit(10);
    }

    free(ISUPPZ);
    free(IWORK);
    free(WORK);
    free(Z);
    free(A);

    return INFO;
}
