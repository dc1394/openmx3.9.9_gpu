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
#include "set_cuda_default_device_from_local_rank.h"
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define measure_time 0
#define GPU_CPU_SWITCH_NUM2 1800

/* task-level threshold for proxy-to-GPU */
#define DCLNO_GPU_PROXY_EIGEN_THRESHOLD_COL  (GPU_CPU_SWITCH_NUM2)

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

/* ------------------------------------------------------------------ */
/* communicator selection                                             */
/* ------------------------------------------------------------------ */

static MPI_Comm DCLNO_GPUProxy_BaseComm(void)
{
    return (mpi_comm_level1 != MPI_COMM_NULL) ? mpi_comm_level1 : MPI_COMM_WORLD;
}

static void DCLNO_GPUProxy_Init(void)
{
    MPI_Comm base_comm;
    int color;

    if (DCLNO_gpu_proxy_initialized) return;

    base_comm = DCLNO_GPUProxy_BaseComm();

    MPI_Comm_split_type(base_comm, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &DCLNO_node_comm);
    MPI_Comm_rank(DCLNO_node_comm, &DCLNO_node_rank);
    MPI_Comm_size(DCLNO_node_comm, &DCLNO_node_size);

    cudaGetDeviceCount(&DCLNO_ngpu);
    if (DCLNO_ngpu <= 0) {
        fprintf(stderr, "DC-LNO GPU proxy: No CUDA device found on node.\n");
        MPI_Abort(base_comm, 1);
    }

    DCLNO_gpu_id = DCLNO_node_rank % DCLNO_ngpu;
    color = DCLNO_gpu_id;

    MPI_Comm_split(DCLNO_node_comm, color, DCLNO_node_rank, &DCLNO_gpu_group_comm);
    MPI_Comm_rank(DCLNO_gpu_group_comm, &DCLNO_gpu_group_rank);
    MPI_Comm_size(DCLNO_gpu_group_comm, &DCLNO_gpu_group_size);

    DCLNO_is_gpu_owner = (DCLNO_gpu_group_rank == 0);

    if (DCLNO_is_gpu_owner) {
        /*
         * Only the GPU owners enter this branch.
         * set_cuda_default_device_from_local_rank() internally calls
         * MPI_Comm_split_type(comm, ...), which is collective on comm and
         * deadlocks if only a subset of ranks calls it.
         */
        wait_cudafunc(cudaSetDevice(DCLNO_gpu_id));
    }

    DCLNO_gpu_proxy_initialized = 1;
}

/* ------------------------------------------------------------------ */
/* utilities                                                          */
/* ------------------------------------------------------------------ */

static double **DCLNO_AllocRealMatrix1B(int n)
{
    int i;
    double **a = (double**)malloc(sizeof(double*)*(n+1));
    for (i=0; i<=n; i++) {
        a[i] = (double*)malloc(sizeof(double)*(n+1));
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

/* ------------------------------------------------------------------ */
/* local dense solve kernel (collinear)                               */
/* ------------------------------------------------------------------ */

static void DCLNO_Solve_Col_Local(int use_gpu,
                                  int NUM,
                                  int NUM2,
                                  double *Smat,
                                  double *Hmat,
                                  double **C,
                                  double *ko)
{
    int i1, j1, l;
    double *Tmp;
    double alpha, beta;
    int BM, BN, BK;

    Tmp = (double*)malloc(sizeof(double)*NUM*NUM);

    /* diagonalize S */
    for (i1=1; i1<=NUM; i1++) {
        for (j1=1; j1<=NUM; j1++) {
            C[i1][j1] = Smat[(j1-1)*NUM + (i1-1)];
        }
    }

    if (use_gpu) {
        Eigen_cusolver_d(C, ko, NUM, NUM);
    }
    else {
        Eigen_lapack_d(C, ko, NUM, NUM);
    }

    for (l=1; l<=NUM; l++) ko[l] = 1.0/sqrt(fabs(ko[l]));

    /* Smat <- U * s^{-1/2} */
    for (i1=1; i1<=NUM; i1++) {
        for (j1=1; j1<=NUM; j1++) {
            Smat[(j1-1)*NUM + (i1-1)] = C[i1][j1]*ko[j1];
        }
    }

    /* Tmp = H * Smat */
    BM = NUM; BN = NUM; BK = NUM;
    if (use_gpu) {
        my_cublasDgemm(CUBLAS_OP_N, CUBLAS_OP_N, BM, BN, BK,
                       Hmat, Smat, Tmp);
    }
    else {
        alpha = 1.0; beta = 0.0;
        F77_NAME(dgemm,DGEMM)
        ("N","N",&BM,&BN,&BK,&alpha,
         Hmat,&BM,Smat,&BK,&beta,Tmp,&BM);
    }

    /* Hmat = Smat^+ * Tmp */
    if (use_gpu) {
        my_cublasDgemm(CUBLAS_OP_C, CUBLAS_OP_N, BM, BN, BK,
                       Smat, Tmp, Hmat);
    }
    else {
        alpha = 1.0; beta = 0.0;
        F77_NAME(dgemm,DGEMM)
        ("C","N",&BM,&BN,&BK,&alpha,
         Smat,&BM,Tmp,&BK,&beta,Hmat,&BM);
    }

    /* diagonalize transformed H */
    for (i1=1; i1<=NUM; i1++) {
        for (j1=1; j1<=NUM; j1++) {
            C[i1][j1] = Hmat[(j1-1)*NUM + (i1-1)];
        }
    }

    if (use_gpu) {
        Eigen_cusolver_d(C, ko, NUM, NUM2);
    }
    else {
        Eigen_lapack_d(C, ko, NUM, NUM2);
    }

    /* Hmat (first NUM2 cols) <- transformed eigenvectors */
    for (i1=1; i1<=NUM; i1++) {
        for (j1=1; j1<=NUM2; j1++) {
            Hmat[(j1-1)*NUM + (i1-1)] = C[i1][j1];
        }
    }

    /* Tmp(:,1:NUM2) = Smat * Hmat(:,1:NUM2) */
    BM = NUM; BN = NUM2; BK = NUM;
    if (use_gpu) {
        my_cublasDgemm(CUBLAS_OP_N, CUBLAS_OP_N, BM, BN, BK,
                       Smat, Hmat, Tmp);
    }
    else {
        alpha = 1.0; beta = 0.0;
        F77_NAME(dgemm,DGEMM)
        ("N","N",&BM,&BN,&BK,&alpha,
         Smat,&BM,Hmat,&BK,&beta,Tmp,&BM);
    }

    /* return C[eig_index][basis_index] */
    for (j1=1; j1<=NUM2; j1++) {
        for (i1=1; i1<=NUM; i1++) {
            C[j1][i1] = Tmp[(j1-1)*NUM + (i1-1)];
        }
    }

    free(Tmp);
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

    allinfo = (int*)malloc(sizeof(int)*3*size);
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

        recv_Sbuf = (double**)calloc(size, sizeof(double*));
        recv_Hbuf = (double**)calloc(size, sizeof(double*));
        recv_Sreq = (MPI_Request*)malloc(sizeof(MPI_Request) * size);
        recv_Hreq = (MPI_Request*)malloc(sizeof(MPI_Request) * size);

        for (src = 0; src < size; src++) {
            recv_Sreq[src] = MPI_REQUEST_NULL;
            recv_Hreq[src] = MPI_REQUEST_NULL;
        }

        for (src = 1; src < size; src++) {
            int s_active = allinfo[3*src + 0];
            int s_num    = allinfo[3*src + 1];

            if (!s_active) continue;

            recv_Sbuf[src] = (double*)malloc(sizeof(double) * s_num * s_num);
            recv_Hbuf[src] = (double*)malloc(sizeof(double) * s_num * s_num);

            MPI_Irecv(recv_Sbuf[src], s_num*s_num, MPI_DOUBLE, src,
                      DCLNO_PROXY_TAG_COL_S, DCLNO_gpu_group_comm, &recv_Sreq[src]);
            MPI_Irecv(recv_Hbuf[src], s_num*s_num, MPI_DOUBLE, src,
                      DCLNO_PROXY_TAG_COL_H, DCLNO_gpu_group_comm, &recv_Hreq[src]);
        }

        /* owner handles its own task first */
        if (active) {
            DCLNO_Solve_Col_Local(1, NUM, NUM2, Smat, Hmat, C, ko);
        }

        /* then serve other ranks in the group */
        for (src=1; src<size; src++) {
            int s_active = allinfo[3*src + 0];
            int s_num    = allinfo[3*src + 1];
            int s_num2   = allinfo[3*src + 2];

            if (!s_active) continue;

            {
                double *Sbuf, *Hbuf, *Cbuf;
                double *kbuf;
                double **Ct;

                MPI_Wait(&recv_Sreq[src], MPI_STATUS_IGNORE);
                MPI_Wait(&recv_Hreq[src], MPI_STATUS_IGNORE);

                Sbuf = recv_Sbuf[src];
                Hbuf = recv_Hbuf[src];
                Cbuf = (double*)malloc(sizeof(double)*s_num*s_num);
                kbuf = (double*)malloc(sizeof(double)*(s_num+2));
                Ct   = DCLNO_AllocRealMatrix1B(s_num);

                DCLNO_Solve_Col_Local(1, s_num, s_num2, Sbuf, Hbuf, Ct, kbuf);

                DCLNO_PackRealC(Ct, s_num, Cbuf);

                MPI_Send(&kbuf[1], s_num2, MPI_DOUBLE, src,
                         DCLNO_PROXY_TAG_COL_EVAL, DCLNO_gpu_group_comm);
                MPI_Send(Cbuf, s_num*s_num, MPI_DOUBLE, src,
                         DCLNO_PROXY_TAG_COL_CVEC, DCLNO_gpu_group_comm);

                DCLNO_FreeRealMatrix1B(Ct, s_num);
                free(Sbuf);
                free(Hbuf);
                free(Cbuf);
                free(kbuf);
            }
        }

        free(recv_Sbuf);
        free(recv_Hbuf);
        free(recv_Sreq);
        free(recv_Hreq);
    }
    else if (active) {
        double *Cbuf = (double*)malloc(sizeof(double)*NUM*NUM);

        MPI_Send(Smat, NUM*NUM, MPI_DOUBLE, 0,
                 DCLNO_PROXY_TAG_COL_S, DCLNO_gpu_group_comm);
        MPI_Send(Hmat, NUM*NUM, MPI_DOUBLE, 0,
                 DCLNO_PROXY_TAG_COL_H, DCLNO_gpu_group_comm);

        MPI_Recv(&ko[1], NUM2, MPI_DOUBLE, 0,
                 DCLNO_PROXY_TAG_COL_EVAL, DCLNO_gpu_group_comm, MPI_STATUS_IGNORE);
        MPI_Recv(Cbuf, NUM*NUM, MPI_DOUBLE, 0,
                 DCLNO_PROXY_TAG_COL_CVEC, DCLNO_gpu_group_comm, MPI_STATUS_IGNORE);

        DCLNO_UnpackRealC(Cbuf, NUM, C);

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
    int             l, i1, j1, i2, ip, po1, m_size, k;
    int             po, loopN, tno1, tno2, h_AN, Gh_AN, iwan, jwan;
    int             MA_AN, GA_AN, GB_AN, tnoA, tnoB, ino, jno;
    int             size_Residues;
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
    double *        BLAS_H, *BLAS_C;
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
    int             use_gpu_proxy;
    int             group_max_atoms, atom_slot;
    int             have_local_atom, use_gpu_task;

    MPI_Comm_size(mpi_comm_level1, &numprocs0);
    MPI_Comm_rank(mpi_comm_level1, &myid0);

    dtime(&TStime);

    time0 = 0.0;
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time6 = 0.0;
    time7 = 0.0;
    time8 = 0.0;
    time9 = 0.0;

    use_gpu_proxy = (scf_eigen_lib_flag == CuSOLVER);
    if (use_gpu_proxy) {
        DCLNO_GPUProxy_Init();
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
        LO_TC = (int**)malloc(sizeof(int*) * 2);
        HO_TC = (int**)malloc(sizeof(int*) * 2);

        for (spin = 0; spin < 2; spin++) {
            LO_TC[spin] = (int*)malloc(sizeof(int) * (Matomnum + 1));
            HO_TC[spin] = (int*)malloc(sizeof(int) * (Matomnum + 1));
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

    Msize = (int*)malloc(sizeof(int) * (Matomnum + 1));
    Max_Msize = 0;
    m_size = 0;

    EVal = (double***)malloc(sizeof(double**) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        EVal[spin] = (double**)malloc(sizeof(double*) * (Matomnum + 1));

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
            EVal[spin][Mc_AN] = (double*)malloc(sizeof(double) * n2);
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
        BLAS_OLP = (double*)malloc(sizeof(double) * Max_Msize * Max_Msize * (SpinP_switch + 1));
        BLAS_allocate_flag = 1;
    }

    BLAS_H = (double*)malloc(sizeof(double) * Max_Msize * Max_Msize * (SpinP_switch + 1));
    BLAS_C = (double*)malloc(sizeof(double) * Max_Msize * Max_Msize);

    ko = (double*)malloc(sizeof(double) * (Max_Msize + 2));
    C  = DCLNO_AllocRealMatrix1B(Max_Msize + 1);

    Sc   = (double*)malloc(sizeof(double) * List_YOUSO[7] * List_YOUSO[7]);
    Hc   = (double*)malloc(sizeof(double) * List_YOUSO[7] * List_YOUSO[7]);
    Stmp = (double*)malloc(sizeof(double) * List_YOUSO[7] * List_YOUSO[7]);
    Htmp = (double*)malloc(sizeof(double) * List_YOUSO[7] * List_YOUSO[7]);

    Snd_H_Size = (int*)malloc(sizeof(int) * numprocs0);
    Rcv_H_Size = (int*)malloc(sizeof(int) * numprocs0);
    Snd_S_Size = (int*)malloc(sizeof(int) * numprocs0);
    Rcv_S_Size = (int*)malloc(sizeof(int) * numprocs0);

    MP = (int*)malloc(sizeof(int) * List_YOUSO[2]);

    /****************************************************
      allocation of Residues
    ****************************************************/

    Residues = (double******)malloc(sizeof(double*****) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        Residues[spin] = (double*****)malloc(sizeof(double****) * Matomnum);
        for (Mc_AN = 0; Mc_AN < Matomnum; Mc_AN++) {
            Gc_AN = M2G[Mc_AN + 1];
            wanA  = WhatSpecies[Gc_AN];
            tno1  = Spe_Total_CNO[wanA];

            Residues[spin][Mc_AN] = (double****)malloc(sizeof(double***) * (FNAN[Gc_AN] + 1));

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                Gh_AN = natn[Gc_AN][h_AN];
                wanB  = WhatSpecies[Gh_AN];
                tno2  = Spe_Total_CNO[wanB];

                Residues[spin][Mc_AN][h_AN] = (double***)malloc(sizeof(double**) * tno1);
                for (i = 0; i < tno1; i++) {
                    Residues[spin][Mc_AN][h_AN][i] = (double**)malloc(sizeof(double*) * tno2);
                }
            }
        }
    }

    /****************************************************
      PDOS
    ****************************************************/

    PDOS_DC = (double***)malloc(sizeof(double**) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        PDOS_DC[spin] = (double**)malloc(sizeof(double*) * Matomnum);
        for (Mc_AN = 0; Mc_AN < Matomnum; Mc_AN++) {
            PDOS_DC[spin][Mc_AN] = (double*)malloc(sizeof(double) * (Msize[Mc_AN + 1] + 3));
        }
    }

    if (firsttime) {
        PrintMemory("Divide_Conquer_LNO(col): EVal", sizeof(double) * m_size, NULL);
        PrintMemory("Divide_Conquer_LNO(col): BLAS_OLP",
                    sizeof(double) * Max_Msize * Max_Msize * (SpinP_switch + 1), NULL);
        PrintMemory("Divide_Conquer_LNO(col): BLAS_H",
                    sizeof(double) * Max_Msize * Max_Msize * (SpinP_switch + 1), NULL);
        PrintMemory("Divide_Conquer_LNO(col): BLAS_C",
                    sizeof(double) * Max_Msize * Max_Msize, NULL);
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
                tmp_array = (double*)malloc(sizeof(double) * size1);

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
                tmp_array2 = (double*)malloc(sizeof(double) * size2);
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
                    tmp_array = (double*)malloc(sizeof(double) * size1);

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
                    tmp_array2 = (double*)malloc(sizeof(double) * size2);
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

                use_gpu_task = (use_gpu_proxy && NUM >= DCLNO_GPU_PROXY_EIGEN_THRESHOLD_COL);

                if (measure_time) dtime(&stime);

                if (!use_gpu_task) {
                    DCLNO_Solve_Col_Local(0, NUM, NUM2,
                                          &BLAS_OLP[spin * NUM * NUM],
                                          &BLAS_H[spin * NUM * NUM],
                                          C, ko);
                }
            }
            else {
                NUM = 0;
                NUM2 = 0;
                use_gpu_task = 0;
            }

            if (use_gpu_proxy) {
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
                        Residues[spin][Mc_AN - 1][h_AN][i][j] = (double*)malloc(sizeof(double) * n2);
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

    dtime(&TEtime);
    return (TEtime - TStime);
}

/* ------------------------------------------------------------------ */
/* noncollinear stub                                                  */
/* ------------------------------------------------------------------ */

static double DC_NonCol(char * mode, int MD_iter, int SCF_iter, int SucceedReadingDMfile,
                        double ***** Hks, double ***** ImNL, double **** OLP0,
                        double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2])
{
    int myid;
    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (myid == Host_ID) {
        fprintf(stderr,
                "Error: this Divide_Conquer_LNO.c is a collinear-only version. "
                "Noncollinear (SpinP_switch==3) is not implemented here.\n");
        fflush(stderr);
    }
    MPI_Abort(mpi_comm_level1, 1);
    return 0.0;
}

static void Save_DOS_NonCol(dcomplex ****** Residues, double **** OLP0, double ** EVal,
                            int * LO_TC, int * HO_TC)
{
    int myid;
    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (myid == Host_ID) {
        fprintf(stderr,
                "Error: Save_DOS_NonCol was called in a collinear-only Divide_Conquer_LNO.c.\n");
        fflush(stderr);
    }
    MPI_Abort(mpi_comm_level1, 1);
}

/* ------------------------------------------------------------------ */
/* DOS writer (collinear)                                             */
/* ------------------------------------------------------------------ */

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
        sprintf(file_eig, "%s%s.Dos.val", filepath, filename);
        fp_eig = fopen(file_eig, "w");
        if (fp_eig == NULL) {
            printf("cannot open a file %s\n", file_eig);
        }
    }

    sprintf(file_ev, "%s%s.Dos.vec%i", filepath, filename, myid);
    fp_ev = fopen(file_ev, "w");
    if (fp_ev == NULL) {
        printf("cannot open a file %s\n", file_ev);
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
