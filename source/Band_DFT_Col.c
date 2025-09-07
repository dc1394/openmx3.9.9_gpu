/**********************************************************************
  Band_DFT_Col.c:

     Band_DFT_Col.c is a subroutine to perform band calculations
     based on a collinear DFT

  Log of Band_DFT_Col.c:

     22/Nov/2001  Released by T. Ozaki

***********************************************************************/

#include "lapack_prototypes.h"
#include "mpi.h"
#include "openmx_common.h"
#include "tran_variables.h"
#include <assert.h>
#include <math.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define measure_time 0

void set_up_blacsgrid_f(int mpi_comm_parent, int np_rows, int np_cols, char layout, int * my_blacs_ctxt, int * my_prow,
                        int * my_pcol);
void set_up_blacs_descriptor_f(int na, int nblk, int my_prow, int my_pcol, int np_rows, int np_cols, int * na_rows,
                               int * na_cols, int sc_desc[9], int my_blacs_ctxt, int * info);

void solve_evp_complex_(int * n1, int * n2, dcomplex * Cs, int * na_rows1, double * ko, dcomplex * Ss, int * na_rows2,
                        int * nblk, int * mpi_comm_rows_int, int * mpi_comm_cols_int);

void elpa_solve_evp_complex_2stage_double_impl_(int * n, int * MaxN, dcomplex * Hs, int * na_rows1, double * ko,
                                                dcomplex * Cs, int * na_rows2, int * nblk, int * na_cols1,
                                                int * mpi_comm_rows_int, int * mpi_comm_cols_int, int * mpiworld);

static void Construct_Band_CsHs(int SCF_iter, int all_knum, int * order_GA, int * MP, double * S1, double * H1,
                                double k1, double k2, double k3, dcomplex * Cs, dcomplex * Hs, int n, int myid2);

static double get_max_value(double localValue);

double Band_DFT_Col(int SCF_iter, int knum_i, int knum_j, int knum_k, int SpinP_switch, double ***** nh,
                    double ***** ImNL, double **** CntOLP, double ***** CDM, double ***** EDM, double Eele0[2],
                    double Eele1[2], int * MP, int * order_GA, double * ko, double * koS, double *** EIGEN, double * H1,
                    double * S1, double * CDM1, double * EDM1, dcomplex ** EVec1, dcomplex * Ss, dcomplex * Cs,
                    dcomplex * Hs, int *** k_op, int * T_k_op, int ** T_k_ID, double * T_KGrids1, double * T_KGrids2,
                    double * T_KGrids3, int myworld1, int * NPROCS_ID1, int * Comm_World1, int * NPROCS_WD1,
                    int * Comm_World_StartID1, MPI_Comm * MPI_CommWD1, int myworld2, int * NPROCS_ID2, int * NPROCS_WD2,
                    int * Comm_World2, int * Comm_World_StartID2, MPI_Comm * MPI_CommWD2)
{
    static int firsttime = 1;
    int        i, j, k, l, m, n, p, wan, MaxN, i0, ks;
    int        i1, i1s, j1, ia, jb, lmax, kmin, kmax, po, po1, spin, s1, e1;
    int        num2, RnB, l1, l2, l3, loop_num, ns, ne;
    int        ct_AN, h_AN, wanA, tnoA, wanB, tnoB;
    int        MA_AN, GA_AN, Anum, num_kloop0;
    int        T_knum, S_knum, E_knum, kloop, kloop0;
    double     av_num, lumos;
    double     time0;
    int        LB_AN, GB_AN, Bnum;
    double     k1, k2, k3, Fkw;
    double     sum, sumi, sum_weights;
    double     Num_State;
    double     My_Num_State;
    double     FermiF, tmp1;
    double     tmp, eig, kw, EV_cut0;
    double     x, Dnum, Dnum2, AcP, ChemP_MAX, ChemP_MIN;
    int *      is1, *ie1;
    int *      is2, *ie2;
    int *      My_NZeros;
    int *      SP_NZeros;
    int *      SP_Atoms;

    int      all_knum;
    dcomplex Ctmp1, Ctmp2;
    int      ii, ij, ik;
    int      BM, BN, BK;
    double   u2, v2, uv, vu;
    double   d1, d2, d3, d4, ReA, ImA;
    double   My_Eele1[2];
    double   TZ, dum, sumE, kRn, si, co;
    double   Resum, ResumE, Redum, Redum2;
    double   Imsum, ImsumE, Imdum, Imdum2;
    double   TStime, TEtime, SiloopTime, EiloopTime;
    double   Stime = 0.0, Etime = 0.0, Stime0, Etime0;
    double   Stime1, Etime1;
    double   FermiEps = 1.0e-13;
    double   x_cut    = 60.0;
    double   My_Eele0[2];

    char   file_EV[YOUSO10];
    FILE * fp_EV;
    char   buf[fp_bsize]; /* setvbuf */
    int    AN, Rn, size_H1;
    int    parallel_mode;
    int    numprocs0, myid0;
    int    ID, ID0, ID1;
    int    numprocs1, myid1;
    int    numprocs2, myid2;
    int    Num_Comm_World1;
    int    Num_Comm_World2;

    int         tag = 999, IDS, IDR;
    MPI_Status  stat;
    MPI_Request request;

    double time1, time2 = 0.0, time3 = 0.0;
    double time4 = 0.0, time5 = 0.0, time6;
    double time7, time8, time9;
    double time10, time11, time12;
    double time81, time82, time83;
    double time84, time85;
    double time51, time11A = 0.0, time11B = 0.0;

    MPI_Comm mpi_comm_rows, mpi_comm_cols;
    int      mpi_comm_rows_int, mpi_comm_cols_int;
    int      info, ig, jg, il, jl, prow, pcol, brow, bcol;
    int      ZERO = 0, ONE = 1;
    dcomplex alpha = {1.0, 0.0};
    dcomplex beta  = {0.0, 0.0};

    int    LOCr, LOCc, node, irow, icol;
    double mC_spin_i1, C_spin_i1;

    int     Max_Num_Snd_EV, Max_Num_Rcv_EV;
    int *   Num_Snd_EV, *Num_Rcv_EV;
    int *   index_Snd_i, *index_Snd_j, *index_Rcv_i, *index_Rcv_j;
    double *EVec_Snd, *EVec_Rcv;
    double *TmpEIGEN, **ReEVec0, **ImEVec0, **ReEVec1, **ImEVec1;

    /* for time */
    dtime(&TStime);

    time1  = 0.0;
    time2  = 0.0;
    time3  = 0.0;
    time4  = 0.0;
    time5  = 0.0;
    time6  = 0.0;
    time7  = 0.0;
    time8  = 0.0;
    time9  = 0.0;
    time10 = 0.0;
    time11 = 0.0;
    time12 = 0.0;
    time81 = 0.0;
    time82 = 0.0;
    time83 = 0.0;
    time84 = 0.0;
    time85 = 0.0;
    time51 = 0.0;

    double        part1      = 0.0;
    double        part2_1    = 0.0;
    double        part2_2    = 0.0;
    double        part2_3    = 0.0;
    double        part2_4    = 0.0;
    double        part2_5    = 0.0;
    double        part3      = 0.0;
    double        part4      = 0.0;
    double        partmul    = 0.0;
    static double part1sum   = 0.0;
    static double part2_1sum = 0.0;
    static double part2_2sum = 0.0;
    static double part2_3sum = 0.0;
    static double part2_4sum = 0.0;
    static double part2_5sum = 0.0;
    static double part3sum   = 0.0;
    static double part4sum   = 0.0;
    static double partmulsum = 0.0;
    double        starttime, endtime;

    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs0);
    MPI_Comm_rank(mpi_comm_level1, &myid0);

    MPI_Barrier(mpi_comm_level1);

    Num_Comm_World1 = SpinP_switch + 1;

    /***********************************************
         for pallalel calculations in myworld1
    ***********************************************/

    MPI_Comm_size(MPI_CommWD1[myworld1], &numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1], &myid1);

    MPI_Comm_rank(MPI_CommWD2[myworld2], &myid2);

    /****************************************************
     find the number of basis functions, n
    ****************************************************/

    n = 0;
    for (i = 1; i <= atomnum; i++) {
        wanA = WhatSpecies[i];
        n += Spe_Total_CNO[wanA];
    }

    /****************************************************
     find TZ
    ****************************************************/

    TZ = 0.0;
    for (i = 1; i <= atomnum; i++) {
        wan = WhatSpecies[i];
        TZ += Spe_Core_Charge[wan];
    }

    /***********************************************
       find the number of states to be solved
    ***********************************************/

    lumos = 0.4 * ((double)TZ - (double)system_charge) / 2.0;
    if (lumos < 50.0)
        lumos = 100.0;
    MaxN = (TZ - system_charge) / 2 + (int)lumos;
    if (n < MaxN)
        MaxN = n;

    /***********************************************
       allocation of arrays
    ***********************************************/

    My_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_Atoms  = (int *)malloc(sizeof(int) * numprocs0);

    TmpEIGEN = (double *)malloc(sizeof(double) * (MaxN + 1));

    ReEVec0 = (double **)malloc(sizeof(double *) * List_YOUSO[7]);
    for (i = 0; i < List_YOUSO[7]; i++) {
        ReEVec0[i] = (double *)malloc(sizeof(double *) * (MaxN + 1));
    }

    ImEVec0 = (double **)malloc(sizeof(double *) * List_YOUSO[7]);
    for (i = 0; i < List_YOUSO[7]; i++) {
        ImEVec0[i] = (double *)malloc(sizeof(double *) * (MaxN + 1));
    }

    ReEVec1 = (double **)malloc(sizeof(double *) * List_YOUSO[7]);
    for (i = 0; i < List_YOUSO[7]; i++) {
        ReEVec1[i] = (double *)malloc(sizeof(double *) * (MaxN + 1));
    }

    ImEVec1 = (double **)malloc(sizeof(double *) * List_YOUSO[7]);
    for (i = 0; i < List_YOUSO[7]; i++) {
        ImEVec1[i] = (double *)malloc(sizeof(double *) * (MaxN + 1));
    }

    /***********************************************
                k-points by regular mesh
    ***********************************************/

    if (way_of_kpoint == 1) {

        /**************************************************************
         k_op[i][j][k]: weight of DOS
                     =0   no calc.
                     =1   G-point
                     =2   which has k<->-k point
            Now, only the relation, E(k)=E(-k), is used.

        Future release: k_op will be used for symmetry operation
        *************************************************************/

        for (i = 0; i < knum_i; i++) {
            for (j = 0; j < knum_j; j++) {
                for (k = 0; k < knum_k; k++) {
                    k_op[i][j][k] = -999;
                }
            }
        }

        for (i = 0; i < knum_i; i++) {
            for (j = 0; j < knum_j; j++) {
                for (k = 0; k < knum_k; k++) {
                    if (k_op[i][j][k] == -999) {
                        k_inversion(i, j, k, knum_i, knum_j, knum_k, &ii, &ij, &ik);
                        if (i == ii && j == ij && k == ik) {
                            k_op[i][j][k] = 1;
                        }

                        else {
                            k_op[i][j][k]    = 2;
                            k_op[ii][ij][ik] = 0;
                        }
                    }
                } /* k */
            } /* j */
        } /* i */

        /***********************************
           one-dimentionalize for MPI
        ************************************/

        T_knum = 0;
        for (i = 0; i < knum_i; i++) {
            for (j = 0; j < knum_j; j++) {
                for (k = 0; k < knum_k; k++) {
                    if (0 < k_op[i][j][k]) {
                        T_knum++;
                    }
                }
            }
        }

        /* set T_KGrids1,2,3 and T_k_op */

        T_knum = 0;
        for (i = 0; i < knum_i; i++) {

            if (knum_i == 1)
                k1 = 0.0;
            else
                k1 = -0.5 + (2.0 * (double)i + 1.0) / (2.0 * (double)knum_i) + Shift_K_Point;

            for (j = 0; j < knum_j; j++) {

                if (knum_j == 1)
                    k2 = 0.0;
                else
                    k2 = -0.5 + (2.0 * (double)j + 1.0) / (2.0 * (double)knum_j) - Shift_K_Point;

                for (k = 0; k < knum_k; k++) {

                    if (knum_k == 1)
                        k3 = 0.0;
                    else
                        k3 = -0.5 + (2.0 * (double)k + 1.0) / (2.0 * (double)knum_k) + 2.0 * Shift_K_Point;

                    if (0 < k_op[i][j][k]) {

                        T_KGrids1[T_knum] = k1;
                        T_KGrids2[T_knum] = k2;
                        T_KGrids3[T_knum] = k3;
                        T_k_op[T_knum]    = k_op[i][j][k];

                        T_knum++;
                    }
                }
            }
        }

        if (myid0 == Host_ID && 0 < level_stdout) {

            printf(" KGrids1: ");
            fflush(stdout);
            for (i = 0; i <= knum_i - 1; i++) {
                if (knum_i == 1)
                    k1 = 0.0;
                else
                    k1 = -0.5 + (2.0 * (double)i + 1.0) / (2.0 * (double)knum_i) + Shift_K_Point;
                printf("%9.5f ", k1);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);

            printf(" KGrids2: ");
            fflush(stdout);

            for (i = 0; i <= knum_j - 1; i++) {
                if (knum_j == 1)
                    k2 = 0.0;
                else
                    k2 = -0.5 + (2.0 * (double)i + 1.0) / (2.0 * (double)knum_j) - Shift_K_Point;
                printf("%9.5f ", k2);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);

            printf(" KGrids3: ");
            fflush(stdout);
            for (i = 0; i <= knum_k - 1; i++) {
                if (knum_k == 1)
                    k3 = 0.0;
                else
                    k3 = -0.5 + (2.0 * (double)i + 1.0) / (2.0 * (double)knum_k) + 2.0 * Shift_K_Point;
                printf("%9.5f ", k3);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);
        }

    }

    /***********************************************
                  Monkhorst-Pack k-points
    ***********************************************/

    else if (way_of_kpoint == 2) {

        T_knum = num_non_eq_kpt;

        for (k = 0; k < num_non_eq_kpt; k++) {
            T_KGrids1[k] = NE_KGrids1[k];
            T_KGrids2[k] = NE_KGrids2[k];
            T_KGrids3[k] = NE_KGrids3[k];
            T_k_op[k]    = NE_T_k_op[k];
        }
    }

    /***********************************************
              calculate the sum of weights
    ***********************************************/

    sum_weights = 0.0;
    for (k = 0; k < T_knum; k++) {
        sum_weights += (double)T_k_op[k];
    }

    /***********************************************
             allocate k-points into processors
    ***********************************************/

    if (numprocs1 < T_knum) {

        /* set parallel_mode */
        parallel_mode = 0;

        /* allocation of kloop to ID */

        for (ID = 0; ID < numprocs1; ID++) {
            tmp    = (double)T_knum / (double)numprocs1;
            S_knum = (int)((double)ID * (tmp + 1.0e-12));
            E_knum = (int)((double)(ID + 1) * (tmp + 1.0e-12)) - 1;
            if (ID == (numprocs1 - 1))
                E_knum = T_knum - 1;
            if (E_knum < 0)
                E_knum = 0;

            for (k = S_knum; k <= E_knum; k++) {
                /* ID in the first level world */
                T_k_ID[myworld1][k] = ID;
            }
        }

        /* find own informations */

        tmp    = (double)T_knum / (double)numprocs1;
        S_knum = (int)((double)myid1 * (tmp + 1.0e-12));
        E_knum = (int)((double)(myid1 + 1) * (tmp + 1.0e-12)) - 1;
        if (myid1 == (numprocs1 - 1))
            E_knum = T_knum - 1;
        if (E_knum < 0)
            E_knum = 0;

        num_kloop0 = E_knum - S_knum + 1;

        MPI_Comm_size(MPI_CommWD2[myworld2], &numprocs2);
        MPI_Comm_rank(MPI_CommWD2[myworld2], &myid2);
    }

    else {

        /* set parallel_mode */
        parallel_mode = 1;
        num_kloop0    = 1;

        Num_Comm_World2 = T_knum;
        MPI_Comm_size(MPI_CommWD2[myworld2], &numprocs2);
        MPI_Comm_rank(MPI_CommWD2[myworld2], &myid2);

        S_knum = myworld2;

        /* allocate k-points into processors */

        for (k = 0; k < T_knum; k++) {
            /* ID in the first level world */
            T_k_ID[myworld1][k] = Comm_World_StartID2[k];
        }
    }

    /****************************************************
     find all_knum
     if (all_knum==1), all the calculation will be made
     by the first diagonalization loop, and the second
     diagonalization will be skipped.
    ****************************************************/

    MPI_Allreduce(&num_kloop0, &all_knum, 1, MPI_INT, MPI_PROD, mpi_comm_level1);

    if (SpinP_switch == 1 && numprocs0 == 1 && all_knum == 1) {
        all_knum = 0;
    }

    // Set the device to be used by OpenACC
    if (scf_eigen_lib_flag == CuSOLVER) {
        // int rank;
        // if (all_knum != 1) {
        //     rank = myid0;
        // } else {
        //     rank = myid2;
        // }

        // OpenACC
        int local_numdevices = acc_get_num_devices(acc_device_nvidia);
        acc_set_device_num(myid0 % local_numdevices, acc_device_nvidia);
    }

    /***********************************************
                cublasmp & cusolverMp initialize
    ***********************************************/

    // Options opts = {
    //     .m = n,
    //     .n = n,
    //     .k = n,
    //     .mbA = nblk,
    //     .nbA = nblk,
    //     .mbB = nblk,
    //     .nbB = nblk,
    //     .mbC = nblk,
    //     .nbC = nblk,
    //     .ia = 1,
    //     .ja = 1,
    //     .ib = 1,
    //     .jb = 1,
    //     .ic = 1,
    //     .jc = 1,
    //     .grid_layout = 'R'
    // };

    // Options2 opts2;

    // Options3 opts3 = {
    //     .n = n,
    //     .mbA = nblk,
    //     .nbA = nblk,
    //     .mbQ = nblk,
    //     .nbQ = nblk,
    //     .ia = 1,
    //     .ja = 1,
    //     .iq = 1,
    //     .jq = 1,
    // };

    // Options4 opts4;
    // if (all_knum == 1) {
    //     init_cublasmp(myworld2, MPI_CommWD2, &opts, &opts2);
    //     init_cusolvermp(myworld2, MPI_CommWD2, &opts2, &opts3, &opts4);
    // }

    if (all_knum != 1 && scf_eigen_lib_flag == CuSOLVER) {
#pragma acc enter data create(Ss[0 : na_rows * na_cols], Hs[0 : na_rows * na_cols], Cs[0 : na_rows * na_cols])
#pragma acc enter data create(ko[0 : n + 1])
    }

    /****************************************************
      if (parallel_mode==1 && all_knum==1)
       make is1, ie1, is2, ie2
    ****************************************************/

    if (all_knum == 1) {

        /* allocation */

        is1 = (int *)malloc(sizeof(int) * numprocs2);
        ie1 = (int *)malloc(sizeof(int) * numprocs2);

        is2 = (int *)malloc(sizeof(int) * numprocs2);
        ie2 = (int *)malloc(sizeof(int) * numprocs2);

        Num_Snd_EV = (int *)malloc(sizeof(int) * numprocs2);
        Num_Rcv_EV = (int *)malloc(sizeof(int) * numprocs2);

        /* make is1 and ie1 */

        if (numprocs2 <= n) {

            av_num = (double)n / (double)numprocs2;

            for (ID = 0; ID < numprocs2; ID++) {
                is1[ID] = (int)(av_num * (double)ID) + 1;
                ie1[ID] = (int)(av_num * (double)(ID + 1));
            }

            is1[0]             = 1;
            ie1[numprocs2 - 1] = n;

        }

        else {

            for (ID = 0; ID < n; ID++) {
                is1[ID] = ID + 1;
                ie1[ID] = ID + 1;
            }
            for (ID = n; ID < numprocs2; ID++) {
                is1[ID] = 1;
                ie1[ID] = 0;
            }
        }

        /* make is2 and ie2 */

        if (numprocs2 <= MaxN) {

            av_num = (double)MaxN / (double)numprocs2;

            for (ID = 0; ID < numprocs2; ID++) {
                is2[ID] = (int)(av_num * (double)ID) + 1;
                ie2[ID] = (int)(av_num * (double)(ID + 1));
            }

            is2[0]             = 1;
            ie2[numprocs2 - 1] = MaxN;
        }

        else {
            for (ID = 0; ID < MaxN; ID++) {
                is2[ID] = ID + 1;
                ie2[ID] = ID + 1;
            }
            for (ID = MaxN; ID < numprocs2; ID++) {
                is2[ID] = 1;
                ie2[ID] = 0;
            }
        }

        /****************************************************************
           making data structure of MPI communicaition for eigenvectors
        ****************************************************************/

        for (ID = 0; ID < numprocs2; ID++) {
            Num_Snd_EV[ID] = 0;
            Num_Rcv_EV[ID] = 0;
        }

        for (i = 0; i < na_rows; i++) {

            ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + my_prow) % np_rows) * nblk + 1;

            po = 0;
            for (ID = 0; ID < numprocs2; ID++) {
                if (is2[ID] <= ig && ig <= ie2[ID]) {
                    po  = 1;
                    ID0 = ID;
                    break;
                }
            }

            if (po == 1)
                Num_Snd_EV[ID0] += na_cols;
        }

        for (ID = 0; ID < numprocs2; ID++) {
            IDS = (myid2 + ID) % numprocs2;
            IDR = (myid2 - ID + numprocs2) % numprocs2;
            if (ID != 0) {
                MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
                MPI_Recv(&Num_Rcv_EV[IDR], 1, MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
                MPI_Wait(&request, &stat);
            } else {
                Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
            }
        }

        Max_Num_Snd_EV = 0;
        Max_Num_Rcv_EV = 0;
        for (ID = 0; ID < numprocs2; ID++) {
            if (Max_Num_Snd_EV < Num_Snd_EV[ID])
                Max_Num_Snd_EV = Num_Snd_EV[ID];
            if (Max_Num_Rcv_EV < Num_Rcv_EV[ID])
                Max_Num_Rcv_EV = Num_Rcv_EV[ID];
        }

        Max_Num_Snd_EV++;
        Max_Num_Rcv_EV++;

        index_Snd_i = (int *)malloc(sizeof(int) * Max_Num_Snd_EV);
        index_Snd_j = (int *)malloc(sizeof(int) * Max_Num_Snd_EV);
        EVec_Snd    = (double *)malloc(sizeof(double) * Max_Num_Snd_EV * 2);
        index_Rcv_i = (int *)malloc(sizeof(int) * Max_Num_Rcv_EV);
        index_Rcv_j = (int *)malloc(sizeof(int) * Max_Num_Rcv_EV);
        EVec_Rcv    = (double *)malloc(sizeof(double) * Max_Num_Rcv_EV * 2);

    } /* if (all_knum==1) */

    /****************************************************
       PrintMemory
    ****************************************************/

    if (firsttime && memoryusage_fileout) {
        PrintMemory("Band_DFT_Col: My_NZeros", sizeof(int) * numprocs0, NULL);
        PrintMemory("Band_DFT_Col: SP_NZeros", sizeof(int) * numprocs0, NULL);
        PrintMemory("Band_DFT_Col: SP_Atoms", sizeof(int) * numprocs0, NULL);
        if (all_knum == 1) {
            PrintMemory("Band_DFT_Col: is1", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: ie1", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: is2", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: ie2", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: Num_Snd_EV", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: Num_Rcv_EV", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: index_Snd_i", sizeof(int) * Max_Num_Snd_EV, NULL);
            PrintMemory("Band_DFT_Col: index_Snd_j", sizeof(int) * Max_Num_Snd_EV, NULL);
            PrintMemory("Band_DFT_Col: EVec_Snd", sizeof(double) * Max_Num_Snd_EV * 2, NULL);
            PrintMemory("Band_DFT_Col: index_Rcv_i", sizeof(int) * Max_Num_Rcv_EV, NULL);
            PrintMemory("Band_DFT_Col: index_Rcv_j", sizeof(int) * Max_Num_Rcv_EV, NULL);
            PrintMemory("Band_DFT_Col: EVec_Rcv", sizeof(double) * Max_Num_Rcv_EV * 2, NULL);
        }
    }

    /****************************************************
       communicate T_k_ID
    ****************************************************/

    if (numprocs0 == 1 && SpinP_switch == 1) {
        for (k = 0; k < T_knum; k++) {
            T_k_ID[1][k] = T_k_ID[0][k];
        }
    } else {
        for (spin = 0; spin <= SpinP_switch; spin++) {
            ID = Comm_World_StartID1[spin];
            MPI_Bcast(&T_k_ID[spin][0], T_knum, MPI_INT, ID, mpi_comm_level1);
        }
    }

    /****************************************************
       store in each processor all the matrix elements
          for overlap and Hamiltonian matrices
    ****************************************************/

    if (measure_time)
        dtime(&Stime);

    if (measure_time)
        dtime(&starttime);

    /* spin=myworld1 */

    spin = myworld1;

    /* set S1 */

    if (SCF_iter == 1 || all_knum != 1) {
        size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    }

diagonalize1:

    /* set H1 */

    if (SpinP_switch == 0) {
        size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    } else if (1 < numprocs0) {

        size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
        size_H1 = Get_OneD_HS_Col(1, nh[1], CDM1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

        if (myworld1) {
            for (i = 0; i < size_H1; i++) {
                H1[i] = CDM1[i];
            }
        }
    } else {
        size_H1 = Get_OneD_HS_Col(1, nh[spin], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    }

    if (measure_time) {
        dtime(&Etime);
        time1 += Etime - Stime;
    }

    /****************************************************
                         start kloop
    ****************************************************/

    dtime(&SiloopTime);

    if (all_knum != 1 && scf_eigen_lib_flag == CuSOLVER) {
        for (int kloop0 = 0; kloop0 < num_kloop0; kloop0++) {
            kloop = S_knum + kloop0;

            k1 = T_KGrids1[kloop];
            k2 = T_KGrids2[kloop];
            k3 = T_KGrids3[kloop];

            /* make S and H */

            // #pragma acc kernels
            // #pragma acc loop independent
            Construct_Band_CsHs(SCF_iter, all_knum, order_GA, MP, S1, H1, k1, k2, k3, Ss, Hs, n, myid2);

#pragma acc update device(Ss[0 : n * n], Hs[0 : n * n])

            /* for blas */

            /* diagonalize S */

            if (measure_time)
                dtime(&Stime);

            if (SCF_iter == 1) {
                /* diagonalize S */

                EigenBand_lapack_openacc(Ss, ko, n, n);
            }

            if (measure_time) {
                dtime(&Etime);
                time2 += Etime - Stime;
            }

            if (SCF_iter == 1) {

                /* minus eigenvalues to 1.0e-14 */

#pragma acc kernels
#pragma acc loop independent
                for (int l = 1; l <= n; l++) {
                    if (ko[l] < 0.0)
                        ko[l] = 1.0e-10;
                }

#pragma acc update self(ko[0 : n + 1])
                for (int l = 1; l <= n; l++) {
                    koS[l] = ko[l];
                }

                /* calculate S*1/sqrt(ko) */

#pragma acc kernels
#pragma acc loop independent
                for (int l = 1; l <= n; l++)
                    ko[l] = 1.0 / sqrt(ko[l]);

                {

#pragma acc kernels
#pragma acc loop independent collapse(2)
                    for (int i1 = 1; i1 <= n; i1++) {
                        for (int j1 = 1; j1 <= n; j1++) {
                            Ss[(j1 - 1) * n + i1 - 1].r *= ko[j1];
                            Ss[(j1 - 1) * n + i1 - 1].i *= ko[j1];
                        }
                    }

                } /* #pragma omp parallel */

                /* S * 1.0/sqrt(ko[l])  */
#pragma acc update           self(Ss[0 : n * n])
            }

            /****************************************************
             1.0/sqrt(ko[l]) * U^t * H * U * 1.0/sqrt(ko[l])
            ****************************************************/

            if (measure_time)
                dtime(&Stime);

#pragma acc kernels
#pragma acc loop independent
            for (int i = 0; i < n * n; i++) {
                Cs[i].r = 0.0;
                Cs[i].i = 0.0;
            }

            my_cublasZgemm_openacc(CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, Hs, Ss, Cs);

            /* pzgemm */

            /* H * U * 1/sqrt(ko) */

            /* 1/sqrt(ko) * U^+ H * U * 1/sqrt(ko) */

#pragma acc kernels
#pragma acc loop independent
            for (int i = 0; i < n * n; i++) {
                Hs[i].r = 0.0;
                Hs[i].i = 0.0;
            }

            my_cublasZgemm_openacc(CUBLAS_OP_C, CUBLAS_OP_N, n, n, n, Ss, Cs, Hs);

#pragma acc kernels
#pragma acc loop independent
            for (int i = 0; i < n * n; i++) {
                Cs[i].r = Hs[i].r;
                Cs[i].i = Hs[i].i;
            }

            if (measure_time) {
                dtime(&Etime);
                time3 += Etime - Stime;
            }

            /* diagonalize H' */

            if (measure_time)
                dtime(&Stime);

            EigenBand_lapack_openacc(Cs, ko, n, MaxN);

            if (measure_time) {
                dtime(&Etime);
                time4 += Etime - Stime;
            }

#pragma acc update     self(ko[0 : n + 1])
            for (int l = 1; l <= MaxN; l++) {
                EIGEN[spin][kloop][l] = ko[l];
            }

            if (3 <= level_stdout && 0 <= kloop) {
                printf(" myid0=%2d spin=%2d kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n", myid0, spin, kloop,
                           T_KGrids1[kloop], T_KGrids2[kloop], T_KGrids3[kloop]);
                for (i1 = 1; i1 <= n; i1++) {
                    if (SpinP_switch == 0)
                        printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n", i1, EIGEN[0][kloop][i1],
                                   EIGEN[0][kloop][i1]);
                    else
                        printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n", i1, EIGEN[0][kloop][i1],
                                   EIGEN[1][kloop][i1]);
                }
            }

            if (measure_time)
                dtime(&Stime);

            if (measure_time) {
                dtime(&Etime);
                time5 += Etime - Stime;
            }
            } /* kloop0 */

        } else {
        for (kloop0 = 0; kloop0 < num_kloop0; kloop0++) {
            if (scf_eigen_lib_flag == CuSOLVER) {
                kloop = S_knum + kloop0;

                k1 = T_KGrids1[kloop];
                k2 = T_KGrids2[kloop];
                k3 = T_KGrids3[kloop];

                /* make S and H */
                double starttimesh = 0.0, endtimesh = 0.0;
                if (measure_time) {
                    dtime(&starttimesh);
                }

                Construct_Band_CsHs(SCF_iter, all_knum, order_GA, MP, S1, H1, k1, k2, k3, Ss, Hs, n, myid2);
                if (measure_time) {
                    dtime(&endtimesh);

                    if (myid0 == 0) {
                        printf("make S and H: %.7f (sec)\n", endtimesh - starttimesh);
                    }
                }

                if (measure_time) {
                    dtime(&endtime);
                    part1 += endtime - starttime;
                    if (SCF_iter != 1) {
                        part1sum += part1;
                    }
                }

                if (measure_time) {
                    MPI_Barrier(mpi_comm_level1);
                }

                if (myid2 == 0) {
                    if (measure_time)
                        dtime(&Stime);

                    if (SCF_iter == 1) {
//#pragma acc enter data create(Hs[0 : n * n], Ss[0 : n * n], Cs[0 : n * n], ko[0 : n + 1])
#pragma acc enter data create(Ss[0 : n * n], ko[0 : n + 1])
                    } else {
//#pragma acc enter data create(Hs[0 : n * n], Cs[0 : n * n], ko[0 : n + 1])
#pragma acc enter data create(ko[0 : n + 1])
#pragma acc enter data copyin(Ss[0 : n * n])
                    }

                    if (SCF_iter == 1) {
//#pragma acc update device(Hs[0 : n * n], Ss[0 : n * n])
#pragma acc update device(Ss[0 : n * n])
                    }
                    /*else {
                #pragma acc update device(Hs[0 : n * n])
                                    }*/

                    if (SCF_iter == 1) {
                        /* diagonalize S */

                        EigenBand_lapack_openacc(Ss, ko, n, n);

                        if (measure_time) {
                            dtime(&Etime);
                            time2 += Etime - Stime;
                            part2_1 += Etime - Stime;
                            if (SCF_iter != 1) {
                                part2_1sum += part2_1;
                            }
                        }

                        if (measure_time)
                            dtime(&Stime);

                        /* minus eigenvalues to 1.0e-14 */

#pragma acc kernels
#pragma acc loop independent
                        for (int l = 1; l <= n; l++) {
                            if (ko[l] < 0.0)
                                ko[l] = 1.0e-10;
                        }

#pragma acc update self(ko[0 : n + 1])
                        for (int l = 1; l <= n; l++) {
                            koS[l] = ko[l];
                        }

                        /* calculate S*1/sqrt(ko) */

#pragma acc kernels
#pragma acc loop independent
                        for (int l = 1; l <= n; l++)
                            ko[l] = 1.0 / sqrt(ko[l]);

                        {

#pragma acc kernels
#pragma acc loop independent collapse(2)
                            for (int i1 = 1; i1 <= n; i1++) {
                                for (int j1 = 1; j1 <= n; j1++) {
                                    Ss[(j1 - 1) * n + i1 - 1].r *= ko[j1];
                                    Ss[(j1 - 1) * n + i1 - 1].i *= ko[j1];
                                }
                            }

                        } /* #pragma omp parallel */

                        /* S * 1.0/sqrt(ko[l])  */
#pragma acc update           self(Ss[0 : n * n])

                        if (measure_time) {
                            dtime(&Etime);
                            part2_2 += Etime - Stime;
                            if (SCF_iter != 1) {
                                part2_2sum += part2_2;
                            }
                        }

                        if (measure_time)
                            dtime(&Stime);
                    }

#pragma acc enter data copyin(Hs[0 : n * n])
#pragma acc enter data create(Cs[0 : n * n])

                    /****************************************************
                                        1/sqrt(ko) * U^t * H * U * 1/sqrt(ko)
                                    ****************************************************/

#pragma acc kernels
#pragma acc loop independent
                    for (int i = 0; i < n * n; i++) {
                        Cs[i].r = 0.0;
                        Cs[i].i = 0.0;
                    }

                    my_cublasZgemm_openacc(CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, Hs, Ss, Cs);

                    /* pzgemm */

                    /* H * U * 1/sqrt(ko) */

                    /* 1/sqrt(ko) * U^+ H * U * 1/sqrt(ko) */

#pragma acc kernels
#pragma acc loop independent
                    for (int i = 0; i < n * n; i++) {
                        Hs[i].r = 0.0;
                        Hs[i].i = 0.0;
                    }

                    my_cublasZgemm_openacc(CUBLAS_OP_C, CUBLAS_OP_N, n, n, n, Ss, Cs, Hs);

#pragma acc kernels
#pragma acc loop independent
                    for (int i = 0; i < n * n; i++) {
                        Cs[i].r = Hs[i].r;
                        Cs[i].i = Hs[i].i;
                    }

                    if (measure_time) {
                        dtime(&Etime);
                        time3 += Etime - Stime;

                        part2_3 += Etime - Stime;
                        if (SCF_iter != 1) {
                            part2_3sum += part2_3;
                        }
                    }

                    /* diagonalize H' */

                    if (measure_time)
                        dtime(&Stime);

#pragma acc update    self(Ss[0 : n * n])
#pragma acc exit data delete(Hs[0 : n * n], Ss[0 : n * n])

                    EigenBand_lapack_openacc(Cs, ko, n, MaxN);

#pragma acc update self(ko[0 : n + 1])
                    for (l = 1; l <= MaxN; l++) {
                        EIGEN[spin][kloop][l] = ko[l];
                    }

                    if (measure_time) {
                        dtime(&Etime);
                        time4 += Etime - Stime;
                        part2_4 += Etime - Stime;
                        if (SCF_iter != 1) {
                            part2_4sum += part2_4;
                        }
                    }

                    if (measure_time)
                        dtime(&Stime);

                    if (measure_time)
                        dtime(&starttime);

#pragma acc enter data create(Hs[0 : n * n])
#pragma acc enter data copyin(Ss[0 : n * n])

                    /****************************************************
                                                                      transformation to the original eigenvectors.
                                                                       NOTE JRCAT-244p and JAIST-2122p
                                                                    ****************************************************/
#pragma acc kernels
#pragma acc loop independent
                    for (int i = 0; i < n * n; i++) {
                        Hs[i] = Cs[i];
                    }

                    /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */

                    my_cublasZgemm_openacc(CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, Hs, Ss, Cs);

#pragma acc update    self(Cs[0 : n * n])
#pragma acc exit data delete(Hs[0 : n * n], Ss[0 : n * n], Cs[0 : n * n], ko[0 : n + 1])

                    if (measure_time) {
                        dtime(&endtime);

                        partmul = endtime - starttime;
                        if (SCF_iter != 1) {
                            partmulsum += partmul;
                        }
                    }
                }

                int desc_global[9];
                int i_zero = 0;
                int i_one  = 1;

                descinit_(desc_global, &n, &n, &n, &n, &i_zero, &i_zero, &ictxt2, &n, &info);
                pzgemr2d_(&n, &n, Cs, &i_one, &i_one, desc_global, Hs, &i_one, &i_one, descH, &ictxt2);
            } else {
                kloop = S_knum + kloop0;

                k1 = T_KGrids1[kloop];
                k2 = T_KGrids2[kloop];
                k3 = T_KGrids3[kloop];

                /* make S and H */

                Construct_Band_CsHs(SCF_iter, all_knum, order_GA, MP, S1, H1, k1, k2, k3, Cs, Hs, n, myid2);

                if (measure_time) {
                    dtime(&endtime);
                    part1 += endtime - starttime;
                    if (SCF_iter != 1) {
                        part1sum += part1;
                    }
                }

                // #pragma acc update device(Hs[0 : na_rows * na_cols], Cs[0 : na_rows * na_cols])

                /* diagonalize S */

                if (measure_time)
                    dtime(&Stime);

                if (parallel_mode == 0 || (SCF_iter == 1 || all_knum != 1)) {

                    MPI_Comm_split(MPI_CommWD2[myworld2], my_pcol, my_prow, &mpi_comm_rows);
                    MPI_Comm_split(MPI_CommWD2[myworld2], my_prow, my_pcol, &mpi_comm_cols);

                    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
                    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

                    if (scf_eigen_lib_flag == 1) {
                        F77_NAME(solve_evp_complex, SOLVE_EVP_COMPLEX)
                        (&n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
                    } else if (scf_eigen_lib_flag == 2) {

#ifndef kcomp
                        int mpiworld;
                        mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
                        F77_NAME(elpa_solve_evp_complex_2stage_double_impl, ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
                        (&n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int,
                         &mpi_comm_cols_int, &mpiworld);
#endif
                    }

                    MPI_Comm_free(&mpi_comm_rows);
                    MPI_Comm_free(&mpi_comm_cols);
                }

                if (measure_time) {
                    dtime(&Etime);
                    time2 += Etime - Stime;
                    part2_1 += Etime - Stime;
                    if (SCF_iter != 1) {
                        part2_1sum += part2_1;
                    }
                }

                if (measure_time) {
                    dtime(&Stime);
                }

                if (SCF_iter == 1 || all_knum != 1) {

                    if (3 <= level_stdout) {
                        printf(" myid0=%2d spin=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n", myid0, spin, kloop,
                               T_KGrids1[kloop], T_KGrids2[kloop], T_KGrids3[kloop]);
                        for (i1 = 1; i1 <= n; i1++) {
                            printf("  Eigenvalues of OLP  %2d  %15.12f\n", i1, ko[i1]);
                        }
                    }

                    /* minus eigenvalues to 1.0e-10 */

                    // #pragma acc kernels
                    // #pragma acc loop independent
                    for (l = 1; l <= n; l++) {
                        if (ko[l] < 0.0) {
                            ko[l] = 1.0e-10;
                        }
                        koS[l] = ko[l];
                    }

                    /* calculate S*1/sqrt(ko) */

                    // #pragma acc kernels
                    // #pragma acc loop independent
                    for (l = 1; l <= n; l++) {
                        ko[l] = 1.0 / sqrt(ko[l]);
                    }

                    /* S * 1.0/sqrt(ko[l]) */

                    // #pragma acc kernels
                    // #pragma acc loop independent
                    for (i = 0; i < na_rows; i++) {
                        // #pragma acc loop independent
                        for (j = 0; j < na_cols; j++) {
                            int jg =
                                np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + my_pcol) % np_cols) * nblk + 1;
                            Ss[j * na_rows + i].r = Ss[j * na_rows + i].r * ko[jg];
                            Ss[j * na_rows + i].i = Ss[j * na_rows + i].i * ko[jg];
                        }
                    }
                }

                // printf("Ss[1][1] = %.7f Ss[1][2] = %.7f Ss[1][3] = %.7f\n", Ss[0].r, Ss[n].r, Ss[2 * n].r);
                // sleep(100);
                // exit(-1);

                /****************************************************
             1.0/sqrt(ko[l]) * U^t * H * U * 1.0/sqrt(ko[l])
            ****************************************************/

                if (measure_time) {
                    dtime(&Etime);
                    part2_2 += Etime - Stime;
                    if (SCF_iter != 1) {
                        part2_2sum += part2_2;
                    }
                }

                if (measure_time)
                    dtime(&Stime);

                /* pzgemm */

                /* H * U * 1.0/sqrt(ko[l]) */

                // #pragma acc kernels
                // #pragma acc loop independent
                for (i = 0; i < na_rows * na_cols; i++) {
                    Cs[i].r = 0.0;
                    Cs[i].i = 0.0;
                }

                // cublasmp_zgemm(CUBLAS_OP_N, CUBLAS_OP_N, Hs, Ss, Cs, &opts, &opts2);

                Cblacs_barrier(ictxt2, "A");
                F77_NAME(pzgemm, PZGEMM)
                ("N", "N", &n, &n, &n, &alpha, Hs, &ONE, &ONE, descH, Ss, &ONE, &ONE, descS, &beta, Cs, &ONE, &ONE,
                 descC);

                // printf("BLAS_C[1][1] = %.7f BLAS_C[1][2] = %.7f BLAS_C[1][3] = %.7f\n", Cs[0].r, Cs[n].r, Cs[2 * n].r);

                /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

                // #pragma acc kernels
                // #pragma acc loop independent
                for (i = 0; i < na_rows * na_cols; i++) {
                    Hs[i].r = 0.0;
                    Hs[i].i = 0.0;
                }

                // cublasmp_zgemm(CUBLAS_OP_C, CUBLAS_OP_N, Ss, Cs, Hs, &opts, &opts2);

                Cblacs_barrier(ictxt2, "C");
                F77_NAME(pzgemm, PZGEMM)
                ("C", "N", &n, &n, &n, &alpha, Ss, &ONE, &ONE, descS, Cs, &ONE, &ONE, descC, &beta, Hs, &ONE, &ONE,
                 descH);

                if (measure_time) {
                    dtime(&Etime);
                    time3 += Etime - Stime;
                    part2_3 += Etime - Stime;
                    if (SCF_iter != 1) {
                        part2_3sum += part2_3;
                    }
                }

                /* diagonalize H' */

                if (measure_time)
                    dtime(&Stime);

                MPI_Comm_split(MPI_CommWD2[myworld2], my_pcol, my_prow, &mpi_comm_rows);
                MPI_Comm_split(MPI_CommWD2[myworld2], my_prow, my_pcol, &mpi_comm_cols);

                mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
                mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

                if (scf_eigen_lib_flag == 1) {
                    F77_NAME(solve_evp_complex, SOLVE_EVP_COMPLEX)
                    (&n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
                } else if (scf_eigen_lib_flag == 2) {

#ifndef kcomp
                    int mpiworld;
                    mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
                    F77_NAME(elpa_solve_evp_complex_2stage_double_impl, ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
                    (&n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int,
                     &mpi_comm_cols_int, &mpiworld);
#endif
                }

                MPI_Comm_free(&mpi_comm_rows);
                MPI_Comm_free(&mpi_comm_cols);

                if (measure_time) {
                    dtime(&Etime);
                    time4 += Etime - Stime;
                    part2_4 += Etime - Stime;
                    if (SCF_iter != 1) {
                        part2_4sum += part2_4;
                    }
                }

                //             if (numprocs0 < n / 2) {
                // //#pragma acc update self(ko[0 : n + 1])
                //             }

                for (l = 1; l <= MaxN; l++) {
                    EIGEN[spin][kloop][l] = ko[l];
                }

                if (3 <= level_stdout && 0 <= kloop) {
                    printf(" myid0=%2d spin=%2d kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n", myid0, spin, kloop,
                           T_KGrids1[kloop], T_KGrids2[kloop], T_KGrids3[kloop]);
                    for (i1 = 1; i1 <= n; i1++) {
                        if (SpinP_switch == 0)
                            printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n", i1, EIGEN[0][kloop][i1],
                                   EIGEN[0][kloop][i1]);
                        else
                            printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n", i1, EIGEN[0][kloop][i1],
                                   EIGEN[1][kloop][i1]);
                    }
                }

                /**************************************************
              if (all_knum==1), wave functions are calculated.
            **************************************************/

                if (measure_time)
                    dtime(&Stime);

                if (all_knum == 1) {
                    // #pragma acc kernels
                    // #pragma acc loop independent
                    for (i = 0; i < na_rows * na_cols; i++) {
                        Hs[i].r = 0.0;
                        Hs[i].i = 0.0;
                    }

                    // cublasmp_zgemm(CUBLAS_OP_T, CUBLAS_OP_T, Cs, Ss, Hs, &opts, &opts2);
                    // #pragma acc update self(Ss[0 : na_rows * na_cols], Hs[0 : na_rows * na_cols], Cs[0 : na_rows * na_cols])

                    F77_NAME(pzgemm, PZGEMM)
                    ("T", "T", &n, &n, &n, &alpha, Cs, &ONE, &ONE, descS, Ss, &ONE, &ONE, descC, &beta, Hs, &ONE, &ONE,
                     descH);
                    Cblacs_barrier(ictxt2, "A");
                }
            }

            if (all_knum == 1) {
                /* MPI communications of Hs and store them to EVec1 */

                for (ID = 0; ID < numprocs2; ID++) {

                    IDS = (myid2 + ID) % numprocs2;
                    IDR = (myid2 - ID + numprocs2) % numprocs2;

                    k = 0;
                    for (i = 0; i < na_rows; i++) {

                        ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + my_prow) % np_rows) * nblk + 1;
                        if (is2[IDS] <= ig && ig <= ie2[IDS]) {

                            for (j = 0; j < na_cols; j++) {
                                jg = np_cols * nblk * ((j) / nblk) + (j) % nblk +
                                     ((np_cols + my_pcol) % np_cols) * nblk + 1;

                                index_Snd_i[k]      = ig;
                                index_Snd_j[k]      = jg;
                                EVec_Snd[2 * k]     = Hs[j * na_rows + i].r;
                                EVec_Snd[2 * k + 1] = Hs[j * na_rows + i].i;

                                k++;
                            }
                        }
                    }

                    if (ID != 0) {

                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
                        }
                        if (Num_Rcv_EV[IDR] != 0) {
                            MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
                        }
                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Wait(&request, &stat);
                        }

                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
                        }
                        if (Num_Rcv_EV[IDR] != 0) {
                            MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
                        }
                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Wait(&request, &stat);
                        }

                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Isend(EVec_Snd, Num_Snd_EV[IDS] * 2, MPI_DOUBLE, IDS, 999, MPI_CommWD2[myworld2],
                                      &request);
                        }
                        if (Num_Rcv_EV[IDR] != 0) {
                            MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR] * 2, MPI_DOUBLE, IDR, 999, MPI_CommWD2[myworld2], &stat);
                        }
                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Wait(&request, &stat);
                        }

                    } else {
                        for (k = 0; k < Num_Snd_EV[IDS]; k++) {
                            index_Rcv_i[k] = index_Snd_i[k];
                            index_Rcv_j[k] = index_Snd_j[k];

                            EVec_Rcv[2 * k]     = EVec_Snd[2 * k];
                            EVec_Rcv[2 * k + 1] = EVec_Snd[2 * k + 1];
                        }
                    }

                    for (k = 0; k < Num_Rcv_EV[IDR]; k++) {

                        ig = index_Rcv_i[k];
                        jg = index_Rcv_j[k];
                        m  = (jg - 1) * (ie2[myid2] - is2[myid2] + 1) + ig - is2[myid2];

                        EVec1[spin][m].r = EVec_Rcv[2 * k];
                        EVec1[spin][m].i = EVec_Rcv[2 * k + 1];
                    }

                } /* ID */
            }

            if (measure_time) {
                dtime(&Etime);
                time5 += Etime - Stime;
                if (scf_eigen_lib_flag == ELPA2 || scf_eigen_lib_flag == CuSOLVER && myid2 == 0) {
                    part2_5 += Etime - Stime;
                    if (SCF_iter != 1) {
                        part2_5sum += part2_5;
                    }
                }
            }

            // if (measure_time == 1) {
            //     MPI_Allreduce(MPI_IN_PLACE, &time2, 1, MPI_DOUBLE, MPI_MAX, MPI_CommWD2[myworld2]);
            //     MPI_Allreduce(MPI_IN_PLACE, &time3, 1, MPI_DOUBLE, MPI_MAX, MPI_CommWD2[myworld2]);
            //     MPI_Allreduce(MPI_IN_PLACE, &time4, 1, MPI_DOUBLE, MPI_MAX, MPI_CommWD2[myworld2]);
            //     MPI_Allreduce(MPI_IN_PLACE, &time5, 1, MPI_DOUBLE, MPI_MAX, MPI_CommWD2[myworld2]);
            // }

        } /* kloop0 */
    }

    if (measure_time)
        dtime(&EiloopTime);

    if (SpinP_switch == 1 && numprocs0 == 1 && spin == 0) {
        spin++;
        goto diagonalize1;
    }

    /****************************************************
       MPI:

       EIGEN
    ****************************************************/

    if (measure_time) {
        MPI_Barrier(mpi_comm_level1);
        dtime(&Stime);
    }

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (kloop = 0; kloop < T_knum; kloop++) {

            /* get ID in the zeroth world */
            ID = Comm_World_StartID1[spin] + T_k_ID[spin][kloop];
            MPI_Bcast(&EIGEN[spin][kloop][0], MaxN + 1, MPI_DOUBLE, ID, mpi_comm_level1);
        }
    }

    if (measure_time) {
        dtime(&Etime);
        time6 += Etime - Stime;
    }

    if (measure_time) {
        dtime(&Stime);
    }

    /**************************************
           find chemical potential
    **************************************/

    /* for XANES */
    if (xanes_calc == 1) {

        for (spin = 0; spin <= SpinP_switch; spin++) {

            po        = 0;
            loop_num  = 0;
            ChemP_MAX = 20.0;
            ChemP_MIN = -20.0;

            do {

                loop_num++;

                ChemP     = 0.50 * (ChemP_MAX + ChemP_MIN);
                Num_State = 0.0;

                for (kloop = 0; kloop < T_knum; kloop++) {

                    for (l = 1; l <= MaxN; l++) {

                        x = (EIGEN[spin][kloop][l] - ChemP) * Beta;

                        if (x <= -x_cut)
                            x = -x_cut;
                        if (x_cut <= x)
                            x = x_cut;
                        FermiF = FermiFunc(x, spin, l, &l, &x);
                        Num_State += FermiF * (double)T_k_op[kloop];
                    }
                }

                if (SpinP_switch == 0)
                    Num_State = 2.0 * Num_State / sum_weights;
                else
                    Num_State = Num_State / sum_weights;
                Dnum = HOMO_XANES[spin] - Num_State;

                if (0.0 <= Dnum)
                    ChemP_MIN = ChemP;
                else
                    ChemP_MAX = ChemP;
                if (fabs(Dnum) < 1.0e-12)
                    po = 1;

            } while (po == 0 && loop_num < 1000);

            ChemP_XANES[spin]  = ChemP;
            Cluster_HOMO[spin] = HOMO_XANES[spin];

        } /* spin loop */

        /* set ChemP */

        ChemP = 0.5 * (ChemP_XANES[0] + ChemP_XANES[1]);

    } /* end of if (xanes_calc==1) */

    /* start of else for if (xanes_calc==1) */

    else {

        if (measure_time)
            dtime(&Stime);

        double Beta_trial1;

        /* first, find ChemP at 1200 K */

        Beta_trial1 = 1.0 / kB / (1200.0 / eV2Hartree);

        po        = 0;
        loop_num  = 0;
        ChemP_MAX = 20.0;
        ChemP_MIN = -20.0;

        do {

            loop_num++;

            ChemP     = 0.50 * (ChemP_MAX + ChemP_MIN);
            Num_State = 0.0;

            for (kloop = 0; kloop < T_knum; kloop++) {
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (l = 1; l <= MaxN; l++) {

                        x = (EIGEN[spin][kloop][l] - ChemP) * Beta_trial1;

                        if (x <= -x_cut)
                            x = -x_cut;
                        if (x_cut <= x)
                            x = x_cut;
                        FermiF = FermiFunc(x, spin, l, &l, &x);

                        Num_State += FermiF * (double)T_k_op[kloop];
                    }
                }
            }

            if (SpinP_switch == 0)
                Num_State = 2.0 * Num_State / sum_weights;
            else
                Num_State = Num_State / sum_weights;

            Dnum = TZ - Num_State - system_charge;

            if (0.0 <= Dnum)
                ChemP_MIN = ChemP;
            else
                ChemP_MAX = ChemP;
            if (fabs(Dnum) < 1.0e-12)
                po = 1;

        } while (po == 0 && loop_num < 1000);

        /* second, find ChemP at the temperatue, starting from the previously found ChemP. */

        po        = 0;
        loop_num  = 0;
        ChemP_MAX = 20.0;
        ChemP_MIN = -20.0;

        do {

            loop_num++;

            if (loop_num != 1) {
                ChemP = 0.50 * (ChemP_MAX + ChemP_MIN);
            }

            Num_State = 0.0;

            for (kloop = 0; kloop < T_knum; kloop++) {
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (l = 1; l <= MaxN; l++) {

                        x = (EIGEN[spin][kloop][l] - ChemP) * Beta;

                        if (x <= -x_cut)
                            x = -x_cut;
                        if (x_cut <= x)
                            x = x_cut;
                        FermiF = FermiFunc(x, spin, l, &l, &x);

                        Num_State += FermiF * (double)T_k_op[kloop];
                    }
                }
            }

            if (SpinP_switch == 0)
                Num_State = 2.0 * Num_State / sum_weights;
            else
                Num_State = Num_State / sum_weights;

            Dnum = TZ - Num_State - system_charge;

            if (0.0 <= Dnum)
                ChemP_MIN = ChemP;
            else
                ChemP_MAX = ChemP;
            if (fabs(Dnum) < 1.0e-12)
                po = 1;
        } while (po == 0 && loop_num < 1000);

    } /* end of else for if (xanes_calc==1) */

    /* for the NEGF calculation */
    if (Solver == 4 && TRAN_ChemP_Band == 0)
        ChemP = 0.5 * (ChemP_e[0] + ChemP_e[1]);

    /****************************************************
             band energy in a finite temperature
    ****************************************************/

    Eele0[0] = 0.0;
    Eele0[1] = 0.0;

    for (kloop = 0; kloop < T_knum; kloop++) {
        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (l = 1; l <= MaxN; l++) {

                if (xanes_calc == 1)
                    x = (EIGEN[spin][kloop][l] - ChemP_XANES[spin]) * Beta;
                else
                    x = (EIGEN[spin][kloop][l] - ChemP) * Beta;

                if (x <= -x_cut)
                    x = -x_cut;
                if (x_cut <= x)
                    x = x_cut;
                FermiF = FermiFunc(x, spin, l, &l, &x);

                Eele0[spin] += FermiF * EIGEN[spin][kloop][l] * (double)T_k_op[kloop];
            }
        }
    }

    if (SpinP_switch == 0) {
        Eele0[0] = Eele0[0] / sum_weights;
        Eele0[1] = Eele0[0];
    } else {
        Eele0[0] = Eele0[0] / sum_weights;
        Eele0[1] = Eele0[1] / sum_weights;
    }

    Uele = Eele0[0] + Eele0[1];

    if (2 <= level_stdout) {
        printf("myid0=%2d ChemP=%lf, Eele0[0]=%lf, Eele0[1]=%lf\n", myid0, ChemP, Eele0[0], Eele0[1]);
    }

    if (measure_time) {
        dtime(&Etime);
        time7 += Etime - Stime;
        part3 += Etime - Stime;
        if (SCF_iter != 1) {
            part3sum += part3;
        }
    }

    /****************************************************
           if all_knum==1, calculate CDM and EDM
    ****************************************************/

    if (measure_time)
        dtime(&Stime);

    if (all_knum == 1) {

        if (measure_time)
            dtime(&Stime0);

        /* initialize CDM1 and EDM1 */

        for (i = 0; i < size_H1; i++) {
            CDM1[i] = 0.0;
            EDM1[i] = 0.0;
        }

        /* calculate CDM and EDM */

        spin  = myworld1;
        kloop = S_knum;

        k1 = T_KGrids1[kloop];
        k2 = T_KGrids2[kloop];
        k3 = T_KGrids3[kloop];

        /* weight of k-point */

        kw = (double)T_k_op[kloop];

        /* pre-calculation of the Fermi function */

        po   = 0;
        kmin = is2[myid2];
        kmax = ie2[myid2];

        for (k = is2[myid2]; k <= ie2[myid2]; k++) {

            eig = EIGEN[spin][kloop][k];

            if (xanes_calc == 1)
                x = (eig - ChemP_XANES[spin]) * Beta;
            else
                x = (eig - ChemP) * Beta;

            if (x <= -x_cut)
                x = -x_cut;
            if (x_cut <= x)
                x = x_cut;
            FermiF = FermiFunc(x, spin, k, &k, &x);

            tmp1 = sqrt(kw * FermiF);

            for (i1 = 1; i1 <= n; i1++) {
                i = (i1 - 1) * (ie2[myid2] - is2[myid2] + 1) + k - is2[myid2];

                EVec1[spin][i].r *= tmp1;
                EVec1[spin][i].i *= tmp1;
            }

            /* find kmax */

            if (FermiF < FermiEps && po == 0) {
                kmax = k;
                po   = 1;
            }
        }

        if (measure_time) {
            dtime(&Etime0);
            time81 += Etime0 - Stime0;
            dtime(&Stime0);
        }

        /* ---------- 必要サイズを決定 ---------------------------------- */
        const int nk      = kmax - kmin + 1; /* k 点数（必ず ≥1） */
        int       max_tno = 0;               /* 系内最大局在軌道数 */
        for (int GA = 1; GA <= atomnum; ++GA) {
            const int wan = WhatSpecies[GA];
            const int tno = Spe_Total_CNO[wan];
            if (tno > max_tno)
                max_tno = tno;
        }

        /* ---------- ヒープ確保（一回だけ）----------------------------- */
        const size_t vec_bytes = (size_t)max_tno * nk * sizeof(double);

        double * buf_Re0  = (double *)malloc(vec_bytes);
        double * buf_Im0  = (double *)malloc(vec_bytes);
        double * buf_Re1  = (double *)malloc(vec_bytes);
        double * buf_Im1  = (double *)malloc(vec_bytes);
        double * TmpEIGEN = (double *)malloc((size_t)nk * sizeof(double));

        /* 行ポインタを作る（配列の配列に相当） */
        double ** ReEVec0 = (double **)malloc((size_t)max_tno * sizeof(double *));
        double ** ImEVec0 = (double **)malloc((size_t)max_tno * sizeof(double *));
        double ** ReEVec1 = (double **)malloc((size_t)max_tno * sizeof(double *));
        double ** ImEVec1 = (double **)malloc((size_t)max_tno * sizeof(double *));

        for (int i = 0; i < max_tno; ++i) {
            ReEVec0[i] = buf_Re0 + (size_t)i * nk;
            ImEVec0[i] = buf_Im0 + (size_t)i * nk;
            ReEVec1[i] = buf_Re1 + (size_t)i * nk;
            ImEVec1[i] = buf_Im1 + (size_t)i * nk;
        }

        /* ---------- Eigen 値をローカルへ -------------------------------- */
        {
            const double * src = &EIGEN[spin][kloop][kmin];
            for (int k = 0; k < nk; ++k)
                TmpEIGEN[k] = src[k];
        }

        const int stride = ie2[myid2] - is2[myid2] + 1;
        size_t    p      = 0; /* CDM1/EDM1 書込オフセット */

        /*========================== GA ループ ===========================*/
        for (int GA_AN = 1; GA_AN <= atomnum; ++GA_AN) {
            const int wanA = WhatSpecies[GA_AN];
            const int tnoA = Spe_Total_CNO[wanA];
            const int Anum = MP[GA_AN];

            /* ----- GA 行列要素を作業バッファへ ------------------------- */
            for (int i = 0; i < tnoA; ++i) {
                const size_t     base = (size_t)(Anum + i - 1) * stride - is2[myid2] + kmin;
                const dcomplex * v    = &EVec1[spin][base];
                double * restrict r   = ReEVec0[i];
                double * restrict im  = ImEVec0[i];
                for (int k = 0; k < nk; ++k) {
                    r[k]  = v[k].r;
                    im[k] = v[k].i;
                }
            }

            /*===================  近接原子 LB ループ  ===================*/
            for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; ++LB_AN) {
                const int GB_AN = natn[GA_AN][LB_AN];
                const int Rn    = ncn[GA_AN][LB_AN];
                const int wanB  = WhatSpecies[GB_AN];
                const int tnoB  = Spe_Total_CNO[wanB];
                const int Bnum  = MP[GB_AN];

                /* ----- フェーズ因子（sin/cos 同時計算） ----------------- */
                const int    l1  = atv_ijk[Rn][1];
                const int    l2  = atv_ijk[Rn][2];
                const int    l3  = atv_ijk[Rn][3];
                const double kRn = k1 * l1 + k2 * l2 + k3 * l3;
                double       si, co;
                sincos(2.0 * PI * kRn, &si, &co);

                /* ----- GB 行列要素を作業バッファへ ---------------------- */
                for (int j = 0; j < tnoB; ++j) {
                    const size_t     base = (size_t)(Bnum + j - 1) * stride - is2[myid2] + kmin;
                    const dcomplex * v    = &EVec1[spin][base];
                    double * restrict r   = ReEVec1[j];
                    double * restrict im  = ImEVec1[j];
                    for (int k = 0; k < nk; ++k) {
                        r[k]  = v[k].r;
                        im[k] = v[k].i;
                    }
                }

                /*================ (i,j) 内側ループ ======================*/
                for (int i = 0; i < tnoA; ++i) {
                    const double * restrict r0  = ReEVec0[i];
                    const double * restrict im0 = ImEVec0[i];

                    for (int j = 0; j < tnoB; ++j, ++p) {
                        const double * restrict r1  = ReEVec1[j];
                        const double * restrict im1 = ImEVec1[j];

                        double d1 = 0.0, d2 = 0.0, d3 = 0.0, d4 = 0.0;
                        int    k = 0;

                        /* ---- k ループを 4 要素アンローリング ------------- */
                        for (; k + 3 < nk; k += 4) {
#define KSTEP(idx)                                                                                                     \
    do {                                                                                                               \
        double reA = r0[idx] * r1[idx] + im0[idx] * im1[idx];                                                          \
        double imA = r0[idx] * im1[idx] - im0[idx] * r1[idx];                                                          \
        d1 += reA;                                                                                                     \
        d2 += imA;                                                                                                     \
        d3 += reA * TmpEIGEN[idx];                                                                                     \
        d4 += imA * TmpEIGEN[idx];                                                                                     \
    } while (0)
                            KSTEP(k);
                            KSTEP(k + 1);
                            KSTEP(k + 2);
                            KSTEP(k + 3);
#undef KSTEP
                        }
                        for (; k < nk; ++k) {
                            double reA = r0[k] * r1[k] + im0[k] * im1[k];
                            double imA = r0[k] * im1[k] - im0[k] * r1[k];
                            d1 += reA;
                            d2 += imA;
                            d3 += reA * TmpEIGEN[k];
                            d4 += imA * TmpEIGEN[k];
                        }

                        CDM1[p] += co * d1 - si * d2;
                        EDM1[p] += co * d3 - si * d4;
                    }
                }
            } /* LB_AN */
        } /* GA_AN */

        free(buf_Re0);
        free(buf_Im0);
        free(buf_Re1);
        free(buf_Im1);
        free(ReEVec0);
        free(ImEVec0);
        free(ReEVec1);
        free(ImEVec1);
        free(TmpEIGEN);

        if (measure_time) {
            dtime(&Etime0);
            time82 += Etime0 - Stime0;
            dtime(&Stime0);
        }

        /* sum of CDM1 and EDM1 by Allreduce in MPI */

        MPI_Allreduce(CDM1, H1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
        for (i = 0; i < size_H1; i++)
            CDM1[i] = H1[i];

        MPI_Allreduce(EDM1, H1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
        for (i = 0; i < size_H1; i++)
            EDM1[i] = H1[i];

        if (measure_time) {
            dtime(&Etime0);
            time83 += Etime0 - Stime0;
            dtime(&Stime0);
        }

        /* store DM1 to a proper place in CDM and EDM */

        p = 0;
        for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {

            MA_AN = F_G2M[GA_AN];
            wanA  = WhatSpecies[GA_AN];
            tnoA  = Spe_Total_CNO[wanA];
            Anum  = MP[GA_AN];
            ID    = G2ID[GA_AN];

            for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                GB_AN = natn[GA_AN][LB_AN];
                wanB  = WhatSpecies[GB_AN];
                tnoB  = Spe_Total_CNO[wanB];
                Bnum  = MP[GB_AN];

                if (myid0 == ID) {

                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {

                            CDM[spin][MA_AN][LB_AN][i][j] = CDM1[p];
                            EDM[spin][MA_AN][LB_AN][i][j] = EDM1[p];

                            /* increment of p */
                            p++;
                        }
                    }
                } else {
                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {
                            /* increment of p */
                            p++;
                        }
                    }
                }

            } /* LB_AN */
        } /* GA_AN */

        if (measure_time) {
            dtime(&Etime0);
            time84 += Etime0 - Stime0;
            dtime(&Stime0);
        }

        /* if necessary, MPI communication of CDM and EDM */

        if (1 < numprocs0 && SpinP_switch == 1) {

            /* set spin */

            if (myworld1 == 0) {
                spin = 1;
            } else {
                spin = 0;
            }

            /* communicate CDM1 and EDM1 */

            for (i = 0; i <= 1; i++) {

                IDS = Comm_World_StartID1[i % 2];
                IDR = Comm_World_StartID1[(i + 1) % 2];

                if (myid0 == IDS) {
                    MPI_Isend(&CDM1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
                }

                if (myid0 == IDR) {
                    MPI_Recv(&H1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Wait(&request, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Isend(&EDM1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
                }

                if (myid0 == IDR) {
                    MPI_Recv(&S1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Wait(&request, &stat);
                }
            }

            MPI_Bcast(&H1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
            MPI_Bcast(&S1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);

            /* put CDM1 and EDM1 into CDM and EDM */

            k = 0;
            for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {

                MA_AN = F_G2M[GA_AN];
                wanA  = WhatSpecies[GA_AN];
                tnoA  = Spe_Total_CNO[wanA];

                for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {

                    GB_AN = natn[GA_AN][LB_AN];
                    wanB  = WhatSpecies[GB_AN];
                    tnoB  = Spe_Total_CNO[wanB];

                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {

                            if (1 <= MA_AN && MA_AN <= Matomnum) {
                                CDM[spin][MA_AN][LB_AN][i][j] = H1[k];
                                EDM[spin][MA_AN][LB_AN][i][j] = S1[k];
                            }

                            k++;
                        }
                    }
                }
            }
        }

        if (measure_time) {
            dtime(&Etime0);
            time85 += Etime0 - Stime0;
        }

    } /* if (all_knum==1) */

    if (measure_time) {
        dtime(&Etime);
        time8 += Etime - Stime;
        part4 += Etime - Stime;
        if (SCF_iter != 1) {
            part4sum += part4;
        }
    }

    dtime(&EiloopTime);

    if (myid0 == Host_ID && 0 < level_stdout && scf_eigen_lib_flag != CuSOLVER) {
        printf("<Band_DFT>  Eigen, time=%lf\n", EiloopTime - SiloopTime);
        fflush(stdout);
    }
    else if (myid0 == Host_ID && 0 < level_stdout && scf_eigen_lib_flag == CuSOLVER) {
        printf("<Band_DFT>  Eigen (GPU-accelerated), time=%lf\n", EiloopTime - SiloopTime);
        fflush(stdout);
    }

    dtime(&SiloopTime);

    /****************************************************
     ****************************************************
       diagonalization for calculating density matrix
     ****************************************************
    ****************************************************/

    if (all_knum != 1) {
        /* spin=myworld1 */

        spin = myworld1;

    diagonalize2:

        /* set S1 */

        size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

        /* set H1 */

        if (SpinP_switch == 0) {
            size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
        } else if (1 < numprocs0) {
            size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
            size_H1 = Get_OneD_HS_Col(1, nh[1], CDM1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

            if (myworld1) {
                for (i = 0; i < size_H1; i++) {
                    H1[i] = CDM1[i];
                }
            }
        } else {
            size_H1 = Get_OneD_HS_Col(1, nh[spin], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
        }

        /* initialize CDM1 and EDM1 */

        for (i = 0; i < size_H1; i++) {
            CDM1[i] = 0.0;
            EDM1[i] = 0.0;
        }

        /* initialize CDM, EDM, and iDM */

        for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
            GA_AN = M2G[MA_AN];
            wanA  = WhatSpecies[GA_AN];
            tnoA  = Spe_Total_CNO[wanA];
            Anum  = MP[GA_AN];
            for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                GB_AN = natn[GA_AN][LB_AN];
                wanB  = WhatSpecies[GB_AN];
                tnoB  = Spe_Total_CNO[wanB];
                Bnum  = MP[GB_AN];

                for (i = 0; i < tnoA; i++) {
                    for (j = 0; j < tnoB; j++) {
                        CDM[spin][MA_AN][LB_AN][i][j] = 0.0;
                        EDM[spin][MA_AN][LB_AN][i][j] = 0.0;

                        iDM[0][0][MA_AN][LB_AN][i][j] = 0.0;
                        iDM[0][1][MA_AN][LB_AN][i][j] = 0.0;
                    }
                }
            }
        }

        /* for kloop */

        if (scf_eigen_lib_flag == CuSOLVER) {
            for (int kloop0 = 0; kloop0 < num_kloop0; kloop0++) {
                int kloop = kloop0 + S_knum;

                double k1 = T_KGrids1[kloop];
                double k2 = T_KGrids2[kloop];
                double k3 = T_KGrids3[kloop];

                /* make S and H */

                Construct_Band_CsHs(SCF_iter, all_knum, order_GA, MP, S1, H1, k1, k2, k3, Ss, Hs, n, myid2);

#pragma acc update device(Hs[0 : n * n], Ss[0 : n * n])

                /* diagonalize S */

                if (measure_time)
                    dtime(&Stime);

                EigenBand_lapack_openacc(Ss, ko, n, n);

                if (measure_time) {
                    dtime(&Etime);
                    time9 += Etime - Stime;
                }

                if (3 <= level_stdout) {
                    printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n", myid0, kloop, T_KGrids1[kloop],
                           T_KGrids2[kloop], T_KGrids3[kloop]);
                    for (i1 = 1; i1 <= n; i1++) {
                        printf("  Eigenvalues of OLP  %2d  %15.12f\n", i1, ko[i1]);
                    }
                }

                /* minus eigenvalues to 1.0e-14 */

#pragma acc kernels
#pragma acc loop independent
                for (int l = 1; l <= n; l++) {
                    if (ko[l] < 0.0) {
                        ko[l] = 1.0e-10;
                    }
                }
#pragma acc update self(ko[0 : n + 1])

                for (int l = 1; l <= n; l++) {
                    koS[l] = ko[l];
                }

                // #pragma acc update device(ko[0 : n + 1])

                /* calculate S*1/sqrt(ko) */

#pragma acc kernels
#pragma acc loop independent
                for (int l = 1; l <= n; l++)
                    ko[l] = 1.0 / sqrt(ko[l]);

#pragma acc kernels
#pragma acc loop independent collapse(2)
                for (int i1 = 1; i1 <= n; i1++) {
                    for (int j1 = 1; j1 <= n; j1++) {
                        Ss[(j1 - 1) * n + i1 - 1].r *= ko[j1];
                        Ss[(j1 - 1) * n + i1 - 1].i *= ko[j1];
                    }
                }

                /* S * 1.0/sqrt(ko[l])  */

                /****************************************************
                      1/sqrt(ko) * U^t * H * U * 1/sqrt(ko)
                ****************************************************/

                for (i = 0; i < na_rows_max * na_cols_max; i++) {
                    Cs[i].r = 0.0;
                    Cs[i].i = 0.0;
                }

                my_cublasZgemm_openacc(CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, Hs, Ss, Cs);

                /* pzgemm */

                /* H * U * 1/sqrt(ko) */

                /* 1/sqrt(ko) * U^+ H * U * 1/sqrt(ko) */

#pragma acc kernels
#pragma acc loop independent
                for (int i = 0; i < n * n; i++) {
                    Hs[i].r = 0.0;
                    Hs[i].i = 0.0;
                }

                my_cublasZgemm_openacc(CUBLAS_OP_C, CUBLAS_OP_N, n, n, n, Ss, Cs, Hs);

#pragma acc kernels
#pragma acc loop independent
                for (int i = 0; i < n * n; i++) {
                    Cs[i].r = Hs[i].r;
                    Cs[i].i = Hs[i].i;
                }

                /* diagonalize H' */

                if (measure_time)
                    dtime(&Stime);

                EigenBand_lapack_openacc(Cs, ko, n, MaxN);

                if (measure_time) {
                    dtime(&Etime);
                    time10 += Etime - Stime;
                }

                if (3 <= level_stdout && 0 <= kloop) {
                    printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n", kloop, T_KGrids1[kloop], T_KGrids2[kloop],
                           T_KGrids3[kloop]);
                    for (i1 = 1; i1 <= n; i1++) {
                        printf("  Eigenvalues of Kohn-Sham(DM) spin=%2d i1=%2d %15.12f\n", spin, i1, ko[i1]);
                    }
                }

                /****************************************************
                  transformation to the original eigenvectors.
                   NOTE JRCAT-244p and JAIST-2122p
                ****************************************************/
#pragma acc kernels
#pragma acc loop independent
                for (int i = 0; i < n * n; i++) {
                    Hs[i] = Cs[i];
                }

                /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */

                my_cublasZgemm_openacc(CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, Hs, Ss, Cs);

#pragma acc update self(Cs[0 : n * n])
                /* Hs are stored to EVec1 */

                int ktmp = 0;
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < n; i++) {

                        EVec1[spin][ktmp].r = Cs[j * n + i].r;
                        EVec1[spin][ktmp].i = Cs[j * n + i].i;

                        ktmp++;
                    }
                }

                /****************************************************
                               calculate DM and EDM
                ****************************************************/

                if (measure_time)
                    dtime(&Stime);

                /* weight of k-point */

                double kw = (double)T_k_op[kloop];

                int po   = 0;
                int kmin = 1;
                int kmax = MaxN;

                if (measure_time)
                    dtime(&Stime1);

                for (int k = 1; k <= MaxN; k++) {

                    double eig = EIGEN[spin][kloop][k];
                    double x;

                    if (xanes_calc == 1)
                        x = (eig - ChemP_XANES[spin]) * Beta;
                    else
                        x = (eig - ChemP) * Beta;

                    if (x <= -x_cut)
                        x = -x_cut;
                    if (x_cut <= x)
                        x = x_cut;
                    double FermiF = FermiFunc(x, spin, k, &k, &x);

                    double tmp1 = sqrt(kw * FermiF);

                    for (int i1 = 1; i1 <= n; i1++) {
                        int i = (i1 - 1) * n + k - 1;
                        EVec1[spin][i].r *= tmp1;
                        EVec1[spin][i].i *= tmp1;
                    }

                    /* find kmax */

                    if (FermiF < FermiEps && po == 0) {
                        kmax = k;
                        po   = 1;
                    }
                }

                if (measure_time) {
                    dtime(&Etime1);
                    time11A += Etime1 - Stime1;
                    dtime(&Stime1);
                }

                /* store EIGEN to a temporary array */

                for (int k = 1; k <= MaxN; k++) {
                    TmpEIGEN[k] = EIGEN[spin][kloop][k];
                }

                /* calculation of CDM1 and EDM1 */

                // int wanAmax = 1;
                // int tnoAmax = 1;
                // int tnoBmax = 1;
                // int i1max = 1;
                // int j1max = 1;
                // int LB_ANmax = 1;
                // int Rnmax = 1;
                // int GB_ANmax = 1;
                // int pmax = 1;
                // for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
                //     if (wanAmax < WhatSpecies[GA_AN]) {
                //         wanAmax = WhatSpecies[GA_AN];
                //     }

                //     if (tnoAmax < Spe_Total_CNO[WhatSpecies[GA_AN]]) {
                //         tnoAmax = Spe_Total_CNO[WhatSpecies[GA_AN]];
                //     }

                //     int wanA = WhatSpecies[GA_AN];
                //     int tnoA = Spe_Total_CNO[wanA];
                //     int Anum = MP[GA_AN];

                //     for (int i = 0; i < tnoA; i++) {
                //         int i1 = (Anum + i - 1) * n - 1;

                //         if (i1max < i1) {
                //             i1max = i1;
                //         }
                //     }

                //     if (LB_ANmax < FNAN[GA_AN]) {
                //         LB_ANmax = FNAN[GA_AN];
                //     }

                //     for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                //         if (Rnmax < ncn[GA_AN][LB_AN]) {
                //             Rnmax = ncn[GA_AN][LB_AN];
                //         }

                //         if (GB_ANmax < natn[GA_AN][LB_AN]) {
                //             GB_ANmax = natn[GA_AN][LB_AN];
                //         }

                //         int GB_AN = natn[GA_AN][LB_AN];
                //         int wanB = WhatSpecies[GB_AN];
                //         int tnoB = Spe_Total_CNO[wanB];
                //         int Bnum = MP[GB_AN];

                //         if (tnoBmax < tnoB) {
                //             tnoBmax = tnoB;
                //         }

                //         for (i = 0; i < tnoA; i++) {
                //             for (j = 0; j < tnoB; j++) {
                //                 /* increment of p */
                //                 pmax++;
                //             }
                //         }
                //     }
                // }

                // int ijmax = i1max > j1max ? i1max : j1max;

                // #pragma acc data copy(ReEVec0[:tnoAmax][:MaxN + 1], ImEVec0[:tnoAmax][:MaxN + 1])
                // #pragma acc data copy(FNAN[:atomnum + 1])
                // #pragma acc data copy(EVec1[:2][:ijmax + MaxN + 1])
                // #pragma acc data copy(Spe_Total_CNO[:wanAmax])
                // #pragma acc data copy(natn[:atomnum + 1][:LB_ANmax], ncn[:atomnum + 1][:LB_ANmax], atv_ijk[:Rnmax][:4])
                // #pragma acc data copy(ReEVec1[:tnoBmax][:MaxN + 1], ImEVec1[:tnoBmax][:MaxN + 1])
                // #pragma acc data copy(MP[:atomnum + 1], WhatSpecies[:atomnum + 1], TmpEIGEN[:MaxN + 1])
                // #pragma acc data copy(CDM1[:pmax], EDM1[:pmax])
                // #pragma acc kernels
                // #pragma acc loop independent gang

                int p = 0;
                for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
                    int wanA = WhatSpecies[GA_AN];
                    int tnoA = Spe_Total_CNO[wanA];
                    int Anum = MP[GA_AN];

                    /* store EVec1 to temporary arrays */

                    for (int i = 0; i < tnoA; i++) {
                        int i1 = (Anum + i - 1) * n - 1;
                        for (int k = 1; k <= MaxN; k++) {
                            ReEVec0[i][k] = EVec1[spin][i1 + k].r;
                            ImEVec0[i][k] = EVec1[spin][i1 + k].i;
                        }
                    }

                    // #pragma acc loop independent vector(1024)
                    for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                        int GB_AN = natn[GA_AN][LB_AN];
                        int Rn    = ncn[GA_AN][LB_AN];
                        int wanB  = WhatSpecies[GB_AN];
                        int tnoB  = Spe_Total_CNO[wanB];
                        int Bnum  = MP[GB_AN];

                        int    l1  = atv_ijk[Rn][1];
                        int    l2  = atv_ijk[Rn][2];
                        int    l3  = atv_ijk[Rn][3];
                        double kRn = k1 * (double)l1 + k2 * (double)l2 + k3 * (double)l3;

                        double si = sin(2.0 * PI * kRn);
                        double co = cos(2.0 * PI * kRn);

                        /* store EVec1 to temporary arrays */

                        for (int j = 0; j < tnoB; j++) {
                            int j1 = (Bnum + j - 1) * n - 1;
                            for (int k = 1; k <= MaxN; k++) {
                                ReEVec1[j][k] = EVec1[spin][j1 + k].r;
                                ImEVec1[j][k] = EVec1[spin][j1 + k].i;
                            }
                        }

                        for (int i = 0; i < tnoA; i++) {
                            for (int j = 0; j < tnoB; j++) {
                                /***************************************************************
                                 Note that the imaginary part is zero,
                                 since

                                 at k
                                 A = (co + i si)(Re + i Im) = (co*Re - si*Im) + i (co*Im + si*Re)
                                 at -k
                                 B = (co - i si)(Re - i Im) = (co*Re - si*Im) - i (co*Im + si*Re)
                                 Thus, Re(A+B) = 2*(co*Re - si*Im)
                                       Im(A+B) = 0
                                ***************************************************************/

                                double d1 = 0.0;
                                double d2 = 0.0;
                                double d3 = 0.0;
                                double d4 = 0.0;

                                // #pragma acc loop reduction(+:d1) reduction(+:d2) reduction(+:d3) reduction(+:d4)
                                for (int k = 1; k <= MaxN; k++) {
                                    double ReA = ReEVec0[i][k] * ReEVec1[j][k] + ImEVec0[i][k] * ImEVec1[j][k];
                                    double ImA = ReEVec0[i][k] * ImEVec1[j][k] - ImEVec0[i][k] * ReEVec1[j][k];

                                    d1 += ReA;
                                    d2 += ImA;
                                    d3 += ReA * TmpEIGEN[k];
                                    d4 += ImA * TmpEIGEN[k];
                                }

                                // int p = (((GA_AN - 1) * (FNAN[GA_AN] + 1) + LB_AN) * tnoA + i) * tnoB + j;

                                // #pragma omp atomic
                                CDM1[p] += co * d1 - si * d2;
                                // #pragma omp atomic
                                EDM1[p] += co * d3 - si * d4;

                                p++;
                            }
                        }
                    }
                } /* GA_AN */

                if (measure_time) {
                    dtime(&Etime1);
                    time11B += Etime1 - Stime1;

                    dtime(&Etime);
                    time11 += Etime - Stime;
                }

            } /* kloop0 */

        } else {
            for (kloop0 = 0; kloop0 < num_kloop0; kloop0++) {

                kloop = kloop0 + S_knum;

                k1 = T_KGrids1[kloop];
                k2 = T_KGrids2[kloop];
                k3 = T_KGrids3[kloop];

                /* make S and H */

                Construct_Band_CsHs(SCF_iter, all_knum, order_GA, MP, S1, H1, k1, k2, k3, Cs, Hs, n, myid2);

                // #pragma acc update device(Hs[0 : na_rows * na_cols], Cs[0 : na_rows * na_cols])

                /* diagonalize S */

                if (measure_time)
                    dtime(&Stime);

                MPI_Comm_split(MPI_CommWD2[myworld2], my_pcol, my_prow, &mpi_comm_rows);
                MPI_Comm_split(MPI_CommWD2[myworld2], my_prow, my_pcol, &mpi_comm_cols);

                mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
                mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
                if (scf_eigen_lib_flag == 1) {
                    F77_NAME(solve_evp_complex, SOLVE_EVP_COMPLEX)
                    (&n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
                } else if (scf_eigen_lib_flag == 2) {

#ifndef kcomp
                    int mpiworld;
                    mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
                    F77_NAME(elpa_solve_evp_complex_2stage_double_impl, ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
                    (&n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int,
                     &mpi_comm_cols_int, &mpiworld);
#endif
                }

                MPI_Comm_free(&mpi_comm_rows);
                MPI_Comm_free(&mpi_comm_cols);

                if (measure_time) {
                    dtime(&Etime);
                    time9 += Etime - Stime;
                }

                if (3 <= level_stdout) {
                    printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n", myid0, kloop, T_KGrids1[kloop],
                           T_KGrids2[kloop], T_KGrids3[kloop]);
                    for (i1 = 1; i1 <= n; i1++) {
                        printf("  Eigenvalues of OLP  %2d  %15.12f\n", i1, ko[i1]);
                    }
                }

                /* minus eigenvalues to 1.0e-14 */

                // #pragma acc kernels
                // #pragma acc loop independent
                for (l = 1; l <= n; l++) {
                    if (ko[l] < 0.0) {
                        ko[l] = 1.0e-10;
                    }
                }

                /* calculate S*1/sqrt(ko) */

                // #pragma acc kernels
                // #pragma acc loop independent
                for (l = 1; l <= n; l++) {
                    ko[l] = 1.0 / sqrt(ko[l]);
                }

                /* S * 1.0/sqrt(ko[l])  */

                // #pragma acc kernels
                // #pragma acc loop independent
                for (i = 0; i < na_rows; i++) {
                    // #pragma acc loop independent
                    for (j = 0; j < na_cols; j++) {
                        int jg =
                            np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + my_pcol) % np_cols) * nblk + 1;
                        Ss[j * na_rows + i].r = Ss[j * na_rows + i].r * ko[jg];
                        Ss[j * na_rows + i].i = Ss[j * na_rows + i].i * ko[jg];
                    }
                }

                /****************************************************
                      1/sqrt(ko) * U^t * H * U * 1/sqrt(ko)
                ****************************************************/

                /* pzgemm */

                /* H * U * 1/sqrt(ko) */

                // #pragma acc kernels
                // #pragma acc loop independent
                for (i = 0; i < na_rows * na_cols; i++) {
                    Cs[i].r = 0.0;
                    Cs[i].i = 0.0;
                }

                // cublasmp_zgemm(CUBLAS_OP_N, CUBLAS_OP_N, Hs, Ss, Cs, &opts, &opts2);

                Cblacs_barrier(ictxt2, "A");
                F77_NAME(pzgemm, PZGEMM)
                ("N", "N", &n, &n, &n, &alpha, Hs, &ONE, &ONE, descH, Ss, &ONE, &ONE, descS, &beta, Cs, &ONE, &ONE,
                 descC);

                /* 1/sqrt(ko) * U^+ H * U * 1/sqrt(ko) */

                // #pragma acc kernels
                // #pragma acc loop independent
                for (i = 0; i < na_rows * na_cols; i++) {
                    Hs[i].r = 0.0;
                    Hs[i].i = 0.0;
                }

                // cublasmp_zgemm(CUBLAS_OP_C, CUBLAS_OP_N, Ss, Cs, Hs, &opts, &opts2);

                Cblacs_barrier(ictxt2, "C");
                F77_NAME(pzgemm, PZGEMM)
                ("C", "N", &n, &n, &n, &alpha, Ss, &ONE, &ONE, descS, Cs, &ONE, &ONE, descC, &beta, Hs, &ONE, &ONE,
                 descH);

                /* diagonalize H' */

                if (measure_time)
                    dtime(&Stime);

                MPI_Comm_split(MPI_CommWD2[myworld2], my_pcol, my_prow, &mpi_comm_rows);
                MPI_Comm_split(MPI_CommWD2[myworld2], my_prow, my_pcol, &mpi_comm_cols);

                mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
                mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

                if (scf_eigen_lib_flag == 1) {
                    F77_NAME(solve_evp_complex, SOLVE_EVP_COMPLEX)
                    (&n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
                } else if (scf_eigen_lib_flag == 2) {

#ifndef kcomp
                    int mpiworld;
                    mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
                    F77_NAME(elpa_solve_evp_complex_2stage_double_impl, ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
                    (&n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int,
                     &mpi_comm_cols_int, &mpiworld);
#endif
                }

                MPI_Comm_free(&mpi_comm_rows);
                MPI_Comm_free(&mpi_comm_cols);

                if (measure_time) {
                    dtime(&Etime);
                    time10 += Etime - Stime;
                }

                if (3 <= level_stdout && 0 <= kloop) {
                    printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n", kloop, T_KGrids1[kloop], T_KGrids2[kloop],
                           T_KGrids3[kloop]);
                    for (i1 = 1; i1 <= n; i1++) {
                        printf("  Eigenvalues of Kohn-Sham(DM) spin=%2d i1=%2d %15.12f\n", spin, i1, ko[i1]);
                    }
                }

                /****************************************************
                  transformation to the original eigenvectors.
                   NOTE JRCAT-244p and JAIST-2122p
                ****************************************************/

                // #pragma acc kernels
                // #pragma acc loop independent
                for (i = 0; i < na_rows * na_cols; i++) {
                    Hs[i].r = 0.0;
                    Hs[i].i = 0.0;
                }

                // cublasmp_zgemm(CUBLAS_OP_T, CUBLAS_OP_T, Cs, Ss, Hs, &opts, &opts2);

                F77_NAME(pzgemm, PZGEMM)
                ("T", "T", &n, &n, &n, &alpha, Cs, &ONE, &ONE, descS, Ss, &ONE, &ONE, descC, &beta, Hs, &ONE, &ONE,
                 descH);
                Cblacs_barrier(ictxt2, "A");
                // #pragma acc update self(Hs[0 : na_rows * na_cols])

                /* Hs are stored to EVec1 */

                k = 0;
                for (j = 0; j < na_cols; j++) {
                    for (i = 0; i < na_rows; i++) {

                        EVec1[spin][k].r = Hs[j * na_rows + i].r;
                        EVec1[spin][k].i = Hs[j * na_rows + i].i;

                        k++;
                    }
                }

                /****************************************************
                               calculate DM and EDM
                ****************************************************/

                if (measure_time)
                    dtime(&Stime);

                /* weight of k-point */

                kw = (double)T_k_op[kloop];

                po   = 0;
                kmin = 1;
                kmax = MaxN;

                if (measure_time)
                    dtime(&Stime1);

                for (k = 1; k <= MaxN; k++) {

                    eig = EIGEN[spin][kloop][k];

                    if (xanes_calc == 1)
                        x = (eig - ChemP_XANES[spin]) * Beta;
                    else
                        x = (eig - ChemP) * Beta;

                    if (x <= -x_cut)
                        x = -x_cut;
                    if (x_cut <= x)
                        x = x_cut;
                    FermiF = FermiFunc(x, spin, k, &k, &x);

                    tmp1 = sqrt(kw * FermiF);

                    for (i1 = 1; i1 <= n; i1++) {
                        i = (i1 - 1) * n + k - 1;
                        EVec1[spin][i].r *= tmp1;
                        EVec1[spin][i].i *= tmp1;
                    }

                    /* find kmax */

                    if (FermiF < FermiEps && po == 0) {
                        kmax = k;
                        po   = 1;
                    }
                }

                if (measure_time) {
                    dtime(&Etime1);
                    time11A += Etime1 - Stime1;
                    dtime(&Stime1);
                }

                /* store EIGEN to a temporary array */

                for (k = 1; k <= MaxN; k++) {
                    TmpEIGEN[k] = EIGEN[spin][kloop][k];
                }

                /* calculation of CDM1 and EDM1 */

                // int wanAmax = 1;
                // int tnoAmax = 1;
                // int tnoBmax = 1;
                // int i1max = 1;
                // int j1max = 1;
                // int LB_ANmax = 1;
                // int Rnmax = 1;
                // int GB_ANmax = 1;
                // int pmax = 1;
                // for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
                //     if (wanAmax < WhatSpecies[GA_AN]) {
                //         wanAmax = WhatSpecies[GA_AN];
                //     }

                //     if (tnoAmax < Spe_Total_CNO[WhatSpecies[GA_AN]]) {
                //         tnoAmax = Spe_Total_CNO[WhatSpecies[GA_AN]];
                //     }

                //     int wanA = WhatSpecies[GA_AN];
                //     int tnoA = Spe_Total_CNO[wanA];
                //     int Anum = MP[GA_AN];

                //     for (int i = 0; i < tnoA; i++) {
                //         int i1 = (Anum + i - 1) * n - 1;

                //         if (i1max < i1) {
                //             i1max = i1;
                //         }
                //     }

                //     if (LB_ANmax < FNAN[GA_AN]) {
                //         LB_ANmax = FNAN[GA_AN];
                //     }

                //     for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                //         if (Rnmax < ncn[GA_AN][LB_AN]) {
                //             Rnmax = ncn[GA_AN][LB_AN];
                //         }

                //         if (GB_ANmax < natn[GA_AN][LB_AN]) {
                //             GB_ANmax = natn[GA_AN][LB_AN];
                //         }

                //         int GB_AN = natn[GA_AN][LB_AN];
                //         int wanB = WhatSpecies[GB_AN];
                //         int tnoB = Spe_Total_CNO[wanB];
                //         int Bnum = MP[GB_AN];

                //         if (tnoBmax < tnoB) {
                //             tnoBmax = tnoB;
                //         }

                //         for (i = 0; i < tnoA; i++) {
                //             for (j = 0; j < tnoB; j++) {
                //                 /* increment of p */
                //                 pmax++;
                //             }
                //         }
                //     }
                // }

                // int ijmax = i1max > j1max ? i1max : j1max;

                // #pragma acc data copy(ReEVec0[:tnoAmax][:MaxN + 1], ImEVec0[:tnoAmax][:MaxN + 1])
                // #pragma acc data copy(FNAN[:atomnum + 1])
                // #pragma acc data copy(EVec1[:2][:ijmax + MaxN + 1])
                // #pragma acc data copy(Spe_Total_CNO[:wanAmax])
                // #pragma acc data copy(natn[:atomnum + 1][:LB_ANmax], ncn[:atomnum + 1][:LB_ANmax], atv_ijk[:Rnmax][:4])
                // #pragma acc data copy(ReEVec1[:tnoBmax][:MaxN + 1], ImEVec1[:tnoBmax][:MaxN + 1])
                // #pragma acc data copy(MP[:atomnum + 1], WhatSpecies[:atomnum + 1], TmpEIGEN[:MaxN + 1])
                // #pragma acc data copy(CDM1[:pmax], EDM1[:pmax])
                // #pragma acc kernels
                // #pragma acc loop independent gang

                int p = 0;
                for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
                    int wanA = WhatSpecies[GA_AN];
                    int tnoA = Spe_Total_CNO[wanA];
                    int Anum = MP[GA_AN];

                    /* store EVec1 to temporary arrays */

                    for (int i = 0; i < tnoA; i++) {
                        int i1 = (Anum + i - 1) * n - 1;
                        for (int k = 1; k <= MaxN; k++) {
                            ReEVec0[i][k] = EVec1[spin][i1 + k].r;
                            ImEVec0[i][k] = EVec1[spin][i1 + k].i;
                        }
                    }

                    // #pragma acc loop independent vector(1024)
                    for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                        int GB_AN = natn[GA_AN][LB_AN];
                        int Rn    = ncn[GA_AN][LB_AN];
                        int wanB  = WhatSpecies[GB_AN];
                        int tnoB  = Spe_Total_CNO[wanB];
                        int Bnum  = MP[GB_AN];

                        int    l1  = atv_ijk[Rn][1];
                        int    l2  = atv_ijk[Rn][2];
                        int    l3  = atv_ijk[Rn][3];
                        double kRn = k1 * (double)l1 + k2 * (double)l2 + k3 * (double)l3;

                        double si = sin(2.0 * PI * kRn);
                        double co = cos(2.0 * PI * kRn);

                        /* store EVec1 to temporary arrays */

                        for (int j = 0; j < tnoB; j++) {
                            int j1 = (Bnum + j - 1) * n - 1;
                            for (int k = 1; k <= MaxN; k++) {
                                ReEVec1[j][k] = EVec1[spin][j1 + k].r;
                                ImEVec1[j][k] = EVec1[spin][j1 + k].i;
                            }
                        }

                        for (int i = 0; i < tnoA; i++) {
                            for (int j = 0; j < tnoB; j++) {
                                /***************************************************************
                                 Note that the imaginary part is zero,
                                 since

                                 at k
                                 A = (co + i si)(Re + i Im) = (co*Re - si*Im) + i (co*Im + si*Re)
                                 at -k
                                 B = (co - i si)(Re - i Im) = (co*Re - si*Im) - i (co*Im + si*Re)
                                 Thus, Re(A+B) = 2*(co*Re - si*Im)
                                       Im(A+B) = 0
                                ***************************************************************/

                                double d1 = 0.0;
                                double d2 = 0.0;
                                double d3 = 0.0;
                                double d4 = 0.0;

                                // #pragma acc loop reduction(+:d1) reduction(+:d2) reduction(+:d3) reduction(+:d4)
                                for (int k = 1; k <= MaxN; k++) {
                                    ReA = ReEVec0[i][k] * ReEVec1[j][k] + ImEVec0[i][k] * ImEVec1[j][k];
                                    ImA = ReEVec0[i][k] * ImEVec1[j][k] - ImEVec0[i][k] * ReEVec1[j][k];

                                    d1 += ReA;
                                    d2 += ImA;
                                    d3 += ReA * TmpEIGEN[k];
                                    d4 += ImA * TmpEIGEN[k];
                                }

                                // int p = (((GA_AN - 1) * (FNAN[GA_AN] + 1) + LB_AN) * tnoA + i) * tnoB + j;

                                CDM1[p] += co * d1 - si * d2;
                                EDM1[p] += co * d3 - si * d4;

                                p++;
                            }
                        }
                    }
                } /* GA_AN */

                if (measure_time) {
                    dtime(&Etime1);
                    time11B += Etime1 - Stime1;

                    dtime(&Etime);
                    time11 += Etime - Stime;
                }
            } /* kloop0 */
        }

        /*******************************************************
             sum of CDM1 and EDM1 by Allreduce in MPI
        *******************************************************/

        if (measure_time)
            dtime(&Stime);

        MPI_Allreduce(&CDM1[0], &H1[0], size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
        MPI_Allreduce(&EDM1[0], &S1[0], size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);

        /* CDM and EDM */

        k = 0;
        for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {

            MA_AN = F_G2M[GA_AN];
            wanA  = WhatSpecies[GA_AN];
            tnoA  = Spe_Total_CNO[wanA];
            ID    = G2ID[GA_AN];

            for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {

                GB_AN = natn[GA_AN][LB_AN];
                wanB  = WhatSpecies[GB_AN];
                tnoB  = Spe_Total_CNO[wanB];

                if (myid0 == ID) {

                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {
                            CDM[spin][MA_AN][LB_AN][i][j] = H1[k];
                            EDM[spin][MA_AN][LB_AN][i][j] = S1[k];
                            k++;
                        }
                    }
                }

                else {
                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {
                            k++;
                        }
                    }
                }
            }
        }

        if (measure_time) {
            dtime(&Etime);
            time12 += Etime - Stime;
        }

        if (SpinP_switch == 1 && numprocs0 == 1 && spin == 0) {
            spin++;
            goto diagonalize2;
        }

        /* if necessary, MPI communication of CDM and EDM */

        if (1 < numprocs0 && SpinP_switch == 1) {

            /* set spin */

            if (myworld1 == 0) {
                spin = 1;
            } else {
                spin = 0;
            }

            /* communicate CDM1 and EDM1 */

            for (i = 0; i <= 1; i++) {

                IDS = Comm_World_StartID1[i % 2];
                IDR = Comm_World_StartID1[(i + 1) % 2];

                if (myid0 == IDS) {
                    MPI_Isend(&H1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
                }

                if (myid0 == IDR) {
                    MPI_Recv(&CDM1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Wait(&request, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Isend(&S1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
                }

                if (myid0 == IDR) {
                    MPI_Recv(&EDM1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Wait(&request, &stat);
                }
            }

            MPI_Bcast(&CDM1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
            MPI_Bcast(&EDM1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);

            /* put CDM1 and EDM1 into CDM and EDM */

            k = 0;
            for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {

                MA_AN = F_G2M[GA_AN];
                wanA  = WhatSpecies[GA_AN];
                tnoA  = Spe_Total_CNO[wanA];

                for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {

                    GB_AN = natn[GA_AN][LB_AN];
                    wanB  = WhatSpecies[GB_AN];
                    tnoB  = Spe_Total_CNO[wanB];

                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {

                            if (1 <= MA_AN && MA_AN <= Matomnum) {
                                CDM[spin][MA_AN][LB_AN][i][j] = CDM1[k];
                                EDM[spin][MA_AN][LB_AN][i][j] = EDM1[k];
                            }

                            k++;
                        }
                    }
                }
            }
        }

    } /* if (all_knum!=1) */

    /****************************************************
             normalization of CDM, EDM, and iDM
    ****************************************************/

    dtime(&EiloopTime);

    if (myid0 == Host_ID && 0 < level_stdout && scf_eigen_lib_flag != CuSOLVER) {
        printf("<Band_DFT>  DM, time=%lf\n", EiloopTime - SiloopTime);
        fflush(stdout);
    }
    else if (myid0 == Host_ID && 0 < level_stdout && scf_eigen_lib_flag == CuSOLVER) {
        printf("<Band_DFT>  DM (GPU-accelerated), time=%lf\n", EiloopTime - SiloopTime);
        fflush(stdout);
    }

    dum = 1.0 / sum_weights;

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
            GA_AN = M2G[MA_AN];
            wanA  = WhatSpecies[GA_AN];
            tnoA  = Spe_Total_CNO[wanA];
            Anum  = MP[GA_AN];
            for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                GB_AN = natn[GA_AN][LB_AN];
                wanB  = WhatSpecies[GB_AN];
                tnoB  = Spe_Total_CNO[wanB];
                Bnum  = MP[GB_AN];

                for (i = 0; i < tnoA; i++) {
                    for (j = 0; j < tnoB; j++) {
                        CDM[spin][MA_AN][LB_AN][i][j]    = CDM[spin][MA_AN][LB_AN][i][j] * dum;
                        EDM[spin][MA_AN][LB_AN][i][j]    = EDM[spin][MA_AN][LB_AN][i][j] * dum;
                        iDM[0][spin][MA_AN][LB_AN][i][j] = iDM[0][spin][MA_AN][LB_AN][i][j] * dum;
                    }
                }
            }
        }
    }

    /****************************************************
                         bond-energies
    ****************************************************/

    My_Eele1[0] = 0.0;
    My_Eele1[1] = 0.0;
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
                    for (spin = 0; spin <= SpinP_switch; spin++) {
                        My_Eele1[spin] += CDM[spin][MA_AN][j][k][l] * nh[spin][MA_AN][j][k][l];
                    }
                }
            }
        }
    }

    /* MPI, My_Eele1 */
    MPI_Barrier(mpi_comm_level1);
    for (spin = 0; spin <= SpinP_switch; spin++) {
        MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }

    if (SpinP_switch == 0) {
        Eele1[1] = Eele1[0];
    }

    if (3 <= level_stdout && myid0 == Host_ID) {
        printf("Eele00=%15.12f Eele01=%15.12f\n", Eele0[0], Eele0[1]);
        printf("Eele10=%15.12f Eele11=%15.12f\n", Eele1[0], Eele1[1]);
    }

    /****************************************************
                          output
    ****************************************************/

    if (myid0 == Host_ID) {

        strcpy(file_EV, ".EV");
        fnjoint(filepath, filename, file_EV);

        if ((fp_EV = fopen(file_EV, "w")) != NULL) {

            setvbuf(fp_EV, buf, _IOFBF, fp_bsize);

            fprintf(fp_EV, "\n");
            fprintf(fp_EV, "***********************************************************\n");
            fprintf(fp_EV, "***********************************************************\n");
            fprintf(fp_EV, "           Eigenvalues (Hartree) of SCF KS-eq.           \n");
            fprintf(fp_EV, "***********************************************************\n");
            fprintf(fp_EV, "***********************************************************\n\n");
            fprintf(fp_EV, "   Chemical Potential (Hartree) = %18.14f\n", ChemP);
            fprintf(fp_EV, "   Number of States             = %18.14f\n", Num_State);
            fprintf(fp_EV, "   Eigenvalues\n");
            fprintf(fp_EV, "              Up-spin           Down-spin\n");

            for (kloop = 0; kloop < T_knum; kloop++) {

                k1 = T_KGrids1[kloop];
                k2 = T_KGrids2[kloop];
                k3 = T_KGrids3[kloop];

                if (0 < T_k_op[kloop]) {

                    fprintf(fp_EV, "\n");
                    fprintf(fp_EV, "   kloop=%i\n", kloop);
                    fprintf(fp_EV, "   k1=%10.5f k2=%10.5f k3=%10.5f\n\n", k1, k2, k3);
                    for (l = 1; l <= MaxN; l++) {
                        if (SpinP_switch == 0) {
                            fprintf(fp_EV, "%5d  %18.14f %18.14f\n", l, EIGEN[0][kloop][l], EIGEN[0][kloop][l]);
                        } else if (SpinP_switch == 1) {
                            fprintf(fp_EV, "%5d  %18.14f %18.14f\n", l, EIGEN[0][kloop][l], EIGEN[1][kloop][l]);
                        }
                    }
                }
            }
            fclose(fp_EV);
        } else {
            printf("Failure of saving the EV file.\n");
            fclose(fp_EV);
        }
    }

    // Destroy cublasmp & cusolverMp
    // if (all_knum == 1) {
    //     destroy_cusolvermp(&opts4);
    //     destroy_cublasmp(&opts2);
    // }

    /****************************************************
                         free arrays
    ****************************************************/

    if (all_knum != 1 && scf_eigen_lib_flag == CuSOLVER) {
#pragma acc exit data delete(Ss[0 : na_rows * na_cols], Hs[0 : na_rows * na_cols], Cs[0 : na_rows * na_cols],          \
                             ko[0 : n + 1])
    }

    if (all_knum == 1) {

        free(EVec_Rcv);
        free(index_Rcv_j);
        free(index_Rcv_i);
        free(EVec_Snd);
        free(index_Snd_j);
        free(index_Snd_i);

        free(Num_Rcv_EV);
        free(Num_Snd_EV);

        free(ie2);
        free(is2);
        free(ie1);
        free(is1);
    }

    free(SP_Atoms);
    free(SP_NZeros);
    free(My_NZeros);

    free(TmpEIGEN);

    for (i = 0; i < List_YOUSO[7]; i++) {
        free(ReEVec0[i]);
    }
    free(ReEVec0);

    for (i = 0; i < List_YOUSO[7]; i++) {
        free(ImEVec0[i]);
    }
    free(ImEVec0);

    for (i = 0; i < List_YOUSO[7]; i++) {
        free(ReEVec1[i]);
    }
    free(ReEVec1);

    for (i = 0; i < List_YOUSO[7]; i++) {
        free(ImEVec1[i]);
    }
    free(ImEVec1);

    /* for PrintMemory and allocation */
    firsttime = 0;

    /* for elapsed time */

    if (measure_time) {
        printf("myid0=%2d time1 =%9.4f\n", myid0, time1);
        fflush(stdout);
        printf("myid0=%2d time2 =%9.4f\n", myid0, time2);
        fflush(stdout);
        printf("myid0=%2d time3 =%9.4f\n", myid0, time3);
        fflush(stdout);
        printf("myid0=%2d time4 =%9.4f\n", myid0, time4);
        fflush(stdout);
        printf("myid0=%2d time5 =%9.4f\n", myid0, time5);
        fflush(stdout);
        printf("myid0=%2d time6 =%9.4f\n", myid0, time6);
        fflush(stdout);
        printf("myid0=%2d time7 =%9.4f\n", myid0, time7);
        fflush(stdout);
        printf("myid0=%2d time8 =%9.4f\n", myid0, time8);
        fflush(stdout);
        printf("myid0=%2d time81=%9.4f\n", myid0, time81);
        fflush(stdout);
        printf("myid0=%2d time82=%9.4f\n", myid0, time82);
        fflush(stdout);
        printf("myid0=%2d time83=%9.4f\n", myid0, time83);
        fflush(stdout);
        printf("myid0=%2d time84=%9.4f\n", myid0, time84);
        fflush(stdout);
        printf("myid0=%2d time85=%9.4f\n", myid0, time85);
        fflush(stdout);
        printf("myid0=%2d time9 =%9.4f\n", myid0, time9);
        fflush(stdout);
        printf("myid0=%2d time10=%9.4f\n", myid0, time10);
        fflush(stdout);
        printf("myid0=%2d time11=%9.4f\n", myid0, time11);
        fflush(stdout);
        printf("myid0=%2d time11A=%9.4f\n", myid0, time11A);
        fflush(stdout);
        printf("myid0=%2d time11B=%9.4f\n", myid0, time11B);
        fflush(stdout);
        printf("myid0=%2d time12=%9.4f\n", myid0, time12);
        fflush(stdout);
    }

    // double part1avescf1   = get_max_value(part1);
    // double part21avescf1  = get_max_value(part2_1);
    // double part22avescf1  = get_max_value(part2_2);
    // double part23avescf1  = get_max_value(part2_3);
    // double part24avescf1  = get_max_value(part2_4);
    // double part25avescf1  = get_max_value(part2_5);
    // double part3avescf1   = get_max_value(part3);
    // double part4avescf1   = get_max_value(part4);
    // double partmulavescf1 = get_max_value(partmul);

    // double part1sumscftotalm1   = get_max_value(part1sum);
    // double part21sumscftotalm1  = get_max_value(part2_1sum);
    // double part22sumscftotalm1  = get_max_value(part2_2sum);
    // double part23sumscftotalm1  = get_max_value(part2_3sum);
    // double part24sumscftotalm1  = get_max_value(part2_4sum);
    // double part25sumscftotalm1  = get_max_value(part2_5sum);
    // double part3sumscftotalm1   = get_max_value(part3sum);
    // double part4sumscftotalm1   = get_max_value(part4sum);
    // double partmulsumscftotalm1 = get_max_value(partmulsum);

    // if (measure_time && SCF_iter == 1 && myid0 == 0) {
    //     printf("myid0=%2d part1 =%9.4f\n", myid0, part1avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part21 =%9.4f\n", myid0, part21avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part22 =%9.4f\n", myid0, part22avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part23 =%9.4f\n", myid0, part23avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part24 =%9.4f\n", myid0, part24avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part25 =%9.4f\n", myid0, part25avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part3 =%9.4f\n", myid0, part3avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part4 =%9.4f\n", myid0, part4avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d partmul =%9.4f\n", myid0, partmulavescf1);
    //     fflush(stdout);
    //     double total = part1avescf1 + part21avescf1 + part22avescf1 + part23avescf1 + part24avescf1 + part25avescf1 +
    //                    part3avescf1 + part4avescf1;

    //     printf("myid0=%2d total =%9.4f\n", myid0, total);
    //     fflush(stdout);
    // }

    // if (measure_time && SCF_iter != 1 && myid0 == 0) {
    //     printf("myid0=%2d part1ave =%9.4f\n", myid0, part1sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part21ave =%9.4f\n", myid0, part21sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part22ave =%9.4f\n", myid0, part22sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part23ave =%9.4f\n", myid0, part23sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part24ave =%9.4f\n", myid0, part24sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part25ave =%9.4f\n", myid0, part25sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part3ave =%9.4f\n", myid0, part3sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part4ave =%9.4f\n", myid0, part4sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d partmul =%9.4f\n", myid0, partmulsumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);

    //     double total = part1sumscftotalm1 + part21sumscftotalm1 + part22sumscftotalm1 + part23sumscftotalm1 +
    //                    part24sumscftotalm1 + part25sumscftotalm1 + part3sumscftotalm1 + part4sumscftotalm1;

    //     printf("myid0=%2d total =%9.4f\n", myid0, total / (double)(SCF_iter - 1));
    //     fflush(stdout);
    // }

    MPI_Barrier(mpi_comm_level1);
    dtime(&TEtime);
    time0 = TEtime - TStime;
    return time0;
}

void Construct_Band_CsHs(int SCF_iter, int all_knum, int * order_GA, int * MP, double * S1, double * H1, double k1,
                         double k2, double k3, dcomplex * Cs, dcomplex * Hs, int n, int myid2)
{
    /* make S and H */

    if (SCF_iter == 1 || all_knum != 1) {
        if (scf_eigen_lib_flag == CuSOLVER && all_knum == 1 && myid2 == 0) {
            for (int i = 0; i < n * n; i++) {
                Cs[i].r = 0.0;
                Cs[i].i = 0.0;
            }
        } else if (scf_eigen_lib_flag == CuSOLVER && all_knum == 1) {
        } else {
            for (int i = 0; i < na_rows; i++) {
                for (int j = 0; j < na_cols; j++) {
                    Cs[j * na_rows + i].r = 0.0;
                    Cs[j * na_rows + i].i = 0.0;
                }
            }
        }
    }

    if (scf_eigen_lib_flag == CuSOLVER && all_knum == 1 && myid2 == 0) {
        for (int i = 0; i < n * n; i++) {
            Hs[i].r = 0.0;
            Hs[i].i = 0.0;
        }
    } else if (scf_eigen_lib_flag == CuSOLVER && all_knum == 1) {
    } else {
        for (int i = 0; i < na_rows; i++) {
            for (int j = 0; j < na_cols; j++) {
                Hs[j * na_rows + i].r = 0.0;
                Hs[j * na_rows + i].i = 0.0;
            }
        }
    }

    int k = 0;
    if (scf_eigen_lib_flag == CuSOLVER && all_knum == 1 && myid2 == 0) {
        if (SCF_iter == 1) {
            for (int AN = 1; AN <= atomnum; AN++) {
                int GA_AN = order_GA[AN];
                int wanA  = WhatSpecies[GA_AN];
                int tnoA  = Spe_Total_CNO[wanA];
                int Anum  = MP[GA_AN];

                for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                    int GB_AN = natn[GA_AN][LB_AN];
                    int Rn    = ncn[GA_AN][LB_AN];
                    int wanB  = WhatSpecies[GB_AN];
                    int tnoB  = Spe_Total_CNO[wanB];
                    int Bnum  = MP[GB_AN];

                    int    l1  = atv_ijk[Rn][1];
                    int    l2  = atv_ijk[Rn][2];
                    int    l3  = atv_ijk[Rn][3];
                    double kRn = k1 * (double)l1 + k2 * (double)l2 + k3 * (double)l3;
                    // phase_real = cos(2π kRn), phase_imag = sin(2π kRn)

                    double phase_real, phase_imag;
                    sincos(2.0 * PI * kRn, &phase_imag, &phase_real);

                    for (int i = 0; i < tnoA; i++) {
                        int ig = Anum + i;

                        int j_start = ig - Bnum;
                        if (j_start < 0) {
                            j_start = 0;
                        } else if (j_start > tnoB) {
                            j_start = tnoB;
                        }

                        k += j_start;

                        for (int j = j_start; j < tnoB; j++) {
                            int jg = Bnum + j;

                            double real_val = H1[k] * phase_real;
                            double imag_val = H1[k] * phase_imag;

                            int idx_jg_ig = (jg - 1) * n + (ig - 1);
                            Hs[idx_jg_ig].r += real_val;
                            Hs[idx_jg_ig].i += imag_val;

                            if (jg > ig) {
                                int idx_ig_jg = (ig - 1) * n + (jg - 1);
                                Hs[idx_ig_jg].r += real_val;
                                Hs[idx_ig_jg].i -= imag_val;
                            }

                            double real_val_S = S1[k] * phase_real;
                            double imag_val_S = S1[k] * phase_imag;

                            int idxC_jg_ig = (jg - 1) * n + (ig - 1);
                            Cs[idxC_jg_ig].r += real_val_S;
                            Cs[idxC_jg_ig].i += imag_val_S;

                            if (jg > ig) {
                                int idxC_ig_jg = (ig - 1) * n + (jg - 1);
                                Cs[idxC_ig_jg].r += real_val_S;
                                Cs[idxC_ig_jg].i -= imag_val_S;
                            }

                            k++;
                        }
                    }
                }
            }
        } else {
            for (int AN = 1; AN <= atomnum; AN++) {
                int GA_AN = order_GA[AN];
                int wanA  = WhatSpecies[GA_AN];
                int tnoA  = Spe_Total_CNO[wanA];
                int Anum  = MP[GA_AN];

                for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                    int GB_AN = natn[GA_AN][LB_AN];
                    int Rn    = ncn[GA_AN][LB_AN];
                    int wanB  = WhatSpecies[GB_AN];
                    int tnoB  = Spe_Total_CNO[wanB];
                    int Bnum  = MP[GB_AN];  // 列オフセット（1-based）

                    int    l1  = atv_ijk[Rn][1];
                    int    l2  = atv_ijk[Rn][2];
                    int    l3  = atv_ijk[Rn][3];
                    double kRn = k1 * (double)l1 + k2 * (double)l2 + k3 * (double)l3;
                    // phase_real = cos(2π kRn), phase_imag = sin(2π kRn)

                    double phase_real, phase_imag;
                    sincos(2.0 * PI * kRn, &phase_imag, &phase_real);

                    for (int i = 0; i < tnoA; i++) {
                        int ig = Anum + i;

                        int j_start = ig - Bnum;
                        if (j_start < 0) {
                            j_start = 0;
                        } else if (j_start > tnoB) {
                            j_start = tnoB;
                        }

                        k += j_start;

                        for (int j = j_start; j < tnoB; j++) {
                            int jg = Bnum + j;

                            double real_val = H1[k] * phase_real;
                            double imag_val = H1[k] * phase_imag;

                            int idx_jg_ig = (jg - 1) * n + (ig - 1);
                            Hs[idx_jg_ig].r += real_val;
                            Hs[idx_jg_ig].i += imag_val;

                            if (jg > ig) {
                                int idx_ig_jg = (ig - 1) * n + (jg - 1);
                                Hs[idx_ig_jg].r += real_val;
                                Hs[idx_ig_jg].i -= imag_val;
                            }
                            k++;
                        }
                    }
                }
            }
        }
    } else if (scf_eigen_lib_flag == CuSOLVER && all_knum == 1) {
    } else {
        for (int AN = 1; AN <= atomnum; AN++) {
            int GA_AN = order_GA[AN];
            int wanA  = WhatSpecies[GA_AN];
            int tnoA  = Spe_Total_CNO[wanA];
            int Anum  = MP[GA_AN];

            for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                int GB_AN = natn[GA_AN][LB_AN];
                int Rn    = ncn[GA_AN][LB_AN];
                int wanB  = WhatSpecies[GB_AN];
                int tnoB  = Spe_Total_CNO[wanB];
                int Bnum  = MP[GB_AN];

                int    l1  = atv_ijk[Rn][1];
                int    l2  = atv_ijk[Rn][2];
                int    l3  = atv_ijk[Rn][3];
                double kRn = k1 * (double)l1 + k2 * (double)l2 + k3 * (double)l3;

                double si = sin(2.0 * PI * kRn);
                double co = cos(2.0 * PI * kRn);

                for (int i = 0; i < tnoA; i++) {

                    int ig   = Anum + i;
                    int brow = (ig - 1) / nblk;
                    int prow = brow % np_rows;

                    for (int j = 0; j < tnoB; j++) {

                        int jg   = Bnum + j;
                        int bcol = (jg - 1) / nblk;
                        int pcol = bcol % np_cols;

                        if (my_prow == prow && my_pcol == pcol) {

                            int il = (brow / np_rows + 1) * nblk + 1;
                            int jl = (bcol / np_cols + 1) * nblk + 1;

                            if (((my_prow + np_rows) % np_rows) >= (brow % np_rows)) {
                                if (my_prow == prow) {
                                    il = il + (ig - 1) % nblk;
                                }
                                il = il - nblk;
                            }

                            if (((my_pcol + np_cols) % np_cols) >= (bcol % np_cols)) {
                                if (my_pcol == pcol) {
                                    jl = jl + (jg - 1) % nblk;
                                }
                                jl = jl - nblk;
                            }

                            if (SCF_iter == 1 || all_knum != 1) {
                                Cs[(jl - 1) * na_rows + il - 1].r += S1[k] * co;
                                Cs[(jl - 1) * na_rows + il - 1].i += S1[k] * si;
                            }

                            Hs[(jl - 1) * na_rows + il - 1].r += H1[k] * co;
                            Hs[(jl - 1) * na_rows + il - 1].i += H1[k] * si;
                        }

                        k++;
                    }
                }
            }
        }
    }
}

static double get_max_value(double local_value)
{
    double global_max;
    MPI_Allreduce(&local_value, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_max;
}