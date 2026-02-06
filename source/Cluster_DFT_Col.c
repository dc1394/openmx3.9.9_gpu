/**********************************************************************************
  Cluster_DFT_Col.c:

     Cluster_DFT_Col.c is a subroutine to perform cluster collinear calculations.

  Log of Cluster_DFT_Col.c:

     21/Feb./2019  Released by T. Ozaki

**********************************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include <assert.h>
#include <math.h>
#include <omp.h>
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

static double Lapack_LU_Dinverse(int n, double * A);
static double Calc_Oscillator_Strength(int n, int UMOmax, int Nocc[2], int * MP, int ** ind2ind, double ** Z0,
                                       double ** A, double ** Z, double ** Ax, double ** Ay, double ** Az, double Utot1,
                                       double Utot2, double XANES_Res[10]);

static void Calc_XANES_Col(int n, int MaxN, int myid1, double ** ko, double ***** nh, double **** CntOLP,
                           double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2], int myworld1,
                           int * NPROCS_ID1, int * Comm_World1, int * NPROCS_WD1, int * Comm_World_StartID1,
                           MPI_Comm * MPI_CommWD1, int * MP, int * is2, int * ie2, double * Ss, double * Cs,
                           double * Hs, double * CDM1, double * EDM1, double * PDM1, int size_H1, int * SP_NZeros,
                           int * SP_Atoms, double ** EVec1, double * Work1);

static void Save_LCAO_Col(int n, int MaxN, int myid1, int * is2, int * ie2, int * MP, double **** OLP0, double ** EVec1,
                          double ** ko, int * NPROCS_WD1, int * Comm_World_StartID1);

static void Save_DOS_Col(int n, int MaxN, int myid1, int * is2, int * ie2, int * MP, double **** OLP0, double ** EVec1,
                         double ** ko, int * NPROCS_WD1, int * Comm_World_StartID1);

static double Calc_DM_Cluster_collinear(int myid0, int numprocs0, int myid1, int numprocs1, int myworld1, int size_H1,
                                        int * is2, int * ie2, int * MP, int n, MPI_Comm * MPI_CommWD1,
                                        int * Comm_World_StartID1, double ***** CDM, double ***** EDM, double ** ko,
                                        double * DM1, double * EDM1, double * PDM1, double * Work1, double ** EVec1,
                                        int * SP_NZeros, int * SP_Atoms);

double Cluster_DFT_Col(char * mode, int SCF_iter, int SpinP_switch, double ** ko, double ***** nh, double **** CntOLP,
                       double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2], int myworld1,
                       int * NPROCS_ID1, int * Comm_World1, int * NPROCS_WD1, int * Comm_World_StartID1,
                       MPI_Comm * MPI_CommWD1, int * MP, int * is2, int * ie2, double * Ss, double * Cs, double * Hs,
                       double * CDM1, double * EDM1, double * PDM1, int size_H1, int * SP_NZeros, int * SP_Atoms,
                       double ** EVec1, double * Work1)
{
    static int firsttime = 1;
    int        i, j, l, n, n2, n1, i1, i1s, j1, k1, l1;
    int        wan, HOMO0, HOMO1;
    int        spin, po, num0, num1, ires;
    int        ct_AN, k, wanA, tnoA, wanB, tnoB;
    int        GA_AN, Anum, loopN, Gc_AN;
    int        MA_AN, LB_AN, GB_AN, Bnum, MaxN;
    int        wan1, mul, m, bcast_flag;
    int *      is1, *ie1;
    double     time0, lumos, av_num;
    double *   OneD_Mat1;
    double *** H;
    double     TZ, my_sum, sum, sumE, max_x = 60.0;
    double     sum0, sum1, sum2, sum3;
    double     My_Eele1[2], tmp1, tmp2;
    double     Num_State, x, FermiF, Dnum, Dnum2;
    double     FermiF2, x2, diffF;
    double     dum, ChemP_MAX, ChemP_MIN, spin_degeneracy;
    double     TStime, TEtime;
    double     FermiEps = 1.0e-13;
    double     EV_cut0;
    double     res;
    int        numprocs0, myid0;
    int        numprocs1, myid1;
    int        ID, p, world_Snd, world_Rcv;
    char *     Name_Angular[Supported_MaxL + 1][2 * (Supported_MaxL + 1) + 1];
    char *     Name_Multiple[20];
    double     OLP_eigen_cut    = Threshold_OLP_Eigen;
    char       file_EV[YOUSO10] = ".EV";
    char       buf[fp_bsize]; /* setvbuf */
    FILE *     fp_EV;
    double     stime, etime;
    double     time1, time2, time3, time4, time5, time6, time7;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    MPI_Comm mpi_comm_rows, mpi_comm_cols;
    int      mpi_comm_rows_int, mpi_comm_cols_int;
    int      info, ig, jg, il, jl, prow, pcol, brow, bcol;
    int      ZERO = 0, ONE = 1;
    double   alpha = 1.0;
    double   beta  = 0.0;
    int      LOCr, LOCc, node, irow, icol;
    double   C_spin_i1, mC_spin_i1;
    int      sp;

    int         ID0, IDS, IDR, Max_Num_Snd_EV, Max_Num_Rcv_EV;
    int *       Num_Snd_EV, *Num_Rcv_EV;
    int *       index_Snd_i, *index_Snd_j, *index_Rcv_i, *index_Rcv_j;
    double *    EVec_Snd, *EVec_Rcv;
    MPI_Status  stat;
    MPI_Request request;

    /* for time */
    MPI_Barrier(mpi_comm_level1);
    dtime(&TStime);

    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs0);
    MPI_Comm_rank(mpi_comm_level1, &myid0);

    MPI_Comm_size(MPI_CommWD1[myworld1], &numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1], &myid1);

    if (scf_eigen_lib_flag == CuSOLVER) {
        // CUDA
        set_cuda_default_device_from_local_rank(MPI_CommWD1[myworld1]);

        // OpenACC
        set_openacc_nvidia_device_from_local_rank(MPI_CommWD1[myworld1]);
    }

    /****************************************************
             calculation of the array size
  ****************************************************/

    n = 0;
    for (i = 1; i <= atomnum; i++) {
        wanA = WhatSpecies[i];
        n    = n + Spe_Total_CNO[wanA];
    }
    n2 = n + 2;

    /****************************************************
   Allocation

   double  H[List_YOUSO[23]][n2][n2]  
  ****************************************************/

    is1 = (int *)malloc(sizeof(int) * numprocs1);
    ie1 = (int *)malloc(sizeof(int) * numprocs1);

    Num_Snd_EV = (int *)malloc(sizeof(int) * numprocs1);
    Num_Rcv_EV = (int *)malloc(sizeof(int) * numprocs1);

    if (measure_time) {
        time1 = 0.0;
        time2 = 0.0;
        time3 = 0.0;
        time4 = 0.0;
        time5 = 0.0;
        time6 = 0.0;
        time7 = 0.0;
    }

    if (SpinP_switch == 0)
        spin_degeneracy = 2.0;
    else if (SpinP_switch == 1)
        spin_degeneracy = 1.0;

    /****************************************************
                   total core charge
  ****************************************************/

    TZ = 0.0;
    for (i = 1; i <= atomnum; i++) {
        wan = WhatSpecies[i];
        TZ  = TZ + Spe_Core_Charge[wan];
    }

    /****************************************************
         find the numbers of partions for MPI
  ****************************************************/

    if (numprocs1 <= n) {

        av_num = (double)n / (double)numprocs1;

        for (ID = 0; ID < numprocs1; ID++) {
            is1[ID] = (int)(av_num * (double)ID) + 1;
            ie1[ID] = (int)(av_num * (double)(ID + 1));
        }

        is1[0]             = 1;
        ie1[numprocs1 - 1] = n;

    }

    else {

        for (ID = 0; ID < n; ID++) {
            is1[ID] = ID + 1;
            ie1[ID] = ID + 1;
        }
        for (ID = n; ID < numprocs1; ID++) {
            is1[ID] = 1;
            ie1[ID] = -2;
        }
    }

    /****************************************************
       1. diagonalize the overlap matrix     
       2. search negative eigenvalues
  ****************************************************/

    if (scf_eigen_lib_flag == CuSOLVER) {
        MPI_Barrier(mpi_comm_level1);
        double starttimesh = 0.0, endtimesh = 0.0;
        if (measure_time) {
            dtime(&starttimesh);
        }
        if (SCF_iter == 1) {
            Overlap_Cluster_Ss(CntOLP, Ss, MP, MPI_CommWD1, myworld1, n);
        }

        if (measure_time) {
            dtime(&endtimesh);

            if (myid0 == 0) {
                printf("make S: %.7f (sec)\n", endtimesh - starttimesh);
            }
        }

        double * hm;

        if (myid1 == 0) {
            hm = (double *)malloc(sizeof(double) * n * n);
        } else {
            hm = (double *)malloc(sizeof(double) * 1);
        }

        if (measure_time) {
            dtime(&starttimesh);
        }
        if (SpinP_switch == 1 && numprocs0 == 1) {
            Hamiltonian_Cluster_Hs(nh[0], hm, MP, 0, MPI_CommWD1, 0, n);
        } else {
            for (spin = 0; spin <= SpinP_switch; spin++) {
                Hamiltonian_Cluster_Hs(nh[spin], hm, MP, spin, MPI_CommWD1, myworld1, n);
            }
        }
        if (measure_time) {
            dtime(&endtimesh);

            if (myid0 == 0) {
                printf("make H: %.7f (sec)\n", endtimesh - starttimesh);
            }
        }

        bool second_time = false;
    diagonalize_gpu:

        if (myid1 == 0) {
#pragma acc enter data copyin(Ss[0 : n * n])
            if (SCF_iter == 1 && !second_time) {
#pragma acc enter data create(ko[0 : SpinP_switch + 1][0 : n + 1])
                Eigen_cusolver_x_openacc2(Ss, ko[0], n, n);

// #pragma acc update self(Ss[0 : n * n])
//             for (int i = 0; i < n * n; i++) {
//                 printf("%.7f ", Ss[i]);
//             }
//             puts("");


                /* minus eigenvalues to 1.0e-10 */
#pragma acc kernels
#pragma acc loop independent
                for (l = 1; l <= n; l++) {
                    if (ko[0][l] < 0.0)
                        ko[0][l] = 1.0e-10;
                }

                /* calculate S*1/sqrt(ko) */
#pragma acc kernels
#pragma acc loop independent
                for (l = 1; l <= n; l++) {
                    ko[0][l] = 1.0 / sqrt(ko[0][l]);
                }

#pragma acc kernels
#pragma acc loop independent
                for (i = 0; i < n; i++) {
#pragma acc loop independent
                    for (j = 0; j < n; j++) {
                        //jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + my_pcol) % np_cols) * nblk + 1;
                        Ss[n * j + i] *= ko[0][j + 1];
                    }
                }

#pragma acc update     self(Ss[0 : n * n])
            } else if (SCF_iter != 1 && !second_time) {
    #pragma acc enter data create(ko[0 : SpinP_switch + 1][0 : n + 1])
            } else if (second_time) {
    #pragma acc enter data copyin(ko[0 : SpinP_switch + 1][0 : n + 1])
            }

            /* find the maximum states in solved eigenvalues */

            if (SCF_iter == 1) {
                MaxN = n;
            } else {

                if (strcasecmp(mode, "scf") == 0)
                    lumos = (double)n * 0.100;
                else if (strcasecmp(mode, "dos") == 0)
                    lumos = (double)n * 0.200;
                else if (strcasecmp(mode, "lcaoout") == 0)
                    lumos = (double)n * 0.200;
                else if (strcasecmp(mode, "xanes") == 0)
                    lumos = (double)n * 0.200;
                else if (strcasecmp(mode, "diag") == 0)
                    lumos = (double)n * 0.200;

                if (lumos < 400.0)
                    lumos = 400.0;
                MaxN = (TZ - system_charge) / 2 + (int)lumos;
                if (n < MaxN)
                    MaxN = n;

                if (cal_partial_charge)
                    MaxN = n;
            }

            /* initialize ko */

#pragma acc kernels
            if (!second_time) {
#pragma acc loop independent
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (i1 = 1; i1 <= n; i1++) {
                        ko[spin][i1] = 10000.0;
                    }
                }
            }

#pragma acc enter data copyin(hm[0 : n * n])
#pragma acc enter data create(Cs[0 : n * n])

            /* spin=myworld1 */

            if (SpinP_switch != 1 || numprocs0 != 1 || !second_time) {
                spin = myworld1;
            } else if (second_time) {
                spin = SpinP_switch;
            }

            if (measure_time)
                dtime(&stime);

            /* pdgemm */

            /* H * U * 1.0/sqrt(ko[l]) */
#pragma acc kernels
#pragma acc loop independent
            for (i = 0; i < n * n; i++) {
                Cs[i] = 0.0;
            }

            my_cublasDgemm_openacc(CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, hm, Ss, Cs);

            /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

#pragma acc kernels
#pragma acc loop independent
            for (int i = 0; i < n * n; i++) {
                hm[i] = 0.0;
            }

            my_cublasDgemm_openacc(CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, Ss, Cs, hm);

#pragma acc kernels
#pragma acc loop independent
            for (int i = 0; i < n; i++) {
#pragma acc loop independent
                for (int j = 0; j < n; j++) {
                    Cs[j * n + i] = hm[j * n + i];
                }
            }

#pragma acc exit data delete(hm[0 : n * n], Ss[0 : n * n])

            Eigen_cusolver_x_openacc2(Cs, ko[spin], n, MaxN);

#pragma acc enter data copyin(Ss[0 : n * n])
#pragma acc enter data create(hm[0 : n * n])

#pragma acc kernels
#pragma acc loop independent
            for (i = 0; i < n * n; i++) {
                hm[i] = 0.0;
            }

            my_cublasDgemm_openacc(CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, Cs, Ss, hm);

#pragma acc update    self(hm[0 : n * n], ko[0 : SpinP_switch + 1][0 : n + 1])
#pragma acc exit data delete(hm[0 : n * n], Ss[0 : n * n], Cs[0 : n * n], ko[0 : SpinP_switch + 1][0 : n + 1])
        }

        int desc_global[9];
        int i_zero = 0;
        int i_one  = 1;

        descinit_(desc_global, &n, &n, &n, &n, &i_zero, &i_zero, &ictxt1, &n, &info);
        pdgemr2d_(&n, &n, hm, &i_one, &i_one, desc_global, Hs, &i_one, &i_one, descH, &ictxt1);

        MPI_Bcast(&MaxN, 1, MPI_INT, 0, MPI_CommWD1[myworld1]);

        if (SpinP_switch != 1 || numprocs0 != 1 || !second_time) {
            spin = myworld1;
        } else if (second_time) {
            spin = SpinP_switch;
        }

        if (numprocs1 <= MaxN) {
            av_num = (double)MaxN / (double)numprocs1;
            for (ID = 0; ID < numprocs1; ID++) {
                is2[ID] = (int)(av_num * (double)ID) + 1;
                ie2[ID] = (int)(av_num * (double)(ID + 1));
            }

            is2[0]             = 1;
            ie2[numprocs1 - 1] = MaxN;
        } else {
            for (ID = 0; ID < MaxN; ID++) {
                is2[ID] = ID + 1;
                ie2[ID] = ID + 1;
            }

            for (ID = MaxN; ID < numprocs1; ID++) {
                is2[ID] = 1;
                ie2[ID] = 0;
            }
        }

        /* making data structure of MPI communicaition for eigenvectors */

        for (ID = 0; ID < numprocs1; ID++) {
            Num_Snd_EV[ID] = 0;
            Num_Rcv_EV[ID] = 0;
        }

        for (i = 0; i < na_rows; i++) {

            ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + my_prow) % np_rows) * nblk + 1;

            po = 0;
            for (ID = 0; ID < numprocs1; ID++) {
                if (is2[ID] <= ig && ig <= ie2[ID]) {
                    po  = 1;
                    ID0 = ID;
                    break;
                }
            }

            if (po == 1)
                Num_Snd_EV[ID0] += na_cols;
        }

        for (ID = 0; ID < numprocs1; ID++) {
            IDS = (myid1 + ID) % numprocs1;
            IDR = (myid1 - ID + numprocs1) % numprocs1;
            if (ID != 0) {
                MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
                MPI_Recv(&Num_Rcv_EV[IDR], 1, MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
                MPI_Wait(&request, &stat);
            } else {
                Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
            }
        }

        Max_Num_Snd_EV = 0;
        Max_Num_Rcv_EV = 0;
        for (ID = 0; ID < numprocs1; ID++) {
            if (Max_Num_Snd_EV < Num_Snd_EV[ID])
                Max_Num_Snd_EV = Num_Snd_EV[ID];
            if (Max_Num_Rcv_EV < Num_Rcv_EV[ID])
                Max_Num_Rcv_EV = Num_Rcv_EV[ID];
        }

        Max_Num_Snd_EV++;
        Max_Num_Rcv_EV++;

        index_Snd_i = (int *)malloc(sizeof(int) * Max_Num_Snd_EV);
        index_Snd_j = (int *)malloc(sizeof(int) * Max_Num_Snd_EV);
        EVec_Snd    = (double *)malloc(sizeof(double) * Max_Num_Snd_EV);
        index_Rcv_i = (int *)malloc(sizeof(int) * Max_Num_Rcv_EV);
        index_Rcv_j = (int *)malloc(sizeof(int) * Max_Num_Rcv_EV);
        EVec_Rcv    = (double *)malloc(sizeof(double) * Max_Num_Rcv_EV);

        /* MPI communications of Hs */

        for (ID = 0; ID < numprocs1; ID++) {

            IDS = (myid1 + ID) % numprocs1;
            IDR = (myid1 - ID + numprocs1) % numprocs1;

            k = 0;
            for (i = 0; i < na_rows; i++) {
                ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + my_prow) % np_rows) * nblk + 1;
                if (is2[IDS] <= ig && ig <= ie2[IDS]) {

                    for (j = 0; j < na_cols; j++) {
                        jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + my_pcol) % np_cols) * nblk + 1;

                        index_Snd_i[k] = ig;
                        index_Snd_j[k] = jg;
                        EVec_Snd[k]    = Hs[j * na_rows + i];
                        k++;
                    }
                }
            }

            if (ID != 0) {

                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
                }
                if (Num_Rcv_EV[IDR] != 0) {
                    MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
                }
                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Wait(&request, &stat);
                }

                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
                }
                if (Num_Rcv_EV[IDR] != 0) {
                    MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
                }
                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Wait(&request, &stat);
                }

                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Isend(EVec_Snd, Num_Snd_EV[IDS], MPI_DOUBLE, IDS, 999, MPI_CommWD1[myworld1], &request);
                }
                if (Num_Rcv_EV[IDR] != 0) {
                    MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR], MPI_DOUBLE, IDR, 999, MPI_CommWD1[myworld1], &stat);
                }
                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Wait(&request, &stat);
                }
            } else {
                for (k = 0; k < Num_Snd_EV[IDS]; k++) {
                    index_Rcv_i[k] = index_Snd_i[k];
                    index_Rcv_j[k] = index_Snd_j[k];
                    EVec_Rcv[k]    = EVec_Snd[k];
                }
            }

            for (k = 0; k < Num_Rcv_EV[IDR]; k++) {
                ig             = index_Rcv_i[k];
                jg             = index_Rcv_j[k];
                m              = (jg - 1) * (ie2[myid1] - is2[myid1] + 1) + ig - is2[myid1];
                EVec1[spin][m] = EVec_Rcv[k];
            }
        }

        if (SpinP_switch == 1 && numprocs0 == 1 && spin == 0) {
            spin++;
            second_time = true;
            Hamiltonian_Cluster_Hs(nh[spin], hm, MP, spin, MPI_CommWD1, spin, n);

            goto diagonalize_gpu;
        }

        free(hm);
    } else {
        MPI_Barrier(mpi_comm_level1);

        if (SCF_iter == 1) {
            Overlap_Cluster_Ss(CntOLP, Cs, MP, MPI_CommWD1, myworld1, n);
        }

        if (SpinP_switch == 1 && numprocs0 == 1) {
            Hamiltonian_Cluster_Hs(nh[0], Hs, MP, 0, MPI_CommWD1, 0, n);
        } else {
            for (spin = 0; spin <= SpinP_switch; spin++) {
                Hamiltonian_Cluster_Hs(nh[spin], Hs, MP, spin, MPI_CommWD1, myworld1, n);
            }
        }

        if (SCF_iter == 1) {

            if (measure_time)
                dtime(&stime);

            if (scf_eigen_lib_flag == ELPA1 || scf_eigen_lib_flag == ELPA2) {
                MPI_Comm_split(MPI_CommWD1[myworld1], my_pcol, my_prow, &mpi_comm_rows);
                MPI_Comm_split(MPI_CommWD1[myworld1], my_prow, my_pcol, &mpi_comm_cols);

                mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
                mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
            }

            if (scf_eigen_lib_flag == 1) {

                F77_NAME(solve_evp_real, SOLVE_EVP_REAL)(&n, &n, Cs, &na_rows, &ko[0][1], Ss, &na_rows, &nblk,
                                                         &mpi_comm_rows_int, &mpi_comm_cols_int);
            }

            else if (scf_eigen_lib_flag == 2) {

#ifndef kcomp

                int mpiworld;
                mpiworld = MPI_Comm_c2f(MPI_CommWD1[myworld1]);

                F77_NAME(elpa_solve_evp_real_2stage_double_impl, ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(
                    &n, &n, Cs, &na_rows, &ko[0][1], Ss, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int,
                    &mpi_comm_cols_int, &mpiworld);

#endif
            }

            if (scf_eigen_lib_flag == ELPA1 || scf_eigen_lib_flag == ELPA2) {
                MPI_Comm_free(&mpi_comm_rows);
                MPI_Comm_free(&mpi_comm_cols);
            }

            /* print to the standard output */

            if (2 <= level_stdout && myid0 == Host_ID) {
                for (l = 1; l <= n; l++) {
                    printf("  Eigenvalues of OLP  %2d  %18.15f\n", l, ko[0][l]);
                    fflush(stdout);
                }
            }

            /* minus eigenvalues to 1.0e-10 */

            for (l = 1; l <= n; l++) {
                if (ko[0][l] < 0.0)
                    ko[0][l] = 1.0e-10;
            }

            /* calculate S*1/sqrt(ko) */

            for (l = 1; l <= n; l++) {
                ko[0][l] = 1.0 / sqrt(ko[0][l]);
            }

            for (i = 0; i < na_rows; i++) {
                for (j = 0; j < na_cols; j++) {
                    jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + my_pcol) % np_cols) * nblk + 1;
                    Ss[j * na_rows + i] = Ss[j * na_rows + i] * ko[0][jg];
                }
            }

            if (measure_time) {
                dtime(&etime);
                time1 += etime - stime;
            }
        }

        /****************************************************
    calculations of eigenvalues for up and down spins

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

        MPI_Barrier(mpi_comm_level1);

        /* find the maximum states in solved eigenvalues */

        if (SCF_iter == 1) {
            MaxN = n;
        } else {

            if (strcasecmp(mode, "scf") == 0)
                lumos = (double)n * 0.100;
            else if (strcasecmp(mode, "dos") == 0)
                lumos = (double)n * 0.200;
            else if (strcasecmp(mode, "lcaoout") == 0)
                lumos = (double)n * 0.200;
            else if (strcasecmp(mode, "xanes") == 0)
                lumos = (double)n * 0.200;
            else if (strcasecmp(mode, "diag") == 0)
                lumos = (double)n * 0.200;

            if (lumos < 400.0)
                lumos = 400.0;
            MaxN = (TZ - system_charge) / 2 + (int)lumos;
            if (n < MaxN)
                MaxN = n;

            if (cal_partial_charge)
                MaxN = n;
        }

        if (numprocs1 <= MaxN) {

            av_num = (double)MaxN / (double)numprocs1;
            for (ID = 0; ID < numprocs1; ID++) {
                is2[ID] = (int)(av_num * (double)ID) + 1;
                ie2[ID] = (int)(av_num * (double)(ID + 1));
            }

            is2[0]             = 1;
            ie2[numprocs1 - 1] = MaxN;
        }

        else {

            for (ID = 0; ID < MaxN; ID++) {
                is2[ID] = ID + 1;
                ie2[ID] = ID + 1;
            }

            for (ID = MaxN; ID < numprocs1; ID++) {
                is2[ID] = 1;
                ie2[ID] = 0;
            }
        }

        /* making data structure of MPI communicaition for eigenvectors */

        for (ID = 0; ID < numprocs1; ID++) {
            Num_Snd_EV[ID] = 0;
            Num_Rcv_EV[ID] = 0;
        }

        for (i = 0; i < na_rows; i++) {

            ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + my_prow) % np_rows) * nblk + 1;

            po = 0;
            for (ID = 0; ID < numprocs1; ID++) {
                if (is2[ID] <= ig && ig <= ie2[ID]) {
                    po  = 1;
                    ID0 = ID;
                    break;
                }
            }

            if (po == 1)
                Num_Snd_EV[ID0] += na_cols;
        }

        for (ID = 0; ID < numprocs1; ID++) {
            IDS = (myid1 + ID) % numprocs1;
            IDR = (myid1 - ID + numprocs1) % numprocs1;
            if (ID != 0) {
                MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
                MPI_Recv(&Num_Rcv_EV[IDR], 1, MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
                MPI_Wait(&request, &stat);
            } else {
                Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
            }
        }

        Max_Num_Snd_EV = 0;
        Max_Num_Rcv_EV = 0;
        for (ID = 0; ID < numprocs1; ID++) {
            if (Max_Num_Snd_EV < Num_Snd_EV[ID])
                Max_Num_Snd_EV = Num_Snd_EV[ID];
            if (Max_Num_Rcv_EV < Num_Rcv_EV[ID])
                Max_Num_Rcv_EV = Num_Rcv_EV[ID];
        }

        Max_Num_Snd_EV++;
        Max_Num_Rcv_EV++;

        index_Snd_i = (int *)malloc(sizeof(int) * Max_Num_Snd_EV);
        index_Snd_j = (int *)malloc(sizeof(int) * Max_Num_Snd_EV);
        EVec_Snd    = (double *)malloc(sizeof(double) * Max_Num_Snd_EV);
        index_Rcv_i = (int *)malloc(sizeof(int) * Max_Num_Rcv_EV);
        index_Rcv_j = (int *)malloc(sizeof(int) * Max_Num_Rcv_EV);
        EVec_Rcv    = (double *)malloc(sizeof(double) * Max_Num_Rcv_EV);

        /* for PrintMemory */

        if (firsttime && memoryusage_fileout) {
            PrintMemory("Cluster_DFT_Col: is1", sizeof(int) * numprocs1, NULL);
            PrintMemory("Cluster_DFT_Col: ie1", sizeof(int) * numprocs1, NULL);
            PrintMemory("Cluster_DFT_Col: Num_Snd_EV", sizeof(int) * numprocs1, NULL);
            PrintMemory("Cluster_DFT_Col: Num_Snd_EV", sizeof(int) * numprocs1, NULL);
            PrintMemory("Cluster_DFT_Col: index_Snd_i", sizeof(int) * Max_Num_Snd_EV, NULL);
            PrintMemory("Cluster_DFT_Col: index_Snd_j", sizeof(int) * Max_Num_Snd_EV, NULL);
            PrintMemory("Cluster_DFT_Col: index_Rcv_i", sizeof(int) * Max_Num_Rcv_EV, NULL);
            PrintMemory("Cluster_DFT_Col: index_Rcv_j", sizeof(int) * Max_Num_Rcv_EV, NULL);
            PrintMemory("Cluster_DFT_Col: EVec_Snd", sizeof(double) * Max_Num_Snd_EV, NULL);
            PrintMemory("Cluster_DFT_Col: EVec_Rcv", sizeof(double) * Max_Num_Rcv_EV, NULL);
        }
        firsttime = 0;

        /* initialize ko */
        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (i1 = 1; i1 <= n; i1++) {
                ko[spin][i1] = 10000.0;
            }
        }

        /* spin=myworld1 */

        spin = myworld1;

    diagonalize:

        if (measure_time)
            dtime(&stime);

        /* pdgemm */

        /* H * U * 1.0/sqrt(ko[l]) */

        for (i = 0; i < na_rows_max * na_cols_max; i++) {
            Cs[i] = 0.0;
        }

        Cblacs_barrier(ictxt1, "A");
        F77_NAME(pdgemm, PDGEMM)("N", "N", &n, &n, &n, &alpha, Hs, &ONE, &ONE, descH, Ss, &ONE, &ONE, descS, &beta, Cs,
                                 &ONE, &ONE, descC);

        /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

        for (i = 0; i < na_rows * na_cols; i++) {
            Hs[i] = 0.0;
        }

        Cblacs_barrier(ictxt1, "C");
        F77_NAME(pdgemm, PDGEMM)("T", "N", &n, &n, &n, &alpha, Ss, &ONE, &ONE, descS, Cs, &ONE, &ONE, descC, &beta, Hs,
                                 &ONE, &ONE, descH);

        if (measure_time) {
            dtime(&etime);
            time2 += etime - stime;
        }

        /* The output C matrix is distributed by column. */

        if (measure_time)
            dtime(&stime);

        if (scf_eigen_lib_flag == ELPA1 || scf_eigen_lib_flag == ELPA2) {
            MPI_Comm_split(MPI_CommWD1[myworld1], my_pcol, my_prow, &mpi_comm_rows);
            MPI_Comm_split(MPI_CommWD1[myworld1], my_prow, my_pcol, &mpi_comm_cols);

            mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
            mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
        }

        if (scf_eigen_lib_flag == 1) {
            F77_NAME(solve_evp_real, SOLVE_EVP_REAL)(&n, &MaxN, Hs, &na_rows, &ko[spin][1], Cs, &na_rows, &nblk,
                                                     &mpi_comm_rows_int, &mpi_comm_cols_int);
        } else if (scf_eigen_lib_flag == 2) {

#ifndef kcomp
            int mpiworld;
            mpiworld = MPI_Comm_c2f(MPI_CommWD1[myworld1]);

            F77_NAME(elpa_solve_evp_real_2stage_double_impl, ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(
                &n, &MaxN, Hs, &na_rows, &ko[spin][1], Cs, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int,
                &mpi_comm_cols_int, &mpiworld);
#endif
        }

        if (scf_eigen_lib_flag == ELPA1 || scf_eigen_lib_flag == ELPA2) {
            MPI_Comm_free(&mpi_comm_rows);
            MPI_Comm_free(&mpi_comm_cols);
        }

        if (measure_time) {
            dtime(&etime);
            time3 += etime - stime;
        }

        /****************************************************
      transformation to the original eigenvectors.
                       NOTE 244P
  ****************************************************/

        if (measure_time)
            dtime(&stime);

        for (i = 0; i < na_rows * na_cols; i++) {
            Hs[i] = 0.0;
        }

        Cblacs_barrier(ictxt1, "A");
        F77_NAME(pdgemm, PDGEMM)("T", "T", &n, &n, &n, &alpha, Cs, &ONE, &ONE, descC, Ss, &ONE, &ONE, descS, &beta, Hs,
                                 &ONE, &ONE, descH);

        /* MPI communications of Hs */

        for (ID = 0; ID < numprocs1; ID++) {

            IDS = (myid1 + ID) % numprocs1;
            IDR = (myid1 - ID + numprocs1) % numprocs1;

            k = 0;
            for (i = 0; i < na_rows; i++) {
                ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + my_prow) % np_rows) * nblk + 1;
                if (is2[IDS] <= ig && ig <= ie2[IDS]) {

                    for (j = 0; j < na_cols; j++) {
                        jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + my_pcol) % np_cols) * nblk + 1;

                        index_Snd_i[k] = ig;
                        index_Snd_j[k] = jg;
                        EVec_Snd[k]    = Hs[j * na_rows + i];
                        k++;
                    }
                }
            }

            if (ID != 0) {

                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
                }
                if (Num_Rcv_EV[IDR] != 0) {
                    MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
                }
                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Wait(&request, &stat);
                }

                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
                }
                if (Num_Rcv_EV[IDR] != 0) {
                    MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
                }
                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Wait(&request, &stat);
                }

                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Isend(EVec_Snd, Num_Snd_EV[IDS], MPI_DOUBLE, IDS, 999, MPI_CommWD1[myworld1], &request);
                }
                if (Num_Rcv_EV[IDR] != 0) {
                    MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR], MPI_DOUBLE, IDR, 999, MPI_CommWD1[myworld1], &stat);
                }
                if (Num_Snd_EV[IDS] != 0) {
                    MPI_Wait(&request, &stat);
                }
            } else {
                for (k = 0; k < Num_Snd_EV[IDS]; k++) {
                    index_Rcv_i[k] = index_Snd_i[k];
                    index_Rcv_j[k] = index_Snd_j[k];
                    EVec_Rcv[k]    = EVec_Snd[k];
                }
            }

            for (k = 0; k < Num_Rcv_EV[IDR]; k++) {
                ig             = index_Rcv_i[k];
                jg             = index_Rcv_j[k];
                m              = (jg - 1) * (ie2[myid1] - is2[myid1] + 1) + ig - is2[myid1];
                EVec1[spin][m] = EVec_Rcv[k];
            }
        }

        if (measure_time) {
            dtime(&etime);
            time4 += etime - stime;
        }

        if (SpinP_switch == 1 && numprocs0 == 1 && spin == 0) {
            spin++;
            Hamiltonian_Cluster_Hs(nh[spin], Hs, MP, spin, MPI_CommWD1, spin, n);
            goto diagonalize;
        }
    }

    /*********************************************** 
    MPI: ko
  ***********************************************/

    if (measure_time)
        dtime(&stime);

    for (sp = 0; sp <= SpinP_switch; sp++) {
        MPI_Bcast(&ko[sp][1], MaxN, MPI_DOUBLE, Comm_World_StartID1[sp], mpi_comm_level1);
    }

    if (strcasecmp(mode, "scf") == 0) {

        if (2 <= level_stdout) {
            for (i1 = 1; i1 <= MaxN; i1++) {
                if (SpinP_switch == 0)
                    printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n", i1, ko[0][i1], ko[0][i1]);
                else
                    printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n", i1, ko[0][i1], ko[1][i1]);
            }
        }

        /* for XANES */
        if (xanes_calc == 1) {

            /****************************************************
              searching of chemical potential
      ****************************************************/

            for (spin = 0; spin <= SpinP_switch; spin++) {

                po    = 0;
                loopN = 0;

                ChemP_MAX = 30.0;
                ChemP_MIN = -30.0;

                do {

                    ChemP     = 0.50 * (ChemP_MAX + ChemP_MIN);
                    Num_State = 0.0;

                    for (i1 = 1; i1 <= MaxN; i1++) {
                        x = (ko[spin][i1] - ChemP) * Beta;
                        if (x <= -max_x)
                            x = -max_x;
                        if (max_x <= x)
                            x = max_x;
                        FermiF = 1.0 / (1.0 + exp(x));
                        Num_State += FermiF;
                    }

                    Dnum = HOMO_XANES[spin] - Num_State;
                    if (0.0 <= Dnum)
                        ChemP_MIN = ChemP;
                    else
                        ChemP_MAX = ChemP;
                    if (fabs(Dnum) < 1.0e-14)
                        po = 1;

                    if (myid1 == Host_ID && 2 <= level_stdout) {
                        printf("spin=%2d ChemP=%15.12f HOMO_XANES=%2d Num_state=%15.12f\n", spin, ChemP,
                               HOMO_XANES[spin], Num_State);
                    }

                    loopN++;

                } while (po == 0 && loopN < 1000);

                ChemP_XANES[spin]  = ChemP;
                Cluster_HOMO[spin] = HOMO_XANES[spin];

            } /* spin */

            /* set ChemP */

            ChemP = 0.5 * (ChemP_XANES[0] + ChemP_XANES[1]);

        } /* end of if (xanes_calc==1) */

        /* start of else for if (xanes_calc==1) */

        else {

            /****************************************************
              searching of chemical potential
      ****************************************************/

            /* first, find ChemP at five times large temperatue */

            po    = 0;
            loopN = 0;

            ChemP_MAX = 30.0;
            ChemP_MIN = -30.0;

            do {

                ChemP     = 0.50 * (ChemP_MAX + ChemP_MIN);
                Num_State = 0.0;

                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (i1 = 1; i1 <= MaxN; i1++) {
                        x = (ko[spin][i1] - ChemP) * Beta * 0.2;
                        if (x <= -max_x)
                            x = -max_x;
                        if (max_x <= x)
                            x = max_x;
                        FermiF = 1.0 / (1.0 + exp(x));

                        Num_State = Num_State + spin_degeneracy * FermiF;
                        if (0.5 < FermiF)
                            Cluster_HOMO[spin] = i1;
                    }
                }

                Dnum = (TZ - Num_State) - system_charge;
                if (0.0 <= Dnum)
                    ChemP_MIN = ChemP;
                else
                    ChemP_MAX = ChemP;
                if (fabs(Dnum) < 1.0e-14)
                    po = 1;

                if (myid1 == Host_ID && 2 <= level_stdout) {
                    printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n", ChemP, TZ, Num_State);
                }

                loopN++;

            } while (po == 0 && loopN < 1000);

            /* second, find ChemP at the temperatue, starting from the previously found ChemP. */

            po    = 0;
            loopN = 0;

            ChemP_MAX = 30.0;
            ChemP_MIN = -30.0;

            do {

                if (loopN != 0) {
                    ChemP = 0.50 * (ChemP_MAX + ChemP_MIN);
                }

                Num_State = 0.0;

                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (i1 = 1; i1 <= MaxN; i1++) {
                        x = (ko[spin][i1] - ChemP) * Beta;
                        if (x <= -max_x)
                            x = -max_x;
                        if (max_x <= x)
                            x = max_x;
                        FermiF = 1.0 / (1.0 + exp(x));

                        Num_State = Num_State + spin_degeneracy * FermiF;
                        if (0.5 < FermiF)
                            Cluster_HOMO[spin] = i1;
                    }
                }

                Dnum = (TZ - Num_State) - system_charge;
                if (0.0 <= Dnum)
                    ChemP_MIN = ChemP;
                else
                    ChemP_MAX = ChemP;
                if (fabs(Dnum) < 1.0e-14)
                    po = 1;

                if (myid1 == Host_ID && 2 <= level_stdout) {
                    printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n", ChemP, TZ, Num_State);
                }

                loopN++;

            } while (po == 0 && loopN < 1000);

            if (2 <= level_stdout) {
                printf("  ChemP=%15.12f\n", ChemP);
            }

            if (measure_time) {
                dtime(&etime);
                time5 += etime - stime;
            }

        } /* end of else for if (xanes_calc==1) */

        /****************************************************
          Energies by summing up eigenvalues
    ****************************************************/

        Eele0[0] = 0.0;
        Eele0[1] = 0.0;

        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (i1 = 1; i1 <= MaxN; i1++) {

                if (xanes_calc == 1)
                    x = (ko[spin][i1] - ChemP_XANES[spin]) * Beta;
                else
                    x = (ko[spin][i1] - ChemP) * Beta;

                if (x <= -max_x)
                    x = -max_x;
                if (max_x <= x)
                    x = max_x;
                FermiF = FermiFunc(x, spin, i1, &po, &x);

                Eele0[spin] += ko[spin][i1] * FermiF;
            }
        }

        if (SpinP_switch == 0) {
            Eele0[1] = Eele0[0];
        }

        /****************************************************
        LCAO coefficients are stored for calculating
                  values of MOs on grids
    ****************************************************/

        if (SpinP_switch == 0) {
            if ((Cluster_HOMO[0] - num_HOMOs + 1) < 1)
                num_HOMOs = Cluster_HOMO[0];
            if ((Cluster_HOMO[0] + num_LUMOs) > MaxN)
                num_LUMOs = MaxN - Cluster_HOMO[0];
        } else if (SpinP_switch == 1) {
            if ((Cluster_HOMO[0] - num_HOMOs + 1) < 1)
                num_HOMOs = Cluster_HOMO[0];
            if ((Cluster_HOMO[1] - num_HOMOs + 1) < 1)
                num_HOMOs = Cluster_HOMO[1];
            if ((Cluster_HOMO[0] + num_LUMOs) > MaxN)
                num_LUMOs = MaxN - Cluster_HOMO[0];
            if ((Cluster_HOMO[1] + num_LUMOs) > MaxN)
                num_LUMOs = MaxN - Cluster_HOMO[1];
        }

        if (myid0 == Host_ID) {
            if (SpinP_switch == 0 && 2 <= level_stdout) {
                printf("  HOMO = %2d\n", Cluster_HOMO[0]);
            } else if (SpinP_switch == 1 && 2 <= level_stdout) {
                printf("  HOMO for up-spin   = %2d\n", Cluster_HOMO[0]);
                printf("  HOMO for down-spin = %2d\n", Cluster_HOMO[1]);
            }
        }

        if (MO_fileout == 1) {

            /* allocation of arrays */
            double * array0;
            int *    is3, *ie3;
            int      numprocs3, ID1;

            array0 = (double *)malloc(sizeof(double) * (n + 2));
            is3    = (int *)malloc(sizeof(int) * numprocs0);
            ie3    = (int *)malloc(sizeof(int) * numprocs0);

            /* HOMOs */

            for (spin = 0; spin <= SpinP_switch; spin++) {

                /* set is3 and ie3 */

                numprocs3 = NPROCS_WD1[spin];

                if (numprocs3 <= MaxN) {

                    av_num = (double)MaxN / (double)numprocs3;
                    for (ID = 0; ID < numprocs3; ID++) {
                        is3[ID] = (int)(av_num * (double)ID) + 1;
                        ie3[ID] = (int)(av_num * (double)(ID + 1));
                    }

                    is3[0]             = 1;
                    ie3[numprocs3 - 1] = MaxN;
                }

                else {
                    for (ID = 0; ID < MaxN; ID++) {
                        is3[ID] = ID + 1;
                        ie3[ID] = ID + 1;
                    }
                    for (ID = MaxN; ID < numprocs3; ID++) {
                        is3[ID] = 1;
                        ie3[ID] = 0;
                    }
                }

                /* loop for j */

                for (j = 0; j < num_HOMOs; j++) {

                    j1 = Cluster_HOMO[spin] - j;

                    /* store eigenvalue */
                    HOMOs_Coef[0][spin][j][0][0].r = ko[spin][j1];

                    /* store EVec1 */
                    if (numprocs0 == 1) {
                        for (k = 0; k < n; k++) {

                            m         = k * (ie2[myid1] - is2[myid1] + 1) + j1 - 1;
                            array0[k] = EVec1[spin][m];
                        }
                    } else {

                        po = 0;
                        for (ID = 0; ID < numprocs3; ID++) {
                            if (is3[ID] <= j1 && j1 <= ie3[ID]) {
                                po  = 1;
                                ID0 = ID;
                                break;
                            }
                        }

                        ID1 = Comm_World_StartID1[spin] + ID0;

                        if (myid0 == ID1) {
                            for (k = 0; k < n; k++) {

                                m         = k * (ie2[myid1] - is2[myid1] + 1) + j1 - is3[ID0];
                                array0[k] = EVec1[spin][m];
                            }
                        }

                        /* MPI communications */
                        MPI_Bcast(array0, n, MPI_DOUBLE, ID1, mpi_comm_level1);
                    }

                    /* store eigenvector */
                    for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
                        wanA = WhatSpecies[GA_AN];
                        tnoA = Spe_Total_CNO[wanA];
                        Anum = MP[GA_AN];
                        for (i = 0; i < tnoA; i++) {
                            HOMOs_Coef[0][spin][j][GA_AN][i].r = array0[Anum + i - 1];
                            HOMOs_Coef[0][spin][j][GA_AN][i].i = 0.0;
                        }
                    }
                }
            }

            /* LUMOs */

            for (spin = 0; spin <= SpinP_switch; spin++) {

                /* set is3 and ie3 */

                numprocs3 = NPROCS_WD1[spin];

                if (numprocs3 <= MaxN) {

                    av_num = (double)MaxN / (double)numprocs3;
                    for (ID = 0; ID < numprocs3; ID++) {
                        is3[ID] = (int)(av_num * (double)ID) + 1;
                        ie3[ID] = (int)(av_num * (double)(ID + 1));
                    }

                    is3[0]             = 1;
                    ie3[numprocs3 - 1] = MaxN;
                }

                else {
                    for (ID = 0; ID < MaxN; ID++) {
                        is3[ID] = ID + 1;
                        ie3[ID] = ID + 1;
                    }
                    for (ID = MaxN; ID < numprocs3; ID++) {
                        is3[ID] = 1;
                        ie3[ID] = -2;
                    }
                }

                /* loop for j */

                for (j = 0; j < num_LUMOs; j++) {

                    j1 = Cluster_HOMO[spin] + 1 + j;

                    /* store eigenvalue */
                    LUMOs_Coef[0][spin][j][0][0].r = ko[spin][j1];

                    /* store EVec1 */
                    if (numprocs0 == 1) {
                        for (k = 0; k < n; k++) {

                            m         = k * (ie2[myid1] - is2[myid1] + 1) + j1 - 1;
                            array0[k] = EVec1[spin][m];
                        }
                    } else {

                        po = 0;
                        for (ID = 0; ID < numprocs3; ID++) {
                            if (is3[ID] <= j1 && j1 <= ie3[ID]) {
                                po  = 1;
                                ID0 = ID;
                                break;
                            }
                        }

                        ID1 = Comm_World_StartID1[spin] + ID0;

                        if (myid0 == ID1) {
                            for (k = 0; k < n; k++) {

                                m         = k * (ie2[myid1] - is2[myid1] + 1) + j1 - is3[ID0];
                                array0[k] = EVec1[spin][m];
                            }
                        }

                        /* MPI communications */
                        MPI_Bcast(array0, n, MPI_DOUBLE, ID1, mpi_comm_level1);
                    }

                    /* store eigenvector */
                    for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
                        wanA = WhatSpecies[GA_AN];
                        tnoA = Spe_Total_CNO[wanA];
                        Anum = MP[GA_AN];
                        for (i = 0; i < tnoA; i++) {
                            LUMOs_Coef[0][spin][j][GA_AN][i].r = array0[Anum + i - 1];
                            LUMOs_Coef[0][spin][j][GA_AN][i].i = 0.0;
                        }
                    }
                }
            }

            /* freeing of array0 */

            free(ie3);
            free(is3);
            free(array0);
        }

        /****************************************************
          density matrix and energy density matrix
                  for up and down spins
    ****************************************************/

        time6 += Calc_DM_Cluster_collinear(myid0, numprocs0, myid1, numprocs1, myworld1, size_H1, is2, ie2, MP, n,
                                           MPI_CommWD1, Comm_World_StartID1, CDM, EDM, ko, CDM1, EDM1, PDM1, Work1,
                                           EVec1, SP_NZeros, SP_Atoms);

        /****************************************************
                        Bond Energies
    ****************************************************/

        My_Eele1[0] = 0.0;
        My_Eele1[1] = 0.0;

        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
                GA_AN = M2G[MA_AN];
                wanA  = WhatSpecies[GA_AN];
                tnoA  = Spe_Total_CNO[wanA];
                for (j = 0; j <= FNAN[GA_AN]; j++) {
                    wanB = WhatSpecies[natn[GA_AN][j]];
                    tnoB = Spe_Total_CNO[wanB];
                    for (k = 0; k < tnoA; k++) {
                        for (l = 0; l < tnoB; l++) {
                            My_Eele1[spin] += CDM[spin][MA_AN][j][k][l] * nh[spin][MA_AN][j][k][l];
                        }
                    }
                }
            }
        }

        /* MPI, My_Eele1 */
        for (spin = 0; spin <= SpinP_switch; spin++) {
            MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        }

        if (SpinP_switch == 0)
            Eele1[1] = Eele1[0];

        if (2 <= level_stdout && myid0 == Host_ID) {
            printf("Eele0[0]=%15.12f Eele0[1]=%15.12f\n", Eele0[0], Eele0[1]);
            printf("Eele1[0]=%15.12f Eele1[1]=%15.12f\n", Eele1[0], Eele1[1]);
        }

        if (measure_time) {
            dtime(&etime);
            time6 += etime - stime;
        }

        /****************************************************
                           Output
    ****************************************************/

        if (measure_time)
            dtime(&stime);

        if (myid0 == Host_ID) {

            sprintf(file_EV, "%s%s.EV", filepath, filename);

            if ((fp_EV = fopen(file_EV, "w")) != NULL) {

                setvbuf(fp_EV, buf, _IOFBF, fp_bsize); /* setvbuf */

                fprintf(fp_EV, "\n");
                fprintf(fp_EV, "***********************************************************\n");
                fprintf(fp_EV, "***********************************************************\n");
                fprintf(fp_EV, "            Eigenvalues (Hartree) for SCF KS-eq.           \n");
                fprintf(fp_EV, "***********************************************************\n");
                fprintf(fp_EV, "***********************************************************\n\n");

                fprintf(fp_EV, "   Chemical Potential (Hartree) = %18.14f\n", ChemP);
                fprintf(fp_EV, "   Number of States             = %18.14f\n", Num_State);
                if (SpinP_switch == 0) {
                    fprintf(fp_EV, "   HOMO = %2d\n", Cluster_HOMO[0]);
                } else if (SpinP_switch == 1) {
                    fprintf(fp_EV, "   HOMO for up-spin   = %2d\n", Cluster_HOMO[0]);
                    fprintf(fp_EV, "   HOMO for down-spin = %2d\n", Cluster_HOMO[1]);
                }

                fprintf(fp_EV, "   Eigenvalues\n");
                fprintf(fp_EV, "                Up-spin            Down-spin\n");
                for (i1 = 1; i1 <= MaxN; i1++) {
                    if (SpinP_switch == 0)
                        fprintf(fp_EV, "      %5d %18.14f %18.14f\n", i1, ko[0][i1], ko[0][i1]);
                    else if (SpinP_switch == 1)
                        fprintf(fp_EV, "      %5d %18.14f %18.14f\n", i1, ko[0][i1], ko[1][i1]);
                }
            }

            /* fclose of fp_EV */
            fclose(fp_EV);
        }

        if (2 <= level_fileout) {

            if (myid0 == Host_ID) {

                sprintf(file_EV, "%s%s.EV", filepath, filename);

                if ((fp_EV = fopen(file_EV, "a")) != NULL) {
                    setvbuf(fp_EV, buf, _IOFBF, fp_bsize); /* setvbuf */
                } else {
                    printf("Failure of saving the EV file.\n");
                }
            }

            if (myid0 == Host_ID) {

                fprintf(fp_EV, "\n");
                fprintf(fp_EV, "***********************************************************\n");
                fprintf(fp_EV, "***********************************************************\n");
                fprintf(fp_EV, "  Eigenvalues (Hartree) and Eigenvectors for SCF KS-eq.  \n");
                fprintf(fp_EV, "***********************************************************\n");
                fprintf(fp_EV, "***********************************************************\n");

                fprintf(fp_EV, "\n\n");
                fprintf(fp_EV, "   Chemical Potential (Hartree) = %18.14f\n", ChemP);
                if (SpinP_switch == 0) {
                    fprintf(fp_EV, "   HOMO = %2d\n", Cluster_HOMO[0]);
                } else if (SpinP_switch == 1) {
                    fprintf(fp_EV, "   HOMO for up-spin   = %2d\n", Cluster_HOMO[0]);
                    fprintf(fp_EV, "   HOMO for down-spin = %2d\n", Cluster_HOMO[1]);
                }

                fprintf(fp_EV, "\n");
                fprintf(fp_EV, "   LCAO coefficients for up (U) and down (D) spins\n\n");
            }

            /* allocation of arrays */
            double * array0;
            int *    is3, *ie3;
            int      numprocs3, ID1;

            array0 = (double *)malloc(sizeof(double) * (n + 2) * 7);
            is3    = (int *)malloc(sizeof(int) * numprocs0);
            ie3    = (int *)malloc(sizeof(int) * numprocs0);

            /* start of loop spin and i */

            num0 = 6;
            num1 = MaxN / num0 + 1 * (MaxN % num0 != 0);

            for (spin = 0; spin <= SpinP_switch; spin++) {

                /* set is3 and ie3 */

                numprocs3 = NPROCS_WD1[spin];

                if (numprocs3 <= MaxN) {

                    av_num = (double)MaxN / (double)numprocs3;
                    for (ID = 0; ID < numprocs3; ID++) {
                        is3[ID] = (int)(av_num * (double)ID) + 1;
                        ie3[ID] = (int)(av_num * (double)(ID + 1));
                    }

                    is3[0]             = 1;
                    ie3[numprocs3 - 1] = MaxN;
                }

                else {
                    for (ID = 0; ID < MaxN; ID++) {
                        is3[ID] = ID + 1;
                        ie3[ID] = ID + 1;
                    }
                    for (ID = MaxN; ID < numprocs3; ID++) {
                        is3[ID] = 1;
                        ie3[ID] = -2;
                    }
                }

                for (i = 1; i <= num1; i++) {

                    if (myid0 == Host_ID) {

                        /* header */
                        fprintf(fp_EV, "\n");
                        for (i1 = -1; i1 <= 0; i1++) {

                            if (i1 == -1)
                                fprintf(fp_EV, "                     ");
                            else if (i1 == 0)
                                fprintf(fp_EV, "                      ");

                            for (j = 1; j <= num0; j++) {
                                j1 = num0 * (i - 1) + j;
                                if (j1 <= MaxN) {
                                    if (i1 == -1) {

                                        if (spin == 0)
                                            fprintf(fp_EV, " %5d (U)", j1);
                                        else if (spin == 1)
                                            fprintf(fp_EV, " %5d (D)", j1);
                                    } else if (i1 == 0) {
                                        fprintf(fp_EV, "  %8.5f", ko[spin][j1]);
                                    }
                                }
                            }
                            fprintf(fp_EV, "\n");
                            if (i1 == 0)
                                fprintf(fp_EV, "\n");
                        }
                    }

                    /* MPI communication of EVec1 */

                    if (numprocs0 == 1) {

                        for (j = 1; j <= num0; j++) {

                            j1 = num0 * (i - 1) + j;

                            if (j1 <= MaxN) {
                                for (k = 0; k < n; k++) {

                                    m                       = k * (ie2[myid1] - is2[myid1] + 1) + j1 - 1;
                                    array0[(j - 1) * n + k] = EVec1[spin][m];
                                }
                            }
                        }
                    } else {

                        for (j = 1; j <= num0; j++) {

                            j1 = num0 * (i - 1) + j;

                            po = 0;
                            for (ID = 0; ID < numprocs3; ID++) {
                                if (is3[ID] <= j1 && j1 <= ie3[ID]) {
                                    po  = 1;
                                    ID0 = ID;
                                    break;
                                }
                            }

                            ID1 = Comm_World_StartID1[spin] + ID0;

                            if (j1 <= MaxN) {

                                if (myid0 == ID1) {
                                    for (k = 0; k < n; k++) {

                                        m                       = k * (ie2[myid1] - is2[myid1] + 1) + j1 - is3[ID0];
                                        array0[(j - 1) * n + k] = EVec1[spin][m];
                                    }
                                }

                                /* MPI communications */
                                MPI_Bcast(&array0[(j - 1) * n], n, MPI_DOUBLE, ID1, mpi_comm_level1);
                            }

                        } /* j */
                    } /* else */

                    /* LCAO coefficients */

                    Name_Angular[0][0] = "s          ";
                    Name_Angular[1][0] = "px         ";
                    Name_Angular[1][1] = "py         ";
                    Name_Angular[1][2] = "pz         ";
                    Name_Angular[2][0] = "d3z^2-r^2  ";
                    Name_Angular[2][1] = "dx^2-y^2   ";
                    Name_Angular[2][2] = "dxy        ";
                    Name_Angular[2][3] = "dxz        ";
                    Name_Angular[2][4] = "dyz        ";
                    Name_Angular[3][0] = "f5z^2-3r^2 ";
                    Name_Angular[3][1] = "f5xz^2-xr^2";
                    Name_Angular[3][2] = "f5yz^2-yr^2";
                    Name_Angular[3][3] = "fzx^2-zy^2 ";
                    Name_Angular[3][4] = "fxyz       ";
                    Name_Angular[3][5] = "fx^3-3*xy^2";
                    Name_Angular[3][6] = "f3yx^2-y^3 ";
                    Name_Angular[4][0] = "g1         ";
                    Name_Angular[4][1] = "g2         ";
                    Name_Angular[4][2] = "g3         ";
                    Name_Angular[4][3] = "g4         ";
                    Name_Angular[4][4] = "g5         ";
                    Name_Angular[4][5] = "g6         ";
                    Name_Angular[4][6] = "g7         ";
                    Name_Angular[4][7] = "g8         ";
                    Name_Angular[4][8] = "g9         ";

                    Name_Multiple[0] = "0";
                    Name_Multiple[1] = "1";
                    Name_Multiple[2] = "2";
                    Name_Multiple[3] = "3";
                    Name_Multiple[4] = "4";
                    Name_Multiple[5] = "5";

                    if (myid0 == Host_ID) {

                        i1 = 1;
                        for (Gc_AN = 1; Gc_AN <= atomnum; Gc_AN++) {

                            wan1 = WhatSpecies[Gc_AN];

                            for (l = 0; l <= Supported_MaxL; l++) {
                                for (mul = 0; mul < Spe_Num_CBasis[wan1][l]; mul++) {
                                    for (m = 0; m < (2 * l + 1); m++) {

                                        if (l == 0 && mul == 0 && m == 0)
                                            fprintf(fp_EV, "%4d %3s %s %s", Gc_AN, SpeName[wan1], Name_Multiple[mul],
                                                    Name_Angular[l][m]);
                                        else
                                            fprintf(fp_EV, "         %s %s", Name_Multiple[mul], Name_Angular[l][m]);
                                        for (j = 1; j <= num0; j++) {
                                            j1 = num0 * (i - 1) + j;
                                            if (0 < i1 && j1 <= MaxN) {
                                                fprintf(fp_EV, "  %8.5f", array0[(j - 1) * n + (i1 - 1)]);
                                            }
                                        }

                                        fprintf(fp_EV, "\n");

                                        i1++;
                                    }
                                }
                            }
                        }
                    }

                } /* i */
            } /* spin */

            /* freeing of arrays */

            free(ie3);
            free(is3);
            free(array0);

            /* fclose of fp_EV */

            if (myid0 == Host_ID) {
                fclose(fp_EV);
            }

        } /* end of if (2<=level_fileout) */

        if (measure_time) {
            dtime(&etime);
            time7 += etime - stime;
        }

    } /* if ( strcasecmp(mode,"scf")==0 ) */

    else if (strcasecmp(mode, "dos") == 0) {
        Save_DOS_Col(n, MaxN, myid1, is2, ie2, MP, CntOLP, EVec1, ko, NPROCS_WD1, Comm_World_StartID1);
    }

    else if (strcasecmp(mode, "lcaoout") == 0) {
        Save_LCAO_Col(n, MaxN, myid1, is2, ie2, MP, CntOLP, EVec1, ko, NPROCS_WD1, Comm_World_StartID1);
    }

    else if (strcasecmp(mode, "xanes") == 0) {
        Calc_XANES_Col(n, MaxN, myid1, ko, nh, CntOLP, CDM, EDM, Eele0, Eele1, myworld1, NPROCS_ID1, Comm_World1,
                       NPROCS_WD1, Comm_World_StartID1, MPI_CommWD1, MP, is2, ie2, Ss, Cs, Hs, CDM1, EDM1, PDM1,
                       size_H1, SP_NZeros, SP_Atoms, EVec1, Work1);
    }

    else if (strcasecmp(mode, "diag") == 0) {
        /* nothing is done. */
    }

    if (measure_time) {
        printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f "
               "time7=%7.3f\n",
               myid0, time1, time2, time3, time4, time5, time6, time7);
        fflush(stdout);
    }

    /****************************************************
                          Free
  ****************************************************/

    free(EVec_Rcv);
    free(index_Rcv_j);
    free(index_Rcv_i);
    free(EVec_Snd);
    free(index_Snd_j);
    free(index_Snd_i);

    free(Num_Rcv_EV);
    free(Num_Snd_EV);
    free(ie1);
    free(is1);

    /* for elapsed time */

    MPI_Barrier(mpi_comm_level1);
    dtime(&TEtime);
    time0 = TEtime - TStime;
    return time0;
}

double Calc_DM_Cluster_collinear(int myid0, int numprocs0, int myid1, int numprocs1, int myworld1, int size_H1,
                                 int * is2, int * ie2, int * MP, int n, MPI_Comm * MPI_CommWD1,
                                 int * Comm_World_StartID1, double ***** CDM, double ***** EDM, double ** ko,
                                 double * DM1, double * EDM1, double * PDM1, double * Work1, double ** EVec1,
                                 int * SP_NZeros, int * SP_Atoms)
{
    int         spin, i, j, i0, j0, k, kmin, kmax, po, p, GA_AN, MA_AN, wanA, tnoA, Anum;
    int         LB_AN, GB_AN, wanB, tnoB, Bnum, i1, j1, ID;
    double      max_x = 60.0, dum;
    double      FermiF, FermiF2, x, x2, diffF, sum1, sum2;
    double      FermiEps = 1.0e-13;
    double      stime, etime, time, lumos;
    MPI_Status  stat;
    MPI_Request request;
    double *    FF, *dFF;

    dtime(&stime);

    /* allocation of arrays */

    FF  = (double *)malloc(sizeof(double) * (n + 1));
    dFF = (double *)malloc(sizeof(double) * (n + 1));

    /* spin=myworld1 */

    spin = myworld1;

calc_dm_collinear:

    /* initialize DM1 */

    for (i = 0; i < size_H1; i++) {
        DM1[i]  = 0.0;
        EDM1[i] = 0.0;
        if (cal_partial_charge) {
            PDM1[i] = 0.0;
        }
    }

    /* pre-calculation of Fermi Function */

    po   = 0;
    kmin = is2[myid1];
    kmax = ie2[myid1];

    for (k = is2[myid1]; k <= ie2[myid1]; k++) {

        if (xanes_calc == 1)
            x = (ko[spin][k] - ChemP_XANES[spin]) * Beta;
        else
            x = (ko[spin][k] - ChemP) * Beta;

        if (x <= -max_x)
            x = -max_x;
        if (max_x <= x)
            x = max_x;
        FermiF = FermiFunc(x, spin, k, &po, &x);

        FF[k] = FermiF;

        if (cal_partial_charge) {

            if (xanes_calc == 1)
                x2 = (ko[spin][k] - (ChemP_XANES[spin] + ene_win_partial_charge)) * Beta;
            else
                x2 = (ko[spin][k] - (ChemP + ene_win_partial_charge)) * Beta;

            if (x2 <= -max_x)
                x2 = -max_x;
            if (max_x <= x2)
                x2 = max_x;
            FermiF2 = FermiFunc(x2, spin, k, &po, &x);
            diffF   = fabs(FermiF - FermiF2);
            dFF[k]  = diffF;
        }
    }

    /* calculation of DM1 */

    p = 0;
    for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {

        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];
        Anum = MP[GA_AN];
        for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            GB_AN = natn[GA_AN][LB_AN];
            wanB  = WhatSpecies[GB_AN];
            tnoB  = Spe_Total_CNO[wanB];
            Bnum  = MP[GB_AN];
            for (i = 0; i < tnoA; i++) {
                for (j = 0; j < tnoB; j++) {

                    i0 = (Anum + i - 1) * (ie2[myid1] - is2[myid1] + 1) - is2[myid1];
                    j0 = (Bnum + j - 1) * (ie2[myid1] - is2[myid1] + 1) - is2[myid1];

                    sum1 = 0.0;
                    sum2 = 0.0;

                    for (k = kmin; k <= kmax; k++) {
                        dum = FF[k] * EVec1[spin][i0 + k] * EVec1[spin][j0 + k];
                        sum1 += dum;
                        sum2 += dum * ko[spin][k];
                    }

                    DM1[p]  = sum1;
                    EDM1[p] = sum2;

                    if (cal_partial_charge) {

                        sum1 = 0.0;
                        for (k = kmin; k <= kmax; k++) {
                            sum1 += dFF[k] * EVec1[spin][i0 + k] * EVec1[spin][j0 + k];
                        }
                        PDM1[p] = sum1;
                    }

                    /* increment of p */
                    p++;
                }
            }
        }
    } /* GA_AN */

    /* MPI_Allreduce */

    MPI_Allreduce(DM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
    for (i = 0; i < size_H1; i++)
        DM1[i] = Work1[i];

    MPI_Allreduce(EDM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
    for (i = 0; i < size_H1; i++)
        EDM1[i] = Work1[i];

    if (cal_partial_charge) {
        MPI_Allreduce(PDM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
        for (i = 0; i < size_H1; i++)
            PDM1[i] = Work1[i];
    }

    /* store DM1 to a proper place */

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

                        CDM[spin][MA_AN][LB_AN][i][j] = DM1[p];
                        EDM[spin][MA_AN][LB_AN][i][j] = EDM1[p];
                        if (cal_partial_charge) {
                            Partial_DM[spin][MA_AN][LB_AN][i][j] = PDM1[p];
                        }

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

    if (SpinP_switch == 1 && numprocs0 == 1 && spin == 0) {
        spin++;
        goto calc_dm_collinear;
    }

    else if (SpinP_switch == 1 && numprocs0 != 1) {

        /* MPI communication of DM1 */

        if (Comm_World_StartID1[0] == myid0) {
            MPI_Isend(DM1, size_H1, MPI_DOUBLE, Comm_World_StartID1[1], 10, mpi_comm_level1, &request);
        }
        if (Comm_World_StartID1[1] == myid0) {
            MPI_Isend(DM1, size_H1, MPI_DOUBLE, Comm_World_StartID1[0], 20, mpi_comm_level1, &request);
        }
        if (Comm_World_StartID1[1] == myid0) {
            MPI_Recv(Work1, size_H1, MPI_DOUBLE, Comm_World_StartID1[0], 10, mpi_comm_level1, &stat);
            MPI_Wait(&request, &stat);
        }
        if (Comm_World_StartID1[0] == myid0) {
            MPI_Recv(Work1, size_H1, MPI_DOUBLE, Comm_World_StartID1[1], 20, mpi_comm_level1, &stat);
            MPI_Wait(&request, &stat);
        }

        MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
        for (i = 0; i < size_H1; i++)
            DM1[i] = Work1[i];

        /* MPI communication of EDM1 */

        if (Comm_World_StartID1[0] == myid0) {
            MPI_Isend(EDM1, size_H1, MPI_DOUBLE, Comm_World_StartID1[1], 10, mpi_comm_level1, &request);
        }
        if (Comm_World_StartID1[1] == myid0) {
            MPI_Isend(EDM1, size_H1, MPI_DOUBLE, Comm_World_StartID1[0], 20, mpi_comm_level1, &request);
        }
        if (Comm_World_StartID1[1] == myid0) {
            MPI_Recv(Work1, size_H1, MPI_DOUBLE, Comm_World_StartID1[0], 10, mpi_comm_level1, &stat);
            MPI_Wait(&request, &stat);
        }
        if (Comm_World_StartID1[0] == myid0) {
            MPI_Recv(Work1, size_H1, MPI_DOUBLE, Comm_World_StartID1[1], 20, mpi_comm_level1, &stat);
            MPI_Wait(&request, &stat);
        }

        MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
        for (i = 0; i < size_H1; i++)
            EDM1[i] = Work1[i];

        /* MPI communication of PDM1 */

        if (cal_partial_charge) {
            if (Comm_World_StartID1[0] == myid0) {
                MPI_Isend(PDM1, size_H1, MPI_DOUBLE, Comm_World_StartID1[1], 10, mpi_comm_level1, &request);
            }
            if (Comm_World_StartID1[1] == myid0) {
                MPI_Isend(PDM1, size_H1, MPI_DOUBLE, Comm_World_StartID1[0], 20, mpi_comm_level1, &request);
            }
            if (Comm_World_StartID1[1] == myid0) {
                MPI_Recv(Work1, size_H1, MPI_DOUBLE, Comm_World_StartID1[0], 10, mpi_comm_level1, &stat);
                MPI_Wait(&request, &stat);
            }
            if (Comm_World_StartID1[0] == myid0) {
                MPI_Recv(Work1, size_H1, MPI_DOUBLE, Comm_World_StartID1[1], 20, mpi_comm_level1, &stat);
                MPI_Wait(&request, &stat);
            }

            MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
            for (i = 0; i < size_H1; i++)
                PDM1[i] = Work1[i];
        }

        /* store DM1 to a proper place */

        if (myworld1 == 0)
            spin = 1;
        else if (myworld1 == 1)
            spin = 0;

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

                            CDM[spin][MA_AN][LB_AN][i][j] = DM1[p];
                            EDM[spin][MA_AN][LB_AN][i][j] = EDM1[p];
                            if (cal_partial_charge) {
                                PDM[spin][MA_AN][LB_AN][i][j] = PDM1[p];
                            }

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
    } /* end of else if (SpinP_switch==1) */

    /* freeing of arrays */

    free(dFF);
    free(FF);

    dtime(&etime);
    return (etime - stime);
}

void Save_DOS_Col(int n, int MaxN, int myid1, int * is2, int * ie2, int * MP, double **** OLP0, double ** EVec1,
                  double ** ko, int * NPROCS_WD1, int * Comm_World_StartID1)
{
    int         spin, i, j, iemin, iemax, GA_AN, k, l;
    int         Anum, Bnum, tnoA, tnoB, wanA, wanB;
    int         MA_AN, LB_AN, GB_AN, MaxL, num, m, p, po;
    int         numprocs, myid, ID, tag, ID0;
    int         i_vec[10];
    double      dum, tmp, av_num;
    char        file_eig[YOUSO10], file_ev[YOUSO10];
    FILE *      fp_eig, *fp_ev;
    MPI_Status  stat;
    MPI_Request request;
    float *     array0;
    float *     SD;
    int *       is3, *ie3;
    int         numprocs3, ID1;

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /* allocation of arrays */

    array0 = (float *)malloc(sizeof(float) * (n + 2));
    is3    = (int *)malloc(sizeof(int) * numprocs);
    ie3    = (int *)malloc(sizeof(int) * numprocs);
    SD     = (float *)malloc(sizeof(float) * (atomnum + 1) * List_YOUSO[7]);

    /* open file pointers */

    if (myid == Host_ID) {

        strcpy(file_eig, ".Dos.val");
        fnjoint(filepath, filename, file_eig);
        if ((fp_eig = fopen(file_eig, "w")) == NULL) {
            printf("cannot open a file %s\n", file_eig);
        }

        strcpy(file_ev, ".Dos.vec");
        fnjoint(filepath, filename, file_ev);
        if ((fp_ev = fopen(file_ev, "w")) == NULL) {
            printf("cannot open a file %s\n", file_ev);
        }
    }

    /* find iemin */

    iemin = n;
    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (i = 1; i < n; i++) {
            if (((ChemP + Dos_Erange[0]) < ko[spin][i]) && (i - 1) < iemin) {
                iemin = i - 1;
                break;
            }
        }
    }
    if (iemin < 1)
        iemin = 1;

    /* find iemax */

    iemax = 1;
    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (i = 1; i <= n; i++) {
            if (((ChemP + Dos_Erange[1]) < ko[spin][i])) {
                if (iemax < i)
                    iemax = i;
                break;
            }
        }
    }
    if (iemax == 1)
        iemax = MaxN;
    if (MaxN < iemax)
        iemax = MaxN;

    /****************************************************
                   save *.Dos.vec
  ****************************************************/

    for (spin = 0; spin <= SpinP_switch; spin++) {

        /* set is3 and ie3 */

        numprocs3 = NPROCS_WD1[spin];

        if (numprocs3 <= MaxN) {

            av_num = (double)MaxN / (double)numprocs3;
            for (ID = 0; ID < numprocs3; ID++) {
                is3[ID] = (int)(av_num * (double)ID) + 1;
                ie3[ID] = (int)(av_num * (double)(ID + 1));
            }

            is3[0]             = 1;
            ie3[numprocs3 - 1] = MaxN;
        }

        else {
            for (ID = 0; ID < MaxN; ID++) {
                is3[ID] = ID + 1;
                ie3[ID] = ID + 1;
            }
            for (ID = MaxN; ID < numprocs3; ID++) {
                is3[ID] = 1;
                ie3[ID] = -2;
            }
        }

        /* loop for p */

        for (p = iemin; p <= iemax; p++) {

            /* store EVec1 */
            if (numprocs == 1) {
                for (k = 0; k < n; k++) {

                    m         = k * (ie2[myid1] - is2[myid1] + 1) + p - 1;
                    array0[k] = (float)EVec1[spin][m];
                }
            } else {

                po = 0;
                for (ID = 0; ID < numprocs3; ID++) {
                    if (is3[ID] <= p && p <= ie3[ID]) {
                        po  = 1;
                        ID0 = ID;
                        break;
                    }
                }

                ID1 = Comm_World_StartID1[spin] + ID0;

                if (myid == ID1) {
                    for (k = 0; k < n; k++) {

                        m         = k * (ie2[myid1] - is2[myid1] + 1) + p - is3[ID0];
                        array0[k] = (float)EVec1[spin][m];
                    }
                }

                /* MPI communications */
                MPI_Bcast(array0, n + 1, MPI_FLOAT, ID1, mpi_comm_level1);
            }

            /* initialize SD */

            for (i = 0; i < (atomnum + 1) * List_YOUSO[7]; i++)
                SD[i] = 0.0;

            i_vec[0] = i_vec[1] = i_vec[2] = 0;
            if (myid == Host_ID)
                fwrite(i_vec, sizeof(int), 3, fp_ev);

            for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {

                GA_AN = M2G[MA_AN];
                wanA  = WhatSpecies[GA_AN];
                tnoA  = Spe_Total_CNO[wanA];
                Anum  = MP[GA_AN];

                for (i = 0; i < tnoA; i++) {
                    for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {

                        GB_AN = natn[GA_AN][LB_AN];
                        wanB  = WhatSpecies[GB_AN];
                        tnoB  = Spe_Total_CNO[wanB];
                        Bnum  = MP[GB_AN];

                        for (j = 0; j < tnoB; j++) {
                            SD[Anum + i] +=
                                array0[Anum - 1 + i] * array0[Bnum - 1 + j] * (float)OLP0[MA_AN][LB_AN][i][j];
                        }

                    } /* LB_AN */
                } /* i */
            } /* MA_AN */

            /* MPI communication */

            for (i = 1; i <= n; i++)
                array0[i] = SD[i];
            MPI_Reduce(array0, SD, n + 1, MPI_FLOAT, MPI_SUM, Host_ID, mpi_comm_level1);

            /* write *.Dos.vec */

            if (myid == Host_ID) {
                fwrite(&SD[1], sizeof(float), n, fp_ev);
            }

        } /* p */
    } /* spin */

    /****************************************************
                   save *.Dos.val
  ****************************************************/

    if (myid == Host_ID) {

        fprintf(fp_eig, "mode        1\n");
        fprintf(fp_eig, "NonCol      0\n");
        fprintf(fp_eig, "N           %d\n", n);
        fprintf(fp_eig, "Nspin       %d\n", SpinP_switch);
        fprintf(fp_eig, "Erange      %lf %lf\n", Dos_Erange[0], Dos_Erange[1]);
        /*  fprintf(fp_eig,"irange      %d %d\n",iemin,iemax); */
        fprintf(fp_eig, "Kgrid       %d %d %d\n", 1, 1, 1);
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

        fprintf(fp_eig, "irange      %d %d\n", iemin, iemax);
        fprintf(fp_eig, "<Eigenvalues\n");

        for (spin = 0; spin <= SpinP_switch; spin++) {
            fprintf(fp_eig, "%d %d %d ", 0, 0, 0);
            for (i = iemin; i <= iemax; i++) {
                fprintf(fp_eig, "%lf ", ko[spin][i]);
                /* printf("%lf ",ko[spin][ie]); */
            }
            fprintf(fp_eig, "\n");
            /* printf("\n"); */
        }
        fprintf(fp_eig, "Eigenvalues>\n");

        printf("write eigenvalues\n");
        printf("write eigenvectors\n");

    } /* if (myid==Host_ID) */

    /* close file pointers */

    if (myid == Host_ID) {
        if (fp_eig)
            fclose(fp_eig);
        if (fp_ev)
            fclose(fp_ev);
    }

    /* freeing of array */

    free(SD);
    free(array0);
    free(is3);
    free(ie3);
}

void Save_LCAO_Col(int n, int MaxN, int myid1, int * is2, int * ie2, int * MP, double **** OLP0, double ** EVec1,
                   double ** ko, int * NPROCS_WD1, int * Comm_World_StartID1)
{
    int         spin, i, j, iemin, iemax, GA_AN, k, l;
    int         Anum, Bnum, tnoA, tnoB, wanA, wanB;
    int         MA_AN, LB_AN, GB_AN, MaxL, num, m, p, po;
    int         numprocs, myid, ID, tag, ID0;
    int         i_vec[10], HOMO[2];
    double      dum, tmp, av_num, FermiF, x;
    double      max_x = 60.0;
    char        operate[YOUSO10];
    FILE *      fp1;
    MPI_Status  stat;
    MPI_Request request;
    double *    array0;
    int *       is3, *ie3;
    int         numprocs3, ID1;

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /* allocation of arrays */

    array0 = (double *)malloc(sizeof(double) * (n + 2));
    is3    = (int *)malloc(sizeof(int) * numprocs);
    ie3    = (int *)malloc(sizeof(int) * numprocs);

    /* open file pointer */

    if (myid == Host_ID) {

        sprintf(operate, "%s%s.lcao", filepath, filename);
        fp1 = fopen(operate, "ab");

        if (fp1 != NULL) {
            remove(operate);
            fclose(fp1);
            fp1 = fopen(operate, "ab");
        }
    }

    /* set iemin and iemax */

    iemin = 1;
    iemax = MaxN;

    /****************************************************
                  find Cluster_HOMO
  ****************************************************/

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (i = 1; i <= MaxN; i++) {
            x = (ko[spin][i] - ChemP) * Beta;
            if (x <= -max_x)
                x = -max_x;
            if (max_x <= x)
                x = max_x;
            FermiF = 1.0 / (1.0 + exp(x));
            if (0.5 < FermiF)
                HOMO[spin] = i;
        }
    }

    /****************************************************
                  save LCAO coefficients
  ****************************************************/

    /* Save parameters */

    if (myid == Host_ID) {
        fwrite(&n, sizeof(int), 1, fp1);
        fwrite(&MaxN, sizeof(int), 1, fp1);
        if (SpinP_switch == 0)
            HOMO[1] = HOMO[0];
        fwrite(HOMO, sizeof(int), 2, fp1);
        fwrite(&SpinP_switch, sizeof(int), 1, fp1);
        fwrite(&Utot, sizeof(double), 1, fp1);
    }

    /* Save array0 */

    for (spin = 0; spin <= SpinP_switch; spin++) {

        /* set is3 and ie3 */

        numprocs3 = NPROCS_WD1[spin];

        if (numprocs3 <= MaxN) {

            av_num = (double)MaxN / (double)numprocs3;
            for (ID = 0; ID < numprocs3; ID++) {
                is3[ID] = (int)(av_num * (double)ID) + 1;
                ie3[ID] = (int)(av_num * (double)(ID + 1));
            }

            is3[0]             = 1;
            ie3[numprocs3 - 1] = MaxN;
        }

        else {
            for (ID = 0; ID < MaxN; ID++) {
                is3[ID] = ID + 1;
                ie3[ID] = ID + 1;
            }
            for (ID = MaxN; ID < numprocs3; ID++) {
                is3[ID] = 1;
                ie3[ID] = -2;
            }
        }

        /* loop for p */

        for (p = iemin; p <= iemax; p++) {

            /* store EVec1 */
            if (numprocs == 1) {
                for (k = 0; k < n; k++) {

                    m         = k * (ie2[myid1] - is2[myid1] + 1) + p - 1;
                    array0[k] = EVec1[spin][m];
                }
            } else {

                po = 0;
                for (ID = 0; ID < numprocs3; ID++) {
                    if (is3[ID] <= p && p <= ie3[ID]) {
                        po  = 1;
                        ID0 = ID;
                        break;
                    }
                }

                ID1 = Comm_World_StartID1[spin] + ID0;

                if (myid == ID1) {
                    for (k = 0; k < n; k++) {

                        m         = k * (ie2[myid1] - is2[myid1] + 1) + p - is3[ID0];
                        array0[k] = EVec1[spin][m];
                    }
                }

                /* MPI communications */
                MPI_Bcast(array0, n, MPI_DOUBLE, ID1, mpi_comm_level1);

            } /* else */

            /* Save array0 */

            if (myid == Host_ID) {
                fwrite(array0, sizeof(double), n, fp1);
            }

        } /* p */
    } /* spin */

    /* close file pointer */

    if (myid == Host_ID) {
        if (fp1)
            fclose(fp1);
    }

    /* freeing of array */

    free(array0);
    free(is3);
    free(ie3);
}

void Calc_XANES_Col(int n, int MaxN, int myid1, double ** ko, double ***** nh, double **** CntOLP, double ***** CDM,
                    double ***** EDM, double Eele0[2], double Eele1[2], int myworld1, int * NPROCS_ID1,
                    int * Comm_World1, int * NPROCS_WD1, int * Comm_World_StartID1, MPI_Comm * MPI_CommWD1, int * MP,
                    int * is2, int * ie2, double * Ss, double * Cs, double * Hs, double * CDM1, double * EDM1,
                    double * PDM1, int size_H1, int * SP_NZeros, int * SP_Atoms, double ** EVec1, double * Work1)

{
    int         spin, i, j, iemin, iemax, GA_AN, k, l, ii, jj;
    int         TNO1, TNO2, ia, jb;
    int         Anum, Bnum, tnoA, tnoB, wanA, wanB;
    int         MA_AN, LB_AN, GB_AN, MaxL, num, m, p, p1, po;
    int         numprocs, myid, ID, tag, ID0, u1, u2, o1;
    int         Num_Excited_States, Num_Occupied_States;
    int         tno0, tno1, Hwan, h_AN, Mc_AN, Gc_AN, BN;
    int         direction, Cwan, Gh_AN, spin0, spin1, spinstate;
    int         i_vec[10], UMOmax, OMOmin;
    double      sum, dum, tmp, av_num, sumx, sumy, sumz;
    char        operate[YOUSO10];
    char        buf[fp_bsize]; /* setvbuf */
    FILE *      fp1;
    MPI_Status  stat;
    MPI_Request request;
    double *    array0;
    int *       is3, *ie3, *MP2;
    int         numprocs3, ID1;

    int         SpinP_switch1, n1, MaxN1;
    double      Utot1, Utot2, Utot3;
    double **   C1, **C2, **Z, **Ax, **Ay, **Az;
    int         Cluster_HOMO1[2], Num_XANES_Out;
    int **      ind2ind;
    double *****OLPmo, **XANES_Out;
    double      XANES_Res[10];
    char        file1[YOUSO10];

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /* standard output */

    if (myid == Host_ID) {
        printf("\n*******************************************************\n");
        printf("               Calculation of XANES spectrum            \n");
        printf("*******************************************************\n\n");
    }

    /* allocation of arrays */

    array0 = (double *)malloc(sizeof(double) * (n + 2));
    is3    = (int *)malloc(sizeof(int) * numprocs);
    ie3    = (int *)malloc(sizeof(int) * numprocs);

    C2 = (double **)malloc(sizeof(double *) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        C2[spin] = (double *)malloc(sizeof(double) * n * MaxN);
    }

    /* set iemin and imax */

    iemin = 1;
    iemax = MaxN;

    /****************************************************
   MPI communicate LCAO coefficients for excited state
  ****************************************************/

    /* MPI communicate LCAO coefficients */

    for (spin = 0; spin <= SpinP_switch; spin++) {

        /* set is3 and ie3 */

        numprocs3 = NPROCS_WD1[spin];

        if (numprocs3 <= MaxN) {

            av_num = (double)MaxN / (double)numprocs3;
            for (ID = 0; ID < numprocs3; ID++) {
                is3[ID] = (int)(av_num * (double)ID) + 1;
                ie3[ID] = (int)(av_num * (double)(ID + 1));
            }

            is3[0]             = 1;
            ie3[numprocs3 - 1] = MaxN;
        }

        else {
            for (ID = 0; ID < MaxN; ID++) {
                is3[ID] = ID + 1;
                ie3[ID] = ID + 1;
            }
            for (ID = MaxN; ID < numprocs3; ID++) {
                is3[ID] = 1;
                ie3[ID] = -2;
            }
        }

        /* loop for p */

        for (p = iemin; p <= iemax; p++) {

            /* store EVec1 */
            if (numprocs == 1) {
                for (k = 0; k < n; k++) {

                    m         = k * (ie2[myid1] - is2[myid1] + 1) + p - 1;
                    array0[k] = EVec1[spin][m];
                }
            } else {

                po = 0;
                for (ID = 0; ID < numprocs3; ID++) {
                    if (is3[ID] <= p && p <= ie3[ID]) {
                        po  = 1;
                        ID0 = ID;
                        break;
                    }
                }

                ID1 = Comm_World_StartID1[spin] + ID0;

                if (myid == ID1) {
                    for (k = 0; k < n; k++) {

                        m         = k * (ie2[myid1] - is2[myid1] + 1) + p - is3[ID0];
                        array0[k] = EVec1[spin][m];
                    }
                }

                /* MPI communications */
                MPI_Bcast(array0, n, MPI_DOUBLE, ID1, mpi_comm_level1);
            }

            /* store array0 to C2 */

            for (k = 0; k < n; k++) {
                C2[spin][(p - iemin) * n + k] = array0[k];
            }

        } /* p */
    } /* spin */

    /*****************************************************
      read the LCAO coefficients for the ground state
   *****************************************************/

    /* open file pointer */

    sprintf(operate, "%s%s", filepath, xanes_gs_file);

    if ((fp1 = fopen(operate, "rb")) != NULL) {

        /* read parameters */

        fread(&n1, sizeof(int), 1, fp1);
        fread(&MaxN1, sizeof(int), 1, fp1);
        fread(Cluster_HOMO1, sizeof(int), 2, fp1);
        fread(&SpinP_switch1, sizeof(int), 1, fp1);
        fread(&Utot1, sizeof(double), 1, fp1);

        /* allocation of C1 */

        C1 = (double **)malloc(sizeof(double *) * (SpinP_switch1 + 1));
        for (spin = 0; spin <= SpinP_switch1; spin++) {
            C1[spin] = (double *)malloc(sizeof(double) * n1 * MaxN1);
        }

        /* read LCAO coefficients */

        for (spin = 0; spin <= SpinP_switch1; spin++) {
            fread(C1[spin], sizeof(double), n1 * MaxN1, fp1);
        }

        /* close file pointer */
        fclose(fp1);
    }

    else {
        if (myid == Host_ID)
            printf("Could not find the xanes.gs.file: %s\n", operate);
        MPI_Finalize();
        exit(0);
    }

    /****************************************************
          start of calculation XANES spectrum 
  ****************************************************/

    /* OLPmo: matrix for momentum operator */

    OLPmo = (double *****)malloc(sizeof(double ****) * 3);
    for (direction = 0; direction < 3; direction++) {

        OLPmo[direction] = (double ****)malloc(sizeof(double ***) * (Matomnum + 1));
        FNAN[0]          = 0;
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN = 0;
                tno0  = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                Cwan  = WhatSpecies[Gc_AN];
                tno0  = Spe_Total_CNO[Cwan];
            }

            OLPmo[direction][Mc_AN] = (double ***)malloc(sizeof(double **) * (FNAN[Gc_AN] + 1));
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno1 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan  = WhatSpecies[Gh_AN];
                    tno1  = Spe_Total_CNO[Hwan];
                }

                OLPmo[direction][Mc_AN][h_AN] = (double **)malloc(sizeof(double *) * tno0);
                for (i = 0; i < tno0; i++) {
                    OLPmo[direction][Mc_AN][h_AN][i] = (double *)malloc(sizeof(double) * tno1);
                }
            }
        }
    }

    /* calculation of momentum matrix elements */

    Calc_OLPmo(OLPmo);

    /* set MP */

    MP2 = (int *)malloc(sizeof(int) * (atomnum + 1));

    p = 0;
    for (Gc_AN = 1; Gc_AN <= atomnum; Gc_AN++) {

        MP2[Gc_AN] = p;
        Cwan       = WhatSpecies[Gc_AN];
        tno0       = Spe_Total_CNO[Cwan];
        p += tno0;
    }

    /* find spinstate and set parameters */

    if (strcmp(Core_Hole_Orbital, "s") == 0) {
        if (Core_Hole_J == 1)
            spinstate = 0;
        else
            spinstate = 1;
    } else if (strcmp(Core_Hole_Orbital, "p") == 0) {
        if (Core_Hole_J <= 3)
            spinstate = 0;
        else
            spinstate = 1;
    } else if (strcmp(Core_Hole_Orbital, "d") == 0) {
        if (Core_Hole_J <= 5)
            spinstate = 0;
        else
            spinstate = 1;
    } else if (strcmp(Core_Hole_Orbital, "f") == 0) {
        if (Core_Hole_J <= 7)
            spinstate = 0;
        else
            spinstate = 1;
    }

    spin0 = spinstate;
    if (spin0 == 0)
        spin1 = 1;
    else
        spin1 = 0;

    /****************************************************
     determine the KS occupied and unoccupied states 
     to be included
  ****************************************************/

    static int xanes_single_flag, xanes_double0_flag, xanes_double1_flag;

    xanes_single_flag  = 1;
    xanes_double0_flag = 0;
    xanes_double1_flag = 0;

    p      = HOMO_XANES[spin0];
    UMOmax = p;

    for (k = p + 1; k <= MaxN; k++) {
        if ((ko[spin0][k] - ko[spin0][p]) < xanes_energy_range)
            UMOmax = k;
    }

    OMOmin = p - 1;
    for (k = p - 1; 1 <= k; k--) {
        if ((ko[spin0][p] - ko[spin0][k]) < xanes_energy_range)
            OMOmin = k;
    }

    /* allocation of XANES_Out */

    p  = HOMO_XANES[spin0];
    p1 = HOMO_XANES[spin1];

    Num_Excited_States  = UMOmax - p;
    Num_Occupied_States = p - OMOmin;

    if (xanes_single_flag == 1)
        Num_XANES_Out = Num_Excited_States;
    if (xanes_double0_flag == 1)
        Num_XANES_Out += Num_Excited_States * (Num_Excited_States - 1) * Num_Occupied_States;
    if (xanes_double1_flag == 1)
        Num_XANES_Out += Num_Excited_States * Num_Excited_States * Num_Occupied_States;

    /* allocation of arrays */

    XANES_Out = (double **)malloc(sizeof(double *) * Num_XANES_Out);
    for (i = 0; i < Num_XANES_Out; i++) {
        XANES_Out[i] = (double *)malloc(sizeof(double) * 10);
    }

    Z = (double **)malloc(sizeof(double *) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        Z[spin] = (double *)malloc(sizeof(double) * UMOmax * UMOmax);
    }

    Ax = (double **)malloc(sizeof(double *) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        Ax[spin] = (double *)malloc(sizeof(double) * UMOmax * UMOmax);
    }

    Ay = (double **)malloc(sizeof(double *) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        Ay[spin] = (double *)malloc(sizeof(double) * UMOmax * UMOmax);
    }

    Az = (double **)malloc(sizeof(double *) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        Az[spin] = (double *)malloc(sizeof(double) * UMOmax * UMOmax);
    }

    ind2ind = (int **)malloc(sizeof(int *) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        ind2ind[spin] = (int *)malloc(sizeof(int) * UMOmax);
    }

    /**************************************************************
   calculations of the overlap, px, py, and pz matrices between 
   Kohn-Sham orbitals
  **************************************************************/

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (ii = 0; ii < UMOmax; ii++) {
            for (jj = 0; jj < UMOmax; jj++) {

                sum  = 0.0;
                sumx = 0.0;
                sumy = 0.0;
                sumz = 0.0;

                for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                    Gc_AN = M2G[Mc_AN];
                    Cwan  = WhatSpecies[Gc_AN];
                    TNO1  = Spe_Total_CNO[Cwan];
                    Anum  = MP2[Gc_AN];

                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                        Gh_AN = natn[Gc_AN][h_AN];
                        Hwan  = WhatSpecies[Gh_AN];
                        TNO2  = Spe_Total_CNO[Hwan];
                        Bnum  = MP2[Gh_AN];

                        for (i = 0; i < TNO1; i++) {

                            ia = Anum + i;

                            for (j = 0; j < TNO2; j++) {

                                jb = Bnum + j;

                                tmp = C1[spin][n * ii + ia] * C2[spin][n * jj + jb];

                                sum += tmp * OLP[0][Mc_AN][h_AN][i][j];
                                sumx += tmp * OLPmo[0][Mc_AN][h_AN][i][j];
                                sumy += tmp * OLPmo[1][Mc_AN][h_AN][i][j];
                                sumz += tmp * OLPmo[2][Mc_AN][h_AN][i][j];
                            }
                        }
                    }
                } /* Mc_AN */

                Z[spin][UMOmax * jj + ii]  = sum;
                Ax[spin][UMOmax * jj + ii] = sumx;
                Ay[spin][UMOmax * jj + ii] = sumy;
                Az[spin][UMOmax * jj + ii] = sumz;

            } /* jj */
        } /* ii */

        /* MPI communication of sum and store to Z */

        MPI_Allreduce(MPI_IN_PLACE, &Z[spin][0], UMOmax * UMOmax, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        MPI_Allreduce(MPI_IN_PLACE, &Ax[spin][0], UMOmax * UMOmax, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        MPI_Allreduce(MPI_IN_PLACE, &Ay[spin][0], UMOmax * UMOmax, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        MPI_Allreduce(MPI_IN_PLACE, &Az[spin][0], UMOmax * UMOmax, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    } /* spin */

    /****************************************************
           contribution of single excitations
  ****************************************************/

    if (xanes_single_flag == 1) {

        /* set parameters */

        for (i = 0; i < Cluster_HOMO[spin1]; i++) {
            ind2ind[spin1][i] = i;
        }

        p                  = HOMO_XANES[spin0];
        Num_Excited_States = UMOmax - p;
        Utot2              = Utot;

        /* loop for k */

        for (k = myid; k < Num_Excited_States; k += numprocs) {

            if (myid == Host_ID) {
                printf("Calc... myid=%2d %4d/%4d\n", myid, k + 1, Num_Excited_States);
                fflush(stdout);
            }

            /* set parameters */

            for (i = 0; i < (p - 1); i++) {
                ind2ind[spin0][i] = i;
            }
            ind2ind[spin0][p - 1] = p - 1 + k;

            /*
      if (myid==Host_ID){
        printf("Utot1=%15.12f Utot2=%15.12f Utot3=%15.12f spin0=%2d spin1=%2d p+k=%2d ko[spin0][p+k]=%15.12f\n",
                Utot1,Utot2,Utot3,spin0,spin1,p+k,ko[spin0][p+k]);
      }
      */

            Utot3 = Utot2 + ko[spin0][p + k] - ko[spin0][p];

            Calc_Oscillator_Strength(n, UMOmax, HOMO_XANES, MP2, ind2ind, C1, C2, Z, Ax, Ay, Az, Utot1, Utot3,
                                     XANES_Res);

            XANES_Out[k][0] = XANES_Res[0];
            XANES_Out[k][1] = XANES_Res[1];
            XANES_Out[k][2] = XANES_Res[2];
            XANES_Out[k][3] = XANES_Res[3];
            XANES_Out[k][4] = XANES_Res[4];
        }
    }

    /* MPI */

    for (k = 0; k < Num_Excited_States; k++) {
        ID = k % numprocs;
        MPI_Bcast(&XANES_Out[k][0], 5, MPI_DOUBLE, ID, mpi_comm_level1);
    }

    /*******************************************************
                       Output result 
  *******************************************************/

    if (myid == Host_ID) {

        sprintf(file1, "%s%s.xanes", filepath, filename);

        if ((fp1 = fopen(file1, "w")) != NULL) {

            setvbuf(fp1, buf, _IOFBF, fp_bsize); /* setvbuf */

            fprintf(fp1, "# core.hole.state: atom=%3d  orbial=%2s index=%2d\n", Core_Hole_Atom, Core_Hole_Orbital,
                    Core_Hole_J);
            fprintf(fp1, "# #1:serial number #2:Excitation energy (eV) #3: Oscillator Strength (OS) #4: x-comp. of OS "
                         "#5: y-comp. of OS #6: z-comp. of OS\n");
            fprintf(fp1, "#\n");

            for (i = 0; i < Num_XANES_Out; i++) {
                fprintf(fp1, "%4d %15.12f %15.12f %15.12f %15.12f %15.12f  # excited state=%4d (spin=%d)\n", i + 1,
                        XANES_Out[i][0], XANES_Out[i][1], XANES_Out[i][2], XANES_Out[i][3], XANES_Out[i][4],
                        HOMO_XANES[spin0] + i, spin0);
            }

            /* fclose of fp1 */
            fclose(fp1);
        }
    }

    /*******************************************************
                    freeing of array
  *******************************************************/

    for (i = 0; i < Num_XANES_Out; i++) {
        free(XANES_Out[i]);
    }
    free(XANES_Out);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        free(Z[spin]);
    }
    free(Z);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        free(Ax[spin]);
    }
    free(Ax);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        free(Ay[spin]);
    }
    free(Ay);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        free(Az[spin]);
    }
    free(Az);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        free(ind2ind[spin]);
    }
    free(ind2ind);

    free(MP2);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        free(C2[spin]);
    }
    free(C2);

    for (spin = 0; spin <= SpinP_switch1; spin++) {
        free(C1[spin]);
    }
    free(C1);

    free(array0);
    free(is3);
    free(ie3);

    for (direction = 0; direction < 3; direction++) {

        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN = 0;
                tno0  = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                Cwan  = WhatSpecies[Gc_AN];
                tno0  = Spe_Total_CNO[Cwan];
            }

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno1 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan  = WhatSpecies[Gh_AN];
                    tno1  = Spe_Total_CNO[Hwan];
                }

                for (i = 0; i < tno0; i++) {
                    free(OLPmo[direction][Mc_AN][h_AN][i]);
                }
                free(OLPmo[direction][Mc_AN][h_AN]);
            }
            free(OLPmo[direction][Mc_AN]);
        }
        free(OLPmo[direction]);
    }
    free(OLPmo);
    OLPmo = NULL;
}

double Calc_Oscillator_Strength(int n, int UMOmax, int Nocc[2], int * MP, int ** ind2ind, double ** Z0, double ** A,
                                double ** Z, double ** Ax, double ** Ay, double ** Az, double Utot1, double Utot2,
                                double XANES_Res[10])
{
    int    ii, jj, Gc_AN, spin, Cwan, Hwan, Mc_AN;
    int    i, j, k, ia, ja, ib, jb, Anum, Bnum, h_AN, Gh_AN, Rn, TNO1, TNO2;
    double tmp, det[2], alldet, sum, allsum, allsumx, allsumy, allsumz;
    double os, osx, osy, osz, p2, px2, py2, pz2;
    double fsum, fsumx, fsumy, fsumz, sumx, sumy, sumz;
    int    numprocs, myid;

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /* set Z0 which is an overlap matrix of two Slater determinants */

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (jj = 0; jj < Nocc[spin]; jj++) {

            j = ind2ind[spin][jj];
            for (ii = 0; ii < Nocc[spin]; ii++) {
                Z0[spin][Nocc[spin] * jj + ii] = Z[spin][UMOmax * j + ii];
            }
        }
    }

    /* calculation of the inverse Z0 */

    for (spin = 0; spin <= SpinP_switch; spin++) {
        det[spin] = Lapack_LU_Dinverse(Nocc[spin], Z0[spin]);
    }

    /* calculation of Z0^{-1} * det */

    alldet = det[0] * det[1];

    for (spin = 0; spin <= SpinP_switch; spin++) {

        for (i = 0; i < Nocc[spin]; i++) {
            for (j = 0; j < Nocc[spin]; j++) {

                tmp                          = Z0[spin][Nocc[spin] * j + i];
                Z0[spin][Nocc[spin] * j + i] = tmp * alldet;
            }
        }
    }

    /* calculation of Ax*(Z^{-1*det)}, Ay*(Z^{-1}*det), and Az*(Z^{-1}*det) */

    for (k = 0; k < 3; k++) {

        /* set Ax, Ay, or Az */

        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (ii = 0; ii < Nocc[spin]; ii++) {

                if (k == 0) {
                    for (jj = 0; jj < Nocc[spin]; jj++) {
                        j                             = ind2ind[spin][jj];
                        A[spin][Nocc[spin] * ii + jj] = Ax[spin][UMOmax * j + ii];
                    }
                }

                else if (k == 1) {
                    for (jj = 0; jj < Nocc[spin]; jj++) {
                        j                             = ind2ind[spin][jj];
                        A[spin][Nocc[spin] * ii + jj] = Ay[spin][UMOmax * j + ii];
                    }
                }

                else if (k == 2) {
                    for (jj = 0; jj < Nocc[spin]; jj++) {
                        j                             = ind2ind[spin][jj];
                        A[spin][Nocc[spin] * ii + jj] = Az[spin][UMOmax * j + ii];
                    }
                }

            } /* ii */
        } /* spin */

        /* A*(Z0^{-1*det)} */

        fsum = 0.0;
        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (i = 0; i < Nocc[spin]; i++) {
                for (j = 0; j < Nocc[spin]; j++) {
                    fsum += A[spin][Nocc[spin] * i + j] * Z0[spin][Nocc[spin] * i + j];
                }
            }
        }

        if (k == 0)
            px2 = fsum * fsum;
        else if (k == 1)
            py2 = fsum * fsum;
        else if (k == 2)
            pz2 = fsum * fsum;

    } /* k */

    p2 = px2 + py2 + pz2;

    os  = p2 / fabs(Utot1 - Utot2);
    osx = px2 / fabs(Utot1 - Utot2);
    osy = py2 / fabs(Utot1 - Utot2);
    osz = pz2 / fabs(Utot1 - Utot2);

    XANES_Res[0] = fabs(Utot1 - Utot2) * 27.2113845;
    XANES_Res[1] = os;
    XANES_Res[2] = osx;
    XANES_Res[3] = osy;
    XANES_Res[4] = osz;

    /*
  if (myid==Host_ID){
    printf(" TE (eV): %10.5f  os: %8.4f  x y z: %8.4f %8.4f %8.4f\n",
             fabs(Utot1-Utot2)*27.2113845,os,osx,osy,osz);
  }
  */
}

double Lapack_LU_Dinverse(int n, double * A)
{
    static char * thisprogram = "Lapack_LU_inverse";
    int *         ipiv;
    double *      work, tmp, det;
    int           lwork;
    int           info, i, j;

    /* L*U factorization */

    ipiv = (int *)malloc(sizeof(int) * n);

    F77_NAME(dgetrf, DGETRF)(&n, &n, A, &n, ipiv, &info);

    if (info != 0) {
        printf("dgetrf failed, info=%i, %s\n", info, thisprogram);
    }

    /* calculation of determinant */

    det = 1.0;
    for (i = 0; i < n; i++) {
        tmp = det;
        det = tmp * A[n * (i) + (i)];
    }

    for (i = 0; i < n; i++) {
        if (ipiv[i] != i + 1) {
            det = -det;
        }
    }

    /* 
    printf("det %15.12f\n",det);   

    for (i=0; i<n; i++){
      printf("i=%2d ipiv=%2d\n",i,ipiv[i]);
    }
    */

    /* inverse L*U factorization */

    lwork = 4 * n;
    work  = (double *)malloc(sizeof(double) * lwork);

    F77_NAME(dgetri, DGETRI)(&n, A, &n, ipiv, work, &lwork, &info);

    if (info != 0) {
        printf("dgetrf failed, info=%i, %s\n", info, thisprogram);
    }

    free(work);
    free(ipiv);

    return det;
}
