/**********************************************************************
  Set_Hamiltonian.c:

     Set_Hamiltonian.c is a subroutine to make Hamiltonian matrix
     within LDA or GGA.

  Log of Set_Hamiltonian.c:

     24/April/2002  Released by T. Ozaki
     17/April/2013  Modified by A.M. Ito

***********************************************************************/

#include "lapack_prototypes.h"
#include "mpi.h"
#include "openmx_common.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define measure_time 0

void Calc_MatrixElements_dVH_Vxc_VNA(int Cnt_kind);

double Set_Hamiltonian(char * mode, int MD_iter, int SCF_iter, int SCF_iter0, int TRAN_Poisson_flag2,
                       int SucceedReadingDMfile, int Cnt_kind, double ***** H0, double ***** HNL, double ***** CDM,
                       double ***** H)
{
    /***************************************************************
      Cnt_kind
        0:  Uncontracted Hamiltonian    
        1:  Contracted Hamiltonian    
  ***************************************************************/

    int    Mc_AN, Gc_AN, Mh_AN, h_AN, Gh_AN;
    int    i, j, k, Cwan, Hwan, NO0, NO1, spin, N, NOLG;
    int    Nc, Ncs, GNc, GRc, Nog, Nh, MN, XC_P_switch;
    double TStime, TEtime;
    int    numprocs, myid;
    double time0, time1, time2, mflops;
    long   Num_C0, Num_C1;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    MPI_Barrier(mpi_comm_level1);
    dtime(&TStime);

    if (myid == Host_ID && strcasecmp(mode, "stdout") == 0 && 0 < level_stdout) {
        printf("<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc...\n");
        fflush(stdout);
    }

    /*****************************************************
                adding H0+HNL+(HCH) to H 
  *****************************************************/

    if (measure_time)
        dtime(&time1);

    /* spin non-collinear */

    if (SpinP_switch == 3) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            Gc_AN = M2G[Mc_AN];
            Cwan  = WhatSpecies[Gc_AN];
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                Gh_AN = natn[Gc_AN][h_AN];

                Hwan = WhatSpecies[Gh_AN];
                for (i = 0; i < Spe_Total_NO[Cwan]; i++) {
                    for (j = 0; j < Spe_Total_NO[Hwan]; j++) {

                        if (ProExpn_VNA == 0) {
                            H[0][Mc_AN][h_AN][i][j] =
                                F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_NL_flag * HNL[0][Mc_AN][h_AN][i][j];
                            H[1][Mc_AN][h_AN][i][j] =
                                F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_NL_flag * HNL[1][Mc_AN][h_AN][i][j];
                            H[2][Mc_AN][h_AN][i][j] = F_NL_flag * HNL[2][Mc_AN][h_AN][i][j];
                            H[3][Mc_AN][h_AN][i][j] = 0.0;
                        } else {
                            H[0][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] +
                                                      F_VNA_flag * HVNA[Mc_AN][h_AN][i][j] +
                                                      F_NL_flag * HNL[0][Mc_AN][h_AN][i][j];
                            H[1][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] +
                                                      F_VNA_flag * HVNA[Mc_AN][h_AN][i][j] +
                                                      F_NL_flag * HNL[1][Mc_AN][h_AN][i][j];
                            H[2][Mc_AN][h_AN][i][j] = F_NL_flag * HNL[2][Mc_AN][h_AN][i][j];
                            H[3][Mc_AN][h_AN][i][j] = 0.0;
                        }

                        /* Effective Hubbard Hamiltonain --- added by MJ */

                        if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch) && F_U_flag == 1 && 2 <= SCF_iter) {
                            H[0][Mc_AN][h_AN][i][j] += H_Hub[0][Mc_AN][h_AN][i][j];
                            H[1][Mc_AN][h_AN][i][j] += H_Hub[1][Mc_AN][h_AN][i][j];
                            H[2][Mc_AN][h_AN][i][j] += H_Hub[2][Mc_AN][h_AN][i][j];
                        }

                        /* core hole Hamiltonain */

                        if (core_hole_state_flag == 1) {
                            H[0][Mc_AN][h_AN][i][j] += HCH[0][Mc_AN][h_AN][i][j];
                            H[1][Mc_AN][h_AN][i][j] += HCH[1][Mc_AN][h_AN][i][j];
                            H[2][Mc_AN][h_AN][i][j] += HCH[2][Mc_AN][h_AN][i][j];
                        }
                    }
                }
            }
        }
    }

    /* spin collinear */

    else {

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            Gc_AN = M2G[Mc_AN];
            Cwan  = WhatSpecies[Gc_AN];
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan  = WhatSpecies[Gh_AN];
                for (i = 0; i < Spe_Total_NO[Cwan]; i++) {
                    for (j = 0; j < Spe_Total_NO[Hwan]; j++) {
                        for (spin = 0; spin <= SpinP_switch; spin++) {

                            if (ProExpn_VNA == 0) {
                                H[spin][Mc_AN][h_AN][i][j] =
                                    F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_NL_flag * HNL[spin][Mc_AN][h_AN][i][j];
                            } else {
                                H[spin][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] +
                                                             F_VNA_flag * HVNA[Mc_AN][h_AN][i][j] +
                                                             F_NL_flag * HNL[spin][Mc_AN][h_AN][i][j];
                            }

                            /* Effective Hubbard Hamiltonain --- added by MJ */
                            if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch) && F_U_flag == 1 && 2 <= SCF_iter) {
                                H[spin][Mc_AN][h_AN][i][j] += H_Hub[spin][Mc_AN][h_AN][i][j];
                            }

                            /* core hole Hamiltonain */
                            if (core_hole_state_flag == 1) {
                                H[spin][Mc_AN][h_AN][i][j] += HCH[spin][Mc_AN][h_AN][i][j];
                            }
                        }
                    }
                }
            }
        }
    }

    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time1=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }

    if (Cnt_kind == 1) {
        Contract_Hamiltonian(H, CntH, OLP, CntOLP);
        if (SO_switch == 1)
            Contract_iHNL(iHNL, iCntHNL);
    }

    /*****************************************************
   calculation of Vpot;
  *****************************************************/

    if (myid == 0 && measure_time)
        dtime(&time1);

    XC_P_switch = 1;
    Set_Vpot(MD_iter, SCF_iter, SCF_iter0, TRAN_Poisson_flag2, XC_P_switch);

    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time2=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }

    /*****************************************************
   calculation of matrix elements for dVH + Vxc (+ VNA)
  *****************************************************/

    Calc_MatrixElements_dVH_Vxc_VNA(Cnt_kind);

    /* for time */
    if (measure_time)
        dtime(&time1);
    MPI_Barrier(mpi_comm_level1);
    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time Barrier=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }

    dtime(&TEtime);
    time0 = TEtime - TStime;
    return time0;
}

void Calc_MatrixElements_dVH_Vxc_VNA(int Cnt_kind)
{
    double time0, time1, time2, mflops;

    if (measure_time)
        dtime(&time1);

    /* === 0. MPI rank/size 取得 ====================================== */
    int myid, numprocs;
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /* === 1. ペア一次元化 (変更なし) ================================== */
    int pair_cnt = 0;
    for (int Mc_AN = 1; Mc_AN <= Matomnum; ++Mc_AN) {
        int Gc_AN = M2G[Mc_AN];
        pair_cnt += FNAN[Gc_AN] + 1;
    }

    int * restrict idx_Mc = (int *)aligned_alloc(64, sizeof(int) * pair_cnt);
    int * restrict idx_h  = (int *)aligned_alloc(64, sizeof(int) * pair_cnt);

    int p = 0;
    for (int Mc_AN = 1; Mc_AN <= Matomnum; ++Mc_AN) {
        int Gc_AN = M2G[Mc_AN];
        for (int h_AN = 0; h_AN <= FNAN[Gc_AN]; ++h_AN) {
            idx_Mc[p] = Mc_AN;
            idx_h[p]  = h_AN;
            ++p;
        }
    }

    /* --- スピン成分数決定 --- */
    const int nspin = (SpinP_switch == 0) ? 1 : (SpinP_switch == 1) ? 2 : 4;

    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time3=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }

    /* numerical integration */

    if (measure_time)
        dtime(&time1);

    /* === 2. ペア並列ループ ========================================= */
//#pragma omp parallel
    {
        /* スレッド専用一時バッファ（orbsB の double 版） */
        double * restrict bufB = NULL;
        size_t buflen          = 0;

//#pragma omp for schedule(guided, 4)
        for (int pair = 0; pair < pair_cnt; ++pair) {

            /*------------------------------------------------------*
             * 2-1. ペア情報を展開                                  *
             *------------------------------------------------------*/
            const int Mc_AN = idx_Mc[pair];
            const int h_AN  = idx_h[pair];
            const int Gc_AN = M2G[Mc_AN];
            const int Gh_AN = natn[Gc_AN][h_AN];
            const int Mh_AN = F_G2M[Gh_AN];

            const int Cwan = WhatSpecies[Gc_AN];
            const int Hwan = WhatSpecies[Gh_AN];

            const int NO0 = (Cnt_kind == 0) ? Spe_Total_NO[Cwan] : Spe_Total_CNO[Cwan];
            const int NO1 = (Cnt_kind == 0) ? Spe_Total_NO[Hwan] : Spe_Total_CNO[Hwan];

            /*------------------------------------------------------*
             * 2-2. 出力行列ポインタ取得                            *
             *------------------------------------------------------*/
            double ** restrict Hij[4];
            if (Cnt_kind == 0) {
                for (int s = 0; s < nspin; ++s)
                    Hij[s] = H[s][Mc_AN][h_AN];
            } else {
                for (int s = 0; s < nspin; ++s)
                    Hij[s] = CntH[s][Mc_AN][h_AN];
            }

            /*------------------------------------------------------*
             * 2-3. 局所グリッドループ                              *
             *------------------------------------------------------*/
            const int NOLG = NumOLG[Mc_AN][h_AN];

            for (int Nog = 0; Nog < NOLG; ++Nog) {
                const int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
                const int MN = MGridListAtom[Mc_AN][Nc];
                const int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

                /* ポテンシャル × 格子体積 (最多 4 スピン) */
                const double GV0 = GridVol * Vpot_Grid[0][MN];
                const double GV1 = (nspin > 1) ? GridVol * Vpot_Grid[1][MN] : 0.0;
                const double GV2 = (nspin > 2) ? GridVol * Vpot_Grid[2][MN] : 0.0;
                const double GV3 = (nspin > 3) ? GridVol * Vpot_Grid[3][MN] : 0.0;

                /* 軌道グリッド（A, B） */
                const float * restrict orbsA = Orbs_Grid[Mc_AN][Nc];

                const int local_B              = (G2ID[Gh_AN] == myid);
                const float * restrict orbsB_f = local_B ? Orbs_Grid[Mh_AN][Nh] : Orbs_Grid_FNAN[Mc_AN][h_AN][Nog];

                /*--------------------------------------------------*
                 * 2-4. orbsB float→double 変換（ペア毎一度のみ）  *
                 *--------------------------------------------------*/
                if (buflen < (size_t)NO1) {
                    free(bufB);
                    bufB   = (double *)aligned_alloc(64, sizeof(double) * NO1);
                    buflen = NO1;
                }
                for (int j = 0; j < NO1; ++j)
                    bufB[j] = (double)orbsB_f[j];

                /*--------------------------------------------------*
                 * 2-5. メイン i ループ                             *
                 *--------------------------------------------------*/
                for (int i = 0; i < NO0; ++i) {
                    const double a = (double)orbsA[i];

                    const double c0 = a * GV0;
                    const double c1 = a * GV1;
                    const double c2 = a * GV2;
                    const double c3 = a * GV3;

                    double * restrict row0 = Hij[0][i];
                    double * restrict row1 = (nspin > 1) ? Hij[1][i] : NULL;
                    double * restrict row2 = (nspin > 2) ? Hij[2][i] : NULL;
                    double * restrict row3 = (nspin > 3) ? Hij[3][i] : NULL;

                    /*----------------------------------------------*
                     * 2-6. j ループ ― OpenMP simd 指示子            *
                     *----------------------------------------------*/
#pragma omp simd aligned(row0, row1, row2, row3, bufB : 64)
                    for (int j = 0; j < NO1; ++j) {
                        const double b = bufB[j];
                        row0[j] += c0 * b;
                        if (nspin >= 2)
                            row1[j] += c1 * b;
                        if (nspin >= 3)
                            row2[j] += c2 * b;
                        if (nspin == 4)
                            row3[j] += c3 * b;
                    }
                } /* i */
            } /* Nog */
        } /* pair */

        /* スレッド一時バッファ解放 */
        free(bufB);
    } /* omp parallel */

    free(idx_Mc);
    free(idx_h);

    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time4=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }
}