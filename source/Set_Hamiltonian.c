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
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define measure_time 0

void Calc_MatrixElements_dVH_Vxc_VNA(int Cnt_kind);
static void Calc_MatrixElements_dVH_Vxc_VNA_CPU(int Cnt_kind);
static void Calc_MatrixElements_dVH_Vxc_VNA_OpenACC(int Cnt_kind);
static void Set_Hamiltonian_Base_OpenACC(int SCF_iter, double *****H0, double *****HNL, double *****H);
static size_t Set_Hamiltonian_Base_OpenACC_DeviceBytes(int SCF_iter, int myid);
static size_t Set_Hamiltonian_MatrixElements_OpenACC_DeviceBytes(int Cnt_kind, int myid);

static void Set_Hamiltonian_abort(const char *where, const char *message, int myid)
{
    if (myid == Host_ID) {
        fprintf(stderr, "Set_Hamiltonian: %s: %s\n", where, message);
        fflush(stderr);
    }
    MPI_Abort(mpi_comm_level1, 1);
    exit(1);
}

static size_t Set_Hamiltonian_checked_add(size_t a, size_t b, const char *label, int myid)
{
    if (b > ((size_t)-1) - a) {
        char message[256];
        snprintf(message, sizeof(message), "size overflow while estimating %s", label);
        Set_Hamiltonian_abort("OpenACC memory check", message, myid);
    }

    return a + b;
}

static size_t Set_Hamiltonian_checked_mul(size_t a, size_t b, const char *label, int myid)
{
    if (a != 0 && b > ((size_t)-1) / a) {
        char message[256];
        snprintf(message, sizeof(message), "size overflow while estimating %s", label);
        Set_Hamiltonian_abort("OpenACC memory check", message, myid);
    }

    return a * b;
}

static size_t Set_Hamiltonian_array_bytes(size_t count, size_t elem_size, const char *label, int myid)
{
    return Set_Hamiltonian_checked_mul(count, elem_size, label, myid);
}

static void Set_Hamiltonian_add_array_bytes(size_t *total, size_t count, size_t elem_size, const char *label, int myid)
{
    size_t bytes = Set_Hamiltonian_array_bytes(count, elem_size, label, myid);
    *total = Set_Hamiltonian_checked_add(*total, bytes, label, myid);
}

static int Set_Hamiltonian_OpenACC_Threshold(void)
{
    int AN, total_basis;

    if (SpinP_switch == 3 && Solver == 3 && scf_eigen_lib_flag == CuSOLVER) {
        return 0;
    }

    if (scf_eigen_lib_flag != CuSOLVER) {
        return 0;
    }

    total_basis = 0;
    for (AN = 1; AN <= atomnum; AN++) {
        total_basis += Spe_Total_CNO[WhatSpecies[AN]];
        if (GPU_CPU_SWITCH_NUM <= total_basis) {
            return 1;
        }
    }

    return 0;
}

static int Set_Hamiltonian_DeviceMemoryOK(size_t required_bytes, const char *where, int myid)
{
    MPI_Comm node_comm, device_comm;
    int local_rank, cuda_device_count, acc_device_count, device_count, device_rank, device_ranks;
    int cuda_device;
    size_t free_bytes, total_bytes;
    cudaError_t cuda_err;
    unsigned long long local_required, group_required, local_free, group_free;
    int cuda_ok;

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &local_rank);

    cuda_device = -1;
    free_bytes = 0;
    total_bytes = 0;
    cuda_ok = 0;

    cuda_err = cudaGetDeviceCount(&cuda_device_count);
    if (cuda_err != cudaSuccess || cuda_device_count <= 0) {
        if (myid == Host_ID || cuda_err != cudaSuccess) {
            fprintf(stderr,
                    "Set_Hamiltonian: rank %d %s: failed to get CUDA device count (%s); switching to CPU path.\n",
                    myid, where, cuda_err == cudaSuccess ? "no CUDA device" : cudaGetErrorString(cuda_err));
            fflush(stderr);
        }
    }
    else {
        acc_device_count = acc_get_num_devices(acc_device_nvidia);
        if (acc_device_count <= 0) {
            if (myid == Host_ID) {
                fprintf(stderr,
                        "Set_Hamiltonian: %s: failed to get OpenACC NVIDIA device count; switching to CPU path.\n",
                        where);
                fflush(stderr);
            }
        }
        else {
            device_count = (cuda_device_count < acc_device_count) ? cuda_device_count : acc_device_count;
            cuda_device = local_rank % device_count;

            cuda_err = cudaSetDevice(cuda_device);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr,
                        "Set_Hamiltonian: rank %d %s: failed to set CUDA device %d (%s); switching to CPU path.\n",
                        myid, where, cuda_device, cudaGetErrorString(cuda_err));
                fflush(stderr);
            }
            else {
                acc_set_device_num(cuda_device, acc_device_nvidia);
                cuda_err = cudaMemGetInfo(&free_bytes, &total_bytes);
                if (cuda_err != cudaSuccess) {
                    fprintf(stderr,
                            "Set_Hamiltonian: rank %d %s: failed to query CUDA memory on device %d (%s); "
                            "switching to CPU path.\n",
                            myid, where, cuda_device, cudaGetErrorString(cuda_err));
                    fflush(stderr);
                }
                else {
                    cuda_ok = 1;
                }
            }
        }
    }

    MPI_Comm_split(node_comm, cuda_ok ? cuda_device : MPI_UNDEFINED, 0, &device_comm);
    MPI_Comm_free(&node_comm);

    if (!cuda_ok) {
        return 0;
    }

    if (required_bytes > (size_t)ULLONG_MAX || free_bytes > (size_t)ULLONG_MAX) {
        Set_Hamiltonian_abort("OpenACC memory check", "memory size does not fit unsigned long long", myid);
    }

    local_required = (unsigned long long)required_bytes;
    local_free = (unsigned long long)free_bytes;
    MPI_Allreduce(&local_required, &group_required, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, device_comm);
    MPI_Allreduce(&local_free, &group_free, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, device_comm);
    MPI_Comm_rank(device_comm, &device_rank);
    MPI_Comm_size(device_comm, &device_ranks);
    MPI_Comm_free(&device_comm);

    if (group_free < group_required) {
        if (device_rank == 0) {
            fprintf(stderr,
                    "Set_Hamiltonian: %s: GPU device %d is shared by %d rank(s), has %.3f MiB free "
                    "(%.3f MiB total on rank %d), but %.3f MiB is required in total; switching to CPU path.\n",
                    where, cuda_device, device_ranks, (double)group_free / (1024.0 * 1024.0),
                    (double)total_bytes / (1024.0 * 1024.0), myid,
                    (double)group_required / (1024.0 * 1024.0));
            fflush(stderr);
        }
        return 0;
    }

    return 1;
}

static int Set_Hamiltonian_Base_Use_OpenACC(int SCF_iter, int myid)
{
    size_t required_bytes;

    if (!Set_Hamiltonian_OpenACC_Threshold()) {
        return 0;
    }

    required_bytes = Set_Hamiltonian_Base_OpenACC_DeviceBytes(SCF_iter, myid);
    return Set_Hamiltonian_DeviceMemoryOK(required_bytes, "base OpenACC path", myid);
}

static int Set_Hamiltonian_MatrixElements_Use_OpenACC(int Cnt_kind, int myid)
{
    size_t required_bytes;

    if (!Set_Hamiltonian_OpenACC_Threshold()) {
        return 0;
    }

    required_bytes = Set_Hamiltonian_MatrixElements_OpenACC_DeviceBytes(Cnt_kind, myid);
    return Set_Hamiltonian_DeviceMemoryOK(required_bytes, "matrix-elements OpenACC path", myid);
}

static void *Set_Hamiltonian_malloc(size_t bytes, const char *name, int myid)
{
    void *p;

    if (bytes == 0) {
        bytes = 1;
    }

    p = malloc(bytes);
    if (p == NULL) {
        char message[256];
        snprintf(message, sizeof(message), "failed to allocate %s", name);
        Set_Hamiltonian_abort("OpenACC", message, myid);
    }

    return p;
}

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

    if (Cnt_kind != 0 && Cnt_kind != 1) {
        Set_Hamiltonian_abort("Set_Hamiltonian", "Cnt_kind must be 0 or 1", myid);
    }

    if (SpinP_switch != 0 && SpinP_switch != 1 && SpinP_switch != 3) {
        Set_Hamiltonian_abort("Set_Hamiltonian", "SpinP_switch must be 0, 1, or 3", myid);
    }

    if (myid == Host_ID && mode != NULL && strcasecmp(mode, "stdout") == 0 && 0 < level_stdout) {
        printf("<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc...\n");
        fflush(stdout);
    }

    /*****************************************************
                adding H0+HNL+(HCH) to H 
  *****************************************************/

    if (measure_time)
        dtime(&time1);

    if (Set_Hamiltonian_Base_Use_OpenACC(SCF_iter, myid)) {
        Set_Hamiltonian_Base_OpenACC(SCF_iter, H0, HNL, H);
    }

    /* spin non-collinear */

    else if (SpinP_switch == 3) {
#pragma omp parallel for if (omp_get_max_threads() > 1) private(Mc_AN, Gc_AN, Cwan, h_AN, Gh_AN, Hwan, i, j) schedule(static)
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

#pragma omp parallel for if (omp_get_max_threads() > 1) private(Mc_AN, Gc_AN, Cwan, h_AN, Gh_AN, Hwan, i, j, spin) schedule(static)
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
    int myid;

    MPI_Comm_rank(mpi_comm_level1, &myid);

    if (Set_Hamiltonian_MatrixElements_Use_OpenACC(Cnt_kind, myid)) {
        Calc_MatrixElements_dVH_Vxc_VNA_OpenACC(Cnt_kind);
    } else {
        Calc_MatrixElements_dVH_Vxc_VNA_CPU(Cnt_kind);
    }
}

static size_t Set_Hamiltonian_Base_OpenACC_DeviceBytes(int SCF_iter, int myid)
{
    int Mc_AN, h_AN;
    int spin_count, pair_count;
    int add_hub, add_hch, use_vna;
    size_t total_mat, total_h, bytes;

    spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);
    add_hub = ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch) && F_U_flag == 1 && 2 <= SCF_iter);
    add_hch = (core_hole_state_flag == 1);
    use_vna = (ProExpn_VNA != 0);

    pair_count = 0;
    total_mat = 0;
    total_h = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int Gh_AN = natn[Gc_AN][h_AN];
            int Hwan = WhatSpecies[Gh_AN];
            size_t mat_size = Set_Hamiltonian_checked_mul((size_t)Spe_Total_NO[Cwan],
                                                          (size_t)Spe_Total_NO[Hwan],
                                                          "base matrix size", myid);
            size_t h_size = Set_Hamiltonian_checked_mul((size_t)spin_count, mat_size,
                                                        "base Hamiltonian size", myid);

            total_mat = Set_Hamiltonian_checked_add(total_mat, mat_size, "base total matrix", myid);
            total_h = Set_Hamiltonian_checked_add(total_h, h_size, "base total Hamiltonian", myid);
            pair_count++;
        }
    }

    bytes = 0;
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(int), "base pair_NO0", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(int), "base pair_NO1", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t), "base pair_mat_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t), "base pair_h_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_h, sizeof(double), "base hbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_mat, sizeof(double), "base h0buf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_h, sizeof(double), "base hnlbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, use_vna ? total_mat : 1u, sizeof(double), "base hvnabuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, add_hub ? total_h : 1u, sizeof(double), "base hhubbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, add_hch ? total_h : 1u, sizeof(double), "base hchbuf", myid);

    return bytes;
}

static size_t Set_Hamiltonian_MatrixElements_OpenACC_DeviceBytes(int Cnt_kind, int myid)
{
    int Mc_AN, h_AN;
    int spin_count, pair_count;
    size_t total_h, total_nolg, total_orbs0, total_orbs1, bytes;

    if (Cnt_kind != 0 && Cnt_kind != 1) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA_OpenACC", "Cnt_kind must be 0 or 1", myid);
    }

    spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);

    pair_count = 0;
    total_h = 0;
    total_nolg = 0;
    total_orbs0 = 0;
    total_orbs1 = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int Gh_AN = natn[Gc_AN][h_AN];
            int Hwan = WhatSpecies[Gh_AN];
            int NOLG = NumOLG[Mc_AN][h_AN];
            int NO0, NO1;
            size_t mat_size, h_size, orbs0_size, orbs1_size;

            if (Cnt_kind == 0) {
                NO0 = Spe_Total_NO[Cwan];
                NO1 = Spe_Total_NO[Hwan];
            } else {
                NO0 = Spe_Total_CNO[Cwan];
                NO1 = Spe_Total_CNO[Hwan];
            }

            mat_size = Set_Hamiltonian_checked_mul((size_t)NO0, (size_t)NO1,
                                                   "matrix-elements matrix size", myid);
            h_size = Set_Hamiltonian_checked_mul((size_t)spin_count, mat_size,
                                                 "matrix-elements Hamiltonian size", myid);
            orbs0_size = Set_Hamiltonian_checked_mul((size_t)NOLG, (size_t)NO0,
                                                     "matrix-elements orbs0 size", myid);
            orbs1_size = Set_Hamiltonian_checked_mul((size_t)NOLG, (size_t)NO1,
                                                     "matrix-elements orbs1 size", myid);

            total_h = Set_Hamiltonian_checked_add(total_h, h_size, "matrix-elements total Hamiltonian", myid);
            total_nolg = Set_Hamiltonian_checked_add(total_nolg, (size_t)NOLG,
                                                     "matrix-elements total NOLG", myid);
            total_orbs0 = Set_Hamiltonian_checked_add(total_orbs0, orbs0_size,
                                                      "matrix-elements total orbs0", myid);
            total_orbs1 = Set_Hamiltonian_checked_add(total_orbs1, orbs1_size,
                                                      "matrix-elements total orbs1", myid);
            pair_count++;
        }
    }

    bytes = 0;
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(int), "matrix-elements pair_NO0", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(int), "matrix-elements pair_NO1", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(int), "matrix-elements pair_NOLG", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t),
                                    "matrix-elements pair_h_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t),
                                    "matrix-elements pair_nolg_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t),
                                    "matrix-elements pair_orbs0_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t),
                                    "matrix-elements pair_orbs1_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_h, sizeof(double), "matrix-elements hbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes,
                                    Set_Hamiltonian_checked_mul((size_t)spin_count, total_nolg,
                                                                "matrix-elements vpotbuf", myid),
                                    sizeof(double), "matrix-elements vpotbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_orbs0, sizeof(Type_Orbs_Grid), "matrix-elements orbs0buf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_orbs1, sizeof(Type_Orbs_Grid), "matrix-elements orbs1buf", myid);

    return bytes;
}

static void Set_Hamiltonian_Base_OpenACC(int SCF_iter, double *****H0, double *****HNL, double *****H)
{
    int Mc_AN, h_AN, Gc_AN, Gh_AN, Cwan, Hwan;
    int numprocs, myid;
    int spin_count, pair_count, pair;
    int add_hub, add_hch, use_vna;
    double f_kin, f_nl, f_vna;
    int *pair_Mc_AN, *pair_h_AN, *pair_NO0, *pair_NO1;
    size_t *pair_mat_offset, *pair_h_offset;
    size_t total_mat, total_h;
    double *hbuf, *h0buf, *hnlbuf, *hvnabuf, *hhubbuf, *hchbuf;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);
    add_hub = ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch) && F_U_flag == 1 && 2 <= SCF_iter);
    add_hch = (core_hole_state_flag == 1);
    use_vna = (ProExpn_VNA != 0);
    f_kin = (double)F_Kin_flag;
    f_nl = (double)F_NL_flag;
    f_vna = (double)F_VNA_flag;

    pair_count = 0;
    total_mat = 0;
    total_h = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int NO0, NO1;

            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            NO0 = Spe_Total_NO[Cwan];
            NO1 = Spe_Total_NO[Hwan];
            total_mat += (size_t)NO0 * (size_t)NO1;
            total_h += (size_t)spin_count * (size_t)NO0 * (size_t)NO1;
            pair_count++;
        }
    }

    pair_Mc_AN = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_Mc_AN", myid);
    pair_h_AN = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_h_AN", myid);
    pair_NO0 = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_NO0", myid);
    pair_NO1 = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_NO1", myid);
    pair_mat_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openacc pair_mat_offset", myid);
    pair_h_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openacc pair_h_offset", myid);
    hbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * total_h, "openacc base hbuf", myid);
    h0buf = (double *)Set_Hamiltonian_malloc(sizeof(double) * total_mat, "openacc base h0buf", myid);
    hnlbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * total_h, "openacc base hnlbuf", myid);
    hvnabuf =
        (double *)Set_Hamiltonian_malloc(sizeof(double) * (use_vna ? total_mat : 1), "openacc base hvnabuf", myid);
    hhubbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * (add_hub ? total_h : 1), "openacc base hhubbuf", myid);
    hchbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * (add_hch ? total_h : 1), "openacc base hchbuf", myid);

    pair = 0;
    total_mat = 0;
    total_h = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int NO0, NO1, spin, i, j;
            size_t mat_size;

            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            NO0 = Spe_Total_NO[Cwan];
            NO1 = Spe_Total_NO[Hwan];
            mat_size = (size_t)NO0 * (size_t)NO1;

            pair_Mc_AN[pair] = Mc_AN;
            pair_h_AN[pair] = h_AN;
            pair_NO0[pair] = NO0;
            pair_NO1[pair] = NO1;
            pair_mat_offset[pair] = total_mat;
            pair_h_offset[pair] = total_h;

            for (i = 0; i < NO0; i++) {
                for (j = 0; j < NO1; j++) {
                    size_t ij = (size_t)i * (size_t)NO1 + (size_t)j;
                    size_t mij = total_mat + ij;
                    h0buf[mij] = H0[0][Mc_AN][h_AN][i][j];
                    if (use_vna) {
                        hvnabuf[mij] = HVNA[Mc_AN][h_AN][i][j];
                    }
                    for (spin = 0; spin < spin_count; spin++) {
                        size_t idx = total_h + (size_t)spin * mat_size + ij;
                        hnlbuf[idx] = (SpinP_switch == 3 && spin == 3) ? 0.0 : HNL[spin][Mc_AN][h_AN][i][j];
                        if (add_hub && !(SpinP_switch == 3 && spin == 3)) {
                            hhubbuf[idx] = H_Hub[spin][Mc_AN][h_AN][i][j];
                        }
                        if (add_hch && !(SpinP_switch == 3 && spin == 3)) {
                            hchbuf[idx] = HCH[spin][Mc_AN][h_AN][i][j];
                        }
                    }
                }
            }

            total_mat += mat_size;
            total_h += (size_t)spin_count * mat_size;
            pair++;
        }
    }

#pragma acc data copyout(hbuf[0:total_h])                                                                                   \
    copyin(pair_NO0[0:pair_count], pair_NO1[0:pair_count], pair_mat_offset[0:pair_count], pair_h_offset[0:pair_count],      \
           h0buf[0:total_mat], hnlbuf[0:total_h], hvnabuf[0:use_vna ? total_mat : 1], hhubbuf[0:add_hub ? total_h : 1],      \
           hchbuf[0:add_hch ? total_h : 1])
    {
#pragma acc parallel loop gang present(hbuf[0:total_h], pair_NO0[0:pair_count], pair_NO1[0:pair_count],                     \
                                           pair_mat_offset[0:pair_count], pair_h_offset[0:pair_count], h0buf[0:total_mat],   \
                                           hnlbuf[0:total_h], hvnabuf[0:use_vna ? total_mat : 1],                            \
                                           hhubbuf[0:add_hub ? total_h : 1], hchbuf[0:add_hch ? total_h : 1])
        for (pair = 0; pair < pair_count; pair++) {
            int NO1 = pair_NO1[pair];
            size_t mat_size = (size_t)pair_NO0[pair] * (size_t)NO1;
            size_t mat_off = pair_mat_offset[pair];
            size_t h_off = pair_h_offset[pair];
            size_t e;

#pragma acc loop vector
            for (e = 0; e < (size_t)spin_count * mat_size; e++) {
                int spin = (int)(e / mat_size);
                size_t ij = e - (size_t)spin * mat_size;
                size_t idx = h_off + e;
                double v;

                if (SpinP_switch == 3) {
                    if (spin == 0 || spin == 1) {
                        v = f_kin * h0buf[mat_off + ij] + (use_vna ? f_vna * hvnabuf[mat_off + ij] : 0.0) +
                            f_nl * hnlbuf[idx];
                    } else if (spin == 2) {
                        v = f_nl * hnlbuf[idx];
                    } else {
                        v = 0.0;
                    }

                    if (spin < 3 && add_hub) {
                        v += hhubbuf[idx];
                    }
                    if (spin < 3 && add_hch) {
                        v += hchbuf[idx];
                    }
                } else {
                    v = f_kin * h0buf[mat_off + ij] + (use_vna ? f_vna * hvnabuf[mat_off + ij] : 0.0) +
                        f_nl * hnlbuf[idx];
                    if (add_hub) {
                        v += hhubbuf[idx];
                    }
                    if (add_hch) {
                        v += hchbuf[idx];
                    }
                }

                hbuf[idx] = v;
            }
        }
    }

    for (pair = 0; pair < pair_count; pair++) {
        int NO0 = pair_NO0[pair];
        int NO1 = pair_NO1[pair];
        int spin, i, j;
        size_t mat_size = (size_t)NO0 * (size_t)NO1;
        size_t h_off = pair_h_offset[pair];
        Mc_AN = pair_Mc_AN[pair];
        h_AN = pair_h_AN[pair];

        for (spin = 0; spin < spin_count; spin++) {
            for (i = 0; i < NO0; i++) {
                for (j = 0; j < NO1; j++) {
                    size_t ij = (size_t)i * (size_t)NO1 + (size_t)j;
                    H[spin][Mc_AN][h_AN][i][j] = hbuf[h_off + (size_t)spin * mat_size + ij];
                }
            }
        }
    }

    free(hchbuf);
    free(hhubbuf);
    free(hvnabuf);
    free(hnlbuf);
    free(h0buf);
    free(hbuf);
    free(pair_h_offset);
    free(pair_mat_offset);
    free(pair_NO1);
    free(pair_NO0);
    free(pair_h_AN);
    free(pair_Mc_AN);
}

static void Calc_MatrixElements_dVH_Vxc_VNA_OpenACC(int Cnt_kind)
{
    int Mc_AN, Gc_AN, h_AN, Gh_AN, Mh_AN, Cwan, Hwan;
    int numprocs, myid;
    int spin_count, pair_count, pair;
    int *pair_Mc_AN, *pair_h_AN, *pair_NO0, *pair_NO1, *pair_NOLG;
    size_t *pair_h_offset, *pair_nolg_offset, *pair_orbs0_offset, *pair_orbs1_offset;
    size_t total_h, total_nolg, total_nolg_all, total_orbs0, total_orbs1;
    double *hbuf, *vpotbuf;
    Type_Orbs_Grid *orbs0buf, *orbs1buf;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    if (Cnt_kind != 0 && Cnt_kind != 1) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA_OpenACC", "Cnt_kind must be 0 or 1", myid);
    }

    if (SpinP_switch != 0 && SpinP_switch != 1 && SpinP_switch != 3) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA_OpenACC", "SpinP_switch must be 0, 1, or 3", myid);
    }

    spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);

    pair_count = 0;
    total_h = 0;
    total_nolg = 0;
    total_orbs0 = 0;
    total_orbs1 = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int NO0, NO1, NOLG;

            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            NOLG = NumOLG[Mc_AN][h_AN];

            if (Cnt_kind == 0) {
                NO0 = Spe_Total_NO[Cwan];
                NO1 = Spe_Total_NO[Hwan];
            } else {
                NO0 = Spe_Total_CNO[Cwan];
                NO1 = Spe_Total_CNO[Hwan];
            }

            total_h += (size_t)spin_count * (size_t)NO0 * (size_t)NO1;
            total_nolg += (size_t)NOLG;
            total_orbs0 += (size_t)NOLG * (size_t)NO0;
            total_orbs1 += (size_t)NOLG * (size_t)NO1;
            pair_count++;
        }
    }

    pair_Mc_AN = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_Mc_AN", myid);
    pair_h_AN = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_h_AN", myid);
    pair_NO0 = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_NO0", myid);
    pair_NO1 = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_NO1", myid);
    pair_NOLG = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_NOLG", myid);
    pair_h_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openacc pair_h_offset", myid);
    pair_nolg_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openacc pair_nolg_offset", myid);
    pair_orbs0_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openacc pair_orbs0_offset", myid);
    pair_orbs1_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openacc pair_orbs1_offset", myid);
    hbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * total_h, "openacc hbuf", myid);
    vpotbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * (size_t)spin_count * total_nolg, "openacc vpotbuf", myid);
    orbs0buf =
        (Type_Orbs_Grid *)Set_Hamiltonian_malloc(sizeof(Type_Orbs_Grid) * total_orbs0, "openacc orbs0buf", myid);
    orbs1buf =
        (Type_Orbs_Grid *)Set_Hamiltonian_malloc(sizeof(Type_Orbs_Grid) * total_orbs1, "openacc orbs1buf", myid);

    total_nolg_all = total_nolg;

    pair = 0;
    total_h = 0;
    total_nolg = 0;
    total_orbs0 = 0;
    total_orbs1 = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int NO0, NO1, NOLG;
            size_t mat_size;
            int spin, i, j, Nog;

            Gh_AN = natn[Gc_AN][h_AN];
            Mh_AN = F_G2M[Gh_AN];
            Hwan = WhatSpecies[Gh_AN];
            NOLG = NumOLG[Mc_AN][h_AN];

            if (Cnt_kind == 0) {
                NO0 = Spe_Total_NO[Cwan];
                NO1 = Spe_Total_NO[Hwan];
            } else {
                NO0 = Spe_Total_CNO[Cwan];
                NO1 = Spe_Total_CNO[Hwan];
            }

            mat_size = (size_t)NO0 * (size_t)NO1;

            pair_Mc_AN[pair] = Mc_AN;
            pair_h_AN[pair] = h_AN;
            pair_NO0[pair] = NO0;
            pair_NO1[pair] = NO1;
            pair_NOLG[pair] = NOLG;
            pair_h_offset[pair] = total_h;
            pair_nolg_offset[pair] = total_nolg;
            pair_orbs0_offset[pair] = total_orbs0;
            pair_orbs1_offset[pair] = total_orbs1;

            for (spin = 0; spin < spin_count; spin++) {
                for (i = 0; i < NO0; i++) {
                    for (j = 0; j < NO1; j++) {
                        size_t idx = total_h + (size_t)spin * mat_size + (size_t)i * (size_t)NO1 + (size_t)j;
                        hbuf[idx] = (Cnt_kind == 0) ? H[spin][Mc_AN][h_AN][i][j] : CntH[spin][Mc_AN][h_AN][i][j];
                    }
                }
            }

            for (Nog = 0; Nog < NOLG; Nog++) {
                int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
                int MN = MGridListAtom[Mc_AN][Nc];
                int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
                Type_Orbs_Grid *orbs1 = (G2ID[Gh_AN] == myid) ? Orbs_Grid[Mh_AN][Nh] : Orbs_Grid_FNAN[Mc_AN][h_AN][Nog];

                for (i = 0; i < NO0; i++) {
                    orbs0buf[total_orbs0 + (size_t)Nog * (size_t)NO0 + (size_t)i] = Orbs_Grid[Mc_AN][Nc][i];
                }
                for (j = 0; j < NO1; j++) {
                    orbs1buf[total_orbs1 + (size_t)Nog * (size_t)NO1 + (size_t)j] = orbs1[j];
                }
                for (spin = 0; spin < spin_count; spin++) {
                    vpotbuf[(size_t)spin * total_nolg_all + total_nolg + (size_t)Nog] =
                        GridVol * Vpot_Grid[spin][MN];
                }
            }

            total_h += (size_t)spin_count * mat_size;
            total_nolg += (size_t)NOLG;
            total_orbs0 += (size_t)NOLG * (size_t)NO0;
            total_orbs1 += (size_t)NOLG * (size_t)NO1;
            pair++;
        }
    }

#pragma acc data copy(hbuf[0:total_h])                                                                                      \
    copyin(pair_NO0[0:pair_count], pair_NO1[0:pair_count], pair_NOLG[0:pair_count], pair_h_offset[0:pair_count],            \
           pair_nolg_offset[0:pair_count], pair_orbs0_offset[0:pair_count], pair_orbs1_offset[0:pair_count],                \
           orbs0buf[0:total_orbs0], orbs1buf[0:total_orbs1], vpotbuf[0:(size_t)spin_count * total_nolg])
    {
#pragma acc parallel loop gang present(hbuf[0:total_h], pair_NO0[0:pair_count], pair_NO1[0:pair_count],                     \
                                           pair_NOLG[0:pair_count], pair_h_offset[0:pair_count],                            \
                                           pair_nolg_offset[0:pair_count], pair_orbs0_offset[0:pair_count],                  \
                                           pair_orbs1_offset[0:pair_count], orbs0buf[0:total_orbs0],                        \
                                           orbs1buf[0:total_orbs1], vpotbuf[0:(size_t)spin_count * total_nolg])
        for (pair = 0; pair < pair_count; pair++) {
            int NO0 = pair_NO0[pair];
            int NO1 = pair_NO1[pair];
            int NOLG = pair_NOLG[pair];
            size_t mat_size = (size_t)NO0 * (size_t)NO1;
            size_t h_off = pair_h_offset[pair];
            size_t nolg_off = pair_nolg_offset[pair];
            size_t orbs0_off = pair_orbs0_offset[pair];
            size_t orbs1_off = pair_orbs1_offset[pair];
            size_t e;

#pragma acc loop vector
            for (e = 0; e < (size_t)spin_count * mat_size; e++) {
                int spin = (int)(e / mat_size);
                size_t ij = e - (size_t)spin * mat_size;
                int i = (int)(ij / (size_t)NO1);
                int j = (int)(ij - (size_t)i * (size_t)NO1);
                size_t hidx = h_off + e;
                double sum = hbuf[hidx];
                int Nog;

#pragma acc loop seq
                for (Nog = 0; Nog < NOLG; Nog++) {
                    sum += vpotbuf[(size_t)spin * total_nolg + nolg_off + (size_t)Nog] *
                           orbs0buf[orbs0_off + (size_t)Nog * (size_t)NO0 + (size_t)i] *
                           orbs1buf[orbs1_off + (size_t)Nog * (size_t)NO1 + (size_t)j];
                }

                hbuf[hidx] = sum;
            }
        }
    }

    for (pair = 0; pair < pair_count; pair++) {
        int NO0 = pair_NO0[pair];
        int NO1 = pair_NO1[pair];
        int spin, i, j;
        size_t mat_size = (size_t)NO0 * (size_t)NO1;
        size_t h_off = pair_h_offset[pair];
        Mc_AN = pair_Mc_AN[pair];
        h_AN = pair_h_AN[pair];

        for (spin = 0; spin < spin_count; spin++) {
            for (i = 0; i < NO0; i++) {
                for (j = 0; j < NO1; j++) {
                    size_t idx = h_off + (size_t)spin * mat_size + (size_t)i * (size_t)NO1 + (size_t)j;
                    if (Cnt_kind == 0) {
                        H[spin][Mc_AN][h_AN][i][j] = hbuf[idx];
                    } else {
                        CntH[spin][Mc_AN][h_AN][i][j] = hbuf[idx];
                    }
                }
            }
        }
    }

    free(orbs1buf);
    free(orbs0buf);
    free(vpotbuf);
    free(hbuf);
    free(pair_orbs1_offset);
    free(pair_orbs0_offset);
    free(pair_nolg_offset);
    free(pair_h_offset);
    free(pair_NOLG);
    free(pair_NO1);
    free(pair_NO0);
    free(pair_h_AN);
    free(pair_Mc_AN);
}

static void Calc_MatrixElements_dVH_Vxc_VNA_CPU(int Cnt_kind)
{
    int    Mc_AN, Gc_AN, Mh_AN, h_AN, Gh_AN;
    int    Nh0, Nh1, Nh2, Nh3;
    int    Nc0, Nc1, Nc2, Nc3;
    int    MN0, MN1, MN2, MN3;
    int    Nloop, OneD_Nloop;
    int *  OneD2spin, *OneD2Mc_AN, *OneD2h_AN;
    int    numprocs, myid;
    double time0, time1, time2, mflops;

    if (measure_time)
        dtime(&time1);

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    if (Cnt_kind != 0 && Cnt_kind != 1) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "Cnt_kind must be 0 or 1", myid);
    }

    if (SpinP_switch != 0 && SpinP_switch != 1 && SpinP_switch != 3) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "SpinP_switch must be 0, 1, or 3", myid);
    }

    /* one-dimensionalization of loops */

    Nloop = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            Nloop++;
        }
    }

    OneD2Mc_AN = NULL;
    OneD2h_AN  = NULL;

    if (0 < Nloop) {
        if ((size_t)Nloop > ((size_t)-1) / sizeof(int)) {
            Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "loop table size overflow", myid);
        }

        OneD2Mc_AN = (int *)malloc(sizeof(int) * (size_t)Nloop);
        OneD2h_AN  = (int *)malloc(sizeof(int) * (size_t)Nloop);

        if (OneD2Mc_AN == NULL || OneD2h_AN == NULL) {
            free(OneD2Mc_AN);
            free(OneD2h_AN);
            Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "failed to allocate loop tables", myid);
        }
    }

    Nloop = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

            OneD2Mc_AN[Nloop] = Mc_AN;
            OneD2h_AN[Nloop]  = h_AN;
            Nloop++;
        }
    }

    OneD_Nloop = Nloop;

    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time3=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }

    /* numerical integration */

    if (measure_time)
        dtime(&time1);

#pragma omp parallel if (omp_get_max_threads() > 1)
    {
        int     Nloop, spin, Mc_AN, h_AN, Gh_AN, Mh_AN, Hwan, NOLG;
        int     Gc_AN, Cwan, NO0, NO1, spin0 = -1, Mc_AN0 = 0;
        int     i, j, Nc, MN, GNA, Nog, Nh, OMPID, Nthrds;
        int     M, N, K, lda, ldb, ldc, ii, jj;
        double  alpha, beta, Vpot;
        double  sum0, sum1, sum2, sum3, sum4;
        double *ChiV0, *Chi1, *ChiV0_2, *C;

        /* allocation of arrays */

        /* AITUNE */
        double ** AI_tmpH[4];
        {
            /* get size of temporary buffer */
            int AI_MaxNO = 0;
            if (Cnt_kind == 0) {
                int spe;
                for (spe = 0; spe < SpeciesNum; spe++) {
                    if (AI_MaxNO < Spe_Total_NO[spe]) {
                        AI_MaxNO = Spe_Total_NO[spe];
                    }
                }
            } else {
                int spe;
                for (spe = 0; spe < SpeciesNum; spe++) {
                    if (AI_MaxNO < Spe_Total_CNO[spe]) {
                        AI_MaxNO = Spe_Total_CNO[spe];
                    }
                }
            }

            int spin;
            for (spin = 0; spin <= SpinP_switch; spin++) {
                size_t ai_maxno = (size_t)AI_MaxNO;
                size_t elems;

                if (AI_MaxNO <= 0) {
                    Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "non-positive orbital buffer size",
                                          myid);
                }

                if (ai_maxno > ((size_t)-1) / sizeof(double *)) {
                    Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "row pointer size overflow", myid);
                }

                if (ai_maxno > ((size_t)-1) / sizeof(double) / ai_maxno) {
                    Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "matrix buffer size overflow", myid);
                }

                elems = ai_maxno * ai_maxno;

                AI_tmpH[spin] = (double **)malloc(sizeof(double *) * ai_maxno);

                int      i;
                double * p = (double *)malloc(sizeof(double) * elems);

                if (AI_tmpH[spin] == NULL || p == NULL) {
                    free(AI_tmpH[spin]);
                    free(p);
                    Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "failed to allocate matrix buffer",
                                          myid);
                }

                for (i = 0; i < AI_MaxNO; i++) {
                    AI_tmpH[spin][i] = p;
                    p += AI_MaxNO;
                }
            }
        }
        /* AITUNE */

        /* starting of one-dimensionalized loop */

#pragma omp for schedule(static, 1) /* guided */       /* AITUNE */
        for (Nloop = 0; Nloop < OneD_Nloop; Nloop++) { /* AITUNE */

            int Mc_AN = OneD2Mc_AN[Nloop];
            int h_AN  = OneD2h_AN[Nloop];
            int Gc_AN = M2G[Mc_AN];
            int Gh_AN = natn[Gc_AN][h_AN];
            int Mh_AN = F_G2M[Gh_AN];
            int Cwan  = WhatSpecies[Gc_AN];
            int Hwan  = WhatSpecies[Gh_AN];
            int GNA   = GridN_Atom[Gc_AN];
            int NOLG  = NumOLG[Mc_AN][h_AN];
            int Gh_AN_is_local = (G2ID[Gh_AN] == myid);

            int NO0, NO1;
            if (Cnt_kind == 0) {
                NO0 = Spe_Total_NO[Cwan];
                NO1 = Spe_Total_NO[Hwan];
            } else {
                NO0 = Spe_Total_CNO[Cwan];
                NO1 = Spe_Total_CNO[Hwan];
            }

            /* quadrature for Hij  */

            /* AITUNE change order of loop */
            if (SpinP_switch == 0) {
                /* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
                        }
                    }
                }

                int Nog;
                for (Nog = 0; Nog < NOLG; Nog++) {

                    int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
                    int MN = MGridListAtom[Mc_AN][Nc];
                    int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
                    Type_Orbs_Grid *orbs1 =
                        Gh_AN_is_local ? Orbs_Grid[Mh_AN][Nh] : Orbs_Grid_FNAN[Mc_AN][h_AN][Nog];
                    Type_Orbs_Grid *orbs0 = Orbs_Grid[Mc_AN][Nc];

                    double AI_tmp_GVVG = GridVol * Vpot_Grid[0][MN];

                    int i;
                    for (i = 0; i < NO0; i++) {

                        double AI_tmp_i = AI_tmp_GVVG * orbs0[i];
                        double *tmp0 = AI_tmpH[0][i];
                        int    j;

                        for (j = 0; j < NO1; j++) {
                            tmp0[j] += AI_tmp_i * orbs1[j];
                        }
                    }

                } /* Nog */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                        }
                    }
                }

            } else if (SpinP_switch == 1) {

                /* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
                        }
                    }
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[1][i][j] = H[1][Mc_AN][h_AN][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
                        }
                    }
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[1][i][j] = CntH[1][Mc_AN][h_AN][i][j];
                        }
                    }
                }

                int Nog;
                for (Nog = 0; Nog < NOLG; Nog++) {

                    int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
                    int MN = MGridListAtom[Mc_AN][Nc];
                    int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
                    Type_Orbs_Grid *orbs1 =
                        Gh_AN_is_local ? Orbs_Grid[Mh_AN][Nh] : Orbs_Grid_FNAN[Mc_AN][h_AN][Nog];
                    Type_Orbs_Grid *orbs0 = Orbs_Grid[Mc_AN][Nc];

                    double AI_tmp_GVVG  = GridVol * Vpot_Grid[0][MN];
                    double AI_tmp_GVVG1 = GridVol * Vpot_Grid[1][MN];

                    int i;
                    for (i = 0; i < NO0; i++) {

                        double AI_tmp_i = AI_tmp_GVVG * orbs0[i];
                        double *tmp0 = AI_tmpH[0][i];
                        int    j;
                        for (j = 0; j < NO1; j++) {
                            tmp0[j] += AI_tmp_i * orbs1[j];
                        }
                    }

                    for (i = 0; i < NO0; i++) {

                        double AI_tmp_i = AI_tmp_GVVG1 * orbs0[i];
                        double *tmp1 = AI_tmpH[1][i];
                        int    j;
                        for (j = 0; j < NO1; j++) {
                            tmp1[j] += AI_tmp_i * orbs1[j];
                        }
                    }

                } /* Nog */

                /* AITUNE copy from temporary buffer */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                        }
                    }
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            H[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                        }
                    }
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
                        }
                    }
                }

            }

            else { /* SpinP_switch==3 */

                /* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
                            AI_tmpH[1][i][j] = H[1][Mc_AN][h_AN][i][j];
                            AI_tmpH[2][i][j] = H[2][Mc_AN][h_AN][i][j];
                            AI_tmpH[3][i][j] = H[3][Mc_AN][h_AN][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
                            AI_tmpH[1][i][j] = CntH[1][Mc_AN][h_AN][i][j];
                            AI_tmpH[2][i][j] = CntH[2][Mc_AN][h_AN][i][j];
                            AI_tmpH[3][i][j] = CntH[3][Mc_AN][h_AN][i][j];
                        }
                    }
                }

                int Nog;

                for (Nog = 0; Nog < NOLG; Nog++) {

                    int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
                    int MN = MGridListAtom[Mc_AN][Nc];
                    int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
                    Type_Orbs_Grid *orbs1 =
                        Gh_AN_is_local ? Orbs_Grid[Mh_AN][Nh] : Orbs_Grid_FNAN[Mc_AN][h_AN][Nog];
                    Type_Orbs_Grid *orbs0 = Orbs_Grid[Mc_AN][Nc];

                    double AI_tmp_GVVG  = GridVol * Vpot_Grid[0][MN];
                    double AI_tmp_GVVG1 = GridVol * Vpot_Grid[1][MN];
                    double AI_tmp_GVVG2 = GridVol * Vpot_Grid[2][MN];
                    double AI_tmp_GVVG3 = GridVol * Vpot_Grid[3][MN];

                    int i;
                    for (i = 0; i < NO0; i++) {

                        double AI_tmp_i = AI_tmp_GVVG * orbs0[i];
                        double *tmp0 = AI_tmpH[0][i];

                        for (j = 0; j < NO1; j++) {
                            tmp0[j] += AI_tmp_i * orbs1[j];
                        }
                    }

                    for (i = 0; i < NO0; i++) {

                        double AI_tmp_i = AI_tmp_GVVG1 * orbs0[i];
                        double *tmp1 = AI_tmpH[1][i];

                        for (j = 0; j < NO1; j++) {
                            tmp1[j] += AI_tmp_i * orbs1[j];
                        }
                    }

                    for (i = 0; i < NO0; i++) {

                        double AI_tmp_i = AI_tmp_GVVG2 * orbs0[i];
                        double *tmp2 = AI_tmpH[2][i];

                        for (j = 0; j < NO1; j++) {
                            tmp2[j] += AI_tmp_i * orbs1[j];
                        }
                    }

                    for (i = 0; i < NO0; i++) {

                        double AI_tmp_i = AI_tmp_GVVG3 * orbs0[i];
                        double *tmp3 = AI_tmpH[3][i];

                        for (j = 0; j < NO1; j++) {
                            tmp3[j] += AI_tmp_i * orbs1[j];
                        }
                    }

                } /* Nog */

                /* AITUNE copy from temporary buffer */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                        }
                    }
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            H[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
                        }
                    }
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            H[2][Mc_AN][h_AN][i][j] = AI_tmpH[2][i][j];
                        }
                    }
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            H[3][Mc_AN][h_AN][i][j] = AI_tmpH[3][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                        }
                    }
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
                        }
                    }
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[2][Mc_AN][h_AN][i][j] = AI_tmpH[2][i][j];
                        }
                    }
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[3][Mc_AN][h_AN][i][j] = AI_tmpH[3][i][j];
                        }
                    }
                }
            }
            /* AITUNE change order of loop */

        } /* Nloop */

        /* freeing of arrays */
        {
            int spin;
            for (spin = 0; spin <= SpinP_switch; spin++) {
                free(AI_tmpH[spin][0]);
                free(AI_tmpH[spin]);
            }
        }

    } /* pragma omp parallel */

    /* freeing of arrays */

    free(OneD2Mc_AN);
    free(OneD2h_AN);

    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time4=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }
}
