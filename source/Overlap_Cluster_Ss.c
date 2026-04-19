/**********************************************************************
  Overlap_Cluster_Ss.c:

     Overlap_Cluster_Ss.c is a subroutine to make an overlap matrix
     for cluster or molecular systems, which is distributed over MPI cores
     according to data distribution of ScaLAPACK.

  Log of Overlap_Cluster_Ss.c:

     19/Nov/2018  Released by T. Ozaki

***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void OverlapClusterSs_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1, 1);
}

static size_t OverlapClusterSs_CheckedAddCount(size_t a, size_t b, const char *label)
{
    if (b > SIZE_MAX - a) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Count overflow in Overlap_Cluster_Ss.c: %s", label);
        OverlapClusterSs_AbortWithMessage(msg);
    }

    return a + b;
}

static size_t OverlapClusterSs_CheckedMulCount(size_t a, size_t b, const char *label)
{
    if (a != 0 && b > SIZE_MAX / a) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Dimension overflow in Overlap_Cluster_Ss.c: %s", label);
        OverlapClusterSs_AbortWithMessage(msg);
    }

    return a * b;
}

static void *OverlapClusterSs_MallocArray(size_t count, size_t elem_size, const char *label)
{
    size_t bytes = OverlapClusterSs_CheckedMulCount(count, elem_size, label);
    void * ptr   = malloc((bytes == 0) ? 1 : bytes);

    if (ptr == NULL) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Out of memory in Overlap_Cluster_Ss.c: %s (%zu bytes)", label, bytes);
        OverlapClusterSs_AbortWithMessage(msg);
    }

    return ptr;
}

static int OverlapClusterSs_ComputeLocalNZeroCount(void)
{
    int    MA_AN, GA_AN, LB_AN, GB_AN;
    int    wanA, wanB, tnoA, tnoB;
    size_t my_nzeros = 0;

    for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        size_t orbitals_in_neighbors = 0;

        GA_AN = M2G[MA_AN];
        wanA  = WhatSpecies[GA_AN];
        tnoA  = Spe_Total_CNO[wanA];

        for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            GB_AN = natn[GA_AN][LB_AN];
            wanB  = WhatSpecies[GB_AN];
            tnoB  = Spe_Total_CNO[wanB];
            orbitals_in_neighbors =
                OverlapClusterSs_CheckedAddCount(orbitals_in_neighbors, (size_t)tnoB, "neighbor orbital count");
        }

        my_nzeros = OverlapClusterSs_CheckedAddCount(
            my_nzeros, OverlapClusterSs_CheckedMulCount((size_t)tnoA, orbitals_in_neighbors, "local S1 segment"),
            "local S1 segment");
    }

    if (my_nzeros > (size_t)INT_MAX) {
        OverlapClusterSs_AbortWithMessage("Local S1 segment exceeds INT_MAX in Overlap_Cluster_Ss.c.");
    }

    return (int)my_nzeros;
}

static int OverlapClusterSs_SetMPAndReturnNum(int *MP)
{
    int    i, wanA;
    size_t orbital_offset = 1;

    for (i = 1; i <= atomnum; i++) {
        if (orbital_offset > (size_t)INT_MAX) {
            OverlapClusterSs_AbortWithMessage(
                "Orbital offsets exceed INT_MAX in Overlap_Cluster_Ss.c.");
        }

        MP[i] = (int)orbital_offset;
        wanA  = WhatSpecies[i];
        orbital_offset =
            OverlapClusterSs_CheckedAddCount(orbital_offset, (size_t)Spe_Total_CNO[wanA], "orbital offsets");
    }

    if (orbital_offset == 0 || orbital_offset - 1u > (size_t)INT_MAX) {
        OverlapClusterSs_AbortWithMessage("Orbital count exceeds INT_MAX in Overlap_Cluster_Ss.c.");
    }

    return (int)(orbital_offset - 1u);
}

static void OverlapClusterSs_PackLocalSegment(double **** OLP0, double *local_S1, int local_nzeros)
{
    int MA_AN, GA_AN, LB_AN, GB_AN;
    int wanA, wanB, tnoA, tnoB;
    int i, j, k;

    k = 0;
    for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        GA_AN = M2G[MA_AN];
        wanA  = WhatSpecies[GA_AN];
        tnoA  = Spe_Total_CNO[wanA];

        for (i = 0; i < tnoA; i++) {
            for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                GB_AN = natn[GA_AN][LB_AN];
                wanB  = WhatSpecies[GB_AN];
                tnoB  = Spe_Total_CNO[wanB];

                for (j = 0; j < tnoB; j++) {
                    local_S1[k] = OLP0[MA_AN][LB_AN][i][j];
                    k++;
                }
            }
        }
    }

    if (k != local_nzeros) {
        OverlapClusterSs_AbortWithMessage("Packed S1 length mismatch in Overlap_Cluster_Ss.c.");
    }
}

static int OverlapClusterSs_FindDenseRoots(int myid, int myid1, int myworld1, int numprocs, int **dense_roots_out)
{
    int  dense_owner = (my_prow == 0 && my_pcol == 0);
    int  local_root_rank;
    int  num_dense_roots;
    int *all_root_ranks;
    int  ID, k;

    (void)myworld1;

    if ((myid1 == 0) != dense_owner) {
        OverlapClusterSs_AbortWithMessage(
            "Inconsistent dense-matrix owner mapping in Overlap_Cluster_Ss.c.");
    }

    local_root_rank = dense_owner ? myid : -1;
    all_root_ranks  = (int *)OverlapClusterSs_MallocArray((size_t)numprocs, sizeof(int), "dense roots");

    MPI_Allgather(&local_root_rank, 1, MPI_INT, all_root_ranks, 1, MPI_INT, mpi_comm_level1);

    num_dense_roots = 0;
    for (ID = 0; ID < numprocs; ID++) {
        if (0 <= all_root_ranks[ID]) {
            num_dense_roots++;
        }
    }

    if (num_dense_roots <= 0) {
        free(all_root_ranks);
        OverlapClusterSs_AbortWithMessage("Failed to identify dense roots in Overlap_Cluster_Ss.c.");
    }

    *dense_roots_out =
        (int *)OverlapClusterSs_MallocArray((size_t)num_dense_roots, sizeof(int), "dense roots list");

    k = 0;
    for (ID = 0; ID < numprocs; ID++) {
        if (0 <= all_root_ranks[ID]) {
            (*dense_roots_out)[k] = all_root_ranks[ID];
            k++;
        }
    }

    free(all_root_ranks);
    return num_dense_roots;
}

static void OverlapClusterSs_BuildDenseFromGathered(const double *S1, const int *order_GA, int *MP, double *Ss, int n,
                                                    int tnum)
{
    int      AN, GA_AN, LB_AN, GB_AN;
    int      wanA, wanB, tnoA, tnoB;
    size_t * atom_offsets;
    size_t   dense_count;
    size_t   offset;

    atom_offsets = (size_t *)OverlapClusterSs_MallocArray((size_t)atomnum + 2u, sizeof(size_t), "atom offsets");

    offset = 0;
    for (AN = 1; AN <= atomnum; AN++) {
        size_t atom_block_count = 0;

        atom_offsets[AN] = offset;
        GA_AN            = order_GA[AN];
        wanA             = WhatSpecies[GA_AN];
        tnoA             = Spe_Total_CNO[wanA];

        for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            GB_AN = natn[GA_AN][LB_AN];
            wanB  = WhatSpecies[GB_AN];
            tnoB  = Spe_Total_CNO[wanB];
            atom_block_count = OverlapClusterSs_CheckedAddCount(
                atom_block_count, OverlapClusterSs_CheckedMulCount((size_t)tnoA, (size_t)tnoB, "atom block span"),
                "atom block span");
        }

        offset = OverlapClusterSs_CheckedAddCount(offset, atom_block_count, "dense assembly offsets");
    }
    atom_offsets[atomnum + 1] = offset;

    if (offset != (size_t)tnum) {
        OverlapClusterSs_AbortWithMessage("Gathered S1 size mismatch in Overlap_Cluster_Ss.c.");
    }

    dense_count = OverlapClusterSs_CheckedMulCount((size_t)n, (size_t)n, "dense Ss size");
    memset(Ss, 0, OverlapClusterSs_CheckedMulCount(dense_count, sizeof(double), "dense Ss bytes"));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (AN = 1; AN <= atomnum; AN++) {
        int    GA_AN_local;
        int    wanA_local;
        int    tnoA_local;
        int    Anum_local;
        size_t k_local;

        GA_AN_local = order_GA[AN];
        wanA_local  = WhatSpecies[GA_AN_local];
        tnoA_local  = Spe_Total_CNO[wanA_local];
        Anum_local  = MP[GA_AN_local];
        k_local     = atom_offsets[AN];

        for (int i = 0; i < tnoA_local; i++) {
            int ig = Anum_local + i;

            for (int LB_AN_local = 0; LB_AN_local <= FNAN[GA_AN_local]; LB_AN_local++) {
                int GB_AN_local;
                int wanB_local;
                int tnoB_local;
                int Bnum_local;

                GB_AN_local = natn[GA_AN_local][LB_AN_local];
                wanB_local  = WhatSpecies[GB_AN_local];
                tnoB_local  = Spe_Total_CNO[wanB_local];
                Bnum_local  = MP[GB_AN_local];

                for (int j = 0; j < tnoB_local; j++) {
                    double value;
                    int    jg;

                    jg = Bnum_local + j;

                    if (ig < 1 || n < ig || jg < 1 || n < jg) {
                        OverlapClusterSs_AbortWithMessage(
                            "Dense matrix index is out of range in Overlap_Cluster_Ss.c.");
                    }

                    value = S1[k_local];
                    Ss[(size_t)(jg - 1) * (size_t)n + (size_t)(ig - 1)] += value;

                    k_local++;
                }
            }
        }

        if (k_local != atom_offsets[AN + 1]) {
            OverlapClusterSs_AbortWithMessage(
                "Atom-local S1 offsets are inconsistent in Overlap_Cluster_Ss.c.");
        }
    }

    free(atom_offsets);
}

static void OverlapClusterSs_CuSolver(double **** OLP0, double *Ss, int *MP, MPI_Comm *MPI_CommWD1, int myworld1,
                                      int n)
{
    int     ID, myid, myid1, numprocs;
    int     NUM;
    int     dummy_int = 0;
    int     local_nzeros;
    int     local_matomnum;
    int     num_dense_roots;
    int     total_nzeros_int;
    int *   recv_nzeros;
    int *   recv_matomnum;
    int *   h1_displs;
    int *   atom_displs;
    int *   dense_roots;
    int *   order_GA = NULL;
    double *local_S1;
    double *gathered_S1 = NULL;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    MPI_Comm_rank(MPI_CommWD1[myworld1], &myid1);

    NUM = OverlapClusterSs_SetMPAndReturnNum(MP);
    if (NUM != n) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Orbital count mismatch in Overlap_Cluster_Ss.c: n=%d, NUM=%d", n, NUM);
        OverlapClusterSs_AbortWithMessage(msg);
    }

    local_nzeros  = OverlapClusterSs_ComputeLocalNZeroCount();
    local_matomnum = Matomnum;
    local_S1      = (double *)OverlapClusterSs_MallocArray((size_t)local_nzeros, sizeof(double), "local S1 segment");
    OverlapClusterSs_PackLocalSegment(OLP0, local_S1, local_nzeros);

    recv_nzeros = (int *)OverlapClusterSs_MallocArray((size_t)numprocs, sizeof(int), "recv_nzeros");
    recv_matomnum = (int *)OverlapClusterSs_MallocArray((size_t)numprocs, sizeof(int), "recv_matomnum");
    h1_displs   = (int *)OverlapClusterSs_MallocArray((size_t)numprocs, sizeof(int), "h1_displs");
    atom_displs = (int *)OverlapClusterSs_MallocArray((size_t)numprocs, sizeof(int), "atom_displs");

    MPI_Allgather(&local_nzeros, 1, MPI_INT, recv_nzeros, 1, MPI_INT, mpi_comm_level1);
    MPI_Allgather(&local_matomnum, 1, MPI_INT, recv_matomnum, 1, MPI_INT, mpi_comm_level1);

    {
        size_t total_nzeros = 0;
        size_t total_atoms  = 0;

        for (ID = 0; ID < numprocs; ID++) {
            if (recv_nzeros[ID] < 0 || recv_matomnum[ID] < 0) {
                OverlapClusterSs_AbortWithMessage(
                    "Negative gather counts detected in Overlap_Cluster_Ss.c.");
            }

            if (total_nzeros > (size_t)INT_MAX || total_atoms > (size_t)INT_MAX) {
                OverlapClusterSs_AbortWithMessage(
                    "Gather displacements exceed INT_MAX in Overlap_Cluster_Ss.c.");
            }

            h1_displs[ID]   = (int)total_nzeros;
            atom_displs[ID] = (int)total_atoms;
            total_nzeros =
                OverlapClusterSs_CheckedAddCount(total_nzeros, (size_t)recv_nzeros[ID], "gathered S1 size");
            total_atoms = OverlapClusterSs_CheckedAddCount(total_atoms, (size_t)recv_matomnum[ID],
                                                           "gathered atom order");
        }

        if (total_nzeros > (size_t)INT_MAX) {
            OverlapClusterSs_AbortWithMessage("Gathered S1 size exceeds INT_MAX in Overlap_Cluster_Ss.c.");
        }

        if (total_atoms != (size_t)atomnum) {
            OverlapClusterSs_AbortWithMessage(
                "Gathered atom order length mismatch in Overlap_Cluster_Ss.c.");
        }

        total_nzeros_int = (int)total_nzeros;
    }

    num_dense_roots = OverlapClusterSs_FindDenseRoots(myid, myid1, myworld1, numprocs, &dense_roots);

    for (ID = 0; ID < num_dense_roots; ID++) {
        int dense_root = dense_roots[ID];

        if (myid == dense_root) {
            gathered_S1 =
                (double *)OverlapClusterSs_MallocArray((size_t)total_nzeros_int, sizeof(double), "gathered S1");
            order_GA = (int *)OverlapClusterSs_MallocArray((size_t)atomnum + 1u, sizeof(int), "order_GA");
        }

        MPI_Gatherv(local_S1, local_nzeros, MPI_DOUBLE, gathered_S1, recv_nzeros, h1_displs, MPI_DOUBLE, dense_root,
                    mpi_comm_level1);
        MPI_Gatherv((0 < Matomnum) ? &M2G[1] : &dummy_int, Matomnum, MPI_INT,
                    (myid == dense_root) ? &order_GA[1] : NULL, recv_matomnum, atom_displs, MPI_INT, dense_root,
                    mpi_comm_level1);

        if (myid == dense_root) {
            OverlapClusterSs_BuildDenseFromGathered(gathered_S1, order_GA, MP, Ss, n, total_nzeros_int);
            free(order_GA);
            free(gathered_S1);
            order_GA    = NULL;
            gathered_S1 = NULL;
        }
    }

    free(dense_roots);
    free(atom_displs);
    free(h1_displs);
    free(recv_matomnum);
    free(recv_nzeros);
    free(local_S1);
}

void Overlap_Cluster_Ss(double **** OLP0, double * Ss, int * MP, MPI_Comm * MPI_CommWD1, int myworld1, int n)
{
    int     i, j, k;
    int     MA_AN, GA_AN, LB_AN, GB_AN, AN;
    int     wanA, wanB, tnoA, tnoB, Anum, Bnum, NUM, tnum;
    int     ID, myid, myid1, numprocs;
    int *   My_NZeros;
    int *   is1, *ie1, *is2;
    int *   My_Matomnum, *order_GA;
    double *S1;
    int     ig, jg, il, jl, prow, pcol, brow, bcol;
    size_t  local_count;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    MPI_Comm_rank(MPI_CommWD1[myworld1], &myid1);
    MPI_Barrier(mpi_comm_level1);

    My_NZeros = (int *)OverlapClusterSs_MallocArray((size_t)numprocs, sizeof(int), "My_NZeros");
    My_Matomnum =
        (int *)OverlapClusterSs_MallocArray((size_t)numprocs, sizeof(int), "My_Matomnum");
    is1      = (int *)OverlapClusterSs_MallocArray((size_t)numprocs, sizeof(int), "is1");
    ie1      = (int *)OverlapClusterSs_MallocArray((size_t)numprocs, sizeof(int), "ie1");
    is2      = (int *)OverlapClusterSs_MallocArray((size_t)numprocs, sizeof(int), "is2");
    order_GA = (int *)OverlapClusterSs_MallocArray((size_t)atomnum + 2u, sizeof(int), "order_GA");

    My_NZeros[myid] = OverlapClusterSs_ComputeLocalNZeroCount();

    for (ID = 0; ID < numprocs; ID++) {
        MPI_Bcast(&My_NZeros[ID], 1, MPI_INT, ID, mpi_comm_level1);
    }

    {
        size_t total_nzeros = 0;
        size_t offset       = 0;

        for (ID = 0; ID < numprocs; ID++) {
            if (My_NZeros[ID] < 0) {
                OverlapClusterSs_AbortWithMessage(
                    "Negative My_NZeros detected in Overlap_Cluster_Ss.c.");
            }

            if (offset > (size_t)INT_MAX) {
                OverlapClusterSs_AbortWithMessage(
                    "S1 offset exceeds INT_MAX in Overlap_Cluster_Ss.c.");
            }

            is1[ID] = (int)offset;
            total_nzeros =
                OverlapClusterSs_CheckedAddCount(total_nzeros, (size_t)My_NZeros[ID], "global S1 size");
            offset = OverlapClusterSs_CheckedAddCount(offset, (size_t)My_NZeros[ID], "S1 offset");

            if (offset > (size_t)INT_MAX) {
                OverlapClusterSs_AbortWithMessage(
                    "S1 offset exceeds INT_MAX in Overlap_Cluster_Ss.c.");
            }

            ie1[ID] = (int)offset - 1;
        }

        if (total_nzeros > (size_t)INT_MAX) {
            OverlapClusterSs_AbortWithMessage(
                "Global S1 size exceeds INT_MAX in Overlap_Cluster_Ss.c.");
        }

        tnum = (int)total_nzeros;
    }

    My_Matomnum[myid] = Matomnum;
    for (ID = 0; ID < numprocs; ID++) {
        MPI_Bcast(&My_Matomnum[ID], 1, MPI_INT, ID, mpi_comm_level1);
    }

    {
        size_t atom_offset = 1;

        for (ID = 0; ID < numprocs; ID++) {
            if (My_Matomnum[ID] < 0) {
                OverlapClusterSs_AbortWithMessage(
                    "Negative My_Matomnum detected in Overlap_Cluster_Ss.c.");
            }

            if (atom_offset > (size_t)INT_MAX) {
                OverlapClusterSs_AbortWithMessage(
                    "Global atom ordering exceeds INT_MAX in Overlap_Cluster_Ss.c.");
            }

            is2[ID] = (int)atom_offset;
            atom_offset =
                OverlapClusterSs_CheckedAddCount(atom_offset, (size_t)My_Matomnum[ID], "global atom ordering");
        }

        if (atom_offset != (size_t)atomnum + 1u) {
            OverlapClusterSs_AbortWithMessage(
                "Inconsistent atom partitioning detected in Overlap_Cluster_Ss.c.");
        }
    }

    for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        order_GA[is2[myid] + MA_AN - 1] = M2G[MA_AN];
    }

    for (ID = 0; ID < numprocs; ID++) {
        MPI_Bcast(&order_GA[is2[ID]], My_Matomnum[ID], MPI_INT, ID, mpi_comm_level1);
    }

    NUM = OverlapClusterSs_SetMPAndReturnNum(MP);
    if (NUM != n) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Orbital count mismatch in Overlap_Cluster_Ss.c: n=%d, NUM=%d", n, NUM);
        OverlapClusterSs_AbortWithMessage(msg);
    }

    S1 = (double *)OverlapClusterSs_MallocArray((size_t)tnum + 1u, sizeof(double), "S1");

    k = is1[myid];
    for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        GA_AN = M2G[MA_AN];
        wanA  = WhatSpecies[GA_AN];
        tnoA  = Spe_Total_CNO[wanA];
        for (i = 0; i < tnoA; i++) {
            for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                GB_AN = natn[GA_AN][LB_AN];
                wanB  = WhatSpecies[GB_AN];
                tnoB  = Spe_Total_CNO[wanB];
                for (j = 0; j < tnoB; j++) {
                    S1[k] = OLP0[MA_AN][LB_AN][i][j];
                    k++;
                }
            }
        }
    }

    for (ID = 0; ID < numprocs; ID++) {
        k = is1[ID];
        MPI_Bcast(&S1[k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
    }

    if (scf_eigen_lib_flag == CuSOLVER && myid1 == 0) {

        for (i = 0; i < n * n; i++) {
            Ss[i] = 0.0;
        }

        k = 0;
        for (AN = 1; AN <= atomnum; AN++) {
            GA_AN = order_GA[AN];
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
                        ig = Anum + i;
                        jg = Bnum + j;

                        if (ig < 1 || n < ig || jg < 1 || n < jg) {
                            OverlapClusterSs_AbortWithMessage(
                                "Dense matrix index is out of range in Overlap_Cluster_Ss.c.");
                        }

                        Ss[(size_t)(jg - 1) * (size_t)n + (size_t)(ig - 1)] += S1[k];
                        k++;
                    }
                }
            }
        }
    } else if (scf_eigen_lib_flag == CuSOLVER) {
    } else {
        local_count = OverlapClusterSs_CheckedMulCount((size_t)na_rows, (size_t)na_cols, "distributed Ss size");

        for (size_t idx = 0; idx < local_count; idx++) {
            Ss[idx] = 0.0;
        }

        k = 0;
        for (AN = 1; AN <= atomnum; AN++) {
            GA_AN = order_GA[AN];
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
                        ig = Anum + i;
                        jg = Bnum + j;

                        if (ig < 1 || n < ig || jg < 1 || n < jg) {
                            OverlapClusterSs_AbortWithMessage(
                                "Global matrix index is out of range in Overlap_Cluster_Ss.c.");
                        }

                        brow = (ig - 1) / nblk;
                        bcol = (jg - 1) / nblk;
                        prow = brow % np_rows;
                        pcol = bcol % np_cols;

                        if (my_prow == prow && my_pcol == pcol) {
                            il = (brow / np_rows + 1) * nblk + 1;
                            jl = (bcol / np_cols + 1) * nblk + 1;

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

                            if (il < 1 || na_rows < il || jl < 1 || na_cols < jl) {
                                OverlapClusterSs_AbortWithMessage(
                                    "Distributed matrix index is out of range in Overlap_Cluster_Ss.c.");
                            }

                            Ss[(size_t)(jl - 1) * (size_t)na_rows + (size_t)(il - 1)] += S1[k];
                        }

                        k++;
                    }
                }
            }
        }
    }

    free(S1);
    free(order_GA);
    free(is2);
    free(ie1);
    free(is1);
    free(My_Matomnum);
    free(My_NZeros);
}
