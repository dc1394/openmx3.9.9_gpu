/**********************************************************************
  Hamiltonian_Cluster.c:

     Hamiltonian_Cluster.c is a subroutine to make a Hamiltonian matrix
     for cluster or molecular systems.

  Log of Hamiltonian_Cluster.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void HamiltonianClusterHs_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1, 1);
}

static size_t HamiltonianClusterHs_CheckedAddCount(size_t a, size_t b, const char *label)
{
    if (b > SIZE_MAX - a) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Count overflow in Hamiltonian_Cluster_Hs.c: %s", label);
        HamiltonianClusterHs_AbortWithMessage(msg);
    }

    return a + b;
}

static size_t HamiltonianClusterHs_CheckedMulCount(size_t a, size_t b, const char *label)
{
    if (a != 0 && b > SIZE_MAX / a) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Dimension overflow in Hamiltonian_Cluster_Hs.c: %s", label);
        HamiltonianClusterHs_AbortWithMessage(msg);
    }

    return a * b;
}

static void *HamiltonianClusterHs_MallocArray(size_t count, size_t elem_size, const char *label)
{
    size_t bytes = HamiltonianClusterHs_CheckedMulCount(count, elem_size, label);
    void * ptr   = malloc((bytes == 0) ? 1 : bytes);

    if (ptr == NULL) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Out of memory in Hamiltonian_Cluster_Hs.c: %s (%zu bytes)", label, bytes);
        HamiltonianClusterHs_AbortWithMessage(msg);
    }

    return ptr;
}

static int HamiltonianClusterHs_ComputeLocalNZeroCount(void)
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
                HamiltonianClusterHs_CheckedAddCount(orbitals_in_neighbors, (size_t)tnoB, "neighbor orbital count");
        }

        my_nzeros = HamiltonianClusterHs_CheckedAddCount(
            my_nzeros, HamiltonianClusterHs_CheckedMulCount((size_t)tnoA, orbitals_in_neighbors, "local H1 segment"),
            "local H1 segment");
    }

    if (my_nzeros > (size_t)INT_MAX) {
        HamiltonianClusterHs_AbortWithMessage("Local H1 segment exceeds INT_MAX in Hamiltonian_Cluster_Hs.c.");
    }

    return (int)my_nzeros;
}

static int HamiltonianClusterHs_SetMPAndReturnNum(int *MP)
{
    int    i, wanA;
    size_t orbital_offset = 1;

    for (i = 1; i <= atomnum; i++) {
        if (orbital_offset > (size_t)INT_MAX) {
            HamiltonianClusterHs_AbortWithMessage(
                "Orbital offsets exceed INT_MAX in Hamiltonian_Cluster_Hs.c.");
        }

        MP[i] = (int)orbital_offset;
        wanA  = WhatSpecies[i];
        orbital_offset =
            HamiltonianClusterHs_CheckedAddCount(orbital_offset, (size_t)Spe_Total_CNO[wanA], "orbital offsets");
    }

    if (orbital_offset == 0 || orbital_offset - 1u > (size_t)INT_MAX) {
        HamiltonianClusterHs_AbortWithMessage("Orbital count exceeds INT_MAX in Hamiltonian_Cluster_Hs.c.");
    }

    return (int)(orbital_offset - 1u);
}

void Hamiltonian_Cluster_Hs(double **** RH, double * Hs, int * MP, int spin, MPI_Comm * MPI_CommWD1, int myworld1, int n)
{
    int     i, j, k;
    int     MA_AN, GA_AN, LB_AN, GB_AN, AN;
    int     wanA, wanB, tnoA, tnoB, Anum, Bnum, NUM, tnum;
    int     ID, myid, numprocs;
    int *   My_NZeros;
    int *   is1, *ie1, *is2;
    int *   My_Matomnum, *order_GA;
    double *H1;
    int     ig, jg, il, jl, prow, pcol, brow, bcol;
    size_t  local_count;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    MPI_Barrier(mpi_comm_level1);

    My_NZeros = (int *)HamiltonianClusterHs_MallocArray((size_t)numprocs, sizeof(int), "My_NZeros");
    My_Matomnum =
        (int *)HamiltonianClusterHs_MallocArray((size_t)numprocs, sizeof(int), "My_Matomnum");
    is1      = (int *)HamiltonianClusterHs_MallocArray((size_t)numprocs, sizeof(int), "is1");
    ie1      = (int *)HamiltonianClusterHs_MallocArray((size_t)numprocs, sizeof(int), "ie1");
    is2      = (int *)HamiltonianClusterHs_MallocArray((size_t)numprocs, sizeof(int), "is2");
    order_GA = (int *)HamiltonianClusterHs_MallocArray((size_t)atomnum + 2u, sizeof(int), "order_GA");

    My_NZeros[myid] = HamiltonianClusterHs_ComputeLocalNZeroCount();

    for (ID = 0; ID < numprocs; ID++) {
        MPI_Bcast(&My_NZeros[ID], 1, MPI_INT, ID, mpi_comm_level1);
    }

    {
        size_t total_nzeros = 0;
        size_t offset       = 0;

        for (ID = 0; ID < numprocs; ID++) {
            if (My_NZeros[ID] < 0) {
                HamiltonianClusterHs_AbortWithMessage(
                    "Negative My_NZeros detected in Hamiltonian_Cluster_Hs.c.");
            }

            if (offset > (size_t)INT_MAX) {
                HamiltonianClusterHs_AbortWithMessage(
                    "H1 offset exceeds INT_MAX in Hamiltonian_Cluster_Hs.c.");
            }

            is1[ID] = (int)offset;
            total_nzeros =
                HamiltonianClusterHs_CheckedAddCount(total_nzeros, (size_t)My_NZeros[ID], "global H1 size");
            offset = HamiltonianClusterHs_CheckedAddCount(offset, (size_t)My_NZeros[ID], "H1 offset");

            if (offset > (size_t)INT_MAX) {
                HamiltonianClusterHs_AbortWithMessage(
                    "H1 offset exceeds INT_MAX in Hamiltonian_Cluster_Hs.c.");
            }

            ie1[ID] = (int)offset - 1;
        }

        if (total_nzeros > (size_t)INT_MAX) {
            HamiltonianClusterHs_AbortWithMessage(
                "Global H1 size exceeds INT_MAX in Hamiltonian_Cluster_Hs.c.");
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
                HamiltonianClusterHs_AbortWithMessage(
                    "Negative My_Matomnum detected in Hamiltonian_Cluster_Hs.c.");
            }

            if (atom_offset > (size_t)INT_MAX) {
                HamiltonianClusterHs_AbortWithMessage(
                    "Global atom ordering exceeds INT_MAX in Hamiltonian_Cluster_Hs.c.");
            }

            is2[ID] = (int)atom_offset;
            atom_offset =
                HamiltonianClusterHs_CheckedAddCount(atom_offset, (size_t)My_Matomnum[ID], "global atom ordering");
        }

        if (atom_offset != (size_t)atomnum + 1u) {
            HamiltonianClusterHs_AbortWithMessage(
                "Inconsistent atom partitioning detected in Hamiltonian_Cluster_Hs.c.");
        }
    }

    for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        order_GA[is2[myid] + MA_AN - 1] = M2G[MA_AN];
    }

    for (ID = 0; ID < numprocs; ID++) {
        MPI_Bcast(&order_GA[is2[ID]], My_Matomnum[ID], MPI_INT, ID, mpi_comm_level1);
    }

    NUM = HamiltonianClusterHs_SetMPAndReturnNum(MP);
    if (NUM != n) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Orbital count mismatch in Hamiltonian_Cluster_Hs.c: n=%d, NUM=%d", n, NUM);
        HamiltonianClusterHs_AbortWithMessage(msg);
    }

    H1 = (double *)HamiltonianClusterHs_MallocArray((size_t)tnum + 1u, sizeof(double), "H1");

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
                    H1[k] = RH[MA_AN][LB_AN][i][j];
                    k++;
                }
            }
        }
    }

    for (ID = 0; ID < numprocs; ID++) {
        k = is1[ID];
        MPI_Bcast(&H1[k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
    }

    if (spin == myworld1) {
        local_count = HamiltonianClusterHs_CheckedMulCount((size_t)na_rows, (size_t)na_cols, "distributed Hs size");

        for (size_t idx = 0; idx < local_count; idx++) {
            Hs[idx] = 0.0;
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
                            HamiltonianClusterHs_AbortWithMessage(
                                "Global matrix index is out of range in Hamiltonian_Cluster_Hs.c.");
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
                                HamiltonianClusterHs_AbortWithMessage(
                                    "Distributed matrix index is out of range in Hamiltonian_Cluster_Hs.c.");
                            }

                            Hs[(size_t)(jl - 1) * (size_t)na_rows + (size_t)(il - 1)] += H1[k];
                        }

                        k++;
                    }
                }
            }
        }
    }

    free(H1);
    free(order_GA);
    free(is2);
    free(ie1);
    free(is1);
    free(My_Matomnum);
    free(My_NZeros);
}
