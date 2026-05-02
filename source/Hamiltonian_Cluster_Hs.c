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

static void HamiltonianClusterHs_PackLocalSegment(double **** RH, double *local_H1, int local_nzeros)
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
                    local_H1[k] = RH[MA_AN][LB_AN][i][j];
                    k++;
                }
            }
        }
    }

    if (k != local_nzeros) {
        HamiltonianClusterHs_AbortWithMessage("Packed H1 length mismatch in Hamiltonian_Cluster_Hs.c.");
    }
}

static int HamiltonianClusterHs_FindDenseRoot(int spin, int myworld1, int myid, int myid1)
{
    int dense_owner = (my_prow == 0 && my_pcol == 0);
    int local_root;
    int local_root_flag;
    int dense_root;
    int num_dense_roots;

    if ((myid1 == 0) != dense_owner) {
        HamiltonianClusterHs_AbortWithMessage(
            "Inconsistent dense-matrix owner mapping in Hamiltonian_Cluster_Hs.c.");
    }

    local_root_flag = (spin == myworld1 && dense_owner) ? 1 : 0;
    local_root      = local_root_flag ? myid : -1;
    dense_root      = -1;

    MPI_Allreduce(&local_root_flag, &num_dense_roots, 1, MPI_INT, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&local_root, &dense_root, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

    if (num_dense_roots != 1 || dense_root < 0) {
        HamiltonianClusterHs_AbortWithMessage("Failed to identify a unique dense root in Hamiltonian_Cluster_Hs.c.");
    }

    return dense_root;
}

static void HamiltonianClusterHs_BuildDenseFromGathered(const double *H1, const int *order_GA, int *MP, double *Hs,
                                                        int n, int tnum)
{
    int      AN, GA_AN, LB_AN, GB_AN;
    int      wanA, wanB, tnoA, tnoB;
    size_t * atom_offsets;
    size_t   dense_count;
    size_t   offset;

    atom_offsets = (size_t *)HamiltonianClusterHs_MallocArray((size_t)atomnum + 2u, sizeof(size_t), "atom offsets");

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
            atom_block_count = HamiltonianClusterHs_CheckedAddCount(
                atom_block_count,
                HamiltonianClusterHs_CheckedMulCount((size_t)tnoA, (size_t)tnoB, "atom block span"),
                "atom block span");
        }

        offset = HamiltonianClusterHs_CheckedAddCount(offset, atom_block_count, "dense assembly offsets");
    }
    atom_offsets[atomnum + 1] = offset;

    if (offset != (size_t)tnum) {
        HamiltonianClusterHs_AbortWithMessage("Gathered H1 size mismatch in Hamiltonian_Cluster_Hs.c.");
    }

    dense_count = HamiltonianClusterHs_CheckedMulCount((size_t)n, (size_t)n, "dense Hs size");
    memset(Hs, 0, HamiltonianClusterHs_CheckedMulCount(dense_count, sizeof(double), "dense Hs bytes"));

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
                        HamiltonianClusterHs_AbortWithMessage(
                            "Dense matrix index is out of range in Hamiltonian_Cluster_Hs.c.");
                    }

                    value = H1[k_local];
                    Hs[(size_t)(jg - 1) * (size_t)n + (size_t)(ig - 1)] += value;

                    k_local++;
                }
            }
        }

        if (k_local != atom_offsets[AN + 1]) {
            HamiltonianClusterHs_AbortWithMessage(
                "Atom-local H1 offsets are inconsistent in Hamiltonian_Cluster_Hs.c.");
        }
    }

    free(atom_offsets);
}

static void HamiltonianClusterHs_CuSolver(double **** RH, double * Hs, int * MP, int spin, MPI_Comm * MPI_CommWD1,
                                          int myworld1, int n)
{
    int    myid, myid1, numprocs;
    int    dense_root;
    int    NUM;
    int    local_nzeros;
    int    local_matomnum;
    int    dummy_int = 0;
    int *  recv_nzeros = NULL;
    int *  recv_matomnum = NULL;
    int *  h1_displs = NULL;
    int *  atom_displs = NULL;
    int *  order_GA = NULL;
    double *local_H1;
    double *gathered_H1 = NULL;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    MPI_Comm_rank(MPI_CommWD1[myworld1], &myid1);

    dense_root = HamiltonianClusterHs_FindDenseRoot(spin, myworld1, myid, myid1);

    NUM = HamiltonianClusterHs_SetMPAndReturnNum(MP);
    if (NUM != n) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Orbital count mismatch in Hamiltonian_Cluster_Hs.c: n=%d, NUM=%d", n, NUM);
        HamiltonianClusterHs_AbortWithMessage(msg);
    }

    local_nzeros  = HamiltonianClusterHs_ComputeLocalNZeroCount();
    local_matomnum = Matomnum;
    local_H1      = (double *)HamiltonianClusterHs_MallocArray((size_t)local_nzeros, sizeof(double), "local H1 segment");
    HamiltonianClusterHs_PackLocalSegment(RH, local_H1, local_nzeros);

    if (myid == dense_root) {
        recv_nzeros = (int *)HamiltonianClusterHs_MallocArray((size_t)numprocs, sizeof(int), "recv_nzeros");
        recv_matomnum =
            (int *)HamiltonianClusterHs_MallocArray((size_t)numprocs, sizeof(int), "recv_matomnum");
    }

    MPI_Gather(&local_nzeros, 1, MPI_INT, recv_nzeros, 1, MPI_INT, dense_root, mpi_comm_level1);
    MPI_Gather(&local_matomnum, 1, MPI_INT, recv_matomnum, 1, MPI_INT, dense_root, mpi_comm_level1);

    if (myid == dense_root) {
        int    ID;
        size_t total_nzeros = 0;
        size_t total_atoms  = 0;

        h1_displs   = (int *)HamiltonianClusterHs_MallocArray((size_t)numprocs, sizeof(int), "h1_displs");
        atom_displs = (int *)HamiltonianClusterHs_MallocArray((size_t)numprocs, sizeof(int), "atom_displs");

        for (ID = 0; ID < numprocs; ID++) {
            if (recv_nzeros[ID] < 0 || recv_matomnum[ID] < 0) {
                HamiltonianClusterHs_AbortWithMessage(
                    "Negative gather counts detected in Hamiltonian_Cluster_Hs.c.");
            }

            if (total_nzeros > (size_t)INT_MAX || total_atoms > (size_t)INT_MAX) {
                HamiltonianClusterHs_AbortWithMessage(
                    "Gather displacements exceed INT_MAX in Hamiltonian_Cluster_Hs.c.");
            }

            h1_displs[ID]   = (int)total_nzeros;
            atom_displs[ID] = (int)total_atoms;
            total_nzeros = HamiltonianClusterHs_CheckedAddCount(total_nzeros, (size_t)recv_nzeros[ID], "gathered H1 size");
            total_atoms  = HamiltonianClusterHs_CheckedAddCount(total_atoms, (size_t)recv_matomnum[ID], "gathered atom order");
        }

        if (total_nzeros > (size_t)INT_MAX) {
            HamiltonianClusterHs_AbortWithMessage("Gathered H1 size exceeds INT_MAX in Hamiltonian_Cluster_Hs.c.");
        }

        if (total_atoms != (size_t)atomnum) {
            HamiltonianClusterHs_AbortWithMessage(
                "Gathered atom order length mismatch in Hamiltonian_Cluster_Hs.c.");
        }

        gathered_H1 = (double *)HamiltonianClusterHs_MallocArray(total_nzeros, sizeof(double), "gathered H1");
        order_GA    = (int *)HamiltonianClusterHs_MallocArray((size_t)atomnum + 1u, sizeof(int), "order_GA");
    }

    MPI_Gatherv(local_H1, local_nzeros, MPI_DOUBLE, gathered_H1, recv_nzeros, h1_displs, MPI_DOUBLE, dense_root,
                mpi_comm_level1);
    MPI_Gatherv((0 < Matomnum) ? &M2G[1] : &dummy_int, Matomnum, MPI_INT, (myid == dense_root) ? &order_GA[1] : NULL,
                recv_matomnum, atom_displs, MPI_INT, dense_root, mpi_comm_level1);

    if (myid == dense_root) {
        int gathered_nzeros = 0;

        if (0 < numprocs) {
            gathered_nzeros = h1_displs[numprocs - 1] + recv_nzeros[numprocs - 1];
        }

        HamiltonianClusterHs_BuildDenseFromGathered(gathered_H1, order_GA, MP, Hs, n, gathered_nzeros);
    }

    free(gathered_H1);
    free(order_GA);
    free(atom_displs);
    free(h1_displs);
    free(recv_matomnum);
    free(recv_nzeros);
    free(local_H1);
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

    if (scf_eigen_lib_flag == CuSOLVER) {
        HamiltonianClusterHs_CuSolver(RH, Hs, MP, spin, MPI_CommWD1, myworld1, n);
        return;
    }

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
