/**********************************************************************
  Force.c:

     Force.c is a subroutine to calculate force on atoms.

  Log of Force.c:

     22/Nov/2001  Released by T. Ozaki
     18/Apr/2013  Force3() modified by A.M. Ito

***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define measure_time 0

#define NVHPC_VERSION ( __NVCOMPILER_MAJOR__ * 10000 \
                       + __NVCOMPILER_MINOR__ * 100   \
                       + __NVCOMPILER_PATCHLEVEL__ )

static void* Force_checked_malloc(size_t size, const char* file, int line)
{
    void* ptr;
    int mpi_initialized = 0;

    if (size == 0) {
        size = 1;
    }

    ptr = malloc(size);
    if (ptr != NULL) {
        return ptr;
    }

    fprintf(stderr, "Force.c: failed to allocate %zu bytes at %s:%d\n", size, file, line);
    fflush(stderr);

    if (MPI_Initialized(&mpi_initialized) == MPI_SUCCESS && mpi_initialized) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    abort();
}

static int Force_env_flag(const char* name, int default_value)
{
    const char* value = getenv(name);

    if (value == NULL || value[0] == '\0') {
        return default_value;
    }

    return atoi(value) != 0;
}

static int Force_collective_env_flag(const char* name, int default_value,
    MPI_Comm comm)
{
    int myid;
    int value = default_value;

    MPI_Comm_rank(comm, &myid);
    if (myid == 0) {
        value = Force_env_flag(name, default_value);
    }
    MPI_Bcast(&value, 1, MPI_INT, 0, comm);

    return value;
}

static void Force_profile_report(const char* label, double local_elapsed,
    MPI_Comm comm, int myid, int numprocs)
{
    double max_elapsed = 0.0;
    double sum_elapsed = 0.0;

    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_elapsed, &sum_elapsed, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (myid == 0) {
        printf("[Force profile] %-12s max=%10.6f s  avg=%10.6f s\n",
            label, max_elapsed, sum_elapsed / (double)numprocs);
        fflush(stdout);
    }
}

static int** Force_alloc_int_matrix(int rows, int cols)
{
    int i;
    int** matrix;
    int* data;

    matrix = (int**)malloc(sizeof(int*) * rows);
    data = (int*)malloc(sizeof(int) * rows * cols);

    for (i = 0; i < rows; i++) {
        matrix[i] = data + i * cols;
    }

    return matrix;
}

static void Force_free_int_matrix(int** matrix)
{
    if (matrix != NULL) {
        free(matrix[0]);
        free(matrix);
    }
}

static double** Force_alloc_double_matrix(int rows, int cols)
{
    int i;
    double** matrix;
    double* data;

    matrix = (double**)malloc(sizeof(double*) * rows);
    data = (double*)malloc(sizeof(double) * rows * cols);

    for (i = 0; i < rows; i++) {
        matrix[i] = data + i * cols;
    }

    return matrix;
}

static void Force_free_double_matrix(double** matrix)
{
    if (matrix != NULL) {
        free(matrix[0]);
        free(matrix);
    }
}

#define malloc(size) Force_checked_malloc((size), __FILE__, __LINE__)

static size_t Force_checked_mul_count(size_t a, size_t b, const char* label)
{
    if (a != 0 && b > SIZE_MAX / a) {
        fprintf(stderr, "Force.c: size overflow for %s\n", label);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    return a * b;
}

static size_t Force_checked_add_count(size_t a, size_t b, const char* label)
{
    if (a > SIZE_MAX - b) {
        fprintf(stderr, "Force.c: size overflow for %s\n", label);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    return a + b;
}

static void Force_accumulate_atom_time_from_terms(const size_t* atom_terms, size_t total_terms, double elapsed)
{
    int Mc_AN, Gc_AN;

    if (atom_terms == NULL || total_terms == 0 || elapsed <= 0.0) {
        return;
    }

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        time_per_atom[Gc_AN] += elapsed * ((double)atom_terms[Mc_AN] / (double)total_terms);
    }
}

/* Device memory used transiently by the force batches (packed item arrays,
   reduction buffers) is claimed through explicit acc_malloc arenas, never
   through data-clause allocation: on a device shared by several MPI ranks a
   feasibility check made at batch begin can go stale while peer ranks claim
   their own stage buffers, and an OpenACC data clause that then runs out of
   memory aborts the whole run.  acc_malloc reports NULL instead, letting the
   caller wait for the peers' transients or fall back to the host. */

static void Force_gpu_pool_flush(void)
{
    if (cudaDeviceSynchronize() == cudaSuccess) acc_clear_freelists();
}

static size_t Force_gpu_arena_off(size_t* pos, size_t bytes)
{
    size_t off = *pos;

    *pos = (off + bytes + 511U) & ~(size_t)511U;
    return off;
}

/* one shot; the caller has a host fallback */
static void* Force_gpu_arena_try(size_t bytes)
{
    void* arena = acc_malloc(bytes);

    if (arena == NULL) {
        /* our own freelist may hoard mismatched freed blocks */
        Force_gpu_pool_flush();
        arena = acc_malloc(bytes);
    }
    return arena;
}

/* mandatory claim: wait out the peers' transient spikes (they release on
   their own, so this always makes progress) and only give up when the
   device stays full for minutes */
static void* Force_gpu_arena_wait(size_t bytes, const char* what)
{
    const int max_wait_ms = 180000;
    int waited_ms = 0, warned = 0;

    for (;;) {
        void* arena = acc_malloc(bytes);

        if (arena != NULL) return arena;
        Force_gpu_pool_flush();

        if (!warned && 2000 <= waited_ms) {
            warned = 1;
            fprintf(stderr,
                "Force %s: waiting for %.1f MiB of device memory held by peer ranks...\n",
                what, (double)bytes / (1024.0 * 1024.0));
            fflush(stderr);
        }
        if (max_wait_ms <= waited_ms) {
            size_t free_b = 0, total_b = 0;

            (void)cudaMemGetInfo(&free_b, &total_b);
            fprintf(stderr,
                "Force %s: %.1f MiB of device memory did not free up within %d s (%.1f MiB free); "
                "reduce the MPI ranks sharing the device.\n",
                what, (double)bytes / (1024.0 * 1024.0), max_wait_ms / 1000,
                (double)free_b / (1024.0 * 1024.0));
            fflush(stderr);
            MPI_Abort(mpi_comm_level1, 1);
        }
        {
            struct timespec ts = { 0, 50 * 1000 * 1000 };

            nanosleep(&ts, NULL);
        }
        waited_ms += 50;
    }
}

static void Force_gpu_reduce_fallback_notice(size_t bytes)
{
    static int printed = 0;
    size_t free_b = 0, total_b = 0;

    if (printed) return;
    printed = 1;
    (void)cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr,
        "Force: a %.1f MiB GPU reduction buffer does not fit (%.1f MiB free); using the host for this reduction.\n",
        (double)bytes / (1024.0 * 1024.0), (double)free_b / (1024.0 * 1024.0));
    fflush(stderr);
}

static void Force_openacc_reduce_xyz(int num_atoms,
    const size_t* atom_offsets,
    size_t total_terms,
    const double* weights,
    const double* vx,
    const double* vy,
    const double* vz,
    double* Fx,
    double* Fy,
    double* Fz)
{
    int Mc_AN;
    size_t nvec, asz = 0;
    size_t o_off, o_w, o_vx, o_vy, o_vz, o_fx, o_fy, o_fz;
    char* arena;

    for (Mc_AN = 0; Mc_AN <= num_atoms; Mc_AN++) {
        Fx[Mc_AN] = 0.0;
        Fy[Mc_AN] = 0.0;
        Fz[Mc_AN] = 0.0;
    }

    if (total_terms == 0) {
        return;
    }

    nvec = (size_t)num_atoms + 1;
    o_off = Force_gpu_arena_off(&asz, sizeof(size_t) * nvec);
    o_w = Force_gpu_arena_off(&asz, sizeof(double) * total_terms);
    o_vx = Force_gpu_arena_off(&asz, sizeof(double) * total_terms);
    o_vy = Force_gpu_arena_off(&asz, sizeof(double) * total_terms);
    o_vz = Force_gpu_arena_off(&asz, sizeof(double) * total_terms);
    o_fx = Force_gpu_arena_off(&asz, sizeof(double) * nvec);
    o_fy = Force_gpu_arena_off(&asz, sizeof(double) * nvec);
    o_fz = Force_gpu_arena_off(&asz, sizeof(double) * nvec);

    arena = (char*)Force_gpu_arena_try(asz);

    if (arena != NULL) {
        acc_memcpy_to_device(arena + o_off, (void*)atom_offsets, sizeof(size_t) * nvec);
        acc_memcpy_to_device(arena + o_w, (void*)weights, sizeof(double) * total_terms);
        acc_memcpy_to_device(arena + o_vx, (void*)vx, sizeof(double) * total_terms);
        acc_memcpy_to_device(arena + o_vy, (void*)vy, sizeof(double) * total_terms);
        acc_memcpy_to_device(arena + o_vz, (void*)vz, sizeof(double) * total_terms);

        {
            const size_t* atom_offsets = (const size_t*)(void*)(arena + o_off);
            const double* weights = (const double*)(void*)(arena + o_w);
            const double* vx = (const double*)(void*)(arena + o_vx);
            const double* vy = (const double*)(void*)(arena + o_vy);
            const double* vz = (const double*)(void*)(arena + o_vz);
            double* Fx = (double*)(void*)(arena + o_fx);
            double* Fy = (double*)(void*)(arena + o_fy);
            double* Fz = (double*)(void*)(arena + o_fz);

#pragma acc parallel loop gang deviceptr(atom_offsets, weights, vx, vy, vz, Fx, Fy, Fz)
            for (Mc_AN = 0; Mc_AN <= num_atoms; Mc_AN++) {
                double sumx = 0.0;
                double sumy = 0.0;
                double sumz = 0.0;

                if (Mc_AN != 0) {
                    size_t start = atom_offsets[Mc_AN - 1];
                    size_t end = atom_offsets[Mc_AN];
                    size_t idx;

#pragma acc loop vector reduction(+:sumx, sumy, sumz)
                    for (idx = start; idx < end; idx++) {
                        double weight = weights[idx];
                        sumx += weight * vx[idx];
                        sumy += weight * vy[idx];
                        sumz += weight * vz[idx];
                    }
                }

                Fx[Mc_AN] = sumx;
                Fy[Mc_AN] = sumy;
                Fz[Mc_AN] = sumz;
            }
        }

        acc_memcpy_from_device(Fx, arena + o_fx, sizeof(double) * nvec);
        acc_memcpy_from_device(Fy, arena + o_fy, sizeof(double) * nvec);
        acc_memcpy_from_device(Fz, arena + o_fz, sizeof(double) * nvec);
        acc_free(arena);
        Force_gpu_pool_flush();
    } else {
        Force_gpu_reduce_fallback_notice(asz);

#pragma omp parallel for schedule(dynamic)
        for (Mc_AN = 1; Mc_AN <= num_atoms; Mc_AN++) {
            double sumx = 0.0;
            double sumy = 0.0;
            double sumz = 0.0;
            size_t start = atom_offsets[Mc_AN - 1];
            size_t end = atom_offsets[Mc_AN];
            size_t idx;

            for (idx = start; idx < end; idx++) {
                double weight = weights[idx];
                sumx += weight * vx[idx];
                sumy += weight * vy[idx];
                sumz += weight * vz[idx];
            }

            Fx[Mc_AN] = sumx;
            Fy[Mc_AN] = sumy;
            Fz[Mc_AN] = sumz;
        }
    }
}

static size_t Force2_Kinetic_OpenACC(double***** H0_src,
    double***** CDM0,
    double* Fx,
    double* Fy,
    double* Fz,
    size_t* atom_terms)
{
    int Mc_AN, Gc_AN, h_AN, q_AN;
    int Gh_AN, Mh_AN, Gq_AN;
    int Hwan, Qwan, i, j, kl;
    int ian, jan;
    size_t total_terms = 0;
    size_t* atom_offsets;
    double* weights;
    double* hx;
    double* hy;
    double* hz;
    size_t idx;

    atom_offsets = (size_t*)malloc(sizeof(size_t) * (Matomnum + 1));
    atom_offsets[0] = 0;
    if (atom_terms != NULL) {
        atom_terms[0] = 0;
    }

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        size_t atom_count = 0;

        Gc_AN = M2G[Mc_AN];
        h_AN = 0;
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        ian = Spe_Total_CNO[Hwan];

        for (q_AN = 0; q_AN <= FNAN[Gc_AN]; q_AN++) {
            Gq_AN = natn[Gc_AN][q_AN];
            Qwan = WhatSpecies[Gq_AN];
            jan = Spe_Total_CNO[Qwan];
            kl = RMI1[Mc_AN][h_AN][q_AN];

            if (0 <= kl) {
                atom_count = Force_checked_add_count(atom_count,
                    Force_checked_mul_count((size_t)ian, (size_t)jan, "Force2 atom block"),
                    "Force2 atom term count");
            }
        }

        total_terms = Force_checked_add_count(total_terms, atom_count, "Force2 total term count");
        atom_offsets[Mc_AN] = total_terms;
        if (atom_terms != NULL) {
            atom_terms[Mc_AN] = atom_count;
        }
    }

    if (total_terms == 0) {
        Force_openacc_reduce_xyz(Matomnum, atom_offsets, 0, NULL, NULL, NULL, NULL, Fx, Fy, Fz);
        free(atom_offsets);
        return 0;
    }

    weights = (double*)malloc(sizeof(double) * total_terms);
    hx = (double*)malloc(sizeof(double) * total_terms);
    hy = (double*)malloc(sizeof(double) * total_terms);
    hz = (double*)malloc(sizeof(double) * total_terms);

    idx = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        double pref;

        Gc_AN = M2G[Mc_AN];
        h_AN = 0;
        Gh_AN = natn[Gc_AN][h_AN];
        Mh_AN = F_G2M[Gh_AN];
        Hwan = WhatSpecies[Gh_AN];
        ian = Spe_Total_CNO[Hwan];

        for (q_AN = 0; q_AN <= FNAN[Gc_AN]; q_AN++) {
            Gq_AN = natn[Gc_AN][q_AN];
            Qwan = WhatSpecies[Gq_AN];
            jan = Spe_Total_CNO[Qwan];
            kl = RMI1[Mc_AN][h_AN][q_AN];

            if (kl < 0) {
                continue;
            }

            pref = (SpinP_switch == 0) ? ((q_AN == 0) ? 2.0 : 4.0) : ((q_AN == 0) ? 1.0 : 2.0);

            for (i = 0; i < ian; i++) {
                for (j = 0; j < jan; j++) {
                    if (SpinP_switch == 0) {
                        weights[idx] = pref * CDM0[0][Mh_AN][kl][i][j];
                    } else {
                        weights[idx] = pref * (CDM0[0][Mh_AN][kl][i][j] + CDM0[1][Mh_AN][kl][i][j]);
                    }

                    hx[idx] = H0_src[1][Mc_AN][q_AN][i][j];
                    hy[idx] = H0_src[2][Mc_AN][q_AN][i][j];
                    hz[idx] = H0_src[3][Mc_AN][q_AN][i][j];
                    idx++;
                }
            }
        }
    }

    Force_openacc_reduce_xyz(Matomnum, atom_offsets, total_terms, weights, hx, hy, hz, Fx, Fy, Fz);

    free(hz);
    free(hy);
    free(hx);
    free(weights);
    free(atom_offsets);

    return total_terms;
}

static size_t Force5_OpenACC(double***** OLP1,
    double***** EDM1,
    double* Fx,
    double* Fy,
    double* Fz,
    size_t* atom_terms)
{
    int Mc_AN, Gc_AN, Cwan, h_AN, Gh_AN, Hwan;
    int i, j;
    size_t total_terms = 0;
    size_t* atom_offsets;
    double* weights;
    double* hx;
    double* hy;
    double* hz;
    size_t idx;

    atom_offsets = (size_t*)malloc(sizeof(size_t) * (Matomnum + 1));
    atom_offsets[0] = 0;
    if (atom_terms != NULL) {
        atom_terms[0] = 0;
    }

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        size_t atom_count = 0;

        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 1; h_AN <= FNAN[Gc_AN]; h_AN++) {
            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];

            atom_count = Force_checked_add_count(atom_count,
                Force_checked_mul_count((size_t)Spe_Total_CNO[Cwan], (size_t)Spe_Total_CNO[Hwan], "Force5 atom block"),
                "Force5 atom term count");
        }

        total_terms = Force_checked_add_count(total_terms, atom_count, "Force5 total term count");
        atom_offsets[Mc_AN] = total_terms;
        if (atom_terms != NULL) {
            atom_terms[Mc_AN] = atom_count;
        }
    }

    if (total_terms == 0) {
        Force_openacc_reduce_xyz(Matomnum, atom_offsets, 0, NULL, NULL, NULL, NULL, Fx, Fy, Fz);
        free(atom_offsets);
        return 0;
    }

    weights = (double*)malloc(sizeof(double) * total_terms);
    hx = (double*)malloc(sizeof(double) * total_terms);
    hy = (double*)malloc(sizeof(double) * total_terms);
    hz = (double*)malloc(sizeof(double) * total_terms);

    idx = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 1; h_AN <= FNAN[Gc_AN]; h_AN++) {
            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];

            for (i = 0; i < Spe_Total_CNO[Cwan]; i++) {
                for (j = 0; j < Spe_Total_CNO[Hwan]; j++) {
                    if (SpinP_switch == 0) {
                        weights[idx] = -4.0 * EDM1[0][Mc_AN][h_AN][i][j];
                    } else {
                        weights[idx] = -2.0 * (EDM1[0][Mc_AN][h_AN][i][j] + EDM1[1][Mc_AN][h_AN][i][j]);
                    }

                    hx[idx] = OLP1[1][Mc_AN][h_AN][i][j];
                    hy[idx] = OLP1[2][Mc_AN][h_AN][i][j];
                    hz[idx] = OLP1[3][Mc_AN][h_AN][i][j];
                    idx++;
                }
            }
        }
    }

    Force_openacc_reduce_xyz(Matomnum, atom_offsets, total_terms, weights, hx, hy, hz, Fx, Fy, Fz);

    free(hz);
    free(hy);
    free(hx);
    free(weights);
    free(atom_offsets);

    return total_terms;
}

static size_t Force4_OpenACC(double* Fx,
    double* Fy,
    double* Fz,
    size_t* atom_terms)
{
    int Mc_AN, Gc_AN, Cwan, Nc, GNc, GRc, MNc;
    size_t total_terms = 0;
    size_t* atom_offsets;
    double* weights;
    double* vx;
    double* vy;
    double* vz;
    size_t idx;

    atom_offsets = (size_t*)malloc(sizeof(size_t) * (Matomnum + 1));
    atom_offsets[0] = 0;
    if (atom_terms != NULL) {
        atom_terms[0] = 0;
    }

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        size_t atom_count;

        Gc_AN = M2G[Mc_AN];
        atom_count = (size_t)GridN_Atom[Gc_AN];
        total_terms = Force_checked_add_count(total_terms, atom_count, "Force4 total term count");
        atom_offsets[Mc_AN] = total_terms;
        if (atom_terms != NULL) {
            atom_terms[Mc_AN] = atom_count;
        }
    }

    if (total_terms == 0) {
        Force_openacc_reduce_xyz(Matomnum, atom_offsets, 0, NULL, NULL, NULL, NULL, Fx, Fy, Fz);
        free(atom_offsets);
        return 0;
    }

    weights = (double*)malloc(sizeof(double) * total_terms);
    vx = (double*)malloc(sizeof(double) * total_terms);
    vy = (double*)malloc(sizeof(double) * total_terms);
    vz = (double*)malloc(sizeof(double) * total_terms);

    idx = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

        for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {
            double Cxyz[4];
            double x, y, z;
            double dx, dy, dz, r;
            double dvx, dvy, dvz;
            double tmp0, den;

            GNc = GridListAtom[Mc_AN][Nc];
            GRc = CellListAtom[Mc_AN][Nc];
            MNc = MGridListAtom[Mc_AN][Nc];

            Get_Grid_XYZ(GNc, Cxyz);
            x = Cxyz[1] + atv[GRc][1];
            y = Cxyz[2] + atv[GRc][2];
            z = Cxyz[3] + atv[GRc][3];
            dx = Gxyz[Gc_AN][1] - x;
            dy = Gxyz[Gc_AN][2] - y;
            dz = Gxyz[Gc_AN][3] - z;
            r = sqrt(dx * dx + dy * dy + dz * dz);

            if (r < 1.0e-10) {
                r = 1.0e-10;
            }

            if (1.0e-14 < r) {
                tmp0 = Dr_VNAF(Cwan, r);
                dvx = tmp0 * dx / r;
                dvy = tmp0 * dy / r;
                dvz = tmp0 * dz / r;
            } else {
                dvx = 0.0;
                dvy = 0.0;
                dvz = 0.0;
            }

            den = Density_Grid[0][MNc] + Density_Grid[1][MNc];
            weights[idx] = den * GridVol;
            vx[idx] = dvx;
            vy[idx] = dvy;
            vz[idx] = dvz;
            idx++;
        }
    }

    Force_openacc_reduce_xyz(Matomnum, atom_offsets, total_terms, weights, vx, vy, vz, Fx, Fy, Fz);

    free(vz);
    free(vy);
    free(vx);
    free(weights);
    free(atom_offsets);

    return total_terms;
}

static void Force_openacc_reduce_terms(size_t total_terms,
    const double* weights,
    const double* vx,
    const double* vy,
    const double* vz,
    double* sumx,
    double* sumy,
    double* sumz)
{
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    size_t idx;

    if (sumx != NULL) {
        *sumx = 0.0;
    }
    if (sumy != NULL) {
        *sumy = 0.0;
    }
    if (sumz != NULL) {
        *sumz = 0.0;
    }

    if (total_terms == 0) {
        return;
    }

    {
        size_t asz = 0;
        const size_t o_w = Force_gpu_arena_off(&asz, sizeof(double) * total_terms);
        const size_t o_vx = Force_gpu_arena_off(&asz, sizeof(double) * total_terms);
        const size_t o_vy = Force_gpu_arena_off(&asz, sizeof(double) * total_terms);
        const size_t o_vz = Force_gpu_arena_off(&asz, sizeof(double) * total_terms);
        char* arena = (char*)Force_gpu_arena_try(asz);

        if (arena != NULL) {
            acc_memcpy_to_device(arena + o_w, (void*)weights, sizeof(double) * total_terms);
            acc_memcpy_to_device(arena + o_vx, (void*)vx, sizeof(double) * total_terms);
            acc_memcpy_to_device(arena + o_vy, (void*)vy, sizeof(double) * total_terms);
            acc_memcpy_to_device(arena + o_vz, (void*)vz, sizeof(double) * total_terms);

            {
                const double* weights = (const double*)(void*)(arena + o_w);
                const double* vx = (const double*)(void*)(arena + o_vx);
                const double* vy = (const double*)(void*)(arena + o_vy);
                const double* vz = (const double*)(void*)(arena + o_vz);

#pragma acc parallel loop reduction(+:sx, sy, sz) deviceptr(weights, vx, vy, vz)
                for (idx = 0; idx < total_terms; idx++) {
                    double weight = weights[idx];
                    sx += weight * vx[idx];
                    sy += weight * vy[idx];
                    sz += weight * vz[idx];
                }
            }

            /* no pool flush here: these reductions repeat with the same
               few arena sizes (per-pair calls), so freelist reuse matters */
            acc_free(arena);
        } else {
            Force_gpu_reduce_fallback_notice(asz);

            for (idx = 0; idx < total_terms; idx++) {
                double weight = weights[idx];
                sx += weight * vx[idx];
                sy += weight * vy[idx];
                sz += weight * vz[idx];
            }
        }
    }

    if (sumx != NULL) {
        *sumx = sx;
    }
    if (sumy != NULL) {
        *sumy = sy;
    }
    if (sumz != NULL) {
        *sumz = sz;
    }
}

static size_t Force4B_pack_trace_terms(double***** CDM0,
    int Mh_AN,
    int kl,
    int ian,
    int jan,
    double pref,
    double** HVNAx,
    double** HVNAy,
    double** HVNAz,
    size_t idx,
    double* weights,
    double* hx,
    double* hy,
    double* hz)
{
    int i, j;

    for (i = 0; i < ian; i++) {
        double* cdm0_row = CDM0[0][Mh_AN][kl][i];
        double* hvnax_row = HVNAx[i];
        double* hvnay_row = HVNAy[i];
        double* hvnaz_row = HVNAz[i];

        if (SpinP_switch == 0) {
            for (j = 0; j < jan; j++) {
                weights[idx] = pref * cdm0_row[j];
                hx[idx] = hvnax_row[j];
                hy[idx] = hvnay_row[j];
                hz[idx] = hvnaz_row[j];
                idx++;
            }
        } else {
            double* cdm1_row = CDM0[1][Mh_AN][kl][i];

            for (j = 0; j < jan; j++) {
                weights[idx] = pref * (cdm0_row[j] + cdm1_row[j]);
                hx[idx] = hvnax_row[j];
                hy[idx] = hvnay_row[j];
                hz[idx] = hvnaz_row[j];
                idx++;
            }
        }
    }

    return idx;
}

static void Force3_gpu_abort(const char* message)
{
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1, 1);
}

/* Batched device evaluation of the part-2 trace of Force3:
   sum over the overlap grid points of every (atom, neighbour) pair of
   Tr[ DM * (dChi0 x Chi1) ] * Vpot.  The per-pair partial forces come back
   to the host and are accumulated per atom in pair order.  Everything here
   is rank-local (no MPI collectives), so each rank decides independently. */

typedef struct {
    size_t dchi_base;
    size_t vpot_base;
    size_t orb1_base;
    size_t dm_off;
    int    vpot_stride;
    int    no0;
    int    no1;
    int    glist_off;
    int    n_olg;
    int    orb1_fnan;
    int    atom;
} Force3GpuPair;

/* Per-rank chunk budget for the batched trace, or zero when the rank should
   stay on the CPU path.  Every rank on the node shares one device here, so
   the resident orbital table plus one transient chunk of every sharing rank
   must fit next to the reserve.  Only the leading checks are collective;
   the returned decision may differ between ranks (the trace path itself has
   no MPI). */
static size_t Force3_GpuChunkBudget(void)
{
    int Mc_AN, node_ranks = 1;
    size_t orbs_bytes = 0, grid_pts = 0, resident_bytes;
    size_t free_bytes = 0, total_bytes = 0;
    size_t usable, budget;
    /* the chunk scratch is mirrored on the host (gl/fn/dm packing), so a
       large cap multiplies host memory by the rank count; keep it small */
    const size_t budget_cap = (size_t)384 * 1024 * 1024;
    const size_t budget_floor = (size_t)192 * 1024 * 1024;
    const size_t reserve = (size_t)256 * 1024 * 1024;
    MPI_Comm node_comm = MPI_COMM_NULL;

    if (scf_eigen_lib_flag != GPUSOLVER) return 0;
    if (!Force_collective_env_flag("OPENMX_FORCE3_GPU", 1, mpi_comm_level1)) return 0;

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_size(node_comm, &node_ranks);
    MPI_Comm_free(&node_comm);
    if (node_ranks < 1) node_ranks = 1;

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        orbs_bytes += (size_t)GridN_Atom[Gc_AN] * (size_t)Spe_Total_CNO[WhatSpecies[Gc_AN]]
            * sizeof(float);
        grid_pts += (size_t)GridN_Atom[Gc_AN];
    }
    /* device-resident tables: the orbital table, the per-spin grid
       potentials, and (with on-device derivatives) the grid coordinates */
    resident_bytes = orbs_bytes
        + grid_pts * (size_t)(SpinP_switch + 1) * sizeof(double)
        + grid_pts * 3 * sizeof(double);

    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return 0;
    if (free_bytes <= reserve) return 0;

    usable = (free_bytes - reserve) / (size_t)node_ranks;
    if (usable <= resident_bytes + budget_floor) return 0;

    budget = usable - resident_bytes;
    if (budget_cap < budget) budget = budget_cap;
    return budget;
}

/* caps of the on-device orbital-derivative evaluation */
#define F3_DORB_L0MAX 3
#define F3_DORB_MULMAX 8

/* The device evaluation covers uncontracted bases with L0 <= 3 (the
   explicit real-harmonic formulas); anything else keeps the host
   Get_dOrbitals path with the derivative upload. */
static int Force3_GpuDorbMode(void)
{
    int w, L0;

    if (Cnt_switch != 0) return 0;
    if (!Force_env_flag("OPENMX_FORCE3_GPU_DORB", 1)) return 0;

    for (w = 0; w < SpeciesNum; w++) {
        if (F3_DORB_L0MAX < Spe_MaxL_Basis[w]) return 0;
        for (L0 = 0; L0 <= Spe_MaxL_Basis[w]; L0++) {
            if (F3_DORB_MULMAX < Spe_Num_Basis[w][L0]) return 0;
        }
        if (Spe_Num_Mesh_PAO[w] < 6) return 0;
    }
    return 1;
}

static void Force3_GpuTrace(const double* dchi_all, const size_t* dchi_off,
    double* vpot_dev_full, const size_t* vpot_off,
    double* xyz_dev_full, const size_t* xyz_off, int dorb_mode,
    size_t chunk_budget)
{
    int Mc_AN, h_AN, myid;
    size_t orbs_local_count = 0;
    size_t* orb_off;
    float* orbs_dev;
    float* orb_stage;
    int chunk_lo;
    const int spins = SpinP_switch + 1;

    /* per-species PAO tables of the on-device orbital derivatives */
    double* pao_rv = NULL;
    double* pao_rwf = NULL;
    size_t* rv_off = NULL;
    size_t* rwf_base = NULL;
    int* sp_mesh = NULL;
    int* sp_maxl = NULL;
    int* sp_nb = NULL;
    size_t pao_rv_count = 0, pao_rwf_count = 0;

    MPI_Comm_rank(mpi_comm_level1, &myid);

    /* ---- flatten the local orbital table (float) into a device-resident
            buffer, one atom block at a time through a staging buffer ---- */

    orb_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)(Matomnum + 2),
        __FILE__, __LINE__);

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int NO0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        orb_off[Mc_AN] = orbs_local_count;
        orbs_local_count += (size_t)GridN_Atom[Gc_AN] * (size_t)NO0;
    }
    orb_off[Matomnum + 1] = orbs_local_count;

    orbs_dev = (float*)Force_gpu_arena_wait(
        sizeof(float) * (orbs_local_count == 0 ? 1 : orbs_local_count),
        "GPU trace orbital table");
    {
        size_t maxorb = 0;

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            size_t cnt = orb_off[Mc_AN + 1] - orb_off[Mc_AN];
            if (maxorb < cnt) maxorb = cnt;
        }
        orb_stage = (float*)Force_checked_malloc(
            sizeof(float) * (maxorb == 0 ? 1 : maxorb), __FILE__, __LINE__);
    }

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int NO0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
        size_t cnt = orb_off[Mc_AN + 1] - orb_off[Mc_AN];
        int Nc;

#pragma omp parallel for schedule(dynamic)
        for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {
            const Type_Orbs_Grid* src = Orbs_Grid[Mc_AN][Nc];
            float* dst = orb_stage + (size_t)Nc * (size_t)NO0;
            int i;

            for (i = 0; i < NO0; i++) dst[i] = (float)src[i];
        }

        if (cnt != 0) {
            acc_memcpy_to_device(orbs_dev + orb_off[Mc_AN], orb_stage, sizeof(float) * cnt);
        }
    }

    /* ---- flatten the PAO radial tables when the device evaluates the
            orbital derivatives itself ---- */

    if (dorb_mode) {
        int w, L0, Mul0, i;

        rv_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)(SpeciesNum + 1), __FILE__, __LINE__);
        rwf_base = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)(SpeciesNum + 1), __FILE__, __LINE__);
        sp_mesh = (int*)Force_checked_malloc(sizeof(int) * (size_t)SpeciesNum, __FILE__, __LINE__);
        sp_maxl = (int*)Force_checked_malloc(sizeof(int) * (size_t)SpeciesNum, __FILE__, __LINE__);
        sp_nb = (int*)Force_checked_malloc(sizeof(int) * (size_t)SpeciesNum * (F3_DORB_L0MAX + 1),
            __FILE__, __LINE__);

        for (w = 0; w < SpeciesNum; w++) {
            int nb_tot = 0;

            rv_off[w] = pao_rv_count;
            rwf_base[w] = pao_rwf_count;
            sp_mesh[w] = Spe_Num_Mesh_PAO[w];
            sp_maxl[w] = Spe_MaxL_Basis[w];
            for (L0 = 0; L0 <= F3_DORB_L0MAX; L0++) {
                int nb = (L0 <= Spe_MaxL_Basis[w]) ? Spe_Num_Basis[w][L0] : 0;

                sp_nb[w * (F3_DORB_L0MAX + 1) + L0] = nb;
                nb_tot += nb;
            }
            pao_rv_count += (size_t)Spe_Num_Mesh_PAO[w];
            pao_rwf_count += (size_t)nb_tot * (size_t)Spe_Num_Mesh_PAO[w];
        }
        rv_off[SpeciesNum] = pao_rv_count;
        rwf_base[SpeciesNum] = pao_rwf_count;

        pao_rv = (double*)Force_checked_malloc(sizeof(double) * (pao_rv_count == 0 ? 1 : pao_rv_count),
            __FILE__, __LINE__);
        pao_rwf = (double*)Force_checked_malloc(sizeof(double) * (pao_rwf_count == 0 ? 1 : pao_rwf_count),
            __FILE__, __LINE__);

        for (w = 0; w < SpeciesNum; w++) {
            size_t rpos = rwf_base[w];

            for (i = 0; i < Spe_Num_Mesh_PAO[w]; i++) {
                pao_rv[rv_off[w] + (size_t)i] = Spe_PAO_RV[w][i];
            }
            for (L0 = 0; L0 <= Spe_MaxL_Basis[w]; L0++) {
                for (Mul0 = 0; Mul0 < Spe_Num_Basis[w][L0]; Mul0++) {
                    for (i = 0; i < Spe_Num_Mesh_PAO[w]; i++) {
                        pao_rwf[rpos++] = Spe_PAO_RWF[w][L0][Mul0][i];
                    }
                }
            }
        }
    }

    {
        const size_t rv_n = (pao_rv_count == 0 ? 1 : pao_rv_count);
        const size_t rwf_n = (pao_rwf_count == 0 ? 1 : pao_rwf_count);
        const size_t spn = (size_t)SpeciesNum;
        const size_t spl = (size_t)SpeciesNum * (F3_DORB_L0MAX + 1);
        double* rv_p = (dorb_mode ? pao_rv : (double*)orb_off);
        double* rwf_p = (dorb_mode ? pao_rwf : (double*)orb_off);
        size_t* rvo_p = (dorb_mode ? rv_off : orb_off);
        size_t* rwb_p = (dorb_mode ? rwf_base : orb_off);
        int* spm_p = (dorb_mode ? sp_mesh : (int*)orb_off);
        int* spx_p = (dorb_mode ? sp_maxl : (int*)orb_off);
        int* spb_p = (dorb_mode ? sp_nb : (int*)orb_off);
        const size_t rv_map = (dorb_mode ? rv_n : 1);
        const size_t rwf_map = (dorb_mode ? rwf_n : 1);
        const size_t sp_map = (dorb_mode ? spn : 1);
        const size_t spl_map = (dorb_mode ? spl : 1);
        const size_t spo_map = (dorb_mode ? spn + 1 : 1);

        /* ---- plan the chunk partition up front; every chunk then reuses one
                set of device buffers sized to the largest chunk.  Per-chunk
                data regions would strand one pool block per distinct size (the
                NVHPC freelist reuses exact sizes only), and the node ranks
                sharing the device run it out of memory. ---- */

        int n_chunks = 0;
        int* chunk_bound = (int*)Force_checked_malloc(sizeof(int) * (size_t)(Matomnum + 2), __FILE__, __LINE__);
        int* plan_pairs = (int*)Force_checked_malloc(sizeof(int) * (size_t)(Matomnum + 1), __FILE__, __LINE__);
        size_t* plan_gl = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)(Matomnum + 1), __FILE__, __LINE__);
        size_t* plan_fn = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)(Matomnum + 1), __FILE__, __LINE__);
        size_t* plan_dm = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)(Matomnum + 1), __FILE__, __LINE__);
        size_t max_pairs = 0, max_gl = 0, max_fn = 0, max_dm = 0;
        size_t max_dchi = 0, max_npts = 0;
        size_t max_atoms = 0;

        chunk_bound[0] = 1;
        chunk_lo = 1;
        while (chunk_lo <= Matomnum) {

            /* ---- grow a chunk of consecutive atoms under the budget ---- */

            int chunk_hi = chunk_lo; /* exclusive */
            int npairs = 0;
            size_t gl_count = 0, fn_count = 0, dm_count = 0;

            while (chunk_hi <= Matomnum) {
                int Gc_AN = M2G[chunk_hi];
                int NO0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
                size_t a_gl = 0, a_fn = 0, a_dm = 0;
                size_t chunk_bytes;
                int a_pairs = 0;

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    int Gh_AN = natn[Gc_AN][h_AN];
                    int NO1 = Spe_Total_CNO[WhatSpecies[Gh_AN]];
                    size_t nolg = (size_t)NumOLG[chunk_hi][h_AN];

                    a_pairs++;
                    a_gl += nolg;
                    a_dm += (size_t)spins * (size_t)NO0 * (size_t)NO1;
                    if (G2ID[Gh_AN] != myid) a_fn += nolg * (size_t)NO1;
                }

                chunk_bytes = (gl_count + a_gl) * 2 * sizeof(int)
                    + (fn_count + a_fn) * sizeof(float)
                    + (dm_count + a_dm) * sizeof(double)
                    + (dchi_off[chunk_hi + 1] - dchi_off[chunk_lo]) * sizeof(double);

                if (chunk_lo < chunk_hi && chunk_budget < chunk_bytes) break;

                gl_count += a_gl;
                fn_count += a_fn;
                dm_count += a_dm;
                npairs += a_pairs;
                chunk_hi++;
            }

            plan_pairs[n_chunks] = npairs;
            plan_gl[n_chunks] = gl_count;
            plan_fn[n_chunks] = fn_count;
            plan_dm[n_chunks] = dm_count;
            n_chunks++;
            chunk_bound[n_chunks] = chunk_hi;

            if (max_pairs < (size_t)npairs) max_pairs = (size_t)npairs;
            if (max_gl < gl_count) max_gl = gl_count;
            if (max_fn < fn_count) max_fn = fn_count;
            if (max_dm < dm_count) max_dm = dm_count;
            {
                const size_t c_dchi = dchi_off[chunk_hi] - dchi_off[chunk_lo];
                const size_t c_npts = (xyz_off[chunk_hi] - xyz_off[chunk_lo]) / 3;

                if (max_dchi < c_dchi) max_dchi = c_dchi;
                if (max_npts < c_npts) max_npts = c_npts;
                if (max_atoms < (size_t)(chunk_hi - chunk_lo)) max_atoms = (size_t)(chunk_hi - chunk_lo);
            }

            chunk_lo = chunk_hi;
        }

        {
            const size_t pm_n = (max_pairs == 0 ? 1 : max_pairs);
            const size_t glb_n = (max_gl == 0 ? 1 : max_gl);
            const size_t fnb_n = (max_fn == 0 ? 1 : max_fn);
            const size_t dmb_n = (max_dm == 0 ? 1 : max_dm);
            const size_t pf_n = 3 * pm_n;
            const size_t pt_n = ((dorb_mode && max_npts != 0) ? max_npts : 1);
            const size_t am_n = ((dorb_mode && max_atoms != 0) ? max_atoms : 1);
            Force3GpuPair* pm = (Force3GpuPair*)Force_checked_malloc(
                sizeof(Force3GpuPair) * pm_n, __FILE__, __LINE__);
            int* pm_h = (int*)Force_checked_malloc(sizeof(int) * pm_n, __FILE__, __LINE__);
            int* gl1 = (int*)Force_checked_malloc(sizeof(int) * glb_n, __FILE__, __LINE__);
            int* gl2 = (int*)Force_checked_malloc(sizeof(int) * glb_n, __FILE__, __LINE__);
            float* fn_buf = (float*)Force_checked_malloc(sizeof(float) * fnb_n, __FILE__, __LINE__);
            double* dm_buf = (double*)Force_checked_malloc(sizeof(double) * dmb_n, __FILE__, __LINE__);
            double* pair_f = (double*)Force_checked_malloc(sizeof(double) * pf_n, __FILE__, __LINE__);
            int* pt_aidx = (int*)Force_checked_malloc(sizeof(int) * pt_n, __FILE__, __LINE__);
            int* a_wan = (int*)Force_checked_malloc(sizeof(int) * am_n, __FILE__, __LINE__);
            int* a_no0 = (int*)Force_checked_malloc(sizeof(int) * am_n, __FILE__, __LINE__);
            size_t* a_dchi = (size_t*)Force_checked_malloc(sizeof(size_t) * am_n, __FILE__, __LINE__);
            size_t* a_pt0 = (size_t*)Force_checked_malloc(sizeof(size_t) * am_n, __FILE__, __LINE__);
            double* dchi_dev;
            int ic;

            /* one arena claim covers the whole chunk-loop footprint
               (scratch, staged chunk buffers, radial tables).  A data
               region here would allocate piecemeal and abort fatally if
               a peer rank grabbed the memory in between; the arena wait
               is non-fatal and the kernels read the same-name shadow
               pointers below, so the kernel bodies stay verbatim */
            size_t arena_pos = 0;
            const size_t o_dchi = Force_gpu_arena_off(&arena_pos, (max_dchi == 0 ? 1 : max_dchi) * sizeof(double));
            const size_t o_rv   = Force_gpu_arena_off(&arena_pos, sizeof(*rv_p) * rv_map);
            const size_t o_rwf  = Force_gpu_arena_off(&arena_pos, sizeof(*rwf_p) * rwf_map);
            const size_t o_rvo  = Force_gpu_arena_off(&arena_pos, sizeof(*rvo_p) * spo_map);
            const size_t o_rwb  = Force_gpu_arena_off(&arena_pos, sizeof(*rwb_p) * spo_map);
            const size_t o_spm  = Force_gpu_arena_off(&arena_pos, sizeof(*spm_p) * sp_map);
            const size_t o_spx  = Force_gpu_arena_off(&arena_pos, sizeof(*spx_p) * sp_map);
            const size_t o_spb  = Force_gpu_arena_off(&arena_pos, sizeof(*spb_p) * spl_map);
            const size_t o_pm   = Force_gpu_arena_off(&arena_pos, sizeof(Force3GpuPair) * pm_n);
            const size_t o_gl1  = Force_gpu_arena_off(&arena_pos, sizeof(int) * glb_n);
            const size_t o_gl2  = Force_gpu_arena_off(&arena_pos, sizeof(int) * glb_n);
            const size_t o_fn   = Force_gpu_arena_off(&arena_pos, sizeof(float) * fnb_n);
            const size_t o_dm   = Force_gpu_arena_off(&arena_pos, sizeof(double) * dmb_n);
            const size_t o_pf   = Force_gpu_arena_off(&arena_pos, sizeof(double) * pf_n);
            const size_t o_pt   = Force_gpu_arena_off(&arena_pos, sizeof(int) * pt_n);
            const size_t o_awan = Force_gpu_arena_off(&arena_pos, sizeof(int) * am_n);
            const size_t o_ano0 = Force_gpu_arena_off(&arena_pos, sizeof(int) * am_n);
            const size_t o_adchi = Force_gpu_arena_off(&arena_pos, sizeof(size_t) * am_n);
            const size_t o_apt0 = Force_gpu_arena_off(&arena_pos, sizeof(size_t) * am_n);
            unsigned char* chunk_arena =
                (unsigned char*)Force_gpu_arena_wait(arena_pos, "GPU trace chunk buffers");

            /* the radial tables are read-only for the whole chunk loop */
            acc_memcpy_to_device(chunk_arena + o_rv,  rv_p,  sizeof(*rv_p) * rv_map);
            acc_memcpy_to_device(chunk_arena + o_rwf, rwf_p, sizeof(*rwf_p) * rwf_map);
            acc_memcpy_to_device(chunk_arena + o_rvo, rvo_p, sizeof(*rvo_p) * spo_map);
            acc_memcpy_to_device(chunk_arena + o_rwb, rwb_p, sizeof(*rwb_p) * spo_map);
            acc_memcpy_to_device(chunk_arena + o_spm, spm_p, sizeof(*spm_p) * sp_map);
            acc_memcpy_to_device(chunk_arena + o_spx, spx_p, sizeof(*spx_p) * sp_map);
            acc_memcpy_to_device(chunk_arena + o_spb, spb_p, sizeof(*spb_p) * spl_map);

            dchi_dev = (double*)(void*)(chunk_arena + o_dchi);

    {
        for (ic = 0; ic < n_chunks; ic++) {

            {
                const int chunk_hi = chunk_bound[ic + 1];
                const int npairs = plan_pairs[ic];
                const size_t gl_count = plan_gl[ic];
                const size_t fn_count = plan_fn[ic];
                const size_t dm_count = plan_dm[ic];
                size_t gl_pos = 0, fn_pos = 0, dm_pos = 0;
                int p = 0, mc;
                size_t dchi_lo, dchi_len;

                chunk_lo = chunk_bound[ic];
                dchi_lo = dchi_off[chunk_lo];
                dchi_len = dchi_off[chunk_hi] - dchi_off[chunk_lo];

                /* offsets first (sequential), the bulk copies in parallel */

                for (mc = chunk_lo; mc < chunk_hi; mc++) {
                    int Gc_AN = M2G[mc];
                    int NO0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];

                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        int Gh_AN = natn[Gc_AN][h_AN];
                        int Mh_AN = F_G2M[Gh_AN];
                        int NO1 = Spe_Total_CNO[WhatSpecies[Gh_AN]];
                        int nolg = NumOLG[mc][h_AN];
                        int is_fnan = (G2ID[Gh_AN] != myid);

                        pm[p].dchi_base = dchi_off[mc] - dchi_lo;
                        /* vpot lives in the full device-resident buffer,
                           so its base offset is global, not chunk-relative */
                        pm[p].vpot_base = vpot_off[mc];
                        pm[p].vpot_stride = GridN_Atom[Gc_AN];
                        pm[p].no0 = NO0;
                        pm[p].no1 = NO1;
                        pm[p].glist_off = (int)gl_pos;
                        pm[p].n_olg = nolg;
                        pm[p].orb1_fnan = is_fnan;
                        pm[p].orb1_base = is_fnan ? fn_pos : orb_off[Mh_AN];
                        pm[p].dm_off = dm_pos;
                        pm[p].atom = mc;
                        pm_h[p] = h_AN;

                        gl_pos += (size_t)nolg;
                        if (is_fnan) fn_pos += (size_t)nolg * (size_t)NO1;
                        dm_pos += (size_t)spins * (size_t)NO0 * (size_t)NO1;
                        p++;
                    }
                }

                if (p != npairs || gl_pos != gl_count || fn_pos != fn_count || dm_pos != dm_count) {
                    Force3_gpu_abort("Force3 GPU trace: inconsistent chunk assembly.");
                }

#pragma omp parallel for schedule(dynamic)
                for (p = 0; p < npairs; p++) {
                    const int mc2 = pm[p].atom;
                    const int h2 = pm_h[p];
                    const int NO0 = pm[p].no0;
                    const int NO1 = pm[p].no1;
                    const int nolg = pm[p].n_olg;
                    size_t pos;

                    memcpy(gl1 + pm[p].glist_off, GListTAtoms1[mc2][h2], sizeof(int) * (size_t)nolg);
                    memcpy(gl2 + pm[p].glist_off, GListTAtoms2[mc2][h2], sizeof(int) * (size_t)nolg);

                    if (pm[p].orb1_fnan) {
                        int Nog, j;

                        for (Nog = 0; Nog < nolg; Nog++) {
                            const Type_Orbs_Grid* src = Orbs_Grid_FNAN[mc2][h2][Nog];
                            float* dst = fn_buf + pm[p].orb1_base + (size_t)Nog * (size_t)NO1;

                            for (j = 0; j < NO1; j++) dst[j] = (float)src[j];
                        }
                    }

                    pos = pm[p].dm_off;
                    {
                        int spin, i;

                        for (spin = 0; spin < spins; spin++) {
                            for (i = 0; i < NO0; i++) {
                                memcpy(dm_buf + pos, DM[0][spin][mc2][h2][i],
                                    sizeof(double) * (size_t)NO1);
                                pos += (size_t)NO1;
                            }
                        }
                    }
                }

                /* ---- stage the chunk into the fixed device buffers ---- */

                {
                    const size_t gl_u = (gl_count == 0 ? 1 : gl_count);
                    const size_t fn_u = (fn_count == 0 ? 1 : fn_count);
                    const size_t dm_u = (dm_count == 0 ? 1 : dm_count);
                    const size_t npairs_u = (size_t)(npairs == 0 ? 1 : npairs);

                    acc_memcpy_to_device(chunk_arena + o_pm,  pm,     sizeof(Force3GpuPair) * npairs_u);
                    acc_memcpy_to_device(chunk_arena + o_gl1, gl1,    sizeof(int) * gl_u);
                    acc_memcpy_to_device(chunk_arena + o_gl2, gl2,    sizeof(int) * gl_u);
                    acc_memcpy_to_device(chunk_arena + o_fn,  fn_buf, sizeof(float) * fn_u);
                    acc_memcpy_to_device(chunk_arena + o_dm,  dm_buf, sizeof(double) * dm_u);
                }

                /* ---- the per-chunk orbital derivatives ---- */

                if (dorb_mode) {

                    {
                        int chunk_atoms = chunk_hi - chunk_lo;
                        size_t npts = (xyz_off[chunk_hi] - xyz_off[chunk_lo]) / 3;
                        const size_t xyz_lo = xyz_off[chunk_lo];
                        double* xyz_chunk = xyz_dev_full + xyz_lo;
                        int a;

                        for (a = 0; a < chunk_atoms; a++) {
                            int mc3 = chunk_lo + a;
                            int Gc_AN = M2G[mc3];
                            size_t p0 = (xyz_off[mc3] - xyz_lo) / 3;
                            size_t pe = (xyz_off[mc3 + 1] - xyz_lo) / 3;
                            size_t q;

                            a_wan[a] = WhatSpecies[Gc_AN];
                            a_no0[a] = Spe_Total_CNO[WhatSpecies[Gc_AN]];
                            a_dchi[a] = dchi_off[mc3] - dchi_lo;
                            a_pt0[a] = p0;
                            for (q = p0; q < pe; q++) pt_aidx[q] = a;
                        }

                        {
                            const int npts_c = (int)npts;
                            const size_t npts_u = (npts == 0 ? 1 : npts);
                            const size_t am_u = (size_t)chunk_atoms;

                            acc_memcpy_to_device(chunk_arena + o_pt,    pt_aidx, sizeof(int) * npts_u);
                            acc_memcpy_to_device(chunk_arena + o_awan,  a_wan,   sizeof(int) * am_u);
                            acc_memcpy_to_device(chunk_arena + o_ano0,  a_no0,   sizeof(int) * am_u);
                            acc_memcpy_to_device(chunk_arena + o_adchi, a_dchi,  sizeof(size_t) * am_u);
                            acc_memcpy_to_device(chunk_arena + o_apt0,  a_pt0,   sizeof(size_t) * am_u);
                            {
                                const int* pt_aidx = (const int*)(const void*)(chunk_arena + o_pt);
                                const int* a_wan = (const int*)(const void*)(chunk_arena + o_awan);
                                const int* a_no0 = (const int*)(const void*)(chunk_arena + o_ano0);
                                const size_t* a_dchi = (const size_t*)(const void*)(chunk_arena + o_adchi);
                                const size_t* a_pt0 = (const size_t*)(const void*)(chunk_arena + o_apt0);
                                const double* rv_p = (const double*)(const void*)(chunk_arena + o_rv);
                                const double* rwf_p = (const double*)(const void*)(chunk_arena + o_rwf);
                                const size_t* rvo_p = (const size_t*)(const void*)(chunk_arena + o_rvo);
                                const size_t* rwb_p = (const size_t*)(const void*)(chunk_arena + o_rwb);
                                const int* spm_p = (const int*)(const void*)(chunk_arena + o_spm);
                                const int* spx_p = (const int*)(const void*)(chunk_arena + o_spx);
                                const int* spb_p = (const int*)(const void*)(chunk_arena + o_spb);

#pragma acc parallel loop gang vector_length(128) deviceptr(dchi_dev, xyz_chunk, \
    pt_aidx, a_wan, a_no0, a_dchi, a_pt0, \
    rv_p, rwf_p, rvo_p, rwb_p, spm_p, spx_p, spb_p)
                                for (int ip = 0; ip < npts_c; ip++) {
                                    const int a2 = pt_aidx[ip];
                                    const int wan = a_wan[a2];
                                    const int NO0 = a_no0[a2];
                                    const size_t Nc = (size_t)ip - a_pt0[a2];
                                    double* dchi = dchi_dev + a_dchi[a2] + Nc * (size_t)NO0 * 3;
                                    const double x = xyz_chunk[3 * (size_t)ip + 0];
                                    const double y = xyz_chunk[3 * (size_t)ip + 1];
                                    const double z = xyz_chunk[3 * (size_t)ip + 2];
                                    const int mesh = spm_p[wan];
                                    const int maxl = spx_p[wan];
                                    const double* rv = rv_p + rvo_p[wan];
                                    double RF[F3_DORB_L0MAX + 1][F3_DORB_MULMAX];
                                    double dRF[F3_DORB_L0MAX + 1][F3_DORB_MULMAX];
                                    double AF[F3_DORB_L0MAX + 1][2 * F3_DORB_L0MAX + 1];
                                    double dAFQ[F3_DORB_L0MAX + 1][2 * F3_DORB_L0MAX + 1];
                                    double dAFP[F3_DORB_L0MAX + 1][2 * F3_DORB_L0MAX + 1];
                                    double xx = x, yy = y, zz = z;
                                    double R, Q, P;
                                    int po = 0;
                                    int L0, Mul0, M0, i1;

                                    /* xyz2spherical */
                                    {
                                        const double Min_r = 10e-15;
                                        const double Rmin0 = 10e-14;
                                        int pass;

                                        for (pass = 0; pass < 2; pass++) {
                                            double dum = xx * xx + yy * yy;
                                            double r = sqrt(dum + zz * zz);
                                            double r1 = sqrt(dum);
                                            double theta, phi, dum1;

                                            if (Min_r <= r) {
                                                if (r < fabs(zz)) dum1 = (zz >= 0.0 ? 1.0 : -1.0);
                                                else dum1 = zz / r;
                                                theta = acos(dum1);

                                                if (Min_r <= r1) {
                                                    if (r1 < fabs(yy)) dum1 = (yy >= 0.0 ? 1.0 : -1.0);
                                                    else dum1 = yy / r1;
                                                    if (0.0 <= xx) phi = asin(dum1);
                                                    else phi = PI - asin(dum1);
                                                } else {
                                                    phi = 0.0;
                                                }
                                            } else {
                                                theta = 0.5 * PI;
                                                phi = 0.0;
                                            }

                                            R = r;
                                            Q = theta;
                                            P = phi;

                                            if (pass == 0 && R < Rmin0) {
                                                xx += Rmin0;
                                                yy += Rmin0;
                                                zz += Rmin0;
                                            } else {
                                                break;
                                            }
                                        }
                                    }

                                    /* radial spline of every (L0, Mul0) */
                                    if (rv[mesh - 1] < R) {
                                        for (L0 = 0; L0 <= maxl; L0++) {
                                            const int nb = spb_p[wan * (F3_DORB_L0MAX + 1) + L0];
                                            for (Mul0 = 0; Mul0 < nb; Mul0++) {
                                                RF[L0][Mul0] = 0.0;
                                                dRF[L0][Mul0] = 0.0;
                                            }
                                        }
                                        po = 1;
                                    } else {
                                        int m;
                                        double rm_or_R;
                                        int below = (R < rv[0]);
                                        double h1, h2, h3, x1, x2, y1v, y2v, y12, y22;
                                        double dum, dum1, dum2, dum3, dum4;
                                        size_t rpos = rwb_p[wan];

                                        if (below) {
                                            m = 4;
                                            rm_or_R = rv[m];
                                        } else {
                                            int mp_min = 0, mp_max = mesh - 1;

                                            do {
                                                m = (mp_min + mp_max) / 2;
                                                if (rv[m] < R) mp_min = m;
                                                else mp_max = m;
                                            } while ((mp_max - mp_min) != 1);
                                            m = mp_max;
                                            rm_or_R = R;
                                        }

                                        /* the boundary meshes fold the outer stencil point back
                                           inward instead of reading past the table */
                                        h2 = rv[m] - rv[m - 1];
                                        h3 = (m == mesh - 1) ? 0.0 : (rv[m + 1] - rv[m]);
                                        h1 = (m == 1) ? 0.0 : (rv[m - 1] - rv[m - 2]);
                                        if (m == mesh - 1) h3 = -(h1 + h2);
                                        if (m == 1) h1 = -(h2 + h3);

                                        x1 = rm_or_R - rv[m - 1];
                                        x2 = rm_or_R - rv[m];
                                        y1v = x1 / h2;
                                        y2v = x2 / h2;
                                        y12 = y1v * y1v;
                                        y22 = y2v * y2v;

                                        dum = h1 + h2;
                                        dum1 = h1 / h2 / dum;
                                        dum2 = h2 / h1 / dum;
                                        dum = h2 + h3;
                                        dum3 = h2 / h3 / dum;
                                        dum4 = h3 / h2 / dum;

                                        for (L0 = 0; L0 <= maxl; L0++) {
                                            const int nb = spb_p[wan * (F3_DORB_L0MAX + 1) + L0];

                                            for (Mul0 = 0; Mul0 < nb; Mul0++) {
                                                const double* rwf = rwf_p + rpos;
                                                double f2 = rwf[m - 1];
                                                double f3 = rwf[m];
                                                double f4 = (m == mesh - 1) ? 0.0 : rwf[m + 1];
                                                double f1 = (m == 1) ? f4 : rwf[m - 2];
                                                double g1, g2, f, df;

                                                rpos += (size_t)mesh;

                                                if (m == mesh - 1) f4 = f1;

                                                dum = f3 - f2;
                                                g1 = dum * dum1 + (f2 - f1) * dum2;
                                                g2 = (f4 - f3) * dum3 + dum * dum4;

                                                f = y22 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2v)
                                                    + y12 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1v);

                                                df = 2.0 * y2v / h2 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2v)
                                                    + y22 * (2.0 * f2 + h2 * g1) / h2
                                                    + 2.0 * y1v / h2 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1v)
                                                    - y12 * (2.0 * f3 - h2 * g2) / h2;

                                                if (below) {
                                                    double aa, bb, cc, dd;
                                                    const double rm = rm_or_R;

                                                    if (L0 == 0) {
                                                        aa = 0.0;
                                                        bb = 0.5 * df / rm;
                                                        cc = 0.0;
                                                        dd = f - bb * rm * rm;
                                                    } else if (L0 == 1) {
                                                        aa = (rm * df - f) / (2.0 * rm * rm * rm);
                                                        bb = 0.0;
                                                        cc = df - 3.0 * aa * rm * rm;
                                                        dd = 0.0;
                                                    } else {
                                                        bb = (3.0 * f - rm * df) / (rm * rm);
                                                        aa = (f - bb * rm * rm) / (rm * rm * rm);
                                                        cc = 0.0;
                                                        dd = 0.0;
                                                    }

                                                    RF[L0][Mul0] = aa * R * R * R + bb * R * R + cc * R + dd;
                                                    dRF[L0][Mul0] = 3.0 * aa * R * R + 2.0 * bb * R + cc;
                                                } else {
                                                    RF[L0][Mul0] = f;
                                                    dRF[L0][Mul0] = df;
                                                }
                                            }
                                        }
                                    }

                                    /* angular parts and assembly */
                                    {
                                        const double Rmin0 = 10e-14;
                                        double dRx = 0.0, dRy = 0.0, dRz = 0.0;
                                        double dQx = 0.0, dQy = 0.0, dQz = 0.0;
                                        double dPx = 0.0, dPy = 0.0, dPz = 0.0;

                                        if (po == 0) {
                                            const double siQ = sin(Q);
                                            const double coQ = cos(Q);
                                            const double siP = sin(P);
                                            const double coP = cos(P);
                                            double dum, dum1, dum2;

                                            dRx = siQ * coP;
                                            dRy = siQ * siP;
                                            dRz = coQ;

                                            if (Rmin0 < R) {
                                                dQx = coQ * coP / R;
                                                dQy = coQ * siP / R;
                                                dQz = -siQ / R;
                                                dPx = -siP / R;
                                                dPy = coP / R;
                                                dPz = 0.0;
                                            }

                                            for (L0 = 0; L0 <= maxl; L0++) {
                                                if (L0 == 0) {
                                                    AF[0][0] = 0.282094791773878;
                                                    dAFQ[0][0] = 0.0;
                                                    dAFP[0][0] = 0.0;
                                                } else if (L0 == 1) {
                                                    dum = 0.48860251190292 * siQ;

                                                    AF[1][0] = dum * coP;
                                                    AF[1][1] = dum * siP;
                                                    AF[1][2] = 0.48860251190292 * coQ;

                                                    dAFQ[1][0] = 0.48860251190292 * coQ * coP;
                                                    dAFQ[1][1] = 0.48860251190292 * coQ * siP;
                                                    dAFQ[1][2] = -0.48860251190292 * siQ;

                                                    dAFP[1][0] = -0.48860251190292 * siP;
                                                    dAFP[1][1] = 0.48860251190292 * coP;
                                                    dAFP[1][2] = 0.0;
                                                } else if (L0 == 2) {
                                                    dum1 = siQ * siQ;
                                                    dum2 = 1.09254843059208 * siQ * coQ;
                                                    AF[2][0] = 0.94617469575756 * coQ * coQ - 0.31539156525252;
                                                    AF[2][1] = 0.54627421529604 * dum1 * (1.0 - 2.0 * siP * siP);
                                                    AF[2][2] = 1.09254843059208 * dum1 * siP * coP;
                                                    AF[2][3] = dum2 * coP;
                                                    AF[2][4] = dum2 * siP;

                                                    dAFQ[2][0] = -1.89234939151512 * siQ * coQ;
                                                    dAFQ[2][1] = 1.09254843059208 * siQ * coQ * (1.0 - 2.0 * siP * siP);
                                                    dAFQ[2][2] = 2.18509686118416 * siQ * coQ * siP * coP;
                                                    dAFQ[2][3] = 1.09254843059208 * (1.0 - 2.0 * siQ * siQ) * coP;
                                                    dAFQ[2][4] = 1.09254843059208 * (1.0 - 2.0 * siQ * siQ) * siP;

                                                    dAFP[2][0] = 0.0;
                                                    dAFP[2][1] = -2.18509686118416 * siQ * siP * coP;
                                                    dAFP[2][2] = 1.09254843059208 * siQ * (1.0 - 2.0 * siP * siP);
                                                    dAFP[2][3] = -1.09254843059208 * coQ * siP;
                                                    dAFP[2][4] = 1.09254843059208 * coQ * coP;
                                                } else {
                                                    AF[3][0] = 0.373176332590116 * (5.0 * coQ * coQ * coQ - 3.0 * coQ);
                                                    AF[3][1] = 0.457045799464466 * coP * siQ * (5.0 * coQ * coQ - 1.0);
                                                    AF[3][2] = 0.457045799464466 * siP * siQ * (5.0 * coQ * coQ - 1.0);
                                                    AF[3][3] = 1.44530572132028 * siQ * siQ * coQ * (coP * coP - siP * siP);
                                                    AF[3][4] = 2.89061144264055 * siQ * siQ * coQ * siP * coP;
                                                    AF[3][5] = 0.590043589926644 * siQ * siQ * siQ * (4.0 * coP * coP * coP - 3.0 * coP);
                                                    AF[3][6] = 0.590043589926644 * siQ * siQ * siQ * (3.0 * siP - 4.0 * siP * siP * siP);

                                                    dAFQ[3][0] = 0.373176332590116 * siQ * (-15.0 * coQ * coQ + 3.0);
                                                    dAFQ[3][1] = 0.457045799464466 * coP * coQ * (15.0 * coQ * coQ - 11.0);
                                                    dAFQ[3][2] = 0.457045799464466 * siP * coQ * (15.0 * coQ * coQ - 11.0);
                                                    dAFQ[3][3] = 1.44530572132028 * (coP * coP - siP * siP) * siQ * (2.0 * coQ * coQ - siQ * siQ);
                                                    dAFQ[3][4] = 2.89061144264055 * coP * siP * siQ * (2.0 * coQ * coQ - siQ * siQ);
                                                    dAFQ[3][5] = 1.770130769779932 * coP * coQ * siQ * siQ * (-3.0 + 4.0 * coP * coP);
                                                    dAFQ[3][6] = 1.770130769779932 * coQ * siP * siQ * siQ * (3.0 - 4.0 * siP * siP);

                                                    dAFP[3][0] = 0.0;
                                                    dAFP[3][1] = 0.457045799464466 * siP * (-5.0 * coQ * coQ + 1.0);
                                                    dAFP[3][2] = 0.457045799464466 * coP * (5.0 * coQ * coQ - 1.0);
                                                    dAFP[3][3] = -5.781222885281120 * coP * coQ * siP * siQ;
                                                    dAFP[3][4] = 2.89061144264055 * coQ * siQ * (coP * coP - siP * siP);
                                                    dAFP[3][5] = 1.770130769779932 * siP * siQ * siQ * (1.0 - 4.0 * coP * coP);
                                                    dAFP[3][6] = 1.770130769779932 * coP * siQ * siQ * (1.0 - 4.0 * siP * siP);
                                                }
                                            }
                                        }

                                        i1 = -1;
                                        for (L0 = 0; L0 <= maxl; L0++) {
                                            const int nb = spb_p[wan * (F3_DORB_L0MAX + 1) + L0];

                                            for (Mul0 = 0; Mul0 < nb; Mul0++) {
                                                for (M0 = 0; M0 <= 2 * L0; M0++) {
                                                    double dChiR, dChiQ, dChiP;

                                                    i1++;
                                                    if (po == 0) {
                                                        dChiR = dRF[L0][Mul0] * AF[L0][M0];
                                                        dChiQ = RF[L0][Mul0] * dAFQ[L0][M0];
                                                        dChiP = RF[L0][Mul0] * dAFP[L0][M0];

                                                        dchi[(size_t)i1 * 3 + 0] = -dRx * dChiR - dQx * dChiQ - dPx * dChiP;
                                                        dchi[(size_t)i1 * 3 + 1] = -dRy * dChiR - dQy * dChiQ - dPy * dChiP;
                                                        dchi[(size_t)i1 * 3 + 2] = -dRz * dChiR - dQz * dChiQ - dPz * dChiP;
                                                    } else {
                                                        dchi[(size_t)i1 * 3 + 0] = 0.0;
                                                        dchi[(size_t)i1 * 3 + 1] = 0.0;
                                                        dchi[(size_t)i1 * 3 + 2] = 0.0;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                    }
                } else if (dchi_len != 0) {
                    acc_memcpy_to_device(dchi_dev, (void*)(dchi_all + dchi_lo),
                        dchi_len * sizeof(double));
                }

                if (0 < npairs) {
                    const int spin_count = spins;
                    const int npairs_c = npairs;

                    {
                        const Force3GpuPair* pm = (const Force3GpuPair*)(const void*)(chunk_arena + o_pm);
                        const int* gl1 = (const int*)(const void*)(chunk_arena + o_gl1);
                        const int* gl2 = (const int*)(const void*)(chunk_arena + o_gl2);
                        const float* fn_buf = (const float*)(const void*)(chunk_arena + o_fn);
                        const double* dm_buf = (const double*)(const void*)(chunk_arena + o_dm);
                        double* pair_f = (double*)(void*)(chunk_arena + o_pf);

#pragma acc parallel loop gang vector_length(128) deviceptr(dchi_dev, vpot_dev_full, orbs_dev, \
    pm, gl1, gl2, fn_buf, dm_buf, pair_f)
                        for (int pp = 0; pp < npairs_c; pp++) {
                            const int NO0 = pm[pp].no0;
                            const int NO1 = pm[pp].no1;
                            const int n_olg = pm[pp].n_olg;
                            const int gl_off = pm[pp].glist_off;
                            const int vstride = pm[pp].vpot_stride;
                            const int use_fn = pm[pp].orb1_fnan;
                            const size_t dbase = pm[pp].dchi_base;
                            const size_t vbase = pm[pp].vpot_base;
                            const size_t obase = pm[pp].orb1_base;
                            const size_t dmoff = pm[pp].dm_off;
                            double fx = 0.0, fy = 0.0, fz = 0.0;

#pragma acc loop vector reduction(+:fx, fy, fz)
                            for (int Nog = 0; Nog < n_olg; Nog++) {
                                const int Nc = gl1[gl_off + Nog];
                                const int Nh = gl2[gl_off + Nog];
                                const float* orb1 = use_fn
                                    ? (fn_buf + obase + (size_t)Nog * (size_t)NO1)
                                    : (orbs_dev + obase + (size_t)Nh * (size_t)NO1);
                                const double* dchi = dchi_dev + dbase + (size_t)Nc * (size_t)NO0 * 3;

                                for (int spin = 0; spin < spin_count; spin++) {
                                    const double* dm = dm_buf + dmoff
                                        + (size_t)spin * (size_t)NO0 * (size_t)NO1;
                                    double tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;

                                    for (int i = 0; i < NO0; i++) {
                                        double tmp0 = 0.0;

                                        for (int j = 0; j < NO1; j++) {
                                            tmp0 += (double)orb1[j] * dm[(size_t)i * (size_t)NO1 + j];
                                        }

                                        tmpx += dchi[(size_t)i * 3 + 0] * tmp0;
                                        tmpy += dchi[(size_t)i * 3 + 1] * tmp0;
                                        tmpz += dchi[(size_t)i * 3 + 2] * tmp0;
                                    }

                                    {
                                        const double Vpt = vpot_dev_full[vbase + (size_t)spin * (size_t)vstride + Nc];

                                        fx += tmpx * Vpt;
                                        fy += tmpy * Vpt;
                                        fz += tmpz * Vpt;
                                    }
                                }
                            }

                            pair_f[3 * (size_t)pp + 0] = fx;
                            pair_f[3 * (size_t)pp + 1] = fy;
                            pair_f[3 * (size_t)pp + 2] = fz;
                        }
                    }

                    acc_memcpy_from_device(pair_f, chunk_arena + o_pf,
                        sizeof(double) * 3U * (size_t)npairs_c);

                    {
                        int q = 0;

                        while (q < npairs) {
                            const int mc_atom = pm[q].atom;
                            const int Gc_AN = M2G[mc_atom];
                            double ax = 0.0, ay = 0.0, az = 0.0;

                            while (q < npairs && pm[q].atom == mc_atom) {
                                ax += pair_f[3 * (size_t)q + 0];
                                ay += pair_f[3 * (size_t)q + 1];
                                az += pair_f[3 * (size_t)q + 2];
                                q++;
                            }

                            Gxyz[Gc_AN][17] += ax * GridVol;
                            Gxyz[Gc_AN][18] += ay * GridVol;
                            Gxyz[Gc_AN][19] += az * GridVol;

                            if (2 <= level_stdout) {
                                printf("<Force>  force(3) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                                    myid, mc_atom, Gc_AN, ax * GridVol, ay * GridVol, az * GridVol);
                                fflush(stdout);
                            }
                        }
                    }
                }

            }
        }
    }

            acc_free(chunk_arena);

            free(a_pt0);
            free(a_dchi);
            free(a_no0);
            free(a_wan);
            free(pt_aidx);
            free(pm_h);
            free(pair_f);
            free(dm_buf);
            free(fn_buf);
            free(gl2);
            free(gl1);
            free(pm);
        }

        free(plan_dm);
        free(plan_fn);
        free(plan_gl);
        free(plan_pairs);
        free(chunk_bound);
    }

    free(sp_nb);
    free(sp_maxl);
    free(sp_mesh);
    free(rwf_base);
    free(rv_off);
    free(pao_rwf);
    free(pao_rv);
    acc_free(orbs_dev);
    free(orb_stage);
    free(orb_off);

    /* return the pooled chunk buffers to CUDA so the next force stage
       measures the true free memory */
    acc_wait_all();
    if (cudaDeviceSynchronize() == cudaSuccess) {
        acc_clear_freelists();
    }
}

static void Force4B_case2_trace_fused(int Mc_AN, int Gc_AN,
    int h_AN, int q_AN, double***** CDM0, Type_DS_VNA***** DS_VNA,
    double* dEx, double* dEy, double* dEz)
{
    int m, n, l;
    const int Gh_AN = natn[Gc_AN][h_AN];
    const int Gq_AN = natn[Gc_AN][q_AN];
    const int Mh_AN = F_G2M[Gh_AN];
    const int Hwan = WhatSpecies[Gh_AN];
    const int Qwan = WhatSpecies[Gq_AN];
    const int ian = Spe_Total_CNO[Hwan];
    const int jan = Spe_Total_CNO[Qwan];
    const int kl = RMI1[Mc_AN][h_AN][q_AN];
    const int kl1 = RMI1[Mc_AN][0][h_AN];
    const int kl2 = RMI1[Mc_AN][0][q_AN];
    const int num_projectors = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];
    const double pref = ((Solver == 5 || Solver == 8 || Solver == 11) || q_AN == h_AN)
        ? ((SpinP_switch == 0) ? 2.0 : 1.0)
        : ((SpinP_switch == 0) ? 4.0 : 2.0);
    const double rcut = Spe_Atom_Cut1[Hwan] + Spe_Atom_Cut1[Qwan];
    const double scale0 = pref * dampingF(rcut, Dis[Gh_AN][kl]);

    for (m = 0; m < ian; m++) {
        const Type_DS_VNA* restrict h0 = DS_VNA[0][Matomnum + 1][kl1][m];
        const Type_DS_VNA* restrict hx = DS_VNA[1][Matomnum + 1][kl1][m];
        const Type_DS_VNA* restrict hy = DS_VNA[2][Matomnum + 1][kl1][m];
        const Type_DS_VNA* restrict hz = DS_VNA[3][Matomnum + 1][kl1][m];
        double* cdm0_row = CDM0[0][Mh_AN][kl][m];
        double* cdm1_row = (SpinP_switch == 0) ? NULL : CDM0[1][Mh_AN][kl][m];

        for (n = 0; n < jan; n++) {
            const Type_DS_VNA* restrict q0 = DS_VNA[0][Matomnum + 1][kl2][n];
            const Type_DS_VNA* restrict qx = DS_VNA[1][Matomnum + 1][kl2][n];
            const Type_DS_VNA* restrict qy = DS_VNA[2][Matomnum + 1][kl2][n];
            const Type_DS_VNA* restrict qz = DS_VNA[3][Matomnum + 1][kl2][n];
            double sumx = 0.0;
            double sumy = 0.0;
            double sumz = 0.0;

#pragma omp simd reduction(+:sumx, sumy, sumz)
            for (l = 0; l < num_projectors; l++) {
                sumx -= (double)(hx[l] * q0[l]) + (double)(h0[l] * qx[l]);
                sumy -= (double)(hy[l] * q0[l]) + (double)(h0[l] * qy[l]);
                sumz -= (double)(hz[l] * q0[l]) + (double)(h0[l] * qz[l]);
            }

            {
                const double cdm = (SpinP_switch == 0)
                    ? cdm0_row[n]
                    : cdm0_row[n] + cdm1_row[n];
                const double scale = scale0 * cdm;
                *dEx += scale * sumx;
                *dEy += scale * sumy;
                *dEz += scale * sumz;
            }
        }
    }
}

static void Force4B_case1_trace_fused(int Mc_AN, int Gc_AN, int q_AN,
    double***** CDM0, Type_DS_VNA***** DS_VNA,
    double* dEx, double* dEy, double* dEz)
{
    int k, m, n, l;
    const int center_atom = natn[Gc_AN][0];
    const int q_atom = natn[Gc_AN][q_AN];
    const int center_local = F_G2M[center_atom];
    const int q_local = F_G2M[q_atom];
    const int q_storage = (q_local <= Matomnum) ? q_local : Matomnum + 1;
    const int center_species = WhatSpecies[center_atom];
    const int q_species = WhatSpecies[q_atom];
    const int ian = Spe_Total_CNO[center_species];
    const int jan = Spe_Total_CNO[q_species];
    const int cdm_kl = RMI1[Mc_AN][0][q_AN];
    const int num_projectors = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];
    const double pref = (SpinP_switch == 0) ? 4.0 : 2.0;
    const double rcut = Spe_Atom_Cut1[center_species] + Spe_Atom_Cut1[q_species];
    const double distance = Dis[Gc_AN][q_AN];
    const double dmp = dampingF(rcut, distance);
    double force_x = 0.0;
    double force_y = 0.0;
    double force_z = 0.0;

    for (k = 0; k <= FNAN[Gc_AN]; k++) {
        const int qk_kl = RMI1[Mc_AN][q_AN][k];
        if (qk_kl >= 0) {
            for (m = 0; m < ian; m++) {
                double* cdm0_row = CDM0[0][center_local][cdm_kl][m];
                double* cdm1_row = (SpinP_switch == 0) ? NULL : CDM0[1][center_local][cdm_kl][m];
                const Type_DS_VNA* restrict hx = DS_VNA[1][Mc_AN][k][m];
                const Type_DS_VNA* restrict hy = DS_VNA[2][Mc_AN][k][m];
                const Type_DS_VNA* restrict hz = DS_VNA[3][Mc_AN][k][m];

                for (n = 0; n < jan; n++) {
                    const Type_DS_VNA* restrict q0 = DS_VNA[0][q_storage][qk_kl][n];
                    double sumx = 0.0;
                    double sumy = 0.0;
                    double sumz = 0.0;

#pragma omp simd reduction(+:sumx, sumy, sumz)
                    for (l = 0; l < num_projectors; l++) {
                        sumx += (double)(hx[l] * q0[l]);
                        sumy += (double)(hy[l] * q0[l]);
                        sumz += (double)(hz[l] * q0[l]);
                    }

                    {
                        const double cdm = (SpinP_switch == 0)
                            ? cdm0_row[n]
                            : cdm0_row[n] + cdm1_row[n];
                        const double scale = pref * cdm;
                        force_x += scale * sumx;
                        force_y += scale * sumy;
                        force_z += scale * sumz;
                    }
                }
            }
        }
    }

    force_x *= dmp;
    force_y *= dmp;
    force_z *= dmp;

    {
        double derivative_scale = 0.0;
        double r = distance;

        if (rcut > r) {
            derivative_scale = deri_dampingF(rcut, r) / dmp;
        }
        if (r < 1.0e-10) {
            r = 1.0e-10;
        }

        if (derivative_scale != 0.0) {
            double trace_hvna = 0.0;
            const int center_cell = ncn[Gc_AN][0];
            const int q_cell = ncn[Gc_AN][q_AN];

            for (m = 0; m < ian; m++) {
                double* cdm0_row = CDM0[0][center_local][cdm_kl][m];
                double* cdm1_row = (SpinP_switch == 0) ? NULL : CDM0[1][center_local][cdm_kl][m];
                for (n = 0; n < jan; n++) {
                    const double cdm = (SpinP_switch == 0)
                        ? cdm0_row[n]
                        : cdm0_row[n] + cdm1_row[n];
                    trace_hvna += pref * cdm * HVNA[Mc_AN][q_AN][m][n];
                }
            }

            const double dx = derivative_scale
                * ((Gxyz[center_atom][1] + atv[center_cell][1])
                    - (Gxyz[q_atom][1] + atv[q_cell][1])) / r;
            const double dy = derivative_scale
                * ((Gxyz[center_atom][2] + atv[center_cell][2])
                    - (Gxyz[q_atom][2] + atv[q_cell][2])) / r;
            const double dz = derivative_scale
                * ((Gxyz[center_atom][3] + atv[center_cell][3])
                    - (Gxyz[q_atom][3] + atv[q_cell][3])) / r;
            force_x += trace_hvna * dx;
            force_y += trace_hvna * dy;
            force_z += trace_hvna * dz;
        }
    }

    *dEx += force_x;
    *dEy += force_y;
    *dEz += force_z;
}

static void dH_U_full(int Mc_AN, int h_AN, int q_AN,
    double***** OLP, double**** v_eff,
    double*** Hx, double*** Hy, double*** Hz);

static void dH_U_NC_full(int Mc_AN, int h_AN, int q_AN,
    double***** OLP, dcomplex***** NC_v_eff,
    dcomplex**** Hx, dcomplex**** Hy, dcomplex**** Hz);

static void dHNL(int where_flag,
    int Mc_AN, int h_AN, int q_AN,
    double****** DS_NL1,
    dcomplex*** Hx, dcomplex*** Hy, dcomplex*** Hz);

static void dHVNA(int where_flag, int Mc_AN, int h_AN, int q_AN,
    Type_DS_VNA***** DS_VNA1,
    double***** TmpHVNA2, double***** TmpHVNA3,
    double** Hx, double** Hy, double** Hz);

static void dHNL_SO(
    double* sumx0r,
    double* sumy0r,
    double* sumz0r,
    double* sumx1r,
    double* sumy1r,
    double* sumz1r,
    double* sumx2r,
    double* sumy2r,
    double* sumz2r,
    double* sumx0i,
    double* sumy0i,
    double* sumz0i,
    double* sumx1i,
    double* sumy1i,
    double* sumz1i,
    double* sumx2i,
    double* sumy2i,
    double* sumz2i,
    double fugou,
    double PFp,
    double PFm,
    double ene_p,
    double ene_m,
    int l2, int* l,
    int Mc_AN, int k, int m,
    int Mj_AN, int kl, int n,
    double****** DS_NL1);

static void dHCH(int where_flag,
    int Mc_AN, int h_AN, int q_AN,
    double***** OLP1,
    dcomplex*** Hx, dcomplex*** Hy, dcomplex*** Hz);

static void dHCH_SO(double* sumx0r, double* sumx0i, double* sumy0r, double* sumy0i, double* sumz0r, double* sumz0i,
    double* sumx1r, double* sumx1i, double* sumy1r, double* sumy1i, double* sumz1r, double* sumz1i,
    double* sumx2r, double* sumx2i, double* sumy2r, double* sumy2i, double* sumz2r, double* sumz2i,
    double fugou,
    int Mc_AN, int k, int m,
    int Mj_AN, int kl, int n,
    int kg, int wakg,
    double penalty_value,
    double***** OLP1);

static void MPI_OLP(double***** OLP1);
static void Force3();
static void Force4();
static size_t Force4_OpenACC(double* Fx, double* Fy, double* Fz, size_t* atom_terms);
static void Force4B(double***** CDM0);

static void Force_HNL(double***** CDM0, double***** iDM0);
static int Force_HNL_GpuBegin(double****** DS_NL);
static void Force_HNL_GpuEnd(void);
static void Force_HNL_GpuArchiveHalo(double****** DS_NL, int Original_Mc_AN, int Gc_AN);
static void Force_HNL_GpuArchiveCase2(double****** DS_NL, int Mc_AN, int Gc_AN);
static void Force_HNL_GpuCase1Run(double***** CDM0);
static void Force_HNL_GpuCase2Run(double***** CDM0);
static void Force_CoreHole(double***** CDM0, double***** iDM0);

double Force(double***** H0,
    double****** DS_NL,
    double***** OLP,
    double***** CDM,
    double***** EDM)
{
    static int firsttime = 1;
    int Nc, GNc, GRc, Cwan, s1, s2, BN_AB;
    int Mc_AN, Gc_AN, MNc, start_q_AN;
    double x, y, z, dx, dy, dz, tmp0, tmp1, tmp2, tmp3;
    double xx, r2, tot_den;
    double sumx, sumy, sumz, r, dege, pref;
    int i, j, k, l, Hwan, Qwan, so, p0, q, q0;
    int h_AN, Gh_AN, q_AN, Gq_AN;
    int ian, jan, kl, spin, spinmax, al, be, p, size_CDM0, size_iDM0;
    int tno0, tno1, tno2, Mh_AN, Mq_AN, n, num, size1, size2;
    int wanA, wanB, Gc_BN;
    int XC_P_switch;
    double time0;
    double dum = 0.0, dge;
    double dEx, dEy, dEz;
    double Cxyz[4];
    double *Fx, *Fy, *Fz;
    dcomplex*** Hx;
    dcomplex*** Hy;
    dcomplex*** Hz;
    double*** HUx = NULL;
    double*** HUy = NULL;
    double*** HUz = NULL;
    dcomplex**** NC_HUx = NULL;
    dcomplex**** NC_HUy = NULL;
    dcomplex**** NC_HUz = NULL;
    double** HVNAx;
    double** HVNAy;
    double** HVNAz;
    double***** CDM0;
    double***** iDM0 = NULL;
    double* tmp_array;
    double* tmp_array2;
    double Re00x, Re00y, Re00z;
    double Re11x, Re11y, Re11z;
    double Re01x, Re01y, Re01z;
    double Im00x, Im00y, Im00z;
    double Im11x, Im11y, Im11z;
    double Im01x, Im01y, Im01z;
    int *Snd_CDM0_Size = NULL, *Rcv_CDM0_Size = NULL;
    int *Snd_iDM0_Size = NULL, *Rcv_iDM0_Size = NULL;
    double TStime, TEtime;
    int numprocs, myid, tag = 999, ID, IDS, IDR;
    double Stime_atom, Etime_atom;
    double profile_stamp, profile_total_start;
    double***** Force2_source = NULL;
    double***** Force5_source = NULL;
    size_t* gpu_atom_terms = NULL;
    size_t gpu_total_terms;
    const int force_profile = Force_collective_env_flag(
        "OPENMX_FORCE_PROFILE", 0, mpi_comm_level1);
    const int use_force_openacc = (scf_eigen_lib_flag == GPUSOLVER
        && Force_collective_env_flag("OPENMX_FORCE_GPU", 1,
            mpi_comm_level1));
    const int use_force4_openacc = (use_force_openacc
        && F_VNA_flag == 1
        && (ProExpn_VNA == 0 || ProExpn_VNA == 1));
    const int use_force2_openacc = (use_force_openacc
        && F_Kin_flag == 1
        && SO_switch == 0
        && Hub_U_switch == 0
        && Constraint_NCS_switch == 0
        && Zeeman_NCS_switch == 0
        && Zeeman_NCO_switch == 0);
    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;
    double stime, etime;

    MPI_Status stat;
    MPI_Request request;

    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    const int use_iDM0 = (SO_switch == 1
        || (Hub_U_switch == 1 && F_U_flag == 1 && SpinP_switch == 3)
        || 1 <= Constraint_NCS_switch
        || Zeeman_NCS_switch == 1
        || Zeeman_NCO_switch == 1);

    MPI_Barrier(mpi_comm_level1);
    dtime(&TStime);
    profile_total_start = TStime;
    profile_stamp = TStime;

    /* the SCF-side device caches were released just before Force; return
       any blocks still parked in this rank's OpenACC freelist so the force
       batches (and the peer ranks) measure the true free device memory */
    if (use_force_openacc) {
        acc_wait_all();
        Force_gpu_pool_flush();
    }

    /****************************************************
     allocation of arrays:
    ****************************************************/

    Fx = (double*)malloc(sizeof(double) * (Matomnum + 1));
    Fy = (double*)malloc(sizeof(double) * (Matomnum + 1));
    Fz = (double*)malloc(sizeof(double) * (Matomnum + 1));

    HVNAx = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
    for (j = 0; j < List_YOUSO[7]; j++) {
        HVNAx[j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
    }

    HVNAy = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
    for (j = 0; j < List_YOUSO[7]; j++) {
        HVNAy[j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
    }

    HVNAz = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
    for (j = 0; j < List_YOUSO[7]; j++) {
        HVNAz[j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
    }

    /* CDM0 */
    size_CDM0 = 0;
    CDM0 = (double*****)malloc(sizeof(double****) * (SpinP_switch + 1));
    for (k = 0; k <= SpinP_switch; k++) {
        CDM0[k] = (double****)malloc(sizeof(double***) * (Matomnum + MatomnumF + 1));
        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN = 0;
                tno0 = 1;
            } else {
                Gc_AN = F_M2G[Mc_AN];
                Cwan = WhatSpecies[Gc_AN];
                tno0 = Spe_Total_CNO[Cwan];
            }

            CDM0[k][Mc_AN] = (double***)malloc(sizeof(double**) * (FNAN[Gc_AN] + 1));
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno1 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    tno1 = Spe_Total_CNO[Hwan];
                }

                CDM0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*) * tno0);
                for (i = 0; i < tno0; i++) {
                    CDM0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double) * tno1);
                    size_CDM0 += tno1;
                }
            }
        }
    }

    Snd_CDM0_Size = (int*)malloc(sizeof(int) * numprocs);
    Rcv_CDM0_Size = (int*)malloc(sizeof(int) * numprocs);

    /* iDM0 */

    if (use_iDM0) {

        size_iDM0 = 0;
        iDM0 = (double*****)malloc(sizeof(double****) * 2);
        for (k = 0; k < 2; k++) {
            iDM0[k] = (double****)malloc(sizeof(double***) * (Matomnum + MatomnumF + 1));
            FNAN[0] = 0;
            for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++) {

                if (Mc_AN == 0) {
                    Gc_AN = 0;
                    tno0 = 1;
                } else {
                    Gc_AN = F_M2G[Mc_AN];
                    Cwan = WhatSpecies[Gc_AN];
                    tno0 = Spe_Total_CNO[Cwan];
                }

                iDM0[k][Mc_AN] = (double***)malloc(sizeof(double**) * (FNAN[Gc_AN] + 1));
                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                    if (Mc_AN == 0) {
                        tno1 = 1;
                    } else {
                        Gh_AN = natn[Gc_AN][h_AN];
                        Hwan = WhatSpecies[Gh_AN];
                        tno1 = Spe_Total_CNO[Hwan];
                    }

                    iDM0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*) * tno0);
                    for (i = 0; i < tno0; i++) {
                        iDM0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double) * tno1);
                        size_iDM0 += tno1;
                    }
                }
            }
        }

        Snd_iDM0_Size = (int*)malloc(sizeof(int) * numprocs);
        Rcv_iDM0_Size = (int*)malloc(sizeof(int) * numprocs);
    }

    /****************************************************
                        PrintMemory
    ****************************************************/

    if (firsttime) {
        PrintMemory("Force: Hx", sizeof(dcomplex) * List_YOUSO[7] * List_YOUSO[7], NULL);
        PrintMemory("Force: Hy", sizeof(dcomplex) * List_YOUSO[7] * List_YOUSO[7], NULL);
        PrintMemory("Force: Hz", sizeof(dcomplex) * List_YOUSO[7] * List_YOUSO[7], NULL);
        PrintMemory("Force: CDM0", sizeof(double) * size_CDM0, NULL);
        if (use_iDM0) {
            PrintMemory("Force: iDM0", sizeof(double) * size_iDM0, NULL);
        }
        firsttime = 0;
    }

    /****************************************************
      CDM to CDM0
    ****************************************************/

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_CNO[Cwan];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            tno2 = Spe_Total_CNO[Hwan];

            for (spin = 0; spin <= SpinP_switch; spin++) {
                for (i = 0; i < tno1; i++) {
                    for (j = 0; j < tno2; j++) {
                        CDM0[spin][Mc_AN][h_AN][i][j] = CDM[spin][Mc_AN][h_AN][i][j];
                    }
                }
            }
        }
    }

    /****************************************************
      iDM to iDM0
    ****************************************************/

    if (use_iDM0) {

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

            Gc_AN = M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[Cwan];

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_CNO[Hwan];

                for (i = 0; i < tno1; i++) {
                    for (j = 0; j < tno2; j++) {
                        iDM0[0][Mc_AN][h_AN][i][j] = iDM[0][0][Mc_AN][h_AN][i][j];
                        iDM0[1][Mc_AN][h_AN][i][j] = iDM[0][1][Mc_AN][h_AN][i][j];
                    }
                }
            }
        }
    }

    /****************************************************
     MPI:

     CDM0
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
            if (F_Snd_Num[IDS] != 0) {

                size1 = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (n = 0; n < F_Snd_Num[IDS]; n++) {
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
                                    size1++;
                                }
                            }
                        }
                    }
                }

                Snd_CDM0_Size[IDS] = size1;
                MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
            } else {
                Snd_CDM0_Size[IDS] = 0;
            }

            /* receiving of size of data */

            if (F_Rcv_Num[IDR] != 0) {
                MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                Rcv_CDM0_Size[IDR] = size2;
            } else {
                Rcv_CDM0_Size[IDR] = 0;
            }

            if (F_Snd_Num[IDS] != 0)
                MPI_Wait(&request, &stat);

        } else {
            Snd_CDM0_Size[IDS] = 0;
            Rcv_CDM0_Size[IDR] = 0;
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

            if (F_Snd_Num[IDS] != 0) {

                size1 = Snd_CDM0_Size[IDS];

                /* allocation of array */

                tmp_array = (double*)malloc(sizeof(double) * size1);

                /* multidimentional array to vector array */

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (n = 0; n < F_Snd_Num[IDS]; n++) {
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
                                    tmp_array[num] = CDM[spin][Mc_AN][h_AN][i][j];
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

            if (F_Rcv_Num[IDR] != 0) {

                size2 = Rcv_CDM0_Size[IDR];

                /* allocation of array */
                tmp_array2 = (double*)malloc(sizeof(double) * size2);

                MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    Mc_AN = F_TopMAN[IDR] - 1;
                    for (n = 0; n < F_Rcv_Num[IDR]; n++) {
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
                                    CDM0[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }
                }

                /* freeing of array */
                free(tmp_array2);
            }

            if (F_Snd_Num[IDS] != 0) {
                MPI_Wait(&request, &stat);
                free(tmp_array); /* freeing of array */
            }
        }
    }

    /****************************************************
     MPI:

     iDM0
    ****************************************************/

    if (use_iDM0) {

        /***********************************
                    set data size
        ************************************/

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {
                tag = 999;

                /* find data size to send block data */
                if (F_Snd_Num[IDS] != 0) {

                    size1 = 0;
                    for (so = 0; so < 2; so++) {
                        for (n = 0; n < F_Snd_Num[IDS]; n++) {
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
                                        size1++;
                                    }
                                }
                            }
                        }
                    }

                    Snd_iDM0_Size[IDS] = size1;
                    MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);

                } else {
                    Snd_iDM0_Size[IDS] = 0;
                }

                /* receiving of size of data */

                if (F_Rcv_Num[IDR] != 0) {
                    MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                    Rcv_iDM0_Size[IDR] = size2;
                } else {
                    Rcv_iDM0_Size[IDR] = 0;
                }

                if (F_Snd_Num[IDS] != 0)
                    MPI_Wait(&request, &stat);

            } else {
                Snd_iDM0_Size[IDS] = 0;
                Rcv_iDM0_Size[IDR] = 0;
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

                if (F_Snd_Num[IDS] != 0) {

                    size1 = Snd_iDM0_Size[IDS];

                    /* allocation of array */

                    tmp_array = (double*)malloc(sizeof(double) * size1);

                    /* multidimentional array to vector array */

                    num = 0;
                    for (so = 0; so < 2; so++) {
                        for (n = 0; n < F_Snd_Num[IDS]; n++) {
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
                                        tmp_array[num] = iDM[0][so][Mc_AN][h_AN][i][j];
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

                if (F_Rcv_Num[IDR] != 0) {

                    size2 = Rcv_iDM0_Size[IDR];

                    /* allocation of array */
                    tmp_array2 = (double*)malloc(sizeof(double) * size2);

                    MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                    num = 0;
                    for (so = 0; so < 2; so++) {
                        Mc_AN = F_TopMAN[IDR] - 1;
                        for (n = 0; n < F_Rcv_Num[IDR]; n++) {
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
                                        iDM0[so][Mc_AN][h_AN][i][j] = tmp_array2[num];
                                        num++;
                                    }
                                }
                            }
                        }
                    }

                    /* freeing of array */
                    free(tmp_array2);
                }

                if (F_Snd_Num[IDS] != 0) {
                    MPI_Wait(&request, &stat);
                    free(tmp_array); /* freeing of array */
                }
            }
        }

    } /* if (use_iDM0) */

    if (force_profile) {
        double now;
        dtime(&now);
        Force_profile_report("setup+CDM", now - profile_stamp,
            mpi_comm_level1, myid, numprocs);
        profile_stamp = now;
    }

    /****************************************************
                        #1 of force

                -\int \delta V_H drho_a/dx dr
                           and
                  force induced from PCC
                +\int V_XC drho_pcc/dx dr
    ****************************************************/

    if (myid == Host_ID && 0 < level_stdout) {
        printf("  Force calculation #1\n");
        fflush(stdout);
    }

    dtime(&stime);

    /*********************************************************
     set RefVxc_Grid, where the CA-LDA exchange-correlation
     functional is alway used.
    *********************************************************/

    XC_P_switch = 1;
    for (BN_AB = 0; BN_AB < My_NumGridB_AB; BN_AB++) {
        tot_den = ADensity_Grid_B[BN_AB] + ADensity_Grid_B[BN_AB];
        if (PCC_switch == 1) {
            tot_den += PCCDensity_Grid_B[0][BN_AB] + PCCDensity_Grid_B[1][BN_AB];
        }
        RefVxc_Grid_B[BN_AB] = XC_Ceperly_Alder(tot_den, XC_P_switch);
    }

    Data_Grid_Copy_B2C_1(RefVxc_Grid_B, RefVxc_Grid);
    Data_Grid_Copy_B2C_1(dVHart_Grid_B, dVHart_Grid);
    Data_Grid_Copy_B2C_2(Vxc_Grid_B, Vxc_Grid);
    Data_Grid_Copy_B2C_2(Density_Grid_B, Density_Grid);

#pragma omp parallel shared(myid, Spe_OpenCore_flag, Spe_Atomic_PCC, Spe_VPS_RV, Spe_VPS_XV, Spe_Num_Mesh_VPS, Spe_PAO_RV, Spe_Atomic_Den, Spe_PAO_XV, Spe_Num_Mesh_PAO, time_per_atom, level_stdout, GridVol, Vxc_Grid, RefVxc_Grid, SpinP_switch, F_Vxc_flag, PCC_switch, dVHart_Grid, F_dVHart_flag, Gxyz, atv, MGridListAtom, CellListAtom, GridListAtom, GridN_Atom, WhatSpecies, M2G, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, Stime_atom, Etime_atom, Gc_AN, Cwan, sumx, sumy, sumz, Nc, GNc, GRc, MNc, Cxyz, x, y, z, dx, dy, dz, r, r2, tmp0, tmp1, tmp2, xx)
    {

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (Mc_AN = (OMPID * Matomnum / Nthrds + 1); Mc_AN < ((OMPID + 1) * Matomnum / Nthrds + 1); Mc_AN++) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];

            sumx = 0.0;
            sumy = 0.0;
            sumz = 0.0;

            for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {

                GNc = GridListAtom[Mc_AN][Nc];
                GRc = CellListAtom[Mc_AN][Nc];
                MNc = MGridListAtom[Mc_AN][Nc];

                Get_Grid_XYZ(GNc, Cxyz);
                x = Cxyz[1] + atv[GRc][1];
                y = Cxyz[2] + atv[GRc][2];
                z = Cxyz[3] + atv[GRc][3];

                dx = Gxyz[Gc_AN][1] - x;
                dy = Gxyz[Gc_AN][2] - y;
                dz = Gxyz[Gc_AN][3] - z;
                r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < 1.0e-20)
                    r2 = 1.0e-20;
                r = sqrt(r2);
                xx = 0.5 * log(r2);

                /* for empty atoms */
                if (r < 1.0e-10)
                    r = 1.0e-10;

                if (1.0e-14 < r) {

                    tmp0 = Dr_KumoF(Spe_Num_Mesh_PAO[Cwan], xx, r,
                        Spe_PAO_XV[Cwan], Spe_PAO_RV[Cwan], Spe_Atomic_Den[Cwan]);

                    tmp1 = dVHart_Grid[MNc] * tmp0 / r * F_dVHart_flag;
                    sumx += tmp1 * dx;
                    sumy += tmp1 * dy;
                    sumz += tmp1 * dz;

                    /* contribution of Exc^(0) */

                    tmp1 = RefVxc_Grid[MNc] * tmp0 / r * F_Vxc_flag;
                    sumx += tmp1 * dx;
                    sumy += tmp1 * dy;
                    sumz += tmp1 * dz;

                    /* partial core correction */
                    if (PCC_switch == 1) {

                        tmp0 = 0.5 * F_Vxc_flag * Dr_KumoF(Spe_Num_Mesh_VPS[Cwan], xx, r, Spe_VPS_XV[Cwan], Spe_VPS_RV[Cwan], Spe_Atomic_PCC[Cwan]);

                        if (SpinP_switch == 0) {
                            tmp2 = 2.0 * Vxc_Grid[0][MNc];
                        } else {
                            if (Spe_OpenCore_flag[Cwan] == 0) {
                                tmp2 = Vxc_Grid[0][MNc] + Vxc_Grid[1][MNc];
                            } else if (Spe_OpenCore_flag[Cwan] == 1) {
                                tmp2 = 2.0 * Vxc_Grid[0][MNc];
                            } else if (Spe_OpenCore_flag[Cwan] == -1) {
                                tmp2 = 2.0 * Vxc_Grid[1][MNc];
                            }
                        }

                        tmp1 = tmp2 * tmp0 / r;
                        sumx -= tmp1 * dx;
                        sumy -= tmp1 * dy;
                        sumz -= tmp1 * dz;

                        /* contribution of Exc^(0) */

                        tmp2 = 2.0 * RefVxc_Grid[MNc];
                        tmp1 = tmp2 * tmp0 / r;
                        sumx += tmp1 * dx;
                        sumy += tmp1 * dy;
                        sumz += tmp1 * dz;
                    }
                }
            }

            Gxyz[Gc_AN][17] = -sumx * GridVol;
            Gxyz[Gc_AN][18] = -sumy * GridVol;
            Gxyz[Gc_AN][19] = -sumz * GridVol;

            if (2 <= level_stdout) {
                printf("<Force>  force(1) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                    myid, Mc_AN, Gc_AN, -sumx * GridVol, -sumy * GridVol, -sumz * GridVol);
                fflush(stdout);
            }

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
        }

    } /* #pragma omp parallel */

    dtime(&etime);
    if (force_profile) {
        Force_profile_report("force#1", etime - stime,
            mpi_comm_level1, myid, numprocs);
    }
    if (myid == 0 && measure_time) {
        printf("Time for force#1=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    /****************************************************
       added by T.Ohwaki

                        #1' of force
       contribution from an artificial wall applied
       in the ESM method so that atoms cannot go beyond
       the boundary of the unit cell along the a-axis.
    ****************************************************/

    if (ESM_switch != 0) {

        double fx, xb, x0, x, a;

        /* modified by AdvanceSoft */
        xb = Grid_Origin[ESM_direction] + tv[1][ESM_direction];
        a = ESM_wall_height / pow(1.89, 3.0);

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

            Gc_AN = M2G[Mc_AN];
            x = Gxyz[Gc_AN][ESM_direction];
            x0 = xb - ESM_wall_position;
            dx = x - x0;

            if (0.0 < dx) {
                fx = 3.0 * a * dx * dx;
            } else {
                fx = 0.0;
            }

            Gxyz[Gc_AN][16 + ESM_direction] += fx;

            /*
            printf("Gc_AN=%2d fx=%15.12f\n",Gc_AN,fx);fflush(stdout);
            */

            /* add an artifical force if required. */

            if (Arti_Force == 1) {
                if (Gc_AN == 1)
                    Gxyz[1][16 + ESM_direction] += Arti_Grad;
                if (myid == 0)
                    printf("    adding force at the proc. 'Force #1' \n");
            }
        }
    }

    /****************************************************
     contraction

     H0
     OLP
    ****************************************************/

    MPI_Barrier(mpi_comm_level1);

    if (Cnt_switch == 1) {

        Cont_Matrix0(H0[0], CntH0[0]);
        Cont_Matrix0(H0[1], CntH0[1]);
        Cont_Matrix0(H0[2], CntH0[2]);
        Cont_Matrix0(H0[3], CntH0[3]);

        Cont_Matrix0(OLP[0], CntOLP[0]);
        Cont_Matrix0(OLP[1], CntOLP[1]);
        Cont_Matrix0(OLP[2], CntOLP[2]);
        Cont_Matrix0(OLP[3], CntOLP[3]);
    }

    if (Hub_U_switch == 1 && Hub_U_occupation == 1) {
        MPI_OLP(OLP);
    }

    MPI_Barrier(mpi_comm_level1);

    /****************************************************
                        #2 of force

     kinetic operator and contribution from the Hubbard
     term with the full representation in calculating
     the occupation number
    ****************************************************/

    dtime(&stime);

    if (myid == Host_ID && 0 < level_stdout) {
        printf("  Force calculation #2%s\n", use_force2_openacc ? " (GPU reduction)" : "");
        fflush(stdout);
    }

    if (use_force2_openacc) {

        Force2_source = (Cnt_switch == 0) ? H0 : CntH0;
        gpu_atom_terms = (size_t*)malloc(sizeof(size_t) * (Matomnum + 1));

        dtime(&Stime_atom);
        gpu_total_terms = Force2_Kinetic_OpenACC(Force2_source, CDM0, Fx, Fy, Fz, gpu_atom_terms);
        dtime(&Etime_atom);
        Force_accumulate_atom_time_from_terms(gpu_atom_terms, gpu_total_terms, Etime_atom - Stime_atom);

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            Gc_AN = M2G[Mc_AN];

            if (2 <= level_stdout) {
                printf("<Force>  force(2) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                    myid, Mc_AN, Gc_AN, Fx[Mc_AN], Fy[Mc_AN], Fz[Mc_AN]);
                fflush(stdout);
            }

            Gxyz[Gc_AN][17] += Fx[Mc_AN];
            Gxyz[Gc_AN][18] += Fy[Mc_AN];
            Gxyz[Gc_AN][19] += Fz[Mc_AN];
        }

        free(gpu_atom_terms);
        gpu_atom_terms = NULL;
    }

    else {
#pragma omp parallel shared(time_per_atom, Gxyz, myid, level_stdout, iDM0, CDM0, CntH0, H0, F_Kin_flag, NC_v_eff, v_eff, OLP, Hub_U_occupation, Cnt_switch, F_NL_flag, List_YOUSO, RMI1, Zeeman_NCO_switch, Zeeman_NCS_switch, Constraint_NCS_switch, F_U_flag, Hub_U_switch, SO_switch, SpinP_switch, Spe_Total_CNO, F_G2M, natn, FNAN, WhatSpecies, M2G, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, Stime_atom, Etime_atom, Gc_AN, Cwan, dEx, dEy, dEz, h_AN, Gh_AN, Mh_AN, Hwan, ian, start_q_AN, q_AN, Gq_AN, Mq_AN, Qwan, jan, kl, so, i, j, k, Hx, Hy, Hz, HUx, HUy, HUz, NC_HUx, NC_HUy, NC_HUz, s1, s2, pref, spinmax, spin)
        {

        /* allocation of arrays */

        Hx = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hx[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hx[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hy = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hy[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hy[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hz = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hz[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hz[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
            && (Hub_U_occupation == 1 || Hub_U_occupation == 2)
            && SpinP_switch != 3) {

            HUx = (double***)malloc(sizeof(double**) * 3);
            for (i = 0; i < 3; i++) {
                HUx[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
                for (j = 0; j < List_YOUSO[7]; j++) {
                    HUx[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
                }
            }

            HUy = (double***)malloc(sizeof(double**) * 3);
            for (i = 0; i < 3; i++) {
                HUy[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
                for (j = 0; j < List_YOUSO[7]; j++) {
                    HUy[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
                }
            }

            HUz = (double***)malloc(sizeof(double**) * 3);
            for (i = 0; i < 3; i++) {
                HUz[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
                for (j = 0; j < List_YOUSO[7]; j++) {
                    HUz[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
                }
            }
        }

        if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
            && (Hub_U_occupation == 1 || Hub_U_occupation == 2)
            && SpinP_switch == 3) {

            NC_HUx = (dcomplex****)malloc(sizeof(dcomplex***) * 2);
            for (i = 0; i < 2; i++) {
                NC_HUx[i] = (dcomplex***)malloc(sizeof(dcomplex**) * 2);
                for (j = 0; j < 2; j++) {
                    NC_HUx[i][j] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
                    for (k = 0; k < List_YOUSO[7]; k++) {
                        NC_HUx[i][j][k] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
                    }
                }
            }

            NC_HUy = (dcomplex****)malloc(sizeof(dcomplex***) * 2);
            for (i = 0; i < 2; i++) {
                NC_HUy[i] = (dcomplex***)malloc(sizeof(dcomplex**) * 2);
                for (j = 0; j < 2; j++) {
                    NC_HUy[i][j] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
                    for (k = 0; k < List_YOUSO[7]; k++) {
                        NC_HUy[i][j][k] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
                    }
                }
            }

            NC_HUz = (dcomplex****)malloc(sizeof(dcomplex***) * 2);
            for (i = 0; i < 2; i++) {
                NC_HUz[i] = (dcomplex***)malloc(sizeof(dcomplex**) * 2);
                for (j = 0; j < 2; j++) {
                    NC_HUz[i][j] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
                    for (k = 0; k < List_YOUSO[7]; k++) {
                        NC_HUz[i][j][k] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
                    }
                }
            }
        }

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (Mc_AN = (OMPID * Matomnum / Nthrds + 1); Mc_AN < ((OMPID + 1) * Matomnum / Nthrds + 1); Mc_AN++) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];

            dEx = 0.0;
            dEy = 0.0;
            dEz = 0.0;

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                Gh_AN = natn[Gc_AN][h_AN];
                Mh_AN = F_G2M[Gh_AN];
                Hwan = WhatSpecies[Gh_AN];
                ian = Spe_Total_CNO[Hwan];

                if (SpinP_switch == 3 && (SO_switch == 1 || (Hub_U_switch == 1 && F_U_flag == 1) || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1))
                    start_q_AN = 0;
                else
                    start_q_AN = h_AN;

                for (q_AN = start_q_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {

                    Gq_AN = natn[Gc_AN][q_AN];
                    Mq_AN = F_G2M[Gq_AN];
                    Qwan = WhatSpecies[Gq_AN];
                    jan = Spe_Total_CNO[Qwan];
                    kl = RMI1[Mc_AN][h_AN][q_AN];

                    if (0 <= kl) {

                        for (so = 0; so < 3; so++) {
                            for (i = 0; i < List_YOUSO[7]; i++) {
                                for (j = 0; j < List_YOUSO[7]; j++) {
                                    Hx[so][i][j] = Complex(0.0, 0.0);
                                    Hy[so][i][j] = Complex(0.0, 0.0);
                                    Hz[so][i][j] = Complex(0.0, 0.0);
                                }
                            }
                        }

                        /****************************************************
                         Contribution from LDA+U with 'full'treatment for
                         counting the occupation number
                        ****************************************************/

                        if ((Hub_U_switch == 1 && F_U_flag == 1) || 1 <= Constraint_NCS_switch
                            || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1) {

                            /* full treatment and collinear case */

                            if (Hub_U_occupation == 1 && SpinP_switch != 3) {

                                /* initialize HUx, HUy, and HUz */

                                for (so = 0; so < 3; so++) {
                                    for (i = 0; i < List_YOUSO[7]; i++) {
                                        for (j = 0; j < List_YOUSO[7]; j++) {
                                            HUx[so][i][j] = 0.0;
                                            HUy[so][i][j] = 0.0;
                                            HUz[so][i][j] = 0.0;
                                        }
                                    }
                                }

                                dH_U_full(Mc_AN, h_AN, q_AN, OLP, v_eff, HUx, HUy, HUz);

                                /* add the contribution to Hx, Hy, and Hz */

                                if (SpinP_switch == 0)
                                    spinmax = 0;
                                else
                                    spinmax = 1;

                                for (spin = 0; spin <= spinmax; spin++) {
                                    for (i = 0; i < ian; i++) {
                                        for (j = 0; j < jan; j++) {
                                            Hx[spin][i][j].r += HUx[spin][i][j];
                                            Hy[spin][i][j].r += HUy[spin][i][j];
                                            Hz[spin][i][j].r += HUz[spin][i][j];
                                        }
                                    }
                                }
                            }

                            /* full treatment and non-collinear case */

                            else if (Hub_U_occupation == 1 && SpinP_switch == 3) {

                                /* initialize NC_HUx, NC_HUy, and NC_HUz */

                                for (s1 = 0; s1 < 2; s1++) {
                                    for (s2 = 0; s2 < 2; s2++) {
                                        for (i = 0; i < List_YOUSO[7]; i++) {
                                            for (j = 0; j < List_YOUSO[7]; j++) {
                                                NC_HUx[s1][s2][i][j] = Complex(0.0, 0.0);
                                                NC_HUy[s1][s2][i][j] = Complex(0.0, 0.0);
                                                NC_HUz[s1][s2][i][j] = Complex(0.0, 0.0);
                                            }
                                        }
                                    }
                                }

                                dH_U_NC_full(Mc_AN, h_AN, q_AN, OLP, NC_v_eff, NC_HUx, NC_HUy, NC_HUz);

                                /******************************************************
                                       add the contribution to Hx, Hy, and Hz

                                       Hx[0] 00
                                       Hx[1] 11
                                       Hx[2] 01
                                ******************************************************/

                                for (i = 0; i < ian; i++) {
                                    for (j = 0; j < jan; j++) {

                                        Hx[0][i][j].r += NC_HUx[0][0][i][j].r;
                                        Hy[0][i][j].r += NC_HUy[0][0][i][j].r;
                                        Hz[0][i][j].r += NC_HUz[0][0][i][j].r;

                                        Hx[1][i][j].r += NC_HUx[1][1][i][j].r;
                                        Hy[1][i][j].r += NC_HUy[1][1][i][j].r;
                                        Hz[1][i][j].r += NC_HUz[1][1][i][j].r;

                                        Hx[2][i][j].r += NC_HUx[0][1][i][j].r;
                                        Hy[2][i][j].r += NC_HUy[0][1][i][j].r;
                                        Hz[2][i][j].r += NC_HUz[0][1][i][j].r;

                                        Hx[0][i][j].i += NC_HUx[0][0][i][j].i;
                                        Hy[0][i][j].i += NC_HUy[0][0][i][j].i;
                                        Hz[0][i][j].i += NC_HUz[0][0][i][j].i;

                                        Hx[1][i][j].i += NC_HUx[1][1][i][j].i;
                                        Hy[1][i][j].i += NC_HUy[1][1][i][j].i;
                                        Hz[1][i][j].i += NC_HUz[1][1][i][j].i;

                                        Hx[2][i][j].i += NC_HUx[0][1][i][j].i;
                                        Hy[2][i][j].i += NC_HUy[0][1][i][j].i;
                                        Hz[2][i][j].i += NC_HUz[0][1][i][j].i;
                                    }
                                }
                            }
                        }

                        /****************************************************
                                           H0 = dKinetic
                        ****************************************************/

                        if (F_Kin_flag == 1) {

                            /* in case of no obital optimization */

                            if (Cnt_switch == 0) {
                                if (h_AN == 0) {
                                    for (i = 0; i < ian; i++) {
                                        for (j = 0; j < jan; j++) {
                                            Hx[0][i][j].r += H0[1][Mc_AN][q_AN][i][j];
                                            Hy[0][i][j].r += H0[2][Mc_AN][q_AN][i][j];
                                            Hz[0][i][j].r += H0[3][Mc_AN][q_AN][i][j];

                                            Hx[1][i][j].r += H0[1][Mc_AN][q_AN][i][j];
                                            Hy[1][i][j].r += H0[2][Mc_AN][q_AN][i][j];
                                            Hz[1][i][j].r += H0[3][Mc_AN][q_AN][i][j];
                                        }
                                    }
                                }

                                else if (h_AN != 0 && q_AN == 0) {
                                    for (i = 0; i < ian; i++) {
                                        for (j = 0; j < jan; j++) {
                                            Hx[0][i][j].r += H0[1][Mc_AN][h_AN][j][i];
                                            Hy[0][i][j].r += H0[2][Mc_AN][h_AN][j][i];
                                            Hz[0][i][j].r += H0[3][Mc_AN][h_AN][j][i];

                                            Hx[1][i][j].r += H0[1][Mc_AN][h_AN][j][i];
                                            Hy[1][i][j].r += H0[2][Mc_AN][h_AN][j][i];
                                            Hz[1][i][j].r += H0[3][Mc_AN][h_AN][j][i];
                                        }
                                    }
                                }
                            }

                            /* in case of obital optimization */

                            else {

                                if (h_AN == 0) {
                                    for (i = 0; i < ian; i++) {
                                        for (j = 0; j < jan; j++) {

                                            Hx[0][i][j].r += CntH0[1][Mc_AN][q_AN][i][j];
                                            Hy[0][i][j].r += CntH0[2][Mc_AN][q_AN][i][j];
                                            Hz[0][i][j].r += CntH0[3][Mc_AN][q_AN][i][j];

                                            Hx[1][i][j].r += CntH0[1][Mc_AN][q_AN][i][j];
                                            Hy[1][i][j].r += CntH0[2][Mc_AN][q_AN][i][j];
                                            Hz[1][i][j].r += CntH0[3][Mc_AN][q_AN][i][j];
                                        }
                                    }
                                }

                                else if (h_AN != 0 && q_AN == 0) {
                                    for (i = 0; i < ian; i++) {
                                        for (j = 0; j < jan; j++) {

                                            Hx[0][i][j].r += CntH0[1][Mc_AN][h_AN][j][i];
                                            Hy[0][i][j].r += CntH0[2][Mc_AN][h_AN][j][i];
                                            Hz[0][i][j].r += CntH0[3][Mc_AN][h_AN][j][i];

                                            Hx[1][i][j].r += CntH0[1][Mc_AN][h_AN][j][i];
                                            Hy[1][i][j].r += CntH0[2][Mc_AN][h_AN][j][i];
                                            Hz[1][i][j].r += CntH0[3][Mc_AN][h_AN][j][i];
                                        }
                                    }
                                }
                            }

                        } /* if F_Kin_flag */

                        /****************************************************
                                          \sum rho*dH
                        ****************************************************/

                        /* non-spin polarization */

                        if (SpinP_switch == 0) {

                            if (q_AN == h_AN)
                                pref = 2.0;
                            else
                                pref = 4.0;

                            for (i = 0; i < Spe_Total_CNO[Hwan]; i++) {
                                for (j = 0; j < Spe_Total_CNO[Qwan]; j++) {
                                    dEx += pref * CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r;
                                    dEy += pref * CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r;
                                    dEz += pref * CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r;
                                }
                            }
                        }

                        /* collinear spin polarized or non-colliear without SO and LDA+U */

                        else if (SpinP_switch == 1 || (SpinP_switch == 3 && SO_switch == 0 && Hub_U_switch == 0 && Constraint_NCS_switch == 0 && Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0)) {

                            if (q_AN == h_AN)
                                pref = 1.0;
                            else
                                pref = 2.0;

                            for (i = 0; i < Spe_Total_CNO[Hwan]; i++) {
                                for (j = 0; j < Spe_Total_CNO[Qwan]; j++) {

                                    dEx += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].r);
                                    dEy += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].r);
                                    dEz += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].r);
                                }
                            }
                        }

                        /* spin collinear with spin-orbit coupling */

                        else if (SpinP_switch == 1 && SO_switch == 1) {
                            printf("Spin-orbit coupling is not supported for collinear DFT calculations.\n");
                            fflush(stdout);
                            MPI_Finalize();
                            exit(0);
                        }

                        /* spin non-collinear with spin-orbit coupling or with LDA+U */

                        else if (SpinP_switch == 3 && (SO_switch == 1 || (Hub_U_switch == 1 && F_U_flag == 1) || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)) {

                            for (i = 0; i < Spe_Total_CNO[Hwan]; i++) {
                                for (j = 0; j < Spe_Total_CNO[Qwan]; j++) {

                                    dEx += CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx[2][i][j].i;

                                    dEy += CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy[2][i][j].i;

                                    dEz += CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz[2][i][j].i;
                                }
                            }
                        }

                    } /* if (0<=kl) */
                } /* q_AN */
            } /* h_AN */

            /****************************************************
                              #2 of Force
            ****************************************************/

            if (2 <= level_stdout) {
                printf("<Force>  force(2) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                    myid, Mc_AN, Gc_AN, dEx, dEy, dEz);
                fflush(stdout);
            }

            Gxyz[Gc_AN][17] += dEx;
            Gxyz[Gc_AN][18] += dEy;
            Gxyz[Gc_AN][19] += dEz;

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

        } /* Mc_AN */

        /* freeing of arrays */

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hx[i][j]);
            }
            free(Hx[i]);
        }
        free(Hx);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hy[i][j]);
            }
            free(Hy[i]);
        }
        free(Hy);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hz[i][j]);
            }
            free(Hz[i]);
        }
        free(Hz);

        if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
            && (Hub_U_occupation == 1 || Hub_U_occupation == 2)
            && SpinP_switch != 3) {

            for (i = 0; i < 3; i++) {
                for (j = 0; j < List_YOUSO[7]; j++) {
                    free(HUx[i][j]);
                }
                free(HUx[i]);
            }
            free(HUx);

            for (i = 0; i < 3; i++) {
                for (j = 0; j < List_YOUSO[7]; j++) {
                    free(HUy[i][j]);
                }
                free(HUy[i]);
            }
            free(HUy);

            for (i = 0; i < 3; i++) {
                for (j = 0; j < List_YOUSO[7]; j++) {
                    free(HUz[i][j]);
                }
                free(HUz[i]);
            }
            free(HUz);
        }

        if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
            && (Hub_U_occupation == 1 || Hub_U_occupation == 2)
            && SpinP_switch == 3) {

            for (i = 0; i < 2; i++) {
                for (j = 0; j < 2; j++) {
                    for (k = 0; k < List_YOUSO[7]; k++) {
                        free(NC_HUx[i][j][k]);
                    }
                    free(NC_HUx[i][j]);
                }
                free(NC_HUx[i]);
            }
            free(NC_HUx);

            for (i = 0; i < 2; i++) {
                for (j = 0; j < 2; j++) {
                    for (k = 0; k < List_YOUSO[7]; k++) {
                        free(NC_HUy[i][j][k]);
                    }
                    free(NC_HUy[i][j]);
                }
                free(NC_HUy[i]);
            }
            free(NC_HUy);

            for (i = 0; i < 2; i++) {
                for (j = 0; j < 2; j++) {
                    for (k = 0; k < List_YOUSO[7]; k++) {
                        free(NC_HUz[i][j][k]);
                    }
                    free(NC_HUz[i][j]);
                }
                free(NC_HUz[i]);
            }
            free(NC_HUz);
        }

        } /* #pragma omp parallel */
    }

    dtime(&etime);
    if (force_profile) {
        Force_profile_report("force#2", etime - stime,
            mpi_comm_level1, myid, numprocs);
    }
    if (myid == 0 && measure_time) {
        printf("Time for force#2=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    /****************************************************
                        #3 of Force

                 dn/dx * (VNA + dVH + Vxc)
              or
                 dn/dx * (dVH + Vxc)
    ****************************************************/

    dtime(&stime);

    if (myid == Host_ID && 0 < level_stdout) {
        printf("  Force calculation #3%s\n",
            (scf_eigen_lib_flag == GPUSOLVER && Force_env_flag("OPENMX_FORCE3_GPU", 1))
                ? " (GPU-accelerated)" : "");
        fflush(stdout);
    }

    Force3();

    dtime(&etime);
    if (force_profile) {
        Force_profile_report("force#3", etime - stime,
            mpi_comm_level1, myid, numprocs);
    }
    if (myid == 0 && measure_time) {
        printf("Time for force#3=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    /****************************************************
                        #4 of Force

         Force4:   n * dVNA/dx
         Force4B:  from separable VNA projectors
    ****************************************************/

    dtime(&stime);

    if (myid == Host_ID && 0 < level_stdout) {
        printf("  Force calculation #4%s\n",
            (ProExpn_VNA == 0 && use_force4_openacc) ? " (GPU reduction)"
            : (ProExpn_VNA == 1 && F_VNA_flag == 1 && scf_eigen_lib_flag == GPUSOLVER &&
               Force_env_flag("OPENMX_FORCE4B_GPU", 1)) ? " (GPU-accelerated)" : "");
        fflush(stdout);
    }

    if (ProExpn_VNA == 0 && F_VNA_flag == 1) {
        if (use_force_openacc) {
            gpu_atom_terms = (size_t*)malloc(sizeof(size_t) * (Matomnum + 1));

            dtime(&Stime_atom);
            gpu_total_terms = Force4_OpenACC(Fx, Fy, Fz, gpu_atom_terms);
            dtime(&Etime_atom);
            Force_accumulate_atom_time_from_terms(gpu_atom_terms, gpu_total_terms, Etime_atom - Stime_atom);

            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
                Gc_AN = M2G[Mc_AN];
                Gxyz[Gc_AN][17] += Fx[Mc_AN];
                Gxyz[Gc_AN][18] += Fy[Mc_AN];
                Gxyz[Gc_AN][19] += Fz[Mc_AN];
            }

            free(gpu_atom_terms);
            gpu_atom_terms = NULL;
        } else {
            Force4();
        }
    } else if (ProExpn_VNA == 1 && F_VNA_flag == 1) {
        Force4B(CDM0);
    }

    dtime(&etime);
    if (force_profile) {
        Force_profile_report("force#4", etime - stime,
            mpi_comm_level1, myid, numprocs);
    }
    if (myid == 0 && measure_time) {
        printf("Time for force#4=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    /****************************************************
                        #5 of Force

                 Contribution from overlap
    ****************************************************/

    dtime(&stime);

    if (myid == Host_ID && 0 < level_stdout) {
        printf("  Force calculation #5%s\n", use_force_openacc ? " (GPU reduction)" : "");
        fflush(stdout);
    }

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Fx[Mc_AN] = 0.0;
        Fy[Mc_AN] = 0.0;
        Fz[Mc_AN] = 0.0;
    }
    if (use_force_openacc) {
        Force5_source = (Cnt_switch == 0) ? OLP : CntOLP;
        gpu_atom_terms = (size_t*)malloc(sizeof(size_t) * (Matomnum + 1));

        dtime(&Stime_atom);
        gpu_total_terms = Force5_OpenACC(Force5_source, EDM, Fx, Fy, Fz, gpu_atom_terms);
        dtime(&Etime_atom);
        Force_accumulate_atom_time_from_terms(gpu_atom_terms, gpu_total_terms, Etime_atom - Stime_atom);

        free(gpu_atom_terms);
        gpu_atom_terms = NULL;
    }

    else {
    // comment out June 2nd, 2023 H. Kawai
#if NVHPC_VERSION >= 250000
        #pragma omp parallel shared(Dis,time_per_atom,Fx,Fy,Fz,CntOLP,OLP,Cnt_switch,EDM,SpinP_switch,Spe_Total_CNO,natn,FNAN,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,dum,dx,dy,dz)
#endif
        {

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (Mc_AN = (OMPID * Matomnum / Nthrds + 1); Mc_AN < ((OMPID + 1) * Matomnum / Nthrds + 1); Mc_AN++) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];

            for (h_AN = 1; h_AN <= FNAN[Gc_AN]; h_AN++) {

                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];

                for (i = 0; i < Spe_Total_CNO[Cwan]; i++) {
                    for (j = 0; j < Spe_Total_CNO[Hwan]; j++) {

                        if (SpinP_switch == 0) {
                            dum = 2.0 * EDM[0][Mc_AN][h_AN][i][j];
                        } else if (SpinP_switch == 1 || SpinP_switch == 3) {
                            dum = EDM[0][Mc_AN][h_AN][i][j] + EDM[1][Mc_AN][h_AN][i][j];
                        }

                        if (Cnt_switch == 0) {

                            dx = dum * OLP[1][Mc_AN][h_AN][i][j];
                            dy = dum * OLP[2][Mc_AN][h_AN][i][j];
                            dz = dum * OLP[3][Mc_AN][h_AN][i][j];
                        } else {
                            dx = dum * CntOLP[1][Mc_AN][h_AN][i][j];
                            dy = dum * CntOLP[2][Mc_AN][h_AN][i][j];
                            dz = dum * CntOLP[3][Mc_AN][h_AN][i][j];
                        }

                        Fx[Mc_AN] = Fx[Mc_AN] - 2.0 * dx;
                        Fy[Mc_AN] = Fy[Mc_AN] - 2.0 * dy;
                        Fz[Mc_AN] = Fz[Mc_AN] - 2.0 * dz;
                    }
                }
            }

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
        }

        } /* #pragma omp parallel */
    }

    dtime(&etime);
    if (force_profile) {
        Force_profile_report("force#5", etime - stime,
            mpi_comm_level1, myid, numprocs);
    }
    if (myid == 0 && measure_time) {
        printf("Time for force#5=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    /****************************************************
                    add #5 of Force
    ****************************************************/

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

        Gc_AN = M2G[Mc_AN];

        Gxyz[Gc_AN][17] += Fx[Mc_AN];
        Gxyz[Gc_AN][18] += Fy[Mc_AN];
        Gxyz[Gc_AN][19] += Fz[Mc_AN];

        if (2 <= level_stdout) {
            printf("<Force>  force(5) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                myid, Mc_AN, Gc_AN, Fx[Mc_AN], Fy[Mc_AN], Fz[Mc_AN]);
            fflush(stdout);
        }
    }

    /****************************************************************
     In case that the dual representation is used for evaluation of
     the occupation number in the LDA+U method, the following force
     term is added.
    ****************************************************************/

    if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
        && (Hub_U_occupation == 1 || Hub_U_occupation == 2)
        && SpinP_switch != 3) {

        HUx = (double***)malloc(sizeof(double**) * 3);
        for (i = 0; i < 3; i++) {
            HUx[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                HUx[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
            }
        }

        HUy = (double***)malloc(sizeof(double**) * 3);
        for (i = 0; i < 3; i++) {
            HUy[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                HUy[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
            }
        }

        HUz = (double***)malloc(sizeof(double**) * 3);
        for (i = 0; i < 3; i++) {
            HUz[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                HUz[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
            }
        }
    }

    if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
        && F_U_flag == 1 && Hub_U_occupation == 2) {

        if (myid == Host_ID)
            printf("  Force calculation for LDA_U with dual\n");
        fflush(stdout);

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            Fx[Mc_AN] = 0.0;
            Fy[Mc_AN] = 0.0;
            Fz[Mc_AN] = 0.0;
        }

        /****************************************************
          if (SpinP_switch!=3)

          collinear case
        ****************************************************/

        if (SpinP_switch != 3) {

            if (SpinP_switch == 0) {
                spinmax = 0;
                dege = 2.0;
            } else {
                spinmax = 1;
                dege = 1.0;
            }

            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                dtime(&Stime_atom);

                Gc_AN = M2G[Mc_AN];
                Cwan = WhatSpecies[Gc_AN];

                for (spin = 0; spin <= spinmax; spin++) {

                    for (h_AN = 1; h_AN <= FNAN[Gc_AN]; h_AN++) {

                        Gh_AN = natn[Gc_AN][h_AN];
                        Mh_AN = F_G2M[Gh_AN];
                        Hwan = WhatSpecies[Gh_AN];

                        /* non-orbital optimization */

                        if (Cnt_switch == 0) {

                            for (i = 0; i < Spe_Total_NO[Cwan]; i++) {
                                for (j = 0; j < Spe_Total_NO[Hwan]; j++) {

                                    tmp1 = 0.0;
                                    tmp2 = 0.0;
                                    tmp3 = 0.0;

                                    for (k = 0; k < Spe_Total_NO[Cwan]; k++) {
                                        tmp1 += v_eff[spin][Mc_AN][i][k] * OLP[1][Mc_AN][h_AN][k][j];
                                        tmp2 += v_eff[spin][Mc_AN][i][k] * OLP[2][Mc_AN][h_AN][k][j];
                                        tmp3 += v_eff[spin][Mc_AN][i][k] * OLP[3][Mc_AN][h_AN][k][j];
                                    }

                                    for (k = 0; k < Spe_Total_NO[Hwan]; k++) {
                                        tmp1 += v_eff[spin][Mh_AN][k][j] * OLP[1][Mc_AN][h_AN][i][k];
                                        tmp2 += v_eff[spin][Mh_AN][k][j] * OLP[2][Mc_AN][h_AN][i][k];
                                        tmp3 += v_eff[spin][Mh_AN][k][j] * OLP[3][Mc_AN][h_AN][i][k];
                                    }

                                    dx = tmp1 * dege * CDM[spin][Mc_AN][h_AN][i][j];
                                    dy = tmp2 * dege * CDM[spin][Mc_AN][h_AN][i][j];
                                    dz = tmp3 * dege * CDM[spin][Mc_AN][h_AN][i][j];

                                    Fx[Mc_AN] += dx;
                                    Fy[Mc_AN] += dy;
                                    Fz[Mc_AN] += dz;
                                }
                            }
                        }

                        /* orbital optimization */

                        else if (Cnt_switch == 1) {

                            /* HUx, HUy, HUz for primitive orbital */

                            for (i = 0; i < Spe_Total_NO[Cwan]; i++) {
                                for (j = 0; j < Spe_Total_NO[Hwan]; j++) {

                                    tmp1 = 0.0;
                                    tmp2 = 0.0;
                                    tmp3 = 0.0;

                                    for (k = 0; k < Spe_Total_NO[Cwan]; k++) {
                                        tmp1 += v_eff[spin][Mc_AN][i][k] * OLP[1][Mc_AN][h_AN][k][j];
                                        tmp2 += v_eff[spin][Mc_AN][i][k] * OLP[2][Mc_AN][h_AN][k][j];
                                        tmp3 += v_eff[spin][Mc_AN][i][k] * OLP[3][Mc_AN][h_AN][k][j];
                                    }

                                    for (k = 0; k < Spe_Total_NO[Hwan]; k++) {
                                        tmp1 += v_eff[spin][Mh_AN][k][j] * OLP[1][Mc_AN][h_AN][i][k];
                                        tmp2 += v_eff[spin][Mh_AN][k][j] * OLP[2][Mc_AN][h_AN][i][k];
                                        tmp3 += v_eff[spin][Mh_AN][k][j] * OLP[3][Mc_AN][h_AN][i][k];
                                    }

                                    HUx[0][i][j] = tmp1;
                                    HUy[0][i][j] = tmp2;
                                    HUz[0][i][j] = tmp3;
                                }
                            }

                            /* contract HUx, HUy, HUz */

                            for (al = 0; al < Spe_Total_CNO[Cwan]; al++) {
                                for (be = 0; be < Spe_Total_CNO[Hwan]; be++) {

                                    tmp1 = 0.0;
                                    tmp2 = 0.0;
                                    tmp3 = 0.0;

                                    for (p = 0; p < Spe_Specified_Num[Cwan][al]; p++) {
                                        p0 = Spe_Trans_Orbital[Cwan][al][p];
                                        for (q = 0; q < Spe_Specified_Num[Hwan][be]; q++) {
                                            q0 = Spe_Trans_Orbital[Hwan][be][q];
                                            tmp0 = CntCoes[Mc_AN][al][p] * CntCoes[Mh_AN][be][q];
                                            tmp1 += tmp0 * HUx[0][p0][q0];
                                            tmp2 += tmp0 * HUy[0][p0][q0];
                                            tmp3 += tmp0 * HUz[0][p0][q0];
                                        }
                                    }

                                    dx = tmp1 * dege * CDM[spin][Mc_AN][h_AN][al][be];
                                    dy = tmp2 * dege * CDM[spin][Mc_AN][h_AN][al][be];
                                    dz = tmp3 * dege * CDM[spin][Mc_AN][h_AN][al][be];

                                    Fx[Mc_AN] += dx;
                                    Fy[Mc_AN] += dy;
                                    Fz[Mc_AN] += dz;
                                }
                            }
                        }
                    }
                }

                dtime(&Etime_atom);
                time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
            }

        }

        /****************************************************
          if (SpinP_switch==3)

          spin non-collinear
        ****************************************************/

        else {

            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                dtime(&Stime_atom);

                Gc_AN = M2G[Mc_AN];
                Cwan = WhatSpecies[Gc_AN];

                for (h_AN = 1; h_AN <= FNAN[Gc_AN]; h_AN++) {

                    Gh_AN = natn[Gc_AN][h_AN];
                    Mh_AN = F_G2M[Gh_AN];
                    Hwan = WhatSpecies[Gh_AN];

                    kl = RMI1[Mc_AN][h_AN][0];

                    for (i = 0; i < Spe_Total_NO[Cwan]; i++) {
                        for (j = 0; j < Spe_Total_NO[Hwan]; j++) {

                            Re00x = 0.0;
                            Re00y = 0.0;
                            Re00z = 0.0;
                            Re11x = 0.0;
                            Re11y = 0.0;
                            Re11z = 0.0;
                            Re01x = 0.0;
                            Re01y = 0.0;
                            Re01z = 0.0;

                            Im00x = 0.0;
                            Im00y = 0.0;
                            Im00z = 0.0;
                            Im11x = 0.0;
                            Im11y = 0.0;
                            Im11z = 0.0;
                            Im01x = 0.0;
                            Im01y = 0.0;
                            Im01z = 0.0;

                            for (k = 0; k < Spe_Total_NO[Cwan]; k++) {

                                Re00x += NC_v_eff[0][0][Mc_AN][i][k].r * OLP[1][Mc_AN][h_AN][k][j];
                                Re00y += NC_v_eff[0][0][Mc_AN][i][k].r * OLP[2][Mc_AN][h_AN][k][j];
                                Re00z += NC_v_eff[0][0][Mc_AN][i][k].r * OLP[3][Mc_AN][h_AN][k][j];

                                Re11x += NC_v_eff[1][1][Mc_AN][i][k].r * OLP[1][Mc_AN][h_AN][k][j];
                                Re11y += NC_v_eff[1][1][Mc_AN][i][k].r * OLP[2][Mc_AN][h_AN][k][j];
                                Re11z += NC_v_eff[1][1][Mc_AN][i][k].r * OLP[3][Mc_AN][h_AN][k][j];

                                Re01x += NC_v_eff[0][1][Mc_AN][i][k].r * OLP[1][Mc_AN][h_AN][k][j];
                                Re01y += NC_v_eff[0][1][Mc_AN][i][k].r * OLP[2][Mc_AN][h_AN][k][j];
                                Re01z += NC_v_eff[0][1][Mc_AN][i][k].r * OLP[3][Mc_AN][h_AN][k][j];

                                Im00x += NC_v_eff[0][0][Mc_AN][i][k].i * OLP[1][Mc_AN][h_AN][k][j];
                                Im00y += NC_v_eff[0][0][Mc_AN][i][k].i * OLP[2][Mc_AN][h_AN][k][j];
                                Im00z += NC_v_eff[0][0][Mc_AN][i][k].i * OLP[3][Mc_AN][h_AN][k][j];

                                Im11x += NC_v_eff[1][1][Mc_AN][i][k].i * OLP[1][Mc_AN][h_AN][k][j];
                                Im11y += NC_v_eff[1][1][Mc_AN][i][k].i * OLP[2][Mc_AN][h_AN][k][j];
                                Im11z += NC_v_eff[1][1][Mc_AN][i][k].i * OLP[3][Mc_AN][h_AN][k][j];

                                Im01x += NC_v_eff[0][1][Mc_AN][i][k].i * OLP[1][Mc_AN][h_AN][k][j];
                                Im01y += NC_v_eff[0][1][Mc_AN][i][k].i * OLP[2][Mc_AN][h_AN][k][j];
                                Im01z += NC_v_eff[0][1][Mc_AN][i][k].i * OLP[3][Mc_AN][h_AN][k][j];
                            }

                            for (k = 0; k < Spe_Total_NO[Hwan]; k++) {

                                Re00x += NC_v_eff[0][0][Mh_AN][k][j].r * OLP[1][Mc_AN][h_AN][i][k];
                                Re00y += NC_v_eff[0][0][Mh_AN][k][j].r * OLP[2][Mc_AN][h_AN][i][k];
                                Re00z += NC_v_eff[0][0][Mh_AN][k][j].r * OLP[3][Mc_AN][h_AN][i][k];

                                Re11x += NC_v_eff[1][1][Mh_AN][k][j].r * OLP[1][Mc_AN][h_AN][i][k];
                                Re11y += NC_v_eff[1][1][Mh_AN][k][j].r * OLP[2][Mc_AN][h_AN][i][k];
                                Re11z += NC_v_eff[1][1][Mh_AN][k][j].r * OLP[3][Mc_AN][h_AN][i][k];

                                Re01x += NC_v_eff[0][1][Mh_AN][k][j].r * OLP[1][Mc_AN][h_AN][i][k];
                                Re01y += NC_v_eff[0][1][Mh_AN][k][j].r * OLP[2][Mc_AN][h_AN][i][k];
                                Re01z += NC_v_eff[0][1][Mh_AN][k][j].r * OLP[3][Mc_AN][h_AN][i][k];

                                Im00x += NC_v_eff[0][0][Mh_AN][k][j].i * OLP[1][Mc_AN][h_AN][i][k];
                                Im00y += NC_v_eff[0][0][Mh_AN][k][j].i * OLP[2][Mc_AN][h_AN][i][k];
                                Im00z += NC_v_eff[0][0][Mh_AN][k][j].i * OLP[3][Mc_AN][h_AN][i][k];

                                Im11x += NC_v_eff[1][1][Mh_AN][k][j].i * OLP[1][Mc_AN][h_AN][i][k];
                                Im11y += NC_v_eff[1][1][Mh_AN][k][j].i * OLP[2][Mc_AN][h_AN][i][k];
                                Im11z += NC_v_eff[1][1][Mh_AN][k][j].i * OLP[3][Mc_AN][h_AN][i][k];

                                Im01x += NC_v_eff[0][1][Mh_AN][k][j].i * OLP[1][Mc_AN][h_AN][i][k];
                                Im01y += NC_v_eff[0][1][Mh_AN][k][j].i * OLP[2][Mc_AN][h_AN][i][k];
                                Im01z += NC_v_eff[0][1][Mh_AN][k][j].i * OLP[3][Mc_AN][h_AN][i][k];
                            }

                            dx = Re00x * CDM0[0][Mc_AN][h_AN][i][j]
                                + Re11x * CDM0[1][Mc_AN][h_AN][i][j]
                                + 2.0 * Re01x * CDM0[2][Mc_AN][h_AN][i][j]
                                - Im00x * iDM0[0][Mc_AN][h_AN][i][j]
                                - Im11x * iDM0[1][Mc_AN][h_AN][i][j]
                                - 2.0 * Im01x * CDM0[3][Mc_AN][h_AN][i][j];

                            dy = Re00y * CDM0[0][Mc_AN][h_AN][i][j]
                                + Re11y * CDM0[1][Mc_AN][h_AN][i][j]
                                + 2.0 * Re01y * CDM0[2][Mc_AN][h_AN][i][j]
                                - Im00y * iDM0[0][Mc_AN][h_AN][i][j]
                                - Im11y * iDM0[1][Mc_AN][h_AN][i][j]
                                - 2.0 * Im01y * CDM0[3][Mc_AN][h_AN][i][j];

                            dz = Re00z * CDM0[0][Mc_AN][h_AN][i][j]
                                + Re11z * CDM0[1][Mc_AN][h_AN][i][j]
                                + 2.0 * Re01z * CDM0[2][Mc_AN][h_AN][i][j]
                                - Im00z * iDM0[0][Mc_AN][h_AN][i][j]
                                - Im11z * iDM0[1][Mc_AN][h_AN][i][j]
                                - 2.0 * Im01z * CDM0[3][Mc_AN][h_AN][i][j];

                            Fx[Mc_AN] += 0.5 * dx;
                            Fy[Mc_AN] += 0.5 * dy;
                            Fz[Mc_AN] += 0.5 * dz;

                            Re00x = 0.0;
                            Re00y = 0.0;
                            Re00z = 0.0;
                            Re11x = 0.0;
                            Re11y = 0.0;
                            Re11z = 0.0;
                            Re01x = 0.0;
                            Re01y = 0.0;
                            Re01z = 0.0;

                            Im00x = 0.0;
                            Im00y = 0.0;
                            Im00z = 0.0;
                            Im11x = 0.0;
                            Im11y = 0.0;
                            Im11z = 0.0;
                            Im01x = 0.0;
                            Im01y = 0.0;
                            Im01z = 0.0;

                            for (k = 0; k < Spe_Total_NO[Hwan]; k++) {

                                Re00x += NC_v_eff[0][0][Mh_AN][j][k].r * OLP[1][Mc_AN][h_AN][i][k];
                                Re00y += NC_v_eff[0][0][Mh_AN][j][k].r * OLP[2][Mc_AN][h_AN][i][k];
                                Re00z += NC_v_eff[0][0][Mh_AN][j][k].r * OLP[3][Mc_AN][h_AN][i][k];

                                Re11x += NC_v_eff[1][1][Mh_AN][j][k].r * OLP[1][Mc_AN][h_AN][i][k];
                                Re11y += NC_v_eff[1][1][Mh_AN][j][k].r * OLP[2][Mc_AN][h_AN][i][k];
                                Re11z += NC_v_eff[1][1][Mh_AN][j][k].r * OLP[3][Mc_AN][h_AN][i][k];

                                Re01x += NC_v_eff[0][1][Mh_AN][j][k].r * OLP[1][Mc_AN][h_AN][i][k];
                                Re01y += NC_v_eff[0][1][Mh_AN][j][k].r * OLP[2][Mc_AN][h_AN][i][k];
                                Re01z += NC_v_eff[0][1][Mh_AN][j][k].r * OLP[3][Mc_AN][h_AN][i][k];

                                Im00x += NC_v_eff[0][0][Mh_AN][j][k].i * OLP[1][Mc_AN][h_AN][i][k];
                                Im00y += NC_v_eff[0][0][Mh_AN][j][k].i * OLP[2][Mc_AN][h_AN][i][k];
                                Im00z += NC_v_eff[0][0][Mh_AN][j][k].i * OLP[3][Mc_AN][h_AN][i][k];

                                Im11x += NC_v_eff[1][1][Mh_AN][j][k].i * OLP[1][Mc_AN][h_AN][i][k];
                                Im11y += NC_v_eff[1][1][Mh_AN][j][k].i * OLP[2][Mc_AN][h_AN][i][k];
                                Im11z += NC_v_eff[1][1][Mh_AN][j][k].i * OLP[3][Mc_AN][h_AN][i][k];

                                Im01x += NC_v_eff[0][1][Mh_AN][j][k].i * OLP[1][Mc_AN][h_AN][i][k];
                                Im01y += NC_v_eff[0][1][Mh_AN][j][k].i * OLP[2][Mc_AN][h_AN][i][k];
                                Im01z += NC_v_eff[0][1][Mh_AN][j][k].i * OLP[3][Mc_AN][h_AN][i][k];
                            }

                            for (k = 0; k < Spe_Total_NO[Cwan]; k++) {

                                Re00x += NC_v_eff[0][0][Mc_AN][k][i].r * OLP[1][Mc_AN][h_AN][k][j];
                                Re00y += NC_v_eff[0][0][Mc_AN][k][i].r * OLP[2][Mc_AN][h_AN][k][j];
                                Re00z += NC_v_eff[0][0][Mc_AN][k][i].r * OLP[3][Mc_AN][h_AN][k][j];

                                Re11x += NC_v_eff[1][1][Mc_AN][k][i].r * OLP[1][Mc_AN][h_AN][k][j];
                                Re11y += NC_v_eff[1][1][Mc_AN][k][i].r * OLP[2][Mc_AN][h_AN][k][j];
                                Re11z += NC_v_eff[1][1][Mc_AN][k][i].r * OLP[3][Mc_AN][h_AN][k][j];

                                Re01x += NC_v_eff[0][1][Mc_AN][k][i].r * OLP[1][Mc_AN][h_AN][k][j];
                                Re01y += NC_v_eff[0][1][Mc_AN][k][i].r * OLP[2][Mc_AN][h_AN][k][j];
                                Re01z += NC_v_eff[0][1][Mc_AN][k][i].r * OLP[3][Mc_AN][h_AN][k][j];

                                Im00x += NC_v_eff[0][0][Mc_AN][k][i].i * OLP[1][Mc_AN][h_AN][k][j];
                                Im00y += NC_v_eff[0][0][Mc_AN][k][i].i * OLP[2][Mc_AN][h_AN][k][j];
                                Im00z += NC_v_eff[0][0][Mc_AN][k][i].i * OLP[3][Mc_AN][h_AN][k][j];

                                Im11x += NC_v_eff[1][1][Mc_AN][k][i].i * OLP[1][Mc_AN][h_AN][k][j];
                                Im11y += NC_v_eff[1][1][Mc_AN][k][i].i * OLP[2][Mc_AN][h_AN][k][j];
                                Im11z += NC_v_eff[1][1][Mc_AN][k][i].i * OLP[3][Mc_AN][h_AN][k][j];

                                Im01x += NC_v_eff[0][1][Mc_AN][k][i].i * OLP[1][Mc_AN][h_AN][k][j];
                                Im01y += NC_v_eff[0][1][Mc_AN][k][i].i * OLP[2][Mc_AN][h_AN][k][j];
                                Im01z += NC_v_eff[0][1][Mc_AN][k][i].i * OLP[3][Mc_AN][h_AN][k][j];
                            }

                            dx = Re00x * CDM0[0][Mh_AN][kl][j][i]
                                + Re11x * CDM0[1][Mh_AN][kl][j][i]
                                + 2.0 * Re01x * CDM0[2][Mh_AN][kl][j][i]
                                - Im00x * iDM0[0][Mh_AN][kl][j][i]
                                - Im11x * iDM0[1][Mh_AN][kl][j][i]
                                - 2.0 * Im01x * CDM0[3][Mh_AN][kl][j][i];

                            dy = Re00y * CDM0[0][Mh_AN][kl][j][i]
                                + Re11y * CDM0[1][Mh_AN][kl][j][i]
                                + 2.0 * Re01y * CDM0[2][Mh_AN][kl][j][i]
                                - Im00y * iDM0[0][Mh_AN][kl][j][i]
                                - Im11y * iDM0[1][Mh_AN][kl][j][i]
                                - 2.0 * Im01y * CDM0[3][Mh_AN][kl][j][i];

                            dz = Re00z * CDM0[0][Mh_AN][kl][j][i]
                                + Re11z * CDM0[1][Mh_AN][kl][j][i]
                                + 2.0 * Re01z * CDM0[2][Mh_AN][kl][j][i]
                                - Im00z * iDM0[0][Mh_AN][kl][j][i]
                                - Im11z * iDM0[1][Mh_AN][kl][j][i]
                                - 2.0 * Im01z * CDM0[3][Mh_AN][kl][j][i];

                            Fx[Mc_AN] += 0.5 * dx;
                            Fy[Mc_AN] += 0.5 * dy;
                            Fz[Mc_AN] += 0.5 * dz;
                        }
                    }
                }

                dtime(&Etime_atom);
                time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
            }
        }

        /****************************************************
          add the contribution
        ****************************************************/

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

            Gc_AN = M2G[Mc_AN];

            Gxyz[Gc_AN][17] += Fx[Mc_AN];
            Gxyz[Gc_AN][18] += Fy[Mc_AN];
            Gxyz[Gc_AN][19] += Fz[Mc_AN];

            if (2 <= level_stdout) {
                printf("<Force>  force(LDA_U_dual) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                    myid, Mc_AN, Gc_AN, Fx[Mc_AN], Fy[Mc_AN], Fz[Mc_AN]);
                fflush(stdout);
            }
        }

    } /* if ( (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
         && F_U_flag==1 && Hub_U_occupation==2) */

    /****************************************************
                   Force arising from HNL
    ****************************************************/

    dtime(&stime);
    Force_HNL(CDM0, iDM0);
    dtime(&etime);
    if (force_profile) {
        Force_profile_report("force-HNL", etime - stime,
            mpi_comm_level1, myid, numprocs);
        profile_stamp = etime;
    }

    /****************************************************
          Force arising from the penalty functional
          to create a core hole
    ****************************************************/

    if (core_hole_state_flag == 1) {
        Force_CoreHole(CDM0, iDM0);
    }

    /****************************************************
     freeing of arrays:
    ****************************************************/

    if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
        && (Hub_U_occupation == 1 || Hub_U_occupation == 2)
        && SpinP_switch != 3) {

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(HUx[i][j]);
            }
            free(HUx[i]);
        }
        free(HUx);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(HUy[i][j]);
            }
            free(HUy[i]);
        }
        free(HUy);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(HUz[i][j]);
            }
            free(HUz[i]);
        }
        free(HUz);
    }

    free(Fx);
    free(Fy);
    free(Fz);

    for (j = 0; j < List_YOUSO[7]; j++) {
        free(HVNAx[j]);
    }
    free(HVNAx);

    for (j = 0; j < List_YOUSO[7]; j++) {
        free(HVNAy[j]);
    }
    free(HVNAy);

    for (j = 0; j < List_YOUSO[7]; j++) {
        free(HVNAz[j]);
    }
    free(HVNAz);

    /* CDM0 */
    for (k = 0; k <= SpinP_switch; k++) {
        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN = 0;
                tno0 = 1;
            } else {
                Gc_AN = F_M2G[Mc_AN];
                Cwan = WhatSpecies[Gc_AN];
                tno0 = Spe_Total_CNO[Cwan];
            }

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno1 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    tno1 = Spe_Total_CNO[Hwan];
                }

                for (i = 0; i < tno0; i++) {
                    free(CDM0[k][Mc_AN][h_AN][i]);
                }
                free(CDM0[k][Mc_AN][h_AN]);
            }
            free(CDM0[k][Mc_AN]);
        }
        free(CDM0[k]);
    }
    free(CDM0);

    free(Snd_CDM0_Size);
    free(Rcv_CDM0_Size);

    /* iDM0 */
    if (use_iDM0) {

        for (k = 0; k < 2; k++) {

            FNAN[0] = 0;
            for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++) {

                if (Mc_AN == 0) {
                    Gc_AN = 0;
                    tno0 = 1;
                } else {
                    Gc_AN = F_M2G[Mc_AN];
                    Cwan = WhatSpecies[Gc_AN];
                    tno0 = Spe_Total_CNO[Cwan];
                }

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                    if (Mc_AN == 0) {
                        tno1 = 1;
                    } else {
                        Gh_AN = natn[Gc_AN][h_AN];
                        Hwan = WhatSpecies[Gh_AN];
                        tno1 = Spe_Total_CNO[Hwan];
                    }

                    for (i = 0; i < tno0; i++) {
                        free(iDM0[k][Mc_AN][h_AN][i]);
                    }
                    free(iDM0[k][Mc_AN][h_AN]);
                }
                free(iDM0[k][Mc_AN]);
            }
            free(iDM0[k]);
        }
        free(iDM0);

        free(Snd_iDM0_Size);
        free(Rcv_iDM0_Size);
    }

    /* for time */

    MPI_Barrier(mpi_comm_level1);
    dtime(&TEtime);
    time0 = TEtime - TStime;

    if (force_profile) {
        Force_profile_report("cleanup", TEtime - profile_stamp,
            mpi_comm_level1, myid, numprocs);
        Force_profile_report("total", TEtime - profile_total_start,
            mpi_comm_level1, myid, numprocs);
    }

    return time0;
}

/* OMP by A. Ito */

#ifndef f3omp
void Force3()
{
    /****************************************************
          #3 of Force
          dn/dx * (VNA + dVH + Vxc)
          or
          dn/dx * (dVH + Vxc)
    ****************************************************/
    /* for OpenMP */

    /* MPI */
    int numprocs, myid;
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /**********************************************************
                main loop for calculation of force #3
    **********************************************************/
    /* shared memory for force */

    double** Vpot_grid = (double**)malloc(sizeof(double*) * (SpinP_switch + 1));
    {
        double* p2 = (double*)malloc(sizeof(double) * (SpinP_switch + 1) * Max_GridN_Atom);

        int spin;
        for (spin = 0; spin < (SpinP_switch + 1); spin++) {
            Vpot_grid[spin] = p2;
            p2 += Max_GridN_Atom;
        }
    }

    double*** dChi0 = (double***)malloc(sizeof(double**) * Max_GridN_Atom);
    {
        double** p2 = (double**)malloc(sizeof(double*) * Max_GridN_Atom * List_YOUSO[7]);
        double* p = (double*)malloc(sizeof(double) * Max_GridN_Atom * List_YOUSO[7] * 3);
        int Nc;
        for (Nc = 0; Nc < Max_GridN_Atom; Nc++) {
            dChi0[Nc] = p2;
            p2 += List_YOUSO[7];
            int i;
            for (i = 0; i < List_YOUSO[7]; i++) {
                dChi0[Nc][i] = p;
                p += 3;
            }
        }
    }

    double sumx = 0.0; /* this must be defined out of parallel pragma */
    double sumy = 0.0;
    double sumz = 0.0;

    /* Batched device path for the part-2 trace: part 1 then writes the
       orbital derivatives and grid potentials of every atom into flat
       buffers, and one chunked kernel replaces the per-atom h_AN loops. */

    int use_gpu_trace = 0;
    int f3_dorb = 0;
    double* dchi_all = NULL;
    size_t* dchi_off = NULL;
    size_t* vpot_off = NULL;
    size_t* xyz_off = NULL;
    /* the per-spin grid potentials and grid coordinates are staged one
       atom at a time and live on the device only; a full host copy of
       either multiplies the node's host memory by the rank count */
    double* vpot_stage = NULL;
    double* xyz_stage = NULL;
    double* vpot_dev_full = NULL;
    double* xyz_dev_full = NULL;

    size_t f3_chunk_budget = Force3_GpuChunkBudget();

    if (f3_chunk_budget != 0) {
        size_t dpos = 0, vpos = 0, xpos = 0, maxgrid = 0;
        int mc;

        f3_dorb = Force3_GpuDorbMode();

        dchi_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)(Matomnum + 2),
            __FILE__, __LINE__);
        vpot_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)(Matomnum + 2),
            __FILE__, __LINE__);
        xyz_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)(Matomnum + 2),
            __FILE__, __LINE__);
        dchi_off[0] = 0;
        vpot_off[0] = 0;
        xyz_off[0] = 0;
        for (mc = 1; mc <= Matomnum; mc++) {
            int ga = M2G[mc];
            int no0 = Spe_Total_CNO[WhatSpecies[ga]];

            dchi_off[mc] = dpos;
            vpot_off[mc] = vpos;
            xyz_off[mc] = xpos;
            dpos += (size_t)GridN_Atom[ga] * (size_t)no0 * 3;
            vpos += (size_t)(SpinP_switch + 1) * (size_t)GridN_Atom[ga];
            xpos += (size_t)GridN_Atom[ga] * 3;
            if (maxgrid < (size_t)GridN_Atom[ga]) maxgrid = (size_t)GridN_Atom[ga];
        }
        dchi_off[Matomnum + 1] = dpos;
        vpot_off[Matomnum + 1] = vpos;
        xyz_off[Matomnum + 1] = xpos;

        /* with the on-device orbital derivatives only the grid coordinates
           travel to the device; otherwise part 1 fills the flat derivative
           buffer on the host */
        if (f3_dorb) {
            xyz_stage = (double*)Force_checked_malloc(
                sizeof(double) * 3 * (maxgrid == 0 ? 1 : maxgrid), __FILE__, __LINE__);
            xyz_dev_full = (double*)Force_gpu_arena_wait(
                sizeof(double) * (xpos == 0 ? 1 : xpos), "GPU trace grid coordinates");
        } else {
            dchi_all = (double*)Force_checked_malloc(sizeof(double) * (dpos == 0 ? 1 : dpos),
                __FILE__, __LINE__);
        }
        vpot_stage = (double*)Force_checked_malloc(
            sizeof(double) * (size_t)(SpinP_switch + 1) * (maxgrid == 0 ? 1 : maxgrid),
            __FILE__, __LINE__);
        vpot_dev_full = (double*)Force_gpu_arena_wait(
            sizeof(double) * (vpos == 0 ? 1 : vpos), "GPU trace grid potentials");
        use_gpu_trace = 1;
    }

#pragma omp parallel
    {

        /* allocation of arrays */

        double** dorbs0 = (double**)malloc(sizeof(double*) * 4);
        {
            int i;
            for (i = 0; i < 4; i++) {
                dorbs0[i] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
            }
        }
        double* orbs1 = (double*)malloc(sizeof(double) * List_YOUSO[7]);

        struct WORK_DORBITAL work_dObs;
        Get_dOrbitals_init(&work_dObs);

#if measure_time
        double time1 = 0.0;
        double time2 = 0.0;
        double last_time;
        double current_time;
#endif

        int Mc_AN;
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

            int Gc_AN = M2G[Mc_AN];
            int Cwan = WhatSpecies[Gc_AN];
            int NO0 = Spe_Total_CNO[Cwan];

            /* part-1 write targets: the flat per-atom slices on the GPU
               path, the shared per-atom buffers on the CPU path */
            double* vpot_dst[4];
            {
                int s;
                for (s = 0; s <= SpinP_switch; s++) {
                    vpot_dst[s] = use_gpu_trace
                        ? (vpot_stage + (size_t)s * (size_t)GridN_Atom[Gc_AN])
                        : Vpot_grid[s];
                }
            }

            /***********************************
                         calc dOrb0
            ***********************************/
#pragma omp barrier /* wait until the previous atom's master update has completed */
#pragma omp master
            {
                sumx = 0.0;
                sumy = 0.0;
                sumz = 0.0;
            }
#pragma omp barrier
#if measure_time
            dtime(&last_time);
#endif

            int Nc;
#pragma omp for
            for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {

                int GNc = GridListAtom[Mc_AN][Nc];
                int GRc = CellListAtom[Mc_AN][Nc];
                int MNc = MGridListAtom[Mc_AN][Nc];

                double Cxyz[4];
                Get_Grid_XYZ(GNc, Cxyz);
                double x = Cxyz[1] + atv[GRc][1];
                double y = Cxyz[2] + atv[GRc][2];
                double z = Cxyz[3] + atv[GRc][3];
                double dx = x - Gxyz[Gc_AN][1];
                double dy = y - Gxyz[Gc_AN][2];
                double dz = z - Gxyz[Gc_AN][3];

                if (f3_dorb) {
                    /* the device kernel evaluates the derivatives itself */
                    size_t xb = (size_t)Nc * 3;

                    xyz_stage[xb + 0] = dx;
                    xyz_stage[xb + 1] = dy;
                    xyz_stage[xb + 2] = dz;
                }
                else {

                if (Cnt_switch == 0) {
                    /* AITUNE201704 : Get_dOrbitals(Cwan, dx, dy, dz, dorbs0); */
                    /* Get_dOrbitals_work(Cwan, dx, dy, dz, dorbs0, work_dObs); */

                    /* start: direct inlining of Get_dOrbitals_work */

                    int wan = Cwan;
                    double x = dx, y = dy, z = dz;
                    int i, i1, i2, i3, i4, j, l, l1;
                    int po, L0, Mul0, M0;
                    int mp_min, mp_max, m;
                    double dum, dum1, dum2, dum3, dum4;
                    double siQ, coQ, siP, coP, a, b, c;
                    double dx, rm, tmp0, tmp1, id, d;
                    double drx, dry, drz, R, Q, P, Rmin;
                    double S_coordinate[3];
                    double** RF = work_dObs.RF;
                    double** dRF = work_dObs.dRF;
                    double** AF = work_dObs.AF;
                    double** dAFQ = work_dObs.dAFQ;
                    double** dAFP = work_dObs.dAFP;
                    double h1, h2, h3, f1, f2, f3, f4, dfx, dfx2;
                    double g1, g2, x1, x2, y1, y2, y12, y22, f, df, df2;
                    double dRx, dRy, dRz, dQx, dQy, dQz, dPx, dPy, dPz;
                    double dChiR, dChiQ, dChiP, h, sum0, sum1;
                    double SH[Supported_MaxL * 2 + 1][2];
                    double dSHt[Supported_MaxL * 2 + 1][2];
                    double dSHp[Supported_MaxL * 2 + 1][2];

                    /* start calc. */

                    Rmin = 10e-14;

                    xyz2spherical(x, y, z, 0.0, 0.0, 0.0, S_coordinate);
                    R = S_coordinate[0];
                    Q = S_coordinate[1];
                    P = S_coordinate[2];

                    if (R < Rmin) {
                        x = x + Rmin;
                        y = y + Rmin;
                        z = z + Rmin;
                        xyz2spherical(x, y, z, 0.0, 0.0, 0.0, S_coordinate);
                        R = S_coordinate[0];
                        Q = S_coordinate[1];
                        P = S_coordinate[2];
                    }

                    po = 0;
                    mp_min = 0;
                    mp_max = Spe_Num_Mesh_PAO[wan] - 1;

                    if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan] - 1] < R) {

                        for (L0 = 0; L0 <= Spe_MaxL_Basis[wan]; L0++) {
                            for (Mul0 = 0; Mul0 < Spe_Num_Basis[wan][L0]; Mul0++) {
                                RF[L0][Mul0] = 0.0;
                                dRF[L0][Mul0] = 0.0;
                            }
                        }

                        po = 1;
                    }

                    else if (R < Spe_PAO_RV[wan][0]) {

                        m = 4;
                        rm = Spe_PAO_RV[wan][m];

                        h1 = Spe_PAO_RV[wan][m - 1] - Spe_PAO_RV[wan][m - 2];
                        h2 = Spe_PAO_RV[wan][m] - Spe_PAO_RV[wan][m - 1];
                        h3 = Spe_PAO_RV[wan][m + 1] - Spe_PAO_RV[wan][m];

                        x1 = rm - Spe_PAO_RV[wan][m - 1];
                        x2 = rm - Spe_PAO_RV[wan][m];
                        y1 = x1 / h2;
                        y2 = x2 / h2;
                        y12 = y1 * y1;
                        y22 = y2 * y2;

                        dum = h1 + h2;
                        dum1 = h1 / h2 / dum;
                        dum2 = h2 / h1 / dum;
                        dum = h2 + h3;
                        dum3 = h2 / h3 / dum;
                        dum4 = h3 / h2 / dum;

                        for (L0 = 0; L0 <= Spe_MaxL_Basis[wan]; L0++) {
                            for (Mul0 = 0; Mul0 < Spe_Num_Basis[wan][L0]; Mul0++) {

                                f1 = Spe_PAO_RWF[wan][L0][Mul0][m - 2];
                                f2 = Spe_PAO_RWF[wan][L0][Mul0][m - 1];
                                f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
                                f4 = Spe_PAO_RWF[wan][L0][Mul0][m + 1];

                                if (m == 1) {
                                    h1 = -(h2 + h3);
                                    f1 = f4;
                                } else if (m == (Spe_Num_Mesh_PAO[wan] - 1)) {
                                    h3 = -(h1 + h2);
                                    f4 = f1;
                                }

                                dum = f3 - f2;
                                g1 = dum * dum1 + (f2 - f1) * dum2;
                                g2 = (f4 - f3) * dum3 + dum * dum4;

                                f = y22 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2)
                                    + y12 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1);

                                df = 2.0 * y2 / h2 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2)
                                    + y22 * (2.0 * f2 + h2 * g1) / h2
                                    + 2.0 * y1 / h2 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1)
                                    - y12 * (2.0 * f3 - h2 * g2) / h2;

                                if (L0 == 0) {
                                    a = 0.0;
                                    b = 0.5 * df / rm;
                                    c = 0.0;
                                    d = f - b * rm * rm;
                                }

                                else if (L0 == 1) {
                                    a = (rm * df - f) / (2.0 * rm * rm * rm);
                                    b = 0.0;
                                    c = df - 3.0 * a * rm * rm;
                                    d = 0.0;
                                }

                                else {
                                    b = (3.0 * f - rm * df) / (rm * rm);
                                    a = (f - b * rm * rm) / (rm * rm * rm);
                                    c = 0.0;
                                    d = 0.0;
                                }

                                RF[L0][Mul0] = a * R * R * R + b * R * R + c * R + d;
                                dRF[L0][Mul0] = 3.0 * a * R * R + 2.0 * b * R + c;
                            }
                        }

                    }

                    else {

                        do {
                            m = (mp_min + mp_max) / 2;
                            if (Spe_PAO_RV[wan][m] < R)
                                mp_min = m;
                            else
                                mp_max = m;
                        } while ((mp_max - mp_min) != 1);
                        m = mp_max;

                        h1 = Spe_PAO_RV[wan][m - 1] - Spe_PAO_RV[wan][m - 2];
                        h2 = Spe_PAO_RV[wan][m] - Spe_PAO_RV[wan][m - 1];
                        h3 = Spe_PAO_RV[wan][m + 1] - Spe_PAO_RV[wan][m];

                        x1 = R - Spe_PAO_RV[wan][m - 1];
                        x2 = R - Spe_PAO_RV[wan][m];
                        y1 = x1 / h2;
                        y2 = x2 / h2;
                        y12 = y1 * y1;
                        y22 = y2 * y2;

                        dum = h1 + h2;
                        dum1 = h1 / h2 / dum;
                        dum2 = h2 / h1 / dum;
                        dum = h2 + h3;
                        dum3 = h2 / h3 / dum;
                        dum4 = h3 / h2 / dum;

                        for (L0 = 0; L0 <= Spe_MaxL_Basis[wan]; L0++) {
                            for (Mul0 = 0; Mul0 < Spe_Num_Basis[wan][L0]; Mul0++) {

                                f1 = Spe_PAO_RWF[wan][L0][Mul0][m - 2];
                                f2 = Spe_PAO_RWF[wan][L0][Mul0][m - 1];
                                f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
                                f4 = Spe_PAO_RWF[wan][L0][Mul0][m + 1];

                                if (m == 1) {
                                    h1 = -(h2 + h3);
                                    f1 = f4;
                                } else if (m == (Spe_Num_Mesh_PAO[wan] - 1)) {
                                    h3 = -(h1 + h2);
                                    f4 = f1;
                                }

                                dum = f3 - f2;
                                g1 = dum * dum1 + (f2 - f1) * dum2;
                                g2 = (f4 - f3) * dum3 + dum * dum4;

                                f = y22 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2)
                                    + y12 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1);

                                df = 2.0 * y2 / h2 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2)
                                    + y2 * y2 * (2.0 * f2 + h2 * g1) / h2
                                    + 2.0 * y1 / h2 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1)
                                    - y1 * y1 * (2.0 * f3 - h2 * g2) / h2;

                                RF[L0][Mul0] = f;
                                dRF[L0][Mul0] = df;
                            }
                        }
                    }

                    /* dr/dx,y,z, dQ/dx,y,z, dP/dx,y,z and dAngular */

                    if (po == 0) {

                        /* Angular */
                        siQ = sin(Q);
                        coQ = cos(Q);
                        siP = sin(P);
                        coP = cos(P);

                        dRx = siQ * coP;
                        dRy = siQ * siP;
                        dRz = coQ;

                        if (Rmin < R) {
                            dQx = coQ * coP / R;
                            dQy = coQ * siP / R;
                            dQz = -siQ / R;
                        } else {
                            dQx = 0.0;
                            dQy = 0.0;
                            dQz = 0.0;
                        }

                        /* RICS note 72P */

                        if (Rmin < R) {
                            dPx = -siP / R;
                            dPy = coP / R;
                            dPz = 0.0;
                        } else {
                            dPx = 0.0;
                            dPy = 0.0;
                            dPz = 0.0;
                        }

                        for (L0 = 0; L0 <= Spe_MaxL_Basis[wan]; L0++) {
                            if (L0 == 0) {
                                AF[0][0] = 0.282094791773878;
                                dAFQ[0][0] = 0.0;
                                dAFP[0][0] = 0.0;
                            } else if (L0 == 1) {
                                dum = 0.48860251190292 * siQ;

                                AF[1][0] = dum * coP;
                                AF[1][1] = dum * siP;
                                AF[1][2] = 0.48860251190292 * coQ;

                                dAFQ[1][0] = 0.48860251190292 * coQ * coP;
                                dAFQ[1][1] = 0.48860251190292 * coQ * siP;
                                dAFQ[1][2] = -0.48860251190292 * siQ;

                                dAFP[1][0] = -0.48860251190292 * siP;
                                dAFP[1][1] = 0.48860251190292 * coP;
                                dAFP[1][2] = 0.0;
                            } else if (L0 == 2) {

                                dum1 = siQ * siQ;
                                dum2 = 1.09254843059208 * siQ * coQ;
                                AF[2][0] = 0.94617469575756 * coQ * coQ - 0.31539156525252;
                                AF[2][1] = 0.54627421529604 * dum1 * (1.0 - 2.0 * siP * siP);
                                AF[2][2] = 1.09254843059208 * dum1 * siP * coP;
                                AF[2][3] = dum2 * coP;
                                AF[2][4] = dum2 * siP;

                                dAFQ[2][0] = -1.89234939151512 * siQ * coQ;
                                dAFQ[2][1] = 1.09254843059208 * siQ * coQ * (1.0 - 2.0 * siP * siP);
                                dAFQ[2][2] = 2.18509686118416 * siQ * coQ * siP * coP;
                                dAFQ[2][3] = 1.09254843059208 * (1.0 - 2.0 * siQ * siQ) * coP;
                                dAFQ[2][4] = 1.09254843059208 * (1.0 - 2.0 * siQ * siQ) * siP;

                                /* RICS note 72P */

                                dAFP[2][0] = 0.0;
                                dAFP[2][1] = -2.18509686118416 * siQ * siP * coP;
                                dAFP[2][2] = 1.09254843059208 * siQ * (1.0 - 2.0 * siP * siP);
                                dAFP[2][3] = -1.09254843059208 * coQ * siP;
                                dAFP[2][4] = 1.09254843059208 * coQ * coP;
                            }

                            else if (L0 == 3) {
                                AF[3][0] = 0.373176332590116 * (5.0 * coQ * coQ * coQ - 3.0 * coQ);
                                AF[3][1] = 0.457045799464466 * coP * siQ * (5.0 * coQ * coQ - 1.0);
                                AF[3][2] = 0.457045799464466 * siP * siQ * (5.0 * coQ * coQ - 1.0);
                                AF[3][3] = 1.44530572132028 * siQ * siQ * coQ * (coP * coP - siP * siP);
                                AF[3][4] = 2.89061144264055 * siQ * siQ * coQ * siP * coP;
                                AF[3][5] = 0.590043589926644 * siQ * siQ * siQ * (4.0 * coP * coP * coP - 3.0 * coP);
                                AF[3][6] = 0.590043589926644 * siQ * siQ * siQ * (3.0 * siP - 4.0 * siP * siP * siP);

                                dAFQ[3][0] = 0.373176332590116 * siQ * (-15.0 * coQ * coQ + 3.0);
                                dAFQ[3][1] = 0.457045799464466 * coP * coQ * (15.0 * coQ * coQ - 11.0);
                                dAFQ[3][2] = 0.457045799464466 * siP * coQ * (15.0 * coQ * coQ - 11.0);
                                dAFQ[3][3] = 1.44530572132028 * (coP * coP - siP * siP) * siQ * (2.0 * coQ * coQ - siQ * siQ);
                                dAFQ[3][4] = 2.89061144264055 * coP * siP * siQ * (2.0 * coQ * coQ - siQ * siQ);
                                dAFQ[3][5] = 1.770130769779932 * coP * coQ * siQ * siQ * (-3.0 + 4.0 * coP * coP);
                                dAFQ[3][6] = 1.770130769779932 * coQ * siP * siQ * siQ * (3.0 - 4.0 * siP * siP);

                                /* RICS note 72P */

                                dAFP[3][0] = 0.0;
                                dAFP[3][1] = 0.457045799464466 * siP * (-5.0 * coQ * coQ + 1.0);
                                dAFP[3][2] = 0.457045799464466 * coP * (5.0 * coQ * coQ - 1.0);
                                dAFP[3][3] = -5.781222885281120 * coP * coQ * siP * siQ;
                                dAFP[3][4] = 2.89061144264055 * coQ * siQ * (coP * coP - siP * siP);
                                dAFP[3][5] = 1.770130769779932 * siP * siQ * siQ * (1.0 - 4.0 * coP * coP);
                                dAFP[3][6] = 1.770130769779932 * coP * siQ * siQ * (1.0 - 4.0 * siP * siP);
                            }

                            else if (4 <= L0) {

                                /* calculation of complex spherical harmonics functions */
                                for (m = -L0; m <= L0; m++) {
                                    ComplexSH(L0, m, Q, P, SH[L0 + m], dSHt[L0 + m], dSHp[L0 + m]);
                                }

                                /* transformation of complex to real */
                                for (i = 0; i < (L0 * 2 + 1); i++) {

                                    sum0 = 0.0;
                                    sum1 = 0.0;
                                    for (j = 0; j < (L0 * 2 + 1); j++) {
                                        sum0 += Comp2Real[L0][i][j].r * SH[j][0] - Comp2Real[L0][i][j].i * SH[j][1];
                                        sum1 += Comp2Real[L0][i][j].r * SH[j][1] + Comp2Real[L0][i][j].i * SH[j][0];
                                    }
                                    AF[L0][i] = sum0 + sum1;

                                    sum0 = 0.0;
                                    sum1 = 0.0;
                                    for (j = 0; j < (L0 * 2 + 1); j++) {
                                        sum0 += Comp2Real[L0][i][j].r * dSHt[j][0] - Comp2Real[L0][i][j].i * dSHt[j][1];
                                        sum1 += Comp2Real[L0][i][j].r * dSHt[j][1] + Comp2Real[L0][i][j].i * dSHt[j][0];
                                    }
                                    dAFQ[L0][i] = sum0 + sum1;

                                    sum0 = 0.0;
                                    sum1 = 0.0;
                                    for (j = 0; j < (L0 * 2 + 1); j++) {
                                        sum0 += Comp2Real[L0][i][j].r * dSHp[j][0] - Comp2Real[L0][i][j].i * dSHp[j][1];
                                        sum1 += Comp2Real[L0][i][j].r * dSHp[j][1] + Comp2Real[L0][i][j].i * dSHp[j][0];
                                    }
                                    dAFP[L0][i] = sum0 + sum1;
                                }
                            }
                        }
                    }

                    /* Chi */
                    i1 = -1;
                    for (L0 = 0; L0 <= Spe_MaxL_Basis[wan]; L0++) {
                        for (Mul0 = 0; Mul0 < Spe_Num_Basis[wan][L0]; Mul0++) {
                            for (M0 = 0; M0 <= 2 * L0; M0++) {

                                i1++;

                                dChiR = dRF[L0][Mul0] * AF[L0][M0];
                                dChiQ = RF[L0][Mul0] * dAFQ[L0][M0];
                                dChiP = RF[L0][Mul0] * dAFP[L0][M0];

                                dorbs0[1][i1] = -dRx * dChiR - dQx * dChiQ - dPx * dChiP;
                                dorbs0[2][i1] = -dRy * dChiR - dQy * dChiQ - dPy * dChiP;
                                dorbs0[3][i1] = -dRz * dChiR - dQz * dChiQ - dPz * dChiP;
                            }
                        }
                    }

                    /* end: direct inlining of Get_dOrbitals_work */

                } else {
                    Get_Cnt_dOrbitals(Mc_AN, dx, dy, dz, dorbs0);
                }

                int i;
                for (i = 0; i < NO0; i++) {
                    double* dchi = use_gpu_trace
                        ? (dchi_all + dchi_off[Mc_AN] + ((size_t)Nc * (size_t)NO0 + (size_t)i) * 3)
                        : dChi0[Nc][i];
                    dchi[0] = dorbs0[1][i];
                    dchi[1] = dorbs0[2][i];
                    dchi[2] = dorbs0[3][i];
                }

                } /* if (f3_dorb) else */

                if (SpinP_switch == 0 || SpinP_switch == 1) {

                    int spin;
                    for (spin = 0; spin <= SpinP_switch; spin++) {

                        double Vpt;
                        if (0 <= MNc) {
                            if (E_Field_switch == 1) {

                                if (ProExpn_VNA == 0) {

                                    Vpt = F_dVHart_flag * dVHart_Grid[MNc]
                                        + F_Vxc_flag * Vxc_Grid[spin][MNc]
                                        + F_VNA_flag * VNA_Grid[MNc]
                                        + F_VEF_flag * VEF_Grid[MNc];

                                } else {

                                    Vpt = F_dVHart_flag * dVHart_Grid[MNc]
                                        + F_Vxc_flag * Vxc_Grid[spin][MNc]
                                        + F_VEF_flag * VEF_Grid[MNc];
                                }

                            } else {
                                if (ProExpn_VNA == 0) {

                                    Vpt = F_dVHart_flag * dVHart_Grid[MNc]
                                        + F_Vxc_flag * Vxc_Grid[spin][MNc]
                                        + F_VNA_flag * VNA_Grid[MNc];

                                } else {

                                    Vpt = F_dVHart_flag * dVHart_Grid[MNc]
                                        + F_Vxc_flag * Vxc_Grid[spin][MNc];
                                }
                            }
                        } else {
                            Vpt = 0.0;
                        }

                        if (SpinP_switch == 0) {
                            vpot_dst[0][Nc] = 4.0 * Vpt;
                        } else if (SpinP_switch == 1) {
                            vpot_dst[spin][Nc] = 2.0 * Vpt;
                        }
                    }
                }

                else if (SpinP_switch == 3) {

                    /* spin non-collinear */

                    double ReVpt11;
                    double ReVpt22;
                    double ReVpt21;
                    double ImVpt21;

                    if (0 <= MNc) {
                        if (E_Field_switch == 1) {

                            if (ProExpn_VNA == 0) {

                                ReVpt11 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[0][MNc]
                                    + F_VNA_flag * VNA_Grid[MNc]
                                    + F_VEF_flag * VEF_Grid[MNc];

                                ReVpt22 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[1][MNc]
                                    + F_VNA_flag * VNA_Grid[MNc]
                                    + F_VEF_flag * VEF_Grid[MNc];

                                ReVpt21 = F_Vxc_flag * Vxc_Grid[2][MNc];
                                ImVpt21 = -F_Vxc_flag * Vxc_Grid[3][MNc];
                            } else {

                                ReVpt11 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[0][MNc]
                                    + F_VEF_flag * VEF_Grid[MNc];

                                ReVpt22 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[1][MNc]
                                    + F_VEF_flag * VEF_Grid[MNc];

                                ReVpt21 = F_Vxc_flag * Vxc_Grid[2][MNc];
                                ImVpt21 = -F_Vxc_flag * Vxc_Grid[3][MNc];
                            }

                        } else {

                            if (ProExpn_VNA == 0) {

                                ReVpt11 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[0][MNc]
                                    + F_VNA_flag * VNA_Grid[MNc];

                                ReVpt22 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[1][MNc]
                                    + F_VNA_flag * VNA_Grid[MNc];

                                ReVpt21 = F_Vxc_flag * Vxc_Grid[2][MNc];
                                ImVpt21 = -F_Vxc_flag * Vxc_Grid[3][MNc];

                            } else {

                                ReVpt11 = F_dVHart_flag * dVHart_Grid[MNc] + F_Vxc_flag * Vxc_Grid[0][MNc];
                                ReVpt22 = F_dVHart_flag * dVHart_Grid[MNc] + F_Vxc_flag * Vxc_Grid[1][MNc];

                                ReVpt21 = F_Vxc_flag * Vxc_Grid[2][MNc];
                                ImVpt21 = -F_Vxc_flag * Vxc_Grid[3][MNc];
                            }
                        }
                    } else {
                        ReVpt11 = 0.0;
                        ReVpt22 = 0.0;
                        ReVpt21 = 0.0;
                        ImVpt21 = 0.0;
                    }

                    vpot_dst[0][Nc] = 2.0 * ReVpt11;
                    vpot_dst[1][Nc] = 2.0 * ReVpt22;
                    vpot_dst[2][Nc] = 4.0 * ReVpt21;
                    vpot_dst[3][Nc] = 4.0 * ImVpt21;
                }

            } /* Nc, here omp barrier is called implicitly because of end of for loop */

            if (use_gpu_trace) {
#pragma omp master
                {
                    if (0 < GridN_Atom[Gc_AN]) {
                        acc_memcpy_to_device(vpot_dev_full + vpot_off[Mc_AN], vpot_stage,
                            sizeof(double) * (size_t)(SpinP_switch + 1) * (size_t)GridN_Atom[Gc_AN]);
                        if (f3_dorb) {
                            acc_memcpy_to_device(xyz_dev_full + xyz_off[Mc_AN], xyz_stage,
                                sizeof(double) * 3 * (size_t)GridN_Atom[Gc_AN]);
                        }
                    }
                }
                /* no barrier needed here: the next atom's leading barrier keeps
                   the other threads from refilling the stage until the master
                   thread has finished the upload */
            }

#if measure_time
            dtime(&current_time);
            time1 += current_time - last_time;
            last_time = current_time;
#endif

            if (!use_gpu_trace) {

            int h_AN;
#pragma omp for schedule(static) reduction(+:sumx, sumy, sumz)
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                int Gh_AN = natn[Gc_AN][h_AN];
                int Mh_AN = F_G2M[Gh_AN];
                int Hwan = WhatSpecies[Gh_AN];
                int NO1 = Spe_Total_CNO[Hwan];

                int Nog;

                for (Nog = 0; Nog < NumOLG[Mc_AN][h_AN]; Nog++) {

                    int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
                    int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

                    double** const ai_dorbs0 = dChi0[Nc];

                    /* set orbs1 */

                    const Type_Orbs_Grid* restrict orbs1_source;
                    if (G2ID[Gh_AN] == myid) {
                        orbs1_source = Orbs_Grid[Mh_AN][Nh];
                    } else {
                        orbs1_source = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog];
                    }
                    int j;
#pragma omp simd
                    for (j = 0; j < NO1; j++) {
                        orbs1[j] = (double)orbs1_source[j];
                    }

                    int spin;
                    for (spin = 0; spin <= SpinP_switch; spin++) {
                        double** dm = DM[0][spin][Mc_AN][h_AN];

                        double tmpx = 0.0;
                        double tmpy = 0.0;
                        double tmpz = 0.0;

                        int i;
                        for (i = 0; i < NO0; i++) {
                            double tmp0 = 0.0;
                            const double* restrict dm_row = dm[i];
                            int j;
#pragma omp simd reduction(+:tmp0)
                            for (j = 0; j < NO1; j++) {
                                tmp0 += orbs1[j] * dm_row[j];
                            }

                            tmpx += ai_dorbs0[i][0] * tmp0;
                            tmpy += ai_dorbs0[i][1] * tmp0;
                            tmpz += ai_dorbs0[i][2] * tmp0;
                        }

                        /* due to difference in the definition between density matrix and density */
                        double Vpt = Vpot_grid[spin][Nc];
                        sumx += tmpx * Vpt;
                        sumy += tmpy * Vpt;
                        sumz += tmpz * Vpt;

                    } /* spin */
                } /* Nog */

            } /* h_AN, here omp barrier is called implicitly because of end of for loop  */

            /***********************************
                        calc force #3
            ***********************************/

#pragma omp master
            {
                Gxyz[Gc_AN][17] += sumx * GridVol;
                Gxyz[Gc_AN][18] += sumy * GridVol;
                Gxyz[Gc_AN][19] += sumz * GridVol;
            }

#if measure_time
            dtime(&current_time);
            time2 += current_time - last_time;
            last_time = current_time;
#endif

#pragma omp master
            if (2 <= level_stdout) {
                printf("<Force>  force(3) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                    myid, Mc_AN, Gc_AN, sumx * GridVol, sumy * GridVol, sumz * GridVol);
                fflush(stdout);

                /*
                printf("<Force>  force(3) myid=%2d  GridN_Atom[Gc_AN] = %d, FNAN[Gc_AN] = %d\n",
                       myid, GridN_Atom[Gc_AN], FNAN[Gc_AN]);
                */
            }

            } /* if (!use_gpu_trace) */

        } /* Mc_AN */

#if measure_time
#pragma omp master
        printf("<Force>  force(3) myid=%2d  time1, 2 = %lf [s], %lf [s]\n", myid, time1, time2);
#endif

        /* freeing of arrays */

        free(orbs1);

        int i;
        for (i = 0; i < 4; i++) {
            free(dorbs0[i]);
        }
        free(dorbs0);

        Get_dOrbitals_free(work_dObs);

    } /* #pragma omp parallel */

    /* batched device evaluation of the skipped part-2 traces */
    if (use_gpu_trace) {
        Force3_GpuTrace(dchi_all, dchi_off, vpot_dev_full, vpot_off,
            xyz_dev_full, xyz_off, f3_dorb, f3_chunk_budget);
    }

    /* free */
    if (xyz_dev_full != NULL) acc_free(xyz_dev_full);
    if (vpot_dev_full != NULL) acc_free(vpot_dev_full);
    free(xyz_stage);
    free(vpot_stage);
    free(dchi_all);
    free(xyz_off);
    free(vpot_off);
    free(dchi_off);

    free(dChi0[0][0]);
    free(dChi0[0]);
    free(dChi0);

    free(Vpot_grid[0]);
    free(Vpot_grid);
}

/* OMP by Ozaki */

#else
void Force3()
{
    /****************************************************
                        #3 of Force

                 dn/dx * (VNA + dVH + Vxc)
             or
                 dn/dx * (dVH + Vxc)
    ****************************************************/

    int Mc_AN, Gc_AN, Cwan, Hwan, NO0, NO1;
    int i, j, k, Nc, Nh, GNc, GRc, MNc, GNh, GRh;
    int h_AN, Gh_AN, Mh_AN, Rnh, spin, Nog;
    double*** dDen_Grid;
    double sum, tmp0, r, dx, dy, dz;
    double sumx, sumy, sumz;
    double x, y, z, x1, y1, z1, Vpt;
    double Cxyz[4];
    double **dorbs0, *orbs1, ***dChi0;
    double ReVpt11, ReVpt22, ReVpt21, ImVpt21;
    int numprocs, myid, tag = 999, ID, IDS, IDR;
    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /**********************************************************
                main loop for calculation of force #3
    **********************************************************/

#pragma omp parallel shared(Orbs_Grid_FNAN, G2ID, myid, level_stdout, GridVol, F_VEF_flag, VEF_Grid, F_VNA_flag, VNA_Grid, F_Vxc_flag, Vxc_Grid, dVHart_Grid, F_dVHart_flag, ProExpn_VNA, E_Field_switch, DM, Orbs_Grid, GListTAtoms2, GListTAtoms1, NumOLG, ncn, F_G2M, natn, FNAN, Max_GridN_Atom, SpinP_switch, List_YOUSO, Cnt_switch, Gxyz, atv, MGridListAtom, CellListAtom, GridListAtom, GridN_Atom, Spe_Total_CNO, WhatSpecies, M2G, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, Gc_AN, Cwan, NO0, Nc, GNc, GRc, MNc, Cxyz, x, y, z, dx, dy, dz, dorbs0, orbs1, dDen_Grid, dChi0, i, k, h_AN, Gh_AN, Mh_AN, Rnh, Hwan, NO1, spin, Nog, Nh, j, sum, tmp0, sumx, sumy, sumz, Vpt, ReVpt11, ReVpt22, ReVpt21, ImVpt21)
    {

        /* allocation of arrays */

        dorbs0 = (double**)malloc(sizeof(double*) * 4);
        for (i = 0; i < 4; i++) {
            dorbs0[i] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
        }

        orbs1 = (double*)malloc(sizeof(double) * List_YOUSO[7]);

        dDen_Grid = (double***)malloc(sizeof(double**) * (SpinP_switch + 1));
        for (i = 0; i < (SpinP_switch + 1); i++) {
            dDen_Grid[i] = (double**)malloc(sizeof(double*) * 3);
            for (k = 0; k < 3; k++) {
                dDen_Grid[i][k] = (double*)malloc(sizeof(double) * Max_GridN_Atom);
            }
        }

        dChi0 = (double***)malloc(sizeof(double**) * 3);
        for (k = 0; k < 3; k++) {
            dChi0[k] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
            for (i = 0; i < List_YOUSO[7]; i++) {
                dChi0[k][i] = (double*)malloc(sizeof(double) * Max_GridN_Atom);
            }
        }

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (Mc_AN = (OMPID * Matomnum / Nthrds + 1); Mc_AN < ((OMPID + 1) * Matomnum / Nthrds + 1); Mc_AN++) {

            Gc_AN = M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            NO0 = Spe_Total_CNO[Cwan];

            /***********************************
                     calc dOrb0
            ***********************************/

            for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {

                GNc = GridListAtom[Mc_AN][Nc];
                GRc = CellListAtom[Mc_AN][Nc];
                MNc = MGridListAtom[Mc_AN][Nc];

                Get_Grid_XYZ(GNc, Cxyz);
                x = Cxyz[1] + atv[GRc][1];
                y = Cxyz[2] + atv[GRc][2];
                z = Cxyz[3] + atv[GRc][3];
                dx = x - Gxyz[Gc_AN][1];
                dy = y - Gxyz[Gc_AN][2];
                dz = z - Gxyz[Gc_AN][3];

                if (Cnt_switch == 0)
                    Get_dOrbitals(Cwan, dx, dy, dz, dorbs0);
                else
                    Get_Cnt_dOrbitals(Mc_AN, dx, dy, dz, dorbs0);

                for (k = 0; k < 3; k++) {
                    for (i = 0; i < NO0; i++) {
                        dChi0[k][i][Nc] = dorbs0[k + 1][i];
                    }
                }
            }

            /***********************************
                    calc dDen_Grid
            ***********************************/

            /* initialize */
            for (i = 0; i <= SpinP_switch; i++) {
                for (k = 0; k < 3; k++) {
                    for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {
                        dDen_Grid[i][k][Nc] = 0.0;
                    }
                }
            }

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                Gh_AN = natn[Gc_AN][h_AN];
                Mh_AN = F_G2M[Gh_AN];
                Rnh = ncn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                NO1 = Spe_Total_CNO[Hwan];

                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (Nog = 0; Nog < NumOLG[Mc_AN][h_AN]; Nog++) {

                        Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
                        Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

                        for (k = 0; k < 3; k++) {
                            for (i = 0; i < NO0; i++) {
                                dorbs0[k][i] = dChi0[k][i][Nc];
                            }
                        }

                        /* set orbs1 */

                        if (G2ID[Gh_AN] == myid) {
                            for (j = 0; j < NO1; j++)
                                orbs1[j] = Orbs_Grid[Mh_AN][Nh][j]; /* AITUNE */
                        } else {
                            for (j = 0; j < NO1; j++)
                                orbs1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j]; /* AITUNE */
                        }

                        for (k = 0; k < 3; k++) {
                            sum = 0.0;
                            for (i = 0; i < NO0; i++) {
                                tmp0 = 0.0;
                                for (j = 0; j < NO1; j++) {
                                    tmp0 += orbs1[j] * DM[0][spin][Mc_AN][h_AN][i][j];
                                }
                                sum += dorbs0[k][i] * tmp0;
                            }

                            /* due to difference in the definition between density matrix and density */
                            if (spin == 3)
                                dDen_Grid[spin][k][Nc] -= sum;
                            else
                                dDen_Grid[spin][k][Nc] += sum;
                        }
                    }
                }
            }

            /***********************************
                     calc force #3
            ***********************************/

            /* spin collinear */

            if (SpinP_switch == 0 || SpinP_switch == 1) {

                sumx = 0.0;
                sumy = 0.0;
                sumz = 0.0;

                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {

                        MNc = MGridListAtom[Mc_AN][Nc];

                        if (0 <= MNc) {
                            if (E_Field_switch == 1) {

                                if (ProExpn_VNA == 0) {

                                    Vpt = F_dVHart_flag * dVHart_Grid[MNc]
                                        + F_Vxc_flag * Vxc_Grid[spin][MNc]
                                        + F_VNA_flag * VNA_Grid[MNc]
                                        + F_VEF_flag * VEF_Grid[MNc];

                                } else {

                                    Vpt = F_dVHart_flag * dVHart_Grid[MNc]
                                        + F_Vxc_flag * Vxc_Grid[spin][MNc]
                                        + F_VEF_flag * VEF_Grid[MNc];
                                }

                            } else {
                                if (ProExpn_VNA == 0) {

                                    Vpt = F_dVHart_flag * dVHart_Grid[MNc]
                                        + F_Vxc_flag * Vxc_Grid[spin][MNc]
                                        + F_VNA_flag * VNA_Grid[MNc];

                                } else {

                                    Vpt = F_dVHart_flag * dVHart_Grid[MNc]
                                        + F_Vxc_flag * Vxc_Grid[spin][MNc];
                                }
                            }
                        } else
                            Vpt = 0.0;

                        sumx += dDen_Grid[spin][0][Nc] * Vpt;
                        sumy += dDen_Grid[spin][1][Nc] * Vpt;
                        sumz += dDen_Grid[spin][2][Nc] * Vpt;
                    }
                }

                if (SpinP_switch == 0) {
                    sumx = 4.0 * sumx;
                    sumy = 4.0 * sumy;
                    sumz = 4.0 * sumz;
                } else if (SpinP_switch == 1) {
                    sumx = 2.0 * sumx;
                    sumy = 2.0 * sumy;
                    sumz = 2.0 * sumz;
                }
            }

            /* spin non-collinear */

            else if (SpinP_switch == 3) {

                sumx = 0.0;
                sumy = 0.0;
                sumz = 0.0;

                for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {

                    MNc = MGridListAtom[Mc_AN][Nc];

                    if (0 <= MNc) {
                        if (E_Field_switch == 1) {

                            if (ProExpn_VNA == 0) {

                                ReVpt11 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[0][MNc]
                                    + F_VNA_flag * VNA_Grid[MNc]
                                    + F_VEF_flag * VEF_Grid[MNc];

                                ReVpt22 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[1][MNc]
                                    + F_VNA_flag * VNA_Grid[MNc]
                                    + F_VEF_flag * VEF_Grid[MNc];

                                ReVpt21 = F_Vxc_flag * Vxc_Grid[2][MNc];
                                ImVpt21 = -F_Vxc_flag * Vxc_Grid[3][MNc];
                            } else {

                                ReVpt11 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[0][MNc]
                                    + F_VEF_flag * VEF_Grid[MNc];

                                ReVpt22 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[1][MNc]
                                    + F_VEF_flag * VEF_Grid[MNc];

                                ReVpt21 = F_Vxc_flag * Vxc_Grid[2][MNc];
                                ImVpt21 = -F_Vxc_flag * Vxc_Grid[3][MNc];
                            }

                        } else {

                            if (ProExpn_VNA == 0) {

                                ReVpt11 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[0][MNc]
                                    + F_VNA_flag * VNA_Grid[MNc];

                                ReVpt22 = F_dVHart_flag * dVHart_Grid[MNc]
                                    + F_Vxc_flag * Vxc_Grid[1][MNc]
                                    + F_VNA_flag * VNA_Grid[MNc];

                                ReVpt21 = F_Vxc_flag * Vxc_Grid[2][MNc];
                                ImVpt21 = -F_Vxc_flag * Vxc_Grid[3][MNc];
                            } else {

                                ReVpt11 = F_dVHart_flag * dVHart_Grid[MNc] + F_Vxc_flag * Vxc_Grid[0][MNc];
                                ReVpt22 = F_dVHart_flag * dVHart_Grid[MNc] + F_Vxc_flag * Vxc_Grid[1][MNc];

                                ReVpt21 = F_Vxc_flag * Vxc_Grid[2][MNc];
                                ImVpt21 = -F_Vxc_flag * Vxc_Grid[3][MNc];
                            }
                        }
                    } else {
                        ReVpt11 = 0.0;
                        ReVpt22 = 0.0;
                        ReVpt21 = 0.0;
                        ImVpt21 = 0.0;
                    }

                    sumx += dDen_Grid[0][0][Nc] * ReVpt11;
                    sumx += dDen_Grid[1][0][Nc] * ReVpt22;
                    sumx += 2.0 * dDen_Grid[2][0][Nc] * ReVpt21;
                    sumx += -2.0 * dDen_Grid[3][0][Nc] * ImVpt21;

                    sumy += dDen_Grid[0][1][Nc] * ReVpt11;
                    sumy += dDen_Grid[1][1][Nc] * ReVpt22;
                    sumy += 2.0 * dDen_Grid[2][1][Nc] * ReVpt21;
                    sumy += -2.0 * dDen_Grid[3][1][Nc] * ImVpt21;

                    sumz += dDen_Grid[0][2][Nc] * ReVpt11;
                    sumz += dDen_Grid[1][2][Nc] * ReVpt22;
                    sumz += 2.0 * dDen_Grid[2][2][Nc] * ReVpt21;
                    sumz += -2.0 * dDen_Grid[3][2][Nc] * ImVpt21;
                }

                sumx = 2.0 * sumx;
                sumy = 2.0 * sumy;
                sumz = 2.0 * sumz;
            }

            Gxyz[Gc_AN][17] += sumx * GridVol;
            Gxyz[Gc_AN][18] += sumy * GridVol;
            Gxyz[Gc_AN][19] += sumz * GridVol;

            if (2 <= level_stdout) {
                printf("<Force>  force(3) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                    myid, Mc_AN, Gc_AN, sumx * GridVol, sumy * GridVol, sumz * GridVol);
                fflush(stdout);
            }

        } /* Mc_AN */

        /* freeing of arrays */

        for (k = 0; k < 3; k++) {
            for (i = 0; i < List_YOUSO[7]; i++) {
                free(dChi0[k][i]);
            }
            free(dChi0[k]);
        }
        free(dChi0);

        for (i = 0; i < (SpinP_switch + 1); i++) {
            for (k = 0; k < 3; k++) {
                free(dDen_Grid[i][k]);
            }
            free(dDen_Grid[i]);
        }
        free(dDen_Grid);

        free(orbs1);

        for (i = 0; i < 4; i++) {
            free(dorbs0[i]);
        }
        free(dorbs0);

    } /* #pragma omp parallel */
}
#endif

void Force4()
{
    /****************************************************
                        #4 of Force

                        n * dVNA/dx
    ****************************************************/

    int Mc_AN, Gc_AN, Cwan, Hwan, NO0, NO1;
    int i, j, k, Nc, Nh, GNc, GRc, MNc;
    int h_AN, Gh_AN, Mh_AN, Rnh, spin, Nog;
    double sum, tmp0, r, dx, dy, dz;
    double dvx, dvy, dvz;
    double sumx, sumy, sumz;
    double x, y, z, den;
    double Cxyz[4];

    /**********************************************************
                main loop for calculation of force #4
    **********************************************************/

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        NO0 = Spe_Total_CNO[Cwan];

        /***********************************
                     summation
        ***********************************/

        sumx = 0.0;
        sumy = 0.0;
        sumz = 0.0;

        for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {

            GNc = GridListAtom[Mc_AN][Nc];
            GRc = CellListAtom[Mc_AN][Nc];
            MNc = MGridListAtom[Mc_AN][Nc];

            Get_Grid_XYZ(GNc, Cxyz);
            x = Cxyz[1] + atv[GRc][1];
            y = Cxyz[2] + atv[GRc][2];
            z = Cxyz[3] + atv[GRc][3];
            dx = Gxyz[Gc_AN][1] - x;
            dy = Gxyz[Gc_AN][2] - y;
            dz = Gxyz[Gc_AN][3] - z;
            r = sqrt(dx * dx + dy * dy + dz * dz);

            /* for empty atoms or finite elemens basis */
            if (r < 1.0e-10)
                r = 1.0e-10;

            if (1.0e-14 < r) {
                tmp0 = Dr_VNAF(Cwan, r);
                dvx = tmp0 * dx / r;
                dvy = tmp0 * dy / r;
                dvz = tmp0 * dz / r;
            } else {
                dvx = 0.0;
                dvy = 0.0;
                dvz = 0.0;
            }

            den = Density_Grid[0][MNc] + Density_Grid[1][MNc];
            sumx += den * dvx;
            sumy += den * dvy;
            sumz += den * dvz;
        }

        Gxyz[Gc_AN][17] += sumx * GridVol;
        Gxyz[Gc_AN][18] += sumy * GridVol;
        Gxyz[Gc_AN][19] += sumz * GridVol;

        /*
        if (2<=level_stdout){
          printf("<Force>  force(4) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                  myid,Mc_AN,Gc_AN,sumx*GridVol,sumy*GridVol,sumz*GridVol);fflush(stdout);
        }
        */
    }
}

void Force_HNL(double***** CDM0, double***** iDM0)
{
    /****************************************************
                    Force arising from HNL
    ****************************************************/

    int Mc_AN, Gc_AN, Cwan, i, j, h_AN, q_AN, Mq_AN, start_q_AN;
    int jan, kl, km, kl1, Qwan, Gq_AN, Gh_AN, Mh_AN, Hwan, ian;
    int l1, l2, l3, l, LL, Mul1, tno0, ncp, so;
    int tno1, tno2, size1, size2, n, kk, num, po, po1, po2;
    int numprocs, myid, tag = 999, ID, IDS, IDR;
    int **S_array, **R_array;
    int S_comm_flag, R_comm_flag;
    int SA_num, q, Sc_AN, GSc_AN, smul;
    int Sc_wan, Sh_AN, GSh_AN, Sh_wan;
    int Sh_AN2, fan, jg, j0, jg0, Mj_AN0;
    int Original_Mc_AN, max_ODNloop;

    double rcutA, rcutB, rcut;
    double dEx, dEy, dEz, ene, pref;
    double Stime_atom, Etime_atom;
    dcomplex ***Hx, ***Hy, ***Hz;
    dcomplex ***Hx0, ***Hy0, ***Hz0;
    dcomplex ***Hx1, ***Hy1, ***Hz1;
    int *Snd_DS_NL_Size, *Rcv_DS_NL_Size;
    int* Indicator;
    int fhnl_gpu = 0;
    double* tmp_array;
    double* tmp_array2;

    /* for OpenMP */
    int OMPID, Nthrds, Nthrds0, Nprocs, Nloop, ODNloop;
    int *OneD2h_AN, *OneD2q_AN;
    double* dEx_threads;
    double* dEy_threads;
    double* dEz_threads;
    double stime, etime;
    double stime1, etime1;

    MPI_Status stat;
    MPI_Request request;

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    dtime(&stime);

    /****************************
         allocation of arrays
    *****************************/

    Indicator = (int*)malloc(sizeof(int) * numprocs);

    S_array = (int**)malloc(sizeof(int*) * numprocs);
    for (ID = 0; ID < numprocs; ID++) {
        S_array[ID] = (int*)malloc(sizeof(int) * 3);
    }

    R_array = (int**)malloc(sizeof(int*) * numprocs);
    for (ID = 0; ID < numprocs; ID++) {
        R_array[ID] = (int*)malloc(sizeof(int) * 3);
    }

    Snd_DS_NL_Size = (int*)malloc(sizeof(int) * numprocs);
    Rcv_DS_NL_Size = (int*)malloc(sizeof(int) * numprocs);

    /* initialize the temporal array storing the force contribution */

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = F_M2G[Mc_AN];
        Gxyz[Gc_AN][41] = 0.0;
        Gxyz[Gc_AN][42] = 0.0;
        Gxyz[Gc_AN][43] = 0.0;
    }

    /*************************************************************
                      contraction of DS_NL
       Note: DS_NL is overwritten by CntDS_NL in Cont_Matrix1().
    *************************************************************/

    if (Cnt_switch == 1) {
        for (so = 0; so < (SO_switch + 1); so++) {
            Cont_Matrix1(DS_NL[so][0], CntDS_NL[so][0]);
            Cont_Matrix1(DS_NL[so][1], CntDS_NL[so][1]);
            Cont_Matrix1(DS_NL[so][2], CntDS_NL[so][2]);
            Cont_Matrix1(DS_NL[so][3], CntDS_NL[so][3]);
        }
    }

    /* batched device path for the scalar-relativistic traces */

    fhnl_gpu = Force_HNL_GpuBegin(DS_NL);

    /*****************************************}**********************
        THE FIRST CASE:
        In case of I=i or I=j
        for d [ \sum_k <i|k>ek<k|j> ]/dRI
    ****************************************************************/

    /*******************************************************
     *******************************************************
         multiplying overlap integrals WITH COMMUNICATION

         In case of I=i or I=j
         for d [ \sum_k <i|k>ek<k|j> ]/dRI
     *******************************************************
     *******************************************************/

    MPI_Barrier(mpi_comm_level1);
    dtime(&stime);

    Hx0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hx0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hx0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    Hy0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hy0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hy0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    Hz0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hz0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hz0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    Hx1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hx1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hx1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    Hy1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hy1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hy1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    Hz1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hz1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hz1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    for (ID = 0; ID < numprocs; ID++) {
        F_Snd_Num_WK[ID] = 0;
        F_Rcv_Num_WK[ID] = 0;
    }

    do {

        /***********************************
                set the size of data
        ************************************/

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            /* find the data size to send the block data */

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS])) {

                size1 = 0;
                n = F_Snd_Num_WK[IDS];

                Mc_AN = Snd_MAN[IDS][n];
                Gc_AN = Snd_GAN[IDS][n];
                Cwan = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_NO[Cwan];

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    tno2 = Spe_Total_VPS_Pro[Hwan];
                    size1 += (VPS_j_dependency[Hwan] + 1) * tno1 * tno2;
                }

                Snd_DS_NL_Size[IDS] = size1;
                MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
            } else {
                Snd_DS_NL_Size[IDS] = 0;
            }

            /* receiving of the size of the data */

            if (0 < (F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR])) {
                MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                Rcv_DS_NL_Size[IDR] = size2;
            } else {
                Rcv_DS_NL_Size[IDR] = 0;
            }

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS]))
                MPI_Wait(&request, &stat);

        } /* ID */

        /***********************************
                   data transfer
        ************************************/

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            /******************************
                  sending of the data
            ******************************/

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS])) {

                size1 = Snd_DS_NL_Size[IDS];

                /* allocation of the array */

                tmp_array = (double*)malloc(sizeof(double) * size1);

                /* multidimentional array to the vector array */

                num = 0;
                n = F_Snd_Num_WK[IDS];

                Mc_AN = Snd_MAN[IDS][n];
                Gc_AN = Snd_GAN[IDS][n];
                Cwan = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_NO[Cwan];

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    tno2 = Spe_Total_VPS_Pro[Hwan];

                    for (so = 0; so <= VPS_j_dependency[Hwan]; so++) {
                        for (i = 0; i < tno1; i++) {
                            for (j = 0; j < tno2; j++) {
                                tmp_array[num] = DS_NL[so][0][Mc_AN][h_AN][i][j];
                                num++;
                            }
                        }
                    }
                }

                MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
            }

            /******************************
              receiving of the block data
            ******************************/

            if (0 < (F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR])) {

                size2 = Rcv_DS_NL_Size[IDR];
                tmp_array2 = (double*)malloc(sizeof(double) * size2);
                MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                /* store */

                num = 0;
                n = F_Rcv_Num_WK[IDR];
                Original_Mc_AN = F_TopMAN[IDR] + n;

                Gc_AN = Rcv_GAN[IDR][n];
                Cwan = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_NO[Cwan];
                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    tno2 = Spe_Total_VPS_Pro[Hwan];

                    for (so = 0; so <= VPS_j_dependency[Hwan]; so++) {
                        for (i = 0; i < tno1; i++) {
                            for (j = 0; j < tno2; j++) {
                                DS_NL[so][0][Matomnum + 1][h_AN][i][j] = tmp_array2[num];
                                num++;
                            }
                        }
                    }
                }

                /* free tmp_array2 */
                free(tmp_array2);

                if (fhnl_gpu) {
                    Force_HNL_GpuArchiveHalo(DS_NL, Original_Mc_AN, Gc_AN);
                }

                /*****************************************************************
                                   multiplying overlap integrals
                *****************************************************************/

                if (!fhnl_gpu) {

                for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                    dtime(&Stime_atom);

                    dEx = 0.0;
                    dEy = 0.0;
                    dEz = 0.0;

                    Gc_AN = M2G[Mc_AN];
                    Cwan = WhatSpecies[Gc_AN];
                    fan = FNAN[Gc_AN];

                    h_AN = 0;
                    Gh_AN = natn[Gc_AN][h_AN];
                    Mh_AN = F_G2M[Gh_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    ian = Spe_Total_CNO[Hwan];

                    n = F_Rcv_Num_WK[IDR];
                    jg = Rcv_GAN[IDR][n];

                    for (j0 = 0; j0 <= fan; j0++) {

                        jg0 = natn[Gc_AN][j0];
                        Mj_AN0 = F_G2M[jg0];

                        po2 = 0;
                        if (Original_Mc_AN == Mj_AN0) {
                            po2 = 1;
                            q_AN = j0;
                        }

                        if (po2 == 1) {

                            Gq_AN = natn[Gc_AN][q_AN];
                            Mq_AN = F_G2M[Gq_AN];
                            Qwan = WhatSpecies[Gq_AN];
                            jan = Spe_Total_CNO[Qwan];
                            kl = RMI1[Mc_AN][h_AN][q_AN];

                            dHNL(0, Mc_AN, h_AN, q_AN, DS_NL, Hx0, Hy0, Hz0);

                            /* contribution of force = Trace(CDM0*dH) */
                            /* spin non-polarization */

                            if (SpinP_switch == 0) {

                                if (q_AN == h_AN)
                                    pref = 2.0;
                                else
                                    pref = 4.0;

                                for (i = 0; i < ian; i++) {
                                    for (j = 0; j < jan; j++) {

                                        dEx += pref * CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r;
                                        dEy += pref * CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r;
                                        dEz += pref * CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r;
                                    }
                                }
                            }

                            /* collinear spin polarized or non-colliear without SO and LDA+U */

                            else if (SpinP_switch == 1 || (SpinP_switch == 3 && SO_switch == 0 && Hub_U_switch == 0 && Constraint_NCS_switch == 0 && Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0)) {

                                if (q_AN == h_AN)
                                    pref = 1.0;
                                else
                                    pref = 2.0;

                                for (i = 0; i < Spe_Total_CNO[Hwan]; i++) {
                                    for (j = 0; j < Spe_Total_CNO[Qwan]; j++) {

                                        dEx += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r);
                                        dEy += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r);
                                        dEz += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r);
                                    }
                                }
                            }

                            /* spin non-collinear with spin-orbit coupling or with LDA+U */

                            else if (SpinP_switch == 3 && (SO_switch == 1 || (Hub_U_switch == 1 && F_U_flag == 1) || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)) {

                                if (q_AN == h_AN) {

                                    for (i = 0; i < Spe_Total_CNO[Hwan]; i++) {
                                        for (j = 0; j < Spe_Total_CNO[Qwan]; j++) {

                                            dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

                                            dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

                                            dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;
                                        }
                                    }
                                }

                                else {

                                    for (i = 0; i < Spe_Total_CNO[Hwan]; i++) { /* Hwan */
                                        for (j = 0; j < Spe_Total_CNO[Qwan]; j++) { /* Qwan  */

                                            dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

                                            dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

                                            dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;

                                        } /* j */
                                    } /* i */

                                    dHNL(0, Mc_AN, q_AN, h_AN, DS_NL, Hx1, Hy1, Hz1);
                                    kl1 = RMI1[Mc_AN][q_AN][h_AN];

                                    for (i = 0; i < Spe_Total_CNO[Qwan]; i++) { /* Qwan */
                                        for (j = 0; j < Spe_Total_CNO[Hwan]; j++) { /* Hwan */

                                            dEx += CDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].r
                                                - iDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].i
                                                + CDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].r
                                                - iDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].i
                                                + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hx1[2][i][j].r
                                                - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hx1[2][i][j].i;

                                            dEy += CDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].r
                                                - iDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].i
                                                + CDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].r
                                                - iDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].i
                                                + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hy1[2][i][j].r
                                                - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hy1[2][i][j].i;

                                            dEz += CDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].r
                                                - iDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].i
                                                + CDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].r
                                                - iDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].i
                                                + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hz1[2][i][j].r
                                                - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hz1[2][i][j].i;

                                        } /* j */
                                    } /* i */
                                }
                            }

                        } /* if (po2==1) */
                    } /* j0 */

                    /* force from #4B */

                    Gxyz[Gc_AN][41] += dEx;
                    Gxyz[Gc_AN][42] += dEy;
                    Gxyz[Gc_AN][43] += dEz;

                    /* timing */
                    dtime(&Etime_atom);
                    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

                } /* Mc_AN */

                } /* if (!fhnl_gpu) */

                /********************************************
                  increment of F_Rcv_Num_WK[IDR]
                ********************************************/

                F_Rcv_Num_WK[IDR]++;

            } /* if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) */

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS])) {

                MPI_Wait(&request, &stat);
                free(tmp_array); /* freeing of array */

                /********************************************
                     increment of F_Snd_Num_WK[IDS]
                ********************************************/

                F_Snd_Num_WK[IDS]++;
            }

        } /* ID */

        /*****************************************************
          check whether all the communications have finished
        *****************************************************/

        po = 0;
        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS]))
                po += F_Snd_Num[IDS] - F_Snd_Num_WK[IDS];
            if (0 < (F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR]))
                po += F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR];
        }

    } while (po != 0);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hx0[i][j]);
        }
        free(Hx0[i]);
    }
    free(Hx0);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hy0[i][j]);
        }
        free(Hy0[i]);
    }
    free(Hy0);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hz0[i][j]);
        }
        free(Hz0[i]);
    }
    free(Hz0);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hx1[i][j]);
        }
        free(Hx1[i]);
    }
    free(Hx1);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hy1[i][j]);
        }
        free(Hy1[i]);
    }
    free(Hy1);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hz1[i][j]);
        }
        free(Hz1[i]);
    }
    free(Hz1);

    dtime(&etime);
    if (myid == 0 && measure_time) {
        printf("Time for part1 of force_NL=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];

        if (2 <= level_stdout) {
            printf("<Force>  force(HNL1) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                myid, Mc_AN, Gc_AN, Gxyz[Gc_AN][41], Gxyz[Gc_AN][42], Gxyz[Gc_AN][43]);
            fflush(stdout);
        }
    }

    /*******************************************************
     *******************************************************
       THE FIRST CASE:
       multiplying overlap integrals WITHOUT COMMUNICATION

       In case of I=i or I=j
       for d [ \sum_k <i|k>ek<k|j> ]/dRI
     *******************************************************
     *******************************************************/

    dtime(&stime);

#pragma omp parallel shared(time_per_atom, Gxyz, CDM0, SpinP_switch, SO_switch, Hub_U_switch, F_U_flag, Constraint_NCS_switch, Zeeman_NCS_switch, Zeeman_NCO_switch, DS_NL, RMI1, FNAN, Spe_Total_CNO, WhatSpecies, F_G2M, natn, M2G, Matomnum, List_YOUSO, F_NL_flag) private(Hx0, Hy0, Hz0, Hx1, Hy1, Hz1, OMPID, Nthrds, Nprocs, Mc_AN, Stime_atom, Etime_atom, dEx, dEy, dEz, Gc_AN, h_AN, Gh_AN, Mh_AN, Hwan, ian, q_AN, Gq_AN, Mq_AN, Qwan, jan, kl, kl1, i, j, kk, pref)
    {

        /* allocation of array */

        Hx0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hx0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hx0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hy0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hy0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hy0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hz0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hz0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hz0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hx1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hx1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hx1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hy1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hy1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hy1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hz1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hz1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hz1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (Mc_AN = (OMPID * Matomnum / Nthrds + 1); Mc_AN < ((OMPID + 1) * Matomnum / Nthrds + 1); Mc_AN++) {

            dtime(&Stime_atom);

            dEx = 0.0;
            dEy = 0.0;
            dEz = 0.0;

            Gc_AN = M2G[Mc_AN];
            h_AN = 0;
            Gh_AN = natn[Gc_AN][h_AN];
            Mh_AN = F_G2M[Gh_AN];
            Hwan = WhatSpecies[Gh_AN];
            ian = Spe_Total_CNO[Hwan];

            for (q_AN = 0; q_AN <= FNAN[Gc_AN]; q_AN++) {

                Gq_AN = natn[Gc_AN][q_AN];
                Mq_AN = F_G2M[Gq_AN];

                if (fhnl_gpu && q_AN != 0)
                    continue;

                if (Mq_AN <= Matomnum) {

                    Qwan = WhatSpecies[Gq_AN];
                    jan = Spe_Total_CNO[Qwan];
                    kl = RMI1[Mc_AN][h_AN][q_AN];

                    dHNL(0, Mc_AN, h_AN, q_AN, DS_NL, Hx0, Hy0, Hz0);

                    if (SpinP_switch == 0) {

                        if (q_AN == h_AN)
                            pref = 2.0;
                        else
                            pref = 4.0;

                        for (i = 0; i < ian; i++) {
                            for (j = 0; j < jan; j++) {

                                dEx += pref * CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r;
                                dEy += pref * CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r;
                                dEz += pref * CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r;
                            }
                        }
                    }

                    /* collinear spin polarized or non-colliear without SO and LDA+U */

                    else if (SpinP_switch == 1 || (SpinP_switch == 3 && SO_switch == 0 && Hub_U_switch == 0 && Constraint_NCS_switch == 0 && Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0)) {

                        if (q_AN == h_AN)
                            pref = 1.0;
                        else
                            pref = 2.0;

                        for (i = 0; i < ian; i++) {
                            for (j = 0; j < jan; j++) {

                                dEx += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r);
                                dEy += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r);
                                dEz += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r);
                            }
                        }
                    }

                    /* spin non-collinear with spin-orbit coupling or with LDA+U */

                    else if (SpinP_switch == 3 && (SO_switch == 1 || (Hub_U_switch == 1 && F_U_flag == 1) || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)) {

                        if (q_AN == h_AN) {

                            for (i = 0; i < Spe_Total_CNO[Hwan]; i++) {
                                for (j = 0; j < Spe_Total_CNO[Qwan]; j++) {

                                    dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

                                    dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

                                    dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;
                                }
                            }
                        }

                        else {

                            for (i = 0; i < Spe_Total_CNO[Hwan]; i++) { /* Hwan */
                                for (j = 0; j < Spe_Total_CNO[Qwan]; j++) { /* Qwan  */

                                    dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

                                    dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

                                    dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;

                                } /* j */
                            } /* i */

                            dHNL(0, Mc_AN, q_AN, h_AN, DS_NL, Hx1, Hy1, Hz1);

                            kl1 = RMI1[Mc_AN][q_AN][h_AN];
                            for (i = 0; i < Spe_Total_CNO[Qwan]; i++) { /* Qwan */
                                for (j = 0; j < Spe_Total_CNO[Hwan]; j++) { /* Hwan */

                                    dEx += CDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].r
                                        - iDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].i
                                        + CDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].r
                                        - iDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].i
                                        + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hx1[2][i][j].r
                                        - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hx1[2][i][j].i;

                                    dEy += CDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].r
                                        - iDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].i
                                        + CDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].r
                                        - iDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].i
                                        + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hy1[2][i][j].r
                                        - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hy1[2][i][j].i;

                                    dEz += CDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].r
                                        - iDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].i
                                        + CDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].r
                                        - iDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].i
                                        + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hz1[2][i][j].r
                                        - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hz1[2][i][j].i;

                                } /* j */
                            } /* i */
                        }
                    }
                }
            }

            /* force from #4B */

            if (F_NL_flag == 1) {
                Gxyz[Gc_AN][41] += dEx;
                Gxyz[Gc_AN][42] += dEy;
                Gxyz[Gc_AN][43] += dEz;
            }

            /* timing */
            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

        } /* Mc_AN */

        /* freeing of array */

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hx0[i][j]);
            }
            free(Hx0[i]);
        }
        free(Hx0);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hy0[i][j]);
            }
            free(Hy0[i]);
        }
        free(Hy0);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hz0[i][j]);
            }
            free(Hz0[i]);
        }
        free(Hz0);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hx1[i][j]);
            }
            free(Hx1[i]);
        }
        free(Hx1);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hy1[i][j]);
            }
            free(Hy1[i]);
        }
        free(Hy1);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hz1[i][j]);
            }
            free(Hz1[i]);
        }
        free(Hz1);

    } /* #pragma omp parallel */

    if (fhnl_gpu) {
        Force_HNL_GpuCase1Run(CDM0);
    }

    dtime(&etime);
    if (myid == 0 && measure_time) {
        printf("Time for part2 of force_NL=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];

        if (2 <= level_stdout) {
            printf("<Force>  force(HNL2) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                myid, Mc_AN, Gc_AN, Gxyz[Gc_AN][41], Gxyz[Gc_AN][42], Gxyz[Gc_AN][43]);
            fflush(stdout);
        }
    }

    /*************************************************************
       THE SECOND CASE:
       In case of I=k with I!=i and I!=j
       d [ \sum_k <i|k>ek<k|j> ]/dRI
    *************************************************************/

    /************************************************************
       MPI communication of DS_NL whose basis part is not located
       on own site but projector part is located on own site.
    ************************************************************/

    MPI_Barrier(mpi_comm_level1);
    dtime(&stime);

    for (ID = 0; ID < numprocs; ID++)
        Indicator[ID] = 0;

    max_ODNloop = 1;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        i = (FNAN[Gc_AN] + 1) * (FNAN[Gc_AN] + 2);
        if (max_ODNloop < i) {
            max_ODNloop = i;
        }
    }

    OneD2h_AN = (int*)malloc(sizeof(int) * max_ODNloop);
    OneD2q_AN = (int*)malloc(sizeof(int) * max_ODNloop);

    for (Mc_AN = 1; Mc_AN <= Max_Matomnum; Mc_AN++) {

        if (Mc_AN <= Matomnum)
            Gc_AN = M2G[Mc_AN];
        else
            Gc_AN = 0;

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            i = Indicator[IDS];
            po = 0;

            Gh_AN = Pro_Snd_GAtom[IDS][i];

            if (Gh_AN != 0) {

                /* find the range with the same global atomic number */

                do {

                    i++;
                    if (Gh_AN != Pro_Snd_GAtom[IDS][i])
                        po = 1;
                } while (po == 0);

                i--;
                SA_num = i - Indicator[IDS] + 1;

                /* find the data size to send the block data */

                size1 = 0;
                for (q = Indicator[IDS]; q <= (Indicator[IDS] + SA_num - 1); q++) {

                    Sc_AN = Pro_Snd_MAtom[IDS][q];
                    GSc_AN = F_M2G[Sc_AN];
                    Sc_wan = WhatSpecies[GSc_AN];
                    tno1 = Spe_Total_CNO[Sc_wan];

                    Sh_AN = Pro_Snd_LAtom[IDS][q];
                    GSh_AN = natn[GSc_AN][Sh_AN];
                    Sh_wan = WhatSpecies[GSh_AN];
                    tno2 = Spe_Total_VPS_Pro[Sh_wan];
                    smul = (VPS_j_dependency[Sh_wan] + 1);

                    size1 += smul * 4 * tno1 * tno2;
                    size1 += 3;
                }

            } /* if (Gh_AN!=0) */

            else {
                SA_num = 0;
                size1 = 0;
            }

            S_array[IDS][0] = Gh_AN;
            S_array[IDS][1] = SA_num;
            S_array[IDS][2] = size1;

            if (ID != 0) {
                MPI_Isend(&S_array[IDS][0], 3, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                MPI_Recv(&R_array[IDR][0], 3, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                MPI_Wait(&request, &stat);
            } else {
                R_array[myid][0] = S_array[myid][0];
                R_array[myid][1] = S_array[myid][1];
                R_array[myid][2] = S_array[myid][2];
            }

            if (R_array[IDR][0] == Gc_AN)
                R_comm_flag = 1;
            else
                R_comm_flag = 0;

            if (ID != 0) {
                MPI_Isend(&R_comm_flag, 1, MPI_INT, IDR, tag, mpi_comm_level1, &request);
                MPI_Recv(&S_comm_flag, 1, MPI_INT, IDS, tag, mpi_comm_level1, &stat);
                MPI_Wait(&request, &stat);
            } else {
                S_comm_flag = R_comm_flag;
            }

            /*****************************************
                          send the data
            *****************************************/

            /* if (S_comm_flag==1) then, send data to IDS */

            if (S_comm_flag == 1) {

                /* allocate tmp_array */

                tmp_array = (double*)malloc(sizeof(double) * size1);

                /* multidimentional array to vector array */

                num = 0;

                for (q = Indicator[IDS]; q <= (Indicator[IDS] + SA_num - 1); q++) {

                    Sc_AN = Pro_Snd_MAtom[IDS][q];
                    GSc_AN = F_M2G[Sc_AN];
                    Sc_wan = WhatSpecies[GSc_AN];
                    tno1 = Spe_Total_CNO[Sc_wan];

                    Sh_AN = Pro_Snd_LAtom[IDS][q];
                    GSh_AN = natn[GSc_AN][Sh_AN];
                    Sh_wan = WhatSpecies[GSh_AN];
                    tno2 = Spe_Total_VPS_Pro[Sh_wan];
                    Sh_AN2 = Pro_Snd_LAtom2[IDS][q];

                    tmp_array[num] = (double)Sc_AN;
                    num++;
                    tmp_array[num] = (double)Sh_AN;
                    num++;
                    tmp_array[num] = (double)Sh_AN2;
                    num++;

                    for (so = 0; so <= VPS_j_dependency[Sh_wan]; so++) {
                        for (kk = 0; kk <= 3; kk++) {
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    tmp_array[num] = DS_NL[so][kk][Sc_AN][Sh_AN][i][j];
                                    num++;
                                }
                            }
                        }
                    }
                }

                if (ID != 0) {
                    MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
                }

                /* update Indicator[IDS] */

                Indicator[IDS] += SA_num;

            } /* if (S_comm_flag==1) */

            /*****************************************
                         receive the data
            *****************************************/

            /* if (R_comm_flag==1) then, receive the data from IDR */

            if (R_comm_flag == 1) {

                size2 = R_array[IDR][2];
                tmp_array2 = (double*)malloc(sizeof(double) * size2);

                if (ID != 0) {
                    MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
                } else {
                    for (i = 0; i < size2; i++)
                        tmp_array2[i] = tmp_array[i];
                }

                /* store */

                num = 0;

                for (n = 0; n < R_array[IDR][1]; n++) {

                    Sc_AN = (int)tmp_array2[num];
                    num++;
                    Sh_AN = (int)tmp_array2[num];
                    num++;
                    Sh_AN2 = (int)tmp_array2[num];
                    num++;

                    GSc_AN = natn[Gc_AN][Sh_AN2];
                    Sc_wan = WhatSpecies[GSc_AN];
                    tno1 = Spe_Total_CNO[Sc_wan];

                    GSh_AN = natn[GSc_AN][Sh_AN];
                    Sh_wan = WhatSpecies[GSh_AN];
                    tno2 = Spe_Total_VPS_Pro[Sh_wan];

                    for (so = 0; so <= VPS_j_dependency[Sh_wan]; so++) {
                        for (kk = 0; kk <= 3; kk++) {
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    DS_NL[so][kk][Matomnum + 1][Sh_AN2][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }
                }

                /* free tmp_array2 */
                free(tmp_array2);

            } /* if (R_comm_flag==1) */

            if (S_comm_flag == 1) {
                if (ID != 0)
                    MPI_Wait(&request, &stat);
                free(tmp_array); /* freeing of array */
            }

        } /* ID */

        if (Mc_AN <= Matomnum) {

            if (fhnl_gpu) {
                Force_HNL_GpuArchiveCase2(DS_NL, Mc_AN, Gc_AN);
            }

            if (!fhnl_gpu) {

            /* get Nthrds0 */
            // comment out May 20th, 2023 H. Kawai
            //
#if NVHPC_VERSION >= 250000
            #pragma omp parallel shared(Nthrds0)
#endif
            {
                Nthrds0 = omp_get_num_threads();
            }

            /* allocation of arrays */
            dEx_threads = (double*)malloc(sizeof(double) * Nthrds0);
            dEy_threads = (double*)malloc(sizeof(double) * Nthrds0);
            dEz_threads = (double*)malloc(sizeof(double) * Nthrds0);

            for (Nloop = 0; Nloop < Nthrds0; Nloop++) {
                dEx_threads[Nloop] = 0.0;
                dEy_threads[Nloop] = 0.0;
                dEz_threads[Nloop] = 0.0;
            }

            /* one-dimensionalize the h_AN and q_AN loops */

            OneD2h_AN = (int*)malloc(sizeof(int) * (FNAN[Gc_AN] + 1) * (FNAN[Gc_AN] + 2));
            OneD2q_AN = (int*)malloc(sizeof(int) * (FNAN[Gc_AN] + 1) * (FNAN[Gc_AN] + 2));

            ODNloop = 0;
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (SpinP_switch == 3 && (SO_switch == 1 || (Hub_U_switch == 1 && F_U_flag == 1) || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
                    || (Solver == 5 || Solver == 8 || Solver == 11))
                    start_q_AN = 0;
                else
                    start_q_AN = h_AN;

                for (q_AN = start_q_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {

                    kl = RMI1[Mc_AN][h_AN][q_AN];

                    if (0 <= kl) {
                        OneD2h_AN[ODNloop] = h_AN;
                        OneD2q_AN[ODNloop] = q_AN;
                        ODNloop++;
                    }
                }
            }

            // comment out April 28th, 2023 H. Kawai
#if NVHPC_VERSION >= 250000
            #pragma omp parallel shared(ODNloop,OneD2h_AN,OneD2q_AN,Mc_AN,Gc_AN,dEx_threads,dEy_threads,dEz_threads,CDM0,SpinP_switch,SO_switch,Hub_U_switch,Constraint_NCS_switch,Zeeman_NCS_switch,Zeeman_NCO_switch,DS_NL,RMI1,Spe_Total_CNO,WhatSpecies,F_G2M,natn,FNAN,List_YOUSO,Solver,F_NL_flag,F_U_flag) private(OMPID,Nthrds,Nprocs,Hx,Hy,Hz,i,j,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,km,Nloop,pref)
#endif
            {

                /* allocation of arrays */

                Hx = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
                for (i = 0; i < 3; i++) {
                    Hx[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        Hx[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
                    }
                }

                Hy = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
                for (i = 0; i < 3; i++) {
                    Hy[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        Hy[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
                    }
                }

                Hz = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
                for (i = 0; i < 3; i++) {
                    Hz[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        Hz[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
                    }
                }

                /* get info. on OpenMP */

                OMPID = omp_get_thread_num();
                Nthrds = omp_get_num_threads();
                Nprocs = omp_get_num_procs();

                for (Nloop = OMPID * ODNloop / Nthrds; Nloop < (OMPID + 1) * ODNloop / Nthrds; Nloop++) {

                    /* get h_AN and q_AN */

                    h_AN = OneD2h_AN[Nloop];
                    q_AN = OneD2q_AN[Nloop];

                    /* set informations on h_AN */

                    Gh_AN = natn[Gc_AN][h_AN];
                    Mh_AN = F_G2M[Gh_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    ian = Spe_Total_CNO[Hwan];

                    /* set informations on q_AN */

                    Gq_AN = natn[Gc_AN][q_AN];
                    Mq_AN = F_G2M[Gq_AN];
                    Qwan = WhatSpecies[Gq_AN];
                    jan = Spe_Total_CNO[Qwan];
                    kl = RMI1[Mc_AN][h_AN][q_AN];
                    km = RMI1[Mc_AN][q_AN][h_AN];

                    if (0 <= kl) {

                        dHNL(1, Mc_AN, h_AN, q_AN, DS_NL, Hx, Hy, Hz);

                        /* contribution of force = Trace(CDM0*dH) */

                        /* spin non-polarization */

                        if (SpinP_switch == 0) {

                            if (Solver == 5 || Solver == 8 || Solver == 11) {
                                pref = 2.0;
                            } else {
                                if (q_AN == h_AN)
                                    pref = 2.0;
                                else
                                    pref = 4.0;
                            }

                            for (i = 0; i < ian; i++) {
                                for (j = 0; j < jan; j++) {
                                    dEx_threads[OMPID] += pref * CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r;
                                    dEy_threads[OMPID] += pref * CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r;
                                    dEz_threads[OMPID] += pref * CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r;
                                }
                            }

                        }

                        /* collinear spin polarized or non-colliear without SO and LDA+U */

                        else if (SpinP_switch == 1 || (SpinP_switch == 3 && SO_switch == 0 && Hub_U_switch == 0 && Constraint_NCS_switch == 0 && Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0)) {

                            if (Solver == 5 || Solver == 8 || Solver == 11) {
                                pref = 1.0;
                            } else {
                                if (q_AN == h_AN)
                                    pref = 1.0;
                                else
                                    pref = 2.0;
                            }

                            for (i = 0; i < ian; i++) {
                                for (j = 0; j < jan; j++) {

                                    dEx_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].r);
                                    dEy_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].r);
                                    dEz_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].r);
                                }
                            }
                        }

                        /* spin non-collinear with spin-orbit coupling or with LDA+U */

                        else if (SpinP_switch == 3 && (SO_switch == 1 || (Hub_U_switch == 1 && F_U_flag == 1) || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)) {

                            pref = 1.0;

                            for (i = 0; i < ian; i++) {
                                for (j = 0; j < jan; j++) {

                                    dEx_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r - iDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].i + CDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].r - iDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].i + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx[2][i][j].r - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx[2][i][j].i);

                                    dEy_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r - iDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].i + CDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].r - iDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].i + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy[2][i][j].r - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy[2][i][j].i);

                                    dEz_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r - iDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].i + CDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].r - iDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].i + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz[2][i][j].r - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz[2][i][j].i);
                                }
                            }
                        }

                    } /* if (0<=kl) */
                } /* Nloop */

                /* freeing of arrays */

                for (i = 0; i < 3; i++) {
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        free(Hx[i][j]);
                    }
                    free(Hx[i]);
                }
                free(Hx);

                for (i = 0; i < 3; i++) {
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        free(Hy[i][j]);
                    }
                    free(Hy[i]);
                }
                free(Hy);

                for (i = 0; i < 3; i++) {
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        free(Hz[i][j]);
                    }
                    free(Hz[i]);
                }
                free(Hz);

            } /* #pragma omp parallel */

            /* sum of dEx_threads */

            dEx = 0.0;
            dEy = 0.0;
            dEz = 0.0;

            if (F_NL_flag == 1) {
                for (Nloop = 0; Nloop < Nthrds0; Nloop++) {
                    dEx += dEx_threads[Nloop];
                    dEy += dEy_threads[Nloop];
                    dEz += dEz_threads[Nloop];
                }

                /* force from #4B */

                Gxyz[Gc_AN][41] += dEx;
                Gxyz[Gc_AN][42] += dEy;
                Gxyz[Gc_AN][43] += dEz;
            }

            if (2 <= level_stdout) {
                printf("<Force>  force(HNL3) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                    myid, Mc_AN, Gc_AN, dEx, dEy, dEz);
                fflush(stdout);
            }

            /* freeing of array */
            free(OneD2q_AN);
            free(OneD2h_AN);
            free(dEx_threads);
            free(dEy_threads);
            free(dEz_threads);

            } /* if (!fhnl_gpu) */

        } /* if (Mc_AN<=Matomnum) */

    } /* Mc_AN */

    if (fhnl_gpu) {
        Force_HNL_GpuCase2Run(CDM0);
        Force_HNL_GpuEnd();
        fhnl_gpu = 0;
    }

    dtime(&etime);
    if (myid == 0 && measure_time) {
        printf("Time for part3 of force_NL=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    /********************************************************
      adding Gxyz[Gc_AN][41,42,43] to Gxyz[Gc_AN][17,18,19]
    ********************************************************/

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];

        if (2 <= level_stdout) {
            printf("<Force>  force(HNL) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                myid, Mc_AN, Gc_AN, Gxyz[Gc_AN][41], Gxyz[Gc_AN][42], Gxyz[Gc_AN][43]);
            fflush(stdout);
        }

        Gxyz[Gc_AN][17] += Gxyz[Gc_AN][41];
        Gxyz[Gc_AN][18] += Gxyz[Gc_AN][42];
        Gxyz[Gc_AN][19] += Gxyz[Gc_AN][43];
    }

    /***********************************
              freeing of arrays
    ************************************/

    free(Indicator);

    for (ID = 0; ID < numprocs; ID++) {
        free(S_array[ID]);
    }
    free(S_array);

    for (ID = 0; ID < numprocs; ID++) {
        free(R_array[ID]);
    }
    free(R_array);

    free(Snd_DS_NL_Size);
    free(Rcv_DS_NL_Size);
}

/* ------------------------------------------------------------------ */
/* Batched device evaluation of the heavy Force4B projector traces.

   The generic (fused) case-1 and case-2 traces are dot products over
   num_projectors of precomputed DS_VNA rows contracted with CDM; only
   they run on the device.  The special pairs (q_AN==0, h_AN==q_AN, ...)
   and the damping corrections stay on the unchanged host path.

   The MPI structure is untouched: the halo blocks received in the
   case-1 communication loop and the per-atom case-2 stagings are
   archived, and one batched kernel per case runs after the loops.

   All three archives live on the DEVICE only (acc_malloc); each atom
   block is packed into a small host staging buffer and uploaded as it
   becomes available.  Host-resident copies of the archives used to add
   up to several GB per rank, which multiplied by the ranks sharing a
   node exhausted host memory on large systems.                        */
/* ------------------------------------------------------------------ */

typedef struct {
    int enabled;
    int num_proj;

    /* local DS_VNA rows, direction-major: flat[dir*stride + block]    */
    float* flat_dev;
    size_t flat_stride;
    size_t* mck_off;   /* (Mc,k) block base: mck_off[mck_base[Mc]+k]   */
    int* mck_base;     /* Mc -> first (Mc,k) slot                      */

    /* case-1 halo archive: direction 0 rows of every received atom    */
    float* halo_dev;
    size_t halo_count;
    size_t* halo_off;  /* (halo_idx, k) block base                     */
    int* halo_base;    /* halo_idx -> first slot                       */

    /* case-2 per-centre archive: 4 directions of the staged slot;
       block sizes follow the NEIGHBOUR's orbital count                 */
    float* c2_dev;
    size_t c2_stride;
    size_t* c2_off;    /* (Mc,k) block base, indexed via mck_base      */

    /* host bounce buffer holding one atom's largest one-direction block */
    float* stage;
    size_t stage_len;
} Force4BGpuContext;

static Force4BGpuContext F4B_gpu = { 0 };

static void Force4B_gpu_abort(const char* message)
{
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1, 1);
}

/* per-item metadata of the batched case-1/case-2 kernels */
typedef struct {
    size_t cdm_off;
    int k_off;
    int k_n;
    int ian;
    int jan;
} Force4BGpuItem;

static int Force4B_GpuBegin(Type_DS_VNA***** DS_VNA)
{
    Force4BGpuContext* g = &F4B_gpu;
    int Mc_AN, k, node_ranks = 1, node_rank = 0;
    size_t pos = 0, slots = 0, halo_slots = 0, c2pos = 0, maxblk = 0;
    size_t free_bytes = 0, total_bytes = 0;
    MPI_Comm node_comm = MPI_COMM_NULL;

    memset(g, 0, sizeof(*g));
    g->num_proj = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];

    if (scf_eigen_lib_flag != GPUSOLVER) return 0;
    if (!Force_collective_env_flag("OPENMX_FORCE4B_GPU", 1, mpi_comm_level1)) return 0;

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_size(node_comm, &node_ranks);
    MPI_Comm_rank(node_comm, &node_rank);
    if (node_ranks < 1) node_ranks = 1;

    /* return every rank's OpenACC freelist (e.g. the Force3 buffers) to CUDA
       before anyone measures the shared free memory */
    acc_wait_all();
    if (cudaDeviceSynchronize() == cudaSuccess) {
        acc_clear_freelists();
    }
    MPI_Barrier(node_comm);
    MPI_Comm_free(&node_comm);

    /* size the flat local table (4 dirs) and the halo/case-2 archives;
       also find the largest one-direction atom block for the staging
       buffer */

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        size_t blk = (size_t)(FNAN[Gc_AN] + 1) * (size_t)Spe_Total_CNO[WhatSpecies[Gc_AN]]
            * (size_t)g->num_proj;
        size_t c2blk = 0;

        slots += (size_t)(FNAN[Gc_AN] + 1);
        pos += blk;
        if (maxblk < blk) maxblk = blk;
        for (k = 0; k <= FNAN[Gc_AN]; k++) {
            c2blk += (size_t)Spe_Total_CNO[WhatSpecies[natn[Gc_AN][k]]] * (size_t)g->num_proj;
        }
        c2pos += c2blk;
        if (maxblk < c2blk) maxblk = c2blk;
    }
    g->flat_stride = pos;
    g->c2_stride = c2pos;

    for (Mc_AN = Matomnum + 1; Mc_AN <= Matomnum + MatomnumF; Mc_AN++) {
        int Gc_AN = F_M2G[Mc_AN];
        size_t blk = (size_t)(FNAN[Gc_AN] + 1) * (size_t)Spe_Total_CNO[WhatSpecies[Gc_AN]]
            * (size_t)g->num_proj;

        halo_slots += (size_t)(FNAN[Gc_AN] + 1);
        g->halo_count += blk;
        if (maxblk < blk) maxblk = blk;
    }

    /* feasibility against the device memory shared by the node ranks */

    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return 0;
    {
        const size_t reserve = (size_t)256 * 1024 * 1024;
        const size_t transients = (size_t)192 * 1024 * 1024;
        size_t need = (4U * g->flat_stride + g->halo_count + 4U * g->c2_stride) * sizeof(float)
            + transients;

        if (free_bytes <= reserve) return 0;
        if ((free_bytes - reserve) / (size_t)node_ranks <= need) {
            if (node_rank == 0) {
                fprintf(stderr,
                    "Force4B GPU: not enough free device memory; CPU fallback.\n");
                fflush(stderr);
            }
            return 0;
        }
    }

    g->mck_base = (int*)Force_checked_malloc(sizeof(int) * (size_t)(Matomnum + 2), __FILE__, __LINE__);
    g->mck_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (slots == 0 ? 1 : slots), __FILE__, __LINE__);
    g->halo_base = (int*)Force_checked_malloc(sizeof(int) * (size_t)(MatomnumF + 2), __FILE__, __LINE__);
    g->halo_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (halo_slots == 0 ? 1 : halo_slots),
        __FILE__, __LINE__);
    g->c2_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (slots == 0 ? 1 : slots), __FILE__, __LINE__);
    g->stage_len = (maxblk == 0 ? 1 : maxblk);
    g->stage = (float*)Force_checked_malloc(sizeof(float) * g->stage_len, __FILE__, __LINE__);

    pos = 0;
    slots = 0;
    c2pos = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int no0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        g->mck_base[Mc_AN] = (int)slots;
        for (k = 0; k <= FNAN[Gc_AN]; k++) {
            int nok = Spe_Total_CNO[WhatSpecies[natn[Gc_AN][k]]];

            g->mck_off[slots] = pos;
            g->c2_off[slots] = c2pos;
            pos += (size_t)no0 * (size_t)g->num_proj;
            c2pos += (size_t)nok * (size_t)g->num_proj;
            slots++;
        }
    }

    halo_slots = 0;
    pos = 0;
    for (Mc_AN = Matomnum + 1; Mc_AN <= Matomnum + MatomnumF; Mc_AN++) {
        int Gc_AN = F_M2G[Mc_AN];
        int no0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        g->halo_base[Mc_AN - Matomnum] = (int)halo_slots;
        for (k = 0; k <= FNAN[Gc_AN]; k++) {
            g->halo_off[halo_slots] = pos;
            pos += (size_t)no0 * (size_t)g->num_proj;
            halo_slots++;
        }
    }

    /* device archives */

    g->flat_dev = (float*)acc_malloc(sizeof(float) * (g->flat_stride == 0 ? 4 : 4U * g->flat_stride));
    g->halo_dev = (float*)acc_malloc(sizeof(float) * (g->halo_count == 0 ? 1 : g->halo_count));
    g->c2_dev = (float*)acc_malloc(sizeof(float) * (g->c2_stride == 0 ? 4 : 4U * g->c2_stride));
    if (g->flat_dev == NULL || g->halo_dev == NULL || g->c2_dev == NULL) {
        if (g->flat_dev != NULL) acc_free(g->flat_dev);
        if (g->halo_dev != NULL) acc_free(g->halo_dev);
        if (g->c2_dev != NULL) acc_free(g->c2_dev);
        free(g->stage);
        free(g->c2_off);
        free(g->halo_off);
        free(g->halo_base);
        free(g->mck_off);
        free(g->mck_base);
        memset(g, 0, sizeof(*g));
        fprintf(stderr, "Force4B GPU: device allocation failed; CPU fallback.\n");
        fflush(stderr);
        return 0;
    }

    /* upload the (premultiplied) local DS_VNA rows one atom block at a
       time through the staging buffer */

    {
        int kk;

        for (kk = 0; kk <= 3; kk++) {
            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
                int Gc_AN = M2G[Mc_AN];
                int no0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
                size_t base = g->mck_off[g->mck_base[Mc_AN]];
                size_t blk = (size_t)(FNAN[Gc_AN] + 1) * (size_t)no0 * (size_t)g->num_proj;
                int k2;

#pragma omp parallel for schedule(dynamic)
                for (k2 = 0; k2 <= FNAN[Gc_AN]; k2++) {
                    size_t off = g->mck_off[g->mck_base[Mc_AN] + k2] - base;
                    int i, l;

                    for (i = 0; i < no0; i++) {
                        const Type_DS_VNA* src = DS_VNA[kk][Mc_AN][k2][i];
                        float* dst = g->stage + off + (size_t)i * (size_t)g->num_proj;

                        for (l = 0; l < g->num_proj; l++) dst[l] = (float)src[l];
                    }
                }

                if (blk != 0) {
                    acc_memcpy_to_device(g->flat_dev + (size_t)kk * g->flat_stride + base,
                        g->stage, sizeof(float) * blk);
                }
            }
        }
    }

    g->enabled = 1;
    return 1;
}

static void Force4B_GpuEnd(void)
{
    Force4BGpuContext* g = &F4B_gpu;

    if (!g->enabled) return;

    if (g->flat_dev != NULL) acc_free(g->flat_dev);
    if (g->halo_dev != NULL) acc_free(g->halo_dev);
    if (g->c2_dev != NULL) acc_free(g->c2_dev);
    acc_clear_freelists();

    free(g->stage);
    free(g->c2_off);
    free(g->halo_off);
    free(g->halo_base);
    free(g->mck_off);
    free(g->mck_base);
    memset(g, 0, sizeof(*g));
}

/* archive the direction-0 block of the halo atom just received into the
   Matomnum+1 slot of the case-1 communication loop */
static void Force4B_GpuArchiveHalo(Type_DS_VNA***** DS_VNA, int Original_Mc_AN, int Gc_AN)
{
    Force4BGpuContext* g = &F4B_gpu;
    int no0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
    int halo_idx = Original_Mc_AN - Matomnum;
    size_t base;
    size_t blk = (size_t)(FNAN[Gc_AN] + 1) * (size_t)no0 * (size_t)g->num_proj;
    int k;

    if (halo_idx < 1 || MatomnumF < halo_idx) {
        Force4B_gpu_abort("Force4B GPU: halo index out of range.");
    }
    base = g->halo_off[g->halo_base[halo_idx]];

#pragma omp parallel for schedule(dynamic)
    for (k = 0; k <= FNAN[Gc_AN]; k++) {
        size_t off = g->halo_off[g->halo_base[halo_idx] + k] - base;
        int i, l;

        for (i = 0; i < no0; i++) {
            const Type_DS_VNA* src = DS_VNA[0][Matomnum + 1][k][i];
            float* dst = g->stage + off + (size_t)i * (size_t)g->num_proj;

            for (l = 0; l < g->num_proj; l++) dst[l] = (float)src[l];
        }
    }

    if (blk != 0) {
        acc_memcpy_to_device(g->halo_dev + base, g->stage, sizeof(float) * blk);
    }
}

/* archive the staged 4-direction slot of the case-2 centre atom Mc_AN */
static void Force4B_GpuArchiveCase2(Type_DS_VNA***** DS_VNA, int Mc_AN, int Gc_AN)
{
    Force4BGpuContext* g = &F4B_gpu;
    size_t base = g->c2_off[g->mck_base[Mc_AN]];
    size_t blk = 0;
    int k, kk;

    for (k = 0; k <= FNAN[Gc_AN]; k++) {
        blk += (size_t)Spe_Total_CNO[WhatSpecies[natn[Gc_AN][k]]] * (size_t)g->num_proj;
    }

    for (kk = 0; kk <= 3; kk++) {

#pragma omp parallel for schedule(dynamic)
        for (k = 0; k <= FNAN[Gc_AN]; k++) {
            int Gk_AN = natn[Gc_AN][k];
            int nok = Spe_Total_CNO[WhatSpecies[Gk_AN]];
            size_t off = g->c2_off[g->mck_base[Mc_AN] + k] - base;
            int i, l;

            for (i = 0; i < nok; i++) {
                const Type_DS_VNA* src = DS_VNA[kk][Matomnum + 1][k][i];
                float* dst = g->stage + off + (size_t)i * (size_t)g->num_proj;

                for (l = 0; l < g->num_proj; l++) dst[l] = (float)src[l];
            }
        }

        if (blk != 0) {
            acc_memcpy_to_device(g->c2_dev + (size_t)kk * g->c2_stride + base,
                g->stage, sizeof(float) * blk);
        }
    }
}

/* the unified case-1 batch: for every local atom, all q_AN != 0 pairs
   (local q reads the flat table, halo q reads the halo archive) */
static void Force4B_GpuCase1Run(double***** CDM0)
{
    Force4BGpuContext* g = &F4B_gpu;
    const int num_proj = g->num_proj;
    int Mc_AN, q_AN, k;
    int nitems = 0, kslots = 0;
    size_t cdm_count = 0;
    Force4BGpuItem* items;
    int* item_atom;
    int* item_q;
    size_t* krowA;
    size_t* krowB;
    int* krow_halo;
    double* cdm_pref;
    double* item_f;
    int p;
    size_t cpos = 0;
    int kpos = 0;

    /* count */
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        for (q_AN = 1; q_AN <= FNAN[Gc_AN]; q_AN++) {
            int jan = Spe_Total_CNO[WhatSpecies[natn[Gc_AN][q_AN]]];

            nitems++;
            cdm_count += (size_t)ian * (size_t)jan;
            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                if (0 <= RMI1[Mc_AN][q_AN][k]) kslots++;
            }
        }
    }

    if (nitems == 0) return;

    items = (Force4BGpuItem*)Force_checked_malloc(sizeof(Force4BGpuItem) * (size_t)nitems, __FILE__, __LINE__);
    item_atom = (int*)Force_checked_malloc(sizeof(int) * (size_t)nitems, __FILE__, __LINE__);
    item_q = (int*)Force_checked_malloc(sizeof(int) * (size_t)nitems, __FILE__, __LINE__);
    krowA = (size_t*)Force_checked_malloc(sizeof(size_t) * (kslots == 0 ? 1 : (size_t)kslots), __FILE__, __LINE__);
    krowB = (size_t*)Force_checked_malloc(sizeof(size_t) * (kslots == 0 ? 1 : (size_t)kslots), __FILE__, __LINE__);
    krow_halo = (int*)Force_checked_malloc(sizeof(int) * (kslots == 0 ? 1 : (size_t)kslots), __FILE__, __LINE__);
    cdm_pref = (double*)Force_checked_malloc(sizeof(double) * (cdm_count == 0 ? 1 : cdm_count), __FILE__, __LINE__);
    item_f = (double*)Force_checked_malloc(sizeof(double) * 3U * (size_t)nitems, __FILE__, __LINE__);

    /* build */
    p = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        for (q_AN = 1; q_AN <= FNAN[Gc_AN]; q_AN++) {
            int Gq_AN = natn[Gc_AN][q_AN];
            int Mq_AN = F_G2M[Gq_AN];
            int jan = Spe_Total_CNO[WhatSpecies[Gq_AN]];
            int cdm_kl = RMI1[Mc_AN][0][q_AN];
            const double pref = (SpinP_switch == 0) ? 4.0 : 2.0;
            int i, j;

            items[p].cdm_off = cpos;
            items[p].k_off = kpos;
            items[p].ian = ian;
            items[p].jan = jan;
            item_atom[p] = Mc_AN;
            item_q[p] = q_AN;

            for (i = 0; i < ian; i++) {
                const double* c0 = CDM0[0][Mc_AN][cdm_kl][i];
                const double* c1 = (SpinP_switch == 0) ? NULL : CDM0[1][Mc_AN][cdm_kl][i];

                for (j = 0; j < jan; j++) {
                    cdm_pref[cpos++] = pref * ((c1 == NULL) ? c0[j] : (c0[j] + c1[j]));
                }
            }

            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                int qk_kl = RMI1[Mc_AN][q_AN][k];

                if (qk_kl < 0) continue;
                krowA[kpos] = g->mck_off[g->mck_base[Mc_AN] + k];
                if (Mq_AN <= Matomnum) {
                    krowB[kpos] = g->mck_off[g->mck_base[Mq_AN] + qk_kl];
                    krow_halo[kpos] = 0;
                } else {
                    krowB[kpos] = g->halo_off[g->halo_base[Mq_AN - Matomnum] + qk_kl];
                    krow_halo[kpos] = 1;
                }
                kpos++;
            }
            items[p].k_n = kpos - items[p].k_off;
            p++;
        }
    }

    if (p != nitems || kpos != kslots || cpos != cdm_count) {
        Force4B_gpu_abort("Force4B GPU: inconsistent case-1 batch.");
    }

    {
        float* flat = g->flat_dev;
        float* halo = g->halo_dev;
        size_t flat_stride = g->flat_stride;
        const int nitems_c = nitems;
        const size_t kslots_c = (kslots == 0 ? 1 : (size_t)kslots);
        const size_t cdm_c = (cdm_count == 0 ? 1 : cdm_count);
        size_t asz = 0;
        const size_t o_items = Force_gpu_arena_off(&asz, sizeof(Force4BGpuItem) * (size_t)nitems_c);
        const size_t o_krowA = Force_gpu_arena_off(&asz, sizeof(size_t) * kslots_c);
        const size_t o_krowB = Force_gpu_arena_off(&asz, sizeof(size_t) * kslots_c);
        const size_t o_khalo = Force_gpu_arena_off(&asz, sizeof(int) * kslots_c);
        const size_t o_cdm = Force_gpu_arena_off(&asz, sizeof(double) * cdm_c);
        const size_t o_itemf = Force_gpu_arena_off(&asz, sizeof(double) * 3U * (size_t)nitems_c);
        char* arena = (char*)Force_gpu_arena_wait(asz, "4B first-case batch");

        acc_memcpy_to_device(arena + o_items, items, sizeof(Force4BGpuItem) * (size_t)nitems_c);
        acc_memcpy_to_device(arena + o_krowA, krowA, sizeof(size_t) * kslots_c);
        acc_memcpy_to_device(arena + o_krowB, krowB, sizeof(size_t) * kslots_c);
        acc_memcpy_to_device(arena + o_khalo, krow_halo, sizeof(int) * kslots_c);
        acc_memcpy_to_device(arena + o_cdm, cdm_pref, sizeof(double) * cdm_c);

        {
            const Force4BGpuItem* items = (const Force4BGpuItem*)(void*)(arena + o_items);
            const size_t* krowA = (const size_t*)(void*)(arena + o_krowA);
            const size_t* krowB = (const size_t*)(void*)(arena + o_krowB);
            const int* krow_halo = (const int*)(void*)(arena + o_khalo);
            const double* cdm_pref = (const double*)(void*)(arena + o_cdm);
            double* item_f = (double*)(void*)(arena + o_itemf);

#pragma acc parallel loop gang vector_length(128) \
    deviceptr(flat, halo, items, krowA, krowB, krow_halo, cdm_pref, item_f)
            for (int pp = 0; pp < nitems_c; pp++) {
                const int ian = items[pp].ian;
                const int jan = items[pp].jan;
                const int k_off = items[pp].k_off;
                const int k_n = items[pp].k_n;
                const size_t cdm_off = items[pp].cdm_off;
                const int mn = ian * jan;
                double fx = 0.0, fy = 0.0, fz = 0.0;

#pragma acc loop vector reduction(+:fx, fy, fz)
                for (int idx = 0; idx < mn; idx++) {
                    const int m = idx / jan;
                    const int n = idx - m * jan;
                    const double w = cdm_pref[cdm_off + (size_t)idx];
                    double sx = 0.0, sy = 0.0, sz = 0.0;

                    for (int kk = 0; kk < k_n; kk++) {
                        const size_t abase = krowA[k_off + kk] + (size_t)m * (size_t)num_proj;
                        const float* bx = (krow_halo[k_off + kk]
                            ? (halo + krowB[k_off + kk])
                            : (flat + krowB[k_off + kk])) + (size_t)n * (size_t)num_proj;
                        const float* a1 = flat + flat_stride + abase;
                        const float* a2 = flat + 2U * flat_stride + abase;
                        const float* a3 = flat + 3U * flat_stride + abase;
                        double tx = 0.0, ty = 0.0, tz = 0.0;

                        for (int l = 0; l < num_proj; l++) {
                            tx += (double)(a1[l] * bx[l]);
                            ty += (double)(a2[l] * bx[l]);
                            tz += (double)(a3[l] * bx[l]);
                        }
                        sx += tx;
                        sy += ty;
                        sz += tz;
                    }

                    fx += w * sx;
                    fy += w * sy;
                    fz += w * sz;
                }

                item_f[3 * (size_t)pp + 0] = fx;
                item_f[3 * (size_t)pp + 1] = fy;
                item_f[3 * (size_t)pp + 2] = fz;
            }
        }

        acc_memcpy_from_device(item_f, arena + o_itemf, sizeof(double) * 3U * (size_t)nitems_c);
        acc_free(arena);
        Force_gpu_pool_flush();
    }

    /* damping tail and accumulation (host, same math as the fused trace) */

    for (p = 0; p < nitems; p++) {
        int mc = item_atom[p];
        int q_AN = item_q[p];
        int Gc_AN = M2G[mc];
        int Gq_AN = natn[Gc_AN][q_AN];
        int jan = items[p].jan;
        int ian = items[p].ian;
        int cdm_kl = RMI1[mc][0][q_AN];
        const double pref = (SpinP_switch == 0) ? 4.0 : 2.0;
        const double rcut = Spe_Atom_Cut1[WhatSpecies[Gc_AN]] + Spe_Atom_Cut1[WhatSpecies[Gq_AN]];
        double r = Dis[Gc_AN][q_AN];
        const double dmp = dampingF(rcut, r);
        double fx = item_f[3 * (size_t)p + 0] * dmp;
        double fy = item_f[3 * (size_t)p + 1] * dmp;
        double fz = item_f[3 * (size_t)p + 2] * dmp;
        double derivative_scale = 0.0;

        if (rcut > r) {
            derivative_scale = deri_dampingF(rcut, r) / dmp;
        }
        if (r < 1.0e-10) {
            r = 1.0e-10;
        }

        if (derivative_scale != 0.0) {
            double trace_hvna = 0.0;
            const int center_cell = ncn[Gc_AN][0];
            const int q_cell = ncn[Gc_AN][q_AN];
            int m, n;

            for (m = 0; m < ian; m++) {
                const double* c0 = CDM0[0][mc][cdm_kl][m];
                const double* c1 = (SpinP_switch == 0) ? NULL : CDM0[1][mc][cdm_kl][m];

                for (n = 0; n < jan; n++) {
                    const double cdm = (c1 == NULL) ? c0[n] : (c0[n] + c1[n]);

                    trace_hvna += pref * cdm * HVNA[mc][q_AN][m][n];
                }
            }

            {
                const double dx = derivative_scale
                    * ((Gxyz[Gc_AN][1] + atv[center_cell][1])
                        - (Gxyz[Gq_AN][1] + atv[q_cell][1])) / r;
                const double dy = derivative_scale
                    * ((Gxyz[Gc_AN][2] + atv[center_cell][2])
                        - (Gxyz[Gq_AN][2] + atv[q_cell][2])) / r;
                const double dz = derivative_scale
                    * ((Gxyz[Gc_AN][3] + atv[center_cell][3])
                        - (Gxyz[Gq_AN][3] + atv[q_cell][3])) / r;

                fx += trace_hvna * dx;
                fy += trace_hvna * dy;
                fz += trace_hvna * dz;
            }
        }

        Gxyz[Gc_AN][41] += fx;
        Gxyz[Gc_AN][42] += fy;
        Gxyz[Gc_AN][43] += fz;
    }

    free(item_f);
    free(cdm_pref);
    free(krow_halo);
    free(krowB);
    free(krowA);
    free(item_q);
    free(item_atom);
    free(items);

    /* the local table and the halo archive are no longer needed after
       the case-1 batch; release the device memory early */
    if (g->flat_dev != NULL) {
        acc_free(g->flat_dev);
        g->flat_dev = NULL;
    }
    if (g->halo_dev != NULL) {
        acc_free(g->halo_dev);
        g->halo_dev = NULL;
    }
    acc_clear_freelists();
}

/* the case-2 batch: all fused-eligible (h,q) pairs of every centre,
   reading the archived 4-direction stagings */
static void Force4B_GpuCase2Run(double***** CDM0)
{
    Force4BGpuContext* g = &F4B_gpu;
    const int num_proj = g->num_proj;
    int Mc_AN, h_AN, q_AN;
    int nitems = 0;
    size_t cdm_count = 0;
    Force4BGpuItem* items;
    size_t* pair_h_off;
    size_t* pair_q_off;
    int* item_atom;
    double* cdm_scale;
    double* item_f;
    int p;
    size_t cpos = 0;

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];

        for (h_AN = 1; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int start_q_AN = (Solver == 5 || Solver == 8 || Solver == 11) ? 0 : h_AN;
            int ian = Spe_Total_CNO[WhatSpecies[natn[Gc_AN][h_AN]]];

            for (q_AN = start_q_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {
                if (q_AN == 0 || h_AN == q_AN) continue;
                if (RMI1[Mc_AN][h_AN][q_AN] < 0) continue;
                nitems++;
                cdm_count += (size_t)ian * (size_t)Spe_Total_CNO[WhatSpecies[natn[Gc_AN][q_AN]]];
            }
        }
    }

    if (nitems == 0) return;

    items = (Force4BGpuItem*)Force_checked_malloc(sizeof(Force4BGpuItem) * (size_t)nitems, __FILE__, __LINE__);
    pair_h_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)nitems, __FILE__, __LINE__);
    pair_q_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)nitems, __FILE__, __LINE__);
    item_atom = (int*)Force_checked_malloc(sizeof(int) * (size_t)nitems, __FILE__, __LINE__);
    cdm_scale = (double*)Force_checked_malloc(sizeof(double) * (cdm_count == 0 ? 1 : cdm_count), __FILE__, __LINE__);
    item_f = (double*)Force_checked_malloc(sizeof(double) * 3U * (size_t)nitems, __FILE__, __LINE__);

    p = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];

        for (h_AN = 1; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int start_q_AN = (Solver == 5 || Solver == 8 || Solver == 11) ? 0 : h_AN;
            int Gh_AN = natn[Gc_AN][h_AN];
            int Mh_AN = F_G2M[Gh_AN];
            int Hwan = WhatSpecies[Gh_AN];
            int ian = Spe_Total_CNO[Hwan];

            for (q_AN = start_q_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {
                int Gq_AN, Qwan, jan, kl, kl1, kl2, i, j;
                double pref, rcut, scale0;

                if (q_AN == 0 || h_AN == q_AN) continue;
                kl = RMI1[Mc_AN][h_AN][q_AN];
                if (kl < 0) continue;

                Gq_AN = natn[Gc_AN][q_AN];
                Qwan = WhatSpecies[Gq_AN];
                jan = Spe_Total_CNO[Qwan];
                kl1 = RMI1[Mc_AN][0][h_AN];
                kl2 = RMI1[Mc_AN][0][q_AN];

                pref = ((Solver == 5 || Solver == 8 || Solver == 11) || q_AN == h_AN)
                    ? ((SpinP_switch == 0) ? 2.0 : 1.0)
                    : ((SpinP_switch == 0) ? 4.0 : 2.0);
                rcut = Spe_Atom_Cut1[Hwan] + Spe_Atom_Cut1[Qwan];
                scale0 = pref * dampingF(rcut, Dis[Gh_AN][kl]);

                items[p].cdm_off = cpos;
                items[p].k_off = 0;
                items[p].k_n = 0;
                items[p].ian = ian;
                items[p].jan = jan;
                pair_h_off[p] = g->c2_off[g->mck_base[Mc_AN] + kl1];
                pair_q_off[p] = g->c2_off[g->mck_base[Mc_AN] + kl2];
                item_atom[p] = Mc_AN;

                for (i = 0; i < ian; i++) {
                    const double* c0 = CDM0[0][Mh_AN][kl][i];
                    const double* c1 = (SpinP_switch == 0) ? NULL : CDM0[1][Mh_AN][kl][i];

                    for (j = 0; j < jan; j++) {
                        cdm_scale[cpos++] = scale0 * ((c1 == NULL) ? c0[j] : (c0[j] + c1[j]));
                    }
                }
                p++;
            }
        }
    }

    if (p != nitems || cpos != cdm_count) {
        Force4B_gpu_abort("Force4B GPU: inconsistent case-2 batch.");
    }

    {
        float* c2 = g->c2_dev;
        size_t flat_stride = g->c2_stride;
        const int nitems_c = nitems;
        const size_t cdm_c = (cdm_count == 0 ? 1 : cdm_count);
        size_t asz = 0;
        const size_t o_items = Force_gpu_arena_off(&asz, sizeof(Force4BGpuItem) * (size_t)nitems_c);
        const size_t o_hoff = Force_gpu_arena_off(&asz, sizeof(size_t) * (size_t)nitems_c);
        const size_t o_qoff = Force_gpu_arena_off(&asz, sizeof(size_t) * (size_t)nitems_c);
        const size_t o_cdm = Force_gpu_arena_off(&asz, sizeof(double) * cdm_c);
        const size_t o_itemf = Force_gpu_arena_off(&asz, sizeof(double) * 3U * (size_t)nitems_c);
        char* arena = (char*)Force_gpu_arena_wait(asz, "4B second-case batch");

        acc_memcpy_to_device(arena + o_items, items, sizeof(Force4BGpuItem) * (size_t)nitems_c);
        acc_memcpy_to_device(arena + o_hoff, pair_h_off, sizeof(size_t) * (size_t)nitems_c);
        acc_memcpy_to_device(arena + o_qoff, pair_q_off, sizeof(size_t) * (size_t)nitems_c);
        acc_memcpy_to_device(arena + o_cdm, cdm_scale, sizeof(double) * cdm_c);

        {
            const Force4BGpuItem* items = (const Force4BGpuItem*)(void*)(arena + o_items);
            const size_t* pair_h_off = (const size_t*)(void*)(arena + o_hoff);
            const size_t* pair_q_off = (const size_t*)(void*)(arena + o_qoff);
            const double* cdm_scale = (const double*)(void*)(arena + o_cdm);
            double* item_f = (double*)(void*)(arena + o_itemf);

#pragma acc parallel loop gang vector_length(128) \
    deviceptr(c2, items, pair_h_off, pair_q_off, cdm_scale, item_f)
            for (int pp = 0; pp < nitems_c; pp++) {
                const int ian = items[pp].ian;
                const int jan = items[pp].jan;
                const size_t cdm_off = items[pp].cdm_off;
                const size_t hb = pair_h_off[pp];
                const size_t qb = pair_q_off[pp];
                const int mn = ian * jan;
                double fx = 0.0, fy = 0.0, fz = 0.0;

#pragma acc loop vector reduction(+:fx, fy, fz)
                for (int idx = 0; idx < mn; idx++) {
                    const int m = idx / jan;
                    const int n = idx - m * jan;
                    const double w = cdm_scale[cdm_off + (size_t)idx];
                    const float* h0 = c2 + hb + (size_t)m * (size_t)num_proj;
                    const float* hx = c2 + flat_stride + hb + (size_t)m * (size_t)num_proj;
                    const float* hy = c2 + 2U * flat_stride + hb + (size_t)m * (size_t)num_proj;
                    const float* hz = c2 + 3U * flat_stride + hb + (size_t)m * (size_t)num_proj;
                    const float* q0 = c2 + qb + (size_t)n * (size_t)num_proj;
                    const float* qx = c2 + flat_stride + qb + (size_t)n * (size_t)num_proj;
                    const float* qy = c2 + 2U * flat_stride + qb + (size_t)n * (size_t)num_proj;
                    const float* qz = c2 + 3U * flat_stride + qb + (size_t)n * (size_t)num_proj;
                    double sx = 0.0, sy = 0.0, sz = 0.0;

                    for (int l = 0; l < num_proj; l++) {
                        sx -= (double)(hx[l] * q0[l]) + (double)(h0[l] * qx[l]);
                        sy -= (double)(hy[l] * q0[l]) + (double)(h0[l] * qy[l]);
                        sz -= (double)(hz[l] * q0[l]) + (double)(h0[l] * qz[l]);
                    }

                    fx += w * sx;
                    fy += w * sy;
                    fz += w * sz;
                }

                item_f[3 * (size_t)pp + 0] = fx;
                item_f[3 * (size_t)pp + 1] = fy;
                item_f[3 * (size_t)pp + 2] = fz;
            }
        }

        acc_memcpy_from_device(item_f, arena + o_itemf, sizeof(double) * 3U * (size_t)nitems_c);
        acc_free(arena);
        Force_gpu_pool_flush();
    }

    for (p = 0; p < nitems; p++) {
        int Gc_AN = M2G[item_atom[p]];

        Gxyz[Gc_AN][41] += item_f[3 * (size_t)p + 0];
        Gxyz[Gc_AN][42] += item_f[3 * (size_t)p + 1];
        Gxyz[Gc_AN][43] += item_f[3 * (size_t)p + 2];
    }

    free(item_f);
    free(cdm_scale);
    free(item_atom);
    free(pair_q_off);
    free(pair_h_off);
    free(items);
}

/* ------------------------------------------------------------------ */
/* Batched device evaluation of the scalar-relativistic Force_HNL
   traces.

   When every species is j-averaged (VPS_j_dependency==0) and no
   spin-orbit / LDA+U / constraint terms are active, both dHNL cases
   reduce to energy-weighted dot products over the nonlocal projector
   rows -- the same shape as the Force4B traces.  The q_AN==0 pairs of
   the first case and the damping corrections stay on the host.

   The MPI structure is untouched: the halo blocks received in the
   first-case communication loop and the per-atom second-case stagings
   are archived, and one batched kernel per case runs after the loops. */
/* ------------------------------------------------------------------ */

typedef struct {
    int enabled;

    /* local DS_NL rows (so=0), direction-major: flat[dir*stride+block];
       the (Mc,k) block holds CNO(Gc) rows of nlp(species of natn) each */
    double* flat;
    double* flat_dev;
    size_t flat_stride;
    size_t* mck_off;
    int* mck_base;

    /* first-case halo archive: direction-0 rows of every received atom */
    double* halo;
    size_t halo_count;
    size_t* halo_off;
    int* halo_base;

    /* second-case per-centre archive: 4 directions of the staged slot;
       the (Mc,k) block holds CNO(natn[Gc][k]) rows of nlp(centre) each */
    double* c2;
    size_t c2_stride;
    size_t* c2_off;

    /* per-species expanded projector energies (one entry per l)        */
    double* ene;
    size_t* ene_off;
    int* sp_nlp;
} ForceHNLGpuContext;

static ForceHNLGpuContext FHNL_gpu = { 0 };

static int Force_HNL_GpuBegin(double****** DS_NL)
{
    ForceHNLGpuContext* g = &FHNL_gpu;
    int Mc_AN, k, w, node_ranks = 1, node_rank = 0;
    size_t pos = 0, slots = 0, halo_slots = 0, ene_pos = 0;
    size_t free_bytes = 0, total_bytes = 0;
    MPI_Comm node_comm = MPI_COMM_NULL;

    memset(g, 0, sizeof(*g));

    if (scf_eigen_lib_flag != GPUSOLVER) return 0;
    if (!Force_collective_env_flag("OPENMX_FORCEHNL_GPU", 1, mpi_comm_level1)) return 0;
    if (F_NL_flag != 1) return 0;

    /* only the real scalar-relativistic trace branches are batched */
    if (!(SpinP_switch == 0 || SpinP_switch == 1 ||
          (SpinP_switch == 3 && SO_switch == 0 && Hub_U_switch == 0 &&
           Constraint_NCS_switch == 0 && Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0))) {
        return 0;
    }
    for (w = 0; w < SpeciesNum; w++) {
        if (VPS_j_dependency[w] != 0) return 0;
    }

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_size(node_comm, &node_ranks);
    MPI_Comm_rank(node_comm, &node_rank);
    if (node_ranks < 1) node_ranks = 1;
    acc_wait_all();
    if (cudaDeviceSynchronize() == cudaSuccess) {
        acc_clear_freelists();
    }
    MPI_Barrier(node_comm);
    MPI_Comm_free(&node_comm);

    /* expanded projector energies per species */

    g->ene_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)(SpeciesNum + 1), __FILE__, __LINE__);
    g->sp_nlp = (int*)Force_checked_malloc(sizeof(int) * (size_t)SpeciesNum, __FILE__, __LINE__);
    for (w = 0; w < SpeciesNum; w++) {
        int l1, nlp = 0;

        g->ene_off[w] = ene_pos;
        for (l1 = 1; l1 <= Spe_Num_RVPS[w]; l1++) {
            nlp += 2 * Spe_VPS_List[w][l1] + 1;
        }
        g->sp_nlp[w] = nlp;
        ene_pos += (size_t)nlp;
    }
    g->ene_off[SpeciesNum] = ene_pos;
    g->ene = (double*)Force_checked_malloc(sizeof(double) * (ene_pos == 0 ? 1 : ene_pos), __FILE__, __LINE__);
    for (w = 0; w < SpeciesNum; w++) {
        int l1, l3, l = 0;

        for (l1 = 1; l1 <= Spe_Num_RVPS[w]; l1++) {
            const double ene = Spe_VNLE[0][w][l1 - 1];
            const int l2 = 2 * Spe_VPS_List[w][l1];

            for (l3 = 0; l3 <= l2; l3++) {
                g->ene[g->ene_off[w] + (size_t)l] = ene;
                l++;
            }
        }
    }

    /* sizes of the flat local table, halo archive and staged archive */

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int no0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        slots += (size_t)(FNAN[Gc_AN] + 1);
        for (k = 0; k <= FNAN[Gc_AN]; k++) {
            pos += (size_t)no0 * (size_t)g->sp_nlp[WhatSpecies[natn[Gc_AN][k]]];
        }
    }
    g->flat_stride = pos;

    for (Mc_AN = Matomnum + 1; Mc_AN <= Matomnum + MatomnumF; Mc_AN++) {
        int Gc_AN = F_M2G[Mc_AN];
        int no0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        halo_slots += (size_t)(FNAN[Gc_AN] + 1);
        for (k = 0; k <= FNAN[Gc_AN]; k++) {
            g->halo_count += (size_t)no0 * (size_t)g->sp_nlp[WhatSpecies[natn[Gc_AN][k]]];
        }
    }

    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        free(g->ene); free(g->sp_nlp); free(g->ene_off);
        memset(g, 0, sizeof(*g));
        return 0;
    }

    g->mck_base = (int*)Force_checked_malloc(sizeof(int) * (size_t)(Matomnum + 2), __FILE__, __LINE__);
    g->mck_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (slots == 0 ? 1 : slots), __FILE__, __LINE__);
    g->c2_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (slots == 0 ? 1 : slots), __FILE__, __LINE__);

    pos = 0;
    slots = 0;
    {
        size_t c2pos = 0;

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            int Gc_AN = M2G[Mc_AN];
            int no0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
            int nlpc = g->sp_nlp[WhatSpecies[Gc_AN]];

            g->mck_base[Mc_AN] = (int)slots;
            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                int nok = Spe_Total_CNO[WhatSpecies[natn[Gc_AN][k]]];

                g->mck_off[slots] = pos;
                g->c2_off[slots] = c2pos;
                pos += (size_t)no0 * (size_t)g->sp_nlp[WhatSpecies[natn[Gc_AN][k]]];
                c2pos += (size_t)nok * (size_t)nlpc;
                slots++;
            }
        }
        g->c2_stride = c2pos;
    }

    /* feasibility: everything is resident at once (tens of MB) */
    {
        const size_t reserve = (size_t)256 * 1024 * 1024;
        const size_t transients = (size_t)128 * 1024 * 1024;
        size_t need = (4U * g->flat_stride + g->halo_count + 4U * g->c2_stride) * sizeof(double)
            + transients;

        if (free_bytes <= reserve ||
            (free_bytes - reserve) / (size_t)node_ranks <= need) {
            if (node_rank == 0) {
                fprintf(stderr,
                    "Force_HNL GPU: not enough free device memory; CPU fallback.\n");
                fflush(stderr);
            }
            free(g->c2_off); free(g->mck_off); free(g->mck_base);
            free(g->ene); free(g->sp_nlp); free(g->ene_off);
            memset(g, 0, sizeof(*g));
            return 0;
        }
    }

    g->flat = (double*)Force_checked_malloc(sizeof(double) * (g->flat_stride == 0 ? 4 : 4U * g->flat_stride),
        __FILE__, __LINE__);
    g->c2 = (double*)Force_checked_malloc(sizeof(double) * (g->c2_stride == 0 ? 4 : 4U * g->c2_stride),
        __FILE__, __LINE__);
    g->halo_base = (int*)Force_checked_malloc(sizeof(int) * (size_t)(MatomnumF + 2), __FILE__, __LINE__);
    g->halo_off = (size_t*)Force_checked_malloc(sizeof(size_t) * (halo_slots == 0 ? 1 : halo_slots),
        __FILE__, __LINE__);
    g->halo = (double*)Force_checked_malloc(sizeof(double) * (g->halo_count == 0 ? 1 : g->halo_count),
        __FILE__, __LINE__);

    halo_slots = 0;
    pos = 0;
    for (Mc_AN = Matomnum + 1; Mc_AN <= Matomnum + MatomnumF; Mc_AN++) {
        int Gc_AN = F_M2G[Mc_AN];
        int no0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        g->halo_base[Mc_AN - Matomnum] = (int)halo_slots;
        for (k = 0; k <= FNAN[Gc_AN]; k++) {
            g->halo_off[halo_slots] = pos;
            pos += (size_t)no0 * (size_t)g->sp_nlp[WhatSpecies[natn[Gc_AN][k]]];
            halo_slots++;
        }
    }

    /* flatten the local so=0 rows of all four directions */

#pragma omp parallel for schedule(dynamic)
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int no0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
        int kk, k2;

        for (kk = 0; kk <= 3; kk++) {
            for (k2 = 0; k2 <= FNAN[Gc_AN]; k2++) {
                int nlp = FHNL_gpu.sp_nlp[WhatSpecies[natn[Gc_AN][k2]]];
                size_t base = (size_t)kk * FHNL_gpu.flat_stride
                    + FHNL_gpu.mck_off[FHNL_gpu.mck_base[Mc_AN] + k2];
                int i;

                for (i = 0; i < no0; i++) {
                    memcpy(FHNL_gpu.flat + base + (size_t)i * (size_t)nlp,
                        DS_NL[0][kk][Mc_AN][k2][i], sizeof(double) * (size_t)nlp);
                }
            }
        }
    }

    {
        size_t flat_len = (g->flat_stride == 0 ? 4 : 4U * g->flat_stride);

        g->flat_dev = (double*)Force_gpu_arena_wait(flat_len * sizeof(double), "HNL flat archive");
        acc_memcpy_to_device(g->flat_dev, g->flat, flat_len * sizeof(double));
    }

    g->enabled = 1;
    return 1;
}

static void Force_HNL_GpuEnd(void)
{
    ForceHNLGpuContext* g = &FHNL_gpu;

    if (!g->enabled) return;

    if (g->flat_dev != NULL) {
        acc_free(g->flat_dev);
        g->flat_dev = NULL;
    }
    acc_clear_freelists();

    free(g->halo);
    free(g->halo_off);
    free(g->halo_base);
    free(g->c2);
    free(g->flat);
    free(g->c2_off);
    free(g->mck_off);
    free(g->mck_base);
    free(g->ene);
    free(g->sp_nlp);
    free(g->ene_off);
    memset(g, 0, sizeof(*g));
}

/* archive the direction-0 block of the halo atom just received into the
   Matomnum+1 slot of the first-case communication loop */
static void Force_HNL_GpuArchiveHalo(double****** DS_NL, int Original_Mc_AN, int Gc_AN)
{
    ForceHNLGpuContext* g = &FHNL_gpu;
    int no0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
    int halo_idx = Original_Mc_AN - Matomnum;
    int k;

    if (halo_idx < 1 || MatomnumF < halo_idx) {
        Force4B_gpu_abort("Force_HNL GPU: halo index out of range.");
    }

#pragma omp parallel for schedule(dynamic)
    for (k = 0; k <= FNAN[Gc_AN]; k++) {
        int nlp = g->sp_nlp[WhatSpecies[natn[Gc_AN][k]]];
        size_t base = g->halo_off[g->halo_base[halo_idx] + k];
        int i;

        for (i = 0; i < no0; i++) {
            memcpy(g->halo + base + (size_t)i * (size_t)nlp,
                DS_NL[0][0][Matomnum + 1][k][i], sizeof(double) * (size_t)nlp);
        }
    }
}

/* archive the staged 4-direction slot of the second-case centre Mc_AN */
static void Force_HNL_GpuArchiveCase2(double****** DS_NL, int Mc_AN, int Gc_AN)
{
    ForceHNLGpuContext* g = &FHNL_gpu;
    int nlpc = g->sp_nlp[WhatSpecies[Gc_AN]];
    int k;

#pragma omp parallel for schedule(dynamic)
    for (k = 0; k <= FNAN[Gc_AN]; k++) {
        int nok = Spe_Total_CNO[WhatSpecies[natn[Gc_AN][k]]];
        int kk, i;

        for (kk = 0; kk <= 3; kk++) {
            size_t base = (size_t)kk * g->c2_stride + g->c2_off[g->mck_base[Mc_AN] + k];

            for (i = 0; i < nok; i++) {
                memcpy(g->c2 + base + (size_t)i * (size_t)nlpc,
                    DS_NL[0][kk][Matomnum + 1][k][i], sizeof(double) * (size_t)nlpc);
            }
        }
    }
}

/* the unified first-case batch: every (local atom, q_AN != 0) pair;
   local q reads the flat table, halo q reads the halo archive */
static void Force_HNL_GpuCase1Run(double***** CDM0)
{
    ForceHNLGpuContext* g = &FHNL_gpu;
    int Mc_AN, q_AN, k;
    int nitems = 0, kslots = 0;
    size_t cdm_count = 0;
    Force4BGpuItem* items;
    int* item_atom;
    int* item_q;
    size_t* krowA;
    size_t* krowB;
    int* krow_halo;
    int* krow_ene;
    int* krow_nlp;
    double* cdm_pref;
    double* item_f;
    int p;
    size_t cpos = 0;
    int kpos = 0;

    /* count */
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        for (q_AN = 1; q_AN <= FNAN[Gc_AN]; q_AN++) {
            if (RMI1[Mc_AN][0][q_AN] < 0) continue;
            nitems++;
            cdm_count += (size_t)ian * (size_t)Spe_Total_CNO[WhatSpecies[natn[Gc_AN][q_AN]]];
            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                if (0 <= RMI1[Mc_AN][q_AN][k]) kslots++;
            }
        }
    }

    if (nitems == 0) return;

    items = (Force4BGpuItem*)Force_checked_malloc(sizeof(Force4BGpuItem) * (size_t)nitems, __FILE__, __LINE__);
    item_atom = (int*)Force_checked_malloc(sizeof(int) * (size_t)nitems, __FILE__, __LINE__);
    item_q = (int*)Force_checked_malloc(sizeof(int) * (size_t)nitems, __FILE__, __LINE__);
    krowA = (size_t*)Force_checked_malloc(sizeof(size_t) * (kslots == 0 ? 1 : (size_t)kslots), __FILE__, __LINE__);
    krowB = (size_t*)Force_checked_malloc(sizeof(size_t) * (kslots == 0 ? 1 : (size_t)kslots), __FILE__, __LINE__);
    krow_halo = (int*)Force_checked_malloc(sizeof(int) * (kslots == 0 ? 1 : (size_t)kslots), __FILE__, __LINE__);
    krow_ene = (int*)Force_checked_malloc(sizeof(int) * (kslots == 0 ? 1 : (size_t)kslots), __FILE__, __LINE__);
    krow_nlp = (int*)Force_checked_malloc(sizeof(int) * (kslots == 0 ? 1 : (size_t)kslots), __FILE__, __LINE__);
    cdm_pref = (double*)Force_checked_malloc(sizeof(double) * (cdm_count == 0 ? 1 : cdm_count), __FILE__, __LINE__);
    item_f = (double*)Force_checked_malloc(sizeof(double) * 3U * (size_t)nitems, __FILE__, __LINE__);

    /* build */
    p = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        for (q_AN = 1; q_AN <= FNAN[Gc_AN]; q_AN++) {
            int Gq_AN, Mq_AN, jan, cdm_kl, i, j;
            const double pref = (SpinP_switch == 0) ? 4.0 : 2.0;

            cdm_kl = RMI1[Mc_AN][0][q_AN];
            if (cdm_kl < 0) continue;

            Gq_AN = natn[Gc_AN][q_AN];
            Mq_AN = F_G2M[Gq_AN];
            jan = Spe_Total_CNO[WhatSpecies[Gq_AN]];

            items[p].cdm_off = cpos;
            items[p].k_off = kpos;
            items[p].ian = ian;
            items[p].jan = jan;
            item_atom[p] = Mc_AN;
            item_q[p] = q_AN;

            for (i = 0; i < ian; i++) {
                const double* c0 = CDM0[0][Mc_AN][cdm_kl][i];
                const double* c1 = (SpinP_switch == 0) ? NULL : CDM0[1][Mc_AN][cdm_kl][i];

                for (j = 0; j < jan; j++) {
                    cdm_pref[cpos++] = pref * ((c1 == NULL) ? c0[j] : (c0[j] + c1[j]));
                }
            }

            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                int qk_kl = RMI1[Mc_AN][q_AN][k];
                int kwan;

                if (qk_kl < 0) continue;
                kwan = WhatSpecies[natn[Gc_AN][k]];
                krowA[kpos] = g->mck_off[g->mck_base[Mc_AN] + k];
                krow_ene[kpos] = (int)g->ene_off[kwan];
                krow_nlp[kpos] = g->sp_nlp[kwan];
                if (Mq_AN <= Matomnum) {
                    krowB[kpos] = g->mck_off[g->mck_base[Mq_AN] + qk_kl];
                    krow_halo[kpos] = 0;
                } else {
                    krowB[kpos] = g->halo_off[g->halo_base[Mq_AN - Matomnum] + qk_kl];
                    krow_halo[kpos] = 1;
                }
                kpos++;
            }
            items[p].k_n = kpos - items[p].k_off;
            p++;
        }
    }

    if (p != nitems || kpos != kslots || cpos != cdm_count) {
        Force4B_gpu_abort("Force_HNL GPU: inconsistent first-case batch.");
    }

    {
        const double* flat = g->flat_dev;
        size_t halo_len = (g->halo_count == 0 ? 1 : g->halo_count);
        size_t ene_len = (g->ene_off[SpeciesNum] == 0 ? 1 : g->ene_off[SpeciesNum]);
        size_t flat_stride = g->flat_stride;
        const int nitems_c = nitems;
        const size_t kslots_c = (size_t)kslots;
        const size_t cdm_c = cdm_count;
        size_t asz = 0;
        const size_t o_items = Force_gpu_arena_off(&asz, sizeof(Force4BGpuItem) * (size_t)nitems_c);
        const size_t o_krowA = Force_gpu_arena_off(&asz, sizeof(size_t) * kslots_c);
        const size_t o_krowB = Force_gpu_arena_off(&asz, sizeof(size_t) * kslots_c);
        const size_t o_khalo = Force_gpu_arena_off(&asz, sizeof(int) * kslots_c);
        const size_t o_kene = Force_gpu_arena_off(&asz, sizeof(int) * kslots_c);
        const size_t o_knlp = Force_gpu_arena_off(&asz, sizeof(int) * kslots_c);
        const size_t o_cdm = Force_gpu_arena_off(&asz, sizeof(double) * cdm_c);
        const size_t o_halo = Force_gpu_arena_off(&asz, sizeof(double) * halo_len);
        const size_t o_ene = Force_gpu_arena_off(&asz, sizeof(double) * ene_len);
        const size_t o_itemf = Force_gpu_arena_off(&asz, sizeof(double) * 3U * (size_t)nitems_c);
        char* arena = (char*)Force_gpu_arena_wait(asz, "HNL first-case batch");

        acc_memcpy_to_device(arena + o_items, items, sizeof(Force4BGpuItem) * (size_t)nitems_c);
        acc_memcpy_to_device(arena + o_krowA, krowA, sizeof(size_t) * kslots_c);
        acc_memcpy_to_device(arena + o_krowB, krowB, sizeof(size_t) * kslots_c);
        acc_memcpy_to_device(arena + o_khalo, krow_halo, sizeof(int) * kslots_c);
        acc_memcpy_to_device(arena + o_kene, krow_ene, sizeof(int) * kslots_c);
        acc_memcpy_to_device(arena + o_knlp, krow_nlp, sizeof(int) * kslots_c);
        acc_memcpy_to_device(arena + o_cdm, cdm_pref, sizeof(double) * cdm_c);
        acc_memcpy_to_device(arena + o_halo, g->halo, sizeof(double) * halo_len);
        acc_memcpy_to_device(arena + o_ene, g->ene, sizeof(double) * ene_len);

        {
            const Force4BGpuItem* items = (const Force4BGpuItem*)(void*)(arena + o_items);
            const size_t* krowA = (const size_t*)(void*)(arena + o_krowA);
            const size_t* krowB = (const size_t*)(void*)(arena + o_krowB);
            const int* krow_halo = (const int*)(void*)(arena + o_khalo);
            const int* krow_ene = (const int*)(void*)(arena + o_kene);
            const int* krow_nlp = (const int*)(void*)(arena + o_knlp);
            const double* cdm_pref = (const double*)(void*)(arena + o_cdm);
            const double* halo = (const double*)(void*)(arena + o_halo);
            const double* ene = (const double*)(void*)(arena + o_ene);
            double* item_f = (double*)(void*)(arena + o_itemf);

#pragma acc parallel loop gang vector_length(128) \
    deviceptr(items, krowA, krowB, krow_halo, krow_ene, krow_nlp, cdm_pref, halo, ene, item_f, flat)
            for (int pp = 0; pp < nitems_c; pp++) {
                const int ian = items[pp].ian;
                const int jan = items[pp].jan;
                const int k_off = items[pp].k_off;
                const int k_n = items[pp].k_n;
                const size_t cdm_off = items[pp].cdm_off;
                const int mn = ian * jan;
                double fx = 0.0, fy = 0.0, fz = 0.0;

#pragma acc loop vector reduction(+:fx, fy, fz)
                for (int idx = 0; idx < mn; idx++) {
                    const int m = idx / jan;
                    const int n = idx - m * jan;
                    const double w = cdm_pref[cdm_off + (size_t)idx];
                    double sx = 0.0, sy = 0.0, sz = 0.0;

                    for (int kk = 0; kk < k_n; kk++) {
                        const int nlp = krow_nlp[k_off + kk];
                        const size_t abase = krowA[k_off + kk] + (size_t)m * (size_t)nlp;
                        const double* b0 = (krow_halo[k_off + kk]
                            ? (halo + krowB[k_off + kk])
                            : (flat + krowB[k_off + kk])) + (size_t)n * (size_t)nlp;
                        const double* a1 = flat + flat_stride + abase;
                        const double* a2 = flat + 2U * flat_stride + abase;
                        const double* a3 = flat + 3U * flat_stride + abase;
                        const double* el = ene + krow_ene[k_off + kk];
                        double tx = 0.0, ty = 0.0, tz = 0.0;

                        for (int l = 0; l < nlp; l++) {
                            tx += el[l] * a1[l] * b0[l];
                            ty += el[l] * a2[l] * b0[l];
                            tz += el[l] * a3[l] * b0[l];
                        }
                        sx += tx;
                        sy += ty;
                        sz += tz;
                    }

                    fx += w * sx;
                    fy += w * sy;
                    fz += w * sz;
                }

                item_f[3 * (size_t)pp + 0] = fx;
                item_f[3 * (size_t)pp + 1] = fy;
                item_f[3 * (size_t)pp + 2] = fz;
            }
        }

        acc_memcpy_from_device(item_f, arena + o_itemf, sizeof(double) * 3U * (size_t)nitems_c);
        acc_free(arena);
        Force_gpu_pool_flush();
    }

    /* damping tail (dmp scaling plus the dQ/dx * HNL term) and the
       per-atom accumulation, exactly as the host dHNL applies them */

    for (p = 0; p < nitems; p++) {
        int mc = item_atom[p];
        int q_AN = item_q[p];
        int Gc_AN = M2G[mc];
        int Gq_AN = natn[Gc_AN][q_AN];
        int ian = items[p].ian;
        int jan = items[p].jan;
        int cdm_kl = RMI1[mc][0][q_AN];
        const double pref = (SpinP_switch == 0) ? 4.0 : 2.0;
        const double rcut = Spe_Atom_Cut1[WhatSpecies[Gc_AN]] + Spe_Atom_Cut1[WhatSpecies[Gq_AN]];
        const double dmp = dampingF(rcut, Dis[Gc_AN][cdm_kl]);
        double fx = item_f[3 * (size_t)p + 0] * dmp;
        double fy = item_f[3 * (size_t)p + 1] * dmp;
        double fz = item_f[3 * (size_t)p + 2] * dmp;
        double r = Dis[Gc_AN][q_AN];
        double tmp = 0.0;

        if (r < rcut) {
            tmp = deri_dampingF(rcut, r) / dmp;
        }
        if (r < 1.0e-10) {
            r = 1.0e-10;
        }

        if (tmp != 0.0) {
            const int Rni = ncn[Gc_AN][0];
            const int Rnj = ncn[Gc_AN][q_AN];
            const double dxv = tmp * ((Gxyz[Gc_AN][1] + atv[Rni][1]) - (Gxyz[Gq_AN][1] + atv[Rnj][1])) / r;
            const double dyv = tmp * ((Gxyz[Gc_AN][2] + atv[Rni][2]) - (Gxyz[Gq_AN][2] + atv[Rnj][2])) / r;
            const double dzv = tmp * ((Gxyz[Gc_AN][3] + atv[Rni][3]) - (Gxyz[Gq_AN][3] + atv[Rnj][3])) / r;
            const int somax = (SpinP_switch == 0) ? 0 : 1;
            double trace = 0.0;
            int so, i, j;

            for (so = 0; so <= somax; so++) {
                for (i = 0; i < ian; i++) {
                    const double* c = CDM0[so][mc][cdm_kl][i];
                    const double* hrow = HNL[so][mc][q_AN][i];

                    for (j = 0; j < jan; j++) {
                        trace += c[j] * hrow[j];
                    }
                }
            }
            trace *= pref;

            fx += trace * dxv;
            fy += trace * dyv;
            fz += trace * dzv;
        }

        Gxyz[Gc_AN][41] += fx;
        Gxyz[Gc_AN][42] += fy;
        Gxyz[Gc_AN][43] += fz;
    }

    free(item_f);
    free(cdm_pref);
    free(krow_nlp);
    free(krow_ene);
    free(krow_halo);
    free(krowB);
    free(krowA);
    free(item_q);
    free(item_atom);
    free(items);
}

/* the second-case batch: all (h,q) pairs of every centre, reading the
   archived 4-direction stagings (the centre's own k=0 row serves as the
   value side of the h_AN==0 pairs) */
static void Force_HNL_GpuCase2Run(double***** CDM0)
{
    ForceHNLGpuContext* g = &FHNL_gpu;
    const int solver_flat = (Solver == 5 || Solver == 8 || Solver == 11);
    int Mc_AN, h_AN, q_AN;
    int nitems = 0;
    size_t cdm_count = 0;
    Force4BGpuItem* items;
    int* item_atom;
    int* item_h;
    int* item_qn;
    int* item_h0;
    int* item_nlp;
    int* item_ene;
    size_t* item_a;
    size_t* item_b;
    double* cdm_pref;
    double* item_f;
    int p;
    size_t cpos = 0;

    /* count */
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int start_q_AN = solver_flat ? 0 : h_AN;

            for (q_AN = start_q_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {
                if (h_AN == 0 && q_AN == 0) continue;
                if (RMI1[Mc_AN][h_AN][q_AN] < 0) continue;
                nitems++;
                cdm_count += (size_t)Spe_Total_CNO[WhatSpecies[natn[Gc_AN][h_AN]]]
                    * (size_t)Spe_Total_CNO[WhatSpecies[natn[Gc_AN][q_AN]]];
            }
        }
    }

    if (nitems == 0) return;

    items = (Force4BGpuItem*)Force_checked_malloc(sizeof(Force4BGpuItem) * (size_t)nitems, __FILE__, __LINE__);
    item_atom = (int*)Force_checked_malloc(sizeof(int) * (size_t)nitems, __FILE__, __LINE__);
    item_h = (int*)Force_checked_malloc(sizeof(int) * (size_t)nitems, __FILE__, __LINE__);
    item_qn = (int*)Force_checked_malloc(sizeof(int) * (size_t)nitems, __FILE__, __LINE__);
    item_h0 = (int*)Force_checked_malloc(sizeof(int) * (size_t)nitems, __FILE__, __LINE__);
    item_nlp = (int*)Force_checked_malloc(sizeof(int) * (size_t)nitems, __FILE__, __LINE__);
    item_ene = (int*)Force_checked_malloc(sizeof(int) * (size_t)nitems, __FILE__, __LINE__);
    item_a = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)nitems, __FILE__, __LINE__);
    item_b = (size_t*)Force_checked_malloc(sizeof(size_t) * (size_t)nitems, __FILE__, __LINE__);
    cdm_pref = (double*)Force_checked_malloc(sizeof(double) * (cdm_count == 0 ? 1 : cdm_count), __FILE__, __LINE__);
    item_f = (double*)Force_checked_malloc(sizeof(double) * 3U * (size_t)nitems, __FILE__, __LINE__);

    /* build */
    p = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int start_q_AN = solver_flat ? 0 : h_AN;
            int Gh_AN = natn[Gc_AN][h_AN];
            int Mh_AN = F_G2M[Gh_AN];
            int ian = Spe_Total_CNO[WhatSpecies[Gh_AN]];

            for (q_AN = start_q_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {
                int Gq_AN, jan, kl, kl1, kl2, i, j;
                double pref;

                if (h_AN == 0 && q_AN == 0) continue;
                kl = RMI1[Mc_AN][h_AN][q_AN];
                if (kl < 0) continue;

                Gq_AN = natn[Gc_AN][q_AN];
                jan = Spe_Total_CNO[WhatSpecies[Gq_AN]];

                if (solver_flat || q_AN == h_AN) {
                    pref = (SpinP_switch == 0) ? 2.0 : 1.0;
                } else {
                    pref = (SpinP_switch == 0) ? 4.0 : 2.0;
                }

                kl1 = RMI1[Mc_AN][0][h_AN];
                kl2 = RMI1[Mc_AN][0][q_AN];
                if (kl1 < 0 || kl2 < 0) {
                    Force4B_gpu_abort("Force_HNL GPU: negative staging index.");
                }

                items[p].cdm_off = cpos;
                items[p].k_off = 0;
                items[p].k_n = 0;
                items[p].ian = ian;
                items[p].jan = jan;
                item_atom[p] = Mc_AN;
                item_h[p] = h_AN;
                item_qn[p] = q_AN;
                item_h0[p] = (h_AN == 0) ? 1 : ((q_AN == 0) ? 2 : 0);
                item_nlp[p] = g->sp_nlp[Cwan];
                item_ene[p] = (int)g->ene_off[Cwan];
                if (h_AN == 0) {
                    /* value side: the centre's own k=0 row of the flat table */
                    item_a[p] = g->mck_off[g->mck_base[Mc_AN] + 0];
                    item_b[p] = g->c2_off[g->mck_base[Mc_AN] + kl2];
                } else {
                    item_a[p] = g->c2_off[g->mck_base[Mc_AN] + kl1];
                    item_b[p] = g->c2_off[g->mck_base[Mc_AN] + kl2];
                }

                for (i = 0; i < ian; i++) {
                    const double* c0 = CDM0[0][Mh_AN][kl][i];
                    const double* c1 = (SpinP_switch == 0) ? NULL : CDM0[1][Mh_AN][kl][i];

                    for (j = 0; j < jan; j++) {
                        cdm_pref[cpos++] = pref * ((c1 == NULL) ? c0[j] : (c0[j] + c1[j]));
                    }
                }
                p++;
            }
        }
    }

    if (p != nitems || cpos != cdm_count) {
        Force4B_gpu_abort("Force_HNL GPU: inconsistent second-case batch.");
    }

    {
        const double* flat = g->flat_dev;
        size_t c2_len = (g->c2_stride == 0 ? 4 : 4U * g->c2_stride);
        size_t ene_len = (g->ene_off[SpeciesNum] == 0 ? 1 : g->ene_off[SpeciesNum]);
        size_t c2_stride = g->c2_stride;
        const int nitems_c = nitems;
        const size_t cdm_c = cdm_count;
        size_t asz = 0;
        const size_t o_items = Force_gpu_arena_off(&asz, sizeof(Force4BGpuItem) * (size_t)nitems_c);
        const size_t o_ih0 = Force_gpu_arena_off(&asz, sizeof(int) * (size_t)nitems_c);
        const size_t o_inlp = Force_gpu_arena_off(&asz, sizeof(int) * (size_t)nitems_c);
        const size_t o_iene = Force_gpu_arena_off(&asz, sizeof(int) * (size_t)nitems_c);
        const size_t o_ia = Force_gpu_arena_off(&asz, sizeof(size_t) * (size_t)nitems_c);
        const size_t o_ib = Force_gpu_arena_off(&asz, sizeof(size_t) * (size_t)nitems_c);
        const size_t o_cdm = Force_gpu_arena_off(&asz, sizeof(double) * cdm_c);
        const size_t o_c2 = Force_gpu_arena_off(&asz, sizeof(double) * c2_len);
        const size_t o_ene = Force_gpu_arena_off(&asz, sizeof(double) * ene_len);
        const size_t o_itemf = Force_gpu_arena_off(&asz, sizeof(double) * 3U * (size_t)nitems_c);
        char* arena = (char*)Force_gpu_arena_wait(asz, "HNL second-case batch");

        acc_memcpy_to_device(arena + o_items, items, sizeof(Force4BGpuItem) * (size_t)nitems_c);
        acc_memcpy_to_device(arena + o_ih0, item_h0, sizeof(int) * (size_t)nitems_c);
        acc_memcpy_to_device(arena + o_inlp, item_nlp, sizeof(int) * (size_t)nitems_c);
        acc_memcpy_to_device(arena + o_iene, item_ene, sizeof(int) * (size_t)nitems_c);
        acc_memcpy_to_device(arena + o_ia, item_a, sizeof(size_t) * (size_t)nitems_c);
        acc_memcpy_to_device(arena + o_ib, item_b, sizeof(size_t) * (size_t)nitems_c);
        acc_memcpy_to_device(arena + o_cdm, cdm_pref, sizeof(double) * cdm_c);
        acc_memcpy_to_device(arena + o_c2, g->c2, sizeof(double) * c2_len);
        acc_memcpy_to_device(arena + o_ene, g->ene, sizeof(double) * ene_len);

        {
            const Force4BGpuItem* items = (const Force4BGpuItem*)(void*)(arena + o_items);
            const int* item_h0 = (const int*)(void*)(arena + o_ih0);
            const int* item_nlp = (const int*)(void*)(arena + o_inlp);
            const int* item_ene = (const int*)(void*)(arena + o_iene);
            const size_t* item_a = (const size_t*)(void*)(arena + o_ia);
            const size_t* item_b = (const size_t*)(void*)(arena + o_ib);
            const double* cdm_pref = (const double*)(void*)(arena + o_cdm);
            const double* c2 = (const double*)(void*)(arena + o_c2);
            const double* ene = (const double*)(void*)(arena + o_ene);
            double* item_f = (double*)(void*)(arena + o_itemf);

#pragma acc parallel loop gang vector_length(128) \
    deviceptr(items, item_h0, item_nlp, item_ene, item_a, item_b, cdm_pref, c2, ene, item_f, flat)
            for (int pp = 0; pp < nitems_c; pp++) {
                const int ian = items[pp].ian;
                const int jan = items[pp].jan;
                const size_t cdm_off = items[pp].cdm_off;
                const int h0 = item_h0[pp];
                const int nlp = item_nlp[pp];
                const size_t a_off = item_a[pp];
                const size_t b_off = item_b[pp];
                const double* el = ene + item_ene[pp];
                const int mn = ian * jan;
                double fx = 0.0, fy = 0.0, fz = 0.0;

#pragma acc loop vector reduction(+:fx, fy, fz)
                for (int idx = 0; idx < mn; idx++) {
                    const int m = idx / jan;
                    const int n = idx - m * jan;
                    const double w = cdm_pref[cdm_off + (size_t)idx];
                    double sx = 0.0, sy = 0.0, sz = 0.0;

                    if (h0 == 1) {
                        /* -sum_l ene * <i|k>(0) * d<j|k>(dir) */
                        const double* a0 = flat + a_off + (size_t)m * (size_t)nlp;
                        const double* b1 = c2 + c2_stride + b_off + (size_t)n * (size_t)nlp;
                        const double* b2 = c2 + 2U * c2_stride + b_off + (size_t)n * (size_t)nlp;
                        const double* b3 = c2 + 3U * c2_stride + b_off + (size_t)n * (size_t)nlp;

                        for (int l = 0; l < nlp; l++) {
                            sx -= el[l] * a0[l] * b1[l];
                            sy -= el[l] * a0[l] * b2[l];
                            sz -= el[l] * a0[l] * b3[l];
                        }
                    } else {
                        /* -sum_l ene * d<i|k>(dir) * <j|k>(0) */
                        const double* a1 = c2 + c2_stride + a_off + (size_t)m * (size_t)nlp;
                        const double* a2 = c2 + 2U * c2_stride + a_off + (size_t)m * (size_t)nlp;
                        const double* a3 = c2 + 3U * c2_stride + a_off + (size_t)m * (size_t)nlp;
                        const double* b0 = c2 + b_off + (size_t)n * (size_t)nlp;

                        for (int l = 0; l < nlp; l++) {
                            sx -= el[l] * a1[l] * b0[l];
                            sy -= el[l] * a2[l] * b0[l];
                            sz -= el[l] * a3[l] * b0[l];
                        }

                        if (h0 == 0) {
                            /* -sum_l ene * <i|k>(0) * d<j|k>(dir) */
                            const double* a0 = c2 + a_off + (size_t)m * (size_t)nlp;
                            const double* b1 = c2 + c2_stride + b_off + (size_t)n * (size_t)nlp;
                            const double* b2 = c2 + 2U * c2_stride + b_off + (size_t)n * (size_t)nlp;
                            const double* b3 = c2 + 3U * c2_stride + b_off + (size_t)n * (size_t)nlp;

                            for (int l = 0; l < nlp; l++) {
                                sx -= el[l] * a0[l] * b1[l];
                                sy -= el[l] * a0[l] * b2[l];
                                sz -= el[l] * a0[l] * b3[l];
                            }
                        }
                    }

                    fx += w * sx;
                    fy += w * sy;
                    fz += w * sz;
                }

                item_f[3 * (size_t)pp + 0] = fx;
                item_f[3 * (size_t)pp + 1] = fy;
                item_f[3 * (size_t)pp + 2] = fz;
            }
        }

        acc_memcpy_from_device(item_f, arena + o_itemf, sizeof(double) * 3U * (size_t)nitems_c);
        acc_free(arena);
        Force_gpu_pool_flush();
    }

    /* damping tail and per-centre accumulation */

    for (p = 0; p < nitems; p++) {
        int mc = item_atom[p];
        int h_AN = item_h[p];
        int q_AN = item_qn[p];
        int Gc_AN = M2G[mc];
        int Gh_AN = natn[Gc_AN][h_AN];
        int Gq_AN = natn[Gc_AN][q_AN];
        int ian = items[p].ian;
        int jan = items[p].jan;
        int kl = RMI1[mc][h_AN][q_AN];
        const double rcut = Spe_Atom_Cut1[WhatSpecies[Gh_AN]] + Spe_Atom_Cut1[WhatSpecies[Gq_AN]];
        const double dmp = dampingF(rcut, Dis[Gh_AN][kl]);
        double fx = item_f[3 * (size_t)p + 0] * dmp;
        double fy = item_f[3 * (size_t)p + 1] * dmp;
        double fz = item_f[3 * (size_t)p + 2] * dmp;

        if ((h_AN == 0 && q_AN != 0) || (h_AN != 0 && q_AN == 0)) {
            const int kld = (h_AN == 0) ? q_AN : h_AN;
            double r = Dis[Gc_AN][kld];
            double tmp = 0.0;

            if (r < rcut) {
                tmp = deri_dampingF(rcut, r) / dmp;
            }
            if (r < 1.0e-10) {
                r = 1.0e-10;
            }

            if (tmp != 0.0) {
                const int Rni = ncn[Gc_AN][h_AN];
                const int Rnj = ncn[Gc_AN][q_AN];
                const double sgn = (h_AN == 0) ? 1.0 : -1.0;
                const double dxv = sgn * tmp
                    * ((Gxyz[Gh_AN][1] + atv[Rni][1]) - (Gxyz[Gq_AN][1] + atv[Rnj][1])) / r;
                const double dyv = sgn * tmp
                    * ((Gxyz[Gh_AN][2] + atv[Rni][2]) - (Gxyz[Gq_AN][2] + atv[Rnj][2])) / r;
                const double dzv = sgn * tmp
                    * ((Gxyz[Gh_AN][3] + atv[Rni][3]) - (Gxyz[Gq_AN][3] + atv[Rnj][3])) / r;
                const int somax = (SpinP_switch == 0) ? 0 : 1;
                const int Mh_AN = F_G2M[Gh_AN];
                double pref;
                double trace = 0.0;
                int so, i, j;

                if (solver_flat || q_AN == h_AN) {
                    pref = (SpinP_switch == 0) ? 2.0 : 1.0;
                } else {
                    pref = (SpinP_switch == 0) ? 4.0 : 2.0;
                }

                for (so = 0; so <= somax; so++) {
                    for (i = 0; i < ian; i++) {
                        const double* c = CDM0[so][Mh_AN][kl][i];

                        if (h_AN == 0) {
                            const double* hrow = HNL[so][mc][kld][i];

                            for (j = 0; j < jan; j++) {
                                trace += c[j] * hrow[j];
                            }
                        } else {
                            for (j = 0; j < jan; j++) {
                                trace += c[j] * HNL[so][mc][kld][j][i];
                            }
                        }
                    }
                }
                trace *= pref;

                fx += trace * dxv;
                fy += trace * dyv;
                fz += trace * dzv;
            }
        }

        Gxyz[Gc_AN][41] += fx;
        Gxyz[Gc_AN][42] += fy;
        Gxyz[Gc_AN][43] += fz;
    }

    free(item_f);
    free(cdm_pref);
    free(item_b);
    free(item_a);
    free(item_ene);
    free(item_nlp);
    free(item_h0);
    free(item_qn);
    free(item_h);
    free(item_atom);
    free(items);
}

void Force4B(double***** CDM0)
{
    /****************************************************
                        #4 of Force

              by the projector expansion of VNA
    ****************************************************/

    int Mc_AN, Gc_AN, Cwan, i, j, h_AN, q_AN, start_q_AN, Mq_AN;
    int jan, kl, Qwan, Gq_AN, Gh_AN, Mh_AN, Hwan, ian;
    int l1, l2, l3, l, LL, Mul1, Num_RVNA, tno0, ncp;
    int tno1, tno2, size1, size2, n, kk, num, po, po1, po2;
    int numprocs, myid, tag = 999, ID, IDS, IDR;
    int **S_array, **R_array;
    int S_comm_flag, R_comm_flag;
    int SA_num, q, Sc_AN, GSc_AN;
    int Sc_wan, Sh_AN, GSh_AN, Sh_wan;
    int Sh_AN2, fan, jg, j0, jg0, Mj_AN0;
    int Original_Mc_AN, max_ODNloop;

    double rcutA, rcutB, rcut;
    double dEx, dEy, dEz, ene, pref;
    double Stime_atom, Etime_atom;
    double **HVNAx, **HVNAy, **HVNAz;
    double *****ActiveHVNA2, *****ActiveHVNA3;
    int* VNA_List;
    int* VNA_List2;
    int *Snd_DS_VNA_Size, *Rcv_DS_VNA_Size;
    int* Indicator;
    Type_DS_VNA* tmp_array;
    Type_DS_VNA* tmp_array2;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs, Nloop, ODNloop;
    int *OneD2h_AN, *OneD2q_AN;
    double stime, etime;
    double stime1, etime1;

    MPI_Status stat;
    MPI_Request request;

    static int counter = 0;
    const int force_profile = Force_collective_env_flag(
        "OPENMX_FORCE_PROFILE", 0, mpi_comm_level1);

    counter++;

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    dtime(&stime);

    /****************************
         allocation of arrays
    *****************************/

    Indicator = (int*)malloc(sizeof(int) * numprocs);

    S_array = Force_alloc_int_matrix(numprocs, 3);
    R_array = Force_alloc_int_matrix(numprocs, 3);

    Snd_DS_VNA_Size = (int*)malloc(sizeof(int) * numprocs);
    Rcv_DS_VNA_Size = (int*)malloc(sizeof(int) * numprocs);

    VNA_List = (int*)malloc(sizeof(int) * (List_YOUSO[34] * (List_YOUSO[35] + 1) + 2));
    VNA_List2 = (int*)malloc(sizeof(int) * (List_YOUSO[34] * (List_YOUSO[35] + 1) + 2));

    /* initialize the temporal array storing the force contribution */

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = F_M2G[Mc_AN];
        Gxyz[Gc_AN][41] = 0.0;
        Gxyz[Gc_AN][42] = 0.0;
        Gxyz[Gc_AN][43] = 0.0;
    }

    /*************************************************************
                   contraction of DS_VNA and HVNA2
    *************************************************************/

    if (Cnt_switch == 1 && ProExpn_VNA == 1) {

        Cont_Matrix2(DS_VNA[0], CntDS_VNA[0]);
        Cont_Matrix2(DS_VNA[1], CntDS_VNA[1]);
        Cont_Matrix2(DS_VNA[2], CntDS_VNA[2]);
        Cont_Matrix2(DS_VNA[3], CntDS_VNA[3]);

        Cont_Matrix3(HVNA2[1], CntHVNA2[1]);
        Cont_Matrix3(HVNA2[2], CntHVNA2[2]);
        Cont_Matrix3(HVNA2[3], CntHVNA2[3]);

        Cont_Matrix4(HVNA3[1], CntHVNA3[1]);
        Cont_Matrix4(HVNA3[2], CntHVNA3[2]);
        Cont_Matrix4(HVNA3[3], CntHVNA3[3]);
    }

    /*************************************************************
                    make VNA_List and VNA_List2
    *************************************************************/

    l = 0;
    for (i = 0; i <= List_YOUSO[35]; i++) { /* max L */
        for (j = 0; j < List_YOUSO[34]; j++) { /* # of radial projectors */
            VNA_List[l] = i;
            VNA_List2[l] = j;
            l++;
        }
    }

    Num_RVNA = List_YOUSO[34] * (List_YOUSO[35] + 1);
    tno2 = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];
    ActiveHVNA2 = (Cnt_switch == 0) ? HVNA2 : CntHVNA2;
    ActiveHVNA3 = (Cnt_switch == 0) ? HVNA3 : CntHVNA3;
    /*
     * The OpenACC Force4B path packs host-side dHVNA products into large
     * temporary arrays and copies them to the device for a scalar reduction.
     * In DC-LNO/GPUSOLVER runs many MPI ranks can share one GPU while GPUSOLVER
     * still owns most device memory, so these transient copies can exhaust
     * device memory.  Keep Force4B on the CPU trace path, which avoids those
     * device buffers and matches the original contraction.
     */
    const int use_force4b_openacc = 0;
    const int use_force4b_fused = Force_collective_env_flag(
        "OPENMX_FORCE4B_FUSED", 1, mpi_comm_level1);
    int f4b_gpu = 0;

    dtime(&etime);
    if (force_profile) {
        Force_profile_report("f4b-setup", etime - stime,
            mpi_comm_level1, myid, numprocs);
    }
    if (myid == 0 && measure_time) {
        printf("Time for part1 of force#4=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    /*****************************************************
     if orbital optimization
     copy CntDS_VNA[0] into DS_VNA[0]
    *****************************************************/

    if (Cnt_switch == 1) {

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

            Gc_AN = F_M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_CNO[Cwan];

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];

                for (i = 0; i < tno0; i++) {

                    l = 0;
                    for (l1 = 0; l1 < Num_RVNA; l1++) {

                        l2 = 2 * VNA_List[l1];
                        for (l3 = 0; l3 <= l2; l3++) {
                            DS_VNA[0][Mc_AN][h_AN][i][l] = CntDS_VNA[0][Mc_AN][h_AN][i][l];
                            l++;
                        }
                    }
                }
            }
        }
    }

    /*****************************************************
       (1) pre-multiplying DS_VNA[kk] with ene
       (2) copy DS_VNA[kk] or CntDS_VNA[kk] into DS_VNA[kk]
    *****************************************************/

    dtime(&stime);

#pragma omp parallel shared(CntDS_VNA, DS_VNA, Cnt_switch, VNA_proj_ene, VNA_List2, VNA_List, Num_RVNA, natn, FNAN, Spe_Total_CNO, WhatSpecies, F_M2G, Matomnum) private(kk, OMPID, Nthrds, Nprocs, Gc_AN, Cwan, tno0, Mc_AN, h_AN, Gh_AN, Hwan, i, l, l1, LL, Mul1, ene, l2, l3)
    {

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (kk = 1; kk <= 3; kk++) {
            for (Mc_AN = (OMPID * Matomnum / Nthrds + 1); Mc_AN < ((OMPID + 1) * Matomnum / Nthrds + 1); Mc_AN++) {

                Gc_AN = F_M2G[Mc_AN];
                Cwan = WhatSpecies[Gc_AN];
                tno0 = Spe_Total_CNO[Cwan];

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan = WhatSpecies[Gh_AN];

                    for (i = 0; i < tno0; i++) {

                        l = 0;
                        for (l1 = 0; l1 < Num_RVNA; l1++) {

                            LL = VNA_List[l1];
                            Mul1 = VNA_List2[l1];

                            ene = VNA_proj_ene[Hwan][LL][Mul1];
                            l2 = 2 * VNA_List[l1];

                            if (Cnt_switch == 0) {
                                for (l3 = 0; l3 <= l2; l3++) {
                                    DS_VNA[kk][Mc_AN][h_AN][i][l] = ene * DS_VNA[kk][Mc_AN][h_AN][i][l];
                                    l++;
                                }
                            }

                            else {
                                for (l3 = 0; l3 <= l2; l3++) {
                                    DS_VNA[kk][Mc_AN][h_AN][i][l] = ene * CntDS_VNA[kk][Mc_AN][h_AN][i][l];
                                    l++;
                                }
                            }
                        }
                    }

                } /* h_AN */
            } /* Mc_AN */
        } /* kk */

    } /* #pragma omp parallel */

    dtime(&etime);
    if (force_profile) {
        Force_profile_report("f4b-premul", etime - stime,
            mpi_comm_level1, myid, numprocs);
    }
    if (myid == 0 && measure_time) {
        printf("Time for part2 of force#4=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    /* Batched device path for the fused case-1/case-2 traces: the halo
       blocks and case-2 stagings are archived during the unchanged MPI
       loops, and one kernel per case runs afterwards. */

    if (use_force4b_fused) {
        f4b_gpu = Force4B_GpuBegin(DS_VNA);
    }

    /*****************************************}**********************
        THE FIRST CASE:
        In case of I=i or I=j
        for d [ \sum_k <i|k>ek<k|j> ]/dRI
    ****************************************************************/

    /*******************************************************
     *******************************************************
        multiplying overlap integrals WITH COMMUNICATION
     *******************************************************
     *******************************************************/

    MPI_Barrier(mpi_comm_level1);
    dtime(&stime);

    for (ID = 0; ID < numprocs; ID++) {
        F_Snd_Num_WK[ID] = 0;
        F_Rcv_Num_WK[ID] = 0;
    }

    do {

        /***********************************
                set the size of data
        ************************************/

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            /* find the data size to send the block data */

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS])) {

                size1 = 0;
                n = F_Snd_Num_WK[IDS];

                Mc_AN = Snd_MAN[IDS][n];
                Gc_AN = Snd_GAN[IDS][n];
                Cwan = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_NO[Cwan];
                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    tno2 = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];
                    size1 += tno1 * tno2;
                }

                Snd_DS_VNA_Size[IDS] = size1;
                MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
            } else {
                Snd_DS_VNA_Size[IDS] = 0;
            }

            /* receiving of the size of the data */

            if (0 < (F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR])) {
                MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                Rcv_DS_VNA_Size[IDR] = size2;
            } else {
                Rcv_DS_VNA_Size[IDR] = 0;
            }

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS]))
                MPI_Wait(&request, &stat);

        } /* ID */

        /***********************************
                  data transfer
        ************************************/

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            /******************************
                   sending of the data
            ******************************/

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS])) {

                size1 = Snd_DS_VNA_Size[IDS];

                /* allocation of the array */

                tmp_array = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA) * size1);

                /* multidimentional array to the vector array */

                num = 0;
                n = F_Snd_Num_WK[IDS];

                Mc_AN = Snd_MAN[IDS][n];
                Gc_AN = Snd_GAN[IDS][n];
                Cwan = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_NO[Cwan];
                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    tno2 = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];

                    for (i = 0; i < tno1; i++) {
                        for (j = 0; j < tno2; j++) {
                            tmp_array[num] = DS_VNA[0][Mc_AN][h_AN][i][j];
                            num++;
                        }
                    }
                }

                MPI_Isend(&tmp_array[0], size1, MPI_Type_DS_VNA, IDS, tag, mpi_comm_level1, &request);
            }

            /******************************
                receiving of the block data
            ******************************/

            if (0 < (F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR])) {

                size2 = Rcv_DS_VNA_Size[IDR];
                tmp_array2 = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA) * size2);
                MPI_Recv(&tmp_array2[0], size2, MPI_Type_DS_VNA, IDR, tag, mpi_comm_level1, &stat);

                /* store */

                num = 0;
                n = F_Rcv_Num_WK[IDR];
                Original_Mc_AN = F_TopMAN[IDR] + n;
                Gc_AN = Rcv_GAN[IDR][n];
                Cwan = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_NO[Cwan];

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    tno2 = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];

                    for (i = 0; i < tno1; i++) {
                        for (j = 0; j < tno2; j++) {
                            DS_VNA[0][Matomnum + 1][h_AN][i][j] = tmp_array2[num];
                            num++;
                        }
                    }
                }

                /* free tmp_array2 */
                free(tmp_array2);

                if (f4b_gpu) {
                    Force4B_GpuArchiveHalo(DS_VNA, Original_Mc_AN, Gc_AN);
                }

                /*****************************************
                       multiplying overlap integrals
                *****************************************/

                if (use_force4b_openacc) {
                    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
                        size_t atom_terms = 0;
                        size_t idx = 0;
                        double* weights = NULL;
                        double* hx = NULL;
                        double* hy = NULL;
                        double* hz = NULL;

                        dtime(&Stime_atom);

                        dEx = 0.0;
                        dEy = 0.0;
                        dEz = 0.0;

                        Gc_AN = M2G[Mc_AN];
                        Cwan = WhatSpecies[Gc_AN];
                        fan = FNAN[Gc_AN];

                        h_AN = 0;
                        Gh_AN = natn[Gc_AN][h_AN];
                        Mh_AN = F_G2M[Gh_AN];
                        Hwan = WhatSpecies[Gh_AN];
                        ian = Spe_Total_CNO[Hwan];

                        n = F_Rcv_Num_WK[IDR];
                        jg = Rcv_GAN[IDR][n];

                        for (j0 = 0; j0 <= fan; j0++) {
                            jg0 = natn[Gc_AN][j0];
                            Mj_AN0 = F_G2M[jg0];

                            if (Original_Mc_AN != Mj_AN0) {
                                continue;
                            }

                            q_AN = j0;
                            Gq_AN = natn[Gc_AN][q_AN];
                            Qwan = WhatSpecies[Gq_AN];
                            jan = Spe_Total_CNO[Qwan];
                            kl = RMI1[Mc_AN][h_AN][q_AN];

                            if (0 <= kl) {
                                atom_terms = Force_checked_add_count(atom_terms,
                                    Force_checked_mul_count((size_t)ian, (size_t)jan, "Force4B comm atom block"),
                                    "Force4B comm atom term count");
                            }
                        }

                        if (atom_terms != 0) {
                            weights = (double*)malloc(sizeof(double) * atom_terms);
                            hx = (double*)malloc(sizeof(double) * atom_terms);
                            hy = (double*)malloc(sizeof(double) * atom_terms);
                            hz = (double*)malloc(sizeof(double) * atom_terms);

                            HVNAx = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
                            HVNAy = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
                            HVNAz = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);

                            for (j0 = 0; j0 <= fan; j0++) {
                                jg0 = natn[Gc_AN][j0];
                                Mj_AN0 = F_G2M[jg0];

                                if (Original_Mc_AN != Mj_AN0) {
                                    continue;
                                }

                                q_AN = j0;
                                Gq_AN = natn[Gc_AN][q_AN];
                                Qwan = WhatSpecies[Gq_AN];
                                jan = Spe_Total_CNO[Qwan];
                                kl = RMI1[Mc_AN][h_AN][q_AN];

                                if (kl < 0) {
                                    continue;
                                }

                                dHVNA(0, Mc_AN, h_AN, q_AN, DS_VNA, ActiveHVNA2, ActiveHVNA3, HVNAx, HVNAy, HVNAz);

                                pref = (SpinP_switch == 0)
                                    ? ((q_AN == h_AN) ? 2.0 : 4.0)
                                    : ((q_AN == h_AN) ? 1.0 : 2.0);

                                idx = Force4B_pack_trace_terms(CDM0, Mh_AN, kl, ian, jan, pref,
                                    HVNAx, HVNAy, HVNAz, idx, weights, hx, hy, hz);
                            }

                            Force_openacc_reduce_terms(atom_terms, weights, hx, hy, hz, &dEx, &dEy, &dEz);

                            Force_free_double_matrix(HVNAx);
                            Force_free_double_matrix(HVNAy);
                            Force_free_double_matrix(HVNAz);
                            free(hz);
                            free(hy);
                            free(hx);
                            free(weights);
                        }

                        Gxyz[Gc_AN][41] += dEx;
                        Gxyz[Gc_AN][42] += dEy;
                        Gxyz[Gc_AN][43] += dEz;

                        dtime(&Etime_atom);
                        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
                    }
                } else {
#pragma omp parallel shared(use_force4b_fused, List_YOUSO, time_per_atom, Gxyz, CDM0, SpinP_switch, DS_VNA, RMI1, Original_Mc_AN, IDR, Rcv_GAN, F_Rcv_Num_WK, Spe_Total_CNO, F_G2M, natn, FNAN, WhatSpecies, M2G, Matomnum, ActiveHVNA2, ActiveHVNA3) private(Stime_atom, Etime_atom, dEx, dEy, dEz, Gc_AN, Mc_AN, Cwan, fan, h_AN, Gh_AN, Mh_AN, Hwan, ian, n, jg, j0, jg0, Mj_AN0, q_AN, Gq_AN, Mq_AN, Qwan, jan, kl, HVNAx, HVNAy, HVNAz, i, j, pref)
                    {

                        /* allocation of array */

                        HVNAx = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
                        HVNAy = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
                        HVNAz = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);

#pragma omp for schedule(static)
                        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                            dtime(&Stime_atom);

                            dEx = 0.0;
                            dEy = 0.0;
                            dEz = 0.0;

                            Gc_AN = M2G[Mc_AN];
                            Cwan = WhatSpecies[Gc_AN];
                            fan = FNAN[Gc_AN];

                            h_AN = 0;
                            Gh_AN = natn[Gc_AN][h_AN];
                            Mh_AN = F_G2M[Gh_AN];
                            Hwan = WhatSpecies[Gh_AN];
                            ian = Spe_Total_CNO[Hwan];

                            n = F_Rcv_Num_WK[IDR];
                            jg = Rcv_GAN[IDR][n];

                            for (j0 = 0; j0 <= fan; j0++) {

                                jg0 = natn[Gc_AN][j0];
                                Mj_AN0 = F_G2M[jg0];

                                if (Original_Mc_AN != Mj_AN0) {
                                    continue;
                                }

                                q_AN = j0;
                                Gq_AN = natn[Gc_AN][q_AN];
                                Mq_AN = F_G2M[Gq_AN];
                                Qwan = WhatSpecies[Gq_AN];
                                jan = Spe_Total_CNO[Qwan];
                                kl = RMI1[Mc_AN][h_AN][q_AN];

                                if (use_force4b_fused && q_AN != 0) {
                                    if (!f4b_gpu) {
                                        Force4B_case1_trace_fused(Mc_AN, Gc_AN, q_AN,
                                            CDM0, DS_VNA, &dEx, &dEy, &dEz);
                                    }
                                    continue;
                                }

                                dHVNA(0, Mc_AN, h_AN, q_AN, DS_VNA, ActiveHVNA2, ActiveHVNA3, HVNAx, HVNAy, HVNAz);

                                /* contribution of force = Trace(CDM0*dH) */

                                if (SpinP_switch == 0) {

                                    pref = (q_AN == h_AN) ? 2.0 : 4.0;

                                    for (i = 0; i < ian; i++) {
                                        double* cdm0_row = CDM0[0][Mh_AN][kl][i];
                                        double* hvnax_row = HVNAx[i];
                                        double* hvnay_row = HVNAy[i];
                                        double* hvnaz_row = HVNAz[i];

                                        for (j = 0; j < jan; j++) {
                                            double cdm = pref * cdm0_row[j];
                                            dEx += cdm * hvnax_row[j];
                                            dEy += cdm * hvnay_row[j];
                                            dEz += cdm * hvnaz_row[j];
                                        }
                                    }
                                }

                                else {

                                    pref = (q_AN == h_AN) ? 1.0 : 2.0;

                                    for (i = 0; i < ian; i++) {
                                        double* cdm0_row = CDM0[0][Mh_AN][kl][i];
                                        double* cdm1_row = CDM0[1][Mh_AN][kl][i];
                                        double* hvnax_row = HVNAx[i];
                                        double* hvnay_row = HVNAy[i];
                                        double* hvnaz_row = HVNAz[i];

                                        for (j = 0; j < jan; j++) {
                                            double cdm = pref * (cdm0_row[j] + cdm1_row[j]);
                                            dEx += cdm * hvnax_row[j];
                                            dEy += cdm * hvnay_row[j];
                                            dEz += cdm * hvnaz_row[j];
                                        }
                                    }
                                }
                            } /* j0 */

                            /* force from #4B */

                            Gxyz[Gc_AN][41] += dEx;
                            Gxyz[Gc_AN][42] += dEy;
                            Gxyz[Gc_AN][43] += dEz;

                            /* timing */
                            dtime(&Etime_atom);
                            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

                        } /* Mc_AN */

                        /* freeing of array */

                        Force_free_double_matrix(HVNAx);
                        Force_free_double_matrix(HVNAy);
                        Force_free_double_matrix(HVNAz);

                    } /* #pragma omp parallel */
                }

                /********************************************
                  increment of F_Rcv_Num_WK[IDR]
                ********************************************/

                F_Rcv_Num_WK[IDR]++;

            } /* if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) */

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS])) {

                MPI_Wait(&request, &stat);
                free(tmp_array); /* freeing of array */

                /********************************************
                     increment of F_Snd_Num_WK[IDS]
                ********************************************/

                F_Snd_Num_WK[IDS]++;
            }

        } /* ID */

        /*****************************************************
          check whether all the communications have finished
        *****************************************************/

        po = 0;
        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS]))
                po += F_Snd_Num[IDS] - F_Snd_Num_WK[IDS];
            if (0 < (F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR]))
                po += F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR];
        }

    } while (po != 0);

    dtime(&etime);
    if (force_profile) {
        Force_profile_report("f4b-case1-r", etime - stime,
            mpi_comm_level1, myid, numprocs);
    }
    if (myid == 0 && measure_time) {
        printf("Time for part3 of force#4=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    /*******************************************************
     *******************************************************
        THE FIRST CASE:
        multiplying overlap integrals WITHOUT COMMUNICATION
     *******************************************************
     *******************************************************/

    dtime(&stime);

    if (use_force4b_openacc) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            size_t atom_terms = 0;
            size_t idx = 0;
            double* weights = NULL;
            double* hx = NULL;
            double* hy = NULL;
            double* hz = NULL;

            dtime(&Stime_atom);

            dEx = 0.0;
            dEy = 0.0;
            dEz = 0.0;

            Gc_AN = M2G[Mc_AN];
            h_AN = 0;
            Gh_AN = natn[Gc_AN][h_AN];
            Mh_AN = F_G2M[Gh_AN];
            Hwan = WhatSpecies[Gh_AN];
            ian = Spe_Total_CNO[Hwan];

            for (q_AN = h_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {
                Gq_AN = natn[Gc_AN][q_AN];
                Mq_AN = F_G2M[Gq_AN];

                if (Mq_AN <= Matomnum) {
                    Qwan = WhatSpecies[Gq_AN];
                    jan = Spe_Total_CNO[Qwan];
                    kl = RMI1[Mc_AN][h_AN][q_AN];

                    if (0 <= kl) {
                        atom_terms = Force_checked_add_count(atom_terms,
                            Force_checked_mul_count((size_t)ian, (size_t)jan, "Force4B local atom block"),
                            "Force4B local atom term count");
                    }
                }
            }

            if (atom_terms != 0) {
                weights = (double*)malloc(sizeof(double) * atom_terms);
                hx = (double*)malloc(sizeof(double) * atom_terms);
                hy = (double*)malloc(sizeof(double) * atom_terms);
                hz = (double*)malloc(sizeof(double) * atom_terms);

                HVNAx = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
                HVNAy = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
                HVNAz = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);

                for (q_AN = h_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {
                    Gq_AN = natn[Gc_AN][q_AN];
                    Mq_AN = F_G2M[Gq_AN];

                    if (Mq_AN <= Matomnum) {
                        Qwan = WhatSpecies[Gq_AN];
                        jan = Spe_Total_CNO[Qwan];
                        kl = RMI1[Mc_AN][h_AN][q_AN];

                        if (kl < 0) {
                            continue;
                        }

                        dHVNA(0, Mc_AN, h_AN, q_AN, DS_VNA, ActiveHVNA2, ActiveHVNA3, HVNAx, HVNAy, HVNAz);

                        pref = (SpinP_switch == 0)
                            ? ((q_AN == h_AN) ? 2.0 : 4.0)
                            : ((q_AN == h_AN) ? 1.0 : 2.0);

                        idx = Force4B_pack_trace_terms(CDM0, Mh_AN, kl, ian, jan, pref,
                            HVNAx, HVNAy, HVNAz, idx, weights, hx, hy, hz);
                    }
                }

                Force_openacc_reduce_terms(atom_terms, weights, hx, hy, hz, &dEx, &dEy, &dEz);

                Force_free_double_matrix(HVNAx);
                Force_free_double_matrix(HVNAy);
                Force_free_double_matrix(HVNAz);
                free(hz);
                free(hy);
                free(hx);
                free(weights);
            }

            Gxyz[Gc_AN][41] += dEx;
            Gxyz[Gc_AN][42] += dEy;
            Gxyz[Gc_AN][43] += dEz;

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

        } /* Mc_AN */
    } else {
#pragma omp parallel shared(use_force4b_fused, time_per_atom, Gxyz, CDM0, SpinP_switch, DS_VNA, RMI1, FNAN, Spe_Total_CNO, WhatSpecies, F_G2M, natn, M2G, Matomnum, List_YOUSO, ActiveHVNA2, ActiveHVNA3) private(HVNAx, HVNAy, HVNAz, Mc_AN, Stime_atom, Etime_atom, dEx, dEy, dEz, Gc_AN, h_AN, Gh_AN, Mh_AN, Hwan, ian, q_AN, Gq_AN, Mq_AN, Qwan, jan, kl, i, j, pref)
        {

            /* allocation of array */

            HVNAx = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
            HVNAy = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
            HVNAz = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);

#pragma omp for schedule(static)
            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                dtime(&Stime_atom);

                dEx = 0.0;
                dEy = 0.0;
                dEz = 0.0;

                Gc_AN = M2G[Mc_AN];
                h_AN = 0;
                Gh_AN = natn[Gc_AN][h_AN];
                Mh_AN = F_G2M[Gh_AN];
                Hwan = WhatSpecies[Gh_AN];
                ian = Spe_Total_CNO[Hwan];

                for (q_AN = h_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {

                    Gq_AN = natn[Gc_AN][q_AN];
                    Mq_AN = F_G2M[Gq_AN];

                    if (Mq_AN <= Matomnum) {

                        Qwan = WhatSpecies[Gq_AN];
                        jan = Spe_Total_CNO[Qwan];
                        kl = RMI1[Mc_AN][h_AN][q_AN];

                        if (use_force4b_fused && q_AN != 0) {
                            if (!f4b_gpu) {
                                Force4B_case1_trace_fused(Mc_AN, Gc_AN, q_AN,
                                    CDM0, DS_VNA, &dEx, &dEy, &dEz);
                            }
                            continue;
                        }

                        dHVNA(0, Mc_AN, h_AN, q_AN, DS_VNA, ActiveHVNA2, ActiveHVNA3, HVNAx, HVNAy, HVNAz);

                        if (SpinP_switch == 0) {

                            pref = (q_AN == h_AN) ? 2.0 : 4.0;

                            for (i = 0; i < ian; i++) {
                                double* cdm0_row = CDM0[0][Mh_AN][kl][i];
                                double* hvnax_row = HVNAx[i];
                                double* hvnay_row = HVNAy[i];
                                double* hvnaz_row = HVNAz[i];

                                for (j = 0; j < jan; j++) {
                                    double cdm = pref * cdm0_row[j];
                                    dEx += cdm * hvnax_row[j];
                                    dEy += cdm * hvnay_row[j];
                                    dEz += cdm * hvnaz_row[j];
                                }
                            }
                        }

                        /* else */

                        else {

                            pref = (q_AN == h_AN) ? 1.0 : 2.0;

                            for (i = 0; i < ian; i++) {
                                double* cdm0_row = CDM0[0][Mh_AN][kl][i];
                                double* cdm1_row = CDM0[1][Mh_AN][kl][i];
                                double* hvnax_row = HVNAx[i];
                                double* hvnay_row = HVNAy[i];
                                double* hvnaz_row = HVNAz[i];

                                for (j = 0; j < jan; j++) {
                                    double cdm = pref * (cdm0_row[j] + cdm1_row[j]);
                                    dEx += cdm * hvnax_row[j];
                                    dEy += cdm * hvnay_row[j];
                                    dEz += cdm * hvnaz_row[j];
                                }
                            }
                        }
                    }
                }

                /* force from #4B */

                Gxyz[Gc_AN][41] += dEx;
                Gxyz[Gc_AN][42] += dEy;
                Gxyz[Gc_AN][43] += dEz;

                /* timing */
                dtime(&Etime_atom);
                time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

            } /* Mc_AN */

            /* freeing of array */

            Force_free_double_matrix(HVNAx);
            Force_free_double_matrix(HVNAy);
            Force_free_double_matrix(HVNAz);

        } /* #pragma omp parallel */
    }

    /* batched device evaluation of every skipped fused case-1 trace */
    if (f4b_gpu) {
        Force4B_GpuCase1Run(CDM0);
    }

    dtime(&etime);
    if (force_profile) {
        Force_profile_report("f4b-case1-l", etime - stime,
            mpi_comm_level1, myid, numprocs);
    }
    if (myid == 0 && measure_time) {
        printf("Time for part4 of force#4=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    /*************************************************************
       THE SECOND CASE:
       In case of I=k with I!=i and I!=j
       d [ \sum_k <i|k>ek<k|j> ]/dRI
    *************************************************************/

    /************************************************************
       MPI communication of DS_VNA whose basis part is not located
       on own site but projector part is located on own site.
    ************************************************************/

    MPI_Barrier(mpi_comm_level1);
    dtime(&stime);

    for (ID = 0; ID < numprocs; ID++)
        Indicator[ID] = 0;

    max_ODNloop = 1;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        i = (FNAN[Gc_AN] + 1) * (FNAN[Gc_AN] + 2);
        if (max_ODNloop < i) {
            max_ODNloop = i;
        }
    }

    OneD2h_AN = (int*)malloc(sizeof(int) * max_ODNloop);
    OneD2q_AN = (int*)malloc(sizeof(int) * max_ODNloop);

    for (Mc_AN = 1; Mc_AN <= Max_Matomnum; Mc_AN++) {

        dtime(&Stime_atom);

        dtime(&stime1);

        if (Mc_AN <= Matomnum)
            Gc_AN = M2G[Mc_AN];
        else
            Gc_AN = 0;

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            i = Indicator[IDS];
            po = 0;

            Gh_AN = Pro_Snd_GAtom[IDS][i];

            if (Gh_AN != 0) {

                /* find the range with the same global atomic number */

                do {

                    i++;
                    if (Gh_AN != Pro_Snd_GAtom[IDS][i])
                        po = 1;
                } while (po == 0);

                i--;
                SA_num = i - Indicator[IDS] + 1;

                /* find the data size to send the block data */

                size1 = 0;
                for (q = Indicator[IDS]; q <= (Indicator[IDS] + SA_num - 1); q++) {

                    Sc_AN = Pro_Snd_MAtom[IDS][q];
                    GSc_AN = F_M2G[Sc_AN];
                    Sc_wan = WhatSpecies[GSc_AN];
                    tno1 = Spe_Total_CNO[Sc_wan];
                    tno2 = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];
                    size1 += 4 * tno1 * tno2;
                    size1 += 3;
                }

            } /* if (Gh_AN!=0) */

            else {
                SA_num = 0;
                size1 = 0;
            }

            S_array[IDS][0] = Gh_AN;
            S_array[IDS][1] = SA_num;
            S_array[IDS][2] = size1;

            if (ID != 0) {
                MPI_Isend(&S_array[IDS][0], 3, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                MPI_Recv(&R_array[IDR][0], 3, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                MPI_Wait(&request, &stat);
            } else {
                R_array[myid][0] = S_array[myid][0];
                R_array[myid][1] = S_array[myid][1];
                R_array[myid][2] = S_array[myid][2];
            }

            if (R_array[IDR][0] == Gc_AN)
                R_comm_flag = 1;
            else
                R_comm_flag = 0;

            if (ID != 0) {
                MPI_Isend(&R_comm_flag, 1, MPI_INT, IDR, tag, mpi_comm_level1, &request);
                MPI_Recv(&S_comm_flag, 1, MPI_INT, IDS, tag, mpi_comm_level1, &stat);
                MPI_Wait(&request, &stat);
            } else {
                S_comm_flag = R_comm_flag;
            }

            /*
        if (counter==2 && ID==8){
          printf("QQQ4 myid=%2d\n",myid);fflush(stdout);
          MPI_Finalize();
          exit(0);
        }
            */

            /*****************************************
                             send the data
            *****************************************/

            /* if (S_comm_flag==1) then, send data to IDS */

            if (S_comm_flag == 1) {

                /* allocate tmp_array */

                tmp_array = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA) * size1);

                /* multidimentional array to vector array */

                num = 0;

                for (q = Indicator[IDS]; q <= (Indicator[IDS] + SA_num - 1); q++) {

                    Sc_AN = Pro_Snd_MAtom[IDS][q];
                    GSc_AN = F_M2G[Sc_AN];
                    Sc_wan = WhatSpecies[GSc_AN];
                    tno1 = Spe_Total_CNO[Sc_wan];

                    Sh_AN = Pro_Snd_LAtom[IDS][q];
                    GSh_AN = natn[GSc_AN][Sh_AN];
                    Sh_wan = WhatSpecies[GSh_AN];
                    tno2 = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];

                    Sh_AN2 = Pro_Snd_LAtom2[IDS][q];

                    tmp_array[num] = (Type_DS_VNA)Sc_AN;
                    num++;
                    tmp_array[num] = (Type_DS_VNA)Sh_AN;
                    num++;
                    tmp_array[num] = (Type_DS_VNA)Sh_AN2;
                    num++;

                    for (kk = 0; kk <= 3; kk++) {
                        for (i = 0; i < tno1; i++) {
                            for (j = 0; j < tno2; j++) {
                                tmp_array[num] = DS_VNA[kk][Sc_AN][Sh_AN][i][j];
                                num++;
                            }
                        }
                    }
                }

                if (ID != 0) {
                    MPI_Isend(&tmp_array[0], size1, MPI_Type_DS_VNA, IDS, tag, mpi_comm_level1, &request);
                }

                /* update Indicator[IDS] */

                Indicator[IDS] += SA_num;

            } /* if (S_comm_flag==1) */

            /*****************************************
                         receive the data
            *****************************************/

            /* if (R_comm_flag==1) then, receive the data from IDR */

            if (R_comm_flag == 1) {

                size2 = R_array[IDR][2];
                tmp_array2 = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA) * size2);

                if (ID != 0) {
                    MPI_Recv(&tmp_array2[0], size2, MPI_Type_DS_VNA, IDR, tag, mpi_comm_level1, &stat);
                } else {
                    for (i = 0; i < size2; i++)
                        tmp_array2[i] = tmp_array[i];
                }

                /* store */

                num = 0;

                for (n = 0; n < R_array[IDR][1]; n++) {

                    Sc_AN = (int)tmp_array2[num];
                    num++;
                    Sh_AN = (int)tmp_array2[num];
                    num++;
                    Sh_AN2 = (int)tmp_array2[num];
                    num++;

                    GSc_AN = natn[Gc_AN][Sh_AN2];
                    Sc_wan = WhatSpecies[GSc_AN];

                    tno1 = Spe_Total_CNO[Sc_wan];
                    tno2 = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];

                    for (kk = 0; kk <= 3; kk++) {
                        for (i = 0; i < tno1; i++) {
                            for (j = 0; j < tno2; j++) {
                                DS_VNA[kk][Matomnum + 1][Sh_AN2][i][j] = tmp_array2[num];
                                num++;
                            }
                        }
                    }
                }

                /* free tmp_array2 */
                free(tmp_array2);

            } /* if (R_comm_flag==1) */

            if (S_comm_flag == 1) {
                if (ID != 0)
                    MPI_Wait(&request, &stat);
                free(tmp_array); /* freeing of array */
            }

        } /* ID */

        dtime(&etime1);
        if (myid == 0 && measure_time) {
            printf("Time for part5A of force#4=%18.5f\n", etime1 - stime1);
            fflush(stdout);
        }

        dtime(&stime1);

        if (Mc_AN <= Matomnum) {

            if (f4b_gpu) {
                Force4B_GpuArchiveCase2(DS_VNA, Mc_AN, Gc_AN);
            }

            /* one-dimensionalize the h_AN and q_AN loops */

            ODNloop = 0;
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Solver == 5 || Solver == 8 || Solver == 11)
                    start_q_AN = 0;
                else
                    start_q_AN = h_AN;

                for (q_AN = start_q_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {

                    kl = RMI1[Mc_AN][h_AN][q_AN];

                    if (0 <= kl) {
                        OneD2h_AN[ODNloop] = h_AN;
                        OneD2q_AN[ODNloop] = q_AN;
                        ODNloop++;
                    }
                }
            }

            dEx = 0.0;
            dEy = 0.0;
            dEz = 0.0;

            if (use_force4b_openacc) {
                size_t atom_terms = 0;
                size_t idx = 0;
                double* weights = NULL;
                double* hx = NULL;
                double* hy = NULL;
                double* hz = NULL;

                for (Nloop = 0; Nloop < ODNloop; Nloop++) {
                    h_AN = OneD2h_AN[Nloop];
                    q_AN = OneD2q_AN[Nloop];
                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    ian = Spe_Total_CNO[Hwan];
                    Gq_AN = natn[Gc_AN][q_AN];
                    Qwan = WhatSpecies[Gq_AN];
                    jan = Spe_Total_CNO[Qwan];
                    kl = RMI1[Mc_AN][h_AN][q_AN];

                    if (0 <= kl) {
                        atom_terms = Force_checked_add_count(atom_terms,
                            Force_checked_mul_count((size_t)ian, (size_t)jan, "Force4B ODN atom block"),
                            "Force4B ODN atom term count");
                    }
                }

                if (atom_terms != 0) {
                    weights = (double*)malloc(sizeof(double) * atom_terms);
                    hx = (double*)malloc(sizeof(double) * atom_terms);
                    hy = (double*)malloc(sizeof(double) * atom_terms);
                    hz = (double*)malloc(sizeof(double) * atom_terms);

                    HVNAx = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
                    HVNAy = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
                    HVNAz = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);

                    for (Nloop = 0; Nloop < ODNloop; Nloop++) {
                        h_AN = OneD2h_AN[Nloop];
                        q_AN = OneD2q_AN[Nloop];
                        Gh_AN = natn[Gc_AN][h_AN];
                        Mh_AN = F_G2M[Gh_AN];
                        Hwan = WhatSpecies[Gh_AN];
                        ian = Spe_Total_CNO[Hwan];
                        Gq_AN = natn[Gc_AN][q_AN];
                        Qwan = WhatSpecies[Gq_AN];
                        jan = Spe_Total_CNO[Qwan];
                        kl = RMI1[Mc_AN][h_AN][q_AN];

                        if (kl < 0) {
                            continue;
                        }

                        dHVNA(1, Mc_AN, h_AN, q_AN, DS_VNA, ActiveHVNA2, ActiveHVNA3, HVNAx, HVNAy, HVNAz);

                        pref = ((Solver == 5 || Solver == 8 || Solver == 11) || q_AN == h_AN)
                            ? ((SpinP_switch == 0) ? 2.0 : 1.0)
                            : ((SpinP_switch == 0) ? 4.0 : 2.0);

                        idx = Force4B_pack_trace_terms(CDM0, Mh_AN, kl, ian, jan, pref,
                            HVNAx, HVNAy, HVNAz, idx, weights, hx, hy, hz);
                    }

                    Force_openacc_reduce_terms(atom_terms, weights, hx, hy, hz, &dEx, &dEy, &dEz);

                    Force_free_double_matrix(HVNAx);
                    Force_free_double_matrix(HVNAy);
                    Force_free_double_matrix(HVNAz);
                    free(hz);
                    free(hy);
                    free(hx);
                    free(weights);
                }
            } else {
#if NVHPC_VERSION >= 250000
                #pragma omp parallel shared(use_force4b_fused,ODNloop,OneD2h_AN,OneD2q_AN,Mc_AN,Gc_AN,CDM0,SpinP_switch,DS_VNA,RMI1,Spe_Total_CNO,WhatSpecies,F_G2M,natn,FNAN,List_YOUSO,Solver,ActiveHVNA2,ActiveHVNA3) private(HVNAx,HVNAy,HVNAz,i,j,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,Nloop,pref) reduction(+:dEx,dEy,dEz)
#endif
                {

                    /* allocation of arrays */

                    HVNAx = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
                    HVNAy = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);
                    HVNAz = Force_alloc_double_matrix(List_YOUSO[7], List_YOUSO[7]);

#pragma omp for schedule(static)
                    for (Nloop = 0; Nloop < ODNloop; Nloop++) {

                        /* get h_AN and q_AN */

                        h_AN = OneD2h_AN[Nloop];
                        q_AN = OneD2q_AN[Nloop];

                        /* set informations on h_AN */

                        Gh_AN = natn[Gc_AN][h_AN];
                        Mh_AN = F_G2M[Gh_AN];
                        Hwan = WhatSpecies[Gh_AN];
                        ian = Spe_Total_CNO[Hwan];

                        /* set informations on q_AN */

                        Gq_AN = natn[Gc_AN][q_AN];
                        Mq_AN = F_G2M[Gq_AN];
                        Qwan = WhatSpecies[Gq_AN];
                        jan = Spe_Total_CNO[Qwan];
                        kl = RMI1[Mc_AN][h_AN][q_AN];

                        if (0 <= kl) {

                            if (use_force4b_fused && h_AN != 0 && q_AN != 0 && h_AN != q_AN) {
                                if (!f4b_gpu) {
                                    Force4B_case2_trace_fused(Mc_AN, Gc_AN, h_AN, q_AN,
                                        CDM0, DS_VNA, &dEx, &dEy, &dEz);
                                }
                                continue;
                            }

                            dHVNA(1, Mc_AN, h_AN, q_AN, DS_VNA, ActiveHVNA2, ActiveHVNA3, HVNAx, HVNAy, HVNAz);

                            /* contribution of force = Trace(CDM0*dH) */

                            /* spin non-polarization */

                            if (SpinP_switch == 0) {

                                pref = ((Solver == 5 || Solver == 8 || Solver == 11) || q_AN == h_AN) ? 2.0 : 4.0;

                                for (i = 0; i < ian; i++) {
                                    double* cdm0_row = CDM0[0][Mh_AN][kl][i];
                                    double* hvnax_row = HVNAx[i];
                                    double* hvnay_row = HVNAy[i];
                                    double* hvnaz_row = HVNAz[i];

                                    for (j = 0; j < jan; j++) {
                                        double cdm = pref * cdm0_row[j];
                                        dEx += cdm * hvnax_row[j];
                                        dEy += cdm * hvnay_row[j];
                                        dEz += cdm * hvnaz_row[j];
                                    }
                                }
                            }

                            /* else */

                            else {

                                pref = ((Solver == 5 || Solver == 8 || Solver == 11) || q_AN == h_AN) ? 1.0 : 2.0;

                                for (i = 0; i < ian; i++) {
                                    double* cdm0_row = CDM0[0][Mh_AN][kl][i];
                                    double* cdm1_row = CDM0[1][Mh_AN][kl][i];
                                    double* hvnax_row = HVNAx[i];
                                    double* hvnay_row = HVNAy[i];
                                    double* hvnaz_row = HVNAz[i];

                                    for (j = 0; j < jan; j++) {
                                        double cdm = pref * (cdm0_row[j] + cdm1_row[j]);
                                        dEx += cdm * hvnax_row[j];
                                        dEy += cdm * hvnay_row[j];
                                        dEz += cdm * hvnaz_row[j];
                                    }
                                }
                            }

                        } /* if (0<=kl) */

                    } /* Nloop */

                    /* freeing of arrays */

                    Force_free_double_matrix(HVNAx);
                    Force_free_double_matrix(HVNAy);
                    Force_free_double_matrix(HVNAz);

                } /* #pragma omp parallel */
            }

            /* force from #4B */

            Gxyz[Gc_AN][41] += dEx;
            Gxyz[Gc_AN][42] += dEy;
            Gxyz[Gc_AN][43] += dEz;

            /* timing */
            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

        } /* if (Mc_AN<=Matomnum) */

        dtime(&etime1);
        if (myid == 0 && measure_time) {
            printf("Time for part5B of force#4=%18.5f\n", etime1 - stime1);
            fflush(stdout);
        }

    } /* Mc_AN */

    /* batched device evaluation of every skipped fused case-2 trace */
    if (f4b_gpu) {
        Force4B_GpuCase2Run(CDM0);
        Force4B_GpuEnd();
    }

    if (2 <= level_stdout) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            Gc_AN = M2G[Mc_AN];
            printf("<Force>  force(4B) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                myid, Mc_AN, Gc_AN, Gxyz[Gc_AN][41], Gxyz[Gc_AN][42], Gxyz[Gc_AN][43]);
            fflush(stdout);
        }
    }

    free(OneD2q_AN);
    free(OneD2h_AN);

    dtime(&etime);
    if (force_profile) {
        Force_profile_report("f4b-case2", etime - stime,
            mpi_comm_level1, myid, numprocs);
    }
    if (myid == 0 && measure_time) {
        printf("Time for part5 of force#4=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];

        if (2 <= level_stdout) {
            printf("<Force>  force(4B) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                myid, Mc_AN, Gc_AN, Gxyz[Gc_AN][41], Gxyz[Gc_AN][42], Gxyz[Gc_AN][43]);
            fflush(stdout);
        }

        Gxyz[Gc_AN][17] += Gxyz[Gc_AN][41];
        Gxyz[Gc_AN][18] += Gxyz[Gc_AN][42];
        Gxyz[Gc_AN][19] += Gxyz[Gc_AN][43];
    }

    /***********************************
              freeing of arrays
    ************************************/

    free(Indicator);
    Force_free_int_matrix(S_array);
    Force_free_int_matrix(R_array);

    free(Snd_DS_VNA_Size);
    free(Rcv_DS_VNA_Size);

    free(VNA_List);
    free(VNA_List2);

}

void dHNL(int where_flag,
    int Mc_AN, int h_AN, int q_AN,
    double****** DS_NL1,
    dcomplex*** Hx, dcomplex*** Hy, dcomplex*** Hz)
{
    int i, j, k, m, n, l, kg, kan, so, deri_kind;
    int ig, ian, jg, jan, kl, kl1, kl2;
    int wakg, l1, l2, l3, Gc_AN, Mi_AN, Mi_AN2, Mj_AN, Mj_AN2;
    int Rni, Rnj, somax;
    double PF[2], sumx, sumy, sumz, ene, dmp, deri_dmp;
    double tmpx, tmpy, tmpz, tmp, r;
    double x0, y0, z0, x1, y1, z1, dx, dy, dz;
    double rcuti, rcutj, rcut;
    double PFp, PFm, ene_p, ene_m;
    dcomplex sumx0, sumy0, sumz0;
    dcomplex sumx1, sumy1, sumz1;
    dcomplex sumx2, sumy2, sumz2;

    /****************************************************
     start calc.
    ****************************************************/

    Gc_AN = M2G[Mc_AN];
    ig = natn[Gc_AN][h_AN];
    Rni = ncn[Gc_AN][h_AN];
    Mi_AN = F_G2M[ig];
    ian = Spe_Total_CNO[WhatSpecies[ig]];
    rcuti = Spe_Atom_Cut1[WhatSpecies[ig]];

    jg = natn[Gc_AN][q_AN];
    Rnj = ncn[Gc_AN][q_AN];
    Mj_AN = F_G2M[jg];
    jan = Spe_Total_CNO[WhatSpecies[jg]];
    rcutj = Spe_Atom_Cut1[WhatSpecies[jg]];

    rcut = rcuti + rcutj;
    kl = RMI1[Mc_AN][h_AN][q_AN];
    dmp = dampingF(rcut, Dis[ig][kl]);

    for (so = 0; so < 3; so++) {
        for (i = 0; i < List_YOUSO[7]; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hx[so][i][j] = Complex(0.0, 0.0);
                Hy[so][i][j] = Complex(0.0, 0.0);
                Hz[so][i][j] = Complex(0.0, 0.0);
            }
        }
    }

    if (h_AN == 0) {

        /****************************************************
                              dH*ep*H
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            wakg = WhatSpecies[kg];
            kan = Spe_Total_VPS_Pro[wakg];
            kl = RMI1[Mc_AN][q_AN][k];

            /****************************************************
                         l-dependent non-local part
            ****************************************************/

            if (0 <= kl && VPS_j_dependency[wakg] == 0 && where_flag == 0) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        l = 0;
                        for (l1 = 1; l1 <= Spe_Num_RVPS[wakg]; l1++) {

                            ene = Spe_VNLE[0][wakg][l1 - 1];
                            if (Spe_VPS_List[wakg][l1] == 0)
                                l2 = 0;
                            else if (Spe_VPS_List[wakg][l1] == 1)
                                l2 = 2;
                            else if (Spe_VPS_List[wakg][l1] == 2)
                                l2 = 4;
                            else if (Spe_VPS_List[wakg][l1] == 3)
                                l2 = 6;

                            if (Mj_AN <= Matomnum)
                                Mj_AN2 = Mj_AN;
                            else
                                Mj_AN2 = Matomnum + 1;

                            for (l3 = 0; l3 <= l2; l3++) {
                                sumx += ene * DS_NL1[0][1][Mc_AN][k][m][l] * DS_NL1[0][0][Mj_AN2][kl][n][l];
                                sumy += ene * DS_NL1[0][2][Mc_AN][k][m][l] * DS_NL1[0][0][Mj_AN2][kl][n][l];
                                sumz += ene * DS_NL1[0][3][Mc_AN][k][m][l] * DS_NL1[0][0][Mj_AN2][kl][n][l];
                                l++;
                            }
                        }

                        Hx[0][m][n].r += sumx;
                        Hy[0][m][n].r += sumy;
                        Hz[0][m][n].r += sumz;

                        Hx[1][m][n].r += sumx;
                        Hy[1][m][n].r += sumy;
                        Hz[1][m][n].r += sumz;

                    } /* n */
                } /* m */

            } /* if */

            /****************************************************
                         j-dependent non-local part
            ****************************************************/

            else if (0 <= kl && VPS_j_dependency[wakg] == 1 && where_flag == 0) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);

                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);

                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        if (Mj_AN <= Matomnum)
                            Mj_AN2 = Mj_AN;
                        else
                            Mj_AN2 = Matomnum + 1;

                        l = 0;
                        for (l1 = 1; l1 <= Spe_Num_RVPS[wakg]; l1++) {

                            ene_p = Spe_VNLE[0][wakg][l1 - 1];
                            ene_m = Spe_VNLE[1][wakg][l1 - 1];

                            if (Spe_VPS_List[wakg][l1] == 0) {
                                l2 = 0;
                                PFp = 1.0;
                                PFm = 0.0;
                            } else if (Spe_VPS_List[wakg][l1] == 1) {
                                l2 = 2;
                                PFp = 2.0 / 3.0;
                                PFm = 1.0 / 3.0;
                            } else if (Spe_VPS_List[wakg][l1] == 2) {
                                l2 = 4;
                                PFp = 3.0 / 5.0;
                                PFm = 2.0 / 5.0;
                            } else if (Spe_VPS_List[wakg][l1] == 3) {
                                l2 = 6;
                                PFp = 4.0 / 7.0;
                                PFm = 3.0 / 7.0;
                            }

                            dHNL_SO(&sumx0.r, &sumy0.r, &sumz0.r,
                                &sumx1.r, &sumy1.r, &sumz1.r,
                                &sumx2.r, &sumy2.r, &sumz2.r,
                                &sumx0.i, &sumy0.i, &sumz0.i,
                                &sumx1.i, &sumy1.i, &sumz1.i,
                                &sumx2.i, &sumy2.i, &sumz2.i,
                                1.0,
                                PFp, PFm,
                                ene_p, ene_m,
                                l2, &l,
                                Mc_AN, k, m,
                                Mj_AN2, kl, n,
                                DS_NL1);
                        }

                        if (q_AN == 0) {

                            l = 0;
                            for (l1 = 1; l1 <= Spe_Num_RVPS[wakg]; l1++) {

                                ene_p = Spe_VNLE[0][wakg][l1 - 1];
                                ene_m = Spe_VNLE[1][wakg][l1 - 1];

                                if (Spe_VPS_List[wakg][l1] == 0) {
                                    l2 = 0;
                                    PFp = 1.0;
                                    PFm = 0.0;
                                } else if (Spe_VPS_List[wakg][l1] == 1) {
                                    l2 = 2;
                                    PFp = 2.0 / 3.0;
                                    PFm = 1.0 / 3.0;
                                } else if (Spe_VPS_List[wakg][l1] == 2) {
                                    l2 = 4;
                                    PFp = 3.0 / 5.0;
                                    PFm = 2.0 / 5.0;
                                } else if (Spe_VPS_List[wakg][l1] == 3) {
                                    l2 = 6;
                                    PFp = 4.0 / 7.0;
                                    PFm = 3.0 / 7.0;
                                }

                                dHNL_SO(&sumx0.r, &sumy0.r, &sumz0.r,
                                    &sumx1.r, &sumy1.r, &sumz1.r,
                                    &sumx2.r, &sumy2.r, &sumz2.r,
                                    &sumx0.i, &sumy0.i, &sumz0.i,
                                    &sumx1.i, &sumy1.i, &sumz1.i,
                                    &sumx2.i, &sumy2.i, &sumz2.i,
                                    -1.0,
                                    PFp, PFm,
                                    ene_p, ene_m,
                                    l2, &l,
                                    Mj_AN2, kl, n,
                                    Mc_AN, k, m,
                                    DS_NL1);
                            }
                        }

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }

        } /* k */

        /****************************************************
                               H*ep*dH
        ****************************************************/

        /* h_AN==0 && q_AN==0 */

        if (q_AN == 0 && VPS_j_dependency[wakg] == 0) {

            for (m = 0; m < ian; m++) {
                for (n = m; n < jan; n++) {

                    tmpx = Hx[0][m][n].r + Hx[0][n][m].r;
                    Hx[0][m][n].r = tmpx;
                    Hx[0][n][m].r = tmpx;
                    Hx[1][m][n].r = tmpx;
                    Hx[1][n][m].r = tmpx;

                    tmpy = Hy[0][m][n].r + Hy[0][n][m].r;
                    Hy[0][m][n].r = tmpy;
                    Hy[0][n][m].r = tmpy;
                    Hy[1][m][n].r = tmpy;
                    Hy[1][n][m].r = tmpy;

                    tmpz = Hz[0][m][n].r + Hz[0][n][m].r;
                    Hz[0][m][n].r = tmpz;
                    Hz[0][n][m].r = tmpz;
                    Hz[1][m][n].r = tmpz;
                    Hz[1][n][m].r = tmpz;
                }
            }
        }

        else if (where_flag == 1) {

            kg = natn[Gc_AN][0];
            wakg = WhatSpecies[kg];
            kan = Spe_Total_VPS_Pro[wakg];
            kl = RMI1[Mc_AN][q_AN][0];

            /****************************************************
                         l-dependent non-local part
            ****************************************************/

            if (VPS_j_dependency[wakg] == 0) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        if (Mj_AN <= Matomnum) {
                            Mj_AN2 = Mj_AN;
                            kl2 = RMI1[Mc_AN][q_AN][0];
                        } else {
                            Mj_AN2 = Matomnum + 1;
                            kl2 = RMI1[Mc_AN][0][q_AN];
                        }

                        l = 0;
                        for (l1 = 1; l1 <= Spe_Num_RVPS[wakg]; l1++) {

                            ene = Spe_VNLE[0][wakg][l1 - 1];
                            if (Spe_VPS_List[wakg][l1] == 0)
                                l2 = 0;
                            else if (Spe_VPS_List[wakg][l1] == 1)
                                l2 = 2;
                            else if (Spe_VPS_List[wakg][l1] == 2)
                                l2 = 4;
                            else if (Spe_VPS_List[wakg][l1] == 3)
                                l2 = 6;

                            for (l3 = 0; l3 <= l2; l3++) {

                                sumx -= ene * DS_NL1[0][0][Mc_AN][0][m][l] * DS_NL1[0][1][Mj_AN2][kl2][n][l];
                                sumy -= ene * DS_NL1[0][0][Mc_AN][0][m][l] * DS_NL1[0][2][Mj_AN2][kl2][n][l];
                                sumz -= ene * DS_NL1[0][0][Mc_AN][0][m][l] * DS_NL1[0][3][Mj_AN2][kl2][n][l];
                                l++;
                            }
                        }

                        Hx[0][m][n].r += sumx;
                        Hy[0][m][n].r += sumy;
                        Hz[0][m][n].r += sumz;

                        Hx[1][m][n].r += sumx;
                        Hy[1][m][n].r += sumy;
                        Hz[1][m][n].r += sumz;
                    }
                }
            }

            /****************************************************
                         j-dependent non-local part
            ****************************************************/

            else if (VPS_j_dependency[wakg] == 1) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);

                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);

                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        if (Mj_AN <= Matomnum) {
                            Mj_AN2 = Mj_AN;
                            kl2 = RMI1[Mc_AN][q_AN][0];
                        } else {
                            Mj_AN2 = Matomnum + 1;
                            kl2 = RMI1[Mc_AN][0][q_AN];
                        }

                        l = 0;
                        for (l1 = 1; l1 <= Spe_Num_RVPS[wakg]; l1++) {

                            ene_p = Spe_VNLE[0][wakg][l1 - 1];
                            ene_m = Spe_VNLE[1][wakg][l1 - 1];

                            if (Spe_VPS_List[wakg][l1] == 0) {
                                l2 = 0;
                                PFp = 1.0;
                                PFm = 0.0;
                            } else if (Spe_VPS_List[wakg][l1] == 1) {
                                l2 = 2;
                                PFp = 2.0 / 3.0;
                                PFm = 1.0 / 3.0;
                            } else if (Spe_VPS_List[wakg][l1] == 2) {
                                l2 = 4;
                                PFp = 3.0 / 5.0;
                                PFm = 2.0 / 5.0;
                            } else if (Spe_VPS_List[wakg][l1] == 3) {
                                l2 = 6;
                                PFp = 4.0 / 7.0;
                                PFm = 3.0 / 7.0;
                            }

                            /* 1 */

                            dHNL_SO(&sumx0.r, &sumy0.r, &sumz0.r,
                                &sumx1.r, &sumy1.r, &sumz1.r,
                                &sumx2.r, &sumy2.r, &sumz2.r,
                                &sumx0.i, &sumy0.i, &sumz0.i,
                                &sumx1.i, &sumy1.i, &sumz1.i,
                                &sumx2.i, &sumy2.i, &sumz2.i,
                                -1.0,
                                PFp, PFm,
                                -ene_p, -ene_m,
                                l2, &l,
                                Mj_AN2, kl2, n,
                                Mc_AN, 0, m,
                                DS_NL1);
                        }

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }
        }

    } /* if (h_AN==0) */

    else if (where_flag == 0) {

        /****************************************************
           H*ep*dH

           if (h_AN!=0 && where_flag==0)
           This happens
           only if
           ( SpinP_switch==3
             &&
             (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1)
               || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1
               || Zeeman_NCO_switch==1)
             &&
             q_AN==0
           )
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            wakg = WhatSpecies[kg];
            kan = Spe_Total_VPS_Pro[wakg];
            kl = RMI1[Mc_AN][h_AN][k];

            if (Mi_AN <= Matomnum)
                Mi_AN2 = Mi_AN;
            else
                Mi_AN2 = Matomnum + 1;

            if (0 <= kl && VPS_j_dependency[wakg] == 1) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);

                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);

                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        l = 0;
                        for (l1 = 1; l1 <= Spe_Num_RVPS[wakg]; l1++) {

                            ene_p = Spe_VNLE[0][wakg][l1 - 1];
                            ene_m = Spe_VNLE[1][wakg][l1 - 1];

                            if (Spe_VPS_List[wakg][l1] == 0) {
                                l2 = 0;
                                PFp = 1.0;
                                PFm = 0.0;
                            } else if (Spe_VPS_List[wakg][l1] == 1) {
                                l2 = 2;
                                PFp = 2.0 / 3.0;
                                PFm = 1.0 / 3.0;
                            } else if (Spe_VPS_List[wakg][l1] == 2) {
                                l2 = 4;
                                PFp = 3.0 / 5.0;
                                PFm = 2.0 / 5.0;
                            } else if (Spe_VPS_List[wakg][l1] == 3) {
                                l2 = 6;
                                PFp = 4.0 / 7.0;
                                PFm = 3.0 / 7.0;
                            }

                            dHNL_SO(&sumx0.r, &sumy0.r, &sumz0.r,
                                &sumx1.r, &sumy1.r, &sumz1.r,
                                &sumx2.r, &sumy2.r, &sumz2.r,
                                &sumx0.i, &sumy0.i, &sumz0.i,
                                &sumx1.i, &sumy1.i, &sumz1.i,
                                &sumx2.i, &sumy2.i, &sumz2.i,
                                -1.0,
                                PFp, PFm,
                                ene_p, ene_m,
                                l2, &l,
                                Mj_AN, k, n,
                                Mi_AN2, kl, m,
                                DS_NL1);
                        }

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }
        }

    }

    /* if (h_AN!=0 && where_flag==1) */

    else {

        /****************************************************
                               dH*ep*H
        ****************************************************/

        kg = natn[Gc_AN][0];
        wakg = WhatSpecies[kg];
        kan = Spe_Total_VPS_Pro[wakg];
        kl1 = RMI1[Mc_AN][0][h_AN];
        kl2 = RMI1[Mc_AN][0][q_AN];

        /****************************************************
                       l-dependent non-local part
        ****************************************************/

        if (VPS_j_dependency[wakg] == 0) {

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    sumx = 0.0;
                    sumy = 0.0;
                    sumz = 0.0;

                    l = 0;
                    for (l1 = 1; l1 <= Spe_Num_RVPS[wakg]; l1++) {

                        ene = Spe_VNLE[0][wakg][l1 - 1];
                        if (Spe_VPS_List[wakg][l1] == 0)
                            l2 = 0;
                        else if (Spe_VPS_List[wakg][l1] == 1)
                            l2 = 2;
                        else if (Spe_VPS_List[wakg][l1] == 2)
                            l2 = 4;
                        else if (Spe_VPS_List[wakg][l1] == 3)
                            l2 = 6;

                        for (l3 = 0; l3 <= l2; l3++) {
                            sumx -= ene * DS_NL1[0][1][Matomnum + 1][kl1][m][l] * DS_NL1[0][0][Matomnum + 1][kl2][n][l];
                            sumy -= ene * DS_NL1[0][2][Matomnum + 1][kl1][m][l] * DS_NL1[0][0][Matomnum + 1][kl2][n][l];
                            sumz -= ene * DS_NL1[0][3][Matomnum + 1][kl1][m][l] * DS_NL1[0][0][Matomnum + 1][kl2][n][l];
                            l++;
                        }
                    }

                    Hx[0][m][n].r = sumx;
                    Hy[0][m][n].r = sumy;
                    Hz[0][m][n].r = sumz;

                    Hx[1][m][n].r = sumx;
                    Hy[1][m][n].r = sumy;
                    Hz[1][m][n].r = sumz;

                    Hx[2][m][n].r = 0.0;
                    Hy[2][m][n].r = 0.0;
                    Hz[2][m][n].r = 0.0;

                    Hx[0][m][n].i = 0.0;
                    Hy[0][m][n].i = 0.0;
                    Hz[0][m][n].i = 0.0;

                    Hx[1][m][n].i = 0.0;
                    Hy[1][m][n].i = 0.0;
                    Hz[1][m][n].i = 0.0;

                    Hx[2][m][n].i = 0.0;
                    Hy[2][m][n].i = 0.0;
                    Hz[2][m][n].i = 0.0;
                }
            }
        }

        /****************************************************
                     j-dependent non-local part
        ****************************************************/

        else if (VPS_j_dependency[wakg] == 1) {

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    sumx0 = Complex(0.0, 0.0);
                    sumy0 = Complex(0.0, 0.0);
                    sumz0 = Complex(0.0, 0.0);

                    sumx1 = Complex(0.0, 0.0);
                    sumy1 = Complex(0.0, 0.0);
                    sumz1 = Complex(0.0, 0.0);

                    sumx2 = Complex(0.0, 0.0);
                    sumy2 = Complex(0.0, 0.0);
                    sumz2 = Complex(0.0, 0.0);

                    l = 0;
                    for (l1 = 1; l1 <= Spe_Num_RVPS[wakg]; l1++) {

                        ene_p = Spe_VNLE[0][wakg][l1 - 1];
                        ene_m = Spe_VNLE[1][wakg][l1 - 1];

                        if (Spe_VPS_List[wakg][l1] == 0) {
                            l2 = 0;
                            PFp = 1.0;
                            PFm = 0.0;
                        } else if (Spe_VPS_List[wakg][l1] == 1) {
                            l2 = 2;
                            PFp = 2.0 / 3.0;
                            PFm = 1.0 / 3.0;
                        } else if (Spe_VPS_List[wakg][l1] == 2) {
                            l2 = 4;
                            PFp = 3.0 / 5.0;
                            PFm = 2.0 / 5.0;
                        } else if (Spe_VPS_List[wakg][l1] == 3) {
                            l2 = 6;
                            PFp = 4.0 / 7.0;
                            PFm = 3.0 / 7.0;
                        }

                        /* 2 */

                        dHNL_SO(&sumx0.r, &sumy0.r, &sumz0.r,
                            &sumx1.r, &sumy1.r, &sumz1.r,
                            &sumx2.r, &sumy2.r, &sumz2.r,
                            &sumx0.i, &sumy0.i, &sumz0.i,
                            &sumx1.i, &sumy1.i, &sumz1.i,
                            &sumx2.i, &sumy2.i, &sumz2.i,
                            1.0,
                            PFp, PFm,
                            -ene_p, -ene_m,
                            l2, &l,
                            Matomnum + 1, kl1, m,
                            Matomnum + 1, kl2, n,
                            DS_NL1);
                    }

                    Hx[0][m][n].r = sumx0.r; /* up-up */
                    Hy[0][m][n].r = sumy0.r; /* up-up */
                    Hz[0][m][n].r = sumz0.r; /* up-up */

                    Hx[1][m][n].r = sumx1.r; /* dn-dn */
                    Hy[1][m][n].r = sumy1.r; /* dn-dn */
                    Hz[1][m][n].r = sumz1.r; /* dn-dn */

                    Hx[2][m][n].r = sumx2.r; /* up-dn */
                    Hy[2][m][n].r = sumy2.r; /* up-dn */
                    Hz[2][m][n].r = sumz2.r; /* up-dn */

                    Hx[0][m][n].i = sumx0.i; /* up-up */
                    Hy[0][m][n].i = sumy0.i; /* up-up */
                    Hz[0][m][n].i = sumz0.i; /* up-up */

                    Hx[1][m][n].i = sumx1.i; /* dn-dn */
                    Hy[1][m][n].i = sumy1.i; /* dn-dn */
                    Hz[1][m][n].i = sumz1.i; /* dn-dn */

                    Hx[2][m][n].i = sumx2.i; /* up-dn */
                    Hy[2][m][n].i = sumy2.i; /* up-dn */
                    Hz[2][m][n].i = sumz2.i; /* up-dn */
                }
            }
        }

        /****************************************************
                               H*ep*dH
        ****************************************************/

        if (q_AN != 0) {

            kg = natn[Gc_AN][0];
            wakg = WhatSpecies[kg];
            kan = Spe_Total_VPS_Pro[wakg];
            kl1 = RMI1[Mc_AN][0][h_AN];
            kl2 = RMI1[Mc_AN][0][q_AN];

            /****************************************************
                           l-dependent non-local part
            ****************************************************/

            if (VPS_j_dependency[wakg] == 0) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        l = 0;
                        for (l1 = 1; l1 <= Spe_Num_RVPS[wakg]; l1++) {

                            ene = Spe_VNLE[0][wakg][l1 - 1];
                            if (Spe_VPS_List[wakg][l1] == 0)
                                l2 = 0;
                            else if (Spe_VPS_List[wakg][l1] == 1)
                                l2 = 2;
                            else if (Spe_VPS_List[wakg][l1] == 2)
                                l2 = 4;
                            else if (Spe_VPS_List[wakg][l1] == 3)
                                l2 = 6;

                            for (l3 = 0; l3 <= l2; l3++) {
                                sumx -= ene * DS_NL1[0][0][Matomnum + 1][kl1][m][l] * DS_NL1[0][1][Matomnum + 1][kl2][n][l];
                                sumy -= ene * DS_NL1[0][0][Matomnum + 1][kl1][m][l] * DS_NL1[0][2][Matomnum + 1][kl2][n][l];
                                sumz -= ene * DS_NL1[0][0][Matomnum + 1][kl1][m][l] * DS_NL1[0][3][Matomnum + 1][kl2][n][l];
                                l++;
                            }
                        }

                        Hx[0][m][n].r += sumx;
                        Hy[0][m][n].r += sumy;
                        Hz[0][m][n].r += sumz;

                        Hx[1][m][n].r += sumx;
                        Hy[1][m][n].r += sumy;
                        Hz[1][m][n].r += sumz;
                    }
                }
            }

            /****************************************************
                          j-dependent non-local part
            ****************************************************/

            else if (VPS_j_dependency[wakg] == 1) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);

                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);

                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        l = 0;
                        for (l1 = 1; l1 <= Spe_Num_RVPS[wakg]; l1++) {

                            ene_p = Spe_VNLE[0][wakg][l1 - 1];
                            ene_m = Spe_VNLE[1][wakg][l1 - 1];

                            if (Spe_VPS_List[wakg][l1] == 0) {
                                l2 = 0;
                                PFp = 1.0;
                                PFm = 0.0;
                            } else if (Spe_VPS_List[wakg][l1] == 1) {
                                l2 = 2;
                                PFp = 2.0 / 3.0;
                                PFm = 1.0 / 3.0;
                            } else if (Spe_VPS_List[wakg][l1] == 2) {
                                l2 = 4;
                                PFp = 3.0 / 5.0;
                                PFm = 2.0 / 5.0;
                            } else if (Spe_VPS_List[wakg][l1] == 3) {
                                l2 = 6;
                                PFp = 4.0 / 7.0;
                                PFm = 3.0 / 7.0;
                            }

                            /* 4 */

                            dHNL_SO(&sumx0.r, &sumy0.r, &sumz0.r,
                                &sumx1.r, &sumy1.r, &sumz1.r,
                                &sumx2.r, &sumy2.r, &sumz2.r,
                                &sumx0.i, &sumy0.i, &sumz0.i,
                                &sumx1.i, &sumy1.i, &sumz1.i,
                                &sumx2.i, &sumy2.i, &sumz2.i,
                                -1.0,
                                PFp, PFm,
                                -ene_p, -ene_m,
                                l2, &l,
                                Matomnum + 1, kl2, n,
                                Matomnum + 1, kl1, m,
                                DS_NL1);
                        }

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }
        }

    } /* else */

    /****************************************************
                 contribution by dampingF
    ****************************************************/

    /* Qij * dH/dx  */

    for (so = 0; so < 3; so++) {
        for (m = 0; m < ian; m++) {
            for (n = 0; n < jan; n++) {

                Hx[so][m][n].r = dmp * Hx[so][m][n].r;
                Hy[so][m][n].r = dmp * Hy[so][m][n].r;
                Hz[so][m][n].r = dmp * Hz[so][m][n].r;

                Hx[so][m][n].i = dmp * Hx[so][m][n].i;
                Hy[so][m][n].i = dmp * Hy[so][m][n].i;
                Hz[so][m][n].i = dmp * Hz[so][m][n].i;
            }
        }
    }

    /* dQij/dx * H */

    if ((h_AN == 0 && q_AN != 0) || (h_AN != 0 && q_AN == 0)) {

        if (h_AN == 0)
            kl = q_AN;
        else if (q_AN == 0)
            kl = h_AN;

        if (SpinP_switch == 0)
            somax = 0;
        else if (SpinP_switch == 1)
            somax = 1;
        else if (SpinP_switch == 3)
            somax = 2;

        r = Dis[Gc_AN][kl];

        if (rcut <= r) {
            deri_dmp = 0.0;
            tmp = 0.0;
        } else {
            deri_dmp = deri_dampingF(rcut, r);
            tmp = deri_dmp / dmp;
        }

        x0 = Gxyz[ig][1] + atv[Rni][1];
        y0 = Gxyz[ig][2] + atv[Rni][2];
        z0 = Gxyz[ig][3] + atv[Rni][3];

        x1 = Gxyz[jg][1] + atv[Rnj][1];
        y1 = Gxyz[jg][2] + atv[Rnj][2];
        z1 = Gxyz[jg][3] + atv[Rnj][3];

        /* for empty atoms or finite elemens basis */
        if (r < 1.0e-10)
            r = 1.0e-10;

        if (h_AN == 0 && q_AN != 0) {
            dx = tmp * (x0 - x1) / r;
            dy = tmp * (y0 - y1) / r;
            dz = tmp * (z0 - z1) / r;
        }

        else if (h_AN != 0 && q_AN == 0) {
            dx = tmp * (x1 - x0) / r;
            dy = tmp * (y1 - y0) / r;
            dz = tmp * (z1 - z0) / r;
        }

        if (SpinP_switch == 0 || SpinP_switch == 1) {

            if (h_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dx;
                            Hy[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dy;
                            Hz[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dz;
                        }
                    }
                }
            }

            else if (q_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dx;
                            Hy[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dy;
                            Hz[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dz;
                        }
                    }
                }
            }
        }

        else if (SpinP_switch == 3) {

            if (h_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dx;
                            Hy[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dy;
                            Hz[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dz;
                        }
                    }
                }
            }

            else if (q_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dx;
                            Hy[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dy;
                            Hz[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dz;
                        }
                    }
                }
            }

            if (SO_switch == 1) {

                if (h_AN == 0) {
                    for (so = 0; so <= somax; so++) {
                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < jan; n++) {
                                Hx[so][m][n].i += iHNL[so][Mc_AN][kl][m][n] * dx;
                                Hy[so][m][n].i += iHNL[so][Mc_AN][kl][m][n] * dy;
                                Hz[so][m][n].i += iHNL[so][Mc_AN][kl][m][n] * dz;
                            }
                        }
                    }
                }

                else if (q_AN == 0) {
                    for (so = 0; so <= somax; so++) {
                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < jan; n++) {
                                Hx[so][m][n].i += iHNL[so][Mc_AN][kl][n][m] * dx;
                                Hy[so][m][n].i += iHNL[so][Mc_AN][kl][n][m] * dy;
                                Hz[so][m][n].i += iHNL[so][Mc_AN][kl][n][m] * dz;
                            }
                        }
                    }
                }
            }
        }
    }
}

void dHVNA(int where_flag, int Mc_AN, int h_AN, int q_AN,
    Type_DS_VNA***** DS_VNA1,
    double***** TmpHVNA2, double***** TmpHVNA3,
    double** Hx, double** Hy, double** Hz)
{
    int i, j, k, m, n, l, kg, kan, so, deri_kind;
    int ig, ian, jg, jan, kl, kl1, kl2, Rni, Rnj;
    int wakg, l1, l2, l3, Gc_AN, Mi_AN, Mj_AN, Mj_AN2, num_projectors;
    double sumx, sumy, sumz, ene, rcuti, rcutj, rcut;
    double tmpx, tmpy, tmpz, dmp, deri_dmp, tmp;
    double dx, dy, dz, x0, y0, z0, x1, y1, z1, r;
    double PFp, PFm, ene_p, ene_m;
    double sumx0, sumy0, sumz0;
    double sumx1, sumy1, sumz1;
    double sumx2, sumy2, sumz2;
    int L, LL, Mul1, Num_RVNA;

    Num_RVNA = List_YOUSO[34] * (List_YOUSO[35] + 1);
    num_projectors = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];

    /****************************************************
     start calc.
    ****************************************************/

    Gc_AN = M2G[Mc_AN];
    ig = natn[Gc_AN][h_AN];
    Rni = ncn[Gc_AN][h_AN];
    Mi_AN = F_G2M[ig];
    ian = Spe_Total_CNO[WhatSpecies[ig]];
    rcuti = Spe_Atom_Cut1[WhatSpecies[ig]];

    jg = natn[Gc_AN][q_AN];
    Rnj = ncn[Gc_AN][q_AN];
    Mj_AN = F_G2M[jg];
    jan = Spe_Total_CNO[WhatSpecies[jg]];
    rcutj = Spe_Atom_Cut1[WhatSpecies[jg]];

    rcut = rcuti + rcutj;
    kl = RMI1[Mc_AN][h_AN][q_AN];
    dmp = dampingF(rcut, Dis[ig][kl]);

    for (m = 0; m < ian; m++) {
        for (n = 0; n < jan; n++) {
            Hx[m][n] = 0.0;
            Hy[m][n] = 0.0;
            Hz[m][n] = 0.0;
        }
    }

    /****************************************************
      two-center integral with orbitals on one-center

      in case of h_AN==0 && q_AN==0
    ****************************************************/

    if (h_AN == 0 && q_AN == 0 && where_flag == 0) {

// #pragma acc kernels
// #pragma acc loop independent
        for (k = 1; k <= FNAN[Gc_AN]; k++) {
// #pragma acc loop independent
            for (m = 0; m < ian; m++) {
// #pragma acc loop independent
                for (n = 0; n < jan; n++) {
// #pragma acc atomic update
                    Hx[m][n] += TmpHVNA2[1][Mc_AN][k][m][n];
// #pragma acc atomic update
                    Hy[m][n] += TmpHVNA2[2][Mc_AN][k][m][n];
// #pragma acc atomic update
                    Hz[m][n] += TmpHVNA2[3][Mc_AN][k][m][n];
                }
            }
        }
    }

    /****************************************************
      two-center integral with orbitals on one-center

      in case of h_AN==q_AN && h_AN!=0
    ****************************************************/

    else if (h_AN == q_AN && h_AN != 0) {

        kl = RMI1[Mc_AN][h_AN][0];

        for (m = 0; m < ian; m++) {
            for (n = 0; n < jan; n++) {

                Hx[m][n] = -TmpHVNA3[1][Mc_AN][h_AN][m][n];
                Hy[m][n] = -TmpHVNA3[2][Mc_AN][h_AN][m][n];
                Hz[m][n] = -TmpHVNA3[3][Mc_AN][h_AN][m][n];
            }
        }
    }

    /****************************************************
               two and three center integrals
               with orbitals on two-center
    ****************************************************/

    else {

        if (h_AN == 0) {

            /****************************************************
                                 dH*ep*H
            ****************************************************/

            for (k = 0; k <= FNAN[Gc_AN]; k++) {

                kg = natn[Gc_AN][k];
                wakg = WhatSpecies[kg];
                kl = RMI1[Mc_AN][q_AN][k];

                /****************************************************
                                     non-local part
                ****************************************************/

                if (0 <= kl && where_flag == 0) {

                    if (Mj_AN <= Matomnum)
                        Mj_AN2 = Mj_AN;
                    else
                        Mj_AN2 = Matomnum + 1;
// #pragma acc kernels
// #pragma acc loop independent
                    for (m = 0; m < ian; m++) {
// #pragma acc loop independent
                        for (n = 0; n < jan; n++) {

                            sumx = 0.0;
                            sumy = 0.0;
                            sumz = 0.0;

// #pragma acc loop reduction(+:sumx) reduction(+:sumy) reduction(+:sumz)
                            for (l = 0; l < num_projectors; l++) {
                                sumx += DS_VNA1[1][Mc_AN][k][m][l] * DS_VNA1[0][Mj_AN2][kl][n][l];
                                sumy += DS_VNA1[2][Mc_AN][k][m][l] * DS_VNA1[0][Mj_AN2][kl][n][l];
                                sumz += DS_VNA1[3][Mc_AN][k][m][l] * DS_VNA1[0][Mj_AN2][kl][n][l];
                            }

// #pragma acc atomic update
                            Hx[m][n] += sumx;
// #pragma acc atomic update
                            Hy[m][n] += sumy;
// #pragma acc atomic update
                            Hz[m][n] += sumz;

                        } /* n */
                    } /* m */

                } /* if */

            } /* k */

            /****************************************************
                                   H*ep*dH
            ****************************************************/

            /* non-local part */

            if (q_AN == 0) {

// #pragma acc kernels
// #pragma acc loop independent
                for (m = 0; m < ian; m++) {
// #pragma acc loop independent
                    for (n = m; n < jan; n++) {

                        tmpx = Hx[m][n] + Hx[n][m];
                        Hx[m][n] = tmpx;
                        Hx[n][m] = tmpx;

                        tmpy = Hy[m][n] + Hy[n][m];
                        Hy[m][n] = tmpy;
                        Hy[n][m] = tmpy;

                        tmpz = Hz[m][n] + Hz[n][m];
                        Hz[m][n] = tmpz;
                        Hz[n][m] = tmpz;
                    }
                }

            }

            else if (where_flag == 1) {

                kg = natn[Gc_AN][0];
                wakg = WhatSpecies[kg];

                /****************************************************
                                     non-local part
                ****************************************************/
// #pragma acc kernels
// #pragma acc loop independent
                for (m = 0; m < ian; m++) {
// #pragma acc loop independent
                    for (n = 0; n < jan; n++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        if (Mj_AN <= Matomnum) {
                            Mj_AN2 = Mj_AN;
                            kl = RMI1[Mc_AN][q_AN][0];
                        } else {
                            Mj_AN2 = Matomnum + 1;
                            kl = RMI1[Mc_AN][0][q_AN];
                        }

// #pragma acc loop reduction(-:sumx) reduction(-:sumy) reduction(-:sumz)
                        for (l = 0; l < num_projectors; l++) {
                            sumx -= DS_VNA1[0][Mc_AN][0][m][l] * DS_VNA1[1][Mj_AN2][kl][n][l];
                            sumy -= DS_VNA1[0][Mc_AN][0][m][l] * DS_VNA1[2][Mj_AN2][kl][n][l];
                            sumz -= DS_VNA1[0][Mc_AN][0][m][l] * DS_VNA1[3][Mj_AN2][kl][n][l];
                        }

// #pragma acc atomic update
                        Hx[m][n] += sumx;
// #pragma acc atomic update
                        Hy[m][n] += sumy;
// #pragma acc atomic update
                        Hz[m][n] += sumz;
                    }
                }
            }

        } /* if (h_AN==0) */

        else {

            /****************************************************
                                 dH*ep*H
            ****************************************************/

            kg = natn[Gc_AN][0];
            wakg = WhatSpecies[kg];
            kl1 = RMI1[Mc_AN][0][h_AN];
            kl2 = RMI1[Mc_AN][0][q_AN];

            /****************************************************
                               non-local part
            ****************************************************/
// #pragma acc kernels
// #pragma acc loop independent
            for (m = 0; m < ian; m++) {
// #pragma acc loop independent
                for (n = 0; n < jan; n++) {

                    sumx = 0.0;
                    sumy = 0.0;
                    sumz = 0.0;
// #pragma acc loop reduction(-:sumx) reduction(-:sumy) reduction(-:sumz)
                    for (l = 0; l < num_projectors; l++) {
                        sumx -= DS_VNA1[1][Matomnum + 1][kl1][m][l] * DS_VNA1[0][Matomnum + 1][kl2][n][l];
                        sumy -= DS_VNA1[2][Matomnum + 1][kl1][m][l] * DS_VNA1[0][Matomnum + 1][kl2][n][l];
                        sumz -= DS_VNA1[3][Matomnum + 1][kl1][m][l] * DS_VNA1[0][Matomnum + 1][kl2][n][l];
                    }

                    Hx[m][n] = sumx;
                    Hy[m][n] = sumy;
                    Hz[m][n] = sumz;
                }
            }

            /****************************************************
                                 H*ep*dH
            ****************************************************/

            if (q_AN != 0) {

                kg = natn[Gc_AN][0];
                wakg = WhatSpecies[kg];
                kl1 = RMI1[Mc_AN][0][h_AN];
                kl2 = RMI1[Mc_AN][0][q_AN];

                /****************************************************
                                    non-local part
                ****************************************************/
// #pragma acc kernels
// #pragma acc loop independent
                for (m = 0; m < ian; m++) {
// #pragma acc loop independent
                    for (n = 0; n < jan; n++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;
// #pragma acc loop reduction(-:sumx) reduction(-:sumy) reduction(-:sumz)
                        for (l = 0; l < num_projectors; l++) {
                            sumx -= DS_VNA1[0][Matomnum + 1][kl1][m][l] * DS_VNA1[1][Matomnum + 1][kl2][n][l];
                            sumy -= DS_VNA1[0][Matomnum + 1][kl1][m][l] * DS_VNA1[2][Matomnum + 1][kl2][n][l];
                            sumz -= DS_VNA1[0][Matomnum + 1][kl1][m][l] * DS_VNA1[3][Matomnum + 1][kl2][n][l];
                        }
// #pragma acc atomic update
                        Hx[m][n] += sumx;
// #pragma acc atomic update
                        Hy[m][n] += sumy;
// #pragma acc atomic update
                        Hz[m][n] += sumz;
                    }
                }
            }
        }
    }

    /****************************************************
                  contribution by dampingF
    ****************************************************/

    /* Qij * dH/dx  */

    for (m = 0; m < ian; m++) {
        for (n = 0; n < jan; n++) {
            Hx[m][n] = dmp * Hx[m][n];
            Hy[m][n] = dmp * Hy[m][n];
            Hz[m][n] = dmp * Hz[m][n];
        }
    }

    /* dQij/dx * H */

    if ((h_AN == 0 && q_AN != 0) || (h_AN != 0 && q_AN == 0)) {

        if (h_AN == 0)
            kl = q_AN;
        else if (q_AN == 0)
            kl = h_AN;

        r = Dis[Gc_AN][kl];

        if (rcut <= r) {
            deri_dmp = 0.0;
            tmp = 0.0;
        } else {
            deri_dmp = deri_dampingF(rcut, r);
            tmp = deri_dmp / dmp;
        }

        x0 = Gxyz[ig][1] + atv[Rni][1];
        x1 = Gxyz[jg][1] + atv[Rnj][1];

        y0 = Gxyz[ig][2] + atv[Rni][2];
        y1 = Gxyz[jg][2] + atv[Rnj][2];

        z0 = Gxyz[ig][3] + atv[Rni][3];
        z1 = Gxyz[jg][3] + atv[Rnj][3];

        /* for empty atoms or finite elemens basis */
        if (r < 1.0e-10)
            r = 1.0e-10;

        if (h_AN == 0) {
            dx = tmp * (x0 - x1) / r;
            dy = tmp * (y0 - y1) / r;
            dz = tmp * (z0 - z1) / r;
        }

        else if (q_AN == 0) {
            dx = tmp * (x1 - x0) / r;
            dy = tmp * (y1 - y0) / r;
            dz = tmp * (z1 - z0) / r;
        }

        if (h_AN == 0) {
            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {
                    Hx[m][n] += HVNA[Mc_AN][kl][m][n] * dx;
                    Hy[m][n] += HVNA[Mc_AN][kl][m][n] * dy;
                    Hz[m][n] += HVNA[Mc_AN][kl][m][n] * dz;
                }
            }
        }

        else if (q_AN == 0) {
            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {
                    Hx[m][n] += HVNA[Mc_AN][kl][n][m] * dx;
                    Hy[m][n] += HVNA[Mc_AN][kl][n][m] * dy;
                    Hz[m][n] += HVNA[Mc_AN][kl][n][m] * dz;
                }
            }
        }
    }
}

void dHNL_SO(
    double* sumx0r,
    double* sumy0r,
    double* sumz0r,
    double* sumx1r,
    double* sumy1r,
    double* sumz1r,
    double* sumx2r,
    double* sumy2r,
    double* sumz2r,
    double* sumx0i,
    double* sumy0i,
    double* sumz0i,
    double* sumx1i,
    double* sumy1i,
    double* sumz1i,
    double* sumx2i,
    double* sumy2i,
    double* sumz2i,
    double fugou,
    double PFp,
    double PFm,
    double ene_p,
    double ene_m,
    int l2, int* l,
    int Mc_AN, int k, int m,
    int Mj_AN, int kl, int n,
    double****** DS_NL1)
{

    int l3, i;
    double tmpx, tmpy, tmpz;
    double tmp0, tmp1, tmp2;
    double tmp3, tmp4, tmp5, tmp6;
    double deri[4];

    /****************************************************
      off-diagonal contribution to up-dn matrix
      for spin non-collinear
    ****************************************************/

    if (SpinP_switch == 3) {

        /* p */
        if (l2 == 2) {

            /* real contribution of l+1/2 to off diagonal up-down matrix */
            tmpx = fugou * (ene_p / 3.0 * DS_NL1[0][1][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] - ene_p / 3.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l]);

            tmpy = fugou * (ene_p / 3.0 * DS_NL1[0][2][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] - ene_p / 3.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l]);

            tmpz = fugou * (ene_p / 3.0 * DS_NL1[0][3][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] - ene_p / 3.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l]);

            *sumx2r += tmpx;
            *sumy2r += tmpy;
            *sumz2r += tmpz;

            /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

            tmpx = fugou * (-ene_p / 3.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] + ene_p / 3.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1]);

            tmpy = fugou * (-ene_p / 3.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] + ene_p / 3.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1]);

            tmpz = fugou * (-ene_p / 3.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] + ene_p / 3.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1]);

            *sumx2i += tmpx;
            *sumy2i += tmpy;
            *sumz2i += tmpz;

            /* real contribution of l-1/2 for to diagonal up-down matrix */

            tmpx = fugou * (ene_m / 3.0 * DS_NL1[1][1][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] - ene_m / 3.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l]);

            tmpy = fugou * (ene_m / 3.0 * DS_NL1[1][2][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] - ene_m / 3.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l]);

            tmpz = fugou * (ene_m / 3.0 * DS_NL1[1][3][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] - ene_m / 3.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l]);

            *sumx2r -= tmpx;
            *sumy2r -= tmpy;
            *sumz2r -= tmpz;

            /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

            tmpx = fugou * (-ene_m / 3.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] + ene_m / 3.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1]);

            tmpy = fugou * (-ene_m / 3.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] + ene_m / 3.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1]);

            tmpz = fugou * (-ene_m / 3.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] + ene_m / 3.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1]);

            *sumx2i -= tmpx;
            *sumy2i -= tmpy;
            *sumz2i -= tmpz;
        }

        /* d */
        if (l2 == 4) {

            tmp0 = sqrt(3.0);
            tmp1 = ene_p / 5.0;
            tmp2 = tmp0 * tmp1;

            /* real contribution of l+1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp2 * DS_NL1[0][i][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3] + tmp2 * DS_NL1[0][i][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l] + tmp1 * DS_NL1[0][i][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3] - tmp1 * DS_NL1[0][i][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] + tmp1 * DS_NL1[0][i][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4] - tmp1 * DS_NL1[0][i][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2]);
            }
            *sumx2r += deri[1];
            *sumy2r += deri[2];
            *sumz2r += deri[3];

            /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp2 * DS_NL1[0][i][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4] - tmp2 * DS_NL1[0][i][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l] + tmp1 * DS_NL1[0][i][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4] - tmp1 * DS_NL1[0][i][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] - tmp1 * DS_NL1[0][i][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3] + tmp1 * DS_NL1[0][i][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2]);
            }
            *sumx2i += deri[1];
            *sumy2i += deri[2];
            *sumz2i += deri[3];

            /* real contribution of l-1/2 for to diagonal up-down matrix */

            tmp1 = ene_m / 5.0;
            tmp2 = tmp0 * tmp1;

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp2 * DS_NL1[1][i][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3] + tmp2 * DS_NL1[1][i][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l] + tmp1 * DS_NL1[1][i][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3] - tmp1 * DS_NL1[1][i][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] + tmp1 * DS_NL1[1][i][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4] - tmp1 * DS_NL1[1][i][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2]);
            }
            *sumx2r -= deri[1];
            *sumy2r -= deri[2];
            *sumz2r -= deri[3];

            /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp2 * DS_NL1[1][i][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4] - tmp2 * DS_NL1[1][i][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l] + tmp1 * DS_NL1[1][i][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4] - tmp1 * DS_NL1[1][i][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] - tmp1 * DS_NL1[1][i][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3] + tmp1 * DS_NL1[1][i][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2]);
            }
            *sumx2i -= deri[1];
            *sumy2i -= deri[2];
            *sumz2i -= deri[3];
        }

        /* f */
        if (l2 == 6) {

            /* real contribution of l+1/2 to off diagonal up-down matrix */

            tmp0 = sqrt(6.0);
            tmp1 = sqrt(3.0 / 2.0);
            tmp2 = sqrt(5.0 / 2.0);

            tmp3 = ene_p / 7.0;
            tmp4 = tmp1 * tmp3; /* sqrt(3.0/2.0) */
            tmp5 = tmp2 * tmp3; /* sqrt(5.0/2.0) */
            tmp6 = tmp0 * tmp3; /* sqrt(6.0)     */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp6 * DS_NL1[0][i][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] + tmp6 * DS_NL1[0][i][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l] - tmp5 * DS_NL1[0][i][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3] + tmp5 * DS_NL1[0][i][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] - tmp5 * DS_NL1[0][i][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4] + tmp5 * DS_NL1[0][i][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] - tmp4 * DS_NL1[0][i][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 5] + tmp4 * DS_NL1[0][i][Mc_AN][k][m][*l + 5] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3] - tmp4 * DS_NL1[0][i][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 6] + tmp4 * DS_NL1[0][i][Mc_AN][k][m][*l + 6] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4]);
            }
            *sumx2r += deri[1];
            *sumy2r += deri[2];
            *sumz2r += deri[3];

            /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp6 * DS_NL1[0][i][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] - tmp6 * DS_NL1[0][i][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l] + tmp5 * DS_NL1[0][i][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4] - tmp5 * DS_NL1[0][i][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] - tmp5 * DS_NL1[0][i][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3] + tmp5 * DS_NL1[0][i][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] + tmp4 * DS_NL1[0][i][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 6] - tmp4 * DS_NL1[0][i][Mc_AN][k][m][*l + 6] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3] - tmp4 * DS_NL1[0][i][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 5] + tmp4 * DS_NL1[0][i][Mc_AN][k][m][*l + 5] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4]);
            }
            *sumx2i += deri[1];
            *sumy2i += deri[2];
            *sumz2i += deri[3];

            /* real contribution of l-1/2 for to diagonal up-down matrix */

            tmp3 = ene_m / 7.0;
            tmp4 = tmp1 * tmp3; /* sqrt(3.0/2.0) */
            tmp5 = tmp2 * tmp3; /* sqrt(5.0/2.0) */
            tmp6 = tmp0 * tmp3; /* sqrt(6.0)     */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp6 * DS_NL1[1][i][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] + tmp6 * DS_NL1[1][i][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l] - tmp5 * DS_NL1[1][i][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3] + tmp5 * DS_NL1[1][i][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] - tmp5 * DS_NL1[1][i][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4] + tmp5 * DS_NL1[1][i][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] - tmp4 * DS_NL1[1][i][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 5] + tmp4 * DS_NL1[1][i][Mc_AN][k][m][*l + 5] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3] - tmp4 * DS_NL1[1][i][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 6] + tmp4 * DS_NL1[1][i][Mc_AN][k][m][*l + 6] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4]);
            }
            *sumx2r -= deri[1];
            *sumy2r -= deri[2];
            *sumz2r -= deri[3];

            /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp6 * DS_NL1[1][i][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] - tmp6 * DS_NL1[1][i][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l] + tmp5 * DS_NL1[1][i][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4] - tmp5 * DS_NL1[1][i][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] - tmp5 * DS_NL1[1][i][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3] + tmp5 * DS_NL1[1][i][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] + tmp4 * DS_NL1[1][i][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 6] - tmp4 * DS_NL1[1][i][Mc_AN][k][m][*l + 6] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3] - tmp4 * DS_NL1[1][i][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 5] + tmp4 * DS_NL1[1][i][Mc_AN][k][m][*l + 5] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4]);
            }
            *sumx2i -= deri[1];
            *sumy2i -= deri[2];
            *sumz2i -= deri[3];
        }
    }

    /****************************************************
        off-diagonal contribution on up-up and dn-dn
    ****************************************************/

    /* p */
    if (l2 == 2) {

        tmpx = fugou * (ene_p / 3.0 * DS_NL1[0][1][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] - ene_p / 3.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l]);

        tmpy = fugou * (ene_p / 3.0 * DS_NL1[0][2][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] - ene_p / 3.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l]);

        tmpz = fugou * (ene_p / 3.0 * DS_NL1[0][3][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] - ene_p / 3.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l]);

        /* contribution of l+1/2 for up spin */
        *sumx0i += -tmpx;
        *sumy0i += -tmpy;
        *sumz0i += -tmpz;

        /* contribution of l+1/2 for down spin */
        *sumx1i += tmpx;
        *sumy1i += tmpy;
        *sumz1i += tmpz;

        tmpx = fugou * (ene_m / 3.0 * DS_NL1[1][1][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] - ene_m / 3.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l]);

        tmpy = fugou * (ene_m / 3.0 * DS_NL1[1][2][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] - ene_m / 3.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l]);

        tmpz = fugou * (ene_m / 3.0 * DS_NL1[1][3][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] - ene_m / 3.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l]);

        /* contribution of l-1/2 for up spin */
        *sumx0i += tmpx;
        *sumy0i += tmpy;
        *sumz0i += tmpz;

        /* contribution of l+1/2 for down spin */
        *sumx1i += -tmpx;
        *sumy1i += -tmpy;
        *sumz1i += -tmpz;
    }

    /* d */
    else if (l2 == 4) {

        tmpx = fugou * (ene_p * 2.0 / 5.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] - ene_p * 2.0 / 5.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] + ene_p * 1.0 / 5.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4] - ene_p * 1.0 / 5.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3]);

        tmpy = fugou * (ene_p * 2.0 / 5.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] - ene_p * 2.0 / 5.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] + ene_p * 1.0 / 5.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4] - ene_p * 1.0 / 5.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3]);

        tmpz = fugou * (ene_p * 2.0 / 5.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] - ene_p * 2.0 / 5.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] + ene_p * 1.0 / 5.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4] - ene_p * 1.0 / 5.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3]);

        /* contribution of l+1/2 for up spin */
        *sumx0i += -tmpx;
        *sumy0i += -tmpy;
        *sumz0i += -tmpz;

        /* contribution of l+1/2 for down spin */
        *sumx1i += tmpx;
        *sumy1i += tmpy;
        *sumz1i += tmpz;

        tmpx = fugou * (ene_m * 2.0 / 5.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] - ene_m * 2.0 / 5.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] + ene_m * 1.0 / 5.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4] - ene_m * 1.0 / 5.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3]);

        tmpy = fugou * (ene_m * 2.0 / 5.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] - ene_m * 2.0 / 5.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] + ene_m * 1.0 / 5.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4] - ene_m * 1.0 / 5.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3]);

        tmpz = fugou * (ene_m * 2.0 / 5.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] - ene_m * 2.0 / 5.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] + ene_m * 1.0 / 5.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4] - ene_m * 1.0 / 5.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3]);

        /* contribution of l-1/2 for up spin */
        *sumx0i += tmpx;
        *sumy0i += tmpy;
        *sumz0i += tmpz;

        /* contribution of l-1/2 for down spin */
        *sumx1i += -tmpx;
        *sumy1i += -tmpy;
        *sumz1i += -tmpz;

    }

    /* f */
    else if (l2 == 6) {

        tmpx = fugou * (ene_p * 1.0 / 7.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] - ene_p * 1.0 / 7.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] + ene_p * 2.0 / 7.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4] - ene_p * 2.0 / 7.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3] + ene_p * 3.0 / 7.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 5] * DS_NL1[0][0][Mj_AN][kl][n][*l + 6] - ene_p * 3.0 / 7.0 * DS_NL1[0][1][Mc_AN][k][m][*l + 6] * DS_NL1[0][0][Mj_AN][kl][n][*l + 5]);

        tmpy = fugou * (ene_p * 1.0 / 7.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] - ene_p * 1.0 / 7.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] + ene_p * 2.0 / 7.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4] - ene_p * 2.0 / 7.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3] + ene_p * 3.0 / 7.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 5] * DS_NL1[0][0][Mj_AN][kl][n][*l + 6] - ene_p * 3.0 / 7.0 * DS_NL1[0][2][Mc_AN][k][m][*l + 6] * DS_NL1[0][0][Mj_AN][kl][n][*l + 5]);

        tmpz = fugou * (ene_p * 1.0 / 7.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 1] * DS_NL1[0][0][Mj_AN][kl][n][*l + 2] - ene_p * 1.0 / 7.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 2] * DS_NL1[0][0][Mj_AN][kl][n][*l + 1] + ene_p * 2.0 / 7.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 3] * DS_NL1[0][0][Mj_AN][kl][n][*l + 4] - ene_p * 2.0 / 7.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 4] * DS_NL1[0][0][Mj_AN][kl][n][*l + 3] + ene_p * 3.0 / 7.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 5] * DS_NL1[0][0][Mj_AN][kl][n][*l + 6] - ene_p * 3.0 / 7.0 * DS_NL1[0][3][Mc_AN][k][m][*l + 6] * DS_NL1[0][0][Mj_AN][kl][n][*l + 5]);

        /* contribution of l+1/2 for up spin */
        *sumx0i += -tmpx;
        *sumy0i += -tmpy;
        *sumz0i += -tmpz;

        /* contribution of l+1/2 for down spin */
        *sumx1i += tmpx;
        *sumy1i += tmpy;
        *sumz1i += tmpz;

        tmpx = fugou * (ene_m * 1.0 / 7.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] - ene_m * 1.0 / 7.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] + ene_m * 2.0 / 7.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4] - ene_m * 2.0 / 7.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3] + ene_m * 3.0 / 7.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 5] * DS_NL1[1][0][Mj_AN][kl][n][*l + 6] - ene_m * 3.0 / 7.0 * DS_NL1[1][1][Mc_AN][k][m][*l + 6] * DS_NL1[1][0][Mj_AN][kl][n][*l + 5]);

        tmpy = fugou * (ene_m * 1.0 / 7.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] - ene_m * 1.0 / 7.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] + ene_m * 2.0 / 7.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4] - ene_m * 2.0 / 7.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3] + ene_m * 3.0 / 7.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 5] * DS_NL1[1][0][Mj_AN][kl][n][*l + 6] - ene_m * 3.0 / 7.0 * DS_NL1[1][2][Mc_AN][k][m][*l + 6] * DS_NL1[1][0][Mj_AN][kl][n][*l + 5]);

        tmpz = fugou * (ene_m * 1.0 / 7.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 1] * DS_NL1[1][0][Mj_AN][kl][n][*l + 2] - ene_m * 1.0 / 7.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 2] * DS_NL1[1][0][Mj_AN][kl][n][*l + 1] + ene_m * 2.0 / 7.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 3] * DS_NL1[1][0][Mj_AN][kl][n][*l + 4] - ene_m * 2.0 / 7.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 4] * DS_NL1[1][0][Mj_AN][kl][n][*l + 3] + ene_m * 3.0 / 7.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 5] * DS_NL1[1][0][Mj_AN][kl][n][*l + 6] - ene_m * 3.0 / 7.0 * DS_NL1[1][3][Mc_AN][k][m][*l + 6] * DS_NL1[1][0][Mj_AN][kl][n][*l + 5]);

        /* contribution of l-1/2 for up spin */
        *sumx0i += tmpx;
        *sumy0i += tmpy;
        *sumz0i += tmpz;

        /* contribution of l-1/2 for down spin */
        *sumx1i += -tmpx;
        *sumy1i += -tmpy;
        *sumz1i += -tmpz;
    }

    /****************************************************
        diagonal contribution on up-up and dn-dn
    ****************************************************/

    for (l3 = 0; l3 <= l2; l3++) {

        /* VNL for j=l+1/2 */

        tmpx = PFp * ene_p * DS_NL1[0][1][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l];
        tmpy = PFp * ene_p * DS_NL1[0][2][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l];
        tmpz = PFp * ene_p * DS_NL1[0][3][Mc_AN][k][m][*l] * DS_NL1[0][0][Mj_AN][kl][n][*l];

        *sumx0r += tmpx;
        *sumy0r += tmpy;
        *sumz0r += tmpz;

        *sumx1r += tmpx;
        *sumy1r += tmpy;
        *sumz1r += tmpz;

        /* VNL for j=l-1/2 */

        tmpx = PFm * ene_m * DS_NL1[1][1][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l];
        tmpy = PFm * ene_m * DS_NL1[1][2][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l];
        tmpz = PFm * ene_m * DS_NL1[1][3][Mc_AN][k][m][*l] * DS_NL1[1][0][Mj_AN][kl][n][*l];

        *sumx0r += tmpx;
        *sumy0r += tmpy;
        *sumz0r += tmpz;

        *sumx1r += tmpx;
        *sumy1r += tmpy;
        *sumz1r += tmpz;

        *l = *l + 1;
    }
}

void dH_U_full(int Mc_AN, int h_AN, int q_AN,
    double***** OLP, double**** v_eff,
    double*** Hx, double*** Hy, double*** Hz)
{
    int i, j, k, m, n, kg, kan, so, deri_kind, Mk_AN;
    int ig, ian, jg, jan, kl, kl1, kl2, spin, spinmax;
    int wakg, l1, l2, l3, Gc_AN, Mi_AN, Mj_AN;
    int Rwan, Lwan, p, p0;
    double PF[2], sumx, sumy, sumz, ene;
    double tmpx, tmpy, tmpz;
    double Lsum0, Lsum1, Lsum2, Lsum3;
    double Rsum0, Rsum1, Rsum2, Rsum3;
    double PFp, PFm, ene_p, ene_m;
    double ***Hx2, ***Hy2, ***Hz2;
    double sumx0, sumy0, sumz0;
    double sumx1, sumy1, sumz1;
    double sumx2, sumy2, sumz2;

    /****************************************************
     allocation of arrays:

     double Hx2[3][List_YOUSO[7]][List_YOUSO[7]];
     double Hy2[3][List_YOUSO[7]][List_YOUSO[7]];
     double Hz2[3][List_YOUSO[7]][List_YOUSO[7]];
    ****************************************************/

    Hx2 = (double***)malloc(sizeof(double**) * 3);
    for (i = 0; i < 3; i++) {
        Hx2[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hx2[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
        }
    }

    Hy2 = (double***)malloc(sizeof(double**) * 3);
    for (i = 0; i < 3; i++) {
        Hy2[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hy2[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
        }
    }

    Hz2 = (double***)malloc(sizeof(double**) * 3);
    for (i = 0; i < 3; i++) {
        Hz2[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hz2[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
        }
    }

    /****************************************************
     start calc.
    ****************************************************/

    if (SpinP_switch == 0)
        spinmax = 0;
    else
        spinmax = 1;

    Gc_AN = M2G[Mc_AN];
    ig = natn[Gc_AN][h_AN];
    Lwan = WhatSpecies[ig];
    Mi_AN = F_G2M[ig]; /* F_G2M should be used */
    ian = Spe_Total_CNO[Lwan];
    jg = natn[Gc_AN][q_AN];
    Rwan = WhatSpecies[jg];
    Mj_AN = F_G2M[jg]; /* F_G2M should be used */
    jan = Spe_Total_CNO[Rwan];

    if (h_AN == 0) {

        /****************************************************
                              dS*ep*S
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl = RMI1[Mc_AN][q_AN][k];

            /****************************************************
                        derivative at h_AN (=Mc_AN)
            ****************************************************/

            if (0 <= kl) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        for (spin = 0; spin <= spinmax; spin++) {

                            sumx = 0.0;
                            sumy = 0.0;
                            sumz = 0.0;

                            if (Cnt_switch == 0) {

                                for (l1 = 0; l1 < kan; l1++) {
                                    for (l2 = 0; l2 < kan; l2++) {
                                        ene = v_eff[spin][Mk_AN][l1][l2];
                                        sumx += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                        sumy += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                        sumz += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                    }
                                }

                            }

                            else if (Cnt_switch == 1) {

                                for (l1 = 0; l1 < kan; l1++) {
                                    for (l2 = 0; l2 < kan; l2++) {
                                        Lsum1 = 0.0;
                                        Lsum2 = 0.0;
                                        Lsum3 = 0.0;
                                        for (p = 0; p < Spe_Specified_Num[Lwan][m]; p++) {
                                            p0 = Spe_Trans_Orbital[Lwan][m][p];
                                            Lsum1 += CntCoes[Mc_AN][m][p] * OLP[1][Mc_AN][k][p0][l1];
                                            Lsum2 += CntCoes[Mc_AN][m][p] * OLP[2][Mc_AN][k][p0][l1];
                                            Lsum3 += CntCoes[Mc_AN][m][p] * OLP[3][Mc_AN][k][p0][l1];
                                        }

                                        Rsum0 = 0.0;
                                        for (p = 0; p < Spe_Specified_Num[Rwan][n]; p++) {
                                            p0 = Spe_Trans_Orbital[Rwan][n][p];
                                            Rsum0 += CntCoes[Mj_AN][n][p] * OLP[0][Mj_AN][kl][p0][l2];
                                        }

                                        ene = v_eff[spin][Mk_AN][l1][l2];
                                        sumx += ene * Lsum1 * Rsum0;
                                        sumy += ene * Lsum2 * Rsum0;
                                        sumz += ene * Lsum3 * Rsum0;
                                    }
                                }
                            }

                            if (k == 0) {
                                Hx[spin][m][n] = sumx;
                                Hy[spin][m][n] = sumy;
                                Hz[spin][m][n] = sumz;

                                Hx[2][m][n] = 0.0;
                                Hy[2][m][n] = 0.0;
                                Hz[2][m][n] = 0.0;
                            } else {
                                Hx[spin][m][n] += sumx;
                                Hy[spin][m][n] += sumy;
                                Hz[spin][m][n] += sumz;
                            }
                        }
                    }
                }
            } /* if */
        } /* k */

        /****************************************************
                              S*ep*dS
        ****************************************************/

        if (q_AN == 0) {
            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {
                    Hx2[0][m][n] = Hx[0][m][n];
                    Hy2[0][m][n] = Hy[0][m][n];
                    Hz2[0][m][n] = Hz[0][m][n];

                    Hx2[1][m][n] = Hx[1][m][n];
                    Hy2[1][m][n] = Hy[1][m][n];
                    Hz2[1][m][n] = Hz[1][m][n];
                }
            }
            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {
                    Hx[0][m][n] = Hx2[0][m][n] + Hx2[0][n][m];
                    Hy[0][m][n] = Hy2[0][m][n] + Hy2[0][n][m];
                    Hz[0][m][n] = Hz2[0][m][n] + Hz2[0][n][m];

                    Hx[1][m][n] = Hx2[1][m][n] + Hx2[1][n][m];
                    Hy[1][m][n] = Hy2[1][m][n] + Hy2[1][n][m];
                    Hz[1][m][n] = Hz2[1][m][n] + Hz2[1][n][m];
                }
            }
        }

        else {

            kg = natn[Gc_AN][0];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl = RMI1[Mc_AN][q_AN][0];

            /****************************************************
                              derivative at k=0
            ****************************************************/

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    for (spin = 0; spin <= spinmax; spin++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        if (Cnt_switch == 0) {

                            for (l1 = 0; l1 < kan; l1++) {
                                for (l2 = 0; l2 < kan; l2++) {
                                    ene = v_eff[spin][Mk_AN][l1][l2];
                                    sumx -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                                    sumy -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                                    sumz -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];
                                }
                            }

                        }

                        else if (Cnt_switch == 1) {

                            for (l1 = 0; l1 < kan; l1++) {
                                for (l2 = 0; l2 < kan; l2++) {

                                    Lsum0 = 0.0;

                                    for (p = 0; p < Spe_Specified_Num[Lwan][m]; p++) {
                                        p0 = Spe_Trans_Orbital[Lwan][m][p];
                                        Lsum0 += CntCoes[Mc_AN][m][p] * OLP[0][Mc_AN][0][p0][l1];
                                    }

                                    Rsum1 = 0.0;
                                    Rsum2 = 0.0;
                                    Rsum3 = 0.0;

                                    for (p = 0; p < Spe_Specified_Num[Rwan][n]; p++) {
                                        p0 = Spe_Trans_Orbital[Rwan][n][p];
                                        Rsum1 += CntCoes[Mj_AN][n][p] * OLP[1][Mj_AN][kl][p0][l2];
                                        Rsum2 += CntCoes[Mj_AN][n][p] * OLP[2][Mj_AN][kl][p0][l2];
                                        Rsum3 += CntCoes[Mj_AN][n][p] * OLP[3][Mj_AN][kl][p0][l2];
                                    }

                                    ene = v_eff[spin][Mk_AN][l1][l2];
                                    sumx -= ene * Lsum0 * Rsum1;
                                    sumy -= ene * Lsum0 * Rsum2;
                                    sumz -= ene * Lsum0 * Rsum3;
                                }
                            }
                        }

                        Hx[spin][m][n] += sumx;
                        Hy[spin][m][n] += sumy;
                        Hz[spin][m][n] += sumz;
                    }
                }
            }
        }

    } /* if (h_AN==0) */

    else {

        /****************************************************
                               dS*ep*S
        ****************************************************/

        kg = natn[Gc_AN][0];
        Mk_AN = F_G2M[kg]; /* F_G2M should be used */
        wakg = WhatSpecies[kg];
        kan = Spe_Total_NO[wakg];
        kl1 = RMI1[Mc_AN][h_AN][0];
        kl2 = RMI1[Mc_AN][q_AN][0];

        for (m = 0; m < ian; m++) {
            for (n = 0; n < jan; n++) {

                for (spin = 0; spin <= spinmax; spin++) {

                    sumx = 0.0;
                    sumy = 0.0;
                    sumz = 0.0;

                    if (Cnt_switch == 0) {

                        for (l1 = 0; l1 < kan; l1++) {
                            for (l2 = 0; l2 < kan; l2++) {
                                ene = v_eff[spin][Mk_AN][l1][l2];
                                sumx -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                                sumy -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                                sumz -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                            }
                        }
                    }

                    else if (Cnt_switch == 1) {

                        for (l1 = 0; l1 < kan; l1++) {
                            for (l2 = 0; l2 < kan; l2++) {

                                Lsum1 = 0.0;
                                Lsum2 = 0.0;
                                Lsum3 = 0.0;
                                for (p = 0; p < Spe_Specified_Num[Lwan][m]; p++) {
                                    p0 = Spe_Trans_Orbital[Lwan][m][p];
                                    Lsum1 += CntCoes[Mi_AN][m][p] * OLP[1][Mi_AN][kl1][p0][l1];
                                    Lsum2 += CntCoes[Mi_AN][m][p] * OLP[2][Mi_AN][kl1][p0][l1];
                                    Lsum3 += CntCoes[Mi_AN][m][p] * OLP[3][Mi_AN][kl1][p0][l1];
                                }

                                Rsum0 = 0.0;
                                for (p = 0; p < Spe_Specified_Num[Rwan][n]; p++) {
                                    p0 = Spe_Trans_Orbital[Rwan][n][p];
                                    Rsum0 += CntCoes[Mj_AN][n][p] * OLP[0][Mj_AN][kl2][p0][l2];
                                }

                                ene = v_eff[spin][Mk_AN][l1][l2];
                                sumx -= ene * Lsum1 * Rsum0;
                                sumy -= ene * Lsum2 * Rsum0;
                                sumz -= ene * Lsum3 * Rsum0;
                            }
                        }
                    }

                    Hx[spin][m][n] = sumx;
                    Hy[spin][m][n] = sumy;
                    Hz[spin][m][n] = sumz;

                    Hx[2][m][n] = 0.0;
                    Hy[2][m][n] = 0.0;
                    Hz[2][m][n] = 0.0;
                }
            }
        }

        /****************************************************
                               S*ep*dS
        ****************************************************/

        if (q_AN == 0) {

            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                kg = natn[Gc_AN][k];
                Mk_AN = F_G2M[kg]; /* F_G2M should be used */
                wakg = WhatSpecies[kg];
                kan = Spe_Total_NO[wakg];
                kl1 = RMI1[Mc_AN][h_AN][k];
                kl2 = RMI1[Mc_AN][q_AN][k];

                if (0 <= kl1) {

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            for (spin = 0; spin <= spinmax; spin++) {

                                sumx = 0.0;
                                sumy = 0.0;
                                sumz = 0.0;

                                if (Cnt_switch == 0) {

                                    for (l1 = 0; l1 < kan; l1++) {
                                        for (l2 = 0; l2 < kan; l2++) {
                                            ene = v_eff[spin][Mk_AN][l1][l2];
                                            sumx += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                            sumy += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                            sumz += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];
                                        }
                                    }

                                }

                                else if (Cnt_switch == 1) {

                                    for (l1 = 0; l1 < kan; l1++) {
                                        for (l2 = 0; l2 < kan; l2++) {

                                            Lsum0 = 0.0;

                                            for (p = 0; p < Spe_Specified_Num[Lwan][m]; p++) {
                                                p0 = Spe_Trans_Orbital[Lwan][m][p];
                                                Lsum0 += CntCoes[Mi_AN][m][p] * OLP[0][Mi_AN][kl1][p0][l1];
                                            }

                                            Rsum1 = 0.0;
                                            Rsum2 = 0.0;
                                            Rsum3 = 0.0;

                                            for (p = 0; p < Spe_Specified_Num[Rwan][n]; p++) {
                                                p0 = Spe_Trans_Orbital[Rwan][n][p];
                                                Rsum1 += CntCoes[Mj_AN][n][p] * OLP[1][Mj_AN][kl2][p0][l2];
                                                Rsum2 += CntCoes[Mj_AN][n][p] * OLP[2][Mj_AN][kl2][p0][l2];
                                                Rsum3 += CntCoes[Mj_AN][n][p] * OLP[3][Mj_AN][kl2][p0][l2];
                                            }

                                            ene = v_eff[spin][Mk_AN][l1][l2];
                                            sumx += ene * Lsum0 * Rsum1;
                                            sumy += ene * Lsum0 * Rsum2;
                                            sumz += ene * Lsum0 * Rsum3;
                                        }
                                    }
                                }

                                Hx[spin][m][n] += sumx;
                                Hy[spin][m][n] += sumy;
                                Hz[spin][m][n] += sumz;
                            }
                        }
                    }
                }
            }
        } /* if (q_AN==0) */

        else {

            kg = natn[Gc_AN][0];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl1 = RMI1[Mc_AN][h_AN][0];
            kl2 = RMI1[Mc_AN][q_AN][0];

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    for (spin = 0; spin <= spinmax; spin++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        if (Cnt_switch == 0) {

                            for (l1 = 0; l1 < kan; l1++) {
                                for (l2 = 0; l2 < kan; l2++) {
                                    ene = v_eff[spin][Mk_AN][l1][l2];
                                    sumx -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    sumy -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    sumz -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];
                                }
                            }
                        }

                        else if (Cnt_switch == 1) {

                            for (l1 = 0; l1 < kan; l1++) {
                                for (l2 = 0; l2 < kan; l2++) {

                                    Lsum0 = 0.0;

                                    for (p = 0; p < Spe_Specified_Num[Lwan][m]; p++) {
                                        p0 = Spe_Trans_Orbital[Lwan][m][p];
                                        Lsum0 += CntCoes[Mi_AN][m][p] * OLP[0][Mi_AN][kl1][p0][l1];
                                    }

                                    Rsum1 = 0.0;
                                    Rsum2 = 0.0;
                                    Rsum3 = 0.0;

                                    for (p = 0; p < Spe_Specified_Num[Rwan][n]; p++) {
                                        p0 = Spe_Trans_Orbital[Rwan][n][p];
                                        Rsum1 += CntCoes[Mj_AN][n][p] * OLP[1][Mj_AN][kl2][p0][l2];
                                        Rsum2 += CntCoes[Mj_AN][n][p] * OLP[2][Mj_AN][kl2][p0][l2];
                                        Rsum3 += CntCoes[Mj_AN][n][p] * OLP[3][Mj_AN][kl2][p0][l2];
                                    }

                                    ene = v_eff[spin][Mk_AN][l1][l2];
                                    sumx -= ene * Lsum0 * Rsum1;
                                    sumy -= ene * Lsum0 * Rsum2;
                                    sumz -= ene * Lsum0 * Rsum3;
                                }
                            }
                        }

                        Hx[spin][m][n] += sumx;
                        Hy[spin][m][n] += sumy;
                        Hz[spin][m][n] += sumz;
                    }
                }
            }
        }
    }

    /****************************************************
     freeing of arrays:

     double Hx2[3][List_YOUSO[7]][List_YOUSO[7]];
     double Hy2[3][List_YOUSO[7]][List_YOUSO[7]];
     double Hz2[3][List_YOUSO[7]][List_YOUSO[7]];
    ****************************************************/

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hx2[i][j]);
        }
        free(Hx2[i]);
    }
    free(Hx2);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hy2[i][j]);
        }
        free(Hy2[i]);
    }
    free(Hy2);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hz2[i][j]);
        }
        free(Hz2[i]);
    }
    free(Hz2);
}

void dH_U_NC_full(int Mc_AN, int h_AN, int q_AN,
    double***** OLP, dcomplex***** NC_v_eff,
    dcomplex**** Hx, dcomplex**** Hy, dcomplex**** Hz)
{
    int i, j, k, m, n, kg, kan, so, deri_kind, Mk_AN;
    int ig, ian, jg, jan, kl, kl1, kl2, spin;
    int wakg, l1, l2, l3, Gc_AN, Mi_AN, Mj_AN;
    int Rwan, Lwan, p, p0, s1, s2;
    double PF[2], sumx, sumy, sumz, ene;
    double tmpx, tmpy, tmpz;
    double Lsum0, Lsum1, Lsum2, Lsum3;
    double Rsum0, Rsum1, Rsum2, Rsum3;
    double PFp, PFm, ene_p, ene_m;
    double Re00x, Re00y, Re00z;
    double Re11x, Re11y, Re11z;
    double Re01x, Re01y, Re01z;
    double Re10x, Re10y, Re10z;
    double Im00x, Im00y, Im00z;
    double Im11x, Im11y, Im11z;
    double Im01x, Im01y, Im01z;
    double Im10x, Im10y, Im10z;

    /****************************************************
     start calc.
    ****************************************************/

    Gc_AN = M2G[Mc_AN];
    ig = natn[Gc_AN][h_AN];
    Lwan = WhatSpecies[ig];
    Mi_AN = F_G2M[ig]; /* F_G2M should be used */
    ian = Spe_Total_CNO[Lwan];
    jg = natn[Gc_AN][q_AN];
    Rwan = WhatSpecies[jg];
    Mj_AN = F_G2M[jg]; /* F_G2M should be used */
    jan = Spe_Total_CNO[Rwan];

    if (h_AN == 0) {

        /****************************************************
                              dS*ep*S
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl = RMI1[Mc_AN][q_AN][k];

            /****************************************************
                        derivative at h_AN (=Mc_AN)
            ****************************************************/

            if (0 <= kl) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        Re00x = 0.0;
                        Re00y = 0.0;
                        Re00z = 0.0;
                        Re11x = 0.0;
                        Re11y = 0.0;
                        Re11z = 0.0;
                        Re01x = 0.0;
                        Re01y = 0.0;
                        Re01z = 0.0;
                        Re10x = 0.0;
                        Re10y = 0.0;
                        Re10z = 0.0;

                        Im00x = 0.0;
                        Im00y = 0.0;
                        Im00z = 0.0;
                        Im11x = 0.0;
                        Im11y = 0.0;
                        Im11z = 0.0;
                        Im01x = 0.0;
                        Im01y = 0.0;
                        Im01z = 0.0;
                        Im10x = 0.0;
                        Im10y = 0.0;
                        Im10z = 0.0;

                        for (l1 = 0; l1 < kan; l1++) {
                            for (l2 = 0; l2 < kan; l2++) {

                                ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
                                Re00x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re00y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re00z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
                                Re11x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re11y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re11z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
                                Re01x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re01y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re01z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
                                Re10x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re10y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re10z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
                                Im00x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im00y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im00z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
                                Im11x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im11y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im11z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
                                Im01x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im01y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im01z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
                                Im10x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im10y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im10z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                            }
                        }

                        if (k == 0) {
                            Hx[0][0][m][n] = Complex(Re00x, Im00x);
                            Hy[0][0][m][n] = Complex(Re00y, Im00y);
                            Hz[0][0][m][n] = Complex(Re00z, Im00z);

                            Hx[1][1][m][n] = Complex(Re11x, Im11x);
                            Hy[1][1][m][n] = Complex(Re11y, Im11y);
                            Hz[1][1][m][n] = Complex(Re11z, Im11z);

                            Hx[0][1][m][n] = Complex(Re01x, Im01x);
                            Hy[0][1][m][n] = Complex(Re01y, Im01y);
                            Hz[0][1][m][n] = Complex(Re01z, Im01z);

                            Hx[1][0][m][n] = Complex(Re10x, Im10x);
                            Hy[1][0][m][n] = Complex(Re10y, Im10y);
                            Hz[1][0][m][n] = Complex(Re10z, Im10z);
                        } else {

                            Hx[0][0][m][n].r += Re00x;
                            Hx[0][0][m][n].i += Im00x;
                            Hy[0][0][m][n].r += Re00y;
                            Hy[0][0][m][n].i += Im00y;
                            Hz[0][0][m][n].r += Re00z;
                            Hz[0][0][m][n].i += Im00z;

                            Hx[1][1][m][n].r += Re11x;
                            Hx[1][1][m][n].i += Im11x;
                            Hy[1][1][m][n].r += Re11y;
                            Hy[1][1][m][n].i += Im11y;
                            Hz[1][1][m][n].r += Re11z;
                            Hz[1][1][m][n].i += Im11z;

                            Hx[0][1][m][n].r += Re01x;
                            Hx[0][1][m][n].i += Im01x;
                            Hy[0][1][m][n].r += Re01y;
                            Hy[0][1][m][n].i += Im01y;
                            Hz[0][1][m][n].r += Re01z;
                            Hz[0][1][m][n].i += Im01z;

                            Hx[1][0][m][n].r += Re10x;
                            Hx[1][0][m][n].i += Im10x;
                            Hy[1][0][m][n].r += Re10y;
                            Hy[1][0][m][n].i += Im10y;
                            Hz[1][0][m][n].r += Re10z;
                            Hz[1][0][m][n].i += Im10z;
                        }

                    } /* n */
                } /* m */
            } /* if */
        } /* k */

        /****************************************************
                              S*ep*dS
        ****************************************************/

        /* ????? */

        if (q_AN == 0) {

            for (s1 = 0; s1 < 2; s1++) {
                for (s2 = 0; s2 < 2; s2++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            Hx[s1][s2][m][n].r = 2.0 * Hx[s1][s2][m][n].r;
                            Hy[s1][s2][m][n].r = 2.0 * Hy[s1][s2][m][n].r;
                            Hz[s1][s2][m][n].r = 2.0 * Hz[s1][s2][m][n].r;

                            Hx[s1][s2][m][n].i = 2.0 * Hx[s1][s2][m][n].i;
                            Hy[s1][s2][m][n].i = 2.0 * Hy[s1][s2][m][n].i;
                            Hz[s1][s2][m][n].i = 2.0 * Hz[s1][s2][m][n].i;
                        }
                    }
                }
            }
        }

        else {

            kg = natn[Gc_AN][0];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl = RMI1[Mc_AN][q_AN][0];

            /****************************************************
                              derivative at k=0
            ****************************************************/

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    Re00x = 0.0;
                    Re00y = 0.0;
                    Re00z = 0.0;
                    Re11x = 0.0;
                    Re11y = 0.0;
                    Re11z = 0.0;
                    Re01x = 0.0;
                    Re01y = 0.0;
                    Re01z = 0.0;
                    Re10x = 0.0;
                    Re10y = 0.0;
                    Re10z = 0.0;

                    Im00x = 0.0;
                    Im00y = 0.0;
                    Im00z = 0.0;
                    Im11x = 0.0;
                    Im11y = 0.0;
                    Im11z = 0.0;
                    Im01x = 0.0;
                    Im01y = 0.0;
                    Im01z = 0.0;
                    Im10x = 0.0;
                    Im10y = 0.0;
                    Im10z = 0.0;

                    for (l1 = 0; l1 < kan; l1++) {
                        for (l2 = 0; l2 < kan; l2++) {

                            ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
                            Re00x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Re00y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Re00z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
                            Re11x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Re11y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Re11z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
                            Re01x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Re01y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Re01z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
                            Re10x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Re10y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Re10z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
                            Im00x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Im00y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Im00z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
                            Im11x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Im11y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Im11z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
                            Im01x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Im01y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Im01z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
                            Im10x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Im10y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Im10z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];
                        }
                    }

                    Hx[0][0][m][n].r += Re00x;
                    Hx[0][0][m][n].i += Im00x;
                    Hy[0][0][m][n].r += Re00y;
                    Hy[0][0][m][n].i += Im00y;
                    Hz[0][0][m][n].r += Re00z;
                    Hz[0][0][m][n].i += Im00z;

                    Hx[1][1][m][n].r += Re11x;
                    Hx[1][1][m][n].i += Im11x;
                    Hy[1][1][m][n].r += Re11y;
                    Hy[1][1][m][n].i += Im11y;
                    Hz[1][1][m][n].r += Re11z;
                    Hz[1][1][m][n].i += Im11z;

                    Hx[0][1][m][n].r += Re01x;
                    Hx[0][1][m][n].i += Im01x;
                    Hy[0][1][m][n].r += Re01y;
                    Hy[0][1][m][n].i += Im01y;
                    Hz[0][1][m][n].r += Re01z;
                    Hz[0][1][m][n].i += Im01z;

                    Hx[1][0][m][n].r += Re10x;
                    Hx[1][0][m][n].i += Im10x;
                    Hy[1][0][m][n].r += Re10y;
                    Hy[1][0][m][n].i += Im10y;
                    Hz[1][0][m][n].r += Re10z;
                    Hz[1][0][m][n].i += Im10z;
                }
            }
        }

    } /* if (h_AN==0) */

    else {

        /****************************************************
                               dS*ep*S
        ****************************************************/

        kg = natn[Gc_AN][0];
        Mk_AN = F_G2M[kg]; /* F_G2M should be used */
        wakg = WhatSpecies[kg];
        kan = Spe_Total_NO[wakg];
        kl1 = RMI1[Mc_AN][h_AN][0];
        kl2 = RMI1[Mc_AN][q_AN][0];

        for (m = 0; m < ian; m++) {
            for (n = 0; n < jan; n++) {

                Re00x = 0.0;
                Re00y = 0.0;
                Re00z = 0.0;
                Re11x = 0.0;
                Re11y = 0.0;
                Re11z = 0.0;
                Re01x = 0.0;
                Re01y = 0.0;
                Re01z = 0.0;
                Re10x = 0.0;
                Re10y = 0.0;
                Re10z = 0.0;

                Im00x = 0.0;
                Im00y = 0.0;
                Im00z = 0.0;
                Im11x = 0.0;
                Im11y = 0.0;
                Im11z = 0.0;
                Im01x = 0.0;
                Im01y = 0.0;
                Im01z = 0.0;
                Im10x = 0.0;
                Im10y = 0.0;
                Im10z = 0.0;

                for (l1 = 0; l1 < kan; l1++) {
                    for (l2 = 0; l2 < kan; l2++) {

                        ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
                        Re00x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re00y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re00z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
                        Re11x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re11y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re11z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
                        Re01x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re01y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re01z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
                        Re10x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re10y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re10z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
                        Im00x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im00y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im00z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
                        Im11x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im11y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im11z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
                        Im01x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im01y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im01z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
                        Im10x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im10y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im10z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                    }
                }

                Hx[0][0][m][n] = Complex(Re00x, Im00x);
                Hy[0][0][m][n] = Complex(Re00y, Im00y);
                Hz[0][0][m][n] = Complex(Re00z, Im00z);

                Hx[1][1][m][n] = Complex(Re11x, Im11x);
                Hy[1][1][m][n] = Complex(Re11y, Im11y);
                Hz[1][1][m][n] = Complex(Re11z, Im11z);

                Hx[0][1][m][n] = Complex(Re01x, Im01x);
                Hy[0][1][m][n] = Complex(Re01y, Im01y);
                Hz[0][1][m][n] = Complex(Re01z, Im01z);

                Hx[1][0][m][n] = Complex(Re10x, Im10x);
                Hy[1][0][m][n] = Complex(Re10y, Im10y);
                Hz[1][0][m][n] = Complex(Re10z, Im10z);
            }
        }

        /****************************************************
                               S*ep*dS
        ****************************************************/

        if (q_AN == 0) {

            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                kg = natn[Gc_AN][k];
                Mk_AN = F_G2M[kg]; /* F_G2M should be used */
                wakg = WhatSpecies[kg];
                kan = Spe_Total_NO[wakg];
                kl1 = RMI1[Mc_AN][h_AN][k];
                kl2 = RMI1[Mc_AN][q_AN][k];

                if (0 <= kl1) {

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            Re00x = 0.0;
                            Re00y = 0.0;
                            Re00z = 0.0;
                            Re11x = 0.0;
                            Re11y = 0.0;
                            Re11z = 0.0;
                            Re01x = 0.0;
                            Re01y = 0.0;
                            Re01z = 0.0;
                            Re10x = 0.0;
                            Re10y = 0.0;
                            Re10z = 0.0;

                            Im00x = 0.0;
                            Im00y = 0.0;
                            Im00z = 0.0;
                            Im11x = 0.0;
                            Im11y = 0.0;
                            Im11z = 0.0;
                            Im01x = 0.0;
                            Im01y = 0.0;
                            Im01z = 0.0;
                            Im10x = 0.0;
                            Im10y = 0.0;
                            Im10z = 0.0;

                            for (l1 = 0; l1 < kan; l1++) {
                                for (l2 = 0; l2 < kan; l2++) {

                                    ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
                                    Re00x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Re00y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Re00z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
                                    Re11x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Re11y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Re11z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
                                    Re01x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Re01y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Re01z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
                                    Re10x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Re10y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Re10z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
                                    Im00x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Im00y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Im00z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
                                    Im11x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Im11y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Im11z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
                                    Im01x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Im01y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Im01z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
                                    Im10x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Im10y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Im10z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];
                                }
                            }

                            Hx[0][0][m][n].r += Re00x;
                            Hx[0][0][m][n].i += Im00x;
                            Hy[0][0][m][n].r += Re00y;
                            Hy[0][0][m][n].i += Im00y;
                            Hz[0][0][m][n].r += Re00z;
                            Hz[0][0][m][n].i += Im00z;

                            Hx[1][1][m][n].r += Re11x;
                            Hx[1][1][m][n].i += Im11x;
                            Hy[1][1][m][n].r += Re11y;
                            Hy[1][1][m][n].i += Im11y;
                            Hz[1][1][m][n].r += Re11z;
                            Hz[1][1][m][n].i += Im11z;

                            Hx[0][1][m][n].r += Re01x;
                            Hx[0][1][m][n].i += Im01x;
                            Hy[0][1][m][n].r += Re01y;
                            Hy[0][1][m][n].i += Im01y;
                            Hz[0][1][m][n].r += Re01z;
                            Hz[0][1][m][n].i += Im01z;

                            Hx[1][0][m][n].r += Re10x;
                            Hx[1][0][m][n].i += Im10x;
                            Hy[1][0][m][n].r += Re10y;
                            Hy[1][0][m][n].i += Im10y;
                            Hz[1][0][m][n].r += Re10z;
                            Hz[1][0][m][n].i += Im10z;
                        }
                    }
                }
            }
        } /* if (q_AN==0) */

        else {

            kg = natn[Gc_AN][0];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl1 = RMI1[Mc_AN][h_AN][0];
            kl2 = RMI1[Mc_AN][q_AN][0];

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    Re00x = 0.0;
                    Re00y = 0.0;
                    Re00z = 0.0;
                    Re11x = 0.0;
                    Re11y = 0.0;
                    Re11z = 0.0;
                    Re01x = 0.0;
                    Re01y = 0.0;
                    Re01z = 0.0;
                    Re10x = 0.0;
                    Re10y = 0.0;
                    Re10z = 0.0;

                    Im00x = 0.0;
                    Im00y = 0.0;
                    Im00z = 0.0;
                    Im11x = 0.0;
                    Im11y = 0.0;
                    Im11z = 0.0;
                    Im01x = 0.0;
                    Im01y = 0.0;
                    Im01z = 0.0;
                    Im10x = 0.0;
                    Im10y = 0.0;
                    Im10z = 0.0;

                    for (l1 = 0; l1 < kan; l1++) {
                        for (l2 = 0; l2 < kan; l2++) {

                            ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
                            Re00x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Re00y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Re00z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
                            Re11x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Re11y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Re11z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
                            Re01x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Re01y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Re01z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
                            Re10x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Re10y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Re10z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
                            Im00x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Im00y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Im00z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
                            Im11x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Im11y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Im11z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
                            Im01x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Im01y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Im01z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
                            Im10x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Im10y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Im10z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];
                        }
                    }

                    Hx[0][0][m][n].r += Re00x;
                    Hx[0][0][m][n].i += Im00x;
                    Hy[0][0][m][n].r += Re00y;
                    Hy[0][0][m][n].i += Im00y;
                    Hz[0][0][m][n].r += Re00z;
                    Hz[0][0][m][n].i += Im00z;

                    Hx[1][1][m][n].r += Re11x;
                    Hx[1][1][m][n].i += Im11x;
                    Hy[1][1][m][n].r += Re11y;
                    Hy[1][1][m][n].i += Im11y;
                    Hz[1][1][m][n].r += Re11z;
                    Hz[1][1][m][n].i += Im11z;

                    Hx[0][1][m][n].r += Re01x;
                    Hx[0][1][m][n].i += Im01x;
                    Hy[0][1][m][n].r += Re01y;
                    Hy[0][1][m][n].i += Im01y;
                    Hz[0][1][m][n].r += Re01z;
                    Hz[0][1][m][n].i += Im01z;

                    Hx[1][0][m][n].r += Re10x;
                    Hx[1][0][m][n].i += Im10x;
                    Hy[1][0][m][n].r += Re10y;
                    Hy[1][0][m][n].i += Im10y;
                    Hz[1][0][m][n].r += Re10z;
                    Hz[1][0][m][n].i += Im10z;
                }
            }
        }
    }
}

void dHCH(int where_flag,
    int Mc_AN, int h_AN, int q_AN,
    double***** OLP1,
    dcomplex*** Hx, dcomplex*** Hy, dcomplex*** Hz)
{
    int i, j, k, m, n, l, kg, kan, so, deri_kind, L;
    int ig, ian, jg, jan, kl, kl1, kl2, mul;
    int wakg, l1, l2, l3, Gc_AN, Mi_AN, Mi_AN2, Mj_AN, Mj_AN2;
    int Rni, Rnj, somax, apply_flag, target_spin;
    double** penalty;
    double penalty_value;
    double PF[2], sumx[2], sumy[2], sumz[2], ene, dmp, deri_dmp;
    double tmpx, tmpy, tmpz, tmp, r;
    double x0, y0, z0, x1, y1, z1, dx, dy, dz;
    double rcuti, rcutj, rcut;
    double PFp, PFm, ene_p, ene_m;
    dcomplex sumx0, sumy0, sumz0;
    dcomplex sumx1, sumy1, sumz1;
    dcomplex sumx2, sumy2, sumz2;

    /****************************************************
     start calc.
    ****************************************************/

    /* set penalty */

    /* OpenMX 3.9.9 stores the already combined core-hole penalty. */
    penalty_value = penalty_value_CoreHole;

    penalty = (double**)malloc(sizeof(double*) * 2);
    for (i = 0; i < 2; i++) {
        penalty[i] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
    }

    wakg = WhatSpecies[Core_Hole_Atom];

    /* set penalty */

    if (VPS_j_dependency[wakg] == 0) {

        for (i = 0; i < Spe_Total_NO[wakg]; i++) {
            penalty[0][i] = 0.0;
            penalty[1][i] = 0.0;
        }

        L = 0;
        for (l = 0; l <= Spe_MaxL_Basis[wakg]; l++) {
            for (mul = 0; mul < Spe_Num_Basis[wakg][l]; mul++) {

                apply_flag = 0;

                if ((strcmp(Core_Hole_Orbital, "s") == 0) && l == 0 && mul == 0) {

                    if (Core_Hole_J == 1)
                        target_spin = 0;
                    else
                        target_spin = 1;

                    apply_flag = 1;
                }

                else if ((strcmp(Core_Hole_Orbital, "p") == 0) && l == 1 && mul == 0) {

                    if (Core_Hole_J <= 3)
                        target_spin = 0;
                    else
                        target_spin = 1;

                    apply_flag = 1;
                }

                else if ((strcmp(Core_Hole_Orbital, "d") == 0) && l == 2 && mul == 0) {

                    if (Core_Hole_J <= 5)
                        target_spin = 0;
                    else
                        target_spin = 1;

                    apply_flag = 1;
                }

                else if ((strcmp(Core_Hole_Orbital, "f") == 0) && l == 3 && mul == 0) {

                    if (Core_Hole_J <= 7)
                        target_spin = 0;
                    else
                        target_spin = 1;

                    apply_flag = 1;
                }

                /* set the penalty into all the states speficied by l and mul */

                if (apply_flag == 1 && Core_Hole_J == 0) {
                    for (i = 0; i < (2 * l + 1); i++) {
                        penalty[0][L + i] = penalty_value;
                        penalty[1][L + i] = penalty_value;
                    }
                }

                /* set the penalty into one of the states speficied by l and mul */

                else if (apply_flag == 1) {
                    penalty[target_spin][L + (Core_Hole_J - 1) % (2 * l + 1)] = penalty_value;
                }

                /* increment of L */

                L += 2 * l + 1;
            }
        }
    }

    /* get information of relevant atoms */

    Gc_AN = M2G[Mc_AN];
    ig = natn[Gc_AN][h_AN];
    Rni = ncn[Gc_AN][h_AN];
    Mi_AN = F_G2M[ig];
    ian = Spe_Total_CNO[WhatSpecies[ig]];
    rcuti = Spe_Atom_Cut1[WhatSpecies[ig]];

    jg = natn[Gc_AN][q_AN];
    Rnj = ncn[Gc_AN][q_AN];
    Mj_AN = F_G2M[jg];
    jan = Spe_Total_CNO[WhatSpecies[jg]];
    rcutj = Spe_Atom_Cut1[WhatSpecies[jg]];

    rcut = rcuti + rcutj;
    kl = RMI1[Mc_AN][h_AN][q_AN];
    dmp = dampingF(rcut, Dis[ig][kl]);

    for (so = 0; so < 3; so++) {
        for (i = 0; i < List_YOUSO[7]; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hx[so][i][j] = Complex(0.0, 0.0);
                Hy[so][i][j] = Complex(0.0, 0.0);
                Hz[so][i][j] = Complex(0.0, 0.0);
            }
        }
    }

    if (h_AN == 0) {

        /****************************************************
                              dH*ep*H
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            wakg = WhatSpecies[kg];
            kan = Spe_Total_CNO[wakg];
            kl = RMI1[Mc_AN][q_AN][k];

            /****************************************************
                         l-dependent non-local part
            ****************************************************/

            if (0 <= kl && VPS_j_dependency[wakg] == 0 && where_flag == 0) {

                if (Mj_AN <= Matomnum)
                    Mj_AN2 = Mj_AN;
                else
                    Mj_AN2 = Matomnum + 1;

                if (kg == Core_Hole_Atom) {

                    /* calculate the multiplication */

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx[0] = 0.0;
                            sumx[1] = 0.0;
                            sumy[0] = 0.0;
                            sumy[1] = 0.0;
                            sumz[0] = 0.0;
                            sumz[1] = 0.0;

                            for (i = 0; i < Spe_Total_CNO[wakg]; i++) {
                                sumx[0] += penalty[0][i] * OLP1[1][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];
                                sumy[0] += penalty[0][i] * OLP1[2][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];
                                sumz[0] += penalty[0][i] * OLP1[3][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];

                                sumx[1] += penalty[1][i] * OLP1[1][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];
                                sumy[1] += penalty[1][i] * OLP1[2][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];
                                sumz[1] += penalty[1][i] * OLP1[3][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];
                            }

                            Hx[0][m][n].r += sumx[0];
                            Hy[0][m][n].r += sumy[0];
                            Hz[0][m][n].r += sumz[0];

                            Hx[1][m][n].r += sumx[1];
                            Hy[1][m][n].r += sumy[1];
                            Hz[1][m][n].r += sumz[1];

                        } /* n */
                    } /* m */

                } /* if (kg==Core_Hole_Atom) */

            } /* if */

            /****************************************************
                         j-dependent non-local part
            ****************************************************/

            else if (0 <= kl && VPS_j_dependency[wakg] == 1 && where_flag == 0) {

                if (Mj_AN <= Matomnum)
                    Mj_AN2 = Mj_AN;
                else
                    Mj_AN2 = Matomnum + 1;

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);
                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);
                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                            &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                            &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                            1.0,
                            Mc_AN, k, m,
                            Mj_AN2, kl, n,
                            kg, wakg,
                            penalty_value,
                            OLP1);

                        if (q_AN == 0) {

                            dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                                &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                                &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                                -1.0,
                                Mj_AN2, kl, n,
                                Mc_AN, k, m,
                                kg, wakg,
                                penalty_value,
                                OLP1);
                        }

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }

        } /* k */

        /****************************************************
                               H*ep*dH
        ****************************************************/

        /* h_AN==0 && q_AN==0 */

        if (q_AN == 0 && VPS_j_dependency[wakg] == 0) {

            for (m = 0; m < ian; m++) {
                for (n = m; n < jan; n++) {

                    tmpx = Hx[0][m][n].r + Hx[0][n][m].r;
                    Hx[0][m][n].r = tmpx;
                    Hx[0][n][m].r = tmpx;

                    tmpy = Hy[0][m][n].r + Hy[0][n][m].r;
                    Hy[0][m][n].r = tmpy;
                    Hy[0][n][m].r = tmpy;

                    tmpz = Hz[0][m][n].r + Hz[0][n][m].r;
                    Hz[0][m][n].r = tmpz;
                    Hz[0][n][m].r = tmpz;

                    tmpx = Hx[1][m][n].r + Hx[1][n][m].r;
                    Hx[1][m][n].r = tmpx;
                    Hx[1][n][m].r = tmpx;

                    tmpy = Hy[1][m][n].r + Hy[1][n][m].r;
                    Hy[1][m][n].r = tmpy;
                    Hy[1][n][m].r = tmpy;

                    tmpz = Hz[1][m][n].r + Hz[1][n][m].r;
                    Hz[1][m][n].r = tmpz;
                    Hz[1][n][m].r = tmpz;
                }
            }
        }

        else if (where_flag == 1) {

            kg = natn[Gc_AN][0];
            wakg = WhatSpecies[kg];
            kan = Spe_Total_VPS_Pro[wakg];
            kl = RMI1[Mc_AN][q_AN][0];

            /****************************************************
                         l-dependent non-local part
            ****************************************************/

            if (VPS_j_dependency[wakg] == 0) {

                if (Mj_AN <= Matomnum) {
                    Mj_AN2 = Mj_AN;
                    kl2 = RMI1[Mc_AN][q_AN][0];
                } else {
                    Mj_AN2 = Matomnum + 1;
                    kl2 = RMI1[Mc_AN][0][q_AN];
                }

                if (kg == Core_Hole_Atom) {

                    /* calculate the multiplication */

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx[0] = 0.0;
                            sumx[1] = 0.0;
                            sumy[0] = 0.0;
                            sumy[1] = 0.0;
                            sumz[0] = 0.0;
                            sumz[1] = 0.0;

                            for (i = 0; i < Spe_Total_CNO[wakg]; i++) {
                                sumx[0] -= penalty[0][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[1][Mj_AN2][kl2][n][i];
                                sumy[0] -= penalty[0][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[2][Mj_AN2][kl2][n][i];
                                sumz[0] -= penalty[0][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[3][Mj_AN2][kl2][n][i];

                                sumx[1] -= penalty[1][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[1][Mj_AN2][kl2][n][i];
                                sumy[1] -= penalty[1][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[2][Mj_AN2][kl2][n][i];
                                sumz[1] -= penalty[1][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[3][Mj_AN2][kl2][n][i];
                            }

                            Hx[0][m][n].r += sumx[0];
                            Hy[0][m][n].r += sumy[0];
                            Hz[0][m][n].r += sumz[0];

                            Hx[1][m][n].r += sumx[1];
                            Hy[1][m][n].r += sumy[1];
                            Hz[1][m][n].r += sumz[1];
                        }
                    }

                } /* if (kg==Core_Hole_Atom) */
            }

            /****************************************************
                         j-dependent non-local part
            ****************************************************/

            else if (VPS_j_dependency[wakg] == 1) {

                if (Mj_AN <= Matomnum) {
                    Mj_AN2 = Mj_AN;
                    kl2 = RMI1[Mc_AN][q_AN][0];
                } else {
                    Mj_AN2 = Matomnum + 1;
                    kl2 = RMI1[Mc_AN][0][q_AN];
                }

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);
                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);
                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        /* 1 */

                        dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                            &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                            &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                            -1.0,
                            Mj_AN2, kl2, n,
                            Mc_AN, 0, m,
                            kg, wakg,
                            -penalty_value,
                            OLP1);

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }
        }

    } /* if (h_AN==0) */

    else if (where_flag == 0) {

        /****************************************************
           H*ep*dH

           if (h_AN!=0 && where_flag==0)
           This happens
           only if
           ( SpinP_switch==3
             &&
             (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1)
               || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1
               || Zeeman_NCO_switch==1)
             &&
             q_AN==0
           )
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            wakg = WhatSpecies[kg];
            kan = Spe_Total_VPS_Pro[wakg];
            kl = RMI1[Mc_AN][h_AN][k];

            if (Mi_AN <= Matomnum)
                Mi_AN2 = Mi_AN;
            else
                Mi_AN2 = Matomnum + 1;

            if (0 <= kl && VPS_j_dependency[wakg] == 1) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);
                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);
                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                            &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                            &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                            -1.0,
                            Mj_AN, k, n,
                            Mi_AN2, kl, m,
                            kg, wakg,
                            penalty_value,
                            OLP1);

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }
        }

    }

    /* if (h_AN!=0 && where_flag==1) */

    else {

        /****************************************************
                               dH*ep*H
        ****************************************************/

        kg = natn[Gc_AN][0];
        wakg = WhatSpecies[kg];
        kan = Spe_Total_VPS_Pro[wakg];
        kl1 = RMI1[Mc_AN][0][h_AN];
        kl2 = RMI1[Mc_AN][0][q_AN];

        /****************************************************
                       l-dependent non-local part
        ****************************************************/

        if (VPS_j_dependency[wakg] == 0) {

            if (kg == Core_Hole_Atom) {

                /* calculate the multiplication */

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx[0] = 0.0;
                        sumx[1] = 0.0;
                        sumy[0] = 0.0;
                        sumy[1] = 0.0;
                        sumz[0] = 0.0;
                        sumz[1] = 0.0;

                        for (i = 0; i < Spe_Total_CNO[wakg]; i++) {

                            sumx[0] -= penalty[0][i] * OLP1[1][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];
                            sumy[0] -= penalty[0][i] * OLP1[2][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];
                            sumz[0] -= penalty[0][i] * OLP1[3][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];

                            sumx[1] -= penalty[1][i] * OLP1[1][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];
                            sumy[1] -= penalty[1][i] * OLP1[2][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];
                            sumz[1] -= penalty[1][i] * OLP1[3][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];
                        }

                        Hx[0][m][n].r = sumx[0];
                        Hy[0][m][n].r = sumy[0];
                        Hz[0][m][n].r = sumz[0];

                        Hx[1][m][n].r = sumx[1];
                        Hy[1][m][n].r = sumy[1];
                        Hz[1][m][n].r = sumz[1];

                        Hx[2][m][n].r = 0.0;
                        Hy[2][m][n].r = 0.0;
                        Hz[2][m][n].r = 0.0;

                        Hx[0][m][n].i = 0.0;
                        Hy[0][m][n].i = 0.0;
                        Hz[0][m][n].i = 0.0;

                        Hx[1][m][n].i = 0.0;
                        Hy[1][m][n].i = 0.0;
                        Hz[1][m][n].i = 0.0;

                        Hx[2][m][n].i = 0.0;
                        Hy[2][m][n].i = 0.0;
                        Hz[2][m][n].i = 0.0;
                    }
                }
            }
        }

        /****************************************************
                     j-dependent non-local part
        ****************************************************/

        else if (VPS_j_dependency[wakg] == 1) {

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    sumx0 = Complex(0.0, 0.0);
                    sumy0 = Complex(0.0, 0.0);
                    sumz0 = Complex(0.0, 0.0);
                    sumx1 = Complex(0.0, 0.0);
                    sumy1 = Complex(0.0, 0.0);
                    sumz1 = Complex(0.0, 0.0);
                    sumx2 = Complex(0.0, 0.0);
                    sumy2 = Complex(0.0, 0.0);
                    sumz2 = Complex(0.0, 0.0);

                    /* 2 */

                    dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                        &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                        &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                        1.0,
                        Matomnum + 1, kl1, m,
                        Matomnum + 1, kl2, n,
                        kg, wakg,
                        -penalty_value,
                        OLP1);

                    Hx[0][m][n].r = sumx0.r; /* up-up */
                    Hy[0][m][n].r = sumy0.r; /* up-up */
                    Hz[0][m][n].r = sumz0.r; /* up-up */

                    Hx[1][m][n].r = sumx1.r; /* dn-dn */
                    Hy[1][m][n].r = sumy1.r; /* dn-dn */
                    Hz[1][m][n].r = sumz1.r; /* dn-dn */

                    Hx[2][m][n].r = sumx2.r; /* up-dn */
                    Hy[2][m][n].r = sumy2.r; /* up-dn */
                    Hz[2][m][n].r = sumz2.r; /* up-dn */

                    Hx[0][m][n].i = sumx0.i; /* up-up */
                    Hy[0][m][n].i = sumy0.i; /* up-up */
                    Hz[0][m][n].i = sumz0.i; /* up-up */

                    Hx[1][m][n].i = sumx1.i; /* dn-dn */
                    Hy[1][m][n].i = sumy1.i; /* dn-dn */
                    Hz[1][m][n].i = sumz1.i; /* dn-dn */

                    Hx[2][m][n].i = sumx2.i; /* up-dn */
                    Hy[2][m][n].i = sumy2.i; /* up-dn */
                    Hz[2][m][n].i = sumz2.i; /* up-dn */
                }
            }
        }

        /****************************************************
                               H*ep*dH
        ****************************************************/

        if (q_AN != 0) {

            kg = natn[Gc_AN][0];
            wakg = WhatSpecies[kg];
            kan = Spe_Total_VPS_Pro[wakg];
            kl1 = RMI1[Mc_AN][0][h_AN];
            kl2 = RMI1[Mc_AN][0][q_AN];

            /****************************************************
                           l-dependent non-local part
            ****************************************************/

            if (VPS_j_dependency[wakg] == 0) {

                if (kg == Core_Hole_Atom) {

                    /* calculate the multiplication */

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx[0] = 0.0;
                            sumx[1] = 0.0;
                            sumy[0] = 0.0;
                            sumy[1] = 0.0;
                            sumz[0] = 0.0;
                            sumz[1] = 0.0;

                            for (i = 0; i < Spe_Total_CNO[wakg]; i++) {

                                sumx[0] -= penalty[0][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[1][Matomnum + 1][kl2][n][i];
                                sumy[0] -= penalty[0][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[2][Matomnum + 1][kl2][n][i];
                                sumz[0] -= penalty[0][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[3][Matomnum + 1][kl2][n][i];

                                sumx[1] -= penalty[1][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[1][Matomnum + 1][kl2][n][i];
                                sumy[1] -= penalty[1][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[2][Matomnum + 1][kl2][n][i];
                                sumz[1] -= penalty[1][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[3][Matomnum + 1][kl2][n][i];
                            }

                            Hx[0][m][n].r += sumx[0];
                            Hy[0][m][n].r += sumy[0];
                            Hz[0][m][n].r += sumz[0];

                            Hx[1][m][n].r += sumx[1];
                            Hy[1][m][n].r += sumy[1];
                            Hz[1][m][n].r += sumz[1];
                        }
                    }
                }
            }

            /****************************************************
                          j-dependent non-local part
            ****************************************************/

            else if (VPS_j_dependency[wakg] == 1) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);
                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);
                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        /* 4 */

                        dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                            &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                            &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                            -1.0,
                            Matomnum + 1, kl2, n,
                            Matomnum + 1, kl1, m,
                            kg, wakg,
                            -penalty_value,
                            OLP1);

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }
        }

    } /* else */

    /****************************************************
                 contribution by dampingF
    ****************************************************/

    /* Qij * dH/dx  */

    for (so = 0; so < 3; so++) {
        for (m = 0; m < ian; m++) {
            for (n = 0; n < jan; n++) {

                Hx[so][m][n].r = dmp * Hx[so][m][n].r;
                Hy[so][m][n].r = dmp * Hy[so][m][n].r;
                Hz[so][m][n].r = dmp * Hz[so][m][n].r;

                Hx[so][m][n].i = dmp * Hx[so][m][n].i;
                Hy[so][m][n].i = dmp * Hy[so][m][n].i;
                Hz[so][m][n].i = dmp * Hz[so][m][n].i;
            }
        }
    }

    /* dQij/dx * H */

    if ((h_AN == 0 && q_AN != 0) || (h_AN != 0 && q_AN == 0)) {

        if (h_AN == 0)
            kl = q_AN;
        else if (q_AN == 0)
            kl = h_AN;

        if (SpinP_switch == 0)
            somax = 0;
        else if (SpinP_switch == 1)
            somax = 1;
        else if (SpinP_switch == 3)
            somax = 2;

        r = Dis[Gc_AN][kl];

        if (rcut <= r) {
            deri_dmp = 0.0;
            tmp = 0.0;
        } else {
            deri_dmp = deri_dampingF(rcut, r);
            tmp = deri_dmp / dmp;
        }

        x0 = Gxyz[ig][1] + atv[Rni][1];
        y0 = Gxyz[ig][2] + atv[Rni][2];
        z0 = Gxyz[ig][3] + atv[Rni][3];

        x1 = Gxyz[jg][1] + atv[Rnj][1];
        y1 = Gxyz[jg][2] + atv[Rnj][2];
        z1 = Gxyz[jg][3] + atv[Rnj][3];

        /* for empty atoms or finite elemens basis */
        if (r < 1.0e-10)
            r = 1.0e-10;

        if (h_AN == 0 && q_AN != 0) {
            dx = tmp * (x0 - x1) / r;
            dy = tmp * (y0 - y1) / r;
            dz = tmp * (z0 - z1) / r;
        }

        else if (h_AN != 0 && q_AN == 0) {
            dx = tmp * (x1 - x0) / r;
            dy = tmp * (y1 - y0) / r;
            dz = tmp * (z1 - z0) / r;
        }

        if (SpinP_switch == 0 || SpinP_switch == 1) {

            if (h_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dx;
                            Hy[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dy;
                            Hz[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dz;
                        }
                    }
                }
            }

            else if (q_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dx;
                            Hy[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dy;
                            Hz[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dz;
                        }
                    }
                }
            }
        }

        else if (SpinP_switch == 3) {

            if (h_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dx;
                            Hy[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dy;
                            Hz[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dz;
                        }
                    }
                }
            }

            else if (q_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dx;
                            Hy[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dy;
                            Hz[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dz;
                        }
                    }
                }
            }

            if (SO_switch == 1) {

                if (h_AN == 0) {
                    for (so = 0; so <= somax; so++) {
                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < jan; n++) {
                                Hx[so][m][n].i += iHCH[so][Mc_AN][kl][m][n] * dx;
                                Hy[so][m][n].i += iHCH[so][Mc_AN][kl][m][n] * dy;
                                Hz[so][m][n].i += iHCH[so][Mc_AN][kl][m][n] * dz;
                            }
                        }
                    }
                }

                else if (q_AN == 0) {
                    for (so = 0; so <= somax; so++) {
                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < jan; n++) {
                                Hx[so][m][n].i += iHCH[so][Mc_AN][kl][n][m] * dx;
                                Hy[so][m][n].i += iHCH[so][Mc_AN][kl][n][m] * dy;
                                Hz[so][m][n].i += iHCH[so][Mc_AN][kl][n][m] * dz;
                            }
                        }
                    }
                }
            }
        }
    }

    /* freeing of array */

    for (i = 0; i < 2; i++) {
        free(penalty[i]);
    }
    free(penalty);
}

void dHCH_SO(double* sumx0r, double* sumx0i, double* sumy0r, double* sumy0i, double* sumz0r, double* sumz0i,
    double* sumx1r, double* sumx1i, double* sumy1r, double* sumy1i, double* sumz1r, double* sumz1i,
    double* sumx2r, double* sumx2i, double* sumy2r, double* sumy2i, double* sumz2r, double* sumz2i,
    double fugou,
    int Mc_AN, int k, int m,
    int Mj_AN, int kl, int n,
    int kg, int wakg,
    double penalty_value,
    double***** OLP1)
{
    int L, L2, l, mul, apply_flag;
    double d12_m12, d12_p12;
    double d32_m12, d32_p12, d32_m32, d32_p32;
    double d52_m12, d52_p12, d52_m32, d52_p32, d52_m52, d52_p52;
    double d72_m12, d72_p12, d72_m32, d72_p32, d72_m52, d72_p52, d72_m72, d72_p72;

    if (kg != Core_Hole_Atom)
        return;

    /****************************************************
                   set penalty coefficients
    ****************************************************/

    L = 0;
    for (l = 0; l <= Spe_MaxL_Basis[wakg]; l++) {
        for (mul = 0; mul < Spe_Num_Basis[wakg][l]; mul++) {

            apply_flag = 0;

            if ((strcmp(Core_Hole_Orbital, "s") == 0) && l == 0 && mul == 0) {

                L2 = 0;
                apply_flag = 1;
            }

            else if ((strcmp(Core_Hole_Orbital, "p") == 0) && l == 1 && mul == 0) {

                L2 = 2;
                apply_flag = 1;
            }

            else if ((strcmp(Core_Hole_Orbital, "d") == 0) && l == 2 && mul == 0) {

                L2 = 4;
                apply_flag = 1;
            }

            else if ((strcmp(Core_Hole_Orbital, "f") == 0) && l == 3 && mul == 0) {

                L2 = 6;
                apply_flag = 1;
            }

            if (apply_flag == 1) {

                /****************************************************
                         set coefficients related to penalty
                ****************************************************/

                if (L2 == 0) {

                    if (Core_Hole_J == 0) {
                        /* for Core_Hole_J==0 */
                        d12_p12 = penalty_value;
                        d12_m12 = penalty_value;
                    }

                    else {
                        /* for Core_Hole_J!=0 */
                        d12_p12 = penalty_value * (Core_Hole_J == 1);
                        d12_m12 = penalty_value * (Core_Hole_J == 2);
                    }
                }

                else if (L2 == 2) {

                    if (Core_Hole_J == 0) {
                        /* for Core_Hole_J==0 */
                        d32_p32 = penalty_value / 3.0;
                        d32_p12 = penalty_value / 3.0;
                        d32_m12 = penalty_value / 3.0;
                        d32_m32 = penalty_value / 3.0;

                        d12_p12 = penalty_value / 3.0;
                        d12_m12 = penalty_value / 3.0;
                    }

                    else {
                        /* for Core_Hole_J!=0 */
                        d32_p32 = penalty_value / 3.0 * (Core_Hole_J == 1);
                        d32_p12 = penalty_value / 3.0 * (Core_Hole_J == 2);
                        d32_m12 = penalty_value / 3.0 * (Core_Hole_J == 3);
                        d32_m32 = penalty_value / 3.0 * (Core_Hole_J == 4);

                        d12_p12 = penalty_value / 3.0 * (Core_Hole_J == 5);
                        d12_m12 = penalty_value / 3.0 * (Core_Hole_J == 6);
                    }
                }

                else if (L2 == 4) {

                    if (Core_Hole_J == 0) {
                        /* for Core_Hole_J==0 */
                        d52_p52 = penalty_value / 5.0;
                        d52_p32 = penalty_value / 5.0;
                        d52_p12 = penalty_value / 5.0;
                        d52_m12 = penalty_value / 5.0;
                        d52_m32 = penalty_value / 5.0;
                        d52_m52 = penalty_value / 5.0;

                        d32_p32 = penalty_value / 5.0;
                        d32_p12 = penalty_value / 5.0;
                        d32_m12 = penalty_value / 5.0;
                        d32_m32 = penalty_value / 5.0;
                    }

                    else {
                        /* for Core_Hole_J!=0 */
                        d52_p52 = penalty_value / 5.0 * (Core_Hole_J == 1);
                        d52_p32 = penalty_value / 5.0 * (Core_Hole_J == 2);
                        d52_p12 = penalty_value / 5.0 * (Core_Hole_J == 3);
                        d52_m12 = penalty_value / 5.0 * (Core_Hole_J == 4);
                        d52_m32 = penalty_value / 5.0 * (Core_Hole_J == 5);
                        d52_m52 = penalty_value / 5.0 * (Core_Hole_J == 6);

                        d32_p32 = penalty_value / 5.0 * (Core_Hole_J == 7);
                        d32_p12 = penalty_value / 5.0 * (Core_Hole_J == 8);
                        d32_m12 = penalty_value / 5.0 * (Core_Hole_J == 9);
                        d32_m32 = penalty_value / 5.0 * (Core_Hole_J == 10);
                    }
                }

                else if (L2 == 6) {

                    if (Core_Hole_J == 0) {
                        /* for Core_Hole_J==0 */
                        d72_p72 = penalty_value / 7.0;
                        d72_p52 = penalty_value / 7.0;
                        d72_p32 = penalty_value / 7.0;
                        d72_p12 = penalty_value / 7.0;
                        d72_m12 = penalty_value / 7.0;
                        d72_m32 = penalty_value / 7.0;
                        d72_m52 = penalty_value / 7.0;
                        d72_m72 = penalty_value / 7.0;

                        d52_p52 = penalty_value / 7.0;
                        d52_p32 = penalty_value / 7.0;
                        d52_p12 = penalty_value / 7.0;
                        d52_m12 = penalty_value / 7.0;
                        d52_m32 = penalty_value / 7.0;
                        d52_m52 = penalty_value / 7.0;
                    }

                    else {
                        /* for Core_Hole_J!=0 */
                        d72_p72 = penalty_value / 7.0 * (Core_Hole_J == 1);
                        d72_p52 = penalty_value / 7.0 * (Core_Hole_J == 2);
                        d72_p32 = penalty_value / 7.0 * (Core_Hole_J == 3);
                        d72_p12 = penalty_value / 7.0 * (Core_Hole_J == 4);
                        d72_m12 = penalty_value / 7.0 * (Core_Hole_J == 5);
                        d72_m32 = penalty_value / 7.0 * (Core_Hole_J == 6);
                        d72_m52 = penalty_value / 7.0 * (Core_Hole_J == 7);
                        d72_m72 = penalty_value / 7.0 * (Core_Hole_J == 8);

                        d52_p52 = penalty_value / 7.0 * (Core_Hole_J == 9);
                        d52_p32 = penalty_value / 7.0 * (Core_Hole_J == 10);
                        d52_p12 = penalty_value / 7.0 * (Core_Hole_J == 11);
                        d52_m12 = penalty_value / 7.0 * (Core_Hole_J == 12);
                        d52_m32 = penalty_value / 7.0 * (Core_Hole_J == 13);
                        d52_m52 = penalty_value / 7.0 * (Core_Hole_J == 14);
                    }
                }

                /****************************************************
                          off-diagonal contribution on up-dn
                             for spin non-collinear
                ****************************************************/

                if (SpinP_switch == 3) {

                    /***************
                           p
                    ***************/

                    if (L2 == 2) {

                        /* real contribution of l+1/2 to off-diagonal up-down matrix */
                        *sumx2r += fugou * (d32_m12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d32_p12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        *sumy2r += fugou * (d32_m12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d32_p12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        *sumz2r += fugou * (d32_m12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d32_p12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        /* imaginary contribution of l+1/2 to off-diagonal up-down matrix */
                        *sumx2i += fugou * (-d32_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d32_p12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);

                        *sumx2i += fugou * (-d32_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d32_p12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);

                        *sumx2i += fugou * (-d32_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d32_p12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);

                        /* real contribution of l-1/2 for to off-diagonal up-down matrix */
                        *sumx2r -= fugou * (d12_m12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d12_p12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        *sumy2r -= fugou * (d12_m12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d12_p12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        *sumz2r -= fugou * (d12_m12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d12_p12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        /* imaginary contribution of l-1/2 to off-diagonal up-down matrix */
                        *sumx2i -= fugou * (-d12_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d12_p12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);

                        *sumy2i -= fugou * (-d12_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d12_p12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);

                        *sumz2i -= fugou * (-d12_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d12_p12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);
                    }

                    /***************
                           d
                    ***************/

                    else if (L2 == 4) {

                        /* real contribution of l+1/2 to off diagonal up-down matrix */

                        *sumx2r += fugou * (-sqrt(3.0) * d52_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d52_m12 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d52_p32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d52_m32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumy2r += fugou * (-sqrt(3.0) * d52_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d52_m12 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d52_p32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d52_m32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumz2r += fugou * (-sqrt(3.0) * d52_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d52_m12 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d52_p32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d52_m32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

                        *sumx2i += fugou * (sqrt(3.0) * d52_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d52_m12 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d52_m32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d52_p32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumy2i += fugou * (sqrt(3.0) * d52_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d52_m12 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d52_m32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d52_p32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumz2i += fugou * (sqrt(3.0) * d52_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d52_m12 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d52_m32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d52_p32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        /* real contribution of l-1/2 for to diagonal up-down matrix */

                        *sumx2r -= fugou * (-sqrt(3.0) * d32_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d32_m12 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d32_p32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d32_m32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumy2r -= fugou * (-sqrt(3.0) * d32_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d32_m12 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d32_p32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d32_m32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumz2r -= fugou * (-sqrt(3.0) * d32_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d32_m12 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d32_p32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d32_m32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

                        *sumx2i -= fugou * (sqrt(3.0) * d32_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d32_m12 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d32_m32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d32_p32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumy2i -= fugou * (sqrt(3.0) * d32_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d32_m12 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d32_m32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d32_p32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumz2i -= fugou * (sqrt(3.0) * d32_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d32_m12 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d32_m32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d32_p32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);
                    }

                    /***************
                                   f
                    ***************/

                    else if (L2 == 6) {

                        /* real contribution of l+1/2 to off diagonal up-down matrix */

                        *sumx2r += fugou * (-sqrt(6.0) * d72_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d72_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d72_p32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d72_m32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d72_p52 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d72_m52 * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumy2r += fugou * (-sqrt(6.0) * d72_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d72_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d72_p32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d72_m32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d72_p52 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d72_m52 * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumz2r += fugou * (-sqrt(6.0) * d72_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d72_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d72_p32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d72_m32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d72_p52 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d72_m52 * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

                        *sumx2i += fugou * (sqrt(6.0) * d72_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d72_m12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d72_p32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d72_m32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d72_p52 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d72_m52 * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumy2i += fugou * (sqrt(6.0) * d72_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d72_m12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d72_p32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d72_m32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d72_p52 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d72_m52 * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumz2i += fugou * (sqrt(6.0) * d72_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d72_m12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d72_p32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d72_m32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d72_p52 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d72_m52 * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        /* real contribution of l-1/2 for to off-diagonal up-down matrix */

                        *sumx2r -= fugou * (-sqrt(6.0) * d52_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d52_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d52_p32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d52_m32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d52_p52 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d52_m52 * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumy2r -= fugou * (-sqrt(6.0) * d52_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d52_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d52_p32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d52_m32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d52_p52 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d52_m52 * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumz2r -= fugou * (-sqrt(6.0) * d52_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d52_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d52_p32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d52_m32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d52_p52 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d52_m52 * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

                        *sumx2i -= fugou * (sqrt(6.0) * d52_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d52_m12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d52_p32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d52_m32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d52_p52 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d52_m52 * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumy2i -= fugou * (sqrt(6.0) * d52_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d52_m12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d52_p32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d52_m32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d52_p52 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d52_m52 * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumz2i -= fugou * (sqrt(6.0) * d52_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d52_m12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d52_p32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d52_m32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d52_p52 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d52_m52 * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);
                    }

                } /* if (SpinP_switch==3) */

                /****************************************************
                    off-diagonal contribution on up-up and dn-dn
                ****************************************************/

                /* p */

                if (L2 == 2) {

                    /* contribution of l+1/2 for up spin */

                    *sumx0i += 0.5 * fugou * (+(d32_m12 - 3.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (-d32_m12 + 3.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumy0i += 0.5 * fugou * (+(d32_m12 - 3.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (-d32_m12 + 3.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumz0i += 0.5 * fugou * (+(d32_m12 - 3.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (-d32_m12 + 3.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    /* contribution of l+1/2 for down spin */

                    *sumx1i += 0.5 * fugou * (+(-d32_p12 + 3.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (+d32_p12 - 3.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumy1i += 0.5 * fugou * (+(-d32_p12 + 3.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (+d32_p12 - 3.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumz1i += 0.5 * fugou * (+(-d32_p12 + 3.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (+d32_p12 - 3.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    /* contribution of l-1/2 for up spin */

                    *sumx0i += fugou * (d12_m12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] - d12_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumy0i += fugou * (d12_m12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] - d12_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumz0i += fugou * (d12_m12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] - d12_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    /* contribution of l-1/2 for down spin */

                    *sumx1i += fugou * (-d12_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + d12_p12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumy1i += fugou * (-d12_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + d12_p12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumz1i += fugou * (-d12_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + d12_p12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);
                }

                /* d */

                else if (L2 == 4) {

                    /* contribution of l+1/2 for up spin */

                    *sumx0i += 0.5 * fugou * (+(d52_m32 - 5.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-d52_m32 + 5.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d52_m12 - 4.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumy0i += 0.5 * fugou * (+(d52_m32 - 5.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-d52_m32 + 5.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d52_m12 - 4.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumz0i += 0.5 * fugou * (+(d52_m32 - 5.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-d52_m32 + 5.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d52_m12 - 4.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    /* contribution of l+1/2 for down spin */

                    *sumx1i += 0.5 * fugou * (-(d52_p32 - 5.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-d52_p32 + 5.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d52_p12 - 4.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumy1i += 0.5 * fugou * (-(d52_p32 - 5.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-d52_p32 + 5.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d52_p12 - 4.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumz1i += 0.5 * fugou * (-(d52_p32 - 5.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-d52_p32 + 5.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d52_p12 - 4.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    /* contribution of l-1/2 for up spin */

                    *sumx0i += 0.5 * fugou * ((4.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (3.0 * d32_m12 - d32_p32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-3.0 * d32_m12 + d32_p32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumy0i += 0.5 * fugou * ((4.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (3.0 * d32_m12 - d32_p32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-3.0 * d32_m12 + d32_p32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumz0i += 0.5 * fugou * ((4.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (3.0 * d32_m12 - d32_p32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-3.0 * d32_m12 + d32_p32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    /* contribution of l-1/2 for down spin */

                    *sumx1i += 0.5 * fugou * ((-4.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (+4.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (3.0 * d32_p12 - d32_m32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-3.0 * d32_p12 + d32_m32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumy1i += 0.5 * fugou * ((-4.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (+4.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (3.0 * d32_p12 - d32_m32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-3.0 * d32_p12 + d32_m32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumz1i += 0.5 * fugou * ((-4.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (+4.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (3.0 * d32_p12 - d32_m32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-3.0 * d32_p12 + d32_m32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);
                }

                /* f */

                else if (L2 == 6) {

                    /* contribution of l+1/2 for up spin */

                    *sumx0i += 0.5 * fugou * ((3.0 * d72_m12 - 5.0 * d72_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d72_m32 - 6.0 * d72_p52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (1.0 * d72_m52 - 7.0 * d72_p72) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumy0i += 0.5 * fugou * ((3.0 * d72_m12 - 5.0 * d72_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d72_m32 - 6.0 * d72_p52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (1.0 * d72_m52 - 7.0 * d72_p72) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumz0i += 0.5 * fugou * ((3.0 * d72_m12 - 5.0 * d72_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d72_m32 - 6.0 * d72_p52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (1.0 * d72_m52 - 7.0 * d72_p72) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    /* contribution of l+1/2 for down spin */

                    *sumx1i += 0.5 * fugou * (-(3.0 * d72_p12 - 5.0 * d72_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d72_p32 - 6.0 * d72_m52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (1.0 * d72_p52 - 7.0 * d72_m72) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumy1i += 0.5 * fugou * (-(3.0 * d72_p12 - 5.0 * d72_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d72_p32 - 6.0 * d72_m52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (1.0 * d72_p52 - 7.0 * d72_m72) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumz1i += 0.5 * fugou * (-(3.0 * d72_p12 - 5.0 * d72_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d72_p32 - 6.0 * d72_m52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (1.0 * d72_p52 - 7.0 * d72_m72) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    /* contribution of l-1/2 for up spin */

                    *sumx0i += 0.5 * fugou * ((4.0 * d52_m12 - 2.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (5.0 * d52_m32 - 1.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (+6.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-6.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumy0i += 0.5 * fugou * ((4.0 * d52_m12 - 2.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (5.0 * d52_m32 - 1.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (+6.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-6.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumz0i += 0.5 * fugou * ((4.0 * d52_m12 - 2.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (5.0 * d52_m32 - 1.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (+6.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-6.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    /* contribution of l-1/2 for down spin */

                    *sumx1i += 0.5 * fugou * (-(4.0 * d52_p12 - 2.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (5.0 * d52_p32 - 1.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (+6.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-6.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumy1i += 0.5 * fugou * (-(4.0 * d52_p12 - 2.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (5.0 * d52_p32 - 1.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (+6.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-6.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumz1i += 0.5 * fugou * (-(4.0 * d52_p12 - 2.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (5.0 * d52_p32 - 1.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (+6.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-6.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);
                }

                /****************************************************
                     diagonal contribution on up-up and dn-dn
                ****************************************************/

                /* s */

                if (L2 == 0) {

                    /* VNL for j=l+1/2 */

                    *sumx0r += d12_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += d12_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += d12_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];

                    *sumx1r += d12_m12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += d12_m12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += d12_m12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];

                    /* note that VNL for j=l-1/2 is zero */
                }

                /* p */

                else if (L2 == 2) {

                    /* VNL for j=l+1/2 */
                    *sumx0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += 0.5 * (4.0 * d32_p12) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumy0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += 0.5 * (4.0 * d32_p12) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumz0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += 0.5 * (4.0 * d32_p12) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumx1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += 0.5 * (4.0 * d32_m12) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumy1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += 0.5 * (4.0 * d32_m12) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumz1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += 0.5 * (4.0 * d32_m12) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    /* VNL for j=l-1/2 */
                    *sumx0r += d12_m12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += d12_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += d12_p12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumy0r += d12_m12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += d12_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += d12_p12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumz0r += d12_m12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += d12_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += d12_p12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumx1r += d12_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += d12_p12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += d12_m12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumy1r += d12_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += d12_p12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += d12_m12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumz1r += d12_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += d12_p12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += d12_m12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                }

                /* d */

                else if (L2 == 4) {

                    /* VNL for j=l+1/2 */
                    *sumx0r += 0.5 * (6.0 * d52_p12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumy0r += 0.5 * (6.0 * d52_p12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumz0r += 0.5 * (6.0 * d52_p12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumx1r += 0.5 * (6.0 * d52_m12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumy1r += 0.5 * (6.0 * d52_m12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumz1r += 0.5 * (6.0 * d52_m12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    /* VNL for j=l-1/2 */
                    *sumx0r += 0.5 * (4.0 * d32_p12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += 0.5 * (4.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += 0.5 * (4.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumy0r += 0.5 * (4.0 * d32_p12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += 0.5 * (4.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += 0.5 * (4.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumz0r += 0.5 * (4.0 * d32_p12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += 0.5 * (4.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += 0.5 * (4.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumx1r += 0.5 * (4.0 * d32_m12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += 0.5 * (4.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += 0.5 * (4.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumy1r += 0.5 * (4.0 * d32_m12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += 0.5 * (4.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += 0.5 * (4.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumz1r += 0.5 * (4.0 * d32_m12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += 0.5 * (4.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += 0.5 * (4.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                }

                /* f */

                else if (L2 == 6) {

                    /* VNL for j=l+1/2 */
                    *sumx0r += 0.5 * (8.0 * d72_p12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumx0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumx0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumy0r += 0.5 * (8.0 * d72_p12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumy0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumy0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumz0r += 0.5 * (8.0 * d72_p12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumz0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumz0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumx1r += 0.5 * (8.0 * d72_m12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumx1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumx1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumy1r += 0.5 * (8.0 * d72_m12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumy1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumy1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumz1r += 0.5 * (8.0 * d72_m12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumz1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumz1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    /* VNL for j=l-1/2 */
                    *sumx0r += 0.5 * (6.0 * d52_p12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumx0r += 0.5 * (6.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumx0r += 0.5 * (6.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumy0r += 0.5 * (6.0 * d52_p12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumy0r += 0.5 * (6.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumy0r += 0.5 * (6.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumz0r += 0.5 * (6.0 * d52_p12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumz0r += 0.5 * (6.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumz0r += 0.5 * (6.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumx1r += 0.5 * (6.0 * d52_m12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumx1r += 0.5 * (6.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumx1r += 0.5 * (6.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumy1r += 0.5 * (6.0 * d52_m12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumy1r += 0.5 * (6.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumy1r += 0.5 * (6.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumz1r += 0.5 * (6.0 * d52_m12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumz1r += 0.5 * (6.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumz1r += 0.5 * (6.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];
                }

            } /* if (apply_flag==1) */

            /* increment of L */

            L += 2 * l + 1;

        } /* mul */
    } /* l */
}

void MPI_OLP(double***** OLP1)
{
    int i, j, h_AN, Gh_AN, Hwan, n;
    int tno1, tno2, Mc_AN, Gc_AN, Cwan;
    int num, k, size1, size2;
    double* tmp_array;
    double* tmp_array2;
    int *Snd_S_Size, *Rcv_S_Size;
    int numprocs, myid, ID, IDS, IDR, tag = 999;

    MPI_Status stat;
    MPI_Request request;

    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /****************************************************
      allocation of arrays:
    ****************************************************/

    Snd_S_Size = (int*)malloc(sizeof(int) * numprocs);
    Rcv_S_Size = (int*)malloc(sizeof(int) * numprocs);

    /******************************************************************
     MPI

     OLP1[1], OLP1[2], and OLP1[3]

     note:

     OLP is used in DC and GDC method, where overlap integrals
     of Matomnum+MatomnumF+MatomnumS+1 are stored.
     However, overlap integrals of Matomnum+MatomnumF+1 are
     stored in Force.c. So, F_TopMAN should be used to refer
     overlap integrals in Force.c, while S_TopMAN should be
     used in DC and GDC routines.

     Although OLP is used in Eff_Hub_Pot.c, the usage is
     consistent with that of DC and GDC routines by the following
     reason:

     DC or GDC:      OLP + Spe_Total_NO   if no orbital optimization
                  CntOLP + Spe_Total_CNO  if orbital optimization

     Eff_Hub_Pot:    OLP + Spe_Total_NO   always since the U-potential
                                          affects to primitive orbital

     If no orbital optimization, both the usages are consistent.
     If orbital optimization, CntOLP and OLP are used in DC(GDC) and
     Eff_Hub_Pot.c, respectively. Therefore, there is no conflict.
    *******************************************************************/

    /***********************************
               set data size
    ************************************/

    for (ID = 0; ID < numprocs; ID++) {

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (ID != 0) {
            tag = 999;

            /* find data size to send block data */
            if (F_Snd_Num[IDS] != 0) {
                size1 = 0;
                for (n = 0; n < F_Snd_Num[IDS]; n++) {
                    Mc_AN = Snd_MAN[IDS][n];
                    Gc_AN = Snd_GAN[IDS][n];
                    Cwan = WhatSpecies[Gc_AN];
                    tno1 = Spe_Total_NO[Cwan];
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        Hwan = WhatSpecies[Gh_AN];
                        tno2 = Spe_Total_NO[Hwan];
                        size1 += 4 * tno1 * tno2;
                    }
                }

                Snd_S_Size[IDS] = size1;
                MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
            } else {
                Snd_S_Size[IDS] = 0;
            }

            /* receiving of size of data */

            if (F_Rcv_Num[IDR] != 0) {
                MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                Rcv_S_Size[IDR] = size2;
            } else {
                Rcv_S_Size[IDR] = 0;
            }

            if (F_Snd_Num[IDS] != 0)
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

            if (F_Snd_Num[IDS] != 0) {

                size1 = Snd_S_Size[IDS];

                /* allocation of array */

                tmp_array = (double*)malloc(sizeof(double) * size1);

                /* multidimentional array to vector array */

                num = 0;

                for (k = 0; k <= 3; k++) {
                    for (n = 0; n < F_Snd_Num[IDS]; n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan = WhatSpecies[Gc_AN];
                        tno1 = Spe_Total_NO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan = WhatSpecies[Gh_AN];
                            tno2 = Spe_Total_NO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    tmp_array[num] = OLP1[k][Mc_AN][h_AN][i][j];
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

            if (F_Rcv_Num[IDR] != 0) {

                size2 = Rcv_S_Size[IDR];

                /* allocation of array */
                tmp_array2 = (double*)malloc(sizeof(double) * size2);

                MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                num = 0;

                for (k = 0; k <= 3; k++) {
                    Mc_AN = F_TopMAN[IDR] - 1; /* F_TopMAN should be used. */
                    for (n = 0; n < F_Rcv_Num[IDR]; n++) {
                        Mc_AN++;
                        Gc_AN = Rcv_GAN[IDR][n];
                        Cwan = WhatSpecies[Gc_AN];
                        tno1 = Spe_Total_NO[Cwan];

                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan = WhatSpecies[Gh_AN];
                            tno2 = Spe_Total_NO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    OLP1[k][Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }
                }

                /* freeing of array */
                free(tmp_array2);
            }

            if (F_Snd_Num[IDS] != 0) {
                MPI_Wait(&request, &stat);
                free(tmp_array); /* freeing of array */
            }
        }
    }

    /****************************************************
      freeing of arrays:
    ****************************************************/

    free(Snd_S_Size);
    free(Rcv_S_Size);
}

void Force_CoreHole(double***** CDM0, double***** iDM0)
{
    /****************************************************
          Force arising from the penalty functional
          to create a core hole
    ****************************************************/

    int Mc_AN, Gc_AN, Cwan, i, j, h_AN, q_AN, Mq_AN, start_q_AN;
    int jan, kl, km, kl1, Qwan, Gq_AN, Gh_AN, Mh_AN, Hwan, ian;
    int l1, l2, l3, l, LL, Mul1, tno0, ncp, so;
    int tno1, tno2, size1, size2, n, kk, num, po, po1, po2;
    int numprocs, myid, tag = 999, ID, IDS, IDR;
    int **S_array, **R_array;
    int S_comm_flag, R_comm_flag;
    int SA_num, q, Sc_AN, GSc_AN, smul;
    int Sc_wan, Sh_AN, GSh_AN, Sh_wan;
    int Sh_AN2, fan, jg, j0, jg0, Mj_AN0;
    int Original_Mc_AN;

    double rcutA, rcutB, rcut;
    double dEx, dEy, dEz, ene, pref;
    double Stime_atom, Etime_atom;
    dcomplex ***Hx, ***Hy, ***Hz;
    dcomplex ***Hx0, ***Hy0, ***Hz0;
    dcomplex ***Hx1, ***Hy1, ***Hz1;
    int *Snd_OLP_Size, *Rcv_OLP_Size;
    int* Indicator;
    double* tmp_array;
    double* tmp_array2;

    /* for OpenMP */
    int OMPID, Nthrds, Nthrds0, Nprocs, Nloop, ODNloop;
    int *OneD2h_AN, *OneD2q_AN;
    double* dEx_threads;
    double* dEy_threads;
    double* dEz_threads;
    double stime, etime;
    double stime1, etime1;

    MPI_Status stat;
    MPI_Request request;

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    dtime(&stime);

    /****************************
         allocation of arrays
    *****************************/

    Indicator = (int*)malloc(sizeof(int) * numprocs);

    S_array = (int**)malloc(sizeof(int*) * numprocs);
    for (ID = 0; ID < numprocs; ID++) {
        S_array[ID] = (int*)malloc(sizeof(int) * 3);
    }

    R_array = (int**)malloc(sizeof(int*) * numprocs);
    for (ID = 0; ID < numprocs; ID++) {
        R_array[ID] = (int*)malloc(sizeof(int) * 3);
    }

    Snd_OLP_Size = (int*)malloc(sizeof(int) * numprocs);
    Rcv_OLP_Size = (int*)malloc(sizeof(int) * numprocs);

    /* initialize the temporal array storing the force contribution */

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = F_M2G[Mc_AN];
        Gxyz[Gc_AN][41] = 0.0;
        Gxyz[Gc_AN][42] = 0.0;
        Gxyz[Gc_AN][43] = 0.0;
    }

    /*****************************************}**********************
        THE FIRST CASE:
        In case of I=i or I=j
        for d [ \sum_k <i|k>ek<k|j> ]/dRI
    ****************************************************************/

    /*******************************************************
     *******************************************************
         multiplying overlap integrals WITH COMMUNICATION

         In case of I=i or I=j
         for d [ \sum_k <i|k>ek<k|j> ]/dRI
     *******************************************************
     *******************************************************/

    MPI_Barrier(mpi_comm_level1);
    dtime(&stime);

    Hx0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hx0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hx0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    Hy0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hy0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hy0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    Hz0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hz0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hz0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    Hx1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hx1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hx1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    Hy1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hy1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hy1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    Hz1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
    for (i = 0; i < 3; i++) {
        Hz1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hz1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
        }
    }

    for (ID = 0; ID < numprocs; ID++) {
        F_Snd_Num_WK[ID] = 0;
        F_Rcv_Num_WK[ID] = 0;
    }

    do {

        /***********************************
                set the size of data
        ************************************/

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            /* find the data size to send the block data */

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS])) {

                size1 = 0;
                n = F_Snd_Num_WK[IDS];

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

                Snd_OLP_Size[IDS] = size1;
                MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
            } else {
                Snd_OLP_Size[IDS] = 0;
            }

            /* receiving of the size of the data */

            if (0 < (F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR])) {
                MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                Rcv_OLP_Size[IDR] = size2;
            } else {
                Rcv_OLP_Size[IDR] = 0;
            }

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS]))
                MPI_Wait(&request, &stat);

        } /* ID */

        /***********************************
                   data transfer
        ************************************/

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            /******************************
                  sending of the data
            ******************************/

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS])) {

                size1 = Snd_OLP_Size[IDS];

                /* allocation of the array */

                tmp_array = (double*)malloc(sizeof(double) * size1);

                /* multidimentional array to the vector array */

                num = 0;
                n = F_Snd_Num_WK[IDS];

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
                            tmp_array[num] = OLP_CH[0][Mc_AN][h_AN][i][j];
                            num++;
                        }
                    }
                }

                MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
            }

            /******************************
              receiving of the block data
            ******************************/

            if (0 < (F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR])) {

                size2 = Rcv_OLP_Size[IDR];
                tmp_array2 = (double*)malloc(sizeof(double) * size2);
                MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                /* store */

                num = 0;
                n = F_Rcv_Num_WK[IDR];
                Original_Mc_AN = F_TopMAN[IDR] + n;

                Gc_AN = Rcv_GAN[IDR][n];
                Cwan = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[Cwan];
                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    tno2 = Spe_Total_CNO[Hwan];

                    for (i = 0; i < tno1; i++) {
                        for (j = 0; j < tno2; j++) {
                            OLP_CH[0][Matomnum + 1][h_AN][i][j] = tmp_array2[num];
                            num++;
                        }
                    }
                }

                /* free tmp_array2 */
                free(tmp_array2);

                /*****************************************************************
                                   multiplying overlap integrals
                *****************************************************************/

                for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                    dtime(&Stime_atom);

                    dEx = 0.0;
                    dEy = 0.0;
                    dEz = 0.0;

                    Gc_AN = M2G[Mc_AN];
                    Cwan = WhatSpecies[Gc_AN];
                    fan = FNAN[Gc_AN];

                    h_AN = 0;
                    Gh_AN = natn[Gc_AN][h_AN];
                    Mh_AN = F_G2M[Gh_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    ian = Spe_Total_CNO[Hwan];

                    n = F_Rcv_Num_WK[IDR];
                    jg = Rcv_GAN[IDR][n];

                    for (j0 = 0; j0 <= fan; j0++) {

                        jg0 = natn[Gc_AN][j0];
                        Mj_AN0 = F_G2M[jg0];

                        po2 = 0;
                        if (Original_Mc_AN == Mj_AN0) {
                            po2 = 1;
                            q_AN = j0;
                        }

                        if (po2 == 1) {

                            Gq_AN = natn[Gc_AN][q_AN];
                            Mq_AN = F_G2M[Gq_AN];
                            Qwan = WhatSpecies[Gq_AN];
                            jan = Spe_Total_CNO[Qwan];
                            kl = RMI1[Mc_AN][h_AN][q_AN];

                            dHCH(0, Mc_AN, h_AN, q_AN, OLP_CH, Hx0, Hy0, Hz0);

                            /* contribution of force = Trace(CDM0*dH) */
                            /* spin non-polarization */

                            if (SpinP_switch == 0) {

                                if (q_AN == h_AN)
                                    pref = 2.0;
                                else
                                    pref = 4.0;

                                for (i = 0; i < ian; i++) {
                                    for (j = 0; j < jan; j++) {

                                        dEx += pref * CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r;
                                        dEy += pref * CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r;
                                        dEz += pref * CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r;
                                    }
                                }
                            }

                            /* collinear spin polarized or non-colliear without SO and LDA+U */

                            else if (SpinP_switch == 1 || (SpinP_switch == 3 && SO_switch == 0 && Hub_U_switch == 0 && Constraint_NCS_switch == 0 && Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0)) {

                                if (q_AN == h_AN)
                                    pref = 1.0;
                                else
                                    pref = 2.0;

                                for (i = 0; i < Spe_Total_CNO[Hwan]; i++) {
                                    for (j = 0; j < Spe_Total_CNO[Qwan]; j++) {

                                        dEx += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r);
                                        dEy += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r);
                                        dEz += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r);
                                    }
                                }
                            }

                            /* spin non-collinear with spin-orbit coupling or with LDA+U */

                            else if (SpinP_switch == 3 && (SO_switch == 1 || (Hub_U_switch == 1 && F_U_flag == 1) || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)) {

                                if (q_AN == h_AN) {

                                    for (i = 0; i < Spe_Total_CNO[Hwan]; i++) {
                                        for (j = 0; j < Spe_Total_CNO[Qwan]; j++) {

                                            dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

                                            dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

                                            dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;
                                        }
                                    }
                                }

                                else {

                                    for (i = 0; i < Spe_Total_CNO[Hwan]; i++) { /* Hwan */
                                        for (j = 0; j < Spe_Total_CNO[Qwan]; j++) { /* Qwan  */

                                            dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

                                            dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

                                            dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                                                - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                                                + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                                                - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                                                + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                                                - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;

                                        } /* j */
                                    } /* i */

                                    dHCH(0, Mc_AN, q_AN, h_AN, OLP_CH, Hx1, Hy1, Hz1);

                                    kl1 = RMI1[Mc_AN][q_AN][h_AN];

                                    for (i = 0; i < Spe_Total_CNO[Qwan]; i++) { /* Qwan */
                                        for (j = 0; j < Spe_Total_CNO[Hwan]; j++) { /* Hwan */

                                            dEx += CDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].r
                                                - iDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].i
                                                + CDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].r
                                                - iDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].i
                                                + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hx1[2][i][j].r
                                                - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hx1[2][i][j].i;

                                            dEy += CDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].r
                                                - iDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].i
                                                + CDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].r
                                                - iDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].i
                                                + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hy1[2][i][j].r
                                                - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hy1[2][i][j].i;

                                            dEz += CDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].r
                                                - iDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].i
                                                + CDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].r
                                                - iDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].i
                                                + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hz1[2][i][j].r
                                                - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hz1[2][i][j].i;

                                        } /* j */
                                    } /* i */
                                }
                            }

                        } /* if (po2==1) */
                    } /* j0 */

                    /* force from #4B */

                    Gxyz[Gc_AN][41] += dEx;
                    Gxyz[Gc_AN][42] += dEy;
                    Gxyz[Gc_AN][43] += dEz;

                    /* timing */
                    dtime(&Etime_atom);
                    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

                } /* Mc_AN */

                /********************************************
                  increment of F_Rcv_Num_WK[IDR]
                ********************************************/

                F_Rcv_Num_WK[IDR]++;

            } /* if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) */

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS])) {

                MPI_Wait(&request, &stat);
                free(tmp_array); /* freeing of array */

                /********************************************
                     increment of F_Snd_Num_WK[IDS]
                ********************************************/

                F_Snd_Num_WK[IDS]++;
            }

        } /* ID */

        /*****************************************************
          check whether all the communications have finished
        *****************************************************/

        po = 0;
        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (0 < (F_Snd_Num[IDS] - F_Snd_Num_WK[IDS]))
                po += F_Snd_Num[IDS] - F_Snd_Num_WK[IDS];
            if (0 < (F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR]))
                po += F_Rcv_Num[IDR] - F_Rcv_Num_WK[IDR];
        }

    } while (po != 0);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hx0[i][j]);
        }
        free(Hx0[i]);
    }
    free(Hx0);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hy0[i][j]);
        }
        free(Hy0[i]);
    }
    free(Hy0);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hz0[i][j]);
        }
        free(Hz0[i]);
    }
    free(Hz0);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hx1[i][j]);
        }
        free(Hx1[i]);
    }
    free(Hx1);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hy1[i][j]);
        }
        free(Hy1[i]);
    }
    free(Hy1);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hz1[i][j]);
        }
        free(Hz1[i]);
    }
    free(Hz1);

    dtime(&etime);
    if (myid == 0 && measure_time) {
        printf("Time for part1 of force_HCH=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];

        if (2 <= level_stdout) {
            printf("<Force>  force(HCH1) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                myid, Mc_AN, Gc_AN, Gxyz[Gc_AN][41], Gxyz[Gc_AN][42], Gxyz[Gc_AN][43]);
            fflush(stdout);
        }
    }

    /*******************************************************
     *******************************************************
       THE FIRST CASE:
       multiplying overlap integrals WITHOUT COMMUNICATION

       In case of I=i or I=j
       for d [ \sum_k <i|k>ek<k|j> ]/dRI
     *******************************************************
     *******************************************************/

    dtime(&stime);

#pragma omp parallel shared(time_per_atom, Gxyz, CDM0, SpinP_switch, SO_switch, Hub_U_switch, F_U_flag, Constraint_NCS_switch, Zeeman_NCS_switch, Zeeman_NCO_switch, OLP_CH, RMI1, FNAN, Spe_Total_CNO, WhatSpecies, F_G2M, natn, M2G, Matomnum, List_YOUSO, F_NL_flag) private(Hx0, Hy0, Hz0, Hx1, Hy1, Hz1, OMPID, Nthrds, Nprocs, Mc_AN, Stime_atom, Etime_atom, dEx, dEy, dEz, Gc_AN, h_AN, Gh_AN, Mh_AN, Hwan, ian, q_AN, Gq_AN, Mq_AN, Qwan, jan, kl, kl1, i, j, kk, pref)
    {

        /* allocation of array */

        Hx0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hx0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hx0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hy0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hy0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hy0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hz0 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hz0[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hz0[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hx1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hx1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hx1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hy1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hy1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hy1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        Hz1 = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
        for (i = 0; i < 3; i++) {
            Hz1[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
            for (j = 0; j < List_YOUSO[7]; j++) {
                Hz1[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
            }
        }

        /* get info. on OpenMP */

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (Mc_AN = (OMPID * Matomnum / Nthrds + 1); Mc_AN < ((OMPID + 1) * Matomnum / Nthrds + 1); Mc_AN++) {

            dtime(&Stime_atom);

            dEx = 0.0;
            dEy = 0.0;
            dEz = 0.0;

            Gc_AN = M2G[Mc_AN];
            h_AN = 0;
            Gh_AN = natn[Gc_AN][h_AN];
            Mh_AN = F_G2M[Gh_AN];
            Hwan = WhatSpecies[Gh_AN];
            ian = Spe_Total_CNO[Hwan];

            for (q_AN = 0; q_AN <= FNAN[Gc_AN]; q_AN++) {

                Gq_AN = natn[Gc_AN][q_AN];
                Mq_AN = F_G2M[Gq_AN];

                if (Mq_AN <= Matomnum) {

                    Qwan = WhatSpecies[Gq_AN];
                    jan = Spe_Total_CNO[Qwan];
                    kl = RMI1[Mc_AN][h_AN][q_AN];

                    dHCH(0, Mc_AN, h_AN, q_AN, OLP_CH, Hx0, Hy0, Hz0);

                    if (SpinP_switch == 0) {

                        if (q_AN == h_AN)
                            pref = 2.0;
                        else
                            pref = 4.0;

                        for (i = 0; i < ian; i++) {
                            for (j = 0; j < jan; j++) {

                                dEx += pref * CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r;
                                dEy += pref * CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r;
                                dEz += pref * CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r;
                            }
                        }
                    }

                    /* collinear spin polarized or non-colliear without SO and LDA+U */

                    else if (SpinP_switch == 1 || (SpinP_switch == 3 && SO_switch == 0 && Hub_U_switch == 0 && Constraint_NCS_switch == 0 && Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0)) {

                        if (q_AN == h_AN)
                            pref = 1.0;
                        else
                            pref = 2.0;

                        for (i = 0; i < ian; i++) {
                            for (j = 0; j < jan; j++) {

                                dEx += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r);
                                dEy += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r);
                                dEz += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r);
                            }
                        }
                    }

                    /* spin non-collinear with spin-orbit coupling or with LDA+U */

                    else if (SpinP_switch == 3 && (SO_switch == 1 || (Hub_U_switch == 1 && F_U_flag == 1) || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)) {

                        if (q_AN == h_AN) {

                            for (i = 0; i < Spe_Total_CNO[Hwan]; i++) {
                                for (j = 0; j < Spe_Total_CNO[Qwan]; j++) {

                                    dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

                                    dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

                                    dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;
                                }
                            }
                        }

                        else {

                            for (i = 0; i < Spe_Total_CNO[Hwan]; i++) { /* Hwan */
                                for (j = 0; j < Spe_Total_CNO[Qwan]; j++) { /* Qwan  */

                                    dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

                                    dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

                                    dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                                        - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                                        + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                                        - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                                        + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                                        - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;

                                } /* j */
                            } /* i */

                            dHCH(0, Mc_AN, q_AN, h_AN, OLP_CH, Hx1, Hy1, Hz1);

                            kl1 = RMI1[Mc_AN][q_AN][h_AN];

                            for (i = 0; i < Spe_Total_CNO[Qwan]; i++) { /* Qwan */
                                for (j = 0; j < Spe_Total_CNO[Hwan]; j++) { /* Hwan */

                                    dEx += CDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].r
                                        - iDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].i
                                        + CDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].r
                                        - iDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].i
                                        + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hx1[2][i][j].r
                                        - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hx1[2][i][j].i;

                                    dEy += CDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].r
                                        - iDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].i
                                        + CDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].r
                                        - iDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].i
                                        + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hy1[2][i][j].r
                                        - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hy1[2][i][j].i;

                                    dEz += CDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].r
                                        - iDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].i
                                        + CDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].r
                                        - iDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].i
                                        + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hz1[2][i][j].r
                                        - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hz1[2][i][j].i;

                                } /* j */
                            } /* i */
                        }
                    }
                }
            }

            /* force from #4B */

            if (F_NL_flag == 1) {
                Gxyz[Gc_AN][41] += dEx;
                Gxyz[Gc_AN][42] += dEy;
                Gxyz[Gc_AN][43] += dEz;
            }

            /* timing */
            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

        } /* Mc_AN */

        /* freeing of array */

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hx0[i][j]);
            }
            free(Hx0[i]);
        }
        free(Hx0);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hy0[i][j]);
            }
            free(Hy0[i]);
        }
        free(Hy0);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hz0[i][j]);
            }
            free(Hz0[i]);
        }
        free(Hz0);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hx1[i][j]);
            }
            free(Hx1[i]);
        }
        free(Hx1);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hy1[i][j]);
            }
            free(Hy1[i]);
        }
        free(Hy1);

        for (i = 0; i < 3; i++) {
            for (j = 0; j < List_YOUSO[7]; j++) {
                free(Hz1[i][j]);
            }
            free(Hz1[i]);
        }
        free(Hz1);

    } /* #pragma omp parallel */

    dtime(&etime);
    if (myid == 0 && measure_time) {
        printf("Time for part2 of force_HCH=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];

        if (2 <= level_stdout) {
            printf("<Force>  force(HCH2) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                myid, Mc_AN, Gc_AN, Gxyz[Gc_AN][41], Gxyz[Gc_AN][42], Gxyz[Gc_AN][43]);
            fflush(stdout);
        }
    }

    /*************************************************************
       THE SECOND CASE:
       In case of I=k with I!=i and I!=j
       d [ \sum_k <i|k>ek<k|j> ]/dRI
    *************************************************************/

    /************************************************************
      MPI communication of OLP_CH whose basis part is not located
      on own site but projector part is located on own site.
    ************************************************************/

    MPI_Barrier(mpi_comm_level1);
    dtime(&stime);

    for (ID = 0; ID < numprocs; ID++)
        Indicator[ID] = 0;

    for (Mc_AN = 1; Mc_AN <= Max_Matomnum; Mc_AN++) {

        if (Mc_AN <= Matomnum)
            Gc_AN = M2G[Mc_AN];
        else
            Gc_AN = 0;

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            i = Indicator[IDS];
            po = 0;

            Gh_AN = Pro_Snd_GAtom[IDS][i];

            if (Gh_AN != 0) {

                /* find the range with the same global atomic number */

                do {

                    i++;
                    if (Gh_AN != Pro_Snd_GAtom[IDS][i])
                        po = 1;
                } while (po == 0);

                i--;
                SA_num = i - Indicator[IDS] + 1;

                /* find the data size to send the block data */

                size1 = 0;
                for (q = Indicator[IDS]; q <= (Indicator[IDS] + SA_num - 1); q++) {

                    Sc_AN = Pro_Snd_MAtom[IDS][q];
                    GSc_AN = F_M2G[Sc_AN];
                    Sc_wan = WhatSpecies[GSc_AN];
                    tno1 = Spe_Total_CNO[Sc_wan];

                    Sh_AN = Pro_Snd_LAtom[IDS][q];
                    GSh_AN = natn[GSc_AN][Sh_AN];
                    Sh_wan = WhatSpecies[GSh_AN];
                    tno2 = Spe_Total_CNO[Sh_wan];

                    size1 += 4 * tno1 * tno2;
                    size1 += 3;
                }

            } /* if (Gh_AN!=0) */

            else {
                SA_num = 0;
                size1 = 0;
            }

            S_array[IDS][0] = Gh_AN;
            S_array[IDS][1] = SA_num;
            S_array[IDS][2] = size1;

            if (ID != 0) {
                MPI_Isend(&S_array[IDS][0], 3, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                MPI_Recv(&R_array[IDR][0], 3, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                MPI_Wait(&request, &stat);
            } else {
                R_array[myid][0] = S_array[myid][0];
                R_array[myid][1] = S_array[myid][1];
                R_array[myid][2] = S_array[myid][2];
            }

            if (R_array[IDR][0] == Gc_AN)
                R_comm_flag = 1;
            else
                R_comm_flag = 0;

            if (ID != 0) {
                MPI_Isend(&R_comm_flag, 1, MPI_INT, IDR, tag, mpi_comm_level1, &request);
                MPI_Recv(&S_comm_flag, 1, MPI_INT, IDS, tag, mpi_comm_level1, &stat);
                MPI_Wait(&request, &stat);
            } else {
                S_comm_flag = R_comm_flag;
            }

            /*****************************************
                          send the data
            *****************************************/

            /* if (S_comm_flag==1) then, send data to IDS */

            if (S_comm_flag == 1) {

                /* allocate tmp_array */

                tmp_array = (double*)malloc(sizeof(double) * size1);

                /* multidimentional array to vector array */

                num = 0;

                for (q = Indicator[IDS]; q <= (Indicator[IDS] + SA_num - 1); q++) {

                    Sc_AN = Pro_Snd_MAtom[IDS][q];
                    GSc_AN = F_M2G[Sc_AN];
                    Sc_wan = WhatSpecies[GSc_AN];
                    tno1 = Spe_Total_CNO[Sc_wan];

                    Sh_AN = Pro_Snd_LAtom[IDS][q];
                    GSh_AN = natn[GSc_AN][Sh_AN];
                    Sh_wan = WhatSpecies[GSh_AN];
                    tno2 = Spe_Total_CNO[Sh_wan];
                    Sh_AN2 = Pro_Snd_LAtom2[IDS][q];

                    tmp_array[num] = (double)Sc_AN;
                    num++;
                    tmp_array[num] = (double)Sh_AN;
                    num++;
                    tmp_array[num] = (double)Sh_AN2;
                    num++;

                    for (kk = 0; kk <= 3; kk++) {
                        for (i = 0; i < tno1; i++) {
                            for (j = 0; j < tno2; j++) {
                                tmp_array[num] = OLP_CH[kk][Sc_AN][Sh_AN][i][j];
                                num++;
                            }
                        }
                    }
                }

                if (ID != 0) {
                    MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
                }

                /* update Indicator[IDS] */

                Indicator[IDS] += SA_num;

            } /* if (S_comm_flag==1) */

            /*****************************************
                         receiving the data
            *****************************************/

            /* if (R_comm_flag==1) then, receive the data from IDR */

            if (R_comm_flag == 1) {

                size2 = R_array[IDR][2];
                tmp_array2 = (double*)malloc(sizeof(double) * size2);

                if (ID != 0) {
                    MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
                } else {
                    for (i = 0; i < size2; i++)
                        tmp_array2[i] = tmp_array[i];
                }

                /* store */

                num = 0;

                for (n = 0; n < R_array[IDR][1]; n++) {

                    Sc_AN = (int)tmp_array2[num];
                    num++;
                    Sh_AN = (int)tmp_array2[num];
                    num++;
                    Sh_AN2 = (int)tmp_array2[num];
                    num++;

                    GSc_AN = natn[Gc_AN][Sh_AN2];
                    Sc_wan = WhatSpecies[GSc_AN];
                    tno1 = Spe_Total_CNO[Sc_wan];

                    GSh_AN = natn[GSc_AN][Sh_AN];
                    Sh_wan = WhatSpecies[GSh_AN];
                    tno2 = Spe_Total_CNO[Sh_wan];

                    for (kk = 0; kk <= 3; kk++) {
                        for (i = 0; i < tno1; i++) {
                            for (j = 0; j < tno2; j++) {
                                OLP_CH[kk][Matomnum + 1][Sh_AN2][i][j] = tmp_array2[num];
                                num++;
                            }
                        }
                    }
                }

                /* free tmp_array2 */
                free(tmp_array2);

            } /* if (R_comm_flag==1) */

            if (S_comm_flag == 1) {
                if (ID != 0)
                    MPI_Wait(&request, &stat);
                free(tmp_array); /* freeing of array */
            }

        } /* ID */

        if (Mc_AN <= Matomnum) {

            /* get Nthrds0 */
#pragma omp parallel shared(Nthrds0)
            {
                Nthrds0 = omp_get_num_threads();
            }

            /* allocation of arrays */
            dEx_threads = (double*)malloc(sizeof(double) * Nthrds0);
            dEy_threads = (double*)malloc(sizeof(double) * Nthrds0);
            dEz_threads = (double*)malloc(sizeof(double) * Nthrds0);

            for (Nloop = 0; Nloop < Nthrds0; Nloop++) {
                dEx_threads[Nloop] = 0.0;
                dEy_threads[Nloop] = 0.0;
                dEz_threads[Nloop] = 0.0;
            }

            /* one-dimensionalize the h_AN and q_AN loops */

            OneD2h_AN = (int*)malloc(sizeof(int) * (FNAN[Gc_AN] + 1) * (FNAN[Gc_AN] + 2));
            OneD2q_AN = (int*)malloc(sizeof(int) * (FNAN[Gc_AN] + 1) * (FNAN[Gc_AN] + 2));

            ODNloop = 0;
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (SpinP_switch == 3 && (SO_switch == 1 || (Hub_U_switch == 1 && F_U_flag == 1) || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
                    || (Solver == 5 || Solver == 8 || Solver == 11))
                    start_q_AN = 0;
                else
                    start_q_AN = h_AN;

                for (q_AN = start_q_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {

                    kl = RMI1[Mc_AN][h_AN][q_AN];

                    if (0 <= kl) {
                        OneD2h_AN[ODNloop] = h_AN;
                        OneD2q_AN[ODNloop] = q_AN;
                        ODNloop++;
                    }
                }
            }

#pragma omp parallel shared(ODNloop, OneD2h_AN, OneD2q_AN, Mc_AN, Gc_AN, dEx_threads, dEy_threads, dEz_threads, CDM0, SpinP_switch, SO_switch, Hub_U_switch, Constraint_NCS_switch, Zeeman_NCS_switch, Zeeman_NCO_switch, OLP_CH, RMI1, Spe_Total_CNO, WhatSpecies, F_G2M, natn, FNAN, List_YOUSO, Solver, F_NL_flag, F_U_flag) private(OMPID, Nthrds, Nprocs, Hx, Hy, Hz, i, j, h_AN, Gh_AN, Mh_AN, Hwan, ian, q_AN, Gq_AN, Mq_AN, Qwan, jan, kl, km, Nloop, pref)
            {

                /* allocation of arrays */

                Hx = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
                for (i = 0; i < 3; i++) {
                    Hx[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        Hx[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
                    }
                }

                Hy = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
                for (i = 0; i < 3; i++) {
                    Hy[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        Hy[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
                    }
                }

                Hz = (dcomplex***)malloc(sizeof(dcomplex**) * 3);
                for (i = 0; i < 3; i++) {
                    Hz[i] = (dcomplex**)malloc(sizeof(dcomplex*) * List_YOUSO[7]);
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        Hz[i][j] = (dcomplex*)malloc(sizeof(dcomplex) * List_YOUSO[7]);
                    }
                }

                /* get info. on OpenMP */

                OMPID = omp_get_thread_num();
                Nthrds = omp_get_num_threads();
                Nprocs = omp_get_num_procs();

                for (Nloop = OMPID * ODNloop / Nthrds; Nloop < (OMPID + 1) * ODNloop / Nthrds; Nloop++) {

                    /* get h_AN and q_AN */

                    h_AN = OneD2h_AN[Nloop];
                    q_AN = OneD2q_AN[Nloop];

                    /* set informations on h_AN */

                    Gh_AN = natn[Gc_AN][h_AN];
                    Mh_AN = F_G2M[Gh_AN];
                    Hwan = WhatSpecies[Gh_AN];
                    ian = Spe_Total_CNO[Hwan];

                    /* set informations on q_AN */

                    Gq_AN = natn[Gc_AN][q_AN];
                    Mq_AN = F_G2M[Gq_AN];
                    Qwan = WhatSpecies[Gq_AN];
                    jan = Spe_Total_CNO[Qwan];
                    kl = RMI1[Mc_AN][h_AN][q_AN];
                    km = RMI1[Mc_AN][q_AN][h_AN];

                    if (0 <= kl) {

                        dHCH(1, Mc_AN, h_AN, q_AN, OLP_CH, Hx, Hy, Hz);

                        /* contribution of force = Trace(CDM0*dH) */

                        /* spin non-polarization */

                        if (SpinP_switch == 0) {

                            if (Solver == 5 || Solver == 8 || Solver == 11) {
                                pref = 2.0;
                            } else {
                                if (q_AN == h_AN)
                                    pref = 2.0;
                                else
                                    pref = 4.0;
                            }

                            for (i = 0; i < ian; i++) {
                                for (j = 0; j < jan; j++) {
                                    dEx_threads[OMPID] += pref * CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r;
                                    dEy_threads[OMPID] += pref * CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r;
                                    dEz_threads[OMPID] += pref * CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r;
                                }
                            }

                        }

                        /* collinear spin polarized or non-colliear without SO and LDA+U */

                        else if (SpinP_switch == 1 || (SpinP_switch == 3 && SO_switch == 0 && Hub_U_switch == 0 && Constraint_NCS_switch == 0 && Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0)) {

                            if (Solver == 5 || Solver == 8 || Solver == 11) {
                                pref = 1.0;
                            } else {
                                if (q_AN == h_AN)
                                    pref = 1.0;
                                else
                                    pref = 2.0;
                            }

                            for (i = 0; i < ian; i++) {
                                for (j = 0; j < jan; j++) {

                                    dEx_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].r);
                                    dEy_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].r);
                                    dEz_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].r);
                                }
                            }
                        }

                        /* spin non-collinear with spin-orbit coupling or with LDA+U */

                        else if (SpinP_switch == 3 && (SO_switch == 1 || (Hub_U_switch == 1 && F_U_flag == 1) || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)) {

                            pref = 1.0;

                            for (i = 0; i < ian; i++) {
                                for (j = 0; j < jan; j++) {

                                    dEx_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r - iDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].i + CDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].r - iDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].i + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx[2][i][j].r - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx[2][i][j].i);

                                    dEy_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r - iDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].i + CDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].r - iDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].i + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy[2][i][j].r - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy[2][i][j].i);

                                    dEz_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r - iDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].i + CDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].r - iDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].i + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz[2][i][j].r - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz[2][i][j].i);
                                }
                            }
                        }

                    } /* if (0<=kl) */
                } /* Nloop */

                /* freeing of arrays */

                for (i = 0; i < 3; i++) {
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        free(Hx[i][j]);
                    }
                    free(Hx[i]);
                }
                free(Hx);

                for (i = 0; i < 3; i++) {
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        free(Hy[i][j]);
                    }
                    free(Hy[i]);
                }
                free(Hy);

                for (i = 0; i < 3; i++) {
                    for (j = 0; j < List_YOUSO[7]; j++) {
                        free(Hz[i][j]);
                    }
                    free(Hz[i]);
                }
                free(Hz);

            } /* #pragma omp parallel */

            /* sum of dEx_threads */

            dEx = 0.0;
            dEy = 0.0;
            dEz = 0.0;

            if (F_NL_flag == 1) {
                for (Nloop = 0; Nloop < Nthrds0; Nloop++) {
                    dEx += dEx_threads[Nloop];
                    dEy += dEy_threads[Nloop];
                    dEz += dEz_threads[Nloop];
                }

                /* force from #4B */

                Gxyz[Gc_AN][41] += dEx;
                Gxyz[Gc_AN][42] += dEy;
                Gxyz[Gc_AN][43] += dEz;
            }

            if (2 <= level_stdout) {
                printf("<Force>  force(HCH3) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                    myid, Mc_AN, Gc_AN, dEx, dEy, dEz);
                fflush(stdout);
            }

            /* freeing of array */
            free(OneD2q_AN);
            free(OneD2h_AN);
            free(dEx_threads);
            free(dEy_threads);
            free(dEz_threads);

        } /* if (Mc_AN<=Matomnum) */

    } /* Mc_AN */

    dtime(&etime);
    if (myid == 0 && measure_time) {
        printf("Time for part3 of force_NL=%18.5f\n", etime - stime);
        fflush(stdout);
    }

    /********************************************************
      adding Gxyz[Gc_AN][41,42,43] to Gxyz[Gc_AN][17,18,19]
    ********************************************************/

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];

        if (2 <= level_stdout) {
            printf("<Force>  force(HCH) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
                myid, Mc_AN, Gc_AN, Gxyz[Gc_AN][41], Gxyz[Gc_AN][42], Gxyz[Gc_AN][43]);
            fflush(stdout);
        }

        Gxyz[Gc_AN][17] += Gxyz[Gc_AN][41];
        Gxyz[Gc_AN][18] += Gxyz[Gc_AN][42];
        Gxyz[Gc_AN][19] += Gxyz[Gc_AN][43];
    }

    /***********************************
              freeing of arrays
    ************************************/

    free(Indicator);

    for (ID = 0; ID < numprocs; ID++) {
        free(S_array[ID]);
    }
    free(S_array);

    for (ID = 0; ID < numprocs; ID++) {
        free(R_array[ID]);
    }
    free(R_array);

    free(Snd_OLP_Size);
    free(Rcv_OLP_Size);
}
