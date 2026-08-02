/**********************************************************************
  Set_Density_Grid_GPU.c:

  A single-owner GPU service for Set_Density_Grid.  The owner gathers
  the unique atom-centred orbital grids once per geometry, gathers only
  CDM on subsequent SCF iterations, constructs the density on the full
  regular grid, and scatters the contiguous B partitions to all ranks.
***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include "set_cuda_default_device_from_local_rank.h"
#include <accel.h>
#include <limits.h>
#include <openacc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(uint32_t) == sizeof(unsigned int), "uint32_t must match MPI_UNSIGNED");
_Static_assert(sizeof(uint16_t) == sizeof(unsigned short), "uint16_t must match MPI_UNSIGNED_SHORT");

enum {
  SDG_COUNT_ATOMS = 0,
  SDG_COUNT_ORBS,
  SDG_COUNT_OUTPUTS,
  SDG_COUNT_PAIRS,
  SDG_COUNT_TERMS,
  SDG_COUNT_CDM,
  SDG_COUNT_FIELDS
};

typedef struct {
  int ready;
  int unavailable;
  int owner;
  int spin_count;
  int nh_bits;
  uint32_t nh_mask;
  int local_cdm_count;
  int device_resident;

  size_t atom_count;
  size_t orbital_count;
  size_t output_count;
  size_t pair_count;
  size_t term_count;
  size_t cdm_count;
  size_t global_grid_count;
  size_t device_bytes;

  uint32_t *atom_out_offset;
  uint32_t *atom_orb_offset;
  uint32_t *atom_pair_offset;
  int *atom_no;

  int *pair_neighbor;
  uint32_t *pair_cdm_offset;

  uint16_t *out_atom;
  uint32_t *out_ptr;
  uint32_t *terms;
  uint32_t *grid_ptr;
  uint32_t *grid_out;
  Type_Orbs_Grid *orbs;
  double *cdm;
  double *rho;

  int *cdm_counts;
  int *cdm_displs;
  int *scatter_counts;
  int *scatter_displs;
} SetDensityGpuCache;

static SetDensityGpuCache SDG_cache = {0};

static int SDG_env_bool(const char *name, int fallback)
{
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0') return fallback;
  return atoi(value) != 0;
}

static size_t SDG_env_mib(const char *name, size_t fallback_mib)
{
  const char *value = getenv(name);
  char *endp = NULL;
  unsigned long long mib;

  if (value == NULL || value[0] == '\0') return fallback_mib * 1024ULL * 1024ULL;
  mib = strtoull(value, &endp, 10);
  if (endp == value || *endp != '\0') return fallback_mib * 1024ULL * 1024ULL;
  if (mib > (unsigned long long)SIZE_MAX / (1024ULL * 1024ULL)) return SIZE_MAX;
  return (size_t)mib * 1024ULL * 1024ULL;
}

static int SDG_bits_for(unsigned int max_value)
{
  int bits = 0;
  while (max_value != 0U) {
    bits++;
    max_value >>= 1;
  }
  return (bits == 0) ? 1 : bits;
}

static int SDG_add_bytes(size_t *total, size_t count, size_t element_size)
{
  size_t bytes;
  if (count != 0 && element_size > SIZE_MAX / count) return 0;
  bytes = count * element_size;
  if (bytes > SIZE_MAX - *total) return 0;
  *total += bytes;
  return 1;
}

static void *SDG_malloc(size_t count, size_t element_size)
{
  size_t bytes;
  if (count != 0 && element_size > SIZE_MAX / count) return NULL;
  bytes = count * element_size;
  return malloc(bytes == 0 ? 1 : bytes);
}

static void SDG_acc_delete_if_present(void *host_ptr, size_t bytes)
{
  if (host_ptr != NULL && bytes != 0 && acc_is_present(host_ptr, bytes)) {
    acc_delete(host_ptr, bytes);
  }
}

static void SDG_delete_device(SetDensityGpuCache *cache)
{
  if (!cache->device_resident) return;

  /* acc_copyin can fail after only a prefix of these mappings was created.
     Query the present table so the same cleanup handles full and partial
     uploads without attempting to delete an absent mapping. */
  SDG_acc_delete_if_present(cache->orbs, cache->orbital_count * sizeof(Type_Orbs_Grid));
  SDG_acc_delete_if_present(cache->atom_out_offset, (cache->atom_count + 1) * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->atom_orb_offset, (cache->atom_count + 1) * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->atom_pair_offset, (cache->atom_count + 1) * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->atom_no, cache->atom_count * sizeof(int));
  SDG_acc_delete_if_present(cache->pair_neighbor, cache->pair_count * sizeof(int));
  SDG_acc_delete_if_present(cache->pair_cdm_offset, cache->pair_count * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->out_atom, cache->output_count * sizeof(uint16_t));
  SDG_acc_delete_if_present(cache->out_ptr, (cache->output_count + 1) * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->terms, cache->term_count * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->grid_ptr, (cache->global_grid_count + 1) * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->grid_out, cache->output_count * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->cdm, cache->cdm_count * sizeof(double));
  SDG_acc_delete_if_present(cache->rho, cache->global_grid_count * (size_t)cache->spin_count * sizeof(double));
  acc_wait_all();

  /* NVHPC keeps storage released by acc_delete in an OpenACC freelist.  The
     following Band/Cluster eigensolver uses cudaMemGetInfo() to decide whether
     GEMMul8 can allocate its workspace, so leaving the roughly multi-GiB
     density epoch in that freelist makes the memory appear unavailable and
     can force a much slower native-cuBLAS path.  Return the cached blocks to CUDA
     before the next SCF diagonalization. */
  acc_clear_freelists();
  cache->device_resident = 0;
}

static void SDG_free_host(SetDensityGpuCache *cache)
{
  SDG_delete_device(cache);
  free(cache->atom_out_offset);
  free(cache->atom_orb_offset);
  free(cache->atom_pair_offset);
  free(cache->atom_no);
  free(cache->pair_neighbor);
  free(cache->pair_cdm_offset);
  free(cache->out_atom);
  free(cache->out_ptr);
  free(cache->terms);
  free(cache->grid_ptr);
  free(cache->grid_out);
  free(cache->orbs);
  free(cache->cdm);
  free(cache->rho);
  free(cache->cdm_counts);
  free(cache->cdm_displs);
  free(cache->scatter_counts);
  free(cache->scatter_displs);
  memset(cache, 0, sizeof(*cache));
}

static void SDG_local_free(void);

void Set_Density_Grid_GPU_Invalidate(void)
{
  SDG_free_host(&SDG_cache);
  SDG_local_free();
}

static void SDG_disable_current_epoch(SetDensityGpuCache *cache)
{
  SDG_free_host(cache);
  cache->unavailable = 1;
}

static int SDG_make_counts(const unsigned long long *rank_counts, int numprocs, int field,
                           int **counts_out, int **displs_out, size_t *total_out)
{
  int *counts = NULL;
  int *displs = NULL;
  size_t total = 0;
  int rank;

  counts = (int *)SDG_malloc((size_t)numprocs, sizeof(int));
  displs = (int *)SDG_malloc((size_t)numprocs, sizeof(int));
  if (counts == NULL || displs == NULL) {
    free(counts);
    free(displs);
    return 0;
  }

  for (rank = 0; rank < numprocs; rank++) {
    unsigned long long count = rank_counts[(size_t)rank * SDG_COUNT_FIELDS + (size_t)field];
    if (count > (unsigned long long)INT_MAX || total > (size_t)INT_MAX || count > (unsigned long long)((size_t)INT_MAX - total)) {
      free(counts);
      free(displs);
      return 0;
    }
    counts[rank] = (int)count;
    displs[rank] = (int)total;
    total += (size_t)count;
  }

  *counts_out = counts;
  *displs_out = displs;
  *total_out = total;
  return 1;
}

static int SDG_compute_local_counts(unsigned long long counts[SDG_COUNT_FIELDS], int spin_count,
                                    int *max_grid, int *max_h)
{
  int Mc_AN, h_AN, Gc_AN, Gh_AN, Cwan, Hwan;
  size_t atoms = 0, orbs = 0, outputs = 0, pairs = 0, terms = 0, cdm = 0;

  *max_grid = 0;
  *max_h = 0;

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
    int NO0;
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_CNO[Cwan];
    atoms++;
    outputs += (size_t)GridN_Atom[Gc_AN];
    orbs += (size_t)GridN_Atom[Gc_AN] * (size_t)NO0;
    if (*max_grid < GridN_Atom[Gc_AN]) *max_grid = GridN_Atom[Gc_AN];
    if (*max_h < FNAN[Gc_AN] + 1) *max_h = FNAN[Gc_AN] + 1;

    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
      int NO1;
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      NO1 = Spe_Total_CNO[Hwan];
      pairs++;
      terms += (size_t)NumOLG[Mc_AN][h_AN];
      cdm += (size_t)spin_count * (size_t)NO0 * (size_t)NO1;
    }
  }

  if (atoms > ULLONG_MAX || orbs > ULLONG_MAX || outputs > ULLONG_MAX || pairs > ULLONG_MAX ||
      terms > ULLONG_MAX || cdm > ULLONG_MAX) return 0;
  counts[SDG_COUNT_ATOMS] = (unsigned long long)atoms;
  counts[SDG_COUNT_ORBS] = (unsigned long long)orbs;
  counts[SDG_COUNT_OUTPUTS] = (unsigned long long)outputs;
  counts[SDG_COUNT_PAIRS] = (unsigned long long)pairs;
  counts[SDG_COUNT_TERMS] = (unsigned long long)terms;
  counts[SDG_COUNT_CDM] = (unsigned long long)cdm;
  return 1;
}

static int SDG_build_cache(int spin_count, int owner, int myid, int numprocs)
{
  SetDensityGpuCache *cache = &SDG_cache;
  unsigned long long local_counts[SDG_COUNT_FIELDS] = {0};
  unsigned long long *rank_counts = NULL;
  int local_max_grid = 0, local_max_h = 0, max_grid = 0, max_h = 0;
  int *atom_counts = NULL, *atom_displs = NULL;
  int *orb_counts = NULL, *orb_displs = NULL;
  int *out_counts = NULL, *out_displs = NULL;
  int *term_counts = NULL, *term_displs = NULL;
  int *pair_counts = NULL, *pair_displs = NULL;
  int *cdm_counts = NULL, *cdm_displs = NULL;
  size_t total_atoms = 0, total_orbs = 0, total_outputs = 0;
  size_t total_pairs = 0, total_terms = 0, total_cdm = 0;
  int ok = 1, all_ok = 1;
  int *local_atom_ids = NULL, *atom_ids = NULL;
  Type_Orbs_Grid *local_orbs = NULL;
  uint32_t *local_out_gn = NULL, *out_gn = NULL;
  uint16_t *local_degree = NULL, *degree = NULL;
  uint32_t *local_ptr = NULL, *local_cursor = NULL, *local_terms = NULL;
  int nh_bits = 0, h_bits = 0;
  uint32_t nh_mask = 0;
  int Mc_AN, h_AN, Gc_AN, Gh_AN, Nc, Nog, i;
  size_t pos;

  if (!SDG_compute_local_counts(local_counts, spin_count, &local_max_grid, &local_max_h)) ok = 0;
  MPI_Allreduce(&local_max_grid, &max_grid, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
  MPI_Allreduce(&local_max_h, &max_h, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

  if (myid == owner) rank_counts = (unsigned long long *)SDG_malloc((size_t)numprocs * SDG_COUNT_FIELDS,
                                                                    sizeof(unsigned long long));
  if (myid == owner && rank_counts == NULL) ok = 0;
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
  if (!all_ok) {
    free(rank_counts);
    return 0;
  }

  MPI_Gather(local_counts, SDG_COUNT_FIELDS, MPI_UNSIGNED_LONG_LONG,
             rank_counts, SDG_COUNT_FIELDS, MPI_UNSIGNED_LONG_LONG, owner, mpi_comm_level1);

  if (myid == owner) {
    ok = SDG_make_counts(rank_counts, numprocs, SDG_COUNT_ATOMS, &atom_counts, &atom_displs, &total_atoms) &&
         SDG_make_counts(rank_counts, numprocs, SDG_COUNT_ORBS, &orb_counts, &orb_displs, &total_orbs) &&
         SDG_make_counts(rank_counts, numprocs, SDG_COUNT_OUTPUTS, &out_counts, &out_displs, &total_outputs) &&
         SDG_make_counts(rank_counts, numprocs, SDG_COUNT_PAIRS, &pair_counts, &pair_displs, &total_pairs) &&
         SDG_make_counts(rank_counts, numprocs, SDG_COUNT_TERMS, &term_counts, &term_displs, &total_terms) &&
         SDG_make_counts(rank_counts, numprocs, SDG_COUNT_CDM, &cdm_counts, &cdm_displs, &total_cdm);

    nh_bits = SDG_bits_for((max_grid <= 1) ? 0U : (unsigned int)(max_grid - 1));
    h_bits = SDG_bits_for((max_h <= 1) ? 0U : (unsigned int)(max_h - 1));
    if (nh_bits + h_bits > 32 || total_atoms != (size_t)atomnum || total_atoms > (size_t)UINT16_MAX ||
        total_outputs > (size_t)UINT32_MAX || total_pairs > (size_t)UINT32_MAX ||
        total_terms > (size_t)UINT32_MAX || total_cdm > (size_t)UINT32_MAX) ok = 0;
    if (nh_bits == 32) ok = 0;
    if (ok) nh_mask = (1U << nh_bits) - 1U;

    if (ok) {
      atom_ids = (int *)SDG_malloc(total_atoms, sizeof(int));
      cache->orbs = (Type_Orbs_Grid *)SDG_malloc(total_orbs, sizeof(Type_Orbs_Grid));
      out_gn = (uint32_t *)SDG_malloc(total_outputs, sizeof(uint32_t));
      degree = (uint16_t *)SDG_malloc(total_outputs, sizeof(uint16_t));
      cache->terms = (uint32_t *)SDG_malloc(total_terms, sizeof(uint32_t));
      cache->cdm = (double *)SDG_malloc(total_cdm, sizeof(double));
      cache->rho = (double *)SDG_malloc((size_t)spin_count * (size_t)Ngrid1 * (size_t)Ngrid2 * (size_t)Ngrid3,
                                        sizeof(double));
      if (atom_ids == NULL || cache->orbs == NULL || out_gn == NULL || degree == NULL || cache->terms == NULL ||
          cache->cdm == NULL || cache->rho == NULL) ok = 0;
    }
  }

  MPI_Bcast(&ok, 1, MPI_INT, owner, mpi_comm_level1);
  if (!ok) goto fail;
  MPI_Bcast(&nh_bits, 1, MPI_INT, owner, mpi_comm_level1);
  MPI_Bcast(&nh_mask, 1, MPI_UNSIGNED, owner, mpi_comm_level1);

  local_atom_ids = (int *)SDG_malloc((size_t)local_counts[SDG_COUNT_ATOMS], sizeof(int));
  local_orbs = (Type_Orbs_Grid *)SDG_malloc((size_t)local_counts[SDG_COUNT_ORBS], sizeof(Type_Orbs_Grid));
  local_out_gn = (uint32_t *)SDG_malloc((size_t)local_counts[SDG_COUNT_OUTPUTS], sizeof(uint32_t));
  local_degree = (uint16_t *)calloc((size_t)(local_counts[SDG_COUNT_OUTPUTS] == 0 ? 1 : local_counts[SDG_COUNT_OUTPUTS]),
                                    sizeof(uint16_t));
  local_ptr = (uint32_t *)SDG_malloc((size_t)local_counts[SDG_COUNT_OUTPUTS] + 1, sizeof(uint32_t));
  local_cursor = (uint32_t *)SDG_malloc((size_t)local_counts[SDG_COUNT_OUTPUTS], sizeof(uint32_t));
  local_terms = (uint32_t *)SDG_malloc((size_t)local_counts[SDG_COUNT_TERMS], sizeof(uint32_t));
  ok = (local_atom_ids != NULL && local_orbs != NULL && local_out_gn != NULL && local_degree != NULL &&
        local_ptr != NULL && local_cursor != NULL && local_terms != NULL);
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
  if (!all_ok) goto fail;

  pos = 0;
  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) local_atom_ids[Mc_AN - 1] = M2G[Mc_AN];

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
    int NO0;
    Gc_AN = M2G[Mc_AN];
    NO0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
    for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {
      for (i = 0; i < NO0; i++) local_orbs[pos++] = Orbs_Grid[Mc_AN][Nc][i];
    }
  }
  if (pos != (size_t)local_counts[SDG_COUNT_ORBS]) ok = 0;

  pos = 0;
  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
    Gc_AN = M2G[Mc_AN];
    for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {
      int GN = GridListAtom[Mc_AN][Nc];
      if (GN < 0 || (size_t)GN >= (size_t)Ngrid1 * (size_t)Ngrid2 * (size_t)Ngrid3) ok = 0;
      local_out_gn[pos++] = (uint32_t)GN;
    }
  }

  pos = 0;
  {
    size_t out_base = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
      Gc_AN = M2G[Mc_AN];
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
        for (Nog = 0; Nog < NumOLG[Mc_AN][h_AN]; Nog++) {
          uint16_t *d;
          Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
          if (Nc < 0 || Nc >= GridN_Atom[Gc_AN]) {
            ok = 0;
            continue;
          }
          d = &local_degree[out_base + (size_t)Nc];
          if (*d == UINT16_MAX) ok = 0;
          else (*d)++;
          pos++;
        }
      }
      out_base += (size_t)GridN_Atom[Gc_AN];
    }
  }
  if (pos != (size_t)local_counts[SDG_COUNT_TERMS]) ok = 0;

  local_ptr[0] = 0;
  for (pos = 0; pos < (size_t)local_counts[SDG_COUNT_OUTPUTS]; pos++) {
    uint64_t next = (uint64_t)local_ptr[pos] + (uint64_t)local_degree[pos];
    if (next > UINT32_MAX) ok = 0;
    local_ptr[pos + 1] = (uint32_t)next;
    local_cursor[pos] = local_ptr[pos];
  }
  if ((size_t)local_ptr[local_counts[SDG_COUNT_OUTPUTS]] != (size_t)local_counts[SDG_COUNT_TERMS]) ok = 0;

  {
    size_t out_base = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
      Gc_AN = M2G[Mc_AN];
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
        for (Nog = 0; Nog < NumOLG[Mc_AN][h_AN]; Nog++) {
          uint32_t slot, code;
          int Nh;
          Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
          Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
          Gh_AN = natn[Gc_AN][h_AN];
          if (Nc < 0 || Nc >= GridN_Atom[Gc_AN] || Nh < 0 || Nh >= GridN_Atom[Gh_AN] ||
              (uint32_t)Nh > nh_mask || ((uint64_t)(unsigned int)h_AN << nh_bits) > UINT32_MAX) {
            ok = 0;
            continue;
          }
          slot = local_cursor[out_base + (size_t)Nc]++;
          code = ((uint32_t)h_AN << nh_bits) | (uint32_t)Nh;
          if (slot >= (uint32_t)local_counts[SDG_COUNT_TERMS]) ok = 0;
          else local_terms[slot] = code;
        }
      }
      out_base += (size_t)GridN_Atom[Gc_AN];
    }
  }

  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
  if (!all_ok) goto fail;

  MPI_Gatherv(local_atom_ids, (int)local_counts[SDG_COUNT_ATOMS], MPI_INT,
              atom_ids, atom_counts, atom_displs, MPI_INT, owner, mpi_comm_level1);
  MPI_Gatherv(local_orbs, (int)local_counts[SDG_COUNT_ORBS], MPI_Type_Orbs_Grid,
              cache->orbs, orb_counts, orb_displs, MPI_Type_Orbs_Grid, owner, mpi_comm_level1);
  MPI_Gatherv(local_out_gn, (int)local_counts[SDG_COUNT_OUTPUTS], MPI_UNSIGNED,
              out_gn, out_counts, out_displs, MPI_UNSIGNED, owner, mpi_comm_level1);
  MPI_Gatherv(local_degree, (int)local_counts[SDG_COUNT_OUTPUTS], MPI_UNSIGNED_SHORT,
              degree, out_counts, out_displs, MPI_UNSIGNED_SHORT, owner, mpi_comm_level1);
  MPI_Gatherv(local_terms, (int)local_counts[SDG_COUNT_TERMS], MPI_UNSIGNED,
              cache->terms, term_counts, term_displs, MPI_UNSIGNED, owner, mpi_comm_level1);

  free(local_atom_ids); local_atom_ids = NULL;
  free(local_orbs); local_orbs = NULL;
  free(local_out_gn); local_out_gn = NULL;
  free(local_degree); local_degree = NULL;
  free(local_ptr); local_ptr = NULL;
  free(local_cursor); local_cursor = NULL;
  free(local_terms); local_terms = NULL;

  if (myid == owner) {
    int *atom_index = NULL;
    uint32_t *grid_degree = NULL, *grid_cursor = NULL;
    size_t a, p, out, grid_count;
    uint64_t out_off = 0, orb_off = 0, pair_off = 0, cdm_off = 0, term_off = 0;

    cache->atom_out_offset = (uint32_t *)SDG_malloc(total_atoms + 1, sizeof(uint32_t));
    cache->atom_orb_offset = (uint32_t *)SDG_malloc(total_atoms + 1, sizeof(uint32_t));
    cache->atom_pair_offset = (uint32_t *)SDG_malloc(total_atoms + 1, sizeof(uint32_t));
    cache->atom_no = (int *)SDG_malloc(total_atoms, sizeof(int));
    cache->pair_neighbor = (int *)SDG_malloc(total_pairs, sizeof(int));
    cache->pair_cdm_offset = (uint32_t *)SDG_malloc(total_pairs, sizeof(uint32_t));
    cache->out_atom = (uint16_t *)SDG_malloc(total_outputs, sizeof(uint16_t));
    cache->out_ptr = (uint32_t *)SDG_malloc(total_outputs + 1, sizeof(uint32_t));
    atom_index = (int *)SDG_malloc((size_t)atomnum + 1, sizeof(int));
    grid_count = (size_t)Ngrid1 * (size_t)Ngrid2 * (size_t)Ngrid3;
    grid_degree = (uint32_t *)calloc(grid_count == 0 ? 1 : grid_count, sizeof(uint32_t));
    grid_cursor = (uint32_t *)SDG_malloc(grid_count, sizeof(uint32_t));
    cache->grid_ptr = (uint32_t *)SDG_malloc(grid_count + 1, sizeof(uint32_t));
    cache->grid_out = (uint32_t *)SDG_malloc(total_outputs, sizeof(uint32_t));
    cache->scatter_counts = (int *)SDG_malloc((size_t)numprocs, sizeof(int));
    cache->scatter_displs = (int *)SDG_malloc((size_t)numprocs, sizeof(int));

    if (cache->atom_out_offset == NULL || cache->atom_orb_offset == NULL || cache->atom_pair_offset == NULL ||
        cache->atom_no == NULL || cache->pair_neighbor == NULL || cache->pair_cdm_offset == NULL ||
        cache->out_atom == NULL || cache->out_ptr == NULL || atom_index == NULL || grid_degree == NULL ||
        grid_cursor == NULL || cache->grid_ptr == NULL || cache->grid_out == NULL ||
        cache->scatter_counts == NULL || cache->scatter_displs == NULL) ok = 0;

    if (ok) {
      for (i = 0; i <= atomnum; i++) atom_index[i] = -1;
      for (a = 0; a < total_atoms; a++) {
        int ga = atom_ids[a];
        if (ga < 1 || ga > atomnum || atom_index[ga] != -1) ok = 0;
        else atom_index[ga] = (int)a;
      }
      for (i = 1; i <= atomnum; i++) if (atom_index[i] < 0) ok = 0;
    }

    for (a = 0; ok && a < total_atoms; a++) {
      int ga = atom_ids[a];
      int no0 = Spe_Total_CNO[WhatSpecies[ga]];
      cache->atom_out_offset[a] = (uint32_t)out_off;
      cache->atom_orb_offset[a] = (uint32_t)orb_off;
      cache->atom_pair_offset[a] = (uint32_t)pair_off;
      cache->atom_no[a] = no0;
      for (Nc = 0; Nc < GridN_Atom[ga]; Nc++) cache->out_atom[out_off + (size_t)Nc] = (uint16_t)a;
      out_off += (uint64_t)GridN_Atom[ga];
      orb_off += (uint64_t)GridN_Atom[ga] * (uint64_t)no0;
      for (h_AN = 0; h_AN <= FNAN[ga]; h_AN++) {
        int gb = natn[ga][h_AN];
        int b = (gb >= 1 && gb <= atomnum) ? atom_index[gb] : -1;
        int no1 = Spe_Total_CNO[WhatSpecies[gb]];
        if (b < 0 || cdm_off > UINT32_MAX) {
          ok = 0;
          break;
        }
        cache->pair_neighbor[pair_off] = b;
        cache->pair_cdm_offset[pair_off] = (uint32_t)cdm_off;
        cdm_off += (uint64_t)spin_count * (uint64_t)no0 * (uint64_t)no1;
        pair_off++;
      }
    }
    if (out_off != total_outputs || orb_off != total_orbs || pair_off != total_pairs || cdm_off != total_cdm) ok = 0;
    if (ok) {
      cache->atom_out_offset[total_atoms] = (uint32_t)out_off;
      cache->atom_orb_offset[total_atoms] = (uint32_t)orb_off;
      cache->atom_pair_offset[total_atoms] = (uint32_t)pair_off;
    }

    cache->out_ptr[0] = 0;
    for (out = 0; ok && out < total_outputs; out++) {
      term_off += (uint64_t)degree[out];
      if (term_off > UINT32_MAX) ok = 0;
      cache->out_ptr[out + 1] = (uint32_t)term_off;
    }
    if (term_off != total_terms) ok = 0;

    for (out = 0; ok && out < total_outputs; out++) {
      uint32_t gn = out_gn[out];
      if ((size_t)gn >= grid_count || grid_degree[gn] == UINT32_MAX) ok = 0;
      else grid_degree[gn]++;
    }

    cache->grid_ptr[0] = 0;
    for (p = 0; ok && p < grid_count; p++) {
      uint64_t next = (uint64_t)cache->grid_ptr[p] + (uint64_t)grid_degree[p];
      if (next > UINT32_MAX) ok = 0;
      cache->grid_ptr[p + 1] = (uint32_t)next;
      grid_cursor[p] = cache->grid_ptr[p];
    }
    if ((size_t)cache->grid_ptr[grid_count] != total_outputs) ok = 0;

    for (out = 0; ok && out < total_outputs; out++) {
      uint32_t gn = out_gn[out];
      uint32_t slot = grid_cursor[gn]++;
      if ((size_t)slot >= total_outputs) ok = 0;
      else cache->grid_out[slot] = (uint32_t)out;
    }

    if (ok) {
      unsigned long long n2d = (unsigned long long)Ngrid1 * (unsigned long long)Ngrid2;
      int rank;
      for (rank = 0; rank < numprocs; rank++) {
        unsigned long long b0 = ((unsigned long long)rank * n2d + (unsigned long long)numprocs - 1ULL) /
                                (unsigned long long)numprocs;
        unsigned long long b1 = ((unsigned long long)(rank + 1) * n2d + (unsigned long long)numprocs - 1ULL) /
                                (unsigned long long)numprocs;
        unsigned long long disp = b0 * (unsigned long long)Ngrid3;
        unsigned long long count = (b1 - b0) * (unsigned long long)Ngrid3;
        if (disp > (unsigned long long)INT_MAX || count > (unsigned long long)INT_MAX) ok = 0;
        else {
          cache->scatter_displs[rank] = (int)disp;
          cache->scatter_counts[rank] = (int)count;
        }
      }
    }

    cache->device_bytes = 0;
    if (ok) {
      ok = SDG_add_bytes(&cache->device_bytes, total_orbs, sizeof(Type_Orbs_Grid)) &&
           SDG_add_bytes(&cache->device_bytes, total_atoms + 1, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_atoms + 1, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_atoms + 1, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_atoms, sizeof(int)) &&
           SDG_add_bytes(&cache->device_bytes, total_pairs, sizeof(int)) &&
           SDG_add_bytes(&cache->device_bytes, total_pairs, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_outputs, sizeof(uint16_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_outputs + 1, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_terms, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, grid_count + 1, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_outputs, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_cdm, sizeof(double)) &&
           SDG_add_bytes(&cache->device_bytes, (size_t)spin_count * grid_count, sizeof(double));
    }

    free(atom_index);
    free(grid_degree);
    free(grid_cursor);
  }

  MPI_Bcast(&ok, 1, MPI_INT, owner, mpi_comm_level1);
  if (!ok) goto fail;

  cache->ready = 1;
  cache->owner = owner;
  cache->spin_count = spin_count;
  cache->nh_bits = nh_bits;
  cache->nh_mask = nh_mask;
  cache->local_cdm_count = (int)local_counts[SDG_COUNT_CDM];
  cache->atom_count = total_atoms;
  cache->orbital_count = total_orbs;
  cache->output_count = total_outputs;
  cache->pair_count = total_pairs;
  cache->term_count = total_terms;
  cache->cdm_count = total_cdm;
  cache->global_grid_count = (size_t)Ngrid1 * (size_t)Ngrid2 * (size_t)Ngrid3;
  if (myid == owner) {
    cache->cdm_counts = cdm_counts; cdm_counts = NULL;
    cache->cdm_displs = cdm_displs; cdm_displs = NULL;
  }

  free(rank_counts);
  free(atom_ids);
  free(out_gn);
  free(degree);
  free(atom_counts); free(atom_displs);
  free(orb_counts); free(orb_displs);
  free(out_counts); free(out_displs);
  free(term_counts); free(term_displs);
  free(pair_counts); free(pair_displs);
  free(cdm_counts); free(cdm_displs);
  return 1;

fail:
  free(local_atom_ids);
  free(local_orbs);
  free(local_out_gn);
  free(local_degree);
  free(local_ptr);
  free(local_cursor);
  free(local_terms);
  free(rank_counts);
  free(atom_ids);
  free(out_gn);
  free(degree);
  free(atom_counts); free(atom_displs);
  free(orb_counts); free(orb_displs);
  free(out_counts); free(out_displs);
  free(term_counts); free(term_displs);
  free(pair_counts); free(pair_displs);
  free(cdm_counts); free(cdm_displs);
  SDG_free_host(cache);
  cache->unavailable = 1;
  return 0;
}

static int SDG_pack_local_cdm(double *****CDM, double *buffer, size_t expected, int spin_count)
{
  size_t p = 0;
  int Mc_AN, h_AN, Gc_AN, Gh_AN, NO0, NO1, spin, i, j;

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
    Gc_AN = M2G[Mc_AN];
    NO0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
      Gh_AN = natn[Gc_AN][h_AN];
      NO1 = Spe_Total_CNO[WhatSpecies[Gh_AN]];
      for (spin = 0; spin < spin_count; spin++) {
        for (i = 0; i < NO0; i++) {
          for (j = 0; j < NO1; j++) {
            if (p >= expected) return 0;
            buffer[p++] = CDM[spin][Mc_AN][h_AN][i][j];
          }
        }
      }
    }
  }
  return p == expected;
}

static int SDG_upload_device(SetDensityGpuCache *cache)
{
  if (cache->device_resident) {
    acc_update_device(cache->cdm, cache->cdm_count * sizeof(double));
    return 1;
  }

  /* Mark the upload as resident before the first allocation so a partial
     failure is rolled back by SDG_delete_device(). */
  cache->device_resident = 1;
  if (acc_copyin(cache->orbs, cache->orbital_count * sizeof(Type_Orbs_Grid)) == NULL ||
      acc_copyin(cache->atom_out_offset, (cache->atom_count + 1) * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->atom_orb_offset, (cache->atom_count + 1) * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->atom_pair_offset, (cache->atom_count + 1) * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->atom_no, cache->atom_count * sizeof(int)) == NULL ||
      acc_copyin(cache->pair_neighbor, cache->pair_count * sizeof(int)) == NULL ||
      acc_copyin(cache->pair_cdm_offset, cache->pair_count * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->out_atom, cache->output_count * sizeof(uint16_t)) == NULL ||
      acc_copyin(cache->out_ptr, (cache->output_count + 1) * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->terms, cache->term_count * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->grid_ptr, (cache->global_grid_count + 1) * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->grid_out, cache->output_count * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->cdm, cache->cdm_count * sizeof(double)) == NULL ||
      acc_create(cache->rho, cache->global_grid_count * (size_t)cache->spin_count * sizeof(double)) == NULL) {
    SDG_delete_device(cache);
    return 0;
  }
  return 1;
}

static void SDG_run_kernel(SetDensityGpuCache *cache)
{
  Type_Orbs_Grid *orbs = cache->orbs;
  uint32_t *atom_out_offset = cache->atom_out_offset;
  uint32_t *atom_orb_offset = cache->atom_orb_offset;
  uint32_t *atom_pair_offset = cache->atom_pair_offset;
  int *atom_no = cache->atom_no;
  int *pair_neighbor = cache->pair_neighbor;
  uint32_t *pair_cdm_offset = cache->pair_cdm_offset;
  uint16_t *out_atom = cache->out_atom;
  uint32_t *out_ptr = cache->out_ptr;
  uint32_t *terms = cache->terms;
  uint32_t *grid_ptr = cache->grid_ptr;
  uint32_t *grid_out = cache->grid_out;
  double *cdm = cache->cdm;
  double *rho = cache->rho;
  size_t atom_count = cache->atom_count;
  size_t orbital_count = cache->orbital_count;
  size_t output_count = cache->output_count;
  size_t pair_count = cache->pair_count;
  size_t term_count = cache->term_count;
  size_t cdm_count = cache->cdm_count;
  size_t grid_count = cache->global_grid_count;
  int spin_count = cache->spin_count;
  int nh_bits = cache->nh_bits;
  uint32_t nh_mask = cache->nh_mask;

#pragma acc parallel loop gang vector_length(128) \
  present(orbs[0:orbital_count], atom_out_offset[0:atom_count+1], atom_orb_offset[0:atom_count+1], \
          atom_pair_offset[0:atom_count+1], atom_no[0:atom_count], pair_neighbor[0:pair_count], \
          pair_cdm_offset[0:pair_count], out_atom[0:output_count], out_ptr[0:output_count+1], \
          terms[0:term_count], grid_ptr[0:grid_count+1], grid_out[0:output_count], \
          cdm[0:cdm_count], rho[0:grid_count*spin_count])
  for (size_t gn = 0; gn < grid_count; gn++) {
    double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

#pragma acc loop seq
    for (uint32_t q = grid_ptr[gn]; q < grid_ptr[gn + 1]; q++) {
      uint32_t out = grid_out[q];
      int a = (int)out_atom[out];
      uint32_t Nc0 = out - atom_out_offset[a];
      int NO0 = atom_no[a];
      size_t orb0 = (size_t)atom_orb_offset[a] + (size_t)Nc0 * (size_t)NO0;
      double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;

#pragma acc loop seq
      for (uint32_t t = out_ptr[out]; t < out_ptr[out + 1]; t++) {
        uint32_t code = terms[t];
        uint32_t Nh = code & nh_mask;
        uint32_t h = code >> nh_bits;
        uint32_t pair = atom_pair_offset[a] + h;
        int b = pair_neighbor[pair];
        int NO1 = atom_no[b];
        size_t orb1 = (size_t)atom_orb_offset[b] + (size_t)Nh * (size_t)NO1;
        size_t mat = (size_t)NO0 * (size_t)NO1;
        size_t base = (size_t)pair_cdm_offset[pair];
        double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;

#pragma acc loop seq
        for (int ii = 0; ii < NO0; ii++) {
          double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
#pragma acc loop seq
          for (int jj = 0; jj < NO1; jj++) {
            double phi1 = (double)orbs[orb1 + (size_t)jj];
            size_t ij = (size_t)ii * (size_t)NO1 + (size_t)jj;
            t0 += phi1 * cdm[base + ij];
            if (spin_count >= 2) t1 += phi1 * cdm[base + mat + ij];
            if (spin_count == 4) {
              t2 += phi1 * cdm[base + 2U * mat + ij];
              t3 += phi1 * cdm[base + 3U * mat + ij];
            }
          }
          {
            double phi0 = (double)orbs[orb0 + (size_t)ii];
            e0 += phi0 * t0;
            if (spin_count >= 2) e1 += phi0 * t1;
            if (spin_count == 4) {
              e2 += phi0 * t2;
              e3 += phi0 * t3;
            }
          }
        }
        a0 += e0;
        if (spin_count >= 2) a1 += e1;
        if (spin_count == 4) {
          a2 += e2;
          a3 += e3;
        }
      }
      g0 += a0;
      if (spin_count >= 2) g1 += a1;
      if (spin_count == 4) {
        g2 += a2;
        g3 += a3;
      }
    }

    rho[gn] = g0;
    if (spin_count >= 2) rho[grid_count + gn] = g1;
    if (spin_count == 4) {
      rho[2U * grid_count + gn] = g2;
      rho[3U * grid_count + gn] = -g3;
    }
  }
}

int Set_Density_Grid_GPU_Service(int Cnt_kind, int Calc_CntOrbital_ON, double *****CDM,
                                 double **Density_Grid_B0, double *elapsed)
{
  SetDensityGpuCache *cache = &SDG_cache;
  int numprocs, myid, owner = Host_ID;
  int enabled = 0, device_count = 0, device = 0;
  int spin_count = SpinP_switch + 1;
  int ok = 1, all_ok = 1;
  double start, end;
  double *local_cdm = NULL;
  size_t reserve_bytes;
  size_t free_bytes = 0, total_bytes = 0;
  int need_release = 0;
  int persistent;

  (void)Calc_CntOrbital_ON;
  dtime(&start);
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  if (myid == owner) {
    int requested = SDG_env_bool("OPENMX_DENSITY_GRID_GPU",
                                SDG_env_bool("OPENMX_SETDENSITY_GPU", 1));
    enabled = requested && scf_eigen_lib_flag == GPUSOLVER && (Solver == 2 || Solver == 3) &&
              Cnt_switch == 0 && (Cnt_kind == 0 || Cnt_kind == 1) &&
              (SpinP_switch == 0 || SpinP_switch == 1 || SpinP_switch == 3);
    if (enabled) {
      cudaError_t status = cudaGetDeviceCount(&device_count);
      if (status != cudaSuccess || device_count <= 0 || acc_get_num_devices(acc_device_nvidia) <= 0 ||
          !gpu_rank_device_usable()) enabled = 0;
    }
  }
  MPI_Bcast(&enabled, 1, MPI_INT, owner, mpi_comm_level1);
  if (!enabled || cache->unavailable) return 0;

  if (!cache->ready) {
    if (!SDG_build_cache(spin_count, owner, myid, numprocs)) return 0;
  }
  if (cache->spin_count != spin_count) {
    Set_Density_Grid_GPU_Invalidate();
    if (!SDG_build_cache(spin_count, owner, myid, numprocs)) return 0;
  }

  reserve_bytes = SDG_env_mib("OPENMX_DENSITY_GRID_GPU_RESERVE_MB", 256);
  if (myid == owner && !cache->device_resident) {
    cudaGetDevice(&device);
    acc_set_device_num(device, acc_device_nvidia);
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        free_bytes < cache->device_bytes || reserve_bytes > free_bytes - cache->device_bytes) need_release = 1;
  }
  MPI_Bcast(&need_release, 1, MPI_INT, owner, mpi_comm_level1);

  if (need_release) {
    /* First return only disposable storage.  In particular, do not destroy
       the collinear Cluster solver context here: it owns the transformed
       overlap matrix, and dropping it makes the next SCF diagonalize the
       overlap again.  OpenACC freelists are per MPI process, so every rank
       returns its deleted blocks before the owner retries cudaMemGetInfo(). */
    acc_wait_all();
    openmx_gemmul8ReleaseWorkspaces();
    acc_clear_freelists();
    MPI_Barrier(mpi_comm_level1);

    if (myid == owner) {
      need_release =
          (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
           free_bytes < cache->device_bytes || reserve_bytes > free_bytes - cache->device_bytes);
    }
    MPI_Bcast(&need_release, 1, MPI_INT, owner, mpi_comm_level1);
  }

  if (need_release && Solver == 2 && SpinP_switch == 3) {
    /* The non-collinear Cluster path keeps its final dense eigenvectors for a
       later scatter.  Preserve them on the host, then return only their device
       copy and its OpenACC freelist allocation. */
    Cluster_DFT_NonCol_DemoteGpuSolverCachedEVec();
    acc_wait_all();
    acc_clear_freelists();
    MPI_Barrier(mpi_comm_level1);
  }

  if (myid == owner && !cache->device_resident) {
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        free_bytes < cache->device_bytes || reserve_bytes > free_bytes - cache->device_bytes) {
      ok = 0;
      if (0 < level_stdout) {
        fprintf(stderr,
                "Set_Density_Grid GPU owner needs %.3f GiB plus %.3f GiB reserve, but only %.3f GiB is free; CPU fallback.\n",
                (double)cache->device_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)reserve_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)free_bytes / (1024.0 * 1024.0 * 1024.0));
        fflush(stderr);
      }
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, owner, mpi_comm_level1);
  if (!ok) {
    SDG_disable_current_epoch(cache);
    return 0;
  }

  local_cdm = (double *)SDG_malloc((size_t)cache->local_cdm_count, sizeof(double));
  ok = (local_cdm != NULL) && SDG_pack_local_cdm(CDM, local_cdm, (size_t)cache->local_cdm_count, spin_count);
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
  if (!all_ok) {
    free(local_cdm);
    SDG_disable_current_epoch(cache);
    return 0;
  }

  MPI_Gatherv(local_cdm, cache->local_cdm_count, MPI_DOUBLE,
              cache->cdm, cache->cdm_counts, cache->cdm_displs, MPI_DOUBLE, owner, mpi_comm_level1);
  free(local_cdm);

  if (myid == owner) {
    if (!SDG_upload_device(cache)) ok = 0;
    if (ok) {
      SDG_run_kernel(cache);
      acc_update_self(cache->rho, cache->global_grid_count * (size_t)spin_count * sizeof(double));
      acc_wait_all();
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, owner, mpi_comm_level1);
  if (!ok) {
    SDG_disable_current_epoch(cache);
    return 0;
  }

  persistent = SDG_env_bool("OPENMX_DENSITY_GRID_GPU_PERSISTENT", 0);
  if (myid == owner && !persistent) SDG_delete_device(cache);

  for (int spin = 0; spin < spin_count; spin++) {
    const double *sendbuf = (myid == owner) ? cache->rho + (size_t)spin * cache->global_grid_count : NULL;
    MPI_Scatterv(sendbuf, cache->scatter_counts, cache->scatter_displs, MPI_DOUBLE,
                 Density_Grid_B0[spin], My_NumGridB_AB, MPI_DOUBLE, owner, mpi_comm_level1);
  }

  Density_Grid_Copy_B2D(Density_Grid_B0);
  dtime(&end);
  *elapsed = end - start;
  return 1;
}

/**********************************************************************
   Distributed (per-rank) GPU quadrature for Set_Density_Grid.

   Every rank evaluates the atomic-sphere density of its own atoms on
   the device, reusing the Set_Hamiltonian matrix-elements tables (the
   central full-sphere orbitals, the FNAN-layout neighbour orbitals and
   the overlap index lists).  When Set_Hamiltonian keeps those tables
   device-resident the present_or_copyin staging below only attaches,
   so a single copy serves both routines.  The kernel reproduces the
   CPU loop exactly: per output point the pair contributions are summed
   in (h_AN, Nog) order and each contribution is
   sum_i orbs0[i] * (sum_j orbs1[j] * CDM[i][j]).  The result fills
   Tmp_Den_Grid and the ordinary A-to-B communication of
   Set_Density_Grid.c distributes it afterwards.
***********************************************************************/

typedef struct {
  int ready;
  int unavailable;
  int spin_count;
  int cnt_kind;
  int node_ranks;
  size_t output_count;     /* sum of GridN_Atom over the rank's atoms */
  size_t term_count;       /* overlap points of the rank's pairs */
  size_t dm_count;         /* packed density-matrix doubles */
  size_t extra_bytes;      /* device bytes owned by this mode alone */
  uint32_t *atom_out_base; /* Matomnum+2, host only */
  uint32_t *out_ptr;       /* CSR over the output points */
  uint32_t *term_pt;       /* CSR payload: global overlap-point index */
  uint32_t *pt_pair;       /* overlap point -> pair */
  double *dm;
  double *tmpden;
  SetHamiltonianMETables t;
} SDGLocalContext;

static SDGLocalContext SDG_local = {0};

static void SDG_local_free(void)
{
  SDGLocalContext *c = &SDG_local;

  free(c->atom_out_base);
  free(c->out_ptr);
  free(c->term_pt);
  free(c->pt_pair);
  free(c->dm);
  free(c->tmpden);
  memset(c, 0, sizeof(*c));
}

static int SDG_local_build(int Cnt_kind, int spin_count)
{
  SDGLocalContext *c = &SDG_local;
  uint32_t *degree = NULL;
  unsigned long long local_need, group_need = 0ULL;
  int ok = 1, all_ok = 1;
  int Mc_AN, p;
  size_t out, pos;

  {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0 ||
        acc_get_num_devices(acc_device_nvidia) <= 0) ok = 0;
  }

  if (ok && !Set_Hamiltonian_GetMatrixElementsTables(Cnt_kind, &c->t)) ok = 0;

  if (ok) {
    c->spin_count = spin_count;
    c->cnt_kind = Cnt_kind;
    c->term_count = c->t.total_nolg;
    c->dm_count = c->t.total_h;
    if (c->term_count > (size_t)UINT32_MAX) ok = 0;
  }

  if (ok) {
    c->atom_out_base = (uint32_t *)SDG_malloc((size_t)Matomnum + 2, sizeof(uint32_t));
    ok = (c->atom_out_base != NULL);
  }
  if (ok) {
    size_t base = 0;
    c->atom_out_base[0] = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
      c->atom_out_base[Mc_AN] = (uint32_t)base;
      base += (size_t)GridN_Atom[M2G[Mc_AN]];
      if (base > (size_t)UINT32_MAX) {
        ok = 0;
        break;
      }
    }
    if (ok) {
      c->atom_out_base[Matomnum + 1] = (uint32_t)base;
      c->output_count = base;
    }
  }

  if (ok) {
    degree = (uint32_t *)calloc(c->output_count == 0 ? 1 : c->output_count, sizeof(uint32_t));
    c->out_ptr = (uint32_t *)SDG_malloc(c->output_count + 1, sizeof(uint32_t));
    c->term_pt = (uint32_t *)SDG_malloc(c->term_count == 0 ? 1 : c->term_count, sizeof(uint32_t));
    c->pt_pair = (uint32_t *)SDG_malloc(c->term_count == 0 ? 1 : c->term_count, sizeof(uint32_t));
    c->dm = (double *)SDG_malloc(c->dm_count == 0 ? 1 : c->dm_count, sizeof(double));
    c->tmpden = (double *)SDG_malloc((size_t)spin_count *
                                     (c->output_count == 0 ? 1 : c->output_count), sizeof(double));
    ok = (degree != NULL && c->out_ptr != NULL && c->term_pt != NULL && c->pt_pair != NULL &&
          c->dm != NULL && c->tmpden != NULL);
  }

  if (ok) {
    for (p = 0; p < c->t.pair_count && ok; p++) {
      size_t base = c->atom_out_base[c->t.pair_Mc_AN[p]];
      size_t lo = c->t.pair_nolg_offset[p];
      size_t n = (size_t)c->t.pair_NOLG[p];

      for (pos = 0; pos < n; pos++) {
        size_t o = base + (size_t)c->t.nolg_Nc[lo + pos];
        if (o >= c->output_count) {
          ok = 0;
          break;
        }
        degree[o]++;
        c->pt_pair[lo + pos] = (uint32_t)p;
      }
    }
  }

  if (ok) {
    uint64_t running = 0;
    c->out_ptr[0] = 0;
    for (out = 0; out < c->output_count; out++) {
      running += (uint64_t)degree[out];
      if (running > UINT32_MAX) {
        ok = 0;
        break;
      }
      c->out_ptr[out + 1] = (uint32_t)running;
      degree[out] = c->out_ptr[out]; /* reuse as insertion cursor */
    }
    if (ok && (size_t)c->out_ptr[c->output_count] != c->term_count) ok = 0;
  }

  if (ok) {
    for (p = 0; p < c->t.pair_count; p++) {
      size_t base = c->atom_out_base[c->t.pair_Mc_AN[p]];
      size_t lo = c->t.pair_nolg_offset[p];
      size_t n = (size_t)c->t.pair_NOLG[p];

      for (pos = 0; pos < n; pos++) {
        size_t o = base + (size_t)c->t.nolg_Nc[lo + pos];
        c->term_pt[degree[o]++] = (uint32_t)(lo + pos);
      }
    }
  }

  free(degree);
  degree = NULL;

  if (ok) {
    c->extra_bytes = 0;
    ok = SDG_add_bytes(&c->extra_bytes, c->output_count + 1, sizeof(uint32_t)) &&
         SDG_add_bytes(&c->extra_bytes, c->term_count, sizeof(uint32_t)) &&
         SDG_add_bytes(&c->extra_bytes, c->term_count, sizeof(uint32_t)) &&
         SDG_add_bytes(&c->extra_bytes, c->dm_count, sizeof(double)) &&
         SDG_add_bytes(&c->extra_bytes, (size_t)spin_count * c->output_count, sizeof(double));
  }

  {
    MPI_Comm node_comm = MPI_COMM_NULL;
    int node_ranks = 1;

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_size(node_comm, &node_ranks);
    MPI_Comm_free(&node_comm);
    c->node_ranks = (node_ranks < 1) ? 1 : node_ranks;
  }

  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
  if (!all_ok) {
    SDG_local_free();
    SDG_local.unavailable = 1;
    return 0;
  }

  /* publish the mode's own transient need so long-lived caches
     (Set_Hamiltonian residency) leave room for it */
  local_need = (unsigned long long)c->extra_bytes;
  MPI_Allreduce(&local_need, &group_need, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mpi_comm_level1);
  OpenMX_GpuPhaseNeed_Register("density_grid_local", (size_t)group_need);

  c->ready = 1;
  return 1;
}

/* Device bytes this call would stage on top of what is already resident. */
static size_t SDG_local_call_bytes(const SDGLocalContext *c)
{
  size_t bytes = c->extra_bytes;

  if (!c->t.orbs0_resident) (void)SDG_add_bytes(&bytes, c->t.total_orbs0, sizeof(Type_Orbs_Grid));
  if (!c->t.orbs1_resident) (void)SDG_add_bytes(&bytes, c->t.total_orbs1, sizeof(Type_Orbs_Grid));
  if (!c->t.nolg_resident) (void)SDG_add_bytes(&bytes, c->t.total_nolg, sizeof(int));
  if (!c->t.meta_resident) {
    (void)SDG_add_bytes(&bytes, (size_t)c->t.pair_count * 2, sizeof(int));
    (void)SDG_add_bytes(&bytes, (size_t)c->t.pair_count * 4, sizeof(size_t));
  }
  return bytes;
}

int Set_Density_Grid_GPU_Local_Prepare(int Cnt_kind, int Calc_CntOrbital_ON)
{
  SDGLocalContext *c = &SDG_local;
  int spin_count = SpinP_switch + 1;
  int enabled, mode = 1;
  size_t reserve_bytes, need;
  size_t free_bytes = 0, total_bytes = 0;

  (void)Calc_CntOrbital_ON;

  /* 0 = off, 1 = auto (only when the shared Set_Hamiltonian tables are
     device-resident, so each call stages just the CSR/DM/density buffers),
     2 = always (stage the shared tables per call too) */
  {
    const char *value = getenv("OPENMX_DENSITY_GRID_GPU_LOCAL");
    if (value != NULL && value[0] != '\0') mode = atoi(value);
  }

  enabled = (mode != 0) &&
            SDG_env_bool("OPENMX_DENSITY_GRID_GPU", SDG_env_bool("OPENMX_SETDENSITY_GPU", 1)) &&
            scf_eigen_lib_flag == GPUSOLVER && (Solver == 2 || Solver == 3) &&
            Cnt_switch == 0 && (Cnt_kind == 0 || Cnt_kind == 1) &&
            (SpinP_switch == 0 || SpinP_switch == 1 || SpinP_switch == 3) &&
            gpu_rank_device_usable();
  if (!enabled) return 0;

  /* Mode 1 rides on the tables Set_Hamiltonian has already built.  Asking
     Set_Hamiltonian_GetMatrixElementsTables below would lazily BUILD the
     multi-GiB host tables on every rank whose Set_Hamiltonian ran on the
     CPU (a crowded device leaves most ranks without tables), and at
     1000-atom scale that simultaneous build alone can exhaust the host
     memory.  Probe for existing tables first — collectively, because
     SDG_local_build contains collectives and every rank must take the
     same branch. */
  if (mode == 1) {
    int my_ready = Set_Hamiltonian_MatrixElementsTables_Ready(Cnt_kind);
    int all_ready = 0;

    MPI_Allreduce(&my_ready, &all_ready, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
    if (!all_ready) return 0;
  }

  if (c->unavailable) return 0;

  if (c->ready && (c->spin_count != spin_count || c->cnt_kind != Cnt_kind)) {
    SDG_local_free();
  }
  if (!c->ready) {
    if (!SDG_local_build(Cnt_kind, spin_count)) return 0;
  }

  /* refresh the shared-table view: the pointers and the residency flags may
     have changed since the epoch was built */
  if (!Set_Hamiltonian_GetMatrixElementsTables(Cnt_kind, &c->t) ||
      c->t.total_nolg != c->term_count || c->t.total_h != c->dm_count) {
    SDG_local_free();
    SDG_local.unavailable = 1;
    return 0;
  }

  if (c->output_count == 0) return 0;

  /* In auto mode the per-call staging of the multi-GiB orbital tables is
     slower than the single-owner service, so engage only when the shared
     tables already live on the device (Set_Hamiltonian residency). */
  if (mode == 1 &&
      !(c->t.orbs0_resident && c->t.orbs1_resident && c->t.nolg_resident && c->t.meta_resident)) {
    return 0;
  }

  reserve_bytes = SDG_env_mib("OPENMX_DENSITY_GRID_GPU_RESERVE_MB", 256);
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return 0;
  need = SDG_local_call_bytes(c);
  if (free_bytes < reserve_bytes) return 0;
  if ((free_bytes - reserve_bytes) / (size_t)c->node_ranks < need) return 0;
  return 1;
}

int Set_Density_Grid_GPU_Local_Run(double *****CDM, double ***Tmp_Den_Grid)
{
  SDGLocalContext *c = &SDG_local;
  const int spin_count = c->spin_count;
  const int pair_count = c->t.pair_count;
  const size_t output_count = c->output_count;
  const size_t term_count = c->term_count;
  const size_t dm_count = c->dm_count;
  const size_t total_nolg = c->t.total_nolg;
  const size_t total_orbs0 = c->t.total_orbs0;
  const size_t total_orbs1 = c->t.total_orbs1;
  const int *pair_NO0 = c->t.pair_NO0;
  const int *pair_NO1 = c->t.pair_NO1;
  const int *nolg_Nc = c->t.nolg_Nc;
  const size_t *pair_h_offset = c->t.pair_h_offset;
  const size_t *pair_nolg_offset = c->t.pair_nolg_offset;
  const size_t *pair_orbs0_offset = c->t.pair_orbs0_offset;
  const size_t *pair_orbs1_offset = c->t.pair_orbs1_offset;
  const Type_Orbs_Grid *orbs0buf = c->t.orbs0buf;
  const Type_Orbs_Grid *orbs1buf = c->t.orbs1buf;
  uint32_t *out_ptr = c->out_ptr;
  uint32_t *term_pt = c->term_pt;
  uint32_t *pt_pair = c->pt_pair;
  double *dm = c->dm;
  double *tmpden = c->tmpden;
  int Mc_AN, spin;

  if (!c->ready || output_count == 0) return 0;
  if (!SDG_pack_local_cdm(CDM, dm, dm_count, spin_count)) return 0;

#pragma acc data copyin(pair_NO0[0:pair_count], pair_NO1[0:pair_count],                            \
                        pair_h_offset[0:pair_count], pair_nolg_offset[0:pair_count],               \
                        pair_orbs0_offset[0:pair_count], pair_orbs1_offset[0:pair_count],          \
                        nolg_Nc[0:total_nolg], orbs0buf[0:total_orbs0], orbs1buf[0:total_orbs1],   \
                        out_ptr[0:output_count+1], term_pt[0:term_count], pt_pair[0:term_count],   \
                        dm[0:dm_count])                                                            \
                 copyout(tmpden[0:(size_t)spin_count*output_count])
  {
#pragma acc parallel loop gang vector_length(128)                                                  \
    present(pair_NO0[0:pair_count], pair_NO1[0:pair_count], pair_h_offset[0:pair_count],           \
            pair_nolg_offset[0:pair_count], pair_orbs0_offset[0:pair_count],                       \
            pair_orbs1_offset[0:pair_count], nolg_Nc[0:total_nolg], orbs0buf[0:total_orbs0],       \
            orbs1buf[0:total_orbs1], out_ptr[0:output_count+1], term_pt[0:term_count],             \
            pt_pair[0:term_count], dm[0:dm_count], tmpden[0:(size_t)spin_count*output_count])
    for (size_t out = 0; out < output_count; out++) {
      double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

#pragma acc loop seq
      for (uint32_t q = out_ptr[out]; q < out_ptr[out + 1]; q++) {
        uint32_t pt = term_pt[q];
        uint32_t pair = pt_pair[pt];
        int NO0 = pair_NO0[pair];
        int NO1 = pair_NO1[pair];
        size_t mat = (size_t)NO0 * (size_t)NO1;
        size_t base = pair_h_offset[pair];
        size_t o0 = pair_orbs0_offset[pair] + (size_t)nolg_Nc[pt] * (size_t)NO0;
        size_t o1 = pair_orbs1_offset[pair] + ((size_t)pt - pair_nolg_offset[pair]) * (size_t)NO1;
        double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;

#pragma acc loop seq
        for (int ii = 0; ii < NO0; ii++) {
          double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
#pragma acc loop seq
          for (int jj = 0; jj < NO1; jj++) {
            double phi1 = (double)orbs1buf[o1 + (size_t)jj];
            size_t ij = (size_t)ii * (size_t)NO1 + (size_t)jj;
            t0 += phi1 * dm[base + ij];
            if (spin_count >= 2) t1 += phi1 * dm[base + mat + ij];
            if (spin_count == 4) {
              t2 += phi1 * dm[base + 2U * mat + ij];
              t3 += phi1 * dm[base + 3U * mat + ij];
            }
          }
          {
            double phi0 = (double)orbs0buf[o0 + (size_t)ii];
            e0 += phi0 * t0;
            if (spin_count >= 2) e1 += phi0 * t1;
            if (spin_count == 4) {
              e2 += phi0 * t2;
              e3 += phi0 * t3;
            }
          }
        }
        g0 += e0;
        if (spin_count >= 2) g1 += e1;
        if (spin_count == 4) {
          g2 += e2;
          g3 += e3;
        }
      }

      tmpden[out] = g0;
      if (spin_count >= 2) tmpden[output_count + out] = g1;
      if (spin_count == 4) {
        tmpden[2U * output_count + out] = g2;
        tmpden[3U * output_count + out] = g3;
      }
    }
  }

  for (spin = 0; spin < spin_count; spin++) {
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
      size_t base = c->atom_out_base[Mc_AN];
      size_t n = (size_t)GridN_Atom[M2G[Mc_AN]];

      memcpy(Tmp_Den_Grid[spin][Mc_AN], tmpden + (size_t)spin * output_count + base,
             sizeof(double) * n);
    }
  }
  return 1;
}
