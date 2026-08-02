/**********************************************************************
  Set_Orbitals_Grid.c:

   Set_Orbitals_Grid.c is a subroutine to calculate the value of basis
   functions on each grid point.

  Log of Set_Orbitals_Grid.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>
#include <openacc.h>
#include "set_cuda_default_device_from_local_rank.h"

/***********************************************************************
   GPU evaluation of the primitive (Cnt_kind==0) basis orbitals.

   The per-point math is a verbatim transcription of the inlined
   Get_Orbitals in the host loops below; the device covers uncontracted
   bases with L0 <= SOG_L0MAX (the explicit real-harmonic formulas).
   Anything else (orbital optimization, L0 >= 4 via ComplexSH) keeps the
   untouched host path.  All transient device memory is claimed through
   one acc_malloc arena so that an out-of-memory device never aborts the
   run: acc_malloc reports NULL and the caller falls back to the host.
   Disable with OPENMX_ORBS_GRID_GPU=0.
***********************************************************************/

#define SOG_L0MAX  3
#define SOG_MULMAX 8

typedef struct {
  int wan;              /* species of the evaluated atom */
  int no;               /* orbitals per grid point */
  int pt0;              /* first flat point index of this pair */
  int pad;
  size_t out;           /* first output slot in the chi buffer */
  double gx, gy, gz;    /* Gxyz of the evaluated atom */
  double ax, ay, az;    /* atv[Rnh]; zero for the part-1 self evaluation */
} SOG_GpuPair;

static int SOG_env_flag(const char *name, int default_value)
{
  const char *value = getenv(name);

  if (value == NULL || value[0] == '\0') return default_value;
  return atoi(value) != 0;
}

static int SOG_gpu_eligible(int Cnt_kind)
{
  static int device_checked = 0, device_ok = 0;
  int w, L0;

  if (Cnt_kind != 0) return 0;
  if (!SOG_env_flag("OPENMX_ORBS_GRID_GPU", 1)) return 0;

  /* the staging buffers and the device stores are float */
  if (sizeof(Type_Orbs_Grid) != sizeof(float)) return 0;

  for (w = 0; w < SpeciesNum; w++) {
    if (SOG_L0MAX < Spe_MaxL_Basis[w]) return 0;
    for (L0 = 0; L0 <= Spe_MaxL_Basis[w]; L0++) {
      if (SOG_MULMAX < Spe_Num_Basis[w][L0]) return 0;
    }
    if (Spe_Num_Mesh_PAO[w] < 6) return 0;
  }

  if (!device_checked) {
    device_checked = 1;
    /* the probe, not a bare device count: a rank whose context or module
       load cannot complete would abort inside acc_malloc otherwise */
    device_ok = gpu_rank_device_usable();
  }
  return device_ok;
}

static size_t SOG_arena_off(size_t *pos, size_t bytes)
{
  size_t off = *pos;

  *pos = (off + bytes + 511U) & ~(size_t)511U;
  return off;
}

/* one shot; the caller has a host fallback */
static void *SOG_arena_try(size_t bytes)
{
  void *arena = acc_malloc(bytes);

  if (arena == NULL) {
    /* our own freelist may hoard mismatched freed blocks */
    if (cudaDeviceSynchronize() == cudaSuccess) acc_clear_freelists();
    arena = acc_malloc(bytes);
  }
  return arena;
}

/* One launch evaluates every grid point of one atom (part 1:
   single_pair && identity_idx, pairs[0] describes the atom itself) or
   every overlap point of all remote neighbours of one atom (part 2).
   The body mirrors the inlined Get_Orbitals of the host loops below;
   keep the two in sync. */
static void SOG_gpu_eval(int npts, int identity_idx, int single_pair,
                         const int *nog, const int *ppr, const SOG_GpuPair *pairs,
                         const int *gla, const int *cla, const double *atvf,
                         const double *rv_all, const double *rwf_all,
                         const size_t *rv_off, const size_t *rwf_base,
                         const int *sp_mesh, const int *sp_maxl, const int *sp_nb,
                         int ng23, int ng3,
                         double g11, double g12, double g13,
                         double g21, double g22, double g23,
                         double g31, double g32, double g33,
                         double org1, double org2, double org3,
                         float *chi)
{
  int ip;

#pragma acc parallel loop gang vector vector_length(128) \
  deviceptr(nog, ppr, pairs, gla, cla, atvf, rv_all, rwf_all, rv_off, rwf_base, \
            sp_mesh, sp_maxl, sp_nb, chi)
  for (ip = 0; ip < npts; ip++) {

    const int pair = (single_pair ? 0 : ppr[ip]);
    const SOG_GpuPair pr = pairs[pair];
    const int wan = pr.wan;
    const int Nc = (identity_idx ? ip : nog[ip]);
    const int GNc = gla[Nc];
    const int GRc = cla[Nc];
    float *out = chi + pr.out + (size_t)(ip - pr.pt0) * (size_t)pr.no;

    /* Get_Grid_XYZ */
    const int n1 = GNc / ng23;
    const int n2 = (GNc - n1 * ng23) / ng3;
    const int n3 = GNc - n1 * ng23 - n2 * ng3;

    const double x = ((double)n1 * g11 + (double)n2 * g21 + (double)n3 * g31 + org1)
                     + atvf[3 * (size_t)GRc + 0] - pr.gx - pr.ax;
    const double y = ((double)n1 * g12 + (double)n2 * g22 + (double)n3 * g32 + org2)
                     + atvf[3 * (size_t)GRc + 1] - pr.gy - pr.ay;
    const double z = ((double)n1 * g13 + (double)n2 * g23 + (double)n3 * g33 + org3)
                     + atvf[3 * (size_t)GRc + 2] - pr.gz - pr.az;

    const int mesh = sp_mesh[wan];
    const int maxl = sp_maxl[wan];
    const double *rv = rv_all + rv_off[wan];

    double RF[SOG_L0MAX + 1][SOG_MULMAX];
    double AF[SOG_L0MAX + 1][2 * SOG_L0MAX + 1];
    double R, Q, P;
    int po = 0;
    int L0, Mul0, M0, i1;

    /* xyz2spherical */
    {
      const double Min_r = 10e-15;
      double dum = x * x + y * y;
      double r = sqrt(dum + z * z);
      double r1 = sqrt(dum);
      double dum1, theta, phi;

      if (Min_r <= r) {

        if (r < fabs(z)) dum1 = (z < 0.0 ? -1.0 : 1.0) * 1.0;
        else dum1 = z / r;

        theta = acos(dum1);

        if (Min_r <= r1) {
          if (0.0 <= x) {

            if (r1 < fabs(y)) dum1 = (y < 0.0 ? -1.0 : 1.0) * 1.0;
            else dum1 = y / r1;

            phi = asin(dum1);
          }
          else {

            if (r1 < fabs(y)) dum1 = (y < 0.0 ? -1.0 : 1.0) * 1.0;
            else dum1 = y / r1;

            phi = PI - asin(dum1);
          }
        }
        else {
          phi = 0.0;
        }
      }
      else {
        theta = 0.5 * PI;
        phi = 0.0;
      }

      R = r;
      Q = theta;
      P = phi;
    }

    /* radial spline of every (L0, Mul0) */
    if (rv[mesh - 1] < R) {
      /* outside the PAO mesh: every RF is zero, so every orbital is zero */
      po = 1;
    }
    else {

      const int below = (R < rv[0]);
      double h1, h2, h3, x1, x2, y1, y2, y12, y22;
      double dum, dum1, dum2, dum3, dum4;
      double rm = 0.0;
      size_t rrow = rwf_base[wan];
      int m;

      if (below) {
        m = 4;
        rm = rv[m];

        h1 = rv[m - 1] - rv[m - 2];
        h2 = rv[m] - rv[m - 1];
        h3 = rv[m + 1] - rv[m];

        x1 = rm - rv[m - 1];
        x2 = rm - rv[m];
      }
      else {
        int mp_min = 0, mp_max = mesh - 1;

        do {
          m = (mp_min + mp_max) / 2;
          if (rv[m] < R) mp_min = m;
          else mp_max = m;
        }
        while ((mp_max - mp_min) != 1);
        m = mp_max;

        h1 = rv[m - 1] - rv[m - 2];
        h2 = rv[m] - rv[m - 1];
        h3 = rv[m + 1] - rv[m];

        x1 = R - rv[m - 1];
        x2 = R - rv[m];
      }

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

      for (L0 = 0; L0 <= maxl; L0++) {
        const int nb = sp_nb[wan * (SOG_L0MAX + 1) + L0];

        for (Mul0 = 0; Mul0 < nb; Mul0++) {

          const double *rw = rwf_all + rrow;
          double f1 = rw[m - 2];
          double f2 = rw[m - 1];
          double f3 = rw[m];
          double f4 = rw[m + 1];
          double g1, g2, f;

          rrow += (size_t)mesh;

          if (m == 1) {
            h1 = -(h2 + h3);
            f1 = f4;
          }
          else if (m == (mesh - 1)) {
            h3 = -(h1 + h2);
            f4 = f1;
          }

          dum = f3 - f2;
          g1 = dum * dum1 + (f2 - f1) * dum2;
          g2 = (f4 - f3) * dum3 + dum * dum4;

          f = y22 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2)
            + y12 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1);

          if (below) {

            double df = 2.0 * y2 / h2 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2)
              + y22 * (2.0 * f2 + h2 * g1) / h2
              + 2.0 * y1 / h2 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1)
              - y12 * (2.0 * f3 - h2 * g2) / h2;
            double a, b, c, d;

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
          }
          else {
            RF[L0][Mul0] = f;
          }
        }
      }
    }

    if (po == 0) {

      /* Angular */
      const double siQ = sin(Q);
      const double coQ = cos(Q);
      const double siP = sin(P);
      const double coP = cos(P);
      double dum, dum1, dum2;

      for (L0 = 0; L0 <= maxl; L0++) {

        if (L0 == 0) {
          AF[0][0] = 0.282094791773878;
        }
        else if (L0 == 1) {
          dum = 0.48860251190292 * siQ;
          AF[1][0] = dum * coP;
          AF[1][1] = dum * siP;
          AF[1][2] = 0.48860251190292 * coQ;
        }
        else if (L0 == 2) {
          dum1 = siQ * siQ;
          dum2 = 1.09254843059208 * siQ * coQ;
          AF[2][0] = 0.94617469575756 * coQ * coQ - 0.31539156525252;
          AF[2][1] = 0.54627421529604 * dum1 * (1.0 - 2.0 * siP * siP);
          AF[2][2] = 1.09254843059208 * dum1 * siP * coP;
          AF[2][3] = dum2 * coP;
          AF[2][4] = dum2 * siP;
        }
        else if (L0 == 3) {
          AF[3][0] = 0.373176332590116 * (5.0 * coQ * coQ * coQ - 3.0 * coQ);
          AF[3][1] = 0.457045799464466 * coP * siQ * (5.0 * coQ * coQ - 1.0);
          AF[3][2] = 0.457045799464466 * siP * siQ * (5.0 * coQ * coQ - 1.0);
          AF[3][3] = 1.44530572132028 * siQ * siQ * coQ * (coP * coP - siP * siP);
          AF[3][4] = 2.89061144264055 * siQ * siQ * coQ * siP * coP;
          AF[3][5] = 0.590043589926644 * siQ * siQ * siQ * (4.0 * coP * coP * coP - 3.0 * coP);
          AF[3][6] = 0.590043589926644 * siQ * siQ * siQ * (3.0 * siP - 4.0 * siP * siP * siP);
        }
      }

      /* Chi */
      i1 = -1;
      for (L0 = 0; L0 <= maxl; L0++) {
        const int nb = sp_nb[wan * (SOG_L0MAX + 1) + L0];

        for (Mul0 = 0; Mul0 < nb; Mul0++) {
          for (M0 = 0; M0 <= 2 * L0; M0++) {
            i1++;
            out[i1] = (float)(RF[L0][Mul0] * AF[L0][M0]);
          }
        }
      }
    }
    else {
      for (i1 = 0; i1 < pr.no; i1++) out[i1] = 0.0f;
    }
  }
}

/* returns 1 when the whole update ran on the device; 0 keeps the host path */
static int Set_Orbitals_Grid_GPU(void)
{
  static int fail_notice_done = 0, active_notice_done = 0;
  int myid;
  int Mc_AN, Gc_AN, Cwan, h_AN, Gh_AN, Hwan, Rnh, NO0, NO1;
  int w, L0, Mul0, i, npair_cap;
  size_t max_pts = 0, max_chi = 0, max_fpts = 0, max_fchi = 0;
  size_t rv_cnt = 0, rwf_cnt = 0, atv_rows, r, cnt;
  size_t *rv_off_h = NULL, *rwf_base_h = NULL;
  int *sp_mesh_h = NULL, *sp_maxl_h = NULL, *sp_nb_h = NULL;
  double *pao_rv = NULL, *pao_rwf = NULL, *atv_flat = NULL;
  float *chi_host = NULL;
  int *ppr_host = NULL, *pair_hAN = NULL;
  SOG_GpuPair *pair_host = NULL;
  unsigned char *arena = NULL;
  size_t pos = 0, arena_bytes;
  size_t o_rv, o_rwf, o_rvo, o_rwb, o_spm, o_spx, o_spb, o_atv;
  size_t o_gla, o_cla, o_nog, o_ppr, o_prs, o_chi;
  double Stime_atom, Etime_atom;

  const int ng23 = Ngrid2 * Ngrid3;
  const int ng3 = Ngrid3;
  const double g11 = gtv[1][1], g12 = gtv[1][2], g13 = gtv[1][3];
  const double g21 = gtv[2][1], g22 = gtv[2][2], g23 = gtv[2][3];
  const double g31 = gtv[3][1], g32 = gtv[3][2], g33 = gtv[3][3];
  const double org1 = Grid_Origin[1], org2 = Grid_Origin[2], org3 = Grid_Origin[3];

  MPI_Comm_rank(mpi_comm_level1, &myid);

  /* ---- per-atom staging maxima ---- */

  npair_cap = 0;
  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
    size_t fpts = 0, fchi = 0;
    int npair = 0;

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_NO[Cwan];

    if (max_pts < (size_t)GridN_Atom[Gc_AN]) max_pts = (size_t)GridN_Atom[Gc_AN];
    cnt = (size_t)GridN_Atom[Gc_AN] * (size_t)NO0;
    if (max_chi < cnt) max_chi = cnt;

    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
      Gh_AN = natn[Gc_AN][h_AN];
      if (G2ID[Gh_AN] != myid && 0 < NumOLG[Mc_AN][h_AN]) {
        NO1 = Spe_Total_NO[WhatSpecies[Gh_AN]];
        fpts += (size_t)NumOLG[Mc_AN][h_AN];
        fchi += (size_t)NumOLG[Mc_AN][h_AN] * (size_t)NO1;
        npair++;
      }
    }
    if (max_fpts < fpts) max_fpts = fpts;
    if (max_fchi < fchi) max_fchi = fchi;
    if (npair_cap < npair) npair_cap = npair;
  }
  if (npair_cap < 1) npair_cap = 1;

  /* ---- flatten the PAO radial tables and atv ---- */

  rv_off_h = (size_t*)malloc(sizeof(size_t) * (size_t)(SpeciesNum + 1));
  rwf_base_h = (size_t*)malloc(sizeof(size_t) * (size_t)(SpeciesNum + 1));
  sp_mesh_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum);
  sp_maxl_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum);
  sp_nb_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum * (SOG_L0MAX + 1));

  if (rv_off_h == NULL || rwf_base_h == NULL || sp_mesh_h == NULL ||
      sp_maxl_h == NULL || sp_nb_h == NULL) goto host_fallback;

  for (w = 0; w < SpeciesNum; w++) {
    rv_off_h[w] = rv_cnt;
    rwf_base_h[w] = rwf_cnt;
    sp_mesh_h[w] = Spe_Num_Mesh_PAO[w];
    sp_maxl_h[w] = Spe_MaxL_Basis[w];
    for (L0 = 0; L0 <= SOG_L0MAX; L0++) {
      int nb = (L0 <= Spe_MaxL_Basis[w] ? Spe_Num_Basis[w][L0] : 0);

      sp_nb_h[w * (SOG_L0MAX + 1) + L0] = nb;
      rwf_cnt += (size_t)nb * (size_t)Spe_Num_Mesh_PAO[w];
    }
    rv_cnt += (size_t)Spe_Num_Mesh_PAO[w];
  }
  rv_off_h[SpeciesNum] = rv_cnt;
  rwf_base_h[SpeciesNum] = rwf_cnt;

  pao_rv = (double*)malloc(sizeof(double) * (rv_cnt == 0 ? 1 : rv_cnt));
  pao_rwf = (double*)malloc(sizeof(double) * (rwf_cnt == 0 ? 1 : rwf_cnt));
  atv_rows = (size_t)TCpyCell + 1;
  atv_flat = (double*)malloc(sizeof(double) * atv_rows * 3);
  chi_host = (float*)malloc(sizeof(float) * ((max_chi < max_fchi ? max_fchi : max_chi) == 0 ? 1 :
                                             (max_chi < max_fchi ? max_fchi : max_chi)));
  ppr_host = (int*)malloc(sizeof(int) * (max_fpts == 0 ? 1 : max_fpts));
  pair_hAN = (int*)malloc(sizeof(int) * (size_t)npair_cap);
  pair_host = (SOG_GpuPair*)malloc(sizeof(SOG_GpuPair) * (size_t)npair_cap);

  if (pao_rv == NULL || pao_rwf == NULL || atv_flat == NULL || chi_host == NULL ||
      ppr_host == NULL || pair_hAN == NULL || pair_host == NULL) goto host_fallback;

  for (w = 0; w < SpeciesNum; w++) {
    size_t rpos = rwf_base_h[w];

    for (i = 0; i < Spe_Num_Mesh_PAO[w]; i++) {
      pao_rv[rv_off_h[w] + (size_t)i] = Spe_PAO_RV[w][i];
    }
    for (L0 = 0; L0 <= Spe_MaxL_Basis[w]; L0++) {
      for (Mul0 = 0; Mul0 < Spe_Num_Basis[w][L0]; Mul0++) {
        for (i = 0; i < Spe_Num_Mesh_PAO[w]; i++) {
          pao_rwf[rpos++] = Spe_PAO_RWF[w][L0][Mul0][i];
        }
      }
    }
  }

  for (r = 0; r < atv_rows; r++) {
    atv_flat[3 * r + 0] = atv[r][1];
    atv_flat[3 * r + 1] = atv[r][2];
    atv_flat[3 * r + 2] = atv[r][3];
  }

  /* ---- one arena claim covers the whole footprint ---- */

  o_rv = SOG_arena_off(&pos, sizeof(double) * (rv_cnt == 0 ? 1 : rv_cnt));
  o_rwf = SOG_arena_off(&pos, sizeof(double) * (rwf_cnt == 0 ? 1 : rwf_cnt));
  o_rvo = SOG_arena_off(&pos, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  o_rwb = SOG_arena_off(&pos, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  o_spm = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum);
  o_spx = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum);
  o_spb = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum * (SOG_L0MAX + 1));
  o_atv = SOG_arena_off(&pos, sizeof(double) * atv_rows * 3);
  o_gla = SOG_arena_off(&pos, sizeof(int) * (max_pts == 0 ? 1 : max_pts));
  o_cla = SOG_arena_off(&pos, sizeof(int) * (max_pts == 0 ? 1 : max_pts));
  o_nog = SOG_arena_off(&pos, sizeof(int) * (max_fpts == 0 ? 1 : max_fpts));
  o_ppr = SOG_arena_off(&pos, sizeof(int) * (max_fpts == 0 ? 1 : max_fpts));
  o_prs = SOG_arena_off(&pos, sizeof(SOG_GpuPair) * (size_t)npair_cap);
  cnt = (max_chi < max_fchi ? max_fchi : max_chi);
  o_chi = SOG_arena_off(&pos, sizeof(float) * (cnt == 0 ? 1 : cnt));
  arena_bytes = pos;

  arena = (unsigned char*)SOG_arena_try(arena_bytes);
  if (arena == NULL) {
    if (!fail_notice_done) {
      fail_notice_done = 1;
      fprintf(stderr,
              "Set_Orbitals_Grid: %.1f MiB of device staging unavailable; using the host path.\n",
              (double)arena_bytes / (1024.0 * 1024.0));
      fflush(stderr);
    }
    goto host_fallback;
  }

  acc_memcpy_to_device(arena + o_rv, pao_rv, sizeof(double) * (rv_cnt == 0 ? 1 : rv_cnt));
  acc_memcpy_to_device(arena + o_rwf, pao_rwf, sizeof(double) * (rwf_cnt == 0 ? 1 : rwf_cnt));
  acc_memcpy_to_device(arena + o_rvo, rv_off_h, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  acc_memcpy_to_device(arena + o_rwb, rwf_base_h, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  acc_memcpy_to_device(arena + o_spm, sp_mesh_h, sizeof(int) * (size_t)SpeciesNum);
  acc_memcpy_to_device(arena + o_spx, sp_maxl_h, sizeof(int) * (size_t)SpeciesNum);
  acc_memcpy_to_device(arena + o_spb, sp_nb_h, sizeof(int) * (size_t)SpeciesNum * (SOG_L0MAX + 1));
  acc_memcpy_to_device(arena + o_atv, atv_flat, sizeof(double) * atv_rows * 3);

  {
    const int *d_gla = (const int*)(const void*)(arena + o_gla);
    const int *d_cla = (const int*)(const void*)(arena + o_cla);
    const int *d_nog = (const int*)(const void*)(arena + o_nog);
    const int *d_ppr = (const int*)(const void*)(arena + o_ppr);
    const SOG_GpuPair *d_prs = (const SOG_GpuPair*)(const void*)(arena + o_prs);
    const double *d_atv = (const double*)(const void*)(arena + o_atv);
    const double *d_rv = (const double*)(const void*)(arena + o_rv);
    const double *d_rwf = (const double*)(const void*)(arena + o_rwf);
    const size_t *d_rvo = (const size_t*)(const void*)(arena + o_rvo);
    const size_t *d_rwb = (const size_t*)(const void*)(arena + o_rwb);
    const int *d_spm = (const int*)(const void*)(arena + o_spm);
    const int *d_spx = (const int*)(const void*)(arena + o_spx);
    const int *d_spb = (const int*)(const void*)(arena + o_spb);
    float *d_chi = (float*)(void*)(arena + o_chi);

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

      int npts, npair;
      size_t fpts, fchi;

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_NO[Cwan];
      npts = GridN_Atom[Gc_AN];

      /* ---- part 1: the atom's own orbitals on its grid ---- */

      if (0 < npts) {

        int Nc;

        acc_memcpy_to_device((void*)d_gla, GridListAtom[Mc_AN], sizeof(int) * (size_t)npts);
        acc_memcpy_to_device((void*)d_cla, CellListAtom[Mc_AN], sizeof(int) * (size_t)npts);

        pair_host[0].wan = Cwan;
        pair_host[0].no = NO0;
        pair_host[0].pt0 = 0;
        pair_host[0].pad = 0;
        pair_host[0].out = 0;
        pair_host[0].gx = Gxyz[Gc_AN][1];
        pair_host[0].gy = Gxyz[Gc_AN][2];
        pair_host[0].gz = Gxyz[Gc_AN][3];
        pair_host[0].ax = 0.0;
        pair_host[0].ay = 0.0;
        pair_host[0].az = 0.0;
        acc_memcpy_to_device((void*)d_prs, pair_host, sizeof(SOG_GpuPair));

        SOG_gpu_eval(npts, 1, 1, d_nog, d_ppr, d_prs, d_gla, d_cla, d_atv,
                     d_rv, d_rwf, d_rvo, d_rwb, d_spm, d_spx, d_spb,
                     ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33,
                     org1, org2, org3, d_chi);

        acc_memcpy_from_device(chi_host, d_chi, sizeof(float) * (size_t)npts * (size_t)NO0);

#pragma omp parallel for private(i)
        for (Nc = 0; Nc < npts; Nc++) {
          const float *src = chi_host + (size_t)Nc * (size_t)NO0;
          Type_Orbs_Grid *dst = Orbs_Grid[Mc_AN][Nc];

          for (i = 0; i < NO0; i++) dst[i] = src[i];
        }
      }

      /* ---- part 2: remote neighbours on the shared grid points ---- */

      npair = 0;
      fpts = 0;
      fchi = 0;
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

        Gh_AN = natn[Gc_AN][h_AN];

        if (G2ID[Gh_AN] != myid && 0 < NumOLG[Mc_AN][h_AN]) {

          const int nolg = NumOLG[Mc_AN][h_AN];
          size_t k;

          Rnh = ncn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          NO1 = Spe_Total_NO[Hwan];

          pair_host[npair].wan = Hwan;
          pair_host[npair].no = NO1;
          pair_host[npair].pt0 = (int)fpts;
          pair_host[npair].pad = 0;
          pair_host[npair].out = fchi;
          pair_host[npair].gx = Gxyz[Gh_AN][1];
          pair_host[npair].gy = Gxyz[Gh_AN][2];
          pair_host[npair].gz = Gxyz[Gh_AN][3];
          pair_host[npair].ax = atv[Rnh][1];
          pair_host[npair].ay = atv[Rnh][2];
          pair_host[npair].az = atv[Rnh][3];
          pair_hAN[npair] = h_AN;

          acc_memcpy_to_device((void*)(d_nog + fpts), GListTAtoms1[Mc_AN][h_AN],
                               sizeof(int) * (size_t)nolg);
          for (k = 0; k < (size_t)nolg; k++) ppr_host[fpts + k] = npair;

          fpts += (size_t)nolg;
          fchi += (size_t)nolg * (size_t)NO1;
          npair++;
        }
      }

      if (0 < npair) {

        int p;

        acc_memcpy_to_device((void*)d_ppr, ppr_host, sizeof(int) * fpts);
        acc_memcpy_to_device((void*)d_prs, pair_host, sizeof(SOG_GpuPair) * (size_t)npair);

        SOG_gpu_eval((int)fpts, 0, 0, d_nog, d_ppr, d_prs, d_gla, d_cla, d_atv,
                     d_rv, d_rwf, d_rvo, d_rwb, d_spm, d_spx, d_spb,
                     ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33,
                     org1, org2, org3, d_chi);

        acc_memcpy_from_device(chi_host, d_chi, sizeof(float) * fchi);

        for (p = 0; p < npair; p++) {

          const int nolg = NumOLG[Mc_AN][pair_hAN[p]];
          const float *base = chi_host + pair_host[p].out;
          Type_Orbs_Grid **dst_rows = Orbs_Grid_FNAN[Mc_AN][pair_hAN[p]];
          int Nog, j;

          NO1 = pair_host[p].no;

#pragma omp parallel for private(j)
          for (Nog = 0; Nog < nolg; Nog++) {
            const float *src = base + (size_t)Nog * (size_t)NO1;
            Type_Orbs_Grid *dst = dst_rows[Nog];

            for (j = 0; j < NO1; j++) dst[j] = src[j];
          }
        }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }
  }

  acc_free(arena);
  /* return the arena to CUDA instead of the process-local freelist, so
     the other ranks sharing this device can actually use the memory */
  if (cudaDeviceSynchronize() == cudaSuccess) acc_clear_freelists();
  free(rv_off_h); free(rwf_base_h); free(sp_mesh_h); free(sp_maxl_h); free(sp_nb_h);
  free(pao_rv); free(pao_rwf); free(atv_flat); free(chi_host);
  free(ppr_host); free(pair_hAN); free(pair_host);

  if (!active_notice_done) {
    active_notice_done = 1;
    if (SOG_env_flag("OPENMX_ORBS_GRID_GPU_VERBOSE", 0) && myid == Host_ID) {
      fprintf(stderr, "Set_Orbitals_Grid: GPU path active\n");
      fflush(stderr);
    }
  }
  return 1;

host_fallback:
  free(rv_off_h); free(rwf_base_h); free(sp_mesh_h); free(sp_maxl_h); free(sp_nb_h);
  free(pao_rv); free(pao_rwf); free(atv_flat); free(chi_host);
  free(ppr_host); free(pair_hAN); free(pair_host);
  return 0;
}




double Set_Orbitals_Grid(int Cnt_kind)
{
  int i,j,n,Mc_AN,Gc_AN,Cwan,NO0,GNc,GRc;
  int Gh_AN,Mh_AN,Rnh,Hwan,NO1,Nog,h_AN;
  long int k,Nc;
  double time0;
  double x,y,z,dx,dy,dz;
  double TStime,TEtime;
  double Cxyz[4];
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom,Etime_atom;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* Orbital values and their grid topology back the density GPU service
     cache.  Drop the previous epoch before any values are regenerated. */
  Set_Density_Grid_GPU_Invalidate();
  Set_Hamiltonian_Invalidate_OpenACC_MatrixElements_Cache();

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  dtime(&TStime);

  /* device fast path; on any ineligibility or allocation failure the
     untouched host loops below produce the same values */
  if (SOG_gpu_eligible(Cnt_kind) && Set_Orbitals_Grid_GPU()) {
    dtime(&TEtime);
    return TEtime - TStime;
  }

  /*****************************************************
                Calculate orbitals on grids
  *****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];

    if (Cnt_kind==0)  NO0 = Spe_Total_NO[Cwan];
    else              NO0 = Spe_Total_CNO[Cwan]; 

#pragma omp parallel shared(Comp2Real,Spe_PAO_RWF,Spe_Num_Basis,Spe_MaxL_Basis,Spe_PAO_RV,Spe_Num_Mesh_PAO,List_YOUSO,Orbs_Grid,Cnt_kind,Gxyz,atv,CellListAtom,GridListAtom,GridN_Atom,Gc_AN,Cwan,Mc_AN,NO0) private(OMPID,Nthrds,Nprocs,Nc,GNc,GRc,Cxyz,x,y,z,dx,dy,dz,i,j)
    {
      double *Chi0;
      double Cxyz0[4]; 
      double **RF;
      double **AF;
      int i,L0,Mul0,M0,i1;
      double S_coordinate[3];
      double dum,dum1,dum2,dum3,dum4,a,b,c,d;
      double siQ,coQ,siP,coP,Q,P,R;
      double rm,df,sum0,sum1;
      double SH[Supported_MaxL*2+1][2];
      double dSHt[Supported_MaxL*2+1][2];
      double dSHp[Supported_MaxL*2+1][2];

      /* Radial */
      int mp_min,mp_max,m,po,wan;
      double h1,h2,h3,f1,f2,f3,f4;
      double g1,g2,x1,x2,y1,y2,y12,y22,f;
      double r,r1,theta,phi,Min_r;

      /* allocation of array */

      Chi0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

      RF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
      for (i=0; i<(List_YOUSO[25]+1); i++){
	RF[i] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
      }

      AF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
      for (i=0; i<(List_YOUSO[25]+1); i++){
	AF[i] = (double*)malloc(sizeof(double)*(2*(List_YOUSO[25]+1)+1));
      }

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Nc=OMPID*GridN_Atom[Gc_AN]/Nthrds; Nc<(OMPID+1)*GridN_Atom[Gc_AN]/Nthrds; Nc++){

	GNc = GridListAtom[Mc_AN][Nc]; 
	GRc = CellListAtom[Mc_AN][Nc];

	Get_Grid_XYZ(GNc,Cxyz);
	x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1]; 
	y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2]; 
	z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

	if (Cnt_kind==0){

          /* Get_Orbitals(Cwan,x,y,z,Chi0); */
          /* start of inlining of Get_Orbitals */

          wan = Cwan;

          /* xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate); */
          /* start of inlining of xyz2spherical */

	  Min_r = 10e-15;
	  dum = x*x + y*y; 
	  r = sqrt(dum + z*z);
	  r1 = sqrt(dum);

	  if (Min_r<=r){

	    if (r<fabs(z))
	      dum1 = sgn(z)*1.0;
	    else
	      dum1 = z/r;

	    theta = acos(dum1);

	    if (Min_r<=r1){
	      if (0.0<=x){

		if (r1<fabs(y))
		  dum1 = sgn(y)*1.0;
		else
		  dum1 = y/r1;        
  
		phi = asin(dum1);
	      }
	      else{

		if (r1<fabs(y))
		  dum1 = sgn(y)*1.0;
		else
		  dum1 = y/r1;        

		phi = PI - asin(dum1);
	      }
	    }
	    else{
	      phi = 0.0;
	    }
	  }
	  else{
	    theta = 0.5*PI;
	    phi = 0.0;
	  }

	  R = r;
	  Q = theta;
	  P = phi;

	  /* end of inlining of xyz2spherical */

	  po = 0;
	  mp_min = 0;
	  mp_max = Spe_Num_Mesh_PAO[wan] - 1;

	  if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan]-1]<R){

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		RF[L0][Mul0] = 0.0;
	      }
	    }

	    po = 1;
	  }

	  else if (R<Spe_PAO_RV[wan][0]){

	    m = 4;
	    rm = Spe_PAO_RV[wan][m];

	    h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
	    h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
	    h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

	    x1 = rm - Spe_PAO_RV[wan][m-1];
	    x2 = rm - Spe_PAO_RV[wan][m];
	    y1 = x1/h2;
	    y2 = x2/h2;
	    y12 = y1*y1;
	    y22 = y2*y2;

	    dum = h1 + h2;
	    dum1 = h1/h2/dum;
	    dum2 = h2/h1/dum;
	    dum = h2 + h3;
	    dum3 = h2/h3/dum;
	    dum4 = h3/h2/dum;

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		if (m==1){
		  h1 = -(h2+h3);
		  f1 = f4;
		}
		else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		  h3 = -(h1+h2);
		  f4 = f1;
		}

		dum = f3 - f2;
		g1 = dum*dum1 + (f2-f1)*dum2;
		g2 = (f4-f3)*dum3 + dum*dum4;

		f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y22*(2.0*f2 + h2*g1)/h2
		  + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
		  - y12*(2.0*f3 - h2*g2)/h2;

		if (L0==0){
		  a = 0.0;
		  b = 0.5*df/rm;
		  c = 0.0;
		  d = f - b*rm*rm;
		}

		else if (L0==1){
		  a = (rm*df - f)/(2.0*rm*rm*rm);
		  b = 0.0;
		  c = df - 3.0*a*rm*rm;
		  d = 0.0;
		}

		else{
		  b = (3.0*f - rm*df)/(rm*rm);
		  a = (f - b*rm*rm)/(rm*rm*rm);
		  c = 0.0;
		  d = 0.0;
		}

		RF[L0][Mul0] = a*R*R*R + b*R*R + c*R + d;

	      }
	    }

	  }

	  else{

	    do{
	      m = (mp_min + mp_max)/2;
	      if (Spe_PAO_RV[wan][m]<R)
		mp_min = m;
	      else 
		mp_max = m;
	    }
	    while((mp_max-mp_min)!=1);
	    m = mp_max;

	    h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
	    h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
	    h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

	    x1 = R - Spe_PAO_RV[wan][m-1];
	    x2 = R - Spe_PAO_RV[wan][m];
	    y1 = x1/h2;
	    y2 = x2/h2;
	    y12 = y1*y1;
	    y22 = y2*y2;

	    dum = h1 + h2;
	    dum1 = h1/h2/dum;
	    dum2 = h2/h1/dum;
	    dum = h2 + h3;
	    dum3 = h2/h3/dum;
	    dum4 = h3/h2/dum;

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		if (m==1){
		  h1 = -(h2+h3);
		  f1 = f4;
		}
		else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		  h3 = -(h1+h2);
		  f4 = f1;
		}

		dum = f3 - f2;
		g1 = dum*dum1 + (f2-f1)*dum2;
		g2 = (f4-f3)*dum3 + dum*dum4;

		f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		RF[L0][Mul0] = f;

	      }
	    } 

	  }

	  if (po==0){

	    /* Angular */
	    siQ = sin(Q);
	    coQ = cos(Q);
	    siP = sin(P);
	    coP = cos(P);

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){

	      if (L0==0){
		AF[0][0] = 0.282094791773878;
	      }
	      else if (L0==1){
		dum = 0.48860251190292*siQ;
		AF[1][0] = dum*coP;
		AF[1][1] = dum*siP;
		AF[1][2] = 0.48860251190292*coQ;
	      }
	      else if (L0==2){
		dum1 = siQ*siQ;
		dum2 = 1.09254843059208*siQ*coQ;
		AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
		AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
		AF[2][2] = 1.09254843059208*dum1*siP*coP;
		AF[2][3] = dum2*coP;
		AF[2][4] = dum2*siP;
	      }

	      else if (L0==3){
		AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
		AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
		AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
		AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
		AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
		AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
		AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
	      }

	      else if (4<=L0){

		/* calculation of complex spherical harmonics functions */
		for(m=-L0; m<=L0; m++){ 
		  ComplexSH(L0,m,Q,P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
		}

		/* transformation of complex to real */
		for (i=0; i<(L0*2+1); i++){

		  sum0 = 0.0;
		  sum1 = 0.0; 
		  for (j=0; j<(L0*2+1); j++){
		    sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1]; 
		    sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0]; 
		  }
		  AF[L0][i] = sum0 + sum1; 
		}              

	      }
	    }
	  }

	  /* Chi0 */  
	  i1 = -1;
	  for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
	      for (M0=0; M0<=2*L0; M0++){
		i1++;
		Chi0[i1] = RF[L0][Mul0]*AF[L0][M0];
	      }
	    }
	  }
	  /* end of inlining of Get_Orbitals */

	}
	else{
          Get_Cnt_Orbitals(Mc_AN,x,y,z,Chi0);
	}

	for (i=0; i<NO0; i++){
	  Orbs_Grid[Mc_AN][Nc][i] = (Type_Orbs_Grid)Chi0[i];/* AITUNE */
	}

      } /* Nc */

      /* freeing of array */

      free(Chi0);

      for (i=0; i<(List_YOUSO[25]+1); i++){
	free(RF[i]);
      }
      free(RF);

      for (i=0; i<(List_YOUSO[25]+1); i++){
	free(AF[i]);
      }
      free(AF);

    } /* #pragma omp parallel */

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  }

  /****************************************************
     Calculate Orbs_Grid_FNAN
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];    

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];

      if (G2ID[Gh_AN]!=myid){

        Mh_AN = F_G2M[Gh_AN];
        Rnh = ncn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];

        if (Cnt_kind==0)  NO1 = Spe_Total_NO[Hwan];
        else              NO1 = Spe_Total_CNO[Hwan];

#pragma omp parallel shared(List_YOUSO,Orbs_Grid_FNAN,NO1,Mh_AN,Hwan,Cnt_kind,Rnh,Gh_AN,Gxyz,atv,NumOLG,Mc_AN,h_AN,GListTAtoms1,GridListAtom,CellListAtom) private(OMPID,Nthrds,Nprocs,Nog,Nc,GNc,GRc,x,y,z,j)
        {

          double *Chi0;
	  double Cxyz0[4]; 
          double **RF;
          double **AF;
	  int i,L0,Mul0,M0,i1;
	  double S_coordinate[3];
	  double dum,dum1,dum2,dum3,dum4,a,b,c,d;
	  double siQ,coQ,siP,coP,Q,P,R;
	  double rm,df,sum0,sum1;
	  double SH[Supported_MaxL*2+1][2];
	  double dSHt[Supported_MaxL*2+1][2];
	  double dSHp[Supported_MaxL*2+1][2];

	  /* Radial */
	  int mp_min,mp_max,m,po,wan;
	  double h1,h2,h3,f1,f2,f3,f4;
	  double g1,g2,x1,x2,y1,y2,y12,y22,f;
          double r,r1,theta,phi,Min_r;

          /* allocation of arrays */

	  Chi0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

	  RF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    RF[i] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
	  }

	  AF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    AF[i] = (double*)malloc(sizeof(double)*(2*(List_YOUSO[25]+1)+1));
	  }

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (Nog=OMPID*NumOLG[Mc_AN][h_AN]/Nthrds; Nog<(OMPID+1)*NumOLG[Mc_AN][h_AN]/Nthrds; Nog++){

	    Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	    GNc = GridListAtom[Mc_AN][Nc];
	    GRc = CellListAtom[Mc_AN][Nc]; 

	    Get_Grid_XYZ(GNc,Cxyz0);

	    x = Cxyz0[1] + atv[GRc][1] - Gxyz[Gh_AN][1] - atv[Rnh][1];
	    y = Cxyz0[2] + atv[GRc][2] - Gxyz[Gh_AN][2] - atv[Rnh][2];
	    z = Cxyz0[3] + atv[GRc][3] - Gxyz[Gh_AN][3] - atv[Rnh][3];

	    if (Cnt_kind==0){

              /* Get_Orbitals(Hwan,x,y,z,Chi0); */
              /* start of inlining of Get_Orbitals */

              wan = Hwan; 

	      /* xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate); */
              /* start of inlining of xyz2spherical */

	      Min_r = 10e-15;
	      dum = x*x + y*y; 
	      r = sqrt(dum + z*z);
	      r1 = sqrt(dum);

	      if (Min_r<=r){

		if (r<fabs(z))
		  dum1 = sgn(z)*1.0;
		else
		  dum1 = z/r;

		theta = acos(dum1);

		if (Min_r<=r1){
		  if (0.0<=x){

		    if (r1<fabs(y))
		      dum1 = sgn(y)*1.0;
		    else
		      dum1 = y/r1;        
  
		    phi = asin(dum1);
		  }
		  else{

		    if (r1<fabs(y))
		      dum1 = sgn(y)*1.0;
		    else
		      dum1 = y/r1;        

		    phi = PI - asin(dum1);
		  }
		}
		else{
		  phi = 0.0;
		}
	      }
	      else{
		theta = 0.5*PI;
		phi = 0.0;
	      }

	      R = r;
	      Q = theta;
	      P = phi;

              /* end of inlining of xyz2spherical */

	      po = 0;
	      mp_min = 0;
	      mp_max = Spe_Num_Mesh_PAO[wan] - 1;

	      if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan]-1]<R){

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		    RF[L0][Mul0] = 0.0;
		  }
		}

		po = 1;
	      }

	      else if (R<Spe_PAO_RV[wan][0]){

		m = 4;
		rm = Spe_PAO_RV[wan][m];

		h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
		h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
		h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

		x1 = rm - Spe_PAO_RV[wan][m-1];
		x2 = rm - Spe_PAO_RV[wan][m];
		y1 = x1/h2;
		y2 = x2/h2;
		y12 = y1*y1;
		y22 = y2*y2;

		dum = h1 + h2;
		dum1 = h1/h2/dum;
		dum2 = h2/h1/dum;
		dum = h2 + h3;
		dum3 = h2/h3/dum;
		dum4 = h3/h2/dum;

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		    f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		    f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		    f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		    f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		    if (m==1){
		      h1 = -(h2+h3);
		      f1 = f4;
		    }
		    else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		      h3 = -(h1+h2);
		      f4 = f1;
		    }

		    dum = f3 - f2;
		    g1 = dum*dum1 + (f2-f1)*dum2;
		    g2 = (f4-f3)*dum3 + dum*dum4;

		    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y22*(2.0*f2 + h2*g1)/h2
		      + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
		      - y12*(2.0*f3 - h2*g2)/h2;

		    if (L0==0){
		      a = 0.0;
		      b = 0.5*df/rm;
		      c = 0.0;
		      d = f - b*rm*rm;
		    }

		    else if (L0==1){
		      a = (rm*df - f)/(2.0*rm*rm*rm);
		      b = 0.0;
		      c = df - 3.0*a*rm*rm;
		      d = 0.0;
		    }

		    else{
		      b = (3.0*f - rm*df)/(rm*rm);
		      a = (f - b*rm*rm)/(rm*rm*rm);
		      c = 0.0;
		      d = 0.0;
		    }

		    RF[L0][Mul0] = a*R*R*R + b*R*R + c*R + d;

		  }
		}

	      }

	      else{

		do{
		  m = (mp_min + mp_max)/2;
		  if (Spe_PAO_RV[wan][m]<R)
		    mp_min = m;
		  else 
		    mp_max = m;
		}
		while((mp_max-mp_min)!=1);
		m = mp_max;

		h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
		h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
		h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

		x1 = R - Spe_PAO_RV[wan][m-1];
		x2 = R - Spe_PAO_RV[wan][m];
		y1 = x1/h2;
		y2 = x2/h2;
		y12 = y1*y1;
		y22 = y2*y2;

		dum = h1 + h2;
		dum1 = h1/h2/dum;
		dum2 = h2/h1/dum;
		dum = h2 + h3;
		dum3 = h2/h3/dum;
		dum4 = h3/h2/dum;

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		    f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		    f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		    f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		    f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		    if (m==1){
		      h1 = -(h2+h3);
		      f1 = f4;
		    }
		    else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		      h3 = -(h1+h2);
		      f4 = f1;
		    }

		    dum = f3 - f2;
		    g1 = dum*dum1 + (f2-f1)*dum2;
		    g2 = (f4-f3)*dum3 + dum*dum4;

		    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		    RF[L0][Mul0] = f;

		  }
		} 

	      }

	      if (po==0){

		/* Angular */
		siQ = sin(Q);
		coQ = cos(Q);
		siP = sin(P);
		coP = cos(P);

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){

		  if (L0==0){
		    AF[0][0] = 0.282094791773878;
		  }
		  else if (L0==1){
		    dum = 0.48860251190292*siQ;
		    AF[1][0] = dum*coP;
		    AF[1][1] = dum*siP;
		    AF[1][2] = 0.48860251190292*coQ;
		  }
		  else if (L0==2){
		    dum1 = siQ*siQ;
		    dum2 = 1.09254843059208*siQ*coQ;
		    AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
		    AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
		    AF[2][2] = 1.09254843059208*dum1*siP*coP;
		    AF[2][3] = dum2*coP;
		    AF[2][4] = dum2*siP;
		  }

		  else if (L0==3){
		    AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
		    AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
		    AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
		    AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
		    AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
		    AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
		    AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
		  }

		  else if (4<=L0){

		    /* calculation of complex spherical harmonics functions */
		    for(m=-L0; m<=L0; m++){ 
		      ComplexSH(L0,m,Q,P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
		    }

		    /* transformation of complex to real */
		    for (i=0; i<(L0*2+1); i++){

		      sum0 = 0.0;
		      sum1 = 0.0; 
		      for (j=0; j<(L0*2+1); j++){
			sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1]; 
			sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0]; 
		      }
		      AF[L0][i] = sum0 + sum1; 
		    }              

		  }
		}
	      }

	      /* Chi0 */  
	      i1 = -1;
	      for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		  for (M0=0; M0<=2*L0; M0++){
		    i1++;
		    Chi0[i1] = RF[L0][Mul0]*AF[L0][M0];
		  }
		}
	      }

              /* end of inlining of Get_Orbitals */

	    } /* if (Cnt_kind==0) */

	    else{
              Get_Cnt_Orbitals(Mh_AN,x,y,z,Chi0);
	    }

	    for (j=0; j<NO1; j++){
	      Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j] = (Type_Orbs_Grid)Chi0[j];/* AITUNE */
	    }

	  } /* Nog */

          /* freeing of arrays */

	  free(Chi0);

	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    free(RF[i]);
	  }
	  free(RF);

	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    free(AF[i]);
	  }
	  free(AF);

        } 
      }
    }
  }

  /* time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}
