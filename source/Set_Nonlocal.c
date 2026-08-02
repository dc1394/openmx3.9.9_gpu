/**********************************************************************
  Set_Nonlocal.c:

     Set_Nonlocal.c is a subroutine to calculate matrix elements
     and the derivatives of nonlocal potentials in the momentum space.

  Log of Set_Nonlocal.c:

     15/Sep/2002  Released by T.Ozaki

***********************************************************************/

#define  measure_time   0

/* thread-private scratch length of the device spherical Bessel evaluation;
   matches asize_lmax+10 of the host Spherical_Bessel.c */
#define  SETNL_SB_ARR   40

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <openacc.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>
#include "set_cuda_default_device_from_local_rank.h"

#ifdef kcomp
static double NLRF_BesselF(int Gensi, int L, int so, double R);
static void Nonlocal0(double *****HNL, double ******DS_NL);
static void Multiply_DS_NL(int Mc_AN, int Mj_AN, int k, int kl,
                           int Cwan, int Hwan, int wakg, dcomplex ***NLH);
#else
inline double NLRF_BesselF(int Gensi, int L, int so, double R);
inline void Nonlocal0(double *****HNL, double ******DS_NL);
inline void Multiply_DS_NL(int Mc_AN, int Mj_AN, int k, int kl,
                           int Cwan, int Hwan, int wakg, dcomplex ***NLH);
#endif

static void SetNonlocal_AbortWithMessage(const char *message)
{
  fprintf(stderr,"%s\n",message);
  fflush(stderr);
  MPI_Abort(mpi_comm_level1,1);
}

static size_t SetNonlocal_CheckedAddCount(size_t a, size_t b, const char *label)
{
  if (b > SIZE_MAX - a){
    char msg[512];
    snprintf(msg,sizeof(msg),"Count overflow in Set_Nonlocal.c: %s",label);
    SetNonlocal_AbortWithMessage(msg);
  }

  return a + b;
}

static size_t SetNonlocal_CheckedMulCount(size_t a, size_t b, const char *label)
{
  if (a != 0 && b > SIZE_MAX/a){
    char msg[512];
    snprintf(msg,sizeof(msg),"Dimension overflow in Set_Nonlocal.c: %s",label);
    SetNonlocal_AbortWithMessage(msg);
  }

  return a * b;
}

static void *SetNonlocal_MallocArray(size_t count, size_t elem_size, const char *label)
{
  size_t bytes;
  void *ptr;

  bytes = SetNonlocal_CheckedMulCount(count,elem_size,label);
  ptr = malloc((bytes == 0) ? 1 : bytes);

  if (ptr == NULL){
    char msg[512];
    snprintf(msg,sizeof(msg),
             "Out of memory in Set_Nonlocal.c: %s (%zu bytes)",label,bytes);
    SetNonlocal_AbortWithMessage(msg);
  }

  return ptr;
}

static int SetNonlocal_SizeTToInt(size_t value, const char *label)
{
  if (value > (size_t)INT_MAX){
    char msg[512];
    snprintf(msg,sizeof(msg),
             "Value exceeds INT_MAX in Set_Nonlocal.c: %s (%zu)",label,value);
    SetNonlocal_AbortWithMessage(msg);
  }

  return (int)value;
}

static long int SetNonlocal_SizeTToLongInt(size_t value, const char *label)
{
  if (value > (size_t)LONG_MAX){
    char msg[512];
    snprintf(msg,sizeof(msg),
             "Value exceeds LONG_MAX in Set_Nonlocal.c: %s (%zu)",label,value);
    SetNonlocal_AbortWithMessage(msg);
  }

  return (long int)value;
}

static size_t SetNonlocal_ArenaOff(size_t *pos, size_t bytes)
{
  size_t off = *pos;

  *pos = (off + bytes + 511U) & ~(size_t)511U;
  return off;
}

/* one shot; the caller shrinks the request or falls back to the host */
static void *SetNonlocal_ArenaTry(size_t bytes)
{
  void *arena = acc_malloc(bytes);

  if (arena == NULL) {
    /* our own freelist may hoard mismatched freed blocks */
    if (cudaDeviceSynchronize() == cudaSuccess) acc_clear_freelists();
    arena = acc_malloc(bytes);
  }
  return arena;
}

static void SetNonlocal_ValidateGlobalState(void)
{
  int spe,L;
  size_t component_count,tmp_count;

  if (SpeciesNum <= 0){
    SetNonlocal_AbortWithMessage("Invalid SpeciesNum in Set_Nonlocal.c.");
  }

  if (OneD_Grid <= 0){
    SetNonlocal_AbortWithMessage("OneD_Grid must be positive in Set_Nonlocal.c.");
  }

  if (Ngrid_NormK <= 0){
    SetNonlocal_AbortWithMessage("Ngrid_NormK must be positive in Set_Nonlocal.c.");
  }

  if (NormK == NULL){
    SetNonlocal_AbortWithMessage("NormK is NULL in Set_Nonlocal.c.");
  }

  if (Spe_NLRF_Bessel == NULL){
    SetNonlocal_AbortWithMessage("Spe_NLRF_Bessel is NULL in Set_Nonlocal.c.");
  }

  if (List_YOUSO[19] <= 0 || List_YOUSO[24] <= 0 || List_YOUSO[25] < 0 || List_YOUSO[30] < 0){
    SetNonlocal_AbortWithMessage("Invalid List_YOUSO dimensions in Set_Nonlocal.c.");
  }

  for (L=1; L<Ngrid_NormK; L++){
    if (!(NormK[L] > NormK[L-1])){
      SetNonlocal_AbortWithMessage("NormK must be strictly increasing in Set_Nonlocal.c.");
    }
  }

  for (spe=0; spe<SpeciesNum; spe++){

    if (Spe_MaxL_Basis[spe] < 0 || Spe_MaxL_Basis[spe] > List_YOUSO[25]){
      char msg[512];
      snprintf(msg,sizeof(msg),
               "Spe_MaxL_Basis exceeds List_YOUSO[25] in Set_Nonlocal.c for species %d.",spe);
      SetNonlocal_AbortWithMessage(msg);
    }

    if (Spe_Total_NO[spe] < 0 || List_YOUSO[7] < Spe_Total_NO[spe]){
      char msg[512];
      snprintf(msg,sizeof(msg),
               "Spe_Total_NO exceeds List_YOUSO[7] in Set_Nonlocal.c for species %d.",spe);
      SetNonlocal_AbortWithMessage(msg);
    }

    for (L=0; L<=Spe_MaxL_Basis[spe]; L++){
      if (Spe_Num_Basis[spe][L] < 0 || List_YOUSO[24] < Spe_Num_Basis[spe][L]){
        char msg[512];
        snprintf(msg,sizeof(msg),
                 "Spe_Num_Basis exceeds List_YOUSO[24] in Set_Nonlocal.c for species %d, L=%d.",
                 spe,L);
        SetNonlocal_AbortWithMessage(msg);
      }
    }

    if (VPS_j_dependency[spe] < 0 || 1 < VPS_j_dependency[spe]){
      char msg[512];
      snprintf(msg,sizeof(msg),
               "Unsupported VPS_j_dependency in Set_Nonlocal.c for species %d.",spe);
      SetNonlocal_AbortWithMessage(msg);
    }

    if (Spe_Num_RVPS[spe] < 0 || List_YOUSO[19] <= Spe_Num_RVPS[spe]){
      char msg[512];
      snprintf(msg,sizeof(msg),
               "Spe_Num_RVPS exceeds buffer bounds in Set_Nonlocal.c for species %d.",spe);
      SetNonlocal_AbortWithMessage(msg);
    }

    component_count = 0;
    for (L=1; L<=Spe_Num_RVPS[spe]; L++){
      if (Spe_VPS_List[spe][L] < 0 || List_YOUSO[30] < Spe_VPS_List[spe][L]){
        char msg[512];
        snprintf(msg,sizeof(msg),
                 "Spe_VPS_List exceeds List_YOUSO[30] in Set_Nonlocal.c for species %d, projector %d.",
                 spe,L);
        SetNonlocal_AbortWithMessage(msg);
      }

      if (VPS_j_dependency[spe] == 1 && 3 < Spe_VPS_List[spe][L]){
        char msg[512];
        snprintf(msg,sizeof(msg),
                 "j-dependent nonlocal projectors above f are unsupported in Set_Nonlocal.c for species %d.",
                 spe);
        SetNonlocal_AbortWithMessage(msg);
      }

      tmp_count = SetNonlocal_CheckedMulCount(2u,(size_t)Spe_VPS_List[spe][L],
                                              "projector magnetic components");
      tmp_count = SetNonlocal_CheckedAddCount(tmp_count,1u,"projector magnetic components");
      component_count = SetNonlocal_CheckedAddCount(component_count,tmp_count,
                                                    "Spe_Total_VPS_Pro");
    }

    if (component_count != (size_t)Spe_Total_VPS_Pro[spe]){
      char msg[512];
      snprintf(msg,sizeof(msg),
               "Spe_Total_VPS_Pro mismatch in Set_Nonlocal.c for species %d.",spe);
      SetNonlocal_AbortWithMessage(msg);
    }
  }
}

static int SetNonlocalUseOpenACC(void)
{
  /* opt out with OPENMX_SETNL_OPENACC=0 (falls back to the OpenMP CPU path) */
  const char *value = getenv("OPENMX_SETNL_OPENACC");

  if (value != NULL && atoi(value) == 0) return 0;
  return (scf_eigen_lib_flag == GPUSOLVER && gpu_rank_device_usable());
}

/*
   Two-phase evaluation of DS_NL:

     (1) every radial Fourier integral of every (atom pair, so, LL,
         basis x projector) combination is reduced in one batched OpenACC
         kernel (chunked over pairs to bound device memory), and

     (2) the angular assembly (spherical harmonics, Gaunt coefficients and
         the complex-to-real transforms), which has no useful GPU shape,
         runs OpenMP-parallel over the pairs like the CPU path.

   The previous implementation walked the pairs serially on the host and
   launched one small kernel per (pair, so) with a full re-upload of the
   Bessel tables, which left all but one host thread idle.

   All transient device memory is claimed through one acc_malloc arena so
   that an exhausted device (e.g. many MPI ranks sharing one GPU) never
   aborts the run: acc_malloc reports NULL, the Bessel chunk is shrunk and
   retried, and if even the smallest footprint does not fit the function
   returns 0 and the caller runs the original host path instead.
   Returns 1 when DS_NL was evaluated here.
*/
static int SetNonlocal_CalcDSNL_OpenACC(double ******DS_NL,
                                         double *Normk_grid,
                                         int basis_l_dim,
                                         int basis_m_dim,
                                         int rvps_dim,
                                         int rvps_sum_dim,
                                         int proj_m_dim,
                                         int grid_dim,
                                         int max_Lmax_Four_Int,
                                         int max_sph_rows,
                                         int *OneD2Mc_AN,
                                         int *OneD2h_AN,
                                         int OneD_Nloop,
                                         double grid_h)
{
  const int nsp = SpeciesNum;
  int w,cw,hw,p;
  int run_max_nLL;
  int total_rows;
  size_t rf_sp_stride,nlrf_sp_stride,rf_all_elems,nlrf_all_elems;
  size_t max_combo,lookup_stride;
  size_t spair_count,row_scan,sph_stride,chunk_cap_elems;
  int chunk_pairs;
  double *RF_all,*NLRF_all;
  int *rf_base_sp,*nlrf_base_sp;
  int *combo_count_sp,*nLL_sp,*combo_rf_off_sp,*combo_nlrf_off_sp,*combo_lookup_sp;
  int *pair_Cwan,*pair_Hwan,*pair_spair,*pair_diag,*pair_nLL;
  double *pair_r;
  size_t *pair_row_base;
  int *row_pair,*row_sph_off,*row_rf_base,*row_nlrf_base;
  double *sums0_all,*sums1_all;
  unsigned char *arena;
  size_t arena_bytes;
  size_t o_normk,o_rf,o_nlrf,o_row_pair,o_row_sph,o_row_rfb,o_row_nlb;
  size_t o_pair_diag,o_pair_r,o_pair_nLL,o_sums0,o_sums1,o_sphb,o_sphbp;
  int device_ok;
  static int fail_notice_done = 0;
  int maxL0,maxL1;
  size_t gaunt_d1,gaunt_d2,gaunt_d3,gaunt_d4,gaunt_elems;
  double *gaunt_tab;
  double t_mark0,t_mark1,t_setup,t_sphb,t_acc,t_ang;

  if (OneD_Nloop <= 0) return 1;

  t_setup = t_sphb = t_acc = t_ang = 0.0;
  if (measure_time) dtime(&t_mark0);

  /**********************************************************************
     per-species radial tables: the Fourier transforms of the basis
     functions and of the nonlocal projectors on the common k grid
  **********************************************************************/

  rf_sp_stride = SetNonlocal_CheckedMulCount(
                   SetNonlocal_CheckedMulCount((size_t)basis_l_dim,(size_t)List_YOUSO[24],
                                               "OpenACC RF species stride"),
                   (size_t)grid_dim,"OpenACC RF species stride");
  rf_all_elems = SetNonlocal_CheckedMulCount((size_t)nsp,rf_sp_stride,"OpenACC RF_all elements");
  (void)SetNonlocal_SizeTToInt(rf_all_elems,"OpenACC RF_all elements");
  RF_all = (double*)SetNonlocal_MallocArray(rf_all_elems,sizeof(double),"OpenACC RF_all");
  memset(RF_all,0,rf_all_elems*sizeof(double));

  nlrf_sp_stride = SetNonlocal_CheckedMulCount((size_t)rvps_sum_dim,(size_t)grid_dim,
                                               "OpenACC NLRF species stride");
  nlrf_all_elems = SetNonlocal_CheckedMulCount((size_t)(2*nsp),nlrf_sp_stride,
                                               "OpenACC NLRF_all elements");
  (void)SetNonlocal_SizeTToInt(nlrf_all_elems,"OpenACC NLRF_all elements");
  NLRF_all = (double*)SetNonlocal_MallocArray(nlrf_all_elems,sizeof(double),"OpenACC NLRF_all");
  memset(NLRF_all,0,nlrf_all_elems*sizeof(double));

  rf_base_sp = (int*)SetNonlocal_MallocArray((size_t)nsp,sizeof(int),"OpenACC rf_base_sp");
  nlrf_base_sp = (int*)SetNonlocal_MallocArray((size_t)(2*nsp),sizeof(int),"OpenACC nlrf_base_sp");
  for (w=0; w<nsp; w++){
    rf_base_sp[w] = SetNonlocal_SizeTToInt((size_t)w*rf_sp_stride,"OpenACC rf base");
    nlrf_base_sp[w] = SetNonlocal_SizeTToInt((size_t)w*nlrf_sp_stride,"OpenACC nlrf base");
    nlrf_base_sp[nsp+w] = SetNonlocal_SizeTToInt(((size_t)nsp+w)*nlrf_sp_stride,"OpenACC nlrf base");
  }

#pragma omp parallel for schedule(dynamic)
  for (w=0; w<nsp; w++){
    int L0,Mul0,so,L,i;

    for (L0=0; L0<=Spe_MaxL_Basis[w]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[w][L0]; Mul0++){
        size_t rf_offset = (size_t)w*rf_sp_stride + (size_t)(L0*List_YOUSO[24] + Mul0)*grid_dim;

        for (i=0; i<grid_dim; i++){
          RF_all[rf_offset + i] = RF_BesselF(w,L0,Mul0,Normk_grid[i]);
        }
      }
    }

    for (so=0; so<=VPS_j_dependency[w]; so++){
      for (L=1; L<=Spe_Num_RVPS[w]; L++){
        size_t nlrf_offset = ((size_t)so*nsp + w)*nlrf_sp_stride + (size_t)L*grid_dim;

        for (i=0; i<grid_dim; i++){
          NLRF_all[nlrf_offset + i] = NLRF_BesselF(w,L,so,Normk_grid[i]);
        }
      }
    }
  }

  /**********************************************************************
     per-species-pair (basis, projector) combination tables and the
     angular momentum cutoff of the Fourier integrals
  **********************************************************************/

  spair_count = SetNonlocal_CheckedMulCount((size_t)nsp,(size_t)nsp,"OpenACC species pairs");
  max_combo = SetNonlocal_CheckedMulCount(
                SetNonlocal_CheckedMulCount((size_t)basis_l_dim,(size_t)List_YOUSO[24],
                                            "OpenACC combo bound"),
                (size_t)rvps_dim,"OpenACC combo bound");
  lookup_stride = SetNonlocal_CheckedMulCount(
                    SetNonlocal_CheckedMulCount((size_t)basis_l_dim,(size_t)List_YOUSO[24],
                                                "OpenACC lookup stride"),
                    (size_t)rvps_sum_dim,"OpenACC lookup stride");

  combo_count_sp = (int*)SetNonlocal_MallocArray(spair_count,sizeof(int),"OpenACC combo_count_sp");
  nLL_sp = (int*)SetNonlocal_MallocArray(spair_count,sizeof(int),"OpenACC nLL_sp");
  combo_rf_off_sp = (int*)SetNonlocal_MallocArray(
                       SetNonlocal_CheckedMulCount(spair_count,max_combo,"OpenACC combo offsets"),
                       sizeof(int),"OpenACC combo_rf_off_sp");
  combo_nlrf_off_sp = (int*)SetNonlocal_MallocArray(
                         SetNonlocal_CheckedMulCount(spair_count,max_combo,"OpenACC combo offsets"),
                         sizeof(int),"OpenACC combo_nlrf_off_sp");
  combo_lookup_sp = (int*)SetNonlocal_MallocArray(
                       SetNonlocal_CheckedMulCount(spair_count,lookup_stride,"OpenACC combo lookup"),
                       sizeof(int),"OpenACC combo_lookup_sp");

  for (cw=0; cw<nsp; cw++){
    for (hw=0; hw<nsp; hw++){
      size_t spair = (size_t)cw*nsp + hw;
      int *lookup = combo_lookup_sp + spair*lookup_stride;
      int *rf_off = combo_rf_off_sp + spair*max_combo;
      int *nlrf_off = combo_nlrf_off_sp + spair*max_combo;
      int L0,Mul0,L,Lmax,Lmax_Four_Int;
      int combo = 0;
      size_t li;

      for (li=0; li<lookup_stride; li++){
        lookup[li] = -1;
      }

      for (L0=0; L0<=Spe_MaxL_Basis[cw]; L0++){
        for (Mul0=0; Mul0<Spe_Num_Basis[cw][L0]; Mul0++){
          for (L=1; L<=Spe_Num_RVPS[hw]; L++){
            lookup[(size_t)(L0*List_YOUSO[24] + Mul0)*rvps_sum_dim + L] = combo;
            rf_off[combo] = (L0*List_YOUSO[24] + Mul0)*grid_dim;
            nlrf_off[combo] = L*grid_dim;
            combo++;
          }
        }
      }
      combo_count_sp[spair] = combo;

      Lmax = -10;
      for (L=1; L<=Spe_Num_RVPS[hw]; L++){
        if (Lmax<Spe_VPS_List[hw][L]) Lmax = Spe_VPS_List[hw][L];
      }
      if (Spe_MaxL_Basis[cw]<Lmax)
        Lmax_Four_Int = 2*Lmax;
      else
        Lmax_Four_Int = 2*Spe_MaxL_Basis[cw];

      if (max_Lmax_Four_Int < Lmax_Four_Int){
        SetNonlocal_AbortWithMessage("Lmax_Four_Int exceeds its bound in Set_Nonlocal.c.");
      }
      nLL_sp[spair] = Lmax_Four_Int + 1;
    }
  }

  /**********************************************************************
     enumerate the radial-integral rows: one row per
     (pair, so, LL, basis x projector combination)
  **********************************************************************/

  pair_Cwan = (int*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(int),"OpenACC pair_Cwan");
  pair_Hwan = (int*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(int),"OpenACC pair_Hwan");
  pair_spair = (int*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(int),"OpenACC pair_spair");
  pair_diag = (int*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(int),"OpenACC pair_diag");
  pair_nLL = (int*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(int),"OpenACC pair_nLL");
  pair_r = (double*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(double),"OpenACC pair_r");
  pair_row_base = (size_t*)SetNonlocal_MallocArray((size_t)OneD_Nloop+1u,sizeof(size_t),
                                                   "OpenACC pair_row_base");

  run_max_nLL = 1;
  row_scan = 0;
  for (p=0; p<OneD_Nloop; p++){
    int Mc_AN = OneD2Mc_AN[p];
    int h_AN = OneD2h_AN[p];
    int Gc_AN = M2G[Mc_AN];
    int Cwan = WhatSpecies[Gc_AN];
    int Gh_AN = natn[Gc_AN][h_AN];
    int Rnh = ncn[Gc_AN][h_AN];
    int Hwan = WhatSpecies[Gh_AN];
    size_t spair = (size_t)Cwan*nsp + Hwan;
    size_t socnt = (size_t)VPS_j_dependency[Hwan] + 1u;
    double dx,dy,dz,r;
    double S_coordinate[3];

    dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
    dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
    dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];
    xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_coordinate);
    r = S_coordinate[0];
    if (r<1.0e-10) r = 1.0e-10;

    pair_Cwan[p] = Cwan;
    pair_Hwan[p] = Hwan;
    pair_spair[p] = SetNonlocal_SizeTToInt(spair,"OpenACC species pair index");
    pair_diag[p] = (h_AN==0);
    pair_nLL[p] = (0<combo_count_sp[spair]) ? nLL_sp[spair] : 0;
    pair_r[p] = r;
    pair_row_base[p] = row_scan;
    row_scan = SetNonlocal_CheckedAddCount(
                 row_scan,
                 SetNonlocal_CheckedMulCount(
                   SetNonlocal_CheckedMulCount(socnt,(size_t)nLL_sp[spair],"OpenACC pair rows"),
                   (size_t)combo_count_sp[spair],"OpenACC pair rows"),
                 "OpenACC row total");
    if (0<combo_count_sp[spair] && run_max_nLL<nLL_sp[spair]){
      run_max_nLL = nLL_sp[spair];
    }
  }
  pair_row_base[OneD_Nloop] = row_scan;
  total_rows = SetNonlocal_SizeTToInt(row_scan,"OpenACC total rows");

  /* same capacity limit as asize_lmax in the host Spherical_Bessel.c */
  if (30 < run_max_nLL){
    SetNonlocal_AbortWithMessage("Lmax exceeds the spherical Bessel capacity in Set_Nonlocal.c.");
  }

  /**********************************************************************
     tabulate the Gaunt coefficients once: the angular loops below would
     otherwise evaluate the factorial-sum Wigner-3j formula tens of
     millions of times for a handful of distinct (L0,M0,L1,M1,LL) tuples
     (the spherical-harmonic index m is fixed to M0-M1 by the selection
     rule, so it needs no axis of its own)
  **********************************************************************/

  maxL0 = 0;
  maxL1 = 0;
  for (w=0; w<nsp; w++){
    int L;

    if (maxL0<Spe_MaxL_Basis[w]) maxL0 = Spe_MaxL_Basis[w];
    for (L=1; L<=Spe_Num_RVPS[w]; L++){
      if (maxL1<Spe_VPS_List[w][L]) maxL1 = Spe_VPS_List[w][L];
    }
  }

  gaunt_d1 = (size_t)(2*maxL0 + 1);
  gaunt_d2 = (size_t)(maxL1 + 1);
  gaunt_d3 = (size_t)(2*maxL1 + 1);
  gaunt_d4 = (size_t)run_max_nLL;
  gaunt_elems = SetNonlocal_CheckedMulCount(
                  SetNonlocal_CheckedMulCount(
                    SetNonlocal_CheckedMulCount(
                      SetNonlocal_CheckedMulCount((size_t)(maxL0 + 1),gaunt_d1,"OpenACC Gaunt table"),
                      gaunt_d2,"OpenACC Gaunt table"),
                    gaunt_d3,"OpenACC Gaunt table"),
                  gaunt_d4,"OpenACC Gaunt table");
  gaunt_tab = (double*)SetNonlocal_MallocArray(gaunt_elems,sizeof(double),"OpenACC gaunt_tab");

#pragma omp parallel for schedule(dynamic)
  for (w=0; w<=maxL0; w++){
    int L0 = w;
    int M0,L1,M1,LL;

    for (M0=-L0; M0<=L0; M0++){
      for (L1=0; L1<=maxL1; L1++){
        for (M1=-L1; M1<=L1; M1++){
          for (LL=0; LL<run_max_nLL; LL++){
            size_t gidx = ((((size_t)L0*gaunt_d1 + (size_t)(L0+M0))*gaunt_d2 + (size_t)L1)*gaunt_d3
                           + (size_t)(L1+M1))*gaunt_d4 + (size_t)LL;
            int m = M0 - M1;

            if (abs(m)<=LL && abs(L1-LL)<=L0 && L0<=(L1+LL)){
              gaunt_tab[gidx] = Gaunt(L0,M0,L1,M1,LL,m);
            }
            else{
              gaunt_tab[gidx] = 0.0;
            }
          }
        }
      }
    }
  }

  sums0_all = (double*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(double),"OpenACC sums0");
  sums1_all = (double*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(double),"OpenACC sums1");

  row_pair = (int*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(int),"OpenACC row_pair");
  row_sph_off = (int*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(int),"OpenACC row_sph_off");
  row_rf_base = (int*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(int),"OpenACC row_rf_base");
  row_nlrf_base = (int*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(int),"OpenACC row_nlrf_base");

#pragma omp parallel for schedule(static)
  for (p=0; p<OneD_Nloop; p++){
    size_t spair = (size_t)pair_spair[p];
    int socnt = VPS_j_dependency[pair_Hwan[p]] + 1;
    int nLL = nLL_sp[spair];
    int ccount = combo_count_sp[spair];
    const int *rf_off = combo_rf_off_sp + spair*max_combo;
    const int *nlrf_off = combo_nlrf_off_sp + spair*max_combo;
    size_t row = pair_row_base[p];
    int so,LL,c;

    for (so=0; so<socnt; so++){
      for (LL=0; LL<nLL; LL++){
        for (c=0; c<ccount; c++){
          row_pair[row] = p;
          row_sph_off[row] = LL*grid_dim;
          row_rf_base[row] = rf_base_sp[pair_Cwan[p]] + rf_off[c];
          row_nlrf_base[row] = nlrf_base_sp[so*nsp + pair_Hwan[p]] + nlrf_off[c];
          row++;
        }
      }
    }
  }

  /**********************************************************************
     batched radial integrals on the device, chunked over the pairs so
     the spherical Bessel tables stay within a fixed memory budget
  **********************************************************************/

  sph_stride = SetNonlocal_CheckedMulCount((size_t)run_max_nLL,(size_t)grid_dim,
                                           "OpenACC SphB stride");

  /* claim the whole device footprint in one arena; when it does not fit,
     shrink the Bessel chunk (more launches, same math) before giving up */
  device_ok = 0;
  arena = NULL;
  arena_bytes = 0;
  chunk_pairs = 1;

  if (0 < total_rows){

    size_t budget_pairs = ((size_t)96*1024*1024)/(sph_stride*sizeof(double));

    if (budget_pairs < 1u) budget_pairs = 1u;
    if ((size_t)OneD_Nloop < budget_pairs) budget_pairs = (size_t)OneD_Nloop;

    for (;;){
      size_t pos = 0;

      chunk_pairs = SetNonlocal_SizeTToInt(budget_pairs,"OpenACC chunk pairs");
      chunk_cap_elems = SetNonlocal_CheckedMulCount((size_t)chunk_pairs,sph_stride,
                                                    "OpenACC SphB chunk");

      o_normk = SetNonlocal_ArenaOff(&pos,sizeof(double)*(size_t)grid_dim);
      o_rf = SetNonlocal_ArenaOff(&pos,sizeof(double)*rf_all_elems);
      o_nlrf = SetNonlocal_ArenaOff(&pos,sizeof(double)*nlrf_all_elems);
      o_row_pair = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)total_rows);
      o_row_sph = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)total_rows);
      o_row_rfb = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)total_rows);
      o_row_nlb = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)total_rows);
      o_pair_diag = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)OneD_Nloop);
      o_pair_r = SetNonlocal_ArenaOff(&pos,sizeof(double)*(size_t)OneD_Nloop);
      o_pair_nLL = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)OneD_Nloop);
      o_sums0 = SetNonlocal_ArenaOff(&pos,sizeof(double)*(size_t)total_rows);
      o_sums1 = SetNonlocal_ArenaOff(&pos,sizeof(double)*(size_t)total_rows);
      o_sphb = SetNonlocal_ArenaOff(&pos,sizeof(double)*chunk_cap_elems);
      o_sphbp = SetNonlocal_ArenaOff(&pos,sizeof(double)*chunk_cap_elems);
      arena_bytes = pos;

      arena = (unsigned char*)SetNonlocal_ArenaTry(arena_bytes);
      if (arena != NULL) break;
      if (chunk_pairs == 1) break;

      budget_pairs = (size_t)chunk_pairs/12u;
      if (budget_pairs < 1u) budget_pairs = 1u;
    }

    if (arena == NULL && !fail_notice_done){
      fail_notice_done = 1;
      fprintf(stderr,
              "Set_Nonlocal: %.1f MiB of device staging unavailable; using the host path.\n",
              (double)arena_bytes/(1024.0*1024.0));
      fflush(stderr);
    }
  }
  else{
    /* no radial integrals to batch; the assembly below just clears DS_NL */
    device_ok = 1;
  }

  if (measure_time){
    dtime(&t_mark1);
    t_setup = t_mark1 - t_mark0;
  }

  if (arena != NULL){

    double *d_normk = (double*)(void*)(arena + o_normk);
    double *d_rf_all = (double*)(void*)(arena + o_rf);
    double *d_nlrf_all = (double*)(void*)(arena + o_nlrf);
    int *d_row_pair = (int*)(void*)(arena + o_row_pair);
    int *d_row_sph_off = (int*)(void*)(arena + o_row_sph);
    int *d_row_rf_base = (int*)(void*)(arena + o_row_rfb);
    int *d_row_nlrf_base = (int*)(void*)(arena + o_row_nlb);
    int *d_pair_diag = (int*)(void*)(arena + o_pair_diag);
    double *d_pair_r = (double*)(void*)(arena + o_pair_r);
    int *d_pair_nLL = (int*)(void*)(arena + o_pair_nLL);
    double *d_sums0 = (double*)(void*)(arena + o_sums0);
    double *d_sums1 = (double*)(void*)(arena + o_sums1);
    double *d_SphB = (double*)(void*)(arena + o_sphb);
    double *d_SphBp = (double*)(void*)(arena + o_sphbp);

    acc_memcpy_to_device(d_normk,Normk_grid,sizeof(double)*(size_t)grid_dim);
    acc_memcpy_to_device(d_rf_all,RF_all,sizeof(double)*rf_all_elems);
    acc_memcpy_to_device(d_nlrf_all,NLRF_all,sizeof(double)*nlrf_all_elems);
    acc_memcpy_to_device(d_row_pair,row_pair,sizeof(int)*(size_t)total_rows);
    acc_memcpy_to_device(d_row_sph_off,row_sph_off,sizeof(int)*(size_t)total_rows);
    acc_memcpy_to_device(d_row_rf_base,row_rf_base,sizeof(int)*(size_t)total_rows);
    acc_memcpy_to_device(d_row_nlrf_base,row_nlrf_base,sizeof(int)*(size_t)total_rows);
    acc_memcpy_to_device(d_pair_diag,pair_diag,sizeof(int)*(size_t)OneD_Nloop);
    acc_memcpy_to_device(d_pair_r,pair_r,sizeof(double)*(size_t)OneD_Nloop);
    acc_memcpy_to_device(d_pair_nLL,pair_nLL,sizeof(int)*(size_t)OneD_Nloop);

    {
      int p0;

      for (p0=0; p0<OneD_Nloop; p0+=chunk_pairs){
        int p1 = p0 + chunk_pairs;
        int row0,row1,pp,i,row;

        if (OneD_Nloop < p1) p1 = OneD_Nloop;
        row0 = SetNonlocal_SizeTToInt(pair_row_base[p0],"OpenACC chunk row start");
        row1 = SetNonlocal_SizeTToInt(pair_row_base[p1],"OpenACC chunk row end");
        if (row1 <= row0) continue;

        if (measure_time) dtime(&t_mark0);

        /* spherical Bessel tables of this chunk's pairs, evaluated on the
           device with the same downward recurrence, rescalings and sin/cos
           normalization as the host Spherical_Bessel.c */
#pragma acc parallel loop gang vector collapse(2) \
            deviceptr(d_pair_nLL,d_normk,d_pair_r,d_SphB,d_SphBp)
        for (pp=p0; pp<p1; pp++){
          for (i=0; i<grid_dim; i++){
            int lmax = d_pair_nLL[pp] - 1;

            if (0 <= lmax){
              double x = d_normk[i]*d_pair_r[pp];
              size_t sph_base = (size_t)(pp - p0)*sph_stride + (size_t)i;
              double tsb[SETNL_SB_ARR];
              double sbv[SETNL_SB_ARR];
              double dsbv[SETNL_SB_ARR];
              int n,m,nmax;
              double invx,vsb0,vsb1,vsb2,vsbi;
              double j0,j1,sf,tmp,si,co,ix;

              nmax = lmax + (int)(1.5*x) + 10;
              if (nmax<30) nmax = 30;

              if (0.0 < x){

                invx = 1.0/x;
                vsb0 = 0.0;
                vsb1 = 1.0e-14;

                for ( n=nmax-1; (lmax+2)<n; n-- ){
                  vsb2 = (2.0*(double)n + 1.0)*invx*vsb1 - vsb0;
                  if (0.0<(vsb2-1.0e+250)){
                    tmp = 1.0/vsb2;
                    vsb1 *= tmp;
                    vsb2 = 1.0;
                  }
                  vsbi = vsb0;
                  vsb0 = vsb1;
                  vsb1 = vsb2;
                }

                n = lmax + 3;
                tsb[n-1] = vsb1;
                tsb[n  ] = vsb0;
                tsb[n+1] = vsbi;

                tmp = tsb[n-1];
                tsb[n-1] /= tmp;
                tsb[n  ] /= tmp;

                for ( n=lmax+2; 0<n; n-- ){
                  tsb[n-1] = (2.0*(double)n + 1.0)*invx*tsb[n] - tsb[n+1];
                  if (1.0e+250<tsb[n-1]){
                    tmp = tsb[n-1];
                    for (m=n-1; m<=lmax+1; m++){
                      tsb[m] /= tmp;
                    }
                  }
                }

                si = sin(x);
                co = cos(x);
                ix = 1.0/x;
                j0 = si*ix;
                j1 = si*ix*ix - co*ix;

                if (fabs(tsb[1])<fabs(tsb[0])) sf = j0/tsb[0];
                else                           sf = j1/tsb[1];

                for ( n=0; n<=lmax+1; n++ ){
                  sbv[n] = tsb[n]*sf;
                }

                dsbv[0] = co*ix - si*ix*ix;
                for ( n=1; n<=lmax; n++ ){
                  dsbv[n] = ( (double)n*sbv[n-1] - ((double)n+1.0)*sbv[n+1] )/(2.0*(double)n + 1.0);
                }
              }

              else{
                for ( n=0; n<=lmax+1; n++ ){
                  sbv[n] = 0.0;
                }
                sbv[0] = 1.0;

                dsbv[0] = 0.0;
                for ( n=1; n<=lmax; n++ ){
                  dsbv[n] = ( (double)n*sbv[n-1] - ((double)n+1.0)*sbv[n+1] )/(2.0*(double)n + 1.0);
                }
              }

              for ( n=0; n<=lmax; n++ ){
                d_SphB[sph_base + (size_t)n*grid_dim] = sbv[n];
                d_SphBp[sph_base + (size_t)n*grid_dim] = dsbv[n];
              }
            }
          }
        }

        if (measure_time){
          dtime(&t_mark1);
          t_sphb += t_mark1 - t_mark0;
          t_mark0 = t_mark1;
        }

#pragma acc parallel loop gang \
            deviceptr(d_row_pair,d_row_sph_off,d_row_rf_base,d_row_nlrf_base, \
                      d_pair_diag,d_normk,d_rf_all,d_nlrf_all,d_SphB,d_SphBp, \
                      d_sums0,d_sums1)
        for (row=row0; row<row1; row++){
          int pp2 = d_row_pair[row];
          size_t sph_base = (size_t)(pp2 - p0)*sph_stride + (size_t)d_row_sph_off[row];
          size_t rfb = (size_t)d_row_rf_base[row];
          size_t nlb = (size_t)d_row_nlrf_base[row];
          double local_sum0 = 0.0;
          double local_sumr = 0.0;
          int ii;

#pragma acc loop vector reduction(+:local_sum0,local_sumr)
          for (ii=0; ii<grid_dim; ii++){
            double coe0,Normk,tmp0,tmp1,tmp2,Bessel_Pro1;

            if (ii==0 || ii==(grid_dim-1)) coe0 = 0.50;
            else                           coe0 = 1.00;

            Normk = d_normk[ii];
            tmp0 = coe0*grid_h*Normk*Normk*d_rf_all[rfb + ii];
            tmp1 = tmp0*d_SphB[sph_base + ii];
            tmp2 = tmp0*Normk*d_SphBp[sph_base + ii];
            Bessel_Pro1 = d_nlrf_all[nlb + ii];
            local_sum0 += tmp1*Bessel_Pro1;
            local_sumr += tmp2*Bessel_Pro1;
          }

          d_sums0[row] = local_sum0;
          d_sums1[row] = d_pair_diag[pp2] ? 0.0 : local_sumr;
        }

        if (measure_time){
          dtime(&t_mark1);
          t_acc += t_mark1 - t_mark0;
        }
      }
    }

    acc_memcpy_from_device(sums0_all,d_sums0,sizeof(double)*(size_t)total_rows);
    acc_memcpy_from_device(sums1_all,d_sums1,sizeof(double)*(size_t)total_rows);
    acc_free(arena);
    /* return the arena to CUDA instead of the process-local freelist, so
       the other ranks sharing this device can actually use the memory */
    if (cudaDeviceSynchronize() == cudaSuccess) acc_clear_freelists();
    device_ok = 1;
  }

  if (measure_time) dtime(&t_mark0);

  /* nothing staged: leave DS_NL to the caller's host path */
  if (!device_ok) goto cleanup;

  /**********************************************************************
     angular assembly, OpenMP-parallel over the pairs (same structure as
     the CPU path, reading the batched radial integrals)
  **********************************************************************/

#pragma omp parallel
  {
    int OMPID,Nthrds,Nloop;
    int i,j,k,l,m;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,Rnh,so;
    int LL,L0,L1,L,Mul0,M0,M1,num0,num1;
    int nLL_p,ccount,combo;
    size_t spair;
    const int *lookup;
    double dx,dy,dz,r;
    double S_coordinate[3];
    double siT,coT,siP,coP;
    double gant,theta,phi;
    double SH[2],dSHt[2],dSHp[2];
    double Stime_atom,Etime_atom;
    double tmp0;
    dcomplex CsumNL0,CsumNLr,CsumNLt,CsumNLp;
    dcomplex Ctmp0,Ctmp1,Ctmp2,Cpow;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
    dcomplex *****TmpNL;
    dcomplex *****TmpNLr;
    dcomplex *****TmpNLt;
    dcomplex *****TmpNLp;
    dcomplex **CmatNL0;
    dcomplex **CmatNLr;
    dcomplex **CmatNLt;
    dcomplex **CmatNLp;

    TmpNL = (dcomplex*****)SetNonlocal_MallocArray((size_t)basis_l_dim,sizeof(dcomplex****),
                                                   "OpenACC TmpNL");
    for (i=0; i<basis_l_dim; i++){
      TmpNL[i] = (dcomplex****)SetNonlocal_MallocArray((size_t)List_YOUSO[24],sizeof(dcomplex***),
                                                       "OpenACC TmpNL[i]");
      for (j=0; j<List_YOUSO[24]; j++){
        TmpNL[i][j] = (dcomplex***)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex**),
                                                           "OpenACC TmpNL[i][j]");
        for (k=0; k<basis_m_dim; k++){
          TmpNL[i][j][k] = (dcomplex**)SetNonlocal_MallocArray((size_t)rvps_dim,sizeof(dcomplex*),
                                                               "OpenACC TmpNL[i][j][k]");
          for (l=0; l<rvps_dim; l++){
            TmpNL[i][j][k][l] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                                   "OpenACC TmpNL[i][j][k][l]");
          }
        }
      }
    }

    TmpNLr = (dcomplex*****)SetNonlocal_MallocArray((size_t)basis_l_dim,sizeof(dcomplex****),
                                                    "OpenACC TmpNLr");
    for (i=0; i<basis_l_dim; i++){
      TmpNLr[i] = (dcomplex****)SetNonlocal_MallocArray((size_t)List_YOUSO[24],sizeof(dcomplex***),
                                                        "OpenACC TmpNLr[i]");
      for (j=0; j<List_YOUSO[24]; j++){
        TmpNLr[i][j] = (dcomplex***)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex**),
                                                            "OpenACC TmpNLr[i][j]");
        for (k=0; k<basis_m_dim; k++){
          TmpNLr[i][j][k] = (dcomplex**)SetNonlocal_MallocArray((size_t)rvps_dim,sizeof(dcomplex*),
                                                                "OpenACC TmpNLr[i][j][k]");
          for (l=0; l<rvps_dim; l++){
            TmpNLr[i][j][k][l] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                                    "OpenACC TmpNLr[i][j][k][l]");
          }
        }
      }
    }

    TmpNLt = (dcomplex*****)SetNonlocal_MallocArray((size_t)basis_l_dim,sizeof(dcomplex****),
                                                    "OpenACC TmpNLt");
    for (i=0; i<basis_l_dim; i++){
      TmpNLt[i] = (dcomplex****)SetNonlocal_MallocArray((size_t)List_YOUSO[24],sizeof(dcomplex***),
                                                        "OpenACC TmpNLt[i]");
      for (j=0; j<List_YOUSO[24]; j++){
        TmpNLt[i][j] = (dcomplex***)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex**),
                                                            "OpenACC TmpNLt[i][j]");
        for (k=0; k<basis_m_dim; k++){
          TmpNLt[i][j][k] = (dcomplex**)SetNonlocal_MallocArray((size_t)rvps_dim,sizeof(dcomplex*),
                                                                "OpenACC TmpNLt[i][j][k]");
          for (l=0; l<rvps_dim; l++){
            TmpNLt[i][j][k][l] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                                    "OpenACC TmpNLt[i][j][k][l]");
          }
        }
      }
    }

    TmpNLp = (dcomplex*****)SetNonlocal_MallocArray((size_t)basis_l_dim,sizeof(dcomplex****),
                                                    "OpenACC TmpNLp");
    for (i=0; i<basis_l_dim; i++){
      TmpNLp[i] = (dcomplex****)SetNonlocal_MallocArray((size_t)List_YOUSO[24],sizeof(dcomplex***),
                                                        "OpenACC TmpNLp[i]");
      for (j=0; j<List_YOUSO[24]; j++){
        TmpNLp[i][j] = (dcomplex***)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex**),
                                                            "OpenACC TmpNLp[i][j]");
        for (k=0; k<basis_m_dim; k++){
          TmpNLp[i][j][k] = (dcomplex**)SetNonlocal_MallocArray((size_t)rvps_dim,sizeof(dcomplex*),
                                                                "OpenACC TmpNLp[i][j][k]");
          for (l=0; l<rvps_dim; l++){
            TmpNLp[i][j][k][l] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                                    "OpenACC TmpNLp[i][j][k][l]");
          }
        }
      }
    }

    CmatNL0 = (dcomplex**)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex*),"OpenACC CmatNL0");
    for (i=0; i<basis_m_dim; i++){
      CmatNL0[i] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                      "OpenACC CmatNL0[i]");
    }

    CmatNLr = (dcomplex**)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex*),"OpenACC CmatNLr");
    for (i=0; i<basis_m_dim; i++){
      CmatNLr[i] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                      "OpenACC CmatNLr[i]");
    }

    CmatNLt = (dcomplex**)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex*),"OpenACC CmatNLt");
    for (i=0; i<basis_m_dim; i++){
      CmatNLt[i] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                      "OpenACC CmatNLt[i]");
    }

    CmatNLp = (dcomplex**)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex*),"OpenACC CmatNLp");
    for (i=0; i<basis_m_dim; i++){
      CmatNLp[i] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                      "OpenACC CmatNLp[i]");
    }

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

    for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

      dtime(&Stime_atom);

      Mc_AN = OneD2Mc_AN[Nloop];
      h_AN  = OneD2h_AN[Nloop];

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      Gh_AN = natn[Gc_AN][h_AN];
      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];

      dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
      dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
      dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];

      xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_coordinate);
      r     = S_coordinate[0];
      theta = S_coordinate[1];
      phi   = S_coordinate[2];

      if (r<1.0e-10) r = 1.0e-10;

      siT = sin(theta);
      coT = cos(theta);
      siP = sin(phi);
      coP = cos(phi);

      spair = (size_t)pair_spair[Nloop];
      nLL_p = nLL_sp[spair];
      ccount = combo_count_sp[spair];
      lookup = combo_lookup_sp + spair*lookup_stride;

      for (so=0; so<=VPS_j_dependency[Hwan]; so++){

        const double *sum0_so = sums0_all + pair_row_base[Nloop] + (size_t)so*nLL_p*ccount;
        const double *sum1_so = sums1_all + pair_row_base[Nloop] + (size_t)so*nLL_p*ccount;

        for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
          for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
            for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
              L1 = Spe_VPS_List[Hwan][L];
              for (M0=-L0; M0<=L0; M0++){
                for (M1=-L1; M1<=L1; M1++){
                  TmpNL[L0][Mul0][L0+M0][L][L1+M1]  = Complex(0.0,0.0);
                  TmpNLr[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
                  TmpNLt[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
                  TmpNLp[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
                }
              }
            }
          }
        }

        for (LL=0; LL<nLL_p; LL++){
          for (m=-LL; m<=LL; m++){

            ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);
            SH[1]   = -SH[1];
            dSHt[1] = -dSHt[1];
            dSHp[1] = -dSHp[1];

            CY  = Complex(SH[0],SH[1]);
            CYt = Complex(dSHt[0],dSHt[1]);
            CYp = Complex(dSHp[0],dSHp[1]);

            for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
              for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
                for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){

                  int Ls;

                  L1 = Spe_VPS_List[Hwan][L];
                  Ls = -L0 + L1 + LL;
                  combo = lookup[(size_t)(L0*List_YOUSO[24] + Mul0)*rvps_sum_dim + L];

                  if (combo<0){
                    continue;
                  }

                  if (abs(L1-LL)<=L0 && L0<=(L1+LL)){

                    Cpow = Im_pow(-1,Ls);
                    CY1  = Cmul(Cpow,CY);
                    CYt1 = Cmul(Cpow,CYt);
                    CYp1 = Cmul(Cpow,CYp);

                    for (M0=-L0; M0<=L0; M0++){

                      M1 = M0 - m;

                      if (abs(M1)<=L1){

                        gant = gaunt_tab[((((size_t)L0*gaunt_d1 + (size_t)(L0+M0))*gaunt_d2
                                           + (size_t)L1)*gaunt_d3 + (size_t)(L1+M1))*gaunt_d4
                                         + (size_t)LL];

                        tmp0 = gant*sum0_so[(size_t)LL*ccount + combo];
                        Ctmp2 = CRmul(CY1,tmp0);
                        TmpNL[L0][Mul0][L0+M0][L][L1+M1] =
                          Cadd(TmpNL[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);

                        tmp0 = gant*sum1_so[(size_t)LL*ccount + combo];
                        Ctmp2 = CRmul(CY1,tmp0);
                        TmpNLr[L0][Mul0][L0+M0][L][L1+M1] =
                          Cadd(TmpNLr[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);

                        tmp0 = gant*sum0_so[(size_t)LL*ccount + combo];
                        Ctmp2 = CRmul(CYt1,tmp0);
                        TmpNLt[L0][Mul0][L0+M0][L][L1+M1] =
                          Cadd(TmpNLt[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);

                        tmp0 = gant*sum0_so[(size_t)LL*ccount + combo];
                        Ctmp2 = CRmul(CYp1,tmp0);
                        TmpNLp[L0][Mul0][L0+M0][L][L1+M1] =
                          Cadd(TmpNLp[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);
                      }
                    }
                  }
                }
              }
            }
          }
        }

        num0 = 0;
        for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
          for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

            num1 = 0;
            for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
              L1 = Spe_VPS_List[Hwan][L];

              for (M0=-L0; M0<=L0; M0++){
                for (M1=-L1; M1<=L1; M1++){

                  CsumNL0 = Complex(0.0,0.0);
                  CsumNLr = Complex(0.0,0.0);
                  CsumNLt = Complex(0.0,0.0);
                  CsumNLp = Complex(0.0,0.0);

                  for (k=-L0; k<=L0; k++){

                    Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

                    Ctmp0 = TmpNL[L0][Mul0][L0+k][L][L1+M1];
                    Ctmp2 = Cmul(Ctmp1,Ctmp0);
                    CsumNL0 = Cadd(CsumNL0,Ctmp2);

                    Ctmp0 = TmpNLr[L0][Mul0][L0+k][L][L1+M1];
                    Ctmp2 = Cmul(Ctmp1,Ctmp0);
                    CsumNLr = Cadd(CsumNLr,Ctmp2);

                    Ctmp0 = TmpNLt[L0][Mul0][L0+k][L][L1+M1];
                    Ctmp2 = Cmul(Ctmp1,Ctmp0);
                    CsumNLt = Cadd(CsumNLt,Ctmp2);

                    Ctmp0 = TmpNLp[L0][Mul0][L0+k][L][L1+M1];
                    Ctmp2 = Cmul(Ctmp1,Ctmp0);
                    CsumNLp = Cadd(CsumNLp,Ctmp2);
                  }

                  CmatNL0[L0+M0][L1+M1] = CsumNL0;
                  CmatNLr[L0+M0][L1+M1] = CsumNLr;
                  CmatNLt[L0+M0][L1+M1] = CsumNLt;
                  CmatNLp[L0+M0][L1+M1] = CsumNLp;
                }
              }

              for (M0=-L0; M0<=L0; M0++){
                for (M1=-L1; M1<=L1; M1++){

                  CsumNL0 = Complex(0.0,0.0);
                  CsumNLr = Complex(0.0,0.0);
                  CsumNLt = Complex(0.0,0.0);
                  CsumNLp = Complex(0.0,0.0);

                  for (k=-L1; k<=L1; k++){

                    Ctmp1 = Cmul(CmatNL0[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
                    CsumNL0 = Cadd(CsumNL0,Ctmp1);

                    Ctmp1 = Cmul(CmatNLr[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
                    CsumNLr = Cadd(CsumNLr,Ctmp1);

                    Ctmp1 = Cmul(CmatNLt[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
                    CsumNLt = Cadd(CsumNLt,Ctmp1);

                    Ctmp1 = Cmul(CmatNLp[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
                    CsumNLp = Cadd(CsumNLp,Ctmp1);
                  }

                  DS_NL[so][0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*CsumNL0.r;

                  if (h_AN!=0){

                    if (fabs(siT)<10e-14){

                      DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
                        -8.0*(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r);

                      DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
                        -8.0*(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r);

                      DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
                        -8.0*(coT*CsumNLr.r - siT/r*CsumNLt.r);
                    }

                    else{

                      DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
                        -8.0*(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r
                             - siP/siT/r*CsumNLp.r);

                      DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
                        -8.0*(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r
                             + coP/siT/r*CsumNLp.r);

                      DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
                        -8.0*(coT*CsumNLr.r - siT/r*CsumNLt.r);
                    }
                  }

                  else{
                    DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
                    DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
                    DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
                  }
                }
              }

              num1 = num1 + 2*L1 + 1;
            }

            num0 = num0 + 2*L0 + 1;
          }
        }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    for (i=0; i<basis_m_dim; i++){
      free(CmatNLp[i]);
    }
    free(CmatNLp);

    for (i=0; i<basis_m_dim; i++){
      free(CmatNLt[i]);
    }
    free(CmatNLt);

    for (i=0; i<basis_m_dim; i++){
      free(CmatNLr[i]);
    }
    free(CmatNLr);

    for (i=0; i<basis_m_dim; i++){
      free(CmatNL0[i]);
    }
    free(CmatNL0);

    for (i=0; i<basis_l_dim; i++){
      for (j=0; j<List_YOUSO[24]; j++){
        for (k=0; k<basis_m_dim; k++){
          for (l=0; l<rvps_dim; l++){
            free(TmpNLp[i][j][k][l]);
          }
          free(TmpNLp[i][j][k]);
        }
        free(TmpNLp[i][j]);
      }
      free(TmpNLp[i]);
    }
    free(TmpNLp);

    for (i=0; i<basis_l_dim; i++){
      for (j=0; j<List_YOUSO[24]; j++){
        for (k=0; k<basis_m_dim; k++){
          for (l=0; l<rvps_dim; l++){
            free(TmpNLt[i][j][k][l]);
          }
          free(TmpNLt[i][j][k]);
        }
        free(TmpNLt[i][j]);
      }
      free(TmpNLt[i]);
    }
    free(TmpNLt);

    for (i=0; i<basis_l_dim; i++){
      for (j=0; j<List_YOUSO[24]; j++){
        for (k=0; k<basis_m_dim; k++){
          for (l=0; l<rvps_dim; l++){
            free(TmpNLr[i][j][k][l]);
          }
          free(TmpNLr[i][j][k]);
        }
        free(TmpNLr[i][j]);
      }
      free(TmpNLr[i]);
    }
    free(TmpNLr);

    for (i=0; i<basis_l_dim; i++){
      for (j=0; j<List_YOUSO[24]; j++){
        for (k=0; k<basis_m_dim; k++){
          for (l=0; l<rvps_dim; l++){
            free(TmpNL[i][j][k][l]);
          }
          free(TmpNL[i][j][k]);
        }
        free(TmpNL[i][j]);
      }
      free(TmpNL[i]);
    }
    free(TmpNL);
  } /* #pragma omp parallel */

  if (measure_time){
    int prof_myid;

    dtime(&t_mark1);
    t_ang = t_mark1 - t_mark0;
    MPI_Comm_rank(mpi_comm_level1,&prof_myid);
    printf("CalcDSNL: myid=%2d setup=%8.4f sphb=%8.4f acc=%8.4f ang=%8.4f\n",
           prof_myid,t_setup,t_sphb,t_acc,t_ang);
    fflush(stdout);
  }

 cleanup:
  free(gaunt_tab);
  free(sums1_all);
  free(sums0_all);
  free(row_nlrf_base);
  free(row_rf_base);
  free(row_sph_off);
  free(row_pair);
  free(pair_row_base);
  free(pair_r);
  free(pair_nLL);
  free(pair_diag);
  free(pair_spair);
  free(pair_Hwan);
  free(pair_Cwan);
  free(combo_lookup_sp);
  free(combo_nlrf_off_sp);
  free(combo_rf_off_sp);
  free(nLL_sp);
  free(combo_count_sp);
  free(nlrf_base_sp);
  free(rf_base_sp);
  free(NLRF_all);
  free(RF_all);

  return device_ok;
}


double Set_Nonlocal(double *****HNL, double ******DS_NL)
{
  double TStime,TEtime;

  dtime(&TStime);

  Nonlocal0(HNL, DS_NL);

  /* for time */
  dtime(&TEtime);
  return TEtime - TStime;
}



void Nonlocal0(double *****HNL, double ******DS_NL)
{
  /****************************************************
   Evaluate matrix elements of nonlocal potentials
   in the KB or Blochl separable form in momentum space.
  ****************************************************/
  static int firsttime=1;
  int i,j,kl,n,m;
  int Mc_AN,Gc_AN,h_AN,k,Cwan,Gh_AN,Hwan,so;
  int tno0,tno1,tno2,i1,j1,p,ct_AN,spin;
  int fan,jg,kg,wakg,jg0,Mj_AN0,j0;
  int basis_l_dim,basis_m_dim,rvps_dim,rvps_sum_dim,proj_m_dim;
  int grid_dim,max_Lmax_Four_Int,max_sph_rows;
  int Mj_AN,num,size1,size2;
  int DSNL_on_device;
  int *Snd_DS_NL_Size,*Rcv_DS_NL_Size;
  int Original_Mc_AN,po;
  double rcutA,rcutB,rcut,dmp;
  double grid_h;
  double time1,time2,time3;
  double stime,etime;
  dcomplex ***NLH;
  double *Normk_grid;
  double *tmp_array;
  double *tmp_array2;
  double TStime,TEtime;
  double Stime_atom,Etime_atom;
  int numprocs,myid,tag=999,ID,IDS,IDR;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OneD_Nloop,*OneD2Mc_AN,*OneD2h_AN;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  SetNonlocal_ValidateGlobalState();

  basis_l_dim = SetNonlocal_SizeTToInt(SetNonlocal_CheckedAddCount((size_t)List_YOUSO[25],1u,
                                                                   "basis_l_dim"),
                                       "basis_l_dim");
  basis_m_dim = SetNonlocal_SizeTToInt(
                  SetNonlocal_CheckedAddCount(
                    SetNonlocal_CheckedMulCount(2u,(size_t)basis_l_dim,"basis_m_dim"),
                    1u,"basis_m_dim"),
                  "basis_m_dim");
  rvps_dim = SetNonlocal_SizeTToInt((size_t)List_YOUSO[19],"rvps_dim");
  rvps_sum_dim = SetNonlocal_SizeTToInt(SetNonlocal_CheckedAddCount((size_t)rvps_dim,1u,
                                                                    "rvps_sum_dim"),
                                        "rvps_sum_dim");
  proj_m_dim = SetNonlocal_SizeTToInt(
                 SetNonlocal_CheckedAddCount(
                   SetNonlocal_CheckedMulCount(2u,(size_t)List_YOUSO[30],"proj_m_dim"),
                   1u,"proj_m_dim"),
                 "proj_m_dim");
  grid_dim = SetNonlocal_SizeTToInt(SetNonlocal_CheckedAddCount((size_t)OneD_Grid,1u,"grid_dim"),
                                    "grid_dim");
  max_Lmax_Four_Int = 2*((List_YOUSO[25] < List_YOUSO[30]) ? List_YOUSO[30] : List_YOUSO[25]);
  max_sph_rows = SetNonlocal_SizeTToInt(SetNonlocal_CheckedAddCount((size_t)max_Lmax_Four_Int,3u,
                                                                    "max_sph_rows"),
                                        "max_sph_rows");
  grid_h = (PAO_Nkmax - Radial_kmin)/(double)OneD_Grid;
  Normk_grid = (double*)SetNonlocal_MallocArray((size_t)grid_dim,sizeof(double),"Normk_grid");
  for (i=0; i<grid_dim; i++){
    Normk_grid[i] = Radial_kmin + (double)i*grid_h;
  }

  /****************************************************
             calculate the size of arrays:
  ****************************************************/

  /*
  Spe_Num_RVPS[Hwan]     ->  YOUSO35*YOUSO34
  Spe_VPS_List[Hwan][L]  ->  0,0,0,.., 1,1,1,.., 2,2,2,..,
  */

  {
    size_t size_SumNL0,size_TmpNL;

    size_SumNL0 = SetNonlocal_CheckedMulCount((size_t)basis_l_dim,(size_t)List_YOUSO[24],
                                              "SumNL0 elements");
    size_SumNL0 = SetNonlocal_CheckedMulCount(size_SumNL0,(size_t)rvps_sum_dim,
                                              "SumNL0 elements");

    size_TmpNL = SetNonlocal_CheckedMulCount((size_t)basis_l_dim,(size_t)List_YOUSO[24],
                                             "TmpNL elements");
    size_TmpNL = SetNonlocal_CheckedMulCount(size_TmpNL,(size_t)basis_m_dim,
                                             "TmpNL elements");
    size_TmpNL = SetNonlocal_CheckedMulCount(size_TmpNL,(size_t)rvps_dim,
                                             "TmpNL elements");
    size_TmpNL = SetNonlocal_CheckedMulCount(size_TmpNL,(size_t)proj_m_dim,
                                             "TmpNL elements");

    /* PrintMemory */
    if (firsttime) {
      PrintMemory("Set_Nonlocal: SumNL0",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_SumNL0,sizeof(double),
                                                "PrintMemory SumNL0 bytes"),
                    "PrintMemory SumNL0 bytes"),NULL);
      PrintMemory("Set_Nonlocal: SumNLr0",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_SumNL0,sizeof(double),
                                                "PrintMemory SumNLr0 bytes"),
                    "PrintMemory SumNLr0 bytes"),NULL);
      PrintMemory("Set_Nonlocal: TmpNL",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_TmpNL,sizeof(dcomplex),
                                                "PrintMemory TmpNL bytes"),
                    "PrintMemory TmpNL bytes"),NULL);
      PrintMemory("Set_Nonlocal: TmpNLr",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_TmpNL,sizeof(dcomplex),
                                                "PrintMemory TmpNLr bytes"),
                    "PrintMemory TmpNLr bytes"),NULL);
      PrintMemory("Set_Nonlocal: TmpNLt",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_TmpNL,sizeof(dcomplex),
                                                "PrintMemory TmpNLt bytes"),
                    "PrintMemory TmpNLt bytes"),NULL);
      PrintMemory("Set_Nonlocal: TmpNLp",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_TmpNL,sizeof(dcomplex),
                                                "PrintMemory TmpNLp bytes"),
                    "PrintMemory TmpNLp bytes"),NULL);
      firsttime=0;
    }
  }

  /* one-dimensionalize the Mc_AN and h_AN loops */

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD_Nloop++;
    }
  }

  OneD2Mc_AN = (int*)SetNonlocal_MallocArray(
                        SetNonlocal_CheckedAddCount((size_t)OneD_Nloop,1u,
                                                    "OneD2Mc_AN length"),
                        sizeof(int),"OneD2Mc_AN");
  OneD2h_AN = (int*)SetNonlocal_MallocArray(
                       SetNonlocal_CheckedAddCount((size_t)OneD_Nloop,1u,
                                                   "OneD2h_AN length"),
                       sizeof(int),"OneD2h_AN");

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD2Mc_AN[OneD_Nloop] = Mc_AN;
      OneD2h_AN[OneD_Nloop] = h_AN;
      OneD_Nloop++;
    }
  }

  /****************************************************
                    calculate DS_NL
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time) dtime(&stime);

  DSNL_on_device = 0;
  if (SetNonlocalUseOpenACC()){
    /* 0 means the device could not stage the batch; recompute on the host */
    DSNL_on_device = SetNonlocal_CalcDSNL_OpenACC(DS_NL,Normk_grid,
                                 basis_l_dim,basis_m_dim,rvps_dim,rvps_sum_dim,proj_m_dim,
                                 grid_dim,max_Lmax_Four_Int,max_sph_rows,
                                 OneD2Mc_AN,OneD2h_AN,OneD_Nloop,grid_h);
  }

  if (!DSNL_on_device){
#pragma omp parallel shared(List_YOUSO,time_per_atom,DS_NL,Comp2Real,OneD_Grid,Spe_Num_RVPS,Spe_VPS_List,Spe_Num_Basis,Spe_MaxL_Basis,PAO_Nkmax,VPS_j_dependency,atv,Gxyz,WhatSpecies,ncn,natn,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,Ngrid_NormK,NormK,Spe_NLRF_Bessel)
    {

      int OMPID,Nthrds,Nprocs,Nloop;
      int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN;
      int Rnh,Hwan,so,i,j,k,l,m;
      int LL,L0,L1,L,Mul0,M0,M1,num0,num1;
      int Lmax,Lmax_Four_Int,Ls;

      double dx,dy,dz,r;
      double S_coordinate[3];
      double siT,coT,siP,coP;
      double gant,theta,phi;
      double SH[2],dSHt[2],dSHp[2];
      double Stime_atom, Etime_atom;
      double Normk,kmin,kmax;
      double Bessel_Pro0,Bessel_Pro1;
      double h,coe0,sj,sjp;
      double tmp0,tmp1,tmp2,tmp3,tmp4;
      double *SphB,*SphBp;
      double *tmp_SphB,*tmp_SphBp;
      double *RF_BesselCache,*NLRF_BesselCache[2];
      double ***SumNL0;
      double ***SumNLr0;
      size_t rf_cache_elems,nlrf_cache_elems,sph_cache_elems;
      int cached_Cwan,cached_Hwan[2];

      dcomplex CsumNL0,CsumNLr,CsumNLt,CsumNLp;
      dcomplex Ctmp0,Ctmp1,Ctmp2,Cpow;
      dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
      dcomplex *****TmpNL;
      dcomplex *****TmpNLr;
      dcomplex *****TmpNLt;
      dcomplex *****TmpNLp;
      dcomplex **CmatNL0;
      dcomplex **CmatNLr;
      dcomplex **CmatNLt;
      dcomplex **CmatNLp;

    /* allocation of arrays */

    TmpNL = (dcomplex*****)SetNonlocal_MallocArray((size_t)basis_l_dim,sizeof(dcomplex****),
                                                   "TmpNL");
    for (i=0; i<basis_l_dim; i++){
      TmpNL[i] = (dcomplex****)SetNonlocal_MallocArray((size_t)List_YOUSO[24],sizeof(dcomplex***),
                                                       "TmpNL[i]");
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNL[i][j] = (dcomplex***)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex**),
	                                                   "TmpNL[i][j]");
	for (k=0; k<basis_m_dim; k++){
	  TmpNL[i][j][k] = (dcomplex**)SetNonlocal_MallocArray((size_t)rvps_dim,sizeof(dcomplex*),
	                                                       "TmpNL[i][j][k]");
	  for (l=0; l<rvps_dim; l++){
	    TmpNL[i][j][k][l] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
	                                                           "TmpNL[i][j][k][l]");
	  }
	}
      }
    }

    TmpNLr = (dcomplex*****)SetNonlocal_MallocArray((size_t)basis_l_dim,sizeof(dcomplex****),
                                                    "TmpNLr");
    for (i=0; i<basis_l_dim; i++){
      TmpNLr[i] = (dcomplex****)SetNonlocal_MallocArray((size_t)List_YOUSO[24],sizeof(dcomplex***),
                                                        "TmpNLr[i]");
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLr[i][j] = (dcomplex***)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex**),
	                                                    "TmpNLr[i][j]");
	for (k=0; k<basis_m_dim; k++){
	  TmpNLr[i][j][k] = (dcomplex**)SetNonlocal_MallocArray((size_t)rvps_dim,sizeof(dcomplex*),
	                                                        "TmpNLr[i][j][k]");
	  for (l=0; l<rvps_dim; l++){
	    TmpNLr[i][j][k][l] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
	                                                            "TmpNLr[i][j][k][l]");
	  }
	}
      }
    }

    TmpNLt = (dcomplex*****)SetNonlocal_MallocArray((size_t)basis_l_dim,sizeof(dcomplex****),
                                                    "TmpNLt");
    for (i=0; i<basis_l_dim; i++){
      TmpNLt[i] = (dcomplex****)SetNonlocal_MallocArray((size_t)List_YOUSO[24],sizeof(dcomplex***),
                                                        "TmpNLt[i]");
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLt[i][j] = (dcomplex***)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex**),
	                                                    "TmpNLt[i][j]");
	for (k=0; k<basis_m_dim; k++){
	  TmpNLt[i][j][k] = (dcomplex**)SetNonlocal_MallocArray((size_t)rvps_dim,sizeof(dcomplex*),
	                                                        "TmpNLt[i][j][k]");
	  for (l=0; l<rvps_dim; l++){
	    TmpNLt[i][j][k][l] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
	                                                            "TmpNLt[i][j][k][l]");
	  }
	}
      }
    }

    TmpNLp = (dcomplex*****)SetNonlocal_MallocArray((size_t)basis_l_dim,sizeof(dcomplex****),
                                                    "TmpNLp");
    for (i=0; i<basis_l_dim; i++){
      TmpNLp[i] = (dcomplex****)SetNonlocal_MallocArray((size_t)List_YOUSO[24],sizeof(dcomplex***),
                                                        "TmpNLp[i]");
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLp[i][j] = (dcomplex***)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex**),
	                                                    "TmpNLp[i][j]");
	for (k=0; k<basis_m_dim; k++){
	  TmpNLp[i][j][k] = (dcomplex**)SetNonlocal_MallocArray((size_t)rvps_dim,sizeof(dcomplex*),
	                                                        "TmpNLp[i][j][k]");
	  for (l=0; l<rvps_dim; l++){
	    TmpNLp[i][j][k][l] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
	                                                            "TmpNLp[i][j][k][l]");
	  }
	}
      }
    }

    SumNL0 = (double***)SetNonlocal_MallocArray((size_t)basis_l_dim,sizeof(double**),"SumNL0");
    for (i=0; i<basis_l_dim; i++){
      SumNL0[i] = (double**)SetNonlocal_MallocArray((size_t)List_YOUSO[24],sizeof(double*),
                                                    "SumNL0[i]");
      for (j=0; j<List_YOUSO[24]; j++){
	SumNL0[i][j] = (double*)SetNonlocal_MallocArray((size_t)rvps_sum_dim,sizeof(double),
	                                                "SumNL0[i][j]");
      }
    }

    SumNLr0 = (double***)SetNonlocal_MallocArray((size_t)basis_l_dim,sizeof(double**),"SumNLr0");
    for (i=0; i<basis_l_dim; i++){
      SumNLr0[i] = (double**)SetNonlocal_MallocArray((size_t)List_YOUSO[24],sizeof(double*),
                                                     "SumNLr0[i]");
      for (j=0; j<List_YOUSO[24]; j++){
	SumNLr0[i][j] = (double*)SetNonlocal_MallocArray((size_t)rvps_sum_dim,sizeof(double),
	                                                 "SumNLr0[i][j]");
      }
    }

    CmatNL0 = (dcomplex**)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex*),"CmatNL0");
    for (i=0; i<basis_m_dim; i++){
      CmatNL0[i] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                      "CmatNL0[i]");
    }

    CmatNLr = (dcomplex**)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex*),"CmatNLr");
    for (i=0; i<basis_m_dim; i++){
      CmatNLr[i] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                      "CmatNLr[i]");
    }

    CmatNLt = (dcomplex**)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex*),"CmatNLt");
    for (i=0; i<basis_m_dim; i++){
      CmatNLt[i] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                      "CmatNLt[i]");
    }

    CmatNLp = (dcomplex**)SetNonlocal_MallocArray((size_t)basis_m_dim,sizeof(dcomplex*),"CmatNLp");
    for (i=0; i<basis_m_dim; i++){
      CmatNLp[i] = (dcomplex*)SetNonlocal_MallocArray((size_t)proj_m_dim,sizeof(dcomplex),
                                                      "CmatNLp[i]");
    }

    rf_cache_elems = SetNonlocal_CheckedMulCount(
                       SetNonlocal_CheckedMulCount((size_t)basis_l_dim,(size_t)List_YOUSO[24],
                                                   "RF_BesselCache elements"),
                       (size_t)grid_dim,"RF_BesselCache elements");
    RF_BesselCache = (double*)SetNonlocal_MallocArray(rf_cache_elems,sizeof(double),
                                                      "RF_BesselCache");

    nlrf_cache_elems = SetNonlocal_CheckedMulCount((size_t)rvps_dim,(size_t)grid_dim,
                                                   "NLRF_BesselCache elements");
    NLRF_BesselCache[0] = (double*)SetNonlocal_MallocArray(nlrf_cache_elems,sizeof(double),
                                                           "NLRF_BesselCache[0]");
    NLRF_BesselCache[1] = (double*)SetNonlocal_MallocArray(nlrf_cache_elems,sizeof(double),
                                                           "NLRF_BesselCache[1]");

    sph_cache_elems = SetNonlocal_CheckedMulCount((size_t)max_sph_rows,(size_t)grid_dim,
                                                  "SphB cache elements");
    SphB = (double*)SetNonlocal_MallocArray(sph_cache_elems,sizeof(double),"SphB");
    SphBp = (double*)SetNonlocal_MallocArray(sph_cache_elems,sizeof(double),"SphBp");
    tmp_SphB = (double*)SetNonlocal_MallocArray((size_t)max_sph_rows,sizeof(double),"tmp_SphB");
    tmp_SphBp = (double*)SetNonlocal_MallocArray((size_t)max_sph_rows,sizeof(double),"tmp_SphBp");

    cached_Cwan = -1;
    cached_Hwan[0] = -1;
    cached_Hwan[1] = -1;

    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    /* one-dimensionalized loop */

    for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

      dtime(&Stime_atom);

      /* get Mc_AN and h_AN */

      Mc_AN = OneD2Mc_AN[Nloop];
      h_AN  = OneD2h_AN[Nloop];

      /* set data on Mc_AN */

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      /* set data on h_AN */

      Gh_AN = natn[Gc_AN][h_AN];
      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];

      dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
      dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
      dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];

      xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_coordinate);
      r     = S_coordinate[0];
      theta = S_coordinate[1];
      phi   = S_coordinate[2];

      /* for empty atoms or finite elemens basis */

      if (r<1.0e-10) r = 1.0e-10;

      /* precalculation of sin and cos */

      siT = sin(theta);
      coT = cos(theta);
      siP = sin(phi);
      coP = cos(phi);

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      h = grid_h;

      if (cached_Cwan != Cwan){
        for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
          for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
            int rf_offset;

            rf_offset = (L0*List_YOUSO[24] + Mul0)*grid_dim;
            for (i=0; i<grid_dim; i++){
              RF_BesselCache[rf_offset + i] = RF_BesselF(Cwan,L0,Mul0,Normk_grid[i]);
            }
          }
        }
        cached_Cwan = Cwan;
      }

      Lmax = -10;
      for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
        if (Lmax<Spe_VPS_List[Hwan][L]) Lmax = Spe_VPS_List[Hwan][L];
      }
      if (Spe_MaxL_Basis[Cwan]<Lmax)
        Lmax_Four_Int = 2*Lmax;
      else
        Lmax_Four_Int = 2*Spe_MaxL_Basis[Cwan];

      for (i=0; i<grid_dim; i++){
        Normk = Normk_grid[i];
        Spherical_Bessel(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
        for(LL=0; LL<=Lmax_Four_Int; LL++){
          SphB[LL*grid_dim + i]  = tmp_SphB[LL];
          SphBp[LL*grid_dim + i] = tmp_SphBp[LL];
        }
      }

      for (so=0; so<=VPS_j_dependency[Hwan]; so++){

	/****************************************************
           Evaluate ovelap integrals <chi0|P> between PAOs
           and progectors of nonlocal potentials.
	****************************************************/
	/****************************************************
                \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3
	****************************************************/

	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	    for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
	      L1 = Spe_VPS_List[Hwan][L];
	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){
		  TmpNL[L0][Mul0][L0+M0][L][L1+M1]  = Complex(0.0,0.0);
		  TmpNLr[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
		  TmpNLt[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
		  TmpNLp[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
		}
	      }
	    }
	  }
	}

        if (cached_Hwan[so] != Hwan){
          for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
            int nlrf_offset;

            nlrf_offset = L*grid_dim;
            for (i=0; i<grid_dim; i++){
              NLRF_BesselCache[so][nlrf_offset + i] = NLRF_BesselF(Hwan,L,so,Normk_grid[i]);
            }
          }
          cached_Hwan[so] = Hwan;
        }

	/* LL loop */

	for(LL=0; LL<=Lmax_Four_Int; LL++){

	  for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	      for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
		SumNL0[L0][Mul0][L] = 0.0;
		SumNLr0[L0][Mul0][L] = 0.0;
	      }
	    }
	  }

	  for (i=0; i<grid_dim; i++){

	    if (i==0 || i==(grid_dim-1)) coe0 = 0.50;
	    else                      coe0 = 1.00;

	    Normk = Normk_grid[i];

	    sj  =  SphB[LL*grid_dim + i];
	    sjp = SphBp[LL*grid_dim + i];

	    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

		Bessel_Pro0 = RF_BesselCache[(L0*List_YOUSO[24] + Mul0)*grid_dim + i];
		tmp0 = coe0*h*Normk*Normk*Bessel_Pro0;
		tmp1 = tmp0*sj;
		tmp2 = tmp0*Normk*sjp;

		for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){

		  Bessel_Pro1 = NLRF_BesselCache[so][L*grid_dim + i];

		  tmp3 = tmp1*Bessel_Pro1;
		  tmp4 = tmp2*Bessel_Pro1;
		  SumNL0[L0][Mul0][L]  += tmp3;
		  SumNLr0[L0][Mul0][L] += tmp4;
		}
	      }
	    }
	  }

	  /* derivatives of "on site" */
	  if (h_AN==0){
	    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
		for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
		  SumNLr0[L0][Mul0][L] = 0.0;
		}
	      }
	    }
	  }

	  /****************************************************
            For "overlap",
            sum_m 8*(-i)^{-L0+L1+l}*
                  C_{L0,-M0,L1,M1,l,m}*
                  \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,
	  ****************************************************/

	  for(m=-LL; m<=LL; m++){

	    ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);
	    SH[1]   = -SH[1];
	    dSHt[1] = -dSHt[1];
	    dSHp[1] = -dSHp[1];

	    CY  = Complex(SH[0],SH[1]);
	    CYt = Complex(dSHt[0],dSHt[1]);
	    CYp = Complex(dSHp[0],dSHp[1]);

	    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
		for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){

		  L1 = Spe_VPS_List[Hwan][L];
		  Ls = -L0 + L1 + LL;

		  if (abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		    Cpow = Im_pow(-1,Ls);
		    CY1  = Cmul(Cpow,CY);
		    CYt1 = Cmul(Cpow,CYt);
		    CYp1 = Cmul(Cpow,CYp);

		    for (M0=-L0; M0<=L0; M0++){

		      M1 = M0 - m;

		      if (abs(M1)<=L1){

			gant = Gaunt(L0,M0,L1,M1,LL,m);

			/* S */

			tmp0 = gant*SumNL0[L0][Mul0][L];
			Ctmp2 = CRmul(CY1,tmp0);
			TmpNL[L0][Mul0][L0+M0][L][L1+M1] =
			  Cadd(TmpNL[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);

			/* dS/dr */

			tmp0 = gant*SumNLr0[L0][Mul0][L];
			Ctmp2 = CRmul(CY1,tmp0);
			TmpNLr[L0][Mul0][L0+M0][L][L1+M1] =
			  Cadd(TmpNLr[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);

			/* dS/dt */

			tmp0 = gant*SumNL0[L0][Mul0][L];
			Ctmp2 = CRmul(CYt1,tmp0);
			TmpNLt[L0][Mul0][L0+M0][L][L1+M1] =
			  Cadd(TmpNLt[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);

			/* dS/dp */

			tmp0 = gant*SumNL0[L0][Mul0][L];
			Ctmp2 = CRmul(CYp1,tmp0);
			TmpNLp[L0][Mul0][L0+M0][L][L1+M1] =
			  Cadd(TmpNLp[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);

		      }
		    }
		  }

		}
	      }
	    }
	  }
	} /* LL */

	/****************************************************
                           complex to real
	****************************************************/

	num0 = 0;
	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

	    num1 = 0;
	    for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
	      L1 = Spe_VPS_List[Hwan][L];

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumNL0 = Complex(0.0,0.0);
		  CsumNLr = Complex(0.0,0.0);
		  CsumNLt = Complex(0.0,0.0);
		  CsumNLp = Complex(0.0,0.0);

		  for (k=-L0; k<=L0; k++){

		    Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

		    /* S */

		    Ctmp0 = TmpNL[L0][Mul0][L0+k][L][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNL0 = Cadd(CsumNL0,Ctmp2);

		    /* dS/dr */

		    Ctmp0 = TmpNLr[L0][Mul0][L0+k][L][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNLr = Cadd(CsumNLr,Ctmp2);

		    /* dS/dt */

		    Ctmp0 = TmpNLt[L0][Mul0][L0+k][L][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNLt = Cadd(CsumNLt,Ctmp2);

		    /* dS/dp */

		    Ctmp0 = TmpNLp[L0][Mul0][L0+k][L][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNLp = Cadd(CsumNLp,Ctmp2);

		  }

		  CmatNL0[L0+M0][L1+M1] = CsumNL0;
		  CmatNLr[L0+M0][L1+M1] = CsumNLr;
		  CmatNLt[L0+M0][L1+M1] = CsumNLt;
		  CmatNLp[L0+M0][L1+M1] = CsumNLp;
		}
	      }

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumNL0 = Complex(0.0,0.0);
		  CsumNLr = Complex(0.0,0.0);
		  CsumNLt = Complex(0.0,0.0);
		  CsumNLp = Complex(0.0,0.0);

		  for (k=-L1; k<=L1; k++){

		    /* S */

		    Ctmp1 = Cmul(CmatNL0[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNL0 = Cadd(CsumNL0,Ctmp1);

		    /* dS/dr */

		    Ctmp1 = Cmul(CmatNLr[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNLr = Cadd(CsumNLr,Ctmp1);

		    /* dS/dt */

		    Ctmp1 = Cmul(CmatNLt[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNLt = Cadd(CsumNLt,Ctmp1);

		    /* dS/dp */

		    Ctmp1 = Cmul(CmatNLp[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNLp = Cadd(CsumNLp,Ctmp1);

		  }

		  DS_NL[so][0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*CsumNL0.r;

		  if (h_AN!=0){

		    if (fabs(siT)<10e-14){

		      DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r);

		      DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r);

		      DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumNLr.r - siT/r*CsumNLt.r);

		    }

		    else{

		      DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r
			     - siP/siT/r*CsumNLp.r);

		      DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r
			     + coP/siT/r*CsumNLp.r);

		      DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumNLr.r - siT/r*CsumNLt.r);

		    }
		  }

		  else{
		    DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  }

		}
	      }

	      num1 = num1 + 2*L1 + 1;
	    }

	    num0 = num0 + 2*L0 + 1;
	  }
	}
      } /* so */


      dtime(&Etime_atom);
#pragma omp atomic
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLp[i]);
    }
    free(CmatNLp);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLt[i]);
    }
    free(CmatNLt);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLr[i]);
    }
    free(CmatNLr);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNL0[i]);
    }
    free(CmatNL0);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	free(SumNLr0[i][j]);
      }
      free(SumNLr0[i]);
    }
    free(SumNLr0);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	free(SumNL0[i][j]);
      }
      free(SumNL0[i]);
    }
    free(SumNL0);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<List_YOUSO[19]; l++){
	    free(TmpNLp[i][j][k][l]);
	  }
	  free(TmpNLp[i][j][k]);
	}
	free(TmpNLp[i][j]);
      }
      free(TmpNLp[i]);
    }
    free(TmpNLp);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<List_YOUSO[19]; l++){
	    free(TmpNLt[i][j][k][l]);
	  }
	  free(TmpNLt[i][j][k]);
	}
	free(TmpNLt[i][j]);
      }
      free(TmpNLt[i]);
    }
    free(TmpNLt);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<List_YOUSO[19]; l++){
	    free(TmpNLr[i][j][k][l]);
	  }
	  free(TmpNLr[i][j][k]);
	}
	free(TmpNLr[i][j]);
      }
      free(TmpNLr[i]);
    }
    free(TmpNLr);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<List_YOUSO[19]; l++){
	    free(TmpNL[i][j][k][l]);
	  }
	  free(TmpNL[i][j][k]);
	}
	free(TmpNL[i][j]);
      }
      free(TmpNL[i]);
    }
    free(TmpNL);

    free(tmp_SphBp);
    free(tmp_SphB);
    free(SphBp);
    free(SphB);
    free(NLRF_BesselCache[1]);
    free(NLRF_BesselCache[0]);
    free(RF_BesselCache);

#pragma omp flush(DS_NL)

    } /* #pragma omp parallel */
  }

  if (measure_time){
    dtime(&etime);
    time1 = etime - stime;
  }

  /*******************************************************
   *******************************************************
     multiplying overlap integrals WITH COMMUNICATION

     MPI: communicate only for k=0
     DS_NL
  *******************************************************
  *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time) dtime(&stime);

  /* allocation of arrays */

  NLH = (dcomplex***)SetNonlocal_MallocArray(3u,sizeof(dcomplex**),"NLH");
  for (k=0; k<3; k++){
    NLH[k] = (dcomplex**)SetNonlocal_MallocArray((size_t)List_YOUSO[7],sizeof(dcomplex*),"NLH[k]");
    for (i=0; i<List_YOUSO[7]; i++){
      NLH[k][i] = (dcomplex*)SetNonlocal_MallocArray((size_t)List_YOUSO[7],sizeof(dcomplex),"NLH[k][i]");
    }
  }

  Snd_DS_NL_Size = (int*)SetNonlocal_MallocArray((size_t)numprocs,sizeof(int),"Snd_DS_NL_Size");
  Rcv_DS_NL_Size = (int*)SetNonlocal_MallocArray((size_t)numprocs,sizeof(int),"Rcv_DS_NL_Size");

  for (ID=0; ID<numprocs; ID++){
    F_Snd_Num_WK[ID] = 0;
    F_Rcv_Num_WK[ID] = 0;
  }

  do {

    /***********************************
          set the size of data
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /* find the data size to send the block data */

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){
        size_t send_count;

        send_count = 0;
        n = F_Snd_Num_WK[IDS];

        Mc_AN = Snd_MAN[IDS][n];
        Gc_AN = Snd_GAN[IDS][n];
        Cwan = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_NO[Cwan];

        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_VPS_Pro[Hwan];
          send_count =
            SetNonlocal_CheckedAddCount(
              send_count,
              SetNonlocal_CheckedMulCount(
                SetNonlocal_CheckedMulCount((size_t)(VPS_j_dependency[Hwan]+1),(size_t)tno1,
                                            "DS_NL send count"),
                (size_t)tno2,"DS_NL send count"),
              "DS_NL send count");
        }

        size1 = SetNonlocal_SizeTToInt(send_count,"Snd_DS_NL_Size[IDS]");
        Snd_DS_NL_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_DS_NL_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        if (size2 < 0){
          SetNonlocal_AbortWithMessage("Received a negative DS_NL block size in Set_Nonlocal.c.");
        }
        Rcv_DS_NL_Size[IDR] = size2;
      }
      else{
        Rcv_DS_NL_Size[IDR] = 0;
      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) )  MPI_Wait(&request,&stat);

    } /* ID */

    /***********************************
               data transfer
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /******************************
         sending of the data
      ******************************/

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

	size1 = Snd_DS_NL_Size[IDS];

	/* allocation of the array */

	tmp_array = (double*)SetNonlocal_MallocArray((size_t)size1,sizeof(double),"tmp_array");

	/* multidimentional array to the vector array */

	num = 0;
	n = F_Snd_Num_WK[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_VPS_Pro[Hwan];

          for (so=0; so<=VPS_j_dependency[Hwan]; so++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
	        tmp_array[num] = DS_NL[so][0][Mc_AN][h_AN][i][j];
	        num++;
	      }
	    }
	  }
	}

        if (num != size1){
          SetNonlocal_AbortWithMessage("Packed DS_NL block size mismatch in Set_Nonlocal.c.");
        }
	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /******************************
        receiving of the block data
      ******************************/

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){

	size2 = Rcv_DS_NL_Size[IDR];
	tmp_array2 = (double*)SetNonlocal_MallocArray((size_t)size2,sizeof(double),"tmp_array2");
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	/* store */

	num = 0;
	n = F_Rcv_Num_WK[IDR];
	Original_Mc_AN = F_TopMAN[IDR] + n;

	Gc_AN = Rcv_GAN[IDR][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_VPS_Pro[Hwan];

          for (so=0; so<=VPS_j_dependency[Hwan]; so++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
	        DS_NL[so][0][Matomnum+1][h_AN][i][j] = tmp_array2[num];
	        num++;
	      }
	    }
	  }
	}

        if (num != size2){
          SetNonlocal_AbortWithMessage("Unpacked DS_NL block size mismatch in Set_Nonlocal.c.");
        }
	/* free tmp_array2 */
	free(tmp_array2);

	/*****************************************************************
                           multiplying overlap integrals
	*****************************************************************/

        for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	  dtime(&Stime_atom);

          Gc_AN = M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          fan = FNAN[Gc_AN];
          rcutA = Spe_Atom_Cut1[Cwan];

          n = F_Rcv_Num_WK[IDR];
          jg = Rcv_GAN[IDR][n];

          for (j0=0; j0<=fan; j0++){

            jg0 = natn[Gc_AN][j0];
            Mj_AN0 = F_G2M[jg0];

            po = 0;
            if (Original_Mc_AN==Mj_AN0){
              po = 1;
              j = j0;
            }

            if (po==1){

	      Hwan = WhatSpecies[jg];
              rcutB = Spe_Atom_Cut1[Hwan];
              rcut = rcutA + rcutB;

	      for (m=0; m<Spe_Total_NO[Cwan]; m++){
		for (n=0; n<Spe_Total_NO[Hwan]; n++){
		  NLH[0][m][n].r = 0.0;     /* <up|VNL|up> */
		  NLH[1][m][n].r = 0.0;     /* <dn|VNL|dn> */
		  NLH[2][m][n].r = 0.0;     /* <up|VNL|dn> */
		  NLH[0][m][n].i = 0.0;
		  NLH[1][m][n].i = 0.0;
		  NLH[2][m][n].i = 0.0;
		}
	      }

              for (k=0; k<=fan; k++){

                kg = natn[Gc_AN][k];
                wakg = WhatSpecies[kg];
                kl = RMI1[Mc_AN][j][k];

                if (0<=kl){

                  Multiply_DS_NL(Mc_AN, Matomnum+1, k, kl, Cwan, Hwan, wakg, NLH);

		} /* if (0<=kl) */

	      } /* k */

              /****************************************************
                      adding NLH to HNL

                 HNL[0] and iHNL[0] for up-up
                 HNL[1] and iHNL[1] for dn-dn
                 HNL[2] and iHNL[2] for up-dn
	      ****************************************************/

              dmp = dampingF(rcut,Dis[Gc_AN][j]);

	      for (p=0; p<List_YOUSO[5]; p++){
		for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
		  for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){

		    HNL[p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].r*F_NL_flag;

		    if (SO_switch==1){
		      iHNL[ p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].i*F_NL_flag;
		      iHNL0[p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].i*F_NL_flag;
		    }
		  }
		}
	      }

	    } /* if (po==1) */
	  } /* j0 */

          dtime(&Etime_atom);
          time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Mc_AN */

        /********************************************
            increment of F_Rcv_Num_WK[IDR]
	********************************************/

        F_Rcv_Num_WK[IDR]++;

      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) {

        MPI_Wait(&request,&stat);
        free(tmp_array);  /* freeing of array */

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
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) po += F_Snd_Num[IDS]-F_Snd_Num_WK[IDS];
      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ) po += F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR];
    }

  } while (po!=0);

  /* freeing of array */

  free(Rcv_DS_NL_Size);
  free(Snd_DS_NL_Size);

  for (k=0; k<3; k++){
    for (i=0; i<List_YOUSO[7]; i++){
      free(NLH[k][i]);
    }
    free(NLH[k]);
  }
  free(NLH);

  if (measure_time){
    dtime(&etime);
    time2 = etime - stime;
  }

  /*******************************************************
   *******************************************************
     multiplying overlap integrals WITHOUT COMMUNICATION
  *******************************************************
  *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time) dtime(&stime);

  /* allocation of arrays */

#pragma omp parallel shared(Matomnum,time_per_atom,HNL,iHNL,iHNL0,F_NL_flag,List_YOUSO,Dis,SpinP_switch,Spe_Total_NO,DS_NL,Spe_VPS_List,Spe_VNLE,Spe_Num_RVPS,VPS_j_dependency,Spe_Total_VPS_Pro,RMI1,F_G2M,natn,Spe_Atom_Cut1,FNAN,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,SO_switch)
  {
    int OMPID,Nthrds,Nprocs,Nloop;
    int Mc_AN,j,Gc_AN,Cwan,fan,jg,i1,j1,i;
    int Mj_AN,Hwan,k,kg,wakg,kl;
    int p,m,n,L,L1,L2,L3;
    double rcutA,rcutB,rcut,sum,ene;
    double Stime_atom, Etime_atom;
    double ene_m,ene_p,dmp;
    double tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
    double PFp,PFm;
    dcomplex ***NLH;
    dcomplex sum0,sum1,sum2;

    /* allocation of arrays */

    NLH = (dcomplex***)SetNonlocal_MallocArray(3u,sizeof(dcomplex**),"OMP NLH");
    for (k=0; k<3; k++){
      NLH[k] = (dcomplex**)SetNonlocal_MallocArray((size_t)List_YOUSO[7],sizeof(dcomplex*),
                                                   "OMP NLH[k]");
      for (i=0; i<List_YOUSO[7]; i++){
	NLH[k][i] = (dcomplex*)SetNonlocal_MallocArray((size_t)List_YOUSO[7],sizeof(dcomplex),
	                                               "OMP NLH[k][i]");
      }
    }

    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    /* one-dimensionalized loop */

    for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

      dtime(&Stime_atom);

      /* get Mc_AN and j */

      Mc_AN = OneD2Mc_AN[Nloop];
      j     = OneD2h_AN[Nloop];

      /* set data on Mc_AN */

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      rcutA = Spe_Atom_Cut1[Cwan];

      /* set data on j */

      jg = natn[Gc_AN][j];
      Mj_AN = F_G2M[jg];

      if (Mj_AN<=Matomnum){

	Hwan = WhatSpecies[jg];
	rcutB = Spe_Atom_Cut1[Hwan];
	rcut = rcutA + rcutB;

	for (m=0; m<Spe_Total_NO[Cwan]; m++){
	  for (n=0; n<Spe_Total_NO[Hwan]; n++){
	    NLH[0][m][n].r = 0.0;     /* <up|VNL|up> */
	    NLH[1][m][n].r = 0.0;     /* <dn|VNL|dn> */
	    NLH[2][m][n].r = 0.0;     /* <up|VNL|dn> */
	    NLH[0][m][n].i = 0.0;
	    NLH[1][m][n].i = 0.0;
	    NLH[2][m][n].i = 0.0;
	  }
	}

	for (k=0; k<=fan; k++){

	  kg = natn[Gc_AN][k];
	  wakg = WhatSpecies[kg];
	  kl = RMI1[Mc_AN][j][k];

	  if (0<=kl){

	    Multiply_DS_NL(Mc_AN, Mj_AN, k, kl, Cwan, Hwan, wakg, NLH);

	  } /* if (0<=kl) */
	} /* k */

	/****************************************************
                       adding NLH to HNL

                 HNL[0] and iHNL[0] for up-up
                 HNL[1] and iHNL[1] for dn-dn
                 HNL[2] and iHNL[2] for up-dn
	****************************************************/

	dmp = dampingF(rcut,Dis[Gc_AN][j]);

	for (p=0; p<List_YOUSO[5]; p++){
	  for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
	    for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){

	      HNL[p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].r*F_NL_flag;

	      if (SO_switch==1){
		iHNL[p][Mc_AN][j][i1][j1]  = dmp*NLH[p][i1][j1].i*F_NL_flag;
		iHNL0[p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].i*F_NL_flag;
	      }
	    }
	  }
	}

      } /* if (Mj_AN<=Matomnum) */

      dtime(&Etime_atom);
#pragma omp atomic
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Nloop */

    /* freeing of array */

    for (k=0; k<3; k++){
      for (i=0; i<List_YOUSO[7]; i++){
	free(NLH[k][i]);
      }
      free(NLH[k]);
    }
    free(NLH);

#pragma omp barrier
#pragma omp flush(HNL,iHNL,iHNL0)

  } /* #pragma omp parallel */

  /****************************************************
  freeing of arrays:
  ****************************************************/

  free(OneD2Mc_AN);
  free(OneD2h_AN);
  free(Normk_grid);

  if (measure_time){
    dtime(&etime);
    time3 = etime - stime;
  }

  if (measure_time){
    printf("Set_Nonlocal: myid=%2d time1=%10.5f time2=%10.5f time3=%10.5f\n",
           myid,time1,time2,time3);
  }

  /****************************************************
   MPI_Barrier
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
}


void Multiply_DS_NL(int Mc_AN, int Mj_AN, int k, int kl,
                    int Cwan, int Hwan, int wakg, dcomplex ***NLH)
{
  int m,n,L,L1,L2,L3,spin,target_spin;
  int GA_AN,GB_AN;
  double ene_core_hole[2][20];
  double sum,sumMul[2],ene,PFp,PFm,ene_p,ene_m;
  double tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
  dcomplex sum0,sum1,sum2;

  /****************************************************
              l-dependent non-local part
  ****************************************************/

  if (VPS_j_dependency[wakg]==0){

    for (m=0; m<Spe_Total_NO[Cwan]; m++){
      for (n=0; n<Spe_Total_NO[Hwan]; n++){

	sum = 0.0;
	L = 0;

	for (L1=1; L1<=Spe_Num_RVPS[wakg]; L1++){

	  ene = Spe_VNLE[0][wakg][L1-1];
	  L2 = 2*Spe_VPS_List[wakg][L1];

	  for (L3=0; L3<=L2; L3++){

	    sum += ene*DS_NL[0][0][Mc_AN][k][m][L]*DS_NL[0][0][Mj_AN][kl][n][L];
	    L++;
	  }

	} /* L1 */

	NLH[0][m][n].r += sum;    /* <up|VNL|up> */
	NLH[1][m][n].r += sum;    /* <dn|VNL|dn> */

      }
    }

  } /* if */

  /****************************************************
               j-dependent non-local part
  ****************************************************/

  else if (VPS_j_dependency[wakg]==1){

    for (m=0; m<Spe_Total_NO[Cwan]; m++){
      for (n=0; n<Spe_Total_NO[Hwan]; n++){

	sum0 = Complex(0.0,0.0);
	sum1 = Complex(0.0,0.0);
	sum2 = Complex(0.0,0.0);

	L = 0;
	for (L1=1; L1<=Spe_Num_RVPS[wakg]; L1++){

	  ene_p = Spe_VNLE[0][wakg][L1-1];
	  ene_m = Spe_VNLE[1][wakg][L1-1];

	  if      (Spe_VPS_List[wakg][L1]==0) { L2=0; PFp=1.0;     PFm=0.0;     }
	  else if (Spe_VPS_List[wakg][L1]==1) { L2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	  else if (Spe_VPS_List[wakg][L1]==2) { L2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	  else if (Spe_VPS_List[wakg][L1]==3) { L2=6; PFp=4.0/7.0; PFm=3.0/7.0; }
          else{
            char msg[512];
            snprintf(msg,sizeof(msg),
                     "Unsupported j-dependent projector angular momentum in Set_Nonlocal.c: species=%d L=%d.",
                     wakg,Spe_VPS_List[wakg][L1]);
            SetNonlocal_AbortWithMessage(msg);
          }

	  /****************************************************
                  off-diagonal contribution on up-dn
                  for spin non-collinear
	  ****************************************************/

	  if (SpinP_switch==3){

	    /***************
                   p
	    ***************/

	    if (L2==2){

	      /* real contribution of l+1/2 to off-diagonal up-down matrix */
	      sum2.r +=
		 ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		-ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L  ];

	      /* imaginary contribution of l+1/2 to off-diagonal up-down matrix */
	      sum2.i +=
		-ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		+ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+1];

	      /* real contribution of l-1/2 for to off-diagonal up-down matrix */
	      sum2.r -=
		 ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		-ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L  ];

	      /* imaginary contribution of l-1/2 to off-diagonal up-down matrix */
	      sum2.i -=
		-ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		+ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+1];
	    }

	    /***************
                   d
	    ***************/

	    else if (L2==4){

	      /* real contribution of l+1/2 to off diagonal up-down matrix */
	      tmp0 = sqrt(3.0);
	      tmp1 = ene_p/5.0;
	      tmp2 = tmp0*tmp1;

	      sum2.r +=
		-tmp2*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp2*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+2];

	      /* imaginary contribution of l+1/2 to off-diagonal up-down matrix */

	      sum2.i +=
		 tmp2*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp2*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+2];

	      /* real contribution of l-1/2 for to off-diagonal up-down matrix */

	      tmp1 = ene_m/5.0;
	      tmp2 = tmp0*tmp1;

	      sum2.r -=
		-tmp2*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp2*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+2];

	      /* imaginary contribution of l-1/2 to off-diagonal up-down matrix */

	      sum2.i -=
		 tmp2*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp2*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+2];

	    }

	    /***************
                   f
	    ***************/

	    else if (L2==6){

	      /* real contribution of l+1/2 to off diagonal up-down matrix */

	      tmp0 = sqrt(6.0);
	      tmp1 = sqrt(3.0/2.0);
	      tmp2 = sqrt(5.0/2.0);

	      tmp3 = ene_p/7.0;
	      tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
	      tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
	      tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

	      sum2.r +=
                -tmp6*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		+tmp6*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+5]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+6]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+6]*DS_NL[0][0][Mj_AN][kl][n][L+4];

	      /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

	      sum2.i +=
                 tmp6*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		-tmp6*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+6]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+6]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+5]*DS_NL[0][0][Mj_AN][kl][n][L+4];

	      /* real contribution of l-1/2 for to diagonal up-down matrix */

	      tmp3 = ene_m/7.0;
	      tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
	      tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
	      tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

	      sum2.r -=
                -tmp6*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		+tmp6*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+5]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+6]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+6]*DS_NL[1][0][Mj_AN][kl][n][L+4];

	      /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

	      sum2.i -=
                 tmp6*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		-tmp6*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+6]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+6]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+5]*DS_NL[1][0][Mj_AN][kl][n][L+4];

	    }

	  }

	  /****************************************************
             off-diagonal contribution on up-up and dn-dn
	  ****************************************************/

	  /* p */

	  if (L2==2){

	    tmp0 =
	       ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+1]
	      -ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L  ];

	    /* contribution of l+1/2 for up spin */
	    sum0.i += -tmp0;

	    /* contribution of l+1/2 for down spin */
	    sum1.i += tmp0;

	    tmp0 =
	       ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+1]
	      -ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L  ];

	    /* contribution of l-1/2 for up spin */
	    sum0.i += tmp0;

	    /* contribution of l-1/2 for down spin */
	    sum1.i += -tmp0;
	  }

	  /* d */

	  else if (L2==4){

	    tmp0 =
	       ene_p*2.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+2]
	      -ene_p*2.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+1]
	      +ene_p*1.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+4]
	      -ene_p*1.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+3];

	    /* contribution of l+1/2 for up spin */
	    sum0.i += -tmp0;

	    /* contribution of l+1/2 for down spin */
	    sum1.i += tmp0;

	    tmp0 =
	       ene_m*2.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+2]
	      -ene_m*2.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+1]
	      +ene_m*1.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+4]
	      -ene_m*1.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+3];

	    /* contribution of l-1/2 for up spin */
	    sum0.i += tmp0;

	    /* contribution of l-1/2 for down spin */
	    sum1.i += -tmp0;

	  }

	  /* f */

	  else if (L2==6){

	    tmp0 =
	       ene_p*1.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+2]
	      -ene_p*1.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+1]
	      +ene_p*2.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+4]
	      -ene_p*2.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+3]
	      +ene_p*3.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+5]*DS_NL[0][0][Mj_AN][kl][n][L+6]
	      -ene_p*3.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+6]*DS_NL[0][0][Mj_AN][kl][n][L+5];

	    /* contribution of l+1/2 for up spin */
	    sum0.i += -tmp0;

	    /* contribution of l+1/2 for down spin */
	    sum1.i += tmp0;

	    tmp0 =
	       ene_m*1.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+2]
	      -ene_m*1.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+1]
	      +ene_m*2.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+4]
	      -ene_m*2.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+3]
	      +ene_m*3.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+5]*DS_NL[1][0][Mj_AN][kl][n][L+6]
	      -ene_m*3.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+6]*DS_NL[1][0][Mj_AN][kl][n][L+5];

	    /* contribution of l-1/2 for up spin */
	    sum0.i += tmp0;

	    /* contribution of l-1/2 for down spin */
	    sum1.i += -tmp0;
	  }

	  /****************************************************
                 diagonal contribution on up-up and dn-dn
	  ****************************************************/

	  for (L3=0; L3<=L2; L3++){

	    /* VNL for j=l+1/2 */
	    sum0.r += PFp*ene_p*DS_NL[0][0][Mc_AN][k][m][L]*DS_NL[0][0][Mj_AN][kl][n][L];
	    sum1.r += PFp*ene_p*DS_NL[0][0][Mc_AN][k][m][L]*DS_NL[0][0][Mj_AN][kl][n][L];

	    /* VNL for j=l-1/2 */
	    sum0.r += PFm*ene_m*DS_NL[1][0][Mc_AN][k][m][L]*DS_NL[1][0][Mj_AN][kl][n][L];
	    sum1.r += PFm*ene_m*DS_NL[1][0][Mc_AN][k][m][L]*DS_NL[1][0][Mj_AN][kl][n][L];

	    L++;
	  }

	}

	NLH[0][m][n].r += sum0.r;    /* <up|VNL|up> */
	NLH[1][m][n].r += sum1.r;    /* <dn|VNL|dn> */
	NLH[2][m][n].r += sum2.r;    /* <up|VNL|dn> */

	NLH[0][m][n].i += sum0.i;    /* <up|VNL|up> */
	NLH[1][m][n].i += sum1.i;    /* <dn|VNL|dn> */
	NLH[2][m][n].i += sum2.i;    /* <up|VNL|dn> */

      } /* n */
    } /* m */

  } /* else if */

  else{
    SetNonlocal_AbortWithMessage("Unsupported VPS_j_dependency in Multiply_DS_NL of Set_Nonlocal.c.");
  }

}





double NLRF_BesselF(int Gensi, int L, int so, double R)
{
  int mp_min,mp_max,m,po;
  double h1,h2,h3,f1,f2,f3,f4;
  double g1,g2,x1,x2,y1,y2,f;
  double result;
  const double *bessel_values;
  const double eps = 1.0e-14;

  if (Ngrid_NormK <= 0){
    SetNonlocal_AbortWithMessage("Ngrid_NormK must be positive in NLRF_BesselF.");
  }

  bessel_values = Spe_NLRF_Bessel[so][Gensi][L];

  if (Ngrid_NormK == 1){
    if (R <= NormK[0]) result = bessel_values[0];
    else               result = 0.0;
    return result;
  }

  if (Ngrid_NormK == 2){
    double x_low,x_high,y_low,y_high;

    x_low = NormK[0];
    x_high = NormK[1];
    y_low = bessel_values[0];
    y_high = bessel_values[1];

    if (R <= x_low) return y_low;
    if (x_high < R) return 0.0;
    if (fabs(x_high - x_low) <= eps) return y_low;

    result = y_low + (y_high - y_low)*(R - x_low)/(x_high - x_low);
    return result;
  }

  mp_min = 0;
  mp_max = Ngrid_NormK - 1;
  po = 0;

  if (R<NormK[0]){
    m = 1;
  }
  else if (NormK[mp_max]<R){
    result = 0.0;
    po = 1;
  }
  else{
    do{
      m = (mp_min + mp_max)/2;
      if (NormK[m]<R)
        mp_min = m;
      else
        mp_max = m;
    }
    while((mp_max-mp_min)!=1);
    m = mp_max;

    if (m<2)
      m = 2;
    else if (Ngrid_NormK<=m)
      m = Ngrid_NormK - 2;
  }

  /****************************************************
                 Spline like interpolation
  ****************************************************/

  if (po==0){

    if (m==1){
      h2 = NormK[m]   - NormK[m-1];
      h3 = NormK[m+1] - NormK[m];

      f2 = bessel_values[m-1];
      f3 = bessel_values[m];
      f4 = bessel_values[m+1];

      h1 = -(h2+h3);
      f1 = f4;
    }
    else if (m==(Ngrid_NormK-1)){
      h1 = NormK[m-1] - NormK[m-2];
      h2 = NormK[m]   - NormK[m-1];

      f1 = bessel_values[m-2];
      f2 = bessel_values[m-1];
      f3 = bessel_values[m];

      h3 = -(h1+h2);
      f4 = f1;
    }
    else{
      h1 = NormK[m-1] - NormK[m-2];
      h2 = NormK[m]   - NormK[m-1];
      h3 = NormK[m+1] - NormK[m];

      f1 = bessel_values[m-2];
      f2 = bessel_values[m-1];
      f3 = bessel_values[m];
      f4 = bessel_values[m+1];
    }

    if (fabs(h1)<=eps || fabs(h2)<=eps || fabs(h3)<=eps ||
        fabs(h1+h2)<=eps || fabs(h2+h3)<=eps){
      x1 = NormK[m-1];
      x2 = NormK[m];

      if (fabs(x2-x1)<=eps){
        result = f2;
      }
      else{
        result = f2 + (f3-f2)*(R - x1)/(x2 - x1);
      }
      return result;
    }

    /****************************************************
                Calculate the value at R
    ****************************************************/

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = R - NormK[m-1];
    x2 = R - NormK[m];
    y1 = x1/h2;
    y2 = x2/h2;

    f =  y2*y2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y1*y1*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    result = f;
  }

  return result;
}
