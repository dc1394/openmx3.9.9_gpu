/**********************************************************************
  Set_ProExpn_VNA.c:

     Set_ProExpn_VNA.c is a subroutine to calculate matrix elements and
     the derivatives of VNA projector expansion in the momentum space.

  Log of Set_ProExpn_VNA.c:

     8/Apr/2004  Released by T.Ozaki

***********************************************************************/

#define  measure_time   0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>
#include <openacc.h>

static double Set_ProExpn(double ****HVNA, Type_DS_VNA *****DS_VNA);
static double Set_VNA2(double ****HVNA, double *****HVNA2);
static double Set_VNA3(double *****HVNA3);
static int SetPro_VNA23_GpuBegin(void);
static void SetPro_VNA23_GpuEnd(void);

#ifdef kcomp
static void Spherical_Bessel2( double x, int lmax, double *sb, double *dsb );
#else
inline void Spherical_Bessel2( double x, int lmax, double *sb, double *dsb );
#endif


double Set_ProExpn_VNA(double ****HVNA, double *****HVNA2, Type_DS_VNA *****DS_VNA)
{
  double time1,time2,time3;
  int vna23_gpu;

  /* separable form */

  time1 = Set_ProExpn(HVNA, DS_VNA);

  /* batched device path for the one-centre pair integrals */

  vna23_gpu = SetPro_VNA23_GpuBegin();

  /* one-center (orbitals) but two-center integrals, <Phi_{LM,L'M'}|VNA> */

  time2 = Set_VNA2(HVNA, HVNA2);

  /* one-center (orbitals) but two-center integrals, <VNA|Phi_{LM,L'M'}> */

  time3 = Set_VNA3(HVNA3);

  if (vna23_gpu) SetPro_VNA23_GpuEnd();

  if (measure_time){
    printf("Time Set_ProExpn=%7.3f Set_VNA2=%7.3f Set_VNA3=%7.3f\n",time1,time2,time3);
  }

  return time1+time2+time3;
}



/* ------------------------------------------------------------------ */
/* Batched device evaluation of the HVNA projector traces.

   HVNA[Mc][j][m][n] = dmp * sum_k sum_L ene_L
                           * sum_l DS_VNA[0][Mc][k][m][l]*DS_VNA[0][Mj][kl][n][l]

   is the same energy-weighted float dot-product contraction as the
   Force4B/Force_HNL first-case traces.  The kernel keeps the exact CPU
   accumulation order (k outermost, then the radial groups, then the
   2*L+1 members with the float product cast to double), so only the
   final float rounding of DS_VNA itself limits the agreement.  The MPI
   pipeline is untouched: received halo blocks are archived and one
   batched kernel runs after the loops.                                */
/* ------------------------------------------------------------------ */

typedef struct {
  int enabled;
  int num_proj;
  int num_rvna;

  /* local direction-0 DS_VNA rows */
  float *flat;
  size_t *mck_off;
  int *mck_base;

  /* halo archive of the received direction-0 blocks */
  float *halo;
  size_t halo_count;
  size_t *halo_off;
  int *halo_base;

  /* per-species group energies and the group lengths */
  double *ene;
  int *l2p1;
} SetProGpuContext;

static SetProGpuContext SetPro_gpu = { 0 };

static void SetPro_gpu_abort(const char *message)
{
  fprintf(stderr,"%s\n",message);
  fflush(stderr);
  MPI_Abort(mpi_comm_level1,1);
}

static void *SetPro_checked_malloc(size_t size)
{
  void *ptr;

  if (size==0) size = 1;
  ptr = malloc(size);
  if (ptr==NULL){
    SetPro_gpu_abort("Set_ProExpn_VNA: malloc failed for the GPU batch.");
  }
  return ptr;
}

static int SetPro_collective_env_flag(const char *name, int default_value)
{
  int myid,value = default_value;
  const char *env;

  MPI_Comm_rank(mpi_comm_level1,&myid);
  if (myid==0){
    env = getenv(name);
    if (env!=NULL && env[0]!='\0') value = (atoi(env)!=0);
  }
  MPI_Bcast(&value,1,MPI_INT,0,mpi_comm_level1);

  return value;
}

static int SetPro_HVNA_GpuBegin(Type_DS_VNA *****DS_VNA, int *VNA_List,
                                int *VNA_List2, int Num_RVNA)
{
  SetProGpuContext *g = &SetPro_gpu;
  int Mc_AN,k,spe,L1,node_ranks = 1,node_rank = 0;
  size_t pos = 0,slots = 0,halo_slots = 0;
  size_t free_bytes = 0,total_bytes = 0;
  MPI_Comm node_comm = MPI_COMM_NULL;

  memset(g,0,sizeof(*g));
  g->num_proj = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
  g->num_rvna = Num_RVNA;

  if (scf_eigen_lib_flag!=GPUSOLVER) return 0;
  if (!SetPro_collective_env_flag("OPENMX_SETPRO_GPU",1)) return 0;

  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
  MPI_Comm_size(node_comm,&node_ranks);
  MPI_Comm_rank(node_comm,&node_rank);
  if (node_ranks<1) node_ranks = 1;
  acc_wait_all();
  if (cudaDeviceSynchronize()==cudaSuccess){
    acc_clear_freelists();
  }
  MPI_Barrier(node_comm);
  MPI_Comm_free(&node_comm);

  /* sizes of the flat local table and the halo archive */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];
    int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

    slots += (size_t)(FNAN[Gc_AN]+1);
    pos += (size_t)(FNAN[Gc_AN]+1)*(size_t)tno*(size_t)g->num_proj;
  }

  for (Mc_AN=Matomnum+1; Mc_AN<=Matomnum+MatomnumF; Mc_AN++){
    int Gc_AN = F_M2G[Mc_AN];
    int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

    halo_slots += (size_t)(FNAN[Gc_AN]+1);
    g->halo_count += (size_t)(FNAN[Gc_AN]+1)*(size_t)tno*(size_t)g->num_proj;
  }

  if (cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess) return 0;
  {
    const size_t reserve = (size_t)256*1024*1024;
    const size_t transients = (size_t)192*1024*1024;
    size_t need = (pos + g->halo_count)*sizeof(float) + transients;

    if (free_bytes<=reserve ||
        (free_bytes - reserve)/(size_t)node_ranks<=need){
      if (node_rank==0){
        fprintf(stderr,
                "Set_ProExpn_VNA GPU: not enough free device memory; CPU fallback.\n");
        fflush(stderr);
      }
      memset(g,0,sizeof(*g));
      return 0;
    }
  }

  g->mck_base = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)(Matomnum+2));
  g->mck_off = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(slots==0 ? 1 : slots));
  g->flat = (float*)SetPro_checked_malloc(sizeof(float)*(pos==0 ? 1 : pos));
  g->halo_base = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)(MatomnumF+2));
  g->halo_off = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(halo_slots==0 ? 1 : halo_slots));
  g->halo = (float*)SetPro_checked_malloc(sizeof(float)*(g->halo_count==0 ? 1 : g->halo_count));
  g->ene = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)SpeciesNum*(size_t)Num_RVNA);
  g->l2p1 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)Num_RVNA);

  for (L1=0; L1<Num_RVNA; L1++){
    g->l2p1[L1] = 2*VNA_List[L1] + 1;
  }
  for (spe=0; spe<SpeciesNum; spe++){
    for (L1=0; L1<Num_RVNA; L1++){
      g->ene[(size_t)spe*Num_RVNA + L1] = VNA_proj_ene[spe][VNA_List[L1]][VNA_List2[L1]];
    }
  }

  {
    size_t flat_len = pos;

    pos = 0;
    slots = 0;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      int Gc_AN = M2G[Mc_AN];
      int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

      g->mck_base[Mc_AN] = (int)slots;
      for (k=0; k<=FNAN[Gc_AN]; k++){
        g->mck_off[slots] = pos;
        pos += (size_t)tno*(size_t)g->num_proj;
        slots++;
      }
    }

    halo_slots = 0;
    pos = 0;
    for (Mc_AN=Matomnum+1; Mc_AN<=Matomnum+MatomnumF; Mc_AN++){
      int Gc_AN = F_M2G[Mc_AN];
      int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

      g->halo_base[Mc_AN-Matomnum] = (int)halo_slots;
      for (k=0; k<=FNAN[Gc_AN]; k++){
        g->halo_off[halo_slots] = pos;
        pos += (size_t)tno*(size_t)g->num_proj;
        halo_slots++;
      }
    }

    /* flatten the local direction-0 rows */

#pragma omp parallel for schedule(dynamic)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      int Gc_AN = M2G[Mc_AN];
      int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];
      int k2,i;

      for (k2=0; k2<=FNAN[Gc_AN]; k2++){
        size_t base = SetPro_gpu.mck_off[SetPro_gpu.mck_base[Mc_AN]+k2];

        for (i=0; i<tno; i++){
          memcpy(SetPro_gpu.flat + base + (size_t)i*(size_t)SetPro_gpu.num_proj,
                 DS_VNA[0][Mc_AN][k2][i],
                 sizeof(float)*(size_t)SetPro_gpu.num_proj);
        }
      }
    }

    {
      float *flat = g->flat;
      size_t len = (flat_len==0 ? 1 : flat_len);
#pragma acc enter data copyin(flat[0:len])
    }
  }

  g->enabled = 1;
  return 1;
}

static void SetPro_HVNA_GpuEnd(void)
{
  SetProGpuContext *g = &SetPro_gpu;
  size_t flat_len = 0;
  int Mc_AN;

  if (!g->enabled) return;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];
    flat_len += (size_t)(FNAN[Gc_AN]+1)
              *(size_t)Spe_Total_NO[WhatSpecies[Gc_AN]]*(size_t)g->num_proj;
  }

  {
    float *flat = g->flat;
    size_t len = (flat_len==0 ? 1 : flat_len);

    if (acc_is_present(flat,len*sizeof(float))){
#pragma acc exit data delete(flat[0:len])
    }
  }
  acc_clear_freelists();

  free(g->l2p1);
  free(g->ene);
  free(g->halo);
  free(g->halo_off);
  free(g->halo_base);
  free(g->flat);
  free(g->mck_off);
  free(g->mck_base);
  memset(g,0,sizeof(*g));
}

/* archive the direction-0 block of the halo atom just received */
static void SetPro_HVNA_GpuArchiveHalo(Type_DS_VNA *****DS_VNA, int Original_Mc_AN,
                                       int Gc_AN)
{
  SetProGpuContext *g = &SetPro_gpu;
  int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];
  int halo_idx = Original_Mc_AN - Matomnum;
  int k;

  if (halo_idx<1 || MatomnumF<halo_idx){
    SetPro_gpu_abort("Set_ProExpn_VNA GPU: halo index out of range.");
  }

#pragma omp parallel for schedule(dynamic)
  for (k=0; k<=FNAN[Gc_AN]; k++){
    size_t base = g->halo_off[g->halo_base[halo_idx]+k];
    int i;

    for (i=0; i<tno; i++){
      memcpy(g->halo + base + (size_t)i*(size_t)g->num_proj,
             DS_VNA[0][Matomnum+1][k][i],
             sizeof(float)*(size_t)g->num_proj);
    }
  }
}

/* the unified batch over every (local atom, neighbour) pair */
static void SetPro_HVNA_GpuRun(double ****HVNA)
{
  SetProGpuContext *g = &SetPro_gpu;
  const int num_proj = g->num_proj;
  const int num_rvna = g->num_rvna;
  int Mc_AN,j,k;
  int nitems = 0,kslots = 0;
  size_t out_count = 0;
  int *item_atom,*item_j,*item_koff,*item_kn,*item_tnoC,*item_tnoH;
  size_t *item_out;
  size_t *krowA,*krowB;
  int *krow_halo,*krow_ene;
  double *out;
  int p;
  size_t opos = 0;
  int kpos = 0;

  /* count */
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];
    int tnoC = Spe_Total_NO[WhatSpecies[Gc_AN]];

    for (j=0; j<=FNAN[Gc_AN]; j++){
      int tnoH = Spe_Total_NO[WhatSpecies[natn[Gc_AN][j]]];

      nitems++;
      out_count += (size_t)tnoC*(size_t)tnoH;
      for (k=0; k<=FNAN[Gc_AN]; k++){
        if (0<=RMI1[Mc_AN][j][k]) kslots++;
      }
    }
  }

  if (nitems==0) return;

  item_atom = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_j = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_koff = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_kn = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_tnoC = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_tnoH = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_out = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(size_t)nitems);
  krowA = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(size_t)(kslots==0 ? 1 : kslots));
  krowB = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(size_t)(kslots==0 ? 1 : kslots));
  krow_halo = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)(kslots==0 ? 1 : kslots));
  krow_ene = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)(kslots==0 ? 1 : kslots));
  out = (double*)SetPro_checked_malloc(sizeof(double)*(out_count==0 ? 1 : out_count));

  /* build */
  p = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];
    int tnoC = Spe_Total_NO[WhatSpecies[Gc_AN]];

    for (j=0; j<=FNAN[Gc_AN]; j++){
      int jg = natn[Gc_AN][j];
      int Mj_AN = F_G2M[jg];
      int tnoH = Spe_Total_NO[WhatSpecies[jg]];

      item_atom[p] = Mc_AN;
      item_j[p] = j;
      item_koff[p] = kpos;
      item_tnoC[p] = tnoC;
      item_tnoH[p] = tnoH;
      item_out[p] = opos;
      opos += (size_t)tnoC*(size_t)tnoH;

      for (k=0; k<=FNAN[Gc_AN]; k++){
        int kl = RMI1[Mc_AN][j][k];

        if (kl<0) continue;
        krowA[kpos] = g->mck_off[g->mck_base[Mc_AN]+k];
        krow_ene[kpos] = WhatSpecies[natn[Gc_AN][k]]*num_rvna;
        if (Mj_AN<=Matomnum){
          krowB[kpos] = g->mck_off[g->mck_base[Mj_AN]+kl];
          krow_halo[kpos] = 0;
        }
        else{
          krowB[kpos] = g->halo_off[g->halo_base[Mj_AN-Matomnum]+kl];
          krow_halo[kpos] = 1;
        }
        kpos++;
      }
      item_kn[p] = kpos - item_koff[p];
      p++;
    }
  }

  if (p!=nitems || kpos!=kslots || opos!=out_count){
    SetPro_gpu_abort("Set_ProExpn_VNA GPU: inconsistent batch.");
  }

  {
    float *flat = g->flat;
    float *halo = g->halo;
    double *ene = g->ene;
    int *l2p1 = g->l2p1;
    size_t flat_len = 0;
    size_t halo_len = (g->halo_count==0 ? 1 : g->halo_count);
    size_t ene_len = (size_t)SpeciesNum*(size_t)num_rvna;
    const int nitems_c = nitems;
    const size_t kslots_c = (size_t)(kslots==0 ? 1 : kslots);
    const size_t out_c = out_count;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      int Gc_AN = M2G[Mc_AN];
      flat_len += (size_t)(FNAN[Gc_AN]+1)
                *(size_t)Spe_Total_NO[WhatSpecies[Gc_AN]]*(size_t)num_proj;
    }
    if (flat_len==0) flat_len = 1;

#pragma acc data copyin(item_koff[0:nitems_c], item_kn[0:nitems_c], \
                        item_tnoC[0:nitems_c], item_tnoH[0:nitems_c], \
                        item_out[0:nitems_c], \
                        krowA[0:kslots_c], krowB[0:kslots_c], \
                        krow_halo[0:kslots_c], krow_ene[0:kslots_c], \
                        halo[0:halo_len], ene[0:ene_len], l2p1[0:num_rvna]) \
                 copyout(out[0:out_c]) \
                 present(flat[0:flat_len])
    {
#pragma acc parallel loop gang vector_length(128) \
    present(item_koff[0:nitems_c], item_kn[0:nitems_c], \
            item_tnoC[0:nitems_c], item_tnoH[0:nitems_c], item_out[0:nitems_c], \
            krowA[0:kslots_c], krowB[0:kslots_c], krow_halo[0:kslots_c], \
            krow_ene[0:kslots_c], halo[0:halo_len], ene[0:ene_len], \
            l2p1[0:num_rvna], flat[0:flat_len], out[0:out_c])
      for (int pp=0; pp<nitems_c; pp++){
        const int tnoC = item_tnoC[pp];
        const int tnoH = item_tnoH[pp];
        const int k_off = item_koff[pp];
        const int k_n = item_kn[pp];
        const size_t out_off = item_out[pp];
        const int mn = tnoC*tnoH;

#pragma acc loop vector
        for (int idx=0; idx<mn; idx++){
          const int m = idx/tnoH;
          const int n = idx - m*tnoH;
          double nlh = 0.0;

          for (int kk=0; kk<k_n; kk++){
            const float *a = flat + krowA[k_off+kk] + (size_t)m*(size_t)num_proj;
            const float *b = (krow_halo[k_off+kk]
                              ? (halo + krowB[k_off+kk])
                              : (flat + krowB[k_off+kk])) + (size_t)n*(size_t)num_proj;
            const double *el = ene + krow_ene[k_off+kk];
            double sum = 0.0;
            int L = 0;

            for (int L1=0; L1<num_rvna; L1++){
              const int g1 = l2p1[L1];
              double tmp0 = 0.0;

              for (int l3=0; l3<g1; l3++){
                tmp0 += (double)(a[L]*b[L]);
                L++;
              }
              sum += el[L1]*tmp0;
            }

            nlh += sum;
          }

          out[out_off + (size_t)idx] = nlh;
        }
      }
    }
  }

  /* damping and store, exactly as the host loops do */

  for (p=0; p<nitems; p++){
    int mc = item_atom[p];
    int j2 = item_j[p];
    int Gc_AN = M2G[mc];
    int jg = natn[Gc_AN][j2];
    int tnoC = item_tnoC[p];
    int tnoH = item_tnoH[p];
    size_t out_off = item_out[p];
    double rcut = Spe_Atom_Cut1[WhatSpecies[Gc_AN]] + Spe_Atom_Cut1[WhatSpecies[jg]];
    double dmp = dampingF(rcut,Dis[Gc_AN][j2]);
    int m,n;

    for (m=0; m<tnoC; m++){
      for (n=0; n<tnoH; n++){
        HVNA[mc][j2][m][n] = dmp*out[out_off + (size_t)m*tnoH + n];
      }
    }
  }

  free(out);
  free(krow_ene);
  free(krow_halo);
  free(krowB);
  free(krowA);
  free(item_out);
  free(item_tnoH);
  free(item_tnoC);
  free(item_kn);
  free(item_koff);
  free(item_j);
  free(item_atom);
}

/* ------------------------------------------------------------------ */
/* Batched device construction of DS_VNA (the <chi0|P> expansion and
   its derivatives).

   Numerically sensitive special functions never run on the device:
   the Gaunt coefficients (factorial-based) are tabulated once on the
   host, the complex spherical harmonics of every pair direction are
   evaluated on the host with the original long-double routine, and the
   radial Bessel tables are the very host arrays.  Only the downward
   Bessel recurrence, the quadrature dot products and the unitary
   transforms run on the device, each keeping the exact CPU loop order.
   Pairs are processed in chunks; the results overwrite DS_VNA rows on
   the host exactly like the original per-pair loop.                   */
/* ------------------------------------------------------------------ */

#define SETPRO_BES_LMAX 30

#pragma acc routine seq
static void SetPro_Spherical_Bessel2_dev(double x, int lmax, double *sb, double *dsb)
{
  /* device copy of Spherical_Bessel2() (xmin==0.0 branch structure) */

  int m,n,nmax;
  double tsb[SETPRO_BES_LMAX+10];
  double vsb0,vsb1,vsb2,vsbi,invx;
  double j0,j1,sf,tmp,si,co,ix,ix2;

  nmax = lmax + 3*x + 20;
  if (nmax<100) nmax = 100;

  if ( 0.0 < x ){

    invx = 1.0/x;

    vsb0 = 0.0;
    vsb1 = 1.0e-14;
    vsbi = 0.0;

    for ( n=nmax-1; (lmax+2)<n; n-- ){

      vsb2 = (2.0*n + 1.0)*invx*vsb1 - vsb0;

      if (1.0e+250<vsb2){
        tmp = 1.0/vsb2;
        vsb2 *= tmp;
        vsb1 *= tmp;
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

      tsb[n-1] = (2.0*n + 1.0)*invx*tsb[n] - tsb[n+1];

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
    ix2 = ix*ix;
    j0 = si*ix;
    j1 = si*ix*ix - co*ix;

    if (fabs(tsb[1])<fabs(tsb[0])) sf = j0/tsb[0];
    else                           sf = j1/tsb[1];

    for ( n=0; n<=lmax+1; n++ ){
      sb[n] = tsb[n]*sf;
    }

    dsb[0] = co*ix - si*ix*ix;
    for ( n=1; n<=lmax; n++ ){
      dsb[n] = ( (double)n*sb[n-1] - (double)(n+1.0)*sb[n+1] )/(2.0*(double)n + 1.0);
    }
  }

  else {

    for ( n=0; n<=lmax; n++ ){
      sb[n] = 0.0;
    }
    sb[0] = 1.0;

    dsb[0] = 0.0;
    for ( n=1; n<=lmax; n++ ){
      dsb[n] = ( (double)n*sb[n-1] - (double)(n+1.0)*sb[n+1] )/(2.0*(double)n + 1.0);
    }
  }
}

/* per-species combo/block tables of the DS_VNA construction batch */
typedef struct {
  int enabled;
  int num_proj;
  int num_rvna;
  int lfi_max;          /* max Lmax_Four_Int over species             */
  int l0max_all;        /* max Spe_MaxL_Basis                          */
  int l1max;            /* List_YOUSO[35]                              */
  int cmb_max;          /* max #(L0,Mul0) combos per species           */
  int blk_max;          /* cmb_max*num_rvna                            */
  int mn_max;           /* max tno*num_proj                            */

  int *sp_lfi;          /* [spe] Lmax_Four_Int                         */
  int *sp_ncmb;         /* [spe] #(L0,Mul0)                            */
  int *cmb_L0;          /* [spe*cmb_max+c] L0                          */
  int *cmb_num0;        /* [spe*cmb_max+c] row offset of the combo     */
  int *vna_L1;          /* [L] VNA_List                                */
  int *vna_num1;        /* [L] column offset                           */

  double *pro00;        /* [spe][cmb][GL_Mesh]                         */
  double *pro01;
  double *vnab;         /* [spe][num_rvna][GL_Mesh]                    */
  double *gnt;          /* Gaunt table                                 */
  double *c2r;          /* Comp2Real, packed per L                     */
  int *c2r_off;         /* [L] offset into c2r (complex index)         */
} SetProGpu2Context;

static SetProGpu2Context SetPro_gpu2 = { 0 };

static int SetPro_DSVNA_GpuBegin(int *VNA_List, int *VNA_List2, int Num_RVNA,
                                 double ****Bessel_Pro00, double ****Bessel_Pro01)
{
  SetProGpu2Context *g = &SetPro_gpu2;
  int spe,L0,Mul0,L,i,k,c;
  size_t free_bytes = 0,total_bytes = 0;
  int node_ranks = 1;
  MPI_Comm node_comm = MPI_COMM_NULL;

  memset(g,0,sizeof(*g));

  if (scf_eigen_lib_flag!=GPUSOLVER) return 0;
  if (!SetPro_collective_env_flag("OPENMX_SETPRO_GPU",1)) return 0;

  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
  MPI_Comm_size(node_comm,&node_ranks);
  if (node_ranks<1) node_ranks = 1;
  MPI_Comm_free(&node_comm);

  if (cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess) return 0;
  {
    const size_t reserve = (size_t)256*1024*1024;
    const size_t need = (size_t)1024*1024*1024; /* chunk transients */

    if (free_bytes<=reserve ||
        (free_bytes - reserve)/(size_t)node_ranks<=need) return 0;
  }

  g->num_proj = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
  g->num_rvna = Num_RVNA;
  g->l1max = List_YOUSO[35];

  g->sp_lfi = (int*)SetPro_checked_malloc(sizeof(int)*SpeciesNum);
  g->sp_ncmb = (int*)SetPro_checked_malloc(sizeof(int)*SpeciesNum);

  g->l0max_all = 0;
  g->cmb_max = 0;
  for (spe=0; spe<SpeciesNum; spe++){
    int Lmax = 0,ncmb = 0;

    for (L=0; L<Num_RVNA; L++){
      if (Lmax<VNA_List[L]) Lmax = VNA_List[L];
    }
    if (Spe_MaxL_Basis[spe]<Lmax) g->sp_lfi[spe] = 2*Lmax;
    else                          g->sp_lfi[spe] = 2*Spe_MaxL_Basis[spe];
    if (g->lfi_max<g->sp_lfi[spe]) g->lfi_max = g->sp_lfi[spe];
    if (g->l0max_all<Spe_MaxL_Basis[spe]) g->l0max_all = Spe_MaxL_Basis[spe];

    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      ncmb += Spe_Num_Basis[spe][L0];
    }
    g->sp_ncmb[spe] = ncmb;
    if (g->cmb_max<ncmb) g->cmb_max = ncmb;
  }
  g->blk_max = g->cmb_max*Num_RVNA;

  g->cmb_L0 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)SpeciesNum*g->cmb_max);
  g->cmb_num0 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)SpeciesNum*g->cmb_max);
  g->vna_L1 = (int*)SetPro_checked_malloc(sizeof(int)*Num_RVNA);
  g->vna_num1 = (int*)SetPro_checked_malloc(sizeof(int)*Num_RVNA);
  g->pro00 = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)SpeciesNum*g->cmb_max*GL_Mesh);
  g->pro01 = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)SpeciesNum*g->cmb_max*GL_Mesh);
  g->vnab = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)SpeciesNum*Num_RVNA*GL_Mesh);

  {
    int mn_max = 0;

    for (spe=0; spe<SpeciesNum; spe++){
      int num0 = 0;

      c = 0;
      for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
        for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
          g->cmb_L0[spe*g->cmb_max+c] = L0;
          g->cmb_num0[spe*g->cmb_max+c] = num0;
          num0 += 2*L0 + 1;
          c++;
        }
      }
      for (; c<g->cmb_max; c++){
        g->cmb_L0[spe*g->cmb_max+c] = -1;
        g->cmb_num0[spe*g->cmb_max+c] = 0;
      }
      if (mn_max<num0*g->num_proj) mn_max = num0*g->num_proj;
    }
    g->mn_max = mn_max;
  }

  {
    int num1 = 0;

    for (L=0; L<Num_RVNA; L++){
      g->vna_L1[L] = VNA_List[L];
      g->vna_num1[L] = num1;
      num1 += 2*VNA_List[L] + 1;
    }
  }

  for (spe=0; spe<SpeciesNum; spe++){
    c = 0;
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
        for (i=0; i<GL_Mesh; i++){
          g->pro00[((size_t)spe*g->cmb_max+c)*GL_Mesh+i] = Bessel_Pro00[spe][L0][Mul0][i];
          g->pro01[((size_t)spe*g->cmb_max+c)*GL_Mesh+i] = Bessel_Pro01[spe][L0][Mul0][i];
        }
        c++;
      }
    }
    for (L=0; L<Num_RVNA; L++){
      for (i=0; i<GL_Mesh; i++){
        g->vnab[((size_t)spe*g->num_rvna+L)*GL_Mesh+i]
          = Spe_VNA_Bessel[spe][VNA_List[L]][VNA_List2[L]][i];
      }
    }
  }

  /* Gaunt table: host factorial arithmetic, once */

  {
    const int nL0 = g->l0max_all + 1;
    const int nM0 = 2*g->l0max_all + 1;
    const int nL1 = g->l1max + 1;
    const int nM1 = 2*g->l1max + 1;
    const int nLL = g->lfi_max + 1;
    const int nm = 2*g->lfi_max + 1;
    size_t gsz = (size_t)nL0*nM0*nL1*nM1*nLL*nm;
    int L1,M0,M1,LL,m;

    g->gnt = (double*)SetPro_checked_malloc(sizeof(double)*gsz);
    for (i=0; i<(int)gsz; i++) g->gnt[i] = 0.0;

    for (L0=0; L0<nL0; L0++){
      for (M0=-L0; M0<=L0; M0++){
        for (L1=0; L1<nL1; L1++){
          for (M1=-L1; M1<=L1; M1++){
            for (LL=0; LL<nLL; LL++){
              for (m=-LL; m<=LL; m++){
                if (M1==M0-m && abs(L1-LL)<=L0 && L0<=(L1+LL)){
                  size_t idx = ((((((size_t)L0*nM0 + (M0+g->l0max_all))*nL1 + L1)*nM1
                                  + (M1+g->l1max))*nLL + LL)*nm + (m+g->lfi_max));
                  g->gnt[idx] = Gaunt(L0,M0,L1,M1,LL,m);
                }
              }
            }
          }
        }
      }
    }
  }

  /* packed Comp2Real up to max(l0max, l1max) */

  {
    const int lc = (g->l0max_all<g->l1max) ? g->l1max : g->l0max_all;
    size_t pos = 0;
    int Lc,M;

    g->c2r_off = (int*)SetPro_checked_malloc(sizeof(int)*(lc+2));
    for (Lc=0; Lc<=lc; Lc++){
      g->c2r_off[Lc] = (int)pos;
      pos += (size_t)(2*Lc+1)*(2*Lc+1);
    }
    g->c2r_off[lc+1] = (int)pos;
    g->c2r = (double*)SetPro_checked_malloc(sizeof(double)*2*pos);
    for (Lc=0; Lc<=lc; Lc++){
      for (M=0; M<(2*Lc+1); M++){
        for (k=0; k<(2*Lc+1); k++){
          size_t idx = (size_t)g->c2r_off[Lc] + (size_t)M*(2*Lc+1) + k;
          g->c2r[2*idx  ] = Comp2Real[Lc][M][k].r;
          g->c2r[2*idx+1] = Comp2Real[Lc][M][k].i;
        }
      }
    }
  }

  g->enabled = 1;
  return 1;
}

static void SetPro_DSVNA_GpuEnd(void)
{
  SetProGpu2Context *g = &SetPro_gpu2;

  if (!g->enabled) return;

  free(g->c2r);
  free(g->c2r_off);
  free(g->gnt);
  free(g->vnab);
  free(g->pro01);
  free(g->pro00);
  free(g->vna_num1);
  free(g->vna_L1);
  free(g->cmb_num0);
  free(g->cmb_L0);
  free(g->sp_ncmb);
  free(g->sp_lfi);
  memset(g,0,sizeof(*g));
}

/* run the whole (Mc_AN,h_AN) list in chunks and store into DS_VNA */
static void SetPro_DSVNA_GpuRun(Type_DS_VNA *****DS_VNA, int OneD_Nloop,
                                int *OneD2Mc_AN, int *OneD2h_AN)
{
  SetProGpu2Context *g = &SetPro_gpu2;
  const int num_proj = g->num_proj;
  const int num_rvna = g->num_rvna;
  const int lfi_max = g->lfi_max;
  const int cmb_max = g->cmb_max;
  const int blk_max = g->blk_max;
  const int l0max_all = g->l0max_all;
  const int l1max = g->l1max;
  const int nM0 = 2*l0max_all + 1;
  const int nM1 = 2*l1max + 1;
  const int nLL = lfi_max + 1;
  const int nm = 2*lfi_max + 1;
  const int nsh = (lfi_max+1)*(lfi_max+1);
  const int mn_max = g->mn_max;
  const int CHUNK = 256;
  int chunk_start;

  double *pr_r,*pr_siT,*pr_coT,*pr_siP,*pr_coP;
  int *pr_spe,*pr_hwan,*pr_h0,*pr_tno;
  double *sh_tab;
  double *sphb,*sphbp;
  double *sum0,*sumr0;
  double *tmpnl;
  float *outbuf;

  const size_t sphb_len = (size_t)CHUNK*nLL*GL_Mesh;
  const size_t sum_len = (size_t)CHUNK*nLL*cmb_max*num_rvna;
  const size_t elem_per_pair = (size_t)cmb_max*nM0*num_rvna*nM1;
  const size_t tmpnl_len = (size_t)CHUNK*elem_per_pair*8; /* 4 arrays x complex */
  const size_t out_len = (size_t)CHUNK*4*(size_t)g->mn_max;

  if (!g->enabled || OneD_Nloop<=0) return;

  pr_r = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_siT = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_coT = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_siP = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_coP = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_spe = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  pr_hwan = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  pr_h0 = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  pr_tno = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  sh_tab = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)CHUNK*nsh*6);
  sphb = (double*)SetPro_checked_malloc(sizeof(double)*sphb_len);
  sphbp = (double*)SetPro_checked_malloc(sizeof(double)*sphb_len);
  sum0 = (double*)SetPro_checked_malloc(sizeof(double)*sum_len);
  sumr0 = (double*)SetPro_checked_malloc(sizeof(double)*sum_len);
  tmpnl = (double*)SetPro_checked_malloc(sizeof(double)*tmpnl_len);
  outbuf = (float*)SetPro_checked_malloc(sizeof(float)*out_len);

  {
    double *pro00 = g->pro00;
    double *pro01 = g->pro01;
    double *vnab = g->vnab;
    double *gnt = g->gnt;
    double *c2r = g->c2r;
    int *c2r_off = g->c2r_off;
    int *sp_lfi = g->sp_lfi;
    int *sp_ncmb = g->sp_ncmb;
    int *cmb_L0 = g->cmb_L0;
    int *cmb_num0 = g->cmb_num0;
    int *vna_L1 = g->vna_L1;
    int *vna_num1 = g->vna_num1;
    const size_t pro_len = (size_t)SpeciesNum*cmb_max*GL_Mesh;
    const size_t vnab_len = (size_t)SpeciesNum*num_rvna*GL_Mesh;
    const size_t gnt_len = (size_t)(l0max_all+1)*nM0*(l1max+1)*nM1*nLL*nm;
    const size_t c2r_len = 2*(size_t)c2r_off[((l0max_all<l1max)?l1max:l0max_all)+1];
    const int lc_max = (l0max_all<l1max) ? l1max : l0max_all;

#pragma acc data copyin(pro00[0:pro_len], pro01[0:pro_len], vnab[0:vnab_len], \
                        gnt[0:gnt_len], c2r[0:c2r_len], c2r_off[0:lc_max+2], \
                        sp_lfi[0:SpeciesNum], sp_ncmb[0:SpeciesNum], \
                        cmb_L0[0:SpeciesNum*cmb_max], cmb_num0[0:SpeciesNum*cmb_max], \
                        vna_L1[0:num_rvna], vna_num1[0:num_rvna], \
                        GL_NormK[0:GL_Mesh]) \
                 create(sphb[0:sphb_len], sphbp[0:sphb_len], \
                        sum0[0:sum_len], sumr0[0:sum_len], tmpnl[0:tmpnl_len], \
                        outbuf[0:out_len], \
                        pr_r[0:CHUNK], pr_siT[0:CHUNK], pr_coT[0:CHUNK], \
                        pr_siP[0:CHUNK], pr_coP[0:CHUNK], pr_spe[0:CHUNK], \
                        pr_hwan[0:CHUNK], pr_h0[0:CHUNK], pr_tno[0:CHUNK], \
                        sh_tab[0:(size_t)CHUNK*nsh*6])
    for (chunk_start=0; chunk_start<OneD_Nloop; chunk_start+=CHUNK){

      const int np = ((OneD_Nloop-chunk_start)<CHUNK) ? (OneD_Nloop-chunk_start) : CHUNK;
      int pp,LLh,mh;

      /* host: pair geometry and the spherical harmonics of every pair */

#pragma omp parallel for schedule(dynamic) private(LLh,mh)
      for (pp=0; pp<np; pp++){
        int Mc_AN = OneD2Mc_AN[chunk_start+pp];
        int h_AN = OneD2h_AN[chunk_start+pp];
        int Gc_AN = M2G[Mc_AN];
        int Cwan = WhatSpecies[Gc_AN];
        int Gh_AN = natn[Gc_AN][h_AN];
        int Rnh = ncn[Gc_AN][h_AN];
        double S_co[3];
        double dx,dy,dz,r,theta,phi;
        double SH[2],dSHt[2],dSHp[2];

        dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
        dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
        dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];

        xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_co);
        r = S_co[0];
        theta = S_co[1];
        phi = S_co[2];
        if (r<1.0e-10) r = 1.0e-10;

        pr_r[pp] = r;
        pr_siT[pp] = sin(theta);
        pr_coT[pp] = cos(theta);
        pr_siP[pp] = sin(phi);
        pr_coP[pp] = cos(phi);
        pr_spe[pp] = Cwan;
        pr_hwan[pp] = WhatSpecies[Gh_AN];
        pr_h0[pp] = (h_AN==0);
        pr_tno[pp] = Spe_Total_NO[Cwan];

        for (LLh=0; LLh<=sp_lfi[Cwan]; LLh++){
          for (mh=-LLh; mh<=LLh; mh++){
            size_t off = ((size_t)pp*nsh + (size_t)LLh*LLh + (mh+LLh))*6;

            ComplexSH(LLh,mh,theta,phi,SH,dSHt,dSHp);
            sh_tab[off+0] = SH[0];
            sh_tab[off+1] = -SH[1];
            sh_tab[off+2] = dSHt[0];
            sh_tab[off+3] = -dSHt[1];
            sh_tab[off+4] = dSHp[0];
            sh_tab[off+5] = -dSHp[1];
          }
        }
      }

#pragma acc update device(pr_r[0:np], pr_siT[0:np], pr_coT[0:np], \
                          pr_siP[0:np], pr_coP[0:np], pr_spe[0:np], \
                          pr_hwan[0:np], pr_h0[0:np], pr_tno[0:np], \
                          sh_tab[0:(size_t)np*nsh*6])

      /* K0: spherical Bessel tables of every (pair, GL point) */

#pragma acc parallel loop gang vector vector_length(128) \
    present(pr_r[0:CHUNK], pr_spe[0:CHUNK], sp_lfi[0:SpeciesNum], \
            GL_NormK[0:GL_Mesh], sphb[0:sphb_len], sphbp[0:sphb_len])
      for (int pt=0; pt<np*GL_Mesh; pt++){
        const int pp2 = pt/GL_Mesh;
        const int i = pt - pp2*GL_Mesh;
        const int lfi = sp_lfi[pr_spe[pp2]];
        double tb[SETPRO_BES_LMAX];
        double tbp[SETPRO_BES_LMAX];
        int LL;

        SetPro_Spherical_Bessel2_dev(GL_NormK[i]*pr_r[pp2],lfi,tb,tbp);
        for (LL=0; LL<=lfi; LL++){
          sphb [((size_t)pp2*nLL + LL)*GL_Mesh + i] = tb[LL];
          sphbp[((size_t)pp2*nLL + LL)*GL_Mesh + i] = tbp[LL];
        }
      }

      /* K1: Gauss-Legendre quadrature sums */

#pragma acc parallel loop gang vector vector_length(128) \
    present(pr_spe[0:CHUNK], pr_hwan[0:CHUNK], pr_h0[0:CHUNK], sp_lfi[0:SpeciesNum], \
            sp_ncmb[0:SpeciesNum], pro00[0:pro_len], pro01[0:pro_len], \
            vnab[0:vnab_len], sphb[0:sphb_len], sphbp[0:sphb_len], \
            sum0[0:sum_len], sumr0[0:sum_len])
      for (size_t q=0; q<(size_t)np*nLL*cmb_max*num_rvna; q++){
        const int Lq = (int)(q % num_rvna);
        const int cq = (int)((q/num_rvna) % cmb_max);
        const int LLq = (int)((q/((size_t)num_rvna*cmb_max)) % nLL);
        const int pp2 = (int)(q/((size_t)num_rvna*cmb_max*nLL));
        const int spe = pr_spe[pp2];

        if (LLq<=sp_lfi[spe] && cq<sp_ncmb[spe]){
          const double *p00 = pro00 + ((size_t)spe*cmb_max+cq)*GL_Mesh;
          const double *p01 = pro01 + ((size_t)spe*cmb_max+cq)*GL_Mesh;
          const double *vb = vnab + ((size_t)pr_hwan[pp2]*num_rvna+Lq)*GL_Mesh;
          const double *sb = sphb + ((size_t)pp2*nLL + LLq)*GL_Mesh;
          const double *sbp = sphbp + ((size_t)pp2*nLL + LLq)*GL_Mesh;
          double tmp1 = 0.0,tmp2 = 0.0;
          int i;

          for (i=0; i<GL_Mesh; i++){
            tmp1 += (p00[i]*sb[i])*vb[i];
            tmp2 += (p01[i]*sbp[i])*vb[i];
          }

          sum0[q] = tmp1;
          sumr0[q] = pr_h0[pp2] ? 0.0 : tmp2;
        }
        else{
          sum0[q] = 0.0;
          sumr0[q] = 0.0;
        }
      }

      /* K2: assemble TmpNL (the LL,m sums with Gaunt x Y_lm weights) */

#pragma acc parallel loop gang vector vector_length(128) \
    present(pr_spe[0:CHUNK], sp_lfi[0:SpeciesNum], sp_ncmb[0:SpeciesNum], \
            cmb_L0[0:SpeciesNum*cmb_max], vna_L1[0:num_rvna], \
            gnt[0:gnt_len], sh_tab[0:(size_t)CHUNK*nsh*6], \
            sum0[0:sum_len], sumr0[0:sum_len], tmpnl[0:tmpnl_len])
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM1);
        const int Lq = (int)((e/nM1) % num_rvna);
        const int M0i = (int)((e/((size_t)nM1*num_rvna)) % nM0);
        const int cq = (int)((e/((size_t)nM1*num_rvna*nM0)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (cq<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cq] : -1;
        const int L1 = vna_L1[Lq];
        const int M0 = M0i - l0max_all;
        const int M1 = M1i - l1max;
        double s_r = 0.0,s_i = 0.0;
        double r_r = 0.0,r_i = 0.0;
        double t_r = 0.0,t_i = 0.0;
        double p_r = 0.0,p_i = 0.0;

        if (L0>=0 && M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){
          const int lfi = sp_lfi[spe];
          int LL,m;

          for (LL=0; LL<=lfi; LL++){

            const int Ls = -L0 + L1 + LL;
            double pw_r,pw_i;

            /* Im_pow(-1,Ls) with the original C modulo semantics */
            if (Ls%2==0){
              if (Ls%4==0){ pw_r = 1.0; pw_i = 0.0; }
              else        { pw_r = -1.0; pw_i = 0.0; }
            }
            else{
              if ((Ls+1)%4==0){ pw_r = 0.0; pw_i = 1.0; }
              else            { pw_r = 0.0; pw_i = -1.0; }
            }

            if ( abs(L1-LL)<=L0 && L0<=(L1+LL) ){

              m = M0 - M1;
              if (-LL<=m && m<=LL){

                const size_t shoff = ((size_t)pp2*nsh + (size_t)LL*LL + (m+LL))*6;
                const double cy_r = sh_tab[shoff+0];
                const double cy_i = sh_tab[shoff+1];
                const double cyt_r = sh_tab[shoff+2];
                const double cyt_i = sh_tab[shoff+3];
                const double cyp_r = sh_tab[shoff+4];
                const double cyp_i = sh_tab[shoff+5];
                const double cy1_r = pw_r*cy_r - pw_i*cy_i;
                const double cy1_i = pw_r*cy_i + pw_i*cy_r;
                const double cyt1_r = pw_r*cyt_r - pw_i*cyt_i;
                const double cyt1_i = pw_r*cyt_i + pw_i*cyt_r;
                const double cyp1_r = pw_r*cyp_r - pw_i*cyp_i;
                const double cyp1_i = pw_r*cyp_i + pw_i*cyp_r;
                const size_t gidx = ((((((size_t)L0*nM0 + (M0+l0max_all))*(l1max+1) + L1)*nM1
                                       + (M1+l1max))*nLL + LL)*nm + (m+lfi_max));
                const double gant = gnt[gidx];
                const size_t sq = (((size_t)pp2*nLL + LL)*cmb_max + cq)*num_rvna + Lq;
                const double tmp2 = gant*sum0[sq];
                const double tmp0 = gant*sumr0[sq];

                s_r += cy1_r*tmp2;
                s_i += cy1_i*tmp2;
                r_r += cy1_r*tmp0;
                r_i += cy1_i*tmp0;
                t_r += cyt1_r*tmp2;
                t_i += cyt1_i*tmp2;
                p_r += cyp1_r*tmp2;
                p_i += cyp1_i*tmp2;
              }
            }
          }
        }

        {
          double *dst = tmpnl + e*8;

          dst[0] = s_r; dst[1] = s_i;
          dst[2] = r_r; dst[3] = r_i;
          dst[4] = t_r; dst[5] = t_i;
          dst[6] = p_r; dst[7] = p_i;
        }
      }

      /* K3: complex-to-real transforms and the DS_VNA elements */

#pragma acc parallel loop gang vector vector_length(128) \
    present(pr_spe[0:CHUNK], pr_h0[0:CHUNK], pr_r[0:CHUNK], pr_siT[0:CHUNK], \
            pr_coT[0:CHUNK], pr_siP[0:CHUNK], pr_coP[0:CHUNK], pr_tno[0:CHUNK], \
            sp_ncmb[0:SpeciesNum], cmb_L0[0:SpeciesNum*cmb_max], \
            cmb_num0[0:SpeciesNum*cmb_max], vna_L1[0:num_rvna], vna_num1[0:num_rvna], \
            c2r[0:c2r_len], c2r_off[0:lc_max+2], tmpnl[0:tmpnl_len], \
            outbuf[0:out_len])
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM1);
        const int Lq = (int)((e/nM1) % num_rvna);
        const int M0i = (int)((e/((size_t)nM1*num_rvna)) % nM0);
        const int cq = (int)((e/((size_t)nM1*num_rvna*nM0)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (cq<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cq] : -1;
        const int L1 = vna_L1[Lq];
        const int M0 = M0i - l0max_all;
        const int M1 = M1i - l1max;

        if (L0>=0 && M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          double s0_r = 0.0,s0_i = 0.0;
          double sr_r = 0.0,sr_i = 0.0;
          double st_r = 0.0,st_i = 0.0;
          double sp_r = 0.0,sp_i = 0.0;
          int k1,k0;

          for (k1=-L1; k1<=L1; k1++){

            /* first transform: Cmat(M0,k1) = sum_k0 conj(C2R0)*TmpNL(k0,k1) */

            double c0_r = 0.0,c0_i = 0.0;
            double cr_r = 0.0,cr_i = 0.0;
            double ct_r = 0.0,ct_i = 0.0;
            double cp_r = 0.0,cp_i = 0.0;

            for (k0=-L0; k0<=L0; k0++){
              const size_t uidx = (size_t)c2r_off[L0] + (size_t)(L0+M0)*(2*L0+1) + (L0+k0);
              const double u_r = c2r[2*uidx];
              const double u_i = -c2r[2*uidx+1];
              const size_t te = ((((size_t)pp2*cmb_max + cq)*nM0 + (k0+l0max_all))
                                 *num_rvna + Lq)*nM1 + (k1+l1max);
              const double *tv = tmpnl + te*8;

              c0_r += u_r*tv[0] - u_i*tv[1];
              c0_i += u_r*tv[1] + u_i*tv[0];
              cr_r += u_r*tv[2] - u_i*tv[3];
              cr_i += u_r*tv[3] + u_i*tv[2];
              ct_r += u_r*tv[4] - u_i*tv[5];
              ct_i += u_r*tv[5] + u_i*tv[4];
              cp_r += u_r*tv[6] - u_i*tv[7];
              cp_i += u_r*tv[7] + u_i*tv[6];
            }

            /* second transform */

            {
              const size_t vidx = (size_t)c2r_off[L1] + (size_t)(L1+M1)*(2*L1+1) + (L1+k1);
              const double v_r = c2r[2*vidx];
              const double v_i = c2r[2*vidx+1];

              s0_r += (c0_r*v_r - c0_i*v_i);
              s0_i += (c0_r*v_i + c0_i*v_r);
              sr_r += (cr_r*v_r - cr_i*v_i);
              sr_i += (cr_r*v_i + cr_i*v_r);
              st_r += (ct_r*v_r - ct_i*v_i);
              st_i += (ct_r*v_i + ct_i*v_r);
              sp_r += (cp_r*v_r - cp_i*v_i);
              sp_i += (cp_r*v_i + cp_i*v_r);
            }
          }

          {
            const int num0 = cmb_num0[spe*cmb_max+cq];
            const int num1 = vna_num1[Lq];
            const int row = num0 + L0 + M0;
            const int col = num1 + L1 + M1;
            const size_t obase = ((size_t)pp2*4)*(size_t)mn_max;
            const size_t oidx = (size_t)row*num_proj + col;
            const double r = pr_r[pp2];
            const double siT = pr_siT[pp2];
            const double coT = pr_coT[pp2];
            const double siP = pr_siP[pp2];
            const double coP = pr_coP[pp2];

            outbuf[obase + oidx] = 8.0*(float)s0_r;

            if (!pr_h0[pp2]){

              if (fabs(siT)<10e-14){
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(float)(siT*coP*sr_r + coT*coP/r*st_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(float)(siT*siP*sr_r + coT*siP/r*st_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(float)(coT*sr_r - siT/r*st_r);
              }
              else{
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(float)(siT*coP*sr_r + coT*coP/r*st_r
                               - siP/siT/r*sp_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(float)(siT*siP*sr_r + coT*siP/r*st_r
                               + coP/siT/r*sp_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(float)(coT*sr_r - siT/r*st_r);
              }
            }
            else{
              outbuf[obase + (size_t)mn_max + oidx] = 0.0f;
              outbuf[obase + 2*(size_t)mn_max + oidx] = 0.0f;
              outbuf[obase + 3*(size_t)mn_max + oidx] = 0.0f;
            }
          }
        }
      }

#pragma acc update self(outbuf[0:(size_t)np*4*(size_t)mn_max])

      /* host: scatter the chunk results into DS_VNA */

#pragma omp parallel for schedule(dynamic)
      for (pp=0; pp<np; pp++){
        int Mc_AN = OneD2Mc_AN[chunk_start+pp];
        int h_AN = OneD2h_AN[chunk_start+pp];
        int tno = pr_tno[pp];
        int kk,i2;

        for (kk=0; kk<=3; kk++){
          for (i2=0; i2<tno; i2++){
            memcpy(DS_VNA[kk][Mc_AN][h_AN][i2],
                   outbuf + ((size_t)pp*4 + kk)*(size_t)mn_max
                          + (size_t)i2*num_proj,
                   sizeof(float)*(size_t)num_proj);
          }
        }
      }

    } /* chunk_start */
  }

  free(outbuf);
  free(tmpnl);
  free(sumr0);
  free(sum0);
  free(sphbp);
  free(sphb);
  free(sh_tab);
  free(pr_tno);
  free(pr_h0);
  free(pr_hwan);
  free(pr_spe);
  free(pr_coP);
  free(pr_siP);
  free(pr_coT);
  free(pr_siT);
  free(pr_r);
}

/* ------------------------------------------------------------------ */
/* Batched device construction of HVNA2 / HVNA3 (the one-centre PAO
   pairs against the crude VNA potential).  Set_VNA2 and Set_VNA3 are
   mirror images: the PAO-product species and the direction sign swap.
   The same host-tabulated Gaunt/Y_lm treatment as the DS_VNA batch is
   used; the upper->lower conjugate copy and both unitary transforms
   keep the exact CPU loop order.                                      */
/* ------------------------------------------------------------------ */

typedef struct {
  int enabled;
  int l0max;            /* max Spe_MaxL_Basis                          */
  int lfi_max;          /* 2*l0max                                     */
  int cmb_max;
  int tno_max;
  int mn_max;           /* tno_max^2                                   */

  int *sp_lfi;          /* [spe] 2*Spe_MaxL_Basis                      */
  int *sp_ncmb;
  int *sp_tno;
  int *cmb_L0;          /* [spe*cmb_max+c]                             */
  int *cmb_num0;

  double *crude;        /* [spe][GL_Mesh]                              */
  double *prod;         /* [spe][c0][c1][LL][GL_Mesh] (c0<=c1 pairs)   */
  double *gnt2;         /* (-1)^|m| Gaunt(L0,M0,L1,M1,LL,-m), m=M1-M0  */
  double *c2r;
  int *c2r_off;
} SetProGpu3Context;

static SetProGpu3Context SetPro_gpu3 = { 0 };

static int SetPro_VNA23_GpuBegin(void)
{
  SetProGpu3Context *g = &SetPro_gpu3;
  int spe,L0,Mul0,L1,Mul1,LL,i,c0,c1,k;
  size_t free_bytes = 0,total_bytes = 0;
  int node_ranks = 1;
  MPI_Comm node_comm = MPI_COMM_NULL;

  memset(g,0,sizeof(*g));

  if (scf_eigen_lib_flag!=GPUSOLVER) return 0;
  if (!SetPro_collective_env_flag("OPENMX_SETPRO_GPU",1)) return 0;

  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
  MPI_Comm_size(node_comm,&node_ranks);
  if (node_ranks<1) node_ranks = 1;
  MPI_Comm_free(&node_comm);

  if (cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess) return 0;
  {
    const size_t reserve = (size_t)256*1024*1024;
    const size_t need = (size_t)512*1024*1024;

    if (free_bytes<=reserve ||
        (free_bytes - reserve)/(size_t)node_ranks<=need) return 0;
  }

  g->sp_lfi = (int*)SetPro_checked_malloc(sizeof(int)*SpeciesNum);
  g->sp_ncmb = (int*)SetPro_checked_malloc(sizeof(int)*SpeciesNum);
  g->sp_tno = (int*)SetPro_checked_malloc(sizeof(int)*SpeciesNum);

  g->l0max = 0;
  g->cmb_max = 0;
  g->tno_max = 0;
  for (spe=0; spe<SpeciesNum; spe++){
    int ncmb = 0,tno = 0;

    g->sp_lfi[spe] = 2*Spe_MaxL_Basis[spe];
    if (g->l0max<Spe_MaxL_Basis[spe]) g->l0max = Spe_MaxL_Basis[spe];
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      ncmb += Spe_Num_Basis[spe][L0];
      tno += Spe_Num_Basis[spe][L0]*(2*L0+1);
    }
    g->sp_ncmb[spe] = ncmb;
    g->sp_tno[spe] = tno;
    if (g->cmb_max<ncmb) g->cmb_max = ncmb;
    if (g->tno_max<tno) g->tno_max = tno;
  }
  g->lfi_max = 2*g->l0max;
  g->mn_max = g->tno_max*g->tno_max;

  g->cmb_L0 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)SpeciesNum*g->cmb_max);
  g->cmb_num0 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)SpeciesNum*g->cmb_max);

  for (spe=0; spe<SpeciesNum; spe++){
    int num0 = 0,c = 0;

    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
        g->cmb_L0[spe*g->cmb_max+c] = L0;
        g->cmb_num0[spe*g->cmb_max+c] = num0;
        num0 += 2*L0 + 1;
        c++;
      }
    }
    for (; c<g->cmb_max; c++){
      g->cmb_L0[spe*g->cmb_max+c] = -1;
      g->cmb_num0[spe*g->cmb_max+c] = 0;
    }
  }

  g->crude = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)SpeciesNum*GL_Mesh);
  for (spe=0; spe<SpeciesNum; spe++){
    for (i=0; i<GL_Mesh; i++){
      g->crude[(size_t)spe*GL_Mesh+i] = Spe_CrudeVNA_Bessel[spe][i];
    }
  }

  {
    const int nLL = g->lfi_max + 1;
    size_t plen = (size_t)SpeciesNum*g->cmb_max*g->cmb_max*nLL*GL_Mesh;
    size_t q;

    g->prod = (double*)SetPro_checked_malloc(sizeof(double)*plen);
    for (q=0; q<plen; q++) g->prod[q] = 0.0;

    for (spe=0; spe<SpeciesNum; spe++){
      int ca = 0;

      for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
        for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
          int cb = 0;

          for (L1=0; L1<=Spe_MaxL_Basis[spe]; L1++){
            for (Mul1=0; Mul1<Spe_Num_Basis[spe][L1]; Mul1++){

              if (L0<=L1){
                for (LL=0; LL<=2*L1; LL++){
                  for (i=0; i<GL_Mesh; i++){
                    g->prod[((((size_t)spe*g->cmb_max+ca)*g->cmb_max+cb)*nLL+LL)*GL_Mesh+i]
                      = Spe_ProductRF_Bessel[spe][L0][Mul0][L1][Mul1][LL][i];
                  }
                }
              }
              cb++;
            }
          }
          ca++;
        }
      }
    }
  }

  /* (-1)^|m| * Gaunt(L0,M0,L1,M1,LL,-m) with m = M1-M0 */

  {
    const int nM = 2*g->l0max + 1;
    const int nLL = g->lfi_max + 1;
    size_t gsz = (size_t)(g->l0max+1)*nM*(g->l0max+1)*nM*nLL;
    int L0v,M0,L1v,M1,m;
    size_t q;

    g->gnt2 = (double*)SetPro_checked_malloc(sizeof(double)*gsz);
    for (q=0; q<gsz; q++) g->gnt2[q] = 0.0;

    for (L0v=0; L0v<=g->l0max; L0v++){
      for (M0=-L0v; M0<=L0v; M0++){
        for (L1v=0; L1v<=g->l0max; L1v++){
          for (M1=-L1v; M1<=L1v; M1++){
            for (LL=0; LL<nLL; LL++){
              m = M1 - M0;
              if (-LL<=m && m<=LL){
                size_t idx = (((((size_t)L0v*nM + (M0+g->l0max))*(g->l0max+1) + L1v)*nM
                               + (M1+g->l0max))*nLL + LL);
                g->gnt2[idx] = pow(-1.0,(double)abs(m))*Gaunt(L0v,M0,L1v,M1,LL,-m);
              }
            }
          }
        }
      }
    }
  }

  {
    const int lc = g->l0max;
    size_t pos = 0;
    int Lc,M;

    g->c2r_off = (int*)SetPro_checked_malloc(sizeof(int)*(lc+2));
    for (Lc=0; Lc<=lc; Lc++){
      g->c2r_off[Lc] = (int)pos;
      pos += (size_t)(2*Lc+1)*(2*Lc+1);
    }
    g->c2r_off[lc+1] = (int)pos;
    g->c2r = (double*)SetPro_checked_malloc(sizeof(double)*2*pos);
    for (Lc=0; Lc<=lc; Lc++){
      for (M=0; M<(2*Lc+1); M++){
        for (k=0; k<(2*Lc+1); k++){
          size_t idx = (size_t)g->c2r_off[Lc] + (size_t)M*(2*Lc+1) + k;
          g->c2r[2*idx  ] = Comp2Real[Lc][M][k].r;
          g->c2r[2*idx+1] = Comp2Real[Lc][M][k].i;
        }
      }
    }
  }

  g->enabled = 1;
  return 1;
}

static void SetPro_VNA23_GpuEnd(void)
{
  SetProGpu3Context *g = &SetPro_gpu3;

  if (!g->enabled) return;

  free(g->c2r);
  free(g->c2r_off);
  free(g->gnt2);
  free(g->prod);
  free(g->crude);
  free(g->cmb_num0);
  free(g->cmb_L0);
  free(g->sp_tno);
  free(g->sp_ncmb);
  free(g->sp_lfi);
  memset(g,0,sizeof(*g));
}

/* run VNA2 (mirror==0) or VNA3 (mirror==1) in pair chunks */
static void SetPro_VNA23_GpuRun(double *****HVNA23, int mirror, int OneD_Nloop,
                                int *OneD2Mc_AN, int *OneD2h_AN)
{
  SetProGpu3Context *g = &SetPro_gpu3;
  const int l0max = g->l0max;
  const int lfi_max = g->lfi_max;
  const int cmb_max = g->cmb_max;
  const int mn_max = g->mn_max;
  const int tno_max = g->tno_max;
  const int nM = 2*l0max + 1;
  const int nLL = lfi_max + 1;
  const int nsh = (lfi_max+1)*(lfi_max+1);
  const int CHUNK = 512;
  const double Dk = PAO_Nkmax - Radial_kmin;
  int chunk_start;

  double *pr_r,*pr_siT,*pr_coT,*pr_siP,*pr_coP;
  int *pr_spe,*pr_vspe,*pr_h0;
  double *sh_tab;
  double *sphb,*sphbp;
  double *sum0,*sumr0;
  double *tmpnl;
  double *outbuf;

  const size_t sphb_len = (size_t)CHUNK*nLL*GL_Mesh;
  const size_t sum_len = (size_t)CHUNK*nLL*cmb_max*cmb_max;
  const size_t elem_per_pair = (size_t)cmb_max*nM*cmb_max*nM;
  const size_t tmpnl_len = (size_t)CHUNK*elem_per_pair*8;
  const size_t out_len = (size_t)CHUNK*4*(size_t)mn_max;

  if (!g->enabled || OneD_Nloop<=0) return;

  pr_r = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_siT = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_coT = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_siP = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_coP = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_spe = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  pr_vspe = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  pr_h0 = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  sh_tab = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)CHUNK*nsh*6);
  sphb = (double*)SetPro_checked_malloc(sizeof(double)*sphb_len);
  sphbp = (double*)SetPro_checked_malloc(sizeof(double)*sphb_len);
  sum0 = (double*)SetPro_checked_malloc(sizeof(double)*sum_len);
  sumr0 = (double*)SetPro_checked_malloc(sizeof(double)*sum_len);
  tmpnl = (double*)SetPro_checked_malloc(sizeof(double)*tmpnl_len);
  outbuf = (double*)SetPro_checked_malloc(sizeof(double)*out_len);

  {
    double *crude = g->crude;
    double *prod = g->prod;
    double *gnt2 = g->gnt2;
    double *c2r = g->c2r;
    int *c2r_off = g->c2r_off;
    int *sp_lfi = g->sp_lfi;
    int *sp_ncmb = g->sp_ncmb;
    int *cmb_L0 = g->cmb_L0;
    int *cmb_num0 = g->cmb_num0;
    const size_t crude_len = (size_t)SpeciesNum*GL_Mesh;
    const size_t prod_len = (size_t)SpeciesNum*cmb_max*cmb_max*nLL*GL_Mesh;
    const size_t gnt2_len = (size_t)(l0max+1)*nM*(l0max+1)*nM*nLL;
    const size_t c2r_len = 2*(size_t)c2r_off[l0max+1];

#pragma acc data copyin(crude[0:crude_len], prod[0:prod_len], gnt2[0:gnt2_len], \
                        c2r[0:c2r_len], c2r_off[0:l0max+2], \
                        sp_lfi[0:SpeciesNum], sp_ncmb[0:SpeciesNum], \
                        cmb_L0[0:SpeciesNum*cmb_max], cmb_num0[0:SpeciesNum*cmb_max], \
                        GL_NormK[0:GL_Mesh], GL_Weight[0:GL_Mesh]) \
                 create(sphb[0:sphb_len], sphbp[0:sphb_len], \
                        sum0[0:sum_len], sumr0[0:sum_len], tmpnl[0:tmpnl_len], \
                        outbuf[0:out_len], \
                        pr_r[0:CHUNK], pr_siT[0:CHUNK], pr_coT[0:CHUNK], \
                        pr_siP[0:CHUNK], pr_coP[0:CHUNK], pr_spe[0:CHUNK], \
                        pr_vspe[0:CHUNK], pr_h0[0:CHUNK], sh_tab[0:(size_t)CHUNK*nsh*6])
    for (chunk_start=0; chunk_start<OneD_Nloop; chunk_start+=CHUNK){

      const int np = ((OneD_Nloop-chunk_start)<CHUNK) ? (OneD_Nloop-chunk_start) : CHUNK;
      int pp,LLh,mh;

#pragma omp parallel for schedule(dynamic) private(LLh,mh)
      for (pp=0; pp<np; pp++){
        int Mc_AN = OneD2Mc_AN[chunk_start+pp];
        int h_AN = OneD2h_AN[chunk_start+pp];
        int Gc_AN = M2G[Mc_AN];
        int Gh_AN = natn[Gc_AN][h_AN];
        int Rnh = ncn[Gc_AN][h_AN];
        int spe_pao,spe_vna;
        double S_co[3];
        double dx,dy,dz,r,theta,phi;
        double SH[2],dSHt[2],dSHp[2];

        if (mirror==0){
          spe_pao = WhatSpecies[Gc_AN];
          spe_vna = WhatSpecies[Gh_AN];
          dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
          dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
          dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];
        }
        else{
          spe_pao = WhatSpecies[Gh_AN];
          spe_vna = WhatSpecies[Gc_AN];
          dx = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rnh][1];
          dy = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rnh][2];
          dz = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rnh][3];
        }

        xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_co);
        r = S_co[0];
        theta = S_co[1];
        phi = S_co[2];
        if (r<1.0e-10) r = 1.0e-10;

        pr_r[pp] = r;
        pr_siT[pp] = sin(theta);
        pr_coT[pp] = cos(theta);
        pr_siP[pp] = sin(phi);
        pr_coP[pp] = cos(phi);
        pr_spe[pp] = spe_pao;
        pr_vspe[pp] = spe_vna;
        pr_h0[pp] = (h_AN==0);

        for (LLh=0; LLh<=sp_lfi[spe_pao]; LLh++){
          for (mh=-LLh; mh<=LLh; mh++){
            size_t off = ((size_t)pp*nsh + (size_t)LLh*LLh + (mh+LLh))*6;

            ComplexSH(LLh,mh,theta,phi,SH,dSHt,dSHp);
            sh_tab[off+0] = SH[0];
            sh_tab[off+1] = SH[1];
            sh_tab[off+2] = dSHt[0];
            sh_tab[off+3] = dSHt[1];
            sh_tab[off+4] = dSHp[0];
            sh_tab[off+5] = dSHp[1];
          }
        }
      }

#pragma acc update device(pr_r[0:np], pr_siT[0:np], pr_coT[0:np], \
                          pr_siP[0:np], pr_coP[0:np], pr_spe[0:np], \
                          pr_vspe[0:np], pr_h0[0:np], sh_tab[0:(size_t)np*nsh*6])

      /* K0: spherical Bessel tables */

#pragma acc parallel loop gang vector vector_length(128) \
    present(pr_r[0:CHUNK], pr_spe[0:CHUNK], sp_lfi[0:SpeciesNum], \
            GL_NormK[0:GL_Mesh], sphb[0:sphb_len], sphbp[0:sphb_len])
      for (int pt=0; pt<np*GL_Mesh; pt++){
        const int pp2 = pt/GL_Mesh;
        const int i = pt - pp2*GL_Mesh;
        const int lfi = sp_lfi[pr_spe[pp2]];
        double tb[SETPRO_BES_LMAX];
        double tbp[SETPRO_BES_LMAX];
        int LL;

        SetPro_Spherical_Bessel2_dev(GL_NormK[i]*pr_r[pp2],lfi,tb,tbp);
        for (LL=0; LL<=lfi; LL++){
          sphb [((size_t)pp2*nLL + LL)*GL_Mesh + i] = tb[LL];
          sphbp[((size_t)pp2*nLL + LL)*GL_Mesh + i] = tbp[LL];
        }
      }

      /* K1: quadrature sums for every (pair, LL, c0<=c1) block */

#pragma acc parallel loop gang vector vector_length(128) \
    present(pr_spe[0:CHUNK], pr_vspe[0:CHUNK], sp_lfi[0:SpeciesNum], sp_ncmb[0:SpeciesNum], \
            cmb_L0[0:SpeciesNum*cmb_max], crude[0:crude_len], prod[0:prod_len], \
            GL_NormK[0:GL_Mesh], GL_Weight[0:GL_Mesh], \
            sphb[0:sphb_len], sphbp[0:sphb_len], sum0[0:sum_len], sumr0[0:sum_len])
      for (size_t q=0; q<(size_t)np*nLL*cmb_max*cmb_max; q++){
        const int cb = (int)(q % cmb_max);
        const int ca = (int)((q/cmb_max) % cmb_max);
        const int LLq = (int)((q/((size_t)cmb_max*cmb_max)) % nLL);
        const int pp2 = (int)(q/((size_t)cmb_max*cmb_max*nLL));
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        double s = 0.0,sr = 0.0;

        if (L0>=0 && L1>=0 && L0<=L1 && LLq<=2*L1 &&
            abs(L1-LLq)<=L0 && L0<=(L1+LLq)){

          const double *cru = crude + (size_t)pr_vspe[pp2]*GL_Mesh;
          const double *pr2 = prod + ((((size_t)spe*cmb_max+ca)*cmb_max+cb)*nLL+LLq)*GL_Mesh;
          const double *sb = sphb + ((size_t)pp2*nLL + LLq)*GL_Mesh;
          const double *sbp = sphbp + ((size_t)pp2*nLL + LLq)*GL_Mesh;
          int i;

          for (i=0; i<GL_Mesh; i++){
            const double Normk = GL_NormK[i];
            const double tmp10 = 0.50*Dk*cru[i]*pr2[i]*GL_Weight[i]*Normk*Normk;

            s += tmp10*sb[i];
            sr += tmp10*sbp[i]*Normk;
          }
        }

        sum0[q] = s;
        sumr0[q] = sr;
      }

      /* K2a: assemble the upper (L0<=L1) TmpHNA elements */

#pragma acc parallel loop gang vector vector_length(128) \
    present(pr_spe[0:CHUNK], sp_lfi[0:SpeciesNum], sp_ncmb[0:SpeciesNum], \
            cmb_L0[0:SpeciesNum*cmb_max], gnt2[0:gnt2_len], \
            sh_tab[0:(size_t)CHUNK*nsh*6], sum0[0:sum_len], sumr0[0:sum_len], \
            tmpnl[0:tmpnl_len])
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM);
        const int cb = (int)((e/nM) % cmb_max);
        const int M0i = (int)((e/((size_t)nM*cmb_max)) % nM);
        const int ca = (int)((e/((size_t)nM*cmb_max*nM)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        const int M0 = M0i - l0max;
        const int M1 = M1i - l0max;
        double s_r = 0.0,s_i = 0.0;
        double r_r = 0.0,r_i = 0.0;
        double t_r = 0.0,t_i = 0.0;
        double p_r = 0.0,p_i = 0.0;

        if (L0>=0 && L1>=0 && L0<=L1 &&
            M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          const int m = M1 - M0;
          int LL;

          for (LL=0; LL<=2*L1; LL++){

            if (abs(L1-LL)<=L0 && L0<=(L1+LL) && -LL<=m && m<=LL){

              const size_t shoff = ((size_t)pp2*nsh + (size_t)LL*LL + (m+LL))*6;
              const double cy_r = sh_tab[shoff+0];
              const double cy_i = sh_tab[shoff+1];
              const double cyt_r = sh_tab[shoff+2];
              const double cyt_i = sh_tab[shoff+3];
              const double cyp_r = sh_tab[shoff+4];
              const double cyp_i = sh_tab[shoff+5];
              const size_t gidx = (((((size_t)L0*nM + (M0+l0max))*(l0max+1) + L1)*nM
                                    + (M1+l0max))*nLL + LL);
              const double gant = gnt2[gidx];
              const size_t sq = (((size_t)pp2*nLL + LL)*cmb_max + ca)*cmb_max + cb;
              const double sv = sum0[sq];
              const double srv = sumr0[sq];
              const double cs_r = cy_r*gant;
              const double cs_i = cy_i*gant;
              const double cst_r = cyt_r*gant;
              const double cst_i = cyt_i*gant;
              const double csp_r = cyp_r*gant;
              const double csp_i = cyp_i*gant;

              s_r += cs_r*sv;
              s_i += cs_i*sv;
              r_r += cs_r*srv;
              r_i += cs_i*srv;
              t_r += cst_r*sv;
              t_i += cst_i*sv;
              p_r += csp_r*sv;
              p_i += csp_i*sv;
            }
          }
        }

        {
          double *dst = tmpnl + e*8;

          dst[0] = s_r; dst[1] = s_i;
          dst[2] = r_r; dst[3] = r_i;
          dst[4] = t_r; dst[5] = t_i;
          dst[6] = p_r; dst[7] = p_i;
        }
      }

      /* K2b: conjugate copy of the upper part to the lower part */

#pragma acc parallel loop gang vector vector_length(128) \
    present(pr_spe[0:CHUNK], sp_ncmb[0:SpeciesNum], cmb_L0[0:SpeciesNum*cmb_max], \
            tmpnl[0:tmpnl_len])
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM);
        const int cb = (int)((e/nM) % cmb_max);
        const int M0i = (int)((e/((size_t)nM*cmb_max)) % nM);
        const int ca = (int)((e/((size_t)nM*cmb_max*nM)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        const int M0 = M0i - l0max;
        const int M1 = M1i - l0max;

        if (L0>=0 && L1>=0 && L0>L1 &&
            M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          const size_t src = ((((size_t)pp2*cmb_max + cb)*nM + M1i)
                              *cmb_max + ca)*nM + M0i;
          const double *sv = tmpnl + src*8;
          double *dst = tmpnl + e*8;

          dst[0] = sv[0]; dst[1] = -sv[1];
          dst[2] = sv[2]; dst[3] = -sv[3];
          dst[4] = sv[4]; dst[5] = -sv[5];
          dst[6] = sv[6]; dst[7] = -sv[7];
        }
        else if (L0>=0 && L1>=0 && L0==L1 && ca==cb && M0==M1 &&
                 M0>=-L0 && M0<=L0){

          /* the host copy loop conjugates the L0==L1 diagonal in place */
          double *dst = tmpnl + e*8;

          dst[1] = -dst[1];
          dst[3] = -dst[3];
          dst[5] = -dst[5];
          dst[7] = -dst[7];
        }
      }

      /* K3: complex-to-real transforms and the HVNA2/3 elements */

#pragma acc parallel loop gang vector vector_length(128) \
    present(pr_spe[0:CHUNK], pr_h0[0:CHUNK], pr_r[0:CHUNK], pr_siT[0:CHUNK], \
            pr_coT[0:CHUNK], pr_siP[0:CHUNK], pr_coP[0:CHUNK], \
            sp_ncmb[0:SpeciesNum], cmb_L0[0:SpeciesNum*cmb_max], \
            cmb_num0[0:SpeciesNum*cmb_max], c2r[0:c2r_len], c2r_off[0:l0max+2], \
            tmpnl[0:tmpnl_len], outbuf[0:out_len])
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM);
        const int cb = (int)((e/nM) % cmb_max);
        const int M0i = (int)((e/((size_t)nM*cmb_max)) % nM);
        const int ca = (int)((e/((size_t)nM*cmb_max*nM)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        const int M0 = M0i - l0max;
        const int M1 = M1i - l0max;

        if (L0>=0 && L1>=0 && M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          double s0_r = 0.0,s0_i = 0.0;
          double sr_r = 0.0,sr_i = 0.0;
          double st_r = 0.0,st_i = 0.0;
          double sp_r = 0.0,sp_i = 0.0;
          int k1,k0;

          for (k1=-L1; k1<=L1; k1++){

            double c0_r = 0.0,c0_i = 0.0;
            double cr_r = 0.0,cr_i = 0.0;
            double ct_r = 0.0,ct_i = 0.0;
            double cp_r = 0.0,cp_i = 0.0;

            for (k0=-L0; k0<=L0; k0++){
              const size_t uidx = (size_t)c2r_off[L0] + (size_t)(L0+M0)*(2*L0+1) + (L0+k0);
              const double u_r = c2r[2*uidx];
              const double u_i = -c2r[2*uidx+1];
              const size_t te = ((((size_t)pp2*cmb_max + ca)*nM + (k0+l0max))
                                 *cmb_max + cb)*nM + (k1+l0max);
              const double *tv = tmpnl + te*8;

              c0_r += u_r*tv[0] - u_i*tv[1];
              c0_i += u_r*tv[1] + u_i*tv[0];
              cr_r += u_r*tv[2] - u_i*tv[3];
              cr_i += u_r*tv[3] + u_i*tv[2];
              ct_r += u_r*tv[4] - u_i*tv[5];
              ct_i += u_r*tv[5] + u_i*tv[4];
              cp_r += u_r*tv[6] - u_i*tv[7];
              cp_i += u_r*tv[7] + u_i*tv[6];
            }

            {
              const size_t vidx = (size_t)c2r_off[L1] + (size_t)(L1+M1)*(2*L1+1) + (L1+k1);
              const double v_r = c2r[2*vidx];
              const double v_i = c2r[2*vidx+1];

              s0_r += (c0_r*v_r - c0_i*v_i);
              s0_i += (c0_r*v_i + c0_i*v_r);
              sr_r += (cr_r*v_r - cr_i*v_i);
              sr_i += (cr_r*v_i + cr_i*v_r);
              st_r += (ct_r*v_r - ct_i*v_i);
              st_i += (ct_r*v_i + ct_i*v_r);
              sp_r += (cp_r*v_r - cp_i*v_i);
              sp_i += (cp_r*v_i + cp_i*v_r);
            }
          }

          {
            const int num0 = cmb_num0[spe*cmb_max+ca];
            const int num1 = cmb_num0[spe*cmb_max+cb];
            const int row = num0 + L0 + M0;
            const int col = num1 + L1 + M1;
            const size_t obase = (size_t)pp2*4*(size_t)mn_max;
            const size_t oidx = (size_t)row*tno_max + col;
            const double r = pr_r[pp2];
            const double siT = pr_siT[pp2];
            const double coT = pr_coT[pp2];
            const double siP = pr_siP[pp2];
            const double coP = pr_coP[pp2];

            outbuf[obase + oidx] = 8.0*s0_r;

            if (!pr_h0[pp2]){

              if (fabs(siT)<10e-14){
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(siT*coP*sr_r + coT*coP/r*st_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(siT*siP*sr_r + coT*siP/r*st_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(coT*sr_r - siT/r*st_r);
              }
              else{
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(siT*coP*sr_r + coT*coP/r*st_r
                        - siP/siT/r*sp_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(siT*siP*sr_r + coT*siP/r*st_r
                        + coP/siT/r*sp_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(coT*sr_r - siT/r*st_r);
              }
            }
            else{
              outbuf[obase + (size_t)mn_max + oidx] = 0.0;
              outbuf[obase + 2*(size_t)mn_max + oidx] = 0.0;
              outbuf[obase + 3*(size_t)mn_max + oidx] = 0.0;
            }
          }
        }
      }

#pragma acc update self(outbuf[0:(size_t)np*4*(size_t)mn_max])

      /* host: scatter into HVNA2 or HVNA3 */

#pragma omp parallel for schedule(dynamic)
      for (pp=0; pp<np; pp++){
        int Mc_AN = OneD2Mc_AN[chunk_start+pp];
        int h_AN = OneD2h_AN[chunk_start+pp];
        int tno = g->sp_tno[pr_spe[pp]];
        int kk,i2,j2;

        for (kk=0; kk<=3; kk++){
          for (i2=0; i2<tno; i2++){
            for (j2=0; j2<tno; j2++){
              HVNA23[kk][Mc_AN][h_AN][i2][j2] =
                outbuf[((size_t)pp*4 + kk)*(size_t)mn_max
                       + (size_t)i2*g->tno_max + j2];
            }
          }
        }
      }

    } /* chunk_start */
  }

  free(outbuf);
  free(tmpnl);
  free(sumr0);
  free(sum0);
  free(sphbp);
  free(sphb);
  free(sh_tab);
  free(pr_h0);
  free(pr_vspe);
  free(pr_spe);
  free(pr_coP);
  free(pr_siP);
  free(pr_coT);
  free(pr_siT);
  free(pr_r);
}

double Set_ProExpn(double ****HVNA, Type_DS_VNA *****DS_VNA)
{
  /****************************************************
   evaluate matrix elements of neutral atom potential
   in the KB or Blochl separable form in momentum space.
  ****************************************************/
  static int firsttime=1;
  int L2,L3,L,i,j,Mc_AN,Gc_AN,h_AN,Cwan,i1,j1;
  int kk,Ls,n,j0,jg0,Mj_AN0,m,L1,GL,Mul1;
  int tno0,tno1,tno2,p,q,po1,po,Original_Mc_AN;
  int L0,Mul0,spe,Gh_AN,Hwan,fan,jg,k,kg,wakg,kl;
  int size_SumNL0,size_TmpNL;
  int Mc_AN2,Gc_AN2,num,size1,size2;
  int *Snd_DS_VNA_Size,*Rcv_DS_VNA_Size;
  double kmin,kmax,Sk,Dk,Normk,dmp;
  Type_DS_VNA *tmp_array;
  Type_DS_VNA *tmp_array2;
  double tmp0,tmp3,tmp4,tmp5,tmp10;
  double ****Bessel_Pro00;
  double ****Bessel_Pro01;
  double ene,sum,h,coe0,sj,sy,sjp,syp;
  double rcutA,rcutB,rcut;
  double TStime,TEtime;
  double stime,etime;
  double Stime_atom,Etime_atom;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  int *VNA_List;
  int *VNA_List2;
  int Num_RVNA;
  double **NLH;
  double time0,time1,time2,time3,time4,time5;
  dcomplex sum0,sum1,sum2;

  MPI_Status stat;
  MPI_Request request;

  int OneD_Nloop,*OneD2Mc_AN,*OneD2h_AN;
  int setpro_gpu = 0;
  int setpro_gpu2 = 0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  /****************************************************
                 allocation of arrays:
  ****************************************************/

  Num_RVNA = List_YOUSO[34]*(List_YOUSO[35] + 1);

  Snd_DS_VNA_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_DS_VNA_Size = (int*)malloc(sizeof(int)*numprocs);

  VNA_List  = (int*)malloc(sizeof(int)*(List_YOUSO[34]*(List_YOUSO[35] + 1)+2) );
  VNA_List2 = (int*)malloc(sizeof(int)*(List_YOUSO[34]*(List_YOUSO[35] + 1)+2) );

  Bessel_Pro00 = (double****)malloc(sizeof(double***)*SpeciesNum);
  for (spe=0; spe<SpeciesNum; spe++){
    Bessel_Pro00[spe] = (double***)malloc(sizeof(double**)*(Spe_MaxL_Basis[spe]+1));
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      Bessel_Pro00[spe][L0] = (double**)malloc(sizeof(double*)*Spe_Num_Basis[spe][L0]);
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
	Bessel_Pro00[spe][L0][Mul0] = (double*)malloc(sizeof(double)*GL_Mesh);
      }
    }
  }

  Bessel_Pro01 = (double****)malloc(sizeof(double***)*SpeciesNum);
  for (spe=0; spe<SpeciesNum; spe++){
    Bessel_Pro01[spe] = (double***)malloc(sizeof(double**)*(Spe_MaxL_Basis[spe]+1));
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      Bessel_Pro01[spe][L0] = (double**)malloc(sizeof(double*)*Spe_Num_Basis[spe][L0]);
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
        Bessel_Pro01[spe][L0][Mul0] = (double*)malloc(sizeof(double)*GL_Mesh);
      }
    }
  }

  size_TmpNL = (List_YOUSO[25]+1)*List_YOUSO[24]*
               (2*(List_YOUSO[25]+1)+1)*Num_RVNA*(2*List_YOUSO[30]+1);
  size_SumNL0 = (List_YOUSO[25]+1)*List_YOUSO[24]*(Num_RVNA+1);

  /* PrintMemory */
  if (firsttime) {
    PrintMemory("Set_ProExpn_VNA: SumNL0",sizeof(double)*size_SumNL0,NULL);
    PrintMemory("Set_ProExpn_VNA: SumNLr0",sizeof(double)*size_SumNL0,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpNL", sizeof(dcomplex)*size_TmpNL,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpNLr",sizeof(dcomplex)*size_TmpNL,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpNLt",sizeof(dcomplex)*size_TmpNL,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpNLp",sizeof(dcomplex)*size_TmpNL,NULL);
    firsttime=0;
  }

  if (measure_time){
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
  }

  /************************************************************
   Spe_Num_RVPS[Hwan]     ->  Num_RVNA = List_YOUSO[34]*(List_YOUSO[35]+1)
   Spe_VPS_List[Hwan][L]  ->  VNA_List[L] = 0,0,0,.., 1,1,1,.., 2,2,2,..,
   List_YOUSO[19]         ->  Num_RVNA
   List_YOUSO[30]         ->  List_YOUSO[35]
  *************************************************************/

  L = 0;
  for (i=0; i<=List_YOUSO[35]; i++){    /* max L */
    for (j=0; j<List_YOUSO[34]; j++){   /* # of radial projectors */
      VNA_List[L]  = i;
      VNA_List2[L] = j;
      L++;
    }
  }

  /************************************************************
   pre-calculate the time consuming part in the 'time1' part
  *************************************************************/

  kmin = Radial_kmin;
  kmax = PAO_Nkmax;
  Sk = kmax + kmin;
  Dk = kmax - kmin;

  for (spe=0; spe<SpeciesNum; spe++){

    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){

#pragma omp parallel shared(Mul0,L0,spe,Bessel_Pro01,Bessel_Pro00,GL_Weight,Dk,GL_NormK) private(i,Normk,tmp10)
	{

	  int OMPID,Nthrds,Nprocs;

	  /* get info. on OpenMP */

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (i=OMPID*GL_Mesh/Nthrds; i<(OMPID+1)*GL_Mesh/Nthrds; i++){

	    Normk = GL_NormK[i];
	    tmp10 = 0.50*Dk*GL_Weight[i]*Normk*Normk;

	    Bessel_Pro00[spe][L0][Mul0][i] = RF_BesselF(spe,L0,Mul0,Normk)*tmp10;
	    Bessel_Pro01[spe][L0][Mul0][i] = Bessel_Pro00[spe][L0][Mul0][i]*Normk;
	  }

#pragma omp flush(Bessel_Pro00,Bessel_Pro01)

	} /* #pragma omp parallel */

      }
    }
  }

  /************************************************************
    start the main calculation
  *************************************************************/

  /* one-dimensionalize the Mc_AN and h_AN loops */

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD_Nloop++;
    }
  }

  OneD2Mc_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));
  OneD2h_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD2Mc_AN[OneD_Nloop] = Mc_AN;
      OneD2h_AN[OneD_Nloop] = h_AN;
      OneD_Nloop++;
    }
  }

  /* batched device construction of DS_VNA */

  setpro_gpu2 = SetPro_DSVNA_GpuBegin(VNA_List, VNA_List2, Num_RVNA,
                                      Bessel_Pro00, Bessel_Pro01);
  if (setpro_gpu2){
    SetPro_DSVNA_GpuRun(DS_VNA, OneD_Nloop, OneD2Mc_AN, OneD2h_AN);
    SetPro_DSVNA_GpuEnd();
  }

  if (!setpro_gpu2){

#pragma omp parallel shared(time_per_atom,DS_VNA,Comp2Real,Spe_VNA_Bessel,Bessel_Pro01,Bessel_Pro00,GL_NormK,List_YOUSO,VNA_List,Num_RVNA,Spe_Num_Basis,Spe_MaxL_Basis,PAO_Nkmax,atv,Gxyz,ncn,natn,Spe_Atom_Cut1,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,time1,time2,time3)

  {
    int i,j,k,l,m,num0,num1;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN;
    int Rnh,Hwan,L0,Mul0,L,L1,M0,M1,LL,Mul1,Ls;
    int OMPID,Nthrds,Nprocs,Nloop;
    int Lmax,Lmax_Four_Int;
    double Stime_atom,Etime_atom;
    double rcutA,rcutB,rcut;
    double dx,dy,dz;
    double kmin,kmax,Sk,Dk;
    double gant,SH[2],dSHt[2],dSHp[2];
    double S_coordinate[3];
    double r,theta,phi;
    double Normk,tmp0,tmp1,tmp2;
    double siT,coT,siP,coP;
    double ***SumNL0;
    double ***SumNLr0;
    double *Bes00;
    double *Bes01;
    double tmp_SphB[30];
    double tmp_SphBp[30];
    double SphB[30][GL_Mesh];
    double SphBp[30][GL_Mesh];
    double stime,etime;

    dcomplex Ctmp1,Ctmp0,Ctmp2;
    dcomplex CsumNL0,CsumNLr,CsumNLt,CsumNLp;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1,Cpow;
    dcomplex *****TmpNL;
    dcomplex *****TmpNLr;
    dcomplex *****TmpNLt;
    dcomplex *****TmpNLp;
    dcomplex **CmatNL0;
    dcomplex **CmatNLr;
    dcomplex **CmatNLt;
    dcomplex **CmatNLp;

    /* allocation of arrays */

    TmpNL = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNL[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNL[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNL[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNL[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    TmpNLr = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLr[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLr[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLr[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNLr[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    TmpNLt = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLt[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLt[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLt[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNLt[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    TmpNLp = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLp[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLp[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLp[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNLp[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    SumNL0 = (double***)malloc(sizeof(double**)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      SumNL0[i] = (double**)malloc(sizeof(double*)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	SumNL0[i][j] = (double*)malloc(sizeof(double)*Num_RVNA);
      }
    }

    SumNLr0 = (double***)malloc(sizeof(double**)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      SumNLr0[i] = (double**)malloc(sizeof(double*)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	SumNLr0[i][j] = (double*)malloc(sizeof(double)*Num_RVNA);
      }
    }

    CmatNL0 = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNL0[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    CmatNLr = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLr[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    CmatNLt = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLt[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    CmatNLp = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLp[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    Bes00 = (double*)malloc(sizeof(double)*GL_Mesh);
    Bes01 = (double*)malloc(sizeof(double)*GL_Mesh);

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
      rcutA = Spe_Atom_Cut1[Cwan];

      /* set data on h_AN */

      Gh_AN = natn[Gc_AN][h_AN];
      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      rcutB = Spe_Atom_Cut1[Hwan];
      rcut = rcutA + rcutB;

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

      /****************************************************
         evaluate ovelap integrals <chi0|P> between PAOs
         and progectors of nonlocal VNA potentials.
      ****************************************************/
      /****************************************************
              \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3
      ****************************************************/

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
        for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
          for (L=0; L<Num_RVNA; L++){
            L1 = VNA_List[L];
            for (M0=-L0; M0<=L0; M0++){
	      for (M1=-L1; M1<=L1; M1++){
		TmpNL[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
		TmpNLr[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
		TmpNLt[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
		TmpNLp[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
	      }
	    }
	  }
	}
      }

      Lmax = -10;
      for (L=0; L<Num_RVNA; L++){
	if (Lmax<VNA_List[L]) Lmax = VNA_List[L];
      }
      if (Spe_MaxL_Basis[Cwan]<Lmax)
	Lmax_Four_Int = 2*Lmax;
      else
        Lmax_Four_Int = 2*Spe_MaxL_Basis[Cwan];

      /* calculate SphB and SphBp */
#ifdef kcomp
#else
#pragma forceinline recursive
#endif
      for (i=0; i<GL_Mesh; i++){
        Normk = GL_NormK[i];
        Spherical_Bessel2(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
        for(LL=0; LL<=Lmax_Four_Int; LL++){
          SphB[LL][i]  = tmp_SphB[LL];
          SphBp[LL][i] = tmp_SphBp[LL];
	}
      }

      /* LL loop */

      for(LL=0; LL<=Lmax_Four_Int; LL++){

	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	    for (L=0; L<Num_RVNA; L++){
	      SumNL0[L0][Mul0][L] = 0.0;
	      SumNLr0[L0][Mul0][L] = 0.0;
	    }
	  }
	}

        /* Gauss-Legendre quadrature */

        if (measure_time) dtime(&stime);

        for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

            for (i=0; i<GL_Mesh; i++){
              Bes00[i] = Bessel_Pro00[Cwan][L0][Mul0][i]*SphB[LL][i];
              Bes01[i] = Bessel_Pro01[Cwan][L0][Mul0][i]*SphBp[LL][i];
            }

            L = 0;
            for (L1=0; L1<=List_YOUSO[35]; L1++){
              for (Mul1=0; Mul1<List_YOUSO[34]; Mul1++){

                tmp1 = 0.0;
                tmp2 = 0.0;

                for (i=0; i<GL_Mesh; i++){
                  tmp1 += Bes00[i]*Spe_VNA_Bessel[Hwan][L1][Mul1][i];
                  tmp2 += Bes01[i]*Spe_VNA_Bessel[Hwan][L1][Mul1][i];
		}

		SumNL0[L0][Mul0][L]  = tmp1;
		SumNLr0[L0][Mul0][L] = tmp2;

                L++;
	      }
	    }
	  }
	}

        if (measure_time && OMPID==0){
          dtime(&etime);
          time1 += etime - stime;
	}

	/* derivatives of "on site" */

	if (h_AN==0){
	  for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	      for (L=0; L<Num_RVNA; L++){
		SumNLr0[L0][Mul0][L] = 0.0;
	      }
	    }
	  }
	}

	/****************************************************
         for "overlap",
	    sum_m 8*(-i)^{-L0+L1+l}*
	          C_{L0,-M0,L1,M1,l,m}*
	          \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,
	****************************************************/

        if (measure_time && OMPID==0) dtime(&stime);

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
	      for (L=0; L<Num_RVNA; L++){

		L1 = VNA_List[L];
		Ls = -L0 + L1 + LL;

	        if ( abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		  Cpow = Im_pow(-1,Ls);
		  CY1  = Cmul(Cpow,CY);
		  CYt1 = Cmul(Cpow,CYt);
		  CYp1 = Cmul(Cpow,CYp);

		  for (M0=-L0; M0<=L0; M0++){

		    M1 = M0 - m;

                    if (abs(M1)<=L1){

		      gant = Gaunt(L0,M0,L1,M1,LL,m);
		      tmp2 = gant*SumNL0[L0][Mul0][L];

		      /* S */

		      TmpNL[L0][Mul0][L0+M0][L][L1+M1].r += CY1.r*tmp2;
		      TmpNL[L0][Mul0][L0+M0][L][L1+M1].i += CY1.i*tmp2;

		      /* dS/dr */

		      tmp0 = gant*SumNLr0[L0][Mul0][L];

		      TmpNLr[L0][Mul0][L0+M0][L][L1+M1].r += CY1.r*tmp0;
		      TmpNLr[L0][Mul0][L0+M0][L][L1+M1].i += CY1.i*tmp0;

		      /* dS/dt */

		      TmpNLt[L0][Mul0][L0+M0][L][L1+M1].r += CYt1.r*tmp2;
		      TmpNLt[L0][Mul0][L0+M0][L][L1+M1].i += CYt1.i*tmp2;

		      /* dS/dp */

		      TmpNLp[L0][Mul0][L0+M0][L][L1+M1].r += CYp1.r*tmp2;
		      TmpNLp[L0][Mul0][L0+M0][L][L1+M1].i += CYp1.i*tmp2;

		    }
		  }
		}

	      }
	    }
	  }
	} /* m */

        if (measure_time && OMPID==0){
          dtime(&etime);
          time2 += etime - stime;
	}

      } /* LL */

      /****************************************************
                        Complex to Real
      ****************************************************/

      if (measure_time) dtime(&stime);

      num0 = 0;
      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

	  num1 = 0;
	  for (L=0; L<Num_RVNA; L++){
	    L1 = VNA_List[L];

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

		  CsumNL0.r += (CmatNL0[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNL0[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNL0.i += (CmatNL0[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNL0[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		  /* dS/dr */

		  CsumNLr.r += (CmatNLr[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNLr[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNLr.i += (CmatNLr[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNLr[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		  /* dS/dt */

		  CsumNLt.r += (CmatNLt[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNLt[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNLt.i += (CmatNLt[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNLt[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		  /* dS/dp */

		  CsumNLp.r += (CmatNLp[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNLp[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNLp.i += (CmatNLp[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNLp[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		}

		DS_VNA[0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*(Type_DS_VNA)CsumNL0.r;

		if (h_AN!=0){

		  if (fabs(siT)<10e-14){

		    DS_VNA[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r);

		    DS_VNA[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r);

		    DS_VNA[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(coT*CsumNLr.r - siT/r*CsumNLt.r);
		  }

		  else{

		    DS_VNA[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r
					- siP/siT/r*CsumNLp.r);

		    DS_VNA[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r
					+ coP/siT/r*CsumNLp.r);

		    DS_VNA[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(coT*CsumNLr.r - siT/r*CsumNLt.r);
		  }

		}

		else{
		  DS_VNA[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  DS_VNA[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  DS_VNA[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		}

	      }
	    }

	    num1 = num1 + 2*L1 + 1;
	  }

	  num0 = num0 + 2*L0 + 1;
	} /* M0 */
      } /* L0 */

      if (measure_time && OMPID==0){
	dtime(&etime);
	time3 += etime - stime;
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNL0[i]);
    }
    free(CmatNL0);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLr[i]);
    }
    free(CmatNLr);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLt[i]);
    }
    free(CmatNLt);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLp[i]);
    }
    free(CmatNLp);

    free(Bes00);
    free(Bes01);

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
	  for (l=0; l<Num_RVNA; l++){
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
	  for (l=0; l<Num_RVNA; l++){
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
	  for (l=0; l<Num_RVNA; l++){
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
	  for (l=0; l<Num_RVNA; l++){
	    free(TmpNL[i][j][k][l]);
	  }
	  free(TmpNL[i][j][k]);
	}
	free(TmpNL[i][j]);
      }
      free(TmpNL[i]);
    }
    free(TmpNL);

#pragma omp flush(DS_VNA)

  } /* #pragma omp parallel */

  } /* if (!setpro_gpu2) */

  /*******************************************************
   *******************************************************
     multiplying overlap integrals WITH COMMUNICATION

     MPI: communicate only for k=0
     DS_VNA
  *******************************************************
  *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time) dtime(&stime);

  /* batched device path for the HVNA traces */

  setpro_gpu = SetPro_HVNA_GpuBegin(DS_VNA, VNA_List, VNA_List2, Num_RVNA);

  /* allocation of array */

  NLH = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    NLH[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

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

	size1 = 0;
	n = F_Snd_Num_WK[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	  size1 += tno1*tno2;
	}

	Snd_DS_VNA_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_DS_VNA_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_DS_VNA_Size[IDR] = size2;
      }
      else{
	Rcv_DS_VNA_Size[IDR] = 0;
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

	size1 = Snd_DS_VNA_Size[IDS];

	/*
	printf("S myid=%2d ID=%2d IDS=%2d IDR=%2d  size1=%2d\n",myid,ID,IDS,IDR,size1);fflush(stdout);
	*/

	/* allocation of the array */

	tmp_array = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size1);

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
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
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

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){

	size2 = Rcv_DS_VNA_Size[IDR];

	/*
	printf("R myid=%2d ID=%2d IDS=%2d IDR=%2d  size2=%2d\n",myid,ID,IDS,IDR,size2);fflush(stdout);
	*/

	tmp_array2 = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size2);
	MPI_Recv(&tmp_array2[0], size2, MPI_Type_DS_VNA, IDR, tag, mpi_comm_level1, &stat);

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
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      DS_VNA[0][Matomnum+1][h_AN][i][j] = tmp_array2[num];
	      num++;
	    }
	  }
	}

	/* free tmp_array2 */
	free(tmp_array2);

	if (setpro_gpu){
	  SetPro_HVNA_GpuArchiveHalo(DS_VNA, Original_Mc_AN, Gc_AN);
	}

	/*****************************************
              multiplying overlap integrals
	*****************************************/

	if (!setpro_gpu){

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

	      for (k=0; k<=fan; k++){

		kg = natn[Gc_AN][k];
		wakg = WhatSpecies[kg];
		kl = RMI1[Mc_AN][j][k];

		if (0<=kl){
#pragma omp parallel for private(m, n, sum, L ,L1, GL, Mul1, ene, L2 ,tmp0, L3) default(shared) if(Spe_Total_NO[Cwan] > 8)
		  for (m=0; m<Spe_Total_NO[Cwan]; m++){
		    for (n=0; n<Spe_Total_NO[Hwan]; n++){

		      sum = 0.0;
		      L = 0;

		      for (L1=0; L1<Num_RVNA; L1++){

			GL = VNA_List[L1];
			Mul1 = VNA_List2[L1];

			ene = VNA_proj_ene[wakg][GL][Mul1];
			L2 = 2*VNA_List[L1];

			tmp0 = 0.0;

			for (L3=0; L3<=L2; L3++){

			  tmp0 += DS_VNA[0][Mc_AN][k][m][L]*DS_VNA[0][Matomnum+1][kl][n][L];

			  L++;
			}

			sum += ene*tmp0;

		      }

		      if (k==0)  NLH[m][n]  = sum;
		      else       NLH[m][n] += sum;

		    } /* n */
		  } /* m */
		} /* if (0<=kl)*/
	      } /* k */

	      /****************************************************
                            NLH to HVNA
	      ****************************************************/

	      dmp = dampingF(rcut,Dis[Gc_AN][j]);

	      for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
		for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){
		  HVNA[Mc_AN][j][i1][j1] = dmp*NLH[i1][j1];
		}
	      }

	    } /* if (po==1) */

	  } /* j */

	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Mc_AN */

	} /* if (!setpro_gpu) */

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

  /* freeing of arrays */

  for (i=0; i<List_YOUSO[7]; i++){
    free(NLH[i]);
  }
  free(NLH);

  if (measure_time){
    dtime(&etime);
    time4 += etime - stime;
  }

  /*******************************************************
   *******************************************************
    multiplying overlap integrals WITHOUT COMMUNICATION
  *******************************************************
  *******************************************************/

  if (measure_time) dtime(&stime);

  if (!setpro_gpu){

#pragma omp parallel shared(List_YOUSO,time_per_atom,HVNA,Dis,DS_VNA,VNA_proj_ene,VNA_List2,VNA_List,Num_RVNA,Spe_Total_NO,RMI1,F_G2M,natn,Spe_Atom_Cut1,FNAN,WhatSpecies,M2G,Matomnum)
  {

    int OMPID,Nthrds,Nprocs;
    int Mc_AN,Gc_AN,Cwan,fan,Mj_AN,Hwan;
    int L,L1,GL,Mul1,L2,L3,i1,j1;
    int k,kg,wakg,kl,i,j,jg,m,n;
    double **NLH;
    double rcutA,rcutB,rcut,sum,ene,tmp0,dmp;
    double Stime_atom,Etime_atom;

    /* allocation of array */

    NLH = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (i=0; i<List_YOUSO[7]; i++){
      NLH[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }

    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      rcutA = Spe_Atom_Cut1[Cwan];

      for (j=0; j<=fan; j++){

	jg = natn[Gc_AN][j];
	Mj_AN = F_G2M[jg];

	if (Mj_AN<=Matomnum){

	  Hwan = WhatSpecies[jg];
	  rcutB = Spe_Atom_Cut1[Hwan];
	  rcut = rcutA + rcutB;

	  for (k=0; k<=fan; k++){

	    kg = natn[Gc_AN][k];
	    wakg = WhatSpecies[kg];
	    kl = RMI1[Mc_AN][j][k];

	    if (0<=kl){

	      for (m=0; m<Spe_Total_NO[Cwan]; m++){
		for (n=0; n<Spe_Total_NO[Hwan]; n++){

		  sum = 0.0;
		  L = 0;

		  for (L1=0; L1<Num_RVNA; L1++){

		    GL = VNA_List[L1];
		    Mul1 = VNA_List2[L1];

		    ene = VNA_proj_ene[wakg][GL][Mul1];
		    L2 = 2*VNA_List[L1];

		    tmp0 = 0.0;

		    for (L3=0; L3<=L2; L3++){
		      tmp0 += DS_VNA[0][Mc_AN][k][m][L]*DS_VNA[0][Mj_AN][kl][n][L];
		      L++;
		    }

		    sum += ene*tmp0;

		  }

		  if (k==0)  NLH[m][n]  = sum;
		  else       NLH[m][n] += sum;

		}
	      }
	    }
	  } /* k */

	  /****************************************************
                            NLH to HVNA
	  ****************************************************/

	  dmp = dampingF(rcut,Dis[Gc_AN][j]);

	  for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
	    for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){
	      HVNA[Mc_AN][j][i1][j1] = dmp*NLH[i1][j1];
	    }
	  }

	} /* if (Mj_AN<=Matomnum) */
      } /* j */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<List_YOUSO[7]; i++){
      free(NLH[i]);
    }
    free(NLH);

#pragma omp flush(HVNA)

  } /* #pragma omp parallel */

  } /* if (!setpro_gpu) */

  if (setpro_gpu){
    SetPro_HVNA_GpuRun(HVNA);
    SetPro_HVNA_GpuEnd();
    setpro_gpu = 0;
  }

  if (measure_time){
    dtime(&etime);
    time5 += etime - stime;
  }

  if (measure_time){
    printf("Set_ProExpn_VNA myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f\n",
            myid,time1,time2,time3,time4,time5);fflush(stdout);
  }

  /****************************************************
                   freeing of arrays:
  ****************************************************/

  free(Snd_DS_VNA_Size);
  free(Rcv_DS_VNA_Size);

  free(VNA_List);
  free(VNA_List2);

  for (spe=0; spe<SpeciesNum; spe++){
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
        free(Bessel_Pro00[spe][L0][Mul0]);
      }
      free(Bessel_Pro00[spe][L0]);
    }
    free(Bessel_Pro00[spe]);
  }
  free(Bessel_Pro00);

  for (spe=0; spe<SpeciesNum; spe++){
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
        free(Bessel_Pro01[spe][L0][Mul0]);
      }
      free(Bessel_Pro01[spe][L0]);
    }
    free(Bessel_Pro01[spe]);
  }
  free(Bessel_Pro01);

  free(OneD2Mc_AN);
  free(OneD2h_AN);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}




double Set_VNA2(double ****HVNA, double *****HVNA2)
{
  /****************************************************
   Evaluate matrix elements of one-center (orbitals)
   but two-center integrals for neutral atom potentials
   in the momentum space.

   <Phi_{LM,L'M'}|VNA>
  ****************************************************/
  static int firsttime=1;
  int L2,L3,L,GL,i,j;
  int k,kl,h_AN,Gh_AN,Rnh,Ls,n;
  int tno0,tno1,tno2,i1,j1,p;
  int Cwan,Hwan,fan,jg,kg,wakg,Lmax;
  int size_TmpHNA;
  int Mc_AN,Gc_AN,Mj_AN,num,size1,size2;
  double time0;
  double tmp2,tmp3,tmp4,tmp5;
  double TStime,TEtime;
  int Num_RVNA;
  /* for OpenMP */
  int OneD_Nloop,*OneD2Mc_AN,*OneD2h_AN;

  dtime(&TStime);

  /****************************************************
  allocation of arrays:
  ****************************************************/

  size_TmpHNA = (List_YOUSO[25]+1)*List_YOUSO[24]*
                (2*(List_YOUSO[25]+1)+1)*
                (List_YOUSO[25]+1)*List_YOUSO[24]*
                (2*(List_YOUSO[25]+1)+1);

  /* PrintMemory */
  if (firsttime) {
    PrintMemory("Set_ProExpn_VNA: TmpHNA",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAr",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAt",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAp",sizeof(double)*size_TmpHNA,NULL);

    firsttime=0;
  }

  /* one-dimensionalize the Mc_AN and h_AN loops */

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD_Nloop++;
    }
  }

  OneD2Mc_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));
  OneD2h_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD2Mc_AN[OneD_Nloop] = Mc_AN;
      OneD2h_AN[OneD_Nloop] = h_AN;
      OneD_Nloop++;
    }
  }

  if (SetPro_gpu3.enabled){
    SetPro_VNA23_GpuRun(HVNA2, 0, OneD_Nloop, OneD2Mc_AN, OneD2h_AN);
  }

  if (!SetPro_gpu3.enabled){

#pragma omp parallel shared(List_YOUSO,time_per_atom,HVNA2,Comp2Real,GL_Weight,Spe_ProductRF_Bessel,Spe_CrudeVNA_Bessel,GL_NormK,Spe_Num_Basis,Spe_MaxL_Basis,PAO_Nkmax,atv,Gxyz,WhatSpecies,ncn,natn,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop)
  {
    int OMPID,Nthrds,Nprocs,Nloop;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN;
    int Rnh,Hwan,L0,Mul0,L1,Mul1,M0,M1,LL;
    int Lmax_Four_Int,i,j,k,l,m,num0,num1;
    double dx,dy,dz;
    double siT,coT,siP,coP;
    double Normk,kmin,kmax,Sk,Dk;
    double gant,r,theta,phi;
    double SH[2],dSHt[2],dSHp[2];
    double S_coordinate[3];
    double Stime_atom,Etime_atom;
    double sum,sumr,sj,sjp,tmp0,tmp1,tmp10;
    double **SphB,**SphBp;
    double *tmp_SphB,*tmp_SphBp;

    dcomplex Ctmp0,Ctmp1,Ctmp2,Cpow;
    dcomplex Csum,Csumr,Csumt,Csump;
    dcomplex CsumS0,CsumSr,CsumSt,CsumSp;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
    dcomplex **CmatS0;
    dcomplex **CmatSr;
    dcomplex **CmatSt;
    dcomplex **CmatSp;
    dcomplex ******TmpHNA;
    dcomplex ******TmpHNAr;
    dcomplex ******TmpHNAt;
    dcomplex ******TmpHNAp;

    /* allocation of arrays */

    TmpHNA = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNA[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNA[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNA[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNA[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNA[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAr = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAr[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAr[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAr[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAr[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAr[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAt = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAt[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAt[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAt[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAt[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAt[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAp = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAp[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAp[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAp[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAp[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAp[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    CmatS0 = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatS0[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSr = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSr[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSt = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSt[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSp = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSp[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

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

      /****************************************************
         evaluate ovelap integrals <Phi_{LM,L'M'}|VNA>
         between VNA and PAO.
      ****************************************************/
      /****************************************************
       \int VNA(k)*Spe_ProductRF_Bessel(k)*jl(k*R) k^2 dk^3
      ****************************************************/

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Cwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Cwan][L1]; Mul1++){
	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){
		  TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0);
		  TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0);
		  TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0);
		  TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0);
		}
	      }
	    }
	  }
	}
      }

      /* allocate SphB and SphBp */

      Lmax_Four_Int = 2*Spe_MaxL_Basis[Cwan];

      SphB = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
      for(LL=0; LL<(Lmax_Four_Int+3); LL++){
	SphB[LL] = (double*)malloc(sizeof(double)*GL_Mesh);
      }

      SphBp = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
      for(LL=0; LL<(Lmax_Four_Int+3); LL++){
	SphBp[LL] = (double*)malloc(sizeof(double)*GL_Mesh);
      }

      tmp_SphB  = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));
      tmp_SphBp = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));

      /* calculate SphB and SphBp */
#ifdef kcomp
#else
#pragma forceinline recursive
#endif
      for (i=0; i<GL_Mesh; i++){
	Normk = GL_NormK[i];
	Spherical_Bessel2(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
	for(LL=0; LL<=Lmax_Four_Int; LL++){
	  SphB[LL][i]  = tmp_SphB[LL];
	  SphBp[LL][i] = tmp_SphBp[LL];
	}
      }

      free(tmp_SphB);
      free(tmp_SphBp);

      /* loops for L0, Mul0, L1, and Mul1 */

      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Cwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Cwan][L1]; Mul1++){

	      if (L0<=L1){
		Lmax_Four_Int = 2*L1;

		/* sum over LL */

		for(LL=0; LL<=Lmax_Four_Int; LL++){

		  if (abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		    sum  = 0.0;
		    sumr = 0.0;

		    /* Gauss-Legendre quadrature */

		    for (i=0; i<GL_Mesh; i++){

		      Normk = GL_NormK[i];

		      sj  =  SphB[LL][i];
		      sjp = SphBp[LL][i];

		      tmp0 = Spe_CrudeVNA_Bessel[Hwan][i];
		      tmp1 = Spe_ProductRF_Bessel[Cwan][L0][Mul0][L1][Mul1][LL][i];
		      tmp10 = 0.50*Dk*tmp0*tmp1*GL_Weight[i]*Normk*Normk;

		      sum  += tmp10*sj;
		      sumr += tmp10*sjp*Normk;
		    }

		    /* sum over m */

		    for (M0=-L0; M0<=L0; M0++){
		      for (M1=-L1; M1<=L1; M1++){

			Csum = Complex(0.0,0.0);
			Csumr = Complex(0.0,0.0);
			Csumt = Complex(0.0,0.0);
			Csump = Complex(0.0,0.0);

			for(m=-LL; m<=LL; m++){

			  if ( (M1-m)==M0){

			    ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);
			    CY  = Complex(SH[0],SH[1]);
			    CYt = Complex(dSHt[0],dSHt[1]);
			    CYp = Complex(dSHp[0],dSHp[1]);

			    gant = pow(-1.0,(double)abs(m))*Gaunt(L0,M0,L1,M1,LL,-m);

			    /* S */

			    Ctmp2 = CRmul(CY,gant);
			    Csum = Cadd(Csum,Ctmp2);

			    /* dS/dt */

			    Ctmp2 = CRmul(CYt,gant);
			    Csumt = Cadd(Csumt,Ctmp2);

			    /* dS/dp */

			    Ctmp2 = CRmul(CYp,gant);
			    Csump = Cadd(Csump,Ctmp2);
			  }

			}

			/* S */
			Ctmp1 = CRmul(Csum,sum);
			TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1]
			  = Cadd(TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dr */
			Ctmp1 = CRmul(Csum,sumr);
			TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1]
			  = Cadd(TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dt */
			Ctmp1 = CRmul(Csumt,sum);
			TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1]
			  = Cadd(TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dp */
			Ctmp1 = CRmul(Csump,sum);
			TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1]
			  = Cadd(TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);
		      }
		    }
		  }
		} /* LL */
	      } /* if (L0<=L1) */
	    }
	  }
	}
      }

      /* free SphB and SphBp */

      for(LL=0; LL<(Lmax_Four_Int+3); LL++){
	free(SphB[LL]);
      }
      free(SphB);

      for(LL=0; LL<(Lmax_Four_Int+3); LL++){
	free(SphBp[LL]);
      }
      free(SphBp);

      /* copy the upper part to the lower part */

      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Cwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Cwan][L1]; Mul1++){
	      if (L0<=L1){
		for (M0=-L0; M0<=L0; M0++){
		  for (M1=-L1; M1<=L1; M1++){

		    TmpHNA[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1]);

		    TmpHNAr[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1]);

		    TmpHNAt[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1]);

		    TmpHNAp[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1]);

		  }
		}
	      }
	    }
	  }
	}
      }

      /****************************************************
 	                 Complex to Real
      ****************************************************/

      num0 = 0;
      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

	  num1 = 0;
	  for (L1=0; L1<=Spe_MaxL_Basis[Cwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Cwan][L1]; Mul1++){

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumS0 = Complex(0.0,0.0);
		  CsumSr = Complex(0.0,0.0);
		  CsumSt = Complex(0.0,0.0);
		  CsumSp = Complex(0.0,0.0);

		  for (k=-L0; k<=L0; k++){

		    Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

		    /* S */

		    Ctmp0 = TmpHNA[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumS0 = Cadd(CsumS0,Ctmp2);

		    /* dS/dr */

		    Ctmp0 = TmpHNAr[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSr = Cadd(CsumSr,Ctmp2);

		    /* dS/dt */

		    Ctmp0 = TmpHNAt[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSt = Cadd(CsumSt,Ctmp2);

		    /* dS/dp */

		    Ctmp0 = TmpHNAp[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSp = Cadd(CsumSp,Ctmp2);

		  }

		  CmatS0[L0+M0][L1+M1] = CsumS0;
		  CmatSr[L0+M0][L1+M1] = CsumSr;
		  CmatSt[L0+M0][L1+M1] = CsumSt;
		  CmatSp[L0+M0][L1+M1] = CsumSp;

		}
	      }

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumS0 = Complex(0.0,0.0);
		  CsumSr = Complex(0.0,0.0);
		  CsumSt = Complex(0.0,0.0);
		  CsumSp = Complex(0.0,0.0);

		  for (k=-L1; k<=L1; k++){

		    /* S */

		    Ctmp1 = Cmul(CmatS0[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumS0 = Cadd(CsumS0,Ctmp1);

		    /* dS/dr */

		    Ctmp1 = Cmul(CmatSr[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSr = Cadd(CsumSr,Ctmp1);

		    /* dS/dt */

		    Ctmp1 = Cmul(CmatSt[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSt = Cadd(CsumSt,Ctmp1);

		    /* dS/dp */

		    Ctmp1 = Cmul(CmatSp[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSp = Cadd(CsumSp,Ctmp1);

		  }

		  HVNA2[0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*CsumS0.r;

		  if (h_AN!=0){

		    if (fabs(siT)<10e-14){

		      HVNA2[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumSr.r + coT*coP/r*CsumSt.r);

		      HVNA2[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumSr.r + coT*siP/r*CsumSt.r);

		      HVNA2[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumSr.r - siT/r*CsumSt.r);
		    }

		    else{

		      HVNA2[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumSr.r + coT*coP/r*CsumSt.r
			     - siP/siT/r*CsumSp.r);

		      HVNA2[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumSr.r + coT*siP/r*CsumSt.r
			     + coP/siT/r*CsumSp.r);

		      HVNA2[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumSr.r - siT/r*CsumSt.r);
		    }
		  }
		  else{
		    HVNA2[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    HVNA2[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    HVNA2[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  }

		}
	      }

	      num1 = num1 + 2*L1 + 1;
	    }
	  }

	  num0 = num0 + 2*L0 + 1;
	}
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNA[i][j][k][l][m]);
	    }
	    free(TmpHNA[i][j][k][l]);
	  }
	  free(TmpHNA[i][j][k]);
	}
	free(TmpHNA[i][j]);
      }
      free(TmpHNA[i]);
    }
    free(TmpHNA);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAr[i][j][k][l][m]);
	    }
	    free(TmpHNAr[i][j][k][l]);
	  }
	  free(TmpHNAr[i][j][k]);
	}
	free(TmpHNAr[i][j]);
      }
      free(TmpHNAr[i]);
    }
    free(TmpHNAr);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAt[i][j][k][l][m]);
	    }
	    free(TmpHNAt[i][j][k][l]);
	  }
	  free(TmpHNAt[i][j][k]);
	}
	free(TmpHNAt[i][j]);
      }
      free(TmpHNAt[i]);
    }
    free(TmpHNAt);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAp[i][j][k][l][m]);
	    }
	    free(TmpHNAp[i][j][k][l]);
	  }
	  free(TmpHNAp[i][j][k]);
	}
	free(TmpHNAp[i][j]);
      }
      free(TmpHNAp[i]);
    }
    free(TmpHNAp);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatS0[i]);
    }
    free(CmatS0);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSr[i]);
    }
    free(CmatSr);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSt[i]);
    }
    free(CmatSt);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSp[i]);
    }
    free(CmatSp);

#pragma omp flush(HVNA2)

  } /* #pragma omp parallel */

  } /* if (!SetPro_gpu3.enabled) */

  /****************************************************
    HVNA[Mc_AN][0] = sum_{h_AN} HVNA2
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (i=0; i<Spe_Total_NO[Cwan]; i++){
      for (j=0; j<Spe_Total_NO[Cwan]; j++){
	HVNA[Mc_AN][0][i][j] = 0.0;
      }
    }

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      for (i=0; i<Spe_Total_NO[Cwan]; i++){
        for (j=0; j<Spe_Total_NO[Cwan]; j++){
          HVNA[Mc_AN][0][i][j] += HVNA2[0][Mc_AN][h_AN][i][j];
	}
      }
    }
  }

  /****************************************************
  freeing of arrays:
  ****************************************************/

  free(OneD2Mc_AN);
  free(OneD2h_AN);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}



double Set_VNA3(double *****HVNA3)
{
  /****************************************************
   Evaluate matrix elements of one-center (orbitals)
   but two-center integrals for neutral atom potentials
   in the momentum space.

   <VNA|Phi_{LM,L'M'}>
  ****************************************************/
  static int firsttime=0; /* due to the same as in Set_VNA2 */
  int L2,L3,L,GL,i,j;
  int k,kl,h_AN,Gh_AN,Rnh,Ls,n;
  int tno0,tno1,tno2,i1,j1,p;
  int Cwan,Hwan,fan,jg,kg,wakg,Lmax;
  int size_TmpHNA;
  int Mc_AN,Gc_AN,Mj_AN,num,size1,size2;
  double time0;
  double tmp2,tmp3,tmp4,tmp5;
  double TStime,TEtime;
  int Num_RVNA;

  /* for OpenMP */
  int OneD_Nloop,*OneD2Mc_AN,*OneD2h_AN;

  dtime(&TStime);

  /****************************************************
  allocation of arrays:
  ****************************************************/

  size_TmpHNA = (List_YOUSO[25]+1)*List_YOUSO[24]*
                (2*(List_YOUSO[25]+1)+1)*
                (List_YOUSO[25]+1)*List_YOUSO[24]*
                (2*(List_YOUSO[25]+1)+1);

  /* PrintMemory */
  if (firsttime) {
    PrintMemory("Set_ProExpn_VNA: TmpHNA",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAr",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAt",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAp",sizeof(double)*size_TmpHNA,NULL);

    firsttime=0;
  }

  /* one-dimensionalize the Mc_AN and h_AN loops */

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD_Nloop++;
    }
  }

  OneD2Mc_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));
  OneD2h_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD2Mc_AN[OneD_Nloop] = Mc_AN;
      OneD2h_AN[OneD_Nloop] = h_AN;
      OneD_Nloop++;
    }
  }

  if (SetPro_gpu3.enabled){
    SetPro_VNA23_GpuRun(HVNA3, 1, OneD_Nloop, OneD2Mc_AN, OneD2h_AN);
  }

  if (!SetPro_gpu3.enabled){

#pragma omp parallel shared(List_YOUSO,time_per_atom,HVNA3,Comp2Real,GL_Weight,Spe_ProductRF_Bessel,Spe_CrudeVNA_Bessel,GL_NormK,Spe_Num_Basis,Spe_MaxL_Basis,PAO_Nkmax,atv,Gxyz,WhatSpecies,ncn,natn,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop)
  {
    int OMPID,Nthrds,Nprocs,Nloop;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN;
    int Rnh,Hwan,L0,Mul0,L1,Mul1,M0,M1,LL;
    int Lmax_Four_Int,i,j,k,l,m,num0,num1;
    double dx,dy,dz;
    double siT,coT,siP,coP;
    double Normk,kmin,kmax,Sk,Dk;
    double gant,r,theta,phi;
    double SH[2],dSHt[2],dSHp[2];
    double S_coordinate[3];
    double Stime_atom,Etime_atom;
    double sum,sumr,sj,sjp,tmp0,tmp1,tmp10;
    double **SphB,**SphBp;
    double *tmp_SphB,*tmp_SphBp;

    dcomplex Ctmp0,Ctmp1,Ctmp2,Cpow;
    dcomplex Csum,Csumr,Csumt,Csump;
    dcomplex CsumS0,CsumSr,CsumSt,CsumSp;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
    dcomplex **CmatS0;
    dcomplex **CmatSr;
    dcomplex **CmatSt;
    dcomplex **CmatSp;
    dcomplex ******TmpHNA;
    dcomplex ******TmpHNAr;
    dcomplex ******TmpHNAt;
    dcomplex ******TmpHNAp;

    /* allocation of arrays */

    TmpHNA = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNA[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNA[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNA[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNA[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNA[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAr = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAr[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAr[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAr[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAr[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAr[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAt = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAt[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAt[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAt[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAt[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAt[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAp = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAp[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAp[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAp[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAp[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAp[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    CmatS0 = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatS0[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSr = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSr[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSt = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSt[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSp = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSp[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

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

      dx = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rnh][1];
      dy = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rnh][2];
      dz = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rnh][3];

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

      /****************************************************
         evaluate ovelap integrals <Phi_{LM,L'M'}|VNA>
         between VNA and PAO.
      ****************************************************/
      /****************************************************
       \int VNA(k)*Spe_ProductRF_Bessel(k)*jl(k*R) k^2 dk^3
      ****************************************************/

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      for (L0=0; L0<=Spe_MaxL_Basis[Hwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Hwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){
	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){
		  TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0);
		  TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0);
		  TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0);
		  TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0);
		}
	      }
	    }
	  }
	}
      }

      /* allocate SphB and SphBp */

      Lmax_Four_Int = 2*Spe_MaxL_Basis[Hwan];

      SphB = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
      for(LL=0; LL<(Lmax_Four_Int+3); LL++){
	SphB[LL] = (double*)malloc(sizeof(double)*GL_Mesh);
      }

      SphBp = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
      for(LL=0; LL<(Lmax_Four_Int+3); LL++){
	SphBp[LL] = (double*)malloc(sizeof(double)*GL_Mesh);
      }

      tmp_SphB  = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));
      tmp_SphBp = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));

      /* calculate SphB and SphBp */
#ifdef kcomp
#else
#pragma forceinline recursive
#endif
      for (i=0; i<GL_Mesh; i++){
	Normk = GL_NormK[i];
	Spherical_Bessel2(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
	for(LL=0; LL<=Lmax_Four_Int; LL++){
	  SphB[LL][i]  = tmp_SphB[LL];
	  SphBp[LL][i] = tmp_SphBp[LL];
	}
      }

      free(tmp_SphB);
      free(tmp_SphBp);

      /* loops for L0, Mul0, L1, and Mul1 */

      for (L0=0; L0<=Spe_MaxL_Basis[Hwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Hwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){

	      if (L0<=L1){

		Lmax_Four_Int = 2*L1;

		/* sum over LL */

		for(LL=0; LL<=Lmax_Four_Int; LL++){

		  if (abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		    sum  = 0.0;
		    sumr = 0.0;

		    /* Gauss-Legendre quadrature */

		    for (i=0; i<GL_Mesh; i++){

		      Normk = GL_NormK[i];

		      sj  =  SphB[LL][i];
		      sjp = SphBp[LL][i];

		      tmp0 = Spe_CrudeVNA_Bessel[Cwan][i];
		      tmp1 = Spe_ProductRF_Bessel[Hwan][L0][Mul0][L1][Mul1][LL][i];
		      tmp10 = 0.50*Dk*tmp0*tmp1*GL_Weight[i]*Normk*Normk;

		      sum  += tmp10*sj;
		      sumr += tmp10*sjp*Normk;
		    }

		    /* sum over m */

		    for (M0=-L0; M0<=L0; M0++){
		      for (M1=-L1; M1<=L1; M1++){

			Csum = Complex(0.0,0.0);
			Csumr = Complex(0.0,0.0);
			Csumt = Complex(0.0,0.0);
			Csump = Complex(0.0,0.0);

			for(m=-LL; m<=LL; m++){

			  if ( (M1-m)==M0){

			    ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);
			    CY  = Complex(SH[0],SH[1]);
			    CYt = Complex(dSHt[0],dSHt[1]);
			    CYp = Complex(dSHp[0],dSHp[1]);

			    gant = pow(-1.0,(double)abs(m))*Gaunt(L0,M0,L1,M1,LL,-m);

			    /* S */

			    Ctmp2 = CRmul(CY,gant);
			    Csum = Cadd(Csum,Ctmp2);

			    /* dS/dt */

			    Ctmp2 = CRmul(CYt,gant);
			    Csumt = Cadd(Csumt,Ctmp2);

			    /* dS/dp */

			    Ctmp2 = CRmul(CYp,gant);
			    Csump = Cadd(Csump,Ctmp2);
			  }

			}

			/* S */
			Ctmp1 = CRmul(Csum,sum);
			TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1]
			  = Cadd(TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dr */
			Ctmp1 = CRmul(Csum,sumr);
			TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1]
			  = Cadd(TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dt */
			Ctmp1 = CRmul(Csumt,sum);
			TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1]
			  = Cadd(TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dp */
			Ctmp1 = CRmul(Csump,sum);
			TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1]
			  = Cadd(TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);
		      }
		    }
		  }
		} /* LL */
	      } /* if (L0<=L1) */
	    }
	  }
	}
      }

      /* free SphB and SphBp */

      for(LL=0; LL<(Lmax_Four_Int+3); LL++){
	free(SphB[LL]);
      }
      free(SphB);

      for(LL=0; LL<(Lmax_Four_Int+3); LL++){
	free(SphBp[LL]);
      }
      free(SphBp);

      /* copy the upper part to the lower part */

      for (L0=0; L0<=Spe_MaxL_Basis[Hwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Hwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){
	      if (L0<=L1){
		for (M0=-L0; M0<=L0; M0++){
		  for (M1=-L1; M1<=L1; M1++){

		    TmpHNA[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1]);

		    TmpHNAr[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1]);

		    TmpHNAt[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1]);

		    TmpHNAp[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1]);

		  }
		}
	      }
	    }
	  }
	}
      }

      /****************************************************
 	                 Complex to Real
      ****************************************************/

      num0 = 0;
      for (L0=0; L0<=Spe_MaxL_Basis[Hwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Hwan][L0]; Mul0++){

	  num1 = 0;
	  for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumS0 = Complex(0.0,0.0);
		  CsumSr = Complex(0.0,0.0);
		  CsumSt = Complex(0.0,0.0);
		  CsumSp = Complex(0.0,0.0);

		  for (k=-L0; k<=L0; k++){

		    Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

		    /* S */

		    Ctmp0 = TmpHNA[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumS0 = Cadd(CsumS0,Ctmp2);

		    /* dS/dr */

		    Ctmp0 = TmpHNAr[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSr = Cadd(CsumSr,Ctmp2);

		    /* dS/dt */

		    Ctmp0 = TmpHNAt[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSt = Cadd(CsumSt,Ctmp2);

		    /* dS/dp */

		    Ctmp0 = TmpHNAp[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSp = Cadd(CsumSp,Ctmp2);

		  }

		  CmatS0[L0+M0][L1+M1] = CsumS0;
		  CmatSr[L0+M0][L1+M1] = CsumSr;
		  CmatSt[L0+M0][L1+M1] = CsumSt;
		  CmatSp[L0+M0][L1+M1] = CsumSp;

		}
	      }

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumS0 = Complex(0.0,0.0);
		  CsumSr = Complex(0.0,0.0);
		  CsumSt = Complex(0.0,0.0);
		  CsumSp = Complex(0.0,0.0);

		  for (k=-L1; k<=L1; k++){

		    /* S */


		    Ctmp1 = Cmul(CmatS0[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumS0 = Cadd(CsumS0,Ctmp1);

		    /* dS/dr */

		    Ctmp1 = Cmul(CmatSr[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSr = Cadd(CsumSr,Ctmp1);

		    /* dS/dt */

		    Ctmp1 = Cmul(CmatSt[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSt = Cadd(CsumSt,Ctmp1);

		    /* dS/dp */

		    Ctmp1 = Cmul(CmatSp[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSp = Cadd(CsumSp,Ctmp1);

		  }

		  HVNA3[0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*CsumS0.r;

		  if (h_AN!=0){

		    if (fabs(siT)<10e-14){

		      HVNA3[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumSr.r + coT*coP/r*CsumSt.r);

		      HVNA3[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumSr.r + coT*siP/r*CsumSt.r);

		      HVNA3[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumSr.r - siT/r*CsumSt.r);
		    }

		    else{

		      HVNA3[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumSr.r + coT*coP/r*CsumSt.r
			     - siP/siT/r*CsumSp.r);

		      HVNA3[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumSr.r + coT*siP/r*CsumSt.r
			     + coP/siT/r*CsumSp.r);

		      HVNA3[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumSr.r - siT/r*CsumSt.r);
		    }
		  }
		  else{
		    HVNA3[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    HVNA3[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    HVNA3[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  }

		}
	      }

	      num1 = num1 + 2*L1 + 1;
	    }
	  }

	  num0 = num0 + 2*L0 + 1;
	}
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNA[i][j][k][l][m]);
	    }
	    free(TmpHNA[i][j][k][l]);
	  }
	  free(TmpHNA[i][j][k]);
	}
	free(TmpHNA[i][j]);
      }
      free(TmpHNA[i]);
    }
    free(TmpHNA);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAr[i][j][k][l][m]);
	    }
	    free(TmpHNAr[i][j][k][l]);
	  }
	  free(TmpHNAr[i][j][k]);
	}
	free(TmpHNAr[i][j]);
      }
      free(TmpHNAr[i]);
    }
    free(TmpHNAr);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAt[i][j][k][l][m]);
	    }
	    free(TmpHNAt[i][j][k][l]);
	  }
	  free(TmpHNAt[i][j][k]);
	}
	free(TmpHNAt[i][j]);
      }
      free(TmpHNAt[i]);
    }
    free(TmpHNAt);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAp[i][j][k][l][m]);
	    }
	    free(TmpHNAp[i][j][k][l]);
	  }
	  free(TmpHNAp[i][j][k]);
	}
	free(TmpHNAp[i][j]);
      }
      free(TmpHNAp[i]);
    }
    free(TmpHNAp);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatS0[i]);
    }
    free(CmatS0);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSr[i]);
    }
    free(CmatSr);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSt[i]);
    }
    free(CmatSt);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSp[i]);
    }
    free(CmatSp);

#pragma omp flush(HVNA3)

  } /* #pragma omp parallel */

  } /* if (!SetPro_gpu3.enabled) */

  /****************************************************
  freeing of arrays:
  ****************************************************/

  free(OneD2Mc_AN);
  free(OneD2h_AN);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}


#define xmin  0.0
#define asize_lmax 30

#ifdef kcomp
static void Spherical_Bessel2( double x, int lmax, double *sb, double *dsb )
#else
inline void Spherical_Bessel2( double x, int lmax, double *sb, double *dsb )
#endif
{
  int m,n,nmax;
  double tsb[asize_lmax+10];
  double invx,vsb0,vsb1,vsb2,vsbi;
  double j0,j1,j0p,j1p,sf,tmp,si,co,ix,ix2;

  if (x<0.0){
    printf("minus x is invalid for Spherical_Bessel\n");
    exit(0);
  }

  /* find an appropriate nmax */

  nmax = lmax + 3*x + 20;
  if (nmax<100) nmax = 100;

  if (asize_lmax<(lmax+1)){
    printf("asize_lmax should be larger than %d in Spherical_Bessel.c\n",lmax+1);
    exit(0);
  }

  /* if x is larger than xmin */

  if ( xmin < x ){

    invx = 1.0/x;

    /* initial values */

    vsb0 = 0.0;
    vsb1 = 1.0e-14;

    /* downward recurrence from nmax-2 to lmax+2 */

    for ( n=nmax-1; (lmax+2)<n; n-- ){

      vsb2 = (2.0*n + 1.0)*invx*vsb1 - vsb0;

      if (1.0e+250<vsb2){
        tmp = 1.0/vsb2;
        vsb2 *= tmp;
        vsb1 *= tmp;
      }

      vsbi = vsb0;
      vsb0 = vsb1;
      vsb1 = vsb2;
    }

    /* downward recurrence from lmax+1 to 0 */

    n = lmax + 3;
    tsb[n-1] = vsb1;
    tsb[n  ] = vsb0;
    tsb[n+1] = vsbi;

    tmp = tsb[n-1];
    tsb[n-1] /= tmp;
    tsb[n  ] /= tmp;

    for ( n=lmax+2; 0<n; n-- ){

      tsb[n-1] = (2.0*n + 1.0)*invx*tsb[n] - tsb[n+1];

      if (1.0e+250<tsb[n-1]){
        tmp = tsb[n-1];
        for (m=n-1; m<=lmax+1; m++){
          tsb[m] /= tmp;
        }
      }
    }

    /* normalization */

    si = sin(x);
    co = cos(x);
    ix = 1.0/x;
    ix2 = ix*ix;
    j0 = si*ix;
    j1 = si*ix*ix - co*ix;

    if (fabs(tsb[1])<fabs(tsb[0])) sf = j0/tsb[0];
    else                           sf = j1/tsb[1];

    /* tsb to sb */

    for ( n=0; n<=lmax+1; n++ ){
      sb[n] = tsb[n]*sf;
    }

    /* derivative of sb */

    dsb[0] = co*ix - si*ix*ix;
    for ( n=1; n<=lmax; n++ ){
      dsb[n] = ( (double)n*sb[n-1] - (double)(n+1.0)*sb[n+1] )/(2.0*(double)n + 1.0);
    }

  }

  /* if x is smaller than xmin */

  else {

    /* sb */

    for ( n=0; n<=lmax; n++ ){
      sb[n] = 0.0;
    }
    sb[0] = 1.0;

    /* derivative of sb */

    dsb[0] = 0.0;
    for ( n=1; n<=lmax; n++ ){
      dsb[n] = ( (double)n*sb[n-1] - (double)(n+1.0)*sb[n+1] )/(2.0*(double)n + 1.0);
    }
  }
}
