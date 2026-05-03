/**********************************************************************
  Set_OLP_Kin.c:

     Set_OLP_Kin.c is a subroutine to calculate the overlap matrix
     and the matrix for the kinetic operator in momentum space.

  Log of Set_OLP_Kin.c:

     15/Oct./2002  Released by T.Ozaki
     25/Nov./2014  Memory allocation modified by A.M. Ito (AITUNE)

***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SETOLPKIN_CALLOC(count, size, label) SetOLPKin_Calloc((count), (size), (label), __LINE__)

static void SetOLPKin_Abort(const char *msg, const char *label, int line)
{
    int rank = MYID_MPI_COMM_WORLD;

    if (!omp_in_parallel()) {
        MPI_Comm_rank(mpi_comm_level1, &rank);
    }

    fprintf(stderr, "Set_OLP_Kin.c:%d: %s", line, msg);
    if (label != NULL) {
        fprintf(stderr, " (%s)", label);
    }
    fprintf(stderr, " on rank %d\n", rank);
    fflush(stderr);

    if (!omp_in_parallel()) {
        MPI_Abort(mpi_comm_level1, EXIT_FAILURE);
    }
    abort();
}

static size_t SetOLPKin_CheckedMul(size_t a, size_t b, const char *label, int line)
{
    if (a != 0 && ((size_t)-1) / a < b) {
        SetOLPKin_Abort("allocation size overflow", label, line);
    }
    return a * b;
}

static size_t SetOLPKin_DimToSize(int value, const char *label, int line)
{
    if (value <= 0) {
        SetOLPKin_Abort("non-positive allocation dimension", label, line);
    }
    return (size_t)value;
}

static void *SetOLPKin_Calloc(size_t count, size_t size, const char *label, int line)
{
    void *ptr;

    (void)SetOLPKin_CheckedMul(count, size, label, line);
    if (count == 0 || size == 0) {
        count = 1;
        size  = 1;
    }

    ptr = calloc(count, size);
    if (ptr == NULL) {
        SetOLPKin_Abort("calloc failed", label, line);
    }

    return ptr;
}

static long int SetOLPKin_SizeToLong(size_t size, const char *label, int line)
{
    if ((size_t)LONG_MAX < size) {
        SetOLPKin_Abort("memory accounting size overflow", label, line);
    }
    return (long int)size;
}

static void SetOLPKin_PrintMemory(char *name, size_t count, size_t item_size)
{
    size_t bytes = SetOLPKin_CheckedMul(count, item_size, name, __LINE__);
    PrintMemory(name, SetOLPKin_SizeToLong(bytes, name, __LINE__), NULL);
}

static void SetOLPKin_ValidateSpecies(int species, const char *label, int line)
{
    int L;
    int max_l;

    if (species < 0 || SpeciesNum <= species) {
        SetOLPKin_Abort("species index is out of range", label, line);
    }

    max_l = Spe_MaxL_Basis[species];
    if (max_l < 0 || List_YOUSO[25] < max_l) {
        SetOLPKin_Abort("basis angular momentum exceeds allocated bounds", label, line);
    }

    if ((INT_MAX - 3) / 2 < max_l) {
        SetOLPKin_Abort("basis angular momentum is too large", label, line);
    }

    for (L = 0; L <= max_l; L++) {
        if (Spe_Num_Basis[species][L] < 0 || List_YOUSO[24] < Spe_Num_Basis[species][L]) {
            SetOLPKin_Abort("basis multiplicity exceeds allocated bounds", label, line);
        }
    }
}

static void SetOLPKin_ValidateSetup(void)
{
    int species;

    if (SpeciesNum <= 0) {
        SetOLPKin_Abort("SpeciesNum must be positive", "SpeciesNum", __LINE__);
    }
    if (List_YOUSO[24] <= 0) {
        SetOLPKin_Abort("basis multiplicity dimension must be positive", "List_YOUSO[24]", __LINE__);
    }
    if (List_YOUSO[25] < 0 || (INT_MAX - 3) / 2 < List_YOUSO[25] ||
        (INT_MAX - 1) / 4 < List_YOUSO[25]) {
        SetOLPKin_Abort("basis angular momentum dimension is invalid", "List_YOUSO[25]", __LINE__);
    }
    if (OneD_Grid <= 0 || OneD_Grid == INT_MAX) {
        SetOLPKin_Abort("1DFFT.NumGridR must be positive", "OneD_Grid", __LINE__);
    }
    if (!isfinite(Radial_kmin) || !isfinite(PAO_Nkmax) || PAO_Nkmax <= Radial_kmin) {
        SetOLPKin_Abort("invalid radial k range", "Radial_kmin/PAO_Nkmax", __LINE__);
    }

    for (species = 0; species < SpeciesNum; species++) {
        SetOLPKin_ValidateSpecies(species, "species setup", __LINE__);
    }
}

static uint64_t SetOLPKin_HashStep(uint64_t hash, int value)
{
    hash ^= (uint64_t)(uint32_t)value;
    hash *= 1099511628211ULL;
    return hash;
}

static double SetOLPKin_Stabilizer(int Gc_AN, int Gh_AN, int Rnh, int row, int col)
{
    uint64_t hash = 1469598103934665603ULL;
    double   unit;

    hash = SetOLPKin_HashStep(hash, Gc_AN);
    hash = SetOLPKin_HashStep(hash, Gh_AN);
    hash = SetOLPKin_HashStep(hash, Rnh);
    hash = SetOLPKin_HashStep(hash, row);
    hash = SetOLPKin_HashStep(hash, col);

    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    hash *= 0xc4ceb9fe1a85ec53ULL;
    hash ^= hash >> 33;

    unit = (double)(hash >> 11) * (1.0 / 9007199254740992.0);
    return (unit - 0.5) * 1.0e-13;
}

static inline dcomplex SetOLPKin_ImPowMinusOne(int Ls)
{
    if (Ls % 2 == 0) {
        if (Ls % 4 == 0) {
            return Complex(1.0, 0.0);
        }
        return Complex(-1.0, 0.0);
    }

    if ((Ls + 1) % 4 == 0) {
        return Complex(0.0, 1.0);
    }
    return Complex(0.0, -1.0);
}

static inline size_t SetOLPKin_GauntIndex(int L0, int M0, int L1, int M1, int l, int m, int l_dim, int m_dim,
                                          int gaunt_l_count, int gaunt_m_dim, int gaunt_m_offset)
{
    return ((((((size_t)L0 * (size_t)m_dim + (size_t)(L0 + M0)) * (size_t)l_dim + (size_t)L1) *
                   (size_t)m_dim +
               (size_t)(L1 + M1)) *
                  (size_t)gaunt_l_count +
              (size_t)l) *
                 (size_t)gaunt_m_dim +
             (size_t)(m + gaunt_m_offset));
}

static double *SetOLPKin_BuildGauntCache(int max_l, int l_dim, int m_dim, int gaunt_l_count, int gaunt_m_dim,
                                         int gaunt_m_offset, size_t *cache_size)
{
    int     L0, M0, L1, M1, l, m;
    size_t  size;
    double *cache;

    size = SetOLPKin_CheckedMul((size_t)l_dim, (size_t)m_dim, "Gaunt cache", __LINE__);
    size = SetOLPKin_CheckedMul(size, (size_t)l_dim, "Gaunt cache", __LINE__);
    size = SetOLPKin_CheckedMul(size, (size_t)m_dim, "Gaunt cache", __LINE__);
    size = SetOLPKin_CheckedMul(size, (size_t)gaunt_l_count, "Gaunt cache", __LINE__);
    size = SetOLPKin_CheckedMul(size, (size_t)gaunt_m_dim, "Gaunt cache", __LINE__);

    cache = (double *)SETOLPKIN_CALLOC(size, sizeof(double), "Gaunt cache");

    for (L0 = 0; L0 <= max_l; L0++) {
        for (M0 = -L0; M0 <= L0; M0++) {
            for (L1 = 0; L1 <= max_l; L1++) {
                for (l = 0; l <= 2 * max_l; l++) {
                    if (abs(L1 - l) <= L0 && L0 <= (L1 + l)) {
                        for (m = -l; m <= l; m++) {
                            M1 = M0 - m;
                            if (abs(M1) <= L1) {
                                cache[SetOLPKin_GauntIndex(L0, M0, L1, M1, l, m, l_dim, m_dim, gaunt_l_count,
                                                           gaunt_m_dim, gaunt_m_offset)] =
                                    Gaunt(L0, M0, L1, M1, l, m);
                            }
                        }
                    }
                }
            }
        }
    }

    if (cache_size != NULL) {
        *cache_size = size;
    }

    return cache;
}

static inline size_t SetOLPKin_RFBesselIndex(int species, int L, int Mul, int i, int l_dim, int mul_dim, int grid_dim)
{
    return (((size_t)species * (size_t)l_dim + (size_t)L) * (size_t)mul_dim + (size_t)Mul) * (size_t)grid_dim +
           (size_t)i;
}

static double *SetOLPKin_BuildRFBesselCache(int l_dim, int mul_dim, int grid_dim, size_t *cache_size)
{
    int     species, L, Mul, i;
    size_t  size;
    double  h, Normk;
    double *cache;

    size = SetOLPKin_CheckedMul((size_t)SpeciesNum, (size_t)l_dim, "RF_Bessel cache", __LINE__);
    size = SetOLPKin_CheckedMul(size, (size_t)mul_dim, "RF_Bessel cache", __LINE__);
    size = SetOLPKin_CheckedMul(size, (size_t)grid_dim, "RF_Bessel cache", __LINE__);

    cache = (double *)SETOLPKIN_CALLOC(size, sizeof(double), "RF_Bessel cache");
    h     = (PAO_Nkmax - Radial_kmin) / (double)(grid_dim - 1);

    for (i = 0; i < grid_dim; i++) {
        Normk = Radial_kmin + (double)i * h;
        for (species = 0; species < SpeciesNum; species++) {
            for (L = 0; L <= Spe_MaxL_Basis[species]; L++) {
                for (Mul = 0; Mul < Spe_Num_Basis[species][L]; Mul++) {
                    cache[SetOLPKin_RFBesselIndex(species, L, Mul, i, l_dim, mul_dim, grid_dim)] =
                        RF_BesselF(species, L, Mul, Normk);
                }
            }
        }
    }

    if (cache_size != NULL) {
        *cache_size = size;
    }

    return cache;
}

#ifdef kcomp
dcomplex ****** Allocate6D_dcomplex(int size_1, int size_2, int size_3, int size_4, int size_5, int size_6);
double ****     Allocate4D_double(int size_1, int size_2, int size_3, int size_4);
dcomplex **     Allocate2D_dcomplex(int size_1, int size_2);
double **       Allocate2D_double(int size_1, int size_2);
void            Free6D_dcomplex(dcomplex ****** buffer);
void            Free4D_double(double **** buffer);
void            Free2D_dcomplex(dcomplex ** buffer);
void            Free2D_double(double ** buffer);
#else
static inline dcomplex ****** Allocate6D_dcomplex(int size_1, int size_2, int size_3, int size_4, int size_5,
                                                  int size_6);
static inline double ****     Allocate4D_double(int size_1, int size_2, int size_3, int size_4);
static inline dcomplex **     Allocate2D_dcomplex(int size_1, int size_2);
static inline double **       Allocate2D_double(int size_1, int size_2);
void                          Free6D_dcomplex(dcomplex ****** buffer);
void                          Free4D_double(double **** buffer);
void                          Free2D_dcomplex(dcomplex ** buffer);
void                          Free2D_double(double ** buffer);
#endif

double Set_OLP_Kin(double ***** OLP, double ***** H0)
{
    /****************************************************
          Evaluate overlap and kinetic integrals
                 in the momentum space
  ****************************************************/
    static int     firsttime = 1;
    static double *Gaunt_cache = NULL;
    static int     Gaunt_cache_max_l = -1;
    static size_t  Gaunt_cache_size = 0;
    static double *RF_Bessel_cache = NULL;
    static int     RF_Bessel_cache_species_num = -1;
    static int     RF_Bessel_cache_l_dim = -1;
    static int     RF_Bessel_cache_mul_dim = -1;
    static int     RF_Bessel_cache_grid_dim = -1;
    static double  RF_Bessel_cache_kmin = 0.0;
    static double  RF_Bessel_cache_kmax = 0.0;
    static size_t  RF_Bessel_cache_size = 0;
    int            l_dim, mul_dim, m_dim, grid_dim;
    int            gaunt_l_count, gaunt_m_dim, gaunt_m_offset;
    size_t         size_SumS0, size_TmpOLP;
    double     time0;
    double     TStime, TEtime;
    int        numprocs, myid;
    int        Mc_AN, Gc_AN, h_AN;
    int        OneD_Nloop;
    int *      OneD2Mc_AN, *OneD2h_AN;

    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    dtime(&TStime);

    /****************************************************
   MPI_Barrier
  ****************************************************/

    MPI_Barrier(mpi_comm_level1);

    SetOLPKin_ValidateSetup();

    l_dim   = List_YOUSO[25] + 1;
    mul_dim = List_YOUSO[24];
    m_dim   = 2 * l_dim + 1;
    grid_dim = OneD_Grid + 1;

    gaunt_l_count = 2 * List_YOUSO[25] + 1;
    gaunt_m_dim = 4 * List_YOUSO[25] + 1;
    gaunt_m_offset = 2 * List_YOUSO[25];

    if (Gaunt_cache == NULL || Gaunt_cache_max_l != List_YOUSO[25]) {
        free(Gaunt_cache);
        Gaunt_cache = SetOLPKin_BuildGauntCache(List_YOUSO[25], l_dim, m_dim, gaunt_l_count, gaunt_m_dim,
                                                gaunt_m_offset, &Gaunt_cache_size);
        Gaunt_cache_max_l = List_YOUSO[25];
    }

    if (RF_Bessel_cache == NULL || RF_Bessel_cache_species_num != SpeciesNum || RF_Bessel_cache_l_dim != l_dim ||
        RF_Bessel_cache_mul_dim != mul_dim || RF_Bessel_cache_grid_dim != grid_dim ||
        RF_Bessel_cache_kmin != Radial_kmin || RF_Bessel_cache_kmax != PAO_Nkmax) {
        free(RF_Bessel_cache);
        RF_Bessel_cache = SetOLPKin_BuildRFBesselCache(l_dim, mul_dim, grid_dim, &RF_Bessel_cache_size);
        RF_Bessel_cache_species_num = SpeciesNum;
        RF_Bessel_cache_l_dim = l_dim;
        RF_Bessel_cache_mul_dim = mul_dim;
        RF_Bessel_cache_grid_dim = grid_dim;
        RF_Bessel_cache_kmin = Radial_kmin;
        RF_Bessel_cache_kmax = PAO_Nkmax;
    }

    /* PrintMemory */

    if (firsttime) {

        size_SumS0 = SetOLPKin_CheckedMul((size_t)l_dim, (size_t)mul_dim, "SumS0", __LINE__);
        size_SumS0 = SetOLPKin_CheckedMul(size_SumS0, (size_t)l_dim, "SumS0", __LINE__);
        size_SumS0 = SetOLPKin_CheckedMul(size_SumS0, (size_t)mul_dim, "SumS0", __LINE__);

        size_TmpOLP = SetOLPKin_CheckedMul((size_t)l_dim, (size_t)mul_dim, "TmpOLP", __LINE__);
        size_TmpOLP = SetOLPKin_CheckedMul(size_TmpOLP, (size_t)m_dim, "TmpOLP", __LINE__);
        size_TmpOLP = SetOLPKin_CheckedMul(size_TmpOLP, (size_t)l_dim, "TmpOLP", __LINE__);
        size_TmpOLP = SetOLPKin_CheckedMul(size_TmpOLP, (size_t)mul_dim, "TmpOLP", __LINE__);
        size_TmpOLP = SetOLPKin_CheckedMul(size_TmpOLP, (size_t)m_dim, "TmpOLP", __LINE__);

        SetOLPKin_PrintMemory("Set_OLP_Kin: SumS0", size_SumS0, sizeof(double));
        SetOLPKin_PrintMemory("Set_OLP_Kin: SumK0", size_SumS0, sizeof(double));
        SetOLPKin_PrintMemory("Set_OLP_Kin: SumSr0", size_SumS0, sizeof(double));
        SetOLPKin_PrintMemory("Set_OLP_Kin: SumKr0", size_SumS0, sizeof(double));
        SetOLPKin_PrintMemory("Set_OLP_Kin: TmpOLP", size_TmpOLP, sizeof(dcomplex));
        SetOLPKin_PrintMemory("Set_OLP_Kin: TmpOLPr", size_TmpOLP, sizeof(dcomplex));
        SetOLPKin_PrintMemory("Set_OLP_Kin: TmpOLPt", size_TmpOLP, sizeof(dcomplex));
        SetOLPKin_PrintMemory("Set_OLP_Kin: TmpOLPp", size_TmpOLP, sizeof(dcomplex));
        SetOLPKin_PrintMemory("Set_OLP_Kin: TmpKin", size_TmpOLP, sizeof(dcomplex));
        SetOLPKin_PrintMemory("Set_OLP_Kin: TmpKinr", size_TmpOLP, sizeof(dcomplex));
        SetOLPKin_PrintMemory("Set_OLP_Kin: TmpKint", size_TmpOLP, sizeof(dcomplex));
        SetOLPKin_PrintMemory("Set_OLP_Kin: TmpKinp", size_TmpOLP, sizeof(dcomplex));
        SetOLPKin_PrintMemory("Set_OLP_Kin: Gaunt cache", Gaunt_cache_size, sizeof(double));
        SetOLPKin_PrintMemory("Set_OLP_Kin: RF_Bessel cache", RF_Bessel_cache_size, sizeof(double));
        firsttime = 0;
    }

    /* one-dimensionalize the Mc_AN and h_AN loops */

    OneD_Nloop = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        if (FNAN[Gc_AN] < 0 || FNAN[Gc_AN] == INT_MAX) {
            SetOLPKin_Abort("invalid FNAN value", "FNAN", __LINE__);
        }
        if (INT_MAX - (FNAN[Gc_AN] + 1) < OneD_Nloop) {
            SetOLPKin_Abort("one-dimensional loop count overflow", "OneD_Nloop", __LINE__);
        }
        OneD_Nloop += FNAN[Gc_AN] + 1;
    }

    OneD2Mc_AN = (int *)SETOLPKIN_CALLOC((size_t)OneD_Nloop + 1, sizeof(int), "OneD2Mc_AN");
    OneD2h_AN  = (int *)SETOLPKIN_CALLOC((size_t)OneD_Nloop + 1, sizeof(int), "OneD2h_AN");

    OneD_Nloop = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            OneD2Mc_AN[OneD_Nloop] = Mc_AN;
            OneD2h_AN[OneD_Nloop]  = h_AN;
            OneD_Nloop++;
        }
    }

    /* OpenMP */

#pragma omp parallel
    {

        int Nloop;
        int NloopStart, NloopEnd;
        int OMPID, Nthrds;
        int Mc_AN, h_AN, Gc_AN, Cwan;
        int Gh_AN, Rnh, Hwan;
        int Ls, L0, Mul0, L1, Mul1, M0, M1;
        int Lmax_Four_Int;
        int i, j, k, l, m;
        int num0, num1;

        double      Stime_atom, Etime_atom;
        double      dx, dy, dz;
        double      S_coordinate[3];
        double      theta, phi, h;
        double      Bessel_Pro0, Bessel_Pro1;
        double      tmp0, tmp1, tmp2, tmp3, tmp4;
        double      siT, coT, siP, coP;
        double      kmin, kmax, Sk, Dk, r;
        double      sj, sjp, coe0, coe1;
        double      Normk, Normk2;
        double      gant, SH[2], dSHt[2], dSHp[2];
        double **   SphB, **SphBp;
        double *    tmp_SphB, *tmp_SphBp;
        const double *RF_Bessel0, *RF_Bessel1;
        double **** SumS0;
        double **** SumK0;
        double **** SumSr0;
        double **** SumKr0;

        dcomplex        CsumS_Lx, CsumS_Ly, CsumS_Lz;
        dcomplex        CsumS0, CsumSr, CsumSt, CsumSp;
        dcomplex        CsumK0, CsumKr, CsumKt, CsumKp;
        dcomplex        Ctmp0, Ctmp1, Ctmp2, Cpow;
        dcomplex        CY, CYt, CYp, CY1, CYt1, CYp1;
        dcomplex ****** TmpOLP;
        dcomplex ****** TmpOLPr;
        dcomplex ****** TmpOLPt;
        dcomplex ****** TmpOLPp;
        dcomplex ****** TmpKin;
        dcomplex ****** TmpKinr;
        dcomplex ****** TmpKint;
        dcomplex ****** TmpKinp;

        dcomplex ** CmatS0;
        dcomplex ** CmatSr;
        dcomplex ** CmatSt;
        dcomplex ** CmatSp;
        dcomplex ** CmatK0;
        dcomplex ** CmatKr;
        dcomplex ** CmatKt;
        dcomplex ** CmatKp;

        /****************************************************************
                          allocation of arrays:
    ****************************************************************/

        /* get info. on OpenMP */

        OMPID  = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        NloopStart = (int)(((size_t)OMPID * (size_t)OneD_Nloop) / (size_t)Nthrds);
        NloopEnd   = (int)((((size_t)OMPID + 1) * (size_t)OneD_Nloop) / (size_t)Nthrds);

        TmpOLP  = NULL;
        TmpOLPr = NULL;
        TmpOLPt = NULL;
        TmpOLPp = NULL;
        TmpKin  = NULL;
        TmpKinr = NULL;
        TmpKint = NULL;
        TmpKinp = NULL;
        SphB    = NULL;
        SphBp   = NULL;
        tmp_SphB  = NULL;
        tmp_SphBp = NULL;
        SumS0   = NULL;
        SumK0   = NULL;
        SumSr0  = NULL;
        SumKr0  = NULL;
        CmatS0  = NULL;
        CmatSr  = NULL;
        CmatSt  = NULL;
        CmatSp  = NULL;
        CmatK0  = NULL;
        CmatKr  = NULL;
        CmatKt  = NULL;
        CmatKp  = NULL;

        if (NloopStart < NloopEnd) {
            TmpOLP  = Allocate6D_dcomplex(l_dim, mul_dim, m_dim, l_dim, mul_dim, m_dim);
            TmpOLPr = Allocate6D_dcomplex(l_dim, mul_dim, m_dim, l_dim, mul_dim, m_dim);
            TmpOLPt = Allocate6D_dcomplex(l_dim, mul_dim, m_dim, l_dim, mul_dim, m_dim);
            TmpOLPp = Allocate6D_dcomplex(l_dim, mul_dim, m_dim, l_dim, mul_dim, m_dim);
            TmpKin  = Allocate6D_dcomplex(l_dim, mul_dim, m_dim, l_dim, mul_dim, m_dim);
            TmpKinr = Allocate6D_dcomplex(l_dim, mul_dim, m_dim, l_dim, mul_dim, m_dim);
            TmpKint = Allocate6D_dcomplex(l_dim, mul_dim, m_dim, l_dim, mul_dim, m_dim);
            TmpKinp = Allocate6D_dcomplex(l_dim, mul_dim, m_dim, l_dim, mul_dim, m_dim);

            SumS0  = Allocate4D_double(l_dim, mul_dim, l_dim, mul_dim);
            SumK0  = Allocate4D_double(l_dim, mul_dim, l_dim, mul_dim);
            SumSr0 = Allocate4D_double(l_dim, mul_dim, l_dim, mul_dim);
            SumKr0 = Allocate4D_double(l_dim, mul_dim, l_dim, mul_dim);

            CmatS0 = Allocate2D_dcomplex(m_dim, m_dim);
            CmatSr = Allocate2D_dcomplex(m_dim, m_dim);
            CmatSt = Allocate2D_dcomplex(m_dim, m_dim);
            CmatSp = Allocate2D_dcomplex(m_dim, m_dim);
            CmatK0 = Allocate2D_dcomplex(m_dim, m_dim);
            CmatKr = Allocate2D_dcomplex(m_dim, m_dim);
            CmatKt = Allocate2D_dcomplex(m_dim, m_dim);
            CmatKp = Allocate2D_dcomplex(m_dim, m_dim);

            SphB  = Allocate2D_double(m_dim, grid_dim);
            SphBp = Allocate2D_double(m_dim, grid_dim);
            tmp_SphB  = (double *)SETOLPKIN_CALLOC((size_t)m_dim, sizeof(double), "tmp_SphB");
            tmp_SphBp = (double *)SETOLPKIN_CALLOC((size_t)m_dim, sizeof(double), "tmp_SphBp");
        }

        /* one-dimensionalized loop */

        for (Nloop = NloopStart; Nloop < NloopEnd; Nloop++) {

            dtime(&Stime_atom);

            /* get Mc_AN and h_AN */

            Mc_AN = OneD2Mc_AN[Nloop];
            h_AN  = OneD2h_AN[Nloop];

            /* set data on Mc_AN */

            Gc_AN = M2G[Mc_AN];
            Cwan  = WhatSpecies[Gc_AN];
            SetOLPKin_ValidateSpecies(Cwan, "central atom species", __LINE__);

            /* set data on h_AN */

            Gh_AN = natn[Gc_AN][h_AN];
            Rnh   = ncn[Gc_AN][h_AN];
            Hwan  = WhatSpecies[Gh_AN];
            SetOLPKin_ValidateSpecies(Hwan, "neighbor atom species", __LINE__);

            dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
            dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
            dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];

            xyz2spherical(dx, dy, dz, 0.0, 0.0, 0.0, S_coordinate);
            r     = S_coordinate[0];
            theta = S_coordinate[1];
            phi   = S_coordinate[2];

            /* for empty atoms or finite elemens basis */
            if (r < 1.0e-10)
                r = 1.0e-10;

            /* precalculation of sin and cos */

            siT = sin(theta);
            coT = cos(theta);
            siP = sin(phi);
            coP = cos(phi);

            /****************************************************
          Overlap and the derivative
              \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,
              \int RL(k)*RL'(k)*j'l(k*R) k^3 dk^3

          Kinetic 
              \int RL(k)*RL'(k)*jl(k*R) k^4 dk^3, 
              \int RL(k)*RL'(k)*j'l(k*R) k^5 dk^3 
      ****************************************************/

            kmin = Radial_kmin;
            kmax = PAO_Nkmax;
            Sk   = kmax + kmin;
            Dk   = kmax - kmin;

            for (L0 = 0; L0 <= Spe_MaxL_Basis[Cwan]; L0++) {
                for (Mul0 = 0; Mul0 < Spe_Num_Basis[Cwan][L0]; Mul0++) {
                    for (L1 = 0; L1 <= Spe_MaxL_Basis[Hwan]; L1++) {
                        for (Mul1 = 0; Mul1 < Spe_Num_Basis[Hwan][L1]; Mul1++) {
                            for (M0 = -L0; M0 <= L0; M0++) {
                                for (M1 = -L1; M1 <= L1; M1++) {

                                    TmpOLP[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1]  = Complex(0.0, 0.0);
                                    TmpOLPr[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] = Complex(0.0, 0.0);
                                    TmpOLPt[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] = Complex(0.0, 0.0);
                                    TmpOLPp[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] = Complex(0.0, 0.0);

                                    TmpKin[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1]  = Complex(0.0, 0.0);
                                    TmpKinr[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] = Complex(0.0, 0.0);
                                    TmpKint[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] = Complex(0.0, 0.0);
                                    TmpKinp[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] = Complex(0.0, 0.0);
                                }
                            }
                        }
                    }
                }
            }

            if (Spe_MaxL_Basis[Cwan] < Spe_MaxL_Basis[Hwan])
                Lmax_Four_Int = 2 * Spe_MaxL_Basis[Hwan];
            else
                Lmax_Four_Int = 2 * Spe_MaxL_Basis[Cwan];

            /* calculate SphB/SphBp and cache radial basis values */

            h = (kmax - kmin) / (double)OneD_Grid;

            for (i = 0; i <= OneD_Grid; i++) {
                Normk = kmin + (double)i * h;

                Spherical_Bessel(Normk * r, Lmax_Four_Int, tmp_SphB, tmp_SphBp);
                for (l = 0; l <= Lmax_Four_Int; l++) {
                    SphB[l][i]  = tmp_SphB[l];
                    SphBp[l][i] = tmp_SphBp[l];
                }
            }

            /* l loop */

            for (l = 0; l <= Lmax_Four_Int; l++) {

                for (L0 = 0; L0 <= Spe_MaxL_Basis[Cwan]; L0++) {
                    for (Mul0 = 0; Mul0 < Spe_Num_Basis[Cwan][L0]; Mul0++) {
                        for (L1 = 0; L1 <= Spe_MaxL_Basis[Hwan]; L1++) {
                            for (Mul1 = 0; Mul1 < Spe_Num_Basis[Hwan][L1]; Mul1++) {
                                SumS0[L0][Mul0][L1][Mul1]  = 0.0;
                                SumK0[L0][Mul0][L1][Mul1]  = 0.0;
                                SumSr0[L0][Mul0][L1][Mul1] = 0.0;
                                SumKr0[L0][Mul0][L1][Mul1] = 0.0;
                            }
                        }
                    }
                }

                for (i = 0; i <= OneD_Grid; i++) {

                    if (i == 0 || i == OneD_Grid)
                        coe0 = 0.50;
                    else
                        coe0 = 1.00;

                    Normk  = kmin + (double)i * h;
                    Normk2 = Normk * Normk;

                    sj  = SphB[l][i];
                    sjp = SphBp[l][i];

                    for (L0 = 0; L0 <= Spe_MaxL_Basis[Cwan]; L0++) {
                        for (Mul0 = 0; Mul0 < Spe_Num_Basis[Cwan][L0]; Mul0++) {

                            RF_Bessel0 = RF_Bessel_cache + SetOLPKin_RFBesselIndex(Cwan, L0, Mul0, 0, l_dim, mul_dim,
                                                                                   grid_dim);
                            Bessel_Pro0 = RF_Bessel0[i];

                            tmp0 = coe0 * h * Normk2 * Bessel_Pro0;
                            tmp1 = tmp0 * sj;
                            tmp2 = tmp0 * Normk * sjp;

                            for (L1 = 0; L1 <= Spe_MaxL_Basis[Hwan]; L1++) {
                                if (!(abs(L1 - l) <= L0 && L0 <= (L1 + l))) {
                                    continue;
                                }
                                for (Mul1 = 0; Mul1 < Spe_Num_Basis[Hwan][L1]; Mul1++) {

                                    RF_Bessel1 = RF_Bessel_cache +
                                                 SetOLPKin_RFBesselIndex(Hwan, L1, Mul1, 0, l_dim, mul_dim, grid_dim);
                                    Bessel_Pro1 = RF_Bessel1[i];

                                    tmp3 = tmp1 * Bessel_Pro1;
                                    tmp4 = tmp2 * Bessel_Pro1;

                                    SumS0[L0][Mul0][L1][Mul1] += tmp3;
                                    SumK0[L0][Mul0][L1][Mul1] += tmp3 * Normk2;

                                    SumSr0[L0][Mul0][L1][Mul1] += tmp4;
                                    SumKr0[L0][Mul0][L1][Mul1] += tmp4 * Normk2;
                                }
                            }
                        }
                    }
                }

                if (h_AN == 0) {
                    for (L0 = 0; L0 <= Spe_MaxL_Basis[Cwan]; L0++) {
                        for (Mul0 = 0; Mul0 < Spe_Num_Basis[Cwan][L0]; Mul0++) {
                            for (L1 = 0; L1 <= Spe_MaxL_Basis[Hwan]; L1++) {
                                for (Mul1 = 0; Mul1 < Spe_Num_Basis[Hwan][L1]; Mul1++) {
                                    SumSr0[L0][Mul0][L1][Mul1] = 0.0;
                                    SumKr0[L0][Mul0][L1][Mul1] = 0.0;
                                }
                            }
                        }
                    }
                }

                /****************************************************
          For overlap and the derivative,
          sum_m 8*(-i)^{-L0+L1+1}*
                C_{L0,-M0,L1,M1,l,m}*Y_{lm}
                \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,

          For kinetic,
          sum_m 4*(-i)^{-L0+L1+1}*
                C_{L0,-M0,L1,M1,l,m}*
                \int RL(k)*RL'(k)*jl(k*R) k^4 dk^3,
        ****************************************************/

                for (m = -l; m <= l; m++) {

                    ComplexSH(l, m, theta, phi, SH, dSHt, dSHp);
                    SH[1]   = -SH[1];
                    dSHt[1] = -dSHt[1];
                    dSHp[1] = -dSHp[1];

                    CY  = Complex(SH[0], SH[1]);
                    CYt = Complex(dSHt[0], dSHt[1]);
                    CYp = Complex(dSHp[0], dSHp[1]);

                    for (L0 = 0; L0 <= Spe_MaxL_Basis[Cwan]; L0++) {
                        for (Mul0 = 0; Mul0 < Spe_Num_Basis[Cwan][L0]; Mul0++) {
                            for (L1 = 0; L1 <= Spe_MaxL_Basis[Hwan]; L1++) {

                                Ls = -L0 + L1 + l;

                                if (abs(L1 - l) <= L0 && L0 <= (L1 + l)) {

                                    Cpow = SetOLPKin_ImPowMinusOne(Ls);
                                    CY1  = Cmul(Cpow, CY);
                                    CYt1 = Cmul(Cpow, CYt);
                                    CYp1 = Cmul(Cpow, CYp);

                                    for (Mul1 = 0; Mul1 < Spe_Num_Basis[Hwan][L1]; Mul1++) {

                                        for (M0 = -L0; M0 <= L0; M0++) {

                                            M1 = M0 - m;

                                            if (abs(M1) <= L1) {

                                                gant = Gaunt_cache[SetOLPKin_GauntIndex(L0, M0, L1, M1, l, m, l_dim,
                                                                                        m_dim, gaunt_l_count,
                                                                                        gaunt_m_dim, gaunt_m_offset)];

                                                /* S */

                                                tmp0  = gant * SumS0[L0][Mul0][L1][Mul1];
                                                Ctmp2 = CRmul(CY1, tmp0);
                                                TmpOLP[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] =
                                                    Cadd(TmpOLP[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1], Ctmp2);

                                                /* dS/dr */

                                                tmp0  = gant * SumSr0[L0][Mul0][L1][Mul1];
                                                Ctmp2 = CRmul(CY1, tmp0);
                                                TmpOLPr[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] =
                                                    Cadd(TmpOLPr[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1], Ctmp2);

                                                /* dS/dt */

                                                tmp0  = gant * SumS0[L0][Mul0][L1][Mul1];
                                                Ctmp2 = CRmul(CYt1, tmp0);
                                                TmpOLPt[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] =
                                                    Cadd(TmpOLPt[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1], Ctmp2);

                                                /* dS/dp */

                                                tmp0  = gant * SumS0[L0][Mul0][L1][Mul1];
                                                Ctmp2 = CRmul(CYp1, tmp0);
                                                TmpOLPp[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] =
                                                    Cadd(TmpOLPp[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1], Ctmp2);

                                                /* K */

                                                tmp0  = gant * SumK0[L0][Mul0][L1][Mul1];
                                                Ctmp2 = CRmul(CY1, tmp0);
                                                TmpKin[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] =
                                                    Cadd(TmpKin[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1], Ctmp2);

                                                /* dK/dr */

                                                tmp0  = gant * SumKr0[L0][Mul0][L1][Mul1];
                                                Ctmp2 = CRmul(CY1, tmp0);
                                                TmpKinr[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] =
                                                    Cadd(TmpKinr[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1], Ctmp2);

                                                /* dK/dt */

                                                tmp0  = gant * SumK0[L0][Mul0][L1][Mul1];
                                                Ctmp2 = CRmul(CYt1, tmp0);
                                                TmpKint[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] =
                                                    Cadd(TmpKint[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1], Ctmp2);

                                                /* dK/dp */

                                                tmp0  = gant * SumK0[L0][Mul0][L1][Mul1];
                                                Ctmp2 = CRmul(CYp1, tmp0);
                                                TmpKinp[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1] =
                                                    Cadd(TmpKinp[L0][Mul0][L0 + M0][L1][Mul1][L1 + M1], Ctmp2);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } /* l */

            /****************************************************
                         Complex to Real
      ****************************************************/

            num0 = 0;
            for (L0 = 0; L0 <= Spe_MaxL_Basis[Cwan]; L0++) {
                for (Mul0 = 0; Mul0 < Spe_Num_Basis[Cwan][L0]; Mul0++) {

                    num1 = 0;
                    for (L1 = 0; L1 <= Spe_MaxL_Basis[Hwan]; L1++) {
                        for (Mul1 = 0; Mul1 < Spe_Num_Basis[Hwan][L1]; Mul1++) {

                            for (M0 = -L0; M0 <= L0; M0++) {
                                for (M1 = -L1; M1 <= L1; M1++) {

                                    CsumS0 = Complex(0.0, 0.0);
                                    CsumSr = Complex(0.0, 0.0);
                                    CsumSt = Complex(0.0, 0.0);
                                    CsumSp = Complex(0.0, 0.0);

                                    CsumK0 = Complex(0.0, 0.0);
                                    CsumKr = Complex(0.0, 0.0);
                                    CsumKt = Complex(0.0, 0.0);
                                    CsumKp = Complex(0.0, 0.0);

                                    for (k = -L0; k <= L0; k++) {

                                        Ctmp1 = Conjg(Comp2Real[L0][L0 + M0][L0 + k]);

                                        /* S */

                                        Ctmp0  = TmpOLP[L0][Mul0][L0 + k][L1][Mul1][L1 + M1];
                                        Ctmp2  = Cmul(Ctmp1, Ctmp0);
                                        CsumS0 = Cadd(CsumS0, Ctmp2);

                                        /* dS/dr */

                                        Ctmp0  = TmpOLPr[L0][Mul0][L0 + k][L1][Mul1][L1 + M1];
                                        Ctmp2  = Cmul(Ctmp1, Ctmp0);
                                        CsumSr = Cadd(CsumSr, Ctmp2);

                                        /* dS/dt */

                                        Ctmp0  = TmpOLPt[L0][Mul0][L0 + k][L1][Mul1][L1 + M1];
                                        Ctmp2  = Cmul(Ctmp1, Ctmp0);
                                        CsumSt = Cadd(CsumSt, Ctmp2);

                                        /* dS/dp */

                                        Ctmp0  = TmpOLPp[L0][Mul0][L0 + k][L1][Mul1][L1 + M1];
                                        Ctmp2  = Cmul(Ctmp1, Ctmp0);
                                        CsumSp = Cadd(CsumSp, Ctmp2);

                                        /* K */

                                        Ctmp0  = TmpKin[L0][Mul0][L0 + k][L1][Mul1][L1 + M1];
                                        Ctmp2  = Cmul(Ctmp1, Ctmp0);
                                        CsumK0 = Cadd(CsumK0, Ctmp2);

                                        /* dK/dr */

                                        Ctmp0  = TmpKinr[L0][Mul0][L0 + k][L1][Mul1][L1 + M1];
                                        Ctmp2  = Cmul(Ctmp1, Ctmp0);
                                        CsumKr = Cadd(CsumKr, Ctmp2);

                                        /* dK/dt */

                                        Ctmp0  = TmpKint[L0][Mul0][L0 + k][L1][Mul1][L1 + M1];
                                        Ctmp2  = Cmul(Ctmp1, Ctmp0);
                                        CsumKt = Cadd(CsumKt, Ctmp2);

                                        /* dK/dp */

                                        Ctmp0  = TmpKinp[L0][Mul0][L0 + k][L1][Mul1][L1 + M1];
                                        Ctmp2  = Cmul(Ctmp1, Ctmp0);
                                        CsumKp = Cadd(CsumKp, Ctmp2);
                                    }

                                    CmatS0[L0 + M0][L1 + M1] = CsumS0;
                                    CmatSr[L0 + M0][L1 + M1] = CsumSr;
                                    CmatSt[L0 + M0][L1 + M1] = CsumSt;
                                    CmatSp[L0 + M0][L1 + M1] = CsumSp;

                                    CmatK0[L0 + M0][L1 + M1] = CsumK0;
                                    CmatKr[L0 + M0][L1 + M1] = CsumKr;
                                    CmatKt[L0 + M0][L1 + M1] = CsumKt;
                                    CmatKp[L0 + M0][L1 + M1] = CsumKp;
                                }
                            }

                            for (M0 = -L0; M0 <= L0; M0++) {
                                for (M1 = -L1; M1 <= L1; M1++) {

                                    CsumS_Lx = Complex(0.0, 0.0);
                                    CsumS_Ly = Complex(0.0, 0.0);
                                    CsumS_Lz = Complex(0.0, 0.0);

                                    CsumS0 = Complex(0.0, 0.0);
                                    CsumSr = Complex(0.0, 0.0);
                                    CsumSt = Complex(0.0, 0.0);
                                    CsumSp = Complex(0.0, 0.0);
                                    CsumK0 = Complex(0.0, 0.0);
                                    CsumKr = Complex(0.0, 0.0);
                                    CsumKt = Complex(0.0, 0.0);
                                    CsumKp = Complex(0.0, 0.0);

                                    for (k = -L1; k <= L1; k++) {

                                        /*** S_Lx ***/

                                        /*  Y k+1 */
                                        if (k < L1) {
                                            coe0    = sqrt((double)((L1 - k) * (L1 + k + 1)));
                                            Ctmp1   = Cmul(CmatS0[L0 + M0][L1 + k + 1], Comp2Real[L1][L1 + M1][L1 + k]);
                                            Ctmp1.r = 0.5 * coe0 * Ctmp1.r;
                                            Ctmp1.i = 0.5 * coe0 * Ctmp1.i;
                                            CsumS_Lx = Cadd(CsumS_Lx, Ctmp1);
                                        }

                                        /*  Y k-1 */
                                        if (-L1 < k) {
                                            coe1    = sqrt((double)((L1 + k) * (L1 - k + 1)));
                                            Ctmp1   = Cmul(CmatS0[L0 + M0][L1 + k - 1], Comp2Real[L1][L1 + M1][L1 + k]);
                                            Ctmp1.r = 0.5 * coe1 * Ctmp1.r;
                                            Ctmp1.i = 0.5 * coe1 * Ctmp1.i;
                                            CsumS_Lx = Cadd(CsumS_Lx, Ctmp1);
                                        }

                                        /*** S_Ly ***/

                                        /*  Y k+1 */

                                        if (k < L1) {
                                            Ctmp1   = Cmul(CmatS0[L0 + M0][L1 + k + 1], Comp2Real[L1][L1 + M1][L1 + k]);
                                            Ctmp2.r = 0.5 * coe0 * Ctmp1.i;
                                            Ctmp2.i = -0.5 * coe0 * Ctmp1.r;
                                            CsumS_Ly = Cadd(CsumS_Ly, Ctmp2);
                                        }

                                        /*  Y k-1 */

                                        if (-L1 < k) {
                                            Ctmp1   = Cmul(CmatS0[L0 + M0][L1 + k - 1], Comp2Real[L1][L1 + M1][L1 + k]);
                                            Ctmp2.r = -0.5 * coe1 * Ctmp1.i;
                                            Ctmp2.i = 0.5 * coe1 * Ctmp1.r;
                                            CsumS_Ly = Cadd(CsumS_Ly, Ctmp2);
                                        }

                                        /*** S_Lz ***/

                                        Ctmp1   = Cmul(CmatS0[L0 + M0][L1 + k], Comp2Real[L1][L1 + M1][L1 + k]);
                                        Ctmp1.r = (double)k * Ctmp1.r;
                                        ;
                                        Ctmp1.i  = (double)k * Ctmp1.i;
                                        CsumS_Lz = Cadd(CsumS_Lz, Ctmp1);

                                        /* S */

                                        Ctmp1  = Cmul(CmatS0[L0 + M0][L1 + k], Comp2Real[L1][L1 + M1][L1 + k]);
                                        CsumS0 = Cadd(CsumS0, Ctmp1);

                                        /* dS/dr */

                                        Ctmp1  = Cmul(CmatSr[L0 + M0][L1 + k], Comp2Real[L1][L1 + M1][L1 + k]);
                                        CsumSr = Cadd(CsumSr, Ctmp1);

                                        /* dS/dt */

                                        Ctmp1  = Cmul(CmatSt[L0 + M0][L1 + k], Comp2Real[L1][L1 + M1][L1 + k]);
                                        CsumSt = Cadd(CsumSt, Ctmp1);

                                        /* dS/dp */

                                        Ctmp1  = Cmul(CmatSp[L0 + M0][L1 + k], Comp2Real[L1][L1 + M1][L1 + k]);
                                        CsumSp = Cadd(CsumSp, Ctmp1);

                                        /* K */

                                        Ctmp1  = Cmul(CmatK0[L0 + M0][L1 + k], Comp2Real[L1][L1 + M1][L1 + k]);
                                        CsumK0 = Cadd(CsumK0, Ctmp1);

                                        /* dK/dr */

                                        Ctmp1  = Cmul(CmatKr[L0 + M0][L1 + k], Comp2Real[L1][L1 + M1][L1 + k]);
                                        CsumKr = Cadd(CsumKr, Ctmp1);

                                        /* dK/dt */

                                        Ctmp1  = Cmul(CmatKt[L0 + M0][L1 + k], Comp2Real[L1][L1 + M1][L1 + k]);
                                        CsumKt = Cadd(CsumKt, Ctmp1);

                                        /* dK/dp */

                                        Ctmp1  = Cmul(CmatKp[L0 + M0][L1 + k], Comp2Real[L1][L1 + M1][L1 + k]);
                                        CsumKp = Cadd(CsumKp, Ctmp1);
                                    }

                                    OLP_L[0][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] = 8.0 * CsumS_Lx.i;
                                    OLP_L[1][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] = 8.0 * CsumS_Ly.i;
                                    OLP_L[2][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] = 8.0 * CsumS_Lz.i;

                                    /* add a small value for stabilization of eigenvalue routine */

                                    OLP[0][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                        8.0 * CsumS0.r +
                                        SetOLPKin_Stabilizer(Gc_AN, Gh_AN, Rnh, num0 + L0 + M0, num1 + L1 + M1);
                                    H0[0][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] = 4.0 * CsumK0.r;

                                    if (h_AN != 0) {

                                        if (fabs(siT) < 10e-14) {

                                            OLP[1][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -8.0 * (siT * coP * CsumSr.r + coT * coP / r * CsumSt.r);

                                            OLP[2][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -8.0 * (siT * siP * CsumSr.r + coT * siP / r * CsumSt.r);

                                            OLP[3][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -8.0 * (coT * CsumSr.r - siT / r * CsumSt.r);

                                            H0[1][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -4.0 * (siT * coP * CsumKr.r + coT * coP / r * CsumKt.r);

                                            H0[2][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -4.0 * (siT * siP * CsumKr.r + coT * siP / r * CsumKt.r);

                                            H0[3][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -4.0 * (coT * CsumKr.r - siT / r * CsumKt.r);

                                        }

                                        else {

                                            OLP[1][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -8.0 * (siT * coP * CsumSr.r + coT * coP / r * CsumSt.r -
                                                        siP / siT / r * CsumSp.r);

                                            OLP[2][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -8.0 * (siT * siP * CsumSr.r + coT * siP / r * CsumSt.r +
                                                        coP / siT / r * CsumSp.r);

                                            OLP[3][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -8.0 * (coT * CsumSr.r - siT / r * CsumSt.r);

                                            H0[1][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -4.0 * (siT * coP * CsumKr.r + coT * coP / r * CsumKt.r -
                                                        siP / siT / r * CsumKp.r);

                                            H0[2][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -4.0 * (siT * siP * CsumKr.r + coT * siP / r * CsumKt.r +
                                                        coP / siT / r * CsumKp.r);

                                            H0[3][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] =
                                                -4.0 * (coT * CsumKr.r - siT / r * CsumKt.r);
                                        }
                                    } else {
                                        OLP[1][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] = 0.0;
                                        OLP[2][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] = 0.0;
                                        OLP[3][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1] = 0.0;
                                        H0[1][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1]  = 0.0;
                                        H0[2][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1]  = 0.0;
                                        H0[3][Mc_AN][h_AN][num0 + L0 + M0][num1 + L1 + M1]  = 0.0;
                                    }
                                }
                            }

                            num1 = num1 + 2 * L1 + 1;
                        }
                    }

                    num0 = num0 + 2 * L0 + 1;
                }
            }

            dtime(&Etime_atom);
#pragma omp atomic update
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
        } /* end of loop for Nloop */

        /* freeing of arrays */
        Free6D_dcomplex(TmpOLP);
        Free6D_dcomplex(TmpOLPr);
        Free6D_dcomplex(TmpOLPt);
        Free6D_dcomplex(TmpOLPp);
        Free6D_dcomplex(TmpKin);
        Free6D_dcomplex(TmpKinr);
        Free6D_dcomplex(TmpKint);
        Free6D_dcomplex(TmpKinp);

        Free4D_double(SumS0);
        Free4D_double(SumK0);
        Free4D_double(SumSr0);
        Free4D_double(SumKr0);

        Free2D_dcomplex(CmatS0);
        Free2D_dcomplex(CmatSr);
        Free2D_dcomplex(CmatSt);
        Free2D_dcomplex(CmatSp);
        Free2D_dcomplex(CmatK0);
        Free2D_dcomplex(CmatKr);
        Free2D_dcomplex(CmatKt);
        Free2D_dcomplex(CmatKp);
        Free2D_double(SphB);
        Free2D_double(SphBp);
        free(tmp_SphB);
        free(tmp_SphBp);

    } /* #pragma omp parallel */

    /****************************************************
                   freeing of arrays:
  ****************************************************/

    free(OneD2h_AN);
    free(OneD2Mc_AN);

    /* for time */
    dtime(&TEtime);
    time0 = TEtime - TStime;

    return time0;
}

#ifdef kcomp
dcomplex ****** Allocate6D_dcomplex(int size_1, int size_2, int size_3, int size_4, int size_5, int size_6)
#else
static inline dcomplex ****** Allocate6D_dcomplex(int size_1, int size_2, int size_3, int size_4, int size_5,
                                                  int size_6)
#endif
{
    int i, j, k, l, m;
    size_t n1, n2, n3, n4, n5, n6;
    size_t n12, n123, n1234, n12345, n123456;

    n1 = SetOLPKin_DimToSize(size_1, "Allocate6D size_1", __LINE__);
    n2 = SetOLPKin_DimToSize(size_2, "Allocate6D size_2", __LINE__);
    n3 = SetOLPKin_DimToSize(size_3, "Allocate6D size_3", __LINE__);
    n4 = SetOLPKin_DimToSize(size_4, "Allocate6D size_4", __LINE__);
    n5 = SetOLPKin_DimToSize(size_5, "Allocate6D size_5", __LINE__);
    n6 = SetOLPKin_DimToSize(size_6, "Allocate6D size_6", __LINE__);

    n12     = SetOLPKin_CheckedMul(n1, n2, "Allocate6D pointers", __LINE__);
    n123    = SetOLPKin_CheckedMul(n12, n3, "Allocate6D pointers", __LINE__);
    n1234   = SetOLPKin_CheckedMul(n123, n4, "Allocate6D pointers", __LINE__);
    n12345  = SetOLPKin_CheckedMul(n1234, n5, "Allocate6D pointers", __LINE__);
    n123456 = SetOLPKin_CheckedMul(n12345, n6, "Allocate6D data", __LINE__);

    dcomplex ****** buffer = (dcomplex ******)SETOLPKIN_CALLOC(n1, sizeof(dcomplex *****), "Allocate6D level 1");
    buffer[0]              = (dcomplex *****)SETOLPKIN_CALLOC(n12, sizeof(dcomplex ****), "Allocate6D level 2");
    buffer[0][0]           = (dcomplex ****)SETOLPKIN_CALLOC(n123, sizeof(dcomplex ***), "Allocate6D level 3");
    buffer[0][0][0]        = (dcomplex ***)SETOLPKIN_CALLOC(n1234, sizeof(dcomplex **), "Allocate6D level 4");
    buffer[0][0][0][0]     = (dcomplex **)SETOLPKIN_CALLOC(n12345, sizeof(dcomplex *), "Allocate6D level 5");
    buffer[0][0][0][0][0]  = (dcomplex *)SETOLPKIN_CALLOC(n123456, sizeof(dcomplex), "Allocate6D data");

    for (i = 0; i < size_1; i++) {
        buffer[i] = buffer[0] + (size_t)i * n2;
        for (j = 0; j < size_2; j++) {
            buffer[i][j] = buffer[0][0] + ((size_t)i * n2 + (size_t)j) * n3;
            for (k = 0; k < size_3; k++) {
                buffer[i][j][k] = buffer[0][0][0] + (((size_t)i * n2 + (size_t)j) * n3 + (size_t)k) * n4;
                for (l = 0; l < size_4; l++) {
                    buffer[i][j][k][l] =
                        buffer[0][0][0][0] +
                        ((((size_t)i * n2 + (size_t)j) * n3 + (size_t)k) * n4 + (size_t)l) * n5;
                    for (m = 0; m < size_5; m++) {
                        buffer[i][j][k][l][m] = buffer[0][0][0][0][0] +
                                                (((((size_t)i * n2 + (size_t)j) * n3 + (size_t)k) * n4 +
                                                  (size_t)l) *
                                                     n5 +
                                                 (size_t)m) *
                                                    n6;
                    }
                }
            }
        }
    }

    return buffer;
}

#ifdef kcomp
double **** Allocate4D_double(int size_1, int size_2, int size_3, int size_4)
#else
static inline double **** Allocate4D_double(int size_1, int size_2, int size_3, int size_4)
#endif
{
    int i, j, k;
    size_t n1, n2, n3, n4;
    size_t n12, n123, n1234;

    n1 = SetOLPKin_DimToSize(size_1, "Allocate4D size_1", __LINE__);
    n2 = SetOLPKin_DimToSize(size_2, "Allocate4D size_2", __LINE__);
    n3 = SetOLPKin_DimToSize(size_3, "Allocate4D size_3", __LINE__);
    n4 = SetOLPKin_DimToSize(size_4, "Allocate4D size_4", __LINE__);

    n12   = SetOLPKin_CheckedMul(n1, n2, "Allocate4D pointers", __LINE__);
    n123  = SetOLPKin_CheckedMul(n12, n3, "Allocate4D pointers", __LINE__);
    n1234 = SetOLPKin_CheckedMul(n123, n4, "Allocate4D data", __LINE__);

    double **** buffer = (double ****)SETOLPKIN_CALLOC(n1, sizeof(double ***), "Allocate4D level 1");
    buffer[0]          = (double ***)SETOLPKIN_CALLOC(n12, sizeof(double **), "Allocate4D level 2");
    buffer[0][0]       = (double **)SETOLPKIN_CALLOC(n123, sizeof(double *), "Allocate4D level 3");
    buffer[0][0][0]    = (double *)SETOLPKIN_CALLOC(n1234, sizeof(double), "Allocate4D data");

    for (i = 0; i < size_1; i++) {
        buffer[i] = buffer[0] + (size_t)i * n2;
        for (j = 0; j < size_2; j++) {
            buffer[i][j] = buffer[0][0] + ((size_t)i * n2 + (size_t)j) * n3;
            for (k = 0; k < size_3; k++) {
                buffer[i][j][k] = buffer[0][0][0] + (((size_t)i * n2 + (size_t)j) * n3 + (size_t)k) * n4;
            }
        }
    }

    return buffer;
}

#ifdef kcomp
dcomplex ** Allocate2D_dcomplex(int size_1, int size_2)
#else
static inline dcomplex ** Allocate2D_dcomplex(int size_1, int size_2)
#endif
{
    int i;
    size_t n1, n2, n12;

    n1  = SetOLPKin_DimToSize(size_1, "Allocate2D_dcomplex size_1", __LINE__);
    n2  = SetOLPKin_DimToSize(size_2, "Allocate2D_dcomplex size_2", __LINE__);
    n12 = SetOLPKin_CheckedMul(n1, n2, "Allocate2D_dcomplex data", __LINE__);

    dcomplex ** buffer = (dcomplex **)SETOLPKIN_CALLOC(n1, sizeof(dcomplex *), "Allocate2D_dcomplex pointers");
    buffer[0]          = (dcomplex *)SETOLPKIN_CALLOC(n12, sizeof(dcomplex), "Allocate2D_dcomplex data");

    for (i = 0; i < size_1; i++) {
        buffer[i] = buffer[0] + (size_t)i * n2;
    }

    return buffer;
}

#ifdef kcomp
double ** Allocate2D_double(int size_1, int size_2)
#else
static inline double ** Allocate2D_double(int size_1, int size_2)
#endif
{
    int i;
    size_t n1, n2, n12;

    n1  = SetOLPKin_DimToSize(size_1, "Allocate2D_double size_1", __LINE__);
    n2  = SetOLPKin_DimToSize(size_2, "Allocate2D_double size_2", __LINE__);
    n12 = SetOLPKin_CheckedMul(n1, n2, "Allocate2D_double data", __LINE__);

    double ** buffer = (double **)SETOLPKIN_CALLOC(n1, sizeof(double *), "Allocate2D_double pointers");
    buffer[0]        = (double *)SETOLPKIN_CALLOC(n12, sizeof(double), "Allocate2D_double data");

    for (i = 0; i < size_1; i++) {
        buffer[i] = buffer[0] + (size_t)i * n2;
    }

    return buffer;
}

void Free6D_dcomplex(dcomplex ****** buffer)
{
    if (buffer == NULL) {
        return;
    }
    free(buffer[0][0][0][0][0]);
    free(buffer[0][0][0][0]);
    free(buffer[0][0][0]);
    free(buffer[0][0]);
    free(buffer[0]);
    free(buffer);
}

void Free4D_double(double **** buffer)
{
    if (buffer == NULL) {
        return;
    }
    free(buffer[0][0][0]);
    free(buffer[0][0]);
    free(buffer[0]);
    free(buffer);
}

void Free2D_dcomplex(dcomplex ** buffer)
{
    if (buffer == NULL) {
        return;
    }
    free(buffer[0]);
    free(buffer);
}

void Free2D_double(double ** buffer)
{
    if (buffer == NULL) {
        return;
    }
    free(buffer[0]);
    free(buffer);
}
