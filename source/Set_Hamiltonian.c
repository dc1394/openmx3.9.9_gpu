/**********************************************************************
  Set_Hamiltonian.c:

     Set_Hamiltonian.c is a subroutine to make Hamiltonian matrix
     within LDA or GGA.

  Log of Set_Hamiltonian.c:

     24/April/2002  Released by T. Ozaki
     17/April/2013  Modified by A.M. Ito

***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "set_cuda_default_device_from_local_rank.h"
#include <accel.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define measure_time 0

/* Runtime phase profiling gated by OPENMX_BAND_PROFILE=1 (one SETHPROF line
   per rank per call on stderr; zero overhead when the variable is unset). */
typedef struct
{
    double memok;
    double base;
    double vpot;
    double matel;
    double barrier;
} SetHamiltonianProfileCounters;

static SetHamiltonianProfileCounters SetH_prof = {0};

static int SetH_ProfileEnabled(void)
{
    static int enabled = -1;

    if (enabled < 0) {
        const char *value = getenv("OPENMX_BAND_PROFILE");
        enabled = (value != NULL && atoi(value) != 0);
    }
    return enabled;
}

#define SETH_PROF_T0(t)                                                                                                \
    do {                                                                                                               \
        if (SetH_ProfileEnabled())                                                                                     \
            dtime(&(t));                                                                                               \
    } while (0)

#define SETH_PROF_ADD(field, t)                                                                                        \
    do {                                                                                                               \
        if (SetH_ProfileEnabled()) {                                                                                   \
            double _prof_t1;                                                                                           \
            dtime(&_prof_t1);                                                                                          \
            SetH_prof.field += _prof_t1 - (t);                                                                         \
        }                                                                                                              \
    } while (0)

void Calc_MatrixElements_dVH_Vxc_VNA(int Cnt_kind);
static void Calc_MatrixElements_dVH_Vxc_VNA_CPU(int Cnt_kind);
static void Set_Hamiltonian_Base_OpenACC(int SCF_iter, double *****H0, double *****HNL, double *****H);
static size_t Set_Hamiltonian_Base_OpenACC_DeviceBytes(int SCF_iter, int myid);
static size_t Set_Hamiltonian_MatrixElements_OpenACC_DeviceBytes(int Cnt_kind, int myid);
static void *Set_Hamiltonian_malloc(size_t bytes, const char *name, int myid);

int Set_Hamiltonian_Cuda_MatrixElements(int pair_count, int spin_count, size_t vpot_len,
                                        double grid_vol, int max_no, int max_output_count,
                                        const int *pair_NO0, const int *pair_NO1, const int *pair_NOLG,
                                        const int *nolg_MN, const int *nolg_Nc,
                                        const size_t *pair_h_offset, const size_t *pair_nolg_offset,
                                        const size_t *pair_orbs0_offset, const size_t *pair_orbs1_offset,
                                        const Type_Orbs_Grid *orbs0buf, const Type_Orbs_Grid *orbs1buf,
                                        const double *vpotgrid, double *hbuf);

static int Set_Hamiltonian_OpenACC_Rank_Selected = 1;
static int Set_Hamiltonian_OpenACC_Work_Rank_Selected = 1;

void Set_Hamiltonian_Set_OpenACC_Rank_Selected(int selected)
{
    Set_Hamiltonian_OpenACC_Rank_Selected = selected ? 1 : 0;
}

int Set_Hamiltonian_OpenACC_Rank_Is_Selected(void)
{
    return Set_Hamiltonian_OpenACC_Rank_Selected;
}

void Set_Hamiltonian_Set_OpenACC_Work_Rank_Selected(int selected)
{
    Set_Hamiltonian_OpenACC_Work_Rank_Selected = selected ? 1 : 0;
}

int Set_Hamiltonian_OpenACC_Work_Rank_Is_Selected(void)
{
    return Set_Hamiltonian_OpenACC_Work_Rank_Selected;
}

enum {
    SET_HAMILTONIAN_PACK_ORDER_COL = 0,
    SET_HAMILTONIAN_PACK_ORDER_NONCOL = 1
};

/* Classes of SCF-invariant arrays that can live on the device, in
   descending order of typical size (= per-iteration transfer savings).
   Partial residency admits a subset in this priority order when the whole
   set does not fit beside the registered GPU-phase needs. */
enum {
    SETH_RES_ORBS1 = 0,
    SETH_RES_ORBS0 = 1,
    SETH_RES_NOLG = 2,
    SETH_RES_META = 3,
    SETH_RES_NCLASS = 4
};

typedef struct {
    int ready;
    int owns;
    int order_mode;
    int size;
    int spin_count;
    int atom_count;
    int *order_GA;
    double *overlap;
    double *h[4];
    double *imnl[3];
} SetHamiltonianGpuSolverPackedCache;

static SetHamiltonianGpuSolverPackedCache Set_Hamiltonian_GpuSolver_Cache = {0};

typedef struct {
    int ready;
    int resident_mask;
    int resident_blocked;
    int cnt_kind;
    int spin_count;
    int pair_count;
    int matomnum;
    size_t total_h;
    size_t total_nolg;
    size_t total_orbs0;
    size_t total_orbs1;
    int *pair_Mc_AN;
    int *pair_h_AN;
    int *pair_NO0;
    int *pair_NO1;
    int *pair_NOLG;
    int *nolg_MN;
    int *nolg_Nc;
    size_t *pair_h_offset;
    size_t *pair_nolg_offset;
    size_t *pair_orbs0_offset;
    size_t *pair_orbs1_offset;
    Type_Orbs_Grid *orbs0buf;
    Type_Orbs_Grid *orbs1buf;
} SetHamiltonianMatrixElementsCache;

static SetHamiltonianMatrixElementsCache Set_Hamiltonian_ME_Cache = {0};

typedef struct {
    SetHamiltonianMatrixElementsCache *cache;
    double *hbuf;
    double *vpotgrid;
    int cnt_kind;
    int myid;
    int max_no;
    int max_output_count;
    int cuda_kernel_supported;
    double pack_seconds;
    double device_seconds;
} SetHamiltonianMatrixElementsWork;

static int Set_Hamiltonian_ME_CacheMatches(const SetHamiltonianMatrixElementsCache *cache, int Cnt_kind);
static void Set_Hamiltonian_MatrixElements_ResidentClassBytes(int Cnt_kind, int myid,
                                                              size_t class_bytes[SETH_RES_NCLASS]);
static size_t Set_Hamiltonian_MatrixElements_TransientBytes(const SetHamiltonianMatrixElementsCache *cache,
                                                            int myid);
static SetHamiltonianMatrixElementsCache *Set_Hamiltonian_Ensure_OpenACC_MatrixElements_Cache(int Cnt_kind,
                                                                                              int myid);
static void Set_Hamiltonian_Prepare_OpenACC_MatrixElements(SetHamiltonianMatrixElementsWork *work,
                                                            int Cnt_kind, int myid);
static void Set_Hamiltonian_Run_OpenACC_MatrixElements(SetHamiltonianMatrixElementsWork *work);
static void Set_Hamiltonian_Finish_OpenACC_MatrixElements(SetHamiltonianMatrixElementsWork *work);

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

static int Set_Hamiltonian_OpenACC_Enabled(void)
{
    return (scf_eigen_lib_flag == GPUSOLVER && gpu_rank_device_usable());
}

static int Set_Hamiltonian_MatrixElements_OpenACC_Enabled(void)
{
    const char *value = getenv("OPENMX_SETHAM_MATRIX_GPU");

    /* This is the expensive part of Set_Hamiltonian.  It is enabled for
       GPUSOLVER by default and can be disabled for comparison/debugging. */
    return (value == NULL) ? 1 : (atoi(value) != 0);
}

static int Set_Hamiltonian_MatrixElements_CudaKernel_Enabled(void)
{
    const char *value = getenv("OPENMX_SETHAM_CUDA_KERNEL");
    return (value == NULL) ? 1 : (atoi(value) != 0);
}

typedef struct {
    MPI_Comm device_comm;
    int selected;
    int use_gpu;
    int cuda_device;
    int device_rank;
    int device_ranks;
    int concurrent_ranks;
    int turn;
    int turns;
    int resident_admit_mask;
    int release_resident;
    size_t resident_group_bytes;
    size_t registered_need_bytes;
    size_t required_bytes;
    size_t peak_bytes;
    size_t free_bytes;
    size_t total_bytes;
    size_t reserve_bytes;
} SetHamiltonianGpuTurnPlan;

typedef struct {
    int valid;
    unsigned char uuid[16];
} SetHamiltonianGpuUuidRecord;

static int Set_Hamiltonian_compare_ull_desc(const void *a, const void *b)
{
    const unsigned long long va = *(const unsigned long long *)a;
    const unsigned long long vb = *(const unsigned long long *)b;
    return (va < vb) ? 1 : ((vb < va) ? -1 : 0);
}

static size_t Set_Hamiltonian_GpuReserveBytes(size_t total_bytes)
{
    size_t reserve = total_bytes / 10U;
    const size_t gemmul8_reserve = 1536ULL * 1024ULL * 1024ULL;
    const char *value = getenv("OPENMX_SETHAM_GPU_RESERVE_MB");

    if (reserve < gemmul8_reserve) reserve = gemmul8_reserve;
    if (value != NULL && value[0] != '\0') {
        char *endp = NULL;
        unsigned long long mib = strtoull(value, &endp, 10);
        if (endp != value && *endp == '\0' && mib <= (unsigned long long)((size_t)-1) / (1024ULL * 1024ULL)) {
            size_t requested = (size_t)mib * 1024ULL * 1024ULL;
            if (reserve < requested) reserve = requested;
        }
    }
    return reserve;
}

static int Set_Hamiltonian_GpuRequestedMaxRanks(void)
{
    /* By default every MPI rank that shares the device may join the first
       wave (the plan clamps to the actual device-group size); the variable
       only imposes an explicit upper cap. */
    const char *value = getenv("OPENMX_SETHAM_GPU_MAX_RANKS_PER_DEVICE");
    long requested = (long)INT_MAX;

    if (value != NULL && value[0] != '\0') {
        char *end = NULL;
        long parsed = strtol(value, &end, 10);
        if (end != value && *end == '\0') requested = parsed;
    }
    if (requested < 1L) requested = 1L;
    if ((long)INT_MAX < requested) requested = (long)INT_MAX;
    return (int)requested;
}

static int Set_Hamiltonian_GpuSerialWaves(void)
{
    const char *value = getenv("OPENMX_SETHAM_GPU_SERIAL_WAVES");

    /* Prefer the hybrid policy on smaller GPUs: the first fitting wave uses
       the GPU while the remaining ranks compute on the CPU concurrently.  A
       large-memory GPU naturally admits every rank into that first wave.
       Set the variable to one to force every later wave through the GPU. */
    return (value == NULL) ? 0 : (atoi(value) != 0);
}

static int Set_Hamiltonian_GpuResidentEnabled(void)
{
    /* Keep the SCF-invariant tables (orbitals, grid indices, pair metadata)
       resident on the device across SCF iterations.  Residency is attempted
       only when every rank sharing the GPU runs in the single wave and the
       resident total leaves the configured floor free for the eigensolver
       and the grid routines.  Set to zero to force per-iteration staging. */
    const char *value = getenv("OPENMX_SETHAM_GPU_RESIDENT");
    return (value == NULL) ? 1 : (atoi(value) != 0);
}

static size_t Set_Hamiltonian_GpuResidentFloorBytes(int have_registered_need)
{
    /* Device memory that must remain free after the resident tables are
       uploaded.  When a GPU phase has published its full-concurrency need
       that need is charged separately, so only a small base margin is
       kept; without any registration the legacy conservative default
       protects the (non-publishing) cluster solvers.  The environment
       variable overrides both defaults. */
    size_t floor_bytes = have_registered_need ? 4096ULL * 1024ULL * 1024ULL
                                              : 16384ULL * 1024ULL * 1024ULL;
    const char *value = getenv("OPENMX_SETHAM_GPU_RESIDENT_FLOOR_MB");

    if (value != NULL && value[0] != '\0') {
        char *endp = NULL;
        unsigned long long mib = strtoull(value, &endp, 10);
        if (endp != value && *endp == '\0' && mib <= (unsigned long long)((size_t)-1) / (1024ULL * 1024ULL)) {
            floor_bytes = (size_t)mib * 1024ULL * 1024ULL;
        }
    }
    return floor_bytes;
}

static size_t Set_Hamiltonian_GpuTestNeed_Now = 0;

static size_t Set_Hamiltonian_GpuTestNeedBytes(int call_index)
{
    /* Testing hook: pretend a GPU phase registered this need (MiB) from the
       given matrix-elements call on (1-based, default 1), so the defer,
       partial-residency and backoff paths can be exercised on any GPU. */
    const char *mb = getenv("OPENMX_SETHAM_GPU_TEST_NEED_MB");
    const char *from = getenv("OPENMX_SETHAM_GPU_TEST_NEED_FROM_CALL");
    unsigned long long mib = 0ULL;
    long from_call = 1L;

    if (mb == NULL || mb[0] == '\0') return 0;
    {
        char *endp = NULL;
        mib = strtoull(mb, &endp, 10);
        if (endp == mb || *endp != '\0') return 0;
    }
    if (from != NULL && from[0] != '\0') {
        char *endp = NULL;
        long parsed = strtol(from, &endp, 10);
        if (endp != from && *endp == '\0') from_call = parsed;
    }
    if (call_index < from_call) return 0;
    if ((unsigned long long)((size_t)-1) / (1024ULL * 1024ULL) < mib) return (size_t)-1;
    return (size_t)mib * 1024ULL * 1024ULL;
}

static SetHamiltonianGpuTurnPlan Set_Hamiltonian_CreateGpuTurnPlan(size_t required_bytes,
                                                                   const size_t *resident_request_class,
                                                                   int resident_active,
                                                                   const char *where, int myid)
{
    SetHamiltonianGpuTurnPlan plan;
    unsigned long long phase_need = 0ULL;
    unsigned long long diag_need = 0ULL;
    unsigned long long group_cls[SETH_RES_NCLASS] = {0ULL, 0ULL, 0ULL, 0ULL};
    SetHamiltonianGpuUuidRecord local_uuid;
    SetHamiltonianGpuUuidRecord *node_uuids = NULL;
    MPI_Comm node_comm = MPI_COMM_NULL;
    int node_rank = 0, node_ranks = 0;
    int cuda_ok = 0, color = MPI_UNDEFINED;
    int local_memory_ok = 0, group_memory_ok = 0;
    unsigned long long local_free = 0, group_free = 0;
    unsigned long long local_total = 0, group_total = 0;
    unsigned long long local_required = 0;
    unsigned long long *requirements = NULL;
    cudaError_t cuda_status = cudaSuccess;

    memset(&plan, 0, sizeof(plan));
    memset(&local_uuid, 0, sizeof(local_uuid));
    plan.device_comm = MPI_COMM_NULL;
    plan.cuda_device = -1;
    plan.required_bytes = required_bytes;
    plan.selected = (Set_Hamiltonian_OpenACC_Work_Rank_Selected && 0 < required_bytes);

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_ranks);

    if (plan.selected) {
        int cuda_devices = 0;
        int acc_devices = acc_get_num_devices(acc_device_nvidia);
        struct cudaDeviceProp prop;

        cuda_status = cudaGetDeviceCount(&cuda_devices);
        if (cuda_status == cudaSuccess && 0 < cuda_devices && 0 < acc_devices) {
            cuda_status = cudaGetDevice(&plan.cuda_device);
        }
        if (cuda_status == cudaSuccess && 0 <= plan.cuda_device &&
            plan.cuda_device < cuda_devices && plan.cuda_device < acc_devices) {
            cuda_status = cudaGetDeviceProperties(&prop, plan.cuda_device);
        }
        if (cuda_status == cudaSuccess && 0 <= plan.cuda_device &&
            plan.cuda_device < cuda_devices && plan.cuda_device < acc_devices) {
            acc_set_device_num(plan.cuda_device, acc_device_nvidia);
            local_uuid.valid = 1;
            memcpy(local_uuid.uuid, prop.uuid.bytes, sizeof(local_uuid.uuid));
            cuda_ok = 1;
        }
    }

    node_uuids = (SetHamiltonianGpuUuidRecord *)Set_Hamiltonian_malloc(
        sizeof(SetHamiltonianGpuUuidRecord) * (size_t)node_ranks, "GPU UUID table", myid);
    MPI_Allgather(&local_uuid, (int)sizeof(local_uuid), MPI_BYTE,
                  node_uuids, (int)sizeof(local_uuid), MPI_BYTE, node_comm);

    if (cuda_ok) {
        for (int rank = 0; rank < node_ranks; rank++) {
            if (node_uuids[rank].valid &&
                memcmp(node_uuids[rank].uuid, local_uuid.uuid, sizeof(local_uuid.uuid)) == 0) {
                color = rank + 1;
                break;
            }
        }
    }
    free(node_uuids);

    MPI_Comm_split(node_comm, color, node_rank, &plan.device_comm);
    MPI_Comm_free(&node_comm);

    if (!plan.selected) return plan;
    if (!cuda_ok || plan.device_comm == MPI_COMM_NULL) {
        fprintf(stderr, "Set_Hamiltonian: rank %d %s cannot initialize CUDA/OpenACC; using CPU.\n", myid, where);
        fflush(stderr);
        return plan;
    }

    MPI_Comm_rank(plan.device_comm, &plan.device_rank);
    MPI_Comm_size(plan.device_comm, &plan.device_ranks);

    /* Deleted OpenACC allocations are private to each MPI process.  Return
       all of them before the physical-GPU group measures its common free
       memory, otherwise the cap can be reduced by stale freelist storage. */
    acc_wait_all();
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status == cudaSuccess) acc_clear_freelists();
    MPI_Barrier(plan.device_comm);

    if (cuda_status == cudaSuccess) {
        size_t free_bytes = 0, total_bytes = 0;
        cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (cuda_status == cudaSuccess && required_bytes <= (size_t)ULLONG_MAX &&
            free_bytes <= (size_t)ULLONG_MAX && total_bytes <= (size_t)ULLONG_MAX) {
            local_memory_ok = 1;
            local_free = (unsigned long long)free_bytes;
            local_total = (unsigned long long)total_bytes;
            local_required = (unsigned long long)required_bytes;
        }
    }

    MPI_Allreduce(&local_memory_ok, &group_memory_ok, 1, MPI_INT, MPI_MIN, plan.device_comm);
    if (!group_memory_ok) {
        if (plan.device_rank == 0) {
            fprintf(stderr, "Set_Hamiltonian: %s failed to query common GPU memory; using CPU.\n", where);
            fflush(stderr);
        }
        return plan;
    }

    MPI_Allreduce(&local_free, &group_free, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, plan.device_comm);
    MPI_Allreduce(&local_total, &group_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, plan.device_comm);
    requirements = (unsigned long long *)Set_Hamiltonian_malloc(
        sizeof(unsigned long long) * (size_t)plan.device_ranks, "GPU turn requirements", myid);
    MPI_Allgather(&local_required, 1, MPI_UNSIGNED_LONG_LONG,
                  requirements, 1, MPI_UNSIGNED_LONG_LONG, plan.device_comm);
    qsort(requirements, (size_t)plan.device_ranks, sizeof(unsigned long long), Set_Hamiltonian_compare_ull_desc);

    plan.free_bytes = (size_t)group_free;
    plan.total_bytes = (size_t)group_total;
    plan.reserve_bytes = Set_Hamiltonian_GpuReserveBytes(plan.total_bytes);

    {
        /* Collect the group-wide resident-class requests and the largest
           need any GPU phase has published for this device.  Both are
           gathered before the admission scan so a resident cache under
           pressure can release even when no rank is admitted this call. */
        unsigned long long local_cls[SETH_RES_NCLASS];
        unsigned long long local_needs[2], group_needs[2] = {0ULL, 0ULL};
        unsigned long long test_need = (unsigned long long)Set_Hamiltonian_GpuTestNeed_Now;

        for (int c = 0; c < SETH_RES_NCLASS; c++) {
            local_cls[c] = (resident_request_class != NULL) ?
                (unsigned long long)resident_request_class[c] : 0ULL;
        }

        /* [0]: the largest need of any registered GPU phase; [1]: the same
           restricted to the diagonalization planners.  The test hook counts
           as a diagonalization need so it exercises the same branches. */
        local_needs[0] = (unsigned long long)OpenMX_GpuPhaseNeed_Max();
        local_needs[1] = (unsigned long long)OpenMX_GpuPhaseNeed_MaxPrefixed("band_");
        if (local_needs[0] < test_need) local_needs[0] = test_need;
        if (local_needs[1] < test_need) local_needs[1] = test_need;

        MPI_Allreduce(local_cls, group_cls, SETH_RES_NCLASS,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, plan.device_comm);
        MPI_Allreduce(local_needs, group_needs, 2,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, plan.device_comm);
        phase_need = group_needs[0];
        diag_need = group_needs[1];
        plan.registered_need_bytes = (size_t)phase_need;

        if (resident_active && 0ULL < phase_need) {
            unsigned long long floor_bytes =
                (unsigned long long)Set_Hamiltonian_GpuResidentFloorBytes(1);

            if (group_free < phase_need || floor_bytes > group_free - phase_need) {
                plan.release_resident = 1;
            }
        }
    }

    {
        /* Start from every MPI rank that shares this GPU and step down until
           the wave fits.  The requirements are sorted in descending order,
           so the peak of the top-c ranks grows monotonically with c and one
           prefix scan yields the largest fitting concurrency. */
        const int requested_max = Set_Hamiltonian_GpuRequestedMaxRanks();
        const int cmax = (plan.device_ranks < requested_max) ? plan.device_ranks : requested_max;
        unsigned long long peak = 0;

        for (int c = 1; c <= cmax; c++) {
            if (ULLONG_MAX - peak < requirements[c - 1]) break;
            peak += requirements[c - 1];

            if (peak <= group_free && (unsigned long long)plan.reserve_bytes <= group_free - peak) {
                plan.concurrent_ranks = c;
                plan.peak_bytes = (size_t)peak;
            }
            else {
                break;
            }
        }
    }
    free(requirements);

    if (plan.concurrent_ranks == 0) {
        if (plan.device_rank == 0) {
            fprintf(stderr,
                    "Set_Hamiltonian: %s does not fit even one GPU rank: free %.3f GiB, reserve %.3f GiB; using CPU.\n",
                    where, (double)plan.free_bytes / (1024.0 * 1024.0 * 1024.0),
                    (double)plan.reserve_bytes / (1024.0 * 1024.0 * 1024.0));
            fflush(stderr);
        }
        return plan;
    }

    plan.use_gpu = 1;
    plan.turn = plan.device_rank / plan.concurrent_ranks;
    plan.turns = (plan.device_ranks + plan.concurrent_ranks - 1) / plan.concurrent_ranks;

    if (!plan.release_resident && plan.concurrent_ranks == plan.device_ranks) {
        /* Residency is a group affair: it is attempted only when every rank
           sharing this GPU runs in the single wave.  The budget respects the
           largest published GPU-phase need (band diagonalization) plus a
           small base margin; without any publication the band solvers defer
           until their first diagonalization has published, while the
           non-publishing solvers keep the legacy conservative floor.  The
           classes are admitted greedily in descending size order, so a
           partial set is kept when the whole set does not fit. */
        unsigned long long budget = 0ULL;
        unsigned long long total_request = 0ULL;
        int have_budget = 0;

        for (int c = 0; c < SETH_RES_NCLASS; c++) total_request += group_cls[c];

        if (0ULL < total_request) {
            if (0ULL < diag_need) {
                unsigned long long floor_bytes =
                    (unsigned long long)Set_Hamiltonian_GpuResidentFloorBytes(1);

                if (phase_need <= group_free && floor_bytes <= group_free - phase_need) {
                    budget = group_free - phase_need - floor_bytes;
                    have_budget = 1;
                }
            }
            else if (Solver == 3) {
                /* Band solvers publish their full k-concurrency need from
                   the first diagonalization on; wait for that number
                   instead of claiming memory the k-owners may need. */
            }
            else {
                /* The cluster diagonalization does not publish its need, so
                   keep the conservative legacy floor even when other phases
                   (density-grid staging) have registered theirs. */
                unsigned long long floor_bytes =
                    (unsigned long long)Set_Hamiltonian_GpuResidentFloorBytes(0);

                if (floor_bytes <= group_free) {
                    budget = group_free - floor_bytes;
                    have_budget = 1;
                }
            }
        }

        if (have_budget) {
            for (int c = 0; c < SETH_RES_NCLASS; c++) {
                if (0ULL < group_cls[c] && group_cls[c] <= budget) {
                    plan.resident_admit_mask |= (1 << c);
                    plan.resident_group_bytes += (size_t)group_cls[c];
                    budget -= group_cls[c];
                }
            }
        }
    }

    if (plan.device_rank == 0) {
        static int last_device_ranks = -1;
        static int last_concurrent = -1;
        if (last_device_ranks != plan.device_ranks || last_concurrent != plan.concurrent_ranks) {
            printf("<Set_Hamiltonian> GPU device %d: %d Hamiltonian rank(s), GPU concurrency=%d, %s=%d, "
                   "peak=%.3f GiB, free=%.3f GiB, reserve=%.3f GiB\n",
                   plan.cuda_device, plan.device_ranks, plan.concurrent_ranks,
                   Set_Hamiltonian_GpuSerialWaves() ? "serialized waves" : "CPU fallback ranks",
                   Set_Hamiltonian_GpuSerialWaves() ? plan.turns : plan.device_ranks - plan.concurrent_ranks,
                   (double)plan.peak_bytes / (1024.0 * 1024.0 * 1024.0),
                   (double)plan.free_bytes / (1024.0 * 1024.0 * 1024.0),
                   (double)plan.reserve_bytes / (1024.0 * 1024.0 * 1024.0));
            fflush(stdout);
            last_device_ranks = plan.device_ranks;
            last_concurrent = plan.concurrent_ranks;
        }
    }

    return plan;
}

static void Set_Hamiltonian_DestroyGpuTurnPlan(SetHamiltonianGpuTurnPlan *plan)
{
    if (plan->device_comm != MPI_COMM_NULL) MPI_Comm_free(&plan->device_comm);
}

static int Set_Hamiltonian_Base_OpenACC_Enabled(void)
{
    /* The packed base add costs more in packing and transfers than the
       trivial CPU loop (measured 10 ms vs 2 ms per iteration), so the GPU
       path remains opt-in.  The quadrature below is the performance-critical
       GPU path and is enabled independently. */
    const char *value = getenv("OPENMX_SETHAM_BASE_GPU");
    if (Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1) return 0;
    return (value != NULL && atoi(value) != 0);
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

static void Set_Hamiltonian_ResidentMaskLabel(int mask, char *buf, size_t buf_len)
{
    static const char *names[SETH_RES_NCLASS] = {"orbs1", "orbs0", "nolg", "meta"};
    size_t off = 0;
    int c;

    buf[0] = '\0';
    for (c = 0; c < SETH_RES_NCLASS; c++) {
        if ((mask & (1 << c)) && off + 8 < buf_len) {
            off += (size_t)snprintf(buf + off, buf_len - off, "%s%s", (0 < off) ? "+" : "", names[c]);
        }
    }
    if (buf[0] == '\0') snprintf(buf, buf_len, "none");
}

static void Set_Hamiltonian_ME_EnterDeviceCache(const SetHamiltonianGpuTurnPlan *plan)
{
    SetHamiltonianMatrixElementsCache *cache = &Set_Hamiltonian_ME_Cache;
    int *pair_NO0 = cache->pair_NO0;
    int *pair_NO1 = cache->pair_NO1;
    int *pair_NOLG = cache->pair_NOLG;
    int *nolg_MN = cache->nolg_MN;
    int *nolg_Nc = cache->nolg_Nc;
    size_t *pair_h_offset = cache->pair_h_offset;
    size_t *pair_nolg_offset = cache->pair_nolg_offset;
    size_t *pair_orbs0_offset = cache->pair_orbs0_offset;
    size_t *pair_orbs1_offset = cache->pair_orbs1_offset;
    Type_Orbs_Grid *orbs0buf = cache->orbs0buf;
    Type_Orbs_Grid *orbs1buf = cache->orbs1buf;
    const int pair_count = cache->pair_count;
    const size_t total_nolg = cache->total_nolg;
    const size_t total_orbs0 = cache->total_orbs0;
    const size_t total_orbs1 = cache->total_orbs1;
    const int add_mask = plan->resident_admit_mask & ~cache->resident_mask;

    if (add_mask == 0 || pair_count <= 0) return;

    if (add_mask & (1 << SETH_RES_ORBS1)) {
#pragma acc enter data copyin(orbs1buf[0:total_orbs1])
    }
    if (add_mask & (1 << SETH_RES_ORBS0)) {
#pragma acc enter data copyin(orbs0buf[0:total_orbs0])
    }
    if (add_mask & (1 << SETH_RES_NOLG)) {
#pragma acc enter data copyin(nolg_MN[0:total_nolg], nolg_Nc[0:total_nolg])
    }
    if (add_mask & (1 << SETH_RES_META)) {
#pragma acc enter data copyin(pair_NO0[0:pair_count], pair_NO1[0:pair_count], pair_NOLG[0:pair_count],                    \
                              pair_h_offset[0:pair_count], pair_nolg_offset[0:pair_count],                                 \
                              pair_orbs0_offset[0:pair_count], pair_orbs1_offset[0:pair_count])
    }

    cache->resident_mask |= add_mask;

    if (plan->device_rank == 0) {
        char label[64];

        Set_Hamiltonian_ResidentMaskLabel(cache->resident_mask, label, sizeof(label));
        printf("<Set_Hamiltonian> GPU resident cache: %d rank(s) hold %.3f GiB (%s) on device %d across SCF iterations\n",
               plan->device_ranks,
               (double)plan->resident_group_bytes / (1024.0 * 1024.0 * 1024.0),
               label, plan->cuda_device);
        fflush(stdout);
    }
}

static void Set_Hamiltonian_ME_ReleaseDeviceCache(void)
{
    SetHamiltonianMatrixElementsCache *cache = &Set_Hamiltonian_ME_Cache;
    int *pair_NO0 = cache->pair_NO0;
    int *pair_NO1 = cache->pair_NO1;
    int *pair_NOLG = cache->pair_NOLG;
    int *nolg_MN = cache->nolg_MN;
    int *nolg_Nc = cache->nolg_Nc;
    size_t *pair_h_offset = cache->pair_h_offset;
    size_t *pair_nolg_offset = cache->pair_nolg_offset;
    size_t *pair_orbs0_offset = cache->pair_orbs0_offset;
    size_t *pair_orbs1_offset = cache->pair_orbs1_offset;
    Type_Orbs_Grid *orbs0buf = cache->orbs0buf;
    Type_Orbs_Grid *orbs1buf = cache->orbs1buf;
    const int pair_count = cache->pair_count;
    const size_t total_nolg = cache->total_nolg;
    const size_t total_orbs0 = cache->total_orbs0;
    const size_t total_orbs1 = cache->total_orbs1;

    const int mask = cache->resident_mask;

    if (mask == 0) return;

    if (mask & (1 << SETH_RES_ORBS1)) {
#pragma acc exit data delete(orbs1buf[0:total_orbs1])
    }
    if (mask & (1 << SETH_RES_ORBS0)) {
#pragma acc exit data delete(orbs0buf[0:total_orbs0])
    }
    if (mask & (1 << SETH_RES_NOLG)) {
#pragma acc exit data delete(nolg_MN[0:total_nolg], nolg_Nc[0:total_nolg])
    }
    if (mask & (1 << SETH_RES_META)) {
#pragma acc exit data delete(pair_NO0[0:pair_count], pair_NO1[0:pair_count], pair_NOLG[0:pair_count],                     \
                             pair_h_offset[0:pair_count], pair_nolg_offset[0:pair_count],                                  \
                             pair_orbs0_offset[0:pair_count], pair_orbs1_offset[0:pair_count])
    }

    /* return the pooled storage to CUDA so the next consumer (eigensolver,
       force batches) measures the true free device memory */
    acc_wait_all();
    if (cudaDeviceSynchronize() == cudaSuccess) {
        acc_clear_freelists();
    }
    cache->resident_mask = 0;
}

void Set_Hamiltonian_Release_OpenACC_DeviceCache(void)
{
    Set_Hamiltonian_ME_ReleaseDeviceCache();
}

/* Non-building probe: report whether the shared matrix-elements tables for
   this Cnt_kind already exist.  Set_Hamiltonian_GetMatrixElementsTables
   builds the multi-GiB host tables on first use, which consumers that only
   want to ride on existing tables (Set_Density_Grid's distributed mode)
   must not trigger — on a crowded device most ranks run Set_Hamiltonian on
   the CPU and never build them, and forcing the build on every such rank
   at once can exhaust the host memory. */
int Set_Hamiltonian_MatrixElementsTables_Ready(int Cnt_kind)
{
    return Set_Hamiltonian_ME_CacheMatches(&Set_Hamiltonian_ME_Cache, Cnt_kind);
}

int Set_Hamiltonian_GetMatrixElementsTables(int Cnt_kind, SetHamiltonianMETables *tables)
{
    SetHamiltonianMatrixElementsCache *cache;
    int myid;

    memset(tables, 0, sizeof(*tables));
    if (Cnt_kind != 0 && Cnt_kind != 1) return 0;
    if (SpinP_switch != 0 && SpinP_switch != 1 && SpinP_switch != 3) return 0;
    if (Matomnum <= 0) return 0;

    MPI_Comm_rank(mpi_comm_level1, &myid);
    cache = Set_Hamiltonian_Ensure_OpenACC_MatrixElements_Cache(Cnt_kind, myid);
    if (!cache->ready || cache->pair_count <= 0) return 0;

    tables->pair_count = cache->pair_count;
    tables->total_h = cache->total_h;
    tables->total_nolg = cache->total_nolg;
    tables->total_orbs0 = cache->total_orbs0;
    tables->total_orbs1 = cache->total_orbs1;
    tables->pair_Mc_AN = cache->pair_Mc_AN;
    tables->pair_h_AN = cache->pair_h_AN;
    tables->pair_NO0 = cache->pair_NO0;
    tables->pair_NO1 = cache->pair_NO1;
    tables->pair_NOLG = cache->pair_NOLG;
    tables->nolg_MN = cache->nolg_MN;
    tables->nolg_Nc = cache->nolg_Nc;
    tables->pair_h_offset = cache->pair_h_offset;
    tables->pair_nolg_offset = cache->pair_nolg_offset;
    tables->pair_orbs0_offset = cache->pair_orbs0_offset;
    tables->pair_orbs1_offset = cache->pair_orbs1_offset;
    tables->orbs0buf = cache->orbs0buf;
    tables->orbs1buf = cache->orbs1buf;
    tables->orbs0_resident = (cache->resident_mask >> SETH_RES_ORBS0) & 1;
    tables->orbs1_resident = (cache->resident_mask >> SETH_RES_ORBS1) & 1;
    tables->nolg_resident = (cache->resident_mask >> SETH_RES_NOLG) & 1;
    tables->meta_resident = (cache->resident_mask >> SETH_RES_META) & 1;
    return 1;
}

static void Set_Hamiltonian_Free_OpenACC_MatrixElements_Cache(void)
{
    SetHamiltonianMatrixElementsCache *cache = &Set_Hamiltonian_ME_Cache;

    Set_Hamiltonian_ME_ReleaseDeviceCache();
    free(cache->pair_Mc_AN);
    free(cache->pair_h_AN);
    free(cache->pair_NO0);
    free(cache->pair_NO1);
    free(cache->pair_NOLG);
    free(cache->nolg_MN);
    free(cache->nolg_Nc);
    free(cache->pair_h_offset);
    free(cache->pair_nolg_offset);
    free(cache->pair_orbs0_offset);
    free(cache->pair_orbs1_offset);
    free(cache->orbs0buf);
    free(cache->orbs1buf);
    memset(cache, 0, sizeof(*cache));
}

void Set_Hamiltonian_Invalidate_OpenACC_MatrixElements_Cache(void)
{
    Set_Hamiltonian_Free_OpenACC_MatrixElements_Cache();
}

static void Set_Hamiltonian_GpuSolver_Free_Cache(void)
{
    SetHamiltonianGpuSolverPackedCache *cache = &Set_Hamiltonian_GpuSolver_Cache;
    int i;

    free(cache->order_GA);
    free(cache->overlap);
    for (i = 0; i < 4; i++) {
        free(cache->h[i]);
    }
    for (i = 0; i < 3; i++) {
        free(cache->imnl[i]);
    }

    memset(cache, 0, sizeof(*cache));
}

void Set_Hamiltonian_Invalidate_GpuSolver_HS_Cache(void)
{
    Set_Hamiltonian_GpuSolver_Free_Cache();
}

static int Set_Hamiltonian_GpuSolver_LocalPackedSize(int myid)
{
    size_t total = 0;

    for (int MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        int GA_AN = M2G[MA_AN];
        int wanA = WhatSpecies[GA_AN];
        int tnoA = Spe_Total_CNO[wanA];
        size_t neighbor_orbitals = 0;

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int wanB = WhatSpecies[GB_AN];
            int tnoB = Spe_Total_CNO[wanB];

            neighbor_orbitals = Set_Hamiltonian_checked_add(neighbor_orbitals, (size_t)tnoB,
                                                            "GpuSolver packed neighbor orbitals", myid);
        }

        total = Set_Hamiltonian_checked_add(
            total,
            Set_Hamiltonian_checked_mul((size_t)tnoA, neighbor_orbitals,
                                        "GpuSolver packed local matrix segment", myid),
            "GpuSolver packed local matrix segment", myid);
    }

    if ((size_t)INT_MAX < total) {
        Set_Hamiltonian_abort("GpuSolver packed cache", "local packed matrix segment exceeds INT_MAX", myid);
    }

    return (int)total;
}

static void Set_Hamiltonian_GpuSolver_PackLocalMatrix(double ****mat, double *local, int local_size, int order_mode,
                                                     int myid)
{
    int k = 0;

    if (mat == NULL) {
        Set_Hamiltonian_abort("GpuSolver packed cache", "NULL sparse matrix", myid);
    }

    for (int MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        int GA_AN = M2G[MA_AN];
        int wanA = WhatSpecies[GA_AN];
        int tnoA = Spe_Total_CNO[wanA];

        if (order_mode == SET_HAMILTONIAN_PACK_ORDER_COL) {
            for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                int GB_AN = natn[GA_AN][LB_AN];
                int wanB = WhatSpecies[GB_AN];
                int tnoB = Spe_Total_CNO[wanB];

                for (int i = 0; i < tnoA; i++) {
                    for (int j = 0; j < tnoB; j++) {
                        local[k++] = mat[MA_AN][LB_AN][i][j];
                    }
                }
            }
        }
        else {
            for (int i = 0; i < tnoA; i++) {
                for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                    int GB_AN = natn[GA_AN][LB_AN];
                    int wanB = WhatSpecies[GB_AN];
                    int tnoB = Spe_Total_CNO[wanB];

                    for (int j = 0; j < tnoB; j++) {
                        local[k++] = mat[MA_AN][LB_AN][i][j];
                    }
                }
            }
        }
    }

    if (k != local_size) {
        Set_Hamiltonian_abort("GpuSolver packed cache", "packed local matrix size mismatch", myid);
    }
}

static void Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(double ****mat, double **target, int order_mode,
                                                             int local_size, int total_size, const int *counts,
                                                            const int *displs, const int *selected_roots,
                                                            MPI_Comm selected_comm, int first_selected_root,
                                                            int myid)
{
    double *local;
    double *recvbuf = NULL;

    *target = NULL;
    local = (double *)Set_Hamiltonian_malloc(sizeof(double) * (size_t)(0 < local_size ? local_size : 1),
                                             "GpuSolver packed local matrix", myid);
    Set_Hamiltonian_GpuSolver_PackLocalMatrix(mat, local, local_size, order_mode, myid);

    if (myid == first_selected_root) {
        recvbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * (size_t)(0 < total_size ? total_size : 1),
                                                   "GpuSolver packed matrix cache", myid);
    }

    MPI_Gatherv(local, local_size, MPI_DOUBLE, recvbuf, (int *)counts, (int *)displs, MPI_DOUBLE, first_selected_root,
                mpi_comm_level1);

    if (selected_roots[myid]) {
        if (myid == first_selected_root) {
            *target = recvbuf;
        }
        else {
            *target = (double *)Set_Hamiltonian_malloc(sizeof(double) * (size_t)(0 < total_size ? total_size : 1),
                                                       "GpuSolver packed matrix cache", myid);
        }
        MPI_Bcast(*target, total_size, MPI_DOUBLE, 0, selected_comm);
    }

    free(local);
}

void Set_Hamiltonian_GpuSolver_SetMP(int *MP)
{
    int Anum = 1;

    if (MP == NULL) {
        return;
    }

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int wanA = WhatSpecies[GA_AN];
        MP[GA_AN] = Anum;
        Anum += Spe_Total_CNO[wanA];
    }
}

void Set_Hamiltonian_Build_GpuSolver_HS_Cache(int use_contracted)
{
    SetHamiltonianGpuSolverPackedCache *cache = &Set_Hamiltonian_GpuSolver_Cache;
    int myid, numprocs;
    int local_selected;
    int selected_count = 0;
    int first_selected_root = -1;
    int local_size, total_size = 0;
    int local_matomnum;
    int *selected_roots = NULL;
    int *counts = NULL;
    int *displs = NULL;
    int *atom_counts = NULL;
    int *atom_displs = NULL;
    int dummy_atom = 0;
    int order_mode;
    int spin_count;
    MPI_Comm selected_comm = MPI_COMM_NULL;
    double ****overlap_src;
    double *****h_src;
    double *****imnl_src;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    Set_Hamiltonian_GpuSolver_Free_Cache();

    if (scf_eigen_lib_flag != GPUSOLVER) {
        return;
    }

    order_mode = (SpinP_switch == 3) ? SET_HAMILTONIAN_PACK_ORDER_NONCOL : SET_HAMILTONIAN_PACK_ORDER_COL;
    spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);
    local_selected = Set_Hamiltonian_OpenACC_Rank_Selected ? 1 : 0;
    local_size = Set_Hamiltonian_GpuSolver_LocalPackedSize(myid);
    local_matomnum = Matomnum;

    selected_roots = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)numprocs, "GpuSolver selected roots", myid);
    counts = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)numprocs, "GpuSolver packed counts", myid);
    displs = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)numprocs, "GpuSolver packed displacements", myid);
    atom_counts = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)numprocs, "GpuSolver atom counts", myid);
    atom_displs = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)numprocs, "GpuSolver atom displacements", myid);

    MPI_Allgather(&local_selected, 1, MPI_INT, selected_roots, 1, MPI_INT, mpi_comm_level1);
    MPI_Allgather(&local_size, 1, MPI_INT, counts, 1, MPI_INT, mpi_comm_level1);
    MPI_Allgather(&local_matomnum, 1, MPI_INT, atom_counts, 1, MPI_INT, mpi_comm_level1);

    for (int ID = 0; ID < numprocs; ID++) {
        if (selected_roots[ID]) {
            if (first_selected_root < 0) {
                first_selected_root = ID;
            }
            selected_count++;
        }
        if (counts[ID] < 0 || atom_counts[ID] < 0) {
            Set_Hamiltonian_abort("GpuSolver packed cache", "negative gather count", myid);
        }
        if ((size_t)INT_MAX < (size_t)total_size + (size_t)counts[ID]) {
            Set_Hamiltonian_abort("GpuSolver packed cache", "packed matrix size exceeds INT_MAX", myid);
        }
        displs[ID] = total_size;
        total_size += counts[ID];
    }

    {
        int atom_total = 0;
        for (int ID = 0; ID < numprocs; ID++) {
            if ((size_t)INT_MAX < (size_t)atom_total + (size_t)atom_counts[ID]) {
                Set_Hamiltonian_abort("GpuSolver packed cache", "atom-order size exceeds INT_MAX", myid);
            }
            atom_displs[ID] = atom_total;
            atom_total += atom_counts[ID];
        }
        if (atom_total != atomnum) {
            Set_Hamiltonian_abort("GpuSolver packed cache", "atom-order length mismatch", myid);
        }
    }

    if (selected_count == 0) {
        free(atom_displs);
        free(atom_counts);
        free(displs);
        free(counts);
        free(selected_roots);
        return;
    }

    MPI_Comm_split(mpi_comm_level1, local_selected ? 1 : MPI_UNDEFINED, myid, &selected_comm);

    cache->ready = 1;
    cache->owns = local_selected;
    cache->order_mode = order_mode;
    cache->size = total_size;
    cache->spin_count = spin_count;
    cache->atom_count = atomnum;

    if (myid == first_selected_root) {
        cache->order_GA = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)(atomnum + 2),
                                                        "GpuSolver atom order cache", myid);
    }

    MPI_Gatherv((0 < Matomnum) ? &M2G[1] : &dummy_atom, Matomnum, MPI_INT,
                (myid == first_selected_root) ? &cache->order_GA[1] : NULL,
                atom_counts, atom_displs, MPI_INT, first_selected_root, mpi_comm_level1);

    if (local_selected) {
        if (myid != first_selected_root) {
            cache->order_GA = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)(atomnum + 2),
                                                            "GpuSolver atom order cache", myid);
        }
        cache->order_GA[0] = 0;
        cache->order_GA[atomnum + 1] = 0;
        MPI_Bcast(&cache->order_GA[1], atomnum, MPI_INT, 0, selected_comm);
    }

    overlap_src = use_contracted ? CntOLP[0] : OLP[0];
    h_src = use_contracted ? CntH : H;
    imnl_src = use_contracted ? iCntHNL : iHNL;

    Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(overlap_src, &cache->overlap, order_mode, local_size, total_size,
                                                    counts, displs, selected_roots, selected_comm,
                                                    first_selected_root, myid);

    for (int spin = 0; spin < spin_count; spin++) {
        Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(h_src[spin], &cache->h[spin], order_mode, local_size,
                                                        total_size, counts, displs, selected_roots, selected_comm,
                                                        first_selected_root, myid);
    }

    if (SpinP_switch == 3 && imnl_src != NULL) {
        for (int comp = 0; comp < 3; comp++) {
            Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(imnl_src[comp], &cache->imnl[comp], order_mode,
                                                            local_size, total_size, counts, displs, selected_roots,
                                                            selected_comm, first_selected_root, myid);
        }
    }

    if (selected_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&selected_comm);
    }

    free(atom_displs);
    free(atom_counts);
    free(displs);
    free(counts);
    free(selected_roots);
}

int Set_Hamiltonian_GpuSolver_Packed_CacheReady(void)
{
    return Set_Hamiltonian_GpuSolver_Cache.ready;
}

int Set_Hamiltonian_GpuSolver_Packed_OwnsCache(void)
{
    SetHamiltonianGpuSolverPackedCache *cache = &Set_Hamiltonian_GpuSolver_Cache;
    return (cache->ready && cache->owns);
}

int Set_Hamiltonian_GpuSolver_Packed_OrderMode(void)
{
    return Set_Hamiltonian_GpuSolver_Cache.order_mode;
}

int Set_Hamiltonian_GpuSolver_Packed_Size(void)
{
    return Set_Hamiltonian_GpuSolver_Cache.size;
}

int *Set_Hamiltonian_GpuSolver_Packed_OrderGA(void)
{
    return Set_Hamiltonian_GpuSolver_Cache.order_GA;
}

double *Set_Hamiltonian_GpuSolver_Packed_Overlap(void)
{
    return Set_Hamiltonian_GpuSolver_Cache.overlap;
}

double *Set_Hamiltonian_GpuSolver_Packed_H(int spin)
{
    if (spin < 0 || 4 <= spin) {
        return NULL;
    }
    return Set_Hamiltonian_GpuSolver_Cache.h[spin];
}

double *Set_Hamiltonian_GpuSolver_Packed_ImNL(int comp)
{
    if (comp < 0 || 3 <= comp) {
        return NULL;
    }
    return Set_Hamiltonian_GpuSolver_Cache.imnl[comp];
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
    int    use_base_openacc;
    int    matrix_openacc_enabled;
    SetHamiltonianGpuTurnPlan base_plan;
    double prof_t0 = 0.0;

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

    memset(&base_plan, 0, sizeof(base_plan));
    base_plan.device_comm = MPI_COMM_NULL;
    matrix_openacc_enabled =
        (Set_Hamiltonian_OpenACC_Enabled() && Set_Hamiltonian_MatrixElements_OpenACC_Enabled());

    SETH_PROF_T0(prof_t0);
    {
        int enabled = Set_Hamiltonian_OpenACC_Enabled() && Set_Hamiltonian_Base_OpenACC_Enabled();
        int local_request = enabled && Set_Hamiltonian_OpenACC_Work_Rank_Selected;
        int any_request = 0;
        size_t required_bytes = (enabled && Set_Hamiltonian_OpenACC_Work_Rank_Selected) ?
            Set_Hamiltonian_Base_OpenACC_DeviceBytes(SCF_iter, myid) : 0;

        /* CreateGpuTurnPlan contains mpi_comm_level1 collectives.  Every rank
           must enter it even when a rank-local GPU initialization failure has
           switched that rank away from GPUSOLVER. */
        MPI_Allreduce(&local_request, &any_request, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
        if (any_request) {
            base_plan = Set_Hamiltonian_CreateGpuTurnPlan(required_bytes, NULL, 0, "base OpenACC path", myid);
        }
    }
    use_base_openacc = base_plan.use_gpu &&
        (Set_Hamiltonian_GpuSerialWaves() || base_plan.turn == 0);
    SETH_PROF_ADD(memok, prof_t0);

    if (myid == Host_ID && mode != NULL && strcasecmp(mode, "stdout") == 0 && 0 < level_stdout) {
        printf("<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc%s...\n",
               (use_base_openacc || matrix_openacc_enabled) ? " (GPU-accelerated)" : "");
        fflush(stdout);
    }

    /*****************************************************
                adding H0+HNL+(HCH) to H
  *****************************************************/

    if (measure_time)
        dtime(&time1);

    SETH_PROF_T0(prof_t0);

    if (base_plan.use_gpu && Set_Hamiltonian_GpuSerialWaves()) {
        for (int turn = 0; turn < base_plan.turns; turn++) {
            MPI_Barrier(base_plan.device_comm);
            if (base_plan.turn == turn) {
                Set_Hamiltonian_Base_OpenACC(SCF_iter, H0, HNL, H);
                acc_wait_all();
                acc_clear_freelists();
            }
            MPI_Barrier(base_plan.device_comm);
        }
    }
    else if (use_base_openacc) {
        Set_Hamiltonian_Base_OpenACC(SCF_iter, H0, HNL, H);
        acc_wait_all();
        acc_clear_freelists();
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

                        if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch ||
                             Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1) &&
                            F_U_flag == 1 && 2 <= SCF_iter) {
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

    Set_Hamiltonian_DestroyGpuTurnPlan(&base_plan);
    SETH_PROF_ADD(base, prof_t0);

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
    SETH_PROF_T0(prof_t0);
    Set_Vpot(MD_iter, SCF_iter, SCF_iter0, TRAN_Poisson_flag2, XC_P_switch);
    SETH_PROF_ADD(vpot, prof_t0);

    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time2=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }

    /*****************************************************
   calculation of matrix elements for dVH + Vxc (+ VNA)
  *****************************************************/

    SETH_PROF_T0(prof_t0);
    Calc_MatrixElements_dVH_Vxc_VNA(Cnt_kind);
    SETH_PROF_ADD(matel, prof_t0);

    /* for time */
    if (measure_time)
        dtime(&time1);
    SETH_PROF_T0(prof_t0);
    MPI_Barrier(mpi_comm_level1);
    SETH_PROF_ADD(barrier, prof_t0);
    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time Barrier=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }

    if (SetH_ProfileEnabled()) {
        fprintf(stderr, "SETHPROF id=%d it=%d memok=%.3f base=%.3f vpot=%.3f matel=%.3f bar=%.3f\n", myid, SCF_iter,
                SetH_prof.memok, SetH_prof.base, SetH_prof.vpot, SetH_prof.matel, SetH_prof.barrier);
        memset(&SetH_prof, 0, sizeof(SetH_prof));
    }

    dtime(&TEtime);
    time0 = TEtime - TStime;
    return time0;
}

void Calc_MatrixElements_dVH_Vxc_VNA(int Cnt_kind)
{
    int myid;
    SetHamiltonianGpuTurnPlan plan;
    SetHamiltonianMatrixElementsWork work;

    MPI_Comm_rank(mpi_comm_level1, &myid);
    memset(&plan, 0, sizeof(plan));
    memset(&work, 0, sizeof(work));
    plan.device_comm = MPI_COMM_NULL;

    {
        int enabled = Set_Hamiltonian_OpenACC_Enabled() && Set_Hamiltonian_MatrixElements_OpenACC_Enabled();
        int local_request = enabled && Set_Hamiltonian_OpenACC_Work_Rank_Selected;
        int any_request = 0;
        int resident_active = 0;
        size_t required_bytes = 0;
        size_t resident_request_class[SETH_RES_NCLASS] = {0, 0, 0, 0};

        Set_Hamiltonian_GpuTestNeed_Now = 0;

        if (local_request) {
            SetHamiltonianMatrixElementsCache *cache = &Set_Hamiltonian_ME_Cache;
            int cache_ok = Set_Hamiltonian_ME_CacheMatches(cache, Cnt_kind);
            int resident_mask = cache_ok ? cache->resident_mask : 0;
            int blocked = cache_ok ? cache->resident_blocked : 0;
            int want_more = Set_Hamiltonian_GpuResidentEnabled() && !blocked;
            static int call_index = 0;

            call_index++;
            Set_Hamiltonian_GpuTestNeed_Now = Set_Hamiltonian_GpuTestNeedBytes(call_index);
            resident_active = (resident_mask != 0);

            if (resident_mask != 0) {
                /* Some classes already live on the device: charge the
                   per-iteration staging plus the classes that still travel,
                   and offer the missing classes for admission. */
                size_t class_bytes[SETH_RES_NCLASS];

                Set_Hamiltonian_MatrixElements_ResidentClassBytes(Cnt_kind, myid, class_bytes);
                required_bytes = Set_Hamiltonian_MatrixElements_TransientBytes(cache, myid);
                for (int c = 0; c < SETH_RES_NCLASS; c++) {
                    if (!(resident_mask & (1 << c))) {
                        required_bytes = Set_Hamiltonian_checked_add(required_bytes, class_bytes[c],
                                                                     "matrix-elements staged classes", myid);
                        if (want_more) resident_request_class[c] = class_bytes[c];
                    }
                }
            }
            else {
                required_bytes = Set_Hamiltonian_MatrixElements_OpenACC_DeviceBytes(Cnt_kind, myid);
                if (want_more) {
                    Set_Hamiltonian_MatrixElements_ResidentClassBytes(Cnt_kind, myid, resident_request_class);
                }
            }
        }

        /* This routine performs node- and device-level MPI collectives, so all
           ranks enter it irrespective of their local accelerator state. */
        MPI_Allreduce(&local_request, &any_request, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
        if (any_request) {
            plan = Set_Hamiltonian_CreateGpuTurnPlan(required_bytes, resident_request_class,
                                                     resident_active, "matrix-elements OpenACC path", myid);
        }
    }

    if (plan.use_gpu && Set_Hamiltonian_GpuSerialWaves()) {
        /* The orbital/topology cache is host-only.  Build it concurrently on
           all selected ranks before serializing access to the physical GPU;
           otherwise its first-use cost is repeated on the critical path of
           every wave. */
        Set_Hamiltonian_Ensure_OpenACC_MatrixElements_Cache(Cnt_kind, myid);
        if (plan.resident_admit_mask != 0) Set_Hamiltonian_ME_EnterDeviceCache(&plan);
        Set_Hamiltonian_Prepare_OpenACC_MatrixElements(&work, Cnt_kind, myid);
        for (int turn = 0; turn < plan.turns; turn++) {
            MPI_Barrier(plan.device_comm);
            if (plan.turn == turn) {
                Set_Hamiltonian_Run_OpenACC_MatrixElements(&work);
                acc_wait_all();
                acc_clear_freelists();
            }
            MPI_Barrier(plan.device_comm);
        }
        Set_Hamiltonian_Finish_OpenACC_MatrixElements(&work);
    }
    else if (plan.use_gpu && plan.turn == 0) {
        Set_Hamiltonian_Ensure_OpenACC_MatrixElements_Cache(Cnt_kind, myid);
        if (plan.resident_admit_mask != 0) Set_Hamiltonian_ME_EnterDeviceCache(&plan);
        Set_Hamiltonian_Prepare_OpenACC_MatrixElements(&work, Cnt_kind, myid);
        Set_Hamiltonian_Run_OpenACC_MatrixElements(&work);
        acc_wait_all();
        acc_clear_freelists();
        Set_Hamiltonian_Finish_OpenACC_MatrixElements(&work);
    }
    else {
        Calc_MatrixElements_dVH_Vxc_VNA_CPU(Cnt_kind);
    }

    if (plan.release_resident) {
        /* A registered GPU phase can no longer reach its full concurrency
           beside the resident tables: yield the memory now (the
           diagonalization follows within this very SCF iteration) and stay
           transient for the rest of this MD step. */
        Set_Hamiltonian_ME_ReleaseDeviceCache();
        Set_Hamiltonian_ME_Cache.resident_blocked = 1;
        if (plan.device_rank == 0) {
            printf("<Set_Hamiltonian> GPU resident cache released: a GPU phase needs %.3f GiB on device %d; "
                   "staying transient for this MD step\n",
                   (double)plan.registered_need_bytes / (1024.0 * 1024.0 * 1024.0),
                   plan.cuda_device);
            fflush(stdout);
        }
    }

    Set_Hamiltonian_DestroyGpuTurnPlan(&plan);
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

static void Set_Hamiltonian_MatrixElements_CountTotals(int Cnt_kind, int myid, int *pair_count_out,
                                                       size_t *total_h_out, size_t *total_nolg_out,
                                                       size_t *total_orbs0_out, size_t *total_orbs1_out)
{
    int Mc_AN, h_AN;
    int spin_count, pair_count;
    size_t total_h, total_nolg, total_orbs0, total_orbs1;

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
        int central_NO0 = (Cnt_kind == 0) ? Spe_Total_NO[Cwan] : Spe_Total_CNO[Cwan];

        total_orbs0 = Set_Hamiltonian_checked_add(
            total_orbs0,
            Set_Hamiltonian_checked_mul((size_t)GridN_Atom[Gc_AN], (size_t)central_NO0,
                                        "matrix-elements unique orbs0 size", myid),
            "matrix-elements total unique orbs0", myid);

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int Gh_AN = natn[Gc_AN][h_AN];
            int Hwan = WhatSpecies[Gh_AN];
            int NOLG = NumOLG[Mc_AN][h_AN];
            int NO0, NO1;
            size_t mat_size, h_size, orbs1_size;

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
            orbs1_size = Set_Hamiltonian_checked_mul((size_t)NOLG, (size_t)NO1,
                                                     "matrix-elements orbs1 size", myid);

            total_h = Set_Hamiltonian_checked_add(total_h, h_size, "matrix-elements total Hamiltonian", myid);
            total_nolg = Set_Hamiltonian_checked_add(total_nolg, (size_t)NOLG,
                                                     "matrix-elements total NOLG", myid);
            total_orbs1 = Set_Hamiltonian_checked_add(total_orbs1, orbs1_size,
                                                      "matrix-elements total orbs1", myid);
            pair_count++;
        }
    }

    *pair_count_out = pair_count;
    *total_h_out = total_h;
    *total_nolg_out = total_nolg;
    *total_orbs0_out = total_orbs0;
    *total_orbs1_out = total_orbs1;
}

static int Set_Hamiltonian_ME_CacheMatches(const SetHamiltonianMatrixElementsCache *cache, int Cnt_kind)
{
    int spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);

    return (cache->ready && cache->cnt_kind == Cnt_kind && cache->spin_count == spin_count &&
            cache->matomnum == Matomnum);
}

/* Per-class bytes of the SCF-invariant arrays that device residency can
   hold: the orbital tables, the grid index tables and the pair metadata. */
static void Set_Hamiltonian_MatrixElements_ResidentClassBytes(int Cnt_kind, int myid,
                                                              size_t class_bytes[SETH_RES_NCLASS])
{
    const SetHamiltonianMatrixElementsCache *cache = &Set_Hamiltonian_ME_Cache;
    int pair_count = 0;
    size_t total_h = 0, total_nolg = 0, total_orbs0 = 0, total_orbs1 = 0, bytes;
    int c;

    for (c = 0; c < SETH_RES_NCLASS; c++) class_bytes[c] = 0;
    if (Matomnum <= 0) return;

    if (Set_Hamiltonian_ME_CacheMatches(cache, Cnt_kind)) {
        pair_count = cache->pair_count;
        total_nolg = cache->total_nolg;
        total_orbs0 = cache->total_orbs0;
        total_orbs1 = cache->total_orbs1;
    }
    else {
        Set_Hamiltonian_MatrixElements_CountTotals(Cnt_kind, myid, &pair_count, &total_h, &total_nolg,
                                                   &total_orbs0, &total_orbs1);
    }

    bytes = 0;
    Set_Hamiltonian_add_array_bytes(&bytes, total_orbs1, sizeof(Type_Orbs_Grid), "matrix-elements orbs1buf", myid);
    class_bytes[SETH_RES_ORBS1] = bytes;

    bytes = 0;
    Set_Hamiltonian_add_array_bytes(&bytes, total_orbs0, sizeof(Type_Orbs_Grid), "matrix-elements orbs0buf", myid);
    class_bytes[SETH_RES_ORBS0] = bytes;

    bytes = 0;
    Set_Hamiltonian_add_array_bytes(&bytes, total_nolg, sizeof(int), "matrix-elements nolg_MN", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_nolg, sizeof(int), "matrix-elements nolg_Nc", myid);
    class_bytes[SETH_RES_NOLG] = bytes;

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
    class_bytes[SETH_RES_META] = bytes;
}

/* Bytes staged per SCF iteration: the Hamiltonian blocks (both ways) and the
   rank-local grid potential slice. */
static size_t Set_Hamiltonian_MatrixElements_TransientBytes(const SetHamiltonianMatrixElementsCache *cache,
                                                            int myid)
{
    size_t bytes = 0;

    Set_Hamiltonian_add_array_bytes(&bytes, cache->total_h, sizeof(double), "matrix-elements hbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes,
                                    Set_Hamiltonian_checked_mul((size_t)cache->spin_count, (size_t)My_NumGridC,
                                                                "matrix-elements vpotgrid", myid),
                                    sizeof(double), "matrix-elements vpotgrid", myid);
    return bytes;
}

static size_t Set_Hamiltonian_MatrixElements_OpenACC_DeviceBytes(int Cnt_kind, int myid)
{
    int pair_count = 0;
    size_t total_h = 0, total_nolg = 0, total_orbs0 = 0, total_orbs1 = 0, bytes;
    size_t rank_overhead_mb = 512u;
    const char *rank_overhead_env;

    if (Matomnum <= 0) return 0;

    Set_Hamiltonian_MatrixElements_CountTotals(Cnt_kind, myid, &pair_count, &total_h, &total_nolg,
                                               &total_orbs0, &total_orbs1);

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
    Set_Hamiltonian_add_array_bytes(&bytes, total_nolg, sizeof(int), "matrix-elements nolg_MN", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_nolg, sizeof(int), "matrix-elements nolg_Nc", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_h, sizeof(double), "matrix-elements hbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes,
                                    Set_Hamiltonian_checked_mul((size_t)((SpinP_switch == 3) ? 4 : (SpinP_switch + 1)),
                                                                (size_t)My_NumGridC,
                                                                "matrix-elements vpotgrid", myid),
                                    sizeof(double), "matrix-elements vpotgrid", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_orbs0, sizeof(Type_Orbs_Grid), "matrix-elements orbs0buf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_orbs1, sizeof(Type_Orbs_Grid), "matrix-elements orbs1buf", myid);

    /* OpenACC/CUDA runtime state and compiler-generated kernel storage are not
       represented by the explicit arrays above.  The default headroom is
       deliberately charged per concurrently active rank; it can be tuned for
       a particular compiler/runtime without weakening the per-device reserve. */
    rank_overhead_env = getenv("OPENMX_SETHAM_GPU_RANK_OVERHEAD_MB");
    if (rank_overhead_env != NULL) {
        char *end = NULL;
        unsigned long long parsed = strtoull(rank_overhead_env, &end, 10);
        if (end != rank_overhead_env && *end == '\0') rank_overhead_mb = (size_t)parsed;
    }
    Set_Hamiltonian_add_array_bytes(&bytes, rank_overhead_mb, 1024u * 1024u,
                                    "matrix-elements OpenACC per-rank overhead", myid);

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

static SetHamiltonianMatrixElementsCache *Set_Hamiltonian_Ensure_OpenACC_MatrixElements_Cache(int Cnt_kind,
                                                                                              int myid)
{
    SetHamiltonianMatrixElementsCache *cache = &Set_Hamiltonian_ME_Cache;
    int spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);
    int pair_count = 0;
    size_t total_h = 0, total_nolg = 0, total_orbs0 = 0, total_orbs1 = 0;

    if (Cnt_kind != 0 && Cnt_kind != 1) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA_OpenACC", "Cnt_kind must be 0 or 1", myid);
    }

    if (SpinP_switch != 0 && SpinP_switch != 1 && SpinP_switch != 3) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA_OpenACC", "SpinP_switch must be 0, 1, or 3", myid);
    }

    if (Set_Hamiltonian_ME_CacheMatches(cache, Cnt_kind)) {
        return cache;
    }

    Set_Hamiltonian_Free_OpenACC_MatrixElements_Cache();

    for (int Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int Cwan = WhatSpecies[Gc_AN];
        int central_NO0 = (Cnt_kind == 0) ? Spe_Total_NO[Cwan] : Spe_Total_CNO[Cwan];

        total_orbs0 += (size_t)GridN_Atom[Gc_AN] * (size_t)central_NO0;

        for (int h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int NO0, NO1, NOLG;
            int Gh_AN = natn[Gc_AN][h_AN];
            int Hwan = WhatSpecies[Gh_AN];
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
            total_orbs1 += (size_t)NOLG * (size_t)NO1;
            pair_count++;
        }
    }

    cache->pair_Mc_AN = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_Mc_AN", myid);
    cache->pair_h_AN = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_h_AN", myid);
    cache->pair_NO0 = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_NO0", myid);
    cache->pair_NO1 = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_NO1", myid);
    cache->pair_NOLG = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openacc pair_NOLG", myid);
    cache->nolg_MN = (int *)Set_Hamiltonian_malloc(sizeof(int) * total_nolg, "openacc host nolg_MN", myid);
    cache->nolg_Nc = (int *)Set_Hamiltonian_malloc(sizeof(int) * total_nolg, "openacc nolg_Nc", myid);
    cache->pair_h_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openacc pair_h_offset", myid);
    cache->pair_nolg_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openacc pair_nolg_offset", myid);
    cache->pair_orbs0_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openacc pair_orbs0_offset", myid);
    cache->pair_orbs1_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openacc pair_orbs1_offset", myid);
    cache->orbs0buf =
        (Type_Orbs_Grid *)Set_Hamiltonian_malloc(sizeof(Type_Orbs_Grid) * total_orbs0, "openacc orbs0buf", myid);
    cache->orbs1buf =
        (Type_Orbs_Grid *)Set_Hamiltonian_malloc(sizeof(Type_Orbs_Grid) * total_orbs1, "openacc orbs1buf", myid);

    int pair = 0;
    total_h = 0;
    total_nolg = 0;
    total_orbs0 = 0;
    total_orbs1 = 0;
    for (int Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int Cwan = WhatSpecies[Gc_AN];
        int central_NO0 = (Cnt_kind == 0) ? Spe_Total_NO[Cwan] : Spe_Total_CNO[Cwan];
        size_t central_orbs0_offset = total_orbs0;

        for (int Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {
            for (int i = 0; i < central_NO0; i++) {
                cache->orbs0buf[central_orbs0_offset + (size_t)Nc * (size_t)central_NO0 + (size_t)i] =
                    Orbs_Grid[Mc_AN][Nc][i];
            }
        }
        total_orbs0 += (size_t)GridN_Atom[Gc_AN] * (size_t)central_NO0;

        for (int h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int NO0, NO1, NOLG;
            size_t mat_size;
            int Gh_AN = natn[Gc_AN][h_AN];
            int Mh_AN = F_G2M[Gh_AN];
            int Hwan = WhatSpecies[Gh_AN];
            NOLG = NumOLG[Mc_AN][h_AN];

            if (Cnt_kind == 0) {
                NO0 = Spe_Total_NO[Cwan];
                NO1 = Spe_Total_NO[Hwan];
            } else {
                NO0 = Spe_Total_CNO[Cwan];
                NO1 = Spe_Total_CNO[Hwan];
            }

            mat_size = (size_t)NO0 * (size_t)NO1;

            cache->pair_Mc_AN[pair] = Mc_AN;
            cache->pair_h_AN[pair] = h_AN;
            cache->pair_NO0[pair] = NO0;
            cache->pair_NO1[pair] = NO1;
            cache->pair_NOLG[pair] = NOLG;
            cache->pair_h_offset[pair] = total_h;
            cache->pair_nolg_offset[pair] = total_nolg;
            cache->pair_orbs0_offset[pair] = central_orbs0_offset;
            cache->pair_orbs1_offset[pair] = total_orbs1;

            for (int Nog = 0; Nog < NOLG; Nog++) {
                int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
                int MN = MGridListAtom[Mc_AN][Nc];
                int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
                Type_Orbs_Grid *orbs1 = (G2ID[Gh_AN] == myid) ? Orbs_Grid[Mh_AN][Nh] : Orbs_Grid_FNAN[Mc_AN][h_AN][Nog];

                cache->nolg_MN[total_nolg + (size_t)Nog] = MN;
                cache->nolg_Nc[total_nolg + (size_t)Nog] = Nc;
                for (int j = 0; j < NO1; j++) {
                    cache->orbs1buf[total_orbs1 + (size_t)Nog * (size_t)NO1 + (size_t)j] = orbs1[j];
                }
            }

            total_h += (size_t)spin_count * mat_size;
            total_nolg += (size_t)NOLG;
            total_orbs1 += (size_t)NOLG * (size_t)NO1;
            pair++;
        }
    }

    cache->ready = 1;
    cache->cnt_kind = Cnt_kind;
    cache->spin_count = spin_count;
    cache->pair_count = pair_count;
    cache->matomnum = Matomnum;
    cache->total_h = total_h;
    cache->total_nolg = total_nolg;
    cache->total_orbs0 = total_orbs0;
    cache->total_orbs1 = total_orbs1;
    return cache;
}

static void Set_Hamiltonian_Prepare_OpenACC_MatrixElements(SetHamiltonianMatrixElementsWork *work,
                                                            int Cnt_kind, int myid)
{
    double start = 0.0, finish = 0.0;
    if (SetH_ProfileEnabled()) dtime(&start);

    SetHamiltonianMatrixElementsCache *cache =
        Set_Hamiltonian_Ensure_OpenACC_MatrixElements_Cache(Cnt_kind, myid);
    const int spin_count = cache->spin_count;
    const int pair_count = cache->pair_count;
    const size_t total_h = cache->total_h;
    const size_t vpot_len = (size_t)My_NumGridC;
    const size_t vpot_count = Set_Hamiltonian_checked_mul((size_t)spin_count, vpot_len,
                                                          "matrix-elements vpotgrid", myid);

    memset(work, 0, sizeof(*work));
    work->cache = cache;
    work->cnt_kind = Cnt_kind;
    work->myid = myid;
    work->cuda_kernel_supported = 1;
    if (pair_count == 0) return;

    work->hbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * total_h, "openacc hbuf", myid);
    work->vpotgrid = (double *)Set_Hamiltonian_malloc(sizeof(double) * vpot_count, "openacc vpotgrid", myid);

    for (int pair = 0; pair < pair_count; pair++) {
        int Mc_AN = cache->pair_Mc_AN[pair];
        int h_AN = cache->pair_h_AN[pair];
        int NO0 = cache->pair_NO0[pair];
        int NO1 = cache->pair_NO1[pair];
        size_t mat_size = (size_t)NO0 * (size_t)NO1;
        size_t h_off = cache->pair_h_offset[pair];

        if (work->max_no < NO0) work->max_no = NO0;
        if (work->max_no < NO1) work->max_no = NO1;
        if ((size_t)INT_MAX < (size_t)spin_count * mat_size) {
            work->cuda_kernel_supported = 0;
        }
        else if (work->max_output_count < (int)((size_t)spin_count * mat_size)) {
            work->max_output_count = (int)((size_t)spin_count * mat_size);
        }

        for (int spin = 0; spin < spin_count; spin++) {
            for (int i = 0; i < NO0; i++) {
                for (int j = 0; j < NO1; j++) {
                    size_t idx = h_off + (size_t)spin * mat_size + (size_t)i * (size_t)NO1 + (size_t)j;
                    work->hbuf[idx] =
                        (Cnt_kind == 0) ? H[spin][Mc_AN][h_AN][i][j] : CntH[spin][Mc_AN][h_AN][i][j];
                }
            }
        }
    }

    /* The device kernel gathers the potential directly through nolg_MN, so
       only the rank-local grid slice is staged: one contiguous row per spin.
       GridVol is applied inside the kernel with the same rounding order as
       the old host-side expansion. */
    for (int spin = 0; spin < spin_count; spin++) {
        memcpy(work->vpotgrid + (size_t)spin * vpot_len, Vpot_Grid[spin], sizeof(double) * vpot_len);
    }

    if (SetH_ProfileEnabled()) {
        dtime(&finish);
        work->pack_seconds = finish - start;
    }
}

static void Set_Hamiltonian_Run_OpenACC_MatrixElements(SetHamiltonianMatrixElementsWork *work)
{
    SetHamiltonianMatrixElementsCache *cache = work->cache;
    const int spin_count = cache->spin_count;
    const int pair_count = cache->pair_count;
    const size_t total_h = cache->total_h;
    const size_t total_nolg = cache->total_nolg;
    const size_t total_orbs0 = cache->total_orbs0;
    const size_t total_orbs1 = cache->total_orbs1;
    const size_t vpot_len = (size_t)My_NumGridC;
    const size_t vpot_count = (size_t)spin_count * vpot_len;
    const double grid_vol = GridVol;
    int *pair_NO0 = cache->pair_NO0;
    int *pair_NO1 = cache->pair_NO1;
    int *pair_NOLG = cache->pair_NOLG;
    int *nolg_MN = cache->nolg_MN;
    int *nolg_Nc = cache->nolg_Nc;
    size_t *pair_h_offset = cache->pair_h_offset;
    size_t *pair_nolg_offset = cache->pair_nolg_offset;
    size_t *pair_orbs0_offset = cache->pair_orbs0_offset;
    size_t *pair_orbs1_offset = cache->pair_orbs1_offset;
    Type_Orbs_Grid *orbs0buf = cache->orbs0buf;
    Type_Orbs_Grid *orbs1buf = cache->orbs1buf;
    double *hbuf = work->hbuf;
    double *vpotgrid = work->vpotgrid;
    double start = 0.0, finish = 0.0;

    if (pair_count == 0) return;
    if (SetH_ProfileEnabled()) dtime(&start);

    /* The copyin clauses have present_or_copyin semantics: when the resident
       cache holds the SCF-invariant arrays on the device they only attach,
       so each iteration transfers the potential slice and the Hamiltonian
       blocks alone. */
#pragma acc data copy(hbuf[0:total_h])                                                                                      \
    copyin(pair_NO0[0:pair_count], pair_NO1[0:pair_count], pair_NOLG[0:pair_count],                                       \
           nolg_MN[0:total_nolg], nolg_Nc[0:total_nolg],                                                                   \
           pair_h_offset[0:pair_count], pair_nolg_offset[0:pair_count],                                                    \
           pair_orbs0_offset[0:pair_count], pair_orbs1_offset[0:pair_count],                                               \
           orbs0buf[0:total_orbs0], orbs1buf[0:total_orbs1], vpotgrid[0:vpot_count])
    {
        int cuda_status = 1;

        if (work->cuda_kernel_supported && Set_Hamiltonian_MatrixElements_CudaKernel_Enabled()) {
#pragma acc host_data use_device(hbuf, pair_NO0, pair_NO1, pair_NOLG, nolg_MN, nolg_Nc, pair_h_offset,                    \
                                 pair_nolg_offset, pair_orbs0_offset, pair_orbs1_offset, orbs0buf, orbs1buf, vpotgrid)
            cuda_status = Set_Hamiltonian_Cuda_MatrixElements(
                pair_count, spin_count, vpot_len, grid_vol, work->max_no, work->max_output_count,
                pair_NO0, pair_NO1, pair_NOLG, nolg_MN, nolg_Nc, pair_h_offset, pair_nolg_offset,
                pair_orbs0_offset, pair_orbs1_offset, orbs0buf, orbs1buf, vpotgrid, hbuf);
            if (cuda_status < 0) {
                char message[160];
                snprintf(message, sizeof(message), "CUDA matrix-elements kernel failed with unrecoverable status %d", cuda_status);
                Set_Hamiltonian_abort("Set_Hamiltonian_Run_OpenACC_MatrixElements", message, work->myid);
            }
            if (cuda_status == 2) {
                /* Transient failure (typically device memory pressure).  The
                   kernel may have partially updated the device H blocks, so
                   restore them from the untouched host copy before redoing
                   this batch with the OpenACC kernel below. */
                static int retry_reports = 0;
                if (retry_reports < 3) {
                    retry_reports++;
                    fprintf(stderr,
                            "<Set_Hamiltonian> rank %d: CUDA matrix-elements kernel failed transiently; redoing this batch with the OpenACC kernel.\n",
                            work->myid);
                    fflush(stderr);
                }
                acc_update_device(hbuf, sizeof(double) * total_h);
            }
        }

        if (cuda_status != 0) {
#pragma acc parallel loop gang present(hbuf[0:total_h], pair_NO0[0:pair_count], pair_NO1[0:pair_count],                    \
                                           nolg_MN[0:total_nolg], nolg_Nc[0:total_nolg],                                   \
                                           pair_NOLG[0:pair_count], pair_h_offset[0:pair_count],                            \
                                           pair_nolg_offset[0:pair_count], pair_orbs0_offset[0:pair_count],                 \
                                           pair_orbs1_offset[0:pair_count], orbs0buf[0:total_orbs0],                        \
                                           orbs1buf[0:total_orbs1], vpotgrid[0:vpot_count])
            for (int pair = 0; pair < pair_count; pair++) {
                int NO0 = pair_NO0[pair];
                int NO1 = pair_NO1[pair];
                int NOLG = pair_NOLG[pair];
                size_t mat_size = (size_t)NO0 * (size_t)NO1;
                size_t h_off = pair_h_offset[pair];
                size_t nolg_off = pair_nolg_offset[pair];
                size_t orbs0_off = pair_orbs0_offset[pair];
                size_t orbs1_off = pair_orbs1_offset[pair];

#pragma acc loop vector
                for (size_t e = 0; e < (size_t)spin_count * mat_size; e++) {
                    int spin = (int)(e / mat_size);
                    size_t ij = e - (size_t)spin * mat_size;
                    int i = (int)(ij / (size_t)NO1);
                    int j = (int)(ij - (size_t)i * (size_t)NO1);
                    size_t hidx = h_off + e;
                    double sum = hbuf[hidx];

#pragma acc loop seq
                    for (int Nog = 0; Nog < NOLG; Nog++) {
                        sum += (grid_vol * vpotgrid[(size_t)spin * vpot_len +
                                                    (size_t)nolg_MN[nolg_off + (size_t)Nog]]) *
                               orbs0buf[orbs0_off + (size_t)nolg_Nc[nolg_off + (size_t)Nog] * (size_t)NO0 + (size_t)i] *
                               orbs1buf[orbs1_off + (size_t)Nog * (size_t)NO1 + (size_t)j];
                    }
                    hbuf[hidx] = sum;
                }
            }
        }
    }

    if (SetH_ProfileEnabled()) {
        dtime(&finish);
        work->device_seconds += finish - start;
    }
}

static void Set_Hamiltonian_Finish_OpenACC_MatrixElements(SetHamiltonianMatrixElementsWork *work)
{
    SetHamiltonianMatrixElementsCache *cache = work->cache;
    const int spin_count = cache->spin_count;
    const int pair_count = cache->pair_count;
    double start = 0.0, finish = 0.0;

    if (SetH_ProfileEnabled()) dtime(&start);
    for (int pair = 0; pair < pair_count; pair++) {
        int NO0 = cache->pair_NO0[pair];
        int NO1 = cache->pair_NO1[pair];
        size_t mat_size = (size_t)NO0 * (size_t)NO1;
        size_t h_off = cache->pair_h_offset[pair];
        int Mc_AN = cache->pair_Mc_AN[pair];
        int h_AN = cache->pair_h_AN[pair];

        for (int spin = 0; spin < spin_count; spin++) {
            for (int i = 0; i < NO0; i++) {
                for (int j = 0; j < NO1; j++) {
                    size_t idx = h_off + (size_t)spin * mat_size + (size_t)i * (size_t)NO1 + (size_t)j;
                    if (work->cnt_kind == 0) H[spin][Mc_AN][h_AN][i][j] = work->hbuf[idx];
                    else CntH[spin][Mc_AN][h_AN][i][j] = work->hbuf[idx];
                }
            }
        }
    }

    free(work->vpotgrid);
    free(work->hbuf);
    work->vpotgrid = NULL;
    work->hbuf = NULL;

    if (SetH_ProfileEnabled()) {
        dtime(&finish);
        fprintf(stderr, "SETHMEPROF id=%d cache_pack=%.3f device=%.3f unpack=%.3f\n",
                work->myid, work->pack_seconds, work->device_seconds, finish - start);
        fflush(stderr);
    }
}

#define SETH_ME_BLK 16
#define SETH_ME_MAXNO 16

/* Blocked spin-unpolarized quadrature for one (Mc_AN,h_AN) pair: grid points
   are staged in panels of SETH_ME_BLK and the NO0xNO1 accumulator block stays
   L1-resident for the whole pair, instead of re-storing all NO0*NO1 partial
   sums for every grid point as the generic loop below does. The j dimension
   is padded to SETH_ME_MAXNO so the inner update vectorizes with full-width
   FMAs. Only the summation order differs from the generic path. */
static void Set_Hamiltonian_ME_Pair_Blocked_Spin0(int Mc_AN, int h_AN, int Mh_AN, int Gh_AN_is_local, int NO0, int NO1,
                                                  int NOLG, double *restrict acc)
{
    const int *restrict nc_list = GListTAtoms1[Mc_AN][h_AN];
    const int *restrict nh_list = GListTAtoms2[Mc_AN][h_AN];
    const int *restrict mgrid   = MGridListAtom[Mc_AN];
    const double *restrict vpot = Vpot_Grid[0];
    Type_Orbs_Grid **restrict og_c = Orbs_Grid[Mc_AN];
    Type_Orbs_Grid **restrict og_h = Gh_AN_is_local ? Orbs_Grid[Mh_AN] : NULL;
    Type_Orbs_Grid **restrict og_f = Gh_AN_is_local ? NULL : Orbs_Grid_FNAN[Mc_AN][h_AN];
    double w[SETH_ME_BLK][SETH_ME_MAXNO];
    double p1[SETH_ME_BLK][SETH_ME_MAXNO];
    int    Nog0, g, i, j;

    for (i = 0; i < NO0 * SETH_ME_MAXNO; i++) {
        acc[i] = 0.0;
    }

    for (g = 0; g < SETH_ME_BLK; g++) {
        for (j = NO1; j < SETH_ME_MAXNO; j++) {
            p1[g][j] = 0.0;
        }
    }

    for (Nog0 = 0; Nog0 < NOLG; Nog0 += SETH_ME_BLK) {
        const int blk = (NOLG - Nog0 < SETH_ME_BLK) ? (NOLG - Nog0) : SETH_ME_BLK;

        for (g = 0; g < blk; g++) {
            const int Nog = Nog0 + g;
            const int Nc  = nc_list[Nog];
            const Type_Orbs_Grid *restrict orbs0 = og_c[Nc];
            const Type_Orbs_Grid *restrict orbs1 = og_h ? og_h[nh_list[Nog]] : og_f[Nog];
            const double v = GridVol * vpot[mgrid[Nc]];

            for (i = 0; i < NO0; i++) {
                w[g][i] = v * (double)orbs0[i];
            }
            for (j = 0; j < NO1; j++) {
                p1[g][j] = (double)orbs1[j];
            }
        }

        for (g = 0; g < blk; g++) {
            const double *restrict p1g = p1[g];
            const double *restrict wg  = w[g];

            for (i = 0; i < NO0; i++) {
                const double a = wg[i];
                double *restrict accrow = &acc[i * SETH_ME_MAXNO];

                for (j = 0; j < SETH_ME_MAXNO; j++) {
                    accrow[j] += a * p1g[j];
                }
            }
        }
    }
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
            if (SpinP_switch == 0 && NO0 <= SETH_ME_MAXNO && NO1 <= SETH_ME_MAXNO) {
                double acc[SETH_ME_MAXNO * SETH_ME_MAXNO] __attribute__((aligned(64)));
                int    i, j;

                Set_Hamiltonian_ME_Pair_Blocked_Spin0(Mc_AN, h_AN, Mh_AN, Gh_AN_is_local, NO0, NO1, NOLG, acc);

                if (Cnt_kind == 0) {
                    for (i = 0; i < NO0; i++) {
                        for (j = 0; j < NO1; j++) {
                            H[0][Mc_AN][h_AN][i][j] += acc[i * SETH_ME_MAXNO + j];
                        }
                    }
                } else {
                    for (i = 0; i < NO0; i++) {
                        for (j = 0; j < NO1; j++) {
                            CntH[0][Mc_AN][h_AN][i][j] += acc[i * SETH_ME_MAXNO + j];
                        }
                    }
                }

            } else if (SpinP_switch == 0) {
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
                            AI_tmpH[1][i][j] = H[1][Mc_AN][h_AN][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
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

                        double orb0 = (double)orbs0[i];
                        double AI_tmp_i0 = AI_tmp_GVVG  * orb0;
                        double AI_tmp_i1 = AI_tmp_GVVG1 * orb0;
                        double *tmp0 = AI_tmpH[0][i];
                        double *tmp1 = AI_tmpH[1][i];
                        int    j;

                        for (j = 0; j < NO1; j++) {
                            double orb1 = (double)orbs1[j];
                            tmp0[j] += AI_tmp_i0 * orb1;
                            tmp1[j] += AI_tmp_i1 * orb1;
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
                            H[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
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

                        double orb0 = (double)orbs0[i];
                        double AI_tmp_i0 = AI_tmp_GVVG  * orb0;
                        double AI_tmp_i1 = AI_tmp_GVVG1 * orb0;
                        double AI_tmp_i2 = AI_tmp_GVVG2 * orb0;
                        double AI_tmp_i3 = AI_tmp_GVVG3 * orb0;
                        double *tmp0 = AI_tmpH[0][i];
                        double *tmp1 = AI_tmpH[1][i];
                        double *tmp2 = AI_tmpH[2][i];
                        double *tmp3 = AI_tmpH[3][i];
                        int    j;

                        for (j = 0; j < NO1; j++) {
                            double orb1 = (double)orbs1[j];
                            tmp0[j] += AI_tmp_i0 * orb1;
                            tmp1[j] += AI_tmp_i1 * orb1;
                            tmp2[j] += AI_tmp_i2 * orb1;
                            tmp3[j] += AI_tmp_i3 * orb1;
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
                            H[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
                            H[2][Mc_AN][h_AN][i][j] = AI_tmpH[2][i][j];
                            H[3][Mc_AN][h_AN][i][j] = AI_tmpH[3][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                            CntH[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
                            CntH[2][Mc_AN][h_AN][i][j] = AI_tmpH[2][i][j];
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
