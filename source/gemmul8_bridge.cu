#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

#include "gemmul8.hpp"

namespace {

constexpr unsigned kDefaultNumModuli = 15u;
constexpr unsigned kMaxNumModuli     = 20u;
constexpr unsigned kDefaultMinFreeAfterMiB = 1536u;
constexpr unsigned kDefaultMaxWorkspacePercent = 50u;
constexpr size_t   kMiB = 1024u * 1024u;

struct WorkspaceKey {
    int          device;
    cudaStream_t stream;

    bool operator==(const WorkspaceKey &other) const
    {
        return device == other.device && stream == other.stream;
    }
};

struct WorkspaceKeyHash {
    std::size_t operator()(const WorkspaceKey &key) const
    {
        return (static_cast<std::size_t>(key.device) << 32) ^ (reinterpret_cast<std::uintptr_t>(key.stream) << 1);
    }
};

struct Workspace {
    void *ptr   = nullptr;
    size_t size = 0;
};

std::mutex g_workspace_mutex;
std::unordered_map<WorkspaceKey, Workspace, WorkspaceKeyHash> g_workspaces;

struct WorkspaceReport {
    size_t      required_bytes = 0;
    size_t      free_bytes     = 0;
    size_t      total_bytes    = 0;
    size_t      reserve_bytes  = 0;
    unsigned    max_workspace_percent = 0;
    const char *reason = "allocation failure";
};

unsigned env_u32(const char *name, unsigned fallback)
{
    const char *value = std::getenv(name);
    char       *end   = nullptr;

    if (value == nullptr || *value == '\0') {
        return fallback;
    }

    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }

    return static_cast<unsigned>(parsed);
}

bool env_bool(const char *name, bool fallback)
{
    const char *value = std::getenv(name);

    if (value == nullptr || *value == '\0') {
        return fallback;
    }

    return value[0] == '1';
}

unsigned env_percent(const char *openmx_env, const char *gemmul8_env, unsigned fallback)
{
    unsigned percent = env_u32(gemmul8_env, fallback);
    percent          = env_u32(openmx_env, percent);

    if (100u < percent) {
        percent = 100u;
    }

    return percent;
}

size_t env_mib(const char *openmx_env, const char *gemmul8_env, unsigned fallback)
{
    unsigned mib = env_u32(gemmul8_env, fallback);
    mib          = env_u32(openmx_env, mib);

    return static_cast<size_t>(mib) * kMiB;
}

bool gemmul8_disabled(const char *openmx_env, const char *gemmul8_env)
{
    bool disabled = env_bool("GEMMUL8_DISABLE", false);
    disabled      = env_bool("OPENMX_GEMMUL8_DISABLE", disabled);
    disabled      = env_bool(gemmul8_env, disabled);
    disabled      = env_bool(openmx_env, disabled);

    return disabled;
}

unsigned gemmul8_num_moduli(const char *openmx_env, const char *gemmul8_env)
{
    unsigned num_moduli = env_u32(gemmul8_env, kDefaultNumModuli);
    num_moduli          = env_u32(openmx_env, num_moduli);

    if (num_moduli < 2u || kMaxNumModuli < num_moduli) {
        num_moduli = kDefaultNumModuli;
    }

    return num_moduli;
}

cudaError_t release_workspace(Workspace &workspace)
{
    if (workspace.ptr == nullptr) {
        workspace.size = 0;
        return cudaSuccess;
    }

    cudaError_t status = cudaFree(workspace.ptr);
    if (status == cudaSuccess) {
        workspace.ptr  = nullptr;
        workspace.size = 0;
    }

    return status;
}

bool workspace_exceeds_fraction(size_t required, size_t total, unsigned max_percent)
{
    return total != 0 && max_percent != 0 && (total * static_cast<size_t>(max_percent)) / 100u < required;
}

bool free_after_workspace_is_too_low(size_t free_bytes, size_t workspace_size, size_t required, size_t reserve)
{
    if (required <= workspace_size) {
        return free_bytes < reserve;
    }

    const size_t extra_required = required - workspace_size;
    return free_bytes < extra_required || free_bytes - extra_required < reserve;
}

template <bool is_complex>
cublasStatus_t ensure_workspace(cublasHandle_t handle, size_t m, size_t n, size_t k, unsigned num_moduli, void **work,
                                WorkspaceReport *report)
{
    cudaStream_t stream = nullptr;
    int          device = -1;

    cublasStatus_t cublas_status = cublasGetStream(handle, &stream);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cublas_status;
    }

    cudaError_t cuda_status = cudaGetDevice(&device);
    if (cuda_status != cudaSuccess) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    const size_t required = gemmul8::workSize<is_complex, gemmul8::Backend::INT8>(m, n, k, num_moduli);
    WorkspaceKey key      = {device, stream};

    if (report != nullptr) {
        report->required_bytes = required;
        report->reserve_bytes =
            env_mib("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", "GEMMUL8_MIN_FREE_AFTER_MB", kDefaultMinFreeAfterMiB);
        report->max_workspace_percent = env_percent("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT",
                                                     "GEMMUL8_MAX_WORKSPACE_PERCENT",
                                                     kDefaultMaxWorkspacePercent);
    }

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    Workspace                  &workspace = g_workspaces[key];

    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    cuda_status        = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (cuda_status == cudaSuccess && report != nullptr) {
        report->free_bytes  = free_bytes;
        report->total_bytes = total_bytes;
    }

    if (cuda_status == cudaSuccess &&
        workspace_exceeds_fraction(required, total_bytes, report != nullptr ? report->max_workspace_percent : 0u)) {
        if (report != nullptr) {
            report->reason = "workspace fraction policy";
        }
        if (release_workspace(workspace) != cudaSuccess) {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        return CUBLAS_STATUS_ALLOC_FAILED;
    }

    if (workspace.size < required) {
        cuda_status = release_workspace(workspace);
        if (cuda_status != cudaSuccess) {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }

        cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (cuda_status == cudaSuccess && report != nullptr) {
            report->free_bytes  = free_bytes;
            report->total_bytes = total_bytes;
        }
    }

    if (cuda_status == cudaSuccess &&
        free_after_workspace_is_too_low(free_bytes, workspace.size, required,
                                        report != nullptr ? report->reserve_bytes : 0u)) {
        if (report != nullptr) {
            report->reason = "free memory reserve policy";
        }
        if (release_workspace(workspace) != cudaSuccess) {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        return CUBLAS_STATUS_ALLOC_FAILED;
    }

    if (workspace.size < required) {
        cuda_status = cudaMalloc(&workspace.ptr, required);
        if (cuda_status != cudaSuccess) {
            if (report != nullptr) {
                report->reason = "cudaMalloc failure";
            }
            return CUBLAS_STATUS_ALLOC_FAILED;
        }
        workspace.size = required;
    }

    *work = workspace.ptr;
    return CUBLAS_STATUS_SUCCESS;
}

template <bool is_complex>
void log_workspace_fallback_once(const WorkspaceReport &report)
{
    static bool warned = false;

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    if (warned) {
        return;
    }

    fprintf(stderr,
            "openmx_gemmul8%sgemm: GEMMul8 workspace fallback by %s; "
            "need %.3f MiB, CUDA free %.3f MiB / total %.3f MiB, "
            "reserve %.3f MiB, max-workspace %u%%. Falling back to native cuBLAS.\n",
            is_complex ? "Z" : "D", report.reason, (double)report.required_bytes / (1024.0 * 1024.0),
            (double)report.free_bytes / (1024.0 * 1024.0), (double)report.total_bytes / (1024.0 * 1024.0),
            (double)report.reserve_bytes / (1024.0 * 1024.0), report.max_workspace_percent);
    fflush(stderr);
    warned = true;
}

} // namespace

extern "C" void openmx_gemmul8ReleaseWorkspaces(void)
{
    std::lock_guard<std::mutex> lock(g_workspace_mutex);

    for (auto it = g_workspaces.begin(); it != g_workspaces.end();) {
        cudaError_t status = release_workspace(it->second);
        if (status == cudaSuccess || it->second.ptr == nullptr) {
            it = g_workspaces.erase(it);
        }
        else {
            ++it;
        }
    }
}

extern "C" cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t handle,
                                               cublasOperation_t transa,
                                               cublasOperation_t transb,
                                               int m,
                                               int n,
                                               int k,
                                               const double *alpha,
                                               const double *A,
                                               int lda,
                                               const double *B,
                                               int ldb,
                                               const double *beta,
                                               double *C,
                                               int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) {
        return CUBLAS_STATUS_SUCCESS;
    }

    const unsigned num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_D", "GEMMUL8_NUM_MOD_D");
    const bool     fastmode   = env_bool("OPENMX_GEMMUL8_FASTMODE_D", env_bool("GEMMUL8_FASTMODE_D", false));
    const cublasOperation_t gemmul8_transa = (transa == CUBLAS_OP_C) ? CUBLAS_OP_T : transa;
    const cublasOperation_t gemmul8_transb = (transb == CUBLAS_OP_C) ? CUBLAS_OP_T : transb;
    void          *work = nullptr;
    WorkspaceReport report;

    if (gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_D", "GEMMUL8_DISABLE_D")) {
        report.reason = "environment disable";
        log_workspace_fallback_once<false>(report);
        return cublasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    cublasStatus_t status =
        ensure_workspace<false>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                                num_moduli, &work, &report);
    if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        log_workspace_fallback_once<false>(report);
        return cublasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    (void)gemmul8::gemm<double, gemmul8::Backend::INT8>(handle, gemmul8_transa, gemmul8_transb, static_cast<size_t>(m),
                                                        static_cast<size_t>(n), static_cast<size_t>(k), alpha, A,
                                                        static_cast<size_t>(lda), B, static_cast<size_t>(ldb), beta, C,
                                                        static_cast<size_t>(ldc), num_moduli, fastmode, work);

    return CUBLAS_STATUS_SUCCESS;
}

extern "C" cublasStatus_t openmx_gemmul8Zgemm(cublasHandle_t handle,
                                               cublasOperation_t transa,
                                               cublasOperation_t transb,
                                               int m,
                                               int n,
                                               int k,
                                               const cuDoubleComplex *alpha,
                                               const cuDoubleComplex *A,
                                               int lda,
                                               const cuDoubleComplex *B,
                                               int ldb,
                                               const cuDoubleComplex *beta,
                                               cuDoubleComplex *C,
                                               int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) {
        return CUBLAS_STATUS_SUCCESS;
    }

    const unsigned num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_Z", "GEMMUL8_NUM_MOD_Z");
    const bool     fastmode   = env_bool("OPENMX_GEMMUL8_FASTMODE_Z", env_bool("GEMMUL8_FASTMODE_Z", false));
    void          *work = nullptr;
    WorkspaceReport report;

    if (gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_Z", "GEMMUL8_DISABLE_Z")) {
        report.reason = "environment disable";
        log_workspace_fallback_once<true>(report);
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    cublasStatus_t status =
        ensure_workspace<true>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                               num_moduli, &work, &report);
    if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        log_workspace_fallback_once<true>(report);
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    (void)gemmul8::gemm<cuDoubleComplex, gemmul8::Backend::INT8>(
        handle, transa, transb, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k), alpha, A,
        static_cast<size_t>(lda), B, static_cast<size_t>(ldb), beta, C, static_cast<size_t>(ldc), num_moduli, fastmode,
        work);

    return CUBLAS_STATUS_SUCCESS;
}
