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

struct WorkspaceKey {
    int            device;
    cublasHandle_t handle;
    cudaStream_t   stream;

    bool operator==(const WorkspaceKey &other) const
    {
        return device == other.device && handle == other.handle && stream == other.stream;
    }
};

struct WorkspaceKeyHash {
    std::size_t operator()(const WorkspaceKey &key) const
    {
        return (static_cast<std::size_t>(key.device) << 32) ^ reinterpret_cast<std::uintptr_t>(key.handle) ^
               (reinterpret_cast<std::uintptr_t>(key.stream) << 1);
    }
};

struct Workspace {
    void *ptr   = nullptr;
    size_t size = 0;
};

std::mutex g_workspace_mutex;
std::unordered_map<WorkspaceKey, Workspace, WorkspaceKeyHash> g_workspaces;

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

unsigned gemmul8_num_moduli(const char *openmx_env, const char *gemmul8_env)
{
    unsigned num_moduli = env_u32(gemmul8_env, kDefaultNumModuli);
    num_moduli          = env_u32(openmx_env, num_moduli);

    if (num_moduli < 2u || kMaxNumModuli < num_moduli) {
        num_moduli = kDefaultNumModuli;
    }

    return num_moduli;
}

template <bool is_complex>
cublasStatus_t ensure_workspace(cublasHandle_t handle, size_t m, size_t n, size_t k, unsigned num_moduli, void **work,
                                size_t *required_bytes_out, size_t *free_bytes_out)
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
    WorkspaceKey key      = {device, handle, stream};

    if (required_bytes_out != nullptr) {
        *required_bytes_out = required;
    }
    if (free_bytes_out != nullptr) {
        *free_bytes_out = 0;
    }

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    Workspace                  &workspace = g_workspaces[key];

    if (workspace.size < required) {
        if (workspace.ptr != nullptr) {
            cuda_status = cudaFree(workspace.ptr);
            if (cuda_status != cudaSuccess) {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            workspace.ptr  = nullptr;
            workspace.size = 0;
        }

        size_t free_bytes  = 0;
        size_t total_bytes = 0;
        cuda_status        = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (cuda_status == cudaSuccess && free_bytes_out != nullptr) {
            *free_bytes_out = free_bytes;
        }
        if (cuda_status == cudaSuccess && free_bytes < required) {
            return CUBLAS_STATUS_ALLOC_FAILED;
        }

        cuda_status = cudaMalloc(&workspace.ptr, required);
        if (cuda_status != cudaSuccess) {
            return CUBLAS_STATUS_ALLOC_FAILED;
        }
        workspace.size = required;
    }

    *work = workspace.ptr;
    return CUBLAS_STATUS_SUCCESS;
}

template <bool is_complex>
void log_workspace_fallback_once(size_t required_bytes, size_t free_bytes)
{
    static bool warned = false;

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    if (warned) {
        return;
    }

    fprintf(stderr,
            "openmx_gemmul8%sgemm: GEMMul8 workspace allocation needs %.3f MiB "
            "(CUDA reported %.3f MiB free); falling back to native cuBLAS.\n",
            is_complex ? "Z" : "D", (double)required_bytes / (1024.0 * 1024.0),
            (double)free_bytes / (1024.0 * 1024.0));
    fflush(stderr);
    warned = true;
}

} // namespace

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
    void          *work       = nullptr;
    size_t         required   = 0;
    size_t         free_bytes = 0;

    cublasStatus_t status =
        ensure_workspace<false>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                                num_moduli, &work, &required, &free_bytes);
    if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        log_workspace_fallback_once<false>(required, free_bytes);
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
    void          *work       = nullptr;
    size_t         required   = 0;
    size_t         free_bytes = 0;

    cublasStatus_t status =
        ensure_workspace<true>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                               num_moduli, &work, &required, &free_bytes);
    if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        log_workspace_fallback_once<true>(required, free_bytes);
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
