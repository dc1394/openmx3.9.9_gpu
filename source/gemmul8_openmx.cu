/*
 * Minimal explicit-instantiation translation unit of GEMMul8 for OpenMX.
 *
 * Upstream GEMMul8 builds one object per (routine, type, backend)
 * combination from the cu_recipe sources via its own make system.
 * OpenMX only needs the INT8-backend gemm for double and cuDoubleComplex
 * plus the matching workSize queries, so this single TU includes the
 * template definitions and instantiates just those four entry points.
 *
 * Compile with -I<GEMMul8>/include -I<GEMMul8>/src.
 */
#include "gemm/gemm_impl.hpp"
#include "worksize/worksize_impl.hpp"

/*
 * The include chain above only declares the device-kernel launchers
 * (see *_declaration.hpp); upstream compiles their definitions as
 * separate per-instantiation objects.  Include the definition headers
 * here so the explicit instantiations below reach them implicitly.
 */
#include "oz2/mod/mod_hi2mid.hpp"
#include "oz2/scaling/fast/scaling.hpp"
#include "oz2/scaling/accu/extract.hpp"
#include "oz2/scaling/accu/scaling.hpp"
#include "oz2/scaling/general/scaling_rowwise.hpp"
#include "oz2/scaling/general/scaling_colwise.hpp"
#include "oz2/undo_scaling/undo_scaling.hpp"

namespace gemmul8 {

template std::vector<double> gemm<double, Backend::INT8, double, double>(
    cublasHandle_t,
    cublasOperation_t, cublasOperation_t,
    size_t, size_t, size_t,
    const double *,
    const double *const, size_t,
    const double *const, size_t,
    const double *,
    double *const, size_t,
    int, bool,
    void *const,
    void *const,
    void *const,
    bool, bool, bool, bool);

template std::vector<double> gemm<cuDoubleComplex, Backend::INT8, cuDoubleComplex, cuDoubleComplex>(
    cublasHandle_t,
    cublasOperation_t, cublasOperation_t,
    size_t, size_t, size_t,
    const cuDoubleComplex *,
    const cuDoubleComplex *const, size_t,
    const cuDoubleComplex *const, size_t,
    const cuDoubleComplex *,
    cuDoubleComplex *const, size_t,
    int, bool,
    void *const,
    void *const,
    void *const,
    bool, bool, bool, bool);

template size_t workSize<false, Backend::INT8, Func::gemm>(
    size_t, size_t, size_t, int, bool, bool, size_t *, size_t *);

template size_t workSize<true, Backend::INT8, Func::gemm>(
    size_t, size_t, size_t, int, bool, bool, size_t *, size_t *);

} // namespace gemmul8
