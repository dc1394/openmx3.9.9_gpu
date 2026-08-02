#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kThreads = 256;
constexpr int kPreferredGridTile = 32;

__global__ void matrix_elements_kernel(
    int pair_count,
    int spin_count,
    std::size_t vpot_len,
    double grid_vol,
    int max_no,
    int grid_tile,
    const int *__restrict__ pair_NO0,
    const int *__restrict__ pair_NO1,
    const int *__restrict__ pair_NOLG,
    const int *__restrict__ nolg_MN,
    const int *__restrict__ nolg_Nc,
    const std::size_t *__restrict__ pair_h_offset,
    const std::size_t *__restrict__ pair_nolg_offset,
    const std::size_t *__restrict__ pair_orbs0_offset,
    const std::size_t *__restrict__ pair_orbs1_offset,
    const float *__restrict__ orbs0buf,
    const float *__restrict__ orbs1buf,
    const double *__restrict__ vpotgrid,
    double *__restrict__ hbuf)
{
    const int pair = static_cast<int>(blockIdx.x);
    const int NO0 = pair_NO0[pair];
    const int NO1 = pair_NO1[pair];
    const int NOLG = pair_NOLG[pair];
    const std::size_t mat_size = static_cast<std::size_t>(NO0) * static_cast<std::size_t>(NO1);
    const std::size_t output_count = static_cast<std::size_t>(spin_count) * mat_size;
    const std::size_t e = static_cast<std::size_t>(blockIdx.y) * blockDim.x + threadIdx.x;

    if (output_count <= static_cast<std::size_t>(blockIdx.y) * blockDim.x) return;

    extern __shared__ unsigned char shared_bytes[];
    double *weighted0 = reinterpret_cast<double *>(shared_bytes);
    float *tile1 = reinterpret_cast<float *>(
        weighted0 + static_cast<std::size_t>(spin_count) * static_cast<std::size_t>(grid_tile) *
                    static_cast<std::size_t>(max_no));

    const bool active = e < output_count;
    int spin = 0;
    int i = 0;
    int j = 0;
    double sum = 0.0;
    const std::size_t h_off = pair_h_offset[pair];
    const std::size_t nolg_off = pair_nolg_offset[pair];
    const std::size_t orbs0_off = pair_orbs0_offset[pair];
    const std::size_t orbs1_off = pair_orbs1_offset[pair];

    if (active) {
        spin = static_cast<int>(e / mat_size);
        const std::size_t ij = e - static_cast<std::size_t>(spin) * mat_size;
        i = static_cast<int>(ij / static_cast<std::size_t>(NO1));
        j = static_cast<int>(ij - static_cast<std::size_t>(i) * static_cast<std::size_t>(NO1));
        sum = hbuf[h_off + e];
    }

    for (int base = 0; base < NOLG; base += grid_tile) {
        const int count = (base + grid_tile <= NOLG) ? grid_tile : (NOLG - base);
        const int weighted0_count = spin_count * count * NO0;
        const int tile1_count = count * NO1;

        for (int index = static_cast<int>(threadIdx.x); index < weighted0_count; index += blockDim.x) {
            const int spin_grid_size = count * NO0;
            const int s = index / spin_grid_size;
            const int spin_index = index - s * spin_grid_size;
            const int grid = spin_index / NO0;
            const int orbital = spin_index - grid * NO0;
            const std::size_t pt = nolg_off + static_cast<std::size_t>(base + grid);
            weighted0[(static_cast<std::size_t>(s) * grid_tile + grid) * max_no + orbital] =
                (grid_vol * vpotgrid[static_cast<std::size_t>(s) * vpot_len +
                                     static_cast<std::size_t>(nolg_MN[pt])]) *
                static_cast<double>(orbs0buf[
                    orbs0_off + static_cast<std::size_t>(nolg_Nc[pt]) * NO0 + orbital]);
        }
        for (int index = static_cast<int>(threadIdx.x); index < tile1_count; index += blockDim.x) {
            const int grid = index / NO1;
            const int orbital = index - grid * NO1;
            tile1[static_cast<std::size_t>(grid) * max_no + orbital] =
                orbs1buf[orbs1_off + static_cast<std::size_t>(base + grid) * NO1 + orbital];
        }
        __syncthreads();

        if (active) {
            for (int grid = 0; grid < count; grid++) {
                sum += weighted0[(static_cast<std::size_t>(spin) * grid_tile + grid) * max_no + i] *
                       static_cast<double>(tile1[static_cast<std::size_t>(grid) * max_no + j]);
            }
        }
        __syncthreads();
    }

    if (active) hbuf[h_off + e] = sum;
}

} // namespace

extern "C" int Set_Hamiltonian_Cuda_MatrixElements(
    int pair_count,
    int spin_count,
    std::size_t vpot_len,
    double grid_vol,
    int max_no,
    int max_output_count,
    const int *pair_NO0,
    const int *pair_NO1,
    const int *pair_NOLG,
    const int *nolg_MN,
    const int *nolg_Nc,
    const std::size_t *pair_h_offset,
    const std::size_t *pair_nolg_offset,
    const std::size_t *pair_orbs0_offset,
    const std::size_t *pair_orbs1_offset,
    const float *orbs0buf,
    const float *orbs1buf,
    const double *vpotgrid,
    double *hbuf)
{
    int device = 0;
    int max_shared = 0;
    int grid_tile = kPreferredGridTile;

    if (pair_count <= 0 || spin_count <= 0 || max_no <= 0 || max_output_count <= 0) return 0;

    /* cudaGetLastError() reports the last unconsumed error of the whole host
       thread, not of the launch below.  A library that attempted a device
       allocation, failed, and recovered by falling back (e.g. the GEMMul8
       cuBLAS hook under memory pressure) leaves that error latched; pop it
       here so it cannot be misattributed to this kernel. */
    {
        cudaError_t stale = cudaGetLastError();
        if (stale != cudaSuccess) {
            static int stale_reports = 0;
            if (stale_reports < 3) {
                stale_reports++;
                fprintf(stderr,
                        "Set_Hamiltonian_Cuda_MatrixElements: cleared stale CUDA error \"%s\" left by an earlier recovered failure; continuing.\n",
                        cudaGetErrorString(stale));
                fflush(stderr);
            }
        }
    }

    if (cudaGetDevice(&device) != cudaSuccess) return -1;
    if (cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device) != cudaSuccess) return -1;

    auto shared_size = [=](int tile) {
        return static_cast<std::size_t>(tile) * static_cast<std::size_t>(max_no) *
               (static_cast<std::size_t>(spin_count) * sizeof(double) + sizeof(float));
    };

    while (1 < grid_tile && static_cast<std::size_t>(max_shared) < shared_size(grid_tile)) grid_tile /= 2;
    if (static_cast<std::size_t>(max_shared) < shared_size(grid_tile)) return 1;

    const dim3 block(kThreads, 1u, 1u);
    const dim3 grid(static_cast<unsigned>(pair_count),
                    static_cast<unsigned>((max_output_count + kThreads - 1) / kThreads), 1u);
    matrix_elements_kernel<<<grid, block, shared_size(grid_tile)>>>(
        pair_count, spin_count, vpot_len, grid_vol, max_no, grid_tile,
        pair_NO0, pair_NO1, pair_NOLG, nolg_MN, nolg_Nc, pair_h_offset, pair_nolg_offset,
        pair_orbs0_offset, pair_orbs1_offset, orbs0buf, orbs1buf, vpotgrid, hbuf);

    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) return 0;

    /* The kernel may have run partially, so the device H blocks are suspect
       either way.  Distinguish a transient failure (context still usable ->
       the caller restores the H blocks and redoes this batch with the
       OpenACC kernel) from a sticky, context-poisoning one. */
    (void)cudaGetLastError();
    if (cudaDeviceSynchronize() == cudaSuccess) {
        static int recover_reports = 0;
        if (recover_reports < 3) {
            recover_reports++;
            fprintf(stderr,
                    "Set_Hamiltonian_Cuda_MatrixElements: CUDA kernel failed with \"%s\" but the context is healthy; deferring to the OpenACC kernel for this batch.\n",
                    cudaGetErrorString(status));
            fflush(stderr);
        }
        return 2;
    }
    (void)cudaGetLastError();
    return -static_cast<int>(status);
}
