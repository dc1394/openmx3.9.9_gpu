#include "utility.h"
#include "openmx_common.h"
#include <assert.h>
#include <stdint.h>

/*
 * This function retrieves the total memory of "GPU 0" from nvidia-smi output
 * in an MPI environment. It uses popen("nvidia-smi") to capture the output
 * and writes it to a temporary file named with the MPI rank.
 *
 * Then it scans each line to find a pattern "xxxMiB / yyyMiB". Specifically,
 * it looks for the substring "MiB /", and from that point, it parses the
 * total memory (yyy) with sscanf.
 *
 * Returns:
 *   - A positive int64_t value representing the total memory (in bytes).
 *   - -1 if no memory info could be found or if an error occurs.
 */
int64_t get_gpu_total_memory_in_bytes()
{
    size_t freeMem  = 0;
    size_t totalMem = 0;
    assert(sizeof(size_t) == sizeof(int64_t));
    wait_cudafunc(cudaMemGetInfo(&freeMem, &totalMem));

    return (int64_t)totalMem;
}