#ifndef _SET_CUDA_DEFAULT_DEVICE_FROM_LOCAL_RANK_H_
#define _SET_CUDA_DEFAULT_DEVICE_FROM_LOCAL_RANK_H_

#include <mpi.h>

int set_cuda_default_device_from_local_rank();
int set_cuda_default_device_from_local_rank_noncollective(void);

/* Shared rank-to-device map used by every site that binds a rank to a GPU.
   Contiguous blocks of local ranks share a device (block map) instead of the
   old round-robin (local_rank % device_count): the Band solver's k-point
   worlds are contiguous rank blocks whose owner ranks sit world-size apart,
   so with np = 16*gpu.num the round-robin map placed every k-owner on
   device 0 and their concurrent eigensolves serialized (measured on A100,
   n=2808: 8 owners on one device solve in 1.08 s each, 2 owners per device
   in 0.28 s).  The block map spreads the owners across the allowed devices
   while keeping the same number of ranks per device, and it must be the
   ONLY map in the binary: a rank's device has to stay identical from init
   to the last SCF step, or the OpenACC resident data of one phase becomes
   invisible to the next.  OPENMX_GPU_RANK_MAP=mod restores the old map. */
int openmx_gpu_map_rank_to_device(int local_rank, int local_size, int device_count);

/* Node-local rank/size without communication (launcher-provided envs,
   falling back to MPI_COMM_WORLD, which is exact on single-node runs). */
int openmx_gpu_local_rank_noncollective(void);
int openmx_gpu_local_size_noncollective(void);

/* One-time, noncollective, per-rank answer to "may this rank touch the
   GPU at all?".  Creates the rank's CUDA context (the dominant per-rank
   device-memory cost) through error-returning CUDA runtime calls and
   forces the OpenACC device initialization while a free-memory margin
   is verified, because past that point acc_malloc and the data
   directives abort instead of failing.  Ranks that fail keep the host
   as their OpenACC compute device and must take the host code paths. */
int gpu_rank_device_usable(void);

#endif // _SET_CUDA_DEFAULT_DEVICE_FROM_LOCAL_RANK_H_
