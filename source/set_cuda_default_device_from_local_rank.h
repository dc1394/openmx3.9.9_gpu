#ifndef _SET_CUDA_DEFAULT_DEVICE_FROM_LOCAL_RANK_H_
#define _SET_CUDA_DEFAULT_DEVICE_FROM_LOCAL_RANK_H_

#include <mpi.h>

int set_cuda_default_device_from_local_rank(MPI_Comm comm);

#endif // _SET_CUDA_DEFAULT_DEVICE_FROM_LOCAL_RANK_H_