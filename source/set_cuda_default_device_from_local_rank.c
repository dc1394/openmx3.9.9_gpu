#include "openmx_common.h"
#include "set_cuda_default_device_from_local_rank.h"
#include <cuda_runtime.h>
#include <mpi.h>

int set_cuda_default_device_from_local_rank(MPI_Comm comm)
{
    // ノード内 communicator を作って local rank を得る
    MPI_Comm shmcomm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

    int deviceCount;
    wait_cudafunc(cudaGetDeviceCount(&deviceCount));

    int local_rank;
    MPI_Comm_rank(shmcomm, &local_rank);

    int dev = -1;
    if (deviceCount > 0) {
        int dev = local_rank % deviceCount;
        wait_cudafunc(cudaSetDevice(local_rank % deviceCount));
    }

    MPI_Comm_free(&shmcomm);

    return dev;
}