#include "openmx_common.h"
#include "set_cuda_default_device_from_local_rank.h"
#include <cuda_runtime.h>
#include <mpi.h>

int set_cuda_default_device_from_local_rank()
{
    // MPI_COMM_WORLD 内でノード共有 communicator を作り、ノード内 local rank を得る。
    // この関数は MPI_COMM_WORLD 上の全 rank が collective に呼ぶ前提。
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

    int deviceCount;
    wait_cudafunc(cudaGetDeviceCount(&deviceCount));

    int local_rank;
    MPI_Comm_rank(shmcomm, &local_rank);

    int dev = -1;
    if (deviceCount > 0) {
        dev = local_rank % deviceCount;
        wait_cudafunc(cudaSetDevice(dev));
    }

    MPI_Comm_free(&shmcomm);

    return dev;
}
