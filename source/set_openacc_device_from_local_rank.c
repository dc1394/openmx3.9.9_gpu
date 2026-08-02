#include "openmx_common.h"
#include "set_openacc_device_from_local_rank.h"
#include "set_cuda_default_device_from_local_rank.h"
#include <stdlib.h>

/* The rank-to-device map lives in set_cuda_default_device_from_local_rank.c
   (openmx_gpu_map_rank_to_device); the OpenACC binding must agree with the
   CUDA one for the whole run, so both go through the same function. */

int set_openacc_device_from_local_rank(acc_device_t devtype)
{
    // MPI_COMM_WORLD 内でノード共有 communicator を作り、ノード内 local rank を得る。
    // この関数は MPI_COMM_WORLD 上の全 rank が collective に呼ぶ前提。
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

    int local_rank = 0;
    int local_size = 1;
    MPI_Comm_rank(shmcomm, &local_rank);
    MPI_Comm_size(shmcomm, &local_size);

    int ndev = acc_get_num_devices(devtype);
    int dev  = -1;

    /* scf.Gpu.Num caps the GPUs used per node (debugging aid) */
    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < ndev) ndev = SCF_Gpu_Num;

    if (ndev > 0) {
        dev = openmx_gpu_map_rank_to_device(local_rank, local_size, ndev);
        acc_set_device_num(dev, devtype);

        // 必要なら明示初期化（好み/実装依存）
        // acc_init(devtype);
    }

    MPI_Comm_free(&shmcomm);
    return dev;  // -1: デバイス無し
}

/* 便利ラッパ（NVIDIA 固定で良ければ） */
int set_openacc_nvidia_device_from_local_rank(void)
{
    return set_openacc_device_from_local_rank(acc_device_nvidia);
}

int set_openacc_device_from_local_rank_noncollective(acc_device_t devtype)
{
    int ndev = acc_get_num_devices(devtype);
    int dev  = -1;

    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < ndev) ndev = SCF_Gpu_Num;

    if (ndev > 0) {
        dev = openmx_gpu_map_rank_to_device(openmx_gpu_local_rank_noncollective(),
                                            openmx_gpu_local_size_noncollective(),
                                            ndev);
        acc_set_device_num(dev, devtype);
    }

    return dev;
}

int set_openacc_nvidia_device_from_local_rank_noncollective(void)
{
    return set_openacc_device_from_local_rank_noncollective(acc_device_nvidia);
}
