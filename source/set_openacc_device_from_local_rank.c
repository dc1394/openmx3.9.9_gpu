#include "set_openacc_device_from_local_rank.h"

int set_openacc_device_from_local_rank(acc_device_t devtype)
{
    // MPI_COMM_WORLD 内でノード共有 communicator を作り、ノード内 local rank を得る。
    // この関数は MPI_COMM_WORLD 上の全 rank が collective に呼ぶ前提。
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

    int local_rank = 0;
    MPI_Comm_rank(shmcomm, &local_rank);

    int ndev = acc_get_num_devices(devtype);
    int dev  = -1;

    if (ndev > 0) {
        dev = local_rank % ndev;
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
