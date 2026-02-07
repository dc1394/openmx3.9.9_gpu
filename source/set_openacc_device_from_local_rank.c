#include "set_openacc_device_from_local_rank.h"

int set_openacc_device_from_local_rank(MPI_Comm comm, acc_device_t devtype)
{
    // ノード内 communicator を作って local rank を得る
    MPI_Comm shmcomm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED,
                        0, MPI_INFO_NULL, &shmcomm);

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
int set_openacc_nvidia_device_from_local_rank(MPI_Comm comm)
{
    return set_openacc_device_from_local_rank(comm, acc_device_nvidia);
}
