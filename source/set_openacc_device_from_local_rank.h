#ifndef _SET_OPENACC_FROM_LOCAL_RANK_H_
#define _SET_OPENACC_FROM_LOCAL_RANK_H_

#include <mpi.h>
#include <openacc.h>   // または環境によって <acc.h>

int set_openacc_device_from_local_rank(acc_device_t devtype);
int set_openacc_nvidia_device_from_local_rank(void);

#endif // _SET_OPENACC_FROM_LOCAL_RANK_H_