#ifndef _LU_ZINVERSE_GPU_H_
#define _LU_ZINVERSE_GPU_H_

#include <cublas_v2.h>

#pragma acc routine
int LU_Zinverse_GPU(int n, cuDoubleComplex *A, cublasHandle_t handle);

#endif
