#ifndef __MCFFT_HALF_H__
#define __MCFFT_HALF_H__

#define __HIP_PLATFORM_HCC__

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma_coop.hpp>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>

typedef struct 
{
    int N, N_batch;
    int radices[9] = { 16, 16, 16, 16, 16, 16, 16, 16, 16 };
    int n_radices;
    int mergings[3] = { 0, 0, 0 };
    int n_mergings;
    void (*layer_0[3])(float2*, float*, float*);
    void (*layer_1[3])(int, float2*, float*, float*);
    float* F_real, * F_imag;
    float* F_real_tmp, * F_imag_tmp;

} mcfftHandle;

// typedef struct{
//     float x;
//     float y;
// } FloatComplex;

void mcfftExec(mcfftHandle plan, float* data);
void mcfftCreate(mcfftHandle* plan, int n, int n_batch);

#endif