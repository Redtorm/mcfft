#ifndef __MCFFT_2D_DEVICE_H__
#define __MCFFT_2D_DEVICE_H__

#define __HIP_PLATFORM_HCC__

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma_coop.hpp>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>

typedef struct 
{
    int Nx, Ny, N_batch;
    int radices_x[3] = {16, 16, 2};
    int radices_y[3] = {16, 16, 2};
    int n_radices_x, n_radices_y;
    int mergings[2] = {0, 0};
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
void mcfftCreate(mcfftHandle* plan, int nx, int ny, int n_batch);

#endif