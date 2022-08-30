#ifndef __MCFFT_1D_DEVICE_H__
#define __MCFFT_1D_DEVICE_H__

#define __HIP_PLATFORM_HCC__

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma_coop.hpp>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>

#include "common.h"

typedef struct 
{
    int N, N_batch;
    int radices[9] = { 16, 16, 16, 16, 16, 16, 16, 16, 16 };
    int n_radices;
    int mergings[3] = { 0, 0, 0 };
    int n_mergings;
    void (*layer_0[3])(MCTYPE2*, MCTYPE*, MCTYPE*);
    void (*layer_1[3])(int, MCTYPE2*, MCTYPE*, MCTYPE*);
    MCTYPE* F_real, * F_imag;
    MCTYPE* F_real_tmp, * F_imag_tmp;

} mcfftHandle;

// typedef struct{
//     MCTYPE x;
//     MCTYPE y;
// } MCTYPEComplex;

void mcfftExec(mcfftHandle plan, MCTYPE* data);
void mcfftCreate(mcfftHandle* plan, int n, int n_batch);

#endif
