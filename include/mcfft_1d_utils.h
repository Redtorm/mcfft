#ifndef __MCFFT_1D_UTILS_H__
#define __MCFFT_1D_UTILS_H__

#define __HIP_PLATFORM_HCC__

#include "mcfft_1d_device.h"

void setup(double* data, int n, int n_batch);
void doit(int iter);
void finalize(double* result);
void generate_data(double* data, int n, int n_batch, int seed = 42);
// const int PRIME = 100003;


#endif