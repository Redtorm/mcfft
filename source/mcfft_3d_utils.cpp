#include "../include/mcfft_3d_device.h"

int *rev_x, *rev_y, Nx, Ny, N_batch;
MCTYPE *in_host;
MCTYPE *in_device;
mcfftHandle plan;

void gen_rev(int N, int rev[], int radices[], int n_radices)
{
    int* tmp_0 = (int*)malloc(sizeof(int) * N);
    int* tmp_1 = (int*)malloc(sizeof(int) * N);
    int now_N = N;
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
        tmp_0[i] = i;
    for (int i = n_radices - 1; i >= 0; --i)
    {
#pragma omp parallel for
        for (int j = 0; j < N; j += now_N)
            for (int k = 0; k < radices[i]; ++k)
                for (int l = 0; l < now_N / radices[i]; ++l)
                {
                    tmp_1[j + l + k * (now_N / radices[i])] = tmp_0[j + l * radices[i] + k];
                }
        now_N /= radices[i];
        std::swap(tmp_0, tmp_1);
    }
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
        rev[i] = tmp_0[i];
}

void setup(double* data, int nx, int ny, int n_batch)
{
    Nx = nx;
    Ny = ny;
    N_batch = n_batch;
    mcfftCreate(&plan, Nx, Ny, N_batch);
    // in_host
    rev_x = (int *)malloc(sizeof(int) * Nx);
    rev_y = (int *)malloc(sizeof(int) * Ny);
    gen_rev(Nx, rev_x, plan.radices_x, plan.n_radices_x);
    gen_rev(Ny, rev_y, plan.radices_y, plan.n_radices_y);
    in_host = (MCTYPE *)malloc(sizeof(MCTYPE) * 2 * Nx * Ny * N_batch);
#pragma omp parallel for
    for (int i = 0; i < N_batch; ++i){
        for (int j = 0; j < Nx; ++j){
            for (int k = 0; k < Ny; ++k){
                in_host[2 * (i * Nx * Ny + j * Ny + k) + 0] = data[2 * (i * Nx * Ny + rev_x[j] * Ny + rev_y[k]) + 0];
                in_host[2 * (i * Nx * Ny + j * Ny + k) + 1] = data[2 * (i * Nx * Ny + rev_x[j] * Ny + rev_y[k]) + 1];
            }
        }    
    }
        
        
        
    //printf("%f...%f\n" , data[55], data[33]);
    //printf("host%f...%f\n" , in_host[55], in_host[33]);
    hipMalloc(&in_device, sizeof(MCTYPE) * Nx * Ny * N_batch * 2);
    hipMemcpy(in_device, in_host, sizeof(MCTYPE) * Nx * Ny * N_batch * 2, hipMemcpyHostToDevice);
}

void finalize(double* result){
   hipMemcpy(in_host, in_device, sizeof(MCTYPE) * Nx * Ny * N_batch * 2, hipMemcpyDeviceToHost);
#pragma omp paralllel for
    for (int i = 0; i < N_batch * Nx * Ny; ++i){
        
        result[2 * i + 0] = (double)in_host[2 * i + 0];
        result[2 * i + 1] = (double)in_host[2 * i + 1];
        //printf("3424host>>>>>>>>> %e...\n" ,__half2MCTYPE(in_host_imag[j * N + i]));

    }
}
void doit(int iter)
{
    for (int t = 0; t < iter; ++t)
        mcfftExec(plan, in_device);
    hipDeviceSynchronize();
}

void generate_data(double *data, int n, int n_batch, int seed)
{
    srand(seed);
    for (int i = 0; i < n_batch; ++i)
        #pragma omp parallel for
        for (int j = 0; j < n; ++j)
        {
            // data[0 + j * 2 + i * n * 2] = 0.0001f * (j % PRIME) / PRIME;
            // data[1 + j * 2 + i * n * 2] = 0.0001f * (j % PRIME) / PRIME;
            data[0 + j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;//0.001f *(j + 1);//0.0001f * rand() / RAND_MAX;
            data[1 + j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;//0.001f *(j + 1);//0.0001f * rand() / RAND_MAX;
        }
}