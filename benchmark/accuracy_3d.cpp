#include <fftw3.h>
#include <cstdio>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "../include/mcfft_3d_utils.h"

extern char *optarg;
extern int optopt;

void fftw3_get_result(double *data, double *result, int nx, int ny, int n_batch)
{
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nx * ny);
    fftw_plan p = fftw_plan_dft_2d(nx, ny, in, in, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int i = 0; i < n_batch; ++i)
    {
        memcpy(in, data + 2 * i * nx * ny, sizeof(fftw_complex) * nx * ny);
        fftw_execute(p);
        memcpy(result + 2 * i * nx * ny, in, sizeof(fftw_complex) * nx * ny);
    }
    fftw_destroy_plan(p);
    fftw_free(in);
}

double get_error(double *tested, double *standard, int nx, int ny, int n_batch)
{
    double error = 0;
    for (int i = 0; i < n_batch; ++i)
#pragma omp parallel for reduction(+ \
                                   : error)
        for (int j = 0; j < nx * ny; ++j)
        {
            double tested_e = tested[0 + j * 2 + i * nx * ny * 2];
            double standard_e = standard[0 + j * 2 + i * nx * ny * 2];
            error += std::min(1.0, std::abs((tested_e - standard_e) / standard_e));
            //error += std::abs((tested_e - standard_e));
            tested_e = tested[1 + j * 2 + i * nx * ny * 2];
            standard_e = standard[1 + j * 2 + i * nx * ny * 2];
            error += std::min(1.0, std::abs((tested_e - standard_e) / standard_e));
            //error += std::abs((tested_e - standard_e));
            //printf("%f...\n" , error);
            // if(std::abs((tested_e - standard_e))>1){
            //     printf("%d...%d\n" , i, j);
            // }
        }
    for (int i = 0; i < 131072; ++i){
        printf("%e ... %e\n" , tested[i], standard[i]);
        // if(std::abs((tested[i]-standard[i]))>1){
        //     printf("%f...\n" , tested[i]-standard[i]);
        // }
        //error += std::abs((tested[i]-standard[i]));
    }
    // //printf("%f...\n" , error);
    return error / nx / ny / n_batch;
}

int main(int argc, char *argv[])
{
    int nx = 256, ny = 256, n_batch = 1, seed = 42;
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "x:y:b:")))
    {
        switch (opt_c)
        {
        case 'x':
            nx = atoi(optarg);
            break;
        case 'y':
            ny = atoi(optarg);
            break;
        case 'b':
            n_batch = atoi(optarg);
            break;
        case '?':
            printf("unkown option %c\n", optopt);
            break;
        default:
            break;
        }
    }

    double *data = (double *)malloc(sizeof(double) * nx * ny * n_batch * 2);
    generate_data(data, nx * ny, n_batch, seed);

    double *standard = (double *)malloc(sizeof(double) * nx * ny * n_batch * 2);
    fftw3_get_result(data, standard, nx, ny, n_batch);

    double *tested = (double *)malloc(sizeof(double) * nx * ny * n_batch * 2);
    setup(data, nx, ny, n_batch);
    doit(1);
    finalize(tested);

    printf("%e\n", get_error(tested, standard, nx, ny, n_batch));

    return 0;
}