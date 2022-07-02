#include <unistd.h>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include "../include/mcfft_2d_utils.h"
#include "rocfft.h"
#include "../vkFFT/vkFFT.h"
#include "../include/utils_VkFFT.h"

#define T_MIN 4
#define MAX_TIMES (1 << 30)

extern char* optarg;
extern int optopt;

double gettime()
{
    timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_usec * 1.0e-6 + tv.tv_sec;
}

void double2float(double * data, float *res, int n){
    for(int i = 0; i < n; i++){
        res[i] = (float)data[i];
    }
}

void rocfft_get_result(float *data, float *result, int nx, int ny, int n_batch){
    const size_t lengths[2] = {(size_t)ny, (size_t)nx};
    bool inplace = true;
    float* x = NULL;
    hipMalloc(&x, nx * ny * 2 * sizeof(float) * n_batch);
    float* y = NULL;
    //hipMalloc(&y, n * 2 * sizeof(float) * n_batch);

    hipMemcpy(x, data, nx * ny * n_batch * 2 * sizeof(float), hipMemcpyHostToDevice);

    rocfft_setup();
    rocfft_status status = rocfft_status_success;

    // Create forward plan
    rocfft_plan forward = NULL;
    status              = rocfft_plan_create(&forward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_complex_forward,
                                rocfft_precision_single,
                                2, // Dimensions
                                lengths, // lengths
                                n_batch, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    // We may need work memory, which is passed via rocfft_execution_info
    rocfft_execution_info forwardinfo = NULL;
    status                            = rocfft_execution_info_create(&forwardinfo);
    assert(status == rocfft_status_success);
    size_t fbuffersize = 0;
    status             = rocfft_plan_get_work_buffer_size(forward, &fbuffersize);
    assert(status == rocfft_status_success);
    void* fbuffer = NULL;
    if(fbuffersize > 0)
    {
        hipMalloc(&fbuffer, fbuffersize);
        status = rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);
        assert(status == rocfft_status_success);
    }

    int iter;
    double runtime;
    for (int i = 1; i <= MAX_TIMES; i <<= 1)
    {
        double t1;
        iter = i;
        t1 = gettime();

        for(int i = 0; i < iter; i++){
            status = rocfft_execute(forward,
                                (void**)&x, // in_buffer
                                (void**)&y, // out_buffer
                                forwardinfo); // execution info
        }
        hipDeviceSynchronize();
        runtime = gettime() - t1;
        if (runtime > T_MIN) break;
    }


    double tflops = 12.0 * nx * ny * std::log(nx * ny) * 1e-12 / std::log(2.0) * n_batch * iter / runtime;
    printf("@rocfft > x: %d, y: %d, n_batch: %d, iter: %d, time per iter: %e, tflops: %lf \n", nx, ny, n_batch, iter, runtime / iter, tflops);

    assert(status == rocfft_status_success);

    hipMemcpy(result, x, nx * ny * n_batch * 2 * sizeof(float), hipMemcpyDeviceToHost);

    // printf("~~~~~~~~~~~~rocfft~~~~~~~~~~~~\n");
    // for(int i = 0; i < 256*256; i++){
        
    //     printf("%e.....%e\n", result[i], result[i]);
    // }
    // printf("~~~~~~~~~~~~~~~~~~~~~~~~\n");
}

void vkfft_get_result(float *data, float *result, int nx, int ny, int n_batch){
    VkGPU vkGPU = {};
    VkFFTConfiguration configuration = {};
	VkFFTApplication app = {};
    configuration.FFTdim = 2; //FFT dimension, 1D, 2D or 3D (default 1).
	configuration.size[0] = nx;//Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z. 
    configuration.size[1] = ny;
	configuration.size[2] = 1;
	//configuration.doublePrecision = true;
	configuration.useLUT = true; //use twiddle factor table
	configuration.makeForwardPlanOnly = true; 
    //configuration.inverseReturnToInputBuffer = 1;
    configuration.numberBatches = n_batch;
    configuration.device = &vkGPU.device;

    uint64_t inputBufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2] * n_batch;
	hipFloatComplex* inputBuffer = 0;
	hipMalloc((void **)&inputBuffer, inputBufferSize);
    
    uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2] * n_batch;
    hipFloatComplex* buffer;
    hipMalloc((void**)&buffer, bufferSize);

    configuration.inputBufferNum = 1;
    configuration.bufferNum = 1;
    configuration.bufferSize = &bufferSize;
    configuration.isInputFormatted = 1;
    configuration.inputBufferStride[0] = configuration.size[0];
    configuration.inputBufferStride[1] = configuration.size[0] * configuration.size[1];
    configuration.inputBufferStride[2] = configuration.size[0] * configuration.size[1] * configuration.size[2];
    configuration.inputBufferSize = &inputBufferSize;
	
	hipMemcpy(inputBuffer, data, bufferSize, hipMemcpyHostToDevice);

    /*   Initialize, get FFT Kernel   */
    initializeVkFFT(&app, configuration);

	VkFFTLaunchParams launchParams = {};
    launchParams.inputBuffer = (void**)&inputBuffer;
	launchParams.buffer = (void**)&buffer;

	//hipMemcpy(tmpbuffer, buffer_input, bufferSize, hipMemcpyHostToDevice);

    int iter;
    double runtime;
    for (int i = 1; i <= MAX_TIMES; i <<= 1)
    {
        double t1;
        iter = i;
        t1 = gettime();
        performVulkanFFT(&vkGPU, &app, &launchParams, -1, iter);
        hipDeviceSynchronize();
        runtime = gettime() - t1;
        if (runtime > T_MIN) break;
    }

    double tflops = 12.0 * nx * ny * std::log(nx * ny) * 1e-12 / std::log(2.0) * n_batch * iter / runtime;
    printf("@vkfft > x: %d, y: %d, n_batch: %d, iter: %d, time per iter: %e, tflops: %lf \n", nx, ny, n_batch, iter, runtime / iter, tflops);
    hipMemcpy(result, buffer, inputBufferSize, hipMemcpyDeviceToHost);

    // printf("~~~~~~~~~~~vkfft~~~~~~~~~~~~~\n");
    // for(int i = 0; i < 256*256; i++){
        
    //     printf("%e.....%e\n", result[i], result[i]);
    // }
    // printf("~~~~~~~~~~~~~~~~~~~~~~~~\n");
    
}

void mcfft_get_result(double *data, double *result, int nx, int ny, int n_batch){
    double runtime;
    int iter;
    setup(data, nx, ny, n_batch);
    for (int i = 1; i <= MAX_TIMES; i <<= 1)
    {
        double t1;
        t1 = gettime();
        doit(i);
        runtime = gettime() - t1;
        iter = i;
        if (runtime > T_MIN) break;
    }
    finalize(result);
    double tflops = 12.0 * nx * ny * std::log(nx * ny) * 1e-12 / std::log(2.0) * n_batch * iter / runtime;
    printf("@mcfft > x: %d, y: %d, n_batch: %d, iter: %d, time per iter: %e, tflops: %lf \n", nx, ny, n_batch, iter, runtime / iter, tflops);
}



int main(int argc, char* argv[])
{
    int nx = 256, ny = 256, n_batch = 1;
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
        case 'b':
            n_batch = atoi(optarg);
        break;
            break;
        case '?':
            printf("unkown option %c\n", optopt);
            break;
        default:
            break;
        }
    }

    double* data = (double*)malloc(sizeof(double) * nx * ny * n_batch * 2);
    generate_data(data, nx * ny, n_batch);
    
    float* out = (float*)malloc(sizeof(float) * nx * ny * n_batch * 2);
    float* in = (float*)malloc(sizeof(float) * nx * ny * n_batch * 2);

    double2float(data, in, nx * ny * n_batch * 2);

    vkfft_get_result(in, out, nx, ny, n_batch);
  
    rocfft_get_result(in, out, nx, ny, n_batch);

    double* res = (double*)malloc(sizeof(double) * nx * ny * n_batch * 2);
    mcfft_get_result(data, res, nx, ny, n_batch);   

    return 0;
}
