#include <unistd.h>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include "../include/mcfft_api.h"
#include "rocfft.h"
#include "../vkFFT/vkFFT.h"
#include "../include/utils_VkFFT.h"

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

void rocfft_get_result(float *data, float *result, int n, int n_batch){
    const size_t length = n;
    bool inplace = false;
    float* x = NULL;
    hipMalloc(&x, n * 2 * sizeof(float) * n_batch);
    float* y = NULL;
    hipMalloc(&y, n * 2 * sizeof(float) * n_batch);

    hipMemcpy(x, data, n * n_batch * 2 * sizeof(float), hipMemcpyHostToDevice);

    rocfft_setup();
    rocfft_status status = rocfft_status_success;

    // Create forward plan
    rocfft_plan forward = NULL;
    status              = rocfft_plan_create(&forward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_complex_forward,
                                rocfft_precision_single,
                                1, // Dimensions
                                &length, // lengths
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

    double t1;
    t1 = gettime();
    int iter = 8192;
    // Execute the forward transform
    for(int i = 0; i < iter; i++){
        status = rocfft_execute(forward,
                            (void**)&x, // in_buffer
                            (void**)&y, // out_buffer
                            forwardinfo); // execution info
	    
    }
    hipDeviceSynchronize();
    double runtime = gettime() - t1;

    printf("rocfft time cost: %e\n", runtime / iter);

    assert(status == rocfft_status_success);

    hipMemcpy(result, y, n * n_batch * 2 * sizeof(float), hipMemcpyDeviceToHost);

    // printf("~~~~~~~~~~~~rocfft~~~~~~~~~~~~\n");
    // for(int i = 0; i < 256*256; i++){
        
    //     printf("%e.....%e\n", result[i], result[i]);
    // }
    // printf("~~~~~~~~~~~~~~~~~~~~~~~~\n");
}

void vkfft_get_result(float *data, float *result, int n, int n_batch){
    VkGPU vkGPU = {};
    VkFFTConfiguration configuration = {};
	VkFFTApplication app = {};
    configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
	configuration.size[0] = n;//Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z. 
    configuration.size[1] = 1;
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

    int iter = 8192;
    double t1;
    t1 = gettime();
    performVulkanFFT(&vkGPU, &app, &launchParams, -1, iter);
    double runtime = gettime() - t1;
    printf("vkfft time cost: %e\n", runtime / iter);
    hipMemcpy(result, buffer, inputBufferSize, hipMemcpyDeviceToHost);

    // printf("~~~~~~~~~~~vkfft~~~~~~~~~~~~~\n");
    // for(int i = 0; i < 256*256; i++){
        
    //     printf("%e.....%e\n", result[i], result[i]);
    // }
    // printf("~~~~~~~~~~~~~~~~~~~~~~~~\n");
    
}



int main(int argc, char* argv[])
{
    int n = 65536, n_batch = 1, max_times = 1 << 30;
    double t_min = 4;
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "n:b:m:")))
    {
        switch (opt_c)
        {
        case 'n':
            n = atoi(optarg);
            break;
        case 'b':
            n_batch = atoi(optarg);
            break;
        case 'm':
            max_times = atoi(optarg);
            break;
        case '?':
            printf("unkown option %c\n", optopt);
            break;
        default:
            break;
        }
    }

    // int value;
    //hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlock, (hipDevice_t)0);
    //hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, (hipDevice_t)0);
    // hipDeviceGetAttribute(&value, hipDeviceAttributeWarpSize, (hipDevice_t)0);
    // printf("~~~~~~~~~~~~%d~~~~~~~~~~~\n", value);

    double* data = (double*)malloc(sizeof(double) * n * n_batch * 2);
    generate_data(data, n, n_batch);
    
    float* rocfft_res = (float*)malloc(sizeof(float) * n * n_batch * 2);
    float* rocfft_in = (float*)malloc(sizeof(float) * n * n_batch * 2);
    double2float(data, rocfft_in, n * n_batch * 2);
    vkfft_get_result(rocfft_in, rocfft_res, n, n_batch);

    
    rocfft_get_result(rocfft_in, rocfft_res, n, n_batch);

    double* result = (double*)malloc(sizeof(double) * n * n_batch * 2);

    double run_time;
    int iter;
    setup(data, n, n_batch);
    for (int i = 1; i <= max_times; i <<= 1)
    {
        double t1;
        t1 = gettime();
        doit(i);
        run_time = gettime() - t1;
        iter = i;
        if (run_time > t_min)
            break;
        // printf("%d\n", iter);
    }
    finalize(result);

    double tflops = 12.0 * n * std::log(n) * 1e-12 / std::log(2.0) * n_batch * iter / run_time;
    printf("n: %d, n_batch: %d, iter: %d, time per iter: %e, tflops: %lf \n", n, n_batch, iter, run_time / iter, tflops);

    return 0;
}
