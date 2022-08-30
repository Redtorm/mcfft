
#define __HIP_PLATFORM_HCC__

#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>

#include <type_traits>


#include "common.hpp"
#include <hip/hip_complex.h>

#define L 256

typedef float Myhalf;
typedef half MCTYPE;
typedef half2 MCTYPE2;

const int WARP_SIZE = 32, rocwmma_M = 16, rocwmma_N = 16, rocwmma_K = 16, CONT_SIZE = 16;

using namespace rocwmma;

void genData(Myhalf * data);
void genData1(Myhalf * data);
void genData2(Myhalf * data);
__global__ void mul(Myhalf *inA, Myhalf *inB);

__global__ void helloGPU(void){

    printf("threadIDx: Data:>>\n");
}


int main(){
    printf("%d\n", sizeof(hipDoubleComplex));
    Myhalf *dataA=(Myhalf*)malloc(L * sizeof(Myhalf));
    Myhalf *dataB=(Myhalf*)malloc(L * sizeof(Myhalf));
    genData(dataA);
    genData2(dataB);

    Myhalf *inDataA;
    Myhalf *inDataB;
    hipMalloc(&inDataA, sizeof(Myhalf) * L);
    hipMalloc(&inDataB, sizeof(Myhalf) * L);
    hipMemcpy(inDataA, dataA, sizeof(Myhalf) * L, hipMemcpyHostToDevice);
    hipMemcpy(inDataB, dataB, sizeof(Myhalf) * L, hipMemcpyHostToDevice);


    int value1, value2, value3;
    hipDeviceGetAttribute(&value1, hipDeviceAttributeMaxSharedMemoryPerBlock, (hipDevice_t)0);
    hipDeviceGetAttribute(&value2, hipDeviceAttributeMaxThreadsPerBlock, (hipDevice_t)0);
    hipDeviceGetAttribute(&value3, hipDeviceAttributeWarpSize, (hipDevice_t)0);
    printf("Max Share Mem:%d, Max Threads Per Block:%d, WarpSize:%d", value1, value2, value3);
    
    dim3 threads = {64, 1};
    
    hipFuncSetAttribute((void*)mul, hipFuncAttributeMaxDynamicSharedMemorySize, sizeof(half) * 256);
    mul<<<1,threads>>>(inDataA, inDataB);

    

    //helloGPU<<<1,1>>>();
    hipDeviceSynchronize();
  
    // half a =__float2half(1.34f);
    // half b =__float2half(1.88f);
    // half2 c = {a, b};
    // half d = c.x;
    // half e = c.y;
    // printf("Data:%f >>>> %f >>> %f\n",__half2float(a), __half2float(d), __half2float(e));

    int a = 10;
    int *k = &a;
    printf("%d\n", k[0]);



    return 0;
}

void genData(Myhalf * data){
    for(int i = 0; i < L; i++){
        data[i] = (i + 1) * 0.01f;
    }
}

void genData1(Myhalf * data){
    for(int i = 0; i < 16; i++){
        for(int j = 0; j<16;j++){
            if(i == j) data[i*16 +j] =1;
            else data[i*16 +j] =0;
        }
    }
}

void genData2(Myhalf * data){
    for(int i = 0; i < 16; i++){
        for(int j = 0; j<16;j++){
            data[i * 16 + j] = j * 16 + i + 1 ;
        }
    }
}


__global__ void mul(Myhalf *inA, Myhalf *inB){
    rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, Myhalf> matrixC;
    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, Myhalf, rocwmma::row_major> matrixA;
    rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, Myhalf, rocwmma::col_major> matrixB;
    rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, Myhalf, rocwmma::col_major> matrixD;
    rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, Myhalf, rocwmma::col_major> matrixE;

   // __shared__ Myhalf smem_in[256];

    __shared__ MCTYPE2 smem_in[256];
    // MCTYPE a = 0.25;
    // MCTYPE b = 0.25;
    // smem_in[1].x = a;
    // smem_in[1].y = b;
    // MCTYPE2 c = {a, b};

    // smem_in[1] = {a, b};

//    double k = float(a);    
    // uint32_t waveIndex = threadIdx.y;
    // Myhalf f=0.0;
    // rocwmma::fill_fragment(matrixC, f);
    // rocwmma::load_matrix_sync(matrixA, inA, 16);
    // rocwmma::load_matrix_sync(matrixB, inB, 32);
    // rocwmma::load_matrix_sync(matrixE, inB, 16);

    // rocwmma::mma_sync(matrixC, matrixA, matrixB, matrixC);

    // rocwmma::store_matrix_sync(smem_in, matrixC, 16, rocwmma::mem_row_major);

    // rocwmma::load_matrix_sync(matrixD, smem_in, 16);

    // // if(threadIdx.x == 0)
    // // for(int i=0;i < 16; i++){
    // //     Myhalf s = smem_in[i];
    // //     printf("threadIDx Data:%f \n", s);
    // // }
    // // if(threadIdx.x == 16)
    // // printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");

    // int raw_col = threadIdx.x % 16;
    // int raw_row = threadIdx.x / 16 * 4;
    

    

    // if(threadIdx.x == 1)
    // for(int i = 0; i < 4; i++){
    //     int tId = threadIdx.x;
    //     int col = raw_col;
    //     int row = raw_row + i;
    //     Myhalf s = matrixA.x[i];
    //     Myhalf ss = matrixB.x[i];
    //     Myhalf sss = matrixC.x[i];
    //     Myhalf ssss = matrixE.x[i];
        
    //     printf("threadIDx:%d Data:A%f >>>> B%f >>> C%f >>> %f (row:%d, col:%d)\n", tId, s, ss, sss, ssss, row, col);
    //     //smem_in[row * 16 + col]
    // }

    // int res1 = 0;
    // int res2 = 0;
    // for(int i = 0; i < 16; i++){
    //     res1 += (i + 1) * (i + 1);
    //     res2 += (i + 1) * (16 * i + 2);
    // }
    // printf("%d  %d\n", res1, res2);
    

    // int cc=0;
    // for(int i =0; i<16;++i){
    //     cc=cc+(i+1)*(i*16+2);
    // }
    // if(threadIdx.x == 0)
    // printf("threadI: Data:>>%d\n", cc);

    // float2 aaa = {1.0, 2.0};
    // float2 bbb = {3.0, 4.0};
    // float2 ccc = __fadd2(aaa,bbb);
    // printf("%f, %f", ccc.x, ccc.y);

}




