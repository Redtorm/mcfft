#define __HIP_PLATFORM_HCC__
#include "../include/mcfft_3d_device.h"

using namespace rocwmma;


const int WARP_SIZE = 64, rocwmma_M = 16, rocwmma_N = 16, rocwmma_K = 16;


__device__ inline void 
complex_mul(rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> &frag_F_real,
            rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> &frag_F_imag,
            rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> &frag_in_real, 
            rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> &frag_in_imag,
            rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> &frag_out_real, 
            rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> &frag_out_imag)
{
    MCTYPE f=0.0;
    rocwmma::fill_fragment(frag_out_real, f);
    rocwmma::fill_fragment(frag_out_imag, f);

    rocwmma::mma_sync(frag_out_real, frag_F_imag, frag_in_imag, frag_out_real);
    for (int i = 0; i < frag_out_real.num_elements; i++)
        frag_out_real.x[i] = -frag_out_real.x[i];
    rocwmma::mma_sync(frag_out_real, frag_F_real, frag_in_real, frag_out_real);

    rocwmma::mma_sync(frag_out_imag, frag_F_real, frag_in_imag, frag_out_imag);
    rocwmma::mma_sync(frag_out_imag, frag_F_imag, frag_in_real, frag_out_imag);
}

__device__ __host__ inline MCTYPE2 W_N_K(int N, int K, int M)
{
    MCTYPE2 t = {cosf(2 * M_PI * K * M / N), -sinf(2 * M_PI * K * M / N)};
    return t;
}


__device__ inline MCTYPE2 cmul(MCTYPE2 a, MCTYPE2 b)
{
    MCTYPE2 t = {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
    return t;
}

__device__ inline MCTYPE2 cadd(MCTYPE2 a, MCTYPE2 b)
{
    MCTYPE2 t = {a.x + b.x, a.y + b.y};
    return t;
}

__device__ inline MCTYPE2 csub(MCTYPE2 a, MCTYPE2 b)
{
    MCTYPE2 t = {a.x - b.x, a.y - b.y};
    return t;
}


template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_256_0(MCTYPE2* data, MCTYPE* F_real, MCTYPE* F_imag)
{
    //__shared__ half smem_in[256 * 2 * 32];
    extern __shared__ MCTYPE2 smem[];
    int tid_block  = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 256 * CONT_SIZE;


    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> frag_F_real;
    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> frag_F_imag;
    rocwmma::load_matrix_sync(frag_F_real, F_real, 16);
    rocwmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 256)
    {
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_imag;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        MCTYPE2 twiddle_unit = W_N_K(256, raw_col, raw_row);
        MCTYPE2 twiddle_factor = W_N_K(256, raw_col, 1);

        int warp_start = i + threadIdx.y * 256;

        for(int j = 0; j < 4; j++){
            int col = raw_col;
            int row = raw_row + j;
            int eid_global = block_start + warp_start + col * 16 + row;
            MCTYPE2 ele = data[eid_global];
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;

        }


        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for(int j = 0; j < 4; j++){
            int col = raw_col;
            int row = raw_row + j;
            int eid_block = warp_start + col * 16 + row;
            smem[eid_block].x = frag_out_real.x[j];
            smem[eid_block].y = frag_out_imag.x[j];

        }

        __syncthreads();


        for(int j = 0; j < 4; j++){

            int col = raw_col;
            int row = raw_row + j;
            int eid_block = warp_start + row * 16 + col;

            MCTYPE2 ele = smem[eid_block]; 
            
            ele = cmul(ele, twiddle_unit);
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;

            twiddle_unit = cmul(twiddle_unit, twiddle_factor);
        }
        

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for(int j = 0; j < 4; j++){
            int col = raw_col;
            int row = raw_row + j;
            int eid_global = block_start + warp_start + row * 16 + col;
            data[eid_global].x = frag_out_real.x[j];
            data[eid_global].y = frag_out_imag.x[j];
        }
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_512_0(MCTYPE2* data, MCTYPE* F_real, MCTYPE* F_imag)
{
    extern __shared__ MCTYPE2 smem[];
    int tid_block  = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 512 * CONT_SIZE;


    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> frag_F_real;
    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> frag_F_imag;
    rocwmma::load_matrix_sync(frag_F_real, F_real, 16);
    rocwmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    for (int i = 0; i < 512 * CONT_SIZE; i += NUM_WARP * 256)
    {
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_imag;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        MCTYPE2 twiddle_unit = W_N_K(256, raw_col, raw_row);
        MCTYPE2 twiddle_factor = W_N_K(256, raw_col, 1);

        int warp_start = i + threadIdx.y * 256;

        for(int j = 0; j < 4; j++){
            int col = raw_col;
            int row = raw_row + j;
            int eid_global = block_start + warp_start + col * 16 + row;
            MCTYPE2 ele = data[eid_global];
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;

        }


        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for(int j = 0; j < 4; j++){
            int col = raw_col;
            int row = raw_row + j;
            int eid_block = warp_start + col * 16 + row;
            smem[eid_block].x = frag_out_real.x[j];
            smem[eid_block].y = frag_out_imag.x[j];

        }

        __syncthreads();


        for(int j = 0; j < 4; j++){

            int col = raw_col;
            int row = raw_row + j;
            int eid_block = warp_start + row * 16 + col;

            MCTYPE2 ele = smem[eid_block]; 
            
            ele = cmul(ele, twiddle_unit);
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;

            twiddle_unit = cmul(twiddle_unit, twiddle_factor);
        }
        

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for(int j = 0; j < 4; j++){
            int col = raw_col;
            int row = raw_row + j;
            int eid_block = warp_start + row * 16 + col;
            smem[eid_block].x = frag_out_real.x[j];
            smem[eid_block].y = frag_out_imag.x[j]; 
        }
    }

    __syncthreads();

    MCTYPE2 twiddle_512 = W_N_K(512, tid_block % 256, 1);
    for(int i = 0; i < 512 * CONT_SIZE; i += NUM_WARP * 64 * 2){
        int t = tid_block / 256;
        int eid_block = i + (512 - 256) * t + tid_block;
        MCTYPE2 ele_0 = smem[eid_block];
        MCTYPE2 ele_1 = cmul(smem[eid_block + 256], twiddle_512);
        data[block_start + eid_block] = cadd(ele_0, ele_1);
        data[block_start + eid_block + 256] = csub(ele_0, ele_1);
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_1024_0(MCTYPE2* data, MCTYPE* F_real, MCTYPE* F_imag)
{
    extern __shared__ MCTYPE2 smem[];
    int tid_block  = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 1024 * CONT_SIZE;


    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> frag_F_real;
    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> frag_F_imag;
    rocwmma::load_matrix_sync(frag_F_real, F_real, 16);
    rocwmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    for (int i = 0; i < 1024 * CONT_SIZE; i += NUM_WARP * 256)
    {
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_imag;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        MCTYPE2 twiddle_unit = W_N_K(256, raw_col, raw_row);
        MCTYPE2 twiddle_factor = W_N_K(256, raw_col, 1);

        int warp_start = i + threadIdx.y * 256;

        for(int j = 0; j < 4; j++){
            int col = raw_col;
            int row = raw_row + j;
            int eid_global = block_start + warp_start + col * 16 + row;
            MCTYPE2 ele = data[eid_global];
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;

        }


        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for(int j = 0; j < 4; j++){
            int col = raw_col;
            int row = raw_row + j;
            int eid_block = warp_start + col * 16 + row;
            smem[eid_block].x = frag_out_real.x[j];
            smem[eid_block].y = frag_out_imag.x[j];

        }

        __syncthreads();


        for(int j = 0; j < 4; j++){

            int col = raw_col;
            int row = raw_row + j;
            int eid_block = warp_start + row * 16 + col;

            MCTYPE2 ele = smem[eid_block]; 
            
            ele = cmul(ele, twiddle_unit);
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;

            twiddle_unit = cmul(twiddle_unit, twiddle_factor);
        }
        

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for(int j = 0; j < 4; j++){
            int col = raw_col;
            int row = raw_row + j;
            int eid_block = warp_start + row * 16 + col;
            smem[eid_block].x = frag_out_real.x[j];
            smem[eid_block].y = frag_out_imag.x[j]; 
        }
    }

    __syncthreads();

    MCTYPE2 twiddle_1024_1 = W_N_K(1024, tid_block % 256, 1);
    MCTYPE2 twiddle_1024_2 = cmul(twiddle_1024_1, twiddle_1024_1);
    MCTYPE2 twiddle_1024_3 = cmul(twiddle_1024_2, twiddle_1024_1);

    for(int i = 0; i < 1024 * CONT_SIZE; i += NUM_WARP * 64 * 4){
        int t = tid_block / 256;
        int eid_block = i + (1024 - 256) * t + tid_block;
        MCTYPE2 ele_0 = smem[eid_block];
        MCTYPE2 ele_1 = smem[eid_block + 256];
        MCTYPE2 ele_2 = smem[eid_block + 512];
        MCTYPE2 ele_3 = smem[eid_block + 768];

        ele_1 = cmul(ele_1, twiddle_1024_1);
        ele_2 = cmul(ele_2, twiddle_1024_2);
        ele_3 = cmul(ele_3, twiddle_1024_3);

        data[block_start + eid_block] = {ele_0.x + ele_1.x + ele_2.x + ele_3.x, ele_0.y + ele_1.y + ele_2.y + ele_3.y};
        data[block_start + eid_block + 256] = {ele_0.x + ele_1.y - ele_2.x - ele_3.y, ele_0.y - ele_1.x - ele_2.y + ele_3.x};
        data[block_start + eid_block + 512] = {ele_0.x - ele_1.x + ele_2.x - ele_3.x, ele_0.y - ele_1.y + ele_2.y - ele_3.y};
        data[block_start + eid_block + 768] = {ele_0.x - ele_1.y - ele_2.x + ele_3.y, ele_0.y + ele_1.x - ele_2.y - ele_3.x};
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_256_1(int step, MCTYPE2* data, MCTYPE* F_real, MCTYPE* F_imag){

    extern __shared__ MCTYPE2 smem[];

    int tid_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.y * step * 256 + blockIdx.x * CONT_SIZE;

    //chunk num per cont
    int chunk_num = CONT_SIZE / 16;

    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> frag_F_real;
    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> frag_F_imag;
    rocwmma::load_matrix_sync(frag_F_real, F_real, 16);
    rocwmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 256){
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_imag;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        int golbal_col = blockIdx.x * CONT_SIZE + threadIdx.y % chunk_num * 16 + raw_col;
        //MCTYPE2 twiddle_unit = W_N_K(step * 16, golbal_col, raw_row);
        //MCTYPE2 twiddle_factor = W_N_K(step * 16, golbal_col, 1);

        //warp start position in block
        int warp_start = i + threadIdx.y / chunk_num * chunk_num * 256 + threadIdx.y % chunk_num * 16;

        for (int j = 0; j < 4; ++j){
            int col = raw_col;
            int row = raw_row + j;
            //element position in block
            int eid_block = warp_start + row * CONT_SIZE + col;

            //element position in golbal memory
            int eid_global = block_start + eid_block / CONT_SIZE * step + eid_block % CONT_SIZE;

            MCTYPE2 in_ele = data[eid_global];

            //in_ele = cmul(in_ele, twiddle_unit);

            frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[j] = in_ele.y;

            //twiddle_unit = cmul(twiddle_unit, twiddle_factor);
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 4; ++j){
            int col = raw_col;
            int row = raw_row + j;
            //element position in block
            int eid_block = warp_start + row * CONT_SIZE + col;

            smem[eid_block].x = frag_out_real.x[j];
            smem[eid_block].y = frag_out_imag.x[j];
        }


    }

    __syncthreads();

    for (int i = 0; i < CONT_SIZE / NUM_WARP; i++){
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_imag;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        int warp_start = i * NUM_WARP * 16 + threadIdx.y * 16;

        //each block hold the same data as above, but the way of looking data has changed
        int golbal_col = i * 8 + threadIdx.y / 2;
        MCTYPE2 twiddle_unit = W_N_K(256, golbal_col, raw_row);
        MCTYPE2 twiddle_factor = W_N_K(256, golbal_col, 1);
        for (int j = 0; j < 4; ++j){
            int col = raw_col;
            int row = raw_row + j;
            int row_length = CONT_SIZE * 16; 
            int eid_block = warp_start + row * row_length + col;

            MCTYPE2 in_ele = smem[eid_block];

           // MCTYPE aa = in_ele.x;

            in_ele = cmul(in_ele, twiddle_unit);

            // MCTYPE bb = in_ele.x;
            // printf("after:%e after:%e\n", aa, bb);

            frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[j] = in_ele.y;
            
            twiddle_unit = cmul(twiddle_unit, twiddle_factor);
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 4; ++j){
            int col = raw_col;
            int row = raw_row + j;
            int row_length = CONT_SIZE * 16;
            int eid_block = warp_start + row * row_length + col;
            smem[eid_block].x = frag_out_real.x[j];
            smem[eid_block].y = frag_out_imag.x[j];

            // int kkk = threadIdx.y;
            // int eeid_block = warp_start + row * CONT_SIZE + col;
            // if(blockIdx.x == 0)
            //     printf("i:%d threadIdx.y:%d warp_start:%d eid_block:%d row:%d col:%d\n", i, kkk, warp_start, eeid_block, row, col);
        }
    }
    __syncthreads();

    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 64)
    {
        int eid = i + tid_block;
        data[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = smem[eid];
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_512_1(int step, MCTYPE2* data, MCTYPE* F_real, MCTYPE* F_imag){

    extern __shared__ MCTYPE2 smem[];

    int tid_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.y * step * 512 + blockIdx.x * CONT_SIZE;

    //chunk num per cont
    int chunk_num = CONT_SIZE / 16;

    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> frag_F_real;
    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::row_major> frag_F_imag;
    rocwmma::load_matrix_sync(frag_F_real, F_real, 16);
    rocwmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    for (int i = 0; i < 512 * CONT_SIZE; i += NUM_WARP * 256){
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_imag;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        //int golbal_col = blockIdx.x * CONT_SIZE + threadIdx.y % chunk_num * 16 + raw_col;
        //MCTYPE2 twiddle_unit = W_N_K(step * 16, golbal_col, raw_row);
        //MCTYPE2 twiddle_factor = W_N_K(step * 16, golbal_col, 1);

        //warp start position in block
        int warp_start = i + threadIdx.y / chunk_num * chunk_num * 256 + threadIdx.y % chunk_num * 16;

        for (int j = 0; j < 4; ++j){
            int col = raw_col;
            int row = raw_row + j;
            //element position in block
            int eid_block = warp_start + row * CONT_SIZE + col;

            //element position in golbal memory
            int eid_global = block_start + eid_block / CONT_SIZE * step + eid_block % CONT_SIZE;

            MCTYPE2 in_ele = data[eid_global];

            //in_ele = cmul(in_ele, twiddle_unit);

            frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[j] = in_ele.y;

            //twiddle_unit = cmul(twiddle_unit, twiddle_factor);
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 4; ++j){
            int col = raw_col;
            int row = raw_row + j;
            //element position in block
            int eid_block = warp_start + row * CONT_SIZE + col;

            smem[eid_block].x = frag_out_real.x[j];
            smem[eid_block].y = frag_out_imag.x[j];
        }


    }

    __syncthreads();

    for (int i = 0; i < 512 * CONT_SIZE / NUM_WARP / 256; i++){
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, MCTYPE, rocwmma::col_major> frag_in_imag;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        int warp_start = i * NUM_WARP * 16 + i * 256 * CONT_SIZE + threadIdx.y * 16;

        //each block hold the same data as above, but the way of looking data has changed
        int golbal_col = i * 8 + threadIdx.y / 2;
        MCTYPE2 twiddle_unit = W_N_K(256, golbal_col, raw_row);
        MCTYPE2 twiddle_factor = W_N_K(256, golbal_col, 1);
        for (int j = 0; j < 4; ++j){
            int col = raw_col;
            int row = raw_row + j;
            int row_length = CONT_SIZE * 16; 
            int eid_block = warp_start + row * row_length + col;

            MCTYPE2 in_ele = smem[eid_block];

           // MCTYPE aa = in_ele.x;

            in_ele = cmul(in_ele, twiddle_unit);

            // MCTYPE bb = in_ele.x;
            // printf("after:%e after:%e\n", aa, bb);

            frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[j] = in_ele.y;
            
            twiddle_unit = cmul(twiddle_unit, twiddle_factor);
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 4; ++j){
            int col = raw_col;
            int row = raw_row + j;
            int row_length = CONT_SIZE * 16;
            int eid_block = warp_start + row * row_length + col;
            smem[eid_block].x = frag_out_real.x[j];
            smem[eid_block].y = frag_out_imag.x[j];
        }
    }
    __syncthreads();

    // half2 twiddle_unit_2 = W_N_K(step * 512, 256 / CONT_SIZE * step);
    MCTYPE2 twiddle_unit = W_N_K(step * 512, 256 / CONT_SIZE * step, 1); // precision improved
    // half2 twiddle_factor = W_N_K(step * 512, blockIdx.x * CONT_SIZE + t_block / CONT_SIZE * step + t_block % CONT_SIZE);
    MCTYPE2 twiddle_factor = W_N_K(step * 512, blockIdx.x * CONT_SIZE + tid_block / CONT_SIZE * step + tid_block % CONT_SIZE, 1); 
    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 64)
    {
        int eid_block = i + tid_block;
        MCTYPE2 ele_0 = smem[eid_block];
        // half2 ele_1 = cmul(smem_in[eid + 256 * CONT_SIZE], twiddle_factor);
        MCTYPE2 ele_1 = cmul(smem[eid_block + 256 * CONT_SIZE], twiddle_factor); // precision improved
        data[block_start + eid_block / CONT_SIZE * step + eid_block % CONT_SIZE] = cadd(ele_0, ele_1);
        eid_block += 256 * CONT_SIZE;
        data[block_start + eid_block / CONT_SIZE * step + eid_block % CONT_SIZE] = csub(ele_0, ele_1);
        // twiddle_factor = cmul(twiddle_factor, twiddle_unit_2);
        twiddle_factor = cmul(twiddle_factor, twiddle_unit); // precision improved
    }
}

void mcfftExec(mcfftHandle plan, MCTYPE* data)
{

    const uint32_t num_warp = 16;
    const uint32_t n_cont[3] = { 32, 16, 8 };

    uint32_t step = 1;
    uint32_t RADIX = 1;
    uint32_t smem_size;
    dim3 threads, blocks;

    RADIX = plan.Ny;
    threads = {64, num_warp};
    smem_size = RADIX * sizeof(MCTYPE2) * n_cont[plan.mergings[0]];
    blocks = { plan.Nx * plan.Ny * plan.N_batch / n_cont[plan.mergings[0]] / RADIX };
    hipFuncSetAttribute((void*)plan.layer_0[plan.mergings[0]], 
                        hipFuncAttributeMaxDynamicSharedMemorySize, 
                        smem_size);
    plan.layer_0[plan.mergings[0]]<<<blocks, threads, smem_size>>>((MCTYPE2 *)data, plan.F_real, plan.F_imag);
    step *= RADIX;

    RADIX = plan.Nx;
    smem_size = RADIX * sizeof(MCTYPE2) * n_cont[plan.mergings[1]];
    blocks = {step / n_cont[plan.mergings[1]], plan.N_batch * plan.Nx * plan.Ny / step / RADIX};
    hipFuncSetAttribute((void*)plan.layer_1[plan.mergings[1]], 
                         hipFuncAttributeMaxDynamicSharedMemorySize, 
                         smem_size);
    plan.layer_1[plan.mergings[1]]<<<blocks, threads, smem_size>>>(step, (MCTYPE2 *)data, plan.F_real, plan.F_imag);
    step *= RADIX;
}

void mcfftCreate(mcfftHandle* plan, int nx, int ny, int n_batch)
{
    plan->Nx = nx;
    plan->Ny = ny;
    plan->N_batch = n_batch;
    // setup functions
    const int num_warp = 16;
    const int n_cont_256 = 32;
    const int n_cont_512 = 16;
    const int n_cont_1024 = 8;
    plan->layer_0[0] = layer_256_0<n_cont_256, num_warp>;
    plan->layer_0[1] = layer_512_0<n_cont_512, num_warp>;
    plan->layer_0[2] = layer_1024_0<n_cont_1024, num_warp>;
    plan->layer_1[0] = layer_256_1<n_cont_256, num_warp>;
    plan->layer_1[1] = layer_512_1<n_cont_512, num_warp>;
    // radices
    switch (nx)
    {
    case 256:
        plan->n_radices_x = 2;
        break;

    case 512:
        plan->n_radices_x = 3;
        plan->mergings[1] = 1;
        break;

    case 1024:
        plan->n_radices_x = 3;
        plan->radices_x[2] = 4;
        plan->mergings[1] = 2;
        break;

    default:
        break;
    }
    switch (ny)
    {
    case 256:
        plan->n_radices_y = 2;
        break;

    case 512:
        plan->n_radices_y = 3;
        plan->mergings[0] = 1;
        break;

    case 1024:
        plan->n_radices_y = 3;
        plan->radices_y[2] = 4;
        plan->mergings[0] = 2;
        break;

    default:
        break;
    }
    // F
    plan->F_real_tmp = (MCTYPE *)malloc(sizeof(MCTYPE) * 256);
    plan->F_imag_tmp = (MCTYPE *)malloc(sizeof(MCTYPE) * 256);
#pragma omp parallel for
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
        {
            plan->F_real_tmp[16 * i + j] = cosf(2 * M_PI * i * j / 16);
            plan->F_imag_tmp[16 * i + j] = -sinf(2 * M_PI * i * j / 16);
        }
    hipMalloc(&plan->F_real, sizeof(MCTYPE) * 256);
    hipMemcpy(plan->F_real, plan->F_real_tmp, sizeof(MCTYPE) * 256, hipMemcpyHostToDevice);
    hipMalloc(&plan->F_imag, sizeof(MCTYPE) * 256);
    hipMemcpy(plan->F_imag, plan->F_imag_tmp, sizeof(MCTYPE) * 256, hipMemcpyHostToDevice);
}

