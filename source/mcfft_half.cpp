#define __HIP_PLATFORM_HCC__
#include "../include/mcfft_half.h"

using namespace rocwmma;


const int WARP_SIZE = 64, rocwmma_M = 16, rocwmma_N = 16, rocwmma_K = 16;


__device__ inline void 
complex_mul(rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::row_major> &frag_F_real,
            rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::row_major> &frag_F_imag,
            rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> &frag_in_real, 
            rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> &frag_in_imag,
            rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> &frag_out_real, 
            rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> &frag_out_imag)
{
    float f=0.0;
    rocwmma::fill_fragment(frag_out_real, f);
    rocwmma::fill_fragment(frag_out_imag, f);

    rocwmma::mma_sync(frag_out_real, frag_F_imag, frag_in_imag, frag_out_real);
    for (int i = 0; i < frag_out_real.num_elements; i++)
        frag_out_real.x[i] = -frag_out_real.x[i];
    rocwmma::mma_sync(frag_out_real, frag_F_real, frag_in_real, frag_out_real);

    rocwmma::mma_sync(frag_out_imag, frag_F_real, frag_in_imag, frag_out_imag);
    rocwmma::mma_sync(frag_out_imag, frag_F_imag, frag_in_real, frag_out_imag);
}

__device__ __host__ inline FloatComplex W_N_K(int N, int K, int M)
{
    FloatComplex res;
    res.x = cosf(2 * M_PI * K * M / N);
    res.y = -sinf(2 * M_PI * K * M / N);
    return res;
}


__device__ inline FloatComplex cmul(FloatComplex a, FloatComplex b)
{
    FloatComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}


template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_256_0(float* in_real, float* in_imag, float* F_real, float* F_imag)
{
    //__shared__ half smem_in[256 * 2 * 32];
    extern __shared__ float smem_in[];
    int tid_block  = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 256 * CONT_SIZE;


    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::row_major> frag_F_real;
    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::row_major> frag_F_imag;
    rocwmma::load_matrix_sync(frag_F_real, F_real, 16);
    rocwmma::load_matrix_sync(frag_F_imag, F_imag, 16);



    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_imag;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        FloatComplex twiddle_unit = W_N_K(256, raw_col, raw_row);
        FloatComplex twiddle_factor = W_N_K(256, raw_col, 1);

        int warp_start_smem = i * 2 + threadIdx.y * 256 * 2;
        int warp_start_gmem = i + threadIdx.y * 256;
        rocwmma::load_matrix_sync(frag_in_real, (float*)(in_real + block_start + warp_start_gmem), 16);
        rocwmma::load_matrix_sync(frag_in_imag, (float*)(in_imag + block_start + warp_start_gmem), 16);


        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);


        rocwmma::store_matrix_sync(smem_in + warp_start_smem, frag_out_real, 16, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(smem_in + warp_start_smem + 256, frag_out_imag, 16, rocwmma::mem_row_major);

        rocwmma::load_matrix_sync(frag_in_real, (float*)(smem_in + warp_start_smem), 16);
        rocwmma::load_matrix_sync(frag_in_imag, (float*)(smem_in + warp_start_smem) + 256, 16);


        
        for (int j = 0; j < 4; ++j)
        {
            FloatComplex in_ele;
            in_ele.x = frag_in_real.x[j];
            in_ele.y = frag_in_imag.x[j];

            in_ele = cmul(in_ele, twiddle_unit);

            frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[j] = in_ele.y;
            twiddle_unit = cmul(twiddle_unit, twiddle_factor);
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        rocwmma::store_matrix_sync(smem_in + warp_start_smem, frag_out_real, 16, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(smem_in + warp_start_smem + 256, frag_out_imag, 16, rocwmma::mem_row_major);
    }

    __syncthreads();
    int num = 0;
    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 64)
    {
        int eid = i + tid_block;
        int stride = NUM_WARP * 64 * 2;
        int t = tid_block / 256;
        int smem_posi = num * stride + t * (512 - 256) + tid_block;
        in_real[block_start + eid] = smem_in[smem_posi];
        in_imag[block_start + eid] = smem_in[smem_posi + 256];
        num++;
    }

}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_512_0(float* in_real, float* in_imag, float* F_real, float* F_imag)
{
    extern __shared__ float smem_in[];
    int tid_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 512 * CONT_SIZE;

    uint32_t waveIndex = threadIdx.y;

    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::row_major> frag_F_real;
    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::row_major> frag_F_imag;
    rocwmma::load_matrix_sync(frag_F_real, F_real, 16);
    rocwmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    for (int i = 0; i < 512 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_tmp0;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_tmp1;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        FloatComplex twiddle_unit = W_N_K(256, raw_col, raw_row);
        FloatComplex twiddle_factor = W_N_K(256, raw_col, 1);

        int warp_start_smem = i * 2 + threadIdx.y * 256 * 2;
        int warp_start_gmem = i + threadIdx.y * 256;
        rocwmma::load_matrix_sync(frag_in_real, (float*)(in_real + block_start + warp_start_gmem), 16);
        rocwmma::load_matrix_sync(frag_in_imag, (float*)(in_imag + block_start + warp_start_gmem), 16);

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        rocwmma::store_matrix_sync(smem_in + warp_start_smem, frag_out_real, 16, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(smem_in + warp_start_smem + 256, frag_out_imag, 16, rocwmma::mem_row_major);

        rocwmma::load_matrix_sync(frag_in_real, (float*)(smem_in + warp_start_smem), 16);
        rocwmma::load_matrix_sync(frag_in_imag, (float*)(smem_in + warp_start_smem) + 256, 16);

        
        for (int j = 0; j < 4; ++j)
        {
            FloatComplex in_ele;
            in_ele.x = frag_in_real.x[j];
            in_ele.y = frag_in_imag.x[j];

            in_ele = cmul(in_ele, twiddle_unit);

            frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[j] = in_ele.y;
            twiddle_unit = cmul(twiddle_unit, twiddle_factor);
        }

       complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        rocwmma::store_matrix_sync(smem_in + warp_start_smem, frag_out_real, 16, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(smem_in + warp_start_smem + 256, frag_out_imag, 16, rocwmma::mem_row_major);

    }

    __syncthreads();
    int num = 0;
    FloatComplex twiddle_512;
    twiddle_512 = W_N_K(512, tid_block % 256, 1);

    for (int i = 0; i < 512 * CONT_SIZE; i += NUM_WARP * 64 * 2)
    {
        int stride = NUM_WARP * 64 * 4;
        int t = tid_block / 256;
        int eid = i + (512 - 256) * t + tid_block;
        int smem_posi = num * stride + t * (1024 - 256) + tid_block;
        FloatComplex ele_0, ele_1;
        ele_0.x = smem_in[smem_posi];
        ele_0.y = smem_in[smem_posi + 256];
        ele_1.x = smem_in[smem_posi + 512];
        ele_1.y = smem_in[smem_posi + 512 + 256];
        ele_1 = cmul(ele_1, twiddle_512);
        in_real[block_start + eid] = ele_0.x + ele_1.x;
        in_imag[block_start + eid] = ele_0.y + ele_1.y;
        in_real[block_start + eid + 256] = ele_0.x - ele_1.x;
        in_imag[block_start + eid + 256] = ele_0.y - ele_1.y;
        num++;
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_1024_0(float* in_real, float* in_imag, float* F_real, float* F_imag)
{
    extern __shared__ float smem_in[];
    int tid_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 1024 * CONT_SIZE;

    uint32_t waveIndex = threadIdx.y;

    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::row_major> frag_F_real;
    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::row_major> frag_F_imag;
    rocwmma::load_matrix_sync(frag_F_real, F_real, 16);
    rocwmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    for (int i = 0; i < 1024 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_tmp0;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_tmp1;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        FloatComplex twiddle_unit = W_N_K(256, raw_col, raw_row);
        FloatComplex twiddle_factor = W_N_K(256, raw_col, 1);

        int warp_start_smem = i * 2 + threadIdx.y * 256 * 2;
        int warp_start_gmem = i + threadIdx.y * 256;
        rocwmma::load_matrix_sync(frag_in_real, (float*)(in_real + block_start + warp_start_gmem), 16);
        rocwmma::load_matrix_sync(frag_in_imag, (float*)(in_imag + block_start + warp_start_gmem), 16);

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        rocwmma::store_matrix_sync(smem_in + warp_start_smem, frag_out_real, 16, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(smem_in + warp_start_smem + 256, frag_out_imag, 16, rocwmma::mem_row_major);

        rocwmma::load_matrix_sync(frag_in_real, (float*)(smem_in + warp_start_smem), 16);
        rocwmma::load_matrix_sync(frag_in_imag, (float*)(smem_in + warp_start_smem) + 256, 16);

        
        for (int j = 0; j < 4; ++j)
        {
            FloatComplex in_ele;
            in_ele.x = frag_in_real.x[j];
            in_ele.y = frag_in_imag.x[j];

            in_ele = cmul(in_ele, twiddle_unit);

            frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[j] = in_ele.y;
            twiddle_unit = cmul(twiddle_unit, twiddle_factor);
        }

       complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        rocwmma::store_matrix_sync(smem_in + warp_start_smem, frag_out_real, 16, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(smem_in + warp_start_smem + 256, frag_out_imag, 16, rocwmma::mem_row_major);

    }

    __syncthreads();
    int num = 0;
    FloatComplex twiddle_1024_1 = W_N_K(1024, tid_block % 256, 1);
    FloatComplex twiddle_1024_2 = cmul(twiddle_1024_1, twiddle_1024_1);
    FloatComplex twiddle_1024_3 = cmul(twiddle_1024_2, twiddle_1024_1);

    for (int i = 0; i < 1024 * CONT_SIZE; i += NUM_WARP * 64 * 4)
    {
        int stride = NUM_WARP * 64 * 8;
        int t = tid_block / 256;
        int eid = i + (1024 - 256) * t + tid_block;
        int smem_posi = num * stride + t * (2048 - 256) + tid_block;

        FloatComplex ele_0, ele_1, ele_2, ele_3;
        ele_0.x = smem_in[smem_posi];
        ele_0.y = smem_in[smem_posi + 256];
        ele_1.x = smem_in[smem_posi + 512];
        ele_1.y = smem_in[smem_posi + 512 + 256];
        ele_2.x = smem_in[smem_posi + 1024];
        ele_2.y = smem_in[smem_posi + 1024 + 256];
        ele_3.x = smem_in[smem_posi + 1536];
        ele_3.y = smem_in[smem_posi + 1536 + 256];
        ele_1 = cmul(ele_1, twiddle_1024_1);
        ele_2 = cmul(ele_2, twiddle_1024_2);
        ele_3 = cmul(ele_3, twiddle_1024_3); 
        in_real[block_start + eid] = ele_0.x + ele_1.x + ele_2.x + ele_3.x;
        in_imag[block_start + eid] = ele_0.y + ele_1.y + ele_2.y + ele_3.y;
        in_real[block_start + eid + 256] = ele_0.x + ele_1.y - ele_2.x - ele_3.y;
        in_imag[block_start + eid + 256] = ele_0.y - ele_1.x - ele_2.y + ele_3.x;
        in_real[block_start + eid + 512] = ele_0.x - ele_1.x + ele_2.x - ele_3.x;
        in_imag[block_start + eid + 512] = ele_0.y - ele_1.y + ele_2.y - ele_3.y;
        in_real[block_start + eid + 768] = ele_0.x - ele_1.y - ele_2.x + ele_3.y;
        in_imag[block_start + eid + 768] = ele_0.y + ele_1.x - ele_2.y - ele_3.x;
        num++;
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_256_1(int step, float* in_real, float* in_imag, float* F_real, float* F_imag){
    int imag_start = CONT_SIZE * 256;
    extern __shared__ float smem[];

    int tid_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.y * step * 256 + blockIdx.x * CONT_SIZE;

    uint32_t waveIndex = threadIdx.y;


    //chunk num per cont
    int chunk_num = CONT_SIZE / 16;

    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::row_major> frag_F_real;
    rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::row_major> frag_F_imag;
    rocwmma::load_matrix_sync(frag_F_real, F_real, 16);
    rocwmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 256){
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_imag;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        int golbal_col = blockIdx.x * CONT_SIZE + threadIdx.y % chunk_num * 16 + raw_col;
        FloatComplex twiddle_unit = W_N_K(step * 16, golbal_col, raw_row);
        FloatComplex twiddle_factor = W_N_K(step * 16, golbal_col, 1);

        //warp start position in block
        int warp_start = i + threadIdx.y / chunk_num * chunk_num * 256 + threadIdx.y % chunk_num * 16;

        for (int j = 0; j < 4; ++j){
            int col = raw_col;
            int row = raw_row + j;
            //element position in block
            int eid_block = warp_start + row * CONT_SIZE + col;

            //element position in golbal memory
            int eid_global = block_start + eid_block / CONT_SIZE * step + eid_block % CONT_SIZE;

            FloatComplex in_ele;
            in_ele.x = in_real[eid_global];
            in_ele.y = in_imag[eid_global];

            
            in_ele = cmul(in_ele, twiddle_unit);
            frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[j] = in_ele.y;

            twiddle_unit = cmul(twiddle_unit, twiddle_factor);
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 4; ++j){
            int col = raw_col;
            int row = raw_row + j;
            //element position in block
            int eid_block = warp_start + row * CONT_SIZE + col;

            smem[eid_block] = frag_out_real.x[j];
            smem[imag_start + eid_block] = frag_out_imag.x[j]; 
        }


    }

    __syncthreads();

    for (int i = 0; i < CONT_SIZE / NUM_WARP; i++){
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> frag_out_real;
        rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, float> frag_out_imag;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_real;
        rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, float, rocwmma::col_major> frag_in_imag;

        int raw_col = threadIdx.x % 16;
        int raw_row = threadIdx.x / 16 * 4;
        int warp_start = i * NUM_WARP * 16 + threadIdx.y * 16;

        //each block hold the same data as above, but the way of looking data has changed
        int golbal_col = blockIdx.x * CONT_SIZE + i * step * (16 * NUM_WARP / CONT_SIZE) + threadIdx.y / chunk_num * step + threadIdx.y % chunk_num * 16 + raw_col;
        FloatComplex twiddle_unit = W_N_K(step * 256, golbal_col, raw_row);
        FloatComplex twiddle_factor = W_N_K(step * 256, golbal_col, 1);
        for (int j = 0; j < 4; ++j){
            int col = raw_col;
            int row = raw_row + j;
            int row_length = CONT_SIZE * 16; 
            int eid_block = warp_start + row * row_length + col;
            FloatComplex in_ele;


            in_ele.x = smem[eid_block];
            in_ele.y = smem[imag_start + eid_block];

           // float aa = in_ele.x;

            in_ele = cmul(in_ele, twiddle_unit);

            // float bb = in_ele.x;
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
            smem[eid_block] = frag_out_real[j];
            smem[imag_start + eid_block] = frag_out_imag[j];

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
        in_real[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = smem[eid];
        in_imag[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = smem[imag_start + eid];
    }
}

void mcfftExec(mcfftHandle plan, float* data_real, float* data_imag)
{
    const uint32_t num_warp = 16;
    const uint32_t n_cont[3] = { 32, 16, 8 };

    uint32_t step = 1;
    uint32_t RADIX = 1;
    uint32_t smem_size;
    dim3 threads, blocks;

    //printfindevice<<<1,1>>>(data);

    // V100
    switch (plan.mergings[0])
    {
    case 0:
        RADIX = 256;
        break;

    case 1:
        RADIX = 512;
        break;

    case 2:
        RADIX = 1024;
        break;

    default:
        break;
    }
    threads = { 64, num_warp };
    blocks = { plan.N * plan.N_batch / n_cont[plan.mergings[0]] / RADIX };
    smem_size = RADIX * 2 * sizeof(float) * n_cont[plan.mergings[0]];

    hipFuncSetAttribute((void*)plan.layer_0[plan.mergings[0]], 
                        hipFuncAttributeMaxDynamicSharedMemorySize, 
                        smem_size);

    plan.layer_0[plan.mergings[0]]<<<blocks, threads, smem_size>>>((float*)data_real, 
                                                                   (float*)data_imag, 
                                                                   plan.F_real, 
                                                                   plan.F_imag);
    step *= RADIX;

    for (int i = 1; i < plan.n_mergings; ++i)
    {
        switch (plan.mergings[i])
        {
        case 0:
            RADIX = 256;
            break;

        case 1:
            RADIX = 512;
            break;

        case 2:
            RADIX = 1024;
            break;

        default:
            break;
        }

        blocks = { step / n_cont[plan.mergings[i]], plan.N_batch * plan.N / step / RADIX };
        hipFuncSetAttribute((void*)plan.layer_1[plan.mergings[i]], 
                            hipFuncAttributeMaxDynamicSharedMemorySize, 
                            RADIX * sizeof(float) * 2 * n_cont[plan.mergings[i]]);


        plan.layer_1[plan.mergings[i]]<<<blocks, threads, smem_size>>>(step, 
                                                                       (float*)data_real,
                                                                       (float*)data_imag, 
                                                                       plan.F_real, 
                                                                       plan.F_imag);
        step *= RADIX;
    }
}

void mcfftCreate(mcfftHandle* plan, int n, int n_batch)
{
    plan->N = n;
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
    //plan->layer_1[1] = layer_512_1<n_cont_512, num_warp>;
    // radices
    switch (n)
    {
    case 256:
        plan->n_radices = 2;
        plan->n_mergings = 1;
        break;

    case 512:
        plan->n_radices = 3;
        plan->radices[2] = 2;
        plan->n_mergings = 1;
        plan->mergings[0] = 1;
        break;

    case 1024:
        plan->n_radices = 3;
        plan->radices[2] = 4;
        plan->n_mergings = 1;
        plan->mergings[0] = 2;
        break;

    case 131072:
        plan->n_radices = 5;
        plan->radices[2] = 2;
        plan->n_mergings = 2;
        plan->mergings[0] = 1;
        break;

    case 262144:
        plan->n_radices = 6;
        plan->radices[2] = 2;
        plan->radices[5] = 2;
        plan->n_mergings = 2;
        plan->mergings[0] = 1;
        plan->mergings[1] = 1;
        break;

    case 524288:
        plan->n_radices = 6;
        plan->radices[2] = 4;
        plan->radices[5] = 2;
        plan->n_mergings = 2;
        plan->mergings[0] = 2;
        plan->mergings[1] = 1;
        break;

    case 16777216:
        plan->n_radices = 6;
        plan->n_mergings = 3;
        break;

    case 33554432:
        plan->n_radices = 7;
        plan->radices[2] = 2;
        plan->n_mergings = 3;
        plan->mergings[0] = 1;
        break;

    case 67108864:
        plan->n_radices = 8;
        plan->radices[2] = 2;
        plan->radices[5] = 2;
        plan->n_mergings = 3;
        plan->mergings[0] = 1;
        plan->mergings[1] = 1;
        break;

    case 134217728:
        plan->n_radices = 9;
        plan->radices[2] = 2;
        plan->radices[5] = 2;
        plan->radices[8] = 2;
        plan->n_mergings = 3;
        plan->mergings[0] = 1;
        plan->mergings[1] = 1;
        plan->mergings[2] = 1;
        break;

    default:
        break;
    }
    // F
    plan->F_real_tmp = (float*)malloc(sizeof(float) * 256);
    plan->F_imag_tmp = (float*)malloc(sizeof(float) * 256);
#pragma omp parallel for
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
        {
            plan->F_real_tmp[16 * i + j] = cosf(2 * M_PI * i * j / 16);
            plan->F_imag_tmp[16 * i + j] = -sinf(2 * M_PI * i * j / 16);
        }
    hipMalloc(&plan->F_real, sizeof(float) * 256);
    hipMemcpy(plan->F_real, plan->F_real_tmp, sizeof(float) * 256, hipMemcpyHostToDevice);
    hipMalloc(&plan->F_imag, sizeof(float) * 256);
    hipMemcpy(plan->F_imag, plan->F_imag_tmp, sizeof(float) * 256, hipMemcpyHostToDevice);
}

