#include <stdio.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>

#ifdef __use_torch__
#include <torch/extension.h>
#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#define half at::Half
#else
#include "gemv.cuh"
#include <curand.h>
#include <cublas_v2.h>
#endif

const int width_element_per_block = 32 * 2;
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
  if (WarpSize >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (WarpSize >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (WarpSize >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (WarpSize >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (WarpSize >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

template <typename T>
__global__ void gemv(T* out, const T* inA, const uint32_t* inB, const T* scales, const uint32_t* qzeros, int32_t groupsize, int32_t size_k, int32_t size_n, uint8_t add_zero_bias) {
  int bid = blockIdx.x;
  //__shared__ T vecA[size_k];
  __shared__ float bsum[2][32][32 + 1];
  float sum[2] = {0, 0};
  const int block_k = ((size_k + 31) / 32 + 7) / 8 * 8;
  int y_start = threadIdx.y * block_k;
  half2 res2 = {};
  half2 res2_1 = {};

  const half2* inA_start = (const half2*)(inA + blockIdx.y * size_k + y_start);

  int n_offset_x = bid * width_element_per_block + threadIdx.x * 2;

  int start_group_id = (y_start / groupsize);
  int compressed_idx = threadIdx.x % 4;
  half2 scale = ((half2*)(scales + start_group_id * size_n + n_offset_x))[0];
  int32_t qzero_p = ((int32_t*)(qzeros + n_offset_x / 8 +
                                start_group_id * ((size_n + 7) / 8)))[0];
  uint8_t zero_1 =(qzero_p >> (8 * (compressed_idx))) & 0xf;
  uint8_t zero_2 = ((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf;
  half2 hzero = __halves2half2(__int2half_rn(zero_1+add_zero_bias),
                               __int2half_rn(zero_2+add_zero_bias));
  half2 scale_h0 = __half2half2(scale.x);
  half2 scale_h1 = __half2half2(scale.y);
  half2 hzero_scale_0 = __half2half2(hzero.x * scale.x);
  half2 hzero_scale_1 = __half2half2(hzero.y * scale.y);

#pragma unroll
  for (int i = 0; i < block_k / 2; i += 4) {  // read half2 * 4
    res2 = {};
    res2_1 = {};
    int k_offset = y_start + i * 2;
    int g_id = k_offset / groupsize;

    const uint32_t* hinB = inB + n_offset_x + k_offset / 8 * size_n;
    uint32_t vbInt1 =
        (n_offset_x < size_n && (k_offset < size_k)) ? hinB[0] : int32_t(0);
    uint32_t vbInt2 = (n_offset_x + 1 < size_n && (k_offset < size_k))
                          ? (hinB)[1]
                          : int32_t(0);
    half2 vb[8];
    uint8_t* qweight_p1 = (uint8_t*)&vbInt1;
    uint8_t* qweight_p2 = (uint8_t*)&vbInt2;

#pragma unroll
    for (int j = 0; j < 4; j++) {
      // vb[j] = __halves2half2(__int2half_rn(((vbInt1 >> (j * 8))) & 0xF),
      //                        __int2half_rn(((vbInt1) >> (j*8+4)) & 0xF));
      // vb[j + 4] = __halves2half2(__int2half_rn(((vbInt2)>>(j*8)) & 0xF),
      //                            __int2half_rn((((vbInt2) >> (j*8+4))) &
      //                            0xF));
      vb[j] = __halves2half2(__int2half_rn(((*(qweight_p1 + j))) & 0xF),
                             __int2half_rn(((*(qweight_p1 + j)) >> 4) & 0xF));
      vb[j + 4] =
          __halves2half2(__int2half_rn(((*(qweight_p2 + j))) & 0xF),
                         __int2half_rn((((*(qweight_p2 + j)) >> 4)) & 0xF));
    }

    if (g_id > start_group_id) {
      scale = ((const half2*)(scales + g_id * size_n + n_offset_x))[0];
      qzero_p = ((const int32_t*)(qzeros + n_offset_x / 8 + g_id * ((size_n + 7) / 8)))[0];
      hzero = __halves2half2(__int2half_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
                             __int2half_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf));
      scale_h0 = __half2half2(scale.x);
      scale_h1 = __half2half2(scale.y);
      hzero_scale_0 = __half2half2(hzero.x * scale.x);
      hzero_scale_1 = __half2half2(hzero.y * scale.y);
      start_group_id++;
    }

    half2 va[4];
    va[0] = (k_offset < size_k) ? ((inA_start))[i] : res2;
    va[1] = (k_offset + 1 < size_k) ? ((inA_start))[i + 1] : res2;
    va[2] = (k_offset + 2 < size_k) ? ((inA_start))[i + 2] : res2;
    va[3] = (k_offset + 3 < size_k) ? ((inA_start))[i + 3] : res2;

#pragma unroll
    for (int j = 0; j < 4; j++) {
      vb[j] = __hfma2(scale_h0, vb[j], -hzero_scale_0);
      res2 = __hfma2(va[j], vb[j], res2);
      vb[4 + j] = __hfma2(scale_h1, vb[4 + j], -hzero_scale_1);
      res2_1 = __hfma2(va[j], vb[4 + j], res2_1);
    }

    sum[0] += __half2float(res2.x) + __half2float(res2.y);
    sum[1] += __half2float(res2_1.x) + __half2float(res2_1.y);
  }
  // sum[0] += __half2float(res2.x);
  // sum[1] +=  __half2float(res2.y);
  bsum[0][threadIdx.x][threadIdx.y] = sum[0];
  bsum[1][threadIdx.x][threadIdx.y] = sum[1];

  __syncthreads();
  sum[0] = 0;
  sum[1] = 0;

#pragma unroll
  for (int i = 0; i < 2; i++) {
    sum[i] = bsum[i][threadIdx.y][threadIdx.x];
    __syncthreads();
    sum[i] = warpReduceSum<32>(sum[i]);
    if (threadIdx.x == 0) {
      out[+blockIdx.y * size_n + bid * width_element_per_block +
          threadIdx.y * 2 + i] = __float2half_rn(sum[i]);
    }
  }
}

#ifdef __use_torch__

void lauch_Gemv_kernel(torch::Tensor& out_fp16, const torch::Tensor& a_fp16, const torch::Tensor& qweight_i32,
                       const torch::Tensor& scale_fp16, const torch::Tensor& qzeros_i32,
                       int bits, int groupsize, uint32_t mat_m, uint32_t mat_k, uint32_t mat_n, uint8_t add_zero_bias) {
  if (bits != 4 || groupsize != 128) {
    printf("only support 4bit quantization, and groupsize must be 128\n");
    abort();
  }
  const int block_k = ((mat_k + 31) / 32 + 7) / 8 * 8;

  dim3 gridDim = {(mat_n + width_element_per_block - 1) / width_element_per_block, mat_m};
  dim3 blockDim = {32, (mat_k + block_k - 1) / block_k};
  using scalar_t = half;

  gemv<scalar_t><<<gridDim, blockDim>>>(out_fp16.data_ptr<scalar_t>(),
                                        a_fp16.data_ptr<scalar_t>(),
                                        (const uint32_t*)(qweight_i32.data_ptr<int32_t>()),
                                        scale_fp16.data_ptr<scalar_t>(),
                                        (const uint32_t*)(qzeros_i32.data_ptr<int32_t>()),
                                        groupsize, mat_k, mat_n, add_zero_bias);
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    abort();
  }
}
#endif