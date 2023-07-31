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

// Define some error checking macros.
#define cudaErrCheck(stat)                     \
  {                                            \
    cudaErrCheck_((stat), __FILE__, __LINE__); \
  }
void cudaErrCheck_(cudaError_t stat, const char* file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
  }
}
#define cublasErrCheck(stat)                     \
  {                                              \
    cublasErrCheck_((stat), __FILE__, __LINE__); \
  }
void cublasErrCheck_(cublasStatus_t stat, const char* file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
  }
}
//#define curandErrCheck(stat)                     \
//  {                                              \
//    curandErrCheck_((stat), __FILE__, __LINE__); \
//  }
//void curandErrCheck_(curandStatus_t stat, const char* file, int line) {
//  if (stat != CURAND_STATUS_SUCCESS) {
//    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
//  }
//}
extern int MATRIX_M;
extern int MATRIX_K;
extern int MATRIX_N;

constexpr int kBlockSize = 256;
//constexpr int kNumWaves = 32;

namespace cuda_quant {
#define FETCH_UINT2(pointer) (reinterpret_cast<uint2*>(&(pointer))[0])
#define FETCH_HALF2(pointer) (reinterpret_cast<half2*>(&(pointer))[0])

template <typename T, int WBITS>
__global__ void DequantizeAndUnpackWeight248(T* out, uint32_t* qweight, T* scale, uint32_t* zeros, int group_size, const int in_features, const int n) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  const int qweight_rows = (in_features * WBITS + 31) / 32;
  const int half_n = n/2;

  const int compress_group_size = 32 / WBITS;
  const int max_num_in_bits = (1 << WBITS) - 1;
  int col_ind = (tid % half_n)*2;
  int weight_in_row = tid / half_n * compress_group_size;
  half2 scale_v = FETCH_HALF2(scale[weight_in_row / group_size*n + col_ind]);
  uint32_t zero_v = zeros[weight_in_row / group_size * (n / compress_group_size) + (col_ind) / compress_group_size];
  int zero_ind = col_ind % compress_group_size;
  uint8_t zv1 = (zero_v >> (zero_ind * WBITS)) & max_num_in_bits;
  uint8_t zv2 = (zero_v >> (zero_ind * WBITS + WBITS)) & max_num_in_bits;
  half2 scale_zeros = __hmul2(__halves2half2(__short2half_rn(zv1), __short2half_rn(zv2)), scale_v);
  half2* out_h2 = reinterpret_cast<half2*>(out);

  uint2 weight_int2 = FETCH_UINT2(qweight[tid * 2]);
  uint32_t weight_v1 = weight_int2.x;
  uint32_t weight_v2 = weight_int2.y;
  // decompress weights
  int remains = in_features - weight_in_row;
  if (remains >= compress_group_size) {
#pragma unroll
    for (int i = 0; i < compress_group_size; i++) {
      uint8_t wv1 = (weight_v1 >> (i * WBITS)) & max_num_in_bits;
      uint8_t wv2 = (weight_v2 >> (i * WBITS)) & max_num_in_bits;
      half2 wv = __halves2half2(__short2half_rn(wv1), __short2half_rn(wv2));
      out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, -scale_zeros);
    }
  } else {
    for (int i = 0; i < remains; i++) {
      uint8_t wv1 = (weight_v1 >> (i * WBITS)) & max_num_in_bits;
      uint8_t wv2 = (weight_v2 >> (i * WBITS)) & max_num_in_bits;
      half2 wv = __halves2half2(__short2half_rn(wv1), __short2half_rn(wv2));
      out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, -scale_zeros);
    }
  }
  }

template <int WBITS>
__device__ __forceinline__ uint8_t iterator_qweight(const uint32_t* ptr, int idx) {
  int start_bits = idx * WBITS;
  int first = start_bits / 32;
  int end_bits = (start_bits + WBITS);
  int second = end_bits / 32;
  start_bits = start_bits % 32;
  end_bits = end_bits % 32;
  if (first == second) {
    return (ptr[first] >> (start_bits)) & ((1 << WBITS) - 1);
  } else {
    uint8_t v = (ptr[first] >> (start_bits));
    v |= ((ptr[second]) & ((1 << (end_bits)) - 1))<< (32-start_bits);
    return v;
  }
}

template <typename T, int WBITS>
__global__ void DequantizeAndUnpackWeight3567(T* out, uint32_t* qweight, T* scale, uint32_t* zeros, int group_size, const int in_features, const int row_n) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  __shared__ uint32_t qweight_shared[WBITS * kBlockSize];
  const int qweight_rows = (in_features * WBITS + 31) / 32;
  const int group_row_n = row_n * WBITS;
  int total_qw = qweight_rows * row_n;

  uint32_t* qweight_thread = qweight_shared + WBITS * threadIdx.x;

  int qweight_start = tid / row_n * group_row_n + tid % row_n;
#pragma unroll
  for (int j = 0; j < WBITS; j++) {
    int ind = qweight_start + row_n * j;
    qweight_thread[j] = ind < total_qw ? qweight[ind] : 0;
  }
  
  const int max_num_in_bits = (1 << WBITS) - 1;
  const int col_ind = (tid % row_n);
  const int compress_group_size = 32;
  const int fp16_weight_in_row = tid / row_n * compress_group_size;
  T scale_v[4];
  const int scale_zero_from = fp16_weight_in_row / group_size;
  const int scale_zero_to = (fp16_weight_in_row + compress_group_size) / group_size;

  // decompress scales
  for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
    scale_v[i] = scale[scale_zero_from_i * row_n + col_ind];
  }

  // decompress zeros
  uint8_t zv1[4] = {0, 0, 0, 0};
  const int zero_col_from = col_ind * WBITS / 32;
  const int zero_col_to = (col_ind + 1) * WBITS / 32;
  const int qzero_width = (row_n * WBITS + 32 - 1) / 32;
  for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
    uint32_t zero_v = zeros[scale_zero_from_i * qzero_width + zero_col_from];
    const int zero_bits_last = (((col_ind)*WBITS) % 32);
    zv1[i] = (zero_v >> zero_bits_last) & max_num_in_bits;
    if (zero_col_from != zero_col_to) {
      const int zero_bits_first = ((col_ind + 1) * WBITS) % 32;
      uint32_t zero_v1 = zeros[scale_zero_from * qzero_width + zero_col_to];
      zv1[i] |= (zero_v1 & ((1 << zero_bits_first) - 1))  << (32-zero_bits_last);
    }
  }

  T scale_zeros[4];
  for (int i = 0; i <= scale_zero_to - scale_zero_from; i++) {
    scale_zeros[i] = __hmul(__short2half_rn(zv1[i]), scale_v[i]);
  }
  half2 scale_2 = __halves2half2(scale_v[0], scale_v[0]);
  half2 scale_zeros_2 = __halves2half2(scale_zeros[0], scale_zeros[0]);
  const int out_offset = ((fp16_weight_in_row) * row_n + col_ind);
  // decompress weights
  int remains = in_features - fp16_weight_in_row;
  if (remains >= compress_group_size) {
#pragma unroll
  for (int i = 0; i < compress_group_size / 2; i++) {
    uint8_t wv1 = 0;
    uint8_t wv2 = 0;
    wv1 = iterator_qweight<WBITS>(qweight_thread, i);
    wv2 = iterator_qweight<WBITS>(qweight_thread, 16 + i);

    half2 wv = __halves2half2(__short2half_rn(wv1), __short2half_rn(wv2));
    if (group_size < 32) {
      scale_2 = __halves2half2(scale_v[i / group_size], scale_v[(i + 16) / group_size]);
      scale_zeros_2 = __halves2half2(scale_zeros[i / group_size], scale_zeros[(i + 16) / group_size]);
    }
    half2 res = __hfma2(wv, scale_2, -scale_zeros_2);
    //if (bid == 0 && threadIdx.x == 1 && i == 1) {
    //  printf("%d,%d,%d,%d,%d,%f,%f\n", col_ind, int(wv1), int(wv2), row_n, col_ind, __half2float(wv.x), __half2float(wv.y));
    //}
    out[out_offset + i* row_n] = res.x;
    out[(out_offset + (i + 16) * row_n )] = res.y;
  }
  } else {
    // decompress weights
    for (int i = 0; i < remains; i++) {
      uint8_t wv1 = iterator_qweight<WBITS>(qweight_thread, i);
      T wv = __short2half_rn(wv1);
      if (group_size < 32) {
        scale_2.x = scale_v[i / group_size];
        scale_zeros_2.x = scale_zeros[i / group_size];
      }
      T res = __hfma(wv, scale_2.x, -scale_zeros_2.x);
      out[out_offset + i * row_n] = res;
    }
  }
}

template <typename T, int WBITS>
__device__ __forceinline__ uchar2 iterator_qweight_v2(const T* ptr, int idx) {
  int start_bits = idx * WBITS;
  int first = start_bits / 32;
  int end_bits = (start_bits + WBITS);
  int second = end_bits / 32;
  start_bits = start_bits % 32;
  end_bits = end_bits % 32;
  uchar2 res;
  if (first == second) {
    res.x = (ptr[first].x >> (start_bits)) & ((1 << WBITS) - 1);
    res.y = (ptr[first].y >> (start_bits)) & ((1 << WBITS) - 1);
    return res;
  } else {
    res.x = (ptr[first].x >> (start_bits));
    res.y = (ptr[first].y >> (start_bits));

    res.x |= ((ptr[second].x) & ((1 << (end_bits)) - 1))<< (32-start_bits);
    res.y |= ((ptr[second].y) & ((1 << (end_bits)) - 1))<< (32-start_bits);
    return res;
  }
}

template <typename T, int WBITS>
__global__ void DequantizeAndUnpackWeight3567_v2(T* out, const uint32_t* qweight, const T* scale, const uint32_t* zeros, int group_size, const int in_features, const int row_n) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  const int qweight_rows = (in_features * WBITS + 31) / 32;
  __shared__ uint2 qweight_shared[WBITS * kBlockSize];
  const int half_n = row_n / 2;

  const int group_row_n = half_n * (WBITS==6?3:WBITS);
  int total_qw = qweight_rows * half_n;

  uint2* qweight_thread = qweight_shared + WBITS * threadIdx.x;

  int qweight_start = tid / half_n * group_row_n + tid % half_n;
  const uint2* qweigh2 = (const uint2*)qweight;
#pragma unroll
  for (int j = 0; j < WBITS; j++) {
    int ind = qweight_start + half_n * j;
    qweight_thread[j] = ind < total_qw ? (qweigh2[ind]) : uint2();
  }
  
  const int max_num_in_bits = (1 << WBITS) - 1;
  const int col_ind = (tid % half_n);
  const int compress_group_size = 32;
  const int fp16_weight_in_row = tid / half_n * compress_group_size;
  half2 scale_v[4];
  const int scale_zero_from = fp16_weight_in_row / group_size;
  const int scale_zero_to = (fp16_weight_in_row + compress_group_size) / group_size;

  // decompress scales
  const half2 *scale2 = reinterpret_cast<const half2*>(scale);
  for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
    scale_v[i] = (scale2[scale_zero_from_i * half_n + col_ind]);
  }

  // decompress zeros
  uchar2 zv1[4];
  int half_col_ind = col_ind * 2;
  const int zero_col_from = half_col_ind * WBITS / 32;
  const int zero_col_to = ((half_col_ind + 1) * WBITS - 1) / 32;
  const int zero_col_to_2 = ((half_col_ind + 2) * WBITS - 1) / 32;
  const int qzero_width = (row_n * WBITS + 32 - 1) / 32;
  for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
    uint32_t zero_v = zeros[scale_zero_from_i * qzero_width + zero_col_from];
    const int zero_bits_last = (((half_col_ind)*WBITS) % 32);
    zv1[i].x = (zero_v >> zero_bits_last) & max_num_in_bits;
    if (zero_col_from != zero_col_to) {
      const int zero_bits_first = ((half_col_ind + 1) * WBITS) % 32;
      uint32_t zero_v1 = zeros[scale_zero_from * qzero_width + zero_col_to];
      zv1[i].x |= (zero_v1 & ((1 << zero_bits_first) - 1)) << (32-zero_bits_last);

      zv1[i].y = (zero_v1 >> zero_bits_first) & max_num_in_bits;
    } else {
      zv1[i].y = (zero_v >> (zero_bits_last+WBITS)) & max_num_in_bits;
    }

    if (zero_col_to != zero_col_to_2) {
      const int zero_bits_first = ((half_col_ind + 2) * WBITS) % 32;
      uint32_t zero_v1 = zeros[scale_zero_from * qzero_width + zero_col_to_2];
      zv1[i].y |= (zero_v1 & ((1 << zero_bits_first) - 1)) << (32 - zero_bits_last - WBITS);
    }
  }

  half2 scale_zeros[4];
  for (int i = 0; i <= scale_zero_to - scale_zero_from; i++) {
    scale_zeros[i] = __hmul2(__halves2half2(__ushort2half_rn(zv1[i].x), __ushort2half_rn(zv1[i].y)), scale_v[i]);
  }
  half2 scale_2 =  scale_v[0];
  half2 scale_zeros_2 = scale_zeros[0];

  const int out_offset = ((fp16_weight_in_row)*half_n + col_ind);
  half2* out_h2 = reinterpret_cast<half2*>(out);
  // decompress weights
  int remains = in_features - fp16_weight_in_row;
  if (remains >= compress_group_size) {
#pragma unroll
  for (int i = 0; i < compress_group_size / 2; i++) {
    uchar2 wv1= iterator_qweight_v2<uint2, WBITS>(qweight_thread, i);
    uchar2 wv2 = iterator_qweight_v2<uint2, WBITS>(qweight_thread, 16 + i);

    half2 wv = __halves2half2(__ushort2half_rn(wv1.x), __ushort2half_rn(wv1.y));
    if (group_size < 32) {
      half2 scale_2 = scale_v[i / group_size];
      half2 scale_zeros_2 = scale_zeros[i / group_size];
    }
    half2 res = __hfma2(wv, scale_2, -scale_zeros_2);
    out_h2[out_offset + i * half_n] = res;

    wv = __halves2half2(__ushort2half_rn(wv2.x), __ushort2half_rn(wv2.y));
    if (group_size < 32) {
      half2 scale_2 = scale_v[(i + 16) / group_size];
      half2 scale_zeros_2 = scale_zeros[(i + 16) / group_size];
    }
    res = __hfma2(wv, scale_2, -scale_zeros_2);
    out_h2[(out_offset + (i + 16) * half_n)] = res;
  }
  } else {
    // decompress weights
    for (int i = 0; i < remains; i++) {
      uchar2 wv1 = iterator_qweight_v2<uint2, WBITS>(qweight_thread, i);

      half2 wv = __halves2half2(__ushort2half_rn(wv1.x), __ushort2half_rn(wv1.y));
      if (group_size < 32) {
        scale_2 = scale_v[i / group_size];
        scale_zeros_2 = scale_zeros[i / group_size];
      }
      half2 res = __hfma2(wv, scale_2, -scale_zeros_2);
      out_h2[out_offset + i * half_n] = res;
    }
  }
  }

__global__ void convertFp32ToFp16(half* out, float* in, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = __float2half(in[idx]);
  }
}
__global__ void convertFp16ToFp32(float* out, half* in, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = __half2float(in[idx]);
  }
}
}  // namespace cuda_quant
#ifndef assert
#define assert(x) \
  if (!(x)) {     \
    abort();      \
  }
#endif

template <typename scalar_t>
static void lauch_dq_248(scalar_t* b_fp16, int32_t* qweight_i32_i, scalar_t* scale_fp16, int32_t* qzeros_i32_i, int bits, int groupsize, uint32_t mat_k, uint32_t mat_n) {
  if constexpr (std::is_same<scalar_t, double>::value) {
    return;
  } else if constexpr (std::is_same<scalar_t, float>::value) {
    return;
  } 
  const uint32_t conpress_ratio = 32 / bits;
  dim3 gridDim = {(mat_n / 2 * ((mat_k + conpress_ratio-1) / conpress_ratio) + kBlockSize - 1) / kBlockSize};
  dim3 blockDim = {kBlockSize};
#ifdef __use_torch__
  auto stream = at::cuda::getCurrentCUDAStream().stream();
#else
  cudaStream_t stream = nullptr;
#endif
  uint32_t* qweight_i32 = reinterpret_cast<uint32_t*>(qweight_i32_i);
  uint32_t* qzeros_i32 = reinterpret_cast<uint32_t*>(qzeros_i32_i);
  using cuda_quant::DequantizeAndUnpackWeight248;
  switch(bits) {
    case 2:
      DequantizeAndUnpackWeight248<half, 2><<<gridDim, blockDim, 0, stream>>>((half*)b_fp16, qweight_i32, (half*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n);
      break;
  case 4:
      DequantizeAndUnpackWeight248<half, 4><<<gridDim, blockDim, 0, stream>>>((half*)b_fp16, qweight_i32, (half*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n);
      break;
  case 8:
      DequantizeAndUnpackWeight248<half, 8><<<gridDim, blockDim, 0, stream>>>((half*)b_fp16, qweight_i32, (half*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n);
      break;
  default:
  printf("error bits\n");
      assert(false);
  }
}

template <typename scalar_t>
static void lauch_dq_3567(scalar_t* b_fp16, int32_t* qweight_i32_i, scalar_t* scale_fp16, int32_t* qzeros_i32_i, int bits, int groupsize, uint32_t mat_k, uint32_t mat_n) {
  if constexpr (std::is_same<scalar_t, double>::value) {
  return;
  } else if constexpr (std::is_same<scalar_t, float>::value) {
  return;
  }
  const uint32_t conpress_ratio = 32;
  dim3 gridDim = {static_cast<unsigned int>((mat_n / 2 * (mat_k + conpress_ratio - 1) / conpress_ratio + kBlockSize - 1) / kBlockSize)};
  dim3 blockDim = {kBlockSize};
#ifdef __use_torch__
  auto stream = at::cuda::getCurrentCUDAStream().stream();
#else
  cudaStream_t stream = nullptr;
#endif
  uint32_t* qweight_i32 = reinterpret_cast<uint32_t*>(qweight_i32_i);
  uint32_t* qzeros_i32 = reinterpret_cast<uint32_t*>(qzeros_i32_i);
  using cuda_quant::DequantizeAndUnpackWeight3567_v2;
  switch (bits) {
  case 3:
      DequantizeAndUnpackWeight3567_v2<half, 3><<<gridDim, blockDim, 0, stream>>>((half*)b_fp16, qweight_i32, (half*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n);
      break;
  case 5:
      DequantizeAndUnpackWeight3567_v2<half, 5><<<gridDim, blockDim, 0, stream>>>((half*)b_fp16, qweight_i32, (half*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n);
      break;
  case 6:
      DequantizeAndUnpackWeight3567_v2<half, 6><<<gridDim, blockDim, 0, stream>>>((half*)b_fp16, qweight_i32, (half*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n);
      break;
  case 7:
      DequantizeAndUnpackWeight3567_v2<half, 7><<<gridDim, blockDim, 0, stream>>>((half*)b_fp16, qweight_i32, (half*)scale_fp16, qzeros_i32, groupsize, mat_k, mat_n);
      break;
  default:
  printf("error bits\n");
      assert(false);
  }
}


#ifdef __use_torch__

void lauch_deqantize_cuda_pt_kernel(torch::Tensor& b_fp16, const torch::Tensor& qweight_i32, const torch::Tensor& scale_fp16, const torch::Tensor& qzeros_i32,
                                    int bits, int groupsize, uint32_t mat_k, uint32_t mat_n) {
  if (bits == 2 || bits == 4 || bits == 8) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(b_fp16.scalar_type(), "lauch_deqantize_cuda_pt_kernel", ([&] {
      lauch_dq_248<scalar_t>(b_fp16.data_ptr<scalar_t>(), qweight_i32.data_ptr<int32_t>(), scale_fp16.data_ptr<scalar_t>(), qzeros_i32.data_ptr<int32_t>(), bits, groupsize, mat_k, mat_n);
    }));
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(b_fp16.scalar_type(), "lauch_deqantize_cuda_pt_kernel", ([&] {
      lauch_dq_3567<scalar_t>(b_fp16.data_ptr<scalar_t>(), qweight_i32.data_ptr<int32_t>(), scale_fp16.data_ptr<scalar_t>(), qzeros_i32.data_ptr<int32_t>(), bits, groupsize, mat_k, mat_n);
    }));
  }
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
#else
void lauch_deqantize_cuda_kernel(short* b_fp16, uint32_t* qweight_i32, short* scale_fp16, uint32_t* qzeros_i32, int bits, int groupsize, uint32_t mat_k, uint32_t mat_n) {
  if (bits == 2 || bits == 4 || bits == 8) {
  lauch_dq_248<short>(b_fp16, (int32_t*)qweight_i32, scale_fp16, (int32_t*)qzeros_i32, bits, groupsize, mat_k, mat_n);
  } else {
  lauch_dq_3567<short>(b_fp16, (int32_t*)qweight_i32, scale_fp16, (int32_t*)qzeros_i32, bits, groupsize, mat_k, mat_n);
  }
}

int QbitGemv(SampleData* p_sampledata) {
  SampleData& sampledata = *p_sampledata;

  float* a_fp32;
  float* b_fp32;
  half* c_fp16;
  half* c_cublas_fp16;
  half* a_fp16;
  half* b_fp16;
  half* b_ref_fp16;
  uint32_t* qweight_i32;
  half* scale_fp16;
  uint32_t* qzeros_i32;
  half* out_fp16;
  half* out_fp16_ref;
  float* c_cublas;
  float* c_fp32;
  float* c_fp32_ref;
  float* b_fp32_ref;
  float* c_host_cublas;
  float* c_host_gemv;
  float* c_host_ref;
  float* b_host_ref;
  float* b_host;
  curandGenerator_t gen;
  cublasHandle_t cublasHandle;
  cudaEvent_t startGEMV;
  cudaEvent_t stopGEMV;
  cudaEvent_t startcublas;
  cudaEvent_t stopcublas;
  cudaErrCheck(cudaEventCreate(&startGEMV));
  cudaErrCheck(cudaEventCreate(&stopGEMV));
  cudaErrCheck(cudaEventCreate(&startcublas));
  cudaErrCheck(cudaEventCreate(&stopcublas));
  cublasErrCheck(cublasCreate(&cublasHandle));

  cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&b_fp32_ref, MATRIX_K * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&b_ref_fp16, MATRIX_K * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&qweight_i32, sampledata.size.qweight_size));
  cudaErrCheck(cudaMalloc((void**)&scale_fp16, sampledata.size.scale_size));
  cudaErrCheck(cudaMalloc((void**)&qzeros_i32, sampledata.size.qzero_size));
  cudaErrCheck(cudaMalloc((void**)&out_fp16, sampledata.size.out_size));
  cudaErrCheck(cudaMalloc((void**)&out_fp16_ref, sampledata.size.out_size));
  cudaErrCheck(
      cudaMalloc((void**)&c_fp32, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(
      cudaMalloc((void**)&c_fp32_ref, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(
      cudaMalloc((void**)&c_fp16, MATRIX_M * MATRIX_N * sizeof(half)));
  cudaErrCheck(
      cudaMalloc((void**)&c_cublas_fp16, MATRIX_M * MATRIX_N * sizeof(half)));
  cudaErrCheck(
      cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
  c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  c_host_gemv = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  c_host_ref = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  b_host_ref = (float*)malloc(MATRIX_K * MATRIX_N * sizeof(float));
  b_host = (float*)malloc(MATRIX_K * MATRIX_N * sizeof(float));

  cudaErrCheck(cudaMemcpy(b_ref_fp16, sampledata.f16weight, sampledata.size.f16weight_size, cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(qweight_i32, sampledata.qweight, sampledata.size.qweight_size, cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(scale_fp16, sampledata.scale, sampledata.size.scale_size, cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(qzeros_i32, sampledata.qzero, sampledata.size.qzero_size, cudaMemcpyHostToDevice));

  int n_iter = 10;
  float alpha = 1.0f;
  float beta = 0.0f;
  printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  // First: using GEMV
  // dim3 gridDim = {static_cast<unsigned int>((MATRIX_N/2 * ((MATRIX_K+ 8-1) / 8) + kBlockSize - 1) / kBlockSize)};
  dim3 gridDim = {static_cast<unsigned int>((MATRIX_N /2* ((MATRIX_K+31)/32)+ kBlockSize - 1) / kBlockSize)};
  //dim3 gridDim = {static_cast<unsigned int>((MATRIX_N * ((MATRIX_K+31)/32)+ kBlockSize - 1) / kBlockSize)};
  dim3 blockDim = {kBlockSize};

  printf("Running with DequantizeAndUnpackWeight...\n");
  if (sampledata.bits == 4){
  // cuda_quant::DequantizeAndUnpackWeight248<half, 4><<<gridDim, blockDim>>>(b_fp16, qweight_i32, scale_fp16, qzeros_i32, 128, MATRIX_K, MATRIX_N);
  cuda_quant::DequantizeAndUnpackWeight3567<half, 4><<<gridDim, blockDim>>>(b_fp16, qweight_i32, scale_fp16, qzeros_i32, 128, MATRIX_K, MATRIX_N);
  // cuda_quant::DequantizeAndUnpackWeight3567_v2<half, 5><<<gridDim, blockDim>>>(b_fp16, qweight_i32, scale_fp16, qzeros_i32, 128, MATRIX_K, MATRIX_N);
  } else {
    cuda_quant::DequantizeAndUnpackWeight3567_v2<half, 5><<<gridDim, blockDim>>>(b_fp16, qweight_i32, scale_fp16, qzeros_i32, 128, MATRIX_K, MATRIX_N);
  }

  cudaErrCheck(cudaEventRecord(startGEMV));
  for (int i = 0; i < n_iter; i++) {
    // cuda_quant::DequantizeAndUnpackWeight3567<half, 5><<<gridDim, blockDim>>>(b_fp16, qweight_i32, scale_fp16, qzeros_i32, 128, MATRIX_K , MATRIX_N);
    if (sampledata.bits == 4) {
      cuda_quant::DequantizeAndUnpackWeight3567_v2<half, 4><<<gridDim, blockDim>>>(b_fp16, qweight_i32, scale_fp16, qzeros_i32, 128, MATRIX_K, MATRIX_N);
    }
    else{
      cuda_quant::DequantizeAndUnpackWeight3567_v2<half, 5><<<gridDim, blockDim>>>(b_fp16, qweight_i32, scale_fp16, qzeros_i32, 128, MATRIX_K, MATRIX_N);
    }
  // cuda_quant::DequantizeAndUnpackWeight248<half, 4><<<gridDim, blockDim>>>(b_fp16, qweight_i32, scale_fp16, qzeros_i32, 128, MATRIX_K, MATRIX_N);
  }
  cudaErrCheck(cudaEventRecord(stopGEMV));
  cuda_quant::convertFp16ToFp32<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(b_fp32, b_fp16, MATRIX_K * MATRIX_N);
  cuda_quant::convertFp16ToFp32<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(b_fp32_ref, b_ref_fp16, MATRIX_K * MATRIX_N);

  printf("............................\n\n");

  //  Error checking
  printf("\nChecking results...\n");
  cudaErrCheck(cudaMemcpy(b_host, b_fp32, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(b_host_ref, b_fp32_ref, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

  // 0.01% relative tolerance. 1e-5 absolute tolerance.
  int errors = 0;
  for (int i = 0; i < MATRIX_K * MATRIX_N; i++) {
    float v1 = b_host[i];
    // float v2 = c_host_cublas[i];
    float v3 = b_host_ref[i];
    if (errors == 0) {
      errors++;
      printf("%d: %f %f\n\n",i, v1, v3);
    }
    if ( abs(v1 - v3) > 1e-3) {
      errors++;
      if (errors < 10)
        printf("%d: %f %f\n",i, v1, v3);
      }
  }
  if (errors > 1) {
    printf("GEMV does not agree with cuBLAS! %d errors!\n", errors);
  } else {
    printf("Results verified: cublas and GEMV agree.\n\n");
    float gemvTime;
    float cublasTime;
    cudaErrCheck(cudaEventSynchronize(stopGEMV));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&gemvTime, startGEMV, stopGEMV));
    //cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    gemvTime /= n_iter;
    cublasTime /= n_iter;
    printf("gemv took %fms, gflops:%f\n", gemvTime, MATRIX_K / 8 * MATRIX_N * 2 / gemvTime / 1e6);
    printf("cublas took %fms, gflops:%f\n", cublasTime, MATRIX_M * MATRIX_N * MATRIX_K * 2 / cublasTime / 1e6);
  }
  cudaErrCheck(cudaEventDestroy(startGEMV));
  cudaErrCheck(cudaEventDestroy(stopGEMV));
  cudaErrCheck(cudaEventDestroy(startcublas));
  cudaErrCheck(cudaEventDestroy(stopcublas));
  cudaErrCheck(cudaFree(a_fp32));
  cudaErrCheck(cudaFree(b_fp32));
  cudaErrCheck(cudaFree(a_fp16));
  cudaErrCheck(cudaFree(b_fp16));
  cudaErrCheck(cudaFree(c_fp32));
  cudaErrCheck(cudaFree(c_cublas));
  free(c_host_cublas);
  free(c_host_gemv);
  //convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  cudaErrCheck(cudaDeviceReset());
  return 0;
}
#endif