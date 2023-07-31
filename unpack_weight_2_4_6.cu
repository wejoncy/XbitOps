#include "gemv.cuh"

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdint>
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
#define curandErrCheck(stat)                     \
  {                                              \
    curandErrCheck_((stat), __FILE__, __LINE__); \
  }
void curandErrCheck_(curandStatus_t stat, const char* file, int line) {
  if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}
const int MATRIX_M = 1;
const int MATRIX_K = 4096;
const int MATRIX_N = 12288;

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;

#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_HALF2(pointer) (reinterpret_cast<half2*>(&(pointer))[0])

template <typename T, int WBITS>
__global__ void DequantizeAndUnpackWeight(T* out, uint32_t* qweight, T* scale, uint32_t* zeros, int group_size, const int m, const int n) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  const int half_n = n/2;
  if (tid >= m * half_n)
    return;

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

  float2 weight_int2 = FETCH_FLOAT2(qweight[tid*2]);
  uint32_t weight_v1 = *reinterpret_cast<uint32_t*>(&weight_int2.x);
  uint32_t weight_v2 = *reinterpret_cast<uint32_t*>(&weight_int2.y);

  //if (bid == 384 && threadIdx.x == 0) {
  //  printf("%d,%d,%d,%d,%d,%f,%f\n", weight_v1, weight_v2, zero_v,zv1,zv2, __half2float(scale_v.x), __half2float(scale_v.y));
  //}
#pragma unroll
  for (int i = 0; i < 32 / WBITS; i++) {
    uint8_t wv1 = (weight_v1 >> (i * WBITS)) & max_num_in_bits;
    uint8_t wv2 = (weight_v2 >> (i * WBITS)) & max_num_in_bits;
    half2 wv = __halves2half2(__short2half_rn(wv1), __short2half_rn(wv2));
    out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, -scale_zeros);
    //if (bid == 384 && threadIdx.x == 0) {
    //  half2 tv=__hfma2(wv, scale_v, -scale_zeros);
    //  printf("%d,%d,%f,%f\n", wv1, wv2, __half2float(tv.x), __half2float(tv.y));
    //}
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

  int n_iter = 4;
  float alpha = 1.0f;
  float beta = 0.0f;
  printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  // First: using GEMV
  dim3 gridDim = {(MATRIX_N * MATRIX_K / 8 + kBlockSize * 2 - 1) / kBlockSize / 2};
  dim3 blockDim = {kBlockSize};

  printf("Running with DequantizeAndUnpackWeight...\n");
  DequantizeAndUnpackWeight<half, 4><<<gridDim, blockDim>>>(b_fp16, qweight_i32, scale_fp16, qzeros_i32, 128, MATRIX_K / 8, MATRIX_N);
  cudaErrCheck(cudaEventRecord(startGEMV));
  for (int i = 0; i < n_iter; i++) {
    DequantizeAndUnpackWeight<half, 4><<<gridDim, blockDim>>>(b_fp16, qweight_i32, scale_fp16, qzeros_i32, 128, MATRIX_K / 8, MATRIX_N);
  }
  cudaErrCheck(cudaEventRecord(stopGEMV));
  convertFp16ToFp32<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(b_fp32, b_fp16, MATRIX_K * MATRIX_N);
  convertFp16ToFp32<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(b_fp32_ref, b_ref_fp16, MATRIX_K * MATRIX_N);

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
      printf("%d: %f %f\n",i, v1, v3);
    }
    if ( abs(v1 - v3) > 1e-3) {
      errors++;
      if (errors < 10)
        printf("%d:%f %f\n",i, v1, v3);
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
  convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  cudaErrCheck(cudaDeviceReset());
  return 0;
}
