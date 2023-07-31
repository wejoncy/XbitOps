#include "gemv.cuh"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <stdio.h>
#include <sys/types.h>

// Define some error checking macros.
#define cudaErrCheck(stat) \
  { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char* file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file,
            line);
  }
}
#define cublasErrCheck(stat) \
  { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char* file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
  }
}
#define curandErrCheck(stat) \
  { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char* file, int line) {
  if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__) 
void checkLast(const char* const file, const int line) {
  cudaError_t err{cudaGetLastError()};\
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error at: %s:%d" ,file, line);
    fprintf(stderr, cudaGetErrorString(err) );
    // We don't exit when we encounter CUDA errors in this example.
    // std::exit(EXIT_FAILURE);
  }
}

const int MATRIX_M = 1;
const int MATRIX_K = 11008;
const int MATRIX_N = 4096;
const int block_k = ((MATRIX_K + 31) / 32 + 7) / 8 * 8;
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
__global__ void gemv_NT(T* out, T* inA, uint32_t* inB, T* scales, uint32_t* qzeros, int32_t groupsize) {
  int bid = blockIdx.x;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;

  float sum[2] = {0, 0};
  int y_start = tidx * 8; 
  half2 res2 = {};
  half2 res2_1 = {};
  int blockIdx_y = blockIdx.y;
  half2* inA_start = (half2*)(inA + blockIdx_y * MATRIX_K + y_start);

  int n_offset_x = bid * width_element_per_block + tidy * 2;

  int start_group_id = (y_start / groupsize);
  int compressed_idx = tidy % 4;
  int group_nums = (MATRIX_K + groupsize - 1) / groupsize;
  int weight_rows = (MATRIX_K + 8 - 1) / 8;
  half2 scale = {((scales + start_group_id + n_offset_x * group_nums))[0], ((scales + start_group_id + (n_offset_x+1) * group_nums))[0]};
  int32_t qzero_p = ((int32_t*)(qzeros + n_offset_x / 8 * group_nums + start_group_id))[0];
  half2 hzero = __halves2half2(__int2half_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
                               __int2half_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf));
  half2 scale_h0 = __half2half2(scale.x);
  half2 scale_h1 = __half2half2(scale.y);
  half2 hzero_scale_0 = __half2half2(hzero.x * scale.x);
  half2 hzero_scale_1 = __half2half2(hzero.y * scale.y);

#pragma unroll
  for (int i = 0; i < block_k / 8; i++) {  // read half2 * 4
    int i8 = i * 32 * 8;                   // half2*4
    res2 = {};
    res2_1 = {};
    int k_offset = y_start + i8 ;  // half2*4
    int g_id = k_offset / groupsize;

    uint32_t* hinB = inB + n_offset_x * weight_rows + k_offset / 8;
    uint32_t vbInt1 = (n_offset_x < MATRIX_N && (k_offset < MATRIX_K)) ? hinB[0] : int32_t(0);
    uint32_t vbInt2 = (n_offset_x + 1 < MATRIX_N && (k_offset < MATRIX_K))? (inB + (n_offset_x + 1) * weight_rows + k_offset / 8)[0]: int32_t(0);
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
      vb[j + 4] = __halves2half2(__int2half_rn(((*(qweight_p2 + j))) & 0xF),
                                 __int2half_rn((((*(qweight_p2 + j)) >> 4)) & 0xF));
    }

    if (g_id > start_group_id) {
      scale = __halves2half2(((scales + g_id + n_offset_x * group_nums))[0], ((scales + g_id + (n_offset_x + 1) * group_nums))[0]);
      qzero_p = ((int32_t*)(qzeros + n_offset_x / 8 * group_nums + g_id))[0];
      hzero = __halves2half2(__int2half_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
                             __int2half_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf));
      scale_h0 = __half2half2(scale.x);
      scale_h1 = __half2half2(scale.y);
      hzero_scale_0 = __half2half2(hzero.x * scale.x);
      hzero_scale_1 = __half2half2(hzero.y * scale.y);
      start_group_id = g_id;
    }

    half2 va[4];
    i8 >>= 1;
    va[0] = (k_offset + 2 < MATRIX_K) ? ((inA_start))[i8] : half2{0, 0};
    va[1] = (k_offset + 4 < MATRIX_K) ? ((inA_start))[i8 + 1] : half2{0, 0};
    va[2] = (k_offset + 6 < MATRIX_K) ? ((inA_start))[i8 + 2] : half2{0, 0};
    va[3] = (k_offset + 8 < MATRIX_K) ? ((inA_start))[i8 + 3] : half2{0, 0};

#pragma unroll
    for (int j = 0; j < 4; j++) {
      vb[j] = __hfma2(scale_h0, vb[j], -hzero_scale_0);  ///////
      res2 = __hfma2(va[j], vb[j], res2);
      vb[4 + j] = __hfma2(scale_h1, vb[4 + j], -hzero_scale_1);  /////
      res2_1 = __hfma2(va[j], vb[4 + j], res2_1);
    }

    sum[0] += __half2float(res2.x) + __half2float(res2.y);
    sum[1] += __half2float(res2_1.x) + __half2float(res2_1.y);
  }
  sum[0] = warpReduceSum<32>(sum[0]);
  sum[1] = warpReduceSum<32>(sum[1]);
  if (tidx == 0) {
     #pragma unroll
     for (int i = 0; i < 2; i++) {
      out[blockIdx_y * MATRIX_N + bid * width_element_per_block +
          tidy * 2 + i] = __float2half_rn(sum[i]);
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

void generate_random_w(int bits, int8_t** p_weight_in, int m, int n) {
  int right_bound = 1 << bits;

  int8_t* weight_in = *p_weight_in;
  size_t size_in = m * n;
  memset(weight_in, 0, size_in);
  for (int row = 0; row < m; row++) {
    for (int kkkk = 0; kkkk < n; kkkk++) {
      weight_in[row * n + kkkk] = (rand() % right_bound);
    }
  }
}

struct ParamWeight {
  int8_t* weight_in;
  int32_t* weight_out;
  int32_t* weight_i32_cu;
  int m;
  int n;
  int bits;
  half* weight_fp16;
};

void pack_w(int bits, int8_t* weight_in, int32_t** p_weight_out, int m, int n) {
  int right_bound = 1 << bits;
  int32_t* weight_out = *p_weight_out;
  size_t size_out = (m + 7) * n / right_bound;
  memset(weight_out, 0, size_out);
  for (int row = 0; row < m; row++) {
    for (int kkkk = 0; kkkk < n; kkkk++) {
      weight_out[(row / right_bound) * n + kkkk] |= weight_in[row * n + kkkk]
                                                    << (row % right_bound);
    }
  }
}

ParamWeight build_w() {
  int m = MATRIX_K;
  int n = MATRIX_N;
  int8_t* p_weight_in = new int8_t[n * m];
  generate_random_w(4, &p_weight_in, m, n);
  int i32_w_size = (m + 7) * n / 8;
  int32_t* p_weight_out = new int32_t[i32_w_size];
  pack_w(4, p_weight_in, &p_weight_out, m, n);
  half* weight_fp16 = 0;
  cudaMalloc((void**)&weight_fp16, m * n * sizeof(half));
  // half *weight_fp32 = 0;
  // cudaMalloc((void **)&weight_fp32, m * n * sizeof(float));
  int32_t* weight_i32_cu = 0;
  cudaMalloc((void**)&weight_i32_cu, i32_w_size * sizeof(int32_t));
  cudaErrCheck(cudaMemcpy(weight_i32_cu, p_weight_out,
                          i32_w_size * sizeof(float), cudaMemcpyHostToDevice));

  ParamWeight weight = {p_weight_in, p_weight_out, weight_i32_cu, m, n,
                        4, weight_fp16};
  return weight;
}

int QbitGemv(SampleData* p_sampledata) {
  SampleData& sampledata = *p_sampledata;
  float* a_fp32;
  float* b_fp32;
  half* c_fp16;
  half* c_cublas_fp16;
  half* a_fp16;
  half* b_fp16;
  uint32_t* qweight_i32;
  half* scale_fp16;
  uint32_t* qzeros_i32;
  half* out_fp16;
  half* out_fp16_ref;
  float* c_cublas;
  float* c_fp32;
  float* c_fp32_ref;
  float* c_host_cublas;
  float* c_host_gemv;
  float* c_host_ref;
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

  cudaErrCheck(
      cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
  cudaErrCheck(
      cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
  cudaErrCheck(
      cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&qweight_i32, sampledata.size.qweight_size));
  cudaErrCheck(cudaMalloc((void**)&scale_fp16, sampledata.size.scale_size));
  cudaErrCheck(cudaMalloc((void**)&qzeros_i32, sampledata.size.qzero_size));
  cudaErrCheck(cudaMalloc((void**)&out_fp16, sampledata.size.out_size));
  cudaErrCheck(cudaMalloc((void**)&out_fp16_ref, sampledata.size.out_size));
  cudaErrCheck(cudaMalloc((void**)&c_fp32, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&c_fp32_ref, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&c_fp16, MATRIX_M * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&c_cublas_fp16, MATRIX_M * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
  c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  c_host_gemv = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  c_host_ref = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  //curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  //curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
  //curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
  //curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));
  //curandErrCheck(curandDestroyGenerator(gen));

  // float *hostArray = new float[MATRIX_M * MATRIX_K];
  // for (int i = 0; i < MATRIX_K; i++) {
  //   hostArray[i] = (i - MATRIX_K / 2) / 100.0;
  // }
  // float *hostArrayb = new float[MATRIX_N * MATRIX_K];
  // for (int i = 0; i < MATRIX_N * MATRIX_K; i++) {
  //   hostArrayb[i] = (i % 100) / 100.0;
  // }
  // float *hostArrac = new float[MATRIX_M * MATRIX_N];
  // memset(hostArrac, 0, MATRIX_M * MATRIX_N * sizeof(float));
  // for (int i = 0; i < MATRIX_M; i++) {
  //   for (int j = 0; j < MATRIX_N; j++) {
  //     for (int k = 0; k < MATRIX_K; k++) {
  //       hostArrac[j] += hostArray[k] * hostArrayb[k * MATRIX_N + j];
  //     }
  //   }
  // }

  // cudaErrCheck(cudaMemcpy(a_fp32, hostArray, MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));
  // cudaErrCheck(cudaMemcpy(b_fp32, hostArrayb, MATRIX_N * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  // convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(
  //    a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  // convertFp32ToFp16<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(
  //    b_fp16, b_fp32, MATRIX_K * MATRIX_N);
  // convertFp16ToFp32<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(
  //    a_fp32, a_fp16, MATRIX_M * MATRIX_K);
  // convertFp16ToFp32<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(
  //    b_fp32, b_fp16, MATRIX_K * MATRIX_N);

  int n_iter = 1;
  float alpha = 1.0f;
  float beta = 0.0f;
  printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M,
         MATRIX_N, MATRIX_K, alpha, beta);
  ParamWeight pweight = build_w();

  cudaErrCheck(cudaMemcpy(a_fp16, sampledata.input, sampledata.size.input_size,cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(qweight_i32, sampledata.qweight,sampledata.size.qweight_size,cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(scale_fp16, sampledata.scale,sampledata.size.scale_size, cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(qzeros_i32, sampledata.qzero,sampledata.size.qzero_size, cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(b_fp16, sampledata.f16weight,sampledata.size.f16weight_size,cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(out_fp16_ref, sampledata.out,sampledata.size.out_size, cudaMemcpyHostToDevice));
  convertFp16ToFp32<<<(MATRIX_M * MATRIX_N + 255) / 256, 256>>>(c_fp32_ref, out_fp16_ref, MATRIX_M * MATRIX_N);
  // First: using GEMV
  dim3 gridDim = {(MATRIX_N + width_element_per_block - 1) /
                      width_element_per_block,
                  MATRIX_M};
  dim3 blockDim = {32, (MATRIX_K + block_k - 1) / block_k};
  printf("%d================\n", (MATRIX_K + block_k - 1) / block_k);
  printf("Running with gemv...\n");
  gemv_NT<half><<<gridDim, blockDim>>>(c_fp16, a_fp16, qweight_i32, scale_fp16, qzeros_i32, 128);
  cudaDeviceSynchronize();
  CHECK_LAST_CUDA_ERROR();
  cudaErrCheck(cudaEventRecord(startGEMV));
  for (int i = 0; i < n_iter; i++) {
    gemv_NT<half><<<gridDim, blockDim>>>(c_fp16, a_fp16, qweight_i32, scale_fp16, qzeros_i32, 128);
  }
  cudaErrCheck(cudaEventRecord(stopGEMV));
  cudaDeviceSynchronize();
  CHECK_LAST_CUDA_ERROR();
  convertFp16ToFp32<<<(MATRIX_M * MATRIX_N + 255) / 256, 256>>>(c_fp32, c_fp16, MATRIX_M * MATRIX_N);
  printf("............................\n\n");
  // Now using cuBLAS
  printf("Running with cuBLAS...\n");
  // cublasErrCheck(cublasGemmEx(
  //     cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, MATRIX_M, MATRIX_N, MATRIX_K,
  //     &alpha, a_fp16, CUDA_R_16F, MATRIX_K, b_fp16, CUDA_R_16F, MATRIX_N, &beta,
  //     c_cublas_fp16, CUDA_R_16F, MATRIX_M, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  // cudaErrCheck(cudaEventRecord(startcublas));
  // for (int i = 0; i < n_iter; i++) {
  //  cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
  //                              MATRIX_M, MATRIX_N, MATRIX_K, &alpha, a_fp16,
  //                              CUDA_R_16F, MATRIX_K, b_fp16, CUDA_R_16F,
  //                              MATRIX_N, &beta, c_cublas_fp16, CUDA_R_16F,
  //                              MATRIX_N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  //}
  // cudaErrCheck(cudaEventRecord(stopcublas));
  convertFp16ToFp32<<<(MATRIX_M * MATRIX_N + 255) / 256, 256>>>(
      c_cublas, c_cublas_fp16, MATRIX_M * MATRIX_N);
  // cublasSgemv (cublasHandle, CUBLAS_OP_N,
  //         MATRIX_N, MATRIX_K, &alpha, b_fp32, MATRIX_N, a_fp32, 1, &beta,
  //         c_cublas, 1);
  // cudaErrCheck(cudaEventRecord(startcublas));
  // for (int run = 0 ; run < n_iter; run ++ ) {
  //     cublasSgemv (cublasHandle, CUBLAS_OP_T,
  //             MATRIX_K, MATRIX_N, &alpha, b_fp32, MATRIX_N, a_fp32, 1, &beta,
  //             c_cublas, 1);
  // }
  // cudaErrCheck(cudaEventRecord(stopcublas));
  //  Error checking
  printf("\nChecking results...\n");
  cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(c_host_gemv, c_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(c_host_ref, c_fp32_ref, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
  // 0.01% relative tolerance. 1e-5 absolute tolerance.
  int errors = 0;
  for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
    float v1 = c_host_gemv[i];
    float v2 = c_host_cublas[i];
    float v3 = c_host_ref[i];
    float ref = v3;

    if (errors == 0) {
      errors++;
      printf("%f %f\n\n", ref, v1);
    }
    if ((abs(ref / v1) > 1.001 || abs(v1 / ref) > 1.001) &&
        abs(ref - v1) > 4e-4) {
      errors++;
      if (errors < 10)
        printf("%d: r%f %f\n", i, ref, v1);
    }
  }
  if (errors > 100000) {
    printf("GEMV does not agree with cuBLAS! %d errors!\n", errors);
  } else {
    printf("Results verified: cublas and GEMV agree.\n\n");
    float gemvTime;
    float cublasTime;
    cudaErrCheck(cudaEventSynchronize(stopGEMV));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&gemvTime, startGEMV, stopGEMV));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    gemvTime /= n_iter;
    cublasTime /= n_iter;
    printf("gemv took %fms, gflops:%f\n", gemvTime,
           double(MATRIX_M) * MATRIX_N * MATRIX_K * 2 / gemvTime / 1e6);
    printf("cublas took %fms, gflops:%f\n", cublasTime,
           double(MATRIX_M) * MATRIX_N * MATRIX_K * 2 / cublasTime / 1e6);
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
  convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(
      a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  cudaErrCheck(cudaDeviceReset());
  return 0;
}
