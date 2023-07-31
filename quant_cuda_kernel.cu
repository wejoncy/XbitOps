#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 700
// adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh
__device__ __forceinline__ void atomicAdd(__half* address, c10::Half val) {
    unsigned int *address_as_ui = reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) - (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short hsum = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
        hsum += val;
        old = reinterpret_cast<size_t>(address) & 2
                 ? (old & 0xffff) | (hsum << 16)
                 : (old & 0xffff0000) | hsum;
        old = atomicCAS(address_as_ui, assumed, old);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
}
#endif
#endif

__global__ void VecQuant4MatMulKernel(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  half* __restrict__ scales,
    const    int* __restrict__ zeros,
    int batch,
    int vec_height, 	
    int height,
    int width,
    int zero_width,
    int groupsize
);

// https://github.com/iwalton3/GPTQ-for-LLaMa/commit/209d16b0187f149bf13318360925cc4f679cb2ea
template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_G(
    const     half2* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  half* __restrict__ scales,
    const       int* __restrict__ zeros,
    const       int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width
);

const int BLOCKWIDTH = 256;
const int BLOCKHEIGHT4 = 32;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

void vecquant4matmul_cuda(half* vec, int32_t* mat,
                          float* mul, half* scales,
                          int32_t* zeros, int groupsize, int vec_height,
                          int batch,int height, int width, int zero_width
) {  
  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernel<<<blocks, threads>>>(
    (half2*) vec,
    mat,
    mul,
    scales,
    zeros,
    batch, vec_height, height, width, zero_width, groupsize
  );
}

__global__ void convertFp32ToFp161(half *out, float *in, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
        out[idx] = __float2half(in[idx]);
  }
}

void vecquant4matmul(short *vec, int32_t *mat, short *mul, short *scales,
                     int32_t *zeros, int groupsize, int m, int k, int n) {
  float *mul_fp32 = new float[m*n];
  int zero_width = (n + 7) / 8;
  half *vec_cu;
  int32_t *mat_cu;
  half *mul_cu;
  float *mul_cu_fp32;
  half *scales_cu;
  int32_t *zeros_cu;
  cudaEvent_t startGEMV;
  cudaEvent_t stopGEMV;
  cudaEvent_t startcublas;
  cudaEvent_t stopcublas;
  (cudaEventCreate(&startGEMV));
  (cudaEventCreate(&stopGEMV));

  (cudaMalloc((void **)&vec_cu, m*k*sizeof(half)));
  (cudaMalloc((void **)&mat_cu, n * k * sizeof(int32_t)));
  (cudaMalloc((void **)&mul_cu, m * n * sizeof(float)));
  (cudaMalloc((void **)&mul_cu_fp32, m * n * sizeof(float)));
  (cudaMalloc((void **)&scales_cu, n * k * sizeof(float)));
  (cudaMalloc((void **)&zeros_cu, n * k * sizeof(int32_t)));

  (cudaMemcpy(vec_cu, vec, m * k * sizeof(half),cudaMemcpyHostToDevice));
  (cudaMemcpy(mat_cu, mat, n * k/8 * sizeof(int32_t), cudaMemcpyHostToDevice));
  (cudaMemcpy(scales_cu, scales, n * k / 8 * sizeof(half),
              cudaMemcpyHostToDevice));
  (cudaMemcpy(zeros_cu, zeros, n * k / groupsize / 8 * sizeof(int32_t),
              cudaMemcpyHostToDevice));

  (cudaEventRecord(startGEMV));
  vecquant4matmul_cuda((half *)vec_cu, mat_cu, mul_cu_fp32, scales_cu, zeros_cu,
                       groupsize, k / 2, m, k, n, zero_width);
  int n_iter = 100;
  for (int i = 0; i < n_iter; i++) {
        vecquant4matmul_cuda((half *)vec_cu, mat_cu, mul_cu_fp32, scales_cu,
                             zeros_cu, groupsize, k / 2, m, k, n, zero_width);
  }
  (cudaEventRecord(stopGEMV));

  //convertFp32ToFp161<<<(m * k + 255) / 256, 256>>>(mul_cu, mul_cu_fp32, m * k);
  //(cudaMemcpy(mul, mul_cu, n * m* sizeof(half),cudaMemcpyDeviceToHost));
  (cudaMemcpy(mul_fp32, mul_cu_fp32, n * m * sizeof(float),
              cudaMemcpyDeviceToHost));
  float gemvTime;

  (cudaEventElapsedTime(&gemvTime, startGEMV, stopGEMV));
  gemvTime /= n_iter;
  printf("github gemv took %fms, gflops:%f\ngithub==", gemvTime,
         double(m) * n * k * 2 / gemvTime / 1e6);
  for(int i=0;i<10;i++){
        printf("%f,", mul_fp32[i]);
  }
}

__global__ void VecQuant4MatMulKernel(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  half* __restrict__ scales,
    const  	 int* __restrict__ zeros,
	int batch,
	int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[256][8];
  int val = threadIdx.x / 8;
  int off = threadIdx.x % 8;
  for (; val < 256; val += BLOCKWIDTH / 8) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0xF), __int2half_rn(val >> 4)
    );
  }

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8; 
  int z_mod = (w % 8) * 4;

  float res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
	float scale_f = __half2float(scales[g * width + w]);
    half2 scale = __float2half2_rn(scale_f);
    half2 zero = __float2half2_rn(-(scale_f * (((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1)));
	
    res2 = {};
    tmp = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xff][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xff][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xff][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xff][off], scale, zero), blockvec[k + 3], res2);
	i += width;
    k += 4;
    res += __half2float(res2.x) + __half2float(res2.y);
  }

  atomicAdd(&mul[b * width + w], res);
}
#if 0
void vecquant4matmul_g_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_SWITCH(vec.type(), "vecquant4matmul_g_cuda",
    AT_DISPATCH_CASE(at::ScalarType::Half, ([&] {
      VecQuant4MatMulKernel_G<<<blocks, threads>>>(
        (half2*) vec.data_ptr<scalar_t>(),
        mat.data_ptr<int>(),
        mul.data_ptr<scalar_t>(),
        scales.data_ptr<scalar_t>(),
        zeros.data_ptr<int>(),
        g_idx.data_ptr<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  ));
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_G(
    const     half2* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	    int* __restrict__ zeros,
    const       int* __restrict__ g_idx,
	int batch,
	int vec_height,
    int height,
    int width,
    int zero_width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[256][8];
  int val = threadIdx.x / 8;
  int off = threadIdx.x % 8;
  for (; val < 256; val += BLOCKWIDTH / 8) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0xF), __int2half_rn(val >> 4)
    );
  }

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  scalar_t res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    res2 = {};
    tmp = as_unsigned(mat[i]);

    int tmp_k = 0;
    half2 scales_tmp[4];
    half2 zeros_tmp[4];
    while (tmp_k < 4) {
      int g = as_int(g_idx[g_h + (k + tmp_k) * 2]);
      int g2 = as_int(g_idx[g_h + (k + tmp_k) * 2 + 1]);
      scalar_t scale_f = scales[g * width + w];
      scalar_t scale_f2 = scales[g2 * width + w];
      half2 scale = __halves2half2(scale_f, scale_f2);
      half2 zero = __halves2half2(
        __hmul(-scale_f, __int2half_rn(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1)),
        __hmul(-scale_f2, __int2half_rn(((as_unsigned(zeros[g2 * zero_width + z_w]) >> z_mod) & 0xF) + 1))
      );
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
      tmp_k += 1;
    }

    res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xff][off], scales_tmp[0], zeros_tmp[0]), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xff][off], scales_tmp[1], zeros_tmp[1]), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xff][off], scales_tmp[2], zeros_tmp[2]), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xff][off], scales_tmp[3], zeros_tmp[3]), blockvec[k + 3], res2);
	i += width;
    k += 4;
    res = __hadd(res, __hadd(res2.x, res2.y));;
  }

  __half* mul2 = (__half*)mul;
  atomicAdd(&mul2[b * width + w], res);
}
#endif