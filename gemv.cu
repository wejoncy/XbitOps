#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
// Define some error checking macros.
#define cudaErrCheck(stat)                         \
    {                                              \
        cudaErrCheck_((stat), __FILE__, __LINE__); \
    }
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}
#define cublasErrCheck(stat)                         \
    {                                                \
        cublasErrCheck_((stat), __FILE__, __LINE__); \
    }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}
#define curandErrCheck(stat)                         \
    {                                                \
        curandErrCheck_((stat), __FILE__, __LINE__); \
    }
void curandErrCheck_(curandStatus_t stat, const char *file, int line)
{
    if (stat != CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}
const int block_k = 128;
const int MATRIX_M=1;
const int MATRIX_K=4096;
const int MATRIX_N=4096;
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}
// if N>= 128
__global__ void Sgemv_v1(
    float* __restrict__ A,
    float* __restrict__ x,
    float* __restrict__ y,
    const int M,
    const int N) {
    // Block index
    int bx = blockIdx.x;
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    const int warp_size = 32;
    int laneId = tx % warp_size;
    int current_row = blockDim.y * bx + ty;
    if (current_row < M) {
        float res = 0;
        int kIteration = (N / warp_size) / 4;
        if (kIteration == 0) kIteration = 1;
        A = &A[current_row * N];
#pragma unroll
        for (int i = 0; i < kIteration; i++) {
            int current_col_vec = (i * warp_size + laneId);
            float4 current_val = reinterpret_cast<float4*>(A)[current_col_vec];
            float4 current_x = reinterpret_cast<float4*>(x)[current_col_vec];
            res += current_val.x * current_x.x;
            res += current_val.y * current_x.y;
            res += current_val.z * current_x.z;
            res += current_val.w * current_x.w;
        }
        res = warpReduceSum<warp_size>(res);
        if (laneId == 0) y[current_row] = res;
    }
}
template<typename T>
__global__ void gemv(T *out, T *inA, T *inB){
    int bid =  blockIdx.x ;
    int btid= threadIdx.y*blockDim.x + threadIdx.x;
    //int tid = blockDim.x *blockDim.y*bid + btid;
    __shared__ T vecA[MATRIX_K] ;
    __shared__ T bsum[32][MATRIX_K / block_k+1];
//    int element_per_th= block_k/32;
//#pragma unroll
//    for(int i=0;i<element_per_th;i++){
//        vecA[btid*element_per_th+i] = inA[btid*element_per_th+i];
//    }
//    __syncthreads();
    T sum=0;
    int y_start = threadIdx.y*block_k;
#pragma unroll
    for(int i=0;i<block_k;i++){
        sum += inA[y_start + i] *inB[(threadIdx.y * block_k + i) * MATRIX_N + bid * 32 + threadIdx.x];
        /*if (bid == 0 && threadIdx.y==1 && threadIdx.x==0) {
            printf("%f,%f===", vecA[y_start + i] , inB[(threadIdx.y * block_k + i) * 4096 + threadIdx.x + bid * 32]);
        }*/
    }
    bsum[threadIdx.x][threadIdx.y]=sum;
    __syncthreads();
    sum = 0;
    //if (threadIdx.y == 0) {
    //    for (int i = 0; i < 32; i++) {
    //        sum += bsum[threadIdx.x][i];
    //    }
    //    out[bid* blockDim.x + threadIdx.x] = sum;
    //}
    #pragma unroll
    for(int i=0;i<1;i++){
        sum=bsum[threadIdx.y][threadIdx.x];
        __syncthreads();
        sum = warpReduceSum<32>(sum);
        if(threadIdx.x==0) {
            out[bid*32 + threadIdx.y] = sum;
        }
    }
}
__global__ void convertFp32ToFp16(half *out, float *in, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = in[idx];
    }
}
int main(int argc, char *argv[])
{
    float *a_fp32;
    float *b_fp32;
    half *c_fp16;
    half *a_fp16;
    half *b_fp16;
    float *c_cublas;
    float *c_fp32;
    float *c_host_cublas;
    float *c_host_gemv;
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
    cudaErrCheck(cudaMalloc((void **)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
    cudaErrCheck(cudaMalloc((void **)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));
    cudaErrCheck(cudaMalloc((void **)&c_fp32, MATRIX_M * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&c_fp16, MATRIX_M * MATRIX_N * sizeof(half)));
    cudaErrCheck(cudaMalloc((void **)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
    c_host_cublas = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    c_host_gemv = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
    curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));
    // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
    //convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
    //convertFp32ToFp16<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(b_fp16, b_fp32, MATRIX_K * MATRIX_N);
    curandErrCheck(curandDestroyGenerator(gen));
    float* hostArray = new float[MATRIX_M * MATRIX_K];
    for (int i = 0; i < MATRIX_K; i++) {
        hostArray[i] = (i) / 100.0;
    }
    float* hostArrayb = new float[MATRIX_N * MATRIX_K];
    for (int i = 0; i < MATRIX_N * MATRIX_K; i++) {
        hostArrayb[i] = (i%4096 ) / 100.0;
    }
    float* hostArrac = new float[MATRIX_M * MATRIX_N];
    memset(hostArrac, 0, MATRIX_M * MATRIX_N * sizeof(float));
    for (int i = 0; i < MATRIX_M; i++) {
        for (int j = 0; j < MATRIX_N; j++) {
            for (int k = 0; k < MATRIX_K; k++) {
                hostArrac[j] += hostArray[k] * hostArrayb[k * MATRIX_N +j];
            }
        }
    }
    /*cudaErrCheck(cudaMemcpy(a_fp32, hostArray, MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(b_fp32, hostArrayb, MATRIX_N * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));*/
    int n_iter = 100;
    float alpha = 1.0f;
    float beta = 0.0f;
    printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
    // First: using GEMV
    dim3 gridDim={MATRIX_N/32};
    dim3 blockDim={32,MATRIX_K/block_k};
    printf("Running with gemv...\n");
    gemv<float><<<gridDim,blockDim>>>(c_fp32, a_fp32, b_fp32);
    cudaErrCheck(cudaEventRecord(startGEMV));
    for (int i = 0; i < n_iter; i++) {
        gemv<float> << <gridDim, blockDim >> > (c_fp32, a_fp32, b_fp32);
        //Sgemv_v1 << <gridDim, blockDim >> > ( b_fp32, a_fp32, c_fp32, MATRIX_K, MATRIX_N);
    }
    cudaErrCheck(cudaEventRecord(stopGEMV));
    printf("............................\n\n");
    // Now using cuBLAS
    printf("Running with cuBLAS...\n");
    //cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
    //                            MATRIX_M,   MATRIX_N, MATRIX_K,
    //                            &alpha,
    //                            a_fp32, CUDA_R_32F, MATRIX_K,
    //                            b_fp32, CUDA_R_32F, MATRIX_N,
    //                            &beta,
    //                            c_cublas, CUDA_R_32F, MATRIX_M,
    //                            CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    //cudaErrCheck(cudaEventRecord(startcublas));
    //for (int i = 0; i < n_iter; i++) {
    //    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
    //                                MATRIX_M, MATRIX_N, MATRIX_K,
    //                                &alpha,
    //                                a_fp32, CUDA_R_32F, MATRIX_K,
    //                                b_fp32, CUDA_R_32F, MATRIX_N,
    //                                &beta,
    //                                c_cublas, CUDA_R_32F, MATRIX_M,
    //                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    //}
    //cudaErrCheck(cudaEventRecord(stopcublas));
    //cublasSgemv (cublasHandle, CUBLAS_OP_T, 
    //        MATRIX_K, MATRIX_N, &alpha, b_fp32, MATRIX_N, a_fp32, 1, &beta, c_cublas, 1);
    //cudaErrCheck(cudaEventRecord(startcublas));
    //for (int run = 0 ; run < n_iter; run ++ ) {
    //    cublasSgemv (cublasHandle, CUBLAS_OP_N, 
    //            MATRIX_N, MATRIX_K, &alpha, b_fp32, MATRIX_N, a_fp32, 1, &beta, c_cublas, 1);
    //}

    //cudaErrCheck(cudaEventRecord(stopcublas));
    // Error checking
    printf("\nChecking results...\n");
    cudaErrCheck(cudaMemcpy(c_host_gemv, c_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    //cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    // 0.01% relative tolerance. 1e-5 absolute tolerance.
    int errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++)
    {
        float v1 = c_host_gemv[i];
        float v2 = c_host_cublas[i];
        //float v3 = hostArrac[i];
        if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-1)
        {
            errors++;
            if (errors < 10)
                printf("%f %f\n", v1, v2);
        }
    }
    if (errors > 0)
    {
        printf("GEMV does not agree with cuBLAS! %d errors!\n", errors);
    }
    else
    {
        printf("Results verified: cublas and GEMV agree.\n\n");
        float gemvTime;
        float cublasTime;
        cudaErrCheck(cudaEventSynchronize(stopGEMV));
        cudaErrCheck(cudaEventSynchronize(stopcublas));
        cudaErrCheck(cudaEventElapsedTime(&gemvTime, startGEMV, stopGEMV));
        cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
        gemvTime/=n_iter;
        cublasTime/=n_iter;
        printf("wmma took %fms, gflops:%f\n", gemvTime, MATRIX_M*MATRIX_N*MATRIX_K*2/gemvTime/1e6);
        printf("cublas took %fms, gflops:%f\n", cublasTime, MATRIX_M*MATRIX_N*MATRIX_K*2/cublasTime/1e6);
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
    cudaErrCheck(cudaDeviceReset());
    return 0;
}
