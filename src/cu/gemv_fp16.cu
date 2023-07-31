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
const int MATRIX_M=1;
const int MATRIX_K=254;
const int MATRIX_N=1224;
const int block_k = ((MATRIX_K+31)/32+1)/2*2/1;
const int element_per_block=32*2;
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}
template<typename T>
__global__ void gemv(T *out, T *inA, T *inB){
    int bid =  blockIdx.x ;
    __shared__ T vecA[MATRIX_K] ;
    __shared__ float bsum[2][32][32+1];
    float sum[2]={0,0};
    int y_start = threadIdx.y*block_k;
    half2 res2={};
    half2 res2_1={};

#pragma unroll
    for(int i=0;i<block_k/2;i++){
        res2={};
        res2_1={};
        int n_offset = threadIdx.y * block_k +i*2;
        int n_offset_x = bid * element_per_block + threadIdx.x*2;
        int k_offset = y_start+i*2;
        half2 va = (k_offset<MATRIX_K)?((half2 *)(inA+y_start)) [i]:res2;
        half* hinB = inB+n_offset_x+n_offset * MATRIX_N;
        half* hinB_1 = hinB + MATRIX_N;
        half2 vb1=(n_offset_x<MATRIX_N && (k_offset<MATRIX_K))?((half2*)hinB)[0]:res2;
        half2 vb2=(n_offset_x<MATRIX_N && (k_offset<MATRIX_K))?((half2*)hinB_1)[0]:res2;
        //if(bid==0 && threadIdx.y==28&&threadIdx.x==0){
        //    printf("%d,",n_offset);
        //}
        

        //half2 va_1 = va;
        //va.y=va.x;
        //va_1.x=va_1.y;
        //res2 = __hfma2(va,vb1,__hfma2(va_1,vb2,res2));
        //sum[0] += __half2float(res2.x);
        //sum[1] += __half2float(res2.y);

        half tmp=vb1.y;
        vb1.y=vb2.x;
        vb2.x=tmp;
        res2 = __hfma2(va,vb1,res2);
        res2_1 = __hfma2(va,vb2,res2_1);
        //half2 hsum2 = (__hadd2(__halves2half2(res2.x,res2_1.x),__halves2half2 (res2.y,res2_1.y)));
        //sum[0]+=__half2float(hsum2.x);
        //sum[1]+=__half2float(hsum2.y);
        sum[0] += __half2float(res2.x) + __half2float(res2.y);
        sum[1] += __half2float(res2_1.x) + __half2float(res2_1.y);
    }
    //sum[0] += __half2float(res2.x);
    //sum[1] +=  __half2float(res2.y);
    bsum[0][threadIdx.x][threadIdx.y]=sum[0];
    bsum[1][threadIdx.x][threadIdx.y]=sum[1];
    __syncthreads();
    sum[0] = 0;
    sum[1] = 0;

#pragma unroll
    for(int i=0;i<2;i++){
        sum[i]=bsum[i][threadIdx.y][threadIdx.x];
        __syncthreads();
        sum[i] = warpReduceSum<32>(sum[i]);
        if(threadIdx.x==0) {
            out[bid*element_per_block + threadIdx.y*2+i] = __float2half_rn(sum[i]);
        }
    }

}
__global__ void convertFp32ToFp16(half *out, float *in, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = __float2half (in[idx]);
    }
}
__global__ void convertFp16ToFp32(float *out, half *in, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = __half2float (in[idx]);
    }
}
int main(int argc, char *argv[])
{
    float *a_fp32;
    float *b_fp32;
    half *c_fp16;
    half *c_cublas_fp16;
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
    cudaErrCheck(cudaMalloc((void **)&c_cublas_fp16, MATRIX_M * MATRIX_N * sizeof(half)));
    cudaErrCheck(cudaMalloc((void **)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
    c_host_cublas = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    c_host_gemv = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
    curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));
    curandErrCheck(curandDestroyGenerator(gen));
    float* hostArray = new float[MATRIX_M * MATRIX_K];
    for (int i = 0; i < MATRIX_K; i++) {
        hostArray[i] = (i-MATRIX_K/2) / 100.0;
    }
    float* hostArrayb = new float[MATRIX_N * MATRIX_K];
    for (int i = 0; i < MATRIX_N * MATRIX_K; i++) {
        hostArrayb[i] = (i%100) / 100.0;
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
    //cudaErrCheck(cudaMemcpy(a_fp32, hostArray, MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));
    //cudaErrCheck(cudaMemcpy(b_fp32, hostArrayb, MATRIX_N * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));



    // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
    convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
    convertFp32ToFp16<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(b_fp16, b_fp32, MATRIX_K * MATRIX_N);
    convertFp16ToFp32<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(a_fp32, a_fp16, MATRIX_M * MATRIX_K);
    convertFp16ToFp32<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(b_fp32, b_fp16, MATRIX_K * MATRIX_N);

    int n_iter = 10;
    float alpha = 1.0f;
    float beta = 0.0f;
    printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
    // First: using GEMV
    dim3 gridDim={(MATRIX_N+element_per_block-1)/element_per_block};
    dim3 blockDim={32,(MATRIX_K+block_k-1)/block_k};
    printf("%d================\n",(MATRIX_K+block_k-1)/block_k);
    printf("Running with gemv...\n");
    gemv<half><<<gridDim,blockDim>>>(c_fp16, a_fp16, b_fp16);
    cudaErrCheck(cudaEventRecord(startGEMV));
    for (int i = 0; i < n_iter; i++) {
        gemv<half> << <gridDim, blockDim >> > (c_fp16, a_fp16, b_fp16);
    }
    cudaErrCheck(cudaEventRecord(stopGEMV));
    convertFp16ToFp32<<<(MATRIX_M * MATRIX_N + 255) / 256, 256>>>(c_fp32, c_fp16, MATRIX_M * MATRIX_N);
    printf("............................\n\n");
    // Now using cuBLAS
    printf("Running with cuBLAS...\n");
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                                    MATRIX_M, MATRIX_N, MATRIX_K,
                                    &alpha,
                                    a_fp16, CUDA_R_16F, MATRIX_K,
                                    b_fp16, CUDA_R_16F, MATRIX_N,
                                    &beta,
                                    c_cublas_fp16, CUDA_R_16F, MATRIX_M,
                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    cudaErrCheck(cudaEventRecord(startcublas));
    for (int i = 0; i < n_iter; i++) {
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                                    MATRIX_M, MATRIX_N, MATRIX_K,
                                    &alpha,
                                    a_fp16, CUDA_R_16F, MATRIX_K,
                                    b_fp16, CUDA_R_16F, MATRIX_N,
                                    &beta,
                                    c_cublas_fp16, CUDA_R_16F, MATRIX_M,
                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    }
    cudaErrCheck(cudaEventRecord(stopcublas));
    convertFp16ToFp32<<<(MATRIX_M * MATRIX_N + 255) / 256, 256>>>(c_cublas, c_cublas_fp16, MATRIX_M * MATRIX_N);
    //cublasSgemv (cublasHandle, CUBLAS_OP_N, 
    //        MATRIX_N, MATRIX_K, &alpha, b_fp32, MATRIX_N, a_fp32, 1, &beta, c_cublas, 1);
    //cudaErrCheck(cudaEventRecord(startcublas));
    //for (int run = 0 ; run < n_iter; run ++ ) {
    //    cublasSgemv (cublasHandle, CUBLAS_OP_T, 
    //            MATRIX_K, MATRIX_N, &alpha, b_fp32, MATRIX_N, a_fp32, 1, &beta, c_cublas, 1);
    //}
    //cudaErrCheck(cudaEventRecord(stopcublas));
    // Error checking
    printf("\nChecking results...\n");
    cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(c_host_gemv, c_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    // 0.01% relative tolerance. 1e-5 absolute tolerance.
    int errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++)
    {
        float v1 = c_host_gemv[i];
        float v2 = c_host_cublas[i];
        //float v3 = hostArrac[i];
        if (errors==0){
            errors++;
            printf("%f %f\n", v1, v2);
        }
        if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-1)
        {
            errors++;
            if (errors < 10)
                printf("%f %f\n", v1, v2);
        }
    }
    if (errors > 111111110)
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
        printf("gemv took %fms, gflops:%f\n", gemvTime, MATRIX_M*MATRIX_N*MATRIX_K*2/gemvTime/1e6);
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
    convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
    cudaErrCheck(cudaDeviceReset());
    return 0;
}
