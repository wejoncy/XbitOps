#include <stdint.h>
#include <stdio.h>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <sys/time.h>

extern int MATRIX_M;
extern int MATRIX_K;
extern int MATRIX_N;
namespace cpu {
const int block_k = ((MATRIX_K + 31) / 32 + 7) / 8 * 8;
const int width_element_per_block = 32 * 2;

typedef unsigned short ushort;
typedef unsigned int uint;

struct uchar2 {
  uint8_t x, y;
};
struct uint2 {
  uint32_t x, y;
};

typedef ushort half;
uint as_uint(const float x) { return *(uint*)&x; }
float as_float(const uint x) { return *(float*)&x; }

float half_to_float(const ushort x) {                 // IEEE-754 16-bit floating-point format (without
                                      // infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5,
                                      // +-5.9604645E-8, 3.311 digits
  const uint e = (x & 0x7C00) >> 10;  // exponent
  const uint m = (x & 0x03FF) << 13;  // mantissa
  const uint v =
      as_uint((float)m) >>
      23;  // evil log2 bit hack to count leading zeros in denormalized format
  return as_float(
      (x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
      ((e == 0) & (m != 0)) *
          ((v - 37) << 23 | ((m << (150 - v)) &
                             0x007FE000)));  // sign : normalized : denormalized
}

ushort float_to_half(const float x) {             // IEEE-754 16-bit floating-point format (without
                                           // infinity): 1-5-10, exp-15, +-131008.0,
                                           // +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
  const uint b = as_uint(x) + 0x00001000;  // round-to-nearest-even: add last bit
                                           // after truncated mantissa
  const uint e = (b & 0x7F800000) >> 23;   // exponent
  const uint m =
      b &
      0x007FFFFF;  // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000
                   // = decimal indicator flag - initial rounding
  return (b & 0x80000000) >> 16 |
         (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
         ((e < 113) & (e > 101)) *
             ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
         (e > 143) * 0x7FFF;  // sign : normalized : denormalized : saturate
}

struct float2 {
  float x, y;
};

struct half2 {
  ushort x, y;
};

float2 __half22float2(half2 h) {
  return {half_to_float(h.x), half_to_float(h.y)};
}
float __half2float(const half h) { return half_to_float(h); }
half __float2half(float f) { return float_to_half(f); }
half __float2half_rn(float f) { return float_to_half(f); }

float __int2float_rn(int x) { return (float)x; }
float __short2half_rn(int x) { return float_to_half((float) x); }
float __ushort2half_rn(int x) { return float_to_half((float)x); }
float2 __halves2float2(float x, float y) { return {(x), (y)}; }
float fmaf(float a, float b, float c) { return a * b + c; }

void __syncthreads() {}

template <int n = 32>
float warpReduceSum(float sum) { return sum; }

template <typename T>
void cpu_gemv(T* out, T* Tweight_unpack, T* inA, uint32_t* inB, T* scales,
              uint32_t* qzeros, int32_t groupsize) {
  for (int bid = 0; bid < 11008 / 64; bid++) {
    float bsum[2][32][32 + 1];

    for (int tidy = 0; tidy < 32; tidy++) {
      for (int tidx = 0; tidx < 32; tidx++) {
        float sum[2] = {0, 0};
        int y_start = tidy * block_k;
        float2 res2 = {};
        float2 res2_1 = {};

        half2* inA_start = (half2*)(inA + 0 * MATRIX_K + y_start);

        int n_offset_x = bid * width_element_per_block + tidx * 2;

        int start_group_id = (y_start / groupsize);
        int compressed_idx = tidx % 4;
        float2 scale = __half22float2(
            ((half2*)(scales + start_group_id * MATRIX_N + n_offset_x))[0]);
        int32_t qzero_p =
            ((int32_t*)(qzeros + n_offset_x / 8 +
                        start_group_id * ((MATRIX_N + 7) / 8)))[0];
        float2 hzero = {
            __int2float_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
            __int2float_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf)};
        float scale_h0 = (scale.x);
        float scale_h1 = (scale.y);
        float hzero_scale_0 = (hzero.x * scale.x);
        float hzero_scale_1 = (hzero.y * scale.y);

#pragma unroll
        for (int i = 0; i < block_k / 2; i += 4) {  // read half2 * 4
          res2 = {};
          res2_1 = {};
          int k_offset = y_start + i * 2;

          // if (k_offset / groupsize > start_group_id)
          {
            int g_id = k_offset / groupsize;  /////////////////////////////////////
            scale = __half22float2(
                ((half2*)(scales + g_id * MATRIX_N + n_offset_x))[0]);
            qzero_p = ((int32_t*)(qzeros + n_offset_x / 8 +
                                  g_id * ((MATRIX_N + 7) / 8)))[0];
            hzero = {__int2float_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
                     __int2float_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) &
                                    0xf)};
            scale_h0 = (scale.x);
            scale_h1 = (scale.y);
            hzero_scale_0 = (hzero.x * scale.x);
            hzero_scale_1 = (hzero.y * scale.y);
          }

          uint32_t* hinB = inB + n_offset_x + k_offset / 8 * MATRIX_N;
          uint32_t vbInt1 = (n_offset_x < MATRIX_N && (k_offset < MATRIX_K))
                                ? hinB[0]
                                : int32_t(0);
          uint32_t vbInt2 = (n_offset_x + 1 < MATRIX_N && (k_offset < MATRIX_K))
                                ? (hinB)[1]
                                : int32_t(0);
          float2 vb[8];
          uint8_t* qweight_p1 = (uint8_t*)&vbInt1;
          uint8_t* qweight_p2 = (uint8_t*)&vbInt2;

#pragma unroll
          for (int j = 0; j < 4; j++) {
            // vb[j] = __halves2float2(__int2float_rn(((vbInt1 >> (j * 8))) &
            //                                        0xF),
            //                         __int2float_rn(((vbInt1) >> (j * 8 + 4)) & 0xF));
            // vb[j + 4] = __halves2float2(__int2float_rn(((vbInt2) >> (j * 8)) &
            //                                            0xF),
            //                             __int2float_rn((((vbInt2) >> (j * 8 + 4))) & 0xF));
            vb[j] = {__int2float_rn(((*(qweight_p1 + j))) & 0xF), __int2float_rn(((*(qweight_p1 + j)) >> 4) & 0xF)};
            vb[j + 4] = {__int2float_rn(((*(qweight_p2 + j))) & 0xF), __int2float_rn((((*(qweight_p2 + j)) >> 4)) & 0xF)};
          }

          float2 va[4];
          va[0] =
              (k_offset < MATRIX_K) ? __half22float2(((inA_start))[i]) : res2;
          va[1] = (k_offset + 1 < MATRIX_K) ? __half22float2((inA_start)[i + 1])
                                            : res2;
          va[2] = (k_offset + 2 < MATRIX_K) ? __half22float2((inA_start)[i + 2])
                                            : res2;
          va[3] = (k_offset + 3 < MATRIX_K) ? __half22float2((inA_start)[i + 3])
                                            : res2;

#pragma unroll
          for (int j = 0; j < 4; j++) {
            int vbx = vb[j].x;
            int vby = vb[j].y;
            vb[j].x = fmaf(scale_h0, vb[j].x, -hzero_scale_0);
            vb[j].y = fmaf(scale_h0, vb[j].y, -hzero_scale_0);
            if ((__half2float(Tweight_unpack[n_offset_x + (k_offset + j * 2) * MATRIX_N]) - (vb[j].x)) > 0.0002) {
              printf("-%d %d", vbx, vby);
              return;
            }
            Tweight_unpack[n_offset_x + (k_offset + j * 2 + 1) * MATRIX_N] == (vb[j].y);
            res2.x = fmaf(va[j].x, vb[j].x, res2.x);
            res2.y = fmaf(va[j].y, vb[j].y, res2.y);
            vb[4 + j].x = fmaf(scale_h1, vb[4 + j].x, -hzero_scale_1);
            vb[4 + j].y = fmaf(scale_h1, vb[4 + j].y, -hzero_scale_1);
            Tweight_unpack[n_offset_x + 1 + (k_offset + j * 2) * MATRIX_N] == (vb[4 + j].x);
            Tweight_unpack[n_offset_x + 1 + (k_offset + j * 2 + 1) * MATRIX_N] == (vb[j + 4].y);
            res2_1.x = fmaf(va[j].x, vb[4 + j].x, res2_1.x);
            res2_1.y = fmaf(va[j].y, vb[4 + j].y, res2_1.y);
          }

          sum[0] += (res2.x) + (res2.y);
          sum[1] += (res2_1.x) + (res2_1.y);
        }

        // sum[0] += __half2float(res2.x);
        // sum[1] +=  __half2float(res2.y);
        bsum[0][tidx][tidy] = sum[0];
        bsum[1][tidx][tidy] = sum[1];

        __syncthreads();
        sum[0] = 0;
        sum[1] = 0;

#pragma unroll
        for (int i = 0; i < 2; i++) {
          sum[i] = bsum[i][tidy][tidx];
          __syncthreads();
          sum[i] = warpReduceSum<32>(sum[i]);
          if (tidx == 0) {
            out[+0 * MATRIX_N + bid * width_element_per_block + tidy * 2 + i] =
                __float2half_rn(sum[i]);
          }
        }
      }
    }
  }
}

half2 __halves2half2(half x, half y) { return {(x), (y)}; }
half __int2half_rn(int x) { return __float2half_rn(float(x)); }
half2 __half2half2(half x) { return {(x), (x)}; }
half2 __hfma2(half2 a, half2 b, half2 c, float neg=1.0) {
  float2 res;
  res.x = __half2float(a.x) * __half2float(b.x) + neg*__half2float(c.x);
  res.y = __half2float(a.y) * __half2float(b.y) + neg*__half2float(c.y);
  return __halves2half2(__float2half_rn(res.x), __float2half_rn(res.y));
}
half __hfma(half a, half b, half c, float neg = 1.0) {
  float res;
  res = __half2float(a) * __half2float(b) + neg * __half2float(c);
  
  return __float2half_rn(res);
}
half2 __hmul2(half2 a, half2 b) {
  float2 res;
  res.x = __half2float(a.x) * __half2float(b.x);
  res.y = __half2float(a.y) * __half2float(b.y);
  return __halves2half2(__float2half_rn(res.x), __float2half_rn(res.y));
}
half __hmul(half a, half b) {
  float res;
  res = __half2float(a) * __half2float(b);
  return __float2half_rn(res);
}
#define __global__
#define __shared__
template <typename T>
__global__ void cpu_gemv_NT(T* out, T* inA, uint32_t* inB, T* scales, uint32_t* qzeros, int32_t groupsize) {
  for (int bid = 0; bid < 11008 / 64; bid++) {
    float bsum[2][32][32 + 1];

    for (int tidy = 0; tidy < 32; tidy++) {
      bsum[0][0][32] = 0;
      int idx=tidy>0?tidy-1:0;
      float rs = std::accumulate(bsum[0][idx], bsum[0][idx] + 32, 0.f);
      rs = std::accumulate(bsum[1][idx], bsum[1][idx] + 32, 0.f);
      for (int tidx = 0; tidx < 32; tidx++) {
        // int bid = blockIdx.x;
        __shared__ T vecA[MATRIX_K];
        //__shared__ float bsum[2][32][32 + 1];
        float sum[2] = {0, 0};
        int y_start = tidx*8;  // tidx;
        half2 res2 = {};
        half2 res2_1 = {};
        int blockIdx_y = 0;
        half2* inA_start = (half2*)(inA + blockIdx_y * MATRIX_K + y_start);

        int n_offset_x = bid * width_element_per_block + tidy * 2;

        int start_group_id = (y_start / groupsize);
        int compressed_idx = tidy % 4;
        int group_nums = (MATRIX_K + groupsize - 1) / groupsize;
        int weight_rows = (MATRIX_K + 8 - 1) / 8;
        half2 scale = __halves2half2(((scales + start_group_id + n_offset_x * group_nums))[0], ((scales + start_group_id + (n_offset_x + 1) * group_nums))[0]);
        int32_t qzero_p = ((int32_t*)(qzeros + n_offset_x / 8 * group_nums + start_group_id))[0];
        half2 hzero = __halves2half2(__int2half_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
                                     __int2half_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf));
        half2 scale_h0 = __half2half2(scale.x);
        half2 scale_h1 = __half2half2(scale.y);
        half2 hzero_scale_0 = __half2half2(__float2half(__half2float(hzero.x) * __half2float(scale.x)));
        half2 hzero_scale_1 = __half2half2(__float2half(__half2float(hzero.y) * __half2float(scale.y)));

        // if ( bid == 0 && tidx == 0 && tidy == 0) {
        //   printf("scale_and_sz: %f %f %f %f\n", (y_start + block_k) / groupsize / 1.0f, start_group_id * 1.0f, scale_and_sz[2], scale_and_sz[3]);
        // }

#pragma unroll
        for (int i = 0; i < block_k / 8; i++) {  // read half2 * 4
          int i8 = i * 32 * 8;                   // half2*4
          res2 = {};
          res2_1 = {};
          int k_offset = y_start + i8 ;  // half2*4
          int g_id = k_offset / groupsize;

          uint32_t* hinB = inB + n_offset_x * weight_rows + k_offset / 8;
          uint32_t vbInt1 = (n_offset_x < MATRIX_N && (k_offset < MATRIX_K)) ? hinB[0] : int32_t(0);
          uint32_t vbInt2 = (n_offset_x + 1 < MATRIX_N && (k_offset < MATRIX_K))
                                ? (inB + (n_offset_x + 1) * weight_rows + k_offset / 8)[0]
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
            vb[j + 4] = __halves2half2(__int2half_rn(((*(qweight_p2 + j))) & 0xF),
                                       __int2half_rn((((*(qweight_p2 + j)) >> 4)) & 0xF));
          }

          if (g_id > start_group_id) {
            scale = __halves2half2(((scales + g_id + n_offset_x * group_nums))[0], ((scales + g_id + (n_offset_x + 1) * group_nums))[0]);
            qzero_p = ((uint32_t*)(qzeros + n_offset_x / 8 * group_nums + g_id))[0];
            hzero = __halves2half2(__int2half_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
                                   __int2half_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf));
            scale_h0 = __half2half2(scale.x);
            scale_h1 = __half2half2(scale.y);
            hzero_scale_0 = __half2half2(__float2half(__half2float(hzero.x) * __half2float(scale.x)));
            hzero_scale_1 = __half2half2(__float2half(__half2float(hzero.y) * __half2float(scale.y)));
            start_group_id = g_id;
          }

          half2 va[4];
          i8 >>= 1;
          va[0] = (k_offset < MATRIX_K) ? ((inA_start))[i8] : half2{0, 0};
          va[1] = (k_offset + 2 < MATRIX_K) ? ((inA_start))[i8 + 1] : half2{0, 0};
          va[2] = (k_offset + 4 < MATRIX_K) ? ((inA_start))[i8 + 2] : half2{0, 0};
          va[3] = (k_offset + 6 < MATRIX_K) ? ((inA_start))[i8 + 3] : half2{0, 0};

#pragma unroll
          for (int j = 0; j < 4; j++) {
            vb[j] = __hfma2(scale_h0, vb[j], hzero_scale_0, -1);  ///////
            res2 = __hfma2(va[j], vb[j], res2);
            vb[4 + j] = __hfma2(scale_h1, vb[4 + j], hzero_scale_1, -1);  /////
            res2_1 = __hfma2(va[j], vb[4 + j], res2_1);
          }

          sum[0] += __half2float(res2.x) + __half2float(res2.y);
          sum[1] += __half2float(res2_1.x) + __half2float(res2_1.y);
        }
        bsum[0][tidy][tidx] = sum[0];
        bsum[1][tidy][tidx] = sum[1];

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
    }
  }
}


const int kBlockSize = 256;
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_HALF2(pointer) (reinterpret_cast<half2*>(&(pointer))[0])

template <typename T, int WBITS>
__global__ void DequantizeAndUnpackWeight248(T* out, uint32_t* qweight, T* scale, uint32_t* zeros, int group_size, const int in_features, const int n) {
  for (int bid = 0; bid < (MATRIX_N * ((MATRIX_K+7) / 8) + kBlockSize * 2 - 1) / kBlockSize / 2; bid++) {
    for (int tid = bid * kBlockSize; tid < (bid+1) * kBlockSize; tid++) {
      const int half_n = n / 2;
      const int qweight_rows = (in_features * WBITS + 31) / 32;

      const int compress_group_size = 32 / WBITS;
      const int max_num_in_bits = (1 << WBITS) - 1;
      int col_ind = (tid % half_n) * 2;
      int weight_in_row = tid / half_n * compress_group_size;
      half2 scale_v = FETCH_HALF2(scale[weight_in_row / group_size * n + col_ind]);
      uint32_t zero_v = zeros[weight_in_row / group_size * (n / compress_group_size) + (col_ind) / compress_group_size];
      int zero_ind = col_ind % compress_group_size;
      uint8_t zv1 = 0;
      uint8_t zv2 = 0;
      if (WBITS == 8) {
        zv1 = ((uint8_t*)&zero_v)[compress_group_size-zero_ind-1];
        zv2 = ((uint8_t*)&zero_v)[compress_group_size-zero_ind-1-1];
      }else{
        zv1 = (zero_v >> (zero_ind * WBITS)) & max_num_in_bits;
        zv2 = (zero_v >> (zero_ind * WBITS + WBITS)) & max_num_in_bits;
      }
      half2 scale_zeros = __hmul2(__halves2half2(__short2half_rn(zv1), __short2half_rn(zv2)), scale_v);
      half2* out_h2 = reinterpret_cast<half2*>(out);

      float2 weight_int2 = FETCH_FLOAT2(qweight[tid * 2]);
      uint32_t weight_v1 = *reinterpret_cast<uint32_t*>(&weight_int2.x);
      uint32_t weight_v2 = *reinterpret_cast<uint32_t*>(&weight_int2.y);

      // decompress weights
      int remains = in_features - weight_in_row;
      if (remains >= compress_group_size) {
#pragma unroll
        for (int i = 0; i < compress_group_size; i++) {
          uint8_t wv1 = 0;
          uint8_t wv2 = 0;
          if (WBITS == 8) {
          zv1 = ((uint8_t*)&weight_v1)[compress_group_size - i - 1];
          zv2 = ((uint8_t*)&weight_v1)[compress_group_size - i - 1];
          } else {
          wv1 = (weight_v1 >> (i * WBITS)) & max_num_in_bits;
          wv2 = (weight_v2 >> (i * WBITS)) & max_num_in_bits;
          }
          half2 wv = __halves2half2(__short2half_rn(wv1), __short2half_rn(wv2));
          out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, scale_zeros, -1);
        }
      } else {
        for (int i = 0; i < remains; i++) {
          uint8_t wv1 = (weight_v1 >> (i * WBITS)) & max_num_in_bits;
          uint8_t wv2 = (weight_v2 >> (i * WBITS)) & max_num_in_bits;
          half2 wv = __halves2half2(__short2half_rn(wv1), __short2half_rn(wv2));
          out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, scale_zeros, -1);
        }
      }
      }
  }
}

#define __device__
#define __forceinline__
template <int WBITS>
__device__ __forceinline__ uint8_t iterator_qweight(const uint32_t* ptr, int idx) {
  int start_bits = idx*WBITS;
  int first = start_bits / 32;
  int end_bits = (start_bits + WBITS);
  int second = end_bits / 32;
  start_bits = start_bits % 32;
  end_bits = end_bits % 32;
  if (first == second) {
    return (ptr[first] >> (start_bits)) & ((1 << WBITS) - 1);
  } else {
    uint8_t v = (ptr[first] >> (start_bits));
    v |= ((ptr[second]) & ((1 << (end_bits) )- 1))<< (32-start_bits);
    return v;
  }
}

template <typename T, int WBITS>
__global__ void DequantizeAndUnpackWeight3567(T* out, uint32_t* qweight, T* scale, uint32_t* zeros, int group_size, const int in_features, const int row_n) {
  for (int bid = 0; bid < (MATRIX_N * ((MATRIX_K + 31) / 32) + kBlockSize - 1) / kBlockSize; bid++) {
    const int qweight_rows = (in_features * WBITS + 31) / 32;
    __shared__ uint32_t qweight_shared[WBITS * kBlockSize];
    for (int tid = bid * kBlockSize; tid < (bid + 1) * kBlockSize; tid++) {
  const int group_row_n = row_n * WBITS;
  int total_qw = qweight_rows * row_n;

  int theadidx_x = tid - kBlockSize * bid;
  uint32_t* qweight_thread = qweight_shared + WBITS * theadidx_x;

  int qweight_start = tid / row_n * group_row_n + tid % row_n;
  for (int j = 0; j < WBITS; j++) {
    int ind = qweight_start + row_n * j;
    qweight_thread[j] = ind < total_qw ? qweight[ind] : 0;
  }
  }
  for (int tid = bid * kBlockSize; tid < (bid + 1) * kBlockSize; tid++) {
  int theadidx_x = tid - kBlockSize * bid;
  int thread_group = theadidx_x * WBITS;
  const int group_row_n = row_n * WBITS;
  const int max_num_in_bits = (1 << WBITS) - 1;
  const int col_ind = (tid % row_n);
  const int compress_group_size = 32;
  const int fp16_weight_in_row = tid / row_n * compress_group_size;
  half scale_v[4];
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

  half scale_zeros[4];
  for (int i = 0; i <= scale_zero_to - scale_zero_from; i++) {
    scale_zeros[i] = __hmul(__short2half_rn(zv1[i]), scale_v[i]);
  }
  half2 scale_2 = __halves2half2(scale_v[0], scale_v[0]);
  half2 scale_zeros_2 = __halves2half2(scale_zeros[0], scale_zeros[0]);
  const uint32_t* qweight_thread = qweight_shared + WBITS * theadidx_x;
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
      half2 scale_2 = __halves2half2(scale_v[i / group_size], scale_v[(i + 16) / group_size]);
      half2 scale_zeros_2 = __halves2half2(scale_zeros[i / group_size], scale_zeros[(i + 16) / group_size]);
    }
    half2 res = __hfma2(wv, scale_2, scale_zeros_2, -1);
    out[((fp16_weight_in_row + i) * row_n + col_ind)] = res.x;
    out[((fp16_weight_in_row + i + 16) * row_n + col_ind)] = res.y;
  }
  } else {
  // decompress weights
  for (int i = 0; i < remains; i++) {
    uint8_t wv1 = iterator_qweight<WBITS>(qweight_thread, i);
    half wv = __short2half_rn(wv1);
    if (group_size < 32) {
      scale_2.x = scale_v[i / group_size];
      scale_zeros_2.x = scale_zeros[i / group_size];
    }
    half res = __hfma(wv, scale_2.x, scale_zeros_2.x, -1);
    out[((fp16_weight_in_row + i) * row_n + col_ind)] = res;
  }
  }
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
  for (int bid = 0; bid < (MATRIX_N / 2 * ((MATRIX_K + 31) / 32)+ kBlockSize - 1) / kBlockSize; bid++) {
    const int qweight_rows = (in_features * WBITS + 31) / 32;
    __shared__ uint2 qweight_shared[WBITS * kBlockSize];
    const int half_n = row_n / 2;

    for (int tid = bid * kBlockSize; tid < (bid + 1) * kBlockSize; tid++) {
      const int group_row_n = half_n * WBITS;
      int total_qw = qweight_rows * half_n;

      int theadidx_x = tid - kBlockSize * bid;
      uint2* qweight_thread = qweight_shared + WBITS * theadidx_x;

      int qweight_start = tid / half_n * group_row_n + tid % half_n;
      const uint2* qweigh2 = (const uint2*)qweight;
    #pragma unroll
      for (int j = 0; j < WBITS; j++) {
        int ind = qweight_start + half_n * j;
        qweight_thread[j] = ind < total_qw ? (qweigh2[ind]) : uint2();
      }
  }
  for (int tid = bid * kBlockSize; tid < (bid + 1) * kBlockSize; tid++) {
  int theadidx_x = tid - kBlockSize * bid;

  uint2* qweight_thread = qweight_shared + WBITS * theadidx_x;

  const int group_row_n = half_n * WBITS;
  const int max_num_in_bits = (1 << WBITS) - 1;
  const int col_ind = (tid % half_n);
  const int compress_group_size = 32;
  const int fp16_weight_in_row = tid / half_n * compress_group_size;
  half2 scale_v[4];
  const int scale_zero_from = fp16_weight_in_row / group_size;
  const int scale_zero_to = (fp16_weight_in_row + compress_group_size) / group_size;

  // decompress scales
  const half2* scale2 = reinterpret_cast<const half2*>(scale);
  for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
    scale_v[i] = (scale2[scale_zero_from_i * half_n + col_ind]);
  }

  // decompress zeros
  uchar2 zv1[4];
  int half_col_ind = col_ind * 2;
  const int zero_col_from = half_col_ind * WBITS / 32;
  const int zero_col_to = ((half_col_ind + 1) * WBITS-1) / 32;
  const int zero_col_to_2 = ((half_col_ind + 2) * WBITS-1) / 32;
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
      zv1[i].y = (zero_v >> (zero_bits_last + WBITS)) & max_num_in_bits;
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
  half2 scale_2 = scale_v[0];
  half2 scale_zeros_2 = scale_zeros[0];

  const int out_offset = ((fp16_weight_in_row)*half_n + col_ind);
  half2* out_h2 = reinterpret_cast<half2*>(out);

  // decompress weights
  int remains = in_features - fp16_weight_in_row;
  if (remains >= compress_group_size){
#pragma unroll
  for (int i = 0; i < compress_group_size / 2; i++) {
    uchar2 wv1 = iterator_qweight_v2<uint2, WBITS>(qweight_thread, i);
    uchar2 wv2 = iterator_qweight_v2<uint2, WBITS>(qweight_thread, 16 + i);

    half2 wv = __halves2half2(__ushort2half_rn(wv1.x), __ushort2half_rn(wv1.y));
    if (group_size < 32) {
      half2 scale_2 = scale_v[i / group_size];
      half2 scale_zeros_2 = scale_zeros[i / group_size];
    }
    half2 res = __hfma2(wv, scale_2, scale_zeros_2, -1);
    out_h2[out_offset + i * half_n] = res;

    // if (bid == 0 && threadIdx.x == 1 && i == 1) {
    //   printf("%d,%d,%d,%d,%d,%f,%f\n", col_ind, int(wv1), int(wv2), row_n, col_ind, __half2float(wv.x), __half2float(wv.y));
    // }
    wv = __halves2half2(__ushort2half_rn(wv2.x), __ushort2half_rn(wv2.y));
    if (group_size < 32) {
      half2 scale_2 = scale_v[(i + 16) / group_size];
      half2 scale_zeros_2 = scale_zeros[(i + 16) / group_size];
    }
    res = __hfma2(wv, scale_2, scale_zeros_2, -1);
    out_h2[(out_offset + (i + 16) * half_n)] = res;
  }
  } else {
  // decompress weights
  for (int i = 0; i < remains; i++) {
    uchar2 wv1 = iterator_qweight_v2<uint2, WBITS>(qweight_thread, i);

    half2 wv = __halves2half2(__ushort2half_rn(wv1.x), __ushort2half_rn(wv1.y));
    if (group_size < 32) {
      half2 scale_2 = scale_v[i / group_size];
      half2 scale_zeros_2 = scale_zeros[i / group_size];
    }
    half2 res = __hfma2(wv, scale_2, scale_zeros_2, -1);
    out_h2[out_offset + i * half_n] = res;
  }
  }
  }
  }
}


}  // namespace cpu
using cpu::half_to_float;
using cpu::ushort;

void cpu_gemv_wrapper(ushort* out, ushort* Tweight_unpack, ushort* inA,
                      uint32_t* inB, ushort* scales, uint32_t* qzeros,
                      int32_t groupsize) {
  // cpu_gemv<ushort>(out, Tweight_unpack, inA, inB, scales, qzeros, groupsize);
  cpu::cpu_gemv_NT<ushort>(out, inA, inB, scales, qzeros, groupsize);
}

long long current_timestamp() {
  struct timeval te;
  gettimeofday(&te, NULL);                                          // get current time
  long long milliseconds = te.tv_sec * 1000LL + te.tv_usec / 1000;  // calculate milliseconds
  // printf("milliseconds: %lld\n", milliseconds);
  return milliseconds;
}

void dq_cpu_lauch(void* out, int32_t* inA, void* scales, int32_t* qzeros,
                  int32_t groupsize, int bits, int matrix_k, int matrix_n) {
  MATRIX_K = matrix_k;
  MATRIX_N = matrix_n;
  if (bits==4) {
    cpu::DequantizeAndUnpackWeight3567_v2<ushort, 4>((ushort*)out, (uint32_t*)inA, (ushort*)scales, (uint32_t*)qzeros, groupsize, (MATRIX_K * 4 + 31) / 32, MATRIX_N);
  } else {
    printf("not support bits %d\n", bits);
  }
}

void dq_wrapper(ushort* out, ushort* out_ref, uint32_t* inA, ushort* scales, uint32_t* qzeros,int32_t bits,
                int32_t groupsize) {
  uint64_t start = current_timestamp();
  // cpu_gemv<ushort>(out, Tweight_unpack, inA, inB, scales, qzeros, groupsize);
  uint64_t cost = current_timestamp() - start;
  start = current_timestamp();
  if (bits==4) {
  //cpu::DequantizeAndUnpackWeight248<ushort, 4>(out, inA, scales, qzeros, groupsize, MATRIX_K, MATRIX_N);
    cpu::DequantizeAndUnpackWeight3567_v2<ushort, 4>(out, inA, scales, qzeros, groupsize, MATRIX_K, MATRIX_N);
  } else {
    cpu::DequantizeAndUnpackWeight3567_v2<ushort, 5>(out, inA, scales, qzeros, groupsize, MATRIX_K, MATRIX_N);
  }
  cost = current_timestamp() - start;
  int cnt = 0;
  for (int i = 0; i < MATRIX_K * MATRIX_N; i++) {
    float a=half_to_float(out_ref[i]);
    float diff = std::abs(half_to_float(out_ref[i]) - half_to_float(out[i]));
    if (cnt < 10 && diff > 0.001) {
      cnt++;
      printf("%d: ref_%f ,got_%f\n", i, half_to_float(out_ref[i]), half_to_float(out[i]));
    }
  }
  printf("done\n");
}