#include "gemv.cuh"
#include <cstdio>
#include <fstream>
#include <unistd.h>

int MATRIX_M=0;
int MATRIX_K=0;
int MATRIX_N=0;

SampleData build_input() {
  MATRIX_M = 1;
  MATRIX_K = 4096;
  MATRIX_N = 12288;

  short *input = new short[MATRIX_M * MATRIX_K];
  int32_t *qweight = new int32_t[MATRIX_N * MATRIX_K / 8];
  short *scale = new short[MATRIX_N * MATRIX_K / 128];
  int32_t *qzero = new int32_t[MATRIX_N / 8 * MATRIX_K / 128];
  short *f16weight = new short[MATRIX_N * MATRIX_K];
  short *out = new short[MATRIX_M * MATRIX_N];
  SampleData data = {
      input,
      qweight,
      scale,
      qzero,
      f16weight,
      out,
      {MATRIX_M * MATRIX_K * sizeof(short), MATRIX_N * MATRIX_K / 8 * sizeof(int32_t),
       MATRIX_N * MATRIX_K / 128 * sizeof(short), MATRIX_N / 8 * MATRIX_K / 128 * sizeof(int32_t),
       MATRIX_N * MATRIX_K * sizeof(short), MATRIX_M * MATRIX_N * sizeof(short)}};

  std::string data_dir = "/home/jicwen/work/gemv_q/qmatmul_3/";
  {

    std::ifstream in(data_dir +"/input.bin", std::ios::in | std::ios::binary);
    in.read((char *)input, data.size.input_size);
  }

  {
    std::ifstream in(data_dir+"qweight.bin",
                     std::ios::in | std::ios::binary);
    in.read((char *)qweight, data.size.qweight_size);
  }

  {
    std::ifstream in(data_dir+"scales.bin",
                     std::ios::in | std::ios::binary);
    in.read((char *)scale, data.size.scale_size);
  }

  {
    std::ifstream in(data_dir+"qzeros.bin",
                     std::ios::in | std::ios::binary);
    in.read((char *)qzero, data.size.qzero_size);
  }

  {
    std::ifstream in(data_dir+"weight.bin",
                     std::ios::in | std::ios::binary);
    in.read((char *)f16weight, data.size.f16weight_size);
  }
  {
    std::ifstream in(data_dir+"out.bin",
                     std::ios::in | std::ios::binary);
    in.read((char *)out, data.size.out_size);
  }
  return data;
}

void vecquant4matmul(short *vec, int32_t *mat, short *mul, short *scales,
                     int32_t *zeros, int groupsize, int m, int k, int n);

void cpu_gemv_wrapper(ushort *out, ushort *Tweight_unpack, ushort *inA,
                      uint32_t *inB, ushort *scales, uint32_t *qzeros,
                      int32_t groupsize);
void dq_wrapper(ushort* out, ushort* out_ref, uint32_t* inB, ushort* scales, uint32_t* qzeros,
                int32_t groupsize);
int main() {
  printf("start\n");
  SampleData data = build_input();
  // vecquant4matmul(data.input, data.qweight, data.out, data.scale, data.qzero, 128, 1,
  //                 4096, 12288);
  // cpu_gemv_wrapper((ushort *)data.out, (ushort *)data.f16weight,
  //                  (ushort *)data.input, (uint32_t *)data.qweight,
  //                  (ushort *)data.scale, (uint32_t *)data.qzero, 128);
  ushort* dq_weight = new ushort[data.size.f16weight_size / sizeof(ushort)];
  dq_wrapper(dq_weight, (ushort*)data.f16weight, (uint32_t*)data.qweight, (ushort*)data.scale, (uint32_t*)data.qzero, 128);
  QbitGemv(&data);

  return 0;
}