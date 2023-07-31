#pragma once
#include <stdint.h>
struct SampleData {
  short *input;
  int32_t *qweight;
  short *scale;
  int32_t *qzero;
  short* f16weight;
  short* out;
  int32_t bits;
  int32_t group_size;
  struct {
    int input_size;
    int qweight_size;
    int scale_size;
    int qzero_size;
    int f16weight_size;
    int out_size;
  } size;
};

extern "C" int QbitGemv(SampleData *data);