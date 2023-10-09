#include <torch/extension.h>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void lauch_deqantize_cuda_pt_kernel(torch::Tensor& b_fp16, const torch::Tensor& qweight_i32,
                                    const torch::Tensor& scale_fp16, const torch::Tensor& qzeros_i32,
                                    int bits, int groupsize, uint32_t mat_k, uint32_t mat_n);

void lauch_Gemv_kernel(torch::Tensor& out_fp16, const torch::Tensor& a_fp16, const torch::Tensor& qweight_i32,
                       const torch::Tensor& scale_fp16, const torch::Tensor& qzeros_i32,
                       int bits, int groupsize, uint32_t mat_m, uint32_t mat_k, uint32_t mat_n);

int MATRIX_M = 0;
int MATRIX_K = 0;
int MATRIX_N = 0;

torch::Tensor dequant_any_bit(const torch::Tensor& qweight, const torch::Tensor& scales,
                              const torch::Tensor& qzeros, int groupsize, int bits, int in_features) {
  CHECK_INPUT(qweight);
  CHECK_INPUT(scales);
  CHECK_INPUT(qzeros);
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2-dimensional");
  TORCH_CHECK(groupsize >= 16, "groupsize must be >= 16");
  TORCH_CHECK(bits >= 1 && bits <= 8, "bits must be >= 1 and <= 8");
  TORCH_CHECK((in_features * bits + 31) / 32 == qweight.size(0), "in_features must be >= 1");
  cudaSetDevice(qweight.device().index());
  at::Tensor output = at::zeros({in_features, qweight.size(1)}, scales.options());
  lauch_deqantize_cuda_pt_kernel(output, qweight, scales, qzeros, bits, groupsize, in_features, qweight.size(1));
  return output;
}

torch::Tensor op_gemv(const torch::Tensor& input_a, const torch::Tensor& qweight,
                      const torch::Tensor& scales, const torch::Tensor& qzeros,
                      int groupsize, int bits, int in_features) {
  CHECK_INPUT(input_a);
  CHECK_INPUT(qweight);
  CHECK_INPUT(scales);
  CHECK_INPUT(qzeros);
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2-dimensional");
  TORCH_CHECK(groupsize >= 16, "groupsize must be >= 16");
  TORCH_CHECK(bits >= 1 && bits <= 8, "bits must be >= 1 and <= 8");
  TORCH_CHECK((in_features * bits + 31) / 32 == qweight.size(0), "in_features must be >= 1");
  TORCH_CHECK(qweight.device().index() == input_a.device().index(), "input and weight must be on the same device");
  cudaSetDevice(qweight.device().index());
  std::vector<int64_t> outputshape ={input_a.size(0), qweight.size(1)};
  uint32_t mat_m = input_a.size(0);
  if (input_a.dim() > 2) {
    outputshape.insert(outputshape.begin()+1, input_a.size(1));
    mat_m *= input_a.size(1);
  }
  at::Tensor output = at::zeros(outputshape, scales.options());
  lauch_Gemv_kernel(output, input_a, qweight, scales, qzeros, bits, groupsize, mat_m, in_features, qweight.size(1));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant", &dequant_any_bit, "dequantize qweight to fp16, \nfunction type: const torch::Tensor& qweight, "
        "const torch::Tensor& scales, const torch::Tensor& qzeros, int groupsize, int bits, int in_features");
  m.def("gemv", &op_gemv, "gemv, \nfunction type: const torch::Tensor& input_a, const torch::Tensor& qweight, "
        "const torch::Tensor& scales, const torch::Tensor& qzeros, int groupsize, int bits, int in_features");
}
