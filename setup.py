import os
import io
from packaging.version import parse, Version
from typing import List, Set
import subprocess
import setuptools

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
# TODO(woosuk): Should we use -O3?
NVCC_FLAGS = ["-O2", "-std=c++17"]
ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
               "-U__CUDA_NO_HALF_OPERATORS__",
               "-U__CUDA_NO_HALF_CONVERSIONS__",
               "-U__CUDA_NO_HALF2_OPERATORS__",
               "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
               "-D__use_torch__",
               "--expt-relaxed-constexpr",
               "--expt-extended-lambda",]

if CUDA_HOME is None:
    raise RuntimeError(
        f"Cannot find CUDA_HOME. CUDA must be available in order to build the package.")


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def check_compatability(compute_capabilities: Set[int]) -> None:
    if os.getenv("CUDA_ARCH", "") == "ALL":
        compute_capabilities = set([70, 75, 80, 86, 89, 90])
        return 
    # Collect the compute capabilities of all available GPUs.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 7:
            raise RuntimeError(
                "GPUs with compute capability less than 7.0 are not supported.")
        compute_capabilities.add(major * 10 + minor)
    # If no GPU is available, add all supported compute capabilities.
    if not compute_capabilities:
        compute_capabilities.extend((70, 75, 80, 86, 90))


compute_capabilities: Set[int] = set()
check_compatability(compute_capabilities)
# Add target compute capabilities to NVCC flags.
for capability in compute_capabilities:
    NVCC_FLAGS += ["-gencode", f"arch=compute_{capability},code=sm_{capability}"]


# Validate the NVCC CUDA version.
nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if nvcc_cuda_version < Version("11.0"):
    raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
if 86 in compute_capabilities and nvcc_cuda_version < Version("11.1"):
    raise RuntimeError(
        "CUDA 11.1 or higher is required for GPUs with compute capability 8.6.")
if 90 in compute_capabilities and nvcc_cuda_version < Version("11.8"):
    raise RuntimeError(
        "CUDA 11.8 or higher is required for GPUs with compute capability 9.0.")

# Use NVCC threads to parallelize the build.
if nvcc_cuda_version >= Version("11.2"):
    num_threads = min(os.cpu_count(), 8)
    NVCC_FLAGS += ["--threads", str(num_threads)]

ext_modules = []
# dq operations.
include_dirs = os.path.dirname(os.path.abspath(__file__))+'/src'
working_dirs = os.path.dirname(os.path.abspath(__file__))

source_file = [
    os.path.join(working_dirs, 'src/dq_torch_ops.cc'),
    os.path.join(working_dirs, 'src/cu/gemv_w4a16_pt.cu'),
    os.path.join(working_dirs, 'src/cu/unpack_weight_2_to_7.cu')]

dq_extension = CUDAExtension(
    name="XbitOps",
    sources=source_file,
    include_dirs=[include_dirs],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(dq_extension)


def get_requirements() -> List[str]:
    return [""]
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements

def read_readme() -> str:
    """Read the README file."""
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()

setuptools.setup(
    name="XbitOps",
    version="0.1.2",
    author="wejoncy",
    license="Apache 2.0",
    description="A high-efficient dequantization method for n-bit GPTQ",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wejoncy/XbitOps",
    project_urls={
        "Homepage": "https://github.com/wejoncy/XbitOps",
        "Documentation": "https://github.com/wejoncy/XbitOps",
        },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(
        exclude=("qmatmul", "src",)),
    python_requires=">=3.7",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
