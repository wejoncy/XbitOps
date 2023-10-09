# XbitOps
This project is mainly for LLM quantization inference.

It supports DQ (dequantize from 2-8bit to float16) and A16Wx_Gemv.

Intergrate exllama to support A16W4 is on the way

# installation
`pip install .`
or 
`python setup.py install`
or
`pip install git+https://github.com/wejoncy/XbitOps.git` for the latest version

if you don't want to compile it.
Just have a try
`pip install -i https://test.pypi.org/simple/ XbitOps`

# Benchmark
Roughly 2times faster then a16w16 gemv

