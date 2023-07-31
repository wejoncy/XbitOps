nvcc gemv_w4a16.cu -O3   -lcuda -lcurand -lcublas --gpu-architecture=compute_70  -o gemv
