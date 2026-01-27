nvcc -c mm.cu -o mm.o
nvcc test_gemm.cpp mm.o -o test -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
./test