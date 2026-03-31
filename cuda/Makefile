CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_75

all: libcudatensor.a

libcudatensor.a: kernels.o
	ar rcs $@ $^

kernels.o: kernels.cu tensor_cuda.h
	$(NVCC) -O3 -Xcompiler -fPIC -arch=$(ARCH) -c kernels.cu -o $@

clean:
	rm -f kernels.o libcudatensor.a
