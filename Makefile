CUDA_HOME ?= /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8

CXX = mpic++
CXXFLAGS = -g -std=c++11 -Wall -Wno-sign-compare -O3 -fPIE \
	   -I$(CUDA_HOME)/include \
	   -I/datasets/leejinho/nccl/build/include \
	   -I/datasets/leejinho/libnvshmem-linux-x86_64-3.2.5_cuda12-archive/include

NVSHMEM_LIB_DIR = /datasets/leejinho/libnvshmem-linux-x86_64-3.2.5_cuda12-archive/lib

LDFLAGS = -L$(CUDA_HOME)/lib64 \
      -L/datasets/leejinho/nccl/build/lib \
      -L/datasets/leejinho/libnvshmem-linux-x86_64-3.2.5_cuda12-archive/lib \
      -lnccl -lmpi -lcudart -lnvshmem_host \
      /datasets/leejinho/libnvshmem-linux-x86_64-3.2.5_cuda12-archive/lib/libnvshmem_device.a

NVCXX = nvcc
NVCXXFLAGS = -g --ptxas-options=-v -std=c++11 -O3 \
                 -rdc=true -ccbin mpic++ -gencode arch=compute_86,code=sm_86 \
		 -Xcompiler -fPIE \
		 -Xcompiler -mno-avx512fp16 \
                 -I$(CUDA_HOME)/include \
                 -I/datasets/leejinho/nccl/build/include \
                 -I/datasets/leejinho/libnvshmem-linux-x86_64-3.2.5_cuda12-archive/include



SRCDIR = src
OBJDIR = obj
CUOBJDIR = cuobj
BINDIR = bin

INCS := $(wildcard $(SRCDIR)/*.h)
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
CUSRCS := $(wildcard $(SRCDIR)/*.cu)
CUOBJS := $(CUSRCS:$(SRCDIR)/%.cu=$(CUOBJDIR)/%.o)

all: $(BINDIR)/hash_join

$(BINDIR):
	@mkdir -p $(BINDIR)

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(CUOBJDIR):
	@mkdir -p $(CUOBJDIR)

$(BINDIR)/hash_join: $(OBJS) $(CUOBJS) | $(BINDIR)
	@echo "Linking $@"
	$(NVCXX) $(NVCXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling C++ $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CUOBJDIR)/%.o: $(SRCDIR)/%.cu | $(CUOBJDIR)
	@echo "Compiling CUDA $<"
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJDIR) $(CUOBJDIR) $(BINDIR)

