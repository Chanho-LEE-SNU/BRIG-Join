# Install prefixes. Override in the environment or on the command line:
#   make NCCL_HOME=/path/to/nccl NVSHMEM_HOME=/path/to/nvshmem
CUDA_HOME    ?= /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
NCCL_HOME    ?= /path/to/nccl
NVSHMEM_HOME ?= /path/to/libnvshmem-linux-x86_64-3.2.5_cuda12-archive

CXX = mpic++
CXXFLAGS = -g -std=c++11 -Wall -Wno-sign-compare -O3 -fPIE \
	   -I$(CUDA_HOME)/include \
	   -I$(NCCL_HOME)/include \
	   -I$(NVSHMEM_HOME)/include

LDFLAGS = -L$(CUDA_HOME)/lib64 \
      -L$(NCCL_HOME)/lib \
      -L$(NVSHMEM_HOME)/lib \
      -lnccl -lmpi -lcudart -lnvshmem_host \
      $(NVSHMEM_HOME)/lib/libnvshmem_device.a

NVCXX = nvcc
NVCXXFLAGS = -g --ptxas-options=-v -std=c++11 -O3 \
                 -rdc=true -ccbin mpic++ -gencode arch=compute_86,code=sm_86 \
		 -Xcompiler -fPIE \
		 -Xcompiler -mno-avx512fp16 \
                 -I$(CUDA_HOME)/include \
                 -I$(NCCL_HOME)/include \
                 -I$(NVSHMEM_HOME)/include



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
