# BRIG-Join

Implementation of **BRIG-Join: Bloom-Filter Assisted Radix Join with Efficient Inter-GPU Communication**, by Chanho Lee and Bongki Moon.

**BRIG-Join** is a Bloom-filter assisted radix join with efficient inter-GPU communication. In conventional multi-GPU radix joins a significant portion of the execution time is spent on inter-GPU data transfer, yet much of the transferred data does not contribute to the final join result. BRIG-Join prunes irrelevant tuples before transfer with Bloom filters. It consists of:

1. **A Bloom-filter-assisted radix join pipeline** for multi-GPU systems that drastically reduces the inter-GPU data transfer.
2. **Efficient Bloom filter construction and querying** that is L2-aware and integrates into the existing radix partitioning phase.
3. **Fused join phases** that collapse multiple phases into a single kernel, eliminating intermediate shared-memory spills to device memory.
4. **A custom GPU-native all-reduce library** that performs bitwise-OR all-reduce entirely within the GPU pipeline.

## Requirements

- NVIDIA HPC SDK 25.3 (CUDA 12.8, NVSHMEM 3.2.5)
- NCCL 2.25.1
- An MPI implementation (`mpic++`); we use HPC-X 2.22.1 from the same SDK
- The `nvidia_peermem` kernel module loaded on the host
- Apptainer (recommended, for a reproducible toolchain)

BRIG-Join targets commodity servers whose GPUs are connected over PCIe, where inter-GPU transfer dominates the join. No NVLink or NVSwitch is required. The evaluation uses four NVIDIA RTX A6000 GPUs (6 MB L2, PCIe 4.0 x16), which is what the `sm_86` target in the `Makefile` refers to.

The join also runs on a single GPU, but with one GPU there is no inter-GPU transfer to prune, so multiple GPUs are recommended.

## Container (recommended)

The container image `nvcr.io/nvidia/nvhpc:25.3-devel-cuda12.8-ubuntu24.04` provides the compilers, CUDA 12.8 and MPI. All commands below are run inside it:

```bash
apptainer exec --nv --writable-tmpfs <path to nvhpc-25.3.sif> bash
```

## Build

The `Makefile` starts with three install prefixes. `CUDA_HOME` is the HPC SDK default; the other two are placeholders, so point them at your installations:

```make
CUDA_HOME    ?= /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
NCCL_HOME    ?= /path/to/nccl/build
NVSHMEM_HOME ?= /path/to/libnvshmem-linux-x86_64-3.2.5_cuda12-archive
```

They can also be overridden without touching the file:

```bash
make NCCL_HOME=/path/to/nccl/build NVSHMEM_HOME=/path/to/nvshmem
```

Then:

```bash
make                            # -> ./bin/hash_join
cd datagen && make && cd ..     # -> ./datagen/gen_dataset
```

`NVCXXFLAGS` targets `sm_86`; change `-gencode` in the `Makefile` for other architectures. `nvcc` and `mpic++` are taken from `PATH`, not from `CUDA_HOME`. Inside the container both are already there. If you're not using a container, make sure they resolve to the CUDA and MPI you intend to build against.


## Prepare datasets

Both relations are raw arrays of `int32` keys in a binary file. `gen_dataset` produces the synthetic workloads:

```bash
DATA_DIR=/path/to/data
mkdir -p "$DATA_DIR"

./datagen/gen_dataset 640000000 640000000 10.0 "$DATA_DIR"
```

The arguments are `<R_size> <S_size> <answer_pct> <out_dir> [seed] [--max <pct>] [--verify]`, where `answer_pct` is the mean percentage of R values that have a match in S. The files land at `$DATA_DIR/gen_R<R>_ans<pct>pct.bin` and `$DATA_DIR/gen_S<S>_ans<pct>pct.bin`.

Without `--read_val 1`, the input files hold the 4-byte keys only and the 4-byte value of each tuple is synthesized from the row index at load time, outside the timed region. With `--read_val 1`, the values are read from a value block that follows the key block in the same file.

## Quick example

Four GPUs, 640M x 640M tuples, 10% match rate:

```bash
nvshmrun -n 4 ./bin/hash_join \
  "$DATA_DIR"/gen_R640000000_ans10.0pct.bin "$DATA_DIR"/gen_S640000000_ans10.0pct.bin \
  "$DATA_DIR"/output.bin 640000000 640000000 0 1024 \
  14 1 1 0.1 1 1 1.4 \
  --partition_size 512 \
  --fused_probe_breakdown 1 \
  --r_replicate_ratio 4.5
```

`-n` is the GPU count; `-n 1` runs the same command on one GPU. See `src/driver.cpp` for the full list of positional arguments and options.

## Repository layout

```
src/
  driver.cpp              entry point, argument parsing
  hash_kernel.cu          table loading and per-GPU slicing
  vlsplit_hash_kernel.cu  two-level radix partitioning
  bloom.cu                Bloom filter construction and probing
  bf_allreduce.cu         GPU-native OR all-reduce
  join_executor.cu        per-chunk join pipeline
  kernel_functions.cu     join kernels
  r_replicate.cu          build-side replication path
  r_shuffle.cu            all-to-all redistribution to the owner GPU
  s_redistribute.cu       candidate rebalancing across GPUs
  sanity_check.cpp        CPU reference join
datagen/                  synthetic dataset generator
```
