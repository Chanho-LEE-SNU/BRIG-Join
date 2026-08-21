Container: `nvcr.io/nvidia/nvhpc:25.3-devel-cuda12.8-ubuntu24.04` (NVIDIA HPC SDK 25.3, CUDA 12.8, NVSHMEM 3.2.5).

```bash
apptainer exec --nv <TODO: path to nvhpc-25.3.sif> bash
```

Run in this order.

```bash
DATA_DIR=/path/to/data
mkdir -p "$DATA_DIR"

cd datagen && make

./gen_dataset 640000000 640000000 10.0 "$DATA_DIR"

cd .. && make

nvshmrun -n 4 ./bin/hash_join \
"$DATA_DIR"/gen_R640000000_ans10.0pct.bin "$DATA_DIR"/gen_S640000000_ans10.0pct.bin \
"$DATA_DIR"/output.bin 640000000 640000000 0 1024 \
14 1 1 0.1 1 1 1.4 \
--partition_size 512 \
--fused_probe_breakdown 1 \
--r_replicate_ratio 4.5
```


# datagen
Without `--read_val 1`, the input files hold the 4-byte keys only and the 4-byte value of each tuple is synthesized from the row index at load time, outside the timed region. With `--read_val 1`, the values are read from a value block that follows the key block in the same file.
