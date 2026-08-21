#ifndef HASH_KERNEL_H
#define HASH_KERNEL_H

#include <vector>
#include <mutex>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

// Splits the host tables into one contiguous slice per GPU.
void split_tables(int device_num,
                  int* const tableA, int* const tableB,
                  int* const tableA_val, int* const tableB_val,
                  long long *tableA_size, long long *tableB_size,
                  int **tableA_split, int **tableB_split,
                  int **tableA_val_split, int **tableB_val_split,
                  const long long R, const long long S);

// Runs the whole multi GPU join. A negative override_* argument keeps the default.
void gpu_main(std::vector<int> *solution, std::vector<int> *solution_ref,
              const std::string& r_inputPath, const std::string& s_inputPath,
              long long R, long long S,
              int use_bf = 0, bool use_nvlink = false, bool use_warp_compact = false,
              double true_false_ratio = 0.5, int s_chunks = 1, bool sanityCheck = false,
              int kernel_fusion = 0, double redist_skew_threshold = 0.0,
              long long override_num_bits = -1, int override_num_hashes = -1,
              int override_partition_rs_split = -1,
              int override_partition_size = -1, int override_second_partition_size = -1,
              int fused_probe_breakdown = 0,
              int read_val = 0,
              int r_shuffle_enable = 0,
              double r_replicate_ratio = -1.0,
              double answer_scale = 1.0);
#endif
