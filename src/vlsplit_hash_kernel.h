#ifndef VSPLIT_HASH_KERNEL_H
#define VSPLIT_HASH_KERNEL_H

#include <vector>
#include <mutex>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "hash_kernel.h"
#include "bloom.cuh"


// Radix partitions R and S in two levels.
void partial_jointable_gpu_partition(
                           const int* const d_adata, const int* const d_bdata, int* const d_oa_data, int* const d_ob_data,
                           int* const d_a_val, int* const d_b_val, int* const d_oa_val, int* const d_ob_val,
                           int2* const d_b_kv_out,
                           long long* const d_a_prefix_data, long long* const d_b_prefix_data,
                           long long* const d_block_index_split_r, long long* const d_block_index_split_s, long long* const d_a_offset, long long* const d_b_offset,
                           long long* const d_second_index_split_r, long long* const d_second_index_split_s, int* const d_answer_num,
                           const long long R,  const long long S, cudaStream_t gpu_stream, cudaEvent_t gpu_event,
                           double& out_partition_sec, double& out_hist_sec, bool skip_r_hist = false);

// Same as partial_jointable_gpu_partition but splits every stage into R and S halves so each is timed separately.
void partial_jointable_gpu_partition_rs_split(
                           const int* const d_adata, const int* const d_bdata, int* const d_oa_data, int* const d_ob_data,
                           int* const d_a_val, int* const d_b_val, int* const d_oa_val, int* const d_ob_val,
                           int2* const d_b_kv_out,
                           long long* const d_a_prefix_data, long long* const d_b_prefix_data,
                           long long* const d_block_index_split_r, long long* const d_block_index_split_s, long long* const d_a_offset, long long* const d_b_offset,
                           long long* const d_second_index_split_r, long long* const d_second_index_split_s, int* const d_answer_num,
                           const long long R,  const long long S, cudaStream_t gpu_stream, cudaEvent_t gpu_event,
                           double& out_partition_sec, double& out_partition_r_sec, double& out_partition_s_sec,
                           double& out_hist_r_sec, double& out_hist_s_sec, bool skip_r_hist = false);

// Partitions only R in two levels. The result stays in d_adata in place.
void partial_jointable_gpu_partition_r_only(
        int* const d_adata, int* const d_oa_data,
        int* const d_a_val, int* const d_oa_val,
        long long* const d_a_prefix_data, long long* const d_block_index_split_r,
        long long* const d_a_offset, long long* const d_second_index_split_r,
        const long long R, cudaStream_t gpu_stream);

// Partitions only S in two levels. The result stays in d_bdata in place.
void partial_jointable_gpu_partition_s_only(
        int* const d_bdata, int* const d_ob_data,
        int* const d_b_val, int* const d_ob_val,
        int2* const d_b_kv_out,
        long long* const d_b_prefix_data, long long* const d_block_index_split_s,
        long long* const d_b_offset, long long* const d_second_index_split_s,
        int* const d_answer_num, const long long S, cudaStream_t gpu_stream);

// Probes the already partitioned R against an S buffer received from a remote PE.
void partial_jointable_gpu_run_only_s(
        const int* const d_adata, const int* const d_bdata,
        const int* const d_a_val, const int* const d_b_val,
        long long* const d_block_index_split_r, long long* const d_block_index_split_s,
        long long* const d_second_index_split_r, long long* const d_second_index_split_s, int* const d_answer, int* const d_answer_val, int* const d_answer_s_val, int* const d_answer_num, int max_int_number_for_shared_memory, int * final_answer_number,
        cudaStream_t gpu_stream, cudaEvent_t gpu_event, int* const d_answer_write_idx);

#endif
