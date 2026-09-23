//
// Created by Chanho LEE on 2023-10-09.
//

#ifndef TEST_KERNEL_FUNCTIONS_H
#define TEST_KERNEL_FUNCTIONS_H

#include <vector>
#include <mutex>
#include <algorithm>
#include "bloom.cuh"

constexpr int SECOND_PASS_SCAN_SIZE = 256;
constexpr int SECOND_PASS_MAX = 4096;
constexpr int PARTITION_MAX = 2048;

// Chooses the block dimension for the prefix and scan kernels.
inline int prefix_scan_block_dim(int ps, int max_threads) { return ps < max_threads ? ps : max_threads; }

// Clears the partition counters and the answer counter.
__global__ void gpu_prefix_init(long long * g_a_prefix_data, long long * g_b_prefix_data, int partition_size, int * g_answer_num);

// Clears the partition counters.
__global__ void gpu_prefix_init(long long * g_a_prefix_data, long long * g_b_prefix_data, int partition_size);

// Turns first level partition counts into bucket boundaries and scatter offsets.
__global__ void gpu_make_simple_prefix(int * g_idata, long long * g_prefix_data, long long * g_block_index_split, long long data_end, int partition_size, long long * g_offset);

// Counts second level sub buckets within each first level bucket and scans them.
__global__ void gpu_run_hist_second_pass(int *g_idata, int *g_odata, long long *g_block_index_split, long long *g_second_index_split, int second_partition_size);

// Scatters keys and payloads into second level sub bucket order.
__global__ void gpu_build_table_second_pass(int *g_idata, int *g_odata, int *g_ival, int *g_oval, long long *g_block_index_split, long long *g_second_index_split, int second_partition_size);

// Same as gpu_build_table_second_pass but writes (key, value) int2 pairs.
__global__ void gpu_build_table_second_pass_kv(int *g_idata, int *g_ival, int2 *g_okv, long long *g_block_index_split, long long *g_second_index_split, int second_partition_size);


extern __device__ int d_probe_overflow_count;
extern __device__ int d_answer_capacity;

// Interleaves separate key and value arrays into an array of (key, value) pairs.
__global__ void gpu_interleave_kv(const int* keys, const int* vals, int2* out, long long n);

// Loads one R sub bucket into a shared memory hash table and probes it with S.
template<bool COLLECT>
__global__ void gpu_probe_with_second_pass(int *g_oa_data, int *g_ob_data, int *g_oa_val, int *g_ob_val, long long * ga_index, long long * gb_index, long long * ga_second_index, long long * gb_second_index, int partition_size, int second_partition_size, int * g_answer, int * g_answer_val, int * g_answer_s_val, int * g_answer_num, int * g_answer_write_idx, long long * d_phase_timers);

// Performs the local probe, the remote S transfer and the cross GPU probe in one kernel.
template<bool COLLECT>
__global__ void gpu_fused_probe_transfer(
    const int* __restrict__ r_data,
    const int* __restrict__ r_val,
    const int* __restrict__ local_s_data,
    const int* __restrict__ local_s_val,
    const int2* nvshmem_send_buf_kv,
    const long long* __restrict__ r_block_idx,
    const long long* __restrict__ r_second_idx,
    long long** all_s_block_idx,
    long long** all_s_second_idx,
    int me, int device_num,
    int ps, int sp,
    int* g_answer, int* g_answer_val, int* g_answer_s_val, int* g_answer_num, int* g_answer_write_idx,
    int ht_size, int chunk_size,
    long long* d_phase_timers,
    int single_remote_peer);


// Builds the first level partition histograms of R and S in one kernel.
__global__ void gpu_run_hist_for_both(int *g_adata, long long *g_a_prefix_data, long long a_data_end, int *g_bdata, long long *g_b_prefix_data, long long b_data_end,
                                      long long block_size, int partition_size);

// Scatters R and S into first level bucket order.
__global__ void gpu_make_sorted_table_for_both(int *g_i_a_data, int *g_o_a_data, int *g_i_a_val, int *g_o_a_val, long long *g_a_block_index_split, long long a_data_end, long long *g_a_offset,
                                               int *g_i_b_data, int *g_o_b_data, int *g_i_b_val, int *g_o_b_val, long long *g_b_block_index_split, long long b_data_end, long long *g_b_offset,
                                               long long block_size, int partition_size);

// Scatters while recording a separate partition offset for every block.
__global__ void gpu_make_block_by_block_offset(int * g_idata, int * g_odata, int * g_prefix_data, int * g_block_index_split, int block_size, int data_end, int partition_size, int * g_offset, int * g_per_block_offset);

// Probes one first level bucket per block with a linear scan.
__global__ void gpu_probe_in_a_single_block(int *g_oa_data, int *g_ob_data, long long * ga_index, long long * gb_index, int second_partition_size, int * g_answer_num);

// Same as gpu_probe_in_a_single_block with extra reporting of the match count.
__global__ void gpu_probe_in_a_single_block_with_check(int *g_oa_data, int *g_ob_data, long long * ga_index, long long * gb_index, int second_partition_size, int * g_answer_num);

// Prints device array values for debugging.
__global__ void print_gpu_value(int *g_data, int offset);

// Prints device array values with a tag for debugging.
__global__ void print_gpu_value(int *g_data, int offset, int check);

#endif //TEST_KERNEL_FUNCTIONS_H
