#ifndef JOIN_EXECUTOR_H
#define JOIN_EXECUTOR_H

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "bloom.cuh"

struct JoinChunkContext {
    int me;
    int device_num;

    int  use_bf;
    bool use_nvlink;
    int  kernel_fusion;
    bool fused_probe_breakdown;

    int*   GPUA_split;
    int*   GPUB_send_buffer;
    int*   GPU_buffer;
    int*   h_pcie_stage;
    int*   GPUA_val_split;
    int*   GPUB_send_buffer_val;
    int*   GPU_buffer_val;
    int*   h_pcie_stage_val;
    int2*  GPUB_send_buffer_kv;

    int*       GPU_OB_split;
    int*       GPU_OB_val_split;
    long long* GPU_B_prefix_split;
    long long* GPU_B_offset;

    long long*  GPU_block_index_split_R;
    long long** GPU_block_index_split_S;
    long long*  GPU_second_index_split_R;
    long long** GPU_second_index_split_S;

    int* GPU_answer;
    int* GPU_answer_val;
    int* GPU_answer_s_val;
    int* GPU_answer_num;
    int* GPU_answer_write_idx;
    int* answer_num_host;

    std::vector<BloomFilter>* bf;
    size_t    bf_words;
    uint32_t* d_bf_global;

    cudaStream_t gpu_stream;
    cudaEvent_t  gpu_event;
    cudaStream_t send_stream;

    int max_int_number_for_shared_memory;

    double* first_pass_probe_sec;
    double* cumulative_bf_exchange_sec;
    double* cumulative_bf_pipeline_sec;
    double* cumulative_pre_send_barrier_sec;
    double* cumulative_inner_barrier_sec;
    double* cumulative_transfer_sec;
    double* cumulative_second_pass_sec;
    double* cumulative_fused_ht_sec;
    double* cumulative_fused_local_sec;
    double* cumulative_fused_get_sec;
    double* cumulative_fused_remote_probe_sec;
    double* cumulative_fused_wall_sec;
    double* cumulative_fused_setup_sec;

    double* cumulative_fp_preamble_sec;
    double* cumulative_fp_memset_sec;
    double* cumulative_fp_initbar_sec;
    double* cumulative_fp_build_sec;
    double* cumulative_fp_local_sec;
    double* cumulative_fp_reduce_sec;

    long long** d_s_block_ptrs;
    long long** d_s_second_ptrs;
    long long*  d_phase_timers;
    long long   d_phase_timers_slots;

    int*    total_answer_num;
    int*    total_overflow_count;
};

// Runs the local probe, the inter GPU transfer and the cross GPU probe for one S chunk.
void join_execute_host(JoinChunkContext& ctx,
                       int* effective_s_ptr, int* effective_s_val, long long actual_b_size);

#endif
