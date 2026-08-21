#include "r_replicate.h"
#include "common.h"
#include "vlsplit_hash_kernel.h"
#include "kernel_functions.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

// Replicates R to every GPU and partitions it. Called once before the chunk loop.
void r_replicate_gather_r(JoinChunkContext& ctx,
                          int* R_full_key, int* R_full_val,
                          int* scratch_key, int* scratch_val,
                          long long* prefix_R, long long* offset_R,
                          const long long* tableA_size, long long R_total,
                          double& t_r_gather)
{
    auto t0 = std::chrono::steady_clock::now();
    const int me = ctx.me, D = ctx.device_num;

    std::vector<long long> off(D);
    long long acc = 0;
    for (int i = 0; i < D; i++) { off[i] = acc; acc += tableA_size[i]; }
    if (acc != R_total) {
        fprintf(stderr, "[R_REPLICATE][PE%d] slice sum (%lld) != R_total (%lld), aborting\n",
                me, acc, R_total);
        return;
    }

    const long long n = tableA_size[me];
    if (n > 0) {
        const size_t bytes = (size_t)n * sizeof(int);
        cudaMemcpyAsync(R_full_key + off[me], ctx.GPUA_split,
                        bytes, cudaMemcpyDeviceToDevice, ctx.gpu_stream);
        cudaMemcpyAsync(R_full_val + off[me], ctx.GPUA_val_split,
                        bytes, cudaMemcpyDeviceToDevice, ctx.gpu_stream);
        for (int p = 0; p < D; p++) {
            if (p == me) continue;
            nvshmem_putmem(R_full_key + off[me], ctx.GPUA_split,     bytes, p);
            nvshmem_putmem(R_full_val + off[me], ctx.GPUA_val_split, bytes, p);
        }
    }
    cudaStreamSynchronize(ctx.gpu_stream);
    nvshmem_quiet();
    nvshmem_barrier_all();

    partial_jointable_gpu_partition_r_only(
        R_full_key, scratch_key, R_full_val, scratch_val,
        prefix_R, ctx.GPU_block_index_split_R, offset_R, ctx.GPU_second_index_split_R,
        R_total, ctx.gpu_stream);

    t_r_gather += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("[PE%d][R_REPLICATE] R replicated+partitioned: %lld elems (%.4f sec)\n",
           me, R_total, t_r_gather);
}

// Partitions the local S and probes it against the replicated R. Returns the match count of this chunk.
int r_replicate_join_chunk(JoinChunkContext& ctx,
                           int* R_full_key, int* R_full_val,
                           int* effective_s_ptr, int* effective_s_val, long long actual_b_size,
                           double& t_partition, double& t_probe)
{
    long long* bidx_S = ctx.GPU_block_index_split_S[ctx.me];
    long long* sidx_S = ctx.GPU_second_index_split_S[ctx.me];

    auto t_p0 = std::chrono::steady_clock::now();
    partial_jointable_gpu_partition_s_only(
        effective_s_ptr, ctx.GPU_OB_split, effective_s_val, ctx.GPU_OB_val_split,
        nullptr, ctx.GPU_B_prefix_split, bidx_S, ctx.GPU_B_offset,
        sidx_S, ctx.GPU_answer_num, actual_b_size, ctx.gpu_stream);
    t_partition += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_p0).count();

    auto t_pr0 = std::chrono::steady_clock::now();
    { int zero = 0; cudaMemcpyToSymbol(d_probe_overflow_count, &zero, sizeof(int)); }
    size_t kv_shm = sizeof(int) * (size_t)ctx.max_int_number_for_shared_memory * 2;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(gpu_probe_with_second_pass<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kv_shm);
        attr_set = true;
    }
    gpu_probe_with_second_pass<false><<<partition_size * second_partition_size, thread_no, kv_shm, ctx.gpu_stream>>>(
        R_full_key, effective_s_ptr, R_full_val, effective_s_val,
        ctx.GPU_block_index_split_R, bidx_S, ctx.GPU_second_index_split_R, sidx_S,
        partition_size, second_partition_size,
        ctx.GPU_answer, ctx.GPU_answer_val, ctx.GPU_answer_s_val,
        ctx.GPU_answer_num, ctx.GPU_answer_write_idx, nullptr);
    cudaStreamSynchronize(ctx.gpu_stream);
    t_probe += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pr0).count();

    { int h = 0; cudaMemcpyFromSymbol(&h, d_probe_overflow_count, sizeof(int));
      *ctx.total_overflow_count += h; }

    int ans = 0;
    cudaMemcpy(&ans, ctx.GPU_answer_num, sizeof(int), cudaMemcpyDeviceToHost);
    printf("[PE%d][R_REPLICATE] S_local=%lld answers=%d\n", ctx.me, actual_b_size, ans);
    return ans;
}
