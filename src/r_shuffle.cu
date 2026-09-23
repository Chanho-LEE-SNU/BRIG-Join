#include "r_shuffle.h"
#include "common.h"
#include "vlsplit_hash_kernel.h"
#include "kernel_functions.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

static long long* g_mat   = nullptr;
static int        g_mat_D = 0;

// Reallocates the size matrix buffer when it is missing or sized for a different PE count.
static void ensure_mat(int D) {
    if (g_mat == nullptr || g_mat_D != D) {
        if (g_mat) nvshmem_free(g_mat);
        g_mat   = (long long*)nvshmem_malloc(sizeof(long long) * (long long)D * D);
        g_mat_D = D;
    }
}

// Preallocates the size matrix buffer used by the all to all exchange.
void r_shuffle_init(int D) { ensure_mat(D); }

// Releases the buffer allocated by r_shuffle_init.
void r_shuffle_free() {
    if (g_mat) { nvshmem_free(g_mat); g_mat = nullptr; g_mat_D = 0; }
}

// Redistributes a partitioned relation to its owner GPU. Returns the received element count, or -1 on failure.
long long all_to_all_to_owner(int* src_key, int* src_val,
                              long long* block_index,
                              int* dst_key, int* dst_val,
                              int me, int D, cudaStream_t stream)
{
    const int P = partition_size;
    if (P % D != 0) {
        if (me == 0)
            fprintf(stderr, "[R_SHUFFLE] partition_size (%d) must be a multiple of device_num (%d)\n", P, D);
        return -1;
    }
    const int bpg = P / D;

    std::vector<long long> h_bidx(P);
    cudaMemcpy(h_bidx.data(), block_index, sizeof(long long) * P, cudaMemcpyDeviceToHost);

    std::vector<long long> soff(D), scnt(D);
    for (int d = 0; d < D; d++) {
        long long start = (d * bpg == 0) ? 0 : h_bidx[d * bpg - 1];
        long long end   = h_bidx[(d + 1) * bpg - 1];
        soff[d] = start;
        scnt[d] = end - start;
    }

    ensure_mat(D);
    {
        std::vector<long long> row(D);
        for (int d = 0; d < D; d++) row[d] = scnt[d];
        cudaMemcpy(&g_mat[me * D], row.data(), sizeof(long long) * D, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        for (int p = 0; p < D; p++) {
            if (p == me) continue;
            nvshmem_putmem(&g_mat[me * D], &g_mat[me * D], sizeof(long long) * D, p);
        }
        nvshmem_quiet();
        nvshmem_barrier_all();
    }
    std::vector<long long> h_mat((long long)D * D);
    cudaMemcpy(h_mat.data(), g_mat, sizeof(long long) * D * D, cudaMemcpyDeviceToHost);
    auto mat = [&](int s, int d) -> long long { return h_mat[(long long)s * D + d]; };

    long long recv_total = 0;
    for (int s = 0; s < D; s++) recv_total += mat(s, me);

    for (int d = 0; d < D; d++) {
        long long cnt = mat(me, d);
        if (cnt <= 0) continue;
        long long doff = 0;
        for (int s = 0; s < me; s++) doff += mat(s, d);
        size_t bytes = (size_t)cnt * sizeof(int);
        if (d == me) {
            cudaMemcpyAsync(dst_key + doff, src_key + soff[d], bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(dst_val + doff, src_val + soff[d], bytes, cudaMemcpyDeviceToDevice, stream);
        } else {
            nvshmem_putmem(dst_key + doff, src_key + soff[d], bytes, d);
            nvshmem_putmem(dst_val + doff, src_val + soff[d], bytes, d);
        }
    }
    cudaStreamSynchronize(stream);
    nvshmem_quiet();
    nvshmem_barrier_all();

    return recv_total;
}

// Gathers R onto its owner GPU and repartitions it. Called once before the chunk loop.
void r_shuffle_gather_r(JoinChunkContext& ctx,
                        int* R_gather_key, int* R_gather_val,
                        int* oa_key, int* oa_val,
                        long long* prefix_R, long long* offset_R,
                        long long tableA_size_me, long long split_size,
                        double& t_r_gather)
{
    auto t0 = std::chrono::steady_clock::now();
    if (ctx.use_bf == 5 || ctx.use_bf == 13) {
        partial_jointable_gpu_partition_r_only(
            ctx.GPUA_split, oa_key, ctx.GPUA_val_split, oa_val,
            prefix_R, ctx.GPU_block_index_split_R, offset_R, ctx.GPU_second_index_split_R,
            tableA_size_me, ctx.gpu_stream);
    }

    long long r_recv = all_to_all_to_owner(
        ctx.GPUA_split, ctx.GPUA_val_split, ctx.GPU_block_index_split_R,
        R_gather_key, R_gather_val, ctx.me, ctx.device_num, ctx.gpu_stream);
    if (r_recv > split_size * 2)
        fprintf(stderr, "[R_SHUFFLE][PE%d] R gather overflow: %lld > %lld (skew). "
                "Increase the GPU count or partition_size.\n", ctx.me, r_recv, split_size * 2);

    partial_jointable_gpu_partition_r_only(
        R_gather_key, oa_key, R_gather_val, oa_val,
        prefix_R, ctx.GPU_block_index_split_R, offset_R, ctx.GPU_second_index_split_R,
        r_recv, ctx.gpu_stream);

    t_r_gather += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("[PE%d][R_SHUFFLE] R gathered+repartitioned: %lld elems (%.4f sec)\n",
           ctx.me, r_recv, t_r_gather);
}

// Gathers into owner buffers in first level bucket order so the first partition pass is folded into the transfer.
static long long all_to_all_bucketed(int* src_key, int* src_val,
                                     long long** all_block_index,
                                     int* dst_key, int* dst_val,
                                     int me, int D, cudaStream_t stream)
{
    const int P = partition_size;
    const int bpg = P / D;

    for (int p = 0; p < D; p++) {
        if (p == me) continue;
        nvshmem_putmem(all_block_index[me], all_block_index[me], sizeof(long long) * P, p);
    }
    nvshmem_quiet();
    nvshmem_barrier_all();

    std::vector<long long> h((long long)D * P);
    for (int s = 0; s < D; s++)
        cudaMemcpy(&h[(long long)s * P], all_block_index[s], sizeof(long long) * P, cudaMemcpyDeviceToHost);
    auto bidx  = [&](int s, int b) -> long long { return h[(long long)s * P + b]; };
    auto ssize = [&](int s, int b) -> long long { return bidx(s, b) - (b == 0 ? 0 : bidx(s, b - 1)); };

    std::vector<long long> h_out(P);
    long long C = 0;
    for (int b = 0; b < P; b++) {
        if (b / bpg == me) {
            long long tot = 0;
            for (int s = 0; s < D; s++) tot += ssize(s, b);
            C += tot;
        }
        h_out[b] = C;
    }
    long long recv_total = C;

    for (int b = 0; b < P; b++) {
        long long cnt = ssize(me, b);
        if (cnt <= 0) continue;
        int g = b / bpg;
        long long base = 0;
        for (int bb = g * bpg; bb < b; bb++)
            for (int s = 0; s < D; s++) base += ssize(s, bb);
        long long pre = 0;
        for (int s = 0; s < me; s++) pre += ssize(s, b);
        long long doff = base + pre;
        long long soff = (b == 0 ? 0 : bidx(me, b - 1));
        size_t bytes = (size_t)cnt * sizeof(int);
        if (g == me) {
            cudaMemcpyAsync(dst_key + doff, src_key + soff, bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(dst_val + doff, src_val + soff, bytes, cudaMemcpyDeviceToDevice, stream);
        } else {
            nvshmem_putmem(dst_key + doff, src_key + soff, bytes, g);
            nvshmem_putmem(dst_val + doff, src_val + soff, bytes, g);
        }
    }
    cudaStreamSynchronize(stream);
    nvshmem_quiet();
    nvshmem_barrier_all();

    cudaMemcpy(all_block_index[me], h_out.data(), sizeof(long long) * P, cudaMemcpyHostToDevice);
    return recv_total;
}

// Gathers the S candidates onto their owner, runs the second pass and probes. Returns the match count of this chunk.
int r_shuffle_join_chunk(JoinChunkContext& ctx,
                         int* R_gather_key, int* R_gather_val,
                         int* effective_s_ptr, int* effective_s_val, long long actual_b_size,
                         long long split_size,
                         double& t_partition, double& t_gather, double& t_probe)
{
    long long* bidx_S = ctx.GPU_block_index_split_S[ctx.me];
    long long* sidx_S = ctx.GPU_second_index_split_S[ctx.me];

    auto t_p0 = std::chrono::steady_clock::now();
    partial_jointable_gpu_partition_s_only(
        effective_s_ptr, ctx.GPU_OB_split, effective_s_val, ctx.GPU_OB_val_split,
        nullptr, ctx.GPU_B_prefix_split, bidx_S, ctx.GPU_B_offset,
        sidx_S, ctx.GPU_answer_num, actual_b_size, ctx.gpu_stream);
    t_partition += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_p0).count();

    auto t_g0 = std::chrono::steady_clock::now();
    long long s_recv = all_to_all_bucketed(
        effective_s_ptr, effective_s_val, ctx.GPU_block_index_split_S,
        ctx.GPU_buffer, ctx.GPU_buffer_val, ctx.me, ctx.device_num, ctx.gpu_stream);
    t_gather += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_g0).count();
    if (s_recv > split_size * 2)
        fprintf(stderr, "[R_SHUFFLE][PE%d] S' gather overflow: %lld > %lld\n",
                ctx.me, s_recv, split_size * 2);

    auto t_p1 = std::chrono::steady_clock::now();
    gpu_run_hist_second_pass<<<partition_size, thread_no, sizeof(int) * second_partition_size * 2, ctx.gpu_stream>>>(
        ctx.GPU_buffer, ctx.GPU_OB_split, bidx_S, sidx_S, second_partition_size);
    gpu_build_table_second_pass<<<partition_size, thread_no, sizeof(int) * second_partition_size, ctx.gpu_stream>>>(
        ctx.GPU_buffer, ctx.GPU_OB_split, ctx.GPU_buffer_val, ctx.GPU_OB_val_split,
        bidx_S, sidx_S, second_partition_size);
    cudaStreamSynchronize(ctx.gpu_stream);
    t_partition += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_p1).count();

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
        R_gather_key, ctx.GPU_OB_split, R_gather_val, ctx.GPU_OB_val_split,
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
    printf("[PE%d][R_SHUFFLE] S'_recv=%lld answers=%d\n", ctx.me, s_recv, ans);
    return ans;
}
