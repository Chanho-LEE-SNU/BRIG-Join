#include "join_executor.h"
#include "bf_allreduce.h"
#include "vlsplit_hash_kernel.h"
#include "kernel_functions.h"
#include "common.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

// Runs the local probe, the inter GPU transfer and the cross GPU probe for one S chunk.
void join_execute_host(JoinChunkContext& ctx,
                       int* effective_s_ptr, int* effective_s_val, long long actual_b_size)
{
    if(ctx.kernel_fusion==0)
    {
        {
            auto t0 = std::chrono::steady_clock::now();
            { int zero = 0; cudaMemcpyToSymbol(d_probe_overflow_count, &zero, sizeof(int)); }
            size_t kv_probe_shm = sizeof(int) * (size_t)ctx.max_int_number_for_shared_memory * 2;
            { static bool set0 = false; if (!set0) {
                cudaFuncSetAttribute(gpu_probe_with_second_pass<false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kv_probe_shm);
                cudaFuncSetAttribute(gpu_probe_with_second_pass<true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kv_probe_shm); set0 = true; } }
            if (ctx.fused_probe_breakdown) {
                gpu_probe_with_second_pass<true><<<
                    partition_size * second_partition_size, thread_no, kv_probe_shm, ctx.gpu_stream>>>(
                        (int*)ctx.GPUA_split, (int*)effective_s_ptr,
                        (int*)ctx.GPUA_val_split, (int*)effective_s_val,
                        ctx.GPU_block_index_split_R, ctx.GPU_block_index_split_S[ctx.me],
                        ctx.GPU_second_index_split_R, ctx.GPU_second_index_split_S[ctx.me],
                        partition_size, second_partition_size,
                        ctx.GPU_answer, ctx.GPU_answer_val, ctx.GPU_answer_s_val, ctx.GPU_answer_num, ctx.GPU_answer_write_idx,
                        ctx.d_phase_timers);
            } else {
                gpu_probe_with_second_pass<false><<<
                    partition_size * second_partition_size, thread_no, kv_probe_shm, ctx.gpu_stream>>>(
                        (int*)ctx.GPUA_split, (int*)effective_s_ptr,
                        (int*)ctx.GPUA_val_split, (int*)effective_s_val,
                        ctx.GPU_block_index_split_R, ctx.GPU_block_index_split_S[ctx.me],
                        ctx.GPU_second_index_split_R, ctx.GPU_second_index_split_S[ctx.me],
                        partition_size, second_partition_size,
                        ctx.GPU_answer, ctx.GPU_answer_val, ctx.GPU_answer_s_val, ctx.GPU_answer_num, ctx.GPU_answer_write_idx,
                        /*d_phase_timers=*/nullptr);
            }
            cudaDeviceSynchronize();
            double kf0_local_kernel_sec = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            *ctx.first_pass_probe_sec += kf0_local_kernel_sec;
            { int h = 0; cudaMemcpyFromSymbol(&h, d_probe_overflow_count, sizeof(int));
            *ctx.total_overflow_count += h; }

            if (ctx.fused_probe_breakdown) {
                const int nb = partition_size * second_partition_size;
                std::vector<long long> h_t(nb * 7);
                cudaMemcpy(h_t.data(), ctx.d_phase_timers,
                           sizeof(long long) * nb * 7, cudaMemcpyDeviceToHost);
                double sum_pre_cyc = 0.0, sum_mem_cyc = 0.0, sum_ibar_cyc = 0.0,
                       sum_build_cyc = 0.0, sum_probe_cyc = 0.0, sum_reduce_cyc = 0.0;
                int valid = 0;
                for (int b = 0; b < nb; b++) {
                    long long s0 = h_t[(long long)b * 7 + 0];
                    long long s1 = h_t[(long long)b * 7 + 1];
                    long long s2 = h_t[(long long)b * 7 + 2];
                    long long s3 = h_t[(long long)b * 7 + 3];
                    long long s4 = h_t[(long long)b * 7 + 4];
                    long long s5 = h_t[(long long)b * 7 + 5];
                    long long s6 = h_t[(long long)b * 7 + 6];
                    if (s6 <= s0) continue;
                    sum_pre_cyc    += (double)(s1 - s0);
                    sum_mem_cyc    += (double)(s2 - s1);
                    sum_ibar_cyc   += (double)(s3 - s2);
                    sum_build_cyc  += (double)(s4 - s3);
                    sum_probe_cyc  += (double)(s5 - s4);
                    sum_reduce_cyc += (double)(s6 - s5);
                    valid++;
                }
                if (valid > 0) {
                    double avg_pre    = sum_pre_cyc    / valid;
                    double avg_mem    = sum_mem_cyc    / valid;
                    double avg_ibar   = sum_ibar_cyc   / valid;
                    double avg_build  = sum_build_cyc  / valid;
                    double avg_probe  = sum_probe_cyc  / valid;
                    double avg_reduce = sum_reduce_cyc / valid;
                    double avg_total  = avg_pre + avg_mem + avg_ibar + avg_build + avg_probe + avg_reduce;
                    if (avg_total > 0.0) {
                        *ctx.cumulative_fp_preamble_sec += (avg_pre    / avg_total) * kf0_local_kernel_sec;
                        *ctx.cumulative_fp_memset_sec   += (avg_mem    / avg_total) * kf0_local_kernel_sec;
                        *ctx.cumulative_fp_initbar_sec  += (avg_ibar   / avg_total) * kf0_local_kernel_sec;
                        *ctx.cumulative_fp_build_sec    += (avg_build  / avg_total) * kf0_local_kernel_sec;
                        *ctx.cumulative_fp_local_sec    += (avg_probe  / avg_total) * kf0_local_kernel_sec;
                        *ctx.cumulative_fp_reduce_sec   += (avg_reduce / avg_total) * kf0_local_kernel_sec;
                    }
                }
            }
            {
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                    fprintf(stderr, "[PE%d] CUDA error after probe (first pass): %s\n",
                            ctx.me, cudaGetErrorString(err));
                //else
                //    fprintf(stderr, "[PE%d] No error after probe (first pass)\n", ctx.me);
                fflush(stderr);
            }
        }

        printf("[PE%d][DBG] overflow blocks=%d\n", ctx.me, *ctx.total_overflow_count);

        long long filtered_size = actual_b_size;
        int* send_data = effective_s_ptr;
        int* send_data_val = effective_s_val;

        if (ctx.use_bf > 0) {
            if (ctx.use_bf == 3) {
                auto t_exc0 = std::chrono::steady_clock::now();
                for (int dst_pe = 0; dst_pe < ctx.device_num; dst_pe++) {
                    if (dst_pe == ctx.me) continue;
                    nvshmem_putmem((*ctx.bf)[ctx.me].getDeviceFilter(),
                                (*ctx.bf)[ctx.me].getDeviceFilter(),
                                sizeof(uint32_t) * ctx.bf_words, dst_pe);
                }
                nvshmem_quiet();
                nvshmem_barrier_all();
                cudaMemcpy(ctx.d_bf_global, (*ctx.bf)[ctx.me].getDeviceFilter(),
                        sizeof(uint32_t) * ctx.bf_words, cudaMemcpyDeviceToDevice);
                for (int p = 0; p < ctx.device_num; p++) {
                    if (p == ctx.me) continue;
                    bf_or_kernel<<<((size_t)ctx.bf_words + 255) / 256, 256>>>(
                        ctx.d_bf_global, (*ctx.bf)[p].getDeviceFilter(), (size_t)ctx.bf_words);
                    cudaDeviceSynchronize();
                }
                nvshmem_barrier_all();
                double this_sec = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_exc0).count();
                *ctx.cumulative_bf_exchange_sec += this_sec;
                printf("[PE%d] Late BF all-reduce (%d PEs, push): %.4f sec\n",
                    ctx.me, ctx.device_num, this_sec);
            }

            if (ctx.use_bf == 1 || ctx.use_bf == 3) {
                {
                    long long* d_hist_check;
                    cudaMalloc(&d_hist_check, sizeof(long long) * partition_size);
                    cudaMemset(d_hist_check, 0, sizeof(long long) * partition_size);
                    long long blk = actual_b_size / block_no;
                    if (actual_b_size % block_no != 0) blk++;
                    queryAndHistOnlyKernel<<<block_no, thread_no, sizeof(int)*partition_size>>>(
                        ctx.d_bf_global, effective_s_ptr, actual_b_size,
                        (*ctx.bf)[ctx.me].getNumHashes(), (*ctx.bf)[ctx.me].getNumBits(),
                        d_hist_check, partition_size, blk);
                    cudaDeviceSynchronize();
                    long long* h_hist = new long long[partition_size];
                    cudaMemcpy(h_hist, d_hist_check,
                            sizeof(long long) * partition_size, cudaMemcpyDeviceToHost);
                    long long total_passing = 0;
                    for (int i = 0; i < partition_size; i++) total_passing += h_hist[i];
                    printf("[PE%d] global BF check on pre-filtered(%lld): %lld pass (%.1f%%)\n",
                        ctx.me, actual_b_size, total_passing,
                        100.0 * total_passing / (actual_b_size > 0 ? actual_b_size : 1));
                    delete[] h_hist;
                    cudaFree(d_hist_check);
                }

                auto t_pipeline_start = std::chrono::steady_clock::now();
                {
                    long long blk = actual_b_size / block_no;
                    if (actual_b_size % block_no != 0) blk++;

                    auto t0 = std::chrono::steady_clock::now();
                    cudaMemset(ctx.GPU_B_prefix_split, 0, sizeof(long long) * partition_size);
                    queryAndHistOnlyKernel<<<block_no, thread_no, sizeof(int)*partition_size>>>(
                        ctx.d_bf_global, effective_s_ptr, actual_b_size,
                        (*ctx.bf)[ctx.me].getNumHashes(), (*ctx.bf)[ctx.me].getNumBits(),
                        ctx.GPU_B_prefix_split, partition_size, blk);
                    cudaDeviceSynchronize();
                    auto t1 = std::chrono::steady_clock::now();
                    printf("[PE%d][BF] histogram:    %.4f sec\n",
                        ctx.me, std::chrono::duration<double>(t1-t0).count());

                    gpu_make_simple_prefix<<<1, prefix_scan_block_dim(partition_size, thread_no),
                        sizeof(long long)*partition_size*2>>>(
                            effective_s_ptr, ctx.GPU_B_prefix_split,
                            ctx.GPU_block_index_split_S[ctx.me],
                            actual_b_size, partition_size, ctx.GPU_B_offset);
                    cudaDeviceSynchronize();
                    auto t2 = std::chrono::steady_clock::now();
                    printf("[PE%d][BF] prefix sum:   %.4f sec\n",
                        ctx.me, std::chrono::duration<double>(t2-t1).count());

                    cudaMemcpy(&filtered_size,
                            &ctx.GPU_block_index_split_S[ctx.me][partition_size-1],
                            sizeof(long long), cudaMemcpyDeviceToHost);
                    printf("[PE%d] Global-BF filtered: %lld / %lld (%.1f%%)\n",
                        ctx.me, filtered_size, actual_b_size,
                        100.0 * filtered_size / (actual_b_size > 0 ? actual_b_size : 1));

                    queryAndScatterKernel<<<block_no, thread_no, 0>>>(
                        ctx.d_bf_global, effective_s_ptr, effective_s_val, actual_b_size,
                        (*ctx.bf)[ctx.me].getNumHashes(), (*ctx.bf)[ctx.me].getNumBits(),
                        ctx.GPUB_send_buffer, ctx.GPUB_send_buffer_val, ctx.GPU_B_offset, partition_size, blk);
                    cudaDeviceSynchronize();
                    auto t3 = std::chrono::steady_clock::now();
                    printf("[PE%d][BF] scatter:      %.4f sec\n",
                        ctx.me, std::chrono::duration<double>(t3-t2).count());

                    gpu_run_hist_second_pass<<<partition_size, thread_no, sizeof(int)*second_partition_size*2>>>(
                        ctx.GPUB_send_buffer, ctx.GPU_OB_split,
                        ctx.GPU_block_index_split_S[ctx.me],
                        ctx.GPU_second_index_split_S[ctx.me], second_partition_size);
                    gpu_build_table_second_pass<<<partition_size, thread_no,
                        sizeof(int)*second_partition_size>>>(
                            ctx.GPUB_send_buffer, ctx.GPU_OB_split,
                            ctx.GPUB_send_buffer_val, ctx.GPU_OB_val_split,
                            ctx.GPU_block_index_split_S[ctx.me],
                            ctx.GPU_second_index_split_S[ctx.me], second_partition_size);
                    cudaDeviceSynchronize();
                    auto t4 = std::chrono::steady_clock::now();
                    printf("[PE%d][BF] 2nd pass sort:%.4f sec\n",
                        ctx.me, std::chrono::duration<double>(t4-t3).count());
                }
                double this_pipeline_sec = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_pipeline_start).count();
                *ctx.cumulative_bf_pipeline_sec += this_pipeline_sec;
                printf("[PE%d][BF] pipeline total: %.4f sec\n", ctx.me, this_pipeline_sec);

                send_data = ctx.GPU_OB_split;
                send_data_val = ctx.GPU_OB_val_split;
            } else {
                filtered_size = actual_b_size;
                send_data = effective_s_ptr;
                send_data_val = effective_s_val;
            }
        } else {
            nvshmem_barrier_all();
            nvshmem_barrier_all();
        }

        auto t_pre_barrier = std::chrono::steady_clock::now();
        nvshmem_barrier_all();
        auto t_after_barrier = std::chrono::steady_clock::now();
        {
            double this_sec = std::chrono::duration<double>(
                t_after_barrier - t_pre_barrier).count();
            *ctx.cumulative_pre_send_barrier_sec += this_sec;
            printf("[PE%d] pre-send barrier wait: %.4f sec\n", ctx.me, this_sec);
        }

        for (int loop = 1; loop < ctx.device_num; loop++) {
            int sending_to = (ctx.me + loop) % ctx.device_num;
            size_t send_bytes = sizeof(int) * filtered_size;

            cudaEvent_t send_ev0, send_ev1;
            cudaEventCreate(&send_ev0);
            cudaEventCreate(&send_ev1);

            auto t_inner_barrier = std::chrono::steady_clock::now();
            nvshmem_barrier_all();
            auto t_inner_barrier_end = std::chrono::steady_clock::now();
            {
                double this_sec = std::chrono::duration<double>(
                    t_inner_barrier_end - t_inner_barrier).count();
                *ctx.cumulative_inner_barrier_sec += this_sec;
                printf("[PE%d] inner barrier wait:  %.4f sec\n", ctx.me, this_sec);
            }

            cudaEventRecord(send_ev0, ctx.send_stream);
            if (ctx.use_nvlink) {
                nvshmem_putmem(ctx.GPU_buffer, send_data,
                            sizeof(int) * filtered_size, sending_to);
                nvshmem_putmem(ctx.GPU_buffer_val, send_data_val,
                            sizeof(int) * filtered_size, sending_to);
            } else {
                cudaDeviceSynchronize();
                size_t sz_data = sizeof(int) * filtered_size;
                cudaMemcpy(ctx.h_pcie_stage, send_data, sz_data, cudaMemcpyDeviceToHost);
                cudaMemcpy(ctx.h_pcie_stage_val, send_data_val, sz_data, cudaMemcpyDeviceToHost);
                int* remote_buf = (int*)nvshmem_ptr(ctx.GPU_buffer, sending_to);
                int* remote_buf_val = (int*)nvshmem_ptr(ctx.GPU_buffer_val, sending_to);
                if (!remote_buf) {
                    printf("[PE%d] WARNING: nvshmem_ptr returned NULL, falling back to nvshmem_putmem\n",
                        ctx.me);
                    nvshmem_putmem(ctx.GPU_buffer, send_data, sz_data, sending_to);
                    nvshmem_putmem(ctx.GPU_buffer_val, send_data_val, sz_data, sending_to);
                } else {
                    cudaMemcpy(remote_buf, ctx.h_pcie_stage, sz_data, cudaMemcpyHostToDevice);
                    cudaMemcpy(remote_buf_val, ctx.h_pcie_stage_val, sz_data, cudaMemcpyHostToDevice);
                }
            }
            cudaEventRecord(send_ev1, ctx.send_stream);
            cudaStreamSynchronize(ctx.send_stream);

            float send_ms;
            cudaEventElapsedTime(&send_ms, send_ev0, send_ev1);
            printf("[SEND/%s] PE%d->PE%d  %.2f MB  %.2f GB/s (%.3f ms)\n",
                ctx.use_nvlink ? "NVLink" : "PCIe",
                ctx.me, sending_to,
                send_bytes / 1e6,
                send_bytes / (send_ms / 1000.0) / 1e9, send_ms);
            cudaEventDestroy(send_ev0);
            cudaEventDestroy(send_ev1);
            cudaDeviceSynchronize();

            auto t_delivery_barrier = std::chrono::steady_clock::now();
            nvshmem_barrier_all();
            auto t_after_delivery = std::chrono::steady_clock::now();
            printf("[PE%d] delivery barrier wait: %.4f sec\n",
                ctx.me, std::chrono::duration<double>(
                    t_after_delivery - t_delivery_barrier).count());

            double iter_transfer_sec = std::chrono::duration<double>(
                t_after_delivery - t_inner_barrier_end).count();
            *ctx.cumulative_transfer_sec += iter_transfer_sec;
            printf("[PE%d] iter%d transfer: %.4f sec  (cumulative: %.4f sec)\n",
                ctx.me, loop, iter_transfer_sec, *ctx.cumulative_transfer_sec);

            { int zero = 0; cudaMemcpyToSymbol(d_probe_overflow_count, &zero, sizeof(int)); }
            int source_pe = (ctx.me - loop + ctx.device_num) % ctx.device_num;
            partial_jointable_gpu_run_only_s(
                ctx.GPUA_split, ctx.GPU_buffer,
                ctx.GPUA_val_split, ctx.GPU_buffer_val,
                ctx.GPU_block_index_split_R, ctx.GPU_block_index_split_S[source_pe],
                ctx.GPU_second_index_split_R, ctx.GPU_second_index_split_S[source_pe],
                ctx.GPU_answer, ctx.GPU_answer_val, ctx.GPU_answer_s_val, ctx.GPU_answer_num,
                ctx.max_int_number_for_shared_memory,
                ctx.answer_num_host, ctx.gpu_stream, ctx.gpu_event,
                ctx.GPU_answer_write_idx);
            { int h = 0; cudaMemcpyFromSymbol(&h, d_probe_overflow_count, sizeof(int));
            *ctx.total_overflow_count += h; }
            {
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                    fprintf(stderr, "[PE%d] CUDA error after inter-GPU join (loop %d): %s\n",
                            ctx.me, loop, cudaGetErrorString(err));
                //else
                //    fprintf(stderr, "[PE%d] No error after inter-GPU join (loop %d)\n",
                //            ctx.me, loop);
                fflush(stderr);
            }

            double iter_second_pass_sec = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_after_delivery).count();
            *ctx.cumulative_second_pass_sec += iter_second_pass_sec;
            printf("[PE%d] iter%d 2nd-pass join: %.4f sec  (cumulative: %.4f sec)\n",
                ctx.me, loop, iter_second_pass_sec, *ctx.cumulative_second_pass_sec);
        }

        // answer_num_host holds a running total, so it is read once after the loop.
        cudaMemcpy(ctx.answer_num_host, ctx.GPU_answer_num,
                   sizeof(int), cudaMemcpyDeviceToHost);
        *ctx.total_answer_num += ctx.answer_num_host[0];
    }
    else if ((ctx.use_bf == 0 || ctx.use_bf == 5 || ctx.use_bf == 7 || ctx.use_bf == 9 || ctx.use_bf == 10 || ctx.use_bf == 13) && ctx.kernel_fusion == 1)
    {
        auto t_fused0 = std::chrono::steady_clock::now();

        { int zero = 0; cudaMemcpyToSymbol(d_probe_overflow_count, &zero, sizeof(int)); }

        // Every PE must have filled its send buffer before any remote GET is issued.
        nvshmem_barrier_all();

        const int ht_size = HT_SIZE;
        const size_t shm  = sizeof(int) * (2 * ht_size + 2 * chunk_size);
        { static bool setf1 = false; if (!setf1) {
            cudaFuncSetAttribute(gpu_fused_probe_transfer<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            cudaFuncSetAttribute(gpu_fused_probe_transfer<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); setf1 = true; } }

        const int num_blocks = partition_size * second_partition_size;
        const long long total_slots = ctx.d_phase_timers_slots;
        long long* d_phase_timers   = ctx.d_phase_timers;
        if (ctx.fused_probe_breakdown) {
            cudaMemset(d_phase_timers, 0, sizeof(long long) * total_slots);
            // The min_get slots start at ULL_MAX so that atomicMin works.
            cudaMemset(d_phase_timers + num_blocks * 5 + 16 + 16, 0xFF,
                       sizeof(long long) * 8);
        }

        auto t_kernel_start = std::chrono::steady_clock::now();
        if (ctx.fused_probe_breakdown) {
            gpu_fused_probe_transfer<true><<<
                partition_size * second_partition_size, thread_no, shm, ctx.gpu_stream>>>(
                    ctx.GPUA_split, ctx.GPUA_val_split, effective_s_ptr, effective_s_val,
                    ctx.GPUB_send_buffer_kv,
                    ctx.GPU_block_index_split_R, ctx.GPU_second_index_split_R,
                    ctx.d_s_block_ptrs, ctx.d_s_second_ptrs,
                    ctx.me, ctx.device_num, partition_size, second_partition_size,
                    ctx.GPU_answer, ctx.GPU_answer_val, ctx.GPU_answer_s_val, ctx.GPU_answer_num, ctx.GPU_answer_write_idx,
                    ht_size, chunk_size, d_phase_timers, -1);
        } else {
            gpu_fused_probe_transfer<false><<<
                partition_size * second_partition_size, thread_no, shm, ctx.gpu_stream>>>(
                    ctx.GPUA_split, ctx.GPUA_val_split, effective_s_ptr, effective_s_val,
                    ctx.GPUB_send_buffer_kv,
                    ctx.GPU_block_index_split_R, ctx.GPU_second_index_split_R,
                    ctx.d_s_block_ptrs, ctx.d_s_second_ptrs,
                    ctx.me, ctx.device_num, partition_size, second_partition_size,
                    ctx.GPU_answer, ctx.GPU_answer_val, ctx.GPU_answer_s_val, ctx.GPU_answer_num, ctx.GPU_answer_write_idx,
                    ht_size, chunk_size, /*d_phase_timers=*/nullptr, -1);
        }

        cudaDeviceSynchronize();
        double kernel_wall_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_kernel_start).count();

        { int h = 0; cudaMemcpyFromSymbol(&h, d_probe_overflow_count, sizeof(int));
          *ctx.total_overflow_count += h; }

        cudaMemcpy(ctx.answer_num_host, ctx.GPU_answer_num,
                   sizeof(int), cudaMemcpyDeviceToHost);
        *ctx.total_answer_num += ctx.answer_num_host[0];

        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "[PE%d] CUDA error in fused probe kernel: %s\n",
                        ctx.me, cudaGetErrorString(err));
            //else
            //    fprintf(stderr, "[PE%d] Fused probe kernel OK\n", ctx.me);
            fflush(stderr);
        }
        printf("[PE%d][fused] local+cross-GPU probe done: %d answers, overflow=%d\n",
               ctx.me, ctx.answer_num_host[0], *ctx.total_overflow_count);

        if (!ctx.fused_probe_breakdown) {
            *ctx.cumulative_fused_wall_sec += kernel_wall_sec;
        } else {
            std::vector<long long> h_timers(total_slots);
            cudaMemcpy(h_timers.data(), d_phase_timers,
                       sizeof(long long) * total_slots, cudaMemcpyDeviceToHost);

            long long* per_iter_get   = &h_timers[num_blocks * 5];
            long long* per_iter_probe = &h_timers[num_blocks * 5 + 8];
            int num_iter = ctx.device_num - 1;
            long long blk0_total_cyc = h_timers[0*5 + 3] - h_timers[0*5 + 0];
            double inv_blk0 = (blk0_total_cyc > 0) ? (kernel_wall_sec / (double)blk0_total_cyc) : 0.0;
            printf("[PE%d][fused] per-iter breakdown (block 0, %d remote PE iters):\n",
                   ctx.me, num_iter);
            for (int i = 0; i < num_iter && i < 8; i++) {
                printf("[PE%d][fused]   iter %d: get=%.4f ms  probe=%.4f ms  (cycles: get=%lld probe=%lld)\n",
                       ctx.me, i,
                       per_iter_get[i]   * inv_blk0 * 1000.0,
                       per_iter_probe[i] * inv_blk0 * 1000.0,
                       per_iter_get[i], per_iter_probe[i]);
            }

            // Aggregate across all blocks. These are per block averages, so they cannot be
            // compared with kernel_wall_sec directly and only show whether block 0 is an outlier.
            const long long agg_base = (long long)num_blocks * 5 + 16;
            long long* agg_sum_get   = &h_timers[agg_base + 0];
            long long* agg_sum_probe = &h_timers[agg_base + 8];
            unsigned long long* agg_min_get = (unsigned long long*)&h_timers[agg_base + 16];
            unsigned long long* agg_max_get = (unsigned long long*)&h_timers[agg_base + 24];
            printf("[PE%d][fused] per-iter aggregate across %d blocks (get cycles):\n",
                   ctx.me, num_blocks);
            for (int i = 0; i < num_iter && i < 8; i++) {
                double avg_get_ms   = (double)agg_sum_get[i]   / num_blocks * inv_blk0 * 1000.0;
                double avg_probe_ms = (double)agg_sum_probe[i] / num_blocks * inv_blk0 * 1000.0;
                double min_get_ms   = (double)agg_min_get[i]                * inv_blk0 * 1000.0;
                double max_get_ms   = (double)agg_max_get[i]                * inv_blk0 * 1000.0;
                printf("[PE%d][fused]   iter %d: get avg=%.4f ms  min=%.4f ms  max=%.4f ms  "
                       "(probe avg=%.4f ms)  (raw cyc: sum_get=%lld min=%llu max=%llu)\n",
                       ctx.me, i,
                       avg_get_ms, min_get_ms, max_get_ms, avg_probe_ms,
                       agg_sum_get[i], (unsigned long long)agg_min_get[i],
                       (unsigned long long)agg_max_get[i]);
            }

            double sum_ht = 0.0, sum_local = 0.0, sum_remote = 0.0;
            double sum_get = 0.0;
            int valid = 0;
            for (int b = 0; b < num_blocks; b++) {
                long long t0 = h_timers[(long long)b*5 + 0];
                long long t1 = h_timers[(long long)b*5 + 1];
                long long t2 = h_timers[(long long)b*5 + 2];
                long long t3 = h_timers[(long long)b*5 + 3];
                long long cg = h_timers[(long long)b*5 + 4];
                if (t3 <= t0) continue;
                sum_ht     += (double)(t1 - t0);
                sum_local  += (double)(t2 - t1);
                sum_remote += (double)(t3 - t2);
                sum_get    += (double)cg;
                valid++;
            }

            if (valid > 0) {
                double avg_ht           = sum_ht     / valid;
                double avg_local        = sum_local  / valid;
                double avg_remote       = sum_remote / valid;
                double avg_get          = sum_get    / valid;
                double avg_probe_remote = avg_remote - avg_get;
                double avg_total        = avg_ht + avg_local + avg_remote;

                // Per block cycle fractions times the wall time, so the parts sum to kernel_wall_sec.
                if (avg_total > 0.0) {
                    *ctx.cumulative_fused_ht_sec           += (avg_ht           / avg_total) * kernel_wall_sec;
                    *ctx.cumulative_fused_local_sec        += (avg_local        / avg_total) * kernel_wall_sec;
                    *ctx.cumulative_fused_get_sec          += (avg_get          / avg_total) * kernel_wall_sec;
                    *ctx.cumulative_fused_remote_probe_sec += (avg_probe_remote / avg_total) * kernel_wall_sec;
                }
                *ctx.cumulative_fused_wall_sec += kernel_wall_sec;
            }

            // d_phase_timers is owned by ctx and must not be freed here.
        }
        // Setup and teardown is the branch wall time minus the kernel wall time.
        {
            double phase1_total_sec_f1 = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_fused0).count();
            *ctx.cumulative_fused_setup_sec += (phase1_total_sec_f1 - kernel_wall_sec);
        }
    }
    else if ((ctx.use_bf == 5 || ctx.use_bf == 7 || ctx.use_bf == 9 || ctx.use_bf == 13) && ctx.kernel_fusion == 2)
    {
        // Hybrid path: the fused kernel handles the local probe and one NVLink peer, while the
        // remaining peers go through the PUT loop. Keeping the in-kernel remote peer count at one
        // avoids the multi peer access pattern of kernel_fusion==1.
        // The NVLink partner is hardcoded as me^1, which assumes a (0,1)(2,3) topology.
        const int nvl_peer = ctx.me ^ 1;
        const bool has_nvl_peer = (nvl_peer < ctx.device_num);

        auto t_fused0 = std::chrono::steady_clock::now();

        { int zero = 0; cudaMemcpyToSymbol(d_probe_overflow_count, &zero, sizeof(int)); }

        // The partition step emits separate key and value arrays here because the PUT loop needs
        // them, so the interleaved send buffer is built explicitly.
        {
            int thr = 256;
            long long blk = (actual_b_size + thr - 1) / thr;
            if (blk < 1) blk = 1;
            if (blk > 65535) blk = 65535;
            gpu_interleave_kv<<<(int)blk, thr, 0, ctx.gpu_stream>>>(
                effective_s_ptr, effective_s_val, ctx.GPUB_send_buffer_kv, actual_b_size);
            cudaStreamSynchronize(ctx.gpu_stream);
        }
        nvshmem_barrier_all();

        const int ht_size = HT_SIZE;
        const size_t shm  = sizeof(int) * (2 * ht_size + 2 * chunk_size);
        { static bool setf2 = false; if (!setf2) {
            cudaFuncSetAttribute(gpu_fused_probe_transfer<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            cudaFuncSetAttribute(gpu_fused_probe_transfer<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); setf2 = true; } }
        const int num_blocks = partition_size * second_partition_size;
        const long long total_slots = ctx.d_phase_timers_slots;
        long long* d_phase_timers   = ctx.d_phase_timers;
        if (ctx.fused_probe_breakdown) {
            cudaMemset(d_phase_timers, 0, sizeof(long long) * total_slots);
            cudaMemset(d_phase_timers + num_blocks * 5 + 16 + 16, 0xFF,
                       sizeof(long long) * 8);
        }

        auto t_kernel_start = std::chrono::steady_clock::now();
        const int f2_peer = has_nvl_peer ? nvl_peer : -1;
        if (ctx.fused_probe_breakdown) {
            gpu_fused_probe_transfer<true><<<
                partition_size * second_partition_size, thread_no, shm, ctx.gpu_stream>>>(
                    ctx.GPUA_split, ctx.GPUA_val_split, effective_s_ptr, effective_s_val,
                    ctx.GPUB_send_buffer_kv,
                    ctx.GPU_block_index_split_R, ctx.GPU_second_index_split_R,
                    ctx.d_s_block_ptrs, ctx.d_s_second_ptrs,
                    ctx.me, ctx.device_num, partition_size, second_partition_size,
                    ctx.GPU_answer, ctx.GPU_answer_val, ctx.GPU_answer_s_val, ctx.GPU_answer_num, ctx.GPU_answer_write_idx,
                    ht_size, chunk_size, d_phase_timers, f2_peer);
        } else {
            gpu_fused_probe_transfer<false><<<
                partition_size * second_partition_size, thread_no, shm, ctx.gpu_stream>>>(
                    ctx.GPUA_split, ctx.GPUA_val_split, effective_s_ptr, effective_s_val,
                    ctx.GPUB_send_buffer_kv,
                    ctx.GPU_block_index_split_R, ctx.GPU_second_index_split_R,
                    ctx.d_s_block_ptrs, ctx.d_s_second_ptrs,
                    ctx.me, ctx.device_num, partition_size, second_partition_size,
                    ctx.GPU_answer, ctx.GPU_answer_val, ctx.GPU_answer_s_val, ctx.GPU_answer_num, ctx.GPU_answer_write_idx,
                    ht_size, chunk_size, /*d_phase_timers=*/nullptr, f2_peer);
        }
        cudaDeviceSynchronize();
        double kernel_wall_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_kernel_start).count();

        { int h = 0; cudaMemcpyFromSymbol(&h, d_probe_overflow_count, sizeof(int));
          *ctx.total_overflow_count += h; }

        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "[PE%d] CUDA error in fused (fusion=2): %s\n",
                        ctx.me, cudaGetErrorString(err));
            fflush(stderr);
        }
        // Snapshot of the answers produced so far, used to isolate the PUT loop contribution.
        int fused_answer_count = 0;
        cudaMemcpy(&fused_answer_count, ctx.GPU_answer_num,
                   sizeof(int), cudaMemcpyDeviceToHost);
        printf("[PE%d][fusion=2][P1-fused] local+NVL(peer=%d) kernel wall = %.4f sec, answers so far = %d\n",
               ctx.me, has_nvl_peer ? nvl_peer : -1, kernel_wall_sec, fused_answer_count);

        if (!ctx.fused_probe_breakdown) {
            *ctx.cumulative_fused_wall_sec += kernel_wall_sec;
        } else {
            std::vector<long long> h_timers(total_slots);
            cudaMemcpy(h_timers.data(), d_phase_timers,
                       sizeof(long long) * total_slots, cudaMemcpyDeviceToHost);

            long long* per_iter_get   = &h_timers[num_blocks * 5];
            long long* per_iter_probe = &h_timers[num_blocks * 5 + 8];
            int num_iter = has_nvl_peer ? 1 : (ctx.device_num - 1);
            long long blk0_total_cyc = h_timers[0*5 + 3] - h_timers[0*5 + 0];
            double inv_blk0 = (blk0_total_cyc > 0) ? (kernel_wall_sec / (double)blk0_total_cyc) : 0.0;
            printf("[PE%d][fusion=2][P1-fused] per-iter breakdown (block 0, %d remote iters):\n",
                   ctx.me, num_iter);
            for (int i = 0; i < num_iter && i < 8; i++) {
                printf("[PE%d][fusion=2][P1-fused]   iter %d: get=%.4f ms  probe=%.4f ms  (cycles: get=%lld probe=%lld)\n",
                       ctx.me, i,
                       per_iter_get[i]   * inv_blk0 * 1000.0,
                       per_iter_probe[i] * inv_blk0 * 1000.0,
                       per_iter_get[i], per_iter_probe[i]);
            }

            const long long agg_base = (long long)num_blocks * 5 + 16;
            long long* agg_sum_get   = &h_timers[agg_base + 0];
            long long* agg_sum_probe = &h_timers[agg_base + 8];
            unsigned long long* agg_min_get = (unsigned long long*)&h_timers[agg_base + 16];
            unsigned long long* agg_max_get = (unsigned long long*)&h_timers[agg_base + 24];
            printf("[PE%d][fusion=2][P1-fused] per-iter aggregate across %d blocks (get cycles):\n",
                   ctx.me, num_blocks);
            for (int i = 0; i < num_iter && i < 8; i++) {
                double avg_get_ms   = (double)agg_sum_get[i]   / num_blocks * inv_blk0 * 1000.0;
                double avg_probe_ms = (double)agg_sum_probe[i] / num_blocks * inv_blk0 * 1000.0;
                double min_get_ms   = (double)agg_min_get[i]                * inv_blk0 * 1000.0;
                double max_get_ms   = (double)agg_max_get[i]                * inv_blk0 * 1000.0;
                printf("[PE%d][fusion=2][P1-fused]   iter %d: get avg=%.4f ms  min=%.4f ms  max=%.4f ms  "
                       "(probe avg=%.4f ms)\n",
                       ctx.me, i, avg_get_ms, min_get_ms, max_get_ms, avg_probe_ms);
            }

            double sum_ht = 0.0, sum_local = 0.0, sum_remote = 0.0;
            double sum_get = 0.0;
            int valid = 0;
            for (int b = 0; b < num_blocks; b++) {
                long long t0 = h_timers[(long long)b*5 + 0];
                long long t1 = h_timers[(long long)b*5 + 1];
                long long t2 = h_timers[(long long)b*5 + 2];
                long long t3 = h_timers[(long long)b*5 + 3];
                long long cg = h_timers[(long long)b*5 + 4];
                if (t3 <= t0) continue;
                sum_ht     += (double)(t1 - t0);
                sum_local  += (double)(t2 - t1);
                sum_remote += (double)(t3 - t2);
                sum_get    += (double)cg;
                valid++;
            }

            if (valid > 0) {
                double avg_ht           = sum_ht     / valid;
                double avg_local        = sum_local  / valid;
                double avg_remote       = sum_remote / valid;
                double avg_get          = sum_get    / valid;
                double avg_probe_remote = avg_remote - avg_get;
                double avg_total        = avg_ht + avg_local + avg_remote;

                if (avg_total > 0.0) {
                    *ctx.cumulative_fused_ht_sec           += (avg_ht           / avg_total) * kernel_wall_sec;
                    *ctx.cumulative_fused_local_sec        += (avg_local        / avg_total) * kernel_wall_sec;
                    *ctx.cumulative_fused_get_sec          += (avg_get          / avg_total) * kernel_wall_sec;
                    *ctx.cumulative_fused_remote_probe_sec += (avg_probe_remote / avg_total) * kernel_wall_sec;
                }
                *ctx.cumulative_fused_wall_sec += kernel_wall_sec;
            }

            // d_phase_timers is owned by ctx and must not be freed here.
        }

        {
            double phase1_total_sec_f2 = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_fused0).count();
            *ctx.cumulative_fused_setup_sec += (phase1_total_sec_f2 - kernel_wall_sec);
        }

        auto t_phase2_start = std::chrono::steady_clock::now();

        auto t_pre_barrier = std::chrono::steady_clock::now();
        nvshmem_barrier_all();
        auto t_after_barrier = std::chrono::steady_clock::now();
        *ctx.cumulative_pre_send_barrier_sec += std::chrono::duration<double>(
            t_after_barrier - t_pre_barrier).count();

        int* send_data = effective_s_ptr;
        int* send_data_val = effective_s_val;
        long long filtered_size = actual_b_size;

        double p2_inner_barrier_sec  = 0.0;
        double p2_transfer_sec       = 0.0;
        double p2_delivery_bar_sec   = 0.0;
        double p2_probe_kernel_sec   = 0.0;
        int    p2_pushes_done        = 0;
        int    p2_probes_done        = 0;

        // With four PEs and the (0,1)(2,3) NVLink pairing, two iterations cover all four cross pair
        // edges. Any other PE count falls back to an (N-1) iteration ring that skips the NVLink pair.
        const bool use_optimized_4pe = (ctx.device_num == 4);

        if (!use_optimized_4pe && ctx.me == 0) {
            fprintf(stderr,
                    "[fusion=2] N=%d != 4: falling back to (N-1)-iter conditional-skip\n",
                    ctx.device_num);
        }

        const int n_p2_iters = use_optimized_4pe ? 2 : (ctx.device_num - 1);

        for (int k = 0; k < n_p2_iters; k++) {
            int sending_to, source_pe;
            bool do_push, do_probe;
            int iter_label;

            if (use_optimized_4pe) {
                const int pair_id    = ctx.me / 2;
                const int idx        = ctx.me % 2;
                const int other_base = (1 - pair_id) * 2;
                int target = other_base + (idx ^ k);
                sending_to = target;
                source_pe  = target;
                do_push  = true;
                do_probe = true;
                iter_label = k;
            } else {
                int loop = k + 1;
                sending_to = (ctx.me + loop) % ctx.device_num;
                source_pe  = (ctx.me - loop + ctx.device_num) % ctx.device_num;
                do_push  = !(has_nvl_peer && sending_to == nvl_peer);
                do_probe = !(has_nvl_peer && source_pe  == nvl_peer);
                iter_label = loop;
            }
            size_t send_bytes = sizeof(int) * filtered_size;

            auto t_ib0 = std::chrono::steady_clock::now();
            nvshmem_barrier_all();
            auto t_ib1 = std::chrono::steady_clock::now();
            double ib_sec = std::chrono::duration<double>(t_ib1 - t_ib0).count();
            p2_inner_barrier_sec += ib_sec;
            *ctx.cumulative_inner_barrier_sec += ib_sec;

            auto t_send0 = std::chrono::steady_clock::now();
            if (do_push) {
                if (ctx.use_nvlink) {
                    nvshmem_putmem(ctx.GPU_buffer, send_data, send_bytes, sending_to);
                    nvshmem_putmem(ctx.GPU_buffer_val, send_data_val, send_bytes, sending_to);
                } else {
                    cudaDeviceSynchronize();
                    cudaMemcpy(ctx.h_pcie_stage, send_data, send_bytes, cudaMemcpyDeviceToHost);
                    cudaMemcpy(ctx.h_pcie_stage_val, send_data_val, send_bytes, cudaMemcpyDeviceToHost);
                    int* remote_buf = (int*)nvshmem_ptr(ctx.GPU_buffer, sending_to);
                    int* remote_buf_val = (int*)nvshmem_ptr(ctx.GPU_buffer_val, sending_to);
                    if (!remote_buf) {
                        nvshmem_putmem(ctx.GPU_buffer, send_data, send_bytes, sending_to);
                        nvshmem_putmem(ctx.GPU_buffer_val, send_data_val, send_bytes, sending_to);
                    } else {
                        cudaMemcpy(remote_buf, ctx.h_pcie_stage, send_bytes, cudaMemcpyHostToDevice);
                        cudaMemcpy(remote_buf_val, ctx.h_pcie_stage_val, send_bytes, cudaMemcpyHostToDevice);
                    }
                }
                cudaDeviceSynchronize();
                p2_pushes_done++;
            }
            auto t_send1 = std::chrono::steady_clock::now();
            double push_sec = std::chrono::duration<double>(t_send1 - t_send0).count();

            auto t_db0 = std::chrono::steady_clock::now();
            nvshmem_barrier_all();
            auto t_db1 = std::chrono::steady_clock::now();
            double db_sec = std::chrono::duration<double>(t_db1 - t_db0).count();
            p2_delivery_bar_sec += db_sec;

            double probe_sec = 0.0;
            int answer_before = 0, answer_after = 0;
            if (do_probe) {
                cudaMemcpy(&answer_before, ctx.GPU_answer_num,
                           sizeof(int), cudaMemcpyDeviceToHost);
                { int zero = 0; cudaMemcpyToSymbol(d_probe_overflow_count, &zero, sizeof(int)); }
                auto t_pr0 = std::chrono::steady_clock::now();
                partial_jointable_gpu_run_only_s(
                    ctx.GPUA_split, ctx.GPU_buffer,
                    ctx.GPUA_val_split, ctx.GPU_buffer_val,
                    ctx.GPU_block_index_split_R, ctx.GPU_block_index_split_S[source_pe],
                    ctx.GPU_second_index_split_R, ctx.GPU_second_index_split_S[source_pe],
                    ctx.GPU_answer, ctx.GPU_answer_val, ctx.GPU_answer_s_val, ctx.GPU_answer_num,
                    ctx.max_int_number_for_shared_memory,
                    ctx.answer_num_host, ctx.gpu_stream, ctx.gpu_event,
                    ctx.GPU_answer_write_idx);
                cudaDeviceSynchronize();
                auto t_pr1 = std::chrono::steady_clock::now();
                probe_sec = std::chrono::duration<double>(t_pr1 - t_pr0).count();
                p2_probe_kernel_sec += probe_sec;
                *ctx.cumulative_second_pass_sec += probe_sec;
                { int h = 0; cudaMemcpyFromSymbol(&h, d_probe_overflow_count, sizeof(int));
                  *ctx.total_overflow_count += h; }
                cudaMemcpy(&answer_after, ctx.GPU_answer_num,
                           sizeof(int), cudaMemcpyDeviceToHost);
                p2_probes_done++;
            }
            double iter_transfer_sec = push_sec + db_sec;
            p2_transfer_sec += iter_transfer_sec;
            *ctx.cumulative_transfer_sec += iter_transfer_sec;

            printf("[PE%d][fusion=2][P2-iter%d] %s send_to=%d %s  src=%d %s  "
                   "ib=%.4f  push=%.4f  delivery=%.4f  probe=%.4f sec  matches=%d\n",
                   ctx.me, iter_label,
                   use_optimized_4pe ? "(2-iter ring)" : "(N-1 ring)",
                   sending_to, do_push  ? "(push)" : "(skip-NVL)",
                   source_pe,  do_probe ? "(probe)" : "(skip-NVL)",
                   ib_sec, push_sec, db_sec, probe_sec,
                   do_probe ? (answer_after - answer_before) : 0);
        }

        double phase2_wall_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_phase2_start).count();

        cudaMemcpy(ctx.answer_num_host, ctx.GPU_answer_num,
                   sizeof(int), cudaMemcpyDeviceToHost);
        *ctx.total_answer_num += ctx.answer_num_host[0];

        printf("[PE%d][fusion=2] ============================================\n", ctx.me);
        printf("[PE%d][fusion=2]   Phase 1 (fused local+NVL):  %.4f sec  answers=%d\n",
               ctx.me, kernel_wall_sec, fused_answer_count);
        printf("[PE%d][fusion=2]   Phase 2 (PUT loop, %d push/%d probe):  %.4f sec\n",
               ctx.me, p2_pushes_done, p2_probes_done, phase2_wall_sec);
        printf("[PE%d][fusion=2]     P2 inner_barrier sum:    %.4f sec\n", ctx.me, p2_inner_barrier_sec);
        printf("[PE%d][fusion=2]     P2 transfer (push+deliv):%.4f sec\n", ctx.me, p2_transfer_sec);
        printf("[PE%d][fusion=2]     P2 probe kernels sum:    %.4f sec\n", ctx.me, p2_probe_kernel_sec);
        printf("[PE%d][fusion=2]   TOTAL (P1 + P2):            %.4f sec  answers=%d\n",
               ctx.me, kernel_wall_sec + phase2_wall_sec, ctx.answer_num_host[0]);
        printf("[PE%d][fusion=2] ============================================\n", ctx.me);
    }
}
