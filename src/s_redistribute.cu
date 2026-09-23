#include "s_redistribute.h"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <vector>
#include <chrono>
#include <cstdio>
#include <algorithm>

namespace {

struct Transfer {
    int       src;
    int       dst;
    long long count;
    long long src_off;
    long long dst_off;
};

} // namespace

// Moves the S candidates that passed the filter between PEs so every PE holds a similar count.
// Returns this PE's count after redistribution.
long long redistribute_survivors(
    int me, int device_num,
    int* d_prefiltered_S, int* d_prefiltered_S_val, long long local_count, long long split_size,
    int* d_send_sym, int* d_recv_sym, int* d_send_sym_val, int* d_recv_sym_val, long long* d_sizes_sym,
    double skew_threshold,
    double& out_plan_sec, double& out_exchange_sec, bool& out_did_exchange,
    long long& out_total)
{
    out_plan_sec     = 0.0;
    out_exchange_sec = 0.0;
    out_did_exchange = false;
    out_total        = 0;

    const int N = device_num;

    auto t_plan0 = std::chrono::steady_clock::now();

    cudaMemcpy(&d_sizes_sym[me], &local_count, sizeof(long long),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    for (int dst = 0; dst < N; dst++) {
        if (dst == me) continue;
        nvshmem_putmem(&d_sizes_sym[me], &d_sizes_sym[me],
                       sizeof(long long), dst);
    }
    nvshmem_quiet();
    nvshmem_barrier_all();

    std::vector<long long> sizes(N, 0);
    cudaMemcpy(sizes.data(), d_sizes_sym, sizeof(long long) * N,
               cudaMemcpyDeviceToHost);

    long long total = 0;
    long long maxsz = 0;
    for (int p = 0; p < N; p++) {
        total += sizes[p];
        if (sizes[p] > maxsz) maxsz = sizes[p];
    }

    out_total = total;

    const double avg       = (N > 0) ? (double)total / (double)N : 0.0;
    const double imbalance = (avg > 0.0) ? (double)maxsz / avg : 1.0;

    const long long base = (N > 0) ? total / N : 0;
    const long long rem  = (N > 0) ? total % N : 0;
    std::vector<long long> tgt(N, 0);
    for (int p = 0; p < N; p++) tgt[p] = base + (p < rem ? 1 : 0);

    std::vector<Transfer> transfers;
    {
        std::vector<long long> surplus(N), sent(N, 0), recvd(N, 0);
        for (int p = 0; p < N; p++) surplus[p] = sizes[p] - tgt[p];
        int di = 0, ri = 0;
        while (true) {
            while (di < N && surplus[di] <= 0) di++;
            while (ri < N && surplus[ri] >= 0) ri++;
            if (di >= N || ri >= N) break;
            long long give = std::min(surplus[di], -surplus[ri]);
            transfers.push_back(Transfer{
                di, ri, give,
                tgt[di]   + sent[di],
                sizes[ri] + recvd[ri]});
            sent[di]    += give;
            recvd[ri]   += give;
            surplus[di] -= give;
            surplus[ri] += give;
        }
    }

    out_plan_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_plan0).count();

    if (total == 0 || imbalance <= skew_threshold) {
        out_did_exchange = false;
        return local_count;
    }

    auto t_exch0 = std::chrono::steady_clock::now();

    const long long my_tgt  = tgt[me];
    const long long my_keep = std::min(local_count, my_tgt);

    if (local_count > 0) {
        cudaMemcpy(d_send_sym, d_prefiltered_S, sizeof(int) * local_count,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_send_sym_val, d_prefiltered_S_val, sizeof(int) * local_count,
                   cudaMemcpyDeviceToDevice);
    }
    if (my_keep > 0) {
        cudaMemcpy(d_recv_sym, d_send_sym, sizeof(int) * my_keep,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_recv_sym_val, d_send_sym_val, sizeof(int) * my_keep,
                   cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();

    nvshmem_barrier_all();

    for (size_t t = 0; t < transfers.size(); t++) {
        const Transfer& tr = transfers[t];
        if (tr.src != me || tr.count <= 0) continue;
        nvshmem_putmem(d_recv_sym + tr.dst_off,
                       d_send_sym + tr.src_off,
                       sizeof(int) * (size_t)tr.count,
                       tr.dst);
        nvshmem_putmem(d_recv_sym_val + tr.dst_off,
                       d_send_sym_val + tr.src_off,
                       sizeof(int) * (size_t)tr.count,
                       tr.dst);
    }
    nvshmem_quiet();
    nvshmem_barrier_all();

    if (my_tgt > 0) {
        cudaMemcpy(d_prefiltered_S, d_recv_sym, sizeof(int) * my_tgt,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_prefiltered_S_val, d_recv_sym_val, sizeof(int) * my_tgt,
                   cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();

    out_exchange_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_exch0).count();
    out_did_exchange = true;
    return my_tgt;
}
