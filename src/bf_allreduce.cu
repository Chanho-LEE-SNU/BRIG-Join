#include "bf_allreduce.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#ifdef __AVX512FP16__
#undef __AVX512FP16__
#endif
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace cg = cooperative_groups;

// ORs src into dst word by word.
__global__ void bf_or_kernel(uint32_t* dst, const uint32_t* src, size_t num_words) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_words) dst[idx] |= src[idx];
}

// ---------------------------------------------------------------------------
// bf_ring_allreduce_fused  (use_bf==7)
// In-kernel ring allreduce: reduce-scatter (N-1 steps) + allgather (N-1 steps).
// Every block pushes an equal share of the send segment, since a single block cannot saturate the link.
// Must be launched with cudaLaunchCooperativeKernel, and d_bf_global must already hold the local filter.
// d_phase_times holds five slots per step for block 0: put, gsync+quiet+barrier, gsync, OR or copy, gsync.
// ---------------------------------------------------------------------------
__global__ void bf_ring_allreduce_fused(
    uint32_t* d_bf_global,
    uint32_t* d_bf_symmetric,
    size_t bf_words,
    int me, int device_num,
    long long* d_phase_times)
{
    cg::grid_group grid = cg::this_grid();

    const int tid         = threadIdx.x;
    const size_t gtid     = (size_t)blockIdx.x * blockDim.x + tid;
    const size_t tot_thr  = (size_t)gridDim.x * blockDim.x;
    const size_t base_seg = bf_words / device_num;
    const int next_pe     = (me + 1) % device_num;

    long long ts0=0, ts1=0, ts2=0, ts3=0, ts4=0, ts5=0;
    int step_idx = 0;

    for (int phase = 0; phase < 2; phase++) {          // 0 = Reduce-Scatter, 1 = Allgather
        for (int s = 0; s < device_num - 1; s++) {
            int send_seg = (phase == 0)
                ? (((me - s    ) % device_num + device_num) % device_num)
                : (((me + 1 - s) % device_num + device_num) % device_num);
            int recv_seg = (phase == 0)
                ? (((me - s - 1) % device_num + device_num) % device_num)
                : (((me     - s) % device_num + device_num) % device_num);
            size_t send_size = (send_seg == device_num - 1)
                               ? (bf_words - base_seg * (device_num - 1)) : base_seg;
            size_t recv_size = (recv_seg == device_num - 1)
                               ? (bf_words - base_seg * (device_num - 1)) : base_seg;

            if (blockIdx.x == 0 && tid == 0) ts0 = clock64();

            const size_t per_blk = (send_size + gridDim.x - 1) / gridDim.x;
            const size_t my_off  = (size_t)blockIdx.x * per_blk;
            if (my_off < send_size) {
                size_t my_cnt = per_blk;
                if (my_off + my_cnt > send_size) my_cnt = send_size - my_off;
                nvshmemx_putmem_block(
                    d_bf_symmetric + send_seg * base_seg + my_off,
                    d_bf_global    + send_seg * base_seg + my_off,
                    my_cnt * sizeof(uint32_t), next_pe);
            }
            if (blockIdx.x == 0 && tid == 0) ts1 = clock64();

            // nvshmem_quiet has PE scope, so block 0 issues it once for the whole grid.
            grid.sync();
            if (blockIdx.x == 0) {
                if (tid == 0) nvshmem_quiet();
                __syncthreads();
                nvshmemx_barrier_all_block();
            }
            if (blockIdx.x == 0 && tid == 0) ts2 = clock64();

            grid.sync();
            if (blockIdx.x == 0 && tid == 0) ts3 = clock64();

            size_t recv_off = recv_seg * base_seg;
            if (phase == 0) {
                for (size_t i = gtid; i < recv_size; i += tot_thr)
                    d_bf_global[recv_off + i] |= d_bf_symmetric[recv_off + i];
            } else {
                for (size_t i = gtid; i < recv_size; i += tot_thr)
                    d_bf_global[recv_off + i]  = d_bf_symmetric[recv_off + i];
            }
            if (blockIdx.x == 0 && tid == 0) ts4 = clock64();

            // All blocks must finish before the next PUT overwrites d_bf_symmetric.
            grid.sync();
            if (blockIdx.x == 0 && tid == 0) {
                ts5 = clock64();
                d_phase_times[step_idx * 5 + 0] = ts1 - ts0;
                d_phase_times[step_idx * 5 + 1] = ts2 - ts1;
                d_phase_times[step_idx * 5 + 2] = ts3 - ts2;
                d_phase_times[step_idx * 5 + 3] = ts4 - ts3;
                d_phase_times[step_idx * 5 + 4] = ts5 - ts4;
            }
            step_idx++;
        }
    }
}

// ---------------------------------------------------------------------------
// bf_pull_allreduce
// Pull based allgather and OR. Each block owns a contiguous chunk of bf_words and issues a
// block collective GET per peer followed by a local OR. There is no barrier and no cross block
// dependency, at the cost of moving (N-1) times bf_words per PE.
// d_phase_times is written by block 0 only, as [step*2+0]=get and [step*2+1]=or.
// ---------------------------------------------------------------------------
__global__ void bf_pull_allreduce(
    uint32_t* d_bf_global,
    uint32_t* d_bf_symmetric,
    uint32_t* d_recv_scratch,
    size_t bf_words,
    int me, int device_num,
    long long* d_phase_times)
{
    const size_t words_per_block = (bf_words + gridDim.x - 1) / gridDim.x;
    const size_t start = (size_t)blockIdx.x * words_per_block;
    const size_t end   = (start + words_per_block < bf_words)
                         ? (start + words_per_block) : bf_words;
    if (start >= end) return;
    const size_t chunk = end - start;

    const int tid = threadIdx.x;
    const int thread_size = blockDim.x;

    for (size_t i = tid; i < chunk; i += thread_size)
        d_bf_global[start + i] = d_bf_symmetric[start + i];
    __syncthreads();

    int step_idx = 0;
    for (int p = 0; p < device_num; p++) {
        if (p == me) continue;

        long long ts0 = 0, ts1 = 0, ts2 = 0;
        if (blockIdx.x == 0 && tid == 0) ts0 = clock64();

        nvshmemx_getmem_block(
            d_recv_scratch + start,
            d_bf_symmetric + start,
            chunk * sizeof(uint32_t), p);
        __syncthreads();
        if (blockIdx.x == 0 && tid == 0) ts1 = clock64();

        for (size_t i = tid; i < chunk; i += thread_size)
            d_bf_global[start + i] |= d_recv_scratch[start + i];
        __syncthreads();
        if (blockIdx.x == 0 && tid == 0) {
            ts2 = clock64();
            d_phase_times[step_idx * 2 + 0] = ts1 - ts0;  // get
            d_phase_times[step_idx * 2 + 1] = ts2 - ts1;  // or
        }
        step_idx++;
    }
}

// Grid sizing is split out of the launcher because the occupancy query can take milliseconds
// and must stay outside the measured region.
static const int BF_ALLREDUCE_THREADS = 1024;

// Returns the grid size of the pull allreduce kernel.
int bf_pull_grid_size()
{
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, bf_pull_allreduce, BF_ALLREDUCE_THREADS, 0);
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    return blocks_per_sm * prop.multiProcessorCount;
}

// Launches the pull allreduce kernel, which does not need a cooperative launch.
static void launch_bf_pull_allreduce(
    uint32_t* d_bf_global,
    uint32_t* d_bf_symmetric,
    uint32_t* d_recv_scratch,
    size_t bf_words, int me, int device_num,
    long long* d_phase_times, int grid_size)
{
    bf_pull_allreduce<<<grid_size, BF_ALLREDUCE_THREADS, 0>>>(
        d_bf_global, d_bf_symmetric, d_recv_scratch,
        bf_words, me, device_num, d_phase_times);
}

// Returns the grid size of the fused ring allreduce kernel.
int bf_fused_grid_size()
{
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, bf_ring_allreduce_fused, BF_ALLREDUCE_THREADS, 0);
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    return blocks_per_sm * prop.multiProcessorCount;
}

// Launches the fused ring allreduce with cudaLaunchCooperativeKernel.
static void launch_bf_ring_allreduce_fused(
    uint32_t* d_bf_global, uint32_t* d_bf_symmetric,
    size_t bf_words, int me, int device_num,
    long long* d_phase_times, int grid_size)
{
    void* args[] = { (void*)&d_bf_global, (void*)&d_bf_symmetric,
                     (void*)&bf_words, (void*)&me, (void*)&device_num,
                     (void*)&d_phase_times };
    cudaError_t e = cudaLaunchCooperativeKernel(
        (void*)bf_ring_allreduce_fused,
        dim3(grid_size), dim3(BF_ALLREDUCE_THREADS),
        args, 0, 0);
    if (e != cudaSuccess) {
        fprintf(stderr, "[PE%d] cudaLaunchCooperativeKernel failed: %s (grid=%d)\n",
                me, cudaGetErrorString(e), grid_size);
    }
}

// Ring OR allreduce over an arbitrary buffer. d_buf must already hold the local data.
void bf_ring_or_allreduce(
    int me, int device_num, uint32_t* d_buf, uint32_t* d_sym, size_t words,
    double& out_transfer_sec, double& out_merge_sec, double& out_allgather_sec)
{
    const int next_pe     = (me + 1) % device_num;
    const size_t base_seg = words / device_num;
    auto seg_size = [&](int seg) -> size_t {
        return (seg == device_num - 1)
               ? (words - base_seg * (device_num - 1)) : base_seg;
    };

    // Phase 1: Reduce-Scatter
    for (int s = 0; s < device_num - 1; s++) {
        int send_seg = ((me - s    ) % device_num + device_num) % device_num;
        int recv_seg = ((me - s - 1) % device_num + device_num) % device_num;
        auto t0 = std::chrono::steady_clock::now();
        nvshmem_putmem(d_sym + send_seg * base_seg,
                       d_buf + send_seg * base_seg,
                       seg_size(send_seg) * sizeof(uint32_t), next_pe);
        nvshmem_quiet();
        nvshmem_barrier_all();
        auto t1 = std::chrono::steady_clock::now();
        size_t rw = seg_size(recv_seg);
        bf_or_kernel<<<(rw + 255) / 256, 256>>>(
            d_buf + recv_seg * base_seg, d_sym + recv_seg * base_seg, rw);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::steady_clock::now();
        out_transfer_sec += std::chrono::duration<double>(t1 - t0).count();
        out_merge_sec    += std::chrono::duration<double>(t2 - t1).count();
    }

    // Phase 2: Allgather
    auto ag0 = std::chrono::steady_clock::now();
    for (int s = 0; s < device_num - 1; s++) {
        int send_seg = ((me + 1 - s) % device_num + device_num) % device_num;
        int recv_seg = ((me     - s) % device_num + device_num) % device_num;
        nvshmem_putmem(d_sym + send_seg * base_seg,
                       d_buf + send_seg * base_seg,
                       seg_size(send_seg) * sizeof(uint32_t), next_pe);
        nvshmem_quiet();
        nvshmem_barrier_all();
        size_t rw = seg_size(recv_seg);
        cudaMemcpy(d_buf + recv_seg * base_seg,
                   d_sym + recv_seg * base_seg,
                   rw * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    }
    out_allgather_sec += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - ag0).count();
}

// =============================================================================
// NCCL has no bitwise OR reduction, so two workarounds are provided.
//   Mode 15 all gathers the filters and ORs them locally, which moves about N/2 times
//   as much data as the ring allreduce.
//   Mode 16 expands each bit to a byte so that max behaves like OR, then compacts the result.
//   That inflates the traffic eightfold, so it runs chunk by chunk.
// Both leave the result in d_bf_global and share the downstream path with use_bf==13.
// =============================================================================

#define BF_NCCLCHECK(cmd) do {                                          \
    ncclResult_t _r = (cmd);                                            \
    if (_r != ncclSuccess) {                                            \
        fprintf(stderr, "[bf_nccl] NCCL error %s:%d '%s'\n",            \
                __FILE__, __LINE__, ncclGetErrorString(_r));            \
        abort();                                                        \
    }                                                                   \
} while (0)

// ORs all np slices of the gathered buffer into dst. The local slice is included.
__global__ void bf_or_gathered_kernel(uint32_t* __restrict__ dst,
                                      const uint32_t* __restrict__ gathered,
                                      size_t words, int np) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride) {
        uint32_t a = gathered[i];
        for (int p = 1; p < np; p++) a |= gathered[(size_t)p * words + i];
        dst[i] = a;
    }
}

// Same as bf_or_gathered_kernel but reads uint4 at a time. Requires words to be a multiple of four
// so that every slice offset stays aligned.
__global__ void bf_or_gathered_kernel_vec4(uint32_t* __restrict__ dst,
                                           const uint32_t* __restrict__ gathered,
                                           size_t words, int np) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t nvec   = words >> 2;
    uint4*       d4     = reinterpret_cast<uint4*>(dst);
    const uint4* g4     = reinterpret_cast<const uint4*>(gathered);
    const size_t vstep  = nvec;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
        uint4 a = g4[i];
        for (int p = 1; p < np; p++) {
            const uint4 b = g4[(size_t)p * vstep + i];
            a.x |= b.x; a.y |= b.y; a.z |= b.z; a.w |= b.w;
        }
        d4[i] = a;
    }
}

// Expands bit b of word i into byte (i*32 + b), writing 32 bytes as eight uint32 stores.
__global__ void bf_expand_bits_to_bytes_kernel(uint8_t* __restrict__ out,
                                               const uint32_t* __restrict__ in,
                                               size_t words) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride) {
        const uint32_t w = in[i];
        uint32_t* o = reinterpret_cast<uint32_t*>(out + i * 32);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t q = w >> (4 * k);
            o[k] =  (q        & 1u)
                 | ((q >> 1)  & 1u) << 8
                 | ((q >> 2)  & 1u) << 16
                 | ((q >> 3)  & 1u) << 24;
        }
    }
}

// Inverse of the expand kernel. Any nonzero byte sets the corresponding bit.
__global__ void bf_compact_bytes_to_bits_kernel(uint32_t* __restrict__ out,
                                                const uint8_t* __restrict__ in,
                                                size_t words) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(in + i * 32);
        uint32_t w = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t v = p[k];
            w |= (uint32_t)((v & 0x000000FFu) != 0) << (4 * k    );
            w |= (uint32_t)((v & 0x0000FF00u) != 0) << (4 * k + 1);
            w |= (uint32_t)((v & 0x00FF0000u) != 0) << (4 * k + 2);
            w |= (uint32_t)((v & 0xFF000000u) != 0) << (4 * k + 3);
        }
        out[i] = w;
    }
}

struct BfNccl {
    ncclComm_t   comm   = nullptr;
    cudaStream_t stream = nullptr;
    int    me = 0, np = 0, mode = 0;
    size_t words = 0;
    int    blocks = 0, threads = 256;

    uint32_t* d_gather = nullptr;

    uint8_t*  d_send_bytes = nullptr;
    uint8_t*  d_recv_bytes = nullptr;
    size_t    chunk_words  = 0;
    int       nchunk       = 0;
    std::vector<cudaEvent_t> ev;
};

// Creates the NCCL communicator. PE 0 generates the unique id and broadcasts it over NVSHMEM
// so that the code does not depend on MPI.
BfNccl* bf_nccl_create(int me, int np, size_t words, int mode) {
    BfNccl* hp = new BfNccl();
    BfNccl& h = *hp;
    h.me = me; h.np = np; h.mode = mode; h.words = words;

    // Source and destination are kept separate because overlapping them is undefined for broadcast.
    ncclUniqueId id;
    const size_t id_bytes = sizeof(ncclUniqueId);
    uint8_t* d_id = (uint8_t*)nvshmem_malloc(id_bytes * 2);
    if (!d_id) {
        fprintf(stderr, "[bf_nccl] nvshmem_malloc(uniqueId) failed\n");
        abort();
    }
    uint8_t* d_id_src = d_id;
    uint8_t* d_id_dst = d_id + id_bytes;
    if (me == 0) {
        BF_NCCLCHECK(ncclGetUniqueId(&id));
        cudaMemcpy(d_id_src, &id, id_bytes, cudaMemcpyHostToDevice);
    }
    nvshmem_barrier_all();
    nvshmem_broadcastmem(NVSHMEM_TEAM_WORLD, d_id_dst, d_id_src, id_bytes, 0);
    nvshmem_barrier_all();
    // Whether the root updates its own destination varies across OpenSHMEM versions,
    // so the root keeps the id it generated.
    if (me != 0) cudaMemcpy(&id, d_id_dst, id_bytes, cudaMemcpyDeviceToHost);
    nvshmem_free(d_id);

    BF_NCCLCHECK(ncclCommInitRank(&h.comm, np, id, me));
    cudaStreamCreateWithFlags(&h.stream, cudaStreamNonBlocking);

    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    h.blocks = std::min(2048, prop.multiProcessorCount * 8);

    if (mode == 15) {
        const size_t bytes = (size_t)np * words * sizeof(uint32_t);
        if (cudaMalloc(&h.d_gather, bytes) != cudaSuccess) {
            fprintf(stderr, "[PE%d] bf_nccl: cudaMalloc(allgather %.2f MB) failed\n",
                    me, bytes / (1024.0 * 1024.0));
            abort();
        }
        cudaMemset(h.d_gather, 0, bytes);
        if (me == 0)
            printf("use_bf==15 NCCL allgather+OR: words=%zu, gather buf=%.2f MB (%d slices)\n",
                   words, bytes / (1024.0 * 1024.0), np);
    } else {
        // The byte expansion is eight times larger, so it runs in chunks.
        // BF_NCCL_CHUNK_MB gives the chunk size in megabytes of expanded data.
        size_t chunk_mb = 64;
        if (const char* e = getenv("BF_NCCL_CHUNK_MB")) { long v = atol(e); if (v > 0) chunk_mb = (size_t)v; }
        h.chunk_words = (chunk_mb * 1024 * 1024) / 32;
        if (h.chunk_words == 0)     h.chunk_words = 1;
        if (h.chunk_words > words)  h.chunk_words = words;
        h.nchunk = (int)((words + h.chunk_words - 1) / h.chunk_words);
        const size_t bytes = h.chunk_words * 32;
        if (cudaMalloc(&h.d_send_bytes, bytes) != cudaSuccess ||
            cudaMalloc(&h.d_recv_bytes, bytes) != cudaSuccess) {
            fprintf(stderr, "[PE%d] bf_nccl: cudaMalloc(byte-expand 2 x %.2f MB) failed\n",
                    me, bytes / (1024.0 * 1024.0));
            abort();
        }
        cudaMemset(h.d_send_bytes, 0, bytes);
        cudaMemset(h.d_recv_bytes, 0, bytes);
        h.ev.resize((size_t)h.nchunk * 4);
        for (auto& e : h.ev) cudaEventCreate(&e);
        if (me == 0)
            printf("use_bf==16 NCCL byte-expand+max: words=%zu, chunk=%zu words (%.2f MB expanded) x %d chunks\n",
                   words, h.chunk_words, bytes / (1024.0 * 1024.0), h.nchunk);
    }

    // NCCL sets up its channels and internal buffers on the first collective, not in
    // ncclCommInitRank. The filter merge runs once per execution, so that setup would land inside
    // the measured merge. A warm up call with the same size and op moves it outside.
    if (np > 1) {
        if (mode == 15)
            BF_NCCLCHECK(ncclAllGather(h.d_gather + (size_t)me * words, h.d_gather,
                                       words, ncclUint32, h.comm, h.stream));
        else
            BF_NCCLCHECK(ncclAllReduce(h.d_send_bytes, h.d_recv_bytes, h.chunk_words * 32,
                                       ncclUint8, ncclMax, h.comm, h.stream));
        cudaStreamSynchronize(h.stream);
    }
    nvshmem_barrier_all();
    return hp;
}

// Releases everything allocated by bf_nccl_create.
void bf_nccl_destroy(BfNccl* hp) {
    if (!hp) return;
    BfNccl& h = *hp;
    nvshmem_barrier_all();
    for (auto& e : h.ev) cudaEventDestroy(e);
    if (h.d_gather)     cudaFree(h.d_gather);
    if (h.d_send_bytes) cudaFree(h.d_send_bytes);
    if (h.d_recv_bytes) cudaFree(h.d_recv_bytes);
    if (h.stream)       cudaStreamDestroy(h.stream);
    if (h.comm)         ncclCommDestroy(h.comm);
    delete hp;
}

// Gathers the N filters with ncclAllGather and ORs them locally.
// The local filter lives on the NVSHMEM symmetric heap, which NVSHMEM backs with virtual memory
// mappings that NCCL can fail to register for IPC. To avoid that, the filter is staged into this
// PE's own slot first and the all gather runs in place.
static void bf_nccl_allgather_or(BfNccl& h, const uint32_t* d_local, uint32_t* d_out,
                                 size_t words,
                                 double& exchange_sec, double& merge_sec,
                                 double& stage_sec) {
    uint32_t* my_slot = h.d_gather + (size_t)h.me * words;

    auto ts0 = std::chrono::steady_clock::now();
    cudaMemcpyAsync(my_slot, d_local, words * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, h.stream);
    cudaStreamSynchronize(h.stream);
    auto t0 = std::chrono::steady_clock::now();

    BF_NCCLCHECK(ncclAllGather(my_slot, h.d_gather, words, ncclUint32, h.comm, h.stream));
    cudaStreamSynchronize(h.stream);
    auto t1 = std::chrono::steady_clock::now();

    if ((words & 3u) == 0)
        bf_or_gathered_kernel_vec4<<<h.blocks, h.threads, 0, h.stream>>>(
            d_out, h.d_gather, words, h.np);
    else
        bf_or_gathered_kernel<<<h.blocks, h.threads, 0, h.stream>>>(
            d_out, h.d_gather, words, h.np);
    cudaStreamSynchronize(h.stream);
    auto t2 = std::chrono::steady_clock::now();

    stage_sec    += std::chrono::duration<double>(t0 - ts0).count();
    exchange_sec += std::chrono::duration<double>(t1 - ts0).count();
    merge_sec    += std::chrono::duration<double>(t2 - t1).count();
}

// Expands bits to bytes, runs an ncclMax allreduce and compacts the result back.
// Every chunk is queued on a single stream because sharing one communicator across several
// streams can deadlock NCCL. The per stage timings come from CUDA events.
static void bf_nccl_byte_max(BfNccl& h, const uint32_t* d_local, uint32_t* d_out,
                             size_t words,
                             double& expand_sec, double& reduce_sec, double& compact_sec) {
    for (int c = 0; c < h.nchunk; c++) {
        const size_t off = (size_t)c * h.chunk_words;
        const size_t n   = std::min(h.chunk_words, words - off);
        cudaEvent_t* e   = &h.ev[(size_t)c * 4];

        cudaEventRecord(e[0], h.stream);
        bf_expand_bits_to_bytes_kernel<<<h.blocks, h.threads, 0, h.stream>>>(
            h.d_send_bytes, d_local + off, n);
        cudaEventRecord(e[1], h.stream);

        BF_NCCLCHECK(ncclAllReduce(h.d_send_bytes, h.d_recv_bytes, n * 32,
                                   ncclUint8, ncclMax, h.comm, h.stream));
        cudaEventRecord(e[2], h.stream);

        bf_compact_bytes_to_bits_kernel<<<h.blocks, h.threads, 0, h.stream>>>(
            d_out + off, h.d_recv_bytes, n);
        cudaEventRecord(e[3], h.stream);
    }
    cudaStreamSynchronize(h.stream);

    for (int c = 0; c < h.nchunk; c++) {
        cudaEvent_t* e = &h.ev[(size_t)c * 4];
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e[0], e[1]); expand_sec  += ms / 1000.0;
        cudaEventElapsedTime(&ms, e[1], e[2]); reduce_sec  += ms / 1000.0;
        cudaEventElapsedTime(&ms, e[2], e[3]); compact_sec += ms / 1000.0;
    }
}

// Merges the per PE Bloom filters using the scheme selected by use_bf.
void bf_allreduce(
    int me, int device_num, int use_bf,
    std::vector<BloomFilter>& bf, size_t bf_words,
    uint32_t* d_bf_global, uint32_t* d_bf_symmetric, uint32_t* h_bf_cpu_or,
    uint32_t* d_recv_scratch, long long* d_phase_times, int bf_grid_size,
    double& bf_exchange_loop_sec, double& bf_exchange_sec,
    double& bf_merge_sec, double& bf_distribute_sec,
    std::vector<long long>* out_fused_cycles, BfNccl* nccl)
{
    // use_bf==1,2: all-to-all push
    if (use_bf == 1 || use_bf == 2) {
        auto t_exc0 = std::chrono::steady_clock::now();

        for (int dst_pe = 0; dst_pe < device_num; dst_pe++) {
            if (dst_pe == me) continue;
            nvshmem_putmem(bf[me].getDeviceFilter(), bf[me].getDeviceFilter(),
                           sizeof(uint32_t)*bf_words, dst_pe);
        }
        nvshmem_quiet();
        nvshmem_barrier_all();
        auto t_after_send = std::chrono::steady_clock::now();

        cudaMemcpy(d_bf_global, bf[me].getDeviceFilter(), sizeof(uint32_t)*bf_words, cudaMemcpyDeviceToDevice);
        for (int p = 0; p < device_num; p++) {
            if (p == me) continue;
            bf_or_kernel<<<((size_t)bf_words+255)/256, 256>>>(d_bf_global, bf[p].getDeviceFilter(), (size_t)bf_words);
            cudaDeviceSynchronize();
        }
        nvshmem_barrier_all();
        auto t_end = std::chrono::steady_clock::now();

        bf_exchange_loop_sec = std::chrono::duration<double>(t_end        - t_exc0      ).count();
        bf_exchange_sec      = std::chrono::duration<double>(t_after_send - t_exc0      ).count();
        bf_merge_sec         = std::chrono::duration<double>(t_end        - t_after_send).count();
        printf("[PE%d] Early BF all-reduce (%d PEs, push): %.4f sec\n", me, device_num, bf_exchange_loop_sec);
    }

    // use_bf==4: hub-and-spoke via PE 0
    if (use_bf == 4) {
        auto t_exc0 = std::chrono::steady_clock::now();

        if (me != 0)
            nvshmem_putmem(bf[me].getDeviceFilter(), bf[me].getDeviceFilter(),
                           sizeof(uint32_t)*bf_words, 0);
        nvshmem_quiet();
        nvshmem_barrier_all();
        auto t_after_send = std::chrono::steady_clock::now();

        if (me == 0) {
            cudaMemcpy(d_bf_global, bf[0].getDeviceFilter(), sizeof(uint32_t)*bf_words, cudaMemcpyDeviceToDevice);
            for (int p = 1; p < device_num; p++) {
                bf_or_kernel<<<((size_t)bf_words+255)/256, 256>>>(d_bf_global, bf[p].getDeviceFilter(), (size_t)bf_words);
                cudaDeviceSynchronize();
            }
            cudaMemcpy(d_bf_symmetric, d_bf_global, sizeof(uint32_t)*bf_words, cudaMemcpyDeviceToDevice);
        }
        nvshmem_barrier_all();
        auto t_after_merge = std::chrono::steady_clock::now();

        if (me == 0) {
            for (int dst = 1; dst < device_num; dst++)
                nvshmem_putmem(d_bf_symmetric, d_bf_symmetric, sizeof(uint32_t)*bf_words, dst);
        }
        nvshmem_quiet();
        nvshmem_barrier_all();

        if (me != 0)
            cudaMemcpy(d_bf_global, d_bf_symmetric, sizeof(uint32_t)*bf_words, cudaMemcpyDeviceToDevice);

        auto t_end = std::chrono::steady_clock::now();
        bf_exchange_loop_sec = std::chrono::duration<double>(t_end         - t_exc0       ).count();
        bf_exchange_sec      = std::chrono::duration<double>(t_after_send  - t_exc0       ).count();
        bf_merge_sec         = std::chrono::duration<double>(t_after_merge - t_after_send ).count();
        bf_distribute_sec    = std::chrono::duration<double>(t_end         - t_after_merge).count();
        printf("[PE%d] BF hub total=%.4f sec  (send=%.4f, merge=%.4f, distribute=%.4f)\n",
               me, bf_exchange_loop_sec, bf_exchange_sec, bf_merge_sec, bf_distribute_sec);
    }

    // use_bf==5,9: ring allreduce — reduce-scatter(N-1 steps) + allgather(N-1 steps)
    if (use_bf == 5 || use_bf == 9) {
        auto t_exc0 = std::chrono::steady_clock::now();

        cudaMemcpy(d_bf_global, bf[me].getDeviceFilter(),
                   sizeof(uint32_t)*bf_words, cudaMemcpyDeviceToDevice);

        int next_pe = (me + 1) % device_num;
        size_t base_seg = bf_words / device_num;

        auto get_seg_size = [&](int seg) -> size_t {
            return (seg == device_num - 1)
                   ? (bf_words - base_seg * (device_num - 1))
                   : base_seg;
        };

        double rs_transfer_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_exc0).count();
        double rs_merge_sec = 0.0;

        // Phase 1: Reduce-Scatter
        for (int s = 0; s < device_num - 1; s++) {
            int send_seg = ((me - s    ) % device_num + device_num) % device_num;
            int recv_seg = ((me - s - 1) % device_num + device_num) % device_num;
            size_t send_bytes = get_seg_size(send_seg) * sizeof(uint32_t);
            size_t recv_words = get_seg_size(recv_seg);

            auto t_s0 = std::chrono::steady_clock::now();
            nvshmem_putmem(d_bf_symmetric + send_seg * base_seg,
                           d_bf_global    + send_seg * base_seg,
                           send_bytes, next_pe);
            nvshmem_quiet();
            nvshmem_barrier_all();
            auto t_s1 = std::chrono::steady_clock::now();

            bf_or_kernel<<<(recv_words+255)/256, 256>>>(
                d_bf_global    + recv_seg * base_seg,
                d_bf_symmetric + recv_seg * base_seg, recv_words);
            cudaDeviceSynchronize();
            auto t_s2 = std::chrono::steady_clock::now();

            rs_transfer_sec += std::chrono::duration<double>(t_s1 - t_s0).count();
            rs_merge_sec    += std::chrono::duration<double>(t_s2 - t_s1).count();
        }
        auto t_after_rs = std::chrono::steady_clock::now();

        // Phase 2: Allgather
        for (int s = 0; s < device_num - 1; s++) {
            int send_seg = ((me + 1 - s) % device_num + device_num) % device_num;
            int recv_seg = ((me     - s) % device_num + device_num) % device_num;
            size_t send_bytes = get_seg_size(send_seg) * sizeof(uint32_t);
            size_t recv_bytes = get_seg_size(recv_seg) * sizeof(uint32_t);

            nvshmem_putmem(d_bf_symmetric + send_seg * base_seg,
                           d_bf_global    + send_seg * base_seg,
                           send_bytes, next_pe);
            nvshmem_quiet();
            nvshmem_barrier_all();

            cudaMemcpy(d_bf_global    + recv_seg * base_seg,
                       d_bf_symmetric + recv_seg * base_seg,
                       recv_bytes, cudaMemcpyDeviceToDevice);
        }

        auto t_end = std::chrono::steady_clock::now();
        bf_exchange_loop_sec = std::chrono::duration<double>(t_end      - t_exc0    ).count();
        bf_exchange_sec      = rs_transfer_sec;
        bf_merge_sec         = rs_merge_sec;
        bf_distribute_sec    = std::chrono::duration<double>(t_end      - t_after_rs).count();
        printf("[PE%d] BF ring-allreduce total=%.4f sec  (rs-transfer=%.4f, rs-merge=%.4f, allgather=%.4f)\n",
               me, bf_exchange_loop_sec, bf_exchange_sec, bf_merge_sec, bf_distribute_sec);
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "[PE%d] CUDA error after ring allreduce: %s\n", me, cudaGetErrorString(err));
            //else
            //    fprintf(stderr, "[PE%d] No error after ring allreduce\n", me);
            fflush(stderr);
        }
    }

    // Same as use_bf==5 except that the merge uses the nvshmem_uint32_or_reduce collective.
    if (use_bf == 13) {
        auto t_exc0 = std::chrono::steady_clock::now();

        // The collective is a host call and does not order against the stream, so the local
        // filter build is synchronized explicitly.
        cudaDeviceSynchronize();
        auto t_after_send = std::chrono::steady_clock::now();

        nvshmem_uint32_or_reduce(NVSHMEM_TEAM_WORLD,
                                 d_bf_symmetric,
                                 bf[me].getDeviceFilter(),
                                 bf_words);
        auto t_after_merge = std::chrono::steady_clock::now();

        cudaMemcpy(d_bf_global, d_bf_symmetric,
                   sizeof(uint32_t) * bf_words, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        nvshmem_barrier_all();
        auto t_end = std::chrono::steady_clock::now();

        bf_exchange_loop_sec = std::chrono::duration<double>(t_end          - t_exc0       ).count();
        bf_exchange_sec      = std::chrono::duration<double>(t_after_send   - t_exc0       ).count();
        bf_merge_sec         = std::chrono::duration<double>(t_after_merge  - t_after_send ).count();
        bf_distribute_sec    = std::chrono::duration<double>(t_end          - t_after_merge).count();
        printf("[PE%d] BF nvshmem or_reduce total=%.4f sec  (merge=%.4f, distribute=%.4f)\n",
               me, bf_exchange_loop_sec, bf_merge_sec, bf_distribute_sec);
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "[PE%d] CUDA error after nvshmem or_reduce: %s\n", me, cudaGetErrorString(err));
            //else
            //    fprintf(stderr, "[PE%d] No error after nvshmem or_reduce\n", me);
            fflush(stderr);
        }
    }

    if (use_bf == 15) {
        if (!nccl) {
            fprintf(stderr, "[PE%d] use_bf==15 requires a BfNccl handle (bf_nccl_create)\n", me);
            abort();
        }
        auto t_exc0 = std::chrono::steady_clock::now();

        // NCCL runs on its own stream, so the local filter build is synchronized first.
        cudaDeviceSynchronize();

        double stage_sec = 0.0;
        if (device_num == 1) {
            cudaMemcpy(d_bf_global, bf[me].getDeviceFilter(),
                       sizeof(uint32_t) * bf_words, cudaMemcpyDeviceToDevice);
        } else {
            bf_nccl_allgather_or(*nccl, bf[me].getDeviceFilter(), d_bf_global, bf_words,
                                 bf_exchange_sec, bf_merge_sec, stage_sec);
        }
        bf_distribute_sec = 0.0;

        auto t_end = std::chrono::steady_clock::now();
        bf_exchange_loop_sec = std::chrono::duration<double>(t_end - t_exc0).count();
        printf("[PE%d] BF nccl allgather+OR total=%.4f sec  (exchange=%.4f [stage=%.4f], local-or=%.4f)\n",
               me, bf_exchange_loop_sec, bf_exchange_sec, stage_sec, bf_merge_sec);
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "[PE%d] CUDA error after nccl allgather+OR: %s\n", me, cudaGetErrorString(err));
            fflush(stderr);
        }
    }

    if (use_bf == 16) {
        if (!nccl) {
            fprintf(stderr, "[PE%d] use_bf==16 requires a BfNccl handle (bf_nccl_create)\n", me);
            abort();
        }
        auto t_exc0 = std::chrono::steady_clock::now();

        cudaDeviceSynchronize();

        if (device_num == 1) {
            cudaMemcpy(d_bf_global, bf[me].getDeviceFilter(),
                       sizeof(uint32_t) * bf_words, cudaMemcpyDeviceToDevice);
        } else {
            // In this mode the timing buckets follow the layout conversion:
            // distribute is the expand, exchange is the ncclMax and merge is the compact.
            bf_nccl_byte_max(*nccl, bf[me].getDeviceFilter(), d_bf_global, bf_words,
                             bf_distribute_sec, bf_exchange_sec, bf_merge_sec);
        }

        auto t_end = std::chrono::steady_clock::now();
        bf_exchange_loop_sec = std::chrono::duration<double>(t_end - t_exc0).count();
        printf("[PE%d] BF nccl byte-expand+max total=%.4f sec  "
               "(expand=%.4f, allreduce=%.4f, compact=%.4f)\n",
               me, bf_exchange_loop_sec, bf_distribute_sec, bf_exchange_sec, bf_merge_sec);
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "[PE%d] CUDA error after nccl byte-expand+max: %s\n", me, cudaGetErrorString(err));
            fflush(stderr);
        }
    }

    // use_bf==7: ring allreduce, in-kernel fused (cooperative kernel)
    if (use_bf == 7) {
        const int grid_size = bf_grid_size;

        auto t_exc0 = std::chrono::steady_clock::now();

        const int num_steps = 2 * (device_num - 1);

        cudaMemcpy(d_bf_global, bf[me].getDeviceFilter(),
                   sizeof(uint32_t) * bf_words, cudaMemcpyDeviceToDevice);
        auto t_after_init = std::chrono::steady_clock::now();

        launch_bf_ring_allreduce_fused(d_bf_global, d_bf_symmetric,
                                        bf_words, me, device_num, d_phase_times, grid_size);
        cudaDeviceSynchronize();
        auto t_after_kernel = std::chrono::steady_clock::now();

        nvshmem_barrier_all();
        auto t_end = std::chrono::steady_clock::now();

        bf_exchange_loop_sec = std::chrono::duration<double>(t_end          - t_exc0       ).count();
        bf_exchange_sec      = std::chrono::duration<double>(t_after_init   - t_exc0       ).count();
        bf_merge_sec         = std::chrono::duration<double>(t_after_kernel - t_after_init ).count();
        bf_distribute_sec    = std::chrono::duration<double>(t_end          - t_after_kernel).count();

        // Phase cycles → caller's vector (printed later in Timing Summary)
        if (out_fused_cycles) {
            out_fused_cycles->resize(num_steps * 5);
            cudaMemcpy(out_fused_cycles->data(), d_phase_times,
                       sizeof(long long) * num_steps * 5, cudaMemcpyDeviceToHost);
        }

        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "[PE%d] CUDA error after fused ring allreduce: %s\n",
                        me, cudaGetErrorString(err));
            //else
            //    fprintf(stderr, "[PE%d] No error after fused ring allreduce\n", me);
            fflush(stderr);
        }
    }

    // use_bf==8: pull-based in-kernel allreduce (no barrier, per-block GET)
    if (use_bf == 8) {
        const int grid_size = bf_grid_size;

        auto t_exc0 = std::chrono::steady_clock::now();

        // 1) Stage local BF into NVSHMEM symmetric buffer (peers fetch from this)
        cudaMemcpy(d_bf_symmetric, bf[me].getDeviceFilter(),
                   sizeof(uint32_t) * bf_words, cudaMemcpyDeviceToDevice);
        nvshmem_barrier_all();
        auto t_after_init = std::chrono::steady_clock::now();

        const int num_steps = device_num - 1;

        launch_bf_pull_allreduce(d_bf_global, d_bf_symmetric, d_recv_scratch,
                                  bf_words, me, device_num, d_phase_times, grid_size);
        cudaDeviceSynchronize();
        auto t_after_kernel = std::chrono::steady_clock::now();

        bf_exchange_loop_sec = std::chrono::duration<double>(t_after_kernel - t_exc0       ).count();
        bf_exchange_sec      = std::chrono::duration<double>(t_after_init   - t_exc0       ).count();
        bf_merge_sec         = std::chrono::duration<double>(t_after_kernel - t_after_init ).count();
        bf_distribute_sec    = 0.0;

        // Phase cycles → caller's vector (use_bf=8 layout: 2 × num_steps)
        if (out_fused_cycles) {
            out_fused_cycles->resize(num_steps * 2);
            cudaMemcpy(out_fused_cycles->data(), d_phase_times,
                       sizeof(long long) * num_steps * 2, cudaMemcpyDeviceToHost);
        }

        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "[PE%d] CUDA error after pull allreduce: %s\n",
                        me, cudaGetErrorString(err));
            //else
            //    fprintf(stderr, "[PE%d] No error after pull allreduce\n", me);
            fflush(stderr);
        }
    }

    // use_bf==6: D2H → POSIX shm CPU OR (AVX2+OpenMP) → H2D
    if (use_bf == 6) {
        auto t_exc0 = std::chrono::steady_clock::now();
        size_t bf_bytes = sizeof(uint32_t) * bf_words;

        cudaMemcpy(h_bf_cpu_or + (size_t)me * bf_words,
                   bf[me].getDeviceFilter(), bf_bytes, cudaMemcpyDeviceToHost);
        nvshmem_barrier_all();
        auto t_after_send = std::chrono::steady_clock::now();

        if (me == 0) {
            uint32_t* base = h_bf_cpu_or;
            size_t avx_n = (bf_words / 8) * 8;
            #pragma omp parallel for schedule(static)
            for (size_t w = 0; w < avx_n; w += 8) {
                __m256i result = _mm256_loadu_si256((const __m256i*)(base + w));
                for (int p = 1; p < device_num; p++) {
                    __m256i b = _mm256_loadu_si256(
                        (const __m256i*)(h_bf_cpu_or + (size_t)p * bf_words + w));
                    result = _mm256_or_si256(result, b);
                }
                _mm256_storeu_si256((__m256i*)(base + w), result);
            }
            for (size_t w = avx_n; w < bf_words; w++) {
                uint32_t r = base[w];
                for (int p = 1; p < device_num; p++)
                    r |= h_bf_cpu_or[(size_t)p * bf_words + w];
                base[w] = r;
            }
        }
        nvshmem_barrier_all();
        auto t_after_merge = std::chrono::steady_clock::now();

        cudaMemcpy(d_bf_global, h_bf_cpu_or, bf_bytes, cudaMemcpyHostToDevice);
        nvshmem_barrier_all();

        auto t_end = std::chrono::steady_clock::now();
        bf_exchange_loop_sec = std::chrono::duration<double>(t_end         - t_exc0       ).count();
        bf_exchange_sec      = std::chrono::duration<double>(t_after_send  - t_exc0       ).count();
        bf_merge_sec         = std::chrono::duration<double>(t_after_merge - t_after_send ).count();
        bf_distribute_sec    = std::chrono::duration<double>(t_end         - t_after_merge).count();
        printf("[PE%d] BF cpu-or total=%.4f sec  (send=%.4f, merge=%.4f, distribute=%.4f)\n",
               me, bf_exchange_loop_sec, bf_exchange_sec, bf_merge_sec, bf_distribute_sec);
    }
}

// =============================================================================
// bf_v2: chunked-stream ring OR-allreduce (use_bf==14)
//
// Experimental replacement for bf_ring_or_allreduce. Each chunk gets its own stream so the step
// boundaries disappear, synchronization uses pairwise signals instead of barriers, and the
// allgather writes straight into the accumulator. The PUTs are host initiated on a stream so the
// copy engine moves the data while the kernel only waits and merges.
//
// The blocking on stream PUT is the default. The non blocking variant never emits its signal
// without a flush, which leaves the merge spinning forever, so using it would require an
// nvshmemx_quiet_on_stream per step.
// =============================================================================
#include <algorithm>
#include <cstdlib>

#ifndef BF_V2_PUT_SIGNAL
#define BF_V2_PUT_SIGNAL nvshmemx_putmem_signal_on_stream
#endif

// Merges the landed chunk into the accumulator.
// A device side wait would spin while holding an SM, which starves the proxy transfers of GPU time
// and deadlocks when peer to peer is unavailable, so all waiting happens on the host stream.
__global__ void bf_v2_merge_only_kernel(uint32_t* __restrict__ acc,
                                        const uint32_t* __restrict__ land,
                                        size_t words) {
  const size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * (size_t)blockDim.x;
  const size_t nvec   = words >> 2;
  uint4*       a4 = reinterpret_cast<uint4*>(acc);
  const uint4* l4 = reinterpret_cast<const uint4*>(land);
  for (size_t i = tid; i < nvec; i += stride) {
    uint4 a = a4[i];
    const uint4 l = l4[i];
    a.x |= l.x; a.y |= l.y; a.z |= l.z; a.w |= l.w;
    a4[i] = a;
  }
  for (size_t i = (nvec << 2) + tid; i < words; i += stride) acc[i] |= land[i];
}

struct BfV2 {
  int    me = 0, np = 0, next = 0, prev = 0;
  size_t words = 0, seg_words = 0, padded = 0, chunk_words = 0;
  int    nchunk = 0, nstream = 0;
  uint32_t* acc  = nullptr;   // symmetric, padded words. in=local, out=merged
  uint32_t* land = nullptr;   // symmetric, (np-1)*nchunk*chunk_words
  uint64_t* sig_rs = nullptr; // symmetric, (np-1)*nchunk
  uint64_t* sig_ag = nullptr; // symmetric, (np-1)*nchunk
  uint64_t* sig_ep = nullptr; // symmetric, nchunk. cross-call credit
  std::vector<cudaStream_t> stream;
  uint64_t call = 0;
  int or_blocks = 0, or_threads = 256;

  inline uint32_t* land_at(int c, int s) const {
    return land + ((size_t)s * nchunk + c) * chunk_words;
  }
  inline uint64_t* rs_at(int c, int s) const { return sig_rs + (size_t)s * nchunk + c; }
  inline uint64_t* ag_at(int c, int s) const { return sig_ag + (size_t)s * nchunk + c; }
  inline cudaStream_t st(int c) const { return stream[c % nstream]; }
};

static inline size_t bf_v2_align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Queues put, wait and merge per chunk on that chunk's stream, so the step dependencies are
// resolved by stream order alone.
static void bf_v2_or_allreduce(BfV2& h) {
  const int N = h.np, nc = h.nchunk;
  const uint64_t base = h.call;

  if (h.call > 0)
    for (int c = 0; c < nc; ++c)
      nvshmemx_signal_wait_until_on_stream(&h.sig_ep[c], NVSHMEM_CMP_GE, base, h.st(c));

  // Reduce scatter phase.
  for (int s = 0; s < N - 1; ++s) {
    const int send_seg = (h.me - s + N) % N;
    const int recv_seg = (h.me - s - 1 + N) % N;
    for (int c = 0; c < nc; ++c) {
      const size_t off = (size_t)c * h.chunk_words;
      const size_t len = std::min(h.chunk_words, h.seg_words - off);
      cudaStream_t st = h.st(c);
      BF_V2_PUT_SIGNAL(h.land_at(c, s),
                       h.acc + (size_t)send_seg * h.seg_words + off,
                       len * sizeof(uint32_t),
                       h.rs_at(c, s), 1ull, NVSHMEM_SIGNAL_ADD, h.next, st);
      nvshmemx_signal_wait_until_on_stream(h.rs_at(c, s), NVSHMEM_CMP_GE, base + 1, st);
      bf_v2_merge_only_kernel<<<h.or_blocks, h.or_threads, 0, st>>>(
          h.acc + (size_t)recv_seg * h.seg_words + off, h.land_at(c, s), len);
    }
  }
  // Tells the predecessor that this chunk's landing buffer has been consumed, which lets a later
  // call reuse it.
  for (int c = 0; c < nc; ++c)
    nvshmemx_signal_op_on_stream(&h.sig_ep[c], 1ull, NVSHMEM_SIGNAL_ADD, h.prev, h.st(c));

  // Allgather phase. It writes straight into the accumulator, with no merge or flow control.
  for (int s = 0; s < N - 1; ++s) {
    const int send_seg = (h.me + 1 - s + N) % N;
    for (int c = 0; c < nc; ++c) {
      const size_t off = (size_t)c * h.chunk_words;
      const size_t len = std::min(h.chunk_words, h.seg_words - off);
      cudaStream_t st = h.st(c);
      if (s > 0)
        nvshmemx_signal_wait_until_on_stream(h.ag_at(c, s - 1), NVSHMEM_CMP_GE, base + 1, st);
      BF_V2_PUT_SIGNAL(h.acc + (size_t)send_seg * h.seg_words + off,
                       h.acc + (size_t)send_seg * h.seg_words + off,
                       len * sizeof(uint32_t),
                       h.ag_at(c, s), 1ull, NVSHMEM_SIGNAL_ADD, h.next, st);
    }
  }
  for (int c = 0; c < nc; ++c)
    nvshmemx_signal_wait_until_on_stream(h.ag_at(c, N - 2), NVSHMEM_CMP_GE, base + 1, h.st(c));

  h.call += 1;
}

// Waits for every chunk stream to drain.
static void bf_v2_wait_all(BfV2& h) {
  for (auto& s : h.stream) cudaStreamSynchronize(s);
}

// Creates the buffers and streams for the chunked stream ring allreduce.
BfV2* bf_v2_create(size_t words, size_t chunk_bytes, int nstream) {
  BfV2* hp = new BfV2();
  BfV2& h = *hp;
  h.me = nvshmem_my_pe(); h.np = nvshmem_n_pes();
  h.next = (h.me + 1) % h.np; h.prev = (h.me - 1 + h.np) % h.np;
  h.words = words;

  // Environment overrides for the chunk size in KB, the stream count and the merge kernel grid.
  if (const char* e = getenv("BF_V2_CHUNK_KB"))  { long v = atol(e); if (v > 0) chunk_bytes = (size_t)v * 1024; }
  if (const char* e = getenv("BF_V2_NSTREAM"))   { int  v = atoi(e); if (v > 0) nstream = v; }
  int or_blocks_override = 0;
  if (const char* e = getenv("BF_V2_OR_BLOCKS")) { int v = atoi(e); if (v > 0) or_blocks_override = v; }
  // Segments are padded to a multiple of four words. A short tail segment would misalign the
  // signal indices between sender and receiver, and the padding is zero so the OR ignores it.
  h.seg_words = bf_v2_align_up((words + h.np - 1) / h.np, 4);
  h.padded    = h.seg_words * h.np;
  h.chunk_words = std::min(bf_v2_align_up(chunk_bytes / sizeof(uint32_t), 4), h.seg_words);
  if (h.chunk_words == 0) h.chunk_words = h.seg_words;
  h.nchunk    = (int)((h.seg_words + h.chunk_words - 1) / h.chunk_words);
  h.nstream   = std::min(nstream, h.nchunk);
  if (h.nstream < 1) h.nstream = 1;

  const size_t nsig = (size_t)(h.np - 1) * h.nchunk;
  h.acc    = (uint32_t*)nvshmem_malloc(h.padded * sizeof(uint32_t));
  h.land   = (uint32_t*)nvshmem_malloc((nsig ? nsig : 1) * h.chunk_words * sizeof(uint32_t));
  h.sig_rs = (uint64_t*)nvshmem_calloc(nsig ? nsig : 1, sizeof(uint64_t));
  h.sig_ag = (uint64_t*)nvshmem_calloc(nsig ? nsig : 1, sizeof(uint64_t));
  h.sig_ep = (uint64_t*)nvshmem_calloc(h.nchunk, sizeof(uint64_t));
  if (!h.acc || !h.land || !h.sig_rs || !h.sig_ag || !h.sig_ep) {
    fprintf(stderr, "[PE%d] bf_v2 nvshmem_malloc failed. Raise NVSHMEM_SYMMETRIC_SIZE.\n", h.me);
    abort();
  }
  cudaMemset(h.acc, 0, h.padded * sizeof(uint32_t));

  h.stream.resize(h.nstream);
  for (auto& s : h.stream) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

  int dev = 0; cudaGetDevice(&dev);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
  h.or_blocks = or_blocks_override ? or_blocks_override
                                   : std::min(1024, p.multiProcessorCount * 4);

  if (h.me == 0)
    printf("use_bf==14 v2 allreduce: words=%zu seg=%zu padded=%zu chunk=%.2fMB nchunk=%d streams=%d or_blocks=%d\n",
           h.words, h.seg_words, h.padded, h.chunk_words * 4.0 / 1e6, h.nchunk, h.nstream, h.or_blocks);

  nvshmem_barrier_all();  // one time, not on the call path
  return hp;
}

// Copies d_buf into the accumulator, merges it and copies the result back.
// The device to device copies are not ordered against the other streams, hence the syncs around them.
void bf_v2_merge(BfV2* hp, uint32_t* d_buf, size_t words) {
  BfV2& h = *hp;
  cudaMemcpy(h.acc, d_buf, words * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  bf_v2_or_allreduce(h);
  bf_v2_wait_all(h);
  cudaMemcpy(d_buf, h.acc, words * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
}

// Returns the internal accumulator, which can be used directly as the filter buffer.
uint32_t* bf_v2_acc(BfV2* hp) { return hp->acc; }

// Merges in place on the accumulator, which the build step must have filled beforehand.
void bf_v2_merge_inplace(BfV2* hp) {
  BfV2& h = *hp;
  cudaDeviceSynchronize();
  bf_v2_or_allreduce(h);
  bf_v2_wait_all(h);
  cudaDeviceSynchronize();
}

// Releases everything allocated by bf_v2_create.
void bf_v2_destroy(BfV2* hp) {
  if (!hp) return;
  BfV2& h = *hp;
  nvshmem_barrier_all();
  if (h.acc)    nvshmem_free(h.acc);
  if (h.land)   nvshmem_free(h.land);
  if (h.sig_rs) nvshmem_free(h.sig_rs);
  if (h.sig_ag) nvshmem_free(h.sig_ag);
  if (h.sig_ep) nvshmem_free(h.sig_ep);
  for (auto& s : h.stream) cudaStreamDestroy(s);
  delete hp;
}
