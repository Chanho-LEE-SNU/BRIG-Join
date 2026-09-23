#include "hash_kernel.h"
#include "vlsplit_hash_kernel.h"
#include "kernel_functions.h"
#include "common.h"
#include <chrono>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include "nccl.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include "bloom.cuh"
#include "bf_allreduce.h"
#include "sanity_check.h"
#include "join_executor.h"
#include "r_shuffle.h"
#include "r_replicate.h"
#include "s_redistribute.h"
#include <vector>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef __AVX512FP16__
#undef __AVX512FP16__
#endif
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <atomic>

int partition_size=128;
int second_partition_size=67;
int thread_no=1024;
int block_no= 31250;
int block_size;
int chunk_size=1024;

using namespace std;


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


// Splits the host tables into one contiguous slice per GPU.
void split_tables(int device_num,
                  int* const tableA, int* const tableB,
                  int* const tableA_val, int* const tableB_val,
                  long long *tableA_size, long long *tableB_size,
                  int **tableA_split, int **tableB_split,
                  int **tableA_val_split, int **tableB_val_split,
                  const long long R, const long long S)
{
    long long r_split_size=R/device_num;
    long long s_split_size=S/device_num;

    for(int i=0;i<device_num;i++)
    {
        tableA_split[i] = &tableA[r_split_size*i];
        tableB_split[i] = &tableB[s_split_size*i];
        tableA_val_split[i] = &tableA_val[r_split_size*i];
        tableB_val_split[i] = &tableB_val[s_split_size*i];
        if(i==device_num-1)
        {
            tableA_size[i]=R-r_split_size*i;
            tableB_size[i]=S-s_split_size*i;
        }
        else {
            tableA_size[i]=r_split_size;
            tableB_size[i]=s_split_size;
        }
    }
    return;
}



int max_int_number_for_shared_memory= HT_SIZE + 1;

// Runs the whole multi GPU join.
void gpu_main(vector<int> *solution, vector<int> *solution_ref,
              const std::string& r_inputPath, const std::string& s_inputPath,
              long long R, long long S,
              int use_bf, bool use_nvlink, bool use_warp_compact, double true_false_ratio, int s_chunks, bool sanityCheck,
              int kernel_fusion, double redist_skew_threshold,
              long long override_num_bits, int override_num_hashes,
              int override_partition_rs_split,
              int override_partition_size, int override_second_partition_size,
              int fused_probe_breakdown, int read_val, int r_shuffle_enable,
              double r_replicate_ratio, double answer_scale) {

    int r_replicate_enable = (r_replicate_ratio >= 0.0);


    // Mode 14 shares the whole pipeline with mode 10 and only swaps the filter merge, so it is
    // normalized to 10 here and the merge choice is carried by use_v2_merge.
    const bool use_v2_merge = (use_bf == 14);
    if (use_v2_merge) use_bf = 10;
    BfV2* bf_v2 = nullptr;

    // Modes 15 and 16 relate to mode 13 the same way, with the merge replaced by an NCCL path.
    const int bf_nccl_mode = (use_bf == 15 || use_bf == 16) ? use_bf : 0;
    if (bf_nccl_mode) use_bf = 13;
    BfNccl* bf_nccl = nullptr;

    std::chrono::duration<double> diff;
    
    auto pre_nvshmem = std::chrono::steady_clock::now();


    nvshmem_init();

    int me = nvshmem_my_pe();
    int device_num = nvshmem_n_pes();
    

    printf("ID: %d // PE: %d\n",me, device_num);

    if (r_shuffle_enable && r_replicate_enable) {
        if (me == 0)
            fprintf(stderr, "[FATAL] --r_shuffle_enable and --r_replicate_ratio cannot be used together\n");
        exit(EXIT_FAILURE);
    }

    if (use_bf < 0 || use_bf == 11 || use_bf == 12 || use_bf > 16) {
        if (me == 0)
            fprintf(stderr, "[FATAL] unsupported use_bf=%d\n", use_bf);
        exit(EXIT_FAILURE);
    }

    // Redistribution only applies when the threshold is on and the mode produces a flat candidate
    // list filtered by the globally merged Bloom filter.
    bool redist_active = (redist_skew_threshold > 0.0) &&
        (use_bf == 1 || use_bf == 2 || use_bf == 4 || use_bf == 5 ||
         use_bf == 6 || use_bf == 7 || use_bf == 8 || use_bf == 9 ||
         use_bf == 10 || use_bf == 13);
    if (redist_skew_threshold > 0.0 && !redist_active && me == 0) {
        fprintf(stderr,
            "[warn] S' redistribute requested but use_bf=%d has no global "
            "filtered survivor set; redistribute disabled\n", use_bf);
    }

    if (me == 0) {
        int native_atomic, perf_rank;
        cudaDeviceGetP2PAttribute(&native_atomic, cudaDevP2PAttrNativeAtomicSupported, 0, 1);
        cudaDeviceGetP2PAttribute(&perf_rank, cudaDevP2PAttrPerformanceRank, 0, 1);
        printf("P2P NativeAtomic: %d (1=NVLink)\n", native_atomic);
        printf("P2P PerformanceRank: %d\n", perf_rank);
    }

    // Upper bound on how much of a relation one GPU processes at a time.
    long long split_size = 200000000LL;

    if(split_size>R)
    {
        split_size=R;
    }
    {
        // ceil(R/N) is too small when R%N == N-1, so the real last slice size is used.
        long long actual_last_r = R - (R / device_num) * (long long)(device_num - 1);
        long long actual_last_s = S - (S / device_num) * (long long)(device_num - 1);
        long long max_per_gpu = max(actual_last_r, actual_last_s);
        if (split_size < max_per_gpu) split_size = max_per_gpu;
    }

    // Filter parameters come first because the automatic partition_size needs num_bits.
    double ln_p = log(true_false_ratio);
    double ln_2 = log(2.0);
    double bits_double = -(double)R * ln_p / (ln_2 * ln_2);
    size_t num_bits = (override_num_bits >= 0) ? (size_t)override_num_bits : (size_t)bits_double;
    int num_hashes  = (override_num_hashes >= 0) ? override_num_hashes : (int)(bits_double / (double)R * ln_2);
    if (me == 0)
        printf("BF params: num_bits=%zu num_hashes=%d (%s)\n", num_bits, num_hashes,
               (override_num_bits >= 0 || override_num_hashes >= 0) ? "override" : "formula");

    // Any partition sizing warning is printed with the timing breakdown.
    std::string part_size_warn;
    const long long ps_target_subbucket = (long long)(max_int_number_for_shared_memory - 1) * 8 / 10; // 8000

    // An explicit value wins. Otherwise mode 10 takes max(by_l2, join_min), where by_l2 keeps a
    // filter segment inside the L2 cache and join_min keeps second_partition_size within the scan
    // width. Every other mode keeps the default of 128.
    if (override_partition_size >= 0) {
        partition_size = override_partition_size;
        if (partition_size > PARTITION_MAX) {
            fprintf(stderr,
                "[PE%d] WARN: --partition_size=%d > PARTITION_MAX=%d, clamped. "
                "(to go higher, raise PARTITION_MAX in kernel_functions.h and the prefix launch opt-in)\n",
                me, partition_size, PARTITION_MAX);
            partition_size = PARTITION_MAX;
        }
    } else if (use_bf == 10) {
        const size_t L2_SEG_TARGET = 4ULL * 1024 * 1024;  // 4MB (L2 6MB)
        size_t bf_bytes = (num_bits + 7) / 8;
        long long by_l2    = (long long)((bf_bytes + L2_SEG_TARGET - 1) / L2_SEG_TARGET);
        // The cap is one below the scan width to absorb the round up to an odd value below.
        long long join_cap = (long long)(SECOND_PASS_SCAN_SIZE - 1) * ps_target_subbucket;
        long long join_min = (split_size + join_cap - 1) / join_cap;
        long long ps = (by_l2 > join_min) ? by_l2 : join_min;
        if (ps < 1) ps = 1;
        if (ps > PARTITION_MAX) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                "[WARN] use_bf==10 partition_size=%lld > PARTITION_MAX=%d (clamped to %d). "
                "The filter segment exceeds 4MB or the sub buckets grow too large, "
                "which may affect timing and accuracy.",
                ps, PARTITION_MAX, PARTITION_MAX);
            part_size_warn = buf;
            ps = PARTITION_MAX;
        }
        partition_size = (int)ps;
        if (me == 0)
            printf("use_bf==10 partition_size auto=%d (by_l2=%lld, join_min=%lld, BF=%.1fMB)\n",
                   partition_size, by_l2, join_min, (double)bf_bytes / (1024.0*1024.0));
    }
    // An explicit value wins. Otherwise it starts at 67 and grows to the next odd value whenever a
    // sub bucket would exceed 80 percent of the shared hash table.
    if (override_second_partition_size >= 0) {
        second_partition_size = override_second_partition_size;
        if (second_partition_size > SECOND_PASS_MAX) {
            fprintf(stderr,
                "[PE%d] WARN: --second_partition_size=%d > SECOND_PASS_MAX=%d, clamped. "
                "(to go higher, raise SECOND_PASS_MAX in kernel_functions.h and the launch opt-in)\n",
                me, second_partition_size, SECOND_PASS_MAX);
            second_partition_size = SECOND_PASS_MAX;
        }
    } else {
        long long ht_size   = max_int_number_for_shared_memory - 1; // = HT_SIZE (common.h)
        long long target    = ht_size * 8 / 10;                     // 80% = 8000
        long long sub_bucket = (split_size + (long long)partition_size * second_partition_size - 1)
                               / ((long long)partition_size * second_partition_size);
        if (sub_bucket > target) {
            second_partition_size = (int)((split_size + (long long)partition_size * target - 1)
                                          / ((long long)partition_size * target));
            if (second_partition_size % 2 == 0) second_partition_size++;
            if (second_partition_size > SECOND_PASS_MAX) {
                if (use_bf == 10) {
                    const char* cause = (partition_size >= PARTITION_MAX)
                        ? "partition_size is already at PARTITION_MAX"
                        : "the given partition_size is too small, try raising --partition_size";
                    char buf[256];
                    snprintf(buf, sizeof(buf),
                        "[WARN] second_partition_size=%d > SECOND_PASS_MAX=%d (clamped). "
                        "partition_size=%d leaves the sub buckets too large: %s. "
                        "Accuracy and timing may be affected.",
                        second_partition_size, SECOND_PASS_MAX, partition_size, cause);
                    if (part_size_warn.empty()) part_size_warn = buf;
                    second_partition_size = SECOND_PASS_MAX;
                    if (second_partition_size % 2 == 0) second_partition_size--;
                } else {
                    fprintf(stderr,
                        "[PE%d] FATAL: second_partition_size=%d exceeds SECOND_PASS_MAX=%d. "
                        "Increase partition_size or raise SECOND_PASS_MAX in kernel_functions.h.\n",
                        me, second_partition_size, SECOND_PASS_MAX);
                    exit(EXIT_FAILURE);
                }
            }
            if (me == 0)
                printf("second_partition_size auto-adjusted to %d (avg sub_bucket was %lld, target %lld)\n",
                       second_partition_size, sub_bucket, target);
        }
    }




    std::vector<BloomFilter> bf;
    bf.reserve(device_num);

    if (me == 0) {
        printf("Number of Devices: %d\n", device_num);
        printf("Block Number: %d\n", block_no);
        //printf("Block Size: %d\n", block_size);
    }

    cudaSetDevice(me);

    // The pinned buffers are allocated after cudaSetDevice so the pages land on the local NUMA node
    // and the input can be read straight into them.
    int *tableA = nullptr, *tableB = nullptr;
    cudaMallocHost((void**)&tableA, sizeof(int) * R);
    cudaMallocHost((void**)&tableB, sizeof(int) * S);
    int *tableA_val = nullptr, *tableB_val = nullptr;
    cudaMallocHost((void**)&tableA_val, sizeof(int) * R);
    cudaMallocHost((void**)&tableB_val, sizeof(int) * S);
    double nfs_r_sec = 0.0, nfs_s_sec = 0.0;
    {
        std::ifstream rf(r_inputPath, std::ios::binary);
        std::ifstream sf(s_inputPath, std::ios::binary);
        if (!rf.is_open() || !sf.is_open()) {
            fprintf(stderr, "[PE%d] Failed to open input files\n", me);
            exit(-1);
        }
        auto t_nfs0 = std::chrono::steady_clock::now();
        rf.read((char*)tableA, sizeof(int) * R);
        nfs_r_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_nfs0).count();
        auto t_nfs1 = std::chrono::steady_clock::now();
        sf.read((char*)tableB, sizeof(int) * S);
        nfs_s_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_nfs1).count();
        if (read_val) {
            // The input stores the key block first and the value block right after it.
            rf.read((char*)tableA_val, sizeof(int) * R);
            sf.read((char*)tableB_val, sizeof(int) * S);
            if (!rf || !sf) {
                fprintf(stderr, "[PE%d] read_val=1 but the input has no value block or it is too short\n", me);
                exit(-1);
            }
        } else {
            // Synthetic payloads derived from the row index, generated outside the Total timer.
            #pragma omp parallel for
            for (long long i = 0; i < R; i++) tableA_val[i] = (int)(i) + 1;
            #pragma omp parallel for
            for (long long i = 0; i < S; i++) tableB_val[i] = (int)(i) + 100000001;
        }
        //printf("[PE%d] NFS→CPU read: R=%.4f sec (%.2f MB/s)  S=%.4f sec (%.2f MB/s)\n",
        //       me,
        //       nfs_r_sec, R * sizeof(int) / nfs_r_sec / 1e6,
        //       nfs_s_sec, S * sizeof(int) / nfs_s_sec / 1e6);
    }
    printf("[PE%d] Pinned host alloc + file read complete\n", me);

    if (sanityCheck && me == 0)
        run_sanity_check(tableA, R, tableB, S);

    for (int i = 0; i < device_num; i++) {
        bf.emplace_back(num_bits, num_hashes);
    }
    size_t bf_words = bf[0].getNumWords();

    int **tableA_split;
    int **tableB_split;
    long long *tableA_size;
    long long *tableB_size;


    tableA_split = new int*[device_num];
    tableB_split = new int*[device_num];
    tableA_size = new long long[device_num];
    tableB_size = new long long[device_num];

    int **tableA_val_split = new int*[device_num];
    int **tableB_val_split = new int*[device_num];




    
    int *GPUA_split;
    cudaMalloc((void **) &GPUA_split, sizeof(int) * split_size);
    int *GPUB_split;
    cudaMalloc((void **) &GPUB_split, sizeof(int) * split_size);
    int *GPUA_val_split;
    cudaMalloc((void **) &GPUA_val_split, sizeof(int) * split_size);
    int *GPUB_val_split;
    cudaMalloc((void **) &GPUB_val_split, sizeof(int) * split_size);

    long long *GPU_A_prefix_split;
    long long *GPU_B_prefix_split;

    int *GPU_B_bloom_filtered_prefix_split;

    cudaMalloc((void **) &GPU_A_prefix_split, sizeof(long long) * partition_size);
    cudaMalloc((void **) &GPU_B_prefix_split, sizeof(long long) * partition_size);
    cudaMalloc((void **) &GPU_B_bloom_filtered_prefix_split, sizeof(int) * partition_size);


    int *GPU_OA_split;
    int *GPU_OB_split;
    cudaMalloc((void **) &GPU_OA_split, sizeof(int) * split_size * 2);//*2 as a backup
    cudaMalloc((void **) &GPU_OB_split, sizeof(int) * split_size * 2);
    int *GPU_OA_val_split;
    int *GPU_OB_val_split;
    cudaMalloc((void **) &GPU_OA_val_split, sizeof(int) * split_size * 2);
    cudaMalloc((void **) &GPU_OB_val_split, sizeof(int) * split_size * 2);

    int *GPU_buffer = (int *)nvshmem_malloc(sizeof(int)*split_size*2);//Recieves data from other GPUs
    int *GPUB_send_buffer = (int *)nvshmem_malloc(sizeof(int)*split_size*2);//Used to send bloom filtered data to other GPUs
    int *GPU_buffer_val = (int *)nvshmem_malloc(sizeof(int)*split_size*2);
    int *GPUB_send_buffer_val = (int *)nvshmem_malloc(sizeof(int)*split_size*2);
    // The gathered R stays resident for the whole S loop, so it needs its own buffer. GPU_buffer
    // cannot be reused because every chunk overwrites it during the S gather.
    int *R_gather_key = nullptr, *R_gather_val = nullptr;
    if (r_shuffle_enable) {
        R_gather_key = (int *)nvshmem_malloc(sizeof(int)*split_size*2);
        R_gather_val = (int *)nvshmem_malloc(sizeof(int)*split_size*2);
        r_shuffle_init(device_num);
    }
    // Buffers that hold a full copy of R on every GPU, plus scratch for the first scatter pass.
    int *R_full_key = nullptr, *R_full_val = nullptr;
    int *R_full_scratch = nullptr, *R_full_scratch_val = nullptr;
    if (r_replicate_enable) {
        R_full_key = (int *)nvshmem_malloc(sizeof(int) * R);
        R_full_val = (int *)nvshmem_malloc(sizeof(int) * R);
        cudaMalloc((void **) &R_full_scratch,     sizeof(int) * R);
        cudaMalloc((void **) &R_full_scratch_val, sizeof(int) * R);
        if (me == 0) {
            double gb = (double)R * sizeof(int) / 1e9;
            printf("[R_REPLICATE] R full copy per GPU: %lld elems "
                   "(symmetric %.2f GB + device %.2f GB)\n", R, 2.0 * gb, 2.0 * gb);
            long long sub = R / ((long long)partition_size * second_partition_size);
            if (sub >= (long long)HT_SIZE * 8 / 10)
                printf("[R_REPLICATE] WARN: R sub-bucket=%lld (HT_SIZE=%d) → probe linear "
                       "fallback risk, raise partition_size or second_partition_size\n",
                       sub, HT_SIZE);
        }
    }
    int2 *GPUB_send_buffer_kv = nullptr;
    if (kernel_fusion == 1 || kernel_fusion == 2)
        GPUB_send_buffer_kv = (int2 *)nvshmem_malloc(sizeof(int2)*split_size*2);

    // Host staging buffers for the PCIe path. Only the data needs staging because the indices are
    // exchanged separately during the shuffle step.
    size_t pcie_stage_bytes = sizeof(int) * split_size * 2;
    int* h_pcie_stage = nullptr;
    int* h_pcie_stage_val = nullptr;
    if (!use_nvlink) {
        cudaMallocHost(&h_pcie_stage, pcie_stage_bytes);
        cudaMallocHost(&h_pcie_stage_val, pcie_stage_bytes);
    }

    long long *GPU_block_index_split_R;
    cudaMalloc((void **) &GPU_block_index_split_R, sizeof(long long) * partition_size);

    // Entry i holds PE i's S partition index. These are symmetric so the shuffle can put into them
    // directly.
    long long **GPU_block_index_split_S = new long long*[device_num];
    for (int i = 0; i < device_num; i++)
        GPU_block_index_split_S[i] = (long long *)nvshmem_malloc(sizeof(long long) * partition_size);

    long long *GPU_A_offset;
    long long *GPU_B_offset;
    cudaMalloc((void **) &GPU_A_offset, sizeof(long long) * partition_size);
    cudaMalloc((void **) &GPU_B_offset, sizeof(long long) * partition_size);

    long long *GPU_second_index_split_R;
    cudaMalloc((void **) &GPU_second_index_split_R,
                         sizeof(long long) * partition_size * second_partition_size);

    long long **GPU_second_index_split_S = new long long*[device_num];
    for (int i = 0; i < device_num; i++)
        GPU_second_index_split_S[i] = (long long *)nvshmem_malloc(sizeof(long long) * partition_size * second_partition_size);

    int *GPU_answer;
    int *GPU_answer_val;
    int *GPU_answer_s_val;
    int *GPU_answer_num;
    int *GPU_answer_write_idx;
    // The answer buffer defaults to split_size and grows with --answer_scale for high fanout joins.
    long long answer_cap = (answer_scale > 1.0) ? (long long)((double)split_size * answer_scale) : split_size;
    if (me == 0)
        printf("answer capacity (per GPU) = %lld (split_size=%lld x answer_scale=%.2f)\n",
               answer_cap, split_size, answer_scale);
    cudaMalloc((void **) &GPU_answer, sizeof(int) * answer_cap);
    cudaMalloc((void **) &GPU_answer_val, sizeof(int) * answer_cap);
    cudaMalloc((void **) &GPU_answer_s_val, sizeof(int) * answer_cap);
    cudaMalloc((void **) &GPU_answer_num, sizeof(int));
    cudaMalloc((void **) &GPU_answer_write_idx, sizeof(int));
    cudaMemset(GPU_answer_write_idx, 0, sizeof(int));

    cudaStream_t gpu_stream;

    cudaEvent_t gpu_event;

    // Warms up the NVSHMEM connections so the first real transfer is not paying setup latency.
    {
        int warmup_peer = (me + 1) % device_num;
        const int WARMUP_ITERS = 5;
        for (int w = 0; w < WARMUP_ITERS; w++) {
            nvshmem_putmem(GPU_buffer, GPU_buffer, sizeof(int), warmup_peer);
            nvshmem_barrier_all();
        }
    }
    


    // Buffers for the global filter prefilter path. The allreduce scratch and timers are allocated
    // here rather than inside the merge so their cost stays out of the merge measurement.
    int*      d_prefiltered_S  = nullptr;
    int*      d_prefiltered_S_val = nullptr;
    uint32_t* d_bf_global      = nullptr;
    uint32_t* d_bf_symmetric   = nullptr;
    long long* d_compact_counter = nullptr;
    uint32_t* h_bf_cpu_or      = nullptr;
    long long* d_sizes_sym     = nullptr;
    uint32_t*  d_bf_recv_scratch = nullptr;
    long long* d_bf_phase_times  = nullptr;
    int        bf_grid_size      = 0;
    if (use_bf > 0) {
        cudaMalloc(&d_prefiltered_S,  sizeof(int)      * split_size);
        cudaMalloc(&d_prefiltered_S_val, sizeof(int)   * split_size);
        cudaMalloc(&d_bf_global,      sizeof(uint32_t) * bf_words);
        cudaMalloc(&d_compact_counter, sizeof(long long));
        if (use_bf == 8) {
            cudaMalloc(&d_bf_recv_scratch, sizeof(uint32_t) * bf_words);
            bf_grid_size = bf_pull_grid_size();
        }
        if (use_bf == 7) bf_grid_size = bf_fused_grid_size();
        if (use_bf == 7 || use_bf == 8) {
            // Sized for the larger of the two layouts used by modes 7 and 8.
            size_t phase_slots = (size_t)2 * (device_num - 1) * 5;
            if (phase_slots == 0) phase_slots = 5;
            cudaMalloc(&d_bf_phase_times, sizeof(long long) * phase_slots);
            cudaMemset(d_bf_phase_times, 0, sizeof(long long) * phase_slots);
        }
        d_sizes_sym = (long long*)nvshmem_malloc(sizeof(long long) * device_num);
        d_bf_symmetric = (uint32_t*)nvshmem_malloc(sizeof(uint32_t) * bf_words);
        cudaMemset(d_bf_symmetric, 0, sizeof(uint32_t) * bf_words);
        if (use_bf == 6) {
            // POSIX shared memory lets every PE map the same physical pages, so no CPU side IPC
            // is needed for the merge.
            size_t shm_size = sizeof(uint32_t) * bf_words * device_num;
            const char* shm_name = "/bf_cpu_or_shmem";
            int shm_fd;
            if (me == 0) {
                shm_unlink(shm_name);
                shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
                ftruncate(shm_fd, (off_t)shm_size);
            }
            nvshmem_barrier_all();
            if (me != 0)
                shm_fd = shm_open(shm_name, O_RDWR, 0666);
            h_bf_cpu_or = (uint32_t*)mmap(nullptr, shm_size,
                                           PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            close(shm_fd);
            cudaHostRegister(h_bf_cpu_or, shm_size, cudaHostRegisterDefault);
        }
        if (bf_nccl_mode)
            bf_nccl = bf_nccl_create(me, device_num, bf_words, bf_nccl_mode);
    }

    // Per partition filter buffers. The filter is cut into partition_size segments of a multiple of
    // 256 bits, and a segment is picked with the same radix the join uses for its first level.
    uint32_t* d_pbf            = nullptr;
    uint32_t* d_pbf_symmetric  = nullptr;
    size_t    per_partition_bits  = 0;
    size_t    per_partition_words = 0;
    size_t    pbf_total_words     = 0;
    long long* h_block_index_r    = nullptr;
    if (use_bf == 10) {
        per_partition_bits  = ((num_bits / (size_t)partition_size + 255) / 256) * 256;
        if (per_partition_bits < 256) per_partition_bits = 256;
        per_partition_words = per_partition_bits / 32;
        pbf_total_words = per_partition_words * (size_t)partition_size;
        h_block_index_r = new long long[partition_size];
        if (use_v2_merge) {
            // The v2 accumulator doubles as the filter buffer, which removes the staging copies and
            // lets the merge run in place. Only the leading pbf_total_words are used and the
            // padding stays zero so the ring ignores it.
            bf_v2 = bf_v2_create(pbf_total_words, 1u << 20, 4);
            d_pbf = bf_v2_acc(bf_v2);
        } else {
            cudaMalloc(&d_pbf, sizeof(uint32_t) * pbf_total_words);
            d_pbf_symmetric = (uint32_t*)nvshmem_malloc(sizeof(uint32_t) * pbf_total_words);
        }
        if (me == 0)
            printf("use_bf==10 partitioned BF: %d segments x %zu bits (%.2f MB/seg, %.2f MB total)\n",
                   partition_size, per_partition_bits,
                   (double)(per_partition_words * sizeof(uint32_t)) / (1024.0*1024.0),
                   (double)(pbf_total_words * sizeof(uint32_t)) / (1024.0*1024.0));
    }

    // Symmetric buffer where PE 0 collects the per PE answer counts.
    int* d_global_answer_counts = (int*)nvshmem_malloc(sizeof(int) * device_num);

    auto after_init = std::chrono::steady_clock::now();
    diff = after_init- pre_nvshmem;
    //std::cout << "Init including NVSHMEM took " << diff.count() << " sec"<< std::endl;


    split_tables(device_num, tableA, tableB, tableA_val, tableB_val, tableA_size, tableB_size, tableA_split, tableB_split, tableA_val_split, tableB_val_split, R, S);
    if(me==0)
    {
        printf("Splitting Complete\n");
        printf("R: ");
        for(int i=0;i<device_num;i++)
        {
            printf("%lld\t", tableA_size[i]);
        }
        printf("\nS: ");
        for(int i=0;i<device_num;i++)
        {
            printf("%lld\t", tableB_size[i]);
        }
        printf("\n");
    }
    
    
    


    int total_answer_num=0;
    int **answer_num =new int * [device_num];


    cudaDeviceSynchronize();
    cudaSetDevice(me);
    cudaStreamCreate(&gpu_stream);
    cudaEventCreate(&gpu_event);


    answer_num[me] = new int[1];

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("PE%d: GPU memory after allocations - Free: %.2f GB / Total: %.2f GB\n", me, free_mem / 1e9, total_mem / 1e9);

    
    






    // Start of the Total measurement, which includes the host to device copies.
    auto fill_start = std::chrono::steady_clock::now();
    cudaMemcpyAsync(GPUA_split, tableA_split[me], sizeof(int)*tableA_size[me], cudaMemcpyHostToDevice, gpu_stream);
    cudaMemcpyAsync(GPUA_val_split, tableA_val_split[me], sizeof(int)*tableA_size[me], cudaMemcpyHostToDevice, gpu_stream);
    cudaStreamSynchronize(gpu_stream);
    double r_h2d_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - fill_start).count();
    //printf("[PE%d] R H2D upload: %lld elements (%.2f MB) took %.4f sec  (%.2f GB/s)\n",
    //       me, tableA_size[me], tableA_size[me] * sizeof(int) / 1e6, r_h2d_sec,
    //       tableA_size[me] * sizeof(int) / r_h2d_sec / 1e9);
    long long s_chunk_size = (tableB_size[me] > 0)
                             ? (tableB_size[me] + s_chunks - 1) / s_chunks : 0;
    if (me == 0)
        printf("S chunks=%d  chunk_size=%lld  S_total=%lld\n",
               s_chunks, s_chunk_size, tableB_size[me]);


    auto t_pre_bf_barrier0 = std::chrono::steady_clock::now();
    nvshmem_barrier_all();
    double pre_bf_barrier_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pre_bf_barrier0).count();
    //printf("[PE%d] pre-BF barrier: %.4f sec\n", me, pre_bf_barrier_sec);

    double bf_build_total_sec  = 0.0;
    double r_partition_first_pass_sec = 0.0;
    double bf_exchange_loop_sec = 0.0;
    double bf_exchange_sec     = 0.0;
    double bf_merge_sec        = 0.0;
    double bf_distribute_sec   = 0.0;
    double bf_prefilter_sec    = 0.0;
    long long pf_survivors_total = 0;
    long long pf_input_total     = 0;

    {
        auto t0 = std::chrono::steady_clock::now();
        if (use_bf == 10) {
            // R is partitioned first and then one kernel per partition builds the filter. Each
            // kernel touches a single segment, so it stays in L2 and the atomics resolve there
            // instead of in global memory. The partition itself is a first pass cost, so it is
            // timed separately from the build.
            auto t_rpart0 = std::chrono::steady_clock::now();
            partial_jointable_gpu_partition_r_only(
                GPUA_split, GPU_OA_split,
                GPUA_val_split, GPU_OA_val_split,
                GPU_A_prefix_split, GPU_block_index_split_R,
                GPU_A_offset, GPU_second_index_split_R,
                tableA_size[me], gpu_stream);
            r_partition_first_pass_sec += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_rpart0).count();

            auto t_build0 = std::chrono::steady_clock::now();
            cudaMemcpy(h_block_index_r, GPU_block_index_split_R,
                       sizeof(long long) * partition_size, cudaMemcpyDeviceToHost);

            cudaMemset(d_pbf, 0, sizeof(uint32_t) * pbf_total_words);

            int threads = 256;
            for (int p = 0; p < partition_size; p++) {
                long long start = (p == 0) ? 0 : h_block_index_r[p - 1];
                long long end   = h_block_index_r[p];
                long long n     = end - start;
                if (n <= 0) continue;
                long long blocks = (n + threads - 1) / threads;
                insertKernelPartitioned<<<blocks, threads, 0, gpu_stream>>>(
                    d_pbf + (size_t)p * per_partition_words, GPUA_split, start, end,
                    num_hashes, per_partition_bits);
            }
            CUDACHECK(cudaGetLastError());
            cudaStreamSynchronize(gpu_stream);
            bf_build_total_sec += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_build0).count();

            // Every PE uses the same radix, so segment p always covers the same partition and a
            // word wise OR across PEs is enough.
            if (device_num > 1) {
                auto t_merge0 = std::chrono::steady_clock::now();
                if (use_v2_merge) {
                    bf_v2_merge_inplace(bf_v2);
                    bf_merge_sec = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_merge0).count();
                } else {
                    bf_ring_or_allreduce(me, device_num, d_pbf, d_pbf_symmetric, pbf_total_words,
                                         bf_exchange_sec, bf_merge_sec, bf_distribute_sec);
                }
                bf_exchange_loop_sec += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_merge0).count();
                CUDACHECK(cudaGetLastError());
            }
        } else if (use_bf == 9) {
            // One pass over R does both the filter insert and the partition histogram, so the
            // partition step below runs with skip_r_hist.
            const int Theta_insert = 4;
            int groups_per_block = thread_no / Theta_insert;                 // 256
            long long blocks_vec = (tableA_size[me] + groups_per_block - 1) / groups_per_block;
            cudaMemset(GPU_A_prefix_split, 0, sizeof(long long) * partition_size);
            insertAndHistKernel<<<blocks_vec, thread_no, sizeof(int) * partition_size>>>(
                bf[me].getDeviceFilter(), GPUA_split, tableA_size[me],
                bf[me].getNumHashes(), bf[me].getNumBits(),
                GPU_A_prefix_split, partition_size, /*block_size unused*/0);
            CUDACHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        } else if (use_bf > 0) {
            bf[me].buildFromTable(GPUA_split, tableA_size[me]);
        }
        // For every mode except 10 this whole block is the filter build.
        if (use_bf != 10)
            bf_build_total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        //std::cout << "The BF creating took " << bf_build_total_sec << " sec"<< std::endl;
    }

    std::vector<long long> bf_fused_cycles;  // use_bf==7: 5*num_steps cycles
    if (use_bf == 1 || use_bf == 2 || use_bf == 4 || use_bf == 5 || use_bf == 6 || use_bf == 7 || use_bf == 8 || use_bf == 9 || use_bf == 13) {
        // Modes 15 and 16 were normalized to 13 earlier, so the original mode is restored here for
        // the merge selection only.
        bf_allreduce(me, device_num, bf_nccl_mode ? bf_nccl_mode : use_bf, bf, bf_words,
                     d_bf_global, d_bf_symmetric, h_bf_cpu_or,
                     d_bf_recv_scratch, d_bf_phase_times, bf_grid_size,
                     bf_exchange_loop_sec, bf_exchange_sec,
                     bf_merge_sec, bf_distribute_sec,
                     &bf_fused_cycles, bf_nccl);
    }

    const bool partition_rs_split = (override_partition_rs_split >= 0)
                                  ? (override_partition_rs_split != 0) : false;

    double first_pass_partition_sec = r_partition_first_pass_sec;
    double cumulative_partition_r_sec = 0.0;
    double cumulative_partition_s_sec = 0.0;
    double cumulative_partition_hist_sec = 0.0;
    double cumulative_partition_hist_r_sec = 0.0;
    double cumulative_partition_hist_s_sec = 0.0;
    double first_pass_probe_sec     = 0.0;
    int total_overflow_count = 0;
    { int cap = (int)answer_cap; cudaMemcpyToSymbol(d_answer_capacity, &cap, sizeof(int)); }
    double cumulative_r_h2d_sec            = r_h2d_sec;
    double cumulative_pre_bf_barrier_sec   = pre_bf_barrier_sec;
    double cumulative_h2d_sec              = 0.0;
    double cumulative_transfer_sec         = 0.0;
    double cumulative_second_pass_sec      = 0.0;
    double cumulative_bf_exchange_sec      = 0.0;
    double cumulative_bf_pipeline_sec      = 0.0;
    double cumulative_pre_send_barrier_sec = 0.0;
    double cumulative_inner_barrier_sec    = 0.0;
    double cumulative_shuffle_sec          = 0.0;
    double cumulative_fused_ht_sec           = 0.0;
    double cumulative_fused_local_sec        = 0.0;
    double cumulative_fused_get_sec          = 0.0;
    double cumulative_fused_remote_probe_sec = 0.0;
    double cumulative_fused_wall_sec         = 0.0;
    double cumulative_fused_setup_sec        = 0.0;
    double cumulative_fp_preamble_sec        = 0.0;
    double cumulative_fp_memset_sec          = 0.0;
    double cumulative_fp_initbar_sec         = 0.0;
    double cumulative_fp_build_sec           = 0.0;
    double cumulative_fp_local_sec           = 0.0;
    double cumulative_fp_reduce_sec          = 0.0;
    double cumulative_redist_plan_sec        = 0.0;
    double cumulative_redist_exchange_sec    = 0.0;
    cudaStream_t send_stream;
    cudaStreamCreate(&send_stream);

    JoinChunkContext jctx;
    jctx.me                              = me;
    jctx.device_num                      = device_num;
    jctx.use_bf                          = use_bf;
    jctx.use_nvlink                      = use_nvlink;
    jctx.kernel_fusion                   = kernel_fusion;
    jctx.fused_probe_breakdown           = (fused_probe_breakdown != 0);
    jctx.GPUA_split                      = GPUA_split;
    jctx.GPUB_send_buffer                = GPUB_send_buffer;
    jctx.GPU_buffer                      = GPU_buffer;
    jctx.h_pcie_stage                    = h_pcie_stage;
    jctx.GPUA_val_split                  = GPUA_val_split;
    jctx.GPUB_send_buffer_val            = GPUB_send_buffer_val;
    jctx.GPU_buffer_val                  = GPU_buffer_val;
    jctx.h_pcie_stage_val                = h_pcie_stage_val;
    jctx.GPUB_send_buffer_kv             = GPUB_send_buffer_kv;
    jctx.GPU_OB_split                    = GPU_OB_split;
    jctx.GPU_OB_val_split                = GPU_OB_val_split;
    jctx.GPU_B_prefix_split              = GPU_B_prefix_split;
    jctx.GPU_B_offset                    = GPU_B_offset;
    jctx.GPU_block_index_split_R         = GPU_block_index_split_R;
    jctx.GPU_block_index_split_S         = GPU_block_index_split_S;
    jctx.GPU_second_index_split_R        = GPU_second_index_split_R;
    jctx.GPU_second_index_split_S        = GPU_second_index_split_S;
    jctx.GPU_answer                      = GPU_answer;
    jctx.GPU_answer_val                  = GPU_answer_val;
    jctx.GPU_answer_s_val                = GPU_answer_s_val;
    jctx.GPU_answer_num                  = GPU_answer_num;
    jctx.GPU_answer_write_idx            = GPU_answer_write_idx;
    jctx.answer_num_host                 = answer_num[me];
    jctx.bf                              = &bf;
    jctx.bf_words                        = bf_words;
    jctx.d_bf_global                     = d_bf_global;
    jctx.gpu_stream                      = gpu_stream;
    jctx.gpu_event                       = gpu_event;
    jctx.send_stream                     = send_stream;
    jctx.max_int_number_for_shared_memory = max_int_number_for_shared_memory;
    jctx.first_pass_probe_sec            = &first_pass_probe_sec;
    jctx.cumulative_bf_exchange_sec      = &cumulative_bf_exchange_sec;
    jctx.cumulative_bf_pipeline_sec      = &cumulative_bf_pipeline_sec;
    jctx.cumulative_pre_send_barrier_sec = &cumulative_pre_send_barrier_sec;
    jctx.cumulative_inner_barrier_sec    = &cumulative_inner_barrier_sec;
    jctx.cumulative_transfer_sec         = &cumulative_transfer_sec;
    jctx.cumulative_second_pass_sec      = &cumulative_second_pass_sec;
    jctx.cumulative_fused_ht_sec           = &cumulative_fused_ht_sec;
    jctx.cumulative_fused_local_sec        = &cumulative_fused_local_sec;
    jctx.cumulative_fused_get_sec          = &cumulative_fused_get_sec;
    jctx.cumulative_fused_remote_probe_sec = &cumulative_fused_remote_probe_sec;
    jctx.cumulative_fused_wall_sec         = &cumulative_fused_wall_sec;
    jctx.cumulative_fused_setup_sec        = &cumulative_fused_setup_sec;
    jctx.cumulative_fp_preamble_sec        = &cumulative_fp_preamble_sec;
    jctx.cumulative_fp_memset_sec          = &cumulative_fp_memset_sec;
    jctx.cumulative_fp_initbar_sec         = &cumulative_fp_initbar_sec;
    jctx.cumulative_fp_build_sec           = &cumulative_fp_build_sec;
    jctx.cumulative_fp_local_sec           = &cumulative_fp_local_sec;
    jctx.cumulative_fp_reduce_sec          = &cumulative_fp_reduce_sec;
    jctx.total_answer_num                = &total_answer_num;
    jctx.total_overflow_count            = &total_overflow_count;

    // Persistent scratch for the fused kernels, allocated once outside the timed region so the
    // comparison against kernel_fusion==0 stays fair.
    jctx.d_s_block_ptrs        = nullptr;
    jctx.d_s_second_ptrs       = nullptr;
    jctx.d_phase_timers        = nullptr;
    jctx.d_phase_timers_slots  = 0;
    // The phase timer buffer is also used by the non fused local probe, so it is always allocated.
    {
        const int num_blocks = partition_size * second_partition_size;
        jctx.d_phase_timers_slots = (long long)num_blocks * 7 + 16 + 32;
        cudaMalloc(&jctx.d_phase_timers,
                   sizeof(long long) * jctx.d_phase_timers_slots);
    }
    if (kernel_fusion == 1 || kernel_fusion == 2) {
        cudaMalloc(&jctx.d_s_block_ptrs,  sizeof(long long*) * device_num);
        cudaMalloc(&jctx.d_s_second_ptrs, sizeof(long long*) * device_num);
        cudaMemcpy(jctx.d_s_block_ptrs,  GPU_block_index_split_S,
                   sizeof(long long*) * device_num, cudaMemcpyHostToDevice);
        cudaMemcpy(jctx.d_s_second_ptrs, GPU_second_index_split_S,
                   sizeof(long long*) * device_num, cudaMemcpyHostToDevice);
    }

    double rs_r_gather_sec = 0.0, rs_s_partition_sec = 0.0, rs_s_gather_sec = 0.0, rs_probe_sec = 0.0;
    double rr_r_gather_sec = 0.0, rr_s_partition_sec = 0.0, rr_probe_sec = 0.0;

    if (r_shuffle_enable && ((use_bf != 5 && use_bf != 10) || partition_size % device_num != 0)) {
        if (me == 0)
            fprintf(stderr, "[R_SHUFFLE] requires use_bf in {5,10} and partition_size %% device_num == 0 "
                    "(use_bf=%d, P=%d, D=%d), falling back to broadcast\n", use_bf, partition_size, device_num);
        r_shuffle_enable = 0;
    }
    if (r_shuffle_enable) {
        redist_active = false;
        r_shuffle_gather_r(jctx, R_gather_key, R_gather_val,
                           GPU_OA_split, GPU_OA_val_split,
                           GPU_A_prefix_split, GPU_A_offset,
                           tableA_size[me], split_size, rs_r_gather_sec);
    }

    if (r_replicate_enable && (use_bf == 1 || use_bf == 3)) {
        if (me == 0)
            fprintf(stderr, "[R_REPLICATE] use_bf=%d uses a dedicated join path and is not supported, "
                    "falling back to broadcast\n", use_bf);
        r_replicate_enable = 0;
    }
    // The gate needs the global candidate total, which only the redistribute planning stage computes.
    if (r_replicate_enable && r_replicate_ratio > 0.0 && !redist_active) {
        if (me == 0)
            fprintf(stderr, "[FATAL] --r_replicate_ratio requires redistribute to be enabled. "
                    "Set a large redist_skew_threshold (for example 50) to collect the sizes without exchanging.\n");
        exit(EXIT_FAILURE);
    }
    bool rr_decided = false;
    long long rr_s_total = 0;

    for (int ci = 0; ci < s_chunks; ci++) {
        long long s_off  = (long long)ci * s_chunk_size;
        long long s_this = tableB_size[me] - s_off;
        if (s_this > s_chunk_size) s_this = s_chunk_size;
        if (s_this < 0) s_this = 0;

        if (s_chunks > 1)
            printf("[PE%d] S chunk %d/%d: offset=%lld size=%lld\n",
                   me, ci+1, s_chunks, s_off, s_this);

        {
            auto t_h2d0 = std::chrono::steady_clock::now();
            if (s_this > 0) {
                cudaMemcpy(GPUB_split, tableB_split[me] + s_off,
                           sizeof(int)*s_this, cudaMemcpyHostToDevice);
                cudaMemcpy(GPUB_val_split, tableB_val_split[me] + s_off,
                           sizeof(int)*s_this, cudaMemcpyHostToDevice);
            }
            cumulative_h2d_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_h2d0).count();
        }

        long long actual_b_size = s_this;
        int* effective_s_ptr = GPUB_split;
        int* effective_s_val = GPUB_val_split;

        if (use_bf == 10 && s_this > 0) {
            auto t_pf0 = std::chrono::steady_clock::now();

            long long pf_block_size = s_this / block_no;
            if (s_this % block_no != 0) pf_block_size++;

            cudaMemset(d_compact_counter, 0, sizeof(long long));
            queryAndCompactKernelPartitioned<<<block_no, thread_no>>>(
                d_pbf, GPUB_split, GPUB_val_split, s_this,
                num_hashes, per_partition_bits, per_partition_words, partition_size,
                d_prefiltered_S, d_prefiltered_S_val, d_compact_counter, pf_block_size);
            cudaDeviceSynchronize();
            CUDACHECK(cudaGetLastError());
            cudaMemcpy(&actual_b_size, d_compact_counter, sizeof(long long), cudaMemcpyDeviceToHost);

            bf_prefilter_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pf0).count();
            pf_survivors_total += actual_b_size;
            pf_input_total     += s_this;
            effective_s_ptr = d_prefiltered_S;
            effective_s_val = d_prefiltered_S_val;
        } else if ((use_bf == 1 || use_bf == 2 || use_bf == 4 || use_bf == 5 || use_bf == 6 || use_bf == 7 || use_bf == 8 || use_bf == 9 || use_bf == 13) && s_this > 0) {
            auto t_pf0 = std::chrono::steady_clock::now();

            long long pf_block_size = s_this / block_no;
            if (s_this % block_no != 0) pf_block_size++;

            // Redistribution needs a flat candidate list, so it forces the compact path.
            if (use_warp_compact || redist_active) {
                cudaMemset(d_compact_counter, 0, sizeof(long long));
                queryAndCompactKernel<<<block_no, thread_no>>>(
                    d_bf_global, GPUB_split, GPUB_val_split, s_this,
                    bf[me].getNumHashes(), bf[me].getNumBits(),
                    d_prefiltered_S, d_prefiltered_S_val, d_compact_counter, pf_block_size);
                cudaDeviceSynchronize();
                {
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                        fprintf(stderr, "[PE%d] CUDA error after queryAndCompactKernel: %s\n", me, cudaGetErrorString(err));
                    //else
                    //    fprintf(stderr, "[PE%d] No error after queryAndCompactKernel\n", me);
                    fflush(stderr);
                }
                cudaMemcpy(&actual_b_size, d_compact_counter, sizeof(long long), cudaMemcpyDeviceToHost);
            } else {
                cudaMemset(GPU_B_prefix_split, 0, sizeof(long long) * partition_size);
                queryAndHistOnlyKernel<<<block_no, thread_no, sizeof(int)*partition_size>>>(
                    d_bf_global, GPUB_split, s_this,
                    bf[me].getNumHashes(), bf[me].getNumBits(),
                    GPU_B_prefix_split, partition_size, pf_block_size);
                cudaDeviceSynchronize();

                gpu_make_simple_prefix<<<1, prefix_scan_block_dim(partition_size, thread_no), sizeof(long long)*partition_size*2>>>(
                    GPUB_split, GPU_B_prefix_split, GPU_block_index_split_S[me],
                    s_this, partition_size, GPU_B_offset);
                cudaDeviceSynchronize();

                cudaMemcpy(&actual_b_size, &GPU_block_index_split_S[me][partition_size-1],
                           sizeof(long long), cudaMemcpyDeviceToHost);

                queryAndScatterKernel<<<block_no, thread_no, 0>>>(
                    d_bf_global, GPUB_split, GPUB_val_split, s_this,
                    bf[me].getNumHashes(), bf[me].getNumBits(),
                    d_prefiltered_S, d_prefiltered_S_val, GPU_B_offset, partition_size, pf_block_size);
                cudaDeviceSynchronize();
            }

            bf_prefilter_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pf0).count();
            pf_survivors_total += actual_b_size;
            pf_input_total     += s_this;
            //printf("[PE%d] chunk%d Global BF pre-filter (%s): %lld / %lld -> %.1f%%  (%.4f sec)\n",
            //       me, ci, use_warp_compact ? "warp compact" : "hist+scatter",
            //       actual_b_size, s_this,
            //       100.0 * actual_b_size / s_this, bf_prefilter_sec);

            effective_s_ptr = d_prefiltered_S;
            effective_s_val = d_prefiltered_S_val;
        }

        // Balances the candidate counts across GPUs while they are still flat and unpartitioned.
        // It contains collectives, so every PE calls it regardless of its own chunk size, and it
        // returns early when the imbalance stays under the threshold.
        if (redist_active) {
            double redist_plan_sec = 0.0, redist_exchange_sec = 0.0;
            bool   redist_did = false;
            actual_b_size = redistribute_survivors(
                me, device_num,
                d_prefiltered_S, d_prefiltered_S_val, actual_b_size, split_size,
                GPUB_send_buffer, GPU_buffer, GPUB_send_buffer_val, GPU_buffer_val, d_sizes_sym,
                redist_skew_threshold,
                redist_plan_sec, redist_exchange_sec, redist_did, rr_s_total);
            cumulative_redist_plan_sec     += redist_plan_sec;
            cumulative_redist_exchange_sec += redist_exchange_sec;
            if (redist_did) { effective_s_ptr = d_prefiltered_S; effective_s_val = d_prefiltered_S_val; }
            //printf("[PE%d] chunk%d S' redistribute: %s -> new_size=%lld "
            //       "(plan %.4f s, exchange %.4f s)\n",
            //       me, ci, redist_did ? "exchanged" : "skipped",
            //       actual_b_size, redist_plan_sec, redist_exchange_sec);
        }

        if (r_shuffle_enable) {
            total_answer_num += r_shuffle_join_chunk(
                jctx, R_gather_key, R_gather_val,
                effective_s_ptr, effective_s_val, actual_b_size, split_size,
                rs_s_partition_sec, rs_s_gather_sec, rs_probe_sec);
            continue;
        }

        // The first chunk decides whether replicating R pays off, using the global candidate total.
        if (r_replicate_enable && !rr_decided) {
            rr_decided = true;
            if (r_replicate_ratio > 0.0) {
                double ratio = (double)(rr_s_total * s_chunks) / (double)R;
                if (me == 0)
                    printf("[R_REPLICATE] S'/R = %.3f (S'~%lld, R=%lld), threshold %.2f -> %s\n",
                           ratio, rr_s_total * s_chunks, R, r_replicate_ratio,
                           ratio >= r_replicate_ratio ? "replicate" : "broadcast");
                if (ratio < r_replicate_ratio) r_replicate_enable = 0;
            }
            if (r_replicate_enable)
                r_replicate_gather_r(jctx, R_full_key, R_full_val,
                                     R_full_scratch, R_full_scratch_val,
                                     GPU_A_prefix_split, GPU_A_offset,
                                     tableA_size, R, rr_r_gather_sec);
        }

        if (r_replicate_enable) {
            total_answer_num += r_replicate_join_chunk(
                jctx, R_full_key, R_full_val,
                effective_s_ptr, effective_s_val, actual_b_size,
                rr_s_partition_sec, rr_probe_sec);
            continue;
        }

        // Only kernel_fusion==1 writes the final S build straight into the interleaved send buffer,
        // which removes a separate pass. The other paths keep the separate arrays because the
        // probe and the PUT loop read them directly.
        int2* s_kv_out = ((use_bf == 0 || use_bf == 5 || use_bf == 7 || use_bf == 9 || use_bf == 10 || use_bf == 13) && kernel_fusion == 1)
                         ? GPUB_send_buffer_kv : nullptr;
        if (use_bf == 10) {
            // R was already sorted before the filter build, so only the S candidates are partitioned.
            auto t_part0 = std::chrono::steady_clock::now();
            partial_jointable_gpu_partition_s_only(
                    effective_s_ptr, GPU_OB_split,
                    effective_s_val, GPU_OB_val_split,
                    s_kv_out,
                    GPU_B_prefix_split, GPU_block_index_split_S[me], GPU_B_offset,
                    GPU_second_index_split_S[me], GPU_answer_num,
                    actual_b_size, gpu_stream);
            first_pass_partition_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_part0).count();
        } else if (partition_rs_split) {
            partial_jointable_gpu_partition_rs_split(
                    GPUA_split, effective_s_ptr, GPU_OA_split, GPU_OB_split,
                    GPUA_val_split, effective_s_val, GPU_OA_val_split, GPU_OB_val_split,
                    s_kv_out,
                    GPU_A_prefix_split, GPU_B_prefix_split,
                    GPU_block_index_split_R, GPU_block_index_split_S[me], GPU_A_offset, GPU_B_offset,
                    GPU_second_index_split_R, GPU_second_index_split_S[me], GPU_answer_num,
                    tableA_size[me], actual_b_size, gpu_stream, gpu_event,
                    first_pass_partition_sec, cumulative_partition_r_sec, cumulative_partition_s_sec,
                    cumulative_partition_hist_r_sec, cumulative_partition_hist_s_sec, /*skip_r_hist=*/(use_bf == 9));
        } else {
            partial_jointable_gpu_partition(
                    GPUA_split, effective_s_ptr, GPU_OA_split, GPU_OB_split,
                    GPUA_val_split, effective_s_val, GPU_OA_val_split, GPU_OB_val_split,
                    s_kv_out,
                    GPU_A_prefix_split, GPU_B_prefix_split,
                    GPU_block_index_split_R, GPU_block_index_split_S[me], GPU_A_offset, GPU_B_offset,
                    GPU_second_index_split_R, GPU_second_index_split_S[me], GPU_answer_num,
                    tableA_size[me], actual_b_size, gpu_stream, gpu_event,
                    first_pass_partition_sec, cumulative_partition_hist_sec, /*skip_r_hist=*/(use_bf == 9));
        }
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "[PE%d] CUDA error after partition: %s\n", me, cudaGetErrorString(err));
            fflush(stderr);
        }

        // Publishes the S index arrays to every GPU so the cross GPU probe knows how much to fetch.
        {
            auto t_shuffle_start = std::chrono::steady_clock::now();
            for (int dst = 0; dst < device_num; dst++) {
                if (dst == me) continue;
                nvshmem_putmem(GPU_block_index_split_S[me],  GPU_block_index_split_S[me],
                            sizeof(long long) * partition_size, dst);
                nvshmem_putmem(GPU_second_index_split_S[me], GPU_second_index_split_S[me],
                            sizeof(long long) * partition_size * second_partition_size, dst);
            }
            nvshmem_quiet();
            nvshmem_barrier_all();
            double this_shuffle_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_shuffle_start).count();
            cumulative_shuffle_sec += this_shuffle_sec;
            //printf("[PE%d] Index shuffle: %.4f sec\n", me, this_shuffle_sec);
        }

        join_execute_host(jctx, effective_s_ptr, effective_s_val, actual_b_size);
    }
    if (jctx.d_s_block_ptrs)  cudaFree(jctx.d_s_block_ptrs);
    if (jctx.d_s_second_ptrs) cudaFree(jctx.d_s_second_ptrs);
    if (jctx.d_phase_timers)  cudaFree(jctx.d_phase_timers);
    cudaStreamDestroy(send_stream);

    



    // Collects the answer counts of every PE on PE 0.
    auto t_agg0 = std::chrono::steady_clock::now();
    nvshmem_int_p(&d_global_answer_counts[me], total_answer_num, 0);
    nvshmem_barrier_all();
    if (me == 0) {
        int* h_counts = new int[device_num];
        cudaMemcpy(h_counts, d_global_answer_counts, sizeof(int) * device_num, cudaMemcpyDeviceToHost);
        int grand_total = 0;
        for (int i = 0; i < device_num; i++) {
            printf("[PE0] PE%d answer count: %d\n", i, h_counts[i]);
            grand_total += h_counts[i];
        }
        printf("=== Grand Total Answer Count (all PEs): %d ===\n", grand_total);
        delete[] h_counts;
    }
    double agg_barrier_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_agg0).count();
    //printf("[PE%d] answer aggregation barrier: %.4f sec\n", me, agg_barrier_sec);

    /*for(int i=0;i<device_num;i++)
    {
        CUDACHECK(cudaSetDevice(i));
        temp_check(recvbuff[i],count,i*count,gpu_stream,gpu_event[i]);
    }

    sync_device_streams(device_num,gpu_stream);*/

    auto fill_end = std::chrono::steady_clock::now();
    //diff = fill_end-fill_start;
    //std::cout << "Total took " << diff.count() << " sec"<< std::endl;
    diff = fill_end-pre_nvshmem;
    //std::cout << "Total including NVSHMEM init took " << diff.count() << " sec"<< std::endl;

    bool check_all=true;

    // ── BF candidate rate ──
    if (use_bf > 0 && pf_input_total > 0)
        printf("[PE%d] BF candidate: %lld / %lld -> %.1f%%\n",
               me, pf_survivors_total, pf_input_total,
               100.0 * pf_survivors_total / pf_input_total);

    // PE 0 collects the timings of every PE and reports the max and mean of each entry.
    if (check_all)
    {
        const char* use_bf_str = (use_bf == 0) ? "0(off)" :
                                 (use_bf == 1) ? "1(global+peer)" :
                                 (use_bf == 2) ? "2(global only)" :
                                 (use_bf == 3) ? "3(late+peer)" :
                                 (use_bf == 4) ? "4(global only, hub-and-spoke)" :
                                 (use_bf == 5) ? "5(global only, ring allreduce)" :
                                 (use_bf == 6) ? "6(hub-and-spoke + CPU OR  )" :
                                 (use_bf == 7) ? "7(ring allreduce, in-kernel fused)" :
                                 (use_bf == 8) ? "8(pull-based in-kernel)" :
                                 (use_bf == 9) ? "9(fused BF-insert + R-hist, ring allreduce)" :
                                 (use_bf == 10) ? (use_v2_merge ? "14(partitioned BF + v2 chunked-stream allreduce)" : "10(partitioned BF)") :
                                 (use_bf == 13) ? ((bf_nccl_mode == 15) ? "15(global only, NCCL allgather + local OR)" :
                                                   (bf_nccl_mode == 16) ? "16(global only, NCCL byte-expand + max)" :
                                                                          "13(global only, nvshmem or_reduce)") :
                                                 "unknown";
        const bool has_early_bf = (use_bf == 1 || use_bf == 2 || use_bf == 4 ||
                                   use_bf == 5 || use_bf == 6 || use_bf == 7 || use_bf == 8 || use_bf == 9 ||
                                   use_bf == 10 || use_bf == 13);
        const bool has_distribute = (use_bf == 4 || use_bf == 5 || use_bf == 6 || use_bf == 7 || use_bf == 9 || use_bf == 10 || use_bf == 13);
        const bool fused_path = ((use_bf == 0 || use_bf == 5 || use_bf == 7 || use_bf == 9 || use_bf == 10 || use_bf == 13) && kernel_fusion > 0 && !r_shuffle_enable && !r_replicate_enable);
        // The alternative join paths leave the broadcast entries at zero, so they are hidden.
        const bool alt_join_path = (r_shuffle_enable || r_replicate_enable);

        double first_pass_net = first_pass_partition_sec + first_pass_probe_sec;
        double accounted = bf_build_total_sec + bf_exchange_loop_sec
                         + bf_prefilter_sec + cumulative_r_h2d_sec + cumulative_pre_bf_barrier_sec
                         + cumulative_h2d_sec + first_pass_net
                         + cumulative_shuffle_sec
                         + cumulative_redist_plan_sec + cumulative_redist_exchange_sec
                         + cumulative_bf_exchange_sec + cumulative_bf_pipeline_sec
                         + cumulative_fused_wall_sec
                         + cumulative_fused_setup_sec
                         + cumulative_pre_send_barrier_sec + cumulative_inner_barrier_sec
                         + cumulative_transfer_sec + cumulative_second_pass_sec
                         + rs_r_gather_sec + rs_s_partition_sec + rs_s_gather_sec + rs_probe_sec
                         + rr_r_gather_sec + rr_s_partition_sec + rr_probe_sec
                         + agg_barrier_sec;
        double total_sec = std::chrono::duration<double>(fill_end - fill_start).count();

        // Every PE builds the same entries in the same order, so the indices line up across PEs.
        std::vector<std::pair<const char*, double>> M;
        auto add = [&](const char* l, double v) { M.emplace_back(l, v); };

        add("  [-1] NFS->CPU R read",   nfs_r_sec);
        add("  [-1] NFS->CPU S read",   nfs_s_sec);
        add("  [0a] R H2D copy",        cumulative_r_h2d_sec);
        add("  [0b] S H2D copy",        cumulative_h2d_sec);
        if (use_bf > 0)  add("  [0c] BF build", bf_build_total_sec);
        if (has_early_bf) {
            add("  [0d] BF early exch+merge", bf_exchange_loop_sec);
            add("    [0d-1] exchange",        bf_exchange_sec);
            add("    [0d-2] merge",           bf_merge_sec);
            if (has_distribute) add("    [0d-3] distribute", bf_distribute_sec);
        }
        if (has_early_bf) add("  [0e] S global pre-filter", bf_prefilter_sec);
        if (use_bf > 0)   add("  [0f] pre-BF barrier",      cumulative_pre_bf_barrier_sec);
        if (redist_active) {
            add("  [1] S' redistribute", cumulative_redist_plan_sec + cumulative_redist_exchange_sec);
            add("    [1-1] Plan",        cumulative_redist_plan_sec);
            add("    [1-2] Exchange",    cumulative_redist_exchange_sec);
        }
        if (!alt_join_path) {
        add("  [2] Local first pass",  first_pass_net);
        add("    [2-1] Partition build", first_pass_partition_sec);
        if (partition_rs_split) {
            add("      [2-1r] R-side",        cumulative_partition_r_sec);
            add("      [2-1s] S-side",        cumulative_partition_s_sec);
            add("      [2-1hr] Histogram R",  cumulative_partition_hist_r_sec);
            add("      [2-1hs] Histogram S",  cumulative_partition_hist_s_sec);
        } else {
            add("      [2-1h] Histogram",     cumulative_partition_hist_sec);
        }
        add("    [2-s] Index shuffle",   cumulative_shuffle_sec);
        if (!fused_path) {
            add("    [2-2] Probe",           first_pass_probe_sec);
            add("      [2-2a1] Preamble(idx)", cumulative_fp_preamble_sec);
            add("      [2-2a2] HT memset",     cumulative_fp_memset_sec);
            add("      [2-2a3] Init barrier",  cumulative_fp_initbar_sec);
            add("      [2-2b] R build",        cumulative_fp_build_sec);
            add("      [2-2c] Local S probe",  cumulative_fp_local_sec);
            add("      [2-2d] Answer reduce",  cumulative_fp_reduce_sec);
        }
        }
        if (r_shuffle_enable) {
            if (use_bf == 10) add("  [2-1] R partition (pre-BF)", first_pass_net);
            add("  [RS] R gather+repart (1x)", rs_r_gather_sec);
            add("  [RS-1] S' (re)partition",   rs_s_partition_sec);
            add("  [RS-2] S' gather (xfer)",   rs_s_gather_sec);
            add("  [RS-3] Local probe",        rs_probe_sec);
        }
        if (r_replicate_enable) {
            if (use_bf == 10) add("  [2-1] R partition (pre-BF)", first_pass_net);
            add("  [RR] R allgather+part (1x)", rr_r_gather_sec);
            add("  [RR-1] S partition",         rr_s_partition_sec);
            add("  [RR-2] Local probe",         rr_probe_sec);
        }
        if (use_bf == 3)                 add("  [3] BF exchange",       cumulative_bf_exchange_sec);
        if (use_bf == 1 || use_bf == 3)  add("  [4] BF filter pipeline", cumulative_bf_pipeline_sec);
        if (fused_path) {
            add("  [4] Fused kernel",      cumulative_fused_wall_sec);
            add("    [4-a] HT init+R ins", cumulative_fused_ht_sec);
            add("    [4-b] Local S probe", cumulative_fused_local_sec);
            add("    [4-c1] NVSHMEM GET",  cumulative_fused_get_sec);
            add("    [4-c2] Remote probe", cumulative_fused_remote_probe_sec);
            add("    [4-s] Setup/teardown", cumulative_fused_setup_sec);
        }
        int idx_transfer = -1, idx_second = -1;
        if (!alt_join_path) {
            add("  [5] Pre-send barrier",  cumulative_pre_send_barrier_sec);
            add("  [5b] Inner barrier",    cumulative_inner_barrier_sec);
            // These two are usually zero on the fused path, so their indices are recorded and
            // PE 0 decides whether to print them.
            idx_transfer = (int)M.size(); add("  [6] Data transfer",  cumulative_transfer_sec);
            idx_second   = (int)M.size(); add("  [7] Cross-GPU join", cumulative_second_pass_sec);
        }
        add("  [8] Answer agg barrier", agg_barrier_sec);
        add("  Accounted",             accounted);
        add("  Unaccounted",           total_sec - accounted);
        add("  Total (start->end)",    total_sec);

        const int NM = (int)M.size();
        const int MAXM = 64;

        double* d_all_timings = (double*)nvshmem_malloc(sizeof(double) * device_num * MAXM);
        for (int i = 0; i < NM && i < MAXM; i++)
            nvshmem_double_p(&d_all_timings[me * MAXM + i], M[i].second, 0);
        nvshmem_barrier_all();

        if (me == 0) {
            double* h = new double[device_num * MAXM];
            cudaMemcpy(h, d_all_timings, sizeof(double) * device_num * MAXM, cudaMemcpyDeviceToHost);

            std::vector<double> vmax(NM), vmean(NM);
            for (int i = 0; i < NM && i < MAXM; i++) {
                double mx = h[0 * MAXM + i], sum = 0.0;
                for (int p = 0; p < device_num; p++) {
                    double v = h[p * MAXM + i];
                    if (v > mx) mx = v;
                    sum += v;
                }
                vmax[i]  = mx;
                vmean[i] = sum / device_num;
            }
            // Percentages are taken against the last entry, which is the total.
            double tot_max  = vmax[NM - 1];
            double tot_mean = vmean[NM - 1];

            printf("\n[PE0] === Timing Summary across %d PEs (use_bf=%s, transport=%s, s_chunks=%d) ===\n",
                   device_num, use_bf_str, use_nvlink ? "NVLink" : "PCIe", s_chunks);
            printf("[PE0]   partition_size=%d second_partition_size=%d\n", partition_size, second_partition_size);
            if (!part_size_warn.empty())
                printf("[PE0]   %s\n", part_size_warn.c_str());
            printf("[PE0] %-28s %11s %8s %11s %8s\n",
                   "metric", "max(s)", "max%", "mean(s)", "mean%");
            for (int i = 0; i < NM && i < MAXM; i++) {
                if (fused_path && (i == idx_transfer || i == idx_second) && vmax[i] <= 0.0)
                    continue;
                // The first two entries are the input reads, which are excluded from Total.
                bool skip_pct = (i < 2);
                if (skip_pct) {
                    printf("[PE0] %-28s %11.4f %8s %11.4f %8s\n",
                           M[i].first, vmax[i], "-", vmean[i], "-");
                } else {
                    double mxp = (tot_max  > 0.0) ? 100.0 * vmax[i]  / tot_max  : 0.0;
                    double mnp = (tot_mean > 0.0) ? 100.0 * vmean[i] / tot_mean : 0.0;
                    printf("[PE0] %-28s %11.4f %7.1f%% %11.4f %7.1f%%\n",
                           M[i].first, vmax[i], mxp, vmean[i], mnp);
                }
            }
            printf("[PE0] ================================================\n\n");
            delete[] h;
        }
        nvshmem_barrier_all();
        nvshmem_free(d_all_timings);
    }

    if(me==0 && check_all == false)
    {
        // ── Timing summary ──────────────────────────────────────────
        double first_pass_net = first_pass_partition_sec + first_pass_probe_sec;
        const char* use_bf_str = (use_bf == 0) ? "0(off)" :
                                 (use_bf == 1) ? "1(global+peer)" :
                                 (use_bf == 2) ? "2(global only)" :
                                 (use_bf == 3) ? "3(late+peer)" :
                                 (use_bf == 4) ? "4(global only, hub-and-spoke)" :
                                 (use_bf == 5) ? "5(global only, ring allreduce)" :
                                 (use_bf == 6) ? "6(hub-and-spoke + CPU OR  )" :
                                 (use_bf == 7) ? "7(ring allreduce, in-kernel fused)" :
                                 (use_bf == 8) ? "8(pull-based in-kernel)" :
                                 (use_bf == 9) ? "9(fused BF-insert + R-hist, ring allreduce)" :
                                 (use_bf == 10) ? (use_v2_merge ? "14(partitioned BF + v2 chunked-stream allreduce)" : "10(partitioned BF)") :
                                 (use_bf == 13) ? ((bf_nccl_mode == 15) ? "15(global only, NCCL allgather + local OR)" :
                                                   (bf_nccl_mode == 16) ? "16(global only, NCCL byte-expand + max)" :
                                                                          "13(global only, nvshmem or_reduce)") :
                                                 "unknown";
        const bool has_early_bf = (use_bf == 1 || use_bf == 2 || use_bf == 4 ||
                                   use_bf == 5 || use_bf == 6 || use_bf == 7 || use_bf == 8 || use_bf == 9 ||
                                   use_bf == 10 || use_bf == 13);
        const bool has_distribute = (use_bf == 4 || use_bf == 5 || use_bf == 6 || use_bf == 7 || use_bf == 9 || use_bf == 10 || use_bf == 13);
        printf("\n[PE%d] === Timing Summary (use_bf=%s, transport=%s, s_chunks=%d) ===\n", me, use_bf_str, use_nvlink ? "NVLink" : "PCIe", s_chunks);
        printf("[PE%d]   partition_size=%d second_partition_size=%d\n", me, partition_size, second_partition_size);
        if (!part_size_warn.empty())
            printf("[PE%d]   %s\n", me, part_size_warn.c_str());
        printf("[PE%d]  [-1] NFS->CPU R read (excluded): %.4f sec (%.2f MB/s)\n", me, nfs_r_sec, R * sizeof(int) / nfs_r_sec / 1e6);
        printf("[PE%d]  [-1] NFS->CPU S read (excluded): %.4f sec (%.2f MB/s)\n", me, nfs_s_sec, S * sizeof(int) / nfs_s_sec / 1e6);
        printf("[PE%d]  ---- everything below is part of Total ----------\n", me);
        printf("[PE%d]  [0a] R H2D copy:                %.4f sec\n", me, cumulative_r_h2d_sec);
        printf("[PE%d]  [0b] S H2D copy (all chunks):   %.4f sec\n", me, cumulative_h2d_sec);
        if (use_bf > 0)
            printf("[PE%d]  [0c] BF build:                  %.4f sec\n", me, bf_build_total_sec);
        if (has_early_bf) {
            printf("[PE%d]  [0d]   BF early exchange+merge:  %.4f sec\n", me, bf_exchange_loop_sec);
            printf("[PE%d]    [0d-1] exchange (send/gather): %.4f sec\n", me, bf_exchange_sec);
            printf("[PE%d]    [0d-2] merge (OR-reduce):      %.4f sec\n", me, bf_merge_sec);
            if (has_distribute)
                printf("[PE%d]    [0d-3] distribute (broadcast): %.4f sec\n", me, bf_distribute_sec);
        }
        if (use_bf == 7 && !bf_fused_cycles.empty()) {
            int num_steps = (int)(bf_fused_cycles.size() / 5);
            long long all_cyc = 0;
            long long agg[5] = {0,0,0,0,0};
            for (int s = 0; s < num_steps; s++)
                for (int p = 0; p < 5; p++) {
                    agg[p] += bf_fused_cycles[s*5 + p];
                    all_cyc += bf_fused_cycles[s*5 + p];
                }
            double inv_all = (all_cyc > 0) ? (bf_merge_sec / (double)all_cyc) : 0.0;
            const char* lbl[5] = {"put      ", "gs+q+bar ", "gsync_a  ", "or/copy  ", "gsync_b  "};
            printf("[PE%d]    [0d-F] Fused kernel phase breakdown (block 0, %d steps):\n",
                   me, num_steps);
            int n_rs = (num_steps / 2);
            for (int s = 0; s < num_steps; s++) {
                long long step_total = 0;
                for (int p = 0; p < 5; p++) step_total += bf_fused_cycles[s*5 + p];
                const char* phase = (s < n_rs) ? "RS" : "AG";
                const char* op    = (s < n_rs) ? "or" : "cp";
                printf("[PE%d]      step %d (%s): put=%.4f  gs+q+bar=%.4f  gsync_a=%.4f  %s=%.4f  gsync_b=%.4f  | total=%.4f ms\n",
                       me, s, phase,
                       bf_fused_cycles[s*5+0] * inv_all * 1000.0,
                       bf_fused_cycles[s*5+1] * inv_all * 1000.0,
                       bf_fused_cycles[s*5+2] * inv_all * 1000.0,
                       op,
                       bf_fused_cycles[s*5+3] * inv_all * 1000.0,
                       bf_fused_cycles[s*5+4] * inv_all * 1000.0,
                       step_total * inv_all * 1000.0);
            }
            printf("[PE%d]    [0d-F] Aggregate (sums to kernel = %.4f sec):\n", me, bf_merge_sec);
            for (int p = 0; p < 5; p++) {
                double secs = (double)agg[p] * inv_all;
                double pct  = (all_cyc > 0) ? (100.0 * agg[p] / all_cyc) : 0.0;
                printf("[PE%d]      %s: %.4f sec  (%.1f%%)\n", me, lbl[p], secs, pct);
            }
        }
        if (use_bf == 8 && !bf_fused_cycles.empty()) {
            int num_steps = (int)(bf_fused_cycles.size() / 2);
            long long all_cyc = 0;
            long long agg_get = 0, agg_or = 0;
            for (int s = 0; s < num_steps; s++) {
                agg_get += bf_fused_cycles[s*2 + 0];
                agg_or  += bf_fused_cycles[s*2 + 1];
            }
            all_cyc = agg_get + agg_or;
            double inv_all = (all_cyc > 0) ? (bf_merge_sec / (double)all_cyc) : 0.0;
            printf("[PE%d]    [0d-P] Pull kernel phase breakdown (block 0, %d peers):\n",
                   me, num_steps);
            for (int s = 0; s < num_steps; s++) {
                int src_pe = (me + s + 1) % device_num;  // matches kernel's iteration order (skips me)
                if (src_pe == me) src_pe = (src_pe + 1) % device_num;
                long long step_total = bf_fused_cycles[s*2 + 0] + bf_fused_cycles[s*2 + 1];
                printf("[PE%d]      step %d: get=%.4f ms  or=%.4f ms  | total=%.4f ms\n",
                       me, s,
                       bf_fused_cycles[s*2 + 0] * inv_all * 1000.0,
                       bf_fused_cycles[s*2 + 1] * inv_all * 1000.0,
                       step_total * inv_all * 1000.0);
            }
            printf("[PE%d]    [0d-P] Aggregate (sums to kernel = %.4f sec):\n", me, bf_merge_sec);
            double gs = (double)agg_get * inv_all;
            double os = (double)agg_or  * inv_all;
            double gp = (all_cyc > 0) ? (100.0 * agg_get / all_cyc) : 0.0;
            double op = (all_cyc > 0) ? (100.0 * agg_or  / all_cyc) : 0.0;
            printf("[PE%d]      get     : %.4f sec  (%.1f%%)\n", me, gs, gp);
            printf("[PE%d]      or      : %.4f sec  (%.1f%%)\n", me, os, op);
        }
        if (has_early_bf)
            printf("[PE%d]  [0e] S global pre-filter (all chunks): %.4f sec\n", me, bf_prefilter_sec);
        if (use_bf > 0)
            printf("[PE%d]  [0f] pre-BF barrier:            %.4f sec\n", me, cumulative_pre_bf_barrier_sec);
        if (redist_active) {
            printf("[PE%d]  [1] S' redistribute (all):      %.4f sec\n", me, cumulative_redist_plan_sec + cumulative_redist_exchange_sec);
            printf("[PE%d]    [1-1]  Plan (size gather):    %.4f sec\n", me, cumulative_redist_plan_sec);
            printf("[PE%d]    [1-2]  Exchange:              %.4f sec\n", me, cumulative_redist_exchange_sec);
        }
        printf("[PE%d]  [2] Local first pass (net):     %.4f sec\n", me, first_pass_net);
        printf("[PE%d]    [2-1]  Partition build:       %.4f sec\n", me, first_pass_partition_sec);
        if (partition_rs_split) {
            printf("[PE%d]      [2-1r] R-side:          %.4f sec\n", me, cumulative_partition_r_sec);
            printf("[PE%d]      [2-1s] S-side:          %.4f sec\n", me, cumulative_partition_s_sec);
            printf("[PE%d]      [2-1hr] Histogram R:    %.4f sec\n", me, cumulative_partition_hist_r_sec);
            printf("[PE%d]      [2-1hs] Histogram S:    %.4f sec\n", me, cumulative_partition_hist_s_sec);
        } else {
            printf("[PE%d]      [2-1h] Histogram (R+S): %.4f sec\n", me, cumulative_partition_hist_sec);
        }
        printf("[PE%d]    [2-s]  Index shuffle (all):   %.4f sec\n", me, cumulative_shuffle_sec);
        if (!((use_bf == 0 || use_bf == 5 || use_bf == 7 || use_bf == 9 || use_bf == 10 || use_bf == 13) && kernel_fusion > 0)) {
            printf("[PE%d]    [2-2]  Probe:                 %.4f sec%s\n", me, first_pass_probe_sec, total_overflow_count > 0 ? " [!] overflow → linear fallback used" : "");
            {
                double fp_tot = cumulative_fp_preamble_sec + cumulative_fp_memset_sec
                              + cumulative_fp_initbar_sec + cumulative_fp_build_sec
                              + cumulative_fp_local_sec + cumulative_fp_reduce_sec;
                if (fp_tot > 0.0) {
                    printf("[PE%d]      [2-2a1] Preamble(idx):   %.4f sec  (%.1f%%)  [overhead]\n",
                           me, cumulative_fp_preamble_sec, 100.0 * cumulative_fp_preamble_sec / fp_tot);
                    printf("[PE%d]      [2-2a2] HT memset(pure): %.4f sec  (%.1f%%)\n",
                           me, cumulative_fp_memset_sec, 100.0 * cumulative_fp_memset_sec / fp_tot);
                    printf("[PE%d]      [2-2a3] Init barrier:    %.4f sec  (%.1f%%)  [overhead]\n",
                           me, cumulative_fp_initbar_sec, 100.0 * cumulative_fp_initbar_sec / fp_tot);
                    printf("[PE%d]      [2-2b] R build:          %.4f sec  (%.1f%%)\n",
                           me, cumulative_fp_build_sec, 100.0 * cumulative_fp_build_sec / fp_tot);
                    printf("[PE%d]      [2-2c] Local S probe:    %.4f sec  (%.1f%%)\n",
                           me, cumulative_fp_local_sec, 100.0 * cumulative_fp_local_sec / fp_tot);
                    printf("[PE%d]      [2-2d] Answer reduce:    %.4f sec  (%.1f%%)\n",
                           me, cumulative_fp_reduce_sec, 100.0 * cumulative_fp_reduce_sec / fp_tot);
                }
            }
        }
        if (use_bf == 3)
            printf("[PE%d]  [3] BF exchange (all chunks):   %.4f sec\n", me, cumulative_bf_exchange_sec);
        if (use_bf == 1 || use_bf == 3)
            printf("[PE%d]  [4] BF filter pipeline (all):  %.4f sec\n", me, cumulative_bf_pipeline_sec);
        if ((use_bf == 0 || use_bf == 5 || use_bf == 7 || use_bf == 9 || use_bf == 10 || use_bf == 13) && kernel_fusion > 0) {
            printf("[PE%d]  [4] Fused kernel (all):         %.4f sec\n", me, cumulative_fused_wall_sec);
            printf("[PE%d]    [4-a] HT init+R insert:       %.4f sec  (%.1f%%)\n", me, cumulative_fused_ht_sec,           cumulative_fused_wall_sec > 0 ? 100.0*cumulative_fused_ht_sec/cumulative_fused_wall_sec : 0.0);
            printf("[PE%d]    [4-b] Local S probe:          %.4f sec  (%.1f%%)\n", me, cumulative_fused_local_sec,        cumulative_fused_wall_sec > 0 ? 100.0*cumulative_fused_local_sec/cumulative_fused_wall_sec : 0.0);
            printf("[PE%d]    [4-c1] NVSHMEM GET:           %.4f sec  (%.1f%%)\n", me, cumulative_fused_get_sec,          cumulative_fused_wall_sec > 0 ? 100.0*cumulative_fused_get_sec/cumulative_fused_wall_sec : 0.0);
            printf("[PE%d]    [4-c2] Remote probe:          %.4f sec  (%.1f%%)\n", me, cumulative_fused_remote_probe_sec, cumulative_fused_wall_sec > 0 ? 100.0*cumulative_fused_remote_probe_sec/cumulative_fused_wall_sec : 0.0);
            printf("[PE%d]    [4-s] Setup/teardown overhead:%.4f sec  \n",
                   me, cumulative_fused_setup_sec);
        }
        printf("[PE%d]  [5] Pre-send barrier (all):     %.4f sec\n", me, cumulative_pre_send_barrier_sec);
        printf("[PE%d]  [5b] Inner barrier (all):       %.4f sec\n", me, cumulative_inner_barrier_sec);
        if (!((use_bf == 0 || use_bf == 5 || use_bf == 7 || use_bf == 9 || use_bf == 10 || use_bf == 13) && kernel_fusion > 0)
            || cumulative_transfer_sec > 0.0)
            printf("[PE%d]  [6] Data transfer+delivery (all):%.4f sec\n", me, cumulative_transfer_sec);
        // The fused kernel absorbs the cross GPU join, so this is usually zero there. The hybrid
        // path still fills it through its PUT loop, hence the conditional print.
        if (!((use_bf == 0 || use_bf == 5 || use_bf == 7 || use_bf == 9 || use_bf == 10 || use_bf == 13) && kernel_fusion > 0)
            || cumulative_second_pass_sec > 0.0)
            printf("[PE%d]  [7] Cross-GPU join (all):       %.4f sec\n", me, cumulative_second_pass_sec);
        printf("[PE%d]  [8] Answer aggregation barrier: %.4f sec\n", me, agg_barrier_sec);
        {
            double accounted = bf_build_total_sec + bf_exchange_loop_sec
                             + bf_prefilter_sec + cumulative_r_h2d_sec + cumulative_pre_bf_barrier_sec
                             + cumulative_h2d_sec + first_pass_net
                             + cumulative_shuffle_sec
                             + cumulative_redist_plan_sec + cumulative_redist_exchange_sec
                             + cumulative_bf_exchange_sec + cumulative_bf_pipeline_sec
                             + cumulative_fused_wall_sec
                             + cumulative_fused_setup_sec
                             + cumulative_pre_send_barrier_sec + cumulative_inner_barrier_sec
                             + cumulative_transfer_sec + cumulative_second_pass_sec
                             + agg_barrier_sec;
            double total_sec = std::chrono::duration<double>(fill_end - fill_start).count();
            printf("[PE%d]  Accounted:                    %.4f sec\n", me, accounted);
            printf("[PE%d]  Unaccounted (overhead etc.):  %.4f sec\n", me, total_sec - accounted);
            printf("[PE%d]  Total (fill_start→end):       %.4f sec\n", me, total_sec);
        }
        if (total_overflow_count > 0)
            printf("[PE%d]  [!] Probe overflow blocks (linear fallback): %d\n", me, total_overflow_count);
        { int h_widx = 0; cudaMemcpy(&h_widx, GPU_answer_write_idx, sizeof(int), cudaMemcpyDeviceToHost);
          if ((unsigned)h_widx > (unsigned)split_size)
              printf("[PE%d]  [!] Answer buffer truncated: %d written, capacity %lld (true count in answer_num)\n",
                     me, h_widx, split_size); }
        printf("[PE%d] ================================================\n\n", me);
    }

    
    /*ncclComm_t comms[2];
    int nDev = 2;
    int devs[2] = { 0, 1};
    int size=S;
    int** sendbuff = (int**)malloc(nDev * sizeof(int*));
    int** recvbuff = (int**)malloc(nDev * sizeof(int*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(int)));
        CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(int)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(int)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(int)));
        CUDACHECK(cudaStreamCreate(s+i));
    }


    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    
    auto nccl_start = std::chrono::steady_clock::now();
    NCCLCHECK(ncclGroupStart());

    for (int me = 0; me < nDev; ++me) {
        int send = me;
        int recv = (me + 1) % nDev;
        
        NCCLCHECK(ncclSend(sendbuff[send], size, ncclInt32, recv, comms[send], s[send]));
        NCCLCHECK(ncclRecv(recvbuff[send], size, ncclInt32, (send - 1 + nDev) % nDev, comms[send], s[send]));
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }
    auto nccl_end = std::chrono::steady_clock::now();
    diff = nccl_end-nccl_start;
    std::cout << "NCCL sending took " << diff.count() << " sec"<< std::endl;*/
    

    // Cleanup
    /*for (int i = 0; i < device_num; i++) {
        ncclCommDestroy(comms[i]);
        cudaStreamDestroy(gpu_stream);
    }*/

    if (h_pcie_stage)      cudaFreeHost(h_pcie_stage);
    if (h_pcie_stage_val)  cudaFreeHost(h_pcie_stage_val);
    if (h_bf_cpu_or) {
        size_t shm_size = sizeof(uint32_t) * bf_words * device_num;
        cudaHostUnregister(h_bf_cpu_or);
        munmap(h_bf_cpu_or, shm_size);
        if (me == 0) shm_unlink("/bf_cpu_or_shmem");
    }
    if (bf_v2)             bf_v2_destroy(bf_v2);
    if (bf_nccl)           bf_nccl_destroy(bf_nccl);
    // On the v2 path d_pbf aliases the accumulator, which bf_v2_destroy already released.
    if (d_pbf && !use_v2_merge) cudaFree(d_pbf);
    if (d_pbf_symmetric)   nvshmem_free(d_pbf_symmetric);
    if (h_block_index_r)   delete[] h_block_index_r;
    if (d_prefiltered_S)   cudaFree(d_prefiltered_S);
    if (d_prefiltered_S_val) cudaFree(d_prefiltered_S_val);
    if (d_bf_global)       cudaFree(d_bf_global);
    if (d_bf_symmetric)    nvshmem_free(d_bf_symmetric);
    if (d_bf_recv_scratch) cudaFree(d_bf_recv_scratch);
    if (d_bf_phase_times)  cudaFree(d_bf_phase_times);
    if (d_compact_counter) cudaFree(d_compact_counter);
    if (d_sizes_sym)       nvshmem_free(d_sizes_sym);
    if (R_gather_key)      nvshmem_free(R_gather_key);
    if (R_gather_val)      nvshmem_free(R_gather_val);
    if (r_shuffle_enable)  r_shuffle_free();
    if (R_full_key)         nvshmem_free(R_full_key);
    if (R_full_val)         nvshmem_free(R_full_val);
    if (R_full_scratch)     cudaFree(R_full_scratch);
    if (R_full_scratch_val) cudaFree(R_full_scratch_val);
    nvshmem_free(d_global_answer_counts);
    for (int i = 0; i < device_num; i++) {
        nvshmem_free(GPU_block_index_split_S[i]);
        nvshmem_free(GPU_second_index_split_S[i]);
    }
    delete[] GPU_block_index_split_S;
    delete[] GPU_second_index_split_S;

    cudaFreeHost(tableA);
    cudaFreeHost(tableB);
    cudaFreeHost(tableA_val);
    cudaFreeHost(tableB_val);
    delete[] tableA_val_split;
    delete[] tableB_val_split;

    return;
}
