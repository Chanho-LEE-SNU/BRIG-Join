#ifndef BF_ALLREDUCE_H
#define BF_ALLREDUCE_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include "bloom.cuh"

// ORs src into dst word by word.
__global__ void bf_or_kernel(uint32_t* dst, const uint32_t* src, size_t num_words);

// Ring OR allreduce over an arbitrary buffer. d_buf must already hold the local data.
void bf_ring_or_allreduce(
    int me, int device_num, uint32_t* d_buf, uint32_t* d_sym, size_t words,
    double& out_transfer_sec, double& out_merge_sec, double& out_allgather_sec);

struct BfV2;

// Creates the buffers and streams for the chunked stream ring allreduce. This is a collective call.
BfV2* bf_v2_create(size_t words, size_t chunk_bytes, int nstream);

// Copies d_buf into the accumulator, merges it and copies the result back.
void  bf_v2_merge(BfV2* h, uint32_t* d_buf, size_t words);

// Releases everything allocated by bf_v2_create. This is a collective call.
void  bf_v2_destroy(BfV2* h);

// Returns the internal accumulator, which can be used directly as the filter buffer.
uint32_t* bf_v2_acc(BfV2* h);

// Merges in place on the accumulator so no copy is needed.
void      bf_v2_merge_inplace(BfV2* h);

// Returns the grid size of the pull allreduce kernel.
int bf_pull_grid_size();

// Returns the grid size of the fused ring allreduce kernel.
int bf_fused_grid_size();

struct BfNccl;

// Creates the NCCL communicator and its scratch buffers. This is a collective call.
BfNccl* bf_nccl_create(int me, int np, size_t words, int mode);

// Releases everything allocated by bf_nccl_create. This is a collective call.
void    bf_nccl_destroy(BfNccl* h);

// Merges the per PE Bloom filters using the scheme selected by use_bf.
// All scratch and timer buffers are preallocated by the caller.
void bf_allreduce(
    int me, int device_num, int use_bf,
    std::vector<BloomFilter>& bf, size_t bf_words,
    uint32_t* d_bf_global, uint32_t* d_bf_symmetric, uint32_t* h_bf_cpu_or,
    uint32_t* d_recv_scratch, long long* d_phase_times, int bf_grid_size,
    double& bf_exchange_loop_sec, double& bf_exchange_sec,
    double& bf_merge_sec, double& bf_distribute_sec,
    std::vector<long long>* out_fused_cycles = nullptr,
    BfNccl* nccl = nullptr);

#endif
