#ifndef R_SHUFFLE_H
#define R_SHUFFLE_H

#include <cuda_runtime.h>
#include "join_executor.h"

// Redistributes a partitioned relation to its owner GPU. Returns the received element count, or -1 on failure.
long long all_to_all_to_owner(int* src_key, int* src_val,
                              long long* block_index,
                              int* dst_key, int* dst_val,
                              int me, int D, cudaStream_t stream);

// Preallocates the size matrix buffer used by the all to all exchange.
void r_shuffle_init(int D);

// Releases the buffer allocated by r_shuffle_init.
void r_shuffle_free();

// Gathers R onto its owner GPU and repartitions it. Called once before the chunk loop.
void r_shuffle_gather_r(JoinChunkContext& ctx,
                        int* R_gather_key, int* R_gather_val,
                        int* oa_key, int* oa_val,
                        long long* prefix_R, long long* offset_R,
                        long long tableA_size_me, long long split_size,
                        double& t_r_gather);

// Gathers the S candidates onto their owner, runs the second pass and probes. Returns the match count of this chunk.
int  r_shuffle_join_chunk(JoinChunkContext& ctx,
                          int* R_gather_key, int* R_gather_val,
                          int* effective_s_ptr, int* effective_s_val, long long actual_b_size,
                          long long split_size,
                          double& t_partition, double& t_gather, double& t_probe);

#endif // R_SHUFFLE_H
