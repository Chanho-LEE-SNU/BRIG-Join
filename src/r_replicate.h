#ifndef R_REPLICATE_H
#define R_REPLICATE_H

#include <cuda_runtime.h>
#include "join_executor.h"

// Replicates R to every GPU and partitions it. Called once before the chunk loop.
void r_replicate_gather_r(JoinChunkContext& ctx,
                          int* R_full_key, int* R_full_val,
                          int* scratch_key, int* scratch_val,
                          long long* prefix_R, long long* offset_R,
                          const long long* tableA_size, long long R_total,
                          double& t_r_gather);

// Partitions the local S and probes it against the replicated R. Returns the match count of this chunk.
int r_replicate_join_chunk(JoinChunkContext& ctx,
                           int* R_full_key, int* R_full_val,
                           int* effective_s_ptr, int* effective_s_val, long long actual_b_size,
                           double& t_partition, double& t_probe);

#endif // R_REPLICATE_H
