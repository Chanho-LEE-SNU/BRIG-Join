//
// Created by Chanho LEE on 2023-10-09.
//

#include "kernel_functions.h"
#include "common.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__device__ int d_probe_overflow_count = 0;
__device__ int d_answer_capacity = 0;

// Builds the first level partition histograms of R and S in one kernel.
__global__ void gpu_run_hist_for_both(int *g_adata, long long *g_a_prefix_data, long long a_data_end, int *g_bdata, long long *g_b_prefix_data, long long b_data_end,
                                      long long block_size, int partition_size)
{
    extern  __shared__  int sdata[];
    int * sdata1;
    int * sdata2;

    sdata1= &sdata[0];
    sdata2= &sdata[partition_size];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_number =  gridDim.x;

    long long starting_point = (long long)blockIdx.x * block_size + tid;
    long long j;
    long long a_end = ((long long)blockIdx.x +1) * block_size;
    long long b_end = a_end;
    int thread_size = blockDim.x;



    if(a_end>a_data_end)
    {
        a_end=a_data_end;
    }

    if(b_end>b_data_end)
    {
        b_end=b_data_end;
    }


    for(int i=tid; i<partition_size; i+=thread_size)
    {
        sdata1[i]=0;
        sdata2[i]=0;
    }

    __syncthreads();

#pragma unroll
    for(j=starting_point;j<a_end;j+=thread_size)
    {
        int a_block_position = (unsigned)g_adata[j]%partition_size;
        atomicAdd(&sdata1[a_block_position],1);

    }

#pragma unroll
    for(j=starting_point;j<b_end;j+=thread_size)
    {
        int b_block_position = (unsigned)g_bdata[j]%partition_size;
        atomicAdd(&sdata2[b_block_position],1);
    }
    __syncthreads();



    for(int i=tid; i<partition_size; i+=thread_size)
    {
        atomicAdd((unsigned long long*)&g_a_prefix_data[i], (unsigned long long)sdata1[i]);
        atomicAdd((unsigned long long*)&g_b_prefix_data[i], (unsigned long long)sdata2[i]);
    }


    return;

}



// Scatters R and S into first level bucket order.
__global__ void gpu_make_sorted_table_for_both(int *g_i_a_data, int *g_o_a_data, int *g_i_a_val, int *g_o_a_val, long long *g_a_block_index_split, long long a_data_end, long long *g_a_offset,
                                               int *g_i_b_data, int *g_o_b_data, int *g_i_b_val, int *g_o_b_val, long long *g_b_block_index_split, long long b_data_end, long long *g_b_offset,
                                               long long block_size, int partition_size)
{
    extern  __shared__  int sdata[];

    int tid = threadIdx.x;
    int thread_size = blockDim.x;
    long long starting_point = (long long)blockIdx.x * block_size;
    long long r_end = ((long long)blockIdx.x +1) * block_size;
    long long s_end= r_end;

    if(r_end>a_data_end)
    {
        r_end=a_data_end;
    }
    if(s_end>b_data_end)
    {
        s_end=b_data_end;
    }
    __syncwarp();

#pragma unroll
        for(long long j=starting_point+tid;j<r_end;j+=thread_size)
        {
            int target_a=g_i_a_data[j];
            int block_position_a = (unsigned)target_a%partition_size;
            long long input_position_a = (long long)atomicAdd((unsigned long long*)&g_a_offset[block_position_a], 1ULL);
            g_o_a_data[input_position_a]=target_a;
            g_o_a_val[input_position_a]=g_i_a_val[j];        }
#pragma unroll
        for (long long j = starting_point + tid; j < s_end; j += thread_size) {
            int target_b = g_i_b_data[j];
            int block_position_b = (unsigned)target_b % partition_size;
            long long input_position_b = (long long)atomicAdd((unsigned long long*)&g_b_offset[block_position_b], 1ULL);
            g_o_b_data[input_position_b] = target_b;
            g_o_b_val[input_position_b] = g_i_b_val[j];
        }
}






// Clears the partition counters and the answer counter.
__global__ void gpu_prefix_init(long long * g_a_prefix_data, long long * g_b_prefix_data, int partition_size, int * g_answer_num)
{
    if(threadIdx.x==0) g_answer_num[0]=0;
    for(int i=threadIdx.x; i<partition_size; i+=blockDim.x)
    {
        g_a_prefix_data[i]=0;
        g_b_prefix_data[i]=0;
    }
}

// Clears the partition counters.
__global__ void gpu_prefix_init(long long * g_a_prefix_data, long long * g_b_prefix_data, int partition_size)
{
    for(int i=threadIdx.x; i<partition_size; i+=blockDim.x)
    {
        g_a_prefix_data[i]=0;
        g_b_prefix_data[i]=0;
    }
}


// Counts second level sub buckets within each first level bucket and scans them.
__global__ void gpu_run_hist_second_pass(int *g_idata, int *g_odata, long long *g_block_index_split, long long *g_second_index_split, int second_partition_size)
{
    extern  __shared__  int sdata[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int thread_size = blockDim.x;

    long long block_starting_point=0;

    if(bid!=0)
    {
        block_starting_point = g_block_index_split[bid-1];
    }
    __syncwarp();

    long long block_ending_point = g_block_index_split[bid];

    for(int i=tid; i<second_partition_size; i+=thread_size)
        sdata[i]=0;
    __syncthreads();

    for(long long j=block_starting_point; j+tid<block_ending_point; j+=thread_size)
    {
        int value = (unsigned)g_idata[j+tid]%second_partition_size;
        atomicAdd(&sdata[value], 1);
    }
    __syncthreads();

    int * sdata1 = &sdata[0];
    int * sdata2 = &sdata[second_partition_size];

    for(int step=1; step<second_partition_size; step*=2)
    {
        for(int i=tid; i<second_partition_size; i+=thread_size)
            sdata2[i] = (i>=step) ? sdata1[i] + sdata1[i-step] : sdata1[i];
        __syncthreads();
        int * sdata_change = sdata1; sdata1 = sdata2; sdata2 = sdata_change;
    }

    for(int i=tid; i<second_partition_size; i+=thread_size)
        g_second_index_split[(long long)bid * second_partition_size + i] = (long long)sdata1[i];

    return;
}

// Scatters keys and payloads into second level sub bucket order.
__global__ void gpu_build_table_second_pass(int *g_idata, int *g_odata, int *g_ival, int *g_oval, long long *g_block_index_split, long long *g_second_index_split, int second_partition_size)
{
    extern  __shared__  int sdata[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int thread_size = blockDim.x;

    long long block_starting_point=0;

    if(bid!=0)
    {
        block_starting_point = g_block_index_split[bid-1];
    }
    __syncwarp();

    long long block_ending_point = g_block_index_split[bid];

    if(tid==0) sdata[0]=0;
    for(int i=tid; i<second_partition_size-1; i+=thread_size)
        sdata[i+1]=(int)g_second_index_split[(long long)bid * second_partition_size+i];
    __syncthreads();


#pragma unroll
    for(long long j=block_starting_point;j+tid<block_ending_point;j+=thread_size)
    {
        int target = g_idata[j+tid];
        int block_position = (unsigned)target%second_partition_size;
        int input_offset = atomicAdd(&sdata[block_position],1);
        long long input_position = block_starting_point + input_offset;

        g_odata[input_position]=target;
        g_oval[input_position]=g_ival[j+tid];    }



    return;
}

// Same as gpu_build_table_second_pass but writes (key, value) int2 pairs.
__global__ void gpu_build_table_second_pass_kv(int *g_idata, int *g_ival, int2 *g_okv, long long *g_block_index_split, long long *g_second_index_split, int second_partition_size)
{
    extern  __shared__  int sdata[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int thread_size = blockDim.x;

    long long block_starting_point=0;
    if(bid!=0)
        block_starting_point = g_block_index_split[bid-1];
    __syncwarp();

    long long block_ending_point = g_block_index_split[bid];

    if(tid==0) sdata[0]=0;
    for(int i=tid; i<second_partition_size-1; i+=thread_size)
        sdata[i+1]=(int)g_second_index_split[(long long)bid * second_partition_size+i];
    __syncthreads();

#pragma unroll
    for(long long j=block_starting_point;j+tid<block_ending_point;j+=thread_size)
    {
        int target = g_idata[j+tid];
        int block_position = (unsigned)target%second_partition_size;
        int input_offset = atomicAdd(&sdata[block_position],1);
        long long input_position = block_starting_point + input_offset;
        g_okv[input_position] = make_int2(target, g_ival[j+tid]);
    }
    return;
}


// Interleaves separate key and value arrays into an array of (key, value) pairs.
__global__ void gpu_interleave_kv(const int* __restrict__ keys, const int* __restrict__ vals,
                                  int2* __restrict__ out, long long n)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = make_int2(keys[i], vals[i]);
}

// Loads one R sub bucket into a shared memory hash table and probes it with S.
template<bool COLLECT>
__global__ void gpu_probe_with_second_pass(int *g_oa_data, int *g_ob_data, int *g_oa_val, int *g_ob_val, long long * ga_index, long long * gb_index, long long * ga_second_index, long long * gb_second_index, int partition_size, int second_partition_size, int * g_answer, int * g_answer_val, int * g_answer_s_val, int * g_answer_num, int * g_answer_write_idx, long long * d_phase_timers)
{

    extern  __shared__  int sdata[];

    long long t_entry = 0, t_after_preamble = 0, t_after_memset = 0, t_after_initbar = 0,
              t_after_build = 0, t_after_probe = 0, t_end = 0;
    if (COLLECT && threadIdx.x == 0) t_entry = clock64();



    int tid = threadIdx.x;
    int thread_block_id = blockIdx.x;
    int thread_size = blockDim.x;
    int block_number =  gridDim.x;



    int max_shared= HT_SIZE;

    int bid = thread_block_id/second_partition_size;
    int second_bid = thread_block_id%second_partition_size;


    long long r_block_start = 0;
    long long s_block_start = 0;

    if (bid != 0) {
        r_block_start = ga_index[bid-1];
        s_block_start = gb_index[bid-1];
    }
    __syncwarp();

    int r_block_starting_offset = 0;
    int s_block_starting_offset = 0;
    if (second_bid != 0) {
        r_block_starting_offset = (int)ga_second_index[thread_block_id-1];
        s_block_starting_offset = (int)gb_second_index[thread_block_id-1];
    }
    __syncwarp();

    int r_block_ending_offset = (int)ga_second_index[thread_block_id];
    int s_block_ending_offset = (int)gb_second_index[thread_block_id];

    long long r_starting_point = r_block_start + r_block_starting_offset;
    long long s_starting_point = s_block_start + s_block_starting_offset;

    long long s_ending_point = s_block_start + s_block_ending_offset;





    int answer_num = 0;

    int loop_size = r_block_ending_offset - r_block_starting_offset;

    const int EMPTY = 0x80000000;
    const int ht_size = max_shared;
    int* ht_val = &sdata[max_shared];

    if (loop_size >= ht_size) {
        if (tid == 0) {
            atomicAdd(&d_probe_overflow_count, 1);
        }
        for (long long i = s_starting_point + tid; i < s_ending_point; i += thread_size) {
            int target = g_ob_data[i];
            for (int j = 0; j < loop_size; j++) {
                if (g_oa_data[r_starting_point + j] == target) {
                    int idx = atomicAdd(g_answer_write_idx, 1);
                    if ((unsigned)idx < (unsigned)d_answer_capacity) {
                        g_answer[idx] = target;
                        g_answer_val[idx] = g_oa_val[r_starting_point + j];
                        g_answer_s_val[idx] = g_ob_val[i];
                    }
                    answer_num++;
                    break;
                }
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            answer_num += __shfl_down_sync(0xffffffff, answer_num, offset);
        if ((tid & 31) == 0) sdata[tid >> 5] = answer_num;
        __syncthreads();
        if (tid == 0) {
            int sum = 0;
            for (int i = 0; i < (thread_size + 31) / 32; i++) sum += sdata[i];
            atomicAdd(&g_answer_num[0], sum);
        }
        if (COLLECT && tid == 0) {
            t_end = clock64();
            d_phase_timers[thread_block_id * 7 + 0] = t_entry;
            d_phase_timers[thread_block_id * 7 + 1] = t_entry;
            d_phase_timers[thread_block_id * 7 + 2] = t_entry;
            d_phase_timers[thread_block_id * 7 + 3] = t_entry;
            d_phase_timers[thread_block_id * 7 + 4] = t_entry;
            d_phase_timers[thread_block_id * 7 + 5] = t_end;
            d_phase_timers[thread_block_id * 7 + 6] = t_end;
        }
        return;
    }

    if (COLLECT && tid == 0) t_after_preamble = clock64();

    for (int i = tid; i < ht_size; i += thread_size)
        sdata[i] = EMPTY;
    if (COLLECT && tid == 0) t_after_memset = clock64();
    __syncthreads();
    if (COLLECT && tid == 0) t_after_initbar = clock64();

    for (int i = tid; i < loop_size; i += thread_size) {
        int val = g_oa_data[r_starting_point + i];
        int slot = (unsigned)val % ht_size;
        while (atomicCAS(&sdata[slot], EMPTY, val) != EMPTY)
            slot = (slot + 1) % ht_size;
        ht_val[slot] = g_oa_val[r_starting_point + i];
    }
    __syncthreads();
    if (COLLECT && tid == 0) t_after_build = clock64();

    for (long long i = s_starting_point + tid; i < s_ending_point; i += thread_size) {
        int target = g_ob_data[i];
        int slot = (unsigned)target % ht_size;
        while (sdata[slot] != EMPTY) {
            if (sdata[slot] == target) {
                int idx = atomicAdd(g_answer_write_idx, 1);
                if ((unsigned)idx < (unsigned)d_answer_capacity) {
                    g_answer[idx] = target;
                    g_answer_val[idx] = ht_val[slot];
                    g_answer_s_val[idx] = g_ob_val[i];
                }
                answer_num++;
            }
            slot = (slot + 1) % ht_size;
        }
    }
    __syncthreads();
    if (COLLECT && tid == 0) t_after_probe = clock64();
    for (int offset = 16; offset > 0; offset >>= 1)
        answer_num += __shfl_down_sync(0xffffffff, answer_num, offset);
    if ((tid & 31) == 0) sdata[tid >> 5] = answer_num;
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < (thread_size + 31) / 32; i++) sum += sdata[i];
        atomicAdd(&g_answer_num[0], sum);
    }
    if (COLLECT && tid == 0) {
        t_end = clock64();
        d_phase_timers[thread_block_id * 7 + 0] = t_entry;
        d_phase_timers[thread_block_id * 7 + 1] = t_after_preamble;
        d_phase_timers[thread_block_id * 7 + 2] = t_after_memset;
        d_phase_timers[thread_block_id * 7 + 3] = t_after_initbar;
        d_phase_timers[thread_block_id * 7 + 4] = t_after_build;
        d_phase_timers[thread_block_id * 7 + 5] = t_after_probe;
        d_phase_timers[thread_block_id * 7 + 6] = t_end;
    }

    return;
}

template __global__ void gpu_probe_with_second_pass<false>(int*, int*, int*, int*, long long*, long long*, long long*, long long*, int, int, int*, int*, int*, int*, int*, long long*);
template __global__ void gpu_probe_with_second_pass<true>(int*, int*, int*, int*, long long*, long long*, long long*, long long*, int, int, int*, int*, int*, int*, int*, long long*);


// Turns first level partition counts into bucket boundaries and scatter offsets.
__global__ void gpu_make_simple_prefix(int * g_idata, long long * g_prefix_data, long long * g_block_index_split, long long data_end, int partition_size, long long * g_offset)
{
    extern  __shared__  long long sdata_ll[];

    int tid = threadIdx.x;
    int thread_size = blockDim.x;

    for(int i=tid; i<partition_size; i+=thread_size)
        sdata_ll[i]=g_prefix_data[i];
    __syncthreads();

    long long * sdata1 = &sdata_ll[0];
    long long * sdata2 = &sdata_ll[partition_size];

    for(int step=1; step<partition_size; step*=2)
    {
        for(int i=tid; i<partition_size; i+=thread_size)
            sdata2[i] = (i>=step) ? sdata1[i] + sdata1[i-step] : sdata1[i];
        __syncthreads();
        long long * sdata_change = sdata1; sdata1 = sdata2; sdata2 = sdata_change;
    }

    for(int i=tid; i<partition_size; i+=thread_size)
        g_block_index_split[i]=sdata1[i];

    if(tid==0) g_offset[0]=0;
    for(int i=tid; i<partition_size-1; i+=thread_size)
        g_offset[i+1]=sdata1[i];

    return;

}



// Probes one first level bucket per block with a linear scan.
__global__ void gpu_probe_in_a_single_block(int *g_oa_data, int *g_ob_data, long long * ga_index, long long * gb_index, int second_partition_size, int * g_answer_num)
{

    extern  __shared__  int sdata[];


    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int thread_size = blockDim.x;
    int block_number =  gridDim.x;

    long long r_block_start = 0;
    long long s_block_start = 0;

    if (bid != 0) {
        r_block_start = ga_index[bid-1];
        s_block_start = gb_index[bid-1];
    }
    __syncwarp();


    long long r_block_end = ga_index[bid];
    long long s_block_end = gb_index[bid];


    int answer_num =0;

#pragma unroll
    for (long long i = s_block_start + tid; i < s_block_end; i += thread_size)
    {
        int target = g_ob_data[i];
        for (long long j = r_block_start; j < r_block_end; j++)
        {
            if (target == g_oa_data[j]) {
                answer_num++;
            }
        }

    }

    atomicAdd(&g_answer_num[0], answer_num);

    return;
}

// Same as gpu_probe_in_a_single_block with extra reporting of the match count.
__global__ void gpu_probe_in_a_single_block_with_check(int *g_oa_data, int *g_ob_data, long long * ga_index, long long * gb_index, int second_partition_size, int * g_answer_num)
{
    extern  __shared__  int sdata[];


    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int thread_size = blockDim.x;
    int block_number =  gridDim.x;

    long long r_block_start = 0;
    long long s_block_start = 0;

    if (bid != 0) {
        r_block_start = ga_index[bid-1];
        s_block_start = gb_index[bid-1];
    }
    __syncwarp();


    long long r_block_end = ga_index[bid];
    long long s_block_end = gb_index[bid];


    if(bid==0 && tid ==0)
    {
        printf("R: (%lld %lld) S: (%lld %lld)\n", r_block_start, r_block_end, s_block_start, s_block_end);
    }

    int answer_num =0;

#pragma unroll
    for (long long i = s_block_start + tid; i < s_block_end; i += thread_size)
    {
        int target = g_ob_data[i];
        for (long long j = r_block_start; j < r_block_end; j++)
        {
            if (target == g_oa_data[j]) {
                answer_num++;
            }
        }

    }

    atomicAdd(&g_answer_num[0], answer_num);
    __syncthreads();
    if(bid==0 && tid==0)
    {
        printf("Answer Num inside the kernel: %d\n",g_answer_num[0]);
    }

    return;
}

// Prints device array values for debugging.
__global__ void print_gpu_value(int *g_data, int offset)
{
    int tid = threadIdx.x;
    printf("%d: %d\n",offset+tid, g_data[offset+tid]);
}

// Prints device array values with a tag for debugging.
__global__ void print_gpu_value(int *g_data, int offset, int check)
{
    int tid = threadIdx.x;
    printf("(%d %d): %d\n",check, offset+tid, g_data[offset+tid]);
}

// Performs the local probe, the remote S transfer and the cross GPU probe in one kernel.
template<bool COLLECT>
__global__ void gpu_fused_probe_transfer(
    const int* __restrict__ r_data,
    const int* __restrict__ r_val,
    const int* __restrict__ local_s_data,
    const int* __restrict__ local_s_val,
    const int2* nvshmem_send_buf_kv,
    const long long* __restrict__ r_block_idx,
    const long long* __restrict__ r_second_idx,
    long long** all_s_block_idx,
    long long** all_s_second_idx,
    int me, int device_num,
    int ps, int sp,
    int* g_answer, int* g_answer_val, int* g_answer_s_val, int* g_answer_num, int* g_answer_write_idx,
    int ht_size, int chunk_size,
    long long* d_phase_timers,
    int single_remote_peer)
{
    extern __shared__ int sdata[];
    int* ht       = sdata;
    int* ht_val   = sdata + ht_size;
    int* tmp      = sdata + 2*ht_size;
    int2* tmp_kv  = (int2*)(sdata + 2*ht_size);

    const int tid         = threadIdx.x;
    const int thread_size = blockDim.x;
    const int tbid        = blockIdx.x;
    const int bid         = tbid / sp;
    const int second_bid  = tbid % sp;
    const int nwarps      = (thread_size + 31) / 32;

    long long t_entry = 0, t_after_ht = 0, t_after_local = 0, t_end = 0;
    long long cum_get_cycles = 0;
    if (COLLECT && tid == 0) t_entry = clock64();

    int ans = 0;

    const long long r_block_start = (bid != 0) ? r_block_idx[bid - 1] : 0LL;
    const int r_sub_start = (second_bid != 0) ? (int)r_second_idx[tbid - 1] : 0;
    const int r_sub_end   = (int)r_second_idx[tbid];
    const long long r_start = r_block_start + (long long)r_sub_start;
    const int loop_size   = r_sub_end - r_sub_start;

    const int EMPTY = 0x80000000;

#define WARP_REDUCE_TO_TMP(ans)                                             \
    do {                                                                    \
        for (int _off = 16; _off > 0; _off >>= 1)                          \
            (ans) += __shfl_down_sync(0xffffffff, (ans), _off);             \
        if ((tid & 31) == 0) tmp[tid >> 5] = (ans);                        \
        __syncthreads();                                                    \
        if (tid == 0) {                                                     \
            int _sum = 0;                                                   \
            for (int _k = 0; _k < nwarps; _k++) _sum += tmp[_k];           \
            atomicAdd(g_answer_num, _sum);                                  \
        }                                                                   \
    } while (0)

    for (int i = tid; i < ht_size; i += thread_size)
        ht[i] = EMPTY;
    __syncthreads();

    if (loop_size >= ht_size) {
        if (tid == 0) atomicAdd(&d_probe_overflow_count, 1);
        if (COLLECT && tid == 0) t_after_ht = clock64();

        {
            const long long s_blk = (bid != 0) ? all_s_block_idx[me][bid - 1] : 0LL;
            const int s_ss = (second_bid != 0) ? (int)all_s_second_idx[me][tbid - 1] : 0;
            const long long s_start = s_blk + (long long)s_ss;
            const long long s_end   = s_blk + (long long)(int)all_s_second_idx[me][tbid];
            for (long long i = s_start + tid; i < s_end; i += thread_size) {
                int2 kv = nvshmem_send_buf_kv[i];
                int tgt = kv.x;
                for (int j = 0; j < loop_size; j++) {
                    if (r_data[r_start + j] == tgt) {
                        int idx = atomicAdd(g_answer_write_idx, 1);
                        if ((unsigned)idx < (unsigned)d_answer_capacity) {
                            g_answer[idx] = tgt;
                            g_answer_val[idx] = r_val[r_start + j];
                            g_answer_s_val[idx] = kv.y;
                        }
                        ans++;
                        break;
                    }
                }
            }
        }
        __syncthreads();
        if (COLLECT && tid == 0) t_after_local = clock64();

        const int n_remote_iters_ov = (single_remote_peer >= 0) ? 1 : (device_num - 1);
        for (int it = 0; it < n_remote_iters_ov; it++) {
            const int src_pe = (single_remote_peer >= 0)
                ? single_remote_peer
                : ((me - (it + 1) + device_num) % device_num);
            const long long s_blk = (bid != 0) ? all_s_block_idx[src_pe][bid - 1] : 0LL;
            const int s_ss = (second_bid != 0) ? (int)all_s_second_idx[src_pe][tbid - 1] : 0;
            const long long s_offset = s_blk + (long long)s_ss;
            const int s_count = (int)all_s_second_idx[src_pe][tbid] - s_ss;

            for (int ci = 0; ci < s_count; ci += chunk_size) {
                const int fetch_n = min(chunk_size, s_count - ci);
                long long get_ts0 = 0;
                if (COLLECT && tid == 0) get_ts0 = clock64();
                nvshmemx_getmem_block(
                    tmp_kv,
                    nvshmem_send_buf_kv + s_offset + (long long)ci,
                    sizeof(int2) * (size_t)fetch_n, src_pe);
                __syncthreads();
                if (COLLECT && tid == 0) cum_get_cycles += clock64() - get_ts0;
                for (int i = tid; i < fetch_n; i += thread_size) {
                    int2 kv = tmp_kv[i];
                    int tgt = kv.x;
                    for (int j = 0; j < loop_size; j++) {
                        if (r_data[r_start + j] == tgt) {
                            int idx = atomicAdd(g_answer_write_idx, 1);
                            if ((unsigned)idx < (unsigned)d_answer_capacity) {
                                g_answer[idx] = tgt;
                                g_answer_val[idx] = r_val[r_start + j];
                                g_answer_s_val[idx] = kv.y;
                            }
                            ans++;
                            break;
                        }
                    }
                }
            }
            __syncthreads();
        }
        WARP_REDUCE_TO_TMP(ans);
        __syncthreads();
        if (COLLECT && tid == 0) {
            t_end = clock64();
            d_phase_timers[tbid * 5 + 0] = t_entry;
            d_phase_timers[tbid * 5 + 1] = t_after_ht;
            d_phase_timers[tbid * 5 + 2] = t_after_local;
            d_phase_timers[tbid * 5 + 3] = t_end;
            d_phase_timers[tbid * 5 + 4] = cum_get_cycles;
        }
        return;
    }

    for (int i = tid; i < loop_size; i += thread_size) {
        int val  = r_data[r_start + i];
        int slot = (unsigned)val % (unsigned)ht_size;
        while (atomicCAS(&ht[slot], EMPTY, val) != EMPTY)
            slot = (slot + 1) % ht_size;
        ht_val[slot] = r_val[r_start + i];
    }
    __syncthreads();
    if (COLLECT && tid == 0) t_after_ht = clock64();

    {
        const long long s_blk = (bid != 0) ? all_s_block_idx[me][bid - 1] : 0LL;
        const int s_ss = (second_bid != 0) ? (int)all_s_second_idx[me][tbid - 1] : 0;
        const long long s_start = s_blk + (long long)s_ss;
        const long long s_end   = s_blk + (long long)(int)all_s_second_idx[me][tbid];
        for (long long i = s_start + tid; i < s_end; i += thread_size) {
            int2 kv  = nvshmem_send_buf_kv[i];
            int tgt  = kv.x;
            int slot = (unsigned)tgt % (unsigned)ht_size;
            while (ht[slot] != EMPTY) {
                if (ht[slot] == tgt) {
                    int idx = atomicAdd(g_answer_write_idx, 1);
                    if ((unsigned)idx < (unsigned)d_answer_capacity) {
                        g_answer[idx] = tgt;
                        g_answer_val[idx] = ht_val[slot];
                        g_answer_s_val[idx] = kv.y;
                    }
                    ans++;
                }
                slot = (slot + 1) % ht_size;
            }
        }
    }
    __syncthreads();
    if (COLLECT && tid == 0) t_after_local = clock64();

    long long per_iter_get[8]   = {0,0,0,0,0,0,0,0};
    long long per_iter_probe[8] = {0,0,0,0,0,0,0,0};

    const int n_remote_iters = (single_remote_peer >= 0) ? 1 : (device_num - 1);
    for (int it = 0; it < n_remote_iters; it++) {
        const int src_pe = (single_remote_peer >= 0)
            ? single_remote_peer
            : ((me - (it + 1) + device_num) % device_num);
        const long long s_blk = (bid != 0) ? all_s_block_idx[src_pe][bid - 1] : 0LL;
        const int s_ss = (second_bid != 0) ? (int)all_s_second_idx[src_pe][tbid - 1] : 0;
        const long long s_offset = s_blk + (long long)s_ss;
        const int s_count = (int)all_s_second_idx[src_pe][tbid] - s_ss;
        const int iter_idx = it;

        for (int ci = 0; ci < s_count; ci += chunk_size) {
            const int fetch_n = min(chunk_size, s_count - ci);
            long long get_ts0 = 0, get_ts1 = 0;
            if (COLLECT && tid == 0) get_ts0 = clock64();
            nvshmemx_getmem_block(
                tmp_kv,
                nvshmem_send_buf_kv + s_offset + (long long)ci,
                sizeof(int2) * (size_t)fetch_n, src_pe);
            __syncthreads();
            if (COLLECT && tid == 0) {
                get_ts1 = clock64();
                cum_get_cycles      += get_ts1 - get_ts0;
                if (iter_idx < 8)
                    per_iter_get[iter_idx] += get_ts1 - get_ts0;
            }
            for (int i = tid; i < fetch_n; i += thread_size) {
                int2 kv  = tmp_kv[i];
                int tgt  = kv.x;
                int slot = (unsigned)tgt % (unsigned)ht_size;
                while (ht[slot] != EMPTY) {
                    if (ht[slot] == tgt) {
                        int idx = atomicAdd(g_answer_write_idx, 1);
                        if ((unsigned)idx < (unsigned)d_answer_capacity) {
                            g_answer[idx] = tgt;
                            g_answer_val[idx] = ht_val[slot];
                            g_answer_s_val[idx] = kv.y;
                        }
                        ans++;
                    }
                    slot = (slot + 1) % ht_size;
                }
            }
            if (COLLECT && tid == 0 && iter_idx < 8) {
                long long probe_end = clock64();
                per_iter_probe[iter_idx] += probe_end - get_ts1;
            }
        }
        __syncthreads();
    }
    WARP_REDUCE_TO_TMP(ans);
    __syncthreads();
    if (COLLECT && tid == 0) {
        t_end = clock64();
        d_phase_timers[tbid * 5 + 0] = t_entry;
        d_phase_timers[tbid * 5 + 1] = t_after_ht;
        d_phase_timers[tbid * 5 + 2] = t_after_local;
        d_phase_timers[tbid * 5 + 3] = t_end;
        d_phase_timers[tbid * 5 + 4] = cum_get_cycles;
    }
    if (COLLECT && blockIdx.x == 0 && tid == 0) {
        const long long base = (long long)gridDim.x * 5;
        for (int i = 0; i < 8; i++) {
            d_phase_timers[base + i]     = per_iter_get[i];
            d_phase_timers[base + 8 + i] = per_iter_probe[i];
        }
    }
    if (COLLECT && tid == 0) {
        const long long agg_base = (long long)gridDim.x * 5 + 16;
        int num_iters = (single_remote_peer >= 0) ? 1 : (device_num - 1);
        if (num_iters > 8) num_iters = 8;
        for (int i = 0; i < num_iters; i++) {
            unsigned long long g = (unsigned long long)per_iter_get[i];
            unsigned long long p = (unsigned long long)per_iter_probe[i];
            atomicAdd((unsigned long long*)&d_phase_timers[agg_base + 0  + i], g);
            atomicAdd((unsigned long long*)&d_phase_timers[agg_base + 8  + i], p);
            atomicMin((unsigned long long*)&d_phase_timers[agg_base + 16 + i], g);
            atomicMax((unsigned long long*)&d_phase_timers[agg_base + 24 + i], g);
        }
    }

#undef WARP_REDUCE_TO_TMP
}

template __global__ void gpu_fused_probe_transfer<false>(
    const int*, const int*, const int*, const int*, const int2*,
    const long long*, const long long*, long long**, long long**,
    int, int, int, int, int*, int*, int*, int*, int*, int, int, long long*, int);
template __global__ void gpu_fused_probe_transfer<true>(
    const int*, const int*, const int*, const int*, const int2*,
    const long long*, const long long*, long long**, long long**,
    int, int, int, int, int*, int*, int*, int*, int*, int, int, long long*, int);

