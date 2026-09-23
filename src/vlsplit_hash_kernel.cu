#include "vlsplit_hash_kernel.h"
#include "kernel_functions.h"
#include "common.h"


using namespace std;


// Radix partitions R and S in two levels.
void partial_jointable_gpu_partition(
                   const int* const d_adata, const int* const d_bdata, int* const d_oa_data, int* const d_ob_data,
                   int* const d_a_val, int* const d_b_val, int* const d_oa_val, int* const d_ob_val,
                   int2* const d_b_kv_out,
                   long long* const d_a_prefix_data, long long* const d_b_prefix_data,
                   long long* const d_block_index_split_r, long long* const d_block_index_split_s, long long* const d_a_offset, long long* const d_b_offset,
                   long long* const d_second_index_split_r, long long* const d_second_index_split_s, int* const d_answer_num,
                   const long long R,  const long long S, cudaStream_t gpu_stream, cudaEvent_t gpu_event,
                   double& out_partition_sec, double& out_hist_sec, bool skip_r_hist)
{
    std::chrono::duration<double> diff;
    auto start_point = std::chrono::steady_clock::now();

    int * temp_check;
    temp_check = new int[block_no*partition_size];

    long long block_size = R/block_no;
    if(R%block_no!=0) block_size++;

    long long s_block_size = S/block_no;
    if(S%block_no!=0) s_block_size++;

    if(s_block_size>block_size) block_size=s_block_size;

    printf("Block number: %d // Block size: %lld // Thread number: %d // Partition size: %d\n", block_no, block_size, thread_no, partition_size);

    cudaMemcpy((void *) temp_check, d_adata, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *) temp_check, d_bdata, sizeof(int), cudaMemcpyDeviceToHost);

    auto point_00 = std::chrono::steady_clock::now();
    diff = point_00-start_point;
    std::cout << "Init took " << diff.count() << " sec" << std::endl;

    if (skip_r_hist) {
        cudaMemsetAsync(d_b_prefix_data, 0, sizeof(long long) * partition_size, gpu_stream);
        cudaMemsetAsync(d_answer_num,   0, sizeof(int),                         gpu_stream);
    } else {
        gpu_prefix_init<<< 1, prefix_scan_block_dim(partition_size, thread_no), 0, gpu_stream >>>(d_a_prefix_data, d_b_prefix_data, partition_size, d_answer_num);
    }

    cudaMemcpy((void *) temp_check, d_bdata, sizeof(int), cudaMemcpyDeviceToHost);
    auto point_01 = std::chrono::steady_clock::now();
    diff = point_01-point_00;
    std::cout << "Prefix took " << diff.count() << " sec" << std::endl;

    gpu_run_hist_for_both<<< block_no, thread_no, sizeof(int) * (partition_size)* 2, gpu_stream >>>((int *)d_adata, d_a_prefix_data, skip_r_hist ? 0 : R, (int *)d_bdata, d_b_prefix_data, S,
                                                                                                    block_size, partition_size);

    cudaMemcpy((void *) temp_check, d_a_prefix_data, sizeof(int), cudaMemcpyDeviceToHost);
    auto point_01_6 = std::chrono::steady_clock::now();
    diff = point_01_6 - point_01;
    out_hist_sec += diff.count();
    std::cout << "Gpu Hist took " << diff.count() << " sec"   << std::endl;

    gpu_make_simple_prefix<<< 1, prefix_scan_block_dim(partition_size, thread_no), sizeof(long long) * partition_size*2, gpu_stream >>>((int *)d_adata, d_a_prefix_data, d_block_index_split_r, R, partition_size, d_a_offset);
    gpu_make_simple_prefix<<< 1, prefix_scan_block_dim(partition_size, thread_no), sizeof(long long) * partition_size*2, gpu_stream >>>((int *)d_bdata, d_b_prefix_data, d_block_index_split_s, S, partition_size, d_b_offset);

    cudaMemcpy((void *) temp_check, d_block_index_split_r, sizeof(int), cudaMemcpyDeviceToHost);
    auto point_01_6_5 = std::chrono::steady_clock::now();
    diff = point_01_6_5 - point_01_6;
    std::cout << "Making Prefix took: " << diff.count() << " sec"    << std::endl;

    auto pre_compute = std::chrono::steady_clock::now();
    gpu_make_sorted_table_for_both<<< block_no, thread_no, sizeof(int) * 0, gpu_stream >>>((int *)d_adata, (int *) d_oa_data, (int *)d_a_val, (int *)d_oa_val, d_block_index_split_r, R, d_a_offset,
                                                                                           (int *)d_bdata, (int *) d_ob_data, (int *)d_b_val, (int *)d_ob_val, d_block_index_split_s, S, d_b_offset,
                                                                                           block_size, partition_size);
    cudaMemcpy((void *) temp_check, d_second_index_split_r, sizeof(long long), cudaMemcpyDeviceToHost);
    auto post_compute = std::chrono::steady_clock::now();
    diff = post_compute-point_01_6_5;
    std::cout << "Building table took " << diff.count() << " sec"<< std::endl;

    gpu_run_hist_second_pass<<< partition_size, thread_no, sizeof(int) * second_partition_size * 2, gpu_stream >>>((int *) d_oa_data, (int *) d_adata, d_block_index_split_r, d_second_index_split_r, second_partition_size);
    cudaMemcpy((void *) temp_check, d_second_index_split_r, sizeof(long long), cudaMemcpyDeviceToHost);
    auto point_01_6_6 = std::chrono::steady_clock::now();
    diff = point_01_6_6-post_compute;
    std::cout << "Hist second pass took " << diff.count() << " sec"<< std::endl;

    gpu_build_table_second_pass<<< partition_size, thread_no, sizeof(int) * (second_partition_size), gpu_stream >>>((int *) d_oa_data, (int *) d_adata, (int *)d_oa_val, (int *)d_a_val, d_block_index_split_r, d_second_index_split_r, second_partition_size);
    cudaMemcpy((void *) temp_check, d_second_index_split_r, sizeof(long long), cudaMemcpyDeviceToHost);

    auto point_01_6_7 = std::chrono::steady_clock::now();
    diff = point_01_6_7 - point_01_6_6;
    std::cout << "Building single table with Hist information took " << diff.count() << " sec"  << std::endl;

    gpu_run_hist_second_pass<<< partition_size, thread_no, sizeof(int) * second_partition_size * 2, gpu_stream >>>((int *) d_ob_data, (int *) d_bdata, d_block_index_split_s, d_second_index_split_s, second_partition_size);
    cudaMemcpy((void *) temp_check, d_second_index_split_s, sizeof(long long), cudaMemcpyDeviceToHost);
    auto point_01_6_8 = std::chrono::steady_clock::now();
    diff = point_01_6_8 - point_01_6_7;
    std::cout << "(S) Hist second pass took " << diff.count() << " sec"  << std::endl;

    if (d_b_kv_out != nullptr)
        gpu_build_table_second_pass_kv<<< partition_size, thread_no, sizeof(int) * (second_partition_size), gpu_stream >>>((int *) d_ob_data, (int *)d_ob_val, d_b_kv_out, d_block_index_split_s, d_second_index_split_s, second_partition_size);
    else
        gpu_build_table_second_pass<<< partition_size, thread_no, sizeof(int) * (second_partition_size), gpu_stream >>>((int *) d_ob_data, (int *) d_bdata, (int *)d_ob_val, (int *)d_b_val, d_block_index_split_s, d_second_index_split_s, second_partition_size);
    cudaMemcpy((void *) temp_check, d_second_index_split_s, sizeof(long long), cudaMemcpyDeviceToHost);
    auto point_01_6_9 = std::chrono::steady_clock::now();
    diff = point_01_6_9 - point_01_6_8;
    std::cout << "(S) Building single table with Hist information took " << diff.count() << " sec"  << std::endl;

    out_partition_sec += std::chrono::duration<double>(point_01_6_9 - start_point).count();

    delete[] temp_check;
    return;
}


// Same as partial_jointable_gpu_partition but splits every stage into R and S halves so each is timed separately.
void partial_jointable_gpu_partition_rs_split(
                   const int* const d_adata, const int* const d_bdata, int* const d_oa_data, int* const d_ob_data,
                   int* const d_a_val, int* const d_b_val, int* const d_oa_val, int* const d_ob_val,
                   int2* const d_b_kv_out,
                   long long* const d_a_prefix_data, long long* const d_b_prefix_data,
                   long long* const d_block_index_split_r, long long* const d_block_index_split_s, long long* const d_a_offset, long long* const d_b_offset,
                   long long* const d_second_index_split_r, long long* const d_second_index_split_s, int* const d_answer_num,
                   const long long R,  const long long S, cudaStream_t gpu_stream, cudaEvent_t gpu_event,
                   double& out_partition_sec, double& out_partition_r_sec, double& out_partition_s_sec,
                   double& out_hist_r_sec, double& out_hist_s_sec, bool skip_r_hist)
{
    auto start_point = std::chrono::steady_clock::now();

    int * temp_check;
    temp_check = new int[block_no*partition_size];

    long long block_size = R/block_no;
    if(R%block_no!=0) block_size++;

    long long s_block_size = S/block_no;
    if(S%block_no!=0) s_block_size++;

    if(s_block_size>block_size) block_size=s_block_size;

    cudaMemcpy((void *) temp_check, d_adata, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *) temp_check, d_bdata, sizeof(int), cudaMemcpyDeviceToHost);

    auto point_00 = std::chrono::steady_clock::now();

    if (skip_r_hist) {
        cudaMemsetAsync(d_b_prefix_data, 0, sizeof(long long) * partition_size, gpu_stream);
        cudaMemsetAsync(d_answer_num,   0, sizeof(int),                         gpu_stream);
    } else {
        gpu_prefix_init<<< 1, prefix_scan_block_dim(partition_size, thread_no), 0, gpu_stream >>>(d_a_prefix_data, d_b_prefix_data, partition_size, d_answer_num);
    }

    cudaMemcpy((void *) temp_check, d_bdata, sizeof(int), cudaMemcpyDeviceToHost);
    auto point_01 = std::chrono::steady_clock::now();

    if (!skip_r_hist) {
        gpu_run_hist_for_both<<< block_no, thread_no, sizeof(int) * (partition_size)* 2, gpu_stream >>>((int *)d_adata, d_a_prefix_data, R, (int *)d_bdata, d_b_prefix_data, 0,
                                                                                                        block_size, partition_size);
    }
    cudaMemcpy((void *) temp_check, d_a_prefix_data, sizeof(int), cudaMemcpyDeviceToHost);
    auto point_hist_r = std::chrono::steady_clock::now();

    gpu_run_hist_for_both<<< block_no, thread_no, sizeof(int) * (partition_size)* 2, gpu_stream >>>((int *)d_adata, d_a_prefix_data, 0, (int *)d_bdata, d_b_prefix_data, S,
                                                                                                    block_size, partition_size);
    cudaMemcpy((void *) temp_check, d_b_prefix_data, sizeof(int), cudaMemcpyDeviceToHost);
    auto point_hist_s = std::chrono::steady_clock::now();

    gpu_make_simple_prefix<<< 1, prefix_scan_block_dim(partition_size, thread_no), sizeof(long long) * partition_size*2, gpu_stream >>>((int *)d_adata, d_a_prefix_data, d_block_index_split_r, R, partition_size, d_a_offset);
    cudaMemcpy((void *) temp_check, d_block_index_split_r, sizeof(int), cudaMemcpyDeviceToHost);
    auto point_prefix_r = std::chrono::steady_clock::now();

    gpu_make_simple_prefix<<< 1, prefix_scan_block_dim(partition_size, thread_no), sizeof(long long) * partition_size*2, gpu_stream >>>((int *)d_bdata, d_b_prefix_data, d_block_index_split_s, S, partition_size, d_b_offset);
    cudaMemcpy((void *) temp_check, d_block_index_split_s, sizeof(int), cudaMemcpyDeviceToHost);
    auto point_prefix_s = std::chrono::steady_clock::now();

    gpu_make_sorted_table_for_both<<< block_no, thread_no, sizeof(int) * 0, gpu_stream >>>((int *)d_adata, (int *) d_oa_data, (int *)d_a_val, (int *)d_oa_val, d_block_index_split_r, R, d_a_offset,
                                                                                           (int *)d_bdata, (int *) d_ob_data, (int *)d_b_val, (int *)d_ob_val, d_block_index_split_s, 0, d_b_offset,
                                                                                           block_size, partition_size);
    cudaMemcpy((void *) temp_check, d_second_index_split_r, sizeof(long long), cudaMemcpyDeviceToHost);
    auto point_scatter_r = std::chrono::steady_clock::now();

    gpu_make_sorted_table_for_both<<< block_no, thread_no, sizeof(int) * 0, gpu_stream >>>((int *)d_adata, (int *) d_oa_data, (int *)d_a_val, (int *)d_oa_val, d_block_index_split_r, 0, d_a_offset,
                                                                                           (int *)d_bdata, (int *) d_ob_data, (int *)d_b_val, (int *)d_ob_val, d_block_index_split_s, S, d_b_offset,
                                                                                           block_size, partition_size);
    cudaMemcpy((void *) temp_check, d_second_index_split_r, sizeof(long long), cudaMemcpyDeviceToHost);
    auto point_scatter_s = std::chrono::steady_clock::now();

    gpu_run_hist_second_pass<<< partition_size, thread_no, sizeof(int) * second_partition_size * 2, gpu_stream >>>((int *) d_oa_data, (int *) d_adata, d_block_index_split_r, d_second_index_split_r, second_partition_size);
    cudaMemcpy((void *) temp_check, d_second_index_split_r, sizeof(long long), cudaMemcpyDeviceToHost);
    gpu_build_table_second_pass<<< partition_size, thread_no, sizeof(int) * (second_partition_size), gpu_stream >>>((int *) d_oa_data, (int *) d_adata, (int *)d_oa_val, (int *)d_a_val, d_block_index_split_r, d_second_index_split_r, second_partition_size);
    cudaMemcpy((void *) temp_check, d_second_index_split_r, sizeof(long long), cudaMemcpyDeviceToHost);
    auto point_sp_r = std::chrono::steady_clock::now();

    gpu_run_hist_second_pass<<< partition_size, thread_no, sizeof(int) * second_partition_size * 2, gpu_stream >>>((int *) d_ob_data, (int *) d_bdata, d_block_index_split_s, d_second_index_split_s, second_partition_size);
    cudaMemcpy((void *) temp_check, d_second_index_split_s, sizeof(long long), cudaMemcpyDeviceToHost);
    if (d_b_kv_out != nullptr)
        gpu_build_table_second_pass_kv<<< partition_size, thread_no, sizeof(int) * (second_partition_size), gpu_stream >>>((int *) d_ob_data, (int *)d_ob_val, d_b_kv_out, d_block_index_split_s, d_second_index_split_s, second_partition_size);
    else
        gpu_build_table_second_pass<<< partition_size, thread_no, sizeof(int) * (second_partition_size), gpu_stream >>>((int *) d_ob_data, (int *) d_bdata, (int *)d_ob_val, (int *)d_b_val, d_block_index_split_s, d_second_index_split_s, second_partition_size);
    cudaMemcpy((void *) temp_check, d_second_index_split_s, sizeof(long long), cudaMemcpyDeviceToHost);
    auto point_sp_s = std::chrono::steady_clock::now();

    out_partition_r_sec += std::chrono::duration<double>(point_hist_r   - point_01).count()
                         + std::chrono::duration<double>(point_prefix_r - point_hist_s).count()
                         + std::chrono::duration<double>(point_scatter_r - point_prefix_s).count()
                         + std::chrono::duration<double>(point_sp_r     - point_scatter_s).count();
    out_partition_s_sec += std::chrono::duration<double>(point_hist_s    - point_hist_r).count()
                         + std::chrono::duration<double>(point_prefix_s  - point_prefix_r).count()
                         + std::chrono::duration<double>(point_scatter_s - point_scatter_r).count()
                         + std::chrono::duration<double>(point_sp_s      - point_sp_r).count();

    out_hist_r_sec += skip_r_hist ? 0.0
                    : std::chrono::duration<double>(point_hist_r - point_01).count();
    out_hist_s_sec += std::chrono::duration<double>(point_hist_s - point_hist_r).count();

    out_partition_sec += std::chrono::duration<double>(point_sp_s - start_point).count();

    delete[] temp_check;
    return;
}


// Partitions only R in two levels. The result stays in d_adata in place.
void partial_jointable_gpu_partition_r_only(
        int* const d_adata, int* const d_oa_data,
        int* const d_a_val, int* const d_oa_val,
        long long* const d_a_prefix_data, long long* const d_block_index_split_r,
        long long* const d_a_offset, long long* const d_second_index_split_r,
        const long long R, cudaStream_t gpu_stream)
{
    long long block_size = R / block_no;
    if (R % block_no != 0) block_size++;

    cudaMemsetAsync(d_a_prefix_data, 0, sizeof(long long) * partition_size, gpu_stream);

    gpu_run_hist_for_both<<< block_no, thread_no, sizeof(int) * (partition_size) * 2, gpu_stream >>>(
            (int *)d_adata, d_a_prefix_data, R, (int *)d_adata, d_a_prefix_data, 0,
            block_size, partition_size);

    gpu_make_simple_prefix<<< 1, prefix_scan_block_dim(partition_size, thread_no), sizeof(long long) * partition_size * 2, gpu_stream >>>(
            (int *)d_adata, d_a_prefix_data, d_block_index_split_r, R, partition_size, d_a_offset);

    gpu_make_sorted_table_for_both<<< block_no, thread_no, 0, gpu_stream >>>(
            (int *)d_adata, (int *)d_oa_data, (int *)d_a_val, (int *)d_oa_val, d_block_index_split_r, R, d_a_offset,
            (int *)d_adata, (int *)d_oa_data, (int *)d_a_val, (int *)d_oa_val, d_block_index_split_r, 0, d_a_offset,
            block_size, partition_size);

    gpu_run_hist_second_pass<<< partition_size, thread_no, sizeof(int) * second_partition_size * 2, gpu_stream >>>(
            (int *)d_oa_data, (int *)d_adata, d_block_index_split_r, d_second_index_split_r, second_partition_size);
    gpu_build_table_second_pass<<< partition_size, thread_no, sizeof(int) * (second_partition_size), gpu_stream >>>(
            (int *)d_oa_data, (int *)d_adata, (int *)d_oa_val, (int *)d_a_val, d_block_index_split_r, d_second_index_split_r, second_partition_size);

    cudaStreamSynchronize(gpu_stream);
    return;
}

// Partitions only S in two levels. The result stays in d_bdata in place.
void partial_jointable_gpu_partition_s_only(
        int* const d_bdata, int* const d_ob_data,
        int* const d_b_val, int* const d_ob_val,
        int2* const d_b_kv_out,
        long long* const d_b_prefix_data, long long* const d_block_index_split_s,
        long long* const d_b_offset, long long* const d_second_index_split_s,
        int* const d_answer_num, const long long S, cudaStream_t gpu_stream)
{
    long long block_size = S / block_no;
    if (S % block_no != 0) block_size++;
    if (block_size < 1) block_size = 1;

    cudaMemsetAsync(d_b_prefix_data, 0, sizeof(long long) * partition_size, gpu_stream);
    cudaMemsetAsync(d_answer_num,    0, sizeof(int),                        gpu_stream);

    gpu_run_hist_for_both<<< block_no, thread_no, sizeof(int) * (partition_size) * 2, gpu_stream >>>(
            (int *)d_bdata, d_b_prefix_data, 0, (int *)d_bdata, d_b_prefix_data, S,
            block_size, partition_size);

    gpu_make_simple_prefix<<< 1, prefix_scan_block_dim(partition_size, thread_no), sizeof(long long) * partition_size * 2, gpu_stream >>>(
            (int *)d_bdata, d_b_prefix_data, d_block_index_split_s, S, partition_size, d_b_offset);

    gpu_make_sorted_table_for_both<<< block_no, thread_no, 0, gpu_stream >>>(
            (int *)d_bdata, (int *)d_ob_data, (int *)d_b_val, (int *)d_ob_val, d_block_index_split_s, 0, d_b_offset,
            (int *)d_bdata, (int *)d_ob_data, (int *)d_b_val, (int *)d_ob_val, d_block_index_split_s, S, d_b_offset,
            block_size, partition_size);

    gpu_run_hist_second_pass<<< partition_size, thread_no, sizeof(int) * second_partition_size * 2, gpu_stream >>>(
            (int *)d_ob_data, (int *)d_bdata, d_block_index_split_s, d_second_index_split_s, second_partition_size);
    if (d_b_kv_out != nullptr) {
        gpu_build_table_second_pass_kv<<< partition_size, thread_no, sizeof(int) * (second_partition_size), gpu_stream >>>(
                (int *)d_ob_data, (int *)d_ob_val, d_b_kv_out, d_block_index_split_s, d_second_index_split_s, second_partition_size);
    } else {
        gpu_build_table_second_pass<<< partition_size, thread_no, sizeof(int) * (second_partition_size), gpu_stream >>>(
                (int *)d_ob_data, (int *)d_bdata, (int *)d_ob_val, (int *)d_b_val, d_block_index_split_s, d_second_index_split_s, second_partition_size);
    }

    cudaStreamSynchronize(gpu_stream);
    return;
}

// Probes the already partitioned R against an S buffer received from a remote PE.
void partial_jointable_gpu_run_only_s(
        const int* const d_adata, const int* const d_bdata,
        const int* const d_a_val, const int* const d_b_val,
        long long* const d_block_index_split_r, long long* const d_block_index_split_s,
        long long* const d_second_index_split_r, long long* const d_second_index_split_s, int* const d_answer, int* const d_answer_val, int* const d_answer_s_val, int* const d_answer_num, int max_int_number_for_shared_memory, int * final_answer_number,
        cudaStream_t gpu_stream, cudaEvent_t gpu_event, int* const d_answer_write_idx)
{
    size_t kv_shm = sizeof(int) * (size_t)max_int_number_for_shared_memory * 2;
    static bool s_probe_attr_set = false;
    if (!s_probe_attr_set) {
        cudaFuncSetAttribute(gpu_probe_with_second_pass<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kv_shm);
        s_probe_attr_set = true;
    }
    gpu_probe_with_second_pass<false><<< partition_size*second_partition_size, thread_no, kv_shm, gpu_stream >>>((int *)d_adata, (int *)d_bdata, (int *)d_a_val, (int *)d_b_val, d_block_index_split_r, d_block_index_split_s,
                                                                                                                                                  d_second_index_split_r, d_second_index_split_s,
                                                                                                                                                partition_size, second_partition_size,d_answer, d_answer_val, d_answer_s_val, d_answer_num, d_answer_write_idx, nullptr);

    cudaMemcpy(final_answer_number, d_answer_num, sizeof(int), cudaMemcpyDeviceToHost);

    return;
}
