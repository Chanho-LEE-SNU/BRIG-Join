#include "bloom.cuh"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// Hash built on the MurmurHash3 finalizer.
__device__ uint32_t hash_function(int32_t value, uint32_t seed) {
    uint32_t h = seed ^ value;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// Vector loads Phi filter words at a time.
template<int Phi>
__device__ void vec_load_words(const uint32_t* ptr, uint32_t* out) {
    if constexpr (Phi == 1) {
        out[0] = *ptr;
    } else if constexpr (Phi == 2) {
        uint2 data = *reinterpret_cast<uint2*>(__builtin_assume_aligned(ptr, 8));
        out[0] = data.x;
        out[1] = data.y;
    } else if constexpr (Phi == 4) {
        uint4 data = *reinterpret_cast<uint4*>(__builtin_assume_aligned(ptr, 16));
        out[0] = data.x;
        out[1] = data.y;
        out[2] = data.z;
        out[3] = data.w;
    }
}

// Sets k bits per element in the filter.
__global__ void insertKernel(uint32_t* filter, const int32_t* data, 
                            size_t num_elements, int num_hashes, size_t num_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        int32_t value = data[idx];
        uint32_t hash1 = hash_function(value, 1);
        uint32_t hash2 = hash_function(value, 2); 
        
        const size_t block_size = 256;
        const size_t words_per_block = block_size / 32;
        size_t num_blocks = num_bits / block_size;
        
        size_t block_idx = hash1 % num_blocks;
        size_t block_start = block_idx * words_per_block;
        
        for (int i = 0; i < num_hashes; i++) {
            uint32_t hash_val = hash1 + i * hash2;
            
            
            size_t bit_pos = hash_val % block_size;
            size_t word_offset = bit_pos / 32;
            size_t bit_idx = bit_pos % 32;
            
            atomicOr(&filter[block_start + word_offset], (1U << bit_idx));
        }
    }
}

// Sets bits inside one block, using a different seed per hash.
__global__ void insertKernel_blocked(uint32_t* filter, const int32_t* data, 
                            size_t num_elements, int num_hashes, size_t num_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        int32_t value = data[idx];
        
        const size_t block_size = 256;
        const size_t words_per_block = block_size / 32;
        size_t num_blocks = num_bits / block_size;
        
        uint32_t h1 = hash_function(value, 1);
        size_t block_idx = h1 % num_blocks;
        size_t block_start = block_idx * words_per_block;
        
        for (int i = 0; i < num_hashes; i++) {
            uint32_t hash_val = hash_function(value, i + 1);
            
            
            size_t bit_pos = hash_val % block_size;
            size_t word_offset = bit_pos / 32;
            size_t bit_idx = bit_pos % 32;
            
            atomicOr(&filter[block_start + word_offset], (1U << bit_idx));
        }
    }
}
// Sets bits across the whole filter using double hashing.
__global__ void insertKernel_double(uint32_t* filter, const int32_t* data, 
                            size_t num_elements, int num_hashes, size_t num_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        int32_t value = data[idx];
        uint32_t h1 = hash_function(value, 1);
        uint32_t h2 = hash_function(value, 2); 
        
        for (int i = 0; i < num_hashes; i++) {
            uint32_t hash_val = h1 + i * h2;
            
            
            size_t bit_pos = hash_val % num_bits;
            size_t word_idx = bit_pos / 32;
            size_t bit_idx = bit_pos % 32;
            
            atomicOr(&filter[word_idx], (1U << bit_idx));
        }
    }
}
// Sets bits across the whole filter without block locality.
__global__ void insertKernel_plain(uint32_t* filter, const int32_t* data, 
                            size_t num_elements, int num_hashes, size_t num_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        int32_t value = data[idx];
        
        for (int i = 0; i < num_hashes; i++) {
            uint32_t hash_val = hash_function(value, i);
            
            
            size_t bit_pos = hash_val % num_bits;
            size_t word_idx = bit_pos / 32;
            size_t bit_idx = bit_pos % 32;
            
            atomicOr(&filter[word_idx], (1U << bit_idx));
        }
    }
}

// Inserts one element per group of Theta cooperating threads.
template<int Theta = 4, int Phi = 2>
__global__ void insertKernel_vectorized(uint32_t* filter, const int32_t* data, 
                                        size_t num_elements, int num_hashes, 
                                        size_t num_bits) {
    const size_t block_size = 256;
    const size_t words_per_block = block_size / 32;
    size_t num_blocks = num_bits / block_size;

    auto tile = cooperative_groups::tiled_partition<Theta>(
        cooperative_groups::this_thread_block()
    );
    int cg_rank = tile.thread_rank();
    int cg_id = threadIdx.x / Theta;

    int global_cg_id = blockIdx.x * (blockDim.x / Theta) + cg_id;
    if (global_cg_id >= num_elements) return;

    int32_t value = data[global_cg_id];

    uint32_t h_block = (cg_rank == 0) ? hash_function(value, 1) : 0;
    h_block = tile.shfl(h_block, 0);

    size_t block_idx = h_block % num_blocks;
    size_t block_start = block_idx * words_per_block;

    size_t my_word_start = cg_rank * Phi;
    
    for (int k = 0; k < num_hashes; k++) {
        uint32_t hash_val = hash_function(value, k + 1);
        size_t bit_pos = hash_val % block_size;
        size_t target_word = bit_pos / 32;
        size_t bit_idx = bit_pos % 32;
        
        if (target_word >= my_word_start && target_word < my_word_start + Phi) {
            atomicOr(&filter[block_start + target_word], (1U << bit_idx));
        }
    }
}



// Queries the filter for every element and records the outcome in results.
__global__ void queryKernel(const uint32_t* filter, const int32_t* data, 
                           bool* results, size_t num_elements, 
                           int num_hashes, size_t num_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        int32_t value = data[idx];
        uint32_t hash1 = hash_function(value, 1);
        uint32_t hash2 = hash_function(value, 2); 
        bool found = true;
        
        const size_t block_size = 256;
        const size_t words_per_block = block_size / 32;
        size_t num_blocks = num_bits / block_size;
        
        size_t block_idx = hash1 % num_blocks;
        size_t block_start = block_idx * words_per_block;
        
        for (int i = 0; i < num_hashes && found; i++) {
            uint32_t hash_val = hash1 + i * hash2;
            
            size_t bit_pos = hash_val % block_size;
            size_t word_offset = bit_pos / 32;
            size_t bit_idx = bit_pos % 32;
            
            if (!(filter[block_start + word_offset] & (1U << bit_idx))) {
                found = false;
            }
        }
        
        results[idx] = found;
    }
}

// Checks bits inside one block, using a different seed per hash.
__global__ void queryKernel_blocked(const uint32_t* filter, const int32_t* data, 
                           bool* results, size_t num_elements, 
                           int num_hashes, size_t num_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        int32_t value = data[idx];
        bool found = true;
        
        const size_t block_size = 256;
        const size_t words_per_block = block_size / 32;
        size_t num_blocks = num_bits / block_size;
        
        uint32_t h1 = hash_function(value, 1);
        size_t block_idx = h1 % num_blocks;
        size_t block_start = block_idx * words_per_block;
        
        for (int i = 0; i < num_hashes && found; i++) {
            uint32_t hash_val = hash_function(value, i+1);
            
            size_t bit_pos = hash_val % block_size;
            size_t word_offset = bit_pos / 32;
            size_t bit_idx = bit_pos % 32;
            
            if (!(filter[block_start + word_offset] & (1U << bit_idx))) {
                found = false;
            }
        }
        
        results[idx] = found;
    }
}

// Checks bits across the whole filter using double hashing.
__global__ void queryKernel_double(const uint32_t* filter, const int32_t* data, 
                           bool* results, size_t num_elements, 
                           int num_hashes, size_t num_bits) {
                            
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        int32_t value = data[idx];
        uint32_t h1 = hash_function(value, 1);
        uint32_t h2 = hash_function(value, 2); 

        bool found = true;
        
        for (int i = 0; i < num_hashes && found; i++) {
            uint32_t hash_val = h1 + i * h2;
            
            
            
            
            size_t bit_pos = hash_val % num_bits;
            size_t word_idx = bit_pos / 32;
            size_t bit_idx = bit_pos % 32;
            
            if (!(filter[word_idx] & (1U << bit_idx))) {
                found = false;
            }
        }
        
        results[idx] = found;
    }
}

// Queries after vector loading the block words. Threads cooperate when Theta is above one.
template<int Theta = 1, int Phi = 4>
__global__ void queryKernel_vectorized(const uint32_t* filter, const int32_t* data, 
                                       bool* results, size_t num_elements, 
                                       int num_hashes, size_t num_bits) {
    const size_t block_size = 256;
    const size_t words_per_block = block_size / 32;
    size_t num_blocks = num_bits / block_size;

    if constexpr (Theta == 1) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements) return;

        int32_t value = data[idx];

        uint32_t h_block = hash_function(value, 1);
        size_t block_idx = h_block % num_blocks;
        size_t block_start = block_idx * words_per_block;

        uint32_t words[8];

        if constexpr (Phi == 4) {
            vec_load_words<4>(&filter[block_start], &words[0]);
            vec_load_words<4>(&filter[block_start + 4], &words[4]);
        } else if constexpr (Phi == 2) {
            vec_load_words<2>(&filter[block_start], &words[0]);
            vec_load_words<2>(&filter[block_start + 2], &words[2]);
            vec_load_words<2>(&filter[block_start + 4], &words[4]);
            vec_load_words<2>(&filter[block_start + 6], &words[6]);
        }
        
        bool found = true;
        #pragma unroll
        for (int i = 0; i < num_hashes && found; i++) {
            uint32_t hash_val = hash_function(value, i + 1);
            size_t bit_pos = hash_val % block_size;
            size_t word_idx = bit_pos / 32;
            size_t bit_idx = bit_pos % 32;
            
            if (!(words[word_idx] & (1U << bit_idx))) {
                found = false;
            }
        }
        
        results[idx] = found;
        
    } else {
        auto tile = cooperative_groups::tiled_partition<Theta>(
            cooperative_groups::this_thread_block()
        );
        int cg_rank = tile.thread_rank();
        int cg_id = threadIdx.x / Theta;
        int global_cg_id = blockIdx.x * (blockDim.x / Theta) + cg_id;
        
        if (global_cg_id >= num_elements) return;
        
        int32_t value = data[global_cg_id];
        
        uint32_t h_block = (cg_rank == 0) ? hash_function(value, 1) : 0;
        h_block = tile.shfl(h_block, 0);

        size_t block_idx = h_block % num_blocks;
        size_t block_start = block_idx * words_per_block;

        size_t my_word_start = cg_rank * Phi;
        uint32_t my_words[4] = {0};

        if (my_word_start < words_per_block) {
            vec_load_words<Phi>(&filter[block_start + my_word_start], my_words);
        }

        bool found = true;
        for (int i = 0; i < num_hashes && found; i++) {
            uint32_t hash_val = hash_function(value, i + 1);
            size_t bit_pos = hash_val % block_size;
            size_t word_idx = bit_pos / 32;
            size_t bit_idx = bit_pos % 32;
            
            bool bit_set = false;
            if (word_idx >= my_word_start && word_idx < my_word_start + Phi) {
                int local_idx = word_idx - my_word_start;
                bit_set = (my_words[local_idx] & (1U << bit_idx)) != 0;
            }
            
            int owner = word_idx / Phi;
            bit_set = tile.shfl(bit_set, owner);
            
            if (!bit_set) {
                found = false;
            }
        }
        
        results[global_cg_id] = found;
    }
}


// Checks bits across the whole filter without block locality.
__global__ void queryKernel_plain(const uint32_t* filter, const int32_t* data, 
                           bool* results, size_t num_elements, 
                           int num_hashes, size_t num_bits) {
                            
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        int32_t value = data[idx];
        bool found = true;
        
        for (int i = 0; i < num_hashes && found; i++) {
            uint32_t hash_val;
            
            hash_val = hash_function(value, i);
            
            
            size_t bit_pos = hash_val % num_bits;
            size_t word_idx = bit_pos / 32;
            size_t bit_idx = bit_pos % 32;
            
            if (!(filter[word_idx] & (1U << bit_idx))) {
                found = false;
            }
        }
        
        results[idx] = found;
    }
}


// Rounds the bit count up to a multiple of 256 and allocates the host and device buffers.
BloomFilter::BloomFilter(size_t bits, int hashes)
    : num_bits(bits), num_hashes(hashes) {

    const size_t BF_BLOCK_BITS = 256;
    num_bits = (num_bits + BF_BLOCK_BITS - 1) / BF_BLOCK_BITS * BF_BLOCK_BITS;

    num_words = num_bits / 32;

    h_filter = new uint32_t[num_words];
    memset(h_filter, 0, num_words * sizeof(uint32_t));

    d_filter = (uint32_t*)nvshmem_malloc(num_words * sizeof(uint32_t));
    CUDA_CHECK(cudaMemset(d_filter, 0, num_words * sizeof(uint32_t)));
    
    std::cout << "Bloom filter created:" << std::endl;
    std::cout << "  bits: " << num_bits << std::endl;
    std::cout << "  hash functions: " << num_hashes << std::endl;
    {
        size_t bytes = num_words * sizeof(uint32_t);
        std::cout << std::fixed << std::setprecision(2);
        if (bytes >= (1024ULL * 1024 * 1024))
            std::cout << "  memory: " << (double)bytes / (1024.0*1024.0*1024.0) << " GB" << std::endl;
        else
            std::cout << "  memory: " << (double)bytes / (1024.0*1024.0) << " MB" << std::endl;
        std::cout << std::defaultfloat;
    }
}

// Releases the host and device filter buffers.
BloomFilter::~BloomFilter() {
    delete[] h_filter;
    nvshmem_free(d_filter);
}

// Inserts every element of a device resident table into the filter.
void BloomFilter::buildFromTable(int *gpu_data, size_t num_elements) {
    {
        cudaError_t pre_err = cudaGetLastError();
        if (pre_err != cudaSuccess) {
            std::cerr << "CUDA error BEFORE kernel launch in buildFromTable: "
                      << cudaGetErrorString(pre_err) << std::endl;
        }
    }

    int threads = 256;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEventRecord(start);

    const int Theta_insert = 4;
    int groups_per_block = threads / Theta_insert;
    int blocks_vec = (num_elements + groups_per_block - 1) / groups_per_block;
    insertKernel_vectorized<Theta_insert,2><<<blocks_vec, threads>>>(d_filter, gpu_data, num_elements, num_hashes, num_bits);

    cudaEventRecord(stop);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Determines exact membership with a linear scan.
__global__ void exactMatchKernel(const int32_t* build_data, size_t build_size,
                                 const int32_t* query_data, size_t query_size,
                                 bool* exact_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= query_size) return;
    
    int32_t query_value = query_data[idx];
    bool found = false;
    
    for (size_t i = 0; i < build_size; i++) {
        if (build_data[i] == query_value) {
            found = true;
            break;
        }
    }
    
    exact_results[idx] = found;
}

// Counts the elements that pass the filter per partition without moving them.
__global__ void queryAndHistOnlyKernel(
    const uint32_t* filter, const int32_t* data, int num_elements,
    int num_hashes, size_t num_bits,
    long long* histogram, int partition_size, int block_sz)
{
    extern __shared__ int sdata[];
    for (int i = threadIdx.x; i < partition_size; i += blockDim.x) sdata[i] = 0;
    __syncthreads();

    const size_t bf_block_size = 256;
    const size_t words_per_block = bf_block_size / 32;
    size_t num_bf_blocks = num_bits / bf_block_size;

    int start = blockIdx.x * block_sz + threadIdx.x;
    int end   = min((blockIdx.x + 1) * block_sz, num_elements);

    for (int j = start; j < end; j += blockDim.x) {
        int32_t value = data[j];

        uint32_t h1 = hash_function(value, 1);
        size_t block_idx = h1 % num_bf_blocks;
        size_t block_start = block_idx * words_per_block;

        bool found = true;
        for (int i = 0; i < num_hashes && found; i++) {
            uint32_t hash_val = hash_function(value, i + 1);
            size_t bit_pos    = hash_val % bf_block_size;
            size_t word_off   = bit_pos / 32;
            size_t bit_idx    = bit_pos % 32;
            if (!(filter[block_start + word_off] & (1U << bit_idx)))
                found = false;
        }

        if (found) {
            atomicAdd(&sdata[((unsigned int)value) % partition_size], 1);
        }
    }

    __syncthreads();
    for (int i = threadIdx.x; i < partition_size; i += blockDim.x)
        atomicAdd((unsigned long long*)&histogram[i], (unsigned long long)sdata[i]);
}

// Compacts the elements that pass the filter into a flat array.
__global__ void queryAndCompactKernel(
    const uint32_t* filter, const int32_t* idata, const int32_t* idata_val, int num_elements,
    int num_hashes, size_t num_bits,
    int32_t* odata, int32_t* odata_val, long long* g_count, int block_sz)
{
    const size_t bf_block_size = 256;
    const size_t words_per_block = bf_block_size / 32;
    size_t num_bf_blocks = num_bits / bf_block_size;

    int lane_id = threadIdx.x & 31;
    int start   = blockIdx.x * block_sz + threadIdx.x;
    int end     = min((blockIdx.x + 1) * block_sz, num_elements);

    for (int j = start; j < end; j += blockDim.x) {
        int32_t value = idata[j];

        uint32_t h1 = hash_function(value, 1);
        size_t block_idx   = h1 % num_bf_blocks;
        size_t block_start = block_idx * words_per_block;

        bool found = true;
        for (int i = 0; i < num_hashes && found; i++) {
            uint32_t hash_val = hash_function(value, i + 1);
            size_t bit_pos  = hash_val % bf_block_size;
            size_t word_off = bit_pos / 32;
            size_t bit_idx  = bit_pos % 32;
            if (!(filter[block_start + word_off] & (1U << bit_idx)))
                found = false;
        }

        uint32_t pass_mask = __ballot_sync(0xffffffff, found);

        if (pass_mask != 0) {
            int leader = __ffs(pass_mask) - 1;
            long long base_ll = 0;
            if (lane_id == leader) {
                base_ll = (long long)atomicAdd((unsigned long long*)g_count, (unsigned long long)__popc(pass_mask));
            }
            int base_lo = __shfl_sync(pass_mask, (int)(base_ll & 0xFFFFFFFF), leader);
            int base_hi = __shfl_sync(pass_mask, (int)(base_ll >> 32), leader);
            long long base = (long long)base_lo | ((long long)base_hi << 32);

            if (found) {
                int my_rank = __popc(pass_mask & ((1u << lane_id) - 1));
                odata[base + my_rank] = value;
                odata_val[base + my_rank] = idata_val[j];
            }
        }
    }
}

// Scatters the elements that pass the filter into their partition slots.
__global__ void queryAndScatterKernel(
    const uint32_t* filter, const int32_t* idata, const int32_t* idata_val, int num_elements,
    int num_hashes, size_t num_bits,
    int32_t* odata, int32_t* odata_val, long long* g_offset, int partition_size, int block_sz)
{
    const size_t bf_block_size = 256;
    const size_t words_per_block = bf_block_size / 32;
    size_t num_bf_blocks = num_bits / bf_block_size;

    int start = blockIdx.x * block_sz + threadIdx.x;
    int end   = min((blockIdx.x + 1) * block_sz, num_elements);

    for (int j = start; j < end; j += blockDim.x) {
        int32_t value = idata[j];

        uint32_t h1 = hash_function(value, 1);
        size_t block_idx = h1 % num_bf_blocks;
        size_t block_start = block_idx * words_per_block;

        bool found = true;
        for (int i = 0; i < num_hashes && found; i++) {
            uint32_t hash_val = hash_function(value, i + 1);
            size_t bit_pos    = hash_val % bf_block_size;
            size_t word_off   = bit_pos / 32;
            size_t bit_idx    = bit_pos % 32;
            if (!(filter[block_start + word_off] & (1U << bit_idx)))
                found = false;
        }

        if (found) {
            long long pos = (long long)atomicAdd((unsigned long long*)&g_offset[((unsigned int)value) % partition_size], 1ULL);
            odata[pos] = value;
            odata_val[pos] = idata_val[j];
        }
    }
}

// Reads R once and performs both the filter insert and the partition histogram.
__global__ void insertAndHistKernel(
    uint32_t* filter, const int32_t* data, size_t num_elements,
    int num_hashes, size_t num_bits,
    long long* a_prefix, int partition_size, long long block_size)
{
    extern __shared__ int sdata[];
    for (int i = threadIdx.x; i < partition_size; i += blockDim.x) sdata[i] = 0;
    __syncthreads();

    const size_t bf_block_size   = 256;
    const size_t words_per_block = bf_block_size / 32;
    size_t num_bf_blocks = num_bits / bf_block_size;

    constexpr int Theta = 4;
    constexpr int Phi   = 2;
    auto tile = cooperative_groups::tiled_partition<Theta>(
        cooperative_groups::this_thread_block());
    int cg_rank = tile.thread_rank();
    int cg_id   = threadIdx.x / Theta;
    int global_cg_id = blockIdx.x * (blockDim.x / Theta) + cg_id;
    size_t my_word_start = (size_t)cg_rank * Phi;

    if (global_cg_id < (long long)num_elements) {
        int32_t value = data[global_cg_id];

        if (cg_rank == 0)
            atomicAdd(&sdata[(unsigned)value % partition_size], 1);

        uint32_t h_block = (cg_rank == 0) ? hash_function(value, 1) : 0;
        h_block = tile.shfl(h_block, 0);
        size_t block_start = (h_block % num_bf_blocks) * words_per_block;

        for (int k = 0; k < num_hashes; k++) {
            uint32_t hash_val = hash_function(value, k + 1);
            size_t bit_pos = hash_val % bf_block_size;
            size_t target_word = bit_pos / 32;
            size_t bit_idx = bit_pos % 32;
            if (target_word >= my_word_start && target_word < my_word_start + Phi) {
                atomicOr(&filter[block_start + target_word], (1U << bit_idx));
            }
        }
    }

    __syncthreads();
    for (int i = threadIdx.x; i < partition_size; i += blockDim.x)
        atomicAdd((unsigned long long*)&a_prefix[i], (unsigned long long)sdata[i]);
}

// Inserts one partition's R range into the filter segment reserved for that partition.
__global__ void insertKernelPartitioned(
    uint32_t* filter_seg, const int32_t* data, long long start, long long end,
    int num_hashes, size_t num_bits_pp)
{
    long long n = end - start;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t value = data[start + idx];

    const size_t bf_block_size   = 256;
    const size_t words_per_block = bf_block_size / 32;
    size_t num_bf_blocks = num_bits_pp / bf_block_size;

    uint32_t h1 = hash_function(value, 1);
    size_t block_start = (h1 % num_bf_blocks) * words_per_block;

    for (int i = 0; i < num_hashes; i++) {
        uint32_t hash_val = hash_function(value, i + 1);
        size_t bit_pos  = hash_val % bf_block_size;
        size_t word_off = bit_pos / 32;
        size_t bit_idx  = bit_pos % 32;
        atomicOr(&filter_seg[block_start + word_off], (1U << bit_idx));
    }
}

// Queries only the caller's own partition segment and compacts the survivors into a flat array.
__global__ void queryAndCompactKernelPartitioned(
    const uint32_t* filter, const int32_t* idata, const int32_t* idata_val, int num_elements,
    int num_hashes, size_t num_bits_pp, size_t per_partition_words, int partition_size,
    int32_t* odata, int32_t* odata_val, long long* g_count, int block_sz)
{
    const size_t bf_block_size   = 256;
    const size_t words_per_block = bf_block_size / 32;
    size_t num_bf_blocks = num_bits_pp / bf_block_size;

    int lane_id = threadIdx.x & 31;
    int start   = blockIdx.x * block_sz + threadIdx.x;
    int end     = min((blockIdx.x + 1) * block_sz, num_elements);

    for (int j = start; j < end; j += blockDim.x) {
        int32_t value = idata[j];

        size_t seg_base = (size_t)((unsigned)value % partition_size) * per_partition_words;
        uint32_t h1 = hash_function(value, 1);
        size_t block_start = seg_base + (h1 % num_bf_blocks) * words_per_block;

        bool found = true;
        for (int i = 0; i < num_hashes && found; i++) {
            uint32_t hash_val = hash_function(value, i + 1);
            size_t bit_pos  = hash_val % bf_block_size;
            size_t word_off = bit_pos / 32;
            size_t bit_idx  = bit_pos % 32;
            if (!(filter[block_start + word_off] & (1U << bit_idx)))
                found = false;
        }

        uint32_t pass_mask = __ballot_sync(0xffffffff, found);
        if (pass_mask != 0) {
            int leader = __ffs(pass_mask) - 1;
            long long base_ll = 0;
            if (lane_id == leader) {
                base_ll = (long long)atomicAdd((unsigned long long*)g_count, (unsigned long long)__popc(pass_mask));
            }
            int base_lo = __shfl_sync(pass_mask, (int)(base_ll & 0xFFFFFFFF), leader);
            int base_hi = __shfl_sync(pass_mask, (int)(base_ll >> 32), leader);
            long long base = (long long)base_lo | ((long long)base_hi << 32);

            if (found) {
                int my_rank = __popc(pass_mask & ((1u << lane_id) - 1));
                odata[base + my_rank] = value;
                odata_val[base + my_rank] = idata_val[j];
            }
        }
    }
}

