#ifndef BLOOM_FILTER_CUH
#define BLOOM_FILTER_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <nvshmem.h>

class BloomFilter {
private:
    uint32_t* d_filter;
    uint32_t* h_filter;
    size_t num_bits;
    size_t num_words;
    int num_hashes;

public:
    BloomFilter(size_t bits, int hashes);
    ~BloomFilter();

    // Inserts every element of a device resident table into the filter.
    void buildFromTable(int *gpu_data, size_t num_elements);

    void queryFromTable(int *gpu_data, size_t num_elements);
    void printStats();
    void verifyNoFalseNegatives(const char* build_file, const char* query_file);

    uint32_t* getDeviceFilter() const { return d_filter; }
    size_t getNumWords() const { return num_words; }
    size_t getNumBits() const { return num_bits; }
    int getNumHashes() const { return num_hashes; }


private:
    void copyToDevice();
    void copyToHost();
};

// Sets k bits per element in the filter.
__global__ void insertKernel(uint32_t* filter, const int32_t* data,
                            size_t num_elements, int num_hashes, size_t num_bits);

// Queries the filter for every element and records the outcome in results.
__global__ void queryKernel(const uint32_t* filter, const int32_t* data,
                           bool* results, size_t num_elements,
                           int num_hashes, size_t num_bits);

// Determines exact membership with a linear scan.
__global__ void exactMatchKernel(const int32_t* build_data, size_t build_size,
                                 const int32_t* query_data, size_t query_size,
                                 bool* exact_results);

// Scatters the elements that pass the filter into their partition slots.
__global__ void queryAndScatterKernel(
    const uint32_t* filter, const int32_t* idata, const int32_t* idata_val, int num_elements,
    int num_hashes, size_t num_bits,
    int32_t* odata, int32_t* odata_val, long long* g_offset, int partition_size, int block_sz);

// Counts the elements that pass the filter per partition without moving them.
__global__ void queryAndHistOnlyKernel(
    const uint32_t* filter, const int32_t* data, int num_elements,
    int num_hashes, size_t num_bits,
    long long* histogram, int partition_size, int block_sz);

// Compacts the elements that pass the filter into a flat array.
__global__ void queryAndCompactKernel(
    const uint32_t* filter, const int32_t* idata, const int32_t* idata_val, int num_elements,
    int num_hashes, size_t num_bits,
    int32_t* odata, int32_t* odata_val, long long* g_count, int block_sz);

// Reads R once and performs both the filter insert and the partition histogram.
__global__ void insertAndHistKernel(
    uint32_t* filter, const int32_t* data, size_t num_elements,
    int num_hashes, size_t num_bits,
    long long* a_prefix, int partition_size, long long block_size);

// Inserts one partition's R range into the filter segment reserved for that partition.
__global__ void insertKernelPartitioned(
    uint32_t* filter_seg, const int32_t* data, long long start, long long end,
    int num_hashes, size_t num_bits_pp);

// Queries only the caller's own partition segment and compacts the survivors into a flat array.
__global__ void queryAndCompactKernelPartitioned(
    const uint32_t* filter, const int32_t* idata, const int32_t* idata_val, int num_elements,
    int num_hashes, size_t num_bits_pp, size_t per_partition_words, int partition_size,
    int32_t* odata, int32_t* odata_val, long long* g_count, int block_sz);

// Hash built on the MurmurHash3 finalizer.
__device__ uint32_t hash_function(int32_t value, uint32_t seed);

#endif // BLOOM_FILTER_CUH
