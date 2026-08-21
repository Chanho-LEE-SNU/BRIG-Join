#include "sanity_check.h"
#include <cstdio>
#include <cmath>
#include <climits>
#include <atomic>
#include <chrono>
#include <omp.h>

// Computes the reference answer count with a CPU hash join to validate the GPU result.
void run_sanity_check(const int* tableA, long long R,
                      const int* tableB, long long S)
{
    printf("[sanityCheck] Building open-addressing HT from R (%lld entries)...\n", R);
    auto cpu_join_start = std::chrono::steady_clock::now();

    const int EMPTY_KEY = INT_MIN;
    const long long HT_CAP  = 1LL << (int)std::ceil(std::log2((double)R * 2));
    const long long HT_MASK = HT_CAP - 1;

    std::atomic<int>* ht_key = new std::atomic<int>[HT_CAP];
    std::atomic<int>* ht_cnt = new std::atomic<int>[HT_CAP];
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < HT_CAP; i++) {
        ht_key[i].store(EMPTY_KEY, std::memory_order_relaxed);
        ht_cnt[i].store(0,         std::memory_order_relaxed);
    }

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < R; i++) {
        int key = tableA[i];
        long long h = ((unsigned long long)(unsigned int)key * 2654435761ULL) & HT_MASK;
        while (true) {
            int expected = EMPTY_KEY;
            if (ht_key[h].compare_exchange_strong(expected, key,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                ht_cnt[h].fetch_add(1, std::memory_order_relaxed);
                break;
            }
            if (expected == key) {
                ht_cnt[h].fetch_add(1, std::memory_order_relaxed);
                break;
            }
            h = (h + 1) & HT_MASK;
        }
    }

    auto cpu_build_end = std::chrono::steady_clock::now();
    printf("[sanityCheck] Probing S (%lld entries) w/ %d threads...\n", S, omp_get_max_threads());

    long long cpu_answer_num = 0;
    #pragma omp parallel for reduction(+:cpu_answer_num) schedule(static)
    for (long long j = 0; j < S; j++) {
        int key = tableB[j];
        long long h = ((unsigned long long)(unsigned int)key * 2654435761ULL) & HT_MASK;
        while (ht_key[h].load(std::memory_order_relaxed) != EMPTY_KEY) {
            if (ht_key[h].load(std::memory_order_relaxed) == key) {
                cpu_answer_num += ht_cnt[h].load(std::memory_order_relaxed);
                break;
            }
            h = (h + 1) & HT_MASK;
        }
    }

    auto cpu_join_end = std::chrono::steady_clock::now();
    printf("[sanityCheck] Answer Num (hash-based): %lld\n", cpu_answer_num);
    printf("[sanityCheck] === CPU Join Timing ===\n");
    printf("[sanityCheck]   Build: %.4f sec\n", std::chrono::duration<double>(cpu_build_end - cpu_join_start).count());
    printf("[sanityCheck]   Probe: %.4f sec\n", std::chrono::duration<double>(cpu_join_end  - cpu_build_end ).count());
    printf("[sanityCheck]   Total: %.4f sec\n", std::chrono::duration<double>(cpu_join_end  - cpu_join_start).count());

    delete[] ht_key;
    delete[] ht_cnt;
}
