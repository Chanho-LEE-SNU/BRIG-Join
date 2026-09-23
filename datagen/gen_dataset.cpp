/*
 * Generate synthetic join datasets in the binary format expected by driver.cpp.
 * R and S are raw arrays of int32 written to binary files.
 *
 * Usage:
 *   ./gen_dataset <R_size> <S_size> <answer_pct> <out_dir> [seed] [--max <pct>] [--verify]
 *
 *   answer_pct : 0.0 ~ 100.0  (mean % of R values that have a match in S)
 *   --max <pct>: (optional) enable LOAD IMBALANCE. The matching values are
 *                packed into the first 25% of S ("hot" region) at this max match
 *                rate, while the remaining 75% ("cold" region) is lowered so the
 *                OVERALL rate still equals answer_pct. R stays uniform. Since
 *                each GPU owns one contiguous N-th of S, the first GPU(s) then
 *                produce most of the join output. Without --max, S is uniform.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <unordered_set>

static void write_bin(const std::string &path, const int *data, long long n)
{
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", path.c_str()); exit(1); }
    long long written = fwrite(data, sizeof(int), n, f);
    if (written != n) { fprintf(stderr, "Write failed for %s\n", path.c_str()); exit(1); }
    fclose(f);
}

static int *read_bin(const std::string &path, long long &n)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open %s for reading\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long long file_size = ftell(f);
    rewind(f);
    n = file_size / sizeof(int);
    int *data = new int[n];
    long long read = fread(data, sizeof(int), n, f);
    if (read != n) { fprintf(stderr, "Read failed for %s\n", path.c_str()); exit(1); }
    fclose(f);
    return data;
}

// Count actual matches between R and S
static long long count_matches(const int *tableR, long long R, const int *tableS, long long S)
{
    std::unordered_set<int> s_values(tableS, tableS + S);
    long long matches = 0;
    for (long long i = 0; i < R; i++) {
        if (s_values.count(tableR[i])) {
            matches++;
        }
    }
    return matches;
}

// Fisher-Yates shuffle using a fast 32-bit LCG.
// Shuffles arr[0..n); pass arr+offset to shuffle a sub-range independently.
static void shuffle(int *arr, long long n, uint64_t seed)
{
    // xoshiro256** would be faster, but pcg32 is fine here
    // Use std::mt19937_64 for quality randomness
    std::mt19937_64 rng(seed);
    for (long long i = n - 1; i > 0; --i) {
        // uniform in [0, i]
        uint64_t j = rng() % (uint64_t)(i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <R_size> <S_size> <answer_pct> <out_dir> [seed] [--max <pct>] [--verify]\n"
            "  answer_pct: 0.0 ~ 100.0  (mean %% of R rows that match something in S)\n"
            "  --max <pct>: (optional) load imbalance: first 25%% of S at this rate,\n"
            "               rest lowered so the overall rate stays answer_pct\n"
            "  --verify:   (optional) verify selectivity after generation\n",
            argv[0]);
        return 1;
    }

    long long R     = atoll(argv[1]);
    long long S     = atoll(argv[2]);
    double    pct   = atof(argv[3]);
    std::string dir = argv[4];

    // Optional args (in any order after the 4 required positionals):
    //   a bare number -> seed,  "--max <pct>" -> imbalance,  "--verify" -> verify
    uint64_t seed    = 42ULL;
    bool     verify  = false;
    double   max_pct = -1.0;   // <0 means "not set" => uniform (no imbalance)
    for (int i = 5; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--verify") {
            verify = true;
        } else if (a == "--max") {
            if (i + 1 >= argc) { fprintf(stderr, "--max requires a value\n"); return 1; }
            max_pct = atof(argv[++i]);
        } else {
            seed = (uint64_t)atoll(argv[i]);
        }
    }

    if (pct < 0.0 || pct > 100.0) { fprintf(stderr, "answer_pct must be 0~100\n"); return 1; }

    // Remove trailing slash
    if (!dir.empty() && dir.back() == '/') dir.pop_back();

    long long k = (long long)(R * pct / 100.0);
    if (k > S) k = S;

    printf("R=%lld  S=%lld  matches=%lld  answer=%.4f%%\n",
           R, S, k, (double)k / R * 100.0);

    // ---- Load-imbalance setup --------------------------------------------
    // When --max is given, concentrate matches into the first HOT_FRAC of S
    // (R stays uniform). Each GPU owns one contiguous N-th of S, so packing the
    // matching values into S's first quarter makes the first GPU(s) produce
    // most of the join output while the rest stay sparse.
    // The total number of matches stays exactly k (so selectivity and the
    // value-range guarantees below are unchanged); we only redistribute them:
    //   hot region  = first HOT_FRAC of S, filled at max_pct match rate
    //   cold region = remaining rows, lowered so the overall rate stays pct
    const double HOT_FRAC = 0.25;
    bool      imbalance = (max_pct >= 0.0);
    long long hot_n   = imbalance ? (long long)(S * HOT_FRAC) : 0;
    long long cold_n  = S - hot_n;
    long long m_hot = 0, m_cold = k;   // matches placed in each region
    if (imbalance) {
        if (max_pct < pct || max_pct > 100.0) {
            fprintf(stderr,
                "--max must satisfy answer_pct <= max <= 100 (got mean=%.4f, max=%.4f)\n",
                pct, max_pct);
            return 1;
        }
        m_hot = (long long)(hot_n * max_pct / 100.0);
        if (m_hot > hot_n) m_hot = hot_n;        // can't exceed rows in hot region
        if (m_hot > k) {                          // would need more matches than exist
            fprintf(stderr,
                "--max=%.4f too high for mean=%.4f: hot region needs %lld matches "
                "but only %lld exist. Require max <= mean / HOT_FRAC = %.4f\n",
                max_pct, pct, m_hot, k, pct / HOT_FRAC);
            return 1;
        }
        m_cold = k - m_hot;
        if (m_cold > cold_n) {                    // leftover matches don't fit the cold region
            fprintf(stderr,
                "Infeasible: cold region holds %lld rows but needs %lld matches.\n",
                cold_n, m_cold);
            return 1;
        }
        printf("Load imbalance: hot %lld rows @ %.4f%% (%lld matches) | "
               "cold %lld rows @ %.4f%% (%lld matches)\n",
               hot_n, (double)m_hot / hot_n * 100.0, m_hot,
               cold_n, (double)m_cold / cold_n * 100.0, m_cold);
    }

    // Allocate
    int *tableR = new int[R];
    int *tableS = new int[S];

    // Value assignment strategy (int range: [-2^31, 2^31-1]):
    //   matching values : [0,                    k)
    //   R-only values   : [k,            INT32_MAX)  - wraps if needed
    //   S-only values   : [INT32_MIN,         -1)  - wraps if needed
    // This ensures R-only and S-only NEVER overlap
    // => Exact selectivity is GUARANTEED: exactly k values will match

    const long long R_ONLY_RANGE = (long long)INT32_MAX - k;  // positive range available

    printf("Filling R (matching + R-only)...\n");
    // R stays uniform: every matching value [0,k) appears once, spread evenly
    // by the full shuffle below.
    for (long long i = 0; i < k; i++)
        tableR[i] = (int)i;  // matching values: [0, k)
    for (long long i = k; i < R; i++) {
        long long offset = (i - k) % R_ONLY_RANGE;
        tableR[i] = (int)(k + offset);  // R-only: [k, INT32_MAX)
    }

    printf("Shuffling R...\n");
    shuffle(tableR, R, seed);

    printf("Filling S (matching + S-only)...\n");
    // Lay out S as [ hot region | cold region ]; each region gets its share of
    // matching values ([0,k), drawn via mc) and S-only values ([R,R+S), via sj).
    // Total matching rows == k and total S-only rows == S-k regardless of the
    // split, so selectivity stays exact. (Non-imbalance => hot_n==0, m_cold==k,
    // i.e. the whole array is one region with k matches: identical to old code.)
    const long long S_ONLY_BASE  = R;                       // S-only values start here
    const long long S_ONLY_RANGE2 = (long long)INT32_MAX - R;
    long long mc = 0;   // next matching value -> [0, k)
    long long sj = 0;   // next S-only value   -> R + (sj % S_ONLY_RANGE2), i.e. [R, R+S)
    long long p  = 0;
    for (long long c = 0; c < m_hot;           c++) tableS[p++] = (int)(mc++);
    for (long long c = 0; c < hot_n - m_hot;   c++) tableS[p++] = (int)(S_ONLY_BASE + (sj++) % S_ONLY_RANGE2);
    for (long long c = 0; c < m_cold;          c++) tableS[p++] = (int)(mc++);
    for (long long c = 0; c < cold_n - m_cold; c++) tableS[p++] = (int)(S_ONLY_BASE + (sj++) % S_ONLY_RANGE2);
    assert(p == S && mc == k && sj == S - k);

    if (imbalance) {
        // Shuffle each region independently so the per-region match density is
        // preserved (the hot region stays dense, the cold region stays sparse).
        printf("Shuffling S (hot + cold regions separately)...\n");
        shuffle(tableS,         hot_n,  seed ^ 0xdeadbeefULL);
        shuffle(tableS + hot_n, cold_n, seed ^ 0x9e3779b97f4a7c15ULL);
    } else {
        printf("Shuffling S...\n");
        shuffle(tableS, S, seed ^ 0xdeadbeefULL);
    }

    // Build output filenames (encode max rate when imbalance is enabled)
    char pct_str[64];
    if (imbalance)
        snprintf(pct_str, sizeof(pct_str), "%.1f_max%.1f", pct, max_pct);
    else
        snprintf(pct_str, sizeof(pct_str), "%.1f", pct);
    std::string r_path = dir + "/gen_R" + std::to_string(R) + "_ans" + pct_str + "pct.bin";
    std::string s_path = dir + "/gen_S" + std::to_string(S) + "_ans" + pct_str + "pct.bin";

    printf("Writing R -> %s\n", r_path.c_str());
    write_bin(r_path, tableR, R);

    printf("Writing S -> %s\n", s_path.c_str());
    write_bin(s_path, tableS, S);

    // Verify selectivity (if requested)
    if (verify) {
        printf("\n=== VERIFICATION ===\n");
        long long actual_matches = count_matches(tableR, R, tableS, S);
        double actual_pct = (double)actual_matches / R * 100.0;
        printf("Expected matches: %lld (%.4f%%)\n", k, (double)k / R * 100.0);
        printf("Actual matches:   %lld (%.4f%%)\n", actual_matches, actual_pct);
        if (actual_matches == k) {
            printf("✓ Selectivity is CORRECT\n");
        } else {
            printf("✗ ERROR: Selectivity mismatch! (expected %lld, got %lld)\n", k, actual_matches);
        }
        printf("===================\n\n");
    } else {
        printf("\n(Selectivity guaranteed by construction)\n\n");
    }

    long long r_mb = R * sizeof(int) / (1 << 20);
    long long s_mb = S * sizeof(int) / (1 << 20);
    printf("Done.  R: %lld MB,  S: %lld MB\n", r_mb, s_mb);
    printf("\nRun with:\n  ./jointable %s %s <output.bin> %lld %lld 0\n",
           r_path.c_str(), s_path.c_str(), R, S);

    delete[] tableR;
    delete[] tableS;
    return 0;
}
