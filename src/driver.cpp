#include <iostream>
#include <string>
#include "hash_kernel.h"
#include "common.h"
#include "nccl.h"

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

// Sorts both result vectors and compares them element by element.
bool compare(vector<int> *const A, vector<int> *const B) {
    if (A->size() != B->size()) return false;
    sort(begin(*A), end(*A));
    sort(begin(*B), end(*B));
    bool flag = true;
    for (int i = 0; i < A->size(); i++) {
        if (A->at(i) != B->at(i)) {
            flag = false;
            break;
        }
    }
    return flag;
}

// Parses the command line arguments and invokes gpu_main.
int main(int argc, char **argv) {
    string r_inputPath = "/datasets/leejinho/data/unique_32000000.bin";
    string s_inputPath = "/datasets/leejinho/data/unique_skew0.00_S32000000.bin";

    long long int R=32000000;
    long long int S=32000000;

    string outputPath = "/datasets/leejinho/data/output_2m.bin";
    bool sanityCheck = false;
    int use_bf           = 0;
    bool use_nvlink      = false;
    bool use_warp_compact = false;

    long long override_num_bits           = -1;
    int       override_num_hashes         = -1;
    int       override_partition_rs_split = -1;
    int       override_partition_size        = -1;
    int       override_second_partition_size = -1;
    int       fused_probe_breakdown          = 0;
    int       read_val                       = 0;
    int       r_shuffle_enable               = 0;
    double    r_replicate_ratio              = -1.0;
    double    answer_scale                   = 1.0;
    {
        vector<char*> fargv;
        fargv.push_back(argv[0]);
        for (int i = 1; i < argc; i++) {
            string a = argv[i];
            if (a.rfind("--", 0) == 0) {
                string key = a.substr(2), val;
                auto eq = key.find('=');
                if (eq != string::npos) { val = key.substr(eq + 1); key = key.substr(0, eq); }
                else if (i + 1 < argc)  { val = argv[++i]; }
                if      (key == "num_bits")              override_num_bits = stoll(val);
                else if (key == "num_hashes")            override_num_hashes = atoi(val.c_str());
                else if (key == "partition_rs_split")    override_partition_rs_split = atoi(val.c_str());
                else if (key == "partition_size")        override_partition_size = atoi(val.c_str());
                else if (key == "second_partition_size") override_second_partition_size = atoi(val.c_str());
                else if (key == "fused_probe_breakdown") fused_probe_breakdown = val.empty() ? 1 : atoi(val.c_str());
                else if (key == "read_val")              read_val = val.empty() ? 1 : atoi(val.c_str());
                else if (key == "r_shuffle_enable")      r_shuffle_enable = val.empty() ? 1 : atoi(val.c_str());
                else if (key == "r_replicate_ratio")     r_replicate_ratio = atof(val.c_str());
                else if (key == "answer_scale")          answer_scale = atof(val.c_str());
                else fprintf(stderr, "Warning: unknown option --%s (ignored)\n", key.c_str());
            } else {
                fargv.push_back(argv[i]);
            }
        }
        static vector<char*> s_fargv = fargv;
        argc = (int)s_fargv.size();
        argv = s_fargv.data();
    }
    if (override_num_bits >= 0)
        printf("Override num_bits: %lld\n", override_num_bits);
    if (override_num_hashes >= 0)
        printf("Override num_hashes: %d\n", override_num_hashes);
    if (override_partition_rs_split >= 0)
        printf("Override partition_rs_split: %d\n", override_partition_rs_split);
    if (override_partition_size >= 0)
        printf("Override partition_size: %d\n", override_partition_size);
    if (override_second_partition_size >= 0)
        printf("Override second_partition_size: %d\n", override_second_partition_size);
    if (read_val)
        printf("read_val: 1 (payloads are read from the input file, [key][value] layout)\n");
    if (r_shuffle_enable)
        printf("r_shuffle_enable: 1 (filter + partition shuffle: redistribute R and S to their owner, then a single probe)\n");
    if (r_replicate_ratio == 0.0)
        printf("r_replicate_ratio: 0 (always replicate the whole R)\n");
    else if (r_replicate_ratio > 0.0)
        printf("r_replicate_ratio: %.2f (replicate R only when global candidates / R reaches this ratio)\n", r_replicate_ratio);
    if (answer_scale != 1.0)
        printf("answer_scale: %.2f (answer buffer = split_size * %.2f)\n", answer_scale, answer_scale);

    if (argc >= 7) {
        r_inputPath = argv[1];
        s_inputPath = argv[2];
        outputPath = argv[3];
        R = stoll(argv[4]);
        S = stoll(argv[5]);
        sanityCheck = atoi(argv[6]);
    } else {
        cout << "./jointable rPath sPath outPath R S sanityCheck "
                "[chunk_size] [use_bf] [use_nvlink] "
                "[use_warp_compact] [true_false_ratio] [s_chunks] [kernel_fusion] [redist_skew_threshold]" << endl;
        cout << "  options: [--num_bits N] [--num_hashes K] [--partition_rs_split 0|1] "
                "[--partition_size N] [--second_partition_size N] [--fused_probe_breakdown 0|1] [--read_val 0|1]" << endl;
        cout << "Use sanityCheck=1 only in local" << endl;
    }
    if (argc >= 8) {
        chunk_size = atoi(argv[7]);
        int min_chunk = (thread_no + 31) / 32;
        if (chunk_size < min_chunk) {
            fprintf(stderr, "Error: chunk_size must be >= nwarps=%d (got %d)\n",
                    min_chunk, chunk_size);
            exit(EXIT_FAILURE);
        }
        printf("chunk_size: %d\n", chunk_size);
    }
    if (argc >= 9)
    {
        use_bf     = atoi(argv[8]);
        printf("BF filtering is %s\n", use_bf ? "enabled" : "disabled");
    }
    if (argc >= 10) {
        use_nvlink = atoi(argv[9]);
        printf("NVLink is %s\n", use_nvlink ? "enabled" : "disabled");
    }
    if (argc >= 11) {
        use_warp_compact = atoi(argv[10]);
        printf("Warp compact pre-filter is %s\n", use_warp_compact ? "enabled" : "disabled");
    }
    double true_false_ratio = 0.1;
    if (argc >= 12) {
        true_false_ratio = atof(argv[11]);
        printf("BF true_false_ratio: %.4f\n", true_false_ratio);
    }
    int s_chunks = 1;
    if (argc >= 13) {
        s_chunks = atoi(argv[12]);
        if (s_chunks <= 0) {
            fprintf(stderr, "Error: s_chunks must be >= 1 (got %d)\n", s_chunks);
            exit(EXIT_FAILURE);
        }
        printf("S chunks: %d\n", s_chunks);
    }
    int kernel_fusion = 0;
    if (argc >= 14) {
        kernel_fusion = atoi(argv[13]);
        const char* kf_str = (kernel_fusion == 0) ? "disabled" :
                             (kernel_fusion == 1) ? "enabled (chunked GET, all peers)" :
                             (kernel_fusion == 2) ? "hybrid (fused local+NVL, NODE via PUT loop)" :
                                                    "unknown";
        printf("Kernel fusion: %d (%s)\n", kernel_fusion, kf_str);
    }
    double redist_skew_threshold = 0.0;
    if (argc >= 15) {
        redist_skew_threshold = atof(argv[14]);
        printf("S' redistribute: %s (skew threshold=%.2f)\n",
               redist_skew_threshold > 0.0 ? "enabled" : "disabled",
               redist_skew_threshold);
    }
    cout << "=====================================" << endl;
    cout << "             Join Tables             " << endl;
    cout << "=====================================" << endl;
    cout << "The size of tableA: " << R << endl;
    cout << "The size of tableB: " << S << endl;
    cout << "Sanity Check(0: false, 1: true): " << sanityCheck << endl;
    cout << "=====================================" << endl << endl;

    vector<int> *solution = new vector<int>();
    vector<int> *solution_ref = new vector<int>();

    gpu_main(solution, solution_ref, r_inputPath, s_inputPath, R, S, use_bf, use_nvlink, use_warp_compact, true_false_ratio, s_chunks, sanityCheck, kernel_fusion, redist_skew_threshold,
             override_num_bits, override_num_hashes, override_partition_rs_split,
             override_partition_size, override_second_partition_size,
             fused_probe_breakdown, read_val, r_shuffle_enable, r_replicate_ratio, answer_scale);

    delete solution;
    delete solution_ref;

    return 0;
}
