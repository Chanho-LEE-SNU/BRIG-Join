#ifndef S_REDISTRIBUTE_H
#define S_REDISTRIBUTE_H

// Moves the S candidates that passed the filter between PEs so every PE holds a similar count.
// Returns this PE's count after redistribution.
long long redistribute_survivors(
    int me, int device_num,
    int* d_prefiltered_S, int* d_prefiltered_S_val, long long local_count, long long split_size,
    int* d_send_sym, int* d_recv_sym, int* d_send_sym_val, int* d_recv_sym_val, long long* d_sizes_sym,
    double skew_threshold,
    double& out_plan_sec, double& out_exchange_sec, bool& out_did_exchange,
    long long& out_total);

#endif
