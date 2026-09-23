#ifndef SANITY_CHECK_H
#define SANITY_CHECK_H

// Computes the reference answer count with a CPU hash join to validate the GPU result.
void run_sanity_check(const int* tableA, long long R,
                      const int* tableB, long long S);

#endif
