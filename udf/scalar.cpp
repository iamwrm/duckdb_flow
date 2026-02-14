#include "crunch.h"

// Pure scalar — one element at a time, no SIMD.
// Compiled with -fno-tree-vectorize -fno-tree-slp-vectorize to prevent
// the compiler from secretly emitting NEON/SSE behind our back.

void crunch_scalar(const int64_t *__restrict in,
                   int64_t       *__restrict out,
                   idx_t n)
{
    for (idx_t i = 0; i < n; i++) {
        auto x = static_cast<uint64_t>(in[i]);
        for (int r = 0; r < CRUNCH_ROUNDS; r++) {
            x ^= (x >> 13);
            x += (x << 7);
            x ^= (x >> 17);
            x += (x << 11);
        }
        out[i] = static_cast<int64_t>(x);
    }
}
