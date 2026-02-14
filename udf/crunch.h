#pragma once
#include <cstdint>

using idx_t = uint64_t;

static constexpr int CRUNCH_ROUNDS = 32;

// Scalar kernel — must be compiled with -fno-tree-vectorize -fno-slp-vectorize
void crunch_scalar(const int64_t *__restrict in,
                   int64_t       *__restrict out,
                   idx_t n);

// SIMD kernel — explicit intrinsics, unrolled
void crunch_simd(const int64_t *__restrict in,
                 int64_t       *__restrict out,
                 idx_t n);
