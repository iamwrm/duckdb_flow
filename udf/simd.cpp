#include "crunch.h"

// ── SIMD headers ────────────────────────────────────────────────────────────
#if defined(__aarch64__) || defined(_M_ARM64)
#   include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#   include <immintrin.h>
#endif

// ── SIMD kernel — explicit intrinsics, 4-register unroll ────────────────────
// Each register holds 2 (NEON/SSE2) or 4 (AVX2) int64 lanes.
// We process 4 registers per iteration so the CPU has 4 independent
// dependency chains to pipeline.  Scalar OoO can't match this.
//
// Mixing: x ^= x>>13;  x += x<<7;  x ^= x>>17;  x += x<<11;  (×32 rounds)
// All ops (shl, srl, xor, add) have 1:1 SIMD counterparts.

void crunch_simd(const int64_t *__restrict in,
                 int64_t       *__restrict out,
                 idx_t n)
{
    idx_t i = 0;

#if defined(__aarch64__) || defined(_M_ARM64)

    // ── NEON: 4 × q-register = 8 elements / iteration ──────────────────
    for (; i + 8 <= n; i += 8) {
        uint64x2_t a = vld1q_u64(reinterpret_cast<const uint64_t *>(in + i));
        uint64x2_t b = vld1q_u64(reinterpret_cast<const uint64_t *>(in + i + 2));
        uint64x2_t c = vld1q_u64(reinterpret_cast<const uint64_t *>(in + i + 4));
        uint64x2_t d = vld1q_u64(reinterpret_cast<const uint64_t *>(in + i + 6));

        for (int r = 0; r < CRUNCH_ROUNDS; r++) {
            a = veorq_u64(a, vshrq_n_u64(a, 13));
            b = veorq_u64(b, vshrq_n_u64(b, 13));
            c = veorq_u64(c, vshrq_n_u64(c, 13));
            d = veorq_u64(d, vshrq_n_u64(d, 13));

            a = vaddq_u64(a, vshlq_n_u64(a, 7));
            b = vaddq_u64(b, vshlq_n_u64(b, 7));
            c = vaddq_u64(c, vshlq_n_u64(c, 7));
            d = vaddq_u64(d, vshlq_n_u64(d, 7));

            a = veorq_u64(a, vshrq_n_u64(a, 17));
            b = veorq_u64(b, vshrq_n_u64(b, 17));
            c = veorq_u64(c, vshrq_n_u64(c, 17));
            d = veorq_u64(d, vshrq_n_u64(d, 17));

            a = vaddq_u64(a, vshlq_n_u64(a, 11));
            b = vaddq_u64(b, vshlq_n_u64(b, 11));
            c = vaddq_u64(c, vshlq_n_u64(c, 11));
            d = vaddq_u64(d, vshlq_n_u64(d, 11));
        }

        vst1q_u64(reinterpret_cast<uint64_t *>(out + i),     a);
        vst1q_u64(reinterpret_cast<uint64_t *>(out + i + 2), b);
        vst1q_u64(reinterpret_cast<uint64_t *>(out + i + 4), c);
        vst1q_u64(reinterpret_cast<uint64_t *>(out + i + 6), d);
    }

    // ── NEON tail: 2 elements ───────────────────────────────────────────
    for (; i + 2 <= n; i += 2) {
        uint64x2_t x = vld1q_u64(reinterpret_cast<const uint64_t *>(in + i));
        for (int r = 0; r < CRUNCH_ROUNDS; r++) {
            x = veorq_u64(x, vshrq_n_u64(x, 13));
            x = vaddq_u64(x, vshlq_n_u64(x, 7));
            x = veorq_u64(x, vshrq_n_u64(x, 17));
            x = vaddq_u64(x, vshlq_n_u64(x, 11));
        }
        vst1q_u64(reinterpret_cast<uint64_t *>(out + i), x);
    }

#elif defined(__x86_64__) || defined(_M_X64)
#   if defined(__AVX2__)

    // ── AVX2: 4 × ymm = 16 elements / iteration ────────────────────────
    for (; i + 16 <= n; i += 16) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i + 4));
        __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i + 8));
        __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i + 12));

        for (int r = 0; r < CRUNCH_ROUNDS; r++) {
            a = _mm256_xor_si256(a, _mm256_srli_epi64(a, 13));
            b = _mm256_xor_si256(b, _mm256_srli_epi64(b, 13));
            c = _mm256_xor_si256(c, _mm256_srli_epi64(c, 13));
            d = _mm256_xor_si256(d, _mm256_srli_epi64(d, 13));

            a = _mm256_add_epi64(a, _mm256_slli_epi64(a, 7));
            b = _mm256_add_epi64(b, _mm256_slli_epi64(b, 7));
            c = _mm256_add_epi64(c, _mm256_slli_epi64(c, 7));
            d = _mm256_add_epi64(d, _mm256_slli_epi64(d, 7));

            a = _mm256_xor_si256(a, _mm256_srli_epi64(a, 17));
            b = _mm256_xor_si256(b, _mm256_srli_epi64(b, 17));
            c = _mm256_xor_si256(c, _mm256_srli_epi64(c, 17));
            d = _mm256_xor_si256(d, _mm256_srli_epi64(d, 17));

            a = _mm256_add_epi64(a, _mm256_slli_epi64(a, 11));
            b = _mm256_add_epi64(b, _mm256_slli_epi64(b, 11));
            c = _mm256_add_epi64(c, _mm256_slli_epi64(c, 11));
            d = _mm256_add_epi64(d, _mm256_slli_epi64(d, 11));
        }

        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i),      a);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i + 4),  b);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i + 8),  c);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i + 12), d);
    }

    // ── AVX2 tail: 4 elements ───────────────────────────────────────────
    for (; i + 4 <= n; i += 4) {
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i));
        for (int r = 0; r < CRUNCH_ROUNDS; r++) {
            x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 13));
            x = _mm256_add_epi64(x, _mm256_slli_epi64(x, 7));
            x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 17));
            x = _mm256_add_epi64(x, _mm256_slli_epi64(x, 11));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), x);
    }

#   else

    // ── SSE2: 4 × xmm = 8 elements / iteration ─────────────────────────
    for (; i + 8 <= n; i += 8) {
        __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i + 2));
        __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i + 4));
        __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i + 6));

        for (int r = 0; r < CRUNCH_ROUNDS; r++) {
            a = _mm_xor_si128(a, _mm_srli_epi64(a, 13));
            b = _mm_xor_si128(b, _mm_srli_epi64(b, 13));
            c = _mm_xor_si128(c, _mm_srli_epi64(c, 13));
            d = _mm_xor_si128(d, _mm_srli_epi64(d, 13));

            a = _mm_add_epi64(a, _mm_slli_epi64(a, 7));
            b = _mm_add_epi64(b, _mm_slli_epi64(b, 7));
            c = _mm_add_epi64(c, _mm_slli_epi64(c, 7));
            d = _mm_add_epi64(d, _mm_slli_epi64(d, 7));

            a = _mm_xor_si128(a, _mm_srli_epi64(a, 17));
            b = _mm_xor_si128(b, _mm_srli_epi64(b, 17));
            c = _mm_xor_si128(c, _mm_srli_epi64(c, 17));
            d = _mm_xor_si128(d, _mm_srli_epi64(d, 17));

            a = _mm_add_epi64(a, _mm_slli_epi64(a, 11));
            b = _mm_add_epi64(b, _mm_slli_epi64(b, 11));
            c = _mm_add_epi64(c, _mm_slli_epi64(c, 11));
            d = _mm_add_epi64(d, _mm_slli_epi64(d, 11));
        }

        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i),     a);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i + 2), b);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i + 4), c);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i + 6), d);
    }

    // ── SSE2 tail: 2 elements ───────────────────────────────────────────
    for (; i + 2 <= n; i += 2) {
        __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
        for (int r = 0; r < CRUNCH_ROUNDS; r++) {
            x = _mm_xor_si128(x, _mm_srli_epi64(x, 13));
            x = _mm_add_epi64(x, _mm_slli_epi64(x, 7));
            x = _mm_xor_si128(x, _mm_srli_epi64(x, 17));
            x = _mm_add_epi64(x, _mm_slli_epi64(x, 11));
        }
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), x);
    }

#   endif
#endif

    // ── Scalar tail ─────────────────────────────────────────────────────
    for (; i < n; i++) {
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
