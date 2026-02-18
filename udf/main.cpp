// DuckDB SIMD vs Scalar UDF benchmark — single translation unit
//
// Previously split across crunch.h / scalar.cpp / simd.cpp.
// crunch_scalar is wrapped in GCC push/pop_options to suppress
// auto-vectorization, replacing the separate-TU -fno-tree-vectorize flag.

#include "duckdb.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(__aarch64__) || defined(_M_ARM64)
#   include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#   include <immintrin.h>
#endif

// ── Shared types / constants ────────────────────────────────────────────────
using idx_t = uint64_t;
static constexpr int CRUNCH_ROUNDS = 32;

// ── SIMD backend label ──────────────────────────────────────────────────────
#if defined(__aarch64__) || defined(_M_ARM64)
#   define SIMD_BACKEND "NEON 4×q-reg (128-bit × 4 = 8 lanes)"
#elif defined(__x86_64__) || defined(_M_X64)
#   if defined(__AVX2__)
#       define SIMD_BACKEND "AVX2 4×ymm (256-bit × 4 = 16 lanes)"
#   else
#       define SIMD_BACKEND "SSE2 4×xmm (128-bit × 4 = 8 lanes)"
#   endif
#else
#   define SIMD_BACKEND "scalar fallback"
#endif

// ── Scalar kernel ───────────────────────────────────────────────────────────
// Suppress auto-vectorization so this stays a true scalar baseline.
// Uses GCC push/pop_options (Clang honours these as a GCC compat extension).
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize,no-tree-slp-vectorize")

static void crunch_scalar(const int64_t *__restrict in,
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

#pragma GCC pop_options

// ── SIMD kernel — explicit intrinsics, 4-register unroll ────────────────────
// Each register holds 2 (NEON/SSE2) or 4 (AVX2) int64 lanes.
// 4 independent dependency chains let the CPU pipeline them in parallel.
static void crunch_simd(const int64_t *__restrict in,
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

// ── UDF callbacks ───────────────────────────────────────────────────────────
static void crunch_scalar_udf(duckdb_function_info, duckdb_data_chunk input, duckdb_vector output) {
    idx_t n   = duckdb_data_chunk_get_size(input);
    auto *in  = static_cast<int64_t *>(duckdb_vector_get_data(duckdb_data_chunk_get_vector(input, 0)));
    auto *out = static_cast<int64_t *>(duckdb_vector_get_data(output));
    crunch_scalar(in, out, n);
}

static void crunch_simd_udf(duckdb_function_info, duckdb_data_chunk input, duckdb_vector output) {
    idx_t n   = duckdb_data_chunk_get_size(input);
    auto *in  = static_cast<int64_t *>(duckdb_vector_get_data(duckdb_data_chunk_get_vector(input, 0)));
    auto *out = static_cast<int64_t *>(duckdb_vector_get_data(output));
    crunch_simd(in, out, n);
}

// ── Helpers ─────────────────────────────────────────────────────────────────
static void check(duckdb_state state, const char *msg) {
    if (state == DuckDBError) {
        std::fprintf(stderr, "FATAL: %s\n", msg);
        std::exit(1);
    }
}

static void register_udf(duckdb_connection con, const char *name, duckdb_scalar_function_t fn) {
    duckdb_scalar_function func = duckdb_create_scalar_function();
    duckdb_scalar_function_set_name(func, name);
    duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_scalar_function_add_parameter(func, bigint);
    duckdb_scalar_function_set_return_type(func, bigint);
    duckdb_scalar_function_set_function(func, fn);
    check(duckdb_register_scalar_function(con, func), name);
    duckdb_destroy_logical_type(&bigint);
    duckdb_destroy_scalar_function(&func);
}

// ── Raw kernel benchmark ────────────────────────────────────────────────────
struct BenchResult {
    double ns_per_elem;
    double melem_per_sec;
    int64_t checksum;
};

static BenchResult bench_kernel(void (*kernel)(const int64_t *, int64_t *, idx_t),
                                const int64_t *in, int64_t *out,
                                idx_t n, int warmup, int iters)
{
    for (int w = 0; w < warmup; w++) kernel(in, out, n);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; it++) kernel(in, out, n);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ns  = std::chrono::duration<double, std::nano>(t1 - t0).count();
    double ns_elem   = total_ns / (static_cast<double>(n) * iters);
    double melem_sec = (static_cast<double>(n) * iters) / (total_ns / 1e3);

    int64_t ck = 0;
    for (idx_t i = 0; i < n; i += 1024) ck += out[i];
    return {ns_elem, melem_sec, ck};
}

// ── DuckDB query benchmark ──────────────────────────────────────────────────
struct QueryBenchResult {
    double ms;
    int64_t row_count;
};

static QueryBenchResult bench_query(duckdb_connection con, const char *sql, int iters) {
    for (int w = 0; w < 2; w++) {
        duckdb_result r;
        duckdb_query(con, sql, &r);
        duckdb_destroy_result(&r);
    }

    double best_ms = 1e18;
    int64_t rows   = 0;
    for (int it = 0; it < iters; it++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        duckdb_result r;
        duckdb_query(con, sql, &r);
        auto t1 = std::chrono::high_resolution_clock::now();
        rows = static_cast<int64_t>(duckdb_row_count(&r));
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best_ms) best_ms = ms;
        duckdb_destroy_result(&r);
    }
    return {best_ms, rows};
}

// ── Main ────────────────────────────────────────────────────────────────────
int main() {
    std::puts("═══════════════════════════════════════════════════════════════════════");
    std::puts("  DuckDB SIMD vs Scalar Benchmark — compute-heavy UDF");
    std::puts("═══════════════════════════════════════════════════════════════════════");
    std::printf("  SIMD backend:  %s\n", SIMD_BACKEND);
    std::printf("  Mixing rounds: %d  (~%d ALU ops/element)\n", CRUNCH_ROUNDS, CRUNCH_ROUNDS * 8);
    std::puts("  Scalar compiled with: #pragma GCC optimize(no-tree-vectorize,no-tree-slp-vectorize)");
    std::puts("  SIMD compiled with:   explicit intrinsics, 4-register unroll\n");

    // ── Part 1: Raw kernel ──────────────────────────────────────────────
    std::puts("── Part 1: Raw kernel (no DuckDB overhead) ───────────────────────────");

    constexpr idx_t SIZES[] = {2048, 65536, 1 << 20, 10 << 20};
    constexpr const char *LABELS[] = {"2K", "64K", "1M", "10M"};

    std::printf("  %-6s │ %11s │ %11s │ %11s │ %11s │ %s\n",
                "N", "scalar ns/el", "SIMD ns/el", "scalar Me/s", "SIMD Me/s", "speedup");
    std::printf("  %-6s─┼─%11s─┼─%11s─┼─%11s─┼─%11s─┼─%s\n",
                "──────", "───────────", "───────────",
                "───────────", "───────────", "───────");

    for (int si = 0; si < static_cast<int>(sizeof(SIZES)/sizeof(SIZES[0])); si++) {
        idx_t n = SIZES[si];
        auto *in  = static_cast<int64_t *>(std::malloc(n * sizeof(int64_t)));
        auto *out = static_cast<int64_t *>(std::malloc(n * sizeof(int64_t)));
        for (idx_t j = 0; j < n; j++) in[j] = static_cast<int64_t>(j + 1);

        int iters = (n <= 65536) ? 500 : (n <= (1 << 20)) ? 50 : 5;
        auto sc = bench_kernel(crunch_scalar, in, out, n, 3, iters);
        auto sm = bench_kernel(crunch_simd,   in, out, n, 3, iters);

        std::printf("  %-6s │ %11.2f │ %11.2f │ %11.2f │ %11.2f │ %.2fx\n",
                    LABELS[si], sc.ns_per_elem, sm.ns_per_elem,
                    sc.melem_per_sec, sm.melem_per_sec,
                    sc.ns_per_elem / sm.ns_per_elem);

        if (sc.checksum == 0 && sm.checksum == 0) std::puts("(zero)");
        std::free(in);
        std::free(out);
    }

    // ── Part 2: DuckDB end-to-end ───────────────────────────────────────
    std::puts("\n── Part 2: DuckDB end-to-end query ──────────────────────────────────");
    std::puts("  SELECT crunch_*(i) FROM generate_series(1, N)\n");

    duckdb_database db;
    duckdb_connection con;
    check(duckdb_open(nullptr, &db), "duckdb_open");
    check(duckdb_connect(db, &con),  "duckdb_connect");

    register_udf(con, "crunch_scalar", crunch_scalar_udf);
    register_udf(con, "crunch_simd",   crunch_simd_udf);

    constexpr int QSIZES[]          = {10000, 100000, 1000000, 10000000};
    constexpr const char *QLABELS[] = {"10K", "100K", "1M", "10M"};

    std::printf("  %-6s │ %11s │ %11s │ %11s │ %11s │ %s\n",
                "N", "scalar ms", "SIMD ms", "scalar Mr/s", "SIMD Mr/s", "speedup");
    std::printf("  %-6s─┼─%11s─┼─%11s─┼─%11s─┼─%11s─┼─%s\n",
                "──────", "───────────", "───────────",
                "───────────", "───────────", "───────");

    for (int qi = 0; qi < static_cast<int>(sizeof(QSIZES)/sizeof(QSIZES[0])); qi++) {
        int n = QSIZES[qi];
        char sql_s[256], sql_v[256];
        std::snprintf(sql_s, sizeof(sql_s),
            "SELECT crunch_scalar(i) FROM generate_series(1, %d) AS t(i)", n);
        std::snprintf(sql_v, sizeof(sql_v),
            "SELECT crunch_simd(i) FROM generate_series(1, %d) AS t(i)", n);

        int iters = (n <= 100000) ? 10 : 3;
        auto sc = bench_query(con, sql_s, iters);
        auto sm = bench_query(con, sql_v, iters);

        double sc_mr = static_cast<double>(sc.row_count) / (sc.ms * 1000.0);
        double sm_mr = static_cast<double>(sm.row_count) / (sm.ms * 1000.0);

        std::printf("  %-6s │ %11.2f │ %11.2f │ %11.2f │ %11.2f │ %.2fx\n",
                    QLABELS[qi], sc.ms, sm.ms, sc_mr, sm_mr, sc.ms / sm.ms);
    }

    // ── Correctness ─────────────────────────────────────────────────────
    std::puts("\n── Correctness ──────────────────────────────────────────────────────");
    {
        const char *sql = "SELECT i, crunch_scalar(i) AS s, crunch_simd(i) AS v "
                          "FROM generate_series(1, 5) AS t(i)";

        duckdb_result result;
        check(duckdb_query(con, sql, &result) == DuckDBSuccess ? DuckDBSuccess : DuckDBError,
              "correctness query");

        idx_t chunks = duckdb_result_chunk_count(result);
        bool pass = true;
        for (idx_t ci = 0; ci < chunks; ci++) {
            duckdb_data_chunk chunk = duckdb_result_get_chunk(result, ci);
            idx_t rows = duckdb_data_chunk_get_size(chunk);
            auto *ci_ = static_cast<int64_t *>(duckdb_vector_get_data(duckdb_data_chunk_get_vector(chunk, 0)));
            auto *cs  = static_cast<int64_t *>(duckdb_vector_get_data(duckdb_data_chunk_get_vector(chunk, 1)));
            auto *cv  = static_cast<int64_t *>(duckdb_vector_get_data(duckdb_data_chunk_get_vector(chunk, 2)));

            for (idx_t r = 0; r < rows; r++) {
                bool ok = cs[r] == cv[r];
                std::printf("  i=%-3ld  scalar=%-21ld  simd=%-21ld  %s\n",
                            ci_[r], cs[r], cv[r], ok ? "✅" : "❌");
                if (!ok) pass = false;
            }
            duckdb_destroy_data_chunk(&chunk);
        }
        duckdb_destroy_result(&result);
        std::printf("  %s\n", pass ? "✅ All values match" : "❌ MISMATCH");
    }

    std::puts("\n═══════════════════════════════════════════════════════════════════════");
    std::puts("  Done");
    std::puts("═══════════════════════════════════════════════════════════════════════");

    duckdb_disconnect(&con);
    duckdb_close(&db);
    return 0;
}
