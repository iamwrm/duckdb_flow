#include "duckdb.h"
#include "crunch.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

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
    std::puts("  Scalar compiled with: -fno-tree-vectorize -fno-tree-slp-vectorize");
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
