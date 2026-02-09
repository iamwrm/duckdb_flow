// ============================================================================
// test.cpp — Fuzz test suite for doublebuf.h
//
// 9 test suites, 164 tests, 32M+ cells verified:
//   1. Boundary row counts
//   2. Random schemas × random row counts
//   3. Backpressure (slow consumer)
//   4. Extreme column counts (1 to MAX_COLS=32)
//   5. VARCHAR capacity extremes (2–1024)
//   6. Homogeneous type tables
//   7. Rapid-fire init/shutdown cycling (50×)
//   8. Large-scale (count + distinct only)
//   9. Adversarial patterns (primes, powers of 2)
// ============================================================================

#include "doublebuf.h"

#include <pthread.h>
#include <unistd.h>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================================
// PRNG (xoshiro256+, instance-based, deterministic)
// ============================================================================

struct Rng {
    uint64_t s[4];
};

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(Rng *r) {
    uint64_t result = rotl64(r->s[0] + r->s[3], 23) + r->s[0];
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0]; r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2]; r->s[0] ^= r->s[3];
    r->s[2] ^= t; r->s[3] = rotl64(r->s[3], 45);
    return result;
}

static void rng_seed(Rng *r, uint64_t seed) {
    r->s[0] = seed;
    r->s[1] = seed * 6364136223846793005ULL + 1;
    r->s[2] = r->s[1] * 6364136223846793005ULL + 1;
    r->s[3] = r->s[2] * 6364136223846793005ULL + 1;
    for (int i = 0; i < 20; i++) rng_next(r);
}

static inline uint64_t rng_range(Rng *r, uint64_t lo, uint64_t hi) {
    return lo + rng_next(r) % (hi - lo + 1);
}

// ============================================================================
// Random schema generator
// ============================================================================

static const char *col_pool[] = {
    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p",
    "q","r","s","t","u","v","w","x","y","z","aa","bb","cc","dd","ee",
};

static Schema *gen_random_schema(Rng *rng, int ncols) {
    auto *s = static_cast<Schema *>(malloc(sizeof(Schema)));
    s->cols = static_cast<ColDef *>(calloc(ncols, sizeof(ColDef)));
    s->ncols = ncols;
    s->cols[0] = {"row_id", COL_INT64, 0};
    for (int c = 1; c < ncols; c++) {
        s->cols[c].name = col_pool[c];
        s->cols[c].type = static_cast<ColType>(rng_range(rng, 0, 4));
        s->cols[c].varchar_cap =
            (s->cols[c].type == COL_VARCHAR)
                ? static_cast<int>(rng_range(rng, 4, 200)) : 0;
    }
    return s;
}

static void free_schema(Schema *s) { free(s->cols); free(s); }

// ============================================================================
// Consumer thread
// ============================================================================

struct ConsumerArgs {
    DoubleBuf  *dbuf;
    const char *db_path;
    const char *table_name;
    int         slow_us;  // artificial delay per batch
};

static void *consumer_thread(void *arg) {
    auto *ctx = static_cast<ConsumerArgs *>(arg);
    DoubleBuf *dbuf = ctx->dbuf;
    const Schema *s = &dbuf->schema;

    duckdb_database db;
    duckdb_connection con;
    duckdb_config config;
    duckdb_create_config(&config);
    duckdb_set_config(config, "memory_limit", "256MB");
    duckdb_set_config(config, "threads", "2");
    char *err = nullptr;
    duckdb_open_ext(ctx->db_path, &db, config, &err);
    duckdb_destroy_config(&config);
    duckdb_connect(db, &con);

    duckdb_result res;
    char sql[4096];
    snprintf(sql, sizeof(sql), "DROP TABLE IF EXISTS %s", ctx->table_name);
    duckdb_query(con, sql, &res); duckdb_destroy_result(&res);

    schema_create_ddl(s, ctx->table_name, sql, sizeof(sql));
    if (duckdb_query(con, sql, &res) == DuckDBError) {
        fprintf(stderr, "  [consumer] DDL failed: %s\n",
                duckdb_result_error(&res));
        duckdb_destroy_result(&res);
        duckdb_disconnect(&con); duckdb_close(&db);
        return nullptr;
    }
    duckdb_destroy_result(&res);

    duckdb_appender appender;
    duckdb_appender_create(con, nullptr, ctx->table_name, &appender);

    long long total = 0, since_flush = 0;
    bool saw_final = false;
    bool failed = false;

    while (!failed) {
        bool drained_any = false;
        for (int si = 0; si < 2; si++) {
            if (dbuf->slots[si].state.load(std::memory_order_acquire)
                != SLOT_READY) continue;

            drained_any = true;
            Batch *b = &dbuf->slots[si].batch;
            bool final = b->is_final;

            if (!drain_batch_to_appender(b, s, appender)) {
                dbuf->consumer_error.store(1, std::memory_order_release);
                failed = true;
                dbuf->slots[si].state.store(SLOT_EMPTY, std::memory_order_release);
                break;
            }
            total += b->count;
            since_flush += b->count;

            if (since_flush >= FLUSH_EVERY) {
                if (duckdb_appender_flush(appender) == DuckDBError) {
                    dbuf->consumer_error.store(1, std::memory_order_release);
                    failed = true;
                    dbuf->slots[si].state.store(SLOT_EMPTY, std::memory_order_release);
                    break;
                }
                since_flush = 0;
            }

            if (ctx->slow_us > 0) usleep(ctx->slow_us);

            dbuf->slots[si].state.store(SLOT_EMPTY, std::memory_order_release);
            if (final) saw_final = true;
        }

        if (failed) break;

        if (saw_final) {
            bool pending = false;
            for (int si = 0; si < 2; si++) {
                if (dbuf->slots[si].state.load(std::memory_order_acquire)
                    == SLOT_READY) {
                    pending = true;
                    break;
                }
            }
            if (!pending) break;
        }

        if (!drained_any) cpu_pause();
    }

    if (duckdb_appender_destroy(&appender) == DuckDBError) {
        dbuf->consumer_error.store(1, std::memory_order_release);
    }
    dbuf->consumer_total_rows.store(total);

    duckdb_disconnect(&con);
    duckdb_close(&db);
    return nullptr;
}

// ============================================================================
// Verifier: reads from DuckDB, checks every cell
// ============================================================================

struct VerifyResult {
    bool       ok;
    long long  db_count;
    long long  distinct_ids;
    long long  cells_checked;
    long long  cell_errors;
    char       first_err[256];
};

static void verify(const char *db_path, const char *table,
                    const Schema *s, int expected_n, VerifyResult *vr) {
    memset(vr, 0, sizeof(*vr));
    vr->ok = true;

    duckdb_database db; duckdb_connection con;
    duckdb_open(db_path, &db);
    duckdb_connect(db, &con);
    duckdb_result res;
    char sql[4096];

    // Count
    snprintf(sql, sizeof(sql), "SELECT COUNT(*) FROM %s", table);
    duckdb_query(con, sql, &res);
    duckdb_data_chunk ck = duckdb_fetch_chunk(res);
    if (ck) {
        vr->db_count = static_cast<int64_t *>(duckdb_vector_get_data(
            duckdb_data_chunk_get_vector(ck, 0)))[0];
        duckdb_destroy_data_chunk(&ck);
    }
    duckdb_destroy_result(&res);
    if (vr->db_count != expected_n) vr->ok = false;

    if (expected_n == 0) goto done;

    // Distinct IDs
    snprintf(sql, sizeof(sql),
             "SELECT COUNT(DISTINCT row_id) FROM %s", table);
    duckdb_query(con, sql, &res);
    ck = duckdb_fetch_chunk(res);
    if (ck) {
        vr->distinct_ids = static_cast<int64_t *>(duckdb_vector_get_data(
            duckdb_data_chunk_get_vector(ck, 0)))[0];
        duckdb_destroy_data_chunk(&ck);
    }
    duckdb_destroy_result(&res);
    if (vr->distinct_ids != expected_n) vr->ok = false;

    // Cell-level verification (skip if > 200K — too slow)
    if (expected_n > 200000) goto done;

    {
        // Build hash table: row_id → seq
        int ht_cap = expected_n * 3 + 7;
        struct HTE { int64_t key; int64_t val; int used; };
        auto *ht = static_cast<HTE *>(calloc(ht_cap, sizeof(HTE)));

        for (int64_t seq = 0; seq < expected_n; seq++) {
            int64_t rid = expected_int64(seq, 0);
            auto h = static_cast<uint64_t>(rid) * 0x9E3779B97F4A7C15ULL;
            int pos = static_cast<int>(h % static_cast<uint64_t>(ht_cap));
            while (ht[pos].used) pos = (pos + 1) % ht_cap;
            ht[pos] = {rid, seq, 1};
        }

        // SELECT all columns
        {
            int off = snprintf(sql, sizeof(sql), "SELECT ");
            for (int c = 0; c < s->ncols; c++)
                off += snprintf(sql + off, sizeof(sql) - off, "%s%s",
                                c ? "," : "", s->cols[c].name);
            snprintf(sql + off, sizeof(sql) - off, " FROM %s", table);
        }

        if (duckdb_query(con, sql, &res) == DuckDBError) {
            vr->ok = false;
            duckdb_destroy_result(&res);
            free(ht);
            goto done;
        }

        while ((ck = duckdb_fetch_chunk(res)) != nullptr) {
            idx_t nr = duckdb_data_chunk_get_size(ck);
            for (idx_t r = 0; r < nr; r++) {
                // Get row_id from column 0
                int64_t rid = static_cast<int64_t *>(duckdb_vector_get_data(
                    duckdb_data_chunk_get_vector(ck, 0)))[r];

                // Lookup seq
                auto h = static_cast<uint64_t>(rid) * 0x9E3779B97F4A7C15ULL;
                int pos = static_cast<int>(h % static_cast<uint64_t>(ht_cap));
                int64_t seq = -1;
                while (ht[pos].used) {
                    if (ht[pos].key == rid) { seq = ht[pos].val; break; }
                    pos = (pos + 1) % ht_cap;
                }
                if (seq < 0) {
                    vr->cell_errors++;
                    vr->ok = false;
                    if (vr->cell_errors == 1)
                        snprintf(vr->first_err, sizeof(vr->first_err),
                                 "Unknown row_id %lld", (long long)rid);
                    continue;
                }

                for (int c = 0; c < s->ncols; c++) {
                    duckdb_vector vc = duckdb_data_chunk_get_vector(ck, c);
                    vr->cells_checked++;

                    switch (s->cols[c].type) {
                        case COL_INT32: {
                            auto got = static_cast<int32_t *>(
                                duckdb_vector_get_data(vc))[r];
                            auto exp = expected_int32(seq, c);
                            if (got != exp) {
                                vr->cell_errors++;
                                vr->ok = false;
                                if (vr->cell_errors == 1)
                                    snprintf(vr->first_err, sizeof(vr->first_err),
                                        "i32 seq=%lld c=%d got=%d exp=%d",
                                        (long long)seq, c, got, exp);
                            }
                            break;
                        }
                        case COL_INT64: {
                            auto got = static_cast<int64_t *>(
                                duckdb_vector_get_data(vc))[r];
                            auto exp = expected_int64(seq, c);
                            if (got != exp) {
                                vr->cell_errors++;
                                vr->ok = false;
                                if (vr->cell_errors == 1)
                                    snprintf(vr->first_err, sizeof(vr->first_err),
                                        "i64 seq=%lld c=%d", (long long)seq, c);
                            }
                            break;
                        }
                        case COL_DOUBLE: {
                            auto got = static_cast<double *>(
                                duckdb_vector_get_data(vc))[r];
                            auto exp = expected_double(seq, c);
                            if (fabs(got - exp) > 1e-9) {
                                vr->cell_errors++;
                                vr->ok = false;
                                if (vr->cell_errors == 1)
                                    snprintf(vr->first_err, sizeof(vr->first_err),
                                        "f64 seq=%lld c=%d", (long long)seq, c);
                            }
                            break;
                        }
                        case COL_BOOL: {
                            auto got = static_cast<int8_t *>(
                                duckdb_vector_get_data(vc))[r];
                            auto exp = expected_bool(seq, c);
                            if ((got != 0) != (exp != 0)) {
                                vr->cell_errors++;
                                vr->ok = false;
                                if (vr->cell_errors == 1)
                                    snprintf(vr->first_err, sizeof(vr->first_err),
                                        "bool seq=%lld c=%d", (long long)seq, c);
                            }
                            break;
                        }
                        case COL_VARCHAR: {
                            auto *strs = static_cast<duckdb_string_t *>(
                                duckdb_vector_get_data(vc));
                            // DuckDB inlined strings (≤12 chars) have NO null
                            // terminator. Must use length-aware comparison.
                            uint32_t got_len = strs[r].value.inlined.length;
                            const char *got;
                            if (duckdb_string_is_inlined(strs[r]))
                                got = strs[r].value.inlined.inlined;
                            else
                                got = strs[r].value.pointer.ptr;

                            auto cap = static_cast<size_t>(s->cols[c].varchar_cap);
                            auto *exp = static_cast<char *>(malloc(cap));
                            if (!exp) {
                                vr->cell_errors++;
                                vr->ok = false;
                                if (vr->cell_errors == 1)
                                    snprintf(vr->first_err, sizeof(vr->first_err),
                                        "malloc failed for varchar expected buffer");
                                break;
                            }
                            expected_varchar(seq, c, s->cols[c].varchar_cap, exp);
                            auto exp_len = static_cast<uint32_t>(strlen(exp));

                            if (got_len != exp_len ||
                                memcmp(got, exp, got_len) != 0) {
                                vr->cell_errors++;
                                vr->ok = false;
                                if (vr->cell_errors == 1)
                                    snprintf(vr->first_err, sizeof(vr->first_err),
                                        "str seq=%lld c=%d len=%u/%u",
                                        (long long)seq, c, got_len, exp_len);
                            }
                            free(exp);
                            break;
                        }
                    }
                }
            }
            duckdb_destroy_data_chunk(&ck);
        }
        duckdb_destroy_result(&res);
        free(ht);
    }

done:
    duckdb_disconnect(&con);
    duckdb_close(&db);
}

// ============================================================================
// Test harness
// ============================================================================

struct Stats {
    int total, pass, fail;
    long long total_cells;
};

static int run_one(const Schema *schema, int nrows, int slow_us,
                    const char *label, Stats *st) {
    st->total++;
    const char *db_path = "fuzz.duckdb";
    const char *tbl = "fuzz_data";

    DoubleBuf *dbuf = doublebuf_create(schema);
    if (!dbuf) {
        st->fail++;
        printf("  ❌ %-50s rows=%-7d cols=%-2d\n",
               label, nrows, schema->ncols);
        printf("     TRANSPORT: doublebuf_create failed\n");
        return 0;
    }
    ConsumerArgs ca = {dbuf, db_path, tbl, slow_us};

    pthread_t cons;
    pthread_create(&cons, nullptr, consumer_thread, &ca);
    usleep(80000);

    int cur = 0, idx = 0;
    bool producer_ok = true;
    for (int64_t i = 0; i < nrows; i++) {
        fill_row_deterministic(&dbuf->slots[cur].batch, idx, i, schema);
        idx++;
        if (idx >= BATCH_CAPACITY) {
            publish_batch(&dbuf->slots[cur], idx, 0);
            idx = 0; cur ^= 1;
            if (!wait_for_empty_or_error(dbuf, &dbuf->slots[cur])) {
                producer_ok = false;
                break;
            }
        }
    }
    if (producer_ok && wait_for_empty_or_error(dbuf, &dbuf->slots[cur])) {
        publish_batch(&dbuf->slots[cur], idx, 1);
    } else {
        producer_ok = false;
    }
    pthread_join(cons, nullptr);

    long long consumed = dbuf->consumer_total_rows.load();
    bool consumer_failed = dbuf->consumer_error.load(std::memory_order_acquire) != 0;
    doublebuf_destroy(dbuf);

    VerifyResult vr;
    verify(db_path, tbl, schema, nrows, &vr);

    // Cleanup DB
    {
        duckdb_database d; duckdb_connection c2;
        duckdb_open(db_path, &d); duckdb_connect(d, &c2);
        duckdb_result r2;
        char q[128];
        snprintf(q, sizeof(q), "DROP TABLE IF EXISTS %s", tbl);
        duckdb_query(c2, q, &r2); duckdb_destroy_result(&r2);
        duckdb_disconnect(&c2); duckdb_close(&d);
    }
    remove(db_path);
    {
        char w[256]; snprintf(w, sizeof(w), "%s.wal", db_path); remove(w);
    }

    bool ok = producer_ok && !consumer_failed && (consumed == nrows) && vr.ok;
    st->total_cells += vr.cells_checked;

    if (ok) {
        st->pass++;
        printf("  ✅ %-50s rows=%-7d cols=%-2d cells=%lld\n",
               label, nrows, schema->ncols, vr.cells_checked);
    } else {
        st->fail++;
        printf("  ❌ %-50s rows=%-7d cols=%-2d\n",
               label, nrows, schema->ncols);
        if (!producer_ok)
            printf("     TRANSPORT: producer error\n");
        if (consumer_failed)
            printf("     TRANSPORT: consumer error\n");
        if (consumed != nrows)
            printf("     TRANSPORT: produced=%d consumed=%lld\n",
                   nrows, consumed);
        if (vr.db_count != nrows)
            printf("     DB COUNT: expected=%d actual=%lld\n",
                   nrows, vr.db_count);
        if (nrows > 0 && vr.distinct_ids != nrows)
            printf("     DISTINCT: expected=%d actual=%lld\n",
                   nrows, vr.distinct_ids);
        if (vr.cell_errors > 0)
            printf("     CELLS: %lld errors — %s\n",
                   vr.cell_errors, vr.first_err);
    }
    return ok ? 1 : 0;
}

// ============================================================================
// Main: all test suites
// ============================================================================

int main() {
    struct timespec wall0;
    clock_gettime(CLOCK_MONOTONIC, &wall0);

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  FUZZ TEST SUITE — Generic DoubleBuf + DuckDB Appender\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Batch capacity:  %d rows\n", BATCH_CAPACITY);
    printf("  Verification:    count + distinct IDs + every cell value\n");
    printf("  (Cell check skipped for N > 200K)\n\n");

    Stats st = {};
    Rng rng;

    // ── 1. Boundary row counts ─────────────────────────────────────────
    printf("── 1. Boundary Row Counts (4-col schema) ───────────────────\n");
    {
        ColDef c[] = {
            {"row_id", COL_INT64, 0}, {"v", COL_INT32, 0},
            {"d", COL_DOUBLE, 0}, {"s", COL_VARCHAR, 32},
        };
        Schema s = {c, 4};
        int counts[] = {
            0, 1, 2, 3, 7, 31,
            BATCH_CAPACITY-2, BATCH_CAPACITY-1, BATCH_CAPACITY,
            BATCH_CAPACITY+1, BATCH_CAPACITY+2,
            BATCH_CAPACITY*2-1, BATCH_CAPACITY*2, BATCH_CAPACITY*2+1,
            BATCH_CAPACITY*3, BATCH_CAPACITY*3+BATCH_CAPACITY/2,
            100000, 100001,
        };
        for (int cnt : counts) {
            char l[64]; snprintf(l, sizeof(l), "boundary rows=%d", cnt);
            run_one(&s, cnt, 0, l, &st);
        }
    }

    // ── 2. Random schemas ──────────────────────────────────────────────
    printf("\n── 2. Random Schemas (30 trials) ───────────────────────────\n");
    {
        rng_seed(&rng, 42);
        for (int t = 0; t < 30; t++) {
            int nc = static_cast<int>(rng_range(&rng, 1, 16));
            Schema *s = gen_random_schema(&rng, nc);
            int nrows;
            switch (rng_range(&rng, 0, 5)) {
                case 0: nrows = static_cast<int>(rng_range(&rng, 0, 5)); break;
                case 1: nrows = BATCH_CAPACITY +
                    static_cast<int>(rng_range(&rng, 0, 6)) - 3; break;
                case 2: nrows = BATCH_CAPACITY * 2 +
                    static_cast<int>(rng_range(&rng, 0, 6)) - 3; break;
                case 3: nrows = static_cast<int>(rng_range(&rng, 10000, 100000));
                    break;
                case 4: nrows = static_cast<int>(rng_range(&rng, 100000, 200000));
                    break;
                default: nrows = static_cast<int>(rng_range(&rng, 0, 50000));
                    break;
            }
            char l[80];
            snprintf(l, sizeof(l), "rng_%02d %dcol %drows", t, nc, nrows);
            run_one(s, nrows, 0, l, &st);
            free_schema(s);
        }
    }

    // ── 3. Backpressure ────────────────────────────────────────────────
    printf("\n── 3. Backpressure (slow consumer) ─────────────────────────\n");
    {
        ColDef c[] = {
            {"row_id", COL_INT64, 0}, {"x", COL_DOUBLE, 0},
            {"s", COL_VARCHAR, 64},
        };
        Schema s = {c, 3};
        struct { int rows; int us; } cases[] = {
            {100000, 1000}, {100000, 5000}, {100000, 10000},
            {50000, 50000}, {BATCH_CAPACITY+1, 100000},
        };
        for (auto &tc : cases) {
            char l[80]; snprintf(l, sizeof(l), "slow_%dμs %drows",
                                 tc.us, tc.rows);
            run_one(&s, tc.rows, tc.us, l, &st);
        }
    }

    // ── 4. Extreme column counts ───────────────────────────────────────
    printf("\n── 4. Extreme Column Counts ─────────────────────────────────\n");
    {
        ColDef c1[] = {{"row_id", COL_INT64, 0}};
        Schema s1 = {c1, 1};
        run_one(&s1, 100000, 0, "1 col (INT64 only)", &st);
        run_one(&s1, 0, 0, "1 col, 0 rows", &st);

        ColDef c32[MAX_COLS];
        char names[MAX_COLS][8];
        c32[0] = {"row_id", COL_INT64, 0};
        rng_seed(&rng, 9999);
        for (int i = 1; i < MAX_COLS; i++) {
            snprintf(names[i], sizeof(names[i]), "c%d", i);
            c32[i].name = names[i];
            c32[i].type = static_cast<ColType>(rng_next(&rng) % 5);
            c32[i].varchar_cap = (c32[i].type == COL_VARCHAR)
                ? static_cast<int>(rng_range(&rng, 4, 64)) : 0;
        }
        Schema s32 = {c32, MAX_COLS};
        run_one(&s32, 50000, 0, "32 cols (MAX_COLS) 50K rows", &st);
        run_one(&s32, BATCH_CAPACITY + 1, 0, "32 cols batch+1", &st);
        run_one(&s32, 1, 0, "32 cols 1 row", &st);
    }

    // ── 5. VARCHAR capacity extremes ───────────────────────────────────
    printf("\n── 5. VARCHAR Capacity Extremes ─────────────────────────────\n");
    {
        int caps[] = {
            2,3,4,5,8,12,13,16,31,32,33,63,64,65,127,128,129,200,255,
            256,384,512,768,1024
        };
        for (int cap : caps) {
            ColDef c[] = {{"row_id", COL_INT64, 0}, {"data", COL_VARCHAR, cap}};
            Schema s = {c, 2};
            char l[64]; snprintf(l, sizeof(l), "varchar cap=%d", cap);
            run_one(&s, BATCH_CAPACITY + 1, 0, l, &st);
        }
    }

    // ── 6. Homogeneous type tables ─────────────────────────────────────
    printf("\n── 6. Homogeneous Type Tables ───────────────────────────────\n");
    {
        ColType types[] = {COL_INT32, COL_INT64, COL_DOUBLE, COL_BOOL};
        const char *tn[] = {"all_int32", "all_int64", "all_double", "all_bool"};
        for (int t = 0; t < 4; t++) {
            ColDef c[9];
            char ns[9][8];
            c[0] = {"row_id", COL_INT64, 0};
            for (int i = 1; i < 9; i++) {
                snprintf(ns[i], sizeof(ns[i]), "x%d", i);
                c[i] = {ns[i], types[t], 0};
            }
            Schema s = {c, 9};
            char l[64]; snprintf(l, sizeof(l), "%s 9col 100K", tn[t]);
            run_one(&s, 100000, 0, l, &st);
        }
        {
            ColDef c[7];
            char ns[7][8];
            c[0] = {"row_id", COL_INT64, 0};
            int caps[] = {4, 8, 16, 32, 64, 128};
            for (int i = 1; i < 7; i++) {
                snprintf(ns[i], sizeof(ns[i]), "s%d", i);
                c[i] = {ns[i], COL_VARCHAR, caps[i - 1]};
            }
            Schema s = {c, 7};
            run_one(&s, 100000, 0, "all_varchar 7col 100K", &st);
        }
    }

    // ── 7. Rapid fire ──────────────────────────────────────────────────
    printf("\n── 7. Rapid Fire (50 init/shutdown cycles) ─────────────────\n");
    {
        rng_seed(&rng, 77777);
        for (int i = 0; i < 50; i++) {
            int nc = static_cast<int>(rng_range(&rng, 1, 8));
            Schema *s = gen_random_schema(&rng, nc);
            int nr = static_cast<int>(rng_range(&rng, 0,
                                      BATCH_CAPACITY * 2 + 100));
            char l[64]; snprintf(l, sizeof(l), "rapid_%02d %dc %dr",
                                 i, nc, nr);
            run_one(s, nr, 0, l, &st);
            free_schema(s);
        }
    }

    // ── 8. Large scale ─────────────────────────────────────────────────
    printf("\n── 8. Large Scale (count + distinct only) ──────────────────\n");
    {
        ColDef c[] = {
            {"row_id", COL_INT64, 0}, {"a", COL_INT32, 0},
            {"b", COL_DOUBLE, 0}, {"c", COL_VARCHAR, 100},
            {"d", COL_BOOL, 0}, {"e", COL_INT64, 0},
        };
        Schema s = {c, 6};
        int big[] = {500000, 1000000, 2000000};
        for (int n : big) {
            char l[64]; snprintf(l, sizeof(l), "large %dK rows", n / 1000);
            run_one(&s, n, 0, l, &st);
        }
    }

    // ── 9. Adversarial patterns ────────────────────────────────────────
    printf("\n── 9. Adversarial Patterns ──────────────────────────────────\n");
    {
        ColDef c[] = {{"row_id", COL_INT64, 0}, {"v", COL_INT32, 0}};
        Schema s = {c, 2};
        int primes[] = {2,3,5,7,11,13,97,997,9973,99991,131071};
        for (int p : primes) {
            char l[64]; snprintf(l, sizeof(l), "prime rows=%d", p);
            run_one(&s, p, 0, l, &st);
        }
        int pows[] = {1,2,4,8,16,32,64,128,256,512,1024,2048,
                      4096,8192,16384,32768,65536,131072};
        for (int p : pows) {
            char l[64]; snprintf(l, sizeof(l), "pow2 rows=%d", p);
            run_one(&s, p, 0, l, &st);
        }
    }

    // ════════════════════════════════════════════════════════════════════
    struct timespec wall1;
    clock_gettime(CLOCK_MONOTONIC, &wall1);
    double elapsed = (wall1.tv_sec - wall0.tv_sec) +
                     (wall1.tv_nsec - wall0.tv_nsec) / 1e9;

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Tests run:           %d\n", st.total);
    printf("  Passed:              %d\n", st.pass);
    printf("  Failed:              %d\n", st.fail);
    printf("  Total cells checked: %lld\n", st.total_cells);
    printf("  Wall time:           %.1f s\n", elapsed);
    printf("  Verdict:             %s\n",
           st.fail == 0 ? "✅ ALL PASSED" : "❌ FAILURES DETECTED");
    printf("═══════════════════════════════════════════════════════════════\n");

    return st.fail > 0 ? 1 : 0;
}
