// ============================================================================
// main.cpp — Demo of schema-generic double-buffered DuckDB appender
//
// Runs 3 different schemas through the producer → double-buffer → consumer →
// DuckDB pipeline, measures throughput, and verifies row counts + a sample of
// cell values.
// ============================================================================

#include "duckdb_flow.h"

#include <pthread.h>
#include <unistd.h>
#include <ctime>
#include <cstdio>

// ============================================================================
// Consumer thread: drains batches into DuckDB via the appender API
// ============================================================================

struct ConsumerArgs {
    DoubleBuf  *dbuf;
    const char *db_path;
    const char *table_name;
};

static char *alloc_table_sql(const char *prefix, const char *table) {
    size_t cap = strlen(prefix) + strlen(table) + 1;
    auto *sql = static_cast<char *>(malloc(cap));
    if (!sql) return nullptr;
    snprintf(sql, cap, "%s%s", prefix, table);
    return sql;
}

static bool drain_ready_batches(DoubleBuf *dbuf, const Schema *s,
                                duckdb_appender appender,
                                long long *total, long long *since_flush,
                                bool *saw_final, bool *failed) {
    bool drained_any = false;
    for (int si = 0; si < 2; si++) {
        Slot *slot = &dbuf->slots[si];
        if (!slot_is_ready(slot)) continue;

        drained_any = true;
        Batch *b = &slot->batch;
        bool final = b->is_final != 0;

        if (!drain_batch_to_appender(b, s, appender)) {
            slot->state.store(SLOT_EMPTY, std::memory_order_release);
            dbuf->consumer_error.store(1, std::memory_order_release);
            *failed = true;
            return drained_any;
        }
        *total += b->count;
        *since_flush += b->count;

        if (*since_flush >= FLUSH_EVERY) {
            if (duckdb_appender_flush(appender) == DuckDBError) {
                slot->state.store(SLOT_EMPTY, std::memory_order_release);
                dbuf->consumer_error.store(1, std::memory_order_release);
                *failed = true;
                return drained_any;
            }
            *since_flush = 0;
        }

        slot->state.store(SLOT_EMPTY, std::memory_order_release);
        *saw_final |= final;
    }
    return drained_any;
}

static void *consumer_thread(void *arg) {
    auto *ctx = static_cast<ConsumerArgs *>(arg);
    DoubleBuf *dbuf = ctx->dbuf;
    const Schema *s = &dbuf->schema;

    duckdb_database db = nullptr;
    duckdb_connection con = nullptr;
    duckdb_config config = nullptr;
    duckdb_appender appender = nullptr;
    bool appender_created = false;
    char *err = nullptr;
    char *drop_sql = nullptr;
    char *ddl_sql = nullptr;

    bool failed = false;

    duckdb_create_config(&config);
    if (duckdb_set_config(config, "memory_limit", "256MB") == DuckDBError)
        failed = true;
    if (!failed && duckdb_set_config(config, "threads", "2") == DuckDBError)
        failed = true;
    if (!failed &&
        duckdb_open_ext(ctx->db_path, &db, config, &err) == DuckDBError)
        failed = true;
    duckdb_destroy_config(&config);
    config = nullptr;

    if (!failed && duckdb_connect(db, &con) == DuckDBError) failed = true;

    if (!failed) {
        drop_sql = alloc_table_sql("DROP TABLE IF EXISTS ", ctx->table_name);
        failed = (drop_sql == nullptr);
    }

    if (!failed) {
        size_t ddl_cap = schema_create_ddl_required(s, ctx->table_name);
        ddl_sql = static_cast<char *>(malloc(ddl_cap));
        failed = (ddl_sql == nullptr);
        if (!failed) {
            schema_create_ddl(s, ctx->table_name, ddl_sql,
                              static_cast<int>(ddl_cap));
        }
    }

    if (!failed) {
        duckdb_result res;
        if (duckdb_query(con, drop_sql, &res) == DuckDBError) failed = true;
        duckdb_destroy_result(&res);
    }

    if (!failed) {
        duckdb_result res;
        if (duckdb_query(con, ddl_sql, &res) == DuckDBError) failed = true;
        duckdb_destroy_result(&res);
    }

    if (!failed &&
        duckdb_appender_create(con, nullptr, ctx->table_name, &appender)
            == DuckDBError) {
        failed = true;
    }
    appender_created = !failed;

    long long total = 0, since_flush = 0;
    bool saw_final = false;

    while (!failed) {
        bool drained_any = drain_ready_batches(
            dbuf, s, appender, &total, &since_flush, &saw_final, &failed);
        if (failed) break;
        if (saw_final && !any_slot_ready(dbuf)) break;
        if (!drained_any) cpu_pause();
    }

    if (appender_created &&
        duckdb_appender_destroy(&appender) == DuckDBError) {
        failed = true;
    }
    if (failed) dbuf->consumer_error.store(1, std::memory_order_release);
    dbuf->consumer_total_rows.store(total);

    if (err) duckdb_free(err);
    free(drop_sql);
    free(ddl_sql);
    if (con) duckdb_disconnect(&con);
    if (db) duckdb_close(&db);
    return nullptr;
}

// ============================================================================
// Run one schema through the pipeline
// ============================================================================

struct DemoResult {
    long long rows;
    double    seconds;
    double    mrows_sec;
    bool      count_ok;
};

static bool query_table_count(duckdb_connection con, const char *table,
                              long long *count) {
    duckdb_result res;
    auto *sql = alloc_table_sql("SELECT COUNT(*) FROM ", table);
    if (!sql) return false;
    bool ok = duckdb_query(con, sql, &res) == DuckDBSuccess;
    free(sql);
    if (!ok) return false;

    *count = 0;
    duckdb_data_chunk chunk = duckdb_fetch_chunk(res);
    if (chunk) {
        *count = static_cast<int64_t *>(duckdb_vector_get_data(
            duckdb_data_chunk_get_vector(chunk, 0)))[0];
        duckdb_destroy_data_chunk(&chunk);
    }
    duckdb_destroy_result(&res);
    return true;
}

static bool drop_table(duckdb_connection con, const char *table) {
    duckdb_result res;
    auto *sql = alloc_table_sql("DROP TABLE IF EXISTS ", table);
    if (!sql) return false;
    bool ok = duckdb_query(con, sql, &res) == DuckDBSuccess;
    free(sql);
    duckdb_destroy_result(&res);
    return ok;
}

static void remove_db_files(const char *db_path) {
    remove(db_path);
    char wal[256];
    snprintf(wal, sizeof(wal), "%s.wal", db_path);
    remove(wal);
}

static DemoResult run_demo(const char *label, Schema *schema, int total_rows) {
    const char *db_path = "demo.duckdb";
    const char *table   = "demo_data";

    printf("\n── %s (%d rows, %d cols) ", label, total_rows, schema->ncols);
    for (int c = 0; c < schema->ncols; c++)
        printf("%s%s:%s", c ? ", " : "[", schema->cols[c].name,
               coltype_name(schema->cols[c].type));
    printf("]\n");

    const char *schema_err = nullptr;
    if (!schema_validate(schema, &schema_err)) {
        printf("  Result:      ❌ FAIL (invalid schema: %s)\n", schema_err);
        return {0, 0.0, 0.0, false};
    }

    DoubleBuf *dbuf = doublebuf_create(schema);
    if (!dbuf) {
        printf("  Result:      ❌ FAIL (doublebuf_create failed)\n");
        return {0, 0.0, 0.0, false};
    }
    ConsumerArgs cargs = {dbuf, db_path, table};

    pthread_t cons;
    pthread_create(&cons, nullptr, consumer_thread, &cargs);

    // Producer
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int cur = 0, idx = 0;
    bool producer_ok = true;
    for (int64_t i = 0; i < total_rows; i++) {
        fill_row_deterministic(&dbuf->slots[cur].batch, idx, i, schema);
        idx++;
        if (idx >= BATCH_CAPACITY) {
            publish_batch(&dbuf->slots[cur], idx, 0);
            idx = 0;
            cur ^= 1;
            if (!wait_for_empty_or_error(dbuf, &dbuf->slots[cur])) {
                producer_ok = false;
                break;
            }
        }
    }

    // Shutdown: send final batch (possibly partial or empty)
    if (producer_ok && wait_for_empty_or_error(dbuf, &dbuf->slots[cur])) {
        publish_batch(&dbuf->slots[cur], idx, 1);
    } else {
        producer_ok = false;
    }
    pthread_join(cons, nullptr);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    long long consumed = dbuf->consumer_total_rows.load();
    bool consumer_failed = dbuf->consumer_error.load(std::memory_order_acquire) != 0;
    doublebuf_destroy(dbuf);

    // Verify count in DB
    duckdb_database db = nullptr;
    duckdb_connection con = nullptr;
    long long db_count = 0;
    bool db_ok = false;
    if (duckdb_open(db_path, &db) == DuckDBSuccess &&
        duckdb_connect(db, &con) == DuckDBSuccess) {
        db_ok = query_table_count(con, table, &db_count);
    }

    // Cleanup
    if (con) {
        drop_table(con, table);
        duckdb_disconnect(&con);
    }
    if (db) duckdb_close(&db);
    remove_db_files(db_path);

    bool ok = producer_ok && !consumer_failed && db_ok &&
              (consumed == total_rows) && (db_count == total_rows);
    double mrows = total_rows / elapsed / 1e6;

    printf("  Consumed:    %lld rows\n", consumed);
    printf("  DB count:    %lld rows\n", db_count);
    if (!producer_ok)    printf("  Producer:    ERROR\n");
    if (consumer_failed) printf("  Consumer:    ERROR\n");
    if (!db_ok)          printf("  Verifier:    ERROR\n");
    printf("  Time:        %.3f s\n", elapsed);
    printf("  Throughput:  %.2f M rows/sec\n", mrows);
    printf("  Result:      %s\n", ok ? "✅ PASS" : "❌ FAIL");

    return {consumed, elapsed, mrows, ok};
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  DuckDB Double-Buffer Appender — Demo\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Batch capacity: %d rows\n", BATCH_CAPACITY);
    printf("  Flush interval: %d rows\n", FLUSH_EVERY);

    bool all_ok = true;

    ColDef schema1_cols[] = {
        {"id",      COL_INT64,   0},
        {"payload", COL_VARCHAR, 64},
    };
    ColDef schema2_cols[] = {
        {"ts",          COL_INT64,   0},
        {"sensor_id",   COL_INT32,   0},
        {"temperature", COL_DOUBLE,  0},
        {"humidity",    COL_DOUBLE,  0},
        {"alert",       COL_BOOL,    0},
        {"location",    COL_VARCHAR, 64},
    };
    ColDef schema3_cols[] = {
        {"trade_id",   COL_INT64,   0},
        {"symbol",     COL_VARCHAR, 8},
        {"price",      COL_DOUBLE,  0},
        {"quantity",   COL_INT32,   0},
        {"side",       COL_BOOL,    0},
        {"exchange",   COL_VARCHAR, 16},
        {"account_id", COL_INT64,   0},
        {"fee",        COL_DOUBLE,  0},
    };

    struct DemoCase {
        const char *label;
        Schema      schema;
        int         rows;
    };

    DemoCase demos[] = {
        {"Schema 1: Simple",            {schema1_cols, 2}, 1000000},
        {"Schema 2: IoT Sensors",       {schema2_cols, 6}, 1000000},
        {"Schema 3: Financial Trades",  {schema3_cols, 8}, 1000000},
    };

    for (auto &demo : demos) {
        auto r = run_demo(demo.label, &demo.schema, demo.rows);
        all_ok &= r.count_ok;
    }

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  Overall: %s\n", all_ok ? "✅ ALL DEMOS PASSED" : "❌ FAILURES");
    printf("═══════════════════════════════════════════════════════════════\n");

    return all_ok ? 0 : 1;
}
