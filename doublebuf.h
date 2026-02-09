#pragma once
// ============================================================================
// doublebuf.h — Schema-generic double-buffered appender for DuckDB
//
// Hot-thread-safe column-oriented batching with lock-free ping-pong handoff.
// The producer (hot thread) never touches DuckDB; the consumer (background
// thread) drains batches through the DuckDB C appender API.
//
// Usage:
//   1. Define a Schema (array of ColDef)
//   2. Create a DoubleBuf
//   3. Producer fills rows via batch_set_* and publishes with publish_batch()
//   4. Consumer drains via drain_batch_to_appender()
//   5. Shutdown: producer sends a batch with is_final=1
// ============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <atomic>
#include <thread>
#include <new>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "duckdb.h"

// ============================================================================
// Schema Definition
// ============================================================================

enum ColType : int {
    COL_INT32   = 0,
    COL_INT64   = 1,
    COL_DOUBLE  = 2,
    COL_BOOL    = 3,
    COL_VARCHAR = 4,
};

struct ColDef {
    const char *name;
    ColType     type;
    int         varchar_cap;   // only for COL_VARCHAR
};

// Legacy/demo baseline used by tests. The implementation itself has no
// internal hard cap on schema columns.
static constexpr int MAX_COLS = 32;

struct Schema {
    ColDef *cols;
    int     ncols;
};

inline const char *coltype_sql_name(ColType t) {
    switch (t) {
        case COL_INT32:   return "INTEGER";
        case COL_INT64:   return "BIGINT";
        case COL_DOUBLE:  return "DOUBLE";
        case COL_BOOL:    return "BOOLEAN";
        case COL_VARCHAR: return "VARCHAR";
    }
    return "VARCHAR";
}

inline const char *coltype_name(ColType t) {
    switch (t) {
        case COL_INT32:   return "INT32";
        case COL_INT64:   return "INT64";
        case COL_DOUBLE:  return "DOUBLE";
        case COL_BOOL:    return "BOOL";
        case COL_VARCHAR: return "VARCHAR";
    }
    return "?";
}

inline bool schema_validate(const Schema *s, const char **err_msg = nullptr) {
    auto fail = [&](const char *msg) -> bool {
        if (err_msg) *err_msg = msg;
        return false;
    };

    if (!s) return fail("schema is null");
    if (!s->cols) return fail("schema columns are null");
    if (s->ncols <= 0) return fail("schema must have at least one column");

    for (int i = 0; i < s->ncols; i++) {
        const ColDef &col = s->cols[i];
        if (!col.name || !col.name[0])
            return fail("column name is null or empty");

        switch (col.type) {
            case COL_INT32:
            case COL_INT64:
            case COL_DOUBLE:
            case COL_BOOL:
                break;
            case COL_VARCHAR:
                if (col.varchar_cap < 2)
                    return fail("varchar_cap must be >= 2");
                break;
            default:
                return fail("invalid column type");
        }
    }
    return true;
}

inline size_t schema_create_ddl_required(const Schema *s, const char *table) {
    size_t need = strlen("CREATE TABLE ") + strlen(table) + strlen(" ()") + 1;
    for (int i = 0; i < s->ncols; i++) {
        need += (i ? 2 : 0);  // ", "
        need += strlen(s->cols[i].name);
        need += 1;            // space
        need += strlen(coltype_sql_name(s->cols[i].type));
    }
    return need;
}

inline void schema_create_ddl(const Schema *s, const char *table,
                              char *buf, int bufsz) {
    if (bufsz <= 0) return;

    int off = snprintf(buf, bufsz, "CREATE TABLE %s (", table);
    if (off < 0 || off >= bufsz) {
        buf[bufsz - 1] = '\0';
        return;
    }

    for (int i = 0; i < s->ncols; i++) {
        int rem = bufsz - off;
        if (rem <= 0) break;
        int w = snprintf(buf + off, rem, "%s%s %s",
                         i ? ", " : "", s->cols[i].name,
                         coltype_sql_name(s->cols[i].type));
        if (w < 0) {
            buf[bufsz - 1] = '\0';
            return;
        }
        if (w >= rem) {
            off = bufsz - 1;
            break;
        }
        off += w;
    }

    if (off < bufsz) {
        snprintf(buf + off, bufsz - off, ")");
    } else {
        buf[bufsz - 1] = '\0';
    }
}

// ============================================================================
// Column-Oriented Batch
// ============================================================================

static constexpr int BATCH_CAPACITY = 32768;

struct Batch {
    void **col_data;
    int   *col_stride;
    int    ncols;
    int    count;
    int    is_final;
};

inline int col_storage_size(const ColDef *col) {
    switch (col->type) {
        case COL_INT32:   return sizeof(int32_t);
        case COL_INT64:   return sizeof(int64_t);
        case COL_DOUBLE:  return sizeof(double);
        case COL_BOOL:    return sizeof(int8_t);
        case COL_VARCHAR: return col->varchar_cap;
    }
    return 1;
}

inline void batch_free(Batch *b) {
    if (!b) return;
    if (b->col_data) {
        for (int c = 0; c < b->ncols; c++) free(b->col_data[c]);
    }
    free(b->col_data);
    free(b->col_stride);
    memset(b, 0, sizeof(*b));
}

inline bool batch_alloc(Batch *b, const Schema *s) {
    memset(b, 0, sizeof(*b));
    b->ncols = s->ncols;
    b->col_data = static_cast<void **>(calloc(s->ncols, sizeof(void *)));
    b->col_stride = static_cast<int *>(calloc(s->ncols, sizeof(int)));
    if (!b->col_data || !b->col_stride) {
        batch_free(b);
        return false;
    }

    for (int c = 0; c < s->ncols; c++) {
        int es = col_storage_size(&s->cols[c]);
        b->col_stride[c] = es;
        b->col_data[c] = calloc(BATCH_CAPACITY, es);
        if (!b->col_data[c]) {
            batch_free(b);
            return false;
        }
    }
    return true;
}

// ── Type-safe setters (hot path — compile to single indexed store) ─────────

inline void batch_set_int32(Batch *b, int col, int row, int32_t v) {
    static_cast<int32_t *>(b->col_data[col])[row] = v;
}
inline void batch_set_int64(Batch *b, int col, int row, int64_t v) {
    static_cast<int64_t *>(b->col_data[col])[row] = v;
}
inline void batch_set_double(Batch *b, int col, int row, double v) {
    static_cast<double *>(b->col_data[col])[row] = v;
}
inline void batch_set_bool(Batch *b, int col, int row, int8_t v) {
    static_cast<int8_t *>(b->col_data[col])[row] = v;
}
inline void batch_set_varchar(Batch *b, int col, int row, const char *v) {
    char *dst = static_cast<char *>(b->col_data[col]) + row * b->col_stride[col];
    strncpy(dst, v, b->col_stride[col] - 1);
    dst[b->col_stride[col] - 1] = '\0';
}

// ── Getters ────────────────────────────────────────────────────────────────

inline int32_t batch_get_int32(const Batch *b, int col, int row) {
    return static_cast<int32_t *>(b->col_data[col])[row];
}
inline int64_t batch_get_int64(const Batch *b, int col, int row) {
    return static_cast<int64_t *>(b->col_data[col])[row];
}
inline double batch_get_double(const Batch *b, int col, int row) {
    return static_cast<double *>(b->col_data[col])[row];
}
inline int8_t batch_get_bool(const Batch *b, int col, int row) {
    return static_cast<int8_t *>(b->col_data[col])[row];
}
inline const char *batch_get_varchar(const Batch *b, int col, int row) {
    return static_cast<const char *>(b->col_data[col]) + row * b->col_stride[col];
}
inline char *batch_get_varchar_mut(Batch *b, int col, int row) {
    return static_cast<char *>(b->col_data[col]) + row * b->col_stride[col];
}

// ============================================================================
// Double Buffer (Lock-Free Ping-Pong)
// ============================================================================

static constexpr int CACHELINE = 64;

enum SlotState : int { SLOT_EMPTY = 0, SLOT_READY = 1 };

struct alignas(CACHELINE) Slot {
    alignas(CACHELINE) std::atomic<int> state{SLOT_EMPTY};
    alignas(CACHELINE) Batch batch{};
};

struct DoubleBuf {
    Slot    slots[2];
    Schema  schema{};
    alignas(CACHELINE) std::atomic<long long> consumer_total_rows{0};
    alignas(CACHELINE) std::atomic<int> consumer_error{0};
};

inline bool slot_is_ready(const Slot *slot) {
    return slot->state.load(std::memory_order_acquire) == SLOT_READY;
}

inline bool any_slot_ready(const DoubleBuf *db) {
    for (int i = 0; i < 2; i++) {
        if (slot_is_ready(&db->slots[i])) return true;
    }
    return false;
}

inline char *dup_cstr(const char *s) {
    size_t n = strlen(s) + 1;
    auto *out = static_cast<char *>(malloc(n));
    if (!out) return nullptr;
    memcpy(out, s, n);
    return out;
}

inline void schema_destroy_copy(Schema *s) {
    if (!s || !s->cols) return;
    for (int i = 0; i < s->ncols; i++) {
        free(const_cast<char *>(s->cols[i].name));
    }
    free(s->cols);
    s->cols = nullptr;
    s->ncols = 0;
}

inline bool schema_clone(const Schema *src, Schema *dst) {
    memset(dst, 0, sizeof(*dst));
    dst->cols = static_cast<ColDef *>(calloc(src->ncols, sizeof(ColDef)));
    if (!dst->cols) return false;
    dst->ncols = src->ncols;

    for (int i = 0; i < src->ncols; i++) {
        dst->cols[i].type = src->cols[i].type;
        dst->cols[i].varchar_cap = src->cols[i].varchar_cap;
        dst->cols[i].name = dup_cstr(src->cols[i].name);
        if (!dst->cols[i].name) {
            schema_destroy_copy(dst);
            return false;
        }
    }
    return true;
}

inline void cpu_pause() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64)
    __asm__ __volatile__("yield");
#else
    std::this_thread::yield();
#endif
}

inline void wait_for_empty(Slot *slot) {
    while (slot->state.load(std::memory_order_acquire) != SLOT_EMPTY)
        cpu_pause();
}

inline bool wait_for_empty_or_error(const DoubleBuf *db, Slot *slot) {
    while (slot->state.load(std::memory_order_acquire) != SLOT_EMPTY) {
        if (db && db->consumer_error.load(std::memory_order_acquire) != 0)
            return false;
        cpu_pause();
    }
    return true;
}

inline void publish_batch(Slot *slot, int count, int is_final) {
    slot->batch.count = count;
    slot->batch.is_final = is_final;
    slot->state.store(SLOT_READY, std::memory_order_release);
}

inline DoubleBuf *doublebuf_create(const Schema *s) {
    const char *schema_err = nullptr;
    if (!schema_validate(s, &schema_err)) return nullptr;

    void *mem = nullptr;
    if (posix_memalign(&mem, CACHELINE, sizeof(DoubleBuf)) != 0 || !mem)
        return nullptr;
    auto *db = new (mem) DoubleBuf{};
    if (!schema_clone(s, &db->schema)) {
        db->~DoubleBuf();
        free(db);
        return nullptr;
    }
    for (int i = 0; i < 2; i++) {
        if (!batch_alloc(&db->slots[i].batch, &db->schema)) {
            for (int j = 0; j < i; j++) batch_free(&db->slots[j].batch);
            schema_destroy_copy(&db->schema);
            db->~DoubleBuf();
            free(db);
            return nullptr;
        }
    }
    return db;
}

inline void doublebuf_destroy(DoubleBuf *db) {
    if (!db) return;
    for (int i = 0; i < 2; i++) batch_free(&db->slots[i].batch);
    schema_destroy_copy(&db->schema);
    db->~DoubleBuf();
    free(db);
}

// ============================================================================
// Consumer: Generic Drain (schema-driven, calls DuckDB appender per cell)
// ============================================================================

static constexpr int FLUSH_EVERY = 500000;

inline duckdb_state append_batch_cell(const Batch *b, const Schema *s, int row,
                                      int c, duckdb_appender appender) {
    switch (s->cols[c].type) {
        case COL_INT32:
            return duckdb_append_int32(appender, batch_get_int32(b, c, row));
        case COL_INT64:
            return duckdb_append_int64(appender, batch_get_int64(b, c, row));
        case COL_DOUBLE:
            return duckdb_append_double(appender, batch_get_double(b, c, row));
        case COL_BOOL:
            return duckdb_append_bool(appender, batch_get_bool(b, c, row));
        case COL_VARCHAR:
            return duckdb_append_varchar(appender, batch_get_varchar(b, c, row));
    }
    return DuckDBError;
}

inline bool drain_batch_to_appender(const Batch *b, const Schema *s,
                                    duckdb_appender appender) {
    for (int row = 0; row < b->count; row++) {
        for (int c = 0; c < s->ncols; c++) {
            if (append_batch_cell(b, s, row, c, appender) == DuckDBError)
                return false;
        }
        if (duckdb_appender_end_row(appender) == DuckDBError) return false;
    }
    return true;
}

// ============================================================================
// Deterministic Data Generation
//
// Given (seq, col_index), every cell value is deterministic.
// Used by producers to fill, and by verifiers to check.
// ============================================================================

inline int32_t expected_int32(int64_t seq, int col) {
    return static_cast<int32_t>((seq * 31 + col * 7) ^ 0xDEAD);
}

inline int64_t expected_int64(int64_t seq, int col) {
    return (seq * 1000003LL + col * 999983LL) ^ 0xCAFEBABE;
}

inline double expected_double(int64_t seq, int col) {
    return static_cast<double>(seq % 100000) * 0.001 + col * 1.1;
}

inline int8_t expected_bool(int64_t seq, int col) {
    return static_cast<int8_t>(((seq + col) % 3) == 0 ? 1 : 0);
}

inline void expected_varchar(int64_t seq, int col, int cap, char *buf) {
    if (!buf || cap <= 0) return;
    snprintf(buf, static_cast<size_t>(cap), "R%lldC%d_%llx",
             static_cast<long long>(seq), col,
             static_cast<unsigned long long>(seq * 37 + col));
}

inline void fill_row_deterministic(Batch *b, int row, int64_t seq,
                                    const Schema *s) {
    for (int c = 0; c < s->ncols; c++) {
        switch (s->cols[c].type) {
            case COL_INT32:
                batch_set_int32(b, c, row, expected_int32(seq, c));
                break;
            case COL_INT64:
                batch_set_int64(b, c, row, expected_int64(seq, c));
                break;
            case COL_DOUBLE:
                batch_set_double(b, c, row, expected_double(seq, c));
                break;
            case COL_BOOL:
                batch_set_bool(b, c, row, expected_bool(seq, c));
                break;
            case COL_VARCHAR:
                expected_varchar(seq, c, s->cols[c].varchar_cap,
                                 batch_get_varchar_mut(b, c, row));
                break;
        }
    }
}
