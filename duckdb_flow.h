#pragma once
/* ============================================================================
   duckdb_flow.h — Schema-generic double-buffered appender for DuckDB

   Hot-thread-safe column-oriented batching with lock-free ping-pong handoff.
   The producer (hot thread) never touches DuckDB; the consumer (background
   thread) drains batches through the DuckDB C appender API.

   Compiles as both C11 and C++17.

   Usage:
     1. Define a Schema (array of ColDef)
     2. Create a DoubleBuf
     3. Producer fills rows via batch_set_* and publishes with publish_batch()
     4. Consumer drains via drain_batch_to_appender()
     5. Shutdown: producer sends a batch with is_final=1
   ========================================================================= */

#ifdef __cplusplus
  #include <cstdint>
  #include <cstdlib>
  #include <cstring>
  #include <cstdio>
  #include <cmath>
  #include <atomic>
  #include <thread>
  #define DFLOW_ATOMIC(T)        std::atomic<T>
  #define DFLOW_ALOAD(x, o)      (x).load(o)
  #define DFLOW_ASTORE(x, v, o)  (x).store((v), (o))
  #define DFLOW_ACQ              std::memory_order_acquire
  #define DFLOW_REL              std::memory_order_release
  #define DFLOW_NULL             nullptr
#else
  #include <stdint.h>
  #include <stdlib.h>
  #include <string.h>
  #include <stdio.h>
  #include <math.h>
  #include <stdatomic.h>
  #include <stdbool.h>
  #include <stdalign.h>
  #include <sched.h>
  #define DFLOW_ATOMIC(T)        _Atomic(T)
  #define DFLOW_ALOAD(x, o)      atomic_load_explicit(&(x), (o))
  #define DFLOW_ASTORE(x, v, o)  atomic_store_explicit(&(x), (v), (o))
  #define DFLOW_ACQ              memory_order_acquire
  #define DFLOW_REL              memory_order_release
  #define DFLOW_NULL             NULL
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "duckdb.h"

/* ============================================================================
   Schema Definition
   ========================================================================= */

typedef enum ColType {
    COL_INT32   = 0,
    COL_INT64   = 1,
    COL_DOUBLE  = 2,
    COL_BOOL    = 3,
    COL_VARCHAR = 4,
    COL_UINT32  = 5,
    COL_UINT64  = 6,
    COL_FLOAT   = 7,
    COL_DECIMAL = 8
} ColType;

typedef struct ColDef {
    const char *name;
    ColType     type;
    int         varchar_cap;    /* COL_VARCHAR: max string length */
    uint8_t     decimal_width;  /* COL_DECIMAL: precision (1-38) */
    uint8_t     decimal_scale;  /* COL_DECIMAL: digits after decimal point */
} ColDef;

/* Legacy/demo baseline used by tests. The implementation itself has no
   internal hard cap on schema columns. */
enum { MAX_COLS = 32 };

typedef struct Schema {
    ColDef *cols;
    int     ncols;
} Schema;

static inline const char *coltype_sql_name(ColType t) {
    switch (t) {
        case COL_INT32:   return "INTEGER";
        case COL_INT64:   return "BIGINT";
        case COL_DOUBLE:  return "DOUBLE";
        case COL_BOOL:    return "BOOLEAN";
        case COL_VARCHAR: return "VARCHAR";
        case COL_UINT32:  return "UINTEGER";
        case COL_UINT64:  return "UBIGINT";
        case COL_FLOAT:   return "FLOAT";
        case COL_DECIMAL: return "DECIMAL";
    }
    return "VARCHAR";
}

static inline const char *coltype_name(ColType t) {
    switch (t) {
        case COL_INT32:   return "INT32";
        case COL_INT64:   return "INT64";
        case COL_DOUBLE:  return "DOUBLE";
        case COL_BOOL:    return "BOOL";
        case COL_VARCHAR: return "VARCHAR";
        case COL_UINT32:  return "UINT32";
        case COL_UINT64:  return "UINT64";
        case COL_FLOAT:   return "FLOAT";
        case COL_DECIMAL: return "DECIMAL";
    }
    return "?";
}

static inline bool schema_validate(const Schema *s, const char **err_msg) {
    if (!s) { if (err_msg) *err_msg = "schema is null"; return false; }
    if (!s->cols) { if (err_msg) *err_msg = "schema columns are null"; return false; }
    if (s->ncols <= 0) { if (err_msg) *err_msg = "schema must have at least one column"; return false; }

    for (int i = 0; i < s->ncols; i++) {
        const ColDef *col = &s->cols[i];
        if (!col->name || !col->name[0]) {
            if (err_msg) *err_msg = "column name is null or empty";
            return false;
        }
        switch (col->type) {
            case COL_INT32:
            case COL_INT64:
            case COL_DOUBLE:
            case COL_BOOL:
            case COL_UINT32:
            case COL_UINT64:
            case COL_FLOAT:
                break;
            case COL_VARCHAR:
                if (col->varchar_cap < 2) {
                    if (err_msg) *err_msg = "varchar_cap must be >= 2";
                    return false;
                }
                break;
            case COL_DECIMAL:
                if (col->decimal_width < 1 || col->decimal_width > 18) {
                    if (err_msg) *err_msg = "decimal_width must be 1-18";
                    return false;
                }
                if (col->decimal_scale > col->decimal_width) {
                    if (err_msg) *err_msg = "decimal_scale must be <= decimal_width";
                    return false;
                }
                break;
            default:
                if (err_msg) *err_msg = "invalid column type";
                return false;
        }
    }
    return true;
}

static inline size_t schema_create_ddl_required(const Schema *s, const char *table) {
    size_t need = strlen("CREATE TABLE ") + strlen(table) + strlen(" ()") + 1;
    for (int i = 0; i < s->ncols; i++) {
        need += (i ? 2 : 0);  /* ", " */
        need += strlen(s->cols[i].name);
        need += 1;            /* space */
        need += strlen(coltype_sql_name(s->cols[i].type));
        if (s->cols[i].type == COL_DECIMAL)
            need += 10;       /* "(xx,yy)" with room to spare */
    }
    return need;
}

static inline void schema_create_ddl(const Schema *s, const char *table,
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
        int w;
        if (s->cols[i].type == COL_DECIMAL) {
            w = snprintf(buf + off, rem, "%s%s DECIMAL(%d,%d)",
                         i ? ", " : "", s->cols[i].name,
                         s->cols[i].decimal_width, s->cols[i].decimal_scale);
        } else {
            w = snprintf(buf + off, rem, "%s%s %s",
                         i ? ", " : "", s->cols[i].name,
                         coltype_sql_name(s->cols[i].type));
        }
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

/* ============================================================================
   Column-Oriented Batch
   ========================================================================= */

enum { BATCH_CAPACITY = 32768 };

typedef struct Batch {
    void **col_data;
    int   *col_stride;
    int    ncols;
    int    count;
    int    is_final;
} Batch;

static inline int col_storage_size(const ColDef *col) {
    switch (col->type) {
        case COL_INT32:   return sizeof(int32_t);
        case COL_INT64:   return sizeof(int64_t);
        case COL_DOUBLE:  return sizeof(double);
        case COL_BOOL:    return sizeof(int8_t);
        case COL_VARCHAR: return col->varchar_cap;
        case COL_UINT32:  return sizeof(uint32_t);
        case COL_UINT64:  return sizeof(uint64_t);
        case COL_FLOAT:   return sizeof(float);
        case COL_DECIMAL: return sizeof(double);
    }
    return 1;
}

static inline void batch_free(Batch *b) {
    if (!b) return;
    if (b->col_data) {
        for (int c = 0; c < b->ncols; c++) free(b->col_data[c]);
    }
    free(b->col_data);
    free(b->col_stride);
    memset(b, 0, sizeof(*b));
}

static inline bool batch_alloc(Batch *b, const Schema *s) {
    memset(b, 0, sizeof(*b));
    b->ncols = s->ncols;
    b->col_data = (void **)calloc(s->ncols, sizeof(void *));
    b->col_stride = (int *)calloc(s->ncols, sizeof(int));
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

/* ── Type-safe setters (hot path — compile to single indexed store) ──── */

static inline void batch_set_int32(Batch *b, int col, int row, int32_t v) {
    ((int32_t *)b->col_data[col])[row] = v;
}
static inline void batch_set_int64(Batch *b, int col, int row, int64_t v) {
    ((int64_t *)b->col_data[col])[row] = v;
}
static inline void batch_set_double(Batch *b, int col, int row, double v) {
    ((double *)b->col_data[col])[row] = v;
}
static inline void batch_set_bool(Batch *b, int col, int row, int8_t v) {
    ((int8_t *)b->col_data[col])[row] = v;
}
static inline void batch_set_varchar(Batch *b, int col, int row, const char *v) {
    char *dst = (char *)b->col_data[col] + row * b->col_stride[col];
    strncpy(dst, v, b->col_stride[col] - 1);
    dst[b->col_stride[col] - 1] = '\0';
}
static inline void batch_set_uint32(Batch *b, int col, int row, uint32_t v) {
    ((uint32_t *)b->col_data[col])[row] = v;
}
static inline void batch_set_uint64(Batch *b, int col, int row, uint64_t v) {
    ((uint64_t *)b->col_data[col])[row] = v;
}
static inline void batch_set_float(Batch *b, int col, int row, float v) {
    ((float *)b->col_data[col])[row] = v;
}
static inline void batch_set_decimal(Batch *b, int col, int row, double v) {
    ((double *)b->col_data[col])[row] = v;
}

/* ── Getters ─────────────────────────────────────────────────────────── */

static inline int32_t batch_get_int32(const Batch *b, int col, int row) {
    return ((int32_t *)b->col_data[col])[row];
}
static inline int64_t batch_get_int64(const Batch *b, int col, int row) {
    return ((int64_t *)b->col_data[col])[row];
}
static inline double batch_get_double(const Batch *b, int col, int row) {
    return ((double *)b->col_data[col])[row];
}
static inline int8_t batch_get_bool(const Batch *b, int col, int row) {
    return ((int8_t *)b->col_data[col])[row];
}
static inline const char *batch_get_varchar(const Batch *b, int col, int row) {
    return (const char *)b->col_data[col] + row * b->col_stride[col];
}
static inline char *batch_get_varchar_mut(Batch *b, int col, int row) {
    return (char *)b->col_data[col] + row * b->col_stride[col];
}
static inline uint32_t batch_get_uint32(const Batch *b, int col, int row) {
    return ((uint32_t *)b->col_data[col])[row];
}
static inline uint64_t batch_get_uint64(const Batch *b, int col, int row) {
    return ((uint64_t *)b->col_data[col])[row];
}
static inline float batch_get_float(const Batch *b, int col, int row) {
    return ((float *)b->col_data[col])[row];
}
static inline double batch_get_decimal(const Batch *b, int col, int row) {
    return ((double *)b->col_data[col])[row];
}

/* ============================================================================
   Double Buffer (Lock-Free Ping-Pong)
   ========================================================================= */

enum { CACHELINE = 64 };

typedef enum SlotState { SLOT_EMPTY = 0, SLOT_READY = 1 } SlotState;

typedef struct Slot {
    alignas(CACHELINE) DFLOW_ATOMIC(int) state;
    alignas(CACHELINE) Batch batch;
} Slot;

typedef struct DoubleBuf {
    Slot    slots[2];
    Schema  schema;
    alignas(CACHELINE) DFLOW_ATOMIC(long long) consumer_total_rows;
    alignas(CACHELINE) DFLOW_ATOMIC(int) consumer_error;
    char    consumer_error_msg[256];
} DoubleBuf;

static inline bool slot_is_ready(const Slot *slot) {
    return DFLOW_ALOAD(slot->state, DFLOW_ACQ) == SLOT_READY;
}

static inline bool any_slot_ready(const DoubleBuf *db) {
    for (int i = 0; i < 2; i++) {
        if (slot_is_ready(&db->slots[i])) return true;
    }
    return false;
}

static inline char *dup_cstr(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = (char *)malloc(n);
    if (!out) return DFLOW_NULL;
    memcpy(out, s, n);
    return out;
}

static inline void schema_destroy_copy(Schema *s) {
    if (!s || !s->cols) return;
    for (int i = 0; i < s->ncols; i++) {
        free((char *)s->cols[i].name);
    }
    free(s->cols);
    s->cols = DFLOW_NULL;
    s->ncols = 0;
}

static inline bool schema_clone(const Schema *src, Schema *dst) {
    memset(dst, 0, sizeof(*dst));
    dst->cols = (ColDef *)calloc(src->ncols, sizeof(ColDef));
    if (!dst->cols) return false;
    dst->ncols = src->ncols;

    for (int i = 0; i < src->ncols; i++) {
        dst->cols[i].type = src->cols[i].type;
        dst->cols[i].varchar_cap = src->cols[i].varchar_cap;
        dst->cols[i].decimal_width = src->cols[i].decimal_width;
        dst->cols[i].decimal_scale = src->cols[i].decimal_scale;
        dst->cols[i].name = dup_cstr(src->cols[i].name);
        if (!dst->cols[i].name) {
            schema_destroy_copy(dst);
            return false;
        }
    }
    return true;
}

static inline void cpu_pause(void) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64)
    __asm__ __volatile__("yield");
#else
  #ifdef __cplusplus
    std::this_thread::yield();
  #else
    sched_yield();
  #endif
#endif
}

static inline void wait_for_empty(Slot *slot) {
    while (DFLOW_ALOAD(slot->state, DFLOW_ACQ) != SLOT_EMPTY)
        cpu_pause();
}

static inline bool wait_for_empty_or_error(const DoubleBuf *db, Slot *slot) {
    while (DFLOW_ALOAD(slot->state, DFLOW_ACQ) != SLOT_EMPTY) {
        if (db && DFLOW_ALOAD(db->consumer_error, DFLOW_ACQ) != 0)
            return false;
        cpu_pause();
    }
    return true;
}

static inline void publish_batch(Slot *slot, int count, int is_final) {
    slot->batch.count = count;
    slot->batch.is_final = is_final;
    DFLOW_ASTORE(slot->state, SLOT_READY, DFLOW_REL);
}

static inline DoubleBuf *doublebuf_create(const Schema *s) {
    const char *schema_err = DFLOW_NULL;
    if (!schema_validate(s, &schema_err)) return DFLOW_NULL;

    void *mem = DFLOW_NULL;
    if (posix_memalign(&mem, CACHELINE, sizeof(DoubleBuf)) != 0 || !mem)
        return DFLOW_NULL;
    memset(mem, 0, sizeof(DoubleBuf));
    DoubleBuf *db = (DoubleBuf *)mem;

    if (!schema_clone(s, &db->schema)) {
        free(db);
        return DFLOW_NULL;
    }
    for (int i = 0; i < 2; i++) {
        if (!batch_alloc(&db->slots[i].batch, &db->schema)) {
            for (int j = 0; j < i; j++) batch_free(&db->slots[j].batch);
            schema_destroy_copy(&db->schema);
            free(db);
            return DFLOW_NULL;
        }
    }
    return db;
}

static inline void doublebuf_destroy(DoubleBuf *db) {
    if (!db) return;
    for (int i = 0; i < 2; i++) batch_free(&db->slots[i].batch);
    schema_destroy_copy(&db->schema);
    free(db);
}

/* Store an error message and set the error flag (first message wins).
   The message write happens-before the release store on consumer_error,
   so it is visible to any thread that observes the flag via acquire load. */
static inline void doublebuf_set_error(DoubleBuf *db, const char *msg) {
    if (db->consumer_error_msg[0] == '\0' && msg) {
        strncpy(db->consumer_error_msg, msg, sizeof(db->consumer_error_msg) - 1);
        db->consumer_error_msg[sizeof(db->consumer_error_msg) - 1] = '\0';
    }
    DFLOW_ASTORE(db->consumer_error, 1, DFLOW_REL);
}

/* ============================================================================
   Consumer: Generic Drain (schema-driven, calls DuckDB appender per cell)
   ========================================================================= */

enum { FLUSH_EVERY = 500000 };

static inline duckdb_state append_batch_cell(const Batch *b, const Schema *s,
                                             int row, int c,
                                             duckdb_appender appender) {
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
        case COL_UINT32:
            return duckdb_append_uint32(appender, batch_get_uint32(b, c, row));
        case COL_UINT64:
            return duckdb_append_uint64(appender, batch_get_uint64(b, c, row));
        case COL_FLOAT:
            return duckdb_append_float(appender, batch_get_float(b, c, row));
        case COL_DECIMAL:
            return duckdb_append_double(appender, batch_get_decimal(b, c, row));
    }
    return DuckDBError;
}

static inline bool drain_batch_to_appender(const Batch *b, const Schema *s,
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

/* ============================================================================
   Deterministic Data Generation

   Given (seq, col_index), every cell value is deterministic.
   Used by producers to fill, and by verifiers to check.
   ========================================================================= */

static inline int32_t expected_int32(int64_t seq, int col) {
    return (int32_t)((seq * 31 + col * 7) ^ 0xDEAD);
}

static inline int64_t expected_int64(int64_t seq, int col) {
    return (seq * 1000003LL + col * 999983LL) ^ 0xCAFEBABE;
}

static inline double expected_double(int64_t seq, int col) {
    return (double)(seq % 100000) * 0.001 + col * 1.1;
}

static inline int8_t expected_bool(int64_t seq, int col) {
    return (int8_t)(((seq + col) % 3) == 0 ? 1 : 0);
}

static inline void expected_varchar(int64_t seq, int col, int cap, char *buf) {
    if (!buf || cap <= 0) return;
    snprintf(buf, (size_t)cap, "R%lldC%d_%llx",
             (long long)seq, col,
             (unsigned long long)(seq * 37 + col));
}

static inline uint32_t expected_uint32(int64_t seq, int col) {
    return (uint32_t)((seq * 31 + col * 7) ^ 0xBEEF);
}

static inline uint64_t expected_uint64(int64_t seq, int col) {
    return (uint64_t)((seq * 1000003ULL + col * 999983ULL) ^ 0xFACEULL);
}

static inline float expected_float(int64_t seq, int col) {
    return (float)(seq % 10000) * 0.01f + col * 1.1f;
}

static inline double expected_decimal(int64_t seq, int col, uint8_t width,
                                      uint8_t scale) {
    /* Generate a deterministic value that fits in DECIMAL(width, scale).
       Integer part has (width - scale) digits, fractional part has scale digits. */
    int64_t int_digits = width - scale;
    int64_t int_max = 1;
    for (uint8_t i = 0; i < int_digits && i < 15; i++) int_max *= 10;
    int_max--;
    int64_t int_part = (int64_t)((seq * 31 + col * 7) % (int_max + 1));

    double frac = 0.0;
    if (scale > 0) {
        int64_t frac_max = 1;
        for (uint8_t i = 0; i < scale && i < 15; i++) frac_max *= 10;
        int64_t frac_part = (int64_t)((seq * 13 + col * 3) % frac_max);
        frac = (double)frac_part / (double)frac_max;
    }
    return (double)int_part + frac;
}

static inline void fill_row_deterministic(Batch *b, int row, int64_t seq,
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
            case COL_UINT32:
                batch_set_uint32(b, c, row, expected_uint32(seq, c));
                break;
            case COL_UINT64:
                batch_set_uint64(b, c, row, expected_uint64(seq, c));
                break;
            case COL_FLOAT:
                batch_set_float(b, c, row, expected_float(seq, c));
                break;
            case COL_DECIMAL:
                batch_set_decimal(b, c, row,
                                 expected_decimal(seq, c, s->cols[c].decimal_width,
                                                  s->cols[c].decimal_scale));
                break;
        }
    }
}
