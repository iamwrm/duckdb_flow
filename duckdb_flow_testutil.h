#pragma once
/* ============================================================================
   duckdb_flow_testutil.h — Deterministic data generation & test helpers

   Given (seq, col_index), every cell value is deterministic and reproducible.
   Used by producers to fill batches and by verifiers to check correctness.
   ========================================================================= */

#include "duckdb_flow.h"

enum { MAX_COLS = 32 };

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

/* ── Expected value generators ─────────────────────────────────────────── */

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
             (long long)seq, col, (unsigned long long)(seq * 37 + col));
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

/* ── Fill a batch row deterministically ────────────────────────────────── */

static inline void fill_row_deterministic(Batch *b, int row, int64_t seq,
                                          const Schema *s) {
    for (int c = 0; c < s->ncols; c++) {
        switch (s->cols[c].type) {
            case COL_INT32:   batch_set_int32(b, c, row, expected_int32(seq, c));   break;
            case COL_INT64:   batch_set_int64(b, c, row, expected_int64(seq, c));   break;
            case COL_DOUBLE:  batch_set_double(b, c, row, expected_double(seq, c)); break;
            case COL_BOOL:    batch_set_bool(b, c, row, expected_bool(seq, c));     break;
            case COL_UINT32:  batch_set_uint32(b, c, row, expected_uint32(seq, c)); break;
            case COL_UINT64:  batch_set_uint64(b, c, row, expected_uint64(seq, c)); break;
            case COL_FLOAT:   batch_set_float(b, c, row, expected_float(seq, c));   break;
            case COL_DECIMAL:
                batch_set_decimal(b, c, row,
                    expected_decimal(seq, c, s->cols[c].decimal_width,
                                     s->cols[c].decimal_scale));
                break;
            case COL_VARCHAR:
                expected_varchar(seq, c, s->cols[c].varchar_cap,
                                 batch_get_varchar_mut(b, c, row));
                break;
        }
    }
}
