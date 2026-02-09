# duckdb_flow

Schema-generic double-buffered appender for [DuckDB](https://duckdb.org/). A lock-free ping-pong buffer sits between a hot producer thread and a background consumer that drains batches through the DuckDB C appender API.

## Architecture

```
Producer (hot thread)         Consumer (background thread)
  fill rows into Batch  ──►  drain Batch to DuckDB appender
        slot[0] ◄──── ping-pong ────► slot[1]
```

- **Producer** fills column-oriented batches via `batch_set_*` and hands them off with `publish_batch()`.
- **Consumer** picks up ready slots and writes rows into DuckDB via `drain_batch_to_appender()`.
- Handoff is lock-free using cache-line-aligned atomic state flags (no mutex, no condition variable).
- The producer never touches DuckDB; the consumer never touches the hot batch.

## Supported column types

| Type | C type | DuckDB type |
|------|--------|-------------|
| `COL_INT32` | `int32_t` | `INTEGER` |
| `COL_INT64` | `int64_t` | `BIGINT` |
| `COL_UINT32` | `uint32_t` | `UINTEGER` |
| `COL_UINT64` | `uint64_t` | `UBIGINT` |
| `COL_FLOAT` | `float` | `FLOAT` |
| `COL_DOUBLE` | `double` | `DOUBLE` |
| `COL_DECIMAL` | `double` | `DECIMAL(w,s)` |
| `COL_BOOL` | `int8_t` | `BOOLEAN` |
| `COL_VARCHAR` | fixed-capacity `char[]` | `VARCHAR` |

For `COL_DECIMAL`, set `decimal_width` (1-18) and `decimal_scale` in `ColDef`.

## Building

Requires CMake 3.14+ and a C++17 compiler. DuckDB is fetched automatically at configure time.

```bash
bash run.sh
```

This configures, builds, runs the demo, and runs the fuzz test suite.

Or manually:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/main        # demo
./build/test_fuzz   # fuzz tests
```

## Usage

```c++
#include "duckdb_flow.h"

// 1. Define a schema
ColDef cols[] = {
    {"id",      COL_INT64,   0},
    {"payload", COL_VARCHAR, 64},
};
Schema schema = {cols, 2};

// 2. Create the double buffer
DoubleBuf *dbuf = doublebuf_create(&schema);

// 3. Producer: fill rows and publish batches
int slot = 0, idx = 0;
for (int64_t i = 0; i < num_rows; i++) {
    batch_set_int64(&dbuf->slots[slot].batch, 0, idx, i);
    batch_set_varchar(&dbuf->slots[slot].batch, 1, idx, "hello");
    if (++idx >= BATCH_CAPACITY) {
        publish_batch(&dbuf->slots[slot], idx, 0);
        idx = 0;
        slot ^= 1;
        wait_for_empty(&dbuf->slots[slot]);
    }
}
// Send final batch
publish_batch(&dbuf->slots[slot], idx, /*is_final=*/1);

// 4. Consumer (on another thread): drain into DuckDB appender
//    drain_batch_to_appender(&batch, &schema, appender);

// 5. Cleanup
doublebuf_destroy(dbuf);
```

## Tests

The fuzz test suite (`test.cpp`) runs 169 tests across 9 categories:

1. Boundary row counts
2. Random schemas x random row counts
3. Backpressure (slow consumer)
4. Extreme column counts (1 to 32)
5. VARCHAR capacity extremes (2-1024)
6. Homogeneous type tables
7. Rapid-fire init/shutdown cycling (50x)
8. Large-scale (up to 2M rows)
9. Adversarial patterns (primes, powers of 2)

## License

MIT
