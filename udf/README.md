# DuckDB SIMD-Vectorized UDF

A benchmark project that registers a compute-heavy scalar UDF in DuckDB via the C API
and compares a true scalar implementation against hand-written SIMD intrinsics (ARM NEON / x86 SSE2 / AVX2).

```
./run.sh        # configure, build, run
```

## Files

| File | Purpose |
|---|---|
| `crunch.h` | Shared interface: `crunch_scalar()` and `crunch_simd()` |
| `scalar.cpp` | Scalar kernel — compiled with `-fno-tree-vectorize -fno-tree-slp-vectorize` |
| `simd.cpp` | SIMD kernel — explicit intrinsics, 4-register unroll |
| `main.cpp` | Benchmark harness + DuckDB UDF registration |
| `CMakeLists.txt` | C++23, fetches prebuilt libduckdb, exports compile_commands.json |
| `run.sh` | One-shot configure/build/run |

## Results (ARM NEON, Ampere Altra)

```
── Raw kernel ──────────────────────────────────
  N      │ scalar ns/el │  SIMD ns/el │ speedup
  2K     │       29.97  │       14.56 │ 2.06x
  64K    │       29.72  │       14.86 │ 2.00x
  1M     │       29.31  │       14.72 │ 1.99x
  10M    │       29.53  │       14.75 │ 2.00x

── DuckDB end-to-end query ─────────────────────
  N      │   scalar ms │     SIMD ms │ speedup
  10K    │        0.38 │        0.23 │ 1.65x
  100K   │        3.03 │        1.59 │ 1.90x
  1M     │       30.07 │       15.29 │ 1.97x
  10M    │      303.41 │      157.12 │ 1.93x
```

Raw kernel: exact 2× (2 int64 lanes per NEON register).
DuckDB end-to-end converges to ~1.95× as N grows and UDF compute dominates query overhead.

## DuckDB C API for Scalar UDFs

Registration:

```c
duckdb_scalar_function func = duckdb_create_scalar_function();
duckdb_scalar_function_set_name(func, "my_func");

duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
duckdb_scalar_function_add_parameter(func, bigint);
duckdb_scalar_function_set_return_type(func, bigint);
duckdb_scalar_function_set_function(func, my_callback);

duckdb_register_scalar_function(con, func);

// cleanup
duckdb_destroy_logical_type(&bigint);
duckdb_destroy_scalar_function(&func);
```

The callback signature — DuckDB calls this once per chunk (~2048 rows):

```c
void my_callback(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    idx_t n   = duckdb_data_chunk_get_size(input);
    auto *in  = (int64_t *)duckdb_vector_get_data(duckdb_data_chunk_get_vector(input, 0));
    auto *out = (int64_t *)duckdb_vector_get_data(output);
    // process n elements...
}
```

Key types: `duckdb_data_chunk` (columnar batch), `duckdb_vector` (one column within a chunk).
Use `duckdb_vector_get_data` to get the raw pointer, `duckdb_data_chunk_get_size` for row count.

DuckDB calls this "vectorized" — it means **batched** (one call per chunk), not SIMD.
The actual SIMD is up to you.

## Lessons Learned

### 1. GCC auto-vectorizes aggressively — verify your "scalar" baseline

GCC 14 with `-O2 -march=native` auto-vectorized our scalar loop into NEON instructions
(SLP vectorization), making "scalar" and "SIMD" identical. The benchmark showed 0.96× — 
a false result.

**Fix:** compile the scalar baseline in a **separate translation unit** with:
```
-fno-tree-vectorize -fno-tree-slp-vectorize
```

The `__attribute__((optimize("no-tree-vectorize")))` function attribute is not enough —
it only disables loop vectorization, not SLP (Superword-Level Parallelism) vectorization.
Separate TU with explicit compile flags is the reliable approach.

**Always check disassembly** (`objdump -d`) to confirm your scalar code is actually scalar.

### 2. SIMD speedup requires throughput-bound workloads, not latency-bound

A simple `x * 3` (one ALU op per element) shows no SIMD benefit because:
- The operation is so cheap it's **memory-bandwidth-bound**, not compute-bound
- CPU out-of-order execution overlaps independent iterations, achieving the same throughput as SIMD

To see real SIMD gains, use a **compute-heavy** kernel (~256 ALU ops/element in our case: 
32 rounds × 8 ops/round of shift-xor-add mixing).

### 3. Dependency chains kill SIMD parallelism — unroll with independent registers

A single SIMD register chain like:
```c
x = veorq_u64(x, vshrq_n_u64(x, 13));  // depends on x
x = vaddq_u64(x, vshlq_n_u64(x, 7));   // depends on new x
```
has the **same latency** as scalar — each instruction waits for the previous one.
The CPU's OoO engine can overlap scalar iterations just as effectively.

**Fix:** unroll with **4 independent registers** (`a`, `b`, `c`, `d`) processing
different elements. This gives the pipeline 4 independent dependency chains to
interleave, which scalar OoO can't match.

```c
// 4 independent chains — CPU can pipeline all 4
a = veorq_u64(a, vshrq_n_u64(a, 13));
b = veorq_u64(b, vshrq_n_u64(b, 13));
c = veorq_u64(c, vshrq_n_u64(c, 13));
d = veorq_u64(d, vshrq_n_u64(d, 13));
```

### 4. NEON/SSE2 lack 64-bit integer multiply

There is no `vmulq_s64` (NEON) or `_mm_mullo_epi64` (SSE2/AVX2).
For `x * 3`, use `(x << 1) + x`. For general compute, stick to
shift / xor / add / subtract — all have 1:1 SIMD counterparts on every platform.

AVX-512 has `_mm512_mullo_epi64` but requiring AVX-512 limits portability.

### 5. DuckDB UDF overhead vanishes at scale

At 10K rows, DuckDB overhead (query planning, `generate_series`, result materialization)
accounts for ~35% of total time, diluting the SIMD advantage to 1.65×.
At 10M rows, UDF compute dominates and the end-to-end speedup (1.93×) nearly matches
the raw kernel (2.0×).

**If your UDF is trivially cheap**, SIMD won't show up in end-to-end benchmarks
because DuckDB's own machinery is the bottleneck.

### 6. Expected speedups by platform

| Platform | Register width | int64 lanes | Unrolled (×4 regs) | Expected speedup |
|---|---|---|---|---|
| ARM NEON | 128-bit | 2 | 8 elements | ~2× |
| x86 SSE2 | 128-bit | 2 | 8 elements | ~2× |
| x86 AVX2 | 256-bit | 4 | 16 elements | ~4× |

Actual speedup = min(SIMD lanes, compute/memory ratio). For compute-bound
workloads the lane count is the ceiling.

### 7. CMake: per-file compile flags

Use `set_source_files_properties` to apply flags to individual files:
```cmake
set_source_files_properties(scalar.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-tree-vectorize;-fno-tree-slp-vectorize"
)
```
Note the **semicolons** — CMake list syntax, not spaces.

### 8. CMake: importing a prebuilt shared library

```cmake
add_library(duckdb SHARED IMPORTED GLOBAL)
set_target_properties(duckdb PROPERTIES
    IMPORTED_LOCATION "${DUCKDB_LIB_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${DUCKDB_INCLUDE_DIR}"
)
# Copy .so next to binaries so RPATH resolves
file(COPY "${DUCKDB_LIB_PATH}" DESTINATION "${CMAKE_BINARY_DIR}")
```

Set `CMAKE_BUILD_RPATH` to `${CMAKE_BINARY_DIR}` so the executable finds the `.so`
at runtime without `LD_LIBRARY_PATH`.
