#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/build"

echo "═══════════════════════════════════════════════════════════"
echo "  DuckDB Double-Buffer — Build & Test"
echo "═══════════════════════════════════════════════════════════"

if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu)"
else
    JOBS=4
fi

# ── Configure + Build ───────────────────────────────────────────
echo ""
echo "── Configuring (CMake) ──────────────────────────────────────"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release

echo ""
echo "── Building ─────────────────────────────────────────────────"
cmake --build "$BUILD" -j"$JOBS" 2>&1

# ── Run Demo ────────────────────────────────────────────────────
echo ""
echo "── Running Demo (main) ──────────────────────────────────────"
"$BUILD/main"
MAIN_RC=$?

# ── Run Fuzz Tests ──────────────────────────────────────────────
echo ""
echo "── Running Fuzz Tests (test_fuzz) ───────────────────────────"
"$BUILD/test_fuzz"
TEST_RC=$?

# ── Run UDF Benchmark ───────────────────────────────────────────
echo ""
echo "── Running UDF Benchmark (udf_bench) ───────────────────────"
"$BUILD/udf_bench"
UDF_RC=$?

# ── Summary ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
if [ $MAIN_RC -eq 0 ] && [ $TEST_RC -eq 0 ] && [ $UDF_RC -eq 0 ]; then
    echo "  ✅  All good — demo, fuzz tests, and UDF benchmark passed"
else
    echo "  ❌  Failures detected"
    [ $MAIN_RC -ne 0 ] && echo "       main exited $MAIN_RC"
    [ $TEST_RC -ne 0 ] && echo "       test_fuzz exited $TEST_RC"
    [ $UDF_RC  -ne 0 ] && echo "       udf_bench exited $UDF_RC"
fi
echo "═══════════════════════════════════════════════════════════"
exit $(( MAIN_RC | TEST_RC | UDF_RC ))
