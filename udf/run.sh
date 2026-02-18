#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"

echo "── Configure ──"
cmake -S "$ROOT" -B "$BUILD"

echo ""
echo "── Build ──"
cmake --build "$BUILD" --target udf_bench -j"$(nproc)"

echo ""
echo "── Run ──"
"$BUILD/udf_bench"
