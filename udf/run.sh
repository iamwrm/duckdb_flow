#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="build"

echo "── Configure ──"
cmake -S . -B "$BUILD_DIR"

echo ""
echo "── Build ──"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo ""
echo "── Run ──"
"$BUILD_DIR/main"
