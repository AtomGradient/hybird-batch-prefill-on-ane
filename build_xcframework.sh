#!/bin/bash
set -euo pipefail

# Build ANE-LM as xcframework for EdgeRuntime integration
# Currently: macOS arm64 only (iOS support TODO)
# Output: build/ANEInference.xcframework

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/xcframework-build"
OUTPUT_DIR="$SCRIPT_DIR/build"
FRAMEWORK_NAME="ANEInference"
NCPU=$(sysctl -n hw.ncpu)

echo "=== Building ANE-LM xcframework ==="

# Clean
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# ── Build macOS arm64 ──
MACOS_DIR="$BUILD_DIR/macos-arm64"
echo ">> Building for macOS arm64..."
cmake -S "$SCRIPT_DIR" -B "$MACOS_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DBUILD_SHARED_LIBS=OFF

cmake --build "$MACOS_DIR" -j"$NCPU"

# ── Merge into single static library ──
# cmake produces libane_inference.a (our code) + vendor .a files in _deps/
# We merge everything into one libane_inference_full.a
echo ">> Merging static libraries..."
FULL_LIB="$MACOS_DIR/libane_inference_full.a"
MERGE_INPUTS="$MACOS_DIR/libane_inference.a"

for vlib in $(find "$MACOS_DIR/_deps" -name "*.a" 2>/dev/null); do
    echo "   + $(basename "$vlib")"
    MERGE_INPUTS="$MERGE_INPUTS $vlib"
done

libtool -static -o "$FULL_LIB" $MERGE_INPUTS
echo ">> Merged library: $(du -h "$FULL_LIB" | cut -f1)"

# ── Verify symbols ──
echo ">> Verifying symbols..."
for sym in _ane_model_load _ane_generate _ane_prefill_only _ane_save_state _tokenizers_encode; do
    if nm "$FULL_LIB" 2>/dev/null | grep -q "T $sym"; then
        echo "   ✅ $sym"
    else
        echo "   ❌ $sym MISSING"
        exit 1
    fi
done

# ── Headers ──
echo ">> Creating headers..."
HEADER_DIR="$BUILD_DIR/headers"
mkdir -p "$HEADER_DIR"
cp "$SCRIPT_DIR/build-lib/headers/ane_inference.h" "$HEADER_DIR/" 2>/dev/null \
    || cp "$SCRIPT_DIR/ane_inference_bridge.h" "$HEADER_DIR/ane_inference.h" 2>/dev/null \
    || {
# Fallback: generate header
cat > "$HEADER_DIR/ane_inference.h" << 'HEADER'
#ifndef ANE_INFERENCE_H
#define ANE_INFERENCE_H
#ifdef __cplusplus
extern "C" {
#endif
typedef void* ane_model_t;
ane_model_t ane_model_load(const char* model_dir);
void ane_model_free(ane_model_t model);
void ane_model_reset(ane_model_t model);
int ane_model_vocab_size(ane_model_t model);
const char* ane_model_architecture(ane_model_t model);
int ane_prefill_only(ane_model_t model, const char* prompt, double* tps_out);
int ane_save_state(ane_model_t model, const char* path, int n_prompt_tokens);
char* ane_generate(ane_model_t model, const char* prompt, int max_tokens, float temperature, double* stats_out);
void ane_free_string(char* str);
#ifdef __cplusplus
}
#endif
#endif
HEADER
}

cat > "$HEADER_DIR/module.modulemap" << 'MODULEMAP'
module ANEInference {
    header "ane_inference.h"
    export *
}
MODULEMAP

# ── Create xcframework ──
echo ">> Creating xcframework..."
rm -rf "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"

xcodebuild -create-xcframework \
    -library "$FULL_LIB" \
    -headers "$HEADER_DIR" \
    -output "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"

echo ""
echo "=== Done ==="
echo "Output: $OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"
echo "Size: $(du -sh "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework" | cut -f1)"
