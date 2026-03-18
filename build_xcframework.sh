#!/bin/bash
set -euo pipefail

# Build ANE-LM as xcframework for EdgeRuntime integration
# Platforms: macOS arm64 + iOS arm64
# Output: build/ANEInference.xcframework

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/xcframework-build"
OUTPUT_DIR="$SCRIPT_DIR/build"
FRAMEWORK_NAME="ANEInference"
NCPU=$(sysctl -n hw.ncpu)

echo "=== Building ANE-LM xcframework (macOS + iOS) ==="

# Clean
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# ── Helper: build for a platform ──
build_platform() {
    local PLATFORM=$1    # macos-arm64 or ios-arm64
    local PLATFORM_DIR="$BUILD_DIR/$PLATFORM"
    local CMAKE_EXTRA_ARGS=("${@:2}")

    echo ""
    echo ">> Building for $PLATFORM..."
    cmake -S "$SCRIPT_DIR" -B "$PLATFORM_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DBUILD_SHARED_LIBS=OFF \
        "${CMAKE_EXTRA_ARGS[@]}"

    cmake --build "$PLATFORM_DIR" -j"$NCPU" --target ane_inference

    # Merge into single static library
    echo ">> Merging static libraries for $PLATFORM..."
    local FULL_LIB="$PLATFORM_DIR/libane_inference_full.a"
    local MERGE_INPUTS="$PLATFORM_DIR/libane_inference.a"

    for vlib in $(find "$PLATFORM_DIR/_deps" -name "*.a" 2>/dev/null); do
        echo "   + $(basename "$vlib")"
        MERGE_INPUTS="$MERGE_INPUTS $vlib"
    done

    libtool -static -o "$FULL_LIB" $MERGE_INPUTS
    echo ">> Merged: $(du -h "$FULL_LIB" | cut -f1)"

    # Verify our symbols exist
    for sym in _ane_model_load _ane_generate _ane_prefill_only _ane_save_state; do
        if nm "$PLATFORM_DIR/libane_inference.a" 2>/dev/null | grep -q "T $sym"; then
            echo "   ✅ $sym"
        else
            echo "   ❌ $sym MISSING"
            exit 1
        fi
    done
}

# ── Build macOS arm64 ──
build_platform "macos-arm64" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0

# ── Build iOS arm64 ──
IOS_SYSROOT=$(xcrun --sdk iphoneos --show-sdk-path)
build_platform "ios-arm64" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="$IOS_SYSROOT" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0

# ── Headers ──
echo ""
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

# ── Create xcframework with both platforms ──
echo ""
echo ">> Creating xcframework..."
rm -rf "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"

xcodebuild -create-xcframework \
    -library "$BUILD_DIR/macos-arm64/libane_inference_full.a" \
    -headers "$HEADER_DIR" \
    -library "$BUILD_DIR/ios-arm64/libane_inference_full.a" \
    -headers "$HEADER_DIR" \
    -output "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"

echo ""
echo "=== Done ==="
echo "Output: $OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"
echo "Size: $(du -sh "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework" | cut -f1)"
echo ""
echo "Platforms:"
xcodebuild -checkFirstLaunchStatus 2>/dev/null || true
plutil -p "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework/Info.plist" | grep -A2 'SupportedPlatform'
