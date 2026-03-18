#!/bin/bash
set -euo pipefail

# Build ANE-LM as xcframework for EdgeRuntime integration
# Builds for: macOS arm64 + iOS arm64
# Output: build/ANEInference.xcframework

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/xcframework-build"
OUTPUT_DIR="$SCRIPT_DIR/build"
FRAMEWORK_NAME="ANEInference"
NCPU=$(sysctl -n hw.ncpu)

echo "=== Building ANE-LM xcframework (macOS + iOS) ==="

# Clean previous build
rm -rf "$BUILD_DIR"

# ── Helper: build one platform, produce a merged static library ──
build_platform() {
    local PLATFORM=$1   # "macos" or "ios"
    local SYSROOT=$2    # SDK path
    local DEPLOY_FLAG=$3 # cmake deployment target flag
    local DEPLOY_VER=$4  # minimum version

    local PDIR="$BUILD_DIR/$PLATFORM-arm64"
    mkdir -p "$PDIR"

    echo ""
    echo ">> Building for $PLATFORM arm64..."
    cmake -S "$SCRIPT_DIR" -B "$PDIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_SYSROOT="$SYSROOT" \
        -D"$DEPLOY_FLAG"="$DEPLOY_VER" \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_XCFRAMEWORK=ON

    cmake --build "$PDIR" -j"$NCPU"

    # Merge all .o from our targets (ane-lm + ane_inference) into one .a
    local LIB="$PDIR/libane_inference_full.a"
    echo ">> Collecting object files..."
    find "$PDIR/CMakeFiles" -name "*.o" -not -path "*/_deps/*" \
        | xargs ar rcs "$LIB"

    # Merge ALL vendor static libraries (_deps/)
    local VENDOR_LIBS=$(find "$PDIR/_deps" -name "*.a" 2>/dev/null || true)
    if [ -n "$VENDOR_LIBS" ]; then
        local MERGE_ARGS="$LIB"
        while IFS= read -r vlib; do
            echo "   + $(basename "$vlib")"
            MERGE_ARGS="$MERGE_ARGS $vlib"
        done <<< "$VENDOR_LIBS"
        libtool -static -o "$LIB.merged" $MERGE_ARGS
        mv "$LIB.merged" "$LIB"
    fi

    echo ">> $PLATFORM library: $(du -h "$LIB" | cut -f1)"

    # Verify key symbols
    if nm "$LIB" 2>/dev/null | grep -q "T _ane_model_load"; then
        echo "   ✅ ane_model_load"
    else
        echo "   ❌ ane_model_load MISSING"; exit 1
    fi
    if nm "$LIB" 2>/dev/null | grep -q "T _tokenizers_encode"; then
        echo "   ✅ tokenizers_encode"
    else
        echo "   ⚠️  tokenizers_encode not found (may be in vendor lib)"
    fi
}

# ── Build macOS ──
MACOS_SDK=$(xcrun --sdk macosx --show-sdk-path)
build_platform "macos" "$MACOS_SDK" "CMAKE_OSX_DEPLOYMENT_TARGET" "14.0"

# ── Build iOS ──
IOS_SDK=$(xcrun --sdk iphoneos --show-sdk-path)
build_platform "ios" "$IOS_SDK" "CMAKE_OSX_DEPLOYMENT_TARGET" "17.0"

# ── Headers ──
echo ""
echo ">> Creating headers..."
HEADER_DIR="$BUILD_DIR/headers"
mkdir -p "$HEADER_DIR"

cat > "$HEADER_DIR/ane_inference.h" << 'HEADER'
#ifndef ANE_INFERENCE_H
#define ANE_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to ANE-LM model
typedef void* ane_model_t;

// Load model from directory. Returns NULL on failure.
ane_model_t ane_model_load(const char* model_dir);

// Free model.
void ane_model_free(ane_model_t model);

// Reset model state (clear caches).
void ane_model_reset(ane_model_t model);

// Get vocab size.
int ane_model_vocab_size(ane_model_t model);

// Get architecture string ("qwen3_5" or "qwen3").
const char* ane_model_architecture(ane_model_t model);

// Prefill only (ANE batch). Returns number of prompt tokens. tps_out receives tok/s.
int ane_prefill_only(ane_model_t model, const char* prompt, double* tps_out);

// Save prefill state to file. Returns 1 on success, 0 on failure.
int ane_save_state(ane_model_t model, const char* path, int n_prompt_tokens);

// Full generate. Returns generated text (caller must free with ane_free_string).
// stats_out: [prompt_tokens, prompt_tps, gen_tokens, gen_tps] (4 doubles)
char* ane_generate(ane_model_t model, const char* prompt,
                   int max_tokens, float temperature,
                   double* stats_out);

// Free string returned by ane_generate.
void ane_free_string(char* str);

#ifdef __cplusplus
}
#endif

#endif // ANE_INFERENCE_H
HEADER

cat > "$HEADER_DIR/module.modulemap" << 'MODULEMAP'
module ANEInference {
    header "ane_inference.h"
    export *
}
MODULEMAP

# ── Create xcframework (macOS + iOS) ──
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
ls -d "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"/*-arm64 2>/dev/null | while read d; do
    echo "  $(basename "$d"): $(du -h "$d"/*.a | cut -f1)"
done
