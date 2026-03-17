#!/bin/bash
set -euo pipefail

# Build ANE-LM as xcframework for EdgeRuntime integration
# Output: build/ANEInference.xcframework

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/xcframework-build"
OUTPUT_DIR="$SCRIPT_DIR/build"
FRAMEWORK_NAME="ANEInference"

echo "=== Building ANE-LM xcframework ==="

# Clean
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/macos-arm64"

# Build for macOS arm64
echo ">> Building for macOS arm64..."
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR/macos-arm64" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_XCFRAMEWORK=ON

cmake --build "$BUILD_DIR/macos-arm64" -j$(sysctl -n hw.ncpu)

# Collect objects into static library
echo ">> Creating static library..."
OBJECTS_DIR="$BUILD_DIR/macos-arm64/CMakeFiles/ane-lm.dir"
STATIC_LIB="$BUILD_DIR/macos-arm64/libane_inference.a"

# Find all .o files from the build
find "$OBJECTS_DIR" -name "*.o" | xargs ar rcs "$STATIC_LIB"

# Also need vendor objects
VENDOR_OBJECTS=$(find "$BUILD_DIR/macos-arm64" -path "*/jinja/*.o" -o -path "*/tokenizers_cpp*.o" 2>/dev/null || true)
if [ -n "$VENDOR_OBJECTS" ]; then
    echo "$VENDOR_OBJECTS" | xargs ar rcs "$STATIC_LIB"
fi

echo ">> Static library: $(du -h "$STATIC_LIB" | cut -f1)"

# Create module map and umbrella header
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

# Create xcframework
echo ">> Creating xcframework..."
rm -rf "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"

xcodebuild -create-xcframework \
    -library "$STATIC_LIB" \
    -headers "$HEADER_DIR" \
    -output "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"

echo ""
echo "=== Done ==="
echo "Output: $OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"
echo "Size: $(du -sh "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework" | cut -f1)"
