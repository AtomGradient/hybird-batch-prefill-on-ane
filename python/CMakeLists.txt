cmake_minimum_required(VERSION 3.20)
project(ane_lm_python LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "")

add_compile_definitions(ACCELERATE_NEW_LAPACK)

# pybind11
find_package(pybind11 REQUIRED)

# Fetch same dependencies as main project
include(FetchContent)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(json tokenizers_cpp)

# Source files (reuse from parent project)
set(PARENT ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(VENDOR_SOURCES
    ${PARENT}/vendor/jinja/value.cpp
    ${PARENT}/vendor/jinja/string.cpp
    ${PARENT}/vendor/jinja/lexer.cpp
    ${PARENT}/vendor/jinja/parser.cpp
    ${PARENT}/vendor/jinja/runtime.cpp
    ${PARENT}/vendor/jinja/caps.cpp
)

set(CORE_SOURCES
    ${PARENT}/common.cpp
    ${PARENT}/core/safetensors.cpp
    ${PARENT}/core/model_loader.cpp
    ${PARENT}/core/tokenizer.cpp
    ${PARENT}/core/cpu_ops.cpp
    ${PARENT}/core/sampling.cpp
    ${PARENT}/core/ane_runtime.cpp
    ${PARENT}/utils.cpp
    ${PARENT}/generate.cpp
)

set(MODEL_SOURCES
    ${PARENT}/models/llm/qwen3_5.cpp
    ${PARENT}/models/llm/qwen3.cpp
)

# Python module
pybind11_add_module(ane_lm
    ane_lm_binding.cpp
    ${VENDOR_SOURCES}
    ${CORE_SOURCES}
    ${MODEL_SOURCES}
)

target_include_directories(ane_lm PRIVATE
    ${PARENT}
    ${PARENT}/include
)

target_link_libraries(ane_lm PRIVATE
    nlohmann_json::nlohmann_json
    tokenizers_cpp
    "-framework Foundation"
    "-framework IOSurface"
    "-framework Accelerate"
    "-lobjc"
)

target_compile_options(ane_lm PRIVATE -O3)
