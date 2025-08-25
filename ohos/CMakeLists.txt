# CMakeLists.txt for OpenHarmony Native C++ Language Model Inference

cmake_minimum_required(VERSION 3.16)
project(QwenInference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenHarmony specific configurations
set(CMAKE_TOOLCHAIN_FILE "$ENV{OHOS_NDK_HOME}/build/cmake/ohos.toolchain.cmake")

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h" "include/*.hpp")

# Create static library
add_library(qwen_inference STATIC ${SOURCES})

# Link libraries
target_link_libraries(qwen_inference
    hilog_ndk.z
    c++
    m
)

# Compiler-specific options
target_compile_options(qwen_inference PRIVATE
    -Wall
    -Wextra
    -ffast-math
    -fno-exceptions  # Disable exceptions for performance
    -fno-rtti        # Disable RTTI for performance
)

# Define preprocessor macros
target_compile_definitions(qwen_inference PRIVATE
    OHOS_PLATFORM=1
    DEFAULT_MODEL_PATH="/data/storage/el2/base/files/model.bin"
    DEFAULT_TOKENIZER_PATH="/data/storage/el2/base/files/tokenizer.bin"
)

# Export headers for external use
target_include_directories(qwen_inference PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Installation rules
install(TARGETS qwen_inference
    EXPORT QwenInferenceTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT QwenInferenceTargets
    FILE QwenInferenceTargets.cmake
    NAMESPACE QwenInference::
    DESTINATION lib/cmake/QwenInference
)

# Create demo executable (optional)
add_executable(qwen_demo demo/main.cpp)
target_link_libraries(qwen_demo qwen_inference)