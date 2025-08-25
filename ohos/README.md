# OpenHarmony Native C++ Qwen Inference

This directory contains the OpenHarmony Native C++ adaptation of the Qwen language model inference engine.

## Project Structure

```
ohos/
├── CMakeLists.txt          # Build configuration
├── config.json             # OpenHarmony app configuration
├── include/                # Header files
│   ├── config.h            # Model configuration
│   ├── memory_manager.h    # Memory management utilities
│   ├── transformer_weights.h # Model weights container
│   ├── run_state.h         # Runtime state management
│   ├── transformer.h       # Main transformer class
│   ├── tokenizer.h         # Text tokenization
│   ├── sampler.h           # Text generation sampling
│   ├── math_utils.h        # Mathematical operations
│   ├── chat.h              # Chat interface
│   └── qwen_inference.h    # Main API header
├── src/                    # Implementation files
│   ├── config.cpp
│   ├── memory_manager.cpp
│   ├── transformer_weights.cpp
│   ├── run_state.cpp
│   ├── transformer.cpp
│   ├── tokenizer.cpp
│   ├── sampler.cpp
│   ├── math_utils.cpp
│   ├── chat.cpp
│   └── qwen_inference.cpp
├── demo/                   # Demo application
│   └── main.cpp
└── README.md
```

## Key Features

- **C++ Object-Oriented Design**: Converted from original C structs to proper C++ classes
- **RAII Memory Management**: Automatic memory management using smart pointers
- **OpenHarmony Integration**: Native integration with OpenHarmony logging and file system
- **Performance Optimized**: Maintains the performance characteristics of the original C implementation
- **Modular Architecture**: Clean separation of concerns with well-defined interfaces

## Building

### Prerequisites

- OpenHarmony NDK installed
- CMake 3.16 or later
- C++17 compatible compiler

### Build Steps

```bash
# Set OpenHarmony NDK environment
export OHOS_NDK_HOME=/path/to/ohos-sdk/native

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=$OHOS_NDK_HOME/build/cmake/ohos.toolchain.cmake

# Build
make -j$(nproc)
```

## Usage

### Basic Usage

```cpp
#include "qwen_inference.h"

using namespace qwen;

int main() {
    // Initialize inference engine
    InferenceEngine engine;
    bool success = engine.initialize(
        "/data/storage/el2/base/files/model.bin",
        "/data/storage/el2/base/files/tokenizer.bin",
        0.0f,  // temperature
        0.9f   // top-p
    );
    
    if (!success) {
        return 1;
    }
    
    // Generate response
    std::string response = engine.generate(
        "What is artificial intelligence?",
        "You are a helpful assistant."
    );
    
    return 0;
}
```

### Advanced Usage with Chat Interface

```cpp
#include "qwen_inference.h"

using namespace qwen;

int main() {
    InferenceEngine engine;
    engine.initialize(model_path, tokenizer_path);
    
    Chat& chat = engine.get_chat();
    
    // Generate with token callback
    std::string response = chat.chat(
        "Explain quantum computing",
        "You are a physics expert.",
        600, // max tokens
        [](const std::string& token) {
            std::cout << token << std::flush;
        }
    );
    
    // Get performance metrics
    const auto& metrics = chat.get_metrics();
    std::cout << "Throughput: " << metrics.decode_throughput << " tokens/s\n";
    
    return 0;
}
```

## API Reference

### InferenceEngine

Main entry point for the inference system.

- `bool initialize(model_path, tokenizer_path, temperature, topp, rng_seed)`
- `std::string generate(prompt, system_prompt, max_tokens)`
- `Chat& get_chat()`
- `const Config& get_config()`

### Chat

Conversational interface with streaming support.

- `std::string chat(user_prompt, system_prompt, max_steps, callback)`
- `std::vector<int> chat_tokens(prompt_tokens, max_steps, show_output)`
- `const InferenceMetrics& get_metrics()`

### Transformer

Core model implementation.

- `bool load_from_checkpoint(checkpoint_path)`
- `void forward(token, pos)`
- `bool is_initialized()`

## Model File Format

The engine expects model files in the same binary format as the original C implementation:

1. **Model file (.bin)**: Contains model configuration and weights
2. **Tokenizer file (.bin)**: Contains vocabulary and tokenization data

Use the provided Python export scripts from the original project to convert models.

## Performance Considerations

- Uses SIMD-optimized memory allocation (32-byte aligned)
- Efficient memory mapping for model weights
- Zero-copy operations where possible
- Optimized matrix multiplication routines
- Minimal memory allocations during inference

## OpenHarmony Integration

- Uses HiLog for logging on OpenHarmony platform
- Adapts file paths for OpenHarmony file system structure
- Compatible with OpenHarmony Native development workflow
- Supports both phone and tablet form factors

## Thread Safety

The current implementation is not thread-safe. Each inference session should use its own `InferenceEngine` instance or external synchronization should be provided.

## License

This project maintains the same license as the original implementation.