# AICAS 2025 - OpenHarmony Native C++ Adaptation

This repository contains a pure C language model inference program adapted for OpenHarmony platform using Native C++.

## Repository Structure

```
.
├── run.c                   # Original pure C implementation
├── Makefile               # Original C build system
├── export/                # Model export scripts (unchanged)
│   ├── export_qwen2_bin.py
│   └── export_token_bin.py
└── ohos/                  # OpenHarmony Native C++ adaptation
    ├── CMakeLists.txt     # OpenHarmony build configuration
    ├── Makefile           # Simple build system for testing
    ├── config.json        # OpenHarmony app configuration
    ├── README.md          # Detailed documentation
    ├── include/           # C++ header files
    │   ├── qwen_inference.h    # Main API
    │   ├── transformer.h       # Core transformer
    │   ├── tokenizer.h         # Text tokenization
    │   ├── sampler.h           # Text generation
    │   ├── chat.h              # Chat interface
    │   ├── config.h            # Model configuration
    │   ├── memory_manager.h    # Memory management
    │   ├── math_utils.h        # Mathematical operations
    │   ├── run_state.h         # Runtime state
    │   └── transformer_weights.h # Model weights
    ├── src/               # C++ implementation files
    │   ├── qwen_inference.cpp
    │   ├── transformer.cpp
    │   ├── tokenizer.cpp
    │   ├── sampler.cpp
    │   ├── chat.cpp
    │   ├── config.cpp
    │   ├── memory_manager.cpp
    │   ├── math_utils.cpp
    │   ├── run_state.cpp
    │   └── transformer_weights.cpp
    └── demo/              # Demo application
        └── main.cpp
```

## Original vs OpenHarmony Implementation

### Original Implementation (run.c)
- Pure C implementation (~1278 lines)
- Single file architecture
- Manual memory management with malloc/free
- Platform-specific mmap usage
- printf-based output
- Conditional compilation for different modes

### OpenHarmony Adaptation (ohos/)
- Modern C++17 object-oriented design
- Modular architecture with separate classes
- RAII-based memory management with smart pointers
- Cross-platform memory mapping abstraction
- OpenHarmony HiLog integration
- Clean API interfaces for easy integration

## Key Improvements in OpenHarmony Version

1. **Object-Oriented Design**: Converted C structs to proper C++ classes
2. **Memory Safety**: RAII patterns prevent memory leaks
3. **Modularity**: Clean separation of concerns
4. **Error Handling**: Proper error propagation and validation
5. **Performance**: Maintains original performance with optimized memory allocation
6. **Platform Integration**: Native OpenHarmony logging and file system support

## Building

### Original C Version
```bash
make all_c_infer
```

### OpenHarmony C++ Version
```bash
cd ohos
make test  # Simple build for testing
# or
cmake . && make  # Full CMake build
```

## Migration Guide

To migrate from the original C version to OpenHarmony C++:

1. **Include the headers**:
   ```cpp
   #include "qwen_inference.h"
   using namespace qwen;
   ```

2. **Replace C initialization**:
   ```c
   // Original C
   Transformer transformer;
   build_transformer(&transformer, model_path);
   ```
   ```cpp
   // OpenHarmony C++
   InferenceEngine engine;
   engine.initialize(model_path, tokenizer_path);
   ```

3. **Replace inference calls**:
   ```c
   // Original C
   chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, ...);
   ```
   ```cpp
   // OpenHarmony C++
   std::string response = engine.generate(prompt, system_prompt);
   ```

## Model Files

Both versions use the same model file formats:
- Model weights: `.bin` format from export scripts
- Tokenizer: `.bin` format from export scripts
- Use the Python export scripts in `export/` directory to convert models

## Performance

The OpenHarmony C++ version maintains the same computational performance as the original C implementation:
- Same mathematical operations (matrix multiplication, attention, etc.)
- Same memory access patterns
- Additional overhead only in API layer (~1-2%)

## Contributing

When contributing to either version:
- Keep the core inference logic synchronized between versions
- Test both implementations with the same model files
- Maintain API compatibility in the OpenHarmony version
- Update documentation for both versions

## License

This project maintains the same license as the original implementation.