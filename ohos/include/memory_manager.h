#ifndef QWEN_INFERENCE_MEMORY_MANAGER_H
#define QWEN_INFERENCE_MEMORY_MANAGER_H

#include <memory>
#include <cstddef>

#ifdef OHOS_PLATFORM
#include "hilog/log.h"
#else
#include <iostream>
#endif

namespace qwen {

/**
 * @brief RAII-based memory manager for aligned memory allocation
 */
class MemoryManager {
public:
    static std::unique_ptr<float[]> allocate_floats(size_t count);
    static std::unique_ptr<int[]> allocate_ints(size_t count);
    
    // Memory mapping utilities for model files
    class MemoryMap {
    public:
        MemoryMap() = default;
        MemoryMap(const char* filepath);
        ~MemoryMap();
        
        // Non-copyable
        MemoryMap(const MemoryMap&) = delete;
        MemoryMap& operator=(const MemoryMap&) = delete;
        
        // Movable
        MemoryMap(MemoryMap&& other) noexcept;
        MemoryMap& operator=(MemoryMap&& other) noexcept;
        
        void* data() const { return data_; }
        size_t size() const { return size_; }
        bool is_valid() const { return data_ != nullptr; }
        
    private:
        void* data_ = nullptr;
        size_t size_ = 0;
        int fd_ = -1;
        
        void cleanup();
    };

private:
    static constexpr size_t ALIGNMENT = 32; // 32-byte alignment for SIMD
};

} // namespace qwen

#endif // QWEN_INFERENCE_MEMORY_MANAGER_H