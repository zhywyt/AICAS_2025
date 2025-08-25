#include "memory_manager.h"
#include <cstring>
#include <cstdlib>

#ifdef OHOS_PLATFORM
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace qwen {

std::unique_ptr<float[]> MemoryManager::allocate_floats(size_t count) {
    // Use aligned allocation for better SIMD performance
    size_t size = count * sizeof(float);
    size_t aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
    void* ptr = nullptr;
    if (posix_memalign(&ptr, ALIGNMENT, aligned_size) != 0) {
        return nullptr;
    }
    
    // Zero initialize
    std::memset(ptr, 0, aligned_size);
    
    return std::unique_ptr<float[]>(static_cast<float*>(ptr));
}

std::unique_ptr<int[]> MemoryManager::allocate_ints(size_t count) {
    size_t size = count * sizeof(int);
    size_t aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
    void* ptr = nullptr;
    if (posix_memalign(&ptr, ALIGNMENT, aligned_size) != 0) {
        return nullptr;
    }
    
    // Zero initialize
    std::memset(ptr, 0, aligned_size);
    
    return std::unique_ptr<int[]>(static_cast<int*>(ptr));
}

MemoryManager::MemoryMap::MemoryMap(const char* filepath) {
    // Open file
    fd_ = open(filepath, O_RDONLY);
    if (fd_ == -1) {
        return;
    }
    
    // Get file size
    off_t file_size = lseek(fd_, 0, SEEK_END);
    if (file_size == -1) {
        cleanup();
        return;
    }
    
    size_ = static_cast<size_t>(file_size);
    
    // Memory map the file
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        cleanup();
        return;
    }
}

MemoryManager::MemoryMap::~MemoryMap() {
    cleanup();
}

MemoryManager::MemoryMap::MemoryMap(MemoryMap&& other) noexcept
    : data_(other.data_), size_(other.size_), fd_(other.fd_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
}

MemoryManager::MemoryMap& MemoryManager::MemoryMap::operator=(MemoryMap&& other) noexcept {
    if (this != &other) {
        cleanup();
        data_ = other.data_;
        size_ = other.size_;
        fd_ = other.fd_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.fd_ = -1;
    }
    return *this;
}

void MemoryManager::MemoryMap::cleanup() {
    if (data_ != nullptr && data_ != MAP_FAILED) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
    size_ = 0;
}

} // namespace qwen