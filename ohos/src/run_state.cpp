#include "run_state.h"

namespace qwen {

bool RunState::initialize(const Config& config) {
    int kv_dim = config.get_kv_dim();
    
    // Allocate all buffers
    x = MemoryManager::allocate_floats(config.dim);
    xb = MemoryManager::allocate_floats(config.dim);
    xb2 = MemoryManager::allocate_floats(config.dim);
    hb = MemoryManager::allocate_floats(config.hidden_dim);
    hb2 = MemoryManager::allocate_floats(config.hidden_dim);
    q = MemoryManager::allocate_floats(config.dim);
    k = MemoryManager::allocate_floats(config.dim);
    v = MemoryManager::allocate_floats(config.dim);
    att = MemoryManager::allocate_floats(config.n_heads * config.seq_len);
    logits = MemoryManager::allocate_floats(config.vocab_size);
    key_cache = MemoryManager::allocate_floats(config.n_layers * config.seq_len * kv_dim);
    value_cache = MemoryManager::allocate_floats(config.n_layers * config.seq_len * kv_dim);
    
    return is_valid();
}

bool RunState::is_valid() const {
    return x && xb && xb2 && hb && hb2 && q && k && v && att && logits && 
           key_cache && value_cache;
}

} // namespace qwen