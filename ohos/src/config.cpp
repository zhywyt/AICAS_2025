#include "config.h"

namespace qwen {

Config::Config(int32_t dim, int32_t hidden_dim, int32_t n_layers, int32_t n_heads,
               int32_t n_kv_heads, int32_t vocab_size, int32_t seq_len)
    : dim(dim), hidden_dim(hidden_dim), n_layers(n_layers), n_heads(n_heads),
      n_kv_heads(n_kv_heads), vocab_size(vocab_size), seq_len(seq_len) {
}

bool Config::is_valid() const {
    return dim > 0 && hidden_dim > 0 && n_layers > 0 && n_heads > 0 &&
           n_kv_heads > 0 && vocab_size > 0 && seq_len > 0 &&
           dim % n_heads == 0 && n_heads % n_kv_heads == 0;
}

} // namespace qwen