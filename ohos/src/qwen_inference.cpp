#include "qwen_inference.h"
#include <ctime>

namespace qwen {

bool InferenceEngine::initialize(const std::string& model_path,
                                const std::string& tokenizer_path,
                                float temperature,
                                float topp,
                                unsigned long long rng_seed) {
    // Create components
    transformer_ = std::make_shared<Transformer>();
    tokenizer_ = std::make_shared<Tokenizer>();
    sampler_ = std::make_shared<Sampler>();
    
    // Load transformer model
    if (!transformer_->load_from_checkpoint(model_path)) {
        return false;
    }
    
    // Build tokenizer
    if (!tokenizer_->build_from_file(tokenizer_path, transformer_->config.vocab_size)) {
        return false;
    }
    
    // Validate parameters
    if (temperature < 0.0f) temperature = 0.0f;
    if (topp < 0.0f || topp > 1.0f) topp = 0.9f;
    if (rng_seed == 0) rng_seed = static_cast<unsigned long long>(std::time(nullptr));
    
    // Initialize sampler
    if (!sampler_->initialize(transformer_->config.vocab_size, temperature, topp, rng_seed)) {
        return false;
    }
    
    // Initialize chat interface
    if (!chat_.initialize(transformer_, tokenizer_, sampler_)) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

std::string InferenceEngine::generate(const std::string& prompt,
                                     const std::string& system_prompt,
                                     int max_tokens) {
    if (!initialized_) {
        return "";
    }
    
    return chat_.chat(prompt, system_prompt, max_tokens);
}

bool InferenceEngine::is_initialized() const {
    return initialized_ && transformer_ && tokenizer_ && sampler_ &&
           transformer_->is_initialized() && tokenizer_->is_valid() && sampler_->is_valid();
}

const Config& InferenceEngine::get_config() const {
    return transformer_->config;
}

} // namespace qwen