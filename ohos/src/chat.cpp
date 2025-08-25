#include "chat.h"
#include <chrono>
#include <iostream>

#ifdef OHOS_PLATFORM
#include "hilog/log.h"
#define LOG_TAG "QwenChat"
#define QWEN_LOG_INFO(fmt, ...) OH_LOG_INFO(LOG_APP, fmt, ##__VA_ARGS__)
#define QWEN_LOG_ERROR(fmt, ...) OH_LOG_ERROR(LOG_APP, fmt, ##__VA_ARGS__)
#else
#define QWEN_LOG_INFO(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
#define QWEN_LOG_ERROR(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif

namespace qwen {

bool Chat::initialize(std::shared_ptr<Transformer> transformer,
                     std::shared_ptr<Tokenizer> tokenizer,
                     std::shared_ptr<Sampler> sampler) {
    transformer_ = transformer;
    tokenizer_ = tokenizer;
    sampler_ = sampler;
    
    return is_initialized();
}

std::string Chat::chat(const std::string& user_prompt,
                      const std::string& system_prompt,
                      int max_steps,
                      TokenCallback callback) {
    if (!is_initialized()) {
        QWEN_LOG_ERROR("Chat not properly initialized");
        return "";
    }
    
    QWEN_LOG_INFO("User prompt: %s", user_prompt.c_str());
    QWEN_LOG_INFO("Answer: ");
    
    // Prepare prompt tokens
    std::vector<int> prompt_tokens = prepare_prompt_tokens(user_prompt, system_prompt);
    
    // Start inference
    long long start_time = get_time_ms();
    long long prefill_time = 0;
    
    std::string response;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    int prompt_idx = 0;
    
    while (pos < max_steps) {
        // Forward the transformer to get logits for the next token
        transformer_->forward(token, pos);
        
        if (prompt_idx < prompt_tokens.size() - 1) {
            // If we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[++prompt_idx];
        } else {
            // Otherwise sample the next token from the logits
            next = sampler_->sample(transformer_->state.get_logits());
        }
        
        pos++;
        
        // Data-dependent terminating condition: we have hit the end token
        if (next == 1) break; // EOS token
        
        // Record prefill time after processing prompt
        if (prompt_idx == prompt_tokens.size() - 1 && prefill_time == 0) {
            prefill_time = get_time_ms();
        }
        
        // Print the token as string, decode it with the tokenizer object
        if (prompt_idx >= prompt_tokens.size() - 1) {
            std::string token_str = tokenizer_->decode(next);
            response += token_str;
            if (callback) {
                callback(token_str);
            }
        }
        
        token = next;
    }
    
    long long end_time = get_time_ms();
    
    // Update metrics
    metrics_.total_tokens = pos;
    metrics_.prefill_time_ms = prefill_time - start_time;
    metrics_.decode_time_ms = end_time - prefill_time;
    
    if (metrics_.prefill_time_ms > 0) {
        metrics_.prefill_throughput = static_cast<float>(prompt_tokens.size()) / (metrics_.prefill_time_ms / 1000.0f);
    }
    
    if (metrics_.decode_time_ms > 0) {
        int decode_tokens = pos - prompt_tokens.size();
        metrics_.decode_throughput = static_cast<float>(decode_tokens) / (metrics_.decode_time_ms / 1000.0f);
    }
    
    QWEN_LOG_INFO("Achieved prefill throughput: %f tokens/s", metrics_.prefill_throughput);
    QWEN_LOG_INFO("Achieved decode throughput: %f tokens/s", metrics_.decode_throughput);
    
    return response;
}

std::vector<int> Chat::chat_tokens(const std::vector<int>& prompt_tokens,
                                  int max_steps,
                                  bool show_output) {
    if (!is_initialized()) {
        return {};
    }
    
    std::vector<int> generated_tokens;
    long long start_time = get_time_ms();
    long long prefill_time = 0;
    
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    int prompt_idx = 0;
    
    while (pos < max_steps) {
        // Forward the transformer to get logits for the next token
        transformer_->forward(token, pos);
        
        if (prompt_idx < prompt_tokens.size() - 1) {
            // If we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[++prompt_idx];
        } else {
            // Otherwise sample the next token from the logits
            next = sampler_->sample(transformer_->state.get_logits());
            
            // Add to generated tokens only after prompt
            if (prompt_idx >= prompt_tokens.size() - 1) {
                generated_tokens.push_back(next);
            }
        }
        
        pos++;
        
        // Data-dependent terminating condition: we have hit the end token
        if (next == 1) break; // EOS token
        
        // Record prefill time
        if (prompt_idx == prompt_tokens.size() - 1 && prefill_time == 0) {
            prefill_time = get_time_ms();
        }
        
        // Show output if requested
        if (show_output && prompt_idx >= prompt_tokens.size() - 1) {
            std::string token_str = tokenizer_->decode(next);
            QWEN_LOG_INFO("%s", token_str.c_str());
        }
        
        token = next;
    }
    
    long long end_time = get_time_ms();
    
    // Update metrics
    metrics_.total_tokens = pos;
    metrics_.prefill_time_ms = prefill_time - start_time;
    metrics_.decode_time_ms = end_time - prefill_time;
    
    return generated_tokens;
}

bool Chat::is_initialized() const {
    return transformer_ && tokenizer_ && sampler_ &&
           transformer_->is_initialized() && tokenizer_->is_valid() && sampler_->is_valid();
}

std::vector<int> Chat::prepare_prompt_tokens(const std::string& user_prompt,
                                            const std::string& system_prompt) {
    std::vector<int> prompt_tokens;
    
    // Clean prompts
    std::string clean_system = clean_prompt(system_prompt);
    std::string clean_user = clean_prompt(user_prompt);
    
    // Encode system prompt
    std::vector<int> system_tokens = tokenizer_->encode(clean_system, false, false);
    std::vector<int> user_tokens = tokenizer_->encode(clean_user, false, false);
    
    // Build chat format tokens manually (simplified version)
    // Format: <|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n
    
    // <|im_start|>
    prompt_tokens.push_back(151644);
    // system
    prompt_tokens.push_back(9125);
    // \n
    prompt_tokens.push_back(198);
    
    // System prompt tokens
    prompt_tokens.insert(prompt_tokens.end(), system_tokens.begin(), system_tokens.end());
    
    // <|im_end|>
    prompt_tokens.push_back(151645);
    // \n
    prompt_tokens.push_back(198);
    
    // <|im_start|>
    prompt_tokens.push_back(151644);
    // user
    prompt_tokens.push_back(882);
    // \n
    prompt_tokens.push_back(198);
    
    // User prompt tokens
    prompt_tokens.insert(prompt_tokens.end(), user_tokens.begin(), user_tokens.end());
    
    // <|im_end|>
    prompt_tokens.push_back(151645);
    // \n
    prompt_tokens.push_back(198);
    
    // <|im_start|>
    prompt_tokens.push_back(151644);
    // assistant
    prompt_tokens.push_back(77091);
    // \n
    prompt_tokens.push_back(198);
    
    return prompt_tokens;
}

std::string Chat::clean_prompt(const std::string& prompt) {
    // Replace spaces with special UTF-8 character for tokenization
    std::string cleaned;
    cleaned.reserve(prompt.length() * 2); // Reserve extra space
    
    for (char c : prompt) {
        if (c == ' ') {
            // Replace space with Ġ (U+0120)
            cleaned += '\xC4';
            cleaned += '\xA0';
        } else {
            cleaned += c;
        }
    }
    
    return cleaned;
}

long long Chat::get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto time_since_epoch = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch).count();
}

} // namespace qwen