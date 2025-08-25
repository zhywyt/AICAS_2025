#include "qwen_inference.h"
#include <iostream>
#include <string>

#ifdef OHOS_PLATFORM
#include "hilog/log.h"
#define LOG_TAG "QwenDemo"
#define QWEN_LOG_INFO(fmt, ...) OH_LOG_INFO(LOG_APP, fmt, ##__VA_ARGS__)
#define QWEN_LOG_ERROR(fmt, ...) OH_LOG_ERROR(LOG_APP, fmt, ##__VA_ARGS__)
#else
#define QWEN_LOG_INFO(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
#define QWEN_LOG_ERROR(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif

using namespace qwen;

int main(int argc, char* argv[]) {
    // Default parameters
#ifdef DEFAULT_MODEL_PATH
    std::string model_path = DEFAULT_MODEL_PATH;
#else
    std::string model_path = "/data/storage/el2/base/files/model.bin";
#endif

#ifdef DEFAULT_TOKENIZER_PATH
    std::string tokenizer_path = DEFAULT_TOKENIZER_PATH;
#else
    std::string tokenizer_path = "/data/storage/el2/base/files/tokenizer.bin";
#endif
    float temperature = 0.0f;
    float topp = 0.9f;
    int max_tokens = 600;
    std::string prompt = "Introduce the car including its history.";
    std::string system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    
    // Parse command line arguments
    if (argc > 1) {
        prompt = argv[1];
    }
    if (argc > 2) {
        model_path = argv[2];
    }
    if (argc > 3) {
        tokenizer_path = argv[3];
    }
    
    QWEN_LOG_INFO("Initializing Qwen Inference Engine...");
    QWEN_LOG_INFO("Model path: %s", model_path.c_str());
    QWEN_LOG_INFO("Tokenizer path: %s", tokenizer_path.c_str());
    
    // Initialize inference engine
    InferenceEngine engine;
    if (!engine.initialize(model_path, tokenizer_path, temperature, topp)) {
        QWEN_LOG_ERROR("Failed to initialize inference engine");
        return 1;
    }
    
    QWEN_LOG_INFO("Engine initialized successfully");
    
    // Print model configuration
    const Config& config = engine.get_config();
    QWEN_LOG_INFO("Model configuration:");
    QWEN_LOG_INFO("  Dimension: %d", config.dim);
    QWEN_LOG_INFO("  Hidden dimension: %d", config.hidden_dim);
    QWEN_LOG_INFO("  Layers: %d", config.n_layers);
    QWEN_LOG_INFO("  Heads: %d", config.n_heads);
    QWEN_LOG_INFO("  KV heads: %d", config.n_kv_heads);
    QWEN_LOG_INFO("  Vocabulary size: %d", config.vocab_size);
    QWEN_LOG_INFO("  Sequence length: %d", config.seq_len);
    
    // Generate response
    QWEN_LOG_INFO("Generating response...");
    std::string response = engine.generate(prompt, system_prompt, max_tokens);
    
    QWEN_LOG_INFO("Generated response: %s", response.c_str());
    
    // Show performance metrics
    const auto& metrics = engine.get_chat().get_metrics();
    QWEN_LOG_INFO("Performance metrics:");
    QWEN_LOG_INFO("  Total tokens: %d", metrics.total_tokens);
    QWEN_LOG_INFO("  Prefill throughput: %.2f tokens/s", metrics.prefill_throughput);
    QWEN_LOG_INFO("  Decode throughput: %.2f tokens/s", metrics.decode_throughput);
    QWEN_LOG_INFO("  Prefill time: %lld ms", metrics.prefill_time_ms);
    QWEN_LOG_INFO("  Decode time: %lld ms", metrics.decode_time_ms);
    
    return 0;
}