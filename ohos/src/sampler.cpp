#include "sampler.h"
#include "math_utils.h"
#include <algorithm>
#include <ctime>

namespace qwen {

bool Sampler::initialize(int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    this->vocab_size = vocab_size;
    this->temperature = temperature;
    this->topp = topp;
    
    // Initialize probindex buffer for top-p sampling
    probindex.resize(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        probindex[i] = ProbIndex(0.0f, i);
    }
    
    // Initialize random number generator
    if (rng_seed == 0) {
        rng_seed = static_cast<unsigned long long>(std::time(nullptr));
    }
    rng.seed(rng_seed);
    
    return true;
}

int Sampler::sample(float* logits) {
    // If temperature is 0.0, just return the argmax
    if (temperature == 0.0f) {
        int next = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[next]) {
                next = i;
            }
        }
        return next;
    }
    
    // Apply temperature to the logits
    for (int q = 0; q < vocab_size; q++) {
        logits[q] /= temperature;
    }
    
    // Apply softmax to the logits to get the probabilities
    softmax(logits, vocab_size);
    
    // Coin flip decision based on topp
    float coin = static_cast<float>(rng()) / static_cast<float>(rng.max());
    
    if (topp <= 0 || topp >= 1) {
        // Simply sample from the predicted probability distribution
        return sample_mult(logits, vocab_size);
    } else {
        // Top-p (nucleus) sampling, truncating the least likely tokens first
        return sample_topp(logits, vocab_size, topp);
    }
}

bool Sampler::is_valid() const {
    return vocab_size > 0 && !probindex.empty();
}

void Sampler::softmax(float* x, int size) {
    MathUtils::softmax(x, size);
}

int Sampler::sample_mult(float* probabilities, int n) {
    // Sample from multinomial probability distribution
    float coin = static_cast<float>(rng()) / static_cast<float>(rng.max());
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of floating point errors
}

int Sampler::sample_topp(float* probabilities, int n, float topp) {
    // Top-p sampling (nucleus sampling) truncates the less likely tokens
    
    // Quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so we only sort the top (1 - topp) / (n - 1) + topp * n = n - (1 - topp) * (n - 1) indices
    
    int n0 = 0;
    // Cutoff we use to decide if we should continue to consider a token
    float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    
    // Sort the values in descending order
    std::sort(probindex.begin(), probindex.begin() + n0, 
              [](const ProbIndex& a, const ProbIndex& b) {
                  return a.prob > b.prob;
              });
    
    // Truncate the list where cumulative probability > topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }
    
    // Sample from the truncated list
    float coin = static_cast<float>(rng()) / static_cast<float>(rng.max()) * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (coin < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of floating point errors
}

} // namespace qwen