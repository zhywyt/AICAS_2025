#include "math_utils.h"
#include <cmath>
#include <cstring>

namespace qwen {

void MathUtils::rmsnorm(float* o, float* x, float* weight, int size) {
    // Calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f; // epsilon to avoid division by zero
    ss = 1.0f / sqrtf(ss);
    
    // Normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void MathUtils::softmax(float* x, int size) {
    // Find max value for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // Exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void MathUtils::matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most expensive operation in the whole forward pass
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void MathUtils::apply_silu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = silu(x[i]);
    }
}

void MathUtils::element_mul(float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] *= b[i];
    }
}

void MathUtils::copy_array(float* dst, const float* src, int size) {
    std::memcpy(dst, src, size * sizeof(float));
}

} // namespace qwen