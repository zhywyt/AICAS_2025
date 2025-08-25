#include "tokenizer.h"
#include <fstream>
#include <algorithm>
#include <cstring>

namespace qwen {

bool Tokenizer::build_from_file(const std::string& tokenizer_path, int vocab_size) {
    this->vocab_size = vocab_size;
    
    std::ifstream file(tokenizer_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read max token length
    file.read(reinterpret_cast<char*>(&max_token_length), sizeof(max_token_length));
    
    // Initialize vocabulary vectors
    vocab.resize(vocab_size);
    vocab_scores.resize(vocab_size);
    
    // Read vocabulary
    for (int i = 0; i < vocab_size; i++) {
        // Read score
        file.read(reinterpret_cast<char*>(&vocab_scores[i]), sizeof(float));
        
        // Read token length
        int len = 0;
        file.read(reinterpret_cast<char*>(&len), sizeof(int));
        
        // Read token string
        vocab[i].resize(len);
        if (len > 0) {
            file.read(&vocab[i][0], len);
        }
    }
    
    if (!file.good()) {
        return false;
    }
    
    file.close();
    
    // Build sorted vocabulary for binary search
    sorted_vocab.resize(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        sorted_vocab[i] = TokenIndex(vocab[i], i);
    }
    
    // Sort by string for binary search
    std::sort(sorted_vocab.begin(), sorted_vocab.end(),
              [](const TokenIndex& a, const TokenIndex& b) {
                  return a.str < b.str;
              });
    
    // Initialize byte pieces for UTF-8 encoding
    for (int i = 0; i < 256; i++) {
        byte_pieces[i * 2] = (unsigned char)i;
        byte_pieces[i * 2 + 1] = '\0';
    }
    
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool bos, bool eos) {
    std::vector<int> tokens;
    
    if (text.empty()) {
        if (bos) tokens.push_back(1); // BOS token
        if (eos) tokens.push_back(2); // EOS token
        return tokens;
    }
    
    if (bos) tokens.push_back(1); // BOS token
    
    // Add each byte of the input text as a token
    encode_internal(text, tokens);
    
    if (eos) tokens.push_back(2); // EOS token
    
    return tokens;
}

std::string Tokenizer::decode(int token) {
    if (token < 0 || token >= vocab_size) {
        return "";
    }
    
    std::string piece = vocab[token];
    
    // Careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // Parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece.c_str(), "<0x%02hhX>", &byte_val) == 1) {
        piece = std::string(1, (char)byte_val);
    }
    
    return piece;
}

bool Tokenizer::is_valid() const {
    return vocab_size > 0 && !vocab.empty() && !vocab_scores.empty() && 
           vocab.size() == vocab_size && vocab_scores.size() == vocab_size;
}

int Tokenizer::str_lookup(const std::string& str) {
    // Binary search in sorted vocabulary
    auto it = std::lower_bound(sorted_vocab.begin(), sorted_vocab.end(), str,
                              [](const TokenIndex& a, const std::string& b) {
                                  return a.str < b;
                              });
    
    if (it != sorted_vocab.end() && it->str == str) {
        return it->id;
    }
    
    return -1; // not found
}

void Tokenizer::encode_internal(const std::string& text, std::vector<int>& tokens) {
    // Start with character-level encoding
    for (char c : text) {
        std::string char_str(1, c);
        int id = str_lookup(char_str);
        if (id != -1) {
            tokens.push_back(id);
        } else {
            // Fall back to byte-level encoding
            unsigned char byte_val = static_cast<unsigned char>(c);
            std::string byte_str = "<0x";
            byte_str += (byte_val >> 4) < 10 ? ('0' + (byte_val >> 4)) : ('A' + (byte_val >> 4) - 10);
            byte_str += (byte_val & 0xF) < 10 ? ('0' + (byte_val & 0xF)) : ('A' + (byte_val & 0xF) - 10);
            byte_str += ">";
            
            id = str_lookup(byte_str);
            if (id != -1) {
                tokens.push_back(id);
            }
        }
    }
    
    // Iteratively merge tokens based on vocabulary scores
    while (true) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            // Check if pair of tokens can be merged
            std::string str = vocab[tokens[i]] + vocab[tokens[i + 1]];
            int id = str_lookup(str);
            if (id != -1 && vocab_scores[id] > best_score) {
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        
        if (best_idx == -1) {
            break; // No more merges possible
        }
        
        // Merge the best pair
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }
}

} // namespace qwen