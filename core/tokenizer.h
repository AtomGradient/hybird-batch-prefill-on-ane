#pragma once

#include <tokenizers_cpp.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ane_lm {

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // Move-only (unique_ptr member)
    Tokenizer(Tokenizer&&) = default;
    Tokenizer& operator=(Tokenizer&&) = default;
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    bool init(const std::string& model_dir);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& ids) const;

    // Apply Jinja chat template to messages, returns formatted prompt string.
    // messages: vector of {role, content} pairs
    // add_generation_prompt: append assistant turn opening
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt = true,
        bool enable_thinking = false) const;

    bool has_chat_template() const { return !chat_template_.empty(); }

    int eos_id() const { return eos_id_; }
    int im_start_id() const { return im_start_id_; }
    int im_end_id() const { return im_end_id_; }
    int vocab_size() const { return vocab_size_; }

private:
    std::unique_ptr<tokenizers::Tokenizer> tok_;
    int vocab_size_ = 0;
    int eos_id_ = 248044;
    int im_start_id_ = 248045;
    int im_end_id_ = 248046;
    std::string chat_template_;
    std::string bos_token_;
    std::string eos_token_;
};

} // namespace ane_lm
