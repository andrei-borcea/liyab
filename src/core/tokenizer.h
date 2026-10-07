// Liyab — GGUF vocabulary: SentencePiece-BPE encoding, byte-level decoding (internal).
#ifndef LIYAB_CORE_TOKENIZER_H
#define LIYAB_CORE_TOKENIZER_H

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "liyab/mmap_loader.h"

namespace liyab {

class Tokenizer {
public:
    enum class Kind { Spm, Gpt2 };

    static Result<Tokenizer> load(const MmapLoader& model);

    // Text → token ids. Implemented for SentencePiece vocabularies ("llama":
    // Llama 2, Mistral, TinyLlama...). Byte-level BPE vocabularies ("gpt2":
    // Llama 3, Qwen) return Unsupported — tokenize on the app side and use
    // the token-id generation API.
    [[nodiscard]] Result<std::vector<int32_t>> encode(std::string_view text, bool add_bos) const;

    // Raw bytes for one token; control tokens decode to "". A multi-byte
    // UTF-8 character may span several tokens.
    [[nodiscard]] const std::string& piece(int32_t id) const;

    [[nodiscard]] Kind kind() const noexcept { return kind_; }
    [[nodiscard]] int32_t vocab_size() const noexcept { return static_cast<int32_t>(texts_.size()); }
    [[nodiscard]] int32_t bos() const noexcept { return bos_; }
    [[nodiscard]] int32_t eos() const noexcept { return eos_; }
    [[nodiscard]] bool add_bos_default() const noexcept { return add_bos_; }
    [[nodiscard]] bool is_end_of_generation(int32_t id) const;

private:
    Kind kind_ = Kind::Spm;
    std::vector<std::string_view> texts_;  // raw vocabulary entries (in the mapping)
    std::vector<float> scores_;
    std::vector<int32_t> types_;
    std::vector<std::string> pieces_;  // decoded bytes per token
    std::unordered_map<std::string_view, int32_t> lookup_;
    std::vector<int32_t> end_of_generation_;
    int32_t bos_ = -1;
    int32_t eos_ = -1;
    int32_t unk_ = -1;
    bool add_bos_ = true;
    bool add_space_prefix_ = true;
};

}  // namespace liyab

#endif  // LIYAB_CORE_TOKENIZER_H
