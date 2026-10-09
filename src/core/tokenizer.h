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

    // Text → token ids, matching llama.cpp. SentencePiece vocabularies
    // ("llama": Llama 2, Mistral, TinyLlama) and byte-level BPE ("gpt2") with
    // the Qwen2 / Llama 3 pre-tokenizers (tokenizer.ggml.pre = qwen2,
    // deepseek-r1-qwen, llama-bpe, llama3); other BPE pre-tokenizers return
    // Unsupported. Special tokens written in the text (e.g. "</s>",
    // "<|im_start|>") map to their ids.
    [[nodiscard]] Result<std::vector<int32_t>> encode(std::string_view text, bool add_bos) const;
    // Encodes `text` that directly follows already-encoded text ending with
    // token `previous` (a prompt that continues the context): no BOS and, for
    // SentencePiece vocabularies, the space prefix only after a control token,
    // as encode() would place it, so the earlier ids followed by these spell
    // the joined text.
    [[nodiscard]] Result<std::vector<int32_t>> encode_continuation(std::string_view text, int32_t previous) const;

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
    Result<std::vector<int32_t>> encode_from(std::string_view text, bool add_bos, bool after_control) const;
    Kind kind_ = Kind::Spm;
    std::vector<std::string_view> texts_;  // raw vocabulary entries (in the mapping)
    std::vector<float> scores_;
    std::vector<int32_t> types_;
    std::vector<std::string> pieces_;  // decoded bytes per token
    std::unordered_map<std::string_view, int32_t> lookup_;
    std::vector<int32_t> end_of_generation_;
    std::vector<int32_t> control_;  // control tokens with text, longest first

    enum class PreTokenizer { None, Qwen2, Qwen35, Llama3 };
    PreTokenizer pre_ = PreTokenizer::None;
    std::string pre_name_;
    std::unordered_map<std::string, int32_t> merge_rank_;  // "left right" -> rank (BPE)

    void encode_plain(std::string_view text, bool space_prefix, std::vector<int32_t>& out) const;
    void encode_bpe(std::string_view text, std::vector<int32_t>& out) const;
    void bpe_word(std::string_view word, std::vector<int32_t>& out) const;
    int32_t bos_ = -1;
    int32_t eos_ = -1;
    int32_t unk_ = -1;
    bool add_bos_ = true;
    bool add_space_prefix_ = true;
};

}  // namespace liyab

#endif  // LIYAB_CORE_TOKENIZER_H
