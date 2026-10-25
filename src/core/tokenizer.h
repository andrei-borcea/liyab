// Liyab — GGUF vocabulary: SentencePiece-BPE, byte-level BPE and unigram encoding,
// decoding (internal). Byte-level BPE pre-tokenizers come from a table keyed
// by tokenizer.ggml.pre (core/pre_tokenizer.h).
#ifndef LIYAB_CORE_TOKENIZER_H
#define LIYAB_CORE_TOKENIZER_H

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/pre_tokenizer.h"
#include "liyab/mmap_loader.h"

namespace liyab {

class Tokenizer {
public:
    enum class Kind { Spm, Gpt2, Unigram };

    // Loads vocabulary metadata. Scores may be F32 or the converter's I32
    // Tekken ranks; lengths and element types must match the vocabulary.
    static Result<Tokenizer> load(const MmapLoader& model);

    // Text → token ids, as the model's Hugging Face tokenizer encodes them.
    // `add_bos` adds the scheme's initial special tokens unless the text
    // already spells them (a rendered chat template wrote them). Usually
    // this is one BOS; chatglm-spm uses [gMASK], sop instead.
    // SentencePiece vocabularies ("llama": Llama 2, Mistral, Gemma, Phi-3)
    // and byte-level BPE ("gpt2") with the pre-tokenizers of
    // PreTokenizer::supported_names(); a BPE vocabulary with another
    // tokenizer.ggml.pre returns Unsupported naming it and the supported
    // ones (the model still loads, and runs on token ids). Special tokens
    // written in the text (e.g. "</s>", "<|im_start|>") map to their ids;
    // the text between them is normalized (NFC for the Qwen families) and
    // pre-tokenized on its own.
    [[nodiscard]] Result<std::vector<int32_t>> encode(std::string_view text, bool add_bos) const;
    // Initial special tokens to insert before `text`, or an empty span when
    // disabled/already written. Borrows this tokenizer; used by segmented
    // encoding to keep automatic prefix tokens outside reusable segments.
    [[nodiscard]] std::span<const int32_t> initial_tokens(std::string_view text, bool add_bos) const;
    // Encodes `text` that directly follows already-encoded text ending with
    // token `previous` (a prompt that continues the context): no BOS and, for
    // SentencePiece vocabularies, the space prefix only after a control token,
    // as encode() would place it, so the earlier ids followed by these spell
    // the joined text.
    [[nodiscard]] Result<std::vector<int32_t>> encode_continuation(std::string_view text, int32_t previous) const;

    // Raw bytes for one token; control tokens decode to "". A multi-byte
    // UTF-8 character may span several tokens.
    [[nodiscard]] const std::string& piece(int32_t id) const;
    // Whether encode() recognises the token's own text in raw text (control
    // and user-defined tokens such as "<|im_end|>").
    [[nodiscard]] bool is_special(int32_t id) const noexcept;
    // The text that token `id` stands for in a prompt given to encode(): a
    // special token's own text, else its piece, without the space a
    // SentencePiece vocabulary adds at the start of a segment
    // (`segment_start`: the token starts the text or follows a special one).
    // Used to find which tokens of a context a text still spells.
    [[nodiscard]] std::string_view prompt_text(int32_t id, bool segment_start) const;

    // The vocabulary entry of a token as the GGUF stores it (a special
    // token's own text, e.g. "<|begin_of_text|>").
    [[nodiscard]] std::string_view token_text(int32_t id) const;

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
    std::vector<bool> special_;     // per token: in control_
    std::vector<bool> rstrip_;      // per token: white space after it in a text is dropped (kRstripRules)
    // Special-token search (find_special): control_ by first byte, the byte
    // pairs that start one, and the bytes that are one on their own.
    std::array<std::vector<int32_t>, 256> special_by_byte_;
    std::vector<uint64_t> special_pairs_;
    std::array<bool, 256> special_single_{};
    // The earliest special token's text at or after `from` (the longest one
    // there): its offset and id, or npos.
    size_t find_special(std::string_view text, size_t from, int32_t& id) const;

    std::optional<PreTokenizer> pre_;  // byte-level BPE: the scheme of tokenizer.ggml.pre
    Status pre_error_;                 // why pre_ is empty for a BPE vocabulary
    struct Merge {
        int32_t rank;  // position in tokenizer.ggml.merges: lower merges first
        int32_t id;    // the token the pair becomes
    };
    std::unordered_map<uint64_t, Merge> merges_;  // BPE: (left id << 32 | right id) -> merge
    std::array<int32_t, 256> byte_token_{};       // BPE/unigram: each byte's token, -1 when absent

    void encode_plain(std::string_view text, bool space_prefix, std::vector<int32_t>& out) const;
    // PLaMo3's integer-score unigram DP, with hard boundaries around its
    // special-token spellings, consecutive spaces and repeated characters.
    void encode_unigram(std::string_view text, std::vector<int32_t>& out) const;
    struct UnigramNode {
        int32_t child = -1, next = -1, token = -1;
        unsigned char byte = 0;
    };
    std::vector<UnigramNode> unigram_trie_;
    std::vector<int32_t> unigram_scores_;
    void encode_bpe(std::string_view text, std::vector<int32_t>& out) const;
    void bpe_word(std::string_view word, std::vector<int32_t>& out) const;
    int32_t bos_ = -1;
    int32_t eos_ = -1;
    int32_t unk_ = -1;
    bool add_bos_ = true;
    bool add_space_prefix_ = true;
    std::vector<int32_t> initial_tokens_;
    std::string initial_text_;  // their literal spelling, when all are special
};

}  // namespace liyab

#endif  // LIYAB_CORE_TOKENIZER_H
