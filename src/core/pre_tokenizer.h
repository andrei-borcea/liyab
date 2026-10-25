// Liyab — byte-level BPE pre-tokenizers as data (internal).
//
// Before merging, a byte-level BPE tokenizer normalizes the text and cuts it
// into pieces with one or more regular expressions; merges never cross a
// piece boundary. GGUF files name the scheme in `tokenizer.ggml.pre`. The
// table in pre_tokenizer.cpp gives, for each supported name, what the model's
// Hugging Face tokenizer.json specifies:
//
//   * the split steps, applied in order, each to every piece of the previous
//     step: a pattern (core/regex.h) and whether the text between matches is
//     kept as pieces of its own (Split "Isolated") or dropped (Split
//     "Removed" with invert, which keeps only the matches);
//   * NFC normalization of the text first (the Qwen tokenizers);
//   * `ignore_merges`: a piece that is a vocabulary entry as a whole is that
//     token, without merging (Llama 3, Hunyuan, Tekken).
//
// Adding a tokenizer family is adding a row, checked against the Hugging Face
// tokenizer with tools/tokenizer_reference.py and liyab-tok. An unknown name
// is an error that lists the supported ones: guessing would encode text into
// ids the model was not trained on, silently.
#ifndef LIYAB_CORE_PRE_TOKENIZER_H
#define LIYAB_CORE_PRE_TOKENIZER_H

#include <string>
#include <string_view>
#include <vector>

#include "core/regex.h"
#include "liyab/types.h"

namespace liyab {

class PreTokenizer {
public:
    // The pre-tokenizer named `name` (tokenizer.ggml.pre; a GGUF without the
    // key uses "default", the GPT-2 scheme). Unsupported names the value and
    // the supported list.
    static Result<PreTokenizer> for_name(std::string_view name);
    // Every supported tokenizer.ggml.pre value, comma-separated.
    static std::string supported_names();

    // Appends the pieces of `text` (already normalized) to `pieces`, as byte
    // ranges of `text`, in order. Every byte of `text` is in exactly one
    // piece unless a step drops what lies between its matches.
    void split(std::string_view text, std::vector<std::string_view>& pieces) const;

    [[nodiscard]] bool nfc() const noexcept { return nfc_; }
    [[nodiscard]] bool ignore_merges() const noexcept { return ignore_merges_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

private:
    struct Step {
        Regex regex;
        bool keep_gaps;
    };
    PreTokenizer() = default;
    std::string name_;
    std::vector<Step> steps_;
    bool nfc_ = false;
    bool ignore_merges_ = false;
    bool right_digits3_ = false;  // isolate decimal runs into groups of three aligned at their right end
};

}  // namespace liyab

#endif  // LIYAB_CORE_PRE_TOKENIZER_H
