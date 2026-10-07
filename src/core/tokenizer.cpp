#include "core/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <optional>
#include <queue>

namespace liyab {

namespace {

constexpr std::string_view kSpmSpace = "\xE2\x96\x81";  // U+2581 '▁'

enum TokenType : int32_t { kNormal = 1, kUnknown = 2, kControl = 3, kUserDefined = 4, kUnused = 5, kByte = 6 };

size_t utf8_len(unsigned char lead) noexcept {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x6) return 2;
    if ((lead >> 4) == 0xE) return 3;
    if ((lead >> 3) == 0x1E) return 4;
    return 1;  // invalid lead byte: treat as a single byte
}

// Decodes one UTF-8 code point; advances `i`. Returns -1 on malformed input.
int32_t next_codepoint(std::string_view s, size_t& i) noexcept {
    const auto lead = static_cast<unsigned char>(s[i]);
    const size_t len = utf8_len(lead);
    if (i + len > s.size()) {
        ++i;
        return -1;
    }
    int32_t cp = len == 1 ? lead : lead & (0x7F >> len);
    for (size_t k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    i += len;
    return cp;
}

// Inverse of GPT-2's bytes_to_unicode(): code point → original byte.
const std::unordered_map<int32_t, uint8_t>& gpt2_byte_decoder() {
    static const std::unordered_map<int32_t, uint8_t> table = [] {
        std::unordered_map<int32_t, uint8_t> m;
        int32_t extra = 0;
        for (int32_t b = 0; b < 256; ++b) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            m[printable ? b : 256 + extra++] = static_cast<uint8_t>(b);
        }
        return m;
    }();
    return table;
}

bool parse_byte_token(std::string_view text, uint8_t& out) {
    if (text.size() != 6 || text.substr(0, 3) != "<0x" || text[5] != '>') return false;
    unsigned value = 0;
    if (std::sscanf(std::string(text.substr(3, 2)).c_str(), "%2x", &value) != 1) return false;
    out = static_cast<uint8_t>(value);
    return true;
}

std::optional<int32_t> token_id(const MmapLoader& model, std::string_view key, int32_t vocab) {
    const auto v = model.get_int(key);
    if (!v || *v < 0 || *v >= vocab) return std::nullopt;
    return static_cast<int32_t>(*v);
}

}  // namespace

Result<Tokenizer> Tokenizer::load(const MmapLoader& model) {
    Tokenizer tok;
    const auto kind = model.get_string("tokenizer.ggml.model");
    if (!kind) return Status(ErrorCode::InvalidModel, "missing tokenizer.ggml.model");
    if (*kind == "llama") tok.kind_ = Kind::Spm;
    else if (*kind == "gpt2") tok.kind_ = Kind::Gpt2;
    else return Status(ErrorCode::Unsupported, "tokenizer type '" + std::string(*kind) + "' is not supported");

    const GgufValue* tokens = model.metadata("tokenizer.ggml.tokens");
    if (!tokens || tokens->type != GgufValue::Type::Array || tokens->array_type != GgufValue::Type::String ||
        tokens->strings.empty()) {
        return Status(ErrorCode::InvalidModel, "missing tokenizer.ggml.tokens");
    }
    const size_t n = tokens->strings.size();
    tok.texts_ = tokens->strings;

    tok.scores_.assign(n, 0.0f);
    if (const GgufValue* s = model.metadata("tokenizer.ggml.scores")) {
        if (s->type != GgufValue::Type::Array || s->array_type != GgufValue::Type::F32 || s->array_size != n) {
            return Status(ErrorCode::InvalidModel, "tokenizer.ggml.scores does not match the vocabulary");
        }
        std::memcpy(tok.scores_.data(), s->array_data, n * sizeof(float));
    }
    tok.types_.assign(n, kNormal);
    if (const GgufValue* t = model.metadata("tokenizer.ggml.token_type")) {
        if (t->type != GgufValue::Type::Array || t->array_type != GgufValue::Type::I32 || t->array_size != n) {
            return Status(ErrorCode::InvalidModel, "tokenizer.ggml.token_type does not match the vocabulary");
        }
        std::memcpy(tok.types_.data(), t->array_data, n * sizeof(int32_t));
    }

    const auto vocab = static_cast<int32_t>(n);
    tok.bos_ = token_id(model, "tokenizer.ggml.bos_token_id", vocab).value_or(-1);
    tok.eos_ = token_id(model, "tokenizer.ggml.eos_token_id", vocab).value_or(-1);
    tok.unk_ = token_id(model, "tokenizer.ggml.unknown_token_id", vocab).value_or(-1);
    if (const GgufValue* v = model.metadata("tokenizer.ggml.add_bos_token")) {
        tok.add_bos_ = v->as_bool().value_or(true);
    } else {
        tok.add_bos_ = tok.kind_ == Kind::Spm;
    }
    if (const GgufValue* v = model.metadata("tokenizer.ggml.add_space_prefix")) {
        tok.add_space_prefix_ = v->as_bool().value_or(true);
    }
    if (tok.bos_ < 0) tok.add_bos_ = false;

    tok.pieces_.resize(n);
    tok.lookup_.reserve(n);
    const auto& byte_decoder = gpt2_byte_decoder();
    for (size_t i = 0; i < n; ++i) {
        const std::string_view text = tok.texts_[i];
        tok.lookup_.emplace(text, static_cast<int32_t>(i));
        std::string& piece = tok.pieces_[i];
        const int32_t type = tok.types_[i];
        if (type == kControl || type == kUnused) continue;

        uint8_t byte = 0;
        if (tok.kind_ == Kind::Spm) {
            if ((type == kByte || type == kNormal) && parse_byte_token(text, byte)) {
                piece.assign(1, static_cast<char>(byte));
                continue;
            }
            for (size_t pos = 0; pos < text.size();) {
                if (text.substr(pos, kSpmSpace.size()) == kSpmSpace) {
                    piece += ' ';
                    pos += kSpmSpace.size();
                } else {
                    piece += text[pos++];
                }
            }
        } else if (type == kUserDefined) {
            piece.assign(text);
        } else {
            bool ok = true;
            for (size_t pos = 0; pos < text.size() && ok;) {
                const auto it = byte_decoder.find(next_codepoint(text, pos));
                ok = it != byte_decoder.end();
                if (ok) piece += static_cast<char>(it->second);
            }
            if (!ok) piece.assign(text);
        }
    }

    // End-of-generation set: EOS/EOT ids plus well-known chat terminators.
    auto add_eog = [&](int32_t id) {
        if (id >= 0 && std::find(tok.end_of_generation_.begin(), tok.end_of_generation_.end(), id) ==
                           tok.end_of_generation_.end()) {
            tok.end_of_generation_.push_back(id);
        }
    };
    add_eog(tok.eos_);
    add_eog(token_id(model, "tokenizer.ggml.eot_token_id", vocab).value_or(-1));
    add_eog(token_id(model, "tokenizer.ggml.eom_token_id", vocab).value_or(-1));
    for (const char* text : {"<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>"}) {
        const auto it = tok.lookup_.find(text);
        if (it != tok.lookup_.end() && tok.types_[static_cast<size_t>(it->second)] == kControl) add_eog(it->second);
    }
    return tok;
}

bool Tokenizer::is_end_of_generation(int32_t id) const {
    return std::find(end_of_generation_.begin(), end_of_generation_.end(), id) != end_of_generation_.end();
}

const std::string& Tokenizer::piece(int32_t id) const {
    static const std::string empty;
    return id >= 0 && static_cast<size_t>(id) < pieces_.size() ? pieces_[static_cast<size_t>(id)] : empty;
}

Result<std::vector<int32_t>> Tokenizer::encode(std::string_view text, bool add_bos) const {
    if (kind_ != Kind::Spm) {
        return Status(ErrorCode::Unsupported,
                      "text encoding for byte-level BPE (gpt2) vocabularies is not implemented; "
                      "pass token ids instead");
    }
    std::vector<int32_t> out;
    if (add_bos && bos_ >= 0) out.push_back(bos_);
    if (text.empty()) return out;

    std::string normalized;
    normalized.reserve(text.size() * 2 + kSpmSpace.size());
    if (add_space_prefix_) normalized += kSpmSpace;
    for (const char c : text) {
        if (c == ' ') normalized += kSpmSpace;
        else normalized += c;
    }

    // SentencePiece BPE: start from UTF-8 characters and repeatedly merge the
    // adjacent pair whose concatenation is the highest-scoring vocab entry.
    struct Symbol {
        int32_t prev;
        int32_t next;
        const char* text;
        size_t size;
    };
    std::vector<Symbol> symbols;
    for (size_t pos = 0; pos < normalized.size();) {
        const size_t len = std::min(utf8_len(static_cast<unsigned char>(normalized[pos])), normalized.size() - pos);
        const auto index = static_cast<int32_t>(symbols.size());
        symbols.push_back({index - 1, index + 1, normalized.data() + pos, len});
        pos += len;
    }
    symbols.back().next = -1;

    struct Bigram {
        int32_t left;
        int32_t right;
        float score;
        size_t size;
    };
    auto worse = [](const Bigram& a, const Bigram& b) {
        return a.score < b.score || (a.score == b.score && a.left > b.left);
    };
    std::priority_queue<Bigram, std::vector<Bigram>, decltype(worse)> queue(worse);
    auto try_add = [&](int32_t left, int32_t right) {
        if (left < 0 || right < 0) return;
        const std::string_view merged(symbols[static_cast<size_t>(left)].text,
                                      symbols[static_cast<size_t>(left)].size + symbols[static_cast<size_t>(right)].size);
        const auto it = lookup_.find(merged);
        if (it == lookup_.end()) return;
        queue.push({left, right, scores_[static_cast<size_t>(it->second)], merged.size()});
    };
    for (size_t i = 1; i < symbols.size(); ++i) try_add(static_cast<int32_t>(i - 1), static_cast<int32_t>(i));

    while (!queue.empty()) {
        const Bigram b = queue.top();
        queue.pop();
        Symbol& left = symbols[static_cast<size_t>(b.left)];
        Symbol& right = symbols[static_cast<size_t>(b.right)];
        if (left.size == 0 || right.size == 0 || left.size + right.size != b.size) continue;  // stale
        left.size += right.size;
        right.size = 0;
        left.next = right.next;
        if (right.next >= 0) symbols[static_cast<size_t>(right.next)].prev = b.left;
        try_add(left.prev, b.left);
        try_add(b.left, left.next);
    }

    for (int32_t i = 0; i >= 0; i = symbols[static_cast<size_t>(i)].next) {
        const Symbol& s = symbols[static_cast<size_t>(i)];
        const auto it = lookup_.find(std::string_view(s.text, s.size));
        if (it != lookup_.end()) {
            out.push_back(it->second);
            continue;
        }
        for (size_t k = 0; k < s.size; ++k) {  // byte fallback
            char byte_text[8];
            std::snprintf(byte_text, sizeof byte_text, "<0x%02X>", static_cast<unsigned char>(s.text[k]));
            const auto byte_it = lookup_.find(byte_text);
            if (byte_it != lookup_.end()) out.push_back(byte_it->second);
            else if (unk_ >= 0) out.push_back(unk_);
            else return Status(ErrorCode::InvalidArgument, "text contains bytes the vocabulary cannot represent");
        }
    }
    return out;
}

}  // namespace liyab
