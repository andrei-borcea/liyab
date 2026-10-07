#include "core/tokenizer.h"

#include <algorithm>
#include <array>
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

// --- Unicode categories for the BPE pre-tokenizer --------------------------
struct CodepointRange {
    uint32_t first;
    uint32_t last;
};
#include "core/unicode_tables.inc"

template <size_t N>
bool in_ranges(const CodepointRange (&ranges)[N], uint32_t cp) noexcept {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < ranges[mid].first) hi = mid;
        else if (cp > ranges[mid].last) lo = mid + 1;
        else return true;
    }
    return false;
}

bool is_letter(uint32_t cp) noexcept { return in_ranges(kLetterRanges, cp); }
bool is_number(uint32_t cp) noexcept { return in_ranges(kNumberRanges, cp); }
bool is_mark(uint32_t cp) noexcept { return in_ranges(kMarkRanges, cp); }
bool is_whitespace(uint32_t cp) noexcept {  // Unicode White_Space property
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

// GPT-2 bytes_to_unicode(): byte -> UTF-8 of its printable stand-in.
const std::array<std::string, 256>& gpt2_byte_encoder() {
    static const std::array<std::string, 256> table = [] {
        std::array<std::string, 256> t;
        int32_t extra = 0;
        for (int32_t b = 0; b < 256; ++b) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            const uint32_t cp = printable ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra++);
            std::string utf8;
            if (cp < 0x80) {
                utf8 += static_cast<char>(cp);
            } else {
                utf8 += static_cast<char>(0xC0 | (cp >> 6));
                utf8 += static_cast<char>(0x80 | (cp & 0x3F));
            }
            t[static_cast<size_t>(b)] = utf8;
        }
        return t;
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

    // Special tokens recognised in raw text: control and user-defined ones.
    for (size_t i = 0; i < n; ++i) {
        if ((tok.types_[i] == kControl || tok.types_[i] == kUserDefined) && !tok.texts_[i].empty()) {
            tok.control_.push_back(static_cast<int32_t>(i));
        }
    }
    if (tok.kind_ == Kind::Gpt2) {
        const auto pre = model.get_string("tokenizer.ggml.pre");
        tok.pre_name_ = pre ? std::string(*pre) : "default";
        if (tok.pre_name_ == "qwen2" || tok.pre_name_ == "deepseek-r1-qwen") tok.pre_ = PreTokenizer::Qwen2;
        else if (tok.pre_name_ == "qwen35") tok.pre_ = PreTokenizer::Qwen35;
        else if (tok.pre_name_ == "llama-bpe" || tok.pre_name_ == "llama3") tok.pre_ = PreTokenizer::Llama3;
        const GgufValue* merges = model.metadata("tokenizer.ggml.merges");
        if (merges == nullptr || merges->array_type != GgufValue::Type::String) {
            return Status(ErrorCode::InvalidModel, "BPE vocabulary without tokenizer.ggml.merges");
        }
        tok.merge_rank_.reserve(merges->strings.size());
        for (size_t r = 0; r < merges->strings.size(); ++r) {
            tok.merge_rank_.emplace(std::string(merges->strings[r]), static_cast<int32_t>(r));
        }
    }
    std::sort(tok.control_.begin(), tok.control_.end(), [&](int32_t a, int32_t b) {
        return tok.texts_[static_cast<size_t>(a)].size() > tok.texts_[static_cast<size_t>(b)].size();
    });

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
    if (kind_ == Kind::Gpt2 && pre_ == PreTokenizer::None) {
        return Status(ErrorCode::Unsupported, "BPE pre-tokenizer '" + pre_name_ +
                                                  "' is not supported yet (qwen2, qwen35 and llama3 are); pass token ids instead");
    }
    std::vector<int32_t> out;
    if (add_bos && bos_ >= 0) out.push_back(bos_);
    // Split around control-token texts; plain segments go through SPM. As in
    // llama.cpp, a segment gets the space prefix when it starts the text or
    // follows a control token.
    bool prev_control = true;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = std::string_view::npos;
        int32_t best_id = -1;
        for (const int32_t id : control_) {
            const size_t at = text.find(texts_[static_cast<size_t>(id)], pos);
            if (at < best) {  // earliest match; longest wins ties (control_ is sorted by length)
                best = at;
                best_id = id;
            }
        }
        const std::string_view plain = text.substr(pos, best == std::string_view::npos ? std::string_view::npos : best - pos);
        if (!plain.empty()) {
            if (kind_ == Kind::Gpt2) encode_bpe(plain, out);
            else encode_plain(plain, prev_control && add_space_prefix_, out);
            prev_control = false;
        }
        if (best == std::string_view::npos) break;
        out.push_back(best_id);
        prev_control = true;
        pos = best + texts_[static_cast<size_t>(best_id)].size();
    }
    if (std::find(out.begin(), out.end(), unk_) != out.end() && unk_ < 0) {
        return Status(ErrorCode::InvalidArgument, "text contains bytes the vocabulary cannot represent");
    }
    return out;
}

void Tokenizer::encode_plain(std::string_view text, bool space_prefix, std::vector<int32_t>& out) const {
    std::string normalized;
    normalized.reserve(text.size() * 2 + kSpmSpace.size());
    if (space_prefix) normalized += kSpmSpace;
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
            else out.push_back(unk_);  // -1 when the vocabulary has no <unk>: reported by encode()
        }
    }
}

// ---------------------------------------------------------------------------
// Byte-level BPE (GPT-2 family: Qwen2, Llama 3)
// ---------------------------------------------------------------------------
void Tokenizer::encode_bpe(std::string_view text, std::vector<int32_t>& out) const {
    // Decode to code points, remembering byte offsets for slicing.
    std::vector<uint32_t> cps;
    std::vector<size_t> offsets;
    for (size_t i = 0; i < text.size();) {
        offsets.push_back(i);
        const int32_t cp = next_codepoint(text, i);
        cps.push_back(cp < 0 ? 0xFFFD : static_cast<uint32_t>(cp));
    }
    offsets.push_back(text.size());
    const size_t n = cps.size();
    constexpr uint32_t kEnd = 0xFFFFFFFF;
    auto cp_at = [&](size_t i) { return i < n ? cps[i] : kEnd; };
    // Word characters: letters, plus combining marks for Qwen3.5 ([\p{L}\p{M}]+).
    const bool marks_in_words = pre_ == PreTokenizer::Qwen35;
    auto letter = [&](size_t i) { return i < n && (is_letter(cps[i]) || (marks_in_words && is_mark(cps[i]))); };
    auto number = [&](size_t i) { return i < n && is_number(cps[i]); };
    auto space = [&](size_t i) { return i < n && is_whitespace(cps[i]); };
    auto lower = [](uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; };

    // Hand-written equivalent of the Qwen2 / Qwen3.5 / Llama 3 pre-tokenizer regex
    // (same structure as llama.cpp's unicode_regex_split_custom_llama3):
    //   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3}  (Qwen2/3.5: \p{N})
    //   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
    // Qwen3.5 reads \p{L} as [\p{L}\p{M}] in the word and symbol alternatives.
    const size_t max_digits = pre_ == PreTokenizer::Llama3 ? 3 : 1;
    size_t start = 0;
    size_t pos = 0;
    auto emit = [&](size_t end) {
        if (end > start) bpe_word(text.substr(offsets[start], offsets[end] - offsets[start]), out);
        start = pos = end;
    };
    while (pos < n) {
        const uint32_t cp = cps[pos];
        if (cp == '\'' && pos + 1 < n) {  // contractions
            const uint32_t c1 = lower(cp_at(pos + 1));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                emit(pos + 2);
                continue;
            }
            if (pos + 2 < n) {
                const uint32_t c2 = lower(cp_at(pos + 2));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    emit(pos + 3);
                    continue;
                }
            }
        }
        if (!(cp == '\r' || cp == '\n' || number(pos)) && (letter(pos) || letter(pos + 1))) {  // [^\r\n\p{L}\p{N}]?\p{L}+
            size_t end = pos + 1;
            while (letter(end)) ++end;
            emit(end);
            continue;
        }
        if (number(pos)) {  // digits, grouped by at most max_digits
            size_t end = pos;
            while (number(end) && end - pos < max_digits) ++end;
            emit(end);
            continue;
        }
        const size_t sym = cp == ' ' ? pos + 1 : pos;  //  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (sym < n && !space(sym) && !letter(sym) && !number(sym)) {
            size_t end = sym;
            while (end < n && !space(end) && !letter(end) && !number(end)) ++end;
            while (cp_at(end) == '\r' || cp_at(end) == '\n') ++end;
            emit(end);
            continue;
        }
        size_t spaces = 0;
        size_t last_newline_end = 0;
        while (space(pos + spaces)) {
            const uint32_t c = cps[pos + spaces];
            if (c == '\r' || c == '\n') last_newline_end = pos + spaces + 1;
            ++spaces;
        }
        if (last_newline_end > 0) {  // \s*[\r\n]+
            emit(last_newline_end);
            continue;
        }
        if (spaces > 1 && pos + spaces < n) {  // \s+(?!\S): leave one space for the next word
            emit(pos + spaces - 1);
            continue;
        }
        if (spaces > 0) {  // \s+
            emit(pos + spaces);
            continue;
        }
        emit(pos + 1);  // anything else: a single code point
    }
}

void Tokenizer::bpe_word(std::string_view word, std::vector<int32_t>& out) const {
    // Byte-level mapping, then merges by rank (lowest first, leftmost on ties).
    const auto& encoder = gpt2_byte_encoder();
    std::vector<std::string> symbols;
    symbols.reserve(word.size());
    for (const char c : word) symbols.push_back(encoder[static_cast<unsigned char>(c)]);
    if (symbols.size() > 1) {
        std::string key;
        for (;;) {
            int32_t best_rank = INT32_MAX;
            size_t best = 0;
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                key.assign(symbols[i]).append(" ").append(symbols[i + 1]);
                const auto it = merge_rank_.find(key);
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best = i;
                }
            }
            if (best_rank == INT32_MAX) break;
            symbols[best] += symbols[best + 1];
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best) + 1);
        }
    }
    for (const std::string& sym : symbols) {
        const auto it = lookup_.find(sym);
        if (it != lookup_.end()) {
            out.push_back(it->second);
            continue;
        }
        // Not in the vocabulary (should not happen for byte-level BPE): fall back to its bytes.
        size_t i = 0;
        while (i < sym.size()) {
            const size_t len = std::min(utf8_len(static_cast<unsigned char>(sym[i])), sym.size() - i);
            const auto byte_it = lookup_.find(std::string_view(sym).substr(i, len));
            out.push_back(byte_it != lookup_.end() ? byte_it->second : unk_);
            i += len;
        }
    }
}

}  // namespace liyab
