#include "core/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <queue>

#include "core/log.h"
#include "core/unicode.h"

namespace liyab {

namespace {

constexpr std::string_view kSpmSpace = "\xE2\x96\x81";  // U+2581 '▁'

enum TokenType : int32_t { kNormal = 1, kUnknown = 2, kControl = 3, kUserDefined = 4, kUnused = 5, kByte = 6 };

size_t utf8_len(unsigned char lead) noexcept { return unicode::utf8_length(lead); }

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

uint64_t merge_key(int32_t left, int32_t right) noexcept {
    return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) | static_cast<uint32_t>(right);
}

std::optional<int32_t> token_id(const MmapLoader& model, std::string_view key, int32_t vocab) {
    const auto v = model.get_int(key);
    if (!v || *v < 0 || *v >= vocab) return std::nullopt;
    return static_cast<int32_t>(*v);
}

// Special tokens that swallow the white space after them (Hugging Face
// added tokens with rstrip), by architecture: GGUF keeps no such flag. The
// Phi-3 family (Phi-3/3.5, Phi-4-mini) strips after every special token but
// the listed ones, so "<|user|>\nHi" encodes as "<|user|>" then "Hi".
struct RstripRule {
    std::string_view architecture;
    std::initializer_list<std::string_view> except;
};
constexpr RstripRule kRstripRules[] = {
    {"phi3", {"<unk>", "<s>", "<|endoftext|>", "<|endofprompt|>"}},
};

}  // namespace

Result<Tokenizer> Tokenizer::load(const MmapLoader& model) {
    Tokenizer tok;
    const auto kind = model.get_string("tokenizer.ggml.model");
    if (!kind) return Status(ErrorCode::InvalidModel, "missing tokenizer.ggml.model");
    if (*kind == "llama" || *kind == "gemma4") tok.kind_ = Kind::Spm;
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
                const auto it = byte_decoder.find(static_cast<int32_t>(unicode::decode_utf8(text, pos)));
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
        // A GGUF without tokenizer.ggml.pre predates the key: GPT-2's scheme, as the format defines.
        const auto pre = model.get_string("tokenizer.ggml.pre");
        auto scheme = PreTokenizer::for_name(pre ? *pre : std::string_view("default"));
        if (scheme) {
            tok.pre_.emplace(std::move(scheme).value());
        } else {
            tok.pre_error_ = scheme.status();
            LIYAB_LOG_WARN("%s: text cannot be encoded, token ids only", tok.pre_error_.message().c_str());
        }
        const GgufValue* merges = model.metadata("tokenizer.ggml.merges");
        if (merges == nullptr || merges->array_type != GgufValue::Type::String) {
            return Status(ErrorCode::InvalidModel, "BPE vocabulary without tokenizer.ggml.merges");
        }
        // Merges by the ids they join: merging then compares integers, never
        // builds strings. A merge whose parts or result are not tokens can
        // never apply and is dropped; the first rank of a pair wins.
        tok.merges_.reserve(merges->strings.size());
        std::string joined;
        for (size_t r = 0; r < merges->strings.size(); ++r) {
            const std::string_view m = merges->strings[r];
            const size_t space = m.find(' ', 1);
            if (space == std::string_view::npos) continue;
            const auto left = tok.lookup_.find(m.substr(0, space));
            const auto right = tok.lookup_.find(m.substr(space + 1));
            joined.assign(m.substr(0, space)).append(m.substr(space + 1));
            const auto merged = tok.lookup_.find(joined);
            if (left == tok.lookup_.end() || right == tok.lookup_.end() || merged == tok.lookup_.end()) continue;
            tok.merges_.emplace(merge_key(left->second, right->second), Merge{static_cast<int32_t>(r), merged->second});
        }
        const auto& encoder = gpt2_byte_encoder();
        for (size_t b = 0; b < 256; ++b) {
            const auto it = tok.lookup_.find(encoder[b]);
            tok.byte_token_[b] = it == tok.lookup_.end() ? -1 : it->second;
        }
    }
    tok.special_.assign(n, false);
    for (const int32_t id : tok.control_) tok.special_[static_cast<size_t>(id)] = true;
    std::sort(tok.control_.begin(), tok.control_.end(), [&](int32_t a, int32_t b) {
        return tok.texts_[static_cast<size_t>(a)].size() > tok.texts_[static_cast<size_t>(b)].size();
    });
    tok.special_pairs_.assign(65536 / 64, 0);
    for (const int32_t id : tok.control_) {
        const std::string_view t = tok.texts_[static_cast<size_t>(id)];
        tok.special_by_byte_[static_cast<unsigned char>(t[0])].push_back(id);  // longest first, as control_
        if (t.size() == 1) {
            tok.special_single_[static_cast<unsigned char>(t[0])] = true;
        } else {
            const size_t pair = static_cast<unsigned char>(t[0]) << 8 | static_cast<unsigned char>(t[1]);
            tok.special_pairs_[pair >> 6] |= 1ull << (pair & 63);
        }
    }

    tok.rstrip_.assign(n, false);
    const auto arch = model.get_string("general.architecture");
    for (const RstripRule& rule : kRstripRules) {
        if (!arch || *arch != rule.architecture) continue;
        for (const int32_t id : tok.control_) {
            const std::string_view t = tok.texts_[static_cast<size_t>(id)];
            tok.rstrip_[static_cast<size_t>(id)] =
                std::find(rule.except.begin(), rule.except.end(), t) == rule.except.end();
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

bool Tokenizer::is_special(int32_t id) const noexcept {
    return id >= 0 && static_cast<size_t>(id) < special_.size() && special_[static_cast<size_t>(id)];
}

std::string_view Tokenizer::prompt_text(int32_t id, bool segment_start) const {
    if (is_special(id)) return texts_[static_cast<size_t>(id)];
    std::string_view text = piece(id);
    if (segment_start && kind_ == Kind::Spm && add_space_prefix_ && text.starts_with(' ')) text.remove_prefix(1);
    return text;
}

Result<std::vector<int32_t>> Tokenizer::encode(std::string_view text, bool add_bos) const {
    if (add_bos && bos_ >= 0 && special_[static_cast<size_t>(bos_)] && text.starts_with(texts_[static_cast<size_t>(bos_)])) {
        add_bos = false;  // the text spells its BOS (a chat template wrote it): no second one
    }
    return encode_from(text, add_bos, true);
}

std::string_view Tokenizer::token_text(int32_t id) const {
    return id >= 0 && static_cast<size_t>(id) < texts_.size() ? texts_[static_cast<size_t>(id)] : std::string_view();
}

Result<std::vector<int32_t>> Tokenizer::encode_continuation(std::string_view text, int32_t previous) const {
    return encode_from(text, false, std::find(control_.begin(), control_.end(), previous) != control_.end());
}

size_t Tokenizer::find_special(std::string_view text, size_t from, int32_t& id) const {
    // One pass over the text: a position is checked only when its first two
    // bytes start some special token, then against the tokens starting with
    // that byte, longest first. Searching the text once per special token
    // instead took 1.3 s for an 80 KB Llama 3 conversation (256 of them).
    for (size_t i = from; i < text.size(); ++i) {
        const auto b0 = static_cast<unsigned char>(text[i]);
        if (!special_single_[b0]) {
            if (i + 1 >= text.size()) break;
            const size_t pair = b0 << 8 | static_cast<unsigned char>(text[i + 1]);
            if (((special_pairs_[pair >> 6] >> (pair & 63)) & 1) == 0) continue;
        }
        for (const int32_t candidate : special_by_byte_[b0]) {
            const std::string_view t = texts_[static_cast<size_t>(candidate)];
            if (text.compare(i, t.size(), t) == 0) {
                id = candidate;
                return i;
            }
        }
    }
    return std::string_view::npos;
}

Result<std::vector<int32_t>> Tokenizer::encode_from(std::string_view text, bool add_bos, bool after_control) const {
    if (kind_ == Kind::Gpt2 && !pre_) return pre_error_;
    std::vector<int32_t> out;
    if (add_bos && bos_ >= 0) out.push_back(bos_);
    // Split around control-token texts; plain segments go through SPM. As in
    // llama.cpp, a segment gets the space prefix when it starts the text or
    // follows a control token.
    bool prev_control = after_control;
    size_t pos = 0;
    while (pos < text.size()) {
        int32_t best_id = -1;
        const size_t best = find_special(text, pos, best_id);
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
        if (rstrip_[static_cast<size_t>(best_id)]) {  // the white space after it is not text
            for (size_t next = pos; pos < text.size() && unicode::is_whitespace(unicode::decode_utf8(text, next));) {
                pos = next;
            }
        }
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
// Byte-level BPE (GPT-2 family: Qwen, Llama 3, DeepSeek, Hunyuan, Phi...)
// ---------------------------------------------------------------------------
void Tokenizer::encode_bpe(std::string_view text, std::vector<int32_t>& out) const {
    // Normalize, cut into pieces with the family's patterns, then merge each
    // piece on its own (core/pre_tokenizer.h).
    std::string normalized;
    if (pre_->nfc() && !unicode::nfc_quick_check(text)) {
        normalized = unicode::to_nfc(text);
        text = normalized;
    }
    thread_local std::vector<std::string_view> pieces;
    pieces.clear();
    pre_->split(text, pieces);
    for (const std::string_view piece : pieces) bpe_word(piece, out);
}

void Tokenizer::bpe_word(std::string_view word, std::vector<int32_t>& out) const {
    if (pre_->ignore_merges()) {  // a piece that is a token as a whole is that token
        const auto& encoder = gpt2_byte_encoder();
        thread_local std::string mapped;
        mapped.clear();
        for (const char c : word) mapped += encoder[static_cast<unsigned char>(c)];
        const auto it = lookup_.find(mapped);
        if (it != lookup_.end()) {
            out.push_back(it->second);
            return;
        }
    }
    // One symbol per byte (its byte-level token), then merges by rank, lowest
    // first and leftmost on ties, over a linked list with a heap of candidate
    // pairs: O(n log n), so a long piece (a run of digits or punctuation)
    // stays cheap. A heap entry whose symbols changed since is stale and skipped.
    struct Symbol {
        int32_t id;
        int32_t prev, next;
    };
    thread_local std::vector<Symbol> symbols;
    symbols.clear();
    for (const char c : word) {
        int32_t id = byte_token_[static_cast<unsigned char>(c)];
        if (id < 0) {
            // A byte the vocabulary lacks (Hunyuan's dense models have no "\r")
            // is <unk>, or left out before merging when there is none, as the
            // Hugging Face tokenizer does: its neighbours then merge together.
            if (unk_ < 0) continue;
            id = unk_;
        }
        const auto index = static_cast<int32_t>(symbols.size());
        symbols.push_back({id, index - 1, index + 1});
    }
    if (symbols.empty()) return;
    symbols.back().next = -1;

    struct Candidate {
        int32_t rank;
        int32_t left;  // symbol index; leftmost wins among equal ranks
        int32_t left_id, right_id, merged;
    };
    auto later = [](const Candidate& a, const Candidate& b) {
        return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
    };
    thread_local std::vector<Candidate> heap;
    heap.clear();
    auto consider = [&](int32_t left) {
        if (left < 0) return;
        const int32_t right = symbols[static_cast<size_t>(left)].next;
        if (right < 0) return;
        const int32_t a = symbols[static_cast<size_t>(left)].id, b = symbols[static_cast<size_t>(right)].id;
        const auto it = merges_.find(merge_key(a, b));
        if (it == merges_.end()) return;
        heap.push_back({it->second.rank, left, a, b, it->second.id});
        std::push_heap(heap.begin(), heap.end(), later);
    };
    for (int32_t i = 0; i + 1 < static_cast<int32_t>(symbols.size()); ++i) consider(i);
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), later);
        const Candidate c = heap.back();
        heap.pop_back();
        Symbol& left = symbols[static_cast<size_t>(c.left)];
        if (left.id != c.left_id || left.next < 0) continue;  // stale
        Symbol& right = symbols[static_cast<size_t>(left.next)];
        if (right.id != c.right_id) continue;  // stale
        left.id = c.merged;
        right.id = -1;  // removed
        left.next = right.next;
        if (left.next >= 0) symbols[static_cast<size_t>(left.next)].prev = c.left;
        consider(left.prev);
        consider(c.left);
    }
    for (int32_t i = 0; i >= 0; i = symbols[static_cast<size_t>(i)].next) out.push_back(symbols[static_cast<size_t>(i)].id);
}

}  // namespace liyab
