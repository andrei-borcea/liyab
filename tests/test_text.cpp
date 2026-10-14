// Text processing: the regex engine, the pre-tokenizer table, NFC
// normalization and byte-level BPE encoding. Splits and normalizations are
// checked against tests/data/text_cases.txt (tools/gen_text_tests.py: Python's
// regex and unicodedata, cross-checked against the Hugging Face tokenizers).
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/pre_tokenizer.h"
#include "core/regex.h"
#include "core/tokenizer.h"
#include "core/unicode.h"
#include "liyab/mmap_loader.h"
#include "test_util.h"

using namespace liyab;

namespace {

// ---------------------------------------------------------------------------
// tests/data/text_cases.txt
// ---------------------------------------------------------------------------
std::string unescape(std::string_view s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out += s[i];
            continue;
        }
        if (s[i + 1] == '\\') {
            out += '\\';
            ++i;
        } else if (s[i + 1] == 'u' && i + 2 < s.size() && s[i + 2] == '{') {
            const size_t close = s.find('}', i);
            unicode::append_utf8(out, static_cast<uint32_t>(std::stoul(std::string(s.substr(i + 3, close - i - 3)), nullptr, 16)));
            i = close;
        } else {
            out += s[i];
        }
    }
    return out;
}

std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        const size_t tab = line.find('\t', start);
        out.push_back(line.substr(start, tab - start));
        if (tab == std::string::npos) return out;
        start = tab + 1;
    }
}

using Spans = std::vector<std::pair<size_t, size_t>>;

Spans parse_spans(const std::string& s) {
    Spans out;
    std::istringstream in(s);
    std::string token;
    size_t prev = 0;
    while (in >> token) {
        const size_t dash = token.find('-');
        const size_t a = dash == std::string::npos ? prev : std::stoul(token.substr(0, dash));
        const size_t b = std::stoul(dash == std::string::npos ? token : token.substr(dash + 1));
        out.emplace_back(a, b);
        prev = b;
    }
    return out;
}

struct Cases {
    std::unordered_map<std::string, std::string> texts, patterns;
    std::vector<std::vector<std::string>> rows;  // R, E, S and N records
    bool found = false;
};

// The cases file, from LIYAB_TEST_DATA on a device (the source tree elsewhere).
const Cases& cases() {
    static const Cases c = [] {
        Cases c;
        std::string dir = LIYAB_TEST_DATA_DIR;
        if (const char* env = std::getenv("LIYAB_TEST_DATA")) dir = env;
        std::ifstream in(dir + "/text_cases.txt");
        c.found = static_cast<bool>(in);
        if (!c.found) std::printf("  skipped: %s/text_cases.txt not found (set LIYAB_TEST_DATA)\n", dir.c_str());
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto f = fields(line);
            if (f[0] == "T") c.texts[f[1]] = unescape(f[2]);
            else if (f[0] == "P") c.patterns[f[1]] = unescape(f[2]);
            else c.rows.push_back(std::move(f));
        }
        return c;
    }();
    return c;
}

// Code points and keys of a text, for the regex engine.
struct Subject {
    std::vector<uint32_t> cps;
    std::vector<uint8_t> keys;
    explicit Subject(std::string_view text) {
        for (size_t i = 0; i < text.size();) {
            cps.push_back(unicode::decode_utf8(text, i));
            keys.push_back(regex_key(cps.back()));
        }
    }
    [[nodiscard]] RegexSubject view() const { return {cps.data(), keys.data(), cps.size()}; }
};

// The same splitting rule as PreTokenizer::split, for one pattern.
Spans split(const Regex& re, bool keep_gaps, const RegexSubject& s) {
    Spans out;
    size_t pos = 0, gap = 0;
    while (pos < s.size) {
        size_t begin = 0, end = 0;
        if (!re.search(s, pos, begin, end)) break;
        if (end == begin) {
            pos = begin + 1;
            continue;
        }
        if (begin > gap && keep_gaps) out.emplace_back(gap, begin);
        out.emplace_back(begin, end);
        pos = gap = end;
    }
    if (gap < s.size && keep_gaps) out.emplace_back(gap, s.size);
    return out;
}

std::string show(const Spans& spans) {
    std::string out;
    for (const auto& [a, b] : spans) out += std::to_string(a) + "-" + std::to_string(b) + " ";
    return out;
}

size_t codepoints(std::string_view text) {
    size_t n = 0;
    for (size_t i = 0; i < text.size(); ++n) (void)unicode::decode_utf8(text, i);
    return n;
}

// ---------------------------------------------------------------------------
// A vocabulary-only byte-level BPE GGUF, written on the fly.
// ---------------------------------------------------------------------------
class GgufWriter {
public:
    void string(const std::string& key, const std::string& value) {
        header(key, 8);
        str(value);
        ++count_;
    }
    void strings(const std::string& key, const std::vector<std::string>& values) {
        header(key, 9);
        u32(8);
        u64(values.size());
        for (const auto& v : values) str(v);
        ++count_;
    }
    void ints(const std::string& key, const std::vector<int32_t>& values) {
        header(key, 9);
        u32(5);
        u64(values.size());
        for (const int32_t v : values) u32(static_cast<uint32_t>(v));
        ++count_;
    }
    bool write(const std::string& path) const {
        std::string out = "GGUF";
        auto put = [&](uint64_t v, int bytes) {
            for (int i = 0; i < bytes; ++i) out += static_cast<char>((v >> (8 * i)) & 0xFF);
        };
        put(3, 4);
        put(0, 8);  // no tensors
        put(count_, 8);
        out += body_;
        while (out.size() % 32 != 0) out += '\0';
        std::ofstream f(path, std::ios::binary);
        f.write(out.data(), static_cast<std::streamsize>(out.size()));
        return static_cast<bool>(f);
    }

private:
    std::string body_;
    uint64_t count_ = 0;
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) body_ += static_cast<char>((v >> (8 * i)) & 0xFF);
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) body_ += static_cast<char>((v >> (8 * i)) & 0xFF);
    }
    void str(const std::string& s) {
        u64(s.size());
        body_ += s;
    }
    void header(const std::string& key, uint32_t type) {
        str(key);
        u32(type);
    }
};

// GPT-2's bytes_to_unicode(): the vocabulary text of each byte.
std::string byte_text(unsigned b) {
    int extra = 0;
    for (unsigned x = 0; x < 256; ++x) {
        const bool printable = (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255);
        const uint32_t cp = printable ? x : 256 + extra++;
        if (x == b) {
            std::string s;
            unicode::append_utf8(s, cp);
            return s;
        }
    }
    return {};
}

struct BpeVocab {
    std::vector<std::string> tokens;
    std::vector<int32_t> types;
    std::vector<std::string> merges;
    int32_t id(const std::string& t) const {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i] == t) return static_cast<int32_t>(i);
        }
        return -1;
    }
};

// 256 byte tokens (without "\r" when `drop_cr`), merges spelling "hello",
// " world", "é" and "\n\n", a token " xyz" no merge reaches, and two specials.
BpeVocab bpe_vocab(bool drop_cr) {
    BpeVocab v;
    for (unsigned b = 0; b < 256; ++b) {
        if (drop_cr && b == '\r') continue;
        v.tokens.push_back(byte_text(b));
        v.types.push_back(1);
    }
    const std::string G = byte_text(' ');  // "Ġ"
    const std::vector<std::pair<std::string, std::string>> merges = {
        {"h", "e"}, {"l", "l"}, {"he", "ll"}, {"hell", "o"}, {G, "w"}, {"o", "r"}, {G + "w", "or"},
        {G + "wor", "l"}, {G + "worl", "d"}, {byte_text(0xC3), byte_text(0xA9)}, {byte_text('\n'), byte_text('\n')}};
    for (const auto& [a, b] : merges) {
        v.merges.push_back(a + " " + b);
        v.tokens.push_back(a + b);
        v.types.push_back(1);
    }
    v.tokens.push_back(G + "xyz");
    v.types.push_back(1);
    for (const char* special : {"<|im_end|>", "<|user|>"}) {
        v.tokens.push_back(special);
        v.types.push_back(3);
    }
    return v;
}

std::string write_bpe(const std::string& name, const BpeVocab& v, const char* pre, const char* arch = "qwen2") {
    GgufWriter w;
    w.string("general.architecture", arch);
    w.string("tokenizer.ggml.model", "gpt2");
    if (pre != nullptr) w.string("tokenizer.ggml.pre", pre);
    w.strings("tokenizer.ggml.tokens", v.tokens);
    w.ints("tokenizer.ggml.token_type", v.types);
    w.strings("tokenizer.ggml.merges", v.merges);
    const std::string path = test::temp_dir() + "/liyab_text_" + name + ".gguf";
    return w.write(path) ? path : std::string();
}

// Encodes with a vocabulary-only GGUF written for the test (removed after).
struct BpeTokenizer {
    std::string path;
    std::unique_ptr<MmapLoader> file;
    std::optional<Tokenizer> tok;
    BpeTokenizer(const std::string& name, const BpeVocab& v, const char* pre, const char* arch = "qwen2")
        : path(write_bpe(name, v, pre, arch)) {
        auto f = MmapLoader::open(path);
        if (!f) return;
        file = std::move(f).value();
        auto t = Tokenizer::load(*file);
        if (t) tok.emplace(std::move(t).value());
    }
    ~BpeTokenizer() { std::remove(path.c_str()); }
    std::vector<int32_t> encode(const std::string& text) const {
        auto ids = tok->encode(text, false);
        return ids ? *ids : std::vector<int32_t>{-99};
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Regex engine
// ---------------------------------------------------------------------------
TEST_CASE("Regex splits match the reference engines (table patterns, edge corpus, engine features)") {
    if (!cases().found) return;
    const Cases& c = cases();
    size_t checked = 0;
    std::unordered_map<std::string, Regex> compiled;
    for (const auto& row : c.rows) {
        if (row[0] != "R") continue;
        const std::string& pattern = c.patterns.at(row[1]);
        auto it = compiled.find(pattern);
        if (it == compiled.end()) {
            auto re = Regex::compile(pattern);
            if (!re) {
                std::printf("  does not compile: %s\n", re.status().message().c_str());
                CHECK(false);
                continue;
            }
            it = compiled.emplace(pattern, std::move(re).value()).first;
        }
        const Subject s(c.texts.at(row[3]));
        const Spans got = split(it->second, row[2] == "1", s.view());
        const Spans want = parse_spans(row.size() > 4 ? row[4] : "");
        if (got != want) {
            std::printf("  pattern %s on text %s:\n    got  %s\n    want %s\n", row[1].c_str(), row[3].c_str(),
                        show(got).c_str(), show(want).c_str());
        }
        CHECK(got == want);
        ++checked;
    }
    CHECK(checked > 600);
}

TEST_CASE("Regex rejects unsupported syntax with a message naming the pattern") {
    if (!cases().found) return;
    size_t checked = 0;
    for (const auto& row : cases().rows) {
        if (row[0] != "E") continue;
        const std::string pattern = unescape(row[1]);
        auto re = Regex::compile(pattern);
        CHECK(!re.has_value());
        if (!re) CHECK(re.status().message().find(pattern) != std::string::npos);
        else std::printf("  compiled although unsupported: %s\n", pattern.c_str());
        ++checked;
    }
    CHECK(checked >= 10);
}

TEST_CASE("Regex match_at and search report positions in code points") {
    auto re = Regex::compile(R"(\p{L}+)");
    REQUIRE(re.has_value());
    const Subject s("12 h\xC3\xA9llo!");
    size_t begin = 0, end = 0;
    REQUIRE(re->search(s.view(), 0, begin, end));
    CHECK(begin == 3 && end == 8);
    CHECK(!re->match_at(s.view(), 0, end));
    CHECK(re->match_at(s.view(), 4, end) && end == 8);
    CHECK(!re->search(s.view(), 8, begin, end));
}

// ---------------------------------------------------------------------------
// Unicode
// ---------------------------------------------------------------------------
TEST_CASE("Unicode categories, white space and UTF-8 decoding") {
    using namespace unicode;
    CHECK(category('A') == Lu && category('a') == Ll && category('5') == Nd && category(' ') == Zs);
    CHECK(category(0x01C5) == Lt && category(0x02B0) == Lm && category(0x4E2D) == Lo);
    CHECK(category(0x0301) == Mn && category(0x0903) == Mc && category(0x20DD) == Me);
    CHECK(category(0x2165) == Nl && category(0x00BD) == No);
    CHECK(category(0x1F600) == So && category(0xE000) == Co && category(0x0378) == Cn && category(0xD800) == Cs);
    CHECK(category(0x110000) == Cn);
    for (const uint32_t cp : {0x09u, 0x0Au, 0x0Du, 0x20u, 0x85u, 0xA0u, 0x2028u, 0x3000u}) CHECK(is_whitespace(cp));
    for (const uint32_t cp : {0x1Cu, 0x1Fu, 0x180Eu, 0x200Bu, 0xFEFFu, 0x61u}) CHECK(!is_whitespace(cp));
    CHECK(ascii_fold(0x17F) == 's' && ascii_fold(0x212A) == 'k' && ascii_fold('Q') == 'q' && ascii_fold(0xE9) == 0);

    const std::string text = "a\xC3\xA9\xF0\x9F\x98\x80\xE2\x82";  // a, é, 😀, then a truncated sequence
    size_t i = 0;
    CHECK(decode_utf8(text, i) == 'a' && i == 1);
    CHECK(decode_utf8(text, i) == 0xE9 && i == 3);
    CHECK(decode_utf8(text, i) == 0x1F600 && i == 7);
    CHECK(decode_utf8(text, i) == 0xFFFD && i == 8);  // one byte at a time
    CHECK(decode_utf8(text, i) == 0xFFFD && i == 9);
}

TEST_CASE("NFC normalization matches Python's unicodedata") {
    if (!cases().found) return;
    size_t checked = 0;
    for (const auto& row : cases().rows) {
        if (row[0] != "N") continue;
        const std::string in = unescape(row[1]), want = unescape(row.size() > 2 ? row[2] : "");
        const std::string got = unicode::to_nfc(in);
        if (got != want) std::printf("  NFC(%s) differs\n", row[1].c_str());
        CHECK(got == want);
        if (unicode::nfc_quick_check(in)) CHECK(in == want);  // the quick check never skips a change
        ++checked;
    }
    CHECK(checked > 300);
    // Malformed bytes pass through untouched, between characters that compose.
    CHECK(unicode::to_nfc("e\xCC\x81\xFF" "e\xCC\x81") == "\xC3\xA9\xFF\xC3\xA9");
    CHECK(unicode::nfc_quick_check("plain ASCII, caf\xC3\xA9"));
}

// ---------------------------------------------------------------------------
// Pre-tokenizer table
// ---------------------------------------------------------------------------
TEST_CASE("Pre-tokenizer table: every row splits like the Hugging Face Split sequence") {
    if (!cases().found) return;
    const Cases& c = cases();
    size_t checked = 0;
    for (const auto& row : c.rows) {
        if (row[0] != "S") continue;
        auto pre = PreTokenizer::for_name(row[1]);
        REQUIRE(pre.has_value());
        const std::string& text = c.texts.at(row[2]);
        std::vector<std::string_view> pieces;
        pre->split(text, pieces);
        Spans got;
        for (const std::string_view p : pieces) {
            const size_t start = codepoints(std::string_view(text.data(), static_cast<size_t>(p.data() - text.data())));
            got.emplace_back(start, start + codepoints(p));
        }
        const Spans want = parse_spans(row.size() > 3 ? row[3] : "");
        if (got != want) {
            std::printf("  %s on text %s:\n    got  %s\n    want %s\n", row[1].c_str(), row[2].c_str(),
                        show(got).c_str(), show(want).c_str());
        }
        CHECK(got == want);
        ++checked;
    }
    CHECK(checked > 200);
}

TEST_CASE("Pre-tokenizer table: names, flags and the error for an unknown name") {
    const std::string names = PreTokenizer::supported_names();
    size_t count = 0;
    for (size_t start = 0; start < names.size();) {
        const size_t comma = names.find(", ", start);
        const std::string name = names.substr(start, comma - start);
        auto pre = PreTokenizer::for_name(name);
        CHECK(pre.has_value());
        ++count;
        start = comma == std::string::npos ? names.size() : comma + 2;
    }
    CHECK(count >= 14);
    for (const char* name : {"default", "gpt-2", "phi-2", "llama-bpe", "llama3", "qwen2", "deepseek-r1-qwen", "qwen35",
                             "hunyuan", "hunyuan-dense", "deepseek-v3", "deepseek-llm", "gpt-4o"}) {
        CHECK(names.find(name) != std::string::npos);
    }
    CHECK(PreTokenizer::for_name("qwen2")->nfc() && PreTokenizer::for_name("qwen35")->nfc());
    CHECK(!PreTokenizer::for_name("llama3")->nfc() && PreTokenizer::for_name("llama3")->ignore_merges());
    CHECK(PreTokenizer::for_name("hunyuan")->ignore_merges() && !PreTokenizer::for_name("qwen2")->ignore_merges());

    auto unknown = PreTokenizer::for_name("starcoder-9");
    REQUIRE(!unknown.has_value());
    CHECK(unknown.status().code() == ErrorCode::Unsupported);
    CHECK(unknown.status().message().find("'starcoder-9'") != std::string::npos);
    CHECK(unknown.status().message().find(names) != std::string::npos);
}

// ---------------------------------------------------------------------------
// Byte-level BPE tokenizer
// ---------------------------------------------------------------------------
TEST_CASE("Byte-level BPE: merges by rank, NFC per family, ignore_merges, specials") {
    const BpeVocab v = bpe_vocab(false);
    const std::string G = byte_text(' ');
    {
        BpeTokenizer t("qwen2", v, "qwen2");
        REQUIRE(t.tok.has_value());
        CHECK(t.encode("hello world") == (std::vector<int32_t>{v.id("hello"), v.id(G + "world")}));
        // Decomposed "é" is composed first (Qwen's tokenizer.json has an NFC normalizer).
        CHECK(t.encode("e\xCC\x81") == (std::vector<int32_t>{v.id(byte_text(0xC3) + byte_text(0xA9))}));
        // " xyz" is a token, but no merge reaches it: three letters after the space byte.
        CHECK(t.encode(" xyz").size() == 4);
        // Specials split the text and map to their ids; the longest one at a position wins.
        CHECK(t.encode("hello<|im_end|>hello") ==
              (std::vector<int32_t>{v.id("hello"), v.id("<|im_end|>"), v.id("hello")}));
        // Digits are single pieces (\p{N}), and every byte of the text is in some token.
        std::string decoded;
        for (const int32_t id : t.encode("a1234 \xF0\x9F\x98\x80\r\n")) decoded += t.tok->piece(id);
        CHECK(decoded == "a1234 \xF0\x9F\x98\x80\r\n");
    }
    {
        BpeTokenizer t("hunyuan", v, "hunyuan");  // the Qwen2 pattern, no NFC, ignore_merges
        REQUIRE(t.tok.has_value());
        CHECK(t.encode("e\xCC\x81").size() == 3);
        CHECK(t.encode(" xyz") == (std::vector<int32_t>{v.id(G + "xyz")}));
    }
    {
        BpeTokenizer t("default", v, nullptr);  // no tokenizer.ggml.pre: GPT-2's scheme
        REQUIRE(t.tok.has_value());
        CHECK(t.encode("hello world") == (std::vector<int32_t>{v.id("hello"), v.id(G + "world")}));
    }
}

TEST_CASE("Byte-level BPE: a byte the vocabulary lacks is left out before merging") {
    const BpeVocab v = bpe_vocab(true);  // no "\r"
    BpeTokenizer t("no_cr", v, "qwen2");
    REQUIRE(t.tok.has_value());
    // "\r\n\r\n" is one piece; without "\r" its newlines merge into one token.
    const std::string nl = byte_text('\n');
    CHECK(t.encode("\r\n\r\n") == (std::vector<int32_t>{v.id(nl + nl)}));
    const BpeVocab full = bpe_vocab(false);
    BpeTokenizer with_cr("with_cr", full, "qwen2");
    REQUIRE(with_cr.tok.has_value());
    CHECK(with_cr.encode("\r\n\r\n").size() == 4);
}

TEST_CASE("Byte-level BPE: Phi-3 specials swallow the white space after them") {
    const BpeVocab v = bpe_vocab(false);
    BpeTokenizer phi("phi3", v, "gpt-4o", "phi3");
    REQUIRE(phi.tok.has_value());
    CHECK(phi.encode("<|user|>\n hello") == (std::vector<int32_t>{v.id("<|user|>"), v.id("hello")}));
    BpeTokenizer qwen("not_phi3", v, "gpt-4o", "qwen2");
    REQUIRE(qwen.tok.has_value());
    CHECK(qwen.encode("<|user|>\n hello").size() > 2);
}

TEST_CASE("Byte-level BPE: an unknown pre-tokenizer loads but cannot encode, naming the supported ones") {
    const BpeVocab v = bpe_vocab(false);
    BpeTokenizer t("unknown_pre", v, "made-up-pre");
    REQUIRE(t.tok.has_value());
    auto ids = t.tok->encode("hello", false);
    REQUIRE(!ids.has_value());
    CHECK(ids.status().code() == ErrorCode::Unsupported);
    CHECK(ids.status().message().find("'made-up-pre'") != std::string::npos);
    CHECK(ids.status().message().find("qwen35") != std::string::npos);
}

TEST_MAIN()
