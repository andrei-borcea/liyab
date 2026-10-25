// Text processing: the regex engine, the pre-tokenizer table, NFC
// normalization, byte-level BPE encoding and chat templates. Splits and normalizations are
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

#include "core/chat_template.h"
#include "core/pre_tokenizer.h"
#include "core/regex.h"
#include "core/tokenizer.h"
#include "core/unicode.h"
#include "liyab/engine.h"
#include "liyab/liyab_c_api.h"
#include "liyab/mmap_loader.h"
#include "test_model.h"
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

TEST_CASE("The converter's joyai-llm alias uses DeepSeek-V3 splitting") {
    auto original = PreTokenizer::for_name("deepseek-v3");
    auto alias = PreTokenizer::for_name("joyai-llm");
    REQUIRE(original.has_value() && alias.has_value());
    for (const std::string text : {"Hello 1234567!\n", "你好 世界 カナ test's", "a\r\nb   c"}) {
        std::vector<std::string_view> a, b;
        original->split(text, a);
        alias->split(text, b);
        CHECK(a == b);
    }
    CHECK(original->nfc() == alias->nfc());
    CHECK(original->ignore_merges() == alias->ignore_merges());
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

TEST_CASE("GLM uses Llama-3 splitting and merge rules") {
    auto original = PreTokenizer::for_name("llama3");
    auto glm = PreTokenizer::for_name("glm4");
    REQUIRE(original.has_value() && glm.has_value());
    for (const std::string text : {"Hello 1234567!\n", "你好 世界 カナ test's", "a\r\nb   c"}) {
        std::vector<std::string_view> a, b;
        original->split(text, a);
        glm->split(text, b);
        CHECK(a == b);
    }
    CHECK(!glm->nfc() && glm->ignore_merges());
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

// ---------------------------------------------------------------------------
// Chat templates
// ---------------------------------------------------------------------------
namespace {

// tests/data/chat_templates.json (tools/chat_template_reference.py).
std::optional<jinja::Value> chat_cases() {
    std::string dir = LIYAB_TEST_DATA_DIR;
    if (const char* env = std::getenv("LIYAB_TEST_DATA")) dir = env;
    std::ifstream in(dir + "/chat_templates.json", std::ios::binary);
    if (!in) {
        std::printf("  skipped: %s/chat_templates.json not found (set LIYAB_TEST_DATA)\n", dir.c_str());
        return std::nullopt;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    auto v = jinja::parse_json(ss.str());
    if (!v) {
        std::printf("  %s\n", v.status().message().c_str());
        return std::nullopt;
    }
    return std::move(v).value();
}

std::string first_difference(const std::string& got, const std::string& want) {
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
    const size_t from = i > 40 ? i - 40 : 0;
    return "at byte " + std::to_string(i) + ":\n    got  " + jinja::Value(got.substr(from, 100)).repr() +
           "\n    want " + jinja::Value(want.substr(from, 100)).repr();
}

}  // namespace

TEST_CASE("Chat templates render byte for byte like transformers' apply_chat_template") {
    const auto data = chat_cases();
    if (!data) return;
    size_t rendered = 0, refused = 0;
    for (const auto& [name, entry] : data->fields().entries) {
        auto tmpl = ChatTemplate::parse(entry.fields().find("template")->as_string());
        if (!tmpl) std::printf("  %s: %s\n", name.c_str(), tmpl.status().message().c_str());
        REQUIRE(tmpl.has_value());
        const jinja::Value& special = *entry.fields().find("special");
        size_t index = 0;
        for (const jinja::Value& c : entry.fields().find("cases")->items()) {
            const std::string request = c.fields().find("request")->to_json();
            auto out = tmpl->render_chat(request, special);
            if (const jinja::Value* want = c.fields().find("output")) {
                if (!out) std::printf("  %s case %zu: %s\n", name.c_str(), index, out.status().message().c_str());
                else if (*out != want->as_string()) {
                    std::printf("  %s case %zu differs %s\n", name.c_str(), index, first_difference(*out, want->as_string()).c_str());
                }
                CHECK(out.has_value() && *out == want->as_string());
                ++rendered;
            } else {
                if (out) std::printf("  %s case %zu: rendered, but transformers refused it\n", name.c_str(), index);
                CHECK(!out.has_value());
                ++refused;
            }
            ++index;
        }
    }
    CHECK(rendered > 100);
    CHECK(refused >= 1);
}

TEST_CASE("Chat templates: prefix stability and the layout of turns") {
    const auto data = chat_cases();
    if (!data) return;
    auto format = [&](const char* name, const char* probe) {
        const jinja::Value& entry = *data->fields().find(name);
        auto tmpl = ChatTemplate::parse(entry.fields().find("template")->as_string());
        auto f = tmpl->format(probe, *entry.fields().find("special"));
        if (!f) std::printf("  %s: %s\n", name, f.status().message().c_str());
        return f.has_value() ? *f : ChatFormat{};
    };
    // Qwen3.6 with the reasoning of past replies kept: exactly the ChatML
    // pieces the app has always written (its saved contexts stay valid).
    const ChatFormat q36 = format("qwen3.6", R"({"enable_thinking": false, "preserve_thinking": true})");
    CHECK(q36.prefix_stable);
    CHECK(q36.system_prefix == "<|im_start|>system\n" && q36.system_suffix == "<|im_end|>\n");
    CHECK(q36.first_user_prefix == "<|im_start|>user\n" && q36.user_prefix == "<|im_start|>user\n");
    CHECK(q36.no_system_head == "<|im_start|>user\n");
    CHECK(q36.user_suffix == "<|im_end|>\n" && q36.assistant_end == "<|im_end|>\n");
    CHECK(q36.generation_prompt == "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    const ChatFormat q36_on =
        format("qwen3.6", R"({"enable_thinking": true, "preserve_thinking": true, "reply_prefix": "Reasoning.\n</think>\n\n"})");
    CHECK(q36_on.prefix_stable && q36_on.generation_prompt == "<|im_start|>assistant\n<think>\n");
    // Qwen3 and Qwen3.5 drop the reasoning of earlier replies: not prefix stable.
    const ChatFormat q35 = format("qwen3.5", R"({"enable_thinking": false})");
    CHECK(!q35.prefix_stable && !q35.reason.empty());
    CHECK(!format("qwen3", R"({"enable_thinking": false})").prefix_stable);
    // Llama 3: the date lines and the user header belong to the first turn.
    const ChatFormat llama = format("llama3.2", R"({"date_string": "26 Jul 2024"})");
    CHECK(llama.prefix_stable);
    CHECK(llama.system_prefix == "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: "
                                 "December 2023\nToday Date: 26 Jul 2024\n\n");
    CHECK(llama.generation_prompt == "<|start_header_id|>assistant<|end_header_id|>\n\n" && llama.assistant_end == "<|eot_id|>");
    // Gemma 3: the system text opens the first user turn.
    const ChatFormat gemma = format("gemma3", "");
    CHECK(gemma.prefix_stable);
    CHECK(gemma.system_prefix == "<bos><start_of_turn>user\n" && gemma.system_suffix == "\n\n" && gemma.first_user_prefix.empty());
    CHECK(gemma.no_system_head == "<bos><start_of_turn>user\n" && gemma.generation_prompt == "<start_of_turn>model\n");
    for (const char* name : {"qwen2.5", "phi3.5", "phi4-mini", "hunyuan-mt"}) {
        const ChatFormat f = format(name, "");
        if (!f.prefix_stable) std::printf("  %s: %s\n", name, f.reason.c_str());
        CHECK(f.prefix_stable);
    }
}

TEST_CASE("Chat templates: Jinja semantics transformers relies on") {
    auto render = [](const char* source, const char* json = "{}") {
        auto t = ChatTemplate::parse(source);
        if (!t) return "PARSE: " + t.status().message();
        auto ctx = jinja::parse_json(json);
        auto out = t->render(*ctx);
        return out ? *out : "ERROR: " + out.status().message();
    };
    // trim_blocks and lstrip_blocks, '-' markers, the source's trailing newline dropped.
    CHECK(render("a\n  {% if true %}\n  x\n  {% endif %}\nb\n") == "a\n  x\nb");
    CHECK(render("{%- if true -%}  a  {%- endif -%}  b {#- c -#}  d") == "abd");
    // A variable set in a loop lives for one iteration; in an if, it leaks.
    CHECK(render(R"({% for i in range(3) %}{% if i == 0 %}{% set x = "a" %}{% endif %}[{{ x }}]{% endfor %}<{{ x }}>)") ==
          "[a][][]<>");
    CHECK(render("{% set x = 1 %}{% if true %}{% set x = 2 %}{% endif %}{{ x }}") == "2");
    CHECK(render("{% set ns = namespace(a=1) %}{% for i in [1, 2] %}{% set ns.a = ns.a + i %}{% endfor %}{{ ns.a }}") == "4");
    // Python's printing and arithmetic.
    CHECK(render(R"({{ none }}|{{ true }}|{{ 1.0 }}|{{ [1, "a", none] }}|{{ {"a": 1} }}|{{ 1/2 }}|{{ -7//2 }}|{{ -7 % 3 }}|{{ 1e20 }}|{{ 1e-05 }})") ==
          "None|True|1.0|[1, 'a', None]|{'a': 1}|0.5|-4|2|1e+20|1e-05");
    CHECK(render(R"({{ "abc"[::-1] }}{{ [1,2,3][-1] }}{{ "a b  c".split() }}{{ " x ".strip() }}{{ "a,b,c".split(",", 1) }})") ==
          "cba3['a', 'b', 'c']x['a', 'b,c']");
    // transformers' tojson: keys in order, non-ASCII kept, ", " / ": " separators.
    CHECK(render(R"({{ {"b": 1, "a": [1, 2.5, "é\n"]}|tojson }}|{{ {"a": {}}|tojson(indent=2) }})") ==
          "{\"b\": 1, \"a\": [1, 2.5, \"\xC3\xA9\\n\"]}|{\n  \"a\": {}\n}");
    // Macros with defaults and keyword arguments; loop variables; filters in expressions.
    CHECK(render("{% macro m(a, b=2) %}{{a}}{{b}}{% endmacro %}{{ m(1) }}{{ m(1, b=3) }}{{ m(b=5, a=4) }}") == "121345");
    CHECK(render("{% for x in [1,2,3] if x > 1 %}{{ loop.index }}{{ x }}{{ loop.length }}{% else %}E{% endfor %}") == "122232");
    CHECK(render("{% for m in [1,2,3] %}{{ loop.previtem }}/{{ loop.nextitem }};{% endfor %}") == "/2;1/3;2/;");
    CHECK(render(R"({{ 'a' + ' b ' | trim + 'c' }})") == "abc");
    // Errors name what failed: raise_exception, undefined access, sandboxed mutation.
    CHECK(render(R"({{ raise_exception("No user query found.") }})") == "ERROR: chat template: No user query found.");
    CHECK(render("{{ x.y }}").starts_with("ERROR: chat template: 'x' is undefined"));
    CHECK(render(R"({% set d = {} %}{% set _ = d.update({"a": 1}) %})").find("unsafe") != std::string::npos);
    CHECK(render("{% while true %}{% endwhile %}").starts_with("PARSE: chat template: unsupported statement"));
    CHECK(render(R"({{ "a" + 1 }})").starts_with("ERROR: chat template: unsupported operand"));
}

namespace {

// Appends string key/values to a GGUF's metadata in place. The inserted bytes
// are a multiple of 32 (a padding key absorbs the rest), so the tensor data
// keeps its alignment and offsets.
bool add_metadata(const std::string& path, const std::vector<std::pair<std::string, std::string>>& kvs) {
    std::ifstream in(path, std::ios::binary);
    std::string file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (file.size() < 24 || file.compare(0, 4, "GGUF") != 0) return false;
    auto u32 = [&](size_t at) { uint32_t v; std::memcpy(&v, file.data() + at, 4); return v; };
    auto u64 = [&](size_t at) { uint64_t v; std::memcpy(&v, file.data() + at, 8); return v; };
    const uint64_t n_kv = u64(16);
    size_t pos = 24;
    auto skip_value = [&](auto& self, uint32_t type) -> void {
        static const size_t sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (type == 8) {
            pos += 8 + u64(pos);
        } else if (type == 9) {
            const uint32_t et = u32(pos);
            const uint64_t count = u64(pos + 4);
            pos += 12;
            for (uint64_t i = 0; i < count; ++i) self(self, et);
        } else {
            pos += sizes[type];
        }
    };
    for (uint64_t i = 0; i < n_kv; ++i) {
        pos += 8 + u64(pos);  // key
        const uint32_t type = u32(pos);
        pos += 4;
        skip_value(skip_value, type);
    }
    std::string added;
    auto put_kv = [&](const std::string& key, const std::string& value) {
        auto put64 = [&](uint64_t v) { added.append(reinterpret_cast<const char*>(&v), 8); };
        put64(key.size());
        added += key;
        const uint32_t type = 8;
        added.append(reinterpret_cast<const char*>(&type), 4);
        put64(value.size());
        added += value;
    };
    for (const auto& [k, v] : kvs) put_kv(k, v);
    const std::string pad_key = "liyab.test.padding";
    const size_t base = added.size() + 8 + pad_key.size() + 4 + 8;
    put_kv(pad_key, std::string((32 - base % 32) % 32, ' '));
    file.insert(pos, added);
    const uint64_t total = n_kv + kvs.size() + 1;
    std::memcpy(file.data() + 16, &total, 8);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(file.data(), static_cast<std::streamsize>(file.size()));
    return static_cast<bool>(out);
}

constexpr const char* kTinyTemplate =
    "{{ bos_token }}{% for m in messages %}{% if m.role == 'tool' %}{{ raise_exception('no tools here') }}{% endif %}"
    "<|{{ m.role }}|>{{ m.content }}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";

}  // namespace

TEST_CASE("Engine and C API: the model's chat template renders, describes its turns, and keeps one BOS") {
    const std::string path = test::temp_dir() + "/liyab_text_templated.gguf";
    test::write_tiny_model(path);
    REQUIRE(add_metadata(path, {{"tokenizer.chat_template", kTinyTemplate}}));
    EngineConfig config;
    config.model_path = path;
    config.backend = BackendKind::Cpu;
    config.thermal_polling = false;
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    Engine& e = **engine;
    CHECK(e.has_chat_template());
    auto text = e.render_chat(R"({"messages": [{"role": "system", "content": "Be brief."},
                                               {"role": "user", "content": "hello"}], "add_generation_prompt": true})");
    REQUIRE(text.has_value());
    CHECK(*text == "<s><|system|>Be brief.</s><|user|>hello</s><|assistant|>");
    // The template wrote the BOS: tokenizing with add_bos adds no second one.
    auto ids = e.tokenize(*text, true);
    REQUIRE(ids.has_value());
    CHECK(ids->size() > 2 && (*ids)[0] == 1 && (*ids)[1] != 1);
    CHECK(e.tokenize("hello", true).value() == e.tokenize("<s>hello", true).value());

    auto format = e.chat_format("");
    REQUIRE(format.has_value());
    CHECK(format->prefix_stable);
    CHECK(format->system_prefix == "<s><|system|>" && format->system_suffix == "</s>");
    CHECK(format->first_user_prefix == "<|user|>" && format->user_prefix == "<|user|>" && format->user_suffix == "</s>");
    CHECK(format->no_system_head == "<s><|user|>");
    CHECK(format->generation_prompt == "<|assistant|>" && format->assistant_end == "</s>");

    auto refused = e.render_chat(R"({"messages": [{"role": "tool", "content": "x"}]})");
    CHECK(!refused.has_value() && refused.status().message().find("no tools here") != std::string::npos);
    CHECK(!e.render_chat("{not json").has_value());

    // The C ABI: the length needed, the text, -1 and liyab_last_error on failure.
    liyab_engine_config c;
    liyab_engine_config_default(&c);
    c.model_path = path.c_str();
    c.backend = LIYAB_BACKEND_CPU;
    c.thermal_polling = 0;
    liyab_engine* handle = nullptr;
    REQUIRE(liyab_engine_create(&c, &handle) == LIYAB_OK);
    const char* request = R"({"messages": [{"role": "user", "content": "hi"}], "add_generation_prompt": true})";
    const int64_t need = liyab_engine_render_chat(handle, request, nullptr, 0);
    CHECK(need == static_cast<int64_t>(std::strlen("<s><|user|>hi</s><|assistant|>")));
    std::vector<char> buffer(static_cast<size_t>(need) + 1);
    CHECK(liyab_engine_render_chat(handle, request, buffer.data(), buffer.size()) == need);
    CHECK(std::string(buffer.data()) == "<s><|user|>hi</s><|assistant|>");
    CHECK(liyab_engine_render_chat(handle, "[1", buffer.data(), buffer.size()) == -1);
    CHECK(std::string(liyab_last_error()).find("JSON") != std::string::npos);
    char json[2048];
    CHECK(liyab_engine_chat_format(handle, nullptr, json, sizeof json) > 0);
    auto parsed = jinja::parse_json(json);
    REQUIRE(parsed.has_value());
    CHECK(parsed->fields().find("prefix_stable")->as_bool());
    CHECK(parsed->fields().find("generation_prompt")->as_string() == "<|assistant|>");
    liyab_engine_destroy(handle);
    engine->reset();
    std::remove(path.c_str());

    // Without a template: Unsupported, and the C ABI says so.
    const std::string plain = test::temp_dir() + "/liyab_text_plain.gguf";
    test::write_tiny_model(plain);
    config.model_path = plain;
    auto bare = Engine::create(config);
    REQUIRE(bare.has_value());
    CHECK(!(*bare)->has_chat_template());
    auto none = (*bare)->render_chat(R"({"messages": []})");
    CHECK(!none.has_value() && none.status().code() == ErrorCode::Unsupported);
    bare->reset();
    std::remove(plain.c_str());
}

TEST_MAIN()
