#include "core/pre_tokenizer.h"

#include <array>
#include <initializer_list>

#include "core/unicode.h"

namespace liyab {

namespace {

// ---------------------------------------------------------------------------
// The table. Patterns are copied from each family's tokenizer.json
// (pre_tokenizer: Split / ByteLevel / Digits) as raw strings. Non-ASCII
// characters are written \x{HEX}: several (U+1FD3, U+212A...) have canonical
// equivalents, and an editor normalizing the file would silently change
// the pattern. "\r" and "\n" stand for the literal CR / LF some files carry;
// the Digits pre-tokenizer (one piece per numeric character) is \p{N}.
// ---------------------------------------------------------------------------

// GPT-2 (Hugging Face ByteLevel with use_regex): also Phi-2.
constexpr const char* kGpt2 =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
// BLOOM's tokenizer.json Split, with its nested character class flattened
// to the equivalent excluded set (cross-checked against Oniguruma).
constexpr const char* kBloom = R"( ?[^\s.,!?\x{2026}\x{3002}\x{FF0C}\x{3001}\x{0964}\x{06D4}\x{060C}()|]+)";
// Llama 3: case-insensitive contractions, digits in groups of up to three.
constexpr const char* kLlama3 =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// Qwen2 / Qwen2.5 / Qwen3, Hunyuan (MT, A13B): single digits.
constexpr const char* kQwen2 =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// Qwen3.5 / Qwen3.6: combining marks belong to words.
constexpr const char* kQwen35 =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// GPT-4o (o200k; Phi-4-mini): upper/lower case aware words with contractions attached.
constexpr const char* kGpt4o =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// DeepSeek V3 / R1 and Hunyuan dense (0.5B..7B): digits, CJK runs, then words and symbols.
constexpr const char* kDigits3 = R"(\p{N}{1,3})";
constexpr const char* kHanKana = R"([\x{4E00}-\x{9FA5}\x{3040}-\x{309F}\x{30A0}-\x{30FF}]+)";
constexpr const char* kDeepSeekV3Words =
    R"([!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// DeepSeek LLM (7B / 67B): newlines, Latin/Greek/Cyrillic-like letters, ASCII and full-width punctuation,
// trailing white space, CJK, then every digit on its own (the Digits pre-tokenizer).
constexpr const char* kNewline = R"([\r\n])";
constexpr const char* kDeepSeekLetters =
    R"(\s?[A-Za-z\x{B5}\x{C0}-\x{D6}\x{D8}-\x{F6}\x{F8}-\x{1BA}\x{1BC}-\x{1BF}\x{1C4}-\x{293}\x{295}-\x{2AF}\x{370}-\x{373}\x{376}\x{377}\x{37B}-\x{37D}\x{37F}\x{386}\x{388}-\x{38A}\x{38C}\x{38E}-\x{3A1}\x{3A3}-\x{3F5}\x{3F7}-\x{481}\x{48A}-\x{52F}\x{531}-\x{556}\x{10A0}-\x{10C5}\x{13A0}-\x{13F5}\x{13F8}-\x{13FD}\x{1C90}-\x{1CBA}\x{1CBD}-\x{1CBF}\x{1D00}-\x{1D2B}\x{1D6B}-\x{1D77}\x{1D79}-\x{1D9A}\x{1E00}-\x{1F15}\x{1F18}-\x{1F1D}\x{1F20}-\x{1F45}\x{1F48}-\x{1F4D}\x{1F50}-\x{1F57}\x{1F59}\x{1F5B}\x{1F5D}\x{1F5F}-\x{1F7D}\x{1F80}-\x{1FB4}\x{1FB6}-\x{1FBC}\x{1FBE}\x{1FC2}-\x{1FC4}\x{1FC6}-\x{1FCC}\x{1FD0}-\x{1FD3}\x{1FD6}-\x{1FDB}\x{1FE0}-\x{1FEC}\x{1FF2}-\x{1FF4}\x{1FF6}-\x{1FFC}\x{2102}\x{2107}\x{210A}-\x{2113}\x{2115}\x{2119}-\x{211D}\x{2124}\x{2126}\x{2128}\x{212A}-\x{212D}\x{212F}-\x{2134}\x{2139}\x{213C}-\x{213F}\x{2145}-\x{2149}\x{214E}\x{2183}\x{2184}\x{2C00}-\x{2C7B}\x{2C7E}-\x{2CE4}\x{2CEB}-\x{2CEE}\x{2CF2}\x{2CF3}\x{A640}-\x{A66D}\x{A680}-\x{A69B}\x{A722}-\x{A76F}\x{A771}-\x{A787}\x{A78B}-\x{A78E}\x{AB70}-\x{ABBF}\x{FB00}-\x{FB06}\x{FB13}-\x{FB17}\x{FF21}-\x{FF3A}\x{FF41}-\x{FF5A}\x{10400}-\x{1044F}\x{104B0}-\x{104D3}\x{104D8}-\x{104FB}\x{10C80}-\x{10CB2}\x{10CC0}-\x{10CF2}\x{118A0}-\x{118DF}\x{1E900}-\x{1E943}]+)";
constexpr const char* kDeepSeekPunct = R"(\s?[!-/:-~\x{FF01}-\x{FF0F}\x{FF1A}-\x{FF5E}\x{2018}-\x{201F}\x{3000}-\x{3002}]+)";
constexpr const char* kTrailingSpace = R"(\s+$)";
constexpr const char* kDeepSeekCjk = R"([\x{4E00}-\x{9FA5}\x{800}-\x{4E00}\x{AC00}-\x{D7FF}]+)";
constexpr const char* kEachDigit = R"(\p{N})";
// Mistral-Nemo and later Mistral models (Tekken): the GPT-4o words without the contractions, single digits.
constexpr const char* kTekken =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// Youtu-LLM: Hangul, CJK punctuation, Bopomofo and Han/kana runs first, then the GPT-4o words with single
// digits. The file writes the first class as \p{Hangul}; script properties are not part of core/regex.h, so it
// is spelled out as the Hangul script's ranges (Unicode 16 Scripts.txt: the set Oniguruma matches, checked code
// point by code point).
constexpr const char* kYoutuCjk =
    R"([\x{1100}-\x{11FF}\x{302E}\x{302F}\x{3131}-\x{318E}\x{3200}-\x{321E}\x{3260}-\x{327E}\x{A960}-\x{A97C}\x{AC00}-\x{D7A3}\x{D7B0}-\x{D7C6}\x{D7CB}-\x{D7FB}\x{FFA0}-\x{FFBE}\x{FFC2}-\x{FFC7}\x{FFCA}-\x{FFCF}\x{FFD2}-\x{FFD7}\x{FFDA}-\x{FFDC}]+|[\x{FF01}\x{2026}\x{201C}\x{201D}\x{2018}\x{2019}\x{2014}\x{FF1A}\x{FF1B}\x{FF0C}\x{3001}-\x{303F}\x{FE30}-\x{FE4F}]+|[\x{3105}-\x{312F}]+|[\x{4E00}-\x{9FA5}\x{3040}-\x{309F}\x{30A0}-\x{30FF}]+)";
constexpr const char* kYoutuWords =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

struct StepSpec {
    const char* pattern;
    bool keep_gaps;  // Split "Isolated"; false: "Removed" with invert (only the matches are kept)
};

struct Row {
    std::initializer_list<std::string_view> names;  // tokenizer.ggml.pre values
    std::initializer_list<StepSpec> steps;
    bool nfc;
    bool ignore_merges;
};

const std::array<Row, 15>& table() {
    static const std::array<Row, 15> rows = {{
        {{"default", "gpt-2", "gpt2", "phi-2"}, {{kGpt2, true}}, false, false},
        {{"llama-bpe", "llama3"}, {{kLlama3, true}}, false, true},
        {{"qwen2", "deepseek-r1-qwen"}, {{kQwen2, true}}, true, false},
        {{"qwen35"}, {{kQwen35, true}}, true, false},
        {{"hunyuan"}, {{kQwen2, true}}, false, true},
        {{"deepseek-v3", "hunyuan-dense"}, {{kDigits3, true}, {kHanKana, true}, {kDeepSeekV3Words, true}}, false, false},
        {{"deepseek-llm"},
         {{kNewline, true}, {kDeepSeekLetters, true}, {kDeepSeekPunct, true}, {kTrailingSpace, true},
          {kDeepSeekCjk, true}, {kEachDigit, true}},
         false, false},
        {{"gpt-4o"}, {{kGpt4o, false}}, false, false},
        // SmolLM / SmolLM2: the Digits pre-tokenizer (each digit alone), then GPT-2's ByteLevel split.
        {{"smollm"}, {{kEachDigit, true}, {kGpt2, true}}, false, false},
        {{"tekken"}, {{kTekken, true}}, false, true},
        {{"youtu"}, {{kYoutuCjk, true}, {kYoutuWords, true}}, false, false},
        {{"bloom"}, {{kBloom, true}}, false, false},
        {{"starcoder", "refact"}, {{kEachDigit, true}, {kGpt2, true}}, false, false},
        {{"gpt-neox", "mpt", "olmo"}, {{kGpt2, true}}, false, false},
        {{"stablelm2"}, {{kQwen2, false}}, false, false},
    }};
    return rows;
}

// Code points and class keys of a text, with each code point's byte offset.
struct Decoded {
    std::vector<uint32_t> cps;
    std::vector<uint8_t> keys;
    std::vector<uint32_t> offsets;  // size + 1 entries
};

}  // namespace

std::string PreTokenizer::supported_names() {
    std::string out;
    for (const Row& row : table()) {
        for (const std::string_view name : row.names) {
            if (!out.empty()) out += ", ";
            out += name;
        }
    }
    return out;
}

Result<PreTokenizer> PreTokenizer::for_name(std::string_view name) {
    for (const Row& row : table()) {
        bool found = false;
        for (const std::string_view n : row.names) found = found || n == name;
        if (!found) continue;
        PreTokenizer pre;
        pre.name_ = std::string(name);
        pre.nfc_ = row.nfc;
        pre.ignore_merges_ = row.ignore_merges;
        for (const StepSpec& step : row.steps) {
            auto regex = Regex::compile(step.pattern);
            if (!regex) return Status(ErrorCode::Internal, "pre-tokenizer '" + pre.name_ + "': " + regex.status().message());
            pre.steps_.push_back({std::move(regex).value(), step.keep_gaps});
        }
        return pre;
    }
    return Status(ErrorCode::Unsupported, "BPE pre-tokenizer '" + std::string(name) +
                                              "' (tokenizer.ggml.pre) is not supported; supported: " + supported_names());
}

void PreTokenizer::split(std::string_view text, std::vector<std::string_view>& pieces) const {
    // Decoded once per call; thread-local buffers keep repeated encodes from allocating.
    thread_local Decoded d;
    d.cps.clear();
    d.keys.clear();
    d.offsets.clear();
    for (size_t i = 0; i < text.size();) {
        d.offsets.push_back(static_cast<uint32_t>(i));
        const uint32_t cp = unicode::decode_utf8(text, i);
        d.cps.push_back(cp);
        d.keys.push_back(regex_key(cp));
    }
    d.offsets.push_back(static_cast<uint32_t>(text.size()));

    // Pieces as code-point ranges; each step splits every piece of the one before.
    thread_local std::vector<std::pair<uint32_t, uint32_t>> current, next;
    current.assign(1, {0u, static_cast<uint32_t>(d.cps.size())});
    if (d.cps.empty()) current.clear();
    for (const Step& step : steps_) {
        next.clear();
        for (const auto& [from, to] : current) {
            // A piece is matched on its own: anchors and lookaheads see its ends.
            const RegexSubject s{d.cps.data() + from, d.keys.data() + from, to - from};
            size_t pos = 0, gap = 0;
            while (pos < s.size) {
                size_t begin = 0, end = 0;
                if (!step.regex.search(s, pos, begin, end)) break;
                if (end == begin) {  // empty matches cut nothing
                    pos = begin + 1;
                    continue;
                }
                if (begin > gap && step.keep_gaps) next.emplace_back(from + gap, from + begin);
                next.emplace_back(from + begin, from + end);
                pos = gap = end;
            }
            if (gap < s.size && step.keep_gaps) next.emplace_back(from + gap, to);
        }
        std::swap(current, next);
    }
    for (const auto& [from, to] : current) {
        pieces.push_back(text.substr(d.offsets[from], d.offsets[to] - d.offsets[from]));
    }
}

}  // namespace liyab
