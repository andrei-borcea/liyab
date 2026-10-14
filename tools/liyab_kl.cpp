// liyab-kl — teacher-forced next-token distributions of a model over a text,
// compared against a reference. Two uses:
//
// - What a lossy option costs in quality: the reference is saved once from
//   the lossless path and compared against a lossy one (e.g. --requant 4).
//
//     liyab-kl -m model.gguf -f text.txt --save ref.bin               # lossless reference
//     liyab-kl -m model.gguf -f text.txt --compare ref.bin --requant 4
//
// - Conformance of an architecture, tokenizer and weights against another
//   implementation: tools/conform_reference.py writes the same reference
//   from Hugging Face transformers, and --max-kl / --min-top1 turn the
//   comparison into a pass/fail gate (exit status 1 on failure).
//
//     liyab-kl -m model.gguf -f text.txt --compare hf.ref --max-kl 0.01 --min-top1 97
//
// The reference holds the text's token ids, then for every position the
// log-probabilities of its top 32 tokens and the log-probability of the
// text's actual next token. Comparing first checks that this build encodes
// the text into the same ids (the tokenizer's conformance; the first
// difference is printed), then prints the KL divergence KL(reference || this
// run), computed over those 32 tokens plus one bucket for the rest (a lower
// bound of the full-vocabulary KL that is tight when the top 32 carry nearly
// all the mass), the top-1 agreement and both perplexities. --ref-tokens
// teacher-forces the reference's ids instead of this build's encoding, so a
// model is measured even while its tokenizer differs (or cannot encode the
// text yet: an unsupported BPE pre-tokenizer). Text is encoded with the
// model's own BOS rule (tokenizer.ggml.add_bos_token). A quantized KV cache
// stores keys per-channel scaled, as the engine does (core/key_scaling.h);
// --no-key-scaling stores them as computed, to measure what the scaling saves.
// The two runs are separate processes: a 35B MoE does not fit twice in RAM.
//
// Uses internal headers: built with the tests (LIYAB_BUILD_TESTS), on the CPU
// backend, with the engine's expert streaming (--memory-budget and
// --expert-cache like liyab-cli; --requant applies only when experts stream).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/thread_pool.h"
#include "core/tokenizer.h"
#include "core/transformer.h"
#include "liyab/backend.h"
#include "liyab/mmap_loader.h"

namespace {

constexpr int32_t kTop = 32;
constexpr int32_t kChunk = 64;
// Reference file: magic, version, token count, token ids (int32), position
// count, positions. Written by --save and by tools/conform_reference.py.
constexpr char kMagic[4] = {'L', 'Y', 'K', 'L'};
constexpr uint32_t kVersion = 2;

struct Position {
    int32_t ids[kTop];
    float logp[kTop];
    float target_logp;  // log-probability of the text's next token
};

int usage() {
    std::fprintf(stderr,
                 "usage: liyab-kl -m model.gguf -f text.txt (--save REF | --compare REF)\n"
                 "                [--requant 4|5] [--expert-mass P] [--memory-budget MB] [--expert-cache MB]\n"
                 "                [--experts N] [--threads N] [--max-tokens N (default 512)]\n"
                 "                [--kv f16|q8_0|q4_0|q4_1 (default q8_0)] [--no-key-scaling] [--ref-tokens]\n"
                 "                [--max-kl KL] [--min-top1 PERCENT]\n"
                 "  --no-key-scaling   quantized caches: store keys as computed, not per-channel scaled\n"
                 "  --ref-tokens       compare on the reference's token ids, not this build's encoding\n"
                 "  --max-kl KL        fail (exit 1) when the mean KL divergence exceeds KL\n"
                 "  --min-top1 P       fail (exit 1) when the top-1 agreement is below P percent\n");
    return 2;
}

// A reference file: the text's token ids and one Position per id but the last.
struct Reference {
    std::vector<int32_t> tokens;
    std::vector<Position> positions;
};

bool read_reference(const std::string& path, Reference& ref) {
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    uint32_t version = 0;
    uint64_t n_tokens = 0, n_positions = 0;
    in.read(magic, sizeof magic);
    in.read(reinterpret_cast<char*>(&version), sizeof version);
    if (!in || std::memcmp(magic, kMagic, sizeof magic) != 0 || version != kVersion) {
        std::fprintf(stderr, "%s is not a liyab-kl reference of version %u (an older file must be saved again)\n",
                     path.c_str(), kVersion);
        return false;
    }
    in.read(reinterpret_cast<char*>(&n_tokens), sizeof n_tokens);
    if (!in || n_tokens < 2 || n_tokens > (uint64_t{1} << 24)) return false;
    ref.tokens.resize(n_tokens);
    in.read(reinterpret_cast<char*>(ref.tokens.data()), static_cast<std::streamsize>(n_tokens * sizeof(int32_t)));
    in.read(reinterpret_cast<char*>(&n_positions), sizeof n_positions);
    if (!in || n_positions != n_tokens - 1) return false;
    ref.positions.resize(n_positions);
    in.read(reinterpret_cast<char*>(ref.positions.data()),
            static_cast<std::streamsize>(n_positions * sizeof(Position)));
    return static_cast<bool>(in);
}

// Log-softmax of one row of logits, in place.
void log_softmax(float* v, size_t n) {
    const float mx = *std::max_element(v, v + n);
    double sum = 0;
    for (size_t i = 0; i < n; ++i) sum += std::exp(static_cast<double>(v[i] - mx));
    const auto lse = static_cast<float>(std::log(sum)) + mx;
    for (size_t i = 0; i < n; ++i) v[i] -= lse;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, text_path, save_path, compare_path;
    int32_t requant = 0, threads = 0, max_tokens = 512;
    float expert_mass = 1.0f;
    int32_t max_experts = 0;
    int64_t budget_mb = 0, expert_cache_mb = -1;
    bool ref_tokens = false;
    bool key_scaling = true;
    liyab::KvCacheType kv_type = liyab::KvCacheType::Q8_0;
    double max_kl = -1.0, min_top1 = -1.0;  // < 0: no gate
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "-m") model_path = next();
        else if (arg == "-f") text_path = next();
        else if (arg == "--save") save_path = next();
        else if (arg == "--compare") compare_path = next();
        else if (arg == "--requant") requant = std::atoi(next());
        else if (arg == "--expert-mass") expert_mass = static_cast<float>(std::atof(next()));
        else if (arg == "--experts") max_experts = std::atoi(next());
        else if (arg == "--memory-budget") budget_mb = std::atoll(next());
        else if (arg == "--expert-cache") expert_cache_mb = std::atoll(next());
        else if (arg == "--threads") threads = std::atoi(next());
        else if (arg == "--max-tokens") max_tokens = std::atoi(next());
        else if (arg == "--ref-tokens") ref_tokens = true;
        else if (arg == "--no-key-scaling") key_scaling = false;
        else if (arg == "--kv") {
            const std::string v = next();
            if (v == "f16") kv_type = liyab::KvCacheType::F16;
            else if (v == "q8_0") kv_type = liyab::KvCacheType::Q8_0;
            else if (v == "q4_0") kv_type = liyab::KvCacheType::Q4_0;
            else if (v == "q4_1") kv_type = liyab::KvCacheType::Q4_1;
            else return usage();
        }
        else if (arg == "--max-kl") max_kl = std::atof(next());
        else if (arg == "--min-top1") min_top1 = std::atof(next());
        else return usage();
    }
    if (model_path.empty() || text_path.empty() || save_path.empty() == compare_path.empty()) return usage();
    if (compare_path.empty() && (ref_tokens || max_kl >= 0 || min_top1 >= 0)) return usage();

    std::ifstream text_file(text_path);
    std::stringstream text;
    text << text_file.rdbuf();
    if (text.str().empty()) {
        std::fprintf(stderr, "empty or unreadable text: %s\n", text_path.c_str());
        return 1;
    }

    liyab::LoaderOptions loader_options;
    loader_options.memory_budget_bytes = budget_mb > 0 ? static_cast<uint64_t>(budget_mb) << 20 : 0;
    auto file = liyab::MmapLoader::open(model_path, loader_options);
    if (!file) {
        std::fprintf(stderr, "%s\n", file.status().to_string().c_str());
        return 1;
    }
    liyab::TransformerOptions options;
    options.context_length = max_tokens + kChunk;
    options.max_batch = kChunk;
    options.kv_type = kv_type;
    options.memory_budget_bytes = loader_options.memory_budget_bytes;
    options.requant_bits = requant;
    options.expert_mass = expert_mass;
    options.max_experts = max_experts;
    options.expert_cache_bytes = expert_cache_mb > 0 ? expert_cache_mb << 20 : expert_cache_mb;
    auto loaded = liyab::Transformer::load(std::move(file).value(), options);
    if (!loaded) {
        std::fprintf(stderr, "%s\n", loaded.status().to_string().c_str());
        return 1;
    }
    auto model = std::move(loaded).value();
    auto tokenizer = liyab::Tokenizer::load(model->file());
    if (!tokenizer) {
        std::fprintf(stderr, "%s\n", tokenizer.status().to_string().c_str());
        return 1;
    }
    auto encoded = tokenizer->encode(text.str(), tokenizer->add_bos_default());
    std::vector<int32_t> tokens;
    if (encoded) {
        tokens = std::move(encoded).value();
    } else if (ref_tokens) {
        // A tokenizer this build cannot run yet (an unsupported pre-tokenizer):
        // the reference's ids are measured instead, so the model still is.
        std::printf("tokens: this build cannot encode the text (%s)\n", encoded.status().to_string().c_str());
    }
    if (tokens.size() < 2 && !(ref_tokens && !encoded)) {
        std::fprintf(stderr, "the text must encode to at least 2 tokens\n");
        return 1;
    }
    if (tokens.size() > static_cast<size_t>(max_tokens)) tokens.resize(static_cast<size_t>(max_tokens));

    Reference ref;
    if (!compare_path.empty()) {
        if (!read_reference(compare_path, ref)) {
            std::fprintf(stderr, "cannot read the reference %s\n", compare_path.c_str());
            return 1;
        }
        // The tokenizer's conformance: the same text must give the same ids.
        const size_t common = std::min(tokens.size(), ref.tokens.size());
        size_t first_diff = 0;
        while (first_diff < common && tokens[first_diff] == ref.tokens[first_diff]) ++first_diff;
        if (tokens.empty()) {
            // Not encoded here (see above): nothing to compare.
        } else if (first_diff == common && tokens.size() == ref.tokens.size()) {
            std::printf("tokens: %zu, identical to the reference\n", tokens.size());
        } else {
            auto at = [&](const std::vector<int32_t>& ids) {
                if (first_diff >= ids.size()) return std::string("(end)");
                return std::to_string(ids[first_diff]) + " '" +
                       (ids[first_diff] < tokenizer->vocab_size() ? tokenizer->piece(ids[first_diff]) : "?") + "'";
            };
            std::printf("tokens: %zu here, %zu in the reference; first difference at token %zu: %s here, %s there\n",
                        tokens.size(), ref.tokens.size(), first_diff, at(tokens).c_str(), at(ref.tokens).c_str());
            if (!ref_tokens) {
                std::fprintf(stderr, "the encodings differ: fix the tokenizer, or measure the model on the "
                                     "reference's ids with --ref-tokens\n");
                return 1;
            }
        }
        if (ref_tokens) {
            for (const int32_t id : ref.tokens) {
                if (id < 0 || id >= model->config().n_vocab) {
                    std::fprintf(stderr, "the reference has token id %d, outside this model's vocabulary\n", id);
                    return 1;
                }
            }
            tokens = ref.tokens;
        }
    }

    liyab::ThreadPool pool(threads);
    auto cpu = liyab::make_cpu_backend(pool);
    const liyab::Route route{cpu.get(), cpu.get(), cpu.get()};
    if (key_scaling && kv_type != liyab::KvCacheType::F16) {  // an F16 cache stores keys as computed
        auto calibration = tokenizer->encode(liyab::key_calibration_text(), tokenizer->add_bos_default());
        const liyab::Status s = calibration ? model->calibrate_keys(*calibration, route, pool) : calibration.status();
        if (!s.is_ok()) {
            std::fprintf(stderr, "key calibration: %s\n", s.to_string().c_str());
            return 1;
        }
    }
    const auto vocab = static_cast<size_t>(model->config().n_vocab);

    // Teacher-forced pass: position t predicts tokens[t + 1].
    std::vector<Position> positions;
    const std::vector<Position>& reference = ref.positions;
    double kl_sum = 0, ref_nll = 0, new_nll = 0;
    std::vector<double> kls;
    int32_t agree = 0;
    std::vector<int32_t> order(vocab);
    for (size_t begin = 0; begin + 1 < tokens.size(); begin += kChunk) {
        const size_t end = std::min(tokens.size() - 1, begin + kChunk);
        const std::span<const int32_t> chunk(tokens.data() + begin, end - begin);
        auto logits = model->forward(chunk, liyab::Transformer::Logits::All, route, pool);
        if (!logits) {
            std::fprintf(stderr, "%s\n", logits.status().to_string().c_str());
            return 1;
        }
        std::vector<float> rows(logits->begin(), logits->end());
        for (size_t r = 0; r < chunk.size(); ++r) {
            const size_t t = begin + r;
            float* row = rows.data() + r * vocab;
            log_softmax(row, vocab);
            const int32_t target = tokens[t + 1];
            if (compare_path.empty()) {
                Position p{};
                for (size_t i = 0; i < vocab; ++i) order[i] = static_cast<int32_t>(i);
                std::partial_sort(order.begin(), order.begin() + kTop, order.end(),
                                  [&](int32_t a, int32_t b) { return row[a] > row[b]; });
                for (int32_t k = 0; k < kTop; ++k) {
                    p.ids[k] = order[static_cast<size_t>(k)];
                    p.logp[k] = row[order[static_cast<size_t>(k)]];
                }
                p.target_logp = row[target];
                positions.push_back(p);
            } else {
                const Position& p = reference[t];
                double kl = 0, p_top = 0, q_top = 0;
                for (int32_t k = 0; k < kTop; ++k) {
                    const double lp = p.logp[k];
                    const double lq = row[p.ids[k]];
                    kl += std::exp(lp) * (lp - lq);
                    p_top += std::exp(lp);
                    q_top += std::exp(lq);
                }
                const double p_rest = std::max(1e-12, 1.0 - p_top);
                const double q_rest = std::max(1e-12, 1.0 - q_top);
                kl += p_rest * std::log(p_rest / q_rest);
                kls.push_back(kl);
                kl_sum += kl;
                agree += std::max_element(row, row + vocab) - row == p.ids[0];
                ref_nll -= p.target_logp;
                new_nll -= row[target];
            }
        }
        std::fprintf(stderr, "\r%zu / %zu positions", end, tokens.size() - 1);
    }
    std::fprintf(stderr, "\n");

    if (compare_path.empty()) {
        std::ofstream out(save_path, std::ios::binary);
        const uint64_t n_tokens = tokens.size();
        const uint64_t n = positions.size();
        out.write(kMagic, sizeof kMagic);
        out.write(reinterpret_cast<const char*>(&kVersion), sizeof kVersion);
        out.write(reinterpret_cast<const char*>(&n_tokens), sizeof n_tokens);
        out.write(reinterpret_cast<const char*>(tokens.data()), static_cast<std::streamsize>(n_tokens * sizeof(int32_t)));
        out.write(reinterpret_cast<const char*>(&n), sizeof n);
        out.write(reinterpret_cast<const char*>(positions.data()), static_cast<std::streamsize>(n * sizeof(Position)));
        double nll = 0;
        for (const Position& p : positions) nll -= p.target_logp;
        std::printf("saved %llu positions to %s; perplexity %.3f\n", static_cast<unsigned long long>(n),
                    save_path.c_str(), std::exp(nll / static_cast<double>(n)));
        return out ? 0 : 1;
    }
    const auto n = static_cast<double>(kls.size());
    std::sort(kls.begin(), kls.end());
    std::printf("%zu positions | KL mean %.5f, median %.5f, p99 %.5f, max %.5f | top-1 agreement %.2f%% | "
                "perplexity %.3f -> %.3f\n",
                kls.size(), kl_sum / n, kls[kls.size() / 2], kls[static_cast<size_t>(0.99 * (n - 1))], kls.back(),
                100.0 * agree / n, std::exp(ref_nll / n), std::exp(new_nll / n));
    bool pass = true;
    if (max_kl >= 0 && kl_sum / n > max_kl) {
        std::printf("FAIL: mean KL %.5f above %.5f\n", kl_sum / n, max_kl);
        pass = false;
    }
    if (min_top1 >= 0 && 100.0 * agree / n < min_top1) {
        std::printf("FAIL: top-1 agreement %.2f%% below %.2f%%\n", 100.0 * agree / n, min_top1);
        pass = false;
    }
    if (pass && (max_kl >= 0 || min_top1 >= 0)) std::printf("PASS\n");
    return pass ? 0 : 1;
}
