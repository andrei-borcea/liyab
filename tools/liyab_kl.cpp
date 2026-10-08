// liyab-kl — measures what a lossy option costs in quality: teacher-forced
// next-token distributions of a model over a text, saved once from the
// lossless path and compared against a lossy one (e.g. --requant 4).
//
//   liyab-kl -m model.gguf -f text.txt --save ref.bin               # lossless reference
//   liyab-kl -m model.gguf -f text.txt --compare ref.bin --requant 4
//
// For every position the reference keeps the log-probabilities of its top 32
// tokens, the remaining probability mass and the log-probability of the
// text's actual next token. The comparison prints the KL divergence
// KL(reference || lossy), computed over those 32 tokens plus one bucket for the
// rest (a lower bound of the full-vocabulary KL that is tight when the top 32
// carry nearly all the mass), the top-1 agreement and both perplexities.
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

struct Position {
    int32_t ids[kTop];
    float logp[kTop];
    float target_logp;  // log-probability of the text's next token
};

int usage() {
    std::fprintf(stderr,
                 "usage: liyab-kl -m model.gguf -f text.txt (--save REF | --compare REF)\n"
                 "                [--requant 4|5] [--expert-mass P] [--memory-budget MB] [--expert-cache MB]\n"
                 "                [--threads N] [--max-tokens N (default 512)]\n");
    return 2;
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
    int64_t budget_mb = 0, expert_cache_mb = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "-m") model_path = next();
        else if (arg == "-f") text_path = next();
        else if (arg == "--save") save_path = next();
        else if (arg == "--compare") compare_path = next();
        else if (arg == "--requant") requant = std::atoi(next());
        else if (arg == "--expert-mass") expert_mass = static_cast<float>(std::atof(next()));
        else if (arg == "--memory-budget") budget_mb = std::atoll(next());
        else if (arg == "--expert-cache") expert_cache_mb = std::atoll(next());
        else if (arg == "--threads") threads = std::atoi(next());
        else if (arg == "--max-tokens") max_tokens = std::atoi(next());
        else return usage();
    }
    if (model_path.empty() || text_path.empty() || save_path.empty() == compare_path.empty()) return usage();

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
    options.memory_budget_bytes = loader_options.memory_budget_bytes;
    options.requant_bits = requant;
    options.expert_mass = expert_mass;
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
    auto encoded = tokenizer->encode(text.str(), true);
    if (!encoded || encoded->size() < 2) {
        std::fprintf(stderr, "the text must encode to at least 2 tokens\n");
        return 1;
    }
    std::vector<int32_t> tokens = std::move(encoded).value();
    if (tokens.size() > static_cast<size_t>(max_tokens)) tokens.resize(static_cast<size_t>(max_tokens));

    liyab::ThreadPool pool(threads);
    auto cpu = liyab::make_cpu_backend(pool);
    const liyab::Route route{cpu.get(), cpu.get(), cpu.get()};
    const auto vocab = static_cast<size_t>(model->config().n_vocab);

    // Teacher-forced pass: position t predicts tokens[t + 1].
    std::vector<Position> positions;
    std::vector<float> lossy_logp;  // compare mode: full rows are needed at the reference's ids
    std::vector<Position> reference;
    if (!compare_path.empty()) {
        std::ifstream in(compare_path, std::ios::binary);
        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof n);
        reference.resize(n);
        in.read(reinterpret_cast<char*>(reference.data()), static_cast<std::streamsize>(n * sizeof(Position)));
        if (!in || n != tokens.size() - 1) {
            std::fprintf(stderr, "reference %s does not match this text (%llu positions)\n", compare_path.c_str(),
                         static_cast<unsigned long long>(n));
            return 1;
        }
    }
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
        const uint64_t n = positions.size();
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
    return 0;
}
