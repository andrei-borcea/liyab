// liyab-reuse — what segment reuse (KV reused away from its prefix,
// core/kv_reuse.h) costs in quality and saves in time to the first token,
// on RAG-shaped prompts: a system prompt, 3 to 5 documents in varying order
// and a question, rendered with the model's chat template.
//
//   liyab-reuse -m model.gguf                                  # the default grid of recompute fractions
//   liyab-reuse -m model.gguf --criterion first --fractions 0.1,0.2
//   liyab-reuse -m model.gguf --shift 300                      # the cost of turning keys alone
//
// The documents are first seen in warm-up prompts (random subsets, random
// orders), which store their KV as segments: processed whole, or with
// --warm-up-reuse F as the engine processes them, the documents stored
// already reused at share F and the new ones kept with rows computed after
// those (so copies made from reused rows are measured too). Each measured
// prompt is then answered three ways, from an empty context:
//  * full: the whole prompt processed (the reference);
//  * prefix: the part before the first document taken as restored by the
//    prefix cache (not timed), the rest processed: lossless;
//  * reuse: that part as restored, the documents taken from their stored
//    copies (turned to their new positions, a fraction recomputed), the
//    rest processed.
// Time to the first token is the prompt's processing plus the step that
// gives the first token's logits. Quality is the KL divergence KL(full ||
// reuse) over the whole vocabulary and the top-1 agreement at the first
// answer token and at each next one up to the full path's greedy answer's
// end of generation (at most --answer of them), teacher-forced on that
// answer; and how many greedy answers come out the same (top-1 everywhere).
// --dump prints the positions that moved most; --no-numbers leaves the
// documents unnumbered.
//
// --shift D measures the turning alone: the rows of a text's first part are
// moved D positions away and back (two moves, each rounding a quantized key
// again), and the rest of the text is processed on them; the KL against the
// unmoved context bounds what moving costs. --calibrate KL runs the engine's
// calibration (kv_reuse.h) against that target, on its own prompts of the
// same corpus, then measures the share it chose (or, when it refuses, the
// largest) on these prompts, to compare the two. Hybrid models are refused
// (see Transformer::reuse_support), with the share of a prompt's work
// reusing their attention blocks' keys and values would skip.
//
// Uses internal headers: built with the tests (LIYAB_BUILD_TESTS), CPU backend.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/chat_template.h"
#include "core/kv_reuse.h"
#include "core/kv_store.h"
#include "core/thread_pool.h"
#include "core/tokenizer.h"
#include "core/transformer.h"
#include "liyab/backend.h"
#include "liyab/mmap_loader.h"

using namespace liyab;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Log-softmax of a row of logits, in double.
std::vector<double> log_probs(std::span<const float> logits) {
    const float mx = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (const float v : logits) sum += std::exp(static_cast<double>(v - mx));
    const double lse = std::log(sum) + mx;
    std::vector<double> out(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) out[i] = logits[i] - lse;
    return out;
}

double kl(const std::vector<double>& p, const std::vector<double>& q) {
    double s = 0.0;
    for (size_t i = 0; i < p.size(); ++i) s += std::exp(p[i]) * (p[i] - q[i]);
    return std::max(s, 0.0);
}

size_t argmax(std::span<const float> row) {
    return static_cast<size_t>(std::max_element(row.begin(), row.end()) - row.begin());
}

struct Run {
    Transformer& model;
    ThreadPool& pool;
    Route route;
    int32_t chunk;

    Status prefill(std::span<const int32_t> tokens) {
        for (size_t i = 0; i < tokens.size(); i += static_cast<size_t>(chunk)) {
            const auto part = tokens.subspan(i, std::min(static_cast<size_t>(chunk), tokens.size() - i));
            auto r = model.forward(part, Transformer::Logits::None, route, pool);
            if (!r) return r.status();
        }
        return Status::ok();
    }
    Result<std::vector<float>> step(int32_t token) {
        auto r = model.forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, route, pool);
        if (!r) return r.status();
        return std::vector<float>(r->begin(), r->end());
    }
};

// Statistics of one fraction over every prompt.
struct Tally {
    double kl_sum = 0.0;
    std::vector<double> kls;
    int64_t agree = 0, positions = 0;
    double first_kl = 0.0;
    int64_t first_agree = 0;
    double ttft = 0.0;
    int64_t identical = 0;  // prompts whose greedy answer is the reference's
    int64_t reused = 0, recomputed = 0, computed = 0;
    double rows_ms = 0.0, blend_ms = 0.0;
    double kept_deviation = 0.0;  // summed over prompts
};

int usage() {
    std::fprintf(stderr,
                 "usage: liyab-reuse -m model.gguf [--kv f16|q8_0|q4_0|q4_1] [--threads N] [--prompts N]\n"
                 "                   [--fractions F,F,...] [--criterion deviation|first|random|attention|impact] [--select-block B]\n"
                 "                   [--first-tokens K] [--absolute] [--answer N] [--seed S] [--plain]\n"
                 "                   [--shift D] [--no-key-scaling] [--calibrate MAX_KL] [--dump] [--no-numbers]\n"
                 "                   [--warm-up-reuse F]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    KvCacheType kv_type = KvCacheType::Q8_0;
    int32_t threads = 0, prompts = 8, answer = 32, shift = 0;
    double calibrate = -1.0;  // >= 0: run the engine's calibration against this target
    float warm_up_reuse = -1.0f;  // >= 0: warm-up prompts reuse what is stored, at this share
    uint64_t seed = 7;
    bool plain = false, key_scaling = true, dump = false, numbered = true;
    std::vector<float> fractions = {0.0f, 0.05f, 0.1f, 0.15f, 0.2f, 0.3f, 0.5f, 1.0f};
    ReuseSettings settings;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "-m") model_path = next();
        else if (arg == "--kv") {
            const std::string v = next();
            if (v == "f16") kv_type = KvCacheType::F16;
            else if (v == "q8_0") kv_type = KvCacheType::Q8_0;
            else if (v == "q4_0") kv_type = KvCacheType::Q4_0;
            else if (v == "q4_1") kv_type = KvCacheType::Q4_1;
            else return usage();
        } else if (arg == "--threads") threads = std::atoi(next());
        else if (arg == "--prompts") prompts = std::atoi(next());
        else if (arg == "--answer") answer = std::atoi(next());
        else if (arg == "--seed") seed = std::strtoull(next(), nullptr, 10);
        else if (arg == "--plain") plain = true;
        else if (arg == "--dump") dump = true;
        else if (arg == "--no-numbers") numbered = false;
        else if (arg == "--shift") shift = std::atoi(next());
        else if (arg == "--calibrate") calibrate = std::atof(next());
        else if (arg == "--warm-up-reuse") warm_up_reuse = std::strtof(next(), nullptr);
        else if (arg == "--no-key-scaling") key_scaling = false;
        else if (arg == "--select-block") settings.select_block = std::atoi(next());
        else if (arg == "--first-tokens") settings.first_tokens = std::atoi(next());
        else if (arg == "--refine") settings.refine = std::atoi(next());
        else if (arg == "--absolute") settings.relative = false;
        else if (arg == "--criterion") {
            const std::string v = next();
            if (v == "deviation") settings.criterion = ReuseCriterion::Deviation;
            else if (v == "first") settings.criterion = ReuseCriterion::First;
            else if (v == "random") settings.criterion = ReuseCriterion::Random;
            else if (v == "attention") settings.criterion = ReuseCriterion::Attention;
            else if (v == "impact") settings.criterion = ReuseCriterion::Impact;
            else return usage();
        } else if (arg == "--fractions") {
            fractions.clear();
            std::stringstream list(next());
            for (std::string item; std::getline(list, item, ',');) fractions.push_back(std::strtof(item.c_str(), nullptr));
        } else return usage();
    }
    if (model_path.empty() || fractions.empty() || prompts < 1) return usage();

    auto file = MmapLoader::open(model_path);
    if (!file) {
        std::fprintf(stderr, "%s\n", file.status().to_string().c_str());
        return 1;
    }
    TransformerOptions options;
    options.context_length = 4096;
    options.max_batch = 64;
    options.kv_type = kv_type;
    // As the engine with segment reuse on: windowed blocks keep every
    // position (a reused segment needs every block's rows).
    options.keep_windowed_rows = true;
    auto loaded = Transformer::load(std::move(file).value(), options);
    if (!loaded) {
        std::fprintf(stderr, "%s\n", loaded.status().to_string().c_str());
        return 1;
    }
    auto model = std::move(loaded).value();
    auto tokenizer = Tokenizer::load(model->file());
    if (!tokenizer) {
        std::fprintf(stderr, "%s\n", tokenizer.status().to_string().c_str());
        return 1;
    }
    ThreadPool pool(threads);
    auto cpu = make_cpu_backend(pool);
    Run run{*model, pool, {cpu.get(), cpu.get(), cpu.get()}, 64};
    const ModelConfig& c = model->config();
    std::printf("model: %s, %d blocks (%d attention), d=%d, %d threads, KV %s\n", c.arch.c_str(), c.n_layers,
                c.n_attn_layers, c.n_embd, pool.max_threads(), std::string(dtype_traits(model->kv_cache().dtype()).name).c_str());

    if (Status s = model->reuse_support(); !s.is_ok()) {
        // What reusing only the attention blocks' keys and values would skip:
        // their key and value projections, against every block weight a
        // prompt token goes through (the output head runs for the last one only).
        double kv_params = 0.0, all_params = 0.0;
        for (const TensorView& t : model->file().tensors()) {
            const std::string_view name = t.name;
            if (t.n_dims < 2 || !name.starts_with("blk.")) continue;
            double active = static_cast<double>(t.elements());
            if (name.ends_with("_exps.weight") && c.n_expert > 0) active = active * c.n_expert_used / c.n_expert;
            all_params += active;
            if (name.ends_with("attn_k.weight") || name.ends_with("attn_v.weight")) kv_params += active;
        }
        std::printf("segment reuse refused: %s\n", s.message().c_str());
        std::printf("reusing the attention blocks' keys and values would skip %.2f%% of a prompt token's matrix work "
                    "(their key and value projections); every block still runs for every token\n",
                    100.0 * kv_params / std::max(all_params, 1.0));
        return 0;
    }

    if (key_scaling && kv_type != KvCacheType::F16) {
        auto calibration = tokenizer->encode(key_calibration_text(), tokenizer->add_bos_default());
        const Status s = calibration ? model->calibrate_keys(*calibration, run.route, pool) : calibration.status();
        if (!s.is_ok()) {
            std::fprintf(stderr, "key calibration: %s\n", s.to_string().c_str());
            return 1;
        }
    }
    const bool bos = tokenizer->add_bos_default();

    // --- The turning alone ---------------------------------------------------
    if (shift != 0) {
        std::vector<size_t> all(kReuseDocuments);
        std::iota(all.begin(), all.end(), size_t{0});
        ReusePromptStyle unnumbered;
        unnumbered.numbered = false;
        auto corpus = reuse_corpus_prompt(unnumbered, all, 0);
        if (!corpus) return 1;
        const std::string text =
            corpus->text.substr(corpus->documents.front().begin,
                                corpus->documents.back().end - corpus->documents.front().begin) + "\n\n";
        auto encoded = tokenizer->encode(text, bos);
        if (!encoded) return 1;
        std::vector<int32_t> t = std::move(encoded).value();
        t.resize(std::min<size_t>(t.size(), 1024));
        const int32_t n = static_cast<int32_t>(t.size()), m = n - 64;
        // The reference: the text's last 64 positions on the unmoved context.
        model->reset();
        if (!run.prefill(std::span(t).first(static_cast<size_t>(m))).is_ok()) return 1;
        auto ref_rows = model->forward(std::span(t).subspan(static_cast<size_t>(m)), Transformer::Logits::All, run.route, pool);
        if (!ref_rows) return 1;
        const std::vector<float> ref(ref_rows->begin(), ref_rows->end());
        // The first m positions moved away and back, then the same 64.
        model->reset();
        if (!run.prefill(std::span(t).first(static_cast<size_t>(m))).is_ok()) return 1;
        KvCache& kv = model->mutable_kv_cache();
        std::vector<uint8_t> image(kv.rows_bytes(m));
        if (!kv.read_rows_at(0, m, image.data(), m, 0).is_ok()) return 1;
        const auto t0 = Clock::now();
        if (!model->move_keys(image.data(), m, 0, m, 0, shift, &pool).is_ok()) return 1;
        const double move_ms = ms_since(t0);
        if (!model->move_keys(image.data(), m, 0, m, shift, 0, &pool).is_ok()) return 1;
        if (!kv.write_rows_at(0, image.data(), m, 0, m).is_ok()) return 1;
        auto got = model->forward(std::span(t).subspan(static_cast<size_t>(m)), Transformer::Logits::All, run.route, pool);
        if (!got) return 1;
        const auto vocab = static_cast<size_t>(c.n_vocab);
        double kl_sum = 0.0, worst = 0.0;
        int agree = 0;
        for (int32_t r = 0; r < 64; ++r) {
            const std::span<const float> a(ref.data() + static_cast<size_t>(r) * vocab, vocab);
            const std::span<const float> b(got->data() + static_cast<size_t>(r) * vocab, vocab);
            const double k = kl(log_probs(a), log_probs(b));
            kl_sum += k;
            worst = std::max(worst, k);
            agree += argmax(a) == argmax(b);
        }
        std::printf("keys of %d positions moved by %d and back: KL mean %.6f, max %.6f, top-1 %.1f%% over the next 64 "
                    "positions; one move took %.2f ms (%.1f us per position, every block)\n",
                    m, shift, kl_sum / 64.0, worst, 100.0 * agree / 64.0, move_ms, 1000.0 * move_ms / m);
        return 0;
    }

    // --- RAG prompts: the corpus in the model's chat template --------------------
    std::optional<ChatTemplate> chat;
    ReusePromptStyle style;
    if (!plain) {
        if (const auto source = model->file().get_string("tokenizer.chat_template")) {
            if (auto parsed = ChatTemplate::parse(*source, tokenizer->chat_sort_tools())) {
                chat.emplace(std::move(parsed).value());
                style.chat = &*chat;
                auto special = [&](const char* name, const char* key) {
                    if (const auto id = model->file().get_int(key); id && *id >= 0 && *id < tokenizer->vocab_size()) {
                        style.special.fields().set(name, jinja::Value(std::string(tokenizer->token_text(static_cast<int32_t>(*id)))));
                    }
                };
                special("bos_token", "tokenizer.ggml.bos_token_id");
                special("eos_token", "tokenizer.ggml.eos_token_id");
                style.thinking_off = source->find("enable_thinking") != std::string::npos;
            }
        }
    }

    // --- The engine's calibration ------------------------------------------------
    // Its result, then the share it chose measured on this tool's prompts.
    float calibrated = -1.0f;
    double calibrated_kl = 0.0, calibrated_bound = 0.0;
    bool calibrated_met = false;
    if (calibrate >= 0.0) {
        auto r = calibrate_reuse(*model, *tokenizer, style, settings, calibrate, run.route, pool, run.chunk);
        if (!r) {
            std::fprintf(stderr, "calibration: %s\n", r.status().to_string().c_str());
            return 1;
        }
        std::printf("calibration against KL %.4f: %s at %.0f%% recomputed, KL %.5f (bound %.5f), top-1 %.1f%%, %.0f ms; "
                    "KL (bound) by share:",
                    calibrate, r->met ? "met" : "not met", 100.0 * r->fraction, r->kl, r->bound, 100.0 * r->top1, r->ms);
        for (size_t i = 0; i < r->kls.size(); ++i) {
            std::printf(" %.0f%%: %.5f (%.5f)", 100.0 * kReuseFractions[i], r->kls[i], r->bounds[i]);
        }
        // Which of its questions carry the KL, at the share kept (or the largest).
        const auto at = static_cast<size_t>(
            std::find(std::begin(kReuseFractions), std::end(kReuseFractions), r->fraction) - std::begin(kReuseFractions));
        std::printf("\nits questions' KL at %.0f%%:", 100.0 * r->fraction);
        if (at < r->question_kls.size()) {
            for (const double q : r->question_kls[at]) std::printf(" %.4f", q);
        }
        std::printf("\n");
        calibrated = r->fraction;
        calibrated_kl = r->kl;
        calibrated_bound = r->bound;
        calibrated_met = r->met;
        if (std::find(fractions.begin(), fractions.end(), calibrated) == fractions.end()) fractions.push_back(calibrated);
    }

    style.numbered = numbered;
    std::printf("prompts: %s, %d measured, %d answer tokens scored after the first\n",
                style.chat != nullptr ? "the model's chat template" : "plain text", prompts, answer);

    KvStore::Config store_config;
    const uint64_t seed_parts[] = {model->context_fingerprint()};
    store_config.seed = kv_seed_key(seed_parts);
    store_config.chunk_tokens = model->kv_cache().config().page_tokens;
    store_config.chunk_bytes = model->kv_cache().page_bytes();
    store_config.ram_bytes = size_t{2} << 30;
    auto store = KvStore::open(store_config);
    if (!store) return 1;

    std::mt19937_64 rng(seed);
    auto pick = [&](size_t count) {
        std::vector<size_t> all(kReuseDocuments);
        for (size_t i = 0; i < all.size(); ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        all.resize(count);
        return all;
    };
    auto encode = [&](const ReusePrompt& p, std::vector<int32_t>& tokens, std::vector<TokenSegment>& segments) {
        tokens.clear();
        segments.clear();
        return encode_segmented(*tokenizer, p.text, 0, p.documents, bos, tokens, segments);
    };

    // Warm-up: every document seen (and stored) in prompts of 3 to 5 others.
    std::set<size_t> seen;
    int warmups = 0;
    int64_t warm_reused = 0;
    while (seen.size() < kReuseDocuments) {
        const std::vector<size_t> docs = pick(3 + rng() % 3);
        auto prompt = reuse_corpus_prompt(style, docs, rng() % kReuseDocuments);
        if (!prompt) {
            std::fprintf(stderr, "%s\n", prompt.status().to_string().c_str());
            return 1;
        }
        std::vector<int32_t> tokens;
        std::vector<TokenSegment> segments;
        if (!encode(*prompt, tokens, segments).is_ok()) return 1;
        model->reset();
        if (warm_up_reuse >= 0.0f && !segments.empty()) {
            // As the engine does: the stored documents reused, the new ones
            // computed after them and kept.
            const auto n = static_cast<int32_t>(tokens.size());
            const int32_t exact = segments.front().begin;
            ReuseSettings w = settings;
            w.fraction = warm_up_reuse;
            if (!run.prefill(std::span(tokens).first(static_cast<size_t>(exact))).is_ok()) return 1;
            auto r = reuse_segments(*model, *store, tokens, segments, exact, n, w, run.route, pool, run.chunk);
            if (!r) {
                std::fprintf(stderr, "warm-up reuse: %s\n", r.status().to_string().c_str());
                return 1;
            }
            warm_reused += r->reused;
            if (!run.prefill(std::span(tokens).subspan(static_cast<size_t>(r->end))).is_ok()) return 1;
        } else if (!run.prefill(tokens).is_ok()) {
            return 1;
        }
        keep_segments(*model, *store, tokens, segments, settings.min_tokens);
        seen.insert(docs.begin(), docs.end());
        ++warmups;
    }
    std::printf("warm-up: %d prompts stored %d documents (%d segment copies)", warmups, static_cast<int>(kReuseDocuments),
                store->stats().segments);
    if (warm_up_reuse >= 0.0f) {
        std::printf(", %lld positions reused at %.0f%% before the new ones were computed", static_cast<long long>(warm_reused),
                    100.0 * warm_up_reuse);
    }
    std::printf("\n\n");

    std::vector<Tally> tallies(fractions.size());
    double full_ttft = 0.0, prefix_ttft = 0.0;
    int64_t prompt_tokens = 0, doc_tokens = 0;
    for (int32_t i = 0; i < prompts; ++i) {
        const std::vector<size_t> docs = pick(3 + rng() % 3);
        const size_t question = docs[rng() % docs.size()];
        auto prompt = reuse_corpus_prompt(style, docs, question);
        if (!prompt) return 1;
        std::vector<int32_t> tokens;
        std::vector<TokenSegment> segments;
        if (!encode(*prompt, tokens, segments).is_ok() || segments.empty()) return 1;
        const auto n = static_cast<int32_t>(tokens.size());
        const int32_t exact = segments.front().begin;
        prompt_tokens += n;
        for (const TokenSegment& s : segments) doc_tokens += s.end - s.begin;

        // Full: the reference, and its greedy answer.
        model->reset();
        auto t0 = Clock::now();
        if (!run.prefill(std::span(tokens).first(static_cast<size_t>(n - 1))).is_ok()) return 1;
        auto first = run.step(tokens.back());
        if (!first) return 1;
        full_ttft += ms_since(t0);
        // The greedy answer, up to its end of generation (included: whether
        // and where the answer ends is part of what reuse may change).
        std::vector<std::vector<double>> ref = {log_probs(*first)};
        std::vector<int32_t> said = {static_cast<int32_t>(argmax(*first))};
        while (static_cast<int32_t>(said.size()) <= answer && !tokenizer->is_end_of_generation(said.back())) {
            auto next = run.step(said.back());
            if (!next) return 1;
            ref.push_back(log_probs(*next));
            said.push_back(static_cast<int32_t>(argmax(*next)));
        }
        said.pop_back();  // the answer tokens fed back: one per scored position after the first

        // Prefix only (lossless): the part before the first document restored.
        model->reset();
        if (!run.prefill(std::span(tokens).first(static_cast<size_t>(exact))).is_ok()) return 1;
        t0 = Clock::now();
        if (!run.prefill(std::span(tokens).subspan(static_cast<size_t>(exact), static_cast<size_t>(n - 1 - exact))).is_ok()) return 1;
        if (!run.step(tokens.back())) return 1;
        prefix_ttft += ms_since(t0);

        for (size_t f = 0; f < fractions.size(); ++f) {
            ReuseSettings s = settings;
            s.fraction = fractions[f];
            s.seed = seed + static_cast<uint64_t>(i);
            model->reset();
            if (!run.prefill(std::span(tokens).first(static_cast<size_t>(exact))).is_ok()) return 1;
            t0 = Clock::now();
            auto reused = reuse_segments(*model, *store, tokens, segments, exact, n - 1, s, run.route, pool, run.chunk);
            if (!reused) {
                std::fprintf(stderr, "reuse: %s\n", reused.status().to_string().c_str());
                return 1;
            }
            if (!run.prefill(std::span(tokens).subspan(static_cast<size_t>(reused->end),
                                                       static_cast<size_t>(n - 1 - reused->end))).is_ok()) {
                return 1;
            }
            auto got = run.step(tokens.back());
            if (!got) return 1;
            Tally& t = tallies[f];
            t.ttft += ms_since(t0);
            t.reused += reused->reused;
            t.recomputed += reused->recomputed;
            t.computed += reused->computed;
            t.rows_ms += reused->rows_ms;
            t.blend_ms += reused->blend_ms;
            t.kept_deviation += reused->kept_deviation;
            std::vector<std::vector<double>> mine = {log_probs(*got)};
            if (!said.empty()) {
                auto rows = model->forward(said, Transformer::Logits::All, run.route, pool);
                if (!rows) return 1;
                const auto vocab = static_cast<size_t>(c.n_vocab);
                for (size_t a = 0; a < said.size(); ++a) {
                    mine.push_back(log_probs(std::span<const float>(rows->data() + a * vocab, vocab)));
                }
            }
            bool identical = true;  // the greedy answer this path gives is the reference's
            if (dump && f == 0) {
                std::string text;
                for (size_t a = 0; a < said.size(); ++a) text += tokenizer->piece(said[a]);
                std::printf("prompt %d (docs", i);
                for (const size_t d : docs) std::printf(" %zu", d);
                std::printf(", question %zu): \"%s\"\n", question, text.c_str());
            }
            for (size_t p = 0; p < mine.size(); ++p) {
                const double k = kl(ref[p], mine[p]);
                const auto ref_best = static_cast<int32_t>(std::max_element(ref[p].begin(), ref[p].end()) - ref[p].begin());
                const auto my_best = static_cast<int32_t>(std::max_element(mine[p].begin(), mine[p].end()) - mine[p].begin());
                const bool same = ref_best == my_best;
                if (dump && f == 0 && k > 0.05) {
                    std::printf("  position %zu: KL %.3f, '%s' %.2f here -> '%s' %.2f reused\n", p, k,
                                tokenizer->piece(ref_best).c_str(), std::exp(ref[p][static_cast<size_t>(ref_best)]),
                                tokenizer->piece(my_best).c_str(), std::exp(mine[p][static_cast<size_t>(my_best)]));
                }
                t.kl_sum += k;
                t.kls.push_back(k);
                t.agree += same;
                identical = identical && same;
                ++t.positions;
                if (p == 0) {
                    t.first_kl += k;
                    t.first_agree += same;
                }
            }
            t.identical += identical;
        }
        std::fprintf(stderr, "\r%d / %d prompts", i + 1, prompts);
    }
    std::fprintf(stderr, "\n");
    const char* names[] = {"deviation", "first", "random", "attention", "impact"};
    std::printf("criterion %s (select block %d, %s, first tokens %d, refine %d); %.0f tokens per prompt, %.0f%% of them in documents\n",
                names[static_cast<int>(settings.criterion)], settings.select_block,
                settings.relative ? "relative" : "absolute", settings.first_tokens, settings.refine,
                static_cast<double>(prompt_tokens) / prompts, 100.0 * static_cast<double>(doc_tokens) / static_cast<double>(prompt_tokens));
    std::printf("time to first token: full %.1f ms, prefix restored %.1f ms\n\n", full_ttft / prompts,
                prefix_ttft / prompts);
    std::printf("fraction | KL mean  median   p95      max      | top-1  | same answer | first token KL top-1 | TTFT ms  vs prefix | reused recomputed | rows ms blend ms | deviation kept\n");
    for (size_t f = 0; f < fractions.size(); ++f) {
        Tally& t = tallies[f];
        std::sort(t.kls.begin(), t.kls.end());
        const auto np = static_cast<double>(t.positions);
        std::printf("%8.2f | %.5f %.5f %.5f %.5f | %5.1f%% | %3d of %3d  | %.5f %5.1f%%       | %8.1f %5.2fx    | %6.0f %6.0f     | %6.1f %8.1f | %5.1f%%\n",
                    fractions[f], t.kl_sum / np, t.kls[t.kls.size() / 2], t.kls[static_cast<size_t>(0.95 * (np - 1))],
                    t.kls.back(), 100.0 * static_cast<double>(t.agree) / np, static_cast<int>(t.identical), prompts,
                    t.first_kl / prompts,
                    100.0 * static_cast<double>(t.first_agree) / prompts, t.ttft / prompts, prefix_ttft / t.ttft,
                    static_cast<double>(t.reused) / prompts, static_cast<double>(t.recomputed) / prompts,
                    t.rows_ms / prompts, t.blend_ms / prompts, 100.0 * t.kept_deviation / prompts);
    }
    if (calibrated >= 0.0f) {
        const size_t f = static_cast<size_t>(std::find(fractions.begin(), fractions.end(), calibrated) - fractions.begin());
        const double measured = tallies[f].kl_sum / static_cast<double>(tallies[f].positions);
        std::printf("\ncalibrated %.0f%% (%s): KL %.5f, bound %.5f on the calibration's prompts; measured here %.5f, %s the "
                    "target %.4f\n",
                    100.0 * calibrated, calibrated_met ? "kept" : "refused, the largest share", calibrated_kl,
                    calibrated_bound, measured, measured <= calibrate ? "within" : "above", calibrate);
    }
    return 0;
}
