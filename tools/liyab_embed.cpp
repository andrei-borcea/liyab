// liyab-embed — pooling models on their own: embeds the texts of a cases file
// or reranks its queries' documents, compares the results with a reference
// written by Hugging Face transformers, and measures throughput.
//
//     liyab-embed -m Qwen3-Embedding-0.6B-f16.gguf --cases tests/data/pooling_cases.json --ref pooling.ref.json
//     liyab-embed -m Qwen3-Reranker-0.6B-BF16.gguf --cases tests/data/pooling_cases.json --ref pooling.ref.json
//     liyab-embed -m Qwen3-Embedding-0.6B-f16.gguf --cases tests/data/pooling_cases.json --bench 5
//
// The cases file (tests/data/pooling_cases.json) holds "embed", texts embedded
// as they are, and "rerank", cases of one query and its documents.
// tools/pooling_reference.py writes the reference from the models' own
// Hugging Face implementations, the way their model cards run them. What runs
// follows the model: a reranker (pooling_type rank, or --labels) reranks,
// anything else embeds (--pooling picks Mean or Last for a model that
// declares none). With a reference, every input's token ids are checked
// first (the first difference is printed), then each vector's cosine with
// the reference's (--min-cosine, default 0.999) or each score's distance from
// it (--max-score-diff, default 0.001); exit status 1 when one fails. Every
// input is also run on its own, and its result must be the same bits as in
// the batch (--no-check-alone skips it).
//
// Uses internal headers: built with the tests (LIYAB_BUILD_TESTS).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/chat_template.h"
#include "core/log.h"
#include "core/pooling.h"
#include "liyab/liyab.h"

namespace {

using liyab::jinja::Value;

int usage() {
    std::fprintf(stderr,
                 "usage: liyab-embed -m model.gguf --cases CASES.json [options]\n"
                 "  --ref REF.json          compare with tools/pooling_reference.py's output (exit 1 on a failure)\n"
                 "  --min-cosine X          embeddings: lowest cosine with the reference accepted (default 0.999)\n"
                 "  --max-score-diff X      rerank: largest score distance from the reference accepted (default 0.001)\n"
                 "  --pooling auto|mean|last  embeddings: the model's own pooling (default) or this one\n"
                 "  --no-normalize          embeddings as pooled (compared with the reference's raw last-token rows)\n"
                 "  --labels A,B            rerank with the LM head's logits of these tokens (no ranking head)\n"
                 "  --template FILE         rerank with this prompt template ({query}, {document})\n"
                 "  --backend auto|cpu|metal|vulkan  (default: the device ranking)\n"
                 "  --kv f16|q8_0|q4_0|q4_1 KV cache type (default: the engine's, F16 for a pooling model)\n"
                 "  --threads N --context N\n"
                 "  --bench N               run the whole set N more times and print throughput\n"
                 "  --no-check-alone        skip running every input on its own\n"
                 "  --quiet                 only the summary\n");
    return 2;
}

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

std::vector<float> floats(const Value& v) {
    std::vector<float> out;
    if (!v.is_array()) return out;
    for (const Value& x : v.items()) out.push_back(static_cast<float>(x.as_float()));
    return out;
}

std::vector<int32_t> ints(const Value& v) {
    std::vector<int32_t> out;
    if (!v.is_array()) return out;
    for (const Value& x : v.items()) out.push_back(static_cast<int32_t>(x.as_int()));
    return out;
}

const Value* field(const Value& object, std::string_view key) {
    return object.is_object() ? object.fields().find(key) : nullptr;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return na > 0.0 && nb > 0.0 ? dot / std::sqrt(na * nb) : (na == nb ? 1.0 : 0.0);
}

// The first position where two id lists differ, or -1.
long first_difference(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return static_cast<long>(i);
    }
    return a.size() == b.size() ? -1 : static_cast<long>(n);
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, cases_path, ref_path, template_path, pooling_name = "auto", backend = "auto", kv;
    std::vector<std::string> labels;
    double min_cosine = 0.999, max_score_diff = 0.001;
    bool normalize = true, check_alone = true, quiet = false;
    int bench = 0, threads = 0, context = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "-m" && i + 1 < argc) model_path = next();
        else if (a == "--cases" && i + 1 < argc) cases_path = next();
        else if (a == "--ref" && i + 1 < argc) ref_path = next();
        else if (a == "--min-cosine" && i + 1 < argc) min_cosine = std::atof(next());
        else if (a == "--max-score-diff" && i + 1 < argc) max_score_diff = std::atof(next());
        else if (a == "--pooling" && i + 1 < argc) pooling_name = next();
        else if (a == "--no-normalize") normalize = false;
        else if (a == "--labels" && i + 1 < argc) {
            std::stringstream ss(next());
            for (std::string l; std::getline(ss, l, ',');) labels.push_back(l);
        }
        else if (a == "--template" && i + 1 < argc) template_path = next();
        else if (a == "--backend" && i + 1 < argc) backend = next();
        else if (a == "--kv" && i + 1 < argc) kv = next();
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(next());
        else if (a == "--context" && i + 1 < argc) context = std::atoi(next());
        else if (a == "--bench" && i + 1 < argc) bench = std::atoi(next());
        else if (a == "--no-check-alone") check_alone = false;
        else if (a == "--quiet") quiet = true;
        else return usage();
    }
    if (model_path.empty() || cases_path.empty()) return usage();

    std::string text;
    if (!read_file(cases_path, text)) {
        std::fprintf(stderr, "cannot read %s\n", cases_path.c_str());
        return 2;
    }
    auto cases = liyab::jinja::parse_json(text);
    if (!cases) {
        std::fprintf(stderr, "%s: %s\n", cases_path.c_str(), cases.status().message().c_str());
        return 2;
    }
    Value ref;
    if (!ref_path.empty()) {
        std::string ref_text;
        auto parsed = read_file(ref_path, ref_text) ? liyab::jinja::parse_json(ref_text)
                                                    : liyab::Result<Value>(liyab::Status(liyab::ErrorCode::IoError, "cannot read"));
        if (!parsed) {
            std::fprintf(stderr, "%s: %s\n", ref_path.c_str(), parsed.status().message().c_str());
            return 2;
        }
        ref = std::move(parsed).value();
    }
    liyab::RerankOptions rerank_options;
    rerank_options.labels = labels;
    if (!template_path.empty() && !read_file(template_path, rerank_options.prompt_template)) {
        std::fprintf(stderr, "cannot read %s\n", template_path.c_str());
        return 2;
    }

    liyab::log::set_level(liyab::log::Level::Warn);
    liyab::EngineConfig config;
    config.model_path = model_path;
    config.n_threads = threads;
    config.context_length = context;
    config.thermal_polling = false;
    if (backend == "cpu") config.backend = liyab::BackendKind::Cpu;
    else if (backend == "metal") config.backend = liyab::BackendKind::Metal;
    else if (backend == "vulkan") config.backend = liyab::BackendKind::Vulkan;
    else if (backend != "auto") return usage();
    if (!kv.empty()) {  // both formats, so the choice applies whatever the model declares
        liyab::KvCacheType type = liyab::KvCacheType::F16;
        if (kv == "q8_0") type = liyab::KvCacheType::Q8_0;
        else if (kv == "q4_0") type = liyab::KvCacheType::Q4_0;
        else if (kv == "q4_1") type = liyab::KvCacheType::Q4_1;
        else if (kv != "f16") return usage();
        config.kv_cache_type = config.pooling_kv_cache_type = type;
    }
    auto created = liyab::Engine::create(config);
    if (!created) {
        std::fprintf(stderr, "%s\n", created.status().to_string().c_str());
        return 2;
    }
    liyab::Engine& engine = **created;

    const bool rerank = engine.pooling() == liyab::Pooling::Rank || !labels.empty();
    liyab::EmbedOptions embed_options;
    embed_options.normalize = normalize;
    if (pooling_name == "mean") embed_options.pooling = liyab::Pooling::Mean;
    else if (pooling_name == "last") embed_options.pooling = liyab::Pooling::Last;
    else if (pooling_name != "auto") return usage();
    {
        // The backend the transformer blocks run on, from the engine's description.
        const std::string description = engine.describe();
        const size_t at = description.find("Attention:");
        const std::string route =
            at == std::string::npos ? "" : description.substr(at, description.find('\n', at) - at);
        const size_t kv_at = description.find("KV cache:");
        const std::string cache =
            kv_at == std::string::npos ? "" : description.substr(kv_at, description.find(',', kv_at) - kv_at);
        std::printf("%s: %s; %s; %s\n", model_path.c_str(), rerank ? "rerank" : "embed", cache.c_str(), route.c_str());
    }

    // The inputs: texts to embed, or (query, documents) cases.
    std::vector<std::string> texts;
    struct Case {
        std::string query;
        std::vector<std::string> documents;
    };
    std::vector<Case> rerank_cases;
    if (!rerank) {
        const Value* list = field(*cases, "embed");
        if (list == nullptr || !list->is_array()) {
            std::fprintf(stderr, "%s has no \"embed\" list\n", cases_path.c_str());
            return 2;
        }
        for (const Value& t : list->items()) texts.push_back(t.as_string());
    } else {
        const Value* list = field(*cases, "rerank");
        if (list == nullptr || !list->is_array()) {
            std::fprintf(stderr, "%s has no \"rerank\" list\n", cases_path.c_str());
            return 2;
        }
        for (const Value& c : list->items()) {
            Case rc;
            if (const Value* q = field(c, "query")) rc.query = q->as_string();
            if (const Value* docs = field(c, "documents"); docs != nullptr && docs->is_array()) {
                for (const Value& d : docs->items()) rc.documents.push_back(d.as_string());
            }
            rerank_cases.push_back(std::move(rc));
        }
    }

    // One run of the whole set: vectors (embed) or per-case scores (rerank).
    std::vector<std::vector<float>> vectors;
    std::vector<std::vector<float>> scores;
    liyab::PoolingStats total;
    auto run_all = [&]() -> bool {
        total = {};
        if (!rerank) {
            const std::vector<std::string_view> views(texts.begin(), texts.end());
            auto r = engine.embed(views, embed_options, &total);
            if (!r) {
                std::fprintf(stderr, "embed: %s\n", r.status().to_string().c_str());
                return false;
            }
            vectors = std::move(r).value();
            return true;
        }
        scores.clear();
        for (const Case& c : rerank_cases) {
            const std::vector<std::string_view> views(c.documents.begin(), c.documents.end());
            liyab::PoolingStats stats;
            auto r = engine.rerank(c.query, views, rerank_options, &stats);
            if (!r) {
                std::fprintf(stderr, "rerank: %s\n", r.status().to_string().c_str());
                return false;
            }
            scores.push_back(std::move(r).value());
            total.tokens += stats.tokens;
            total.reused_tokens += stats.reused_tokens;
            total.ms += stats.ms;
        }
        return true;
    };
    if (!run_all()) return 1;
    const size_t inputs = rerank ? [&] {
        size_t n = 0;
        for (const Case& c : rerank_cases) n += c.documents.size();
        return n;
    }() : texts.size();
    std::printf("%zu %s, %d tokens (%d shared with a neighbour) in %.0f ms\n", inputs, rerank ? "pairs" : "texts",
                total.tokens, total.reused_tokens, total.ms);

    bool ok = true;
    // Token ids against the reference's: the tokenizer and the input layout first.
    if (!ref_path.empty()) {
        const Value* ref_ids = field(ref, rerank ? "rerank_ids" : "embed_ids");
        if (ref_ids == nullptr || !ref_ids->is_array()) {
            std::fprintf(stderr, "the reference has no %s\n", rerank ? "rerank_ids" : "embed_ids");
            return 2;
        }
        const auto eos_meta = engine.model_metadata("tokenizer.ggml.add_eos_token");
        const bool add_eos = eos_meta && *eos_meta == "true";
        const auto eos_id = engine.model_metadata("tokenizer.ggml.eos_token_id");
        auto ids_of = [&](std::string_view input) {
            auto encoded = engine.tokenize(input, true);
            std::vector<int32_t> ids = encoded ? std::move(encoded).value() : std::vector<int32_t>{};
            if (add_eos && eos_id) ids.push_back(std::atoi(eos_id->c_str()));
            return ids;
        };
        std::string tmpl = rerank_options.prompt_template;
        if (tmpl.empty()) tmpl = engine.model_metadata("tokenizer.chat_template.rerank").value_or("");
        size_t checked = 0, differing = 0;
        auto check = [&](const std::vector<int32_t>& mine, const Value& theirs, const std::string& name) {
            const std::vector<int32_t> expected = ints(theirs);
            ++checked;
            if (const long at = first_difference(mine, expected); at >= 0) {
                ++differing;
                std::printf("  ids differ: %s at token %ld (%zu vs %zu tokens): %d vs %d\n", name.c_str(), at,
                            mine.size(), expected.size(), at < static_cast<long>(mine.size()) ? mine[at] : -1,
                            at < static_cast<long>(expected.size()) ? expected[at] : -1);
            }
        };
        if (!rerank) {
            for (size_t i = 0; i < texts.size() && i < ref_ids->items().size(); ++i) {
                check(ids_of(texts[i]), ref_ids->items()[i], "input " + std::to_string(i));
            }
        } else {
            for (size_t c = 0; c < rerank_cases.size() && c < ref_ids->items().size(); ++c) {
                const Value& case_ids = ref_ids->items()[c];
                for (size_t d = 0; d < rerank_cases[c].documents.size() && d < case_ids.items().size(); ++d) {
                    auto prompt = liyab::format_rerank_prompt(tmpl, rerank_cases[c].query, rerank_cases[c].documents[d]);
                    check(ids_of(prompt ? *prompt : std::string()), case_ids.items()[d],
                          "case " + std::to_string(c) + " document " + std::to_string(d));
                }
            }
        }
        std::printf("token ids: %zu of %zu inputs as the reference's\n", checked - differing, checked);
        ok = ok && differing == 0;
    }

    if (!ref_path.empty() && !rerank) {
        const char* key = normalize ? (embed_options.pooling == liyab::Pooling::Mean ? "embed_mean" : "embed")
                                    : "embed_last_raw";
        const Value* ref_vectors = field(ref, key);
        if (ref_vectors == nullptr || ref_vectors->items().size() != vectors.size()) {
            std::fprintf(stderr, "the reference has no %s for %zu texts\n", key, vectors.size());
            return 2;
        }
        double worst = 1.0, sum = 0.0, worst_rel = 0.0;
        size_t worst_at = 0;
        for (size_t i = 0; i < vectors.size(); ++i) {
            const std::vector<float> expected = floats(ref_vectors->items()[i]);
            const double c = cosine(vectors[i], expected);
            double diff = 0.0, norm = 0.0;
            for (size_t j = 0; j < expected.size() && j < vectors[i].size(); ++j) {
                diff += (static_cast<double>(vectors[i][j]) - expected[j]) * (static_cast<double>(vectors[i][j]) - expected[j]);
                norm += static_cast<double>(expected[j]) * expected[j];
            }
            const double rel = norm > 0.0 ? std::sqrt(diff / norm) : 0.0;
            worst_rel = std::max(worst_rel, rel);
            sum += c;
            if (c < worst) {
                worst = c;
                worst_at = i;
            }
            if (!quiet) std::printf("  %2zu  cosine %.8f  relative L2 error %.2e\n", i, c, rel);
        }
        std::printf("%s against the reference: cosine min %.8f (input %zu), mean %.8f; relative L2 error max %.2e\n",
                    key, worst, worst_at, sum / static_cast<double>(vectors.size()), worst_rel);
        if (worst < min_cosine) {
            std::printf("FAIL: cosine below %.4f\n", min_cosine);
            ok = false;
        }
    }
    if (!ref_path.empty() && rerank) {
        const Value* ref_scores = field(ref, "rerank");
        if (ref_scores == nullptr || ref_scores->items().size() != scores.size()) {
            std::fprintf(stderr, "the reference has no rerank scores for %zu cases\n", scores.size());
            return 2;
        }
        double worst = 0.0;
        for (size_t c = 0; c < scores.size(); ++c) {
            const std::vector<float> expected = floats(ref_scores->items()[c]);
            for (size_t d = 0; d < scores[c].size() && d < expected.size(); ++d) {
                const double diff = std::fabs(static_cast<double>(scores[c][d]) - expected[d]);
                worst = std::max(worst, diff);
                if (!quiet) {
                    std::printf("  case %zu document %zu  score %.6f  reference %.6f  |diff| %.2e\n", c, d, scores[c][d],
                                expected[d], diff);
                }
            }
        }
        std::printf("scores against the reference: largest |diff| %.2e\n", worst);
        if (worst > max_score_diff) {
            std::printf("FAIL: a score is more than %.0e from the reference\n", max_score_diff);
            ok = false;
        }
    }

    // Each input on its own must give the bits it gave in the batch.
    if (check_alone) {
        size_t same = 0, total_inputs = 0;
        if (!rerank) {
            for (size_t i = 0; i < texts.size(); ++i) {
                const std::string_view one = texts[i];
                auto r = engine.embed(std::span<const std::string_view>(&one, 1), embed_options);
                ++total_inputs;
                if (r && r->front() == vectors[i]) ++same;
            }
        } else {
            for (size_t c = 0; c < rerank_cases.size(); ++c) {
                for (size_t d = 0; d < rerank_cases[c].documents.size(); ++d) {
                    const std::string_view one = rerank_cases[c].documents[d];
                    auto r = engine.rerank(rerank_cases[c].query, std::span<const std::string_view>(&one, 1),
                                           rerank_options);
                    ++total_inputs;
                    if (r && r->front() == scores[c][d]) ++same;
                }
            }
        }
        std::printf("alone vs in the batch: %zu of %zu the same bits\n", same, total_inputs);
    }

    if (bench > 0) {
        std::vector<double> times;
        for (int b = 0; b < bench; ++b) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!run_all()) return 1;
            times.push_back(seconds_since(t0));
        }
        std::sort(times.begin(), times.end());
        const double median = times[times.size() / 2], best = times.front();
        std::printf("throughput (%d runs): median %.1f %s/s, %.0f tokens/s (best %.1f %s/s)\n", bench,
                    static_cast<double>(inputs) / median, rerank ? "pairs" : "texts",
                    static_cast<double>(total.tokens) / median, static_cast<double>(inputs) / best,
                    rerank ? "pairs" : "texts");
    }
    return ok ? 0 : 1;
}
