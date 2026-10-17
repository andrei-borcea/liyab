// Pooling models: embeddings and reranking (core/pooling.h, Engine::embed /
// rerank, the C API) on synthetic GGUFs. The vectors are checked against the
// transformer run on its own (last-token states are the LM head's input; a
// mean is the mean of the rows; a ranking head's softmax over its labels),
// and the engine's own rules: EOS from the metadata, the same bits alone or in
// any batch and order with prefixes shared, refusals, the context left empty
// and a conversation set aside, hybrid models and sliding windows, the KV
// format of pooling models. tools/liyab-embed checks real models against
// Hugging Face transformers.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <deque>
#include <string>
#include <thread>
#include <vector>

#include "core/architecture.h"
#include "core/pooling.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "core/transformer.h"
#include "liyab/liyab.h"
#include "liyab/liyab_c_api.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;

namespace {

std::string temp_path(const std::string& name) {
    return test::temp_dir() + "/liyab_pooling_" + std::to_string(test::process_id()) + "_" + name;
}

// Models written once per process and removed at exit.
const std::string& model_path(const std::string& name, const test::TinyModelSpec& spec) {
    struct Written {
        std::deque<std::pair<std::string, std::string>> files;
        ~Written() {
            for (const auto& file : files) std::remove(file.second.c_str());
        }
    };
    static Written written;
    for (const auto& [n, p] : written.files) {
        if (n == name) return p;
    }
    written.files.emplace_back(name, temp_path(name + ".gguf"));
    test::write_tiny_model(written.files.back().second, spec);
    return written.files.back().second;
}

// Float32 weights, so independent computations agree closely.
test::TinyModelSpec f32_spec() {
    test::TinyModelSpec s;
    s.attn_type = s.ffn_type = s.embd_type = s.output_type = DType::F32;
    return s;
}

// An embedding model: last-token pooling, inputs end with EOS.
test::TinyModelSpec embedding_spec(int32_t add_eos = 1) {
    test::TinyModelSpec s = f32_spec();
    s.pooling_type = 3;
    s.add_eos = add_eos;
    return s;
}

constexpr char kRerankTemplate[] = "query: {query} document: {document}";

// A reranker: a two-label head [yes, no] and a rerank prompt.
test::TinyModelSpec reranker_spec() {
    test::TinyModelSpec s = f32_spec();
    s.pooling_type = 4;
    s.cls_rows = 2;
    s.cls_labels = {"yes", "no"};
    s.rerank_template = kRerankTemplate;
    return s;
}

EngineConfig cpu_config(const std::string& path) {
    EngineConfig c;
    c.model_path = path;
    c.n_threads = 4;
    c.thermal_polling = false;
    c.power.profile = PowerProfile::Performance;
    c.backend = BackendKind::Cpu;  // the bit-exactness the cases check is the CPU backend's
    return c;
}

std::unique_ptr<Engine> make_engine(const EngineConfig& config) {
    auto e = Engine::create(config);
    if (!e) {
        std::printf("  engine: %s\n", e.status().to_string().c_str());
        return nullptr;
    }
    return std::move(e).value();
}

std::vector<int32_t> tokens_of(Engine& engine, std::string_view text, bool eos) {
    auto ids = engine.tokenize(text, true);
    std::vector<int32_t> out = ids ? ids.value() : std::vector<int32_t>{};
    if (eos) out.push_back(2);  // the tiny vocabulary's </s>
    return out;
}

// The transformer on its own: runs `tokens` from an empty context (F16 KV,
// as the engine stores a pooling model's cache) in passes of 64 and returns
// the final hidden state of every token (rows of n_embd floats).
std::vector<float> hidden_rows(const std::string& path, const std::vector<int32_t>& tokens) {
    auto file = MmapLoader::open(path);
    if (!file) return {};
    TransformerOptions options;
    options.kv_type = KvCacheType::F16;
    auto model = Transformer::load(std::move(file).value(), options);
    if (!model) return {};
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const auto d = static_cast<size_t>((*model)->config().n_embd);
    std::vector<float> rows(tokens.size() * d);
    for (size_t i = 0; i < tokens.size(); i += 64) {
        const size_t n = std::min<size_t>(64, tokens.size() - i);
        if (!(*model)->forward(std::span<const int32_t>(tokens.data() + i, n), Transformer::Logits::None, route, pool)) {
            return {};
        }
        std::vector<int32_t> ids(n);
        for (size_t r = 0; r < n; ++r) ids[r] = static_cast<int32_t>(r);
        if (!(*model)->final_hidden(ids, std::span<float>(rows.data() + i * d, n * d)).is_ok()) return {};
    }
    return rows;
}

// A 2-D F32 tensor of `path` as floats (rows of ne[0]).
std::vector<float> tensor_values(const std::string& path, const std::string& name) {
    auto file = MmapLoader::open(path);
    if (!file) return {};
    const TensorView* t = (*file)->tensor(name);
    if (t == nullptr || t->type != DType::F32) return {};
    const auto* p = reinterpret_cast<const float*>(t->data);
    return {p, p + t->elements()};
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m;
}

double norm(const std::vector<float>& v) {
    double ss = 0.0;
    for (const float x : v) ss += static_cast<double>(x) * x;
    return std::sqrt(ss);
}

Result<std::vector<std::vector<float>>> embed(Engine& e, const std::vector<std::string>& texts,
                                              const EmbedOptions& options = {}, PoolingStats* stats = nullptr) {
    const std::vector<std::string_view> views(texts.begin(), texts.end());
    return e.embed(views, options, stats);
}

Result<std::vector<float>> rerank(Engine& e, std::string_view query, const std::vector<std::string>& documents,
                                  const RerankOptions& options = {}, PoolingStats* stats = nullptr) {
    const std::vector<std::string_view> views(documents.begin(), documents.end());
    return e.rerank(query, views, options, stats);
}

// "hello" repeated: one ▁hello token each, so the token count is known.
std::string hellos(int n) {
    std::string s;
    for (int i = 0; i < n; ++i) s += i == 0 ? "hello" : " hello";
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// What a GGUF declares
// ---------------------------------------------------------------------------
TEST_CASE("pooling metadata: type, EOS, ranking head, labels and prompt; unusable heads are reported") {
    // The files stay open: a PoolingModel's head points into its file's tensors.
    std::vector<std::unique_ptr<MmapLoader>> files;
    auto read = [&](const test::TinyModelSpec& spec, const std::string& name) -> Result<PoolingModel> {
        auto file = MmapLoader::open(model_path(name, spec));
        if (!file) return file.status();
        files.push_back(std::move(file).value());
        const MmapLoader& f = *files.back();
        auto config = describe_model(f);
        if (!config) return config.status();
        CHECK(declares_pooling(f) == (spec.pooling_type >= 1 && spec.pooling_type <= 4));
        return read_pooling_model(f, *config);
    };
    auto plain = read(f32_spec(), "plain");
    REQUIRE(plain.has_value());
    CHECK(plain->declared == Pooling::None && !plain->add_eos && plain->head == nullptr && plain->head_error.is_ok());

    auto embedding = read(embedding_spec(), "embedding");
    REQUIRE(embedding.has_value());
    CHECK(embedding->declared == Pooling::Last && embedding->add_eos && embedding->eos == 2);

    auto ranker = read(reranker_spec(), "reranker");
    REQUIRE(ranker.has_value());
    CHECK(ranker->declared == Pooling::Rank && ranker->head != nullptr && ranker->head->rows() == 2);
    CHECK(ranker->labels == std::vector<std::string>({"yes", "no"}));
    CHECK(ranker->rerank_template == kRerankTemplate && ranker->head_bias.empty());

    test::TinyModelSpec biased = reranker_spec();
    biased.cls_bias = true;
    auto with_bias = read(biased, "reranker_bias");
    REQUIRE(with_bias.has_value());
    CHECK(with_bias->head != nullptr && with_bias->head_bias.size() == 2);

    test::TinyModelSpec headless = reranker_spec();
    headless.cls_rows = 0;
    auto no_head = read(headless, "reranker_headless");
    REQUIRE(no_head.has_value());
    CHECK(no_head->head == nullptr && no_head->head_error.code() == ErrorCode::Unsupported);

    test::TinyModelSpec mismatched = reranker_spec();
    mismatched.cls_rows = 3;
    auto bad_rows = read(mismatched, "reranker_rows");
    REQUIRE(bad_rows.has_value());
    CHECK(bad_rows->head == nullptr && bad_rows->head_error.code() == ErrorCode::InvalidModel);

    test::TinyModelSpec two_layer = reranker_spec();
    two_layer.cls_dense = true;
    auto dense = read(two_layer, "reranker_dense");
    REQUIRE(dense.has_value());
    CHECK(dense->head == nullptr && dense->head_error.code() == ErrorCode::Unsupported);

    // A pooling type outside the GGUF numbering: refused by the read, but the
    // model still loads, and only the pooling calls report it.
    test::TinyModelSpec odd = f32_spec();
    odd.pooling_type = 7;
    CHECK(read(odd, "pooling_7").status().code() == ErrorCode::InvalidModel);
    auto engine = make_engine(cpu_config(model_path("pooling_7", odd)));
    REQUIRE(engine != nullptr);
    CHECK(embed(*engine, {"hello"}).status().code() == ErrorCode::InvalidModel);
    CHECK(engine->pooling() == Pooling::None);
}

TEST_CASE("rerank prompt templates: each placeholder replaced once, in the template only") {
    auto p = format_rerank_prompt("Q: {query}\nD: {document}!", "what?", "this.");
    REQUIRE(p.has_value());
    CHECK(*p == "Q: what?\nD: this.!");
    // Document first; a query that spells a placeholder stays text.
    auto swapped = format_rerank_prompt("{document} | {query}", "{document}", "doc");
    REQUIRE(swapped.has_value());
    CHECK(*swapped == "doc | {document}");
    auto repeated = format_rerank_prompt("{query}{document}{query}", "a", "b");
    REQUIRE(repeated.has_value());
    CHECK(*repeated == "ab{query}");
    CHECK(format_rerank_prompt("{query} only", "a", "b").status().code() == ErrorCode::InvalidArgument);
    CHECK(format_rerank_prompt("{document} only", "a", "b").status().code() == ErrorCode::InvalidArgument);
}

TEST_CASE("finishing vectors and scores: dimensions, unit length, softmax of the first label, sigmoid of one") {
    const std::vector<float> values = {3, 4, 12, 0, 0, 0, 0, 0};  // two vectors of 4
    auto all = finish_embeddings(values, 4, 0, true);
    REQUIRE(all.size() == 2);
    CHECK(std::fabs(all[0][0] - 3.0f / 13.0f) < 1e-7f && std::fabs(all[0][2] - 12.0f / 13.0f) < 1e-7f);
    CHECK(all[1] == std::vector<float>(4, 0.0f));  // a zero vector stays zero
    auto two = finish_embeddings(values, 4, 2, true);
    CHECK(two[0].size() == 2 && std::fabs(two[0][0] - 0.6f) < 1e-7f && std::fabs(two[0][1] - 0.8f) < 1e-7f);
    auto raw = finish_embeddings(values, 4, 3, false);
    CHECK(raw[0] == std::vector<float>({3, 4, 12}));

    auto pairs = label_scores(std::vector<float>{2.0f, -1.0f, 0.0f, 0.0f, -800.0f, 800.0f}, 2);
    REQUIRE(pairs.size() == 3);
    CHECK(std::fabs(pairs[0] - 1.0f / (1.0f + std::exp(-3.0f))) < 1e-7f);
    CHECK(pairs[1] == 0.5f);
    CHECK(pairs[2] == 0.0f);  // no overflow on extreme logits
    auto single = label_scores(std::vector<float>{0.0f, 2.0f}, 1);
    CHECK(single.size() == 2 && single[0] == 0.5f && std::fabs(single[1] - 1.0f / (1.0f + std::exp(-2.0f))) < 1e-7f);
    auto three = label_scores(std::vector<float>{1.0f, 1.0f, 1.0f}, 3);
    CHECK(three.size() == 1 && std::fabs(three[0] - 1.0f / 3.0f) < 1e-7f);
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------
TEST_CASE("embed: a last-token vector is the LM head's input; unit length; dimensions") {
    const std::string& path = model_path("embedding", embedding_spec());
    auto engine = make_engine(cpu_config(path));
    REQUIRE(engine != nullptr);
    CHECK(engine->pooling() == Pooling::Last);
    const auto d = static_cast<size_t>(engine->embedding_size());
    CHECK(d == 128);

    EmbedOptions raw;
    raw.normalize = false;
    auto pooled = embed(*engine, {"hello world"}, raw);
    REQUIRE(pooled.has_value() && pooled->size() == 1 && pooled->front().size() == d);
    const std::vector<float>& h = pooled->front();

    // The output head applied to the vector gives the transformer's own
    // logits for the last token: the vector is the state the LM head reads.
    const std::vector<int32_t> tokens = tokens_of(*engine, "hello world", true);
    auto file = MmapLoader::open(path);
    REQUIRE(file.has_value());
    TransformerOptions options;
    options.kv_type = KvCacheType::F16;
    auto model = Transformer::load(std::move(file).value(), options);
    REQUIRE(model.has_value());
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    auto logits = (*model)->forward(tokens, Transformer::Logits::Last, route, pool);
    REQUIRE(logits.has_value());
    const std::vector<float> output = tensor_values(path, "output.weight");
    REQUIRE(output.size() == logits->size() * d);
    double worst = 0.0;
    for (size_t r = 0; r < logits->size(); ++r) {
        double dot = 0.0;
        for (size_t j = 0; j < d; ++j) dot += static_cast<double>(output[r * d + j]) * h[j];
        worst = std::max(worst, std::fabs(dot - (*logits)[r]));
    }
    CHECK(worst < 1e-4);
    // ... and final_hidden() of that pass is the vector, bit for bit.
    std::vector<float> last(d);
    const int32_t row = static_cast<int32_t>(tokens.size()) - 1;
    REQUIRE((*model)->final_hidden(std::span<const int32_t>(&row, 1), last).is_ok());
    CHECK(last == h);
    const int32_t outside = row + 1;
    CHECK((*model)->final_hidden(std::span<const int32_t>(&outside, 1), last).code() == ErrorCode::InvalidArgument);

    auto unit = embed(*engine, {"hello world"});
    REQUIRE(unit.has_value());
    CHECK(std::fabs(norm(unit->front()) - 1.0) < 1e-6);
    const double length = norm(h);
    for (size_t j = 0; j < d; ++j) CHECK(std::fabs(unit->front()[j] - h[j] / length) < 1e-6);

    EmbedOptions shorter;
    shorter.dimensions = 16;
    auto first16 = embed(*engine, {"hello world"}, shorter);
    REQUIRE(first16.has_value() && first16->front().size() == 16);
    CHECK(std::fabs(norm(first16->front()) - 1.0) < 1e-6);
    const std::vector<float> head(h.begin(), h.begin() + 16);
    for (size_t j = 0; j < 16; ++j) CHECK(std::fabs(first16->front()[j] - head[j] / norm(head)) < 1e-6);
    shorter.dimensions = static_cast<int32_t>(d) + 1;
    CHECK(embed(*engine, {"hello"}, shorter).status().code() == ErrorCode::InvalidArgument);
    shorter.dimensions = -1;
    CHECK(embed(*engine, {"hello"}, shorter).status().code() == ErrorCode::InvalidArgument);
    auto none = embed(*engine, {});
    CHECK(none.has_value() && none->empty());
}

TEST_CASE("embed: EOS from the metadata; the same bits alone, in any batch or order; prefixes shared") {
    auto with_eos = make_engine(cpu_config(model_path("embedding", embedding_spec(1))));
    auto without_eos = make_engine(cpu_config(model_path("embedding_no_eos", embedding_spec(0))));
    REQUIRE(with_eos != nullptr && without_eos != nullptr);
    // The same weights: an EOS the metadata asks for is the EOS written in the text.
    auto a = embed(*with_eos, {"hello"});
    auto b = embed(*without_eos, {"hello</s>"});
    REQUIRE(a.has_value() && b.has_value());
    CHECK(a->front() == b->front());
    auto c = embed(*without_eos, {"hello"});
    REQUIRE(c.has_value());
    CHECK(c->front() != a->front());

    // Shared prefixes (one longer than a 64-token pass), duplicates, a
    // sequence that is a prefix of another.
    const std::vector<std::string> texts = {
        hellos(70) + " world", "world hello", hellos(70), "hello world", "hello", hellos(3), "hello world",
        "world", hellos(70) + " world world"};
    PoolingStats stats;
    auto batch = embed(*with_eos, texts, {}, &stats);
    REQUIRE(batch.has_value() && batch->size() == texts.size());
    int32_t tokens = 0;
    for (const std::string& t : texts) tokens += static_cast<int32_t>(tokens_of(*with_eos, t, true).size());
    CHECK(stats.tokens == tokens);
    CHECK(stats.reused_tokens > 140);  // the 70 hellos at least twice
    CHECK(stats.ms > 0.0);
    for (size_t i = 0; i < texts.size(); ++i) {
        auto alone = embed(*with_eos, {texts[i]});
        REQUIRE(alone.has_value());
        CHECK(alone->front() == (*batch)[i]);
    }
    std::vector<std::string> reversed(texts.rbegin(), texts.rend());
    auto backwards = embed(*with_eos, reversed);
    REQUIRE(backwards.has_value());
    for (size_t i = 0; i < texts.size(); ++i) CHECK((*backwards)[texts.size() - 1 - i] == (*batch)[i]);
    CHECK((*batch)[3] == (*batch)[6]);  // the duplicate
}

TEST_CASE("embed: mean pooling is the mean of every token's final state, across passes") {
    const std::string& path = model_path("embedding", embedding_spec());
    auto engine = make_engine(cpu_config(path));
    REQUIRE(engine != nullptr);
    const std::vector<std::string> texts = {"hello world", hellos(90) + " world"};
    EmbedOptions mean;
    mean.pooling = Pooling::Mean;
    mean.normalize = false;
    PoolingStats stats;
    auto pooled = embed(*engine, texts, mean, &stats);
    REQUIRE(pooled.has_value());
    CHECK(stats.reused_tokens == 0);  // a mean needs every row: nothing is shared
    const auto d = static_cast<size_t>(engine->embedding_size());
    for (size_t i = 0; i < texts.size(); ++i) {
        const std::vector<int32_t> tokens = tokens_of(*engine, texts[i], true);
        const std::vector<float> rows = hidden_rows(path, tokens);
        REQUIRE(rows.size() == tokens.size() * d);
        std::vector<float> expected(d);
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t r = 0; r < tokens.size(); ++r) sum += rows[r * d + j];
            expected[j] = static_cast<float>(sum / static_cast<double>(tokens.size()));
        }
        CHECK(max_abs_diff((*pooled)[i], expected) < 1e-5);
    }
    // The model's own pooling (Last) is the last row of the same states.
    EmbedOptions last;
    last.normalize = false;
    auto lasts = embed(*engine, texts, last);
    REQUIRE(lasts.has_value());
    const std::vector<int32_t> tokens = tokens_of(*engine, texts[1], true);
    const std::vector<float> rows = hidden_rows(path, tokens);
    REQUIRE(!rows.empty());
    CHECK((*lasts)[1] == std::vector<float>(rows.end() - static_cast<std::ptrdiff_t>(d), rows.end()));
}

TEST_CASE("embed: which pooling applies, inputs longer than the context, truncation") {
    auto generative = make_engine(cpu_config(model_path("plain", f32_spec())));
    REQUIRE(generative != nullptr);
    CHECK(generative->pooling() == Pooling::None);
    CHECK(embed(*generative, {"hello"}).status().code() == ErrorCode::Unsupported);
    EmbedOptions last;
    last.pooling = Pooling::Last;
    CHECK(embed(*generative, {"hello"}, last).has_value());  // explicit pooling: any model
    for (const Pooling p : {Pooling::None, Pooling::Cls, Pooling::Rank}) {
        EmbedOptions o;
        o.pooling = p;
        CHECK(embed(*generative, {"hello"}, o).status().code() == ErrorCode::InvalidArgument);
    }
    auto ranker = make_engine(cpu_config(model_path("reranker", reranker_spec())));
    REQUIRE(ranker != nullptr);
    CHECK(embed(*ranker, {"hello"}).status().code() == ErrorCode::Unsupported);
    test::TinyModelSpec cls = f32_spec();
    cls.pooling_type = 2;
    auto first_token = make_engine(cpu_config(model_path("pooling_cls", cls)));
    REQUIRE(first_token != nullptr);
    CHECK(embed(*first_token, {"hello"}).status().code() == ErrorCode::Unsupported);

    // A context of 32: BOS + 31 hellos + EOS does not fit; cut, it is BOS + 30 hellos + EOS.
    EngineConfig small = cpu_config(model_path("embedding", embedding_spec()));
    small.context_length = 32;
    auto engine = make_engine(small);
    REQUIRE(engine != nullptr && engine->context_length() == 32);
    auto refused = embed(*engine, {"hello", hellos(31)});
    CHECK(refused.status().code() == ErrorCode::InvalidArgument);
    CHECK(refused.status().message().find("input 1") != std::string::npos);
    EmbedOptions cut;
    cut.truncate = true;
    auto truncated = embed(*engine, {hellos(31)}, cut);
    auto fitting = embed(*engine, {hellos(30)});
    REQUIRE(truncated.has_value() && fitting.has_value());
    CHECK(truncated->front() == fitting->front());
}

TEST_CASE("embed: the context is left empty; a conversation in progress is set aside in the prefix cache") {
    EngineConfig config = cpu_config(model_path("plain", f32_spec()));
    config.prefix_cache_mb = 64;
    auto engine = make_engine(config);
    REQUIRE(engine != nullptr);
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.max_tokens = 6;
    auto reply = [&](Engine& e, const std::string& prompt, GenerationStats* stats = nullptr) {
        std::string out;
        auto r = e.generate(prompt, greedy, [&](std::string_view piece, int32_t) {
            out += piece;
            return true;
        });
        if (r && stats != nullptr) *stats = r.value();
        return out;
    };
    const std::string prompt = hellos(150) + " world";
    const std::string first = reply(*engine, prompt);
    EmbedOptions last;
    last.pooling = Pooling::Last;
    REQUIRE(embed(*engine, {"world hello", "hello"}, last).has_value());
    // The next turn continues the conversation: restored, not processed again,
    // and the reply is the one a fresh engine gives.
    const std::string next = prompt + first + " world hello";
    GenerationStats stats;
    const std::string continued = reply(*engine, next, &stats);
    CHECK(stats.cached_prefix_tokens >= 150);
    auto fresh = make_engine(cpu_config(model_path("plain", f32_spec())));
    REQUIRE(fresh != nullptr);
    CHECK(reply(*fresh, next) == continued);
    // Without a prefix cache the context is simply empty: same reply again.
    auto plain = make_engine(cpu_config(model_path("plain", f32_spec())));
    REQUIRE(plain != nullptr);
    const std::string before = reply(*plain, prompt);
    REQUIRE(embed(*plain, {"world"}, last).has_value());
    CHECK(reply(*plain, prompt) == before);
}

TEST_CASE("embed: a pooling model's KV cache is F16 by default (pooling_kv_cache_type), others keep theirs") {
    auto pooled = make_engine(cpu_config(model_path("embedding", embedding_spec())));
    auto generative = make_engine(cpu_config(model_path("plain", f32_spec())));
    REQUIRE(pooled != nullptr && generative != nullptr);
    CHECK(pooled->describe().find("KV cache:   f16") != std::string::npos);
    CHECK(generative->describe().find("KV cache:   q8_0") != std::string::npos);
    CHECK(pooled->describe().find("Pooling:    embeddings, last token, 128 values per vector, inputs end with EOS\n") !=
          std::string::npos);
    CHECK(generative->describe().find("Pooling:") == std::string::npos);
    auto ranker = make_engine(cpu_config(model_path("reranker", reranker_spec())));
    REQUIRE(ranker != nullptr);
    CHECK(ranker->describe().find("Pooling:    reranking, ranking head of 2 labels\n") != std::string::npos);
    EngineConfig quantized = cpu_config(model_path("embedding", embedding_spec()));
    quantized.pooling_kv_cache_type = KvCacheType::Q8_0;
    auto q8 = make_engine(quantized);
    REQUIRE(q8 != nullptr);
    CHECK(q8->describe().find("KV cache:   q8_0") != std::string::npos);
    auto a = embed(*pooled, {"hello world"});
    auto b = embed(*q8, {"hello world"});
    REQUIRE(a.has_value() && b.has_value());
    double dot = 0.0;
    for (size_t j = 0; j < a->front().size(); ++j) dot += static_cast<double>(a->front()[j]) * b->front()[j];
    CHECK(dot > 0.99);  // close, not the same bits
}

TEST_CASE("embed: hybrid models start every input over; sliding windows; still the same bits alone") {
    test::TinyModelSpec hybrid = embedding_spec();
    hybrid.arch = "qwen35";
    hybrid.delta_net_interval = 2;
    hybrid.n_layers = 4;
    auto engine = make_engine(cpu_config(model_path("embedding_hybrid", hybrid)));
    REQUIRE(engine != nullptr);
    const std::vector<std::string> texts = {"hello world", "hello world hello", hellos(80), hellos(80) + " world"};
    PoolingStats stats;
    auto batch = embed(*engine, texts, {}, &stats);
    REQUIRE(batch.has_value());
    CHECK(stats.reused_tokens == 0);  // recurrent states do not rewind to an arbitrary position
    for (size_t i = 0; i < texts.size(); ++i) {
        auto alone = embed(*engine, {texts[i]});
        REQUIRE(alone.has_value());
        CHECK(alone->front() == (*batch)[i]);
    }

    EngineConfig windowed = cpu_config(model_path("embedding", embedding_spec()));
    windowed.sliding_window = 16;
    windowed.kv_sink_tokens = 4;
    auto window = make_engine(windowed);
    REQUIRE(window != nullptr);
    const std::vector<std::string> long_texts = {hellos(150) + " world", hellos(150) + " world world", hellos(20)};
    auto long_batch = embed(*window, long_texts);
    REQUIRE(long_batch.has_value());
    for (size_t i = 0; i < long_texts.size(); ++i) {
        auto alone = embed(*window, {long_texts[i]});
        REQUIRE(alone.has_value());
        CHECK(alone->front() == (*long_batch)[i]);
    }
}

TEST_CASE("embed: cancel() stops a call; the engine works on afterwards") {
    auto engine = make_engine(cpu_config(model_path("embedding", embedding_spec())));
    REQUIRE(engine != nullptr);
    std::vector<std::string> texts;
    for (int i = 0; i < 40; ++i) texts.push_back(hellos(60 + i));
    std::atomic<bool> done{false};
    std::thread canceller([&] {
        while (!done.load()) engine->cancel();
    });
    auto cancelled = embed(*engine, texts);
    done.store(true);
    canceller.join();
    CHECK(cancelled.status().code() == ErrorCode::Cancelled);
    auto again = embed(*engine, {texts[0]});
    auto fresh = make_engine(cpu_config(model_path("embedding", embedding_spec())));
    REQUIRE(again.has_value() && fresh != nullptr);
    auto expected = embed(*fresh, {texts[0]});
    REQUIRE(expected.has_value());
    CHECK(again->front() == expected->front());
}

// ---------------------------------------------------------------------------
// Reranking
// ---------------------------------------------------------------------------
TEST_CASE("rerank: the ranking head's softmax over its labels, from the last token of the pair's prompt") {
    for (const bool bias : {false, true}) {
        test::TinyModelSpec spec = reranker_spec();
        spec.cls_bias = bias;
        const std::string& path = model_path(bias ? "reranker_bias" : "reranker", spec);
        auto engine = make_engine(cpu_config(path));
        REQUIRE(engine != nullptr);
        CHECK(engine->pooling() == Pooling::Rank);
        const std::vector<std::string> documents = {"world", "hello world", hellos(80), "hello"};
        PoolingStats stats;
        auto scores = rerank(*engine, "hello hello", documents, {}, &stats);
        REQUIRE(scores.has_value() && scores->size() == documents.size());
        CHECK(stats.reused_tokens > 0);  // the template's start and the query
        const std::vector<float> head = tensor_values(path, "cls.output.weight");
        const std::vector<float> head_bias = bias ? tensor_values(path, "cls.output.bias") : std::vector<float>(2, 0.0f);
        const auto d = static_cast<size_t>(engine->embedding_size());
        REQUIRE(head.size() == 2 * d && head_bias.size() == 2);
        for (size_t i = 0; i < documents.size(); ++i) {
            auto prompt = format_rerank_prompt(kRerankTemplate, "hello hello", documents[i]);
            REQUIRE(prompt.has_value());
            const std::vector<int32_t> tokens = tokens_of(*engine, *prompt, false);
            const std::vector<float> rows = hidden_rows(path, tokens);
            REQUIRE(!rows.empty());
            const float* h = rows.data() + rows.size() - d;
            double logit[2];
            for (size_t l = 0; l < 2; ++l) {
                logit[l] = head_bias[l];
                for (size_t j = 0; j < d; ++j) logit[l] += static_cast<double>(head[l * d + j]) * h[j];
            }
            const double expected = 1.0 / (1.0 + std::exp(logit[1] - logit[0]));
            CHECK(std::fabs((*scores)[i] - expected) < 1e-5);
            auto alone = rerank(*engine, "hello hello", {documents[i]});
            REQUIRE(alone.has_value());
            CHECK(alone->front() == (*scores)[i]);
        }
    }
    // One output: its sigmoid.
    test::TinyModelSpec single = reranker_spec();
    single.cls_rows = 1;
    single.cls_labels.clear();
    const std::string& path = model_path("reranker_single", single);
    auto engine = make_engine(cpu_config(path));
    REQUIRE(engine != nullptr);
    auto scores = rerank(*engine, "hello", {"world"});
    REQUIRE(scores.has_value());
    auto prompt = format_rerank_prompt(kRerankTemplate, "hello", "world");
    const std::vector<float> rows = hidden_rows(path, tokens_of(*engine, *prompt, false));
    const std::vector<float> head = tensor_values(path, "cls.output.weight");
    const auto d = static_cast<size_t>(engine->embedding_size());
    REQUIRE(!rows.empty() && head.size() == d);
    double logit = 0.0;
    for (size_t j = 0; j < d; ++j) logit += static_cast<double>(head[j]) * rows[rows.size() - d + j];
    CHECK(std::fabs(scores->front() - 1.0 / (1.0 + std::exp(-logit))) < 1e-5);
}

TEST_CASE("rerank: labels score with the LM head as a head of the same rows does; templates and refusals") {
    // A head made of the output head's rows of ▁hello and ▁world.
    test::TinyModelSpec spec = reranker_spec();
    spec.cls_labels = {"hello", "world"};
    spec.cls_rows_from = {test::tiny_token("\xE2\x96\x81hello"), test::tiny_token("\xE2\x96\x81world")};
    auto engine = make_engine(cpu_config(model_path("reranker_lm_rows", spec)));
    REQUIRE(engine != nullptr);
    const std::vector<std::string> documents = {"world", "hello world", hellos(70)};
    auto by_head = rerank(*engine, "hello", documents);
    RerankOptions labels;
    labels.labels = {"hello", "world"};
    auto by_labels = rerank(*engine, "hello", documents, labels);
    REQUIRE(by_head.has_value() && by_labels.has_value());
    for (size_t i = 0; i < documents.size(); ++i) CHECK(std::fabs((*by_head)[i] - (*by_labels)[i]) < 1e-6);

    // A generative model: labels and a template make it a reranker.
    auto generative = make_engine(cpu_config(model_path("plain", f32_spec())));
    REQUIRE(generative != nullptr);
    CHECK(rerank(*generative, "hello", {"world"}).status().code() == ErrorCode::InvalidArgument);  // no template
    RerankOptions own;
    own.prompt_template = kRerankTemplate;
    CHECK(rerank(*generative, "hello", {"world"}, own).status().code() == ErrorCode::Unsupported);  // no head
    own.labels = {"hello", "world"};
    auto scored = rerank(*generative, "hello", {"world", "hello"}, own);
    REQUIRE(scored.has_value());
    for (const float s : *scored) CHECK(s > 0.0f && s < 1.0f);
    own.labels = {"hello world", "world"};  // two tokens
    CHECK(rerank(*generative, "hello", {"world"}, own).status().code() == ErrorCode::InvalidArgument);
    own.labels = {"hello", "world"};
    own.prompt_template = "{query} without a document";
    CHECK(rerank(*generative, "hello", {"world"}, own).status().code() == ErrorCode::InvalidArgument);

    // A template given replaces the model's; a pair too long for the context is refused.
    auto ranker = make_engine(cpu_config(model_path("reranker", reranker_spec())));
    REQUIRE(ranker != nullptr);
    RerankOptions other;
    other.prompt_template = "{document} vs {query}";
    auto a = rerank(*ranker, "hello", {"world"});
    auto b = rerank(*ranker, "hello", {"world"}, other);
    REQUIRE(a.has_value() && b.has_value());
    CHECK(a->front() != b->front());
    EngineConfig small = cpu_config(model_path("reranker", reranker_spec()));
    small.context_length = 32;
    auto tight = make_engine(small);
    REQUIRE(tight != nullptr);
    auto refused = rerank(*tight, "hello", {"world", hellos(40)});
    CHECK(refused.status().code() == ErrorCode::InvalidArgument);
    CHECK(refused.status().message().find("document 1") != std::string::npos);
    auto none = rerank(*ranker, "hello", {});
    CHECK(none.has_value() && none->empty());
}

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------
TEST_CASE("C API: embed and rerank, parameters, buffers and errors") {
    liyab_embed_params params;
    liyab_embed_params_default(&params);
    CHECK(params.pooling == LIYAB_POOLING_AUTO && params.dimensions == 0 && params.normalize == 1 &&
          params.truncate == 0);
    liyab_rerank_params rparams;
    liyab_rerank_params_default(&rparams);
    CHECK(rparams.prompt_template == nullptr && rparams.labels == nullptr && rparams.n_labels == 0);
    liyab_engine_config config;
    liyab_engine_config_default(&config);
    CHECK(config.pooling_kv_cache_type == LIYAB_KV_F16);
    config.backend = LIYAB_BACKEND_CPU;
    config.thermal_polling = 0;
    config.n_threads = 4;
    config.model_path = model_path("embedding", embedding_spec()).c_str();
    liyab_engine* engine = nullptr;
    REQUIRE(liyab_engine_create(&config, &engine) == LIYAB_OK);
    CHECK(liyab_engine_pooling(engine) == LIYAB_POOLING_LAST);
    CHECK(liyab_engine_embedding_size(engine) == 128);
    CHECK(liyab_engine_pooling(nullptr) == -1 && liyab_engine_embedding_size(nullptr) == 0);

    const char* texts[] = {"hello world", "world!"};
    const size_t lengths[] = {11, 5};  // the second without its '!'
    std::vector<float> out(2 * 128);
    liyab_pooling_stats stats{};
    REQUIRE(liyab_engine_embed(engine, texts, lengths, 2, &params, out.data(), out.size(), &stats) == LIYAB_OK);
    CHECK(stats.tokens > 0 && stats.ms > 0.0);
    auto cpp = make_engine(cpu_config(model_path("embedding", embedding_spec())));
    REQUIRE(cpp != nullptr);
    auto expected = embed(*cpp, {"hello world", "world"});
    REQUIRE(expected.has_value());
    CHECK(std::vector<float>(out.begin(), out.begin() + 128) == (*expected)[0]);
    CHECK(std::vector<float>(out.begin() + 128, out.end()) == (*expected)[1]);
    // NUL-terminated texts, fewer dimensions, a buffer too small.
    params.dimensions = 8;
    std::vector<float> small(2 * 8);
    CHECK(liyab_engine_embed(engine, texts, nullptr, 2, &params, small.data(), small.size(), nullptr) == LIYAB_OK);
    CHECK(liyab_engine_embed(engine, texts, nullptr, 2, &params, small.data(), small.size() - 1, nullptr) ==
          LIYAB_ERR_INVALID_ARGUMENT);
    params.pooling = static_cast<liyab_pooling>(9);
    CHECK(liyab_engine_embed(engine, texts, nullptr, 2, &params, small.data(), small.size(), nullptr) ==
          LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_engine_embed(nullptr, texts, nullptr, 2, nullptr, out.data(), out.size(), nullptr) ==
          LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_engine_embed(engine, nullptr, nullptr, 2, nullptr, out.data(), out.size(), nullptr) ==
          LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_engine_embed(engine, texts, nullptr, 0, nullptr, nullptr, 0, nullptr) == LIYAB_OK);
    // A reranker call on an embedding model: no prompt template.
    float score = 0.0f;
    CHECK(liyab_engine_rerank(engine, "hello", texts, nullptr, 1, nullptr, &score, nullptr) ==
          LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(std::string(liyab_last_error()).find("rerank") != std::string::npos);
    liyab_engine_destroy(engine);

    config.model_path = model_path("reranker", reranker_spec()).c_str();
    REQUIRE(liyab_engine_create(&config, &engine) == LIYAB_OK);
    CHECK(liyab_engine_pooling(engine) == LIYAB_POOLING_RANK);
    float scores[2] = {};
    REQUIRE(liyab_engine_rerank(engine, "hello", texts, nullptr, 2, nullptr, scores, &stats) == LIYAB_OK);
    auto ranker = make_engine(cpu_config(model_path("reranker", reranker_spec())));
    REQUIRE(ranker != nullptr);
    auto cpp_scores = rerank(*ranker, "hello", {"hello world", "world!"});
    REQUIRE(cpp_scores.has_value());
    CHECK(scores[0] == (*cpp_scores)[0] && scores[1] == (*cpp_scores)[1]);
    const char* labels[] = {"hello", "world"};
    rparams.labels = labels;
    rparams.n_labels = 2;
    rparams.prompt_template = "{query} / {document}";
    REQUIRE(liyab_engine_rerank(engine, "hello", texts, lengths, 2, &rparams, scores, nullptr) == LIYAB_OK);
    CHECK(scores[0] > 0.0f && scores[0] < 1.0f);
    rparams.labels = nullptr;
    CHECK(liyab_engine_rerank(engine, "hello", texts, lengths, 2, &rparams, scores, nullptr) ==
          LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_engine_rerank(engine, nullptr, texts, nullptr, 2, nullptr, scores, nullptr) ==
          LIYAB_ERR_INVALID_ARGUMENT);
    liyab_engine_destroy(engine);
}

TEST_MAIN()
