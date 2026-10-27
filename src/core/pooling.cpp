#include "core/pooling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "core/quant.h"
#include "core/transformer.h"

namespace liyab {

namespace {

// Tensor names of a ranking head, as GGUF converters write them: the output
// projection over the labels, its optional bias, and the dense layer that a
// two-layer head (an encoder cross-encoder's dense + tanh) puts before it.
constexpr char kHeadWeight[] = "cls.output.weight";
constexpr char kHeadBias[] = "cls.output.bias";
constexpr char kHeadDense[] = "cls.weight";

}  // namespace

bool declares_pooling(const MmapLoader& f) {
    const auto arch = f.get_string("general.architecture");
    if (!arch) return false;
    const auto type = f.get_int(std::string(*arch) + ".pooling_type");
    return type && *type >= static_cast<int64_t>(Pooling::Mean) && *type <= static_cast<int64_t>(Pooling::Rank);
}

Result<PoolingModel> read_pooling_model(const MmapLoader& f, const ModelConfig& c) {
    PoolingModel m;
    const std::string pooling_key = c.arch + ".pooling_type";
    if (const GgufValue* v = f.metadata(pooling_key)) {
        const auto type = v->as_int();
        if (!type || *type < 0 || *type > 4) return Status(ErrorCode::InvalidModel, "invalid " + pooling_key);
        m.declared = static_cast<Pooling>(*type);
    }
    if (const GgufValue* v = f.metadata("tokenizer.ggml.add_eos_token")) m.add_eos = v->as_bool().value_or(false);
    const auto eos = f.get_int("tokenizer.ggml.eos_token_id");
    if (eos && *eos >= 0 && *eos < c.n_vocab) m.eos = static_cast<int32_t>(*eos);
    if (m.eos < 0) m.add_eos = false;  // nothing valid to append

    if (const GgufValue* v = f.metadata(c.arch + ".classifier.output_labels")) {
        for (const std::string_view label : v->strings) m.labels.emplace_back(label);
    }
    if (const auto tmpl = f.get_string("tokenizer.chat_template.rerank")) m.rerank_template = std::string(*tmpl);

    // Dense projections after pooling: each [in -> out], chained from n_embd.
    m.embedding_width = c.n_embd;
    for (const char* name : {"dense_2.weight", "dense_3.weight"}) {
        const TensorView* t = f.tensor(name);
        if (t == nullptr) {
            if (f.tensor("dense_3.weight") != nullptr) {
                return Status(ErrorCode::InvalidModel, "dense_3.weight without dense_2.weight");
            }
            break;
        }
        if (t->n_dims != 2 || t->cols() != m.embedding_width || t->rows() < 1 ||
            t->rows() > std::numeric_limits<int32_t>::max()) {
            return Status(ErrorCode::InvalidModel, std::string(name) + " does not continue the pooled vector's width " +
                                                       std::to_string(m.embedding_width));
        }
        if (f.tensor(std::string(name, std::string_view(name).size() - 6) + "bias") != nullptr) {
            return Status(ErrorCode::Unsupported, std::string(name) + ": dense projections with a bias are not supported");
        }
        m.dense.push_back(t);
        m.embedding_width = static_cast<int32_t>(t->rows());
    }

    const TensorView* head = f.tensor(kHeadWeight);
    if (head == nullptr) {
        if (m.declared == Pooling::Rank) {
            m.head_error = Status(ErrorCode::Unsupported, std::string("the model declares rank pooling but has no ") +
                                                              kHeadWeight);
        }
        return m;
    }
    if (f.tensor(kHeadDense) != nullptr) {
        m.head_error = Status(ErrorCode::Unsupported, std::string("two-layer ranking heads (") + kHeadDense +
                                                          ") are not supported: only a single projection over the labels");
        return m;
    }
    if (head->cols() != c.n_embd || head->rows() < 1 || head->ne[2] != 1 || head->ne[3] != 1) {
        m.head_error = Status(ErrorCode::InvalidModel, std::string(kHeadWeight) + " is not [n_embd x labels]");
        return m;
    }
    if (!m.labels.empty() && static_cast<size_t>(head->rows()) != m.labels.size()) {
        m.head_error = Status(ErrorCode::InvalidModel, std::string(kHeadWeight) + " has " +
                                                           std::to_string(head->rows()) + " rows for " +
                                                           std::to_string(m.labels.size()) + " labels");
        return m;
    }
    if (const TensorView* bias = f.tensor(kHeadBias)) {
        if (bias->elements() != head->rows()) {
            m.head_error = Status(ErrorCode::InvalidModel, std::string(kHeadBias) + " does not match the head's rows");
            return m;
        }
        m.head_bias.resize(static_cast<size_t>(head->rows()));
        quant::dequantize_row(bias->type, bias->data, m.head_bias.data(), head->rows());
    }
    m.head = head;
    return m;
}

Result<std::string> format_rerank_prompt(std::string_view tmpl, std::string_view query, std::string_view document) {
    constexpr std::string_view kQuery = "{query}", kDocument = "{document}";
    const size_t q = tmpl.find(kQuery);
    const size_t d = tmpl.find(kDocument);
    if (q == std::string_view::npos || d == std::string_view::npos) {
        return Status(ErrorCode::InvalidArgument, "a rerank prompt template needs both {query} and {document}");
    }
    // Both placeholders are located in the template before anything is
    // substituted, so a query that contains "{document}" stays text.
    struct Slot {
        size_t at;
        std::string_view placeholder, value;
    };
    Slot slots[2] = {{q, kQuery, query}, {d, kDocument, document}};
    if (slots[1].at < slots[0].at) std::swap(slots[0], slots[1]);
    std::string out;
    out.reserve(tmpl.size() + query.size() + document.size());
    size_t from = 0;
    for (const Slot& s : slots) {
        out.append(tmpl.substr(from, s.at - from));
        out.append(s.value);
        from = s.at + s.placeholder.size();
    }
    out.append(tmpl.substr(from));
    return out;
}

Result<PooledRun> run_pooled(Transformer& model, std::span<const std::vector<int32_t>> sequences,
                             const PoolTarget& target, const Route& route, ThreadPool& pool, int32_t chunk,
                             const ForwardHooks* hooks) {
    const ModelConfig& c = model.config();
    const size_t d = static_cast<size_t>(c.n_embd);
    const bool logits = !target.logit_ids.empty();
    const bool mean = !logits && target.pooling == Pooling::Mean;
    if (!logits && target.pooling != Pooling::Mean && target.pooling != Pooling::Last) {
        return Status(ErrorCode::InvalidArgument, "run_pooled: pooling must be Mean or Last");
    }
    for (const int32_t id : target.logit_ids) {
        if (id < 0 || id >= c.n_vocab) return Status(ErrorCode::InvalidArgument, "run_pooled: label token out of range");
    }
    const size_t width = logits ? target.logit_ids.size() : d;
    const bool causal = c.causal;

    PooledRun run;
    run.values.assign(sequences.size() * width, 0.0f);
    // Lexicographic order of the tokens puts the sequences that share a
    // prefix (the same instruction, the same query) next to each other, so
    // one comparison with the sequence before finds what can be kept.
    std::vector<size_t> order(sequences.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return std::lexicographical_compare(sequences[a].begin(), sequences[a].end(), sequences[b].begin(),
                                            sequences[b].end());
    });

    model.reset();
    const std::vector<int32_t>* previous = nullptr;
    std::vector<float> rows;     // final hidden states of a pass (Mean)
    std::vector<int32_t> ids;    // their row indices
    std::vector<double> sum(d);  // Mean: running sum of a sequence's rows
    for (const size_t index : order) {
        const std::vector<int32_t>& seq = sequences[index];
        if (seq.empty()) return Status(ErrorCode::InvalidArgument, "run_pooled: empty sequence");
        if (seq.size() > static_cast<size_t>(model.context_length())) {
            return Status(ErrorCode::InvalidArgument, "run_pooled: sequence longer than the context");
        }
        run.tokens += static_cast<int64_t>(seq.size());
        // The KV of the prefix shared with the sequence before stays; the
        // last token is always run, as its row is what is pooled.
        size_t keep = 0;
        if (previous != nullptr && !mean && causal) {
            const size_t most = std::min(previous->size(), seq.size() - 1);
            size_t common = 0;
            while (common < most && (*previous)[common] == seq[common]) ++common;
            keep = static_cast<size_t>(model.restorable_prefix(static_cast<int32_t>(common)));
        }
        if (keep == 0) {
            model.reset();
        } else if (!model.truncate(static_cast<int32_t>(keep)).is_ok()) {
            model.reset();  // a sliding window released those pages: start over
            keep = 0;
        }
        run.reused += static_cast<int64_t>(keep);
        std::fill(sum.begin(), sum.end(), 0.0);
        float* out = run.values.data() + index * width;
        // A bidirectional model reads the sequence in one pass.
        const size_t step = causal ? static_cast<size_t>(std::max(chunk, 1)) : seq.size();
        for (size_t i = keep; i < seq.size(); i += step) {
            const size_t n = std::min(step, seq.size() - i);
            const bool last_pass = i + n == seq.size();
            const std::span<const int32_t> part(seq.data() + i, n);
            const auto want = logits && last_pass ? Transformer::Logits::Last : Transformer::Logits::None;
            auto result = model.forward(part, want, route, pool, hooks);
            if (!result) return result.status();
            if (logits) {
                if (last_pass) {
                    const std::span<const float> row = result.value();
                    for (size_t j = 0; j < width; ++j) out[j] = row[static_cast<size_t>(target.logit_ids[j])];
                }
            } else if (mean) {
                ids.resize(n);
                std::iota(ids.begin(), ids.end(), int32_t{0});
                rows.resize(n * d);
                LIYAB_RETURN_IF_ERROR(model.final_hidden(ids, rows));
                for (size_t r = 0; r < n; ++r) {
                    const float* h = rows.data() + r * d;
                    for (size_t j = 0; j < d; ++j) sum[j] += h[j];
                }
            } else if (last_pass) {
                const int32_t last = static_cast<int32_t>(n) - 1;
                LIYAB_RETURN_IF_ERROR(model.final_hidden(std::span<const int32_t>(&last, 1), std::span<float>(out, d)));
            }
        }
        if (mean) {
            for (size_t j = 0; j < d; ++j) out[j] = static_cast<float>(sum[j] / static_cast<double>(seq.size()));
        }
        previous = &seq;
    }
    return run;
}

std::vector<std::vector<float>> finish_embeddings(std::span<const float> values, size_t size, int32_t dimensions,
                                                  bool normalize) {
    const size_t keep = dimensions > 0 ? std::min(size, static_cast<size_t>(dimensions)) : size;
    const size_t count = size == 0 ? 0 : values.size() / size;
    std::vector<std::vector<float>> out(count);
    for (size_t i = 0; i < count; ++i) {
        const float* v = values.data() + i * size;
        out[i].assign(v, v + keep);
        if (!normalize) continue;
        double ss = 0.0;
        for (const float x : out[i]) ss += static_cast<double>(x) * x;
        if (ss <= 0.0) continue;  // a zero vector has no direction: left as is
        const double scale = 1.0 / std::sqrt(ss);
        for (float& x : out[i]) x = static_cast<float>(x * scale);
    }
    return out;
}

std::vector<float> label_scores(std::span<const float> logits, size_t labels) {
    std::vector<float> scores;
    if (labels == 0) return scores;
    scores.reserve(logits.size() / labels);
    for (size_t at = 0; at + labels <= logits.size(); at += labels) {
        const float* l = logits.data() + at;
        if (labels == 1) {
            scores.push_back(static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(l[0])))));
            continue;
        }
        // Softmax over the labels, in double: P(first) = 1 / sum_j exp(l_j - l_0).
        double sum = 0.0;
        for (size_t j = 0; j < labels; ++j) sum += std::exp(static_cast<double>(l[j]) - static_cast<double>(l[0]));
        scores.push_back(static_cast<float>(1.0 / sum));
    }
    return scores;
}

Result<std::vector<float>> project_pooled(const PoolingModel& model, Backend& cpu, std::span<const float> pooled,
                                          int32_t n_embd) {
    if (n_embd <= 0 || pooled.size() % static_cast<size_t>(n_embd) != 0) {
        return Status(ErrorCode::InvalidArgument, "project_pooled: incomplete pooled vectors");
    }
    std::vector<float> values(pooled.begin(), pooled.end());
    auto width = static_cast<size_t>(n_embd);
    const size_t n = width == 0 ? 0 : pooled.size() / width;
    for (const TensorView* t : model.dense) {
        std::vector<float> out(n * static_cast<size_t>(t->rows()));
        if (n > 0) LIYAB_RETURN_IF_ERROR(cpu.matmul(*t, values.data(), out.data(), static_cast<int32_t>(n)));
        values = std::move(out);
        width = static_cast<size_t>(t->rows());
    }
    return values;
}

Result<std::vector<float>> head_scores(const PoolingModel& model, Backend& cpu, std::span<const float> hidden,
                                       int32_t n_embd) {
    if (model.head == nullptr) {
        if (!model.head_error.is_ok()) return model.head_error;
        return Status(ErrorCode::Unsupported, "the model has no ranking head (cls.output.weight)");
    }
    const auto d = static_cast<size_t>(n_embd);
    const size_t n = d == 0 ? 0 : hidden.size() / d;
    const auto labels = static_cast<size_t>(model.head->rows());
    std::vector<float> logits(n * labels);
    // The backend reads any layout the loader gave the head (file or repacked).
    if (n > 0) LIYAB_RETURN_IF_ERROR(cpu.matmul(*model.head, hidden.data(), logits.data(), static_cast<int32_t>(n)));
    if (!model.head_bias.empty()) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < labels; ++j) logits[i * labels + j] += model.head_bias[j];
        }
    }
    return label_scores(logits, labels);
}

}  // namespace liyab
