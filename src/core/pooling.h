// Liyab — pooling models (internal): the decoder's final hidden states turned
// into one vector per text (embeddings) or one relevance score per (query,
// document) pair (rerankers). The public contract is Engine::embed() and
// Engine::rerank() in liyab/engine.h; this module holds what a GGUF declares
// about pooling and the runs behind both calls.
//
// Everything model-specific is data read once at load (PoolingModel): the
// pooling type, whether inputs end with EOS, the ranking head and its labels,
// the prompt of a pair. A new embedding model or reranker built from blocks
// the engine runs needs no code here.
#ifndef LIYAB_CORE_POOLING_H
#define LIYAB_CORE_POOLING_H

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "compat/architecture.h"
#include "liyab/backend.h"
#include "liyab/engine.h"
#include "liyab/experimental/forward_hooks.h"
#include "liyab/mmap_loader.h"

namespace liyab {

class ThreadPool;
class Transformer;
struct Route;

// What a model's GGUF says about pooling.
struct PoolingModel {
    Pooling declared = Pooling::None;  // <arch>.pooling_type (absent: None)
    // tokenizer.ggml.add_eos_token: every input ends with `eos` (the
    // tokenizer's tokenizer.ggml.eos_token_id), as a Hugging Face
    // post-processor appends it.
    bool add_eos = false;
    int32_t eos = -1;
    // Ranking head: cls.output.weight [n_embd x labels] and its optional bias.
    // nullptr when the file has none, or one Liyab cannot apply (head_error says why).
    const TensorView* head = nullptr;
    std::vector<float> head_bias;
    std::vector<std::string> labels;  // <arch>.classifier.output_labels (may be empty)
    Status head_error;                // why there is no usable head (ok when the file simply has none)
    std::string rerank_template;      // tokenizer.chat_template.rerank (empty: none)
    // Dense projections of the pooled vector, applied in order before the
    // normalization (sentence-transformers' Dense modules, as the converter
    // writes them: dense_2.weight then dense_3.weight, no bias, identity
    // activation; EmbeddingGemma 768 -> 3072 -> 768). Empty: none.
    // embedding_width is what the last one returns (n_embd without them).
    std::vector<const TensorView*> dense;
    int32_t embedding_width = 0;
};

// Whether `file` declares a pooling model (<arch>.pooling_type mean, cls,
// last or rank), from its metadata alone, before anything is loaded: such a
// model gets EngineConfig::pooling_kv_cache_type.
bool declares_pooling(const MmapLoader& file);

// Reads the pooling metadata of `file` (described as `config`). Never fails
// the load of a generative model: what cannot be used is recorded
// (head_error) and reported by the call that needs it. InvalidModel only
// for a pooling_type outside 0..4.
Result<PoolingModel> read_pooling_model(const MmapLoader& file, const ModelConfig& config);

// `tmpl` with its first "{query}" and first "{document}" replaced (in the
// template only: text inside the query is never read as a placeholder).
// InvalidArgument when either placeholder is missing.
Result<std::string> format_rerank_prompt(std::string_view tmpl, std::string_view query, std::string_view document);

// What run_pooled() returns for each sequence.
struct PoolTarget {
    // Mean or Last: the final hidden states pooled (n_embd floats per sequence).
    Pooling pooling = Pooling::Last;
    // Non-empty: instead, the LM head's logits of these token ids at the last
    // position (logit_ids.size() floats per sequence).
    std::vector<int32_t> logit_ids;
};

struct PooledRun {
    std::vector<float> values;  // per sequence, in input order: n_embd or logit_ids.size() floats
    int64_t tokens = 0;         // tokens of every sequence
    int64_t reused = 0;         // of them, whose KV a neighbour computed
};

// Runs `sequences` (token ids, none empty, each fitting the context) through
// `model`, starting from an empty context, one after another in the
// lexicographic order of their tokens so that sequences sharing a prefix are
// neighbours. Each keeps the KV of the token prefix it shares with the one
// before (at most all but its last token: the last row is always computed),
// as far as the model can rewind (any position on attention-only models;
// hybrid ones start over, their recurrent states cannot rewind to an
// arbitrary point), and processes only the rest, in passes of at most
// `chunk` tokens. A bidirectional model (ModelConfig::causal false) reads
// each sequence whole in one pass and never shares: every row depends on
// the tokens after it. Mean pooling needs every row of a sequence, so its
// sequences never share. On the CPU backend a row's result does not depend
// on the pass it is computed in, so sharing changes no bit. `hooks` (may be
// nullptr) is passed to every pass (its cancel flag stops the run:
// Cancelled). Leaves the context holding the last sequence run, or undefined
// after an error: the caller resets it.
Result<PooledRun> run_pooled(Transformer& model, std::span<const std::vector<int32_t>> sequences,
                             const PoolTarget& target, const Route& route, ThreadPool& pool, int32_t chunk,
                             const ForwardHooks* hooks);

// Keeps the first `dimensions` values of each `size`-value vector of `values`
// (0: all) and L2-normalizes them when `normalize`; one vector per entry of
// the result. A zero vector stays zero.
std::vector<std::vector<float>> finish_embeddings(std::span<const float> values, size_t size, int32_t dimensions,
                                                  bool normalize);

// The model's dense projections (PoolingModel::dense) applied to pooled
// vectors (`pooled`: n_embd floats each), on `cpu`: embedding_width floats
// per vector. Without projections, `pooled` as it is.
Result<std::vector<float>> project_pooled(const PoolingModel& model, Backend& cpu, std::span<const float> pooled,
                                          int32_t n_embd);

// Scores of pooled last-token states (`hidden`, n_embd floats each) under the
// model's ranking head, computed on `cpu` (whatever layout the head has):
// logits = head · h + bias; with several labels the softmax probability of
// the first, with one its sigmoid.
Result<std::vector<float>> head_scores(const PoolingModel& model, Backend& cpu, std::span<const float> hidden,
                                       int32_t n_embd);

// The softmax probability of the first of each group of `labels` logits
// (`logits`: groups of `labels` floats), or the sigmoid when there is one.
std::vector<float> label_scores(std::span<const float> logits, size_t labels);

}  // namespace liyab

#endif  // LIYAB_CORE_POOLING_H
