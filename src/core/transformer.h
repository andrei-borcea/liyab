// Liyab — decoder-only transformer over memory-mapped GGUF weights (internal).
//
// What the model is comes from describe_model() (core/architecture.h): the
// architecture's row of facts the tensors cannot tell, the hyperparameters,
// and one BlockPlan per block resolved from the tensors present. This class
// binds the weights to those plans and runs them: each block has a token
// mixer — softmax attention (GQA, separate or fused QKV projections, RoPE in
// either pairing with a per-block table, LongRoPE factors or YaRN, optional QKV
// bias, per-head Q/K norm before or after RoPE, optional sigmoid output gate,
// optional sliding window, optional soft cap), multi-head latent attention
// (absorbed queries over one cached [RoPE key | latent] row per position,
// see LatentAttention) or a Gated DeltaNet linear-attention recurrence —
// followed by a gated dense FFN (SwiGLU or GeGLU, separate or fused gate and
// up projections) or a mixture of experts (router + top-k routed experts +
// optional gated shared expert), each optionally followed by its own norm
// before the residual add. So hybrid models (Qwen3.5: three DeltaNet blocks
// per attention block), models that alternate windowed and global attention
// (Gemma 2 and 3), models with fused projections (Phi-3), latent attention
// (DeepSeek) and plain transformers (Llama, Mistral, Qwen2/3) share one
// forward pass. Tensor precision is per
// tensor, so mixed-precision files run unconverted.
#ifndef LIYAB_CORE_TRANSFORMER_H
#define LIYAB_CORE_TRANSFORMER_H

#include <algorithm>
#include <array>
#include <functional>
#include <iosfwd>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/architecture.h"
#include "core/expert_store.h"
#include "core/key_scaling.h"
#include "core/lora.h"
#include "core/quant.h"
#include "liyab/backend.h"
#include "liyab/experimental/forward_hooks.h"
#include "liyab/kv_cache.h"
#include "liyab/triple_buffer_loader.h"
#include "liyab/mmap_loader.h"

namespace liyab {

class ThreadPool;

// Final logit soft-cap (Gemma 2): replaces every x of `logits` by
// cap * tanh(x / cap), in place. Large spans are split into chunks on `pool`;
// each element is computed by the same expression whatever thread runs it, so
// the result is bit-identical for every thread count. Small spans run on the
// calling thread (a fork/join costs more than they do). Precondition: cap > 0.
// Not reentrant on the same pool (ThreadPool::parallel_for's rule).
void apply_final_softcap(std::span<float> logits, float cap, ThreadPool& pool);

class ThreadPool;

// Which backend runs which matmuls for one forward pass. `cpu` is the
// fallback when a backend reports Unsupported for a tensor.
struct Route {
    Backend* attention = nullptr;
    Backend* ffn = nullptr;
    Backend* cpu = nullptr;
};

// How multi-head latent attention (ModelConfig::mla) attends in a forward
// pass. Both forms read the same cache of latents and compute the same
// function; they round differently (see Transformer::latent_attention_mixer).
enum class LatentForm : uint8_t {
    // Per pass, the form LatentFormChooser predicts faster from the pass's
    // shape and a ratio measured once (calibrate_latent_form(); 1 until
    // then): fixed per device, never re-measured on the fly. A single-token
    // pass (decoding) is absorbed.
    Auto,
    Absorbed,  // every pass: absorbed queries over the cached [k_pe | latent] rows
    Expanded,  // every pass of two tokens or more: per-head keys and values expanded from the latents
};

// Picks latent attention's form for a pass: a pure function of the pass's
// shape and one calibrated number. Each form's work is counted in
// multiply-adds of the parts that differ between them (work()); `ratio` is
// what an expanded multiply-add costs on this device relative to an absorbed
// one, measured once (Transformer::calibrate_latent_form()) and then fixed, so
// the threshold on batch size and context is the device's own, not a
// constant, while the same pass always takes the same form: the same prompt
// gives the same bits on that device in every run, whatever the load. Before
// any calibration the ratio is 1 (the counts decide alone). Not thread-safe
// (one per Transformer, used from forward()).
class LatentFormChooser {
public:
    // Multiply-adds of the form-dependent parts of a pass of `n` tokens over
    // `visible` (token, visible position) pairs in total, `positions` of them
    // distinct (the union the expanded form expands): per token the absorbed
    // form projects each head's query onto the latent and the weighted
    // latent back out (kv_rank x (nope + v) per head) and attends over
    // rope_dim + kv_rank channels, accumulating kv_rank; the expanded form
    // expands each distinct position's keys and values (kv_rank x (nope +
    // v) per head, plus the weights' decoding, counted as one more position)
    // and attends over nope + rope_dim channels, accumulating v.
    struct Work {
        double absorbed = 0.0;
        double expanded = 0.0;
    };
    [[nodiscard]] static Work work(const ModelConfig& c, int64_t n, int64_t visible, int64_t positions);
    // True when the expanded form is predicted faster for `w`: ratio x
    // expanded work < absorbed work (a tie stays absorbed).
    [[nodiscard]] bool expand(const Work& w) const noexcept { return ratio_ * w.expanded < w.absorbed; }
    [[nodiscard]] float ratio() const noexcept { return ratio_; }
    // InvalidArgument outside [kMinRatio, kMaxRatio] or not finite; takes the
    // ratio as given (callers round measurements with round_ratio()).
    Status set_ratio(float ratio) noexcept;
    // A measured ratio rounded to steps of 1/8 octave and clamped to
    // [kMinRatio, kMaxRatio], so measurements a little apart (a second
    // calibration, a similar device) usually land on the same value.
    [[nodiscard]] static float round_ratio(double measured) noexcept;
    static constexpr float kMinRatio = 1.0f / 16.0f;
    static constexpr float kMaxRatio = 16.0f;

private:
    float ratio_ = 1.0f;
};

struct TransformerOptions {
    // 0: automatic (kKvBudgetBytes of KV, 4096..32768, at most the training
    // context; windowed attention blocks counted for their window only).
    int32_t context_length = 0;
    // Windowed attention blocks keep every position, as the others (the
    // single-table layout): segment reuse needs every block's rows of a
    // segment wherever it lies. Costs full attention's KV memory on those
    // blocks; false (default): they keep their window (KvCacheConfig::
    // layer_windows).
    bool keep_windowed_rows = false;
    int32_t sliding_window = 0;   // 0: full attention
    int32_t sink_tokens = 8;      // attention-sink anchors kept with a sliding window
    KvCacheType kv_type = KvCacheType::Q8_0;
    int32_t max_batch = 64;       // largest rollback the caller performs (speculative k + 1)
    // MoE models: RAM for routed experts streamed from storage (ExpertStore).
    // -1: automatic (stream when the model does not fit in RAM, budget = free
    // RAM minus the resident weights and a safety margin), 0: never (experts
    // are read through the mapping), > 0: always, with this budget.
    int64_t expert_cache_bytes = -1;
    // > 0: total memory the engine may keep resident (see usable_memory_bytes()).
    uint64_t memory_budget_bytes = 0;
    // Memory the embedder would like set aside in that budget for its own
    // use (the engine's prefix cache). With streamed experts it comes only
    // from what the expert cache can spare above kTightCache of the experts,
    // so it never starves the cache nor causes a lossy requantization
    // (MemoryPlan::reserved_bytes says how much); otherwise it is the
    // embedder's to take.
    size_t reserve_bytes = 0;
    // With expert streaming: convert the resident Q8_0 matrices to Q4_K (4)
    // or Q5_K (5) at load, before the expert cache is sized (lossy; 0: off;
    // -1: only when memory is tight, see Transformer::kTightCache).
    int32_t requant_bits = 0;
    // MoE: per token, run only the fewest top-ranked experts of the top-k
    // whose router probabilities cover this fraction of the top-k's total
    // (renormalized over the kept ones when the model normalizes its top-k).
    // Lossy; >= 1 runs all top-k (default).
    float expert_mass = 1.0f;
    // MoE: at most this many experts per token (the top ones by router
    // score, renormalized like the model's top-k); 0 or >= the model's top-k:
    // the model's own count. Lossy when it cuts.
    int32_t max_experts = 0;
    // MoE with streamed experts: skip a chosen expert not in RAM yet when its
    // weight share is below this for every token using it (Engine's
    // moe_skip_slow). Lossy; 0: off.
    float skip_slow = 0.0f;
    // Repack the resident Q4_K / Q5_K / Q6_K / Q8_0 matrices (Q4_K_R8 / Q5_K_R8 /
    // Q6_K_R8 / Q8_0_R4) for the CPU's i8mm kernels (set only when the CPU
    // computes every block; lossless).
    bool repack_cpu = false;
    // Latent attention (DeepSeek-V2/V3 and models converted alike): the
    // form passes attend in. Ignored by other models.
    LatentForm latent_form = LatentForm::Auto;
    // Contexts the embedder will make besides the live one (make_context():
    // the engine's sequences), and the rollback window it will give them.
    // Only for the memory plan of a MoE whose experts stream: each one's
    // recurrent states, kStateSnapshots snapshots and rollback checkpoints
    // (hybrid models) come out of the expert cache, as the live context's
    // snapshots do. 0: none (the plan of a single context).
    int32_t extra_contexts = 0;
    int32_t extra_rollback_window = 0;
};

class Transformer {
public:
    // One sequence's state; see "Several sequences" below.
    class Context;

    static Result<std::unique_ptr<Transformer>> load(std::unique_ptr<MmapLoader> file, const TransformerOptions& options);

    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] const MmapLoader& file() const noexcept { return *file_; }
    [[nodiscard]] MmapLoader& mutable_file() noexcept { return *file_; }
    [[nodiscard]] const KvCache& kv_cache() const noexcept { return *kv_; }
    [[nodiscard]] int32_t context_length() const noexcept { return context_length_; }
    [[nodiscard]] int32_t n_past() const noexcept { return n_past_; }

    enum class Logits { None, Last, All };
    // Appends `tokens` at positions [n_past, n_past + n) and returns logits
    // (n_vocab floats per row): none (prefill), the last token's, or every
    // token's (speculative verification). Valid until the next forward().
    // `hooks` (optional) enables early exit (single-token Logits::Last steps
    // only) and attention-head masking.
    Result<std::span<const float>> forward(std::span<const int32_t> tokens, Logits logits, const Route& route,
                                           ThreadPool& pool, const ForwardHooks* hooks = nullptr);
    // ---- Tree verification (speculative decoding) ----
    // One pass over a tree of tokens: tokens[0] continues the context and
    // token t > 0 continues token parents[t] (parents[0] = -1, 0 <= parents[t]
    // < t), so the tree's branches are alternative continuations verified
    // together. Every token is computed as if its branch (its ancestors, then
    // itself) alone followed the context: at position n_past() + its depth,
    // attending to the context and to its ancestors only (the same visible
    // positions, in the same order and spans, as a linear pass over its
    // branch), recurrent (DeltaNet) blocks continuing from the state after
    // its parent. So its row of logits (one per token, in token order, as
    // Logits::All) has the bits a linear pass over its branch gives it
    // (attention never depends on the batch; the CPU backend's matmuls give a
    // row the same bits in any batch). Latent attention runs in the absorbed
    // form, as decoding does. Token t's KV rows are held at slot n_past() + t
    // until keep_path(), which must come next: forward() and forward_tree()
    // refuse to run before it; truncate() to n_past() or below drops the tree.
    // Preconditions: tokens.size() <= max_batch() and, on hybrid models,
    // <= rollback_window() (every token's recurrent state is kept until
    // keep_path(), in the rollback checkpoints). InvalidArgument for parents
    // that do not form such a tree or a batch too large; Unsupported for a
    // hybrid model whose rollback window is too small. No hooks.
    Result<std::span<const float>> forward_tree(std::span<const int32_t> tokens, std::span<const int32_t> parents,
                                                const Route& route, ThreadPool& pool);
    // After forward_tree(): keeps the branch `path` (token indices of the
    // tree pass: path[0] = 0, each next one a child of the one before) as
    // the next path.size() positions of the context and drops the rest of
    // the tree: the branch's KV rows move to their positions
    // (KvCache::copy_position(), none when the branch is the tree's first
    // tokens), the recurrent states become those after path.back(), and the
    // rollback checkpoints those of the branch's positions, so truncate()
    // can still return into it. Exactly the context a linear pass over the
    // branch leaves. InvalidArgument without a pending tree or for a path
    // that is not a branch of it (the tree stays pending).
    Status keep_path(std::span<const int32_t> path);
    // Whether a forward_tree() pass awaits keep_path().
    [[nodiscard]] bool tree_pending() const noexcept { return tree_base_ >= 0; }
    // Pooling models (embeddings, ranking heads; core/pooling.h): the final
    // hidden states of rows `rows` of the last forward() pass (indices into
    // its tokens), after the output norm and before the LM head, n_embd floats
    // each, written in order to `out` (at least rows.size() * n_embd floats).
    // A pass leaves its residual stream in place whatever logits it returned,
    // so this costs one norm per row and no pass of its own, and gives the
    // very values the LM head multiplies. Valid until the next forward(), not
    // after one that exited early. InvalidArgument for a row outside the last
    // pass or a short `out`.
    Status final_hidden(std::span<const int32_t> rows, std::span<float> out) const;
    // Block after which the last forward() exited early, or -1.
    [[nodiscard]] int32_t last_exit_layer() const noexcept { return last_exit_layer_; }
    // FFN blocks skipped by ForwardHooks::ffn_skip during the last forward().
    [[nodiscard]] int32_t last_ffn_skips() const noexcept { return last_ffn_skips_; }
    // Attention output projection of block `layer` (used by the head pruner);
    // nullptr for blocks whose mixer is not attention.
    [[nodiscard]] const TensorView* attn_output(int32_t layer) const {
        return layers_[static_cast<size_t>(layer)].w[kO];
    }
    // FFN projections {gate, up, down} of block `layer` (experimental
    // sparsification). With BlockPlan::fused_gate_up, gate is nullptr and up
    // is the fused [gate | up] matrix; MoE blocks have none of them.
    [[nodiscard]] std::array<const TensorView*, 3> ffn_weights(int32_t layer) const {
        const Layer& L = layers_[static_cast<size_t>(layer)];
        return {L.w[kGate], L.w[kUp], L.w[kDown]};
    }
    // Routed-expert streaming (MoE models larger than RAM); nullptr when the
    // experts are read in place from the mapping.
    [[nodiscard]] const ExpertStore* expert_store() const noexcept { return expert_store_.get(); }

    // How expert streaming uses memory (all zero when experts are read in place).
    struct MemoryPlan {
        size_t resident_bytes = 0;      // weights kept in RAM (after any requantization)
        size_t expert_bytes = 0;        // routed experts on storage
        size_t expert_cache_bytes = 0;  // RAM for cached experts
        size_t recommended_bytes = 0;   // memory budget at which the cache holds kComfortableCache of the experts
        int32_t requant_bits = 0;       // resident Q8_0 matrices converted to Q4_K (4) / Q5_K (5); 0: none
        size_t reserved_bytes = 0;      // of TransformerOptions::reserve_bytes, what the plan set aside
    };
    [[nodiscard]] const MemoryPlan& memory_plan() const noexcept { return memory_plan_; }
    // Below this share of the experts cached, hit rates fall off and nearly
    // every token waits for storage; automatic requantization aims above it.
    static constexpr double kTightCache = 0.10;
    // A share at which most expert uses come from RAM (~85% hits at 12-15% on
    // Qwen3.6-35B-A3B): what the recommended budget is computed for.
    static constexpr double kComfortableCache = 0.15;
    // KV memory the automatic context length is sized for.
    static constexpr size_t kKvBudgetBytes = size_t{256} << 20;
    // Attention cuts each token's visible positions into min(kAttentionMaxSpans,
    // visible / kAttentionMinSpan) equal spans (at least one) and merges them
    // in order: a function of the token's own position, whatever the batch or
    // the threads (see attention()). The spans are a decoding step's tasks, so
    // they must split evenly among the threads: the 2 KV heads of Qwen3.6
    // give 168 tasks at most, a multiple of 1-4 and 6-8 threads. With 32 tasks
    // (16 spans), 7 threads took 5 tasks' time for 4.6 tasks' work.
    static constexpr int32_t kAttentionMinSpan = 32;
    static constexpr int32_t kAttentionMaxSpans = 84;
    // Latent attention's expanded form holds one tile of expanded keys and
    // values at a time: the largest power of two of positions in
    // [kLatentTileMin, kLatentTileMax] whose rows fit kLatentTileBytes (256
    // positions, 5 MiB, on DeepSeek-V2-Lite's 16 heads of 192 + 128).
    static constexpr size_t kLatentTileBytes = size_t{8} << 20;
    static constexpr int32_t kLatentTileMin = 16;
    static constexpr int32_t kLatentTileMax = 256;
    [[nodiscard]] ExpertStore* expert_store() noexcept { return expert_store_.get(); }
    // Bytes held by the recurrent (DeltaNet) states; 0 for plain transformers.
    [[nodiscard]] size_t recurrent_state_bytes() const noexcept;
    // Whether the context carries a state bound to its position, beyond KV
    // rows any prefix of it can be cut from: recurrent (DeltaNet) states, or
    // the rows of the last window of windowed attention blocks, whose older
    // rows the cache lets go (KvCache::layer_windows). Such a context returns
    // only to where that state is kept (snapshots, the rollback margin).
    [[nodiscard]] bool has_position_state() const noexcept;
    // Positions of the state snapshots held (snapshot_state()), ascending.
    [[nodiscard]] std::vector<int32_t> snapshot_positions() const;
    [[nodiscard]] int32_t max_batch() const noexcept { return max_batch_; }

    // Wall time spent in each phase of forward(), in ms, cumulative over every
    // call (callers diff two snapshots). A steady_clock read per phase and
    // block, a few hundred per token: noise next to the matmuls they time.
    struct PhaseTimes {
        double attention = 0.0;      // softmax-attention mixers, projections included
        double delta_net = 0.0;      // Gated DeltaNet mixers, projections included
        double router = 0.0;         // MoE routers, top-k selection and expert prediction
        double experts = 0.0;        // routed experts, waits for streamed ones included
        double shared_expert = 0.0;  // MoE shared expert
        double dense_ffn = 0.0;      // dense SwiGLU FFNs
        double lm_head = 0.0;        // final norm and output projection
    };
    [[nodiscard]] const PhaseTimes& phase_times() const noexcept { return phases_; }

    // Expert prediction accuracy (expert streaming only), cumulative like
    // PhaseTimes: per MoE block, the experts predicted ahead of time and how
    // many of them the router then chose. used / predicted is the precision;
    // the rest are reads that only cost flash bandwidth and cache space.
    struct ExpertPredictions {
        uint64_t predicted = 0;
        uint64_t used = 0;
        uint64_t skipped = 0;  // chosen experts skipped instead of waited for (skip_slow)
    };
    [[nodiscard]] const ExpertPredictions& expert_predictions() const noexcept { return predictions_; }

    // Latent attention: the form passes attend in (TransformerOptions::
    // latent_form), changeable between passes; the cache holds latents in
    // either form, so a context continues in the other. No effect on other
    // models.
    [[nodiscard]] LatentForm latent_form() const noexcept { return latent_form_; }
    void set_latent_form(LatentForm form) noexcept { latent_form_ = form; }
    // Passes run in each form so far (cumulative; passes of models without
    // latent attention are not counted), and the automatic choice's ratio.
    struct LatentStats {
        int64_t absorbed_passes = 0;
        int64_t expanded_passes = 0;
    };
    [[nodiscard]] const LatentStats& latent_stats() const noexcept { return latent_stats_; }
    [[nodiscard]] const LatentFormChooser& latent_chooser() const noexcept { return latent_chooser_; }
    // LatentForm::Auto's calibration: times both forms of the first latent
    // block on a synthetic pass (256 tokens after 768 cached positions of
    // random latents, at most half the context; best of 3 each, the
    // form-dependent parts only, on `route`), sets the chooser's ratio from
    // it (LatentFormChooser::round_ratio) and returns it. 0.18 s on
    // Youtu-LLM-2B (Mac CPU). Forgets the context. Measured only when called:
    // forward() never re-measures, so the ratio (and every pass's form) stays
    // fixed until the next call or set_latent_ratio(). Returns 1 without
    // measuring for models without latent attention.
    Result<float> calibrate_latent_form(const Route& route, ThreadPool& pool);
    // Uses a ratio measured earlier on this device (e.g. kept by the engine
    // in its cache_dir). InvalidArgument as LatentFormChooser::set_ratio.
    Status set_latent_ratio(float ratio) noexcept { return latent_chooser_.set_ratio(ratio); }

    // Discards cached positions >= n (speculative-decoding rollback, a chat
    // turn that rewrites the end of the context). KV caches simply drop
    // positions, as long as the windowed blocks still hold the rows of the
    // window before n (the rollback margin kept behind the end, a snapshot's
    // pinned window: InvalidArgument otherwise, see KvCache::truncate()).
    // Recurrent (DeltaNet) states cannot be rewound, so hybrid models restore
    // a copy instead: they accept n == 0, n == n_past(), any n whose position
    // n - 1 is still inside the rollback window, or the position of a state
    // snapshot (Unsupported otherwise). Snapshots past n are discarded.
    Status truncate(int32_t n);
    // After forward() returned Cancelled (ForwardHooks::cancel) partway
    // through its blocks: returns the context to n_past(), the position the
    // pass started from. Recurrent states come back from a snapshot taken
    // there (the caller takes one before a cancellable pass of a hybrid
    // model); without one, Unsupported and the context must be reset.
    Status abandon_pass();
    // The longest prefix of at most n positions truncate() accepts: n itself
    // for models without a position state, else the longest whose windowed
    // rows are still kept (KvCache::restorable()) and, on hybrid models, the
    // nearest checkpoint or snapshot at or below n among those (0 if none).
    [[nodiscard]] int32_t restorable_prefix(int32_t n) const noexcept;
    // Models with a position state (has_position_state()): keeps the state
    // at the current position, so truncate() can return exactly here later:
    // a copy of the recurrent states, and a pin on the windowed blocks'
    // window before it (KvCache::pin(): those pages stay, no copy). The engine
    // takes one at the end of every prompt (`pin`: the next chat turn
    // re-sends that prompt, and its end, the last reply or an edited turn,
    // may differ) and at the end of every prefill (a draft being typed ahead,
    // which changes as the user edits). At most kStateSnapshots are kept: the
    // oldest (usually the system prompt), the last pinned one, and the two
    // newest others; their memory is set aside from the expert cache budget
    // at load. Without a pinned slot, drafts typed ahead replaced the end of
    // the conversation, and an edit before it fell back to the system prompt
    // (2838 tokens processed again, 151 s on a 35B MoE). Two others, because
    // a draft is prepared twice: its words, then the whole message as it
    // would be sent (time line and reply opening included); typing on goes
    // back to the words, not to the last prompt. No-op without a position
    // state or at position 0.
    void snapshot_state(bool pin = false);
    static constexpr int32_t kStateSnapshots = 4;
    // Quantized KV caches: per-channel key scaling (core/key_scaling.h).
    // Measures the factors on `tokens` (key_calibration_text(), encoded) and
    // uses them from then on. Forgets the context. With an F16 cache or no
    // attention block there is nothing to measure and no factors are set.
    Status calibrate_keys(std::span<const int32_t> tokens, const Route& route, ThreadPool& pool);
    // The factors in use, KeyScalingShape order; empty: keys stored as computed.
    [[nodiscard]] std::span<const float> key_scales() const noexcept { return key_scale_; }
    // Uses factors calibrate_keys() measured earlier on this model (e.g. in
    // another process); empty turns scaling off. The context must be empty.
    // InvalidArgument for the wrong count, a factor outside
    // [1/kMaxKeyScale, kMaxKeyScale], a quantized-cache-only call on an F16
    // cache, or a non-empty context.
    Status set_key_scales(std::span<const float> scales);
    // Writes what the context holds (n_past(), the KV rows of positions
    // [0, n_past()) of the blocks keeping every position as rows image pages,
    // the key scales they were stored with, the recurrent states at n_past(),
    // below n_past() the oldest and the pinned state snapshots, and the
    // windowed blocks' pages of the window before n_past() and before those
    // snapshots, whole) with a fingerprint of the model, the cache
    // settings and its per-block layout, so read_state() accepts it only for
    // the same model file loaded the same way (format version 4).
    // Unsupported with a sliding window (released positions are gone).
    // IoError when the stream fails.
    Status write_state(std::ostream& out) const;
    // A context held outside the model, as the engine's KV store keeps it
    // (core/kv_store.h): positions [0, n), the rows image page `page(p)`
    // returns for each p < ceil(n / page_tokens) (page_bytes() bytes, valid
    // until the next call), the recurrent states at n and the snapshots below
    // n; with windowed blocks, the windowed rows of the pages of the window
    // before n that `window_page(p)` returns (KvCache::read_window_rows()'s
    // image of page_tokens positions, window_page_bytes(); valid until the
    // next call), and no snapshots.
    struct SnapshotImage {
        int32_t pos = 0;
        bool pinned = false;
        std::span<const float> data;  // recurrent_state_bytes() / sizeof(float) values
    };
    struct ContextImage {
        int32_t n = 0;
        std::function<const uint8_t*(int32_t page)> page;
        std::function<const uint8_t*(int32_t page)> window_page;  // windowed blocks only
        std::span<const float> state;  // empty without recurrent blocks
        std::vector<SnapshotImage> snapshots;
    };
    // Writes `image` in write_state()'s format, as if it were the context
    // (the key scales in use, which the store's chunks were stored with).
    // Unsupported with a sliding window; InvalidArgument for an image that
    // does not fit this model, or when a window page of the window before n
    // is missing (window_page() returned nullptr);
    // IoError when the stream fails.
    Status write_state(std::ostream& out, const ContextImage& image) const;
    // Replaces the context with one write_state() saved, its snapshots
    // included (truncate() returns to them as in the context that was saved),
    // and takes a state snapshot at its end (pinned unless a saved one is).
    // Reads the formats without snapshots or key scales too. Keys stored with
    // other scales than the ones in use (an older file: none) are converted
    // to them. Files older than version 4 (every block's rows in one table's
    // pages) are converted to the per-block layout: windowed blocks keep the
    // window before the end (a hybrid model's snapshots below it, whose
    // windows those files did not keep apart, are dropped). InvalidArgument
    // when the fingerprint differs or a version 4 file has another per-block
    // layout (other windows); the context is empty after any failure.
    Status read_state(std::istream& in);
    // What the context's bytes depend on (the fingerprint read_state()
    // checks, the page size, the per-block layout and the key scales in use):
    // the root of the KV store's keys, together with what the engine adds
    // (kernels, route). Unchanged for caches without windowed blocks.
    [[nodiscard]] uint64_t context_fingerprint() const noexcept;
    // The state images write_state() keeps, ascending: the oldest and the
    // pinned snapshot below n_past(), then n_past() itself, with the pinned
    // flags read_state() gives them back (the end pinned unless a snapshot
    // below is). Empty without a position state or at position 0. (With
    // windowed blocks these are the restore points whose windows the engine's
    // store keeps, page by page: KvCache::read_window_rows().)
    struct KeptState {
        int32_t pos = 0;
        bool pinned = false;
    };
    [[nodiscard]] std::vector<KeptState> kept_states() const;
    // The recurrent states at `pos`, one of kept_states(), as the parts of a
    // state image in order ([conv | ssm] of each recurrent block; one part
    // for a snapshot), read in place: written to storage without a copy.
    // Valid until the context changes; empty for another position.
    [[nodiscard]] std::vector<std::span<const float>> kept_state_parts(int32_t pos) const;

    // Float storage for recurrent-state images that is not zero-filled when
    // sized: every value is written before it is read, and a 35B's image is
    // 62.8 MiB, whose zero-fill and first touch cost as much as reading it
    // back from storage.
    template <typename T>
    struct UninitializedAllocator : std::allocator<T> {
        template <typename U>
        struct rebind {
            using other = UninitializedAllocator<U>;
        };
        UninitializedAllocator() = default;
        template <typename U>
        UninitializedAllocator(const UninitializedAllocator<U>&) noexcept {}
        template <typename U>
        void construct(U* p) noexcept {
            ::new (static_cast<void*>(p)) U;  // default-initialized: left as it is
        }
        template <typename U, typename... Args>
        void construct(U* p, Args&&... args) {
            ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
        }
    };
    using StateBuffer = std::vector<float, UninitializedAllocator<float>>;

    // Restoring a context image in place (the engine's KV store), with no
    // intermediate copy of the recurrent states. begin_restore() keeps the
    // buffers of the current snapshots for reuse (the context is being
    // replaced: they are already the right size and in memory);
    // recurrent_parts() are the live recurrent states, in image order, for a
    // state image to be read into; take_snapshot_buffer() gives the buffer
    // for one snapshot image (a kept one, else a new one, not zero-filled).
    // adopt_context() then makes positions [0, n) the context.
    void begin_restore();
    [[nodiscard]] std::vector<std::span<float>> recurrent_parts();
    [[nodiscard]] StateBuffer take_snapshot_buffer();
    struct OwnedSnapshot {
        int32_t pos = 0;
        bool pinned = false;
        StateBuffer data;  // recurrent_state_bytes() / sizeof(float) values
    };
    // Takes positions [0, n) as the context once the caller has written their
    // KV rows (KvCache::reserve_context() and write_rows(), and with windowed
    // blocks write_window_rows() for the window before n and before each
    // snapshot; rows below n_past() may be left as they are where they hold
    // the same tokens) and its recurrent states (recurrent_parts()): frees the
    // pages past n, moves `snapshots` in as the snapshots below n (ascending,
    // fewer than kStateSnapshots), discards the rollback checkpoints, then
    // takes a snapshot at n, pinned when `pin_end`, in a buffer begin_restore()
    // kept when one is left (the others are freed); the snapshots' windows are
    // pinned. InvalidArgument for an image that does not fit (sizes,
    // positions, unmapped pages, windowed rows missing); the context is then
    // empty.
    Status adopt_context(int32_t n, std::vector<OwnedSnapshot> snapshots, bool pin_end);

    // Keeps a copy of every recurrent state after each of the last
    // `positions` processed tokens, so truncate() can return to any of them
    // (speculative verification needs draft count + 1). Costs positions x
    // recurrent_state_bytes() of memory and one state copy per token and
    // recurrent block; 0 (the default) turns it off. No effect on models
    // without recurrent blocks. Discards existing checkpoints.
    void set_rollback_window(int32_t positions);
    [[nodiscard]] int32_t rollback_window() const noexcept { return rollback_window_; }
    // Whether forward() records checkpoints (default true). Callers that know
    // a pass will never be rolled back (a plain decoding step) turn it off to
    // skip the state copies; positions processed meanwhile cannot be returned to.
    void set_checkpointing(bool on) noexcept { checkpointing_ = on; }
    [[nodiscard]] KvCache& mutable_kv_cache() noexcept { return *kv_; }
    void reset() noexcept;

    // ---- KV reused away from its prefix (segment reuse, core/kv_reuse.h) ----
    // Whether this model's KV rows can be moved to other positions and
    // partly recomputed there, and why not: Unsupported for hybrid models
    // (their recurrent blocks must run over the whole prompt, in order),
    // with a sliding-window cache or windowed attention blocks (released
    // positions are gone), and when a
    // Q/K norm after RoPE zeroes a rotated key channel (its rotation could
    // not be undone).
    [[nodiscard]] Status reuse_support() const;
    // A context of its own for background work (segment reuse's calibration
    // while the engine is idle): swaps the KV cache and the position count
    // with `kv` and `n_past`, so the work runs on another cache and the
    // conversation's comes back untouched by swapping again. `kv` comes from
    // make_kv_cache(). Unsupported with recurrent blocks or windowed
    // attention blocks (their states and snapshots are part of a context
    // too); InvalidArgument for a cache of another layout.
    Status swap_context(std::unique_ptr<KvCache>& kv, int32_t& n_past);
    // An empty KV cache laid out like the context's (format, pages, rows)
    // for `positions` positions; its pages are mapped as they are used.
    [[nodiscard]] Result<std::unique_ptr<KvCache>> make_kv_cache(int32_t positions) const;
    // Rewrites the keys of slots [first, first + count) of `image` (an image
    // of `slots` positions in KvCache::read_rows()' layout), stored for
    // positions from + i, as this model stores the keys of positions to + i.
    // Exact up to rounding for every RoPE this engine runs, since RoPE turns
    // each rotated pair by an angle proportional to the position: each pair
    // is turned back by its stored position's angle and on by the new one's,
    // with each block's own table (pairing, partial rotation, frequency
    // factors, LongRoPE, YaRN, the windowed blocks' base, latent attention's
    // RoPE part); RoPE's magnitude cancels. Key scales and a Q/K norm after
    // RoPE are taken out before the rotation and put back after it. Only the
    // blocks of a quantized row that hold rotated channels are dequantized
    // and quantized again: values, unrotated channels (latent attention's
    // latent) and every row of a move by 0 keep their bits. `pool`
    // (optional) spreads the rows over its threads. InvalidArgument for
    // slots outside the image or negative positions; reuse_support()'s
    // errors otherwise.
    Status move_keys(uint8_t* image, int32_t slots, int32_t first, int32_t count, int32_t from, int32_t to,
                     ThreadPool* pool = nullptr) const;
    // The token embeddings of `tokens` (n_embd floats each, scaled as
    // forward() scales them) into `x`: the residual stream entering block 0.
    // InvalidArgument for a token outside the vocabulary or a short `x`.
    Status embed(std::span<const int32_t> tokens, std::span<float> x) const;
    // Runs block `layer` for tokens at `positions` (ascending, each below
    // n_past(): their pages mapped, every row a query reads written) whose
    // residual streams entering the block are the rows of `x` (n_embd floats
    // each), as forward() would for a batch at those positions: their KV rows
    // at the block are written, each token attends to every cached position
    // up to its own (the rows of the others as the cache holds them), and
    // `x` is left with the streams after the block. Lets a caller recompute
    // some positions of a context block by block, keeping the rest. Passes
    // of at most `chunk` tokens in order, the block's weights acquired once;
    // no hooks. Latent attention runs in the absorbed form here whatever
    // latent_form() says: the expanded form's tiles assume a contiguous pass,
    // so rows recomputed here equal an absorbed prefill's and differ from an
    // expanded one's by the two forms' rounding. Attention-only models
    // (Unsupported otherwise, see reuse_support()); InvalidArgument for
    // positions or sizes that do not fit.
    Status run_block_at(int32_t layer, std::span<const int32_t> positions, std::span<float> x, int32_t chunk,
                        const Route& route, ThreadPool& pool);
    // The KV cache slot of block `layer`, or -1 when its mixer keeps none.
    [[nodiscard]] int32_t kv_slot_of(int32_t layer) const noexcept {
        return layers_[static_cast<size_t>(layer)].kv_slot;
    }
    // The queries the last pass's attention scored with, at the last block it
    // ran (run_block_at: of its last pass of at most `chunk` tokens): n_head x
    // head_dim floats per token, rotated and key-scaled as the cached keys
    // they meet (a latent-attention head's absorbed query). Valid until the
    // next pass.
    [[nodiscard]] std::span<const float> last_queries() const noexcept { return q_; }

    // Streams transformer blocks through `source` instead of reading the
    // mapping in place. Block l is item `items[l]` of the source (laid out as
    // MmapLoader::layer_range(l)), or read in place when items[l] < 0; an
    // empty `items` streams every block (item i = block i). nullptr restores
    // in-place reads. `source` must outlive its use here.
    // `shared`: the source's slots are accelerator-shared memory
    // (Backend::allocate_shared), so streamed blocks may use the normal route;
    // otherwise they run on the CPU (see BlockWeights::streamed()).
    void set_layer_source(TripleBufferLoader* source, std::vector<int32_t> items = {}, bool shared = false) {
        layer_source_ = source;
        layer_items_ = std::move(items);
        layer_shared_ = shared;
    }

    // ---- LoRA adapters (core/lora.h) ----
    // Applies `adapter` at `scale` (s = scale · alpha / rank per matrix) to
    // every later pass until changed: each adapted matrix's matmul output
    // gets s · B·(A·x), computed on route.cpu (A·x in the base matmuls'
    // batch when they run there too; LoraRunner) after the base matmul (dense
    // matrices, the LM head, and per-expert stacks of a mixture of experts,
    // expert e's slice for the tokens routed to it). nullptr: none, and
    // passes cost what they did without adapters (one branch per matmul).
    // Checks every target first: InvalidArgument for a tensor the model does
    // not have or of another shape, Unsupported for one the forward pass does
    // not multiply as a whole (latent attention's per-head attn_k_b /
    // attn_v_b); the adapter in use is then none. The context is left as it
    // is: the caller decides whether rows computed with another adapter may
    // be kept (the engine never keeps them). `adapter` must outlive its use.
    // Not applied by the router's expert prediction, which only prefetches.
    // `pool` (optional) spreads the updates of large matmuls over its
    // threads (the same bits on any number); it must be the pool the CPU
    // backend runs on, or one not in use during passes.
    Status set_adapter(const LoraAdapter* adapter, float scale = 1.0f, ThreadPool* pool = nullptr);
    [[nodiscard]] const LoraAdapter* adapter() const noexcept { return adapter_; }

    // ---- Several sequences (the engine's sequences) ----
    // Everything above works on the live context: the KV cache, the
    // positions it holds, the recurrent states with their rollback
    // checkpoints and snapshots, a pending tree pass and the adapter bound
    // with set_adapter(). Other sequences' contexts are parked in Context
    // objects, sharing the weights, the scratch and the expert cache.
    //
    // A new, empty context laid out as the live one (KV cache settings,
    // recurrent states, rollback window); no adapter. Its KV pages are
    // mapped as it is used, its recurrent states are allocated now
    // (recurrent_state_bytes(), zero).
    [[nodiscard]] Result<std::unique_ptr<Context>> make_context() const;
    // Makes the context parked in `other` the live one and parks the live
    // one in `other`: O(1), no copy. An empty Context() (made with its
    // default constructor) may take the live one, leaving no live context:
    // swap again before any other call. Not during a pass.
    void swap_context(Context& other) noexcept;
    // One sequence's part of a forward_batch() pass: `tokens` appended to
    // `context` (nullptr: the live context), its logits (None: none, Last:
    // its last token's, All: every token's) and whether its recurrent states
    // are checkpointed for a rollback (set_checkpointing(), per item).
    struct BatchItem {
        Context* context = nullptr;
        std::span<const int32_t> tokens;
        Logits logits = Logits::Last;
        bool checkpoint = false;
    };
    // One pass over several sequences: every item's tokens go through each
    // block together, so the matmuls run over all of their rows at once (a
    // mixture of experts reads the union of the experts they choose), while
    // each item attends to its own KV cache at its own positions, continues
    // its own recurrent states, applies its own context's adapter and, with
    // latent attention, attends in the form its own pass would take. Each
    // item's rows are the bits forward() over its tokens alone gives on the
    // CPU backend (a row's matmul does not depend on the batch around it,
    // and attention depends on the token's position only), and its context
    // ends where that forward() leaves it. `out[i]` receives item i's
    // logits (n_vocab floats per row; empty for None), valid until the
    // next pass. Hooks: head masking, FFN matmul overrides and `cancel`
    // apply to every item; early exit and FFN skipping are single-token
    // decoding steps' and are refused (InvalidArgument), as are items
    // sharing a context, an empty item, a context awaiting keep_path() and
    // a token outside the vocabulary. ContextFull as forward(). Cancelled
    // between blocks: every item's context goes back to where the pass
    // started (abandon_pass(); a hybrid context without a snapshot there
    // is reset, which the caller sees as n_past() 0). On any other failure
    // the contexts keep their positions as before the pass, but, as after a
    // failed forward(), recurrent states may be partly advanced: callers
    // reset the contexts (the engine forgets each sequence's).
    Status forward_batch(std::span<const BatchItem> items, const Route& route, ThreadPool& pool,
                         std::vector<std::span<const float>>& out, const ForwardHooks* hooks = nullptr);
    // Positions a parked context holds (n_past() of the live one).
    [[nodiscard]] static int32_t positions_of(const Context& context) noexcept;

private:
    // Matrices of one block, by role; unused roles stay nullptr.
    enum WeightRole : size_t {
        kQ, kK, kV, kO,              // attention (kQ yields [q | gate] per head when gated)
        kQkv, kZ, kAlpha, kBeta, kSsmOut,  // DeltaNet: conv input, output gate, decay, write strength, output
        kGate, kUp, kDown,           // SwiGLU FFN
        kRouter, kGateExps, kUpExps, kDownExps,  // MoE: router [n_embd x n_expert], experts stacked on dim 2
                                                 // (fused_gate_up: kUpExps holds [gate | up], no kGateExps)
        kGateShared, kUpShared, kDownShared,     // MoE: shared expert
        // Latent attention (kQ: attn_q when queries are not low-rank, kO: attn_output):
        // attn_q_a / attn_q_b, attn_kv_a_mqa [n_embd x (kv_rank + rope_dim)], and per head
        // attn_k_b [nope_dim -> kv_rank] (absorbs a query) and attn_v_b [kv_rank -> v_dim] (expands a value)
        kQa, kQb, kKvA, kKb, kVb,
        kWeightRoles
    };
    // The bound weights of one block (its structure is config_.blocks[layer]).
    struct Layer {
        int32_t kv_slot = -1;     // layer index inside the KV cache (attention)
        int32_t state_slot = -1;  // index into states_ (DeltaNet)
        int32_t window = 0;       // attention: visible positions (BlockPlan::windowed); 0: all
        int32_t rope_table = 0;   // attention: index into rope_tables_
        std::array<const TensorView*, kWeightRoles> w{};
        std::vector<float> attn_norm, ffn_norm, q_norm, k_norm, bq, bk, bv;
        std::vector<float> post_mixer_norm, post_ffn_norm;  // optional (ArchSpec): applied before each residual add
        std::vector<float> q_a_norm, kv_a_norm;  // latent attention: norms of the low-rank query and of the latent
        std::vector<float> conv1d;   // [channel][kernel] taps
        std::vector<float> ssm_a;    // per value head: -exp(A_log)
        std::vector<float> ssm_dt;   // per value head: decay bias
        std::vector<float> ssm_norm; // gated RMSNorm weight, per head dim
        std::vector<float> shared_gate;   // optional: sigmoid(shared_gate . x) scales the shared expert
        std::vector<float> expert_bias;   // optional: added to probabilities for selection only
    };
    // Recurrent memory of one DeltaNet block.
    struct RecurrentState {
        std::vector<float> conv;  // last (kernel - 1) inputs per channel, oldest first: [channel][kernel - 1]
        std::vector<float> ssm;   // per value head a [head_dim (value) x head_dim (key)] matrix
        // Rollback ring: slot s holds [conv | ssm] after the token at the
        // position checkpoint_pos_[s] (slot = position % rollback window).
        std::vector<float> checkpoints;
    };

    // Weight views of one block, rebased into a triple-buffer slot when a
    // layer source is attached. Releases the slot on destruction.
    class BlockWeights;

    // How one group of blocks rotates queries and keys: pair i turns by
    // inv_freq[i] / scale radians per position. Most models have one table;
    // windowed blocks with ModelConfig::rope_base_swa have a second.
    struct RopeTable {
        std::vector<float> inv_freq;  // per rotary pair, frequency factors included
        float scale = 1.0f;           // linear position scaling
        float magnitude = 1.0f;       // multiplies the rotated dimensions (ModelConfig::rope_magnitude)
    };

    Transformer() = default;
    Status bind_weights();
    void rope(float* vec, int32_t n_heads, int32_t pos, const RopeTable& table) const;
    // The same over heads `stride` floats apart (rope_dim channels rotated at
    // the start of each): latent attention's expanded queries.
    void rope(float* vec, int32_t n_heads, size_t stride, int32_t pos, const RopeTable& table) const;
    // One sequence's part of the current pass (segs_): its tokens are rows
    // [first, first + n) of the batch, and the state they read and write is
    // its context's, the live one's (forward(), and every pass of a single
    // sequence) or a parked one's (forward_batch()). Resolved once per pass,
    // so the blocks never look a context up.
    struct BoundAdapter;
    using AdapterMap = std::unordered_map<std::string_view, BoundAdapter>;
    struct Segment {
        int32_t first = 0;
        int32_t n = 0;
        KvCache* kv = nullptr;
        int32_t n_past = 0;  // positions held before the pass
        std::vector<RecurrentState>* states = nullptr;
        int32_t rollback = 0;     // its rollback window (ring slots)
        bool checkpoint = false;  // record checkpoints of its last `rollback` tokens
        const AdapterMap* adapter = nullptr;  // nullptr: none
        bool expanded = false;    // latent attention in the expanded form
        Logits logits = Logits::None;
        Context* context = nullptr;  // nullptr: the live context
    };
    // Position of batch row t of segment s: s.n_past + its index in s,
    // s.n_past + its depth in a tree pass (forward_tree), or positions_[t]
    // while run_block_at() runs a batch at scattered positions. Its RoPE
    // angle and what it may see follow from it.
    [[nodiscard]] int32_t pos_of(const Segment& s, int32_t t) const noexcept {
        if (!positions_.empty()) return positions_[static_cast<size_t>(t - s.first)];
        return s.n_past + (tree_depth_.empty() ? t - s.first : tree_depth_[static_cast<size_t>(t - s.first)]);
    }
    // Cache slot holding row t's KV rows: its position, except in a tree
    // pass, where tokens at one depth are alternatives and each has the slot
    // n_past + its index until keep_path().
    [[nodiscard]] int32_t row_of(const Segment& s, int32_t t) const noexcept {
        return positions_.empty() ? s.n_past + (t - s.first) : positions_[static_cast<size_t>(t - s.first)];
    }
    // The segment of batch row t (the only one outside forward_batch()).
    [[nodiscard]] size_t segment_of(int32_t t) const noexcept {
        return segs_.size() == 1 ? 0 : static_cast<size_t>(seg_of_[static_cast<size_t>(t)]);
    }
    // The live context as a segment of the pass: rows [first, first + n).
    [[nodiscard]] Segment live_segment(int32_t first, int32_t n, Logits logits) noexcept;
    // A pass of the live context alone over n rows (segs_, adapters).
    void begin_live_pass(int32_t n, Logits logits) noexcept;
    // Sizes the per-token scratch for a batch of n tokens.
    void size_batch(int32_t n);
    // One block over the batch in x_ (n rows, token t at pos_of(t)): the
    // token mixer and the FFN with their norms and residual adds, as
    // forward() runs it (it is forward()'s loop body).
    Status run_block(int32_t layer, const BlockWeights& w, int32_t n, const Route& route, ThreadPool& pool,
                     const HeadMaskHook* head_mask, FfnSkipHook* ffn_skip, FfnMatmulHook* ffn_hook);
    // Mixers: read the normalized input from xb_, leave the block output (before
    // the residual add) in xb_.
    Status attention_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route, ThreadPool& pool,
                           const uint8_t* head_mask);
    Status delta_net_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route, ThreadPool& pool);
    // Multi-head latent attention (ModelConfig::mla), in the form forward()
    // chose for the pass (latent_expanded_): absorbed queries against the
    // cached [k_pe | latent] rows and the weighted latent expanded per head,
    // or keys and values expanded per head from the cached latents
    // (latent_expanded()); then attn_output.
    Status latent_attention_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                  ThreadPool& pool, const uint8_t* head_mask);
    // Latent attention's projections of the n normalized rows in xb_: each
    // token's key row [k_pe | RMSNorm(latent)] into k_ (unrotated) and, with
    // `queries`, its queries [q_nope | q_pe] per head into mla_q_
    // (unrotated). On `backend`.
    Status latent_project(int32_t layer, const BlockWeights& w, int32_t n, const Route& route, Backend* backend,
                          bool queries);
    // Absorbed form: each head's query from mla_q_ as [q_pe | W_kb,h^T
    // q_nope,h] into q_ (unrotated). The per-head matrices run on the CPU.
    Status latent_absorb(const BlockWeights& w, int32_t n, const Route& route);
    // Expanded form, once segment s's key rows are stored and its queries in
    // mla_q_ rotated: attends each head over the positions its tokens see,
    // their keys [W_kb,h c | k_pe] and values W_vb,h c expanded from the
    // cached rows tile by tile, and leaves each of its tokens' heads (v_dim
    // channels each) in att_ (rows of other segments untouched). See the
    // definition for the layout and why the result does not depend on the
    // batch.
    Status latent_expanded(int32_t layer, const BlockWeights& w, const Segment& s, ThreadPool& pool,
                           const uint8_t* head_mask);
    // The positions the tokens of segment s see in a block with `window`,
    // merged into ascending disjoint ranges [first, end) (mla_union_), and
    // the number of (token, visible position) pairs.
    int64_t latent_visible(const Segment& s, int32_t window);
    // Expanded form: positions per tile of expanded keys and values, from
    // the model's shape only (so tile bounds, which set the order a token's
    // softmax is merged in, never depend on the pass).
    [[nodiscard]] int32_t latent_tile() const noexcept;
    // Per-head slices of a [cols, rows, n_head] tensor as matmul items, one
    // contiguous input and output block per head: input head h, token t at
    // in + (h * n + t) * cols, output at out + (h * n + t) * rows.
    Status per_head_matmul(const Route& route, const TensorView& stacked, const float* in, float* out, int32_t n);
    // BlockPlan::fused_qkv: projects xb_ through the block's attn_qkv on
    // `backend` and splits each token's [q | k | v] into q_, k_ and v_.
    Status project_fused_qkv(const Route& route, Backend* backend, const BlockWeights& w, int32_t n);
    // FFNs: read the normalized input from xb_, leave the output in xb_.
    Status dense_ffn(int32_t layer, const BlockWeights& w, int32_t n, const Route& route, FfnMatmulHook* hook);
    Status moe_ffn(int32_t layer, const BlockWeights& w, int32_t n, const Route& route);
    // How many of a token's top-k experts run under expert_mass_ and
    // max_experts_: `order`
    // starts with the top-k by selection score, `probs` are their router
    // probabilities and `top_sum` their total.
    [[nodiscard]] size_t experts_to_run(std::span<const float> probs, std::span<const int32_t> order,
                                        float top_sum) const;
    // Expert prefetch: applies block `layer`'s router (and FFN norm) to the
    // current residual stream (x_) and queues the top-k experts it picks in
    // the ExpertStore. forward() calls it for block l + 1 once block l's mixer
    // ran: that residual predicts block l + 1's choice better than the one
    // entering block l (83% vs 77% of the top-8 on Qwen3.6-35B-A3B), while the
    // reads still overlap block l's experts and block l + 1's mixer.
    void predict_experts(int32_t layer, int32_t n, const Route& route);
    Status attach_expert_store(int64_t budget_option, uint64_t memory_budget, size_t reserve, int32_t requant_bits,
                               bool repack);
    // Softmax attention of the batch's queries (q_) over the cached keys and
    // values of attention block `kv_slot`, each segment's tokens over its own
    // cache (segments in the expanded latent form are left out: see
    // latent_expanded); `window` > 0 limits each query to its last `window`
    // positions (within what the KV cache keeps visible). Leaves each
    // token's heads in att_, value_dim channels per head.
    void attention(int32_t kv_slot, int32_t window, int32_t n, ThreadPool& pool, const uint8_t* head_mask);
    [[nodiscard]] KeyScalingShape key_scaling_shape() const;
    // Per-head Q/K norms of block `L` on one token's queries (n_head heads;
    // nullptr: none) and keys (n_head_kv heads), where the block has them.
    void qk_norm(const Layer& L, float* q, float* k) const;
    // Calibration (key_ranges_ set): one token's unrotated queries and keys of
    // block `L`, as the ranges need them whatever the Q/K norm order.
    void add_key_ranges(const Layer& L, const float* q, const float* k);
    // Divides one token's keys (n_head_kv heads) by the key scales of
    // attention block `kv_slot` and multiplies its queries (n_head heads;
    // nullptr: none) by them. No-op without scales.
    void scale_keys(int32_t kv_slot, float* q, float* k) const;
    Status matmul(const Route& route, Backend* backend, const TensorView& w, const float* x, float* y, int32_t n);
    // Grouped variant (shared input); falls back to the CPU as a whole group.
    Status matmul_group(const Route& route, Backend* backend, std::span<const TensorView* const> ws, const float* x,
                        std::span<float* const> ys, int32_t n);
    // The active adapters' items for matmuls ys[i] = x · ws[i]ᵀ of n rows
    // (matrices an adapter does not adapt are skipped) into adapter_items_:
    // with one segment, its adapter over all n rows; in a pass of several,
    // each segment's adapter over its own rows (adapter_rows_ while the
    // logits of forward_batch() are computed). False when none.
    bool collect_adapter_items(std::span<const TensorView* const> ws, const float* x, std::span<float* const> ys,
                               int32_t n);
    // Final norm + LM head over `rows` hidden rows starting at `first_row` of
    // x_, then the final soft-cap (if any) on `pool`.
    Status compute_logits(const Route& route, size_t first_row, int32_t rows, ThreadPool& pool);
    // The same over the hidden rows `rows` of x_ (any order), into logits_
    // rows 0, 1, ... (forward_batch()).
    Status compute_logits_rows(const Route& route, std::span<const int32_t> rows, ThreadPool& pool);
    // Early exit: writes K/V for blocks [from, n_layers) of the single token
    // at `pos` from the current hidden state, so later tokens stay coherent.
    Status propagate_kv(const Route& route, int32_t from, int32_t pos);

    std::unique_ptr<MmapLoader> file_;
    ModelConfig config_;
    int32_t context_length_ = 0;
    int32_t max_batch_ = 0;
    int32_t n_past_ = 0;
    std::span<const int32_t> positions_;  // run_block_at(): the batch's positions (see pos_of)
    // The current pass's segments (one but in forward_batch()), each row's
    // segment when there are several, and forward_batch()'s tokens one
    // segment after another.
    std::vector<Segment> segs_;
    std::vector<int32_t> seg_of_;
    std::vector<int32_t> batch_tokens_;
    // forward_batch()'s logits pass: per segment with an adapter, its rows
    // among the logits rows ({first, count, adapter}); empty otherwise.
    struct AdapterRows {
        int32_t first = 0;
        int32_t n = 0;
        const AdapterMap* adapter = nullptr;
    };
    std::vector<AdapterRows> adapter_rows_;
    bool logits_rows_ = false;  // forward_batch()'s logits pass: adapters by adapter_rows_
    bool adapters_in_pass_ = false;
    // forward_batch() of one token per sequence: decoding steps, which the
    // expert store serves as decoding (moe_ffn), however many rows the pass has.
    bool steps_only_ = false;
    // Tree pass (forward_tree): per token its depth, its parent (-1: the
    // root) and the batch index of its ancestor at each depth up to its own
    // (tree_ancestors_[t * tree_stride_ + d]), which attention reads the rows
    // of positions n_past_ + d through. tree_depth_ is empty outside the
    // pass; tree_parent_ stays until keep_path(), with tree_base_ (n_past_
    // before the pass, -1 when no tree is pending).
    std::vector<int32_t> tree_depth_, tree_parent_, tree_ancestors_;
    int32_t tree_stride_ = 0;
    int32_t tree_base_ = -1;
    int32_t last_exit_layer_ = -1;
    int32_t last_ffn_skips_ = 0;
    std::unique_ptr<KvCache> kv_;
    TripleBufferLoader* layer_source_ = nullptr;
    std::vector<int32_t> layer_items_;
    bool layer_shared_ = false;

    const TensorView* token_embd_ = nullptr;
    const TensorView* output_ = nullptr;
    std::vector<float> output_norm_;
    std::vector<Layer> layers_;
    std::vector<RecurrentState> states_;
    std::unique_ptr<ExpertStore> expert_store_;
    MemoryPlan memory_plan_;
    size_t extra_context_bytes_ = 0;  // TransformerOptions::extra_contexts' recurrent states, set aside at load
    int64_t decode_steps_ = 0;  // expert cache aging clock
    PhaseTimes phases_;
    int32_t rollback_window_ = 0;
    bool checkpointing_ = true;
    std::vector<int32_t> checkpoint_pos_;  // per ring slot: the position its checkpoints hold, -1 none
    [[nodiscard]] std::array<int64_t, 9> state_fingerprint() const noexcept;
    Status write_image(std::ostream& out, int32_t n, const std::function<const uint8_t*(int32_t)>& page,
                       const std::function<const uint8_t*(int32_t)>* window_page, std::span<const float> state,
                       std::span<const SnapshotImage> snapshots, bool live) const;
    struct StateSnapshot {
        int32_t pos = 0;           // n_past() when taken
        StateBuffer data;          // [conv | ssm] of every recurrent block, in states_ order
        bool pinned = false;       // the end of the last prompt (see snapshot_state)
    };
    std::vector<StateSnapshot> snapshots_;  // ascending pos, all <= n_past_; windowed rows pinned in the cache
    // Pins the windowed blocks' window before each snapshot (KvCache::pin()).
    void pin_snapshots();
    // Snapshot buffers begin_restore() kept for the context being restored
    // (at most kStateSnapshots with snapshots_): no allocation, first touch
    // or zero-fill per restore.
    std::vector<StateBuffer> spare_buffers_;
    ExpertPredictions predictions_;
    std::vector<int32_t> predicted_;  // sorted experts predicted for block predicted_layer_
    int32_t predicted_layer_ = -1;
    // Prefetch by confidence: every guess of block predicted_layer_ with its
    // best rank among the tokens' top-k, and per rank how often a guess was
    // then chosen. A rank is prefetched only while its guesses are right more
    // often than not: a right guess saves one read's wait, a wrong one costs
    // one read (flash time and energy) and evicts a cached expert.
    struct Guess {
        int32_t expert;
        int32_t rank;
    };
    std::vector<Guess> guesses_;
    std::vector<double> rank_guessed_, rank_used_;  // decayed counts per rank
    float expert_mass_ = 1.0f;          // TransformerOptions::expert_mass
    int32_t max_experts_ = 0;           // TransformerOptions::max_experts
    float skip_slow_ = 0.0f;            // TransformerOptions::skip_slow
    std::vector<int32_t> experts_kept_;  // per token of the current batch: top-k ranks run
    std::vector<RopeTable> rope_tables_;  // [0]: every block's but the windowed ones with their own base
    std::vector<float> key_scale_;          // per (attention block, KV head, channel); empty: none
    std::unique_ptr<KeyRanges> key_ranges_;  // while calibrate_keys() runs

    // Scratch, sized for the current batch.
    std::vector<float> x_, xb_, q_, k_, v_, att_, hb_, hb2_, logits_;
    std::vector<float> part_max_, part_sum_, part_out_;  // attention: partial softmax per span (attention())
    std::vector<KvCache::RowView> row_views_;           // attention: each segment's rows of the block
    std::vector<int32_t> attn_rows_;                    // attention: the rows attending, when not every row
    std::vector<quant::BlockQ8_0> q8_queries_;          // attention: Q8_0 queries for quantized caches
    std::vector<float> qg_, gate_;                     // gated attention: raw [q | gate] rows, gates
    std::vector<float> qkv_;                           // fused attention projection: raw [q | k | v] rows
    std::vector<float> mla_q_, mla_qa_, mla_kv_;       // latent attention: queries, low-rank queries, [latent | k_pe]
    std::vector<float> mla_heads_in_, mla_heads_out_;  // latent attention: per-head matmul inputs and outputs
    std::vector<TensorView> mla_head_views_;           // latent attention: per-head slices of attn_k_b / attn_v_b
    std::vector<Backend::MatmulItem> mla_items_;
    // Latent attention, expanded form: packed expansion weights, a tile's
    // latents and expanded rows, the positions they hold, per (token, head)
    // running softmax maximum and sum, the batch's visible ranges.
    std::vector<float> mla_wp_, mla_c_, mla_ex_, mla_max_, mla_sum_;
    std::vector<size_t> mla_col_offset_;
    std::vector<int32_t> mla_rows_;
    std::vector<std::pair<int32_t, int32_t>> mla_union_;
    LatentForm latent_form_ = LatentForm::Auto;  // TransformerOptions::latent_form
    LatentFormChooser latent_chooser_;
    LatentStats latent_stats_;
    bool latent_expanded_ = false;  // the current pass's form
    double latent_ms_ = 0.0;        // the current pass's form-dependent time, every latent block
    std::vector<float> mix_, z_, alpha_, beta_, dn_;  // DeltaNet projections and output
    std::vector<float> router_, moe_out_, ein_, eout_, eh_, eh2_, shared_out_, shared_scale_;  // MoE scratch
    std::vector<float> ranked_out_;  // MoE: weighted output of each (token, top-k rank), summed in rank order
    std::vector<float> pred_in_, pred_logits_;                       // expert prediction scratch
    std::vector<float> x_block_in_;  // residual entering the block (FFN-skip hook only)
    // The active adapter (set_adapter): its targets by base tensor name, with
    // their scales; empty when none (the only check a pass pays then).
    struct BoundAdapter {
        const LoraTarget* target = nullptr;
        float scale = 0.0f;
    };
    const LoraAdapter* adapter_ = nullptr;
    AdapterMap adapter_targets_;
    LoraRunner adapter_runner_;
    ThreadPool* adapter_pool_ = nullptr;
    std::vector<LoraRunner::Item> adapter_items_;
    std::vector<Backend::MatmulItem> adapter_batch_;  // base and A·x matmuls of one group
};

// A parked sequence's state (Transformer::make_context(), swap_context(),
// forward_batch()): what the live context holds in the transformer's own
// members. Default-constructed it is empty (no KV cache): only a place for
// swap_context() to park the live context in. Not thread-safe; used under
// the embedder's lock on the transformer.
class Transformer::Context {
public:
    Context() = default;
    ~Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    // Whether it holds a context (else it is an empty place to park one).
    [[nodiscard]] bool holds() const noexcept { return kv_ != nullptr; }
    // KV cache bytes it maps now.
    [[nodiscard]] size_t kv_bytes() const noexcept { return kv_ ? kv_->bytes_in_use() : 0; }

private:
    friend class Transformer;
    std::unique_ptr<KvCache> kv_;
    int32_t n_past_ = 0;
    std::vector<RecurrentState> states_;
    std::vector<int32_t> checkpoint_pos_;
    int32_t rollback_window_ = 0;
    bool checkpointing_ = true;
    std::vector<StateSnapshot> snapshots_;
    std::vector<StateBuffer> spare_buffers_;
    int32_t tree_base_ = -1;
    std::vector<int32_t> tree_parent_;
    const LoraAdapter* adapter_ = nullptr;
    AdapterMap adapter_targets_;
};

}  // namespace liyab

#endif  // LIYAB_CORE_TRANSFORMER_H
