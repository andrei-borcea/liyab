// Liyab — inference engine: ties the mmap loader, device detection, power
// manager, backends and (optionally) speculative decoding together.
#ifndef LIYAB_ENGINE_H
#define LIYAB_ENGINE_H

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "liyab/device_detect.h"
#include "liyab/grammar.h"
#include "liyab/power_manager.h"
#include "liyab/types.h"

namespace liyab {

// Experimental features: require a build with LIYAB_ENABLE_EXPERIMENTAL=ON,
// otherwise Engine::create() fails with ErrorCode::Unsupported. Both apply to
// non-speculative decoding only. See include/liyab/experimental/.
struct ExperimentalConfig {
    bool early_exit = false;              // skip remaining blocks on confident tokens (lossy)
    float early_exit_threshold = 0.98f;   // max softmax probability needed to exit
    int32_t early_exit_min_layer = -1;    // -1: n_layers / 2
    int32_t early_exit_interval = 4;      // probe every N blocks
    bool head_pruning = false;            // mask low-importance heads when hot / LowPower (lossy)
    float head_keep_ratio = 0.75f;        // heads kept per block while pruning
    bool egls = false;                    // entropy-guided FFN skipping (lossy)
    float egls_threshold = 0.002f;        // skip a block's FFN when its entropy delta is below this
    bool tdss = false;                    // 2:4 sparse FFN weights while thermally throttled (lossy)
    bool tdss_force = false;              // use the sparse FFN even when cool (benchmarks)
    // (KV dedup, the experimental persistent prefix KV cache, is retired: the
    // prefix cache's store keeps a shared prefix once, on storage across
    // processes, for every model, within its caps; see prefix_cache_dir.)
};

struct EngineConfig {
    std::string model_path;           // GGUF file (F32/F16/Q8_0/Q4_0/Q4_1 tensors)
    std::string draft_model_path;     // optional: enables speculative decoding
    int32_t n_threads = 0;            // 0: performance cores, minus one when there is no efficiency cluster
    int32_t context_length = 0;       // 0: automatic: what 256 MiB of KV cache holds (4096..32768), at most the training context
    int32_t sliding_window = 0;       // 0: full attention; else attend to the last N tokens
    int32_t kv_sink_tokens = 8;       // with a sliding window: first N tokens stay visible (attention sinks)
    // Paged KV cache format. Q8_0 (the default) halves F16's memory. For
    // models whose keys carry channels far larger than the rest (Qwen2.5,
    // Qwen3), keys are stored with per-channel scales measured once per model
    // (see cache_dir), which takes their mean KL against F16 from 0.05-0.07 to
    // ~0.002; other models keep their keys as computed (README, KV cache). F16
    // needs no scales. Q4_0 / Q4_1 (INT4, -72% / -69% vs F16) are scaled the
    // same way and still trade accuracy for memory on long contexts.
    KvCacheType kv_cache_type = KvCacheType::Q8_0;
    // Stream transformer blocks through the triple-buffered loader (3 block
    // buffers: fetch | prepare | execute) instead of reading the mmap in place.
    // For models larger than RAM only: weight memory is bounded at 3 blocks,
    // but every token re-reads every block from storage, so decode speed is
    // capped by flash bandwidth / model size. With LIYAB_ENABLE_EXPERIMENTAL
    // stage 1 uses O_DIRECT / io_uring.
    bool triple_buffer_loading = false;
    // Mixture-of-experts models: RAM cache for routed experts streamed from
    // storage with direct I/O, in MiB. -1: automatic (only when the model does
    // not fit in RAM; uses the free RAM left after the other weights), 0:
    // never stream experts, > 0: always, with this cache size.
    int64_t expert_cache_mb = -1;
    // Total memory the engine may keep resident (weights, expert cache,
    // streaming slots), in MiB; 0: limited by free RAM only. Streaming
    // decisions use min(free RAM, budget). Set it on platforms with per-app
    // caps the OS counters do not show (HyperOS stops apps above 6 GiB PSS).
    int64_t memory_budget_mb = 0;
    // With expert streaming (MoE): convert the resident Q8_0 matrices
    // (attention / DeltaNet projections, shared experts) to Q4_K (4) or Q5_K
    // (5) at load. Lossy, opt-in: fewer bytes read per token, and the memory
    // saved goes to the expert cache. Converted on every core at load;
    // ignored when experts are not streamed. 0: off. -1: automatic, only when
    // the expert cache would otherwise hold under 10% of the experts (Q5_K if
    // that frees enough memory, else Q4_K); see Engine::memory_plan().
    int32_t requant_bits = 0;
    // MoE: per token, run only the fewest top-ranked experts of the top-k
    // whose router probabilities cover this fraction of the top-k's total
    // (e.g. 0.9), renormalized over the kept ones. Lossy, opt-in: fewer
    // expert reads and less compute per token. >= 1: all top-k (default).
    float moe_expert_mass = 1.0f;
    // MoE: at most this many experts per token (the top ones, renormalized);
    // 0: the model's own top-k. Lossy when below it.
    int32_t moe_max_experts = 0;
    // MoE with streamed experts: a chosen expert that is not in RAM yet is
    // skipped instead of waited for when, for every token using it, its
    // router weight is below this share of the token's experts (the others
    // renormalized). Lossy, opt-in: cuts stalls on low-impact experts, and the
    // output then depends on I/O timing. 0: off.
    float moe_skip_slow = 0.0f;
    // Prefix cache, MiB of RAM at most (0: off): contexts another
    // conversation's prompt replaced (or an embed() / rerank() call) are kept
    // and restored when a prompt continues one, instead of processing it
    // again, cut back to where the two diverge (whenever that gives back more
    // than the live context keeps). A prompt replaces the context when it
    // shares less than half of it, or leaves it before the end of its last
    // prompt and rewinding would process at least 128 tokens again (another
    // chat after a long shared system prompt, an edited message; on a hybrid
    // model, back to its nearest state snapshot); a draft typed ahead and
    // then edited after the last reply does not. Contexts are kept in a
    // content-addressed store of KV chunks: one KV page of tokens each, keyed
    // by every token before it plus the model, cache layout, key scales and
    // kernels, so a prefix conversations share (system prompt, tools) is
    // stored once and any prompt starting with it restores it; recurrent
    // (DeltaNet) states are kept as images at the positions the context's
    // state snapshots hold, and a hybrid model restores up to the deepest
    // one its prompt reaches. Least recently used chunks out first. Taken out
    // of memory_budget_mb; on a MoE whose experts stream, only as far as the
    // expert cache can spare: it never takes the cache under 10% of the
    // experts, so it never causes a requantization (requant_bits -1). On
    // Qwen3.6-35B-A3B a 2000-token conversation costs ~21 MiB of chunks plus
    // 62.8 MiB per state image. Off with a sliding window (released positions
    // are gone) or a draft model (it keeps no reusable context).
    int64_t prefix_cache_mb = 0;
    // Prefix cache on storage: what RAM does not hold (prefix_cache_mb,
    // possibly 0) goes to files in this directory (created if missing), at
    // most prefix_cache_disk_mb (0: off), least recently used out first, and
    // is read back when a prompt continues it. No RAM spent: worth it
    // wherever RAM is tight and storage is fast (UFS 4 writes or reads a
    // ~100 MB context in a fraction of a second; processing it again takes
    // minutes on a large MoE). The files outlive the engine: when destroyed,
    // it sets its context aside there too, and a later engine (or process)
    // with the same model and settings restores from them; other models'
    // files are kept and count against the cap. Writes go through a
    // temporary file and a rename; when it opens the directory the engine
    // removes what it cannot trust (interrupted writes, damaged files, other
    // format versions, the ctx-*.state files of builds before the store).
    // One engine at a time uses a directory (an OS lock on kv.lock there):
    // another one, in this process or another, runs without the storage tier.
    // Chunks are written compressed, losslessly (F16 rows ~1.17x smaller,
    // Q8_0 ~1.04-1.09x), and decoded on the engine's threads when read back;
    // the cap counts the bytes written. Earlier builds' raw files stay readable.
    std::string prefix_cache_dir;
    int64_t prefix_cache_disk_mb = 0;
    // Chunks on storage are compressed losslessly (README, Prefix cache).
    // true re-quantizes them to 4 bits instead (keys per channel over the
    // chunk's positions, values in 32-channel blocks): 3.4x smaller than an
    // F16 cache's rows, 1.8x than Q8_0, at a measured cost (README); lossy,
    // so off by default. A lossy engine keys its chunks apart: they are never
    // restored by an engine without it, nor its lossless ones by it.
    bool prefix_cache_lossy = false;
    // What the engine measures once per model and keeps, in this directory
    // (created if missing): today the key scales of a quantized KV cache
    // (README, KV cache), a few KB per model, at most 16 models, least
    // recently used out first. Empty: measured again at every load (on the
    // reference phone 0.27 s for a 4B model, 1.4 s for a 35B MoE whose experts
    // stream, 7.4 s for a dense 27B larger than RAM).
    std::string cache_dir;
    // Most drafts per speculative step (k). 3 by default: a verification pass
    // then holds 4 tokens, exactly one tile of the repacked CPU kernels.
    int32_t draft_tokens = 3;
    // Without a draft model: draft up to draft_tokens tokens from the text
    // itself, by the suffix drafter or by context lookup (suffix_drafts).
    // Same output distribution, no extra weights, any model; pays off when
    // the text repeats earlier text and on models that verify k tokens for
    // about the cost of one (dense models in RAM).
    bool lookup_drafts = false;
    // Learn on the device how many drafts per step pay off (0..draft_tokens;
    // see SpeculativeDecoder). false: always draft_tokens, which keeps sampled
    // output reproducible for a fixed seed.
    bool adaptive_drafts = true;
    // How lookup_drafts drafts. true (the default, measured ahead of context
    // lookup on every task, README "Speculative decoding"): the suffix drafter
    // (core/suffix_drafter.h), the longest earlier match (2..32 tokens) in
    // this conversation or in earlier ones of this engine and the
    // continuation most of its occurrences agree on, verified at a length
    // chosen per step from calibrated acceptance probabilities and the
    // measured cost of a step per draft count (adaptive_drafts). Earlier
    // conversations are kept in RAM for the engine's lifetime, at most 262144
    // tokens (~2 MiB) in at most 64 sequences, least recently used out first.
    // Sampled output is the same tokens as plain decoding draws from the seed
    // (CPU backend), adaptive or not. false: context lookup
    // (SpeculativeDecoder::lookup). No effect without lookup_drafts or with a
    // draft model. Either way draft_tokens is the most drafts per step, and
    // hybrid models keep draft_tokens + 1 recurrent-state checkpoints.
    bool suffix_drafts = true;
    std::optional<BackendKind> backend;  // force one backend; default: device ranking
    std::optional<bool> streaming;       // force mmap streaming mode; default: by RAM
    PowerConfig power;                   // profile, thermal thresholds, pacing
    bool thermal_polling = true;         // background thermal sampling thread (while prefilling or generating)
    ExperimentalConfig experimental;
    // KV cache format of a pooling model (one whose GGUF declares
    // <arch>.pooling_type: an embedding model or a reranker, see Pooling),
    // used instead of kv_cache_type. Such a model holds one input in its
    // cache at a time and keeps no conversation, so the memory a quantized
    // cache saves is small, while its rounding moves every vector and score:
    // on Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B, Q8_0 kept a cosine of
    // 0.9995 with transformers' embeddings and moved scores by up to 0.004,
    // against 0.9999997 and 0.00006 with F16 (README, Embeddings and reranking).
    KvCacheType pooling_kv_cache_type = KvCacheType::F16;
};

// GGUF `general.architecture` values this build can run, comma-separated
// (e.g. "llama,mistral,qwen2,..."), so front ends can filter models before
// downloading them.
LIYAB_API std::string supported_architectures();

// Receives each decoded piece (complete UTF-8 only) and the token that
// completed it. Return false to stop generation.
using TokenCallback = std::function<bool(std::string_view piece, int32_t token)>;

// Structured output: called after each decoded piece with the output so far;
// returns text the output must continue with (empty: none). The engine runs
// it through the model in one batched pass, as cheap as a short prefill, and
// emits it like generated text; decoding resumes after it. That pass also
// yields the next token's logits, so forced text costs no decoding step of
// its own (on the CPU backend the next token is the same bits a separate step
// would give; with the experimental early exit or EGLS the next token keeps
// its own step). Used for the fixed parts of a tool call and to complete a
// name once only one is possible. Called on the generating thread, only at
// whole-UTF-8 boundaries, after every token that ends one, also with
// speculative decoding: tokens a speculative step returned past that point
// are dropped. Forced text that ends inside a character (a cut buffer) is
// emitted up to it, and the rest with the token that completes it. Forced
// tokens count in GenerationStats::forced_tokens, not generated_tokens.
using ForceCallback = std::function<std::string(std::string_view generated)>;

// Constrained decoding (structured output, see liyab/grammar.h). While a
// grammar is active, every token whose bytes it cannot accept at that point
// is masked before sampling (the sampler's temperature / top-k / top-p see
// only allowed tokens), so the output matches it: a JSON Schema's valid
// values, any JSON, a grammar text. Details:
//  * End-of-generation tokens are allowed only where the grammar accepts.
//  * Text every continuation shares (a schema's keys, separators, brackets,
//    the rest of a unique enum value) is forced, not sampled: it goes through
//    the model in one batched pass, as ForceCallback text does, and counts in
//    GenerationStats::forced_tokens. Forcing stops at a choice and never ends
//    inside a character.
//  * Token healing: when a longer token the grammar allows starts with the
//    forced text's last token (` {"` after a forced ` {`), that last token is
//    not forced; the model samples it or the longer one, under the mask, so
//    it goes on from the tokenization it was trained on. A forced token the
//    model would have merged with its next choice skews that choice.
//  * Speculative decoding (a draft model or lookup drafts) is off while a
//    grammar is active: tokens are decoded one at a time, since drafts would
//    need a mask at every verified position. Unconstrained stretches of the
//    same generation use it as usual.
//  * max_tokens and cancellation can still cut a match short; the caller
//    sees it from GenerationStats (generated_tokens == max_tokens,
//    cancelled).
//  * The engine keeps each grammar's states, transitions and masks (at most
//    32 MiB per grammar; the 16 most recently used grammars, 64 MiB together
//    between generations), so a grammar used again
//    (a request with the same tools) does not pay for its masks twice.
//    GenerationStats::constrained_tokens and grammar_ms report the cost.

// A continuation of a structured generation (ContinuationCallback): `text`
// is forced, as ForceCallback's is; then, when `grammar` is set, the output
// continues with a span that matches it. A span ends when its match is
// complete and nothing more can follow (e.g. the closing brace of a JSON
// object), and the callback is asked again at once: its next text and span
// join the same forced pass.
struct Continuation {
    std::string text;
    std::optional<Grammar> grammar;
};
// Called with the output so far before the first token (empty output) and,
// while no span is active, after every token that ends a whole character
// (as ForceCallback), also with speculative decoding (tokens a step returned
// past that point are dropped). Called on the generating thread.
using ContinuationCallback = std::function<Continuation(std::string_view generated)>;

// Options of Engine::generate. At most one of the two is set (both:
// InvalidArgument).
struct GenerateOptions {
    // The whole output matches this grammar. Generation ends as soon as the
    // match is complete and nothing can follow (no end-of-generation token is
    // needed), or at an end-of-generation token where the grammar accepts.
    std::optional<Grammar> grammar;
    // Forced text and constrained spans at chosen points of the output (a
    // tool call's syntax forced, its arguments constrained by their schema).
    ContinuationCallback continuation;
};

// How a model's chat template lays out a conversation, found by rendering
// probe conversations (Engine::chat_format). A front end that keeps a
// conversation's context between turns builds each prompt from these pieces
// instead of rendering the whole conversation again, so the prompt of one
// turn is always a prefix of the next one's (replies are kept exactly as
// generated):
//
//   with a system message:    system_prefix + SYSTEM + system_suffix + first_user_prefix
//   without one:              no_system_head
//   then the first user turn: USER + user_suffix + generation_prompt + REPLY + assistant_end,
//   every later user turn:    user_prefix + USER + user_suffix + generation_prompt + REPLY + assistant_end
//
// Only valid when prefix_stable: the template renders n turns as a prefix
// of n + 1 turns, every reply as generation_prompt + reply + assistant_end,
// and every user turn alike. Templates that render earlier turns differently
// once the conversation goes on (Qwen3 / Qwen3.5 drop the reasoning of past
// replies) are not, and `reason` says what differs.
struct ChatFormat {
    bool prefix_stable = false;
    std::string reason;
    std::string system_prefix, system_suffix;
    std::string first_user_prefix;  // between the system block and the first user text
    std::string no_system_head;     // before the first user text when there is no system message
    std::string user_prefix, user_suffix;
    std::string generation_prompt;  // what add_generation_prompt appends: the reply's opening
    std::string assistant_end;      // what closes a reply
};

// ---- Pooling models: embeddings and reranking -----------------------------
// A pooling model turns a text into one vector (an embedding model such as
// Qwen3-Embedding) or a (query, document) pair into a relevance score (a
// reranker such as Qwen3-Reranker), from the decoder's final hidden states.
// What a model does is data from its GGUF, never per-model code:
//  * <arch>.pooling_type (the GGUF numbering below) says how hidden states
//    become one vector;
//  * tokenizer.ggml.add_eos_token says whether every input ends with the EOS
//    token, as the model's Hugging Face tokenizer appends it (Qwen3-Embedding:
//    <|endoftext|>); a file converted without that flag needs the EOS token's
//    text at the end of each input;
//  * a reranker carries a ranking head (cls.output.weight, one row per label
//    of <arch>.classifier.output_labels, optional cls.output.bias) and the
//    prompt of one pair (tokenizer.chat_template.rerank, with "{query}" and
//    "{document}" where they go).

// How a model pools (<arch>.pooling_type), and what EmbedOptions asks for.
enum class Pooling : int32_t {
    Auto = -1,  // EmbedOptions only: the model's own pooling_type
    None = 0,   // the model declares none (a generative model)
    Mean = 1,   // the mean of every token's final hidden state
    Cls = 2,    // the first token's: encoder models; in a causal decoder it does not depend on the rest of the text
    Last = 3,   // the last token's: decoder embedding models (Qwen3-Embedding)
    Rank = 4,   // a ranking head over the last token's state: rerankers (Engine::rerank)
};

struct EmbedOptions {
    // Auto: the model's pooling_type, which must be Mean or Last (a model
    // without one, or a reranker, is refused: Unsupported). Mean or Last apply
    // to any model, a generative one included. Cls, None and Rank: InvalidArgument.
    Pooling pooling = Pooling::Auto;
    // > 0: keep only the first `dimensions` values of each vector, for models
    // trained for it (Matryoshka representation learning: Qwen3-Embedding
    // supports 32..1024), before normalizing. 0: all embedding_size() values;
    // more than embedding_size(): InvalidArgument.
    int32_t dimensions = 0;
    // L2-normalize every vector (unit length, so a dot product is the cosine).
    // false: the pooled hidden state as is.
    bool normalize = true;
    // An input longer than the context (Engine::context_length(), its BOS and
    // EOS included) is InvalidArgument; true: cut its text's tokens to fit,
    // keeping BOS and EOS (as a Hugging Face tokenizer truncates).
    bool truncate = false;
};

struct RerankOptions {
    // The prompt of one pair, with "{query}" and "{document}" where they go
    // (each replaced once, in the template only); empty: the model's own
    // (tokenizer.chat_template.rerank). Qwen3-Reranker's asks a fixed
    // instruction; a template of your own carries another.
    std::string prompt_template;
    // Empty: score with the model's ranking head. Otherwise score with the LM
    // head's logits of these tokens (each label's text must be exactly one
    // token), the first being the positive one: for a reranker converted as a
    // plain causal model, without a ranking head (Qwen3-Reranker: {"yes",
    // "no"}, with its prompt in prompt_template).
    std::vector<std::string> labels;
};

// What an embed() or rerank() call processed.
struct PoolingStats {
    int32_t tokens = 0;         // tokens of every input as encoded, special tokens included
    int32_t reused_tokens = 0;  // of them, shared with another input and not processed again
    double ms = 0.0;            // wall time of the call
};

class LIYAB_API Engine {
public:
    static Result<std::unique_ptr<Engine>> create(const EngineConfig& config);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Generates a continuation of `prompt` (already chat-formatted by the
    // caller). The context is reused across calls: when `prompt` continues the
    // text the engine already processed (previous prompt + its generated
    // reply, or a prefill()), that text keeps the tokens it was processed as
    // (re-encoding a reply need not reproduce the tokens that were generated)
    // and only the rest is processed, which makes chat turns start fast.
    // A prompt that leaves that text partway keeps its tokens up to the last
    // one whose text it still starts with. Otherwise the longest common token
    // prefix is kept: any length on
    // attention-only models; on hybrid (DeltaNet) models, whose recurrent
    // states cannot rewind, up to the nearest state snapshot, taken at the end
    // of every prompt and prefill (Transformer::snapshot_state), so an edited
    // last reply or a dropped turn recomputes only what follows that point.
    // Output is the same as from a fresh context, except that a reused reply
    // keeps the tokenization it was generated with. On the CPU backend it is
    // also the same bits however the prompt was split into passes, however
    // many threads ran and whatever the expert cache held or I/O timing did,
    // so greedy decoding is reproducible (moe_skip_slow aside). Not
    // reentrant: a concurrent call returns ErrorCode::Busy.
    Result<GenerationStats> generate(std::string_view prompt, const SamplingParams& params,
                                     const TokenCallback& on_token);
    // Same, with forced continuations (see ForceCallback), with or without
    // speculative decoding (draft model or lookup).
    Result<GenerationStats> generate(std::string_view prompt, const SamplingParams& params,
                                     const TokenCallback& on_token, const ForceCallback& force);
    // Same, with constrained decoding (see GenerateOptions).
    Result<GenerationStats> generate(std::string_view prompt, const SamplingParams& params,
                                     const TokenCallback& on_token, const GenerateOptions& options);
    // Same, from token ids (for vocabularies whose text encoder Liyab lacks);
    // reuse follows token prefixes only.
    Result<GenerationStats> generate_tokens(std::span<const int32_t> prompt, const SamplingParams& params,
                                            const TokenCallback& on_token);

    // Processes `text` into the context without generating, so a later
    // generate() whose prompt starts with it skips that work (e.g. the system
    // prompt, prepared right after loading). Follows the same reuse rules as
    // generate(). Not reentrant (Busy); cancel() stops it between chunks.
    Status prefill(std::string_view text, bool add_bos = true);
    // Forgets the context (the next call starts from scratch, or from what
    // the prefix cache keeps of its prompt).
    void reset_context();
    // Saves the context (e.g. a prefilled system prompt) to `path`: KV pages,
    // recurrent states (with the oldest and the pinned state snapshots on
    // hybrid models), and the tokens and text they encode. Written to a
    // temporary file then renamed, so `path` is either the old file or the new
    // one. Unsupported with a draft model or a sliding window. Busy while a
    // generation runs.
    Status save_state(const std::string& path);
    // Saves the context that `text` continues (a conversation's text so far):
    // the live one when it does, else the longest one the prefix cache set
    // aside for it, written from its store as kept (nothing is restored or
    // processed), so a conversation another prompt replaced (a request
    // through an app's API) can still be saved. InvalidArgument when no
    // context continues `text`; Unsupported when a hybrid model's set-aside
    // context lost its recurrent states (they did not fit the cache).
    Status save_state(const std::string& path, std::string_view text);
    // Replaces the context with one save_state() wrote for this model file
    // loaded with the same cache settings, so the next prompt that continues
    // its text starts at once instead of being processed again (on a 35B MoE a
    // 900-token system prompt takes about a minute); a prompt that diverges
    // from it keeps what the saved context would have kept (on hybrid models,
    // up to its restored snapshots). Returns the tokens restored. InvalidArgument for another model or settings, IoError for a
    // damaged file; the context is empty after a failure.
    Result<int32_t> load_state(const std::string& path);

    // Gives memory back to the OS while the engine is idle (e.g. the app is in
    // the background): empties the MoE expert cache, and drops constrained
    // decoding's caches (vocabulary trie, grammar states and masks; rebuilt
    // on the next constrained generation). The model stays loaded,
    // context included; the next generation reads its experts from storage
    // again, so its first tokens are slower until the cache refills. Returns
    // the expert cache's bytes released (0 for models without streamed
    // experts). Call it between generations, from the thread that generates.
    size_t trim_memory();

    // How a MoE model with streamed experts uses memory; all zero otherwise.
    struct MemoryPlan {
        uint64_t resident_bytes = 0;      // weights kept in RAM (after any requantization)
        uint64_t expert_bytes = 0;        // routed experts read from storage
        uint64_t expert_cache_bytes = 0;  // RAM caching experts
        uint64_t recommended_bytes = 0;   // memory budget at which most expert uses would come from RAM
        int32_t requant_bits = 0;         // resident Q8_0 matrices converted at load (4 / 5); 0: none
    };
    [[nodiscard]] MemoryPlan memory_plan() const;
    // The context length in positions (EngineConfig::context_length, resolved when 0).
    [[nodiscard]] int32_t context_length() const noexcept;

    // The expert cache's hot list (MoE with streamed experts; no-ops
    // otherwise): the cached experts by use, recorded by trim_memory().
    // warm_memory() queues them behind any demand read, so the cache refills
    // in one background pass of large reads once the model is used again
    // instead of one wait per expert; returns how many were queued.
    // save_expert_profile() writes the list (a few KB) for a later process,
    // load_expert_profile() takes it back (InvalidArgument for another model).
    size_t warm_memory();
    Status save_expert_profile(const std::string& path) const;
    Status load_expert_profile(const std::string& path);

    // Thread-safe: stops an in-flight generate() at the next token boundary.
    void cancel() noexcept;
    // Thread-safe: stops an in-flight prefill() between two blocks of the
    // model (within a fraction of a second, also inside a long pass); it
    // returns Cancelled and keeps the context processed so far, so the next
    // prompt that continues the text resumes there. Never affects generate(),
    // so a scheduler can preempt background prefills without a race with the
    // foreground request that follows.
    void cancel_prefill() noexcept;

    // `add_bos` prepends BOS only for models that use one (tokenizer.ggml.add_bos_token),
    // and not to a text that already starts with the BOS token's text (a
    // chat template wrote it).
    [[nodiscard]] Result<std::vector<int32_t>> tokenize(std::string_view text, bool add_bos) const;
    [[nodiscard]] std::string token_to_piece(int32_t token) const;

    [[nodiscard]] const DeviceInfo& device() const noexcept;
    [[nodiscard]] PowerManager& power() noexcept;
    // Model, backends and memory summary for logs / UIs.
    [[nodiscard]] std::string describe() const;

    // Live counters for monitoring UIs; cumulative since creation, so callers
    // sample them periodically and divide the deltas by the elapsed time.
    // Thread-safe: may be read while a generation runs on another thread.
    struct Counters {
        double accelerator_busy_ms = 0.0;  // time the GPU spent on this engine's work (0: none / not tracked)
        uint64_t storage_bytes_read = 0;   // weights streamed from storage (expert cache + block streaming)
        uint64_t tokens_generated = 0;     // decoded tokens, all generations
    };
    [[nodiscard]] Counters counters() const;
    // A scalar GGUF metadata value of the loaded model as text (strings as-is,
    // numbers in decimal, booleans as "true"/"false"), e.g. the publisher's
    // recommended "general.sampling.temp". nullopt when absent or an array.
    [[nodiscard]] std::optional<std::string> model_metadata(std::string_view key) const;

    // ---- Chat template (the model's tokenizer.chat_template) ----------------
    // Rendered by Liyab's own Jinja interpreter (core/chat_template.h), as
    // transformers' apply_chat_template renders it. False when the model has
    // no template or Liyab cannot parse it (the load log says why).
    [[nodiscard]] bool has_chat_template() const noexcept;
    // Renders a conversation. `request_json` is a JSON object: `messages`
    // (role / content objects, also tool_calls and tool turns where the
    // template supports them), optional `tools` (a list of tool schemas),
    // optional `add_generation_prompt` (default false), and any other key as
    // a template variable (`enable_thinking`, `preserve_thinking`...). The
    // special-token variables (bos_token, eos_token...) come from the model.
    // The text starts with the BOS token's text when the template writes it
    // (Llama 3, Gemma); tokenize() and generate() then add no second BOS.
    // InvalidArgument for malformed JSON, a raise_exception() of the template
    // or an evaluation error, with the message; Unsupported without a
    // template. Thread-safe; does not touch the context.
    [[nodiscard]] Result<std::string> render_chat(std::string_view request_json) const;
    // The template's layout (see ChatFormat). `probe_json`: a JSON object of
    // template variables to render with (as in render_chat; "" for none),
    // plus an optional `reply_prefix`, the text a reply starts with after the
    // generation prompt in that mode (e.g. reasoning closed by "</think>"
    // when thinking is on). A template that is not prefix stable is not an
    // error: the result says so. Unsupported without a template. Thread-safe.
    [[nodiscard]] Result<ChatFormat> chat_format(std::string_view probe_json) const;

    // ---- Pooling models (see Pooling) ---------------------------------------
    // What the model declares (<arch>.pooling_type): None for a generative
    // model, Last (Qwen3-Embedding) or Mean for an embedding model, Rank for a
    // reranker. Thread-safe.
    [[nodiscard]] Pooling pooling() const noexcept;
    // Values per vector embed() returns without EmbedOptions::dimensions: the
    // model's hidden size (1024 for Qwen3-Embedding-0.6B). Thread-safe.
    [[nodiscard]] int32_t embedding_size() const noexcept;
    // One vector per text, in order. Each text is encoded as generate()
    // encodes a prompt (special tokens written in it map to their ids, BOS
    // when the model uses one) followed by EOS when the GGUF asks for it
    // (see Pooling), so an instruction is part of the text: Qwen3-Embedding
    // queries are "Instruct: {task}\nQuery:{query}", its documents plain.
    // Contract:
    //  * every input holds at least one token and at most context_length()
    //    positions (else InvalidArgument naming it, unless options.truncate);
    //  * vectors have embedding_size() values (options.dimensions when set),
    //    unit length unless options.normalize is false;
    //  * the texts run one after another in the context, sorted so that those
    //    sharing a token prefix (an instruction) are neighbours: each keeps
    //    the KV of the prefix it shares with the one before and processes
    //    only the rest (Last pooling, attention-only models). On the CPU
    //    backend a vector is the same bits whatever the batch around it;
    //  * the context is used and left empty: a conversation in progress is
    //    set aside in the prefix cache when one is configured (prefix_cache_mb
    //    or prefix_cache_disk_mb), as when another prompt replaces it, and is
    //    processed again otherwise;
    //  * an engine serves one call at a time: Busy while a generation, a
    //    prefill or another embed() / rerank() runs; cancel() stops it between
    //    passes and inside them (Cancelled);
    //  * Unsupported when the pooling cannot apply (see EmbedOptions::pooling).
    // `stats` (optional) receives the tokens processed and reused and the time.
    Result<std::vector<std::vector<float>>> embed(std::span<const std::string_view> texts,
                                                  const EmbedOptions& options = {}, PoolingStats* stats = nullptr);
    // Relevance of each document to `query`, in order: the score of the pair's
    // prompt (RerankOptions::prompt_template or the model's rerank template,
    // "{query}" and "{document}" replaced, encoded like an embed() input) from
    // the last token's state: with the model's ranking head, the softmax
    // probability of its first label ("yes" for Qwen3-Reranker: P(yes) over
    // {yes, no}, as its model card normalizes), or the sigmoid of a head's
    // single output; with RerankOptions::labels, the softmax probability of
    // the first label's token among the labels' LM-head logits. Scores are in
    // (0, 1), higher is more relevant; they are not sorted. Errors: no prompt
    // template (InvalidArgument) or no ranking head and no labels
    // (Unsupported); a pair longer than context_length() (InvalidArgument
    // naming it; no truncation). Otherwise the same contract as embed():
    // pairs share the KV of their common prefix (the instruction and the
    // query), the context is left empty, one call at a time, cancellable.
    Result<std::vector<float>> rerank(std::string_view query, std::span<const std::string_view> documents,
                                      const RerankOptions& options = {}, PoolingStats* stats = nullptr);

    // ---- Prefix cache (EngineConfig::prefix_cache_mb, prefix_cache_dir) ------
    // What its store holds and how it was used since the engine was created.
    // All zero without a prefix cache. Thread-safe: may be read while a
    // generation runs (the figures are those after the last use).
    struct PrefixCacheStats {
        uint64_t ram_bytes = 0;          // KV chunks and state images in RAM
        uint64_t ram_cap_bytes = 0;      // prefix_cache_mb as granted (0: no RAM tier)
        uint64_t storage_bytes = 0;      // the store's files in prefix_cache_dir, every model's
        uint64_t storage_cap_bytes = 0;  // prefix_cache_disk_mb (0: no storage tier)
        int32_t chunks = 0;              // this model's, in RAM and on storage
        int32_t state_images = 0;        // recurrent-state images (hybrid models)
        int32_t contexts = 0;            // stored contexts (texts a prompt is matched against)
        uint64_t set_asides = 0;         // contexts set aside
        uint64_t chunks_written = 0;     // chunks they added
        uint64_t chunks_reused = 0;      // chunks they found stored already (shared prefixes)
        double set_aside_ms = 0.0;       // total time setting aside
        uint64_t restores = 0;           // prompts that restored a prefix
        uint64_t restored_tokens = 0;    // positions restored
        double restore_ms = 0.0;         // total time restoring
    };
    [[nodiscard]] PrefixCacheStats prefix_cache_stats() const;

private:
    struct Impl;
    explicit Engine(std::unique_ptr<Impl> impl);
    // generate_tokens() with the busy lock held; `text`, when given, is the
    // text `prompt` encodes, remembered for the next call's reuse.
    Result<GenerationStats> generate_locked(std::span<const int32_t> prompt, const SamplingParams& params,
                                            const TokenCallback& on_token, const std::string_view* text,
                                            const ForceCallback* force = nullptr,
                                            const GenerateOptions* options = nullptr);
    std::unique_ptr<Impl> impl_;
};

}  // namespace liyab

#endif  // LIYAB_ENGINE_H
