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
    // CPU matmuls of Q4_K / Q5_K / Q6_K weights over two or more activation rows (prefill, speculative
    // verification) run as 2x2 SMMLA tiles (needs i8mm). Exact: the same bits as the dot-product kernels. Measured
    // slower on the Snapdragon 8 Elite for Q5_K and Q6_K, so it stays opt-in; ignored on CPUs without i8mm.
    // The tiles read the weights as stored, so this also keeps the i8mm repacked layouts (the default on such
    // CPUs) off, whatever it costs the matmuls the tiles do not cover.
    bool i8mm_tiles = false;
    // Measurement only: logs how well the streamed-expert predictor (and two cheaper-to-learn corrections of it)
    // guess the experts the router picks, every 64 decode tokens, at INFO level. The output is unchanged. Needs a
    // mixture-of-experts model (else Unsupported); it prints nothing unless the experts stream from storage.
    bool prerouter_trace = false;
    // Streamed MoE experts as a two-tier cache: the RAM cache, then the kernel page cache. 0 off (the default);
    // 1 every expert read is buffered, so what the RAM cache evicts stays in the page cache and comes back at
    // memory speed (and outside the process's PSS); 2 buffered prefetches, direct reads for experts the router
    // already chose. The resident weights then move out of the file mapping into anonymous memory, which expert
    // pages in the page cache would otherwise evict. Exact output. Measured slower on the Snapdragon 8 Elite
    // (buffered misses are slower than direct ones and hold the I/O threads urgent reads wait for). Needs
    // streamed experts (else it only warns); Unsupported on Windows. GenerationStats::expert_page_cache_loads
    // counts the experts it served.
    int32_t page_cache_mode = 0;
    // (KV dedup, the experimental persistent prefix KV cache, is retired: the
    // prefix cache's store keeps a shared prefix once, on storage across
    // processes, for every model, within its caps; see prefix_cache_dir.)
};

// How a model with multi-head latent attention (DeepSeek-V2/V3 and models
// converted to their architecture) attends; see EngineConfig::latent_form.
enum class LatentAttentionForm : int32_t {
    Auto = 0,      // per pass, by its batch and context against a threshold measured once per device; decoding absorbed
    Absorbed = 1,  // always over the cached latents with absorbed queries
    Expanded = 2,  // every pass of two tokens or more over per-head keys and values expanded from the latents
};

struct EngineConfig {
    std::string model_path;           // GGUF file (F32/F16/Q8_0/Q4_0/Q4_1 tensors)
    // optional: enables speculative decoding. The draft's token table (tokenizer.ggml.tokens) must agree with the
    // target's over the ids both have; either may be padded further (Qwen2.5-Coder 0.5B: 151,936 ids, 14B: 152,064).
    std::string draft_model_path;
    int32_t n_threads = 0;            // 0: performance cores, minus one when there is no efficiency cluster
    // 0: automatic: what 256 MiB of KV cache holds (4096..32768), at most the
    // training context; windowed attention blocks count for their window only.
    int32_t context_length = 0;
    int32_t sliding_window = 0;       // 0: full attention; else attend to the last N tokens
    int32_t kv_sink_tokens = 8;       // with a sliding window: first N tokens stay visible (attention sinks)
    // Paged KV cache format. Q8_0 (the default) halves F16's memory. For
    // models whose keys carry channels far larger than the rest (Qwen2.5,
    // Qwen3), keys are stored with per-channel scales measured once per model
    // (see cache_dir), which takes their mean KL against F16 from 0.05-0.07 to
    // ~0.002; other models keep their keys as computed (README, KV cache). F16
    // needs no scales. Q4_0 / Q4_1 (INT4, -72% / -69% vs F16) are scaled the
    // same way and still trade accuracy for memory on long contexts. The
    // scales are measured on a fixed text: a model whose tokenizer cannot
    // encode text (an unknown pre-tokenizer; it runs on token ids) loads with
    // its keys as computed instead, with a warning in the log and a "Keys:"
    // line in describe().
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
    // Lossy, opt-in: convert resident matrices to Q4_K (4) or Q5_K (5) at
    // load, so each token reads fewer bytes. Converted on every core at load.
    //  * With expert streaming (MoE): the resident Q8_0 matrices (attention /
    //    recurrent projections, shared experts); the memory saved goes to the
    //    expert cache. Ignored when experts are not streamed.
    //  * Models without experts: the Q8_0 / Q6_K matrices (and, for 4, Q5_K)
    //    of the feed-forward blocks (gate, up, down) and of the attention
    //    query and output projections, the roles that cost least quality per
    //    byte saved; keys, values, the output head and the embeddings keep
    //    their bytes. Qwen2.5-Coder-14B Q5_K_M with 4 decodes 27% faster on
    //    a Mac's CPU (19.0 -> 24.1 tok/s) at a mean KL of 0.007-0.009
    //    against the file's own weights (perplexity +1.3%), and loads in 14 s
    //    instead of 2.3 s (the conversion runs at every load); with 5 only
    //    the Q6_K / Q8_0 matrices change, which is little on a K-quant file.
    //    Ignored by models streamed from storage in blocks.
    // 0: off. -1: automatic, MoE expert streaming only, when the expert cache
    // would otherwise hold under 10% of the experts (Q5_K if that frees enough
    // memory, else Q4_K); see Engine::memory_plan().
    int32_t requant_bits = 0;
    // MiB of cache_dir the engine may fill with the weights it converts at load (requant_bits on a model without
    // experts, and the CPU's repack into the i8mm layouts): the first load writes them as a file of about the
    // model's size, later loads of the same model map it instead of converting again (a 14B lowered with
    // requant_bits 4: 14 s of conversion becomes 0.4 s; repacked only: 2.3 s becomes under 0.5 s). The directory
    // keeps at most this many MiB of them, least recently used out first (enforced when a file is written); a file
    // larger than the cap, or a volume left under 1 GiB free, is not written. 0 (default): off. Needs cache_dir; models with experts are not cached.
    int64_t converted_weights_mb = 0;
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
    // content-addressed store of KV chunks (whole KV pages, about 2 MiB, at
    // most 256 tokens), each keyed by every token before it plus the model,
    // cache layout, key scales, kernels and the LoRA adapter selected (see
    // AdapterInfo), so a prefix conversations share
    // (system prompt, tools) is stored once and any prompt starting with it
    // restores it; recurrent mixer and compression states are kept as images at the
    // positions the context's state snapshots hold, and a hybrid model
    // restores up to the deepest one its prompt reaches. Least recently used chunks out first. Taken out
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
    // Chunks are written compressed losslessly only where that pays on the
    // device (the read time saved exceeds the decode time, both measured once
    // and kept in cache_dir; on fast flash or SSD nothing pays and chunks stay
    // raw), and decoded on the engine's threads when read back; the cap
    // counts the bytes written. Earlier builds' raw files stay readable.
    std::string prefix_cache_dir;
    int64_t prefix_cache_disk_mb = 0;
    // Chunks on storage are compressed losslessly where that pays (README,
    // Prefix cache). true re-quantizes them to 4 bits instead (keys per channel over the
    // chunk's positions, values in 32-channel blocks): 3.4x smaller than an
    // F16 cache's rows, 1.8x than Q8_0, at a measured cost (README); lossy,
    // so off by default. A lossy engine keys its chunks apart: they are never
    // restored by an engine without it, nor its lossless ones by it.
    bool prefix_cache_lossy = false;
    // What the engine measures once per model and keeps, in this directory
    // (created if missing): today the key scales of a quantized KV cache
    // (README, KV cache), a few KB per model, at most 16 models, least
    // recently used out first, and what reading and decoding prefix-cache
    // chunks cost on this device (kv-costs-*.bin, 24 bytes each, at most 16), the latent-attention form ratio
    // (latent-*.bin), what a verification pass costs (verify-*.bin), the read speeds the model-fit report timed
    // (host-*.bin: liyab_describe_model_fit, `liyab-cli --doctor`; one per host and thread count, at most 4) and, with automatic routing, which route each
    // class of passes took (route-*.bin, 64 bytes; a later run adopts it without the probe passes; each of these
    // kinds is 16 files at most). With converted_weights_mb, also the
    // weights converted at load (weights-*.bin, each about a model's size: capped in bytes, not in files).
    // Empty: measured again at every load (on the
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
    // about the cost of one (dense models in RAM). Auto (the default) decides
    // per model and device: the engine times a plain step and a verification
    // pass of draft_tokens + 1 tokens once (about 0.3 s on a 14B; kept in
    // cache_dir under the model's identity, in memory for the process
    // without one), and drafts only where the pass costs at most
    // kLookupMaxVerifyRatio times the step: dense models in RAM, not MoE
    // models whose experts stream from storage, whose pass reads the union
    // of its tokens' experts. Which models that is follows from the
    // measurement, not from a list. Engine::lookup_drafts() reports the
    // outcome. On / Off skip the measurement.
    enum class LookupDrafts : int32_t { Off = 0, On = 1, Auto = 2 };
    LookupDrafts lookup_drafts = LookupDrafts::Auto;
    // Auto drafts only where a pass of draft_tokens + 1 tokens costs at most this many plain steps. With the 3
    // drafts of the default, 2.5 steps still pays when about 2 of 3 drafts are accepted (code edits: 3.4 tokens per
    // pass); the learned draft count (adaptive_drafts) then backs off on text that repeats nothing.
    static constexpr double kLookupMaxVerifyRatio = 2.5;
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
    // (SpeculativeDecoder::lookup). No effect with lookup_drafts Off or with a
    // draft model. Either way draft_tokens is the most drafts per step, and
    // hybrid models keep draft_tokens + 1 recurrent-state checkpoints.
    bool suffix_drafts = true;
    // Force one backend (always used, with the CPU for what it does not run). Unset: the device's ranking, and
    // with an accelerator present each class of pass (decoding and drafts' verification; prompts) runs on the
    // accelerated route or on the CPU alone, whichever timing of the first passes shows faster by a margin
    // (core/route_picker.h; describe() reports it). The two give results within float rounding of each other.
    std::optional<BackendKind> backend;
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
    // Latent attention only (ignored by other models): the form prefill
    // passes attend in. Both forms compute the same function over the same
    // cache of latents and round differently. Auto times the two forms once
    // per model and device at load (0.18 s on a 2B model), keeps the
    // result in cache_dir (in memory for the process without one; measured
    // again when the file is missing or damaged) and then picks each pass's
    // form from its batch size and context alone: the same prompt gives the
    // same bits on that device in every run, whatever the load; another
    // device may pick otherwise and round differently. Absorbed or Expanded
    // give the same bits on every device of a kernel set. Decoding steps are
    // always absorbed.
    LatentAttentionForm latent_form = LatentAttentionForm::Auto;
    // Segment reuse: KV reused away from its prefix. Lossy and off by
    // default: see PromptSegment for what it does and what it changes. On,
    // the engine measures once per model and cache settings the share of a
    // reused segment's tokens it processes again to keep the output within
    // segment_reuse_max_kl of processing the whole prompt, and keeps the
    // result in cache_dir (measured again at every load without one). The
    // measurement does not hold up loading: it runs in the background once
    // no call has used the engine for 2 s, a stage at a time on a context of
    // its own (the conversation's is left as it was), and a call arriving
    // meanwhile stops it between two blocks, so a request waits a fraction
    // of a second at most; the stage runs again at the next idle time. In
    // all about as much work as processing 15,000 tokens (minutes for a 1B
    // model on a Mac's CPU). Until it ends, segment reuse is off: marked
    // prompts are processed as usual (Engine::segment_reuse() says how far it
    // got). Needs the prefix cache's store (prefix_cache_mb or
    // prefix_cache_disk_mb), whose caps the stored segments share; refused,
    // with the reason in Engine::segment_reuse(), for hybrid models, with a
    // sliding window or a draft model, or when no share on the calibration's
    // grid meets the target. A segment is reused with every block's rows, so
    // on models with windowed attention blocks (Gemma 2 / 3, Phi-3) it makes
    // those blocks keep every position, as the others, instead of only their
    // window: full attention's KV memory, a shorter automatic context.
    bool segment_reuse = false;
    // The target: the mean KL divergence of the next-token distributions,
    // KL(whole prompt processed || segments reused), over the answers to
    // RAG-shaped prompts in the model's chat template (core/kv_reuse.h), with
    // a margin for what those prompts cannot tell apart.
    float segment_reuse_max_kl = 0.03f;
    // LoRA adapters (Engine::load_adapter): RAM at most for the loaded
    // adapters' tensors, in MiB. Several can be loaded at once; past the cap
    // the least recently selected ones are released and mapped again from
    // their files when selected. 0: adapters off (load_adapter() is
    // Unsupported).
    int64_t adapter_memory_mb = 256;
    // Suffix drafts (lookup_drafts with suffix_drafts): verify the likeliest
    // nodes of the drafter's continuation tree, alternative branches
    // included, in one pass (true, the default), or only its best branch.
    // Either way the step verifies at most draft_tokens drafted tokens, chosen
    // by the same cost model, and keeps the longest accepted branch; the
    // output is the same tokens (plain decoding's). No effect on other drafts.
    bool draft_tree = true;
    // Sequences the engine keeps at once, its default one included (see
    // Engine::open_sequence): 1 (the default) keeps one conversation, as
    // before, and open_sequence() is Unsupported. A ceiling only: nothing is
    // reserved at load, so while no sequence besides the default one is
    // open the engine is exactly the engine of max_sequences 1 (the same
    // memory plan, expert cache, requantization decision and output). Each
    // sequence opened keeps a context of its own while it is open: KV pages
    // as it uses them (up to context_length() positions) and, on hybrid
    // models, its recurrent states, rollback checkpoints and up to 4 state
    // snapshots (62.8 MiB each on Qwen3.6-35B-A3B), which open_sequence()
    // takes out of the expert cache (see there). Unsupported with a draft
    // model (its cache follows one sequence).
    int32_t max_sequences = 1;
};

// How the engine schedules a sequence's passes against the others'
// (Engine::Sequence::set_priority). The user's turn is Foreground; work the
// user does not wait for (an agent running in the background, prefetching)
// is Background, which yields: its passes run when no foreground one is
// waiting, its decoding steps join a foreground pass only where that costs
// the foreground step little (passes of one more sequence measured at most
// 1.25 times as long; one step in 64 that would be kept out joins anyway,
// to re-measure), and a background prompt being processed is stopped
// between two blocks (its work so far kept) when a foreground sequence
// needs the model.
enum class SequencePriority : int32_t {
    Foreground = 0,
    Background = 1,
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

// ---- Segment reuse: KV reused away from its prefix (lossy, opt-in) ---------
// An agent's prompts put documents, memory snippets and tool results in the
// middle, so a change before them makes everything after them be processed
// again. A caller marks such parts of a prompt (PromptSegment: a system
// block, tools, a document, a tool result, a conversation turn), and with
// EngineConfig::segment_reuse on, the engine keeps the KV of each marked
// part and reuses it wherever the same part appears again:
//  * Exact reuse first. The part of the prompt the live context or the
//    prefix cache holds is kept or restored as before, losslessly; segments
//    are reused only after it.
//  * A segment's KV is kept in the prefix cache's store (within its caps,
//    least recently used out first, on storage across runs when it has a
//    storage tier) once a prompt marking it has been processed, keyed by its
//    tokens and the position it was computed at.
//  * Reused elsewhere, its keys are turned from the positions they were
//    computed at to the new ones (RoPE: exact up to rounding, every variant
//    the engine runs), then part of its tokens is processed again with
//    attention to everything now before them, the ones whose keys and
//    values moved the most when the first blocks were run again; the rest
//    keep their reused rows. The share is calibrated per model
//    (segment_reuse_max_kl).
//  * It changes the output: the reused rows lack attention to what now
//    precedes them, which recomputing only part of them does not fully
//    restore. A request opts in by marking segments; one that does not never
//    sees reused rows: a context holding them is processed again from the
//    first one (it is set aside in the prefix cache only up to there, and
//    save_state() records where they start).
//  * Text a segment holds is encoded on its own (at its boundaries the
//    prompt is split even where encoding it whole would merge two tokens),
//    so its tokens are the same wherever it appears. Segments shorter than
//    16 tokens are processed as usual.
//  * Not on hybrid models (their recurrent blocks must run over the whole
//    prompt in order), with a sliding window or a draft model (no store):
//    there, marked prompts are processed as usual, losslessly, and
//    Engine::segment_reuse() says why.
// GenerationStats and Engine::PrefixCacheStats count what was reused.
//
// A marked part of a prompt, [begin, end): byte offsets into a prompt's
// text, on whole UTF-8 characters (token indices into a prompt given as
// tokens). The segments of one prompt ascend and do not overlap.
struct PromptSegment {
    size_t begin = 0;
    size_t end = 0;
};

// ---- LoRA adapters -----------------------------------------------------------
// A LoRA adapter (a GGUF the common converters write from a PEFT adapter:
// lora_a / lora_b per adapted matrix, adapter.lora.alpha) changes some of the
// model's matrices by a low-rank update W + s·B·A. Liyab never merges it: each
// adapted matmul adds s·B·(A·x) to the base product, computed by the
// engine's own CPU kernels (A·x batched with the base matmuls, the update over
// a transposed B; see README "LoRA adapters"), so several adapters
// stay loaded next to one model and the one in use is chosen per generation
// (Engine::set_adapter), and a pass without one costs what it did.
//  * s = scale · alpha / rank (scale from set_adapter, 1 by default; each
//    matrix's own rank; alpha 0 or absent: s = scale), as PEFT computes it.
//  * Dense matrices of every block (attention, recurrent mixers, FFN, the shared
//    expert, the router, latent attention's whole projections, compression,
//    indexer and residual-stream projections) and the LM
//    head; a mixture of experts' per-expert stacks (each expert its slice).
//    Refused: adapters of the token embeddings, of latent attention's
//    per-head attn_k_b / attn_v_b or combined attn_kv_b, shared-KV's grouped
//    attn_output_a, activated LoRA, other adapter types.
//  * Contexts are kept apart by adapter (and scale): one computed with
//    adapter X is never continued with Y or without it. Selecting another
//    adapter sets the live context aside in the prefix cache under X (when
//    there is one) and starts from what the cache keeps under Y; the cache's
//    chunks are keyed by the adapter too (its file's hash), on storage
//    across runs as well; save_state() records the adapter, and
//    load_state() refuses a context saved under another one.
//  * Speculative decoding verifies with the adapter (the target model's
//    passes apply it); a draft model drafts without it, which may lower the
//    acceptance rate but not change the output's distribution.
//  * Segment reuse is off while an adapter is selected (its calibration is
//    the base model's): marked prompts are processed as usual, exactly.
//  * Embeddings and reranking apply the selected adapter as well.
struct AdapterInfo {
    std::string name;
    std::string path;
    float alpha = 0.0f;        // adapter.lora.alpha (0: none; s = scale)
    int32_t rank = 0;          // the largest rank of its matrices
    int32_t matrices = 0;      // adapted matrices (a per-expert stack counts once)
    uint64_t bytes = 0;        // its tensors, counted against adapter_memory_mb
    bool resident = false;     // mapped now (else mapped again when selected)
    bool active = false;       // the one the default sequence's passes apply (Sequence::set_adapter for others)
    float scale = 0.0f;        // set_adapter()'s scale, when active
};

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
    // Segment reuse (see PromptSegment): the prompt's marked parts, which
    // makes this request opt in to it. Independent of the two above. Ignored
    // (the prompt processed as usual) while segment reuse is unavailable.
    std::vector<PromptSegment> segments;
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
    // attention-only models; on models with recurrent mixers or compression, whose
    // states cannot rewind, up to the nearest state snapshot, taken at the end
    // of every prompt and prefill (Transformer::snapshot_state), so an edited
    // last reply or a dropped turn recomputes only what follows that point.
    // Output is the same as from a fresh context, except that a reused reply
    // keeps the tokenization it was generated with. On the CPU backend it is
    // also the same bits however the prompt was split into passes, however
    // many threads ran and whatever the expert cache held or I/O timing did,
    // so greedy decoding is reproducible (moe_skip_slow aside). Runs on the
    // engine's default sequence: a concurrent call on it returns
    // ErrorCode::Busy; calls on other sequences run alongside (open_sequence).
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
    // hybrid models), the rows windowed attention blocks keep (the window
    // before the end and before those snapshots), and the tokens and text
    // they encode. Files written before the per-block KV layout load
    // converted. Written to a
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
    // Planned at load for one conversation, whatever max_sequences allows.
    // Thread-safe.
    struct MemoryPlan {
        uint64_t resident_bytes = 0;      // weights kept in RAM (after any requantization)
        uint64_t expert_bytes = 0;        // routed experts read from storage
        uint64_t expert_cache_bytes = 0;  // RAM caching experts now: the load's plan minus what open sequences borrowed
        uint64_t recommended_bytes = 0;   // memory budget at which most expert uses would come from RAM
        int32_t requant_bits = 0;         // matrices converted to Q4_K / Q5_K at load (4 / 5); 0: none
    };
    [[nodiscard]] MemoryPlan memory_plan() const;
    // Actual dense-weight placement after load. File bytes include GGUF
    // metadata and non-block tensors; streamed_bytes counts block ranges read
    // through the 3-slot pipeline. mmap_streaming is a paging hint and can be
    // true even when no block pipeline was attached (e.g. split GGUF).
    // Immutable after load and safe to read while generation runs.
    struct WeightLayout {
        uint64_t file_bytes = 0;
        uint64_t streamed_bytes = 0;
        int32_t streamed_blocks = 0;
        int32_t total_blocks = 0;
        bool mmap_streaming = false;
    };
    [[nodiscard]] WeightLayout weight_layout() const noexcept;
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

    // Thread-safe: stops an in-flight generate() (or embed() / rerank()) on
    // the default sequence at the next token boundary; never another
    // sequence's (Sequence::cancel()).
    void cancel() noexcept;
    // Thread-safe: stops an in-flight prefill() between two blocks of the
    // model (within a fraction of a second, also inside a long pass); it
    // returns Cancelled and keeps the context processed so far, so the next
    // prompt that continues the text resumes there. Never affects generate(),
    // so a scheduler can preempt background prefills without a race with the
    // foreground request that follows.
    void cancel_prefill() noexcept;

    // Tokenizes UTF-8 into owned ids; thread-safe, without changing context.
    // `add_bos` requests the model's initial special tokens: normally BOS
    // (tokenizer.ggml.add_bos_token), or [gMASK], sop for chatglm-spm. A
    // prefix already spelled by the text is not added again. Unsupported
    // for an unknown tokenizer scheme; InvalidArgument for unencodable bytes.
    [[nodiscard]] Result<std::vector<int32_t>> tokenize(std::string_view text, bool add_bos) const;
    [[nodiscard]] std::string token_to_piece(int32_t token) const;

    [[nodiscard]] const DeviceInfo& device() const noexcept;
    [[nodiscard]] PowerManager& power() noexcept;
    // Model, backends and memory summary for logs / UIs. A "Keys:" line
    // appears only when a quantized KV cache stores keys without the scales
    // the model may need (see kv_cache_type), and says why.
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

    // ---- Drafts from the text (EngineConfig::lookup_drafts) -------------------
    struct LookupDraftsInfo {
        bool active = false;          // the engine drafts from the text
        double verify_ratio = 0.0;    // Auto: cost of a pass of draft_tokens + 1 tokens in plain steps; 0: not measured
        bool from_cache = false;      // Auto: the measurement was read from cache_dir (or an earlier engine of the process)
    };
    [[nodiscard]] LookupDraftsInfo lookup_drafts() const noexcept;

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
    // model's hidden size (1024 for Qwen3-Embedding-0.6B), or the width of
    // the dense projections the GGUF applies after pooling (dense_2 /
    // dense_3: 768 for EmbeddingGemma). Thread-safe.
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
    //    unit length unless options.normalize is false; the pooled states go
    //    through the model's dense projections first when it has them;
    //  * the texts run one after another in the context, sorted so that those
    //    sharing a token prefix (an instruction) are neighbours: each keeps
    //    the KV of the prefix it shares with the one before and processes
    //    only the rest (Last pooling, attention-only causal models; a
    //    bidirectional model such as EmbeddingGemma reads each text whole in
    //    one pass; token generation on such a model returns Unsupported).
    //    On the CPU backend a vector is the same bits whatever the
    //    batch around it;
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
        int32_t chunk_tokens = 0;        // positions per chunk: whole KV pages, about 2 MiB
        uint64_t set_asides = 0;         // contexts set aside
        uint64_t chunks_written = 0;     // chunks they added
        uint64_t chunks_reused = 0;      // chunks they found stored already (shared prefixes)
        double set_aside_ms = 0.0;       // total time setting aside
        uint64_t restores = 0;           // prompts that restored a prefix
        uint64_t restored_tokens = 0;    // positions restored
        double restore_ms = 0.0;         // total time restoring
        // Segment reuse (see PromptSegment):
        int32_t segments = 0;                  // stored segment copies (their chunks are in `chunks`)
        uint64_t segments_kept = 0;            // segments stored
        uint64_t segment_reuses = 0;           // segments reused
        uint64_t segment_reused_tokens = 0;    // positions reused away from their prefix
        uint64_t segment_recomputed_tokens = 0;  // of them, processed again
    };
    [[nodiscard]] PrefixCacheStats prefix_cache_stats() const;

    // ---- Segment reuse (see PromptSegment) ----------------------------------
    // generate_tokens() with options (a grammar, a continuation, marked
    // segments as token indices into `prompt`); see generate() with
    // GenerateOptions.
    Result<GenerationStats> generate_tokens(std::span<const int32_t> prompt, const SamplingParams& params,
                                            const TokenCallback& on_token, const GenerateOptions& options);
    // prefill() of a text with marked segments (byte offsets): stored
    // segments are reused as generate() reuses them, and the segments are
    // kept once processed. InvalidArgument for segments that are not
    // ascending, overlap, leave the text or cut a character.
    Status prefill(std::string_view text, std::span<const PromptSegment> segments, bool add_bos = true);
    // Whether segment reuse is on, and what it was calibrated to: OK, or why
    // not (off in EngineConfig, the model or settings refuse it, no stored
    // segments without a prefix cache, the target not met), or Busy while
    // the calibration runs in the background (`calibrating`). Thread-safe.
    struct SegmentReuseInfo {
        Status status;
        float fraction = 0.0f;  // share of a reused segment's tokens processed again
        double kl = 0.0;        // the calibration's mean KL at that share
        double top1 = 0.0;      // and its top-1 agreement with processing the whole prompt, in [0, 1]
        bool from_cache = false;  // the calibration was read from cache_dir
        bool calibrating = false;  // the calibration runs in the background: segment reuse is off until it ends
        float progress = 0.0f;     // share of the calibration's work done, in [0, 1]
        double bound = 0.0;        // the calibration's upper bound on the KL at that share (what met the target)
    };
    [[nodiscard]] SegmentReuseInfo segment_reuse() const;

    // ---- LoRA adapters (see AdapterInfo) -------------------------------------
    // Loads the adapter at `path` under `name` (not selected). Checks it
    // against the model: its architecture, every adapted matrix's name and
    // shape. InvalidArgument for an empty or taken name, another
    // architecture or a matrix that does not fit; Unsupported for what
    // AdapterInfo lists as refused, or with adapter_memory_mb 0; OutOfMemory
    // when it alone exceeds adapter_memory_mb; InvalidModel / IoError for a
    // file that is not a LoRA adapter or cannot be read. Busy while another
    // call runs. Other adapters may be released to fit the cap (not the
    // selected one).
    Status load_adapter(std::string_view name, const std::string& path);
    // Forgets an adapter (selected: the engine goes back to the base model,
    // as set_adapter("")). InvalidArgument for an unknown name; Busy.
    Status unload_adapter(std::string_view name);
    // Selects the adapter later calls apply (generate, prefill, embed,
    // rerank): `name` at `scale` (see AdapterInfo), "" for none (the base
    // model, the default). A context computed otherwise is set aside in the
    // prefix cache (when there is one) and the next prompt starts from what
    // the cache keeps for this adapter, or from scratch; selecting the
    // adapter in use at the same scale keeps the context. An adapter
    // released under the cap is mapped again from its file (IoError /
    // InvalidModel when it changed into something that is not an adapter, or
    // InvalidArgument when it no longer fits the model; the engine is then
    // without an adapter). InvalidArgument for an unknown name or a scale
    // that is not finite; Busy while another call runs.
    Status set_adapter(std::string_view name, float scale = 1.0f);
    // The loaded adapters, in load order. Thread-safe.
    [[nodiscard]] std::vector<AdapterInfo> adapters() const;

    // ---- Several sequences decoded together ----------------------------------
    // The engine's own calls (generate, prefill, save_state...) work on its
    // default sequence. open_sequence() opens another one, up to
    // EngineConfig::max_sequences in all: a conversation with a context of
    // its own (KV cache, recurrent states and snapshots, prompt text, LoRA
    // adapter selection, speculative drafts), sharing the weights, the
    // thread pool, the expert cache and the prefix cache. Calls on different
    // sequences run at once, each on its caller's thread (callbacks
    // included), and their forward passes run together: one pass over every
    // waiting sequence's tokens, the matmuls over all of their rows (each
    // weight read once for all of them; a mixture of experts reads the
    // union of the experts they choose), each sequence attending to its own
    // cache. Contract:
    //  * Each sequence's output is what it gives alone: on the CPU backend
    //    the same bits (a row's matmul does not depend on the rows around it,
    //    attention on its own position only), so greedy decoding gives the
    //    same tokens and sampling the same draws from the same seed. With the
    //    experimental lossy hooks (early exit, EGLS, TDSS, head pruning)
    //    sequences take turns instead of sharing passes, so that holds too.
    //  * A call that runs alone (no call on another sequence) runs exactly as
    //    a single conversation always has: the same passes, speculative
    //    drafts (draft model aside: see max_sequences), pacing per token.
    //    While several run, steps are plain (one token, no drafts): the pass
    //    shared with the others is the batch drafts would otherwise fill
    //    (README measures both). Pacing then holds each shared pass to the
    //    token rate, so every sequence in it gets a token per slot.
    //  * A prompt is processed in passes of at most max_batch tokens while
    //    other sequences decode (a pass's time is what their next token
    //    waits for), except with latent attention, whose passes keep their
    //    size so each takes the form it takes alone.
    //  * Priorities (SequencePriority) and per-sequence cancellation; see
    //    Sequence. Busy when another call runs on the same sequence.
    //  * The prefix cache is shared: a context any sequence set aside (or
    //    one the store kept from earlier runs) is restored by whichever
    //    sequence's prompt continues it, under the same adapter.
    //  * GenerationStats counts each call's own tokens and passes; its expert
    //    and phase figures are the engine's over the call's time, other
    //    sequences' passes included.
    //  * embed(), rerank(), save_state(), load_state() and segment reuse
    //    work on the default sequence; while one runs, the others' passes
    //    wait for it.
    //  * Memory: an open sequence's context exists only while it is open.
    //    Its KV pages are mapped as it uses them. On a hybrid model with
    //    streamed experts, open_sequence() first takes its recurrent states,
    //    rollback checkpoints and snapshots (their most) out of the expert
    //    cache, in whole cache slots: the experts there are evicted (waiting
    //    for reads in flight; other sequences keep decoding on the rest of
    //    the cache) and their pages returned to the OS, so the engine's
    //    footprint does not grow; closing the sequence frees its context and
    //    gives those slots back (MemoryPlan::expert_cache_bytes shows the
    //    cache in use). Nothing is ever requantized after load. Without
    //    streamed experts, or on models without recurrent state, the
    //    context is simply allocated on open and freed on close.
    // Unsupported with max_sequences 1 or a draft model; Busy when
    // max_sequences are open; OutOfMemory when the expert cache cannot spare
    // the sequence's states without going below its minimum (a block's
    // top-k plus the reads in flight per slot size class), in which case
    // nothing changes. Thread-safe; it waits for a pass running on other
    // sequences to end.
    class Sequence;
    Result<std::unique_ptr<Sequence>> open_sequence(SequencePriority priority = SequencePriority::Foreground);
    // How sequences shared passes since the engine was created. Thread-safe.
    struct SchedulerStats {
        uint64_t passes = 0;         // passes run for sequences while several were in a call
        uint64_t shared_passes = 0;  // of them, passes of two sequences or more
        uint64_t shared_rows = 0;    // tokens of those shared passes
        uint64_t preempted = 0;      // background passes stopped for foreground work (run again later)
        int32_t open_sequences = 0;  // open now, the default one included
        int32_t active_sequences = 0;  // in a call now
        double pass_ms = 0.0;        // time in those passes
        double gather_ms = 0.0;      // time leaders waited for sequences about to submit (see open_sequence)
    };
    [[nodiscard]] SchedulerStats scheduler_stats() const;

private:
    struct Impl;
    explicit Engine(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// A sequence of an Engine (Engine::open_sequence): the calls the engine has
// for its default sequence, on this one's context. Destroying it sets its
// context aside in the prefix cache (when there is one), frees it, and
// waits for a call running on it, which it cancels first; destroy every
// Sequence before its Engine. Thread-safe as the engine's calls are: one
// call at a time on it (Busy otherwise), any number on other sequences;
// cancel() and cancel_prefill() from any thread.
class LIYAB_API Engine::Sequence {
public:
    ~Sequence();
    Sequence(const Sequence&) = delete;
    Sequence& operator=(const Sequence&) = delete;

    // An id unique among the engine's sequences (the default one is 0).
    [[nodiscard]] int32_t id() const noexcept { return id_; }
    // Engine::generate on this sequence's context. `options` may carry a
    // grammar, a continuation and segments as there.
    Result<GenerationStats> generate(std::string_view prompt, const SamplingParams& params,
                                     const TokenCallback& on_token, const GenerateOptions& options = {});
    // With forced continuations (ForceCallback), as Engine::generate.
    Result<GenerationStats> generate(std::string_view prompt, const SamplingParams& params,
                                     const TokenCallback& on_token, const ForceCallback& force);
    // Engine::generate_tokens on this sequence.
    Result<GenerationStats> generate_tokens(std::span<const int32_t> prompt, const SamplingParams& params,
                                            const TokenCallback& on_token);
    // Engine::prefill on this sequence (cancel_prefill() stops it).
    Status prefill(std::string_view text, bool add_bos = true);
    // Forgets its context (waits for a call running on it).
    void reset_context();
    // Engine::set_adapter for this sequence only: its generations and
    // prefills apply `name` at `scale` ("" none), whatever the others apply
    // (passes shared with them apply each one's own adapter to its rows).
    // Busy while a call runs on it.
    Status set_adapter(std::string_view name, float scale = 1.0f);
    // Its priority (see SequencePriority), taken into account from its next
    // pass on. Thread-safe.
    void set_priority(SequencePriority priority) noexcept;
    [[nodiscard]] SequencePriority priority() const noexcept;
    // Thread-safe: stops its generate() at the next token boundary; its
    // prefill() between two passes (between two blocks when its pass runs
    // alone or with background work only). Never another sequence's call.
    void cancel() noexcept;
    void cancel_prefill() noexcept;

private:
    friend class Engine;
    Sequence(Engine& engine, int32_t id) : engine_(engine), id_(id) {}
    Engine& engine_;
    int32_t id_;
};

}  // namespace liyab

#endif  // LIYAB_ENGINE_H
