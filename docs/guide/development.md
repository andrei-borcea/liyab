# Testing, limitations and development

[← README](../../README.md) · [Documentation index](README.md)

Commands and source paths in examples are relative to the repository root.

## On this page

- [Experimental modules](#experimental-modules)
- [Testing](#testing)
- [Regression run on the phone](#regression-run-on-the-phone)
- [Where a process spends its cores](#where-a-process-spends-its-cores)
- [Roadmap](#roadmap)
- [Known limitations](#known-limitations)
- [Project layout](#project-layout)
- [Contributing](#contributing)
- [License](#license)

---

## Experimental modules

These are built with `-DLIYAB_ENABLE_EXPERIMENTAL=ON` (`scripts/build_android.sh --experimental`; the app's build
includes them, all off by default). Enable them through `EngineConfig::experimental`, the C config or `liyab-cli
--early-exit P --prune-heads R --egls T --tdss hot|always --i8mm-tiles --prerouter-trace --page-cache-mode N`.

| Module | What it does | Measured | Verdict |
| :--- | :--- | :--- | :--- |
| **Early exit** | Projects the hidden state through the LM head after probed blocks; exits when top-1 > 0.98 and writes the skipped blocks' K/V from the exited state | Forced exit at block 6: +53% (synthetic) | Lossy for models not trained for it |
| **Head pruning** | Ranks query heads by `Wo` column norm; while throttled, skips the least important heads' attention | 50% heads: +12% at 1,536 context | Lossy |
| **EGLS** | Skips a block's FFN when the residual's entropy barely changes | TinyLlama: +5–48% | Output diverges early: not viable without a model trained for it |
| **i8mm 2×2 tiles** (`i8mm_tiles`) | Q4_K / Q5_K / Q6_K CPU matmuls over two or more activation rows (prefill, speculative verification) as 2×2 SMMLA tiles: two weight rows against two activation rows, 32 multiply-adds per instruction. The integer sums and the float epilogue (`epilogue_k4`, `epilogue_scaled`) are the dot-product kernels', so replies are bit-identical. The tiles read the weights as stored, so the flag keeps the i8mm repacked layouts off | Snapdragon 8 Elite, 2048×8192 × 4 rows: Q4_K 681 → 601–617 µs, Q5_K 683 → 863–895 µs, Q6_K 756 → 953–1019 µs (measured before the repacked layouts existed) | Exact, but slower than the default except Q4_K: the zips that build the tiles, the per-sub-block scale multiplies and the exact epilogue cost more than the halved multiply-adds. Ignored without i8mm |
| **Page-cache tier** (`page_cache_mode`) | Streamed MoE experts as a second cache level: experts the kernel still holds in its page cache are copied from there (memory speed, outside the process's PSS, which HyperOS kills on) instead of read with `O_DIRECT`. Mode 1 buffers every read, so what the RAM cache evicts stays in the page cache; mode 2 buffers prefetches and reads the experts the router already chose directly. Residency is checked with `mincore` on the model mapping. The resident weights move to anonymous memory (GPU-shared memory when a GPU holds them), because expert pages in the page cache evicted the mapped ones (31k instead of 5k major faults per 48 tokens). `GenerationStats::expert_page_cache_loads` counts the experts it served | Qwen3.6-35B-A3B, Snapdragon 8 Elite, 64 tokens, A/B: flash reads 283 → 180–215 MiB/token with 40–55 experts/token copied from the page cache; decode cold 7.04 → 4.83 (mode 1) / 6.02 (mode 2) tok/s, warm 6.98 → 5.16 / 6.23. Moving the resident weights alone was neutral (7.04 vs 7.04) | Exact, but slower: buffered cold reads run at ~2.5 GB/s against ~3.7 GB/s for `O_DIRECT` and hold the I/O threads that urgent reads wait for. Needs streamed experts (else it warns); `Unsupported` on Windows |
| **Prerouter trace** (`prerouter_trace`) | Measures, on decode steps of a streamed mixture-of-experts model, the precision@k of the engine's guess of the next block's experts (its router run on the residual stream at hand) against the experts the router picked. Two corrections are scored next to it: a co-occurrence prior learned online (how often expert b of block l follows the experts of block l − 1) and the experts the block ran for the previous token. Seven candidates (`PrerouterTrace::candidates()`) are logged at INFO level every 64 tokens (stderr in `liyab-cli`, logcat on Android) | Counts only: it never changes which experts are prefetched or run, so replies are identical | A measurement for choosing a model-agnostic prerouter, not a feature. Needs experts that stream from storage (else it logs a warning and nothing more) and an MoE model (else `Unsupported`); sees single-token decode steps only |
| **JIT unpacker** | Emits unrolled AArch64 NEON unpack routines at runtime (W^X) | +37% for a 4,096-element row | Row-sized routines only; not on the hot path; unavailable on iOS |
| **TDSS (2:4 sparsity)** | While throttled, FFN switches to a 2:4 magnitude-pruned copy (`TBL` gather + `SDOT`) | −54% decode on the phone | Counterproductive without sparse units |
| **io_uring loader** | `O_DIRECT` reads via raw io_uring syscalls | — | Android denies `io_uring_setup` to apps; kept for platforms that allow it |

KV dedup, which persisted a prompt's KV pages to files keyed by hash for later sessions, is now part of the prefix
cache: its store keeps a shared prefix once, on storage across runs, for hybrid models too, within its caps (KV
dedup's files had no bound). The C config's `kv_dedup_dir` is kept for the struct's layout; the engine ignores it
and removes the snapshot files it left there.

---

## Kernel coverage and conformance

The engine sends a matrix product to a backend, and a backend that answers `Unsupported` has that one product run on
the CPU instead. `liyab-kernels` (and `test_kernels`, which enforces the same rules) shows what that means on a host
by running it: for each of the 27 weight formats (every type `dtype_from_ggml` recognizes but `I32`; a format added
there is checked without touching the check) and each backend, it builds random valid weights, multiplies them by
activations and compares the result with `quant::dequantize_row` of the same bytes, accumulated in double precision.
The code and its reasoning are in [`src/core/kernel_conformance.h`](../../src/core/kernel_conformance.h).

* **Random valid weights, no format knowledge.** A block is redrawn until its decoded values are finite and their
  largest magnitude lies in [0.05, 8], so scale fields never give NaN or values 10⁹ times the others.
* **A tolerance of 1e-5 of Σ|w·x|.** The activations lie on the int8 grid the CPU quantizes to (multiples of 2⁻⁶,
  one of them ±127·2⁻⁶ in every run of 32), so quantizing them loses nothing and only float rounding is left:
  correct kernels deviate by ~5·10⁻⁸ (the tool prints the largest deviation seen). A wrong scale, sign or bit in any
  block shows hundreds of times over; a result rounded through half precision (an fp16 accumulator) fails too.
  A deliberate one-bit shift error in the scale decoding of the Metal Q4_K/Q5_K kernels was reported at 4,900–15,000×
  the tolerance, and only in those two formats.
* **Two placements.** Ordinary memory 32 bytes into a page (what a memory-mapped model is), and, where the backend
  has `allocate_shared`, shared memory, filled and then refilled in place: a backend that keeps a copy of the old
  contents is reported as wrong "after a refill".
* **Shapes** with 1 and several blocks per row, row counts that are not tile multiples (37, 5, 8), and 1, 2, 5 and 33
  activation rows.
* **The claim.** `Backend::supports(type)` must be true exactly when the type ran in some placement and shape.
* **Batching.** An n-row call, n one-row calls and a grouped call must give the same bits on the CPU backend
  (speculative verification and concurrent sequences rely on it); other backends are only reported (`=` or `~`).
* **Throughput** (unless `--no-timing`): GB/s of weights for one activation row, the per-row cost of a 16-row call
  relative to it (about 1: the weights are read again for every row) and the multiply-add rate of a 256-row call in
  TFLOPS, on a 128 MiB matrix, in shared memory where the backend has it. Indicative; `liyab-bench` gives stable CPU numbers.

What it shows today (Apple M4 Pro, 29–31 October 2026): the CPU runs all 27 formats right and batch-invariant with
the NEON, AVX2 (an x86-64 build under Rosetta 2) and scalar kernel sets. Metal claims 9 (F32, F16, Q4_0, Q4_1, Q8_0,
Q4_K, Q5_K, Q6_K, Q1_0), runs them right in both placements, and the other 18 run on the CPU. Vulkan, from its source, runs
7 types from ordinary memory (F32, F16, Q4_0, Q4_1, Q4_K, Q5_0, Q8_0) and 23 from shared memory; the check has not
yet been run on a device. [Throughput by format](benchmarks.md#weight-formats-on-the-m4-pro-liyab-kernels).

---

## Testing

The tests depend on nothing beyond the compiler. They generate small GGUF models on the fly, so nothing needs to be
downloaded. One check is in Python (standard library only) and is registered when CMake finds a Python 3
interpreter: `describe_model_registry`.

| Suite | Covers |
| :--- | :--- |
| `test_mmap` | Mapping, GGUF parsing and malformed-file rejection, split models, prefetcher, triple-buffer pipeline, converted tensors saved to a file and mapped back (refused without any change when damaged or for another id), the converted-weights cache's byte cap and least-recently-used eviction |
| `test_device_detect` | SoC classification, backend ranking, sysfs thermal parsing, headroom bands, pacing, emergency guards, sampling only during work |
| `test_text` | The regex engine, the pre-tokenizer table and NFC/NFKC against `tests/data/text_cases.txt` (Python's `regex` and `unicodedata`, cross-checked with the Hugging Face tokenizers' own splitting; NFKC matches its Unicode 9 normalizer); rejected regex syntax; byte-level BPE on generated vocabularies: merges, NFC per family, whole-piece lookup, missing bytes, Phi-3 special tokens, unknown pre-tokenizers; chat templates byte for byte against transformers (`tests/data/chat_templates.json`), Jinja semantics, layouts of turns and prefix stability, `render_chat` / `chat_format` through the engine and the C API, one BOS |
| `test_engine` | All 25 quant formats bit-exact; kernels, each format the same bits for every batch size and layout; on x86-64 the AVX2 set against the scalar one in the same process (identical activation quantization and decodes, close dots, -128 weights); long-context attention the same bits for every batch split, thread count and KV cache type; tokenizers; transformer vs an independent float reference, for every query-group size; batching and rollback; MoE vs its dense twin; fused QKV and gate/up projections vs the separate matrices, bit for bit; block plans (windowed blocks from the architecture or the file, fused projections, LongRoPE, YaRN); streamed experts vs in-place reads; fused `[gate \| up]` experts vs the split tensors, bit for bit, in place and streamed (two reads per expert instead of three); latent attention (absorbed, over cached latents) vs an independent reference that expands per-head keys and values, batched equal to token by token, key scaling over latent rows (values included), through the engine with saved contexts and in every `latent_form`; its expanded form vs the absorbed form and the reference, the same bits for every split into passes and thread count over several tiles of expanded rows (F16, Q8_0 with key scaling, a sliding window with sinks), no further from an F16 cache than the absorbed form, absorbed decoding after an expanded prefill; the automatic form's work counts and threshold, calibrated once and then the same forms and bits while other threads load every core, the ratio kept in `cache_dir` (engines give the same sampled reply, models given the kept ratio the same logits bit for bit; damaged files measured again; in memory without a `cache_dir`); hybrid snapshots, also restored from a saved context; windowed blocks keeping only their window (Gemma 2 and 3, Phi-3, F16 and Q8_0) giving the logits of blocks keeping every position bit for bit over long prompts, decoding, rollback inside the margin and back to a snapshot, refused beyond it, in less memory; their saved contexts restored exactly, older files converted and other layouts refused; their conversations brought back by the prefix cache (replaced, edited back to a snapshot, saved as kept, from storage in a later engine) and their live context reused back to a snapshot; context reuse, also after cancelled work and saves and when a context's text ends inside a character; prefix cache in RAM and on storage (persisting for a later engine within its cap, a shared system prompt kept once and restored for a new conversation, a chat after a long system prompt set aside and restored while an edited draft is not, damaged files removed with the output unchanged, one engine per directory); key scaling (factor rule, Q8_0 against F16 on large key channels, saved contexts converted, factors kept in `cache_dir` and measured again when damaged; a model whose text cannot be encoded loads with keys unscaled and says so); cancelled prefill; forced continuations (one pass shared with the next token, the same output with or without speculative decoding, forced text cut inside a character); speculative decoding; suffix drafts (suffix arrays, the branch most occurrences agree on, earlier conversations stored once and within their caps, LRU, calibrated acceptance, draft length from acceptance and step cost, greedy and seeded sampling equal to plain decoding on attention-only and hybrid models, a stop inside a speculative step leaving the context plain decoding leaves); tree verification (the continuation tree likeliest first, siblings' probabilities within their parent's, calibrated from tree outcomes; a tree pass giving every branch a linear pass's bits and keep_path() that pass's context, rollbacks into it included, on dense, windowed, latent, hybrid and MoE models with Q8_0 and F16 caches; trees verified by the decoder giving plain decoding's tokens, greedy and seeded, and keeping a branch other than the likeliest); Metal vs CPU; C API |
| `test_golden` | Every supported architecture's logits, pinned bit for bit per kernel set in `tests/data/golden_logits.txt` (the Mac and the Snapdragon phone give the same bits): one synthetic model per path of the forward pass (each architecture; QK norms before and after RoPE, gated attention, partial RoPE, `rope_freqs`, linear scaling, tied output; scaled embeddings, GeGLU, norms after the attention and the FFN, windowed and global blocks with their own RoPE bases, soft caps on the attention scores and the logits, the file's q·k scale; fused QKV and gate/up projections, LongRoPE short and long factors, YaRN (its own attention factor, and DeepSeek's log multiplier under key scaling); F16 KV, Q8_0 KV with key scaling (also measured through Q/K norms after RoPE and per-block RoPE tables), sliding window (also over windowed blocks); MoE with softmax or sigmoid gating, selection bias, shared expert and its gate, streamed experts, fused `[gate \| up]` experts in place and streamed (the split model's hashes); DeltaNet hybrids; latent attention with Q8_0 and F16 caches, key scaling over its keys and the values in them, low-rank queries, a dense leading block before MoE blocks with a shared expert, YaRN with DeepSeek's log multiplier, each in the absorbed and the expanded form, the expanded one also with a sliding window and sinks) |
| `test_grammar` | Grammars for constrained decoding: grammar text (literals, UTF-8 classes, repetition, `%until`, `%json`, `%schema`, `%string`), JSON Schema accepted and refused (each refusal names the keyword and where), matching byte by byte with characters split anywhere, forced text stopping at choices and inside characters; masks over the vocabulary trie (split UTF-8 through byte tokens, end of generation only where the grammar accepts); generation on a random-weight model sampled hot, where every output parses and validates against its schema (checked by an independent validator); forced text with constrained spans, the same with plain, lookup and draft-model decoding; token healing (a forced token that a longer allowed token starts with is left to the model, one nothing extends stays forced); an XML call's bounded string value closing with the call (with `%string` it could run over the closing tag and keep the span open); determinism; unconstrained generation unchanged; the C API. With `LIYAB_GRAMMAR_MODEL=a.gguf,b.gguf`: tool-call arguments from real models and the cost of masking per token |
| `test_pooling` | Embeddings and reranking: the pooling metadata (type, EOS, ranking head and labels, prompt; unusable heads reported, the model still loading), prompt templates, normalization and `dimensions`, scores of one or several labels; a last-token vector is the state the LM head reads, a mean the mean of every token's state across passes, a score the head's softmax (with and without bias) or the LM head's over `labels`; the same bits alone, in any batch and order with shared prefixes, on hybrid models and through a sliding window; which pooling applies and what is refused, inputs past the context, truncation; the context left empty and a conversation restored from the prefix cache; the F16 KV of pooling models; cancellation; the C API |
| `test_kv_cache` | Paging and reuse, sinks over 5,000 positions, quantization accuracy, values read from the key rows (latent attention) bit for bit in half the memory; per-block layout: one group byte for byte the single table with its memory, windowed layers keeping their window and the others every position from one pool of one page size (pages reused across groups), rollback inside the margin and refused beyond it, a pinned position keeping its window so the context returns to it exactly, rows images over several pool pages, window pages out and back bit for bit (values in keys too, the pages a context and its pins keep), older single-table pages converted, key scaling over the rows kept, the longest context in a fixed budget; pages out as chunk images and back bit for bit (whole, partial, values in keys); the KV store on synthetic chunks: a shared prefix kept once, a prompt restored to where it diverges, hybrid restores to the deepest state reached with the snapshots below and an image shared by two chunks, windowed blocks restored to the deepest position whose window pages are whole (with the snapshots below, never past a missing page), both caps never exceeded with LRU eviction leaf first and a context used again outliving newer ones, RAM spilling to storage and back, a context too large for both, files persisting in LRU order, one store per directory, other seeds' files kept and never returned, interrupted writes, foreign and damaged files and broken chains removed; segments kept as contexts no prompt's chain reaches, found by their tokens whatever position they were computed at, the most recently used copy first, a segment that starts another kept apart, on storage for a later store; chunk encodings: lossless bit for bit for F16, Q8_0, Q4_0 and Q4_1, two planes and one, whole and partial chunks, on one thread and on a pool, malformed payloads failing cleanly, the 4-bit encoding within one step of what it quantizes over; compressed chunks on storage counted by their files, read by a later store, older raw files still read, a lossy store's chunks kept apart from a lossless one's, a damaged compressed payload removed when read, chunks of two pages read into parts cut off the page boundary; coding only where the given costs say it pays (raw files otherwise), measured costs kept and found again, reads ahead in order, with a chunk skipped and out of order; the store's seed switched while open (each seed's contexts returned only under it, in RAM and on storage, chunks, states and records found by a later store and its records read back) |
| `test_kv_reuse` | KV reused away from its prefix: rows copied out of and into any positions bit for bit (values in keys too); keys moved to other positions equal to the keys computed there, within two rounding steps, for every RoPE variant (adjacent and half pairs, partial rotation, `rope_freqs` with linear scaling, YaRN, LongRoPE, Q/K norms after RoPE, windowed and global RoPE tables, latent attention's RoPE key) and every cache format, key scales included, with values, unrotated latents and moves by 0 keeping their bits; blocks run at scattered positions giving the rows and logits of a whole pass bit for bit (dense, windowed, latent attention, MoE); hybrid models and sliding-window caches refused; segments reused with every token recomputed giving a full prefill's rows and logits bit for bit for every criterion (dense, key-scaled, windowed, latent attention), stored in chunks of one, two and four pages and starting inside a page, a copy reused where it was computed needing none, a damaged copy processed instead; a segment's tokens the same wherever it appears, BOS outside it, malformed ranges refused; through the engine: segments kept and reused elsewhere, one longer than two of the store's chunks reused whole, a request that marks none getting a fresh context's reply (the reused rows processed again, set aside and saved only up to the first one, restored state included), refusals with their reasons (off, no prefix cache, a hybrid model, a target never met, read back from `cache_dir`) leaving marked prompts exact; the calibration in stages, stopped and resumed, keeping the smallest share whose bound, and every larger share's, is within its target (a question-guided criterion with a pass per question); in the background: the engine created at once, marked prompts processed as usual until the calibration ends, a call stopping it without `Busy` and finding its context as it left it, the result read back from `cache_dir`, an engine closed while it runs; the C API (`Busy` while calibrating) |
| `test_sequences` | Several sequences: `forward_batch()` giving every sequence the bits `forward()` gives it alone, whatever the others in the pass do (prompts in passes, decoding, sequences joining later), on dense, gated, partial-RoPE, Q/K-normed and key-scaled models (Q8_0, F16, Q4_0 caches), windowed blocks (Gemma 2/3, Phi-3), hybrids with rollback checkpoints (and a rollback in a parked context after a shared pass), MoE in place and streamed (fused gate/up too), latent attention absorbed, expanded and automatic (forms mixed in one pass), LoRA adapters per sequence (dense, per-expert, shared expert); a cancelled pass returning every context to its start; what is refused. Through the engine: concurrent generations (greedy and seeded sampling) equal to each one's output alone on those models, with suffix drafts, under one grammar, with per-sequence adapters, one cancelled while the others go on, a background prompt stopped for the user's chat (its first token before the background prompt ends), a context one sequence set aside restored by another, the limits (`Busy`, `max_sequences`, a draft model); the C API. Sequence memory: with no sequence open, `max_sequences` 4 gives the memory plan, expert cache, requantization decision and greedy output of `max_sequences` 1 (dense, hybrid, hybrid MoE streamed, and a hybrid MoE whose tight cache requantizes); an open sequence takes its states out of the expert cache and closing gives them back (out of order too), plain models and in-place experts borrowing nothing; a cache at its minimum refusing the open (`OutOfMemory`, C API too) with nothing changed; the expert store lending and repaying slots while other threads acquire experts, each one's bytes the file's; sequences opened and closed while another decodes, its output unchanged |
| `test_lora` | LoRA adapters: base model + adapter giving the logits of the model with the adapter merged into its weights (prefill and decoding), for dense blocks and the LM head, F32 / F16 / BF16 / Q8_0 adapter tensors, QKV bias, a mixture of experts' routed experts (split and fused gate/up), shared expert and router, hybrid DeltaNet blocks and latent attention's whole projections; no adapter giving the base bits back; malformed files and unsupported targets refused (half pairs, stray tensors, mismatched ranks, missing or misshapen matrices, token embeddings, activated LoRA, latent attention's per-head matrices); through the engine: adapters loaded, listed and selected per generation (the merged model's replies, the base model's without, a context continued only under the same adapter and scale), speculative decoding verifying with the adapter (tree passes under it giving every branch a linear pass's bits), the prefix cache and saved states keeping contexts apart by adapter (in RAM, on storage across engines), the memory cap releasing the least recently selected and mapping it again, refusals; the C API |
| `test_kernels` | Every weight format the loader recognizes (all 27) on every backend of the host, against a reference decoded from the same bytes ([what is checked](#kernel-coverage-and-conformance)): the CPU backend right on every format and giving the same bits for any batch, Metal and Vulkan doing exactly what `supports()` says, shared regions refilled in place; and the check itself failing on a wrong answer, a refusal, an error, a missing write, a false claim, a batch-dependent result and a backend that keeps the old contents of a region; the bytes a decoded token reads of a model file (no embedding rows beside a separate head, the whole table when tied, the router's share of a mixture's experts), the decode ceiling from them and measured speeds (a refused format at the CPU's speed, unknown when untimed, the streaming message), and the host profile kept per SoC, memory size and thread count (`test_engine` covers `liyab_describe_model_fit` through the C API) |
| `test_backends` | Per-matmul latency per backend, grouped submissions, batched CPU matmuls equal to one matmul per item, the final logit soft-cap giving the same bits on 1 and N threads (and its cost on a 256k vocabulary); the engine's own tanh within 1.32 ulp of the true one, odd, monotone and saturating, its results hashed to the same pinned value on every platform; the route picker's schedule, margin, warm-up and CPU probe slices, and its kept decisions (only clear ones, adopted without measuring, damaged ones ignored); the directory of measured values (read back, bounded, least recently used out, damaged files ignored); Metal's K-quant matvecs, fused FFN and prompt-block attention (F16 and Q8_0 caches, 64–256 channels, grouped heads, soft-cap, blocks that are not a multiple of 32 rows) against double-precision references, and the jobs it refuses |
| `test_experimental` | Experimental modules, plus throughput and I/O benchmarks (`LIYAB_BENCH=0` skips them) |
| `describe_model_registry` | `tools/describe_model.py --self-test`: every row of the registry in `architecture.cpp` read and printed back as the file writes it, so the tool that proposes rows keeps the registry's format |
| `app/test` | Tool-call formats; the app's own tool calls: each tool's call grammar in both formats, the system prompt that describes the tools pinned byte for byte, every tool's calls admitted by its grammar and read back valid (checked apart from the grammar), what a call cannot be (another tool, undeclared, missing or reordered parameters, times out of shape), each shape of a time read as local time, XML string values under a length bound or an unchecked format that cannot take in their closing tag, a scripted model calling through the structure the app generates with (reasoning first, a reply without a call untouched, the structure run on another isolate), the step log's cost line; with the engine built on the host (`build/libliyab.dylib`, or `LIYAB_LIBRARY`) the engine itself compiles and matches those grammars, else those tests are skipped; with `LIYAB_TOOL_MODEL=a.gguf,b.gguf` on a Mac, real models answer questions for each tool through the app's prompt and structure, every call valid, and the grammar's cost per token is printed; chat history replay; typing ahead (when words, the whole message and pastes are prepared); prompts from a model template's pieces (Qwen3.6's equal the built-in ChatML; Gemma's first turn); the local API's OpenAI, Anthropic and gRPC servers over real sockets, with tools: call grammars in both formats (the same texts `test_grammar` checks against the engine), when calls are spans and when they are forced, calls split from text as they stream, `tool_calls` / `tool_use` with streamed deltas, history with calls and results, `tool_choice`, `parallel_tool_calls`, `response_format`, and what is refused; embeddings (one call per request, floats and base64, `dimensions`, usage) and reranking (sorted results, `top_n`, documents echoed or not, every path) over real sockets, and what a pooling model or another model refuses; the local API's LoRA adapters (listed after the model by OpenAI, Anthropic and gRPC, selected by `model` or `adapter`, an unknown one refused); with `LIYAB_LIBRARY` and `LIYAB_TEST_MODEL=a.gguf`, the engine's lanes: an idle lane holding no sequence (the memory plan of no lane), a generation on the API's lane opening its sequence and running alongside the chat's, each reply what it is alone, the sequence closed once the lane is idle and opened again for the next request, one generation stopped without the other, the lanes closed with the model |
| `test_ssm` | Mamba-1/Mamba-2 scans and convolution against independent scalar equations, SIMD tails, nonzero encoded projections in all 27 formats and four KV types, projection/scan-state ablations, prefill/decode/rollback logit and state invariants, repacked projections, Metal/Vulkan projection routing when available, incomplete block rejection and unsupported parallel attention/SSM refusal |

Benchmarks and tools built with the tests:

* `liyab-kernels`: which weight formats each backend runs on this host, and whether it runs them right
  ([details below](#kernel-coverage-and-conformance)). Exit status 1 when anything is wrong, so a build for a new
  host (Android, x86-64, Windows) can be gated on it. `--backend cpu|metal|vulkan`, `--only q4_k,iq2_xs`,
  `--threads N`, `--no-timing`. `--model model.gguf` adds how that model will run on this host: it times the formats
  the model holds (those making up at least 0.5% of what a token reads) on the CPU and every backend asked for, then
  prints the weights by format, whether the file fits in the memory available for weights (else the engine streams
  layers), and for each backend the decode rate its measured read speeds allow at most. A decoded token reads every
  weight once, except the embedding table when a separate output head exists (one row) and the experts a mixture's
  router does not pick; a format a backend refuses is counted at the CPU's speed. It is a ceiling, not a forecast: it
  leaves out attention over the cache, recurrent state and launch costs, so the engine reaches about 80% of it (M4 Pro,
  CPU: 27B Q4_K_M 16.7 against 13.3 tokens/s, 14B Q5_K_M 24.6 against about 20). Automatic routing still decides by
  timing real passes ([`src/core/model_bytes.h`](../../src/core/model_bytes.h),
  [`model_fit.h`](../../src/core/model_fit.h)). `liyab-cli --doctor -m model.gguf [--cache-dir DIR]` prints the same
  report without the conformance run (`liyab_describe_model_fit` in the C API): it times only the formats the model
  holds that the host profile lacks, on the CPU and on each accelerator the host offers, and keeps the speeds in
  `cache_dir` as `host-*.bin` ([`src/core/host_profile.h`](../../src/core/host_profile.h): one small file per SoC,
  memory size, kernel set, thread count and backend set, at most 4). On the M4 Pro the 14B Q5_K_M costs 2.8 s the
  first time and 0.4 s after; the 27B Q4_K_M then times only Q4_K, since Q6_K is known. The timings are short and move
  by about ±10% between runs. Without a `cache_dir` they are taken on every call.
* `liyab-bench`: CPU matvec throughput on a MoE's decode shapes, against the RAM read ceiling. It prints the kernel
  set first; on x86-64, `LIYAB_KERNELS=scalar` runs the same cases on the scalar kernels. `--prefill model.gguf`
  times a model's prefill instead: prompts of each `--tokens` length (synthetic token ids) in passes of `--batch`
  tokens on the CPU, best of `--repeat` runs; `--latent-form auto|absorbed|expanded` sets how a latent-attention
  model attends (auto measures its ratio first, as the engine does at load, and prints it), and the line reports
  the passes each form took.
* `liyab-kl`: teacher-forced next-token distributions compared against a reference, for two uses.
  * **The price of a lossy option.** It saves the distributions from the lossless path, then reports KL divergence,
    top-1 agreement and perplexity for a lossy run. The two runs are separate processes, so a model that fills the
    RAM fits once. A quantized KV cache stores keys scaled, as the engine does; `--no-key-scaling` stores them
    as computed, to measure what the scaling saves. `--batch N` sets the tokens per pass (64), and
    `--latent-form absorbed|expanded` pins how a latent-attention model attends, to measure one form against the
    other. `--store-codec lossless|lossy4` measures how the prefix
    cache stores chunks: every KV page the text fills is encoded and decoded as the store would, later positions
    attending to the decoded rows; it prints the ratio and the encode / decode time per MiB, and a lossless
    encoding must give the same bytes back.
  * **Conformance with another implementation.** `tools/conform_reference.py` writes the same reference from
    Hugging Face transformers (needs `torch` and `transformers`). The comparison first checks that the text encodes
    into the same token ids, printing the first difference, then measures the forward pass; `--max-kl` and
    `--min-top1` make it a pass/fail gate, `--ref-tokens` measures the model on the reference's ids while a
    tokenizer still differs (or cannot encode the text yet), and `--kv f16` keeps the KV cache's own rounding out of the measurement. `tests/data/conform_text.txt` is the text used (prose, Italian, code, numbers, URLs,
    CJK, Greek, Cyrillic, Hebrew, Arabic, emoji, tabs and trailing spaces). The script's `--config` sets
    configuration fields before the model is built (a GGUF copy given YaRN metadata is compared with the model
    configured the same way), and `--attn-implementation` picks transformers' attention kernel (`eager` for
    Gemma 2: the default one leaves its soft cap out). The text is read byte for byte, `\r\n` included.
    [LoRA adapters](inference-features.md#lora-adapters) are checked the same way: the script's `--lora DIR` applies a PEFT adapter
    with PEFT (needs `peft`), and `liyab-kl --lora ADAPTER.gguf [--lora-scale S]` applies the same adapter
    converted to GGUF.

  ```bash
  tools/conform_reference.py --model Qwen/Qwen3-0.6B --text tests/data/conform_text.txt --out qwen3.ref
  build/liyab-kl -m Qwen3-0.6B-BF16.gguf -f tests/data/conform_text.txt --compare qwen3.ref --max-kl 0.01 --min-top1 97
  tools/conform_reference.py --model Qwen/Qwen2.5-0.5B-Instruct --lora my-peft-adapter --text tests/data/conform_text.txt \
      --out lora.ref
  build/liyab-kl -m qwen2.5-0.5b-instruct-f16.gguf --lora my-adapter-f16.gguf -f tests/data/conform_text.txt \
      --compare lora.ref --kv f16 --max-kl 0.001
  ```
* `liyab-experts` and `tools/expert_study.py`: measurements on a MoE's routed experts, for the
  [cold-expert study](../research/cold-experts.md). `liyab-experts capture` runs a text one token at a time
  through a backend that wraps the CPU one and records routing counts and traces, the input second moments of
  every expert and sample inputs (the engine is not modified); `to-q8` copies a GGUF with its experts in Q8_0;
  `dequant` / `dequant-raw` decode tensors or raw blocks with the engine's dequantizer. `expert_study.py`
  (needs `numpy`, `torch`, `gguf`) computes spectra, builds candidate representations and their errors, writes
  model variants with the least used experts replaced (for `liyab-kl`), and replays routing traces through an
  LFU cache.
* `liyab-spec-bench`: speculative decoding without a draft model on assistant tasks (a code edit, a summary, a
  repeated tool output over two turns, free chat, and that tool conversation again as a new one). It compares
  plain decoding with lookup and suffix drafts, greedy and unpaced, one mode's engine at a time (each plans its
  memory as if alone; the order flips every second repetition so heat evens out), and reports tok/s, accepted drafts, tokens per pass and whether each reply equals the
  plain one (`--repeat R`: medians of R runs, and the best run, the least disturbed on a shared host). A mode
  `lookup:K:fixed` or `suffix:K:fixed` verifies every draft on offer, so its acceptance and tokens per pass do not
  depend on timing; `suffix:K:linear` verifies only the suffix drafter's best branch instead of its tree;
  `self:K` uses the target as its own draft model, every draft accepted (a check of batched verification).
  `--requant 4|5` converts the weights at load as `liyab-cli` does. A
  reply that differs from the plain one is printed where it differs.

  ```bash
  build/liyab-spec-bench -m Qwen3-0.6B-BF16.gguf --modes plain,lookup,suffix,suffix:8
  ```
* `liyab-seq-bench`: [several sequences decoded together](memory-and-serving.md#context-memory-and-serving). For 1 to 4 sequences, each
  with a writing task of its own (prompt processed first, not timed), it runs their generations at once and the
  same generations one after another, greedy and unpaced, on fresh engines, the two back to back in alternating
  order so they share the host's load. It reports each sequence's tok/s together, the total together and in
  turns, the gain (their ratio), whether every reply together equals its reply alone, and the scheduler's
  counters (passes shared, time in a pass, time a leader waited for the others). `--lookup` turns on drafts from the text
  itself (`lookup_drafts`), which a sequence uses while it runs alone, to compare sharing passes with drafting; `--repeat R` prints medians of R runs.

  ```bash
  build/liyab-seq-bench -m qwen2.5-0.5b-instruct-q8_0.gguf --sequences 1,2,3,4 --repeat 3 --threads 8
  ```
* `liyab-tok`: a model's tokenizer on its own, from the full GGUF or a vocabulary-only one. It prints the ids of a
  text, compares them with reference ids (first difference shown with the pieces around it, exit status 1), or
  measures encoding speed (`--bench N`). `tools/tokenizer_reference.py` writes the reference from the model's
  `tokenizer.json` (needs `tokenizers`), and `tools/gguf_vocab.py` writes a vocabulary-only GGUF from the header
  of a local or published GGUF, read with HTTP range requests, so a tokenizer is checked without downloading
  weights.

  ```bash
  tools/gguf_vocab.py --repo bartowski/tencent_Hunyuan-0.5B-Instruct-GGUF \
      --filename tencent_Hunyuan-0.5B-Instruct-IQ2_M.gguf --out hunyuan.gguf      # ~5 MB of the file
  tools/tokenizer_reference.py --model tencent/Hunyuan-0.5B-Instruct --text tests/data/tokenizer_edge.txt --out hunyuan.ids
  build/liyab-tok -m hunyuan.gguf -f tests/data/tokenizer_edge.txt --ids hunyuan.ids
  ```
  `liyab-tok -m model.gguf --render request.json` applies the GGUF's chat template to a request, and `--format
  PROBE` prints its layout of turns and whether it is prefix-stable.
* `tools/gguf_fuse_experts.py` copies a GGUF with each block's `ffn_gate_exps` and `ffn_up_exps` fused into one
  `ffn_gate_up_exps` (standard library only, rows copied byte for byte): the copy must give the original's logits
  bit for bit, in place and streamed, which checks the fused-expert path on any real MoE file.
* `liyab-embed`: an embedding model or a reranker on its own. It embeds the texts of a cases file or reranks its
  queries' documents, checks every input's token ids and then each vector's cosine (`--min-cosine`, default
  0.999) or each score's distance (`--max-score-diff`, default 0.001) against `tools/pooling_reference.py`'s
  reference, written from Hugging Face transformers the way the model cards run the models (needs `torch` and
  `transformers`). It also runs every input on its own and checks it gives the same bits as in the batch, and
  `--bench N` measures texts or pairs per second. `--pooling`, `--no-normalize`, `--labels` and `--template`
  select the other paths.

  ```bash
  tools/pooling_reference.py --embedding Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \
      --cases tests/data/pooling_cases.json --out pooling.ref.json
  build/liyab-embed -m Qwen3-Embedding-0.6B-f16.gguf --cases tests/data/pooling_cases.json --ref pooling.ref.json
  build/liyab-embed -m Qwen3-Reranker-0.6B-BF16.gguf --cases tests/data/pooling_cases.json --ref pooling.ref.json
  ```

* `liyab-reuse`: what segment reuse costs in quality and saves in time to the first token, on RAG-shaped prompts
  (a system prompt, 3 to 5 documents of its own corpus in varying order and a question, through the model's chat
  template). Documents are first stored by warm-up prompts (processed whole, or with `--warm-up-reuse F` as the
  engine processes them: the documents stored already reused at share F, the new ones kept with rows computed
  after those); each measured prompt is then answered from a full prefill (the reference, and its greedy
  answer), with the part before the first document taken as restored (lossless), and with the documents reused
  at each recompute share of `--fractions`. It reports KL(full || reuse) over the whole vocabulary and the top-1
  agreement at the first answer token and each next one up to the reference answer's end of generation (at
  most `--answer`, teacher-forced on the reference), how many greedy answers come out the same, and the times
  to the first token. `--criterion`, `--select-block`, `--refine`, `--first-tokens` and `--absolute` select how
  the recomputed tokens are chosen, `--no-numbers` leaves the documents unnumbered and `--dump` prints the
  positions that moved most. `--shift D` measures turning keys alone (a text's rows moved D positions away and
  back). `--calibrate KL` runs the engine's calibration against that target (with its questions' own KL), then
  measures the share it kept (the largest when it refuses) on the tool's own prompts and prints both; a hybrid
  model is refused with the share of a prompt token's work reusing its attention blocks' keys and values would
  skip.

  ```bash
  build/liyab-reuse -m Qwen3-0.6B-BF16.gguf --prompts 8 --fractions 0,0.1,0.2,0.3
  ```
* `tools/describe_model.py`: the registry row a model needs, established from its Hugging Face implementation
  and a GGUF of it, or the precise list of parts Liyab does not implement yet; it never guesses a fact it cannot
  establish. `tools/reference_trace.py` runs a reduced copy of the transformers model with random weights and
  recomputes what it observes (block order and residual adds, norm formulas, RoPE pairing and frequencies, Q/K norm
  order, attention scale, window and soft-cap, FFN activation, MoE routing, embedding and logit scaling); the GGUF's
  metadata, tensor names and a few weights of both checkpoints (one key projection, one block's norms, read with
  HTTP range requests when remote) give the rest. `--gate` adds the row to a scratch copy of the sources (the
  repository is not touched), builds `liyab-kl` and `liyab-tok` there, and accepts the row only if the tokenizer
  gives the model's own ids on three texts and `liyab-kl --kv f16` passes against transformers. How each fact is
  established and what the tool said about the supported architectures and others:
  [docs/model-descriptions.md](../model-descriptions.md).
  Architecture traces use explicit causal masks so eager score implementations can be observed alongside
  transformers' attention registry, observe the actual LayerNorm epsilon when it is a function argument,
  and check scalar activation parameters against the checkpoint and GGUF.
  Parallel blocks bind mixer and FFN norms independently, with a shared input norm when one tensor is absent.
  Attention traces recognize repeated GQA keys and values before matching their source projections.
  Observed GELU modes distinguish exact erf from tanh even when tiny inputs make their outputs nearly equal.
  Classic host operations are checked with GPU projections through prefill, decode and rollback as well.
  LayerNorm affine steps and ReLU/ReLU² have NEON, dispatched AVX2 and scalar paths; GELU/xIELU retain scalar
  libm for their rounded transcendental functions.
  Odd FFN widths exercise SIMD tails across batching and fused/split projections on every KV type.
  Reference traces also bound typed child FFN/expert configurations, keeping original residual/head dimensions.
  Unsupported gated xIELU FFNs are refused when describing the file.
  Learned position embeddings combined with an embedding norm are refused, including bias-only norms.
  Optional attention-output bias follows both ordinary and latent attention projections, in either latent form.
  Local PyTorch `.bin` checkpoints, including indexed shards, use `weights_only` memory maps for tensor inspection;
  remote inspection requires safetensors. `python tests/test_describe_model.py` checks both local readers with
  the reference environment, including row bounds, saved gradient flags and lazy shard loading.

  ```bash
  tools/describe_model.py --self-test     # the registry parser and printer against architecture.cpp
  tools/describe_model.py --model unsloth/gemma-2-2b-it \
      --gguf hf://bartowski/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q8_0.gguf     # a few MB of the file
  tools/describe_model.py --model baidu/ERNIE-4.5-0.3B-PT --gguf ERNIE-4.5-0.3B-PT-F16.gguf --gate
  ```
* `tools/tiny_checkpoint.py`: a small checkpoint of a model with random weights, built from its configuration (few
  blocks covering every kind, 4 heads of the model's own head size, narrow FFNs, 16 experts) and saved with the
  model's own tokenizer files, so an architecture is checked against transformers without downloading its weights:
  convert it to GGUF, then `conform_reference.py` and `liyab-kl` as for a real model
  ([docs/model-descriptions.md](../model-descriptions.md#without-the-weights)).

  ```bash
  tools/tiny_checkpoint.py --model Qwen/Qwen3-0.6B --out tiny-qwen3
  ```
* `tools/compression_reference.py PREFIX --out REPORT.json` compares a fixture emitted by
  `build/test_compression --export-fixture PREFIX [--original-geometry] [--indexer-only]` with native transformers
  shared-KV, compressor and residual modules. It reports logits and top-k selection agreement. The default
  fixture retains all three attention kinds; `--indexer-only` isolates first-layer selection. This is component
  evidence with dense FFNs and disabled sinks, not a full-model gate or a tokenizer check.
* `tools/gen_text_tests.py` regenerates `tests/data/text_cases.txt` (needs `regex` and `tokenizers`), and
  `tools/chat_template_reference.py` regenerates `tests/data/chat_templates.json` from transformers.
* `tools/gen_quant_vectors.py` and `tools/gen_ssm_quant_vectors.py` regenerate `tests/data/quant_vectors.bin` and
  `ssm_quant_vectors.bin`: random blocks of every stored format with the values llama.cpp's reference decoder gives
  (gguf-py 0.19.0; for Q1_0, which gguf-py cannot decode yet, llama.cpp's own libggml, found through `GGML_LIB` or
  the system library path: `tools/quant_reference.py`). A new format is appended to the end of the list, so the
  blocks of the earlier ones do not change. Tests that run every format take their count from the loader's format
  list (`conformance::weight_formats()`), so a format missing from the vectors fails them.
* `tools/hf_survey.py fetch DIR` / `report DIR`: what Hugging Face publishes against what Liyab runs, from the public
  API (no token): the most downloaded GGUF repositories by architecture, pipeline and quantization tag, and the
  text-generation repositories by `quantization_config` and tensor dtype. Supported architectures are read from the
  registry in `src/compat/architecture.cpp`, recognized tensor types from `src/core/types.cpp`, so the report
  needs no maintenance when either changes.
* `LIYAB_BENCH_ATTENTION=N test_engine`: attention cost at an *N*-token context.
* `LIYAB_GOLDEN_UPDATE=1 test_golden`: records the logit hashes of the kernel set this build runs (CPU architecture and `scalar` / `neon` / `neon+dotprod` / `avx2`; on x86-64 run it once with `LIYAB_KERNELS=scalar` too), after a change meant to move outputs
  (say why in the commit). A platform with no hashes yet is skipped with a note.

On a device:

```bash
scripts/build_android.sh --experimental
adb shell mkdir -p /data/local/tmp/liyab
adb push build/android/arm64-v8a/{libliyab.so,liyab-cli,test_*} /data/local/tmp/liyab/
adb shell 'cd /data/local/tmp/liyab && export LD_LIBRARY_PATH=. TMPDIR=/data/local/tmp/liyab &&
           for t in test_*; do ./$t || exit 1; done'
```

The tests remove the files they write, so repeated device runs leave nothing behind.

### Regression run on the phone

`scripts/phone_regression.py` is run on every build. It builds for Android, runs the unit tests on the phone, then
plays one short chat (a system prompt and a question, then a follow-up that continues it) through `liyab-cli` and
through the app's local API:

* **Latency:** the load, the first token of the conversation (its whole prompt processed) and the first token of
  the follow-up (only the new turn processed).
* **Speed:** prompt and decode tokens per second, CLI and app.
* **Correctness:** the greedy replies must equal the stored reference, the CLI's three runs must agree, and the
  app's replies must equal the CLI's.
* **Memory:** the peak resident memory of the CLI and of the app.

Programs started through `adb shell` inherit adbd's `oom_score_adj` of -1000, so Android can never stop them: one
that takes more memory than the phone has makes the kernel kill everything else and stall until the phone
reboots. The script raises its commands' `oom_score_adj` to 1000, so a measurement is the first to go; do the same
when running engine binaries by hand (`adb shell 'echo 1000 > /proc/self/oom_score_adj; ./liyab-cli ...'`).

```bash
scripts/phone_regression.py --note "what changed"           # CLI part; models in /data/local/tmp/liyab (default moe.gguf)
LIYAB_API_TOKEN=... scripts/phone_regression.py             # the app too, with the token from its Local API settings
scripts/phone_regression.py --accept                        # after an intended change of the outputs
```

* **Same settings.** The CLI runs with the app's defaults (CPU, Balanced, 5,500 MiB, automatic requantization), so
  the two compare directly. Like the app, it keeps the KV cache's key scales in a cache directory, so only a
  model's first run measures them. The log records whether the app had all the cores (its cpuset), because Android
  confines background apps.
* **History.** Every run is appended to `tests/device/runs.jsonl`, with its commit. The CLI numbers are the medians
  of three runs (`--repeat`), and each number is compared with the median of the five runs before it. A change
  counts only beyond a tolerance, so a warmer phone does not raise alarms. The references live in
  `tests/device/references.json`. The exit status is 1 when a test failed, a reply changed or a number got worse.
  On the reference phone the 35B's chat measures, in the CLI: first token 5.0–5.2 s (a 180-token prompt), follow-up
  first token 1.5–1.6 s (27 new tokens), decode ~10.3 tok/s, peak RSS 4.6–4.7 GB; Qwen3.5-4B: first token ~4 s,
  follow-up 0.75–0.84 s, decode ~11.9 tok/s in the CLI and 14–16 tok/s in the app, peak RSS 2.9 GB.
* **Never in the way.** The phone is someone's phone, so the run waits while its screen is on. It also waits, before
  a CLI run, while the app holds a model in RAM, and it waits for the battery to cool down before each measurement.
  If the phone is picked up or unplugged mid-step, the step stops at once (a request to the app is closed, which ends
  its generation) and runs again once the phone is idle and back. The app is never stopped or opened; when the OS
  has frozen it in the background, the run says so instead of waiting for an API that cannot answer.

### Where a process spends its cores

`scripts/thread_cpu.py` samples a process on the phone over adb: every few seconds it reads
`/proc/<pid>/task/*/stat` in one call, turns each thread's user + system time since the last sample into CPU % of
one core, and prints the busiest threads with the core each last ran on. It starts with the cores the process may
use (`Cpus_allowed_list`) and its cpuset, since Android confines background apps. `--out` keeps a capture as JSON
and `--compare` sets two captures side by side, per thread name. The engine's decode workers are named
`liyab-worker-N`; the thread that calls the engine also decodes (in the app a Dart thread, in the CLI its main
thread). It only reads `/proc`: it never starts or stops an app.

```bash
scripts/thread_cpu.py --app com.liyab.chat --duration 30 --out app.json    # while the app answers
scripts/thread_cpu.py --name liyab-cli --duration 30 --out cli.json        # while the CLI decodes the same prompt
scripts/thread_cpu.py --compare app.json cli.json                          # average CPU % per thread name, A vs B
```

It is the tool for the gap between the app's decode speed and the CLI's on the same model: whether the app's
workers get fewer cores or slower ones, or other threads (Flutter's UI and raster threads, Dart) compete with them.

---

## Roadmap

Implemented features are described above. These are next, in order:

1. **Model support as data** (in progress). In place: an architecture is one row of facts its tensors cannot tell
   (`src/compat/architecture.cpp`) and every block gets a plan from the tensors present, so a model made of existing
   pieces needs no engine code; every architecture is pinned bit for bit (`test_golden`) and checked against
   Hugging Face (`liyab-kl`); YaRN RoPE and logit soft-capping are read from the file (Gemma 2 runs), and experts
   may come as fused `[gate | up]` tensors; multi-head latent attention caches latents (DeepSeek-V2-Lite,
   Youtu-LLM). A model's row is established from its Hugging Face implementation and a GGUF and verified by the
   conformance gate (`tools/describe_model.py`), which regenerates every row in the registry. Next:
   * sparse segment relocation (GLM-5 and DeepSeek-V3.2 have tiny sparse gates);
   * later, multimodal and classification models, and encoder-only embedding models (BERT, XLM-R: bidirectional
     attention, learned positions, LayerNorm with bias, WordPiece and Unigram vocabularies).
2. **Op fusion and fewer synchronizations.** One command buffer per layer on Metal and Vulkan (today: per matmul),
   and fewer fork/joins per CPU block.
3. **CPU and GPU in parallel.** A per-operation split calibrated on the device, so both work on every token.
4. **Speculative decoding with draft-guided expert prefetch.** The draft's routing tells the store which experts the
   target will need. Then verification of the suffix drafts' whole tree in one pass, with tree attention masks.
5. **Several sequences, next.** [Shared passes](memory-and-serving.md#context-memory-and-serving) are in place. Next: multi-row kernels
   for F16 / BF16 weights, which today compute each row's dot product on its own (a shared pass gains little
   there); speculative drafts inside shared passes (each sequence's tree as a segment of the pass); one page pool
   for every sequence's KV cache, so a sequence's pages go back to the others when its context shrinks.
6. **LoRA adapters, next.** [Adapters](inference-features.md#lora-adapters) for dense and MoE layers switched per request are in place.
   Next: the update computed on the GPU next to the base product (today the CPU adds it after an accelerator's
   matmul), adapters of the token embeddings, latent attention's separate/combined per-head matrices and
   shared-KV's grouped first output projection, and a lower fixed cost per
   adapted matmul at low rank.
7. **Local memory and retrieval** on the [pooling models](inference-features.md#embeddings-and-reranking): an index over the user's own
   data, an intent router, and an embedding model kept loaded next to the generative one; several inputs in one
   pass (a mask per input), which the GPU path needs most: a pass costs it ~35 ms of fixed work, against ~3 ms on
   the CPU.
8. **Two tiers.** A small model always loaded for the assistant sheet and tools, and the large one in a separate
   process with its own memory limit, called when the small one escalates.
9. **Smaller experts.** A [measured study](../research/cold-experts.md) chose an importance-weighted Q3_K tier
   (a format the engine already decodes): with half of each block's least used experts in it, `liyab-kl` measured
   KL 0.016 and 94.7% top-1 agreement; with all experts, 0.040 and 91.2%, 29% fewer bytes per expert. Next: Q3_K's
   decode cost on the phone, then the tier (a mixed file written once on the device). Binary factorization
   (LittleBit-style) and a non-uniform 3-bit codebook were measured and not chosen.
10. **The agent.** Planned in this order:
    * voice input (on-device speech recognition);
    * actions confirmed by the user (alarms, events, replies);
    * reliable message reading (persisted notification history, RCS);
    * local memory over your own data;
    * the screen content through a voice-interaction service.
11. **Platforms.** The iOS app, which finishes work in the background through a continued-processing task and shows
    it as a Live Activity (Dynamic Island, lock screen); the app on Linux and Windows (the same Flutter app and
    command bar as on macOS), which first needs the engine's [Windows port](building.md#windows-x64) built and tested on
    Windows and the x86-64 kernels validated on native x86 machines, then AVX-VNNI / AVX-512 paths; NPU compute (QNN HTP,
    NeuroPilot); app localization.
12. **Energy.** Joules per token measured on battery, as a first-class benchmark next to tokens per second.

## Known limitations

* **Reproducibility on a GPU.** Replies are reproducible on the CPU backend, the default. With the GPU, a thermal
  emergency moves the work to the CPU, whose kernels round differently from the GPU's, so a reply can change from
  that token on.
* **NPUs** are detected and ranked, but have no compute kernels yet; their work runs on the GPU or CPU.
* **Prefix cache on dense hybrid models.** A prompt processed in 64-token passes leaves the model's oldest state
  snapshot at its first pass, not at the system prompt's end, and contexts set aside keep that one (as saved
  contexts do). So a new conversation gets the system prompt back from the store only where it was processed in
  one pass (a MoE whose experts stream: 1,024-token passes); the app restores it from its own saved state instead.
  Attention-only models restore any prefix.
* **Vulkan:**
  * batched prefill runs the matvec kernel once per token (no tiled GEMM yet);
  * models that fit in RAM are copied into GPU memory;
  * MoE experts always run on the CPU.
* **Metal:**
  * submits one command buffer per matmul (or per group, or per fused FFN) and waits for it, which costs about
    0.25 ms each: small models decode faster on the CPU, and automatic routing sends them there;
  * its Q4_K and Q5_K matvec kernels read 78 and 88 GB/s, where the CPU's read 250 and the Q4_0 kernel 208: a dense
    Q4_K_M model decodes about three times slower on Metal than on the CPU of the same Mac;
  * batches of 16 rows or more use matrix kernels (tiles of 32 tokens, of 64 from 64 rows) that sit at 5–5.6
    TFLOPS for every format, and whose sums are not the matvec kernels' bits (verification batches of fewer rows keep
    the exact ones);
  * the attention of a prompt block (16 rows or more) runs on the GPU when it is the plain case: one causal sequence,
    an F16 or Q8_0 cache, 64, 96, 128 or 256 channels per head, no window, attention sinks, ALiBi, head pruning, tree
    or scattered positions (`Backend::attention`); the layer's rows are copied into shared buffers first, and the
    kernel reaches about 5 TFLOPS (0.26 s of a 9.5 s 1,555-token prompt on the 14B). Other blocks, and every
    decoding step, keep the CPU attention, whose rounding depends on the position only;
  * the GPU waits for the CPU's norms, RoPE and cache writes and for each command's submission between products: it
    is busy about 90% of a 14B prompt (8.8 s of 9.7);
  * the matrix and attention kernels' fragment loops are unrolled by pragma: left rolled, the compiler keeps the
    accumulators in memory and they run 6–8 times slower.
* **Kernels:** TQ1_0 and NVFP4 use a generic decode path on the CPU. Lattice I-quants are 2–3× slower than Q4_K. Only
  Q4_K/Q5_K/Q6_K/Q8_0 have i8mm layouts.
* **Automatic routing** measures the first passes of each class; a model that decodes in a few milliseconds spends
  them mostly on the slower route (three of the first five steps). A prompt block is measured once on the CPU, on
  32 tokens of a chunk, not all 256, which cost a 14B prompt 0.6 s instead of 12. With `cache_dir` the decision is
  kept (`route-*.bin`) when it is clear (the accelerator wins, or the CPU by 1.25× or more) and a later run adopts it
  without the probe passes: the first 1,400-token prompt of the 14B Q5_K_M on the M4 Pro takes 8.8 s instead of 9.8 s.
  Engines that name no backend can give a different text for the same prompt when the routes they pick differ, since
  the accelerator's sums differ from the CPU's in the last bits; the engine tests name the CPU for that reason.
* **x86-64:**
  * the AVX2 kernels have been run only under Rosetta 2 on Apple silicon, not yet on an Intel or AMD CPU, and the
    `x86_64-*` golden hashes come from there (macOS libm); Linux x86-64 builds are not tested, nor is Windows (below);
  * their native speed is not measured. Rosetta's timings do not carry over: it translates `VPSIGNB` (~13 ns) and
    `VPSHUFB` (~12 ns) into long sequences, so there the AVX2 set is 2.3–4.3× faster than the scalar one on Q4_K
    and F32 but 1.5–2.5× slower on Q8_0, Q4_0 and Q6_K, whose scalar loops the compiler vectorizes with SSE2.
    Decoding there (4 threads, best of 3) went from 7.5 to 14.6 tok/s on Qwen3-0.6B BF16 and from 21 to 24 on
    Qwen2.5-0.5B F16, and stayed at 33 on its Q8_0, with the same greedy text;
  * Q2_K, Q3_K, TQ1_0/TQ2_0, NVFP4 and the lattice I-quants (IQ1–IQ3) have no AVX2 kernel and use the generic
    decode path; there are no AVX-VNNI, AVX-512 or repacked layouts, and CPUs without AVX2 run scalar kernels;
  * the transformer's element-wise loops (norms, softmax, RoPE) are plain C++ left to the compiler, and the DeltaNet
    recurrence has a NEON path but no x86 one.
* **Windows:**
  * the [port](building.md#windows-x64) has not been compiled or run on Windows yet: it was written against the Win32 API for
    MSVC and clang-cl and checked only through the macOS builds (arm64, x86-64, and a `-pedantic` x86-64 build that
    finds no GNU language extension in the code Windows compiles), so its first Windows build may still need fixes;
  * `test_golden` compares Windows' results with the `x86_64-*` hashes recorded on macOS. The forward pass calls
    the C runtime's `exp`, `tanh`, `log1p`, `pow`, `sin` and `cos`, which Windows implements with its own code: if a
    result rounds differently there, those hashes need entries of their own;
  * x64 only (Windows on Arm is not ported); no GPU or NPU backend; no thermal readings; the expert cache is never
    locked in RAM;
  * the tests and tools rely on their manifest's UTF-8 code page (Windows 10 1903 and later) for non-ASCII paths
    of their own, such as a temporary directory under a user name with accents.
* **Architecture limits:** registry presence is not a full-model conformance claim. Gemma 4 and DeepSeek-V4 text have
  [synthetic composition gates](../liyab-compat.md), without trained-quality or production-export gates.
  ChatGLM 3, IQuest-Coder, Sarvam-MoE and cross-family stage-2 compositions remain unverified or incomplete; see [the coverage tiers](models.md#architecture-coverage-and-validation). Group routing, Gemma 3n and Phi-2
  are implemented and have the evidence recorded above. Exported tokenizer and chat metadata can still be wrong:
  the engine follows the file, and `tools/describe_model.py --gate` checks it against the source.
* **Latent attention:** the automatic form's threshold is the device's own, so two devices can round a long
  prompt's attention differently (each repeats its own bits run after run). A pass's form also follows its
  shape, so the same token can round differently when a prompt reaches the engine split into other passes
  (typed ahead, after a restored prefix, recomputed by segment reuse, which always attends absorbed);
  `latent_form` pins one form when the bits must not depend on the device or the split (not exposed through
  the C ABI and `liyab-cli` yet). A new measurement takes
  removing its file from `cache_dir`. The expanded form runs on the CPU.
* **YaRN:** a file can state only one of the reference's two mscales (`yarn_log_multiplier`); when it does, they are
  taken as equal, as on every DeepSeek model so far.
* **Windowed blocks (Gemma 2 and 3, Phi-3):** keeping only their window, a context rewinds only inside the
  64-position rollback margin and to its snapshots (the end of each prompt, the oldest one), as a hybrid model
  does; a further rewind is processed again from the nearest of those (or the prefix cache). From the prefix
  cache, a prompt that leaves a stored context inside a page restores to that page's start or to a snapshot
  (at most 63 tokens processed again that a model without windowed blocks would restore). With `segment_reuse`
  on they keep every position, at full attention's memory. The prefix cache is off for a model whose windowed
  blocks have several windows or sit next to recurrent blocks (including the new recurrent/windowed combinations). Window pages are
  stored raw on storage (the lossless chunk coding is not applied to them yet).
* **Gemma 2 and 3:** Gemma 2 27B and Gemma 3 27B scale their
  queries by 1/√(d/heads), which their GGUFs do not record: they are not supported (they run with 1/√head_dim,
  unless the file is given that scale as `attention.scale`).
* **DeltaNet** runs its recurrence token by token on the CPU, also during prefill: no chunked parallel scan or GPU
  kernel yet.
* **Embeddings and reranking:** decoder models only (no encoder-only models). The inputs of a call run one pass
  each: on the M4 Pro a pass's fixed cost is ~3 ms on the CPU (one-word texts: 304 per second on Qwen3-Embedding
  Q8_0, against ~33 ms of work for a 37-token text) and ~35 ms on Metal (29 per second). F16 and BF16 weights
  multiply row by row on the CPU (no tiled float kernel yet), which makes their files several times slower to embed
  with than Q8_0 ones.
* **LoRA adapters:** the update runs on the CPU, also when the base matmuls run on a GPU. Refused: adapters of the
  token embeddings, of latent attention's per-head `attn_k_b` / `attn_v_b` or combined `attn_kv_b`, of shared-KV's
  grouped `attn_output_a`, and activated LoRA. Segment reuse is off
  while an adapter is selected. A draft model drafts without the adapter. On small models the update's fixed cost
  per matmul shows at low rank (+6% per token at rank 8 on a 0.5B model, see [LoRA adapters](inference-features.md#lora-adapters)).
* **Constrained decoding:** object properties are written in the order the schema lists them; `oneOf` is matched as
  `anyOf` (overlapping branches are not told apart); speculative decoding pauses while a grammar is active.
* **Split GGUF:** dense split models use the older `mmap` streaming window. A split model opened through the system
  picker cannot find its other parts.
* **App:**
  * iOS is not built yet;
  * the local API takes text only (no images), and tools only through OpenAI and Anthropic requests;
  * the app keeps one model loaded: to serve embeddings or reranking, the embedding model or the reranker is loaded
    in place of the chat model (the chat then asks for a chat model), and gRPC carries generation only;
  * Android may stop Liyab in the background (which stops the API too). Once Liyab's work is done, HyperOS also
    freezes it there (cgroup freezer): a frozen Liyab runs no code, so an idle model is not released until Liyab
    runs again. The local API escapes this only while its listening service runs. Android starts that service when
    Liyab leaves the screen, and in a process it started for the notification listener (see the local API). A
    process Android started in the background for another reason (a download job) does not listen until Liyab is
    opened, nor does any process when notification access is off or the system does not start the listener's
    process on its own (HyperOS / MIUI: only with Liyab's Autostart allowed, which Settings asks for);
  * pages that need a full activity may not open from the assistant sheet's window;
  * on macOS: Apple silicon only; the app is signed ad hoc, so macOS may ask for its permissions again after a
    rebuild; other apps' notifications cannot be read;
  * Linux and Windows are not built yet.

---

## Project layout

```
include/liyab/          public headers: C++ API, C ABI (liyab_c_api.h), experimental/
src/compat/             liyab-compat: architecture facts, model adaptation, validation and block plans
src/core/               engine, transformer, pooling models
                        (embeddings, reranking), loader, KV cache and
                        its key scaling, expert store, direct I/O, triple buffer, quant kernels, tokenizer
                        (pre-tokenizer table, regex engine, Unicode tables), chat template interpreter,
                        sampler, speculative decoding (draft budget, suffix drafter), the prefix cache's KV
                        store (kv_store.h), segment reuse (kv_reuse.h), LoRA adapters (lora.h), power manager,
                        device detection, thread pool, OS services (os.h:
                        memory, UTF-8 paths, file replacement, file locks; POSIX and Windows)
src/backends/           cpu/ (its NEON and AVX2 kernels are in src/core/quant*), metal/, vulkan/ (+ shaders/),
                        qnn/ and neuropilot/ (runtime probes)
src/experimental/       early exit, head pruning, EGLS, TDSS, JIT unpacker, io_uring loader
src/c_api/              C ABI implementation
tools/                  liyab-cli, liyab-kernels, liyab-bench, liyab-kl, liyab-tok, liyab-experts, liyab-spec-bench, liyab-seq-bench,
                        liyab-embed, liyab-reuse, conform_reference.py, tokenizer_reference.py, chat_template_reference.py
                        and pooling_reference.py (Hugging Face references), describe_model.py and
                        reference_trace.py (registry rows established from the Hugging Face implementation and
                        verified),
                        gguf_vocab.py, gguf_fuse_experts.py, gguf_set_keys.py, expert_study.py (cold-expert study),
                        hf_survey.py (formats and architectures Hugging Face publishes, against ours),
                        table/vector/test-case generators, benchmark and brand image generators
tests/                  self-contained unit tests and benchmarks; device/: the phone regression run's
                        references and log (scripts/phone_regression.py)
app/                    the Liyab app (Flutter): lib/engine (FFI + scheduler), lib/agent (tools),
                        lib/api (local OpenAI / Anthropic / gRPC API), lib/assist (phone sheet, desktop
                        command bar), proto/ (gRPC interface), android/ and macos/ (native runners: shortcut,
                        command-bar window, device and data channels)
docs/benchmarks/        measurements (results.json) and the charts drawn from them
docs/research/          measured studies (cold-experts.md: smaller representations of rarely used experts)
docs/model-descriptions.md  how tools/describe_model.py establishes a model's registry row, and what it said
                        about the supported architectures, new models and unsupported ones
docs/brand/             the Liyab mark, app icon, the README's animated images and the social preview
                        (tools/gen_brand_svgs.py)
scripts/                build_android.sh, build_ios.sh, build_flutter_app.sh, phone_regression.py, thread_cpu.py
cmake/                  embed_spirv.cmake (Vulkan shaders into the library); windows/: the UTF-8 code page
                        manifest of the Windows executables
```

## Contributing

Liyab is meant to be the engine anyone can run on any device, and to stay easy to extend.

* **Ground rules.**
  * Optimizations must be generic and self-calibrating.
  * Lossy changes are opt-in and measured with `liyab-kl`.
  * Speed claims come from alternating A/B runs on a real device.
  * Documentation changes in the same commit as the code it describes: this README, the headers in
    `include/liyab/`, and `--help` texts.
* **Adding a model.** An architecture whose blocks are made of the pieces above (attention, multi-head latent
  attention or Gated DeltaNet, dense or MoE FFN) needs only a row in the registry of `src/compat/architecture.cpp`:
  how RoPE pairs dimensions, the names of the norms before the FFN and after the attention and the FFN, the MoE
  renormalization default, whether Q/K norms come before or after RoPE, the FFN's activation, whether embeddings are
  scaled, which blocks use the sliding window and their RoPE base. `describe_model()` plans each block from the
  tensors present (mixer, separate or fused projections and experts, gated or windowed attention, dense or MoE FFN,
  shared expert) and binds optional tensors where they exist; what the file states (RoPE scaling up to YaRN, soft
  caps, the q·k scale, latent attention's widths) is read from it.
  Add a `test_golden` case and check a BF16/F16 GGUF against its Hugging Face model (`tools/conform_reference.py`,
  then `liyab-kl --compare … --kv f16`). When the weights are too large to fetch, a small random model built by
  transformers' own code for the architecture and converted to GGUF checks the same paths. New kinds of blocks still
  need engine code (see the [roadmap](#roadmap)).
  `tools/describe_model.py --model <id> --gguf <file> --gate` does the description and the conformance run for you:
  it establishes the row from the reference implementation and the GGUF, or lists the parts Liyab lacks, then
  builds a scratch copy with the row and accepts it only if the tokenizer and `liyab-kl` checks pass
  ([docs/model-descriptions.md](../model-descriptions.md)). An `ArchSpec` field added to `architecture.h` needs
  its rule in that tool too: until then it refuses to propose rows.
* **Before a pull request:** `ctest --test-dir build --output-on-failure` on the host, the test suites on an ARM64
  device for kernel changes, and `flutter analyze && flutter test` in `app/` for app changes.

The full guide is in [CONTRIBUTING.md](../../CONTRIBUTING.md). Issues have templates for bugs, features and **model
support requests**; results from devices we have not tested are welcome. Report security problems privately (see
[SECURITY.md](../../SECURITY.md)); everyone follows the [Code of Conduct](../../CODE_OF_CONDUCT.md). If you use Liyab in your
work, [CITATION.cff](../../CITATION.cff) has the citation.

## License

Apache 2.0 (see [`LICENSE`](../../LICENSE)). The low-bit NEON dot kernels are adapted from llama.cpp / ggml (MIT); see
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
