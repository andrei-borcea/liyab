# Memory, streaming and serving

[← README](../../README.md) · [Documentation index](README.md)

Commands and source paths in examples are relative to the repository root.

## On this page

- [Models larger than RAM](#models-larger-than-ram)
- [Context, memory and serving](#context-memory-and-serving)

---

## Models larger than RAM

Liyab streams dense weights when a GGUF exceeds usable memory after a 1 GiB reserve for context, activations and
the host app. `memory_budget_mb` caps usable memory when set. The mmap loader and block pipeline share this
threshold. If Liyab streams blocks, it reserves three slots before choosing resident blocks.

`liyab_engine_weight_layout` reports file size, streamed bytes and streamed block count. Use
`liyab_engine_memory_plan` for MoE experts. On desktop, Liyab checks the loaded layout before raising its cap. It
saves the chat between requests and commits the new cap only after a successful reload.

* **Storage that is fast only when used right.** UFS 4 reaches ~4.4 GB/s with 2–4 direct requests of ≥ 256 KiB in
  flight, random or sequential alike, but stays under 1.2 GB/s with 16 KiB requests (see the
  [flash chart](benchmarks.md#charts)).
  * `DirectFile` reads with `O_DIRECT` (`F_NOCACHE` on Apple, `FILE_FLAG_NO_BUFFERING` on Windows), so there is no
    page-cache copy and resident weights are not evicted. It splits large reads into concurrent 4 MiB requests.
  * On FUSE (Android's `/storage/emulated`), direct reads were seen to return success without the file's bytes, so
    Liyab never uses them there, and it self-tests direct against buffered reads before trusting any file.
* **Dense models.** As many blocks as fit stay resident. The others stream through a 3-slot pipeline (fetch →
  prepare → execute), spread evenly through the stack so storage keeps reading while resident blocks compute. With
  Vulkan, slots and resident blocks live in GPU-shared memory, where the native kernels decode them in place.
* **Mixture-of-experts models.** Everything but the routed experts stays resident. Experts go through the
  `ExpertStore`:
  * **A RAM cache sized by the memory plan.** It holds one entry per expert (gate/up/down) in slot size classes per
    matrix layout. Eviction is LFU with periodic halving. A freshly loaded expert is not evicted before the block it
    was loaded for has used it.
  * **Prediction one block ahead.** After a block's mixer, the next block's router is applied to the residual stream
    and the experts it picks are prefetched, ordered by rank and only while that rank's guesses pay off. On
    Qwen3.6-35B-A3B about 79–83% of guesses are chosen. Wrong guesses are settled as soon as the real router
    chooses: queued ones are dropped, and ones already read become the next to evict.
  * **Reads shaped for flash.** Four I/O threads read the experts. The thread that starts an expert reads its gate
    matrix and hands up and down to idle threads, so a waiting forward pass gets its expert after one matrix's read
    instead of three. A fused `[gate | up]` matrix is contiguous per expert: one read, then down, two in all.
  * **Compute that overlaps I/O.** Experts already in RAM run first, in one batched pass (gate/up, then down). The
    shared expert runs while the routed ones wait for a read.
  * **Batch-aware.** For a multi-token pass (prefill), experts are rearranged into the i8mm layouts on the I/O
    thread as they arrive. Prefill runs in passes of 1,024 tokens, each reading the union of its tokens' experts
    once. Such passes give the experts no LFU credit, so a long prompt does not evict the experts a conversation
    keeps using. A pass of [several sequences](#context-memory-and-serving)' decoding steps, one token each, is
    decoding: their experts are read in the file's layout and credited as a single sequence's are (served as a
    prefill, two sequences on Qwen3-30B-A3B decoded at 0.60x the speed of taking turns; as decoding, 0.96x).
  * **Deterministic.** Each token's expert outputs are added in router order, whatever order they arrived in, and a
    repacked expert rounds like one in the file's layout. Results therefore depend neither on I/O timing nor on what
    the cache held: streamed and in-place experts give bit-identical logits (tested).
* **Memory plan.** `Engine::memory_plan()` reports the resident bytes, the experts' bytes, the expert cache and the
  budget at which 15% of the experts would be cached, so a front end can tell the user when a budget is too small.
  With `requant_bits = -1` (the app's default), resident Q8_0 matrices are converted to Q5_K or Q4_K at load **only**
  when the expert cache would otherwise hold under 10% of the experts. That is the point where nearly every token
  would wait on storage.

**GLM-5.2 laptop sizing.** Its official configuration (`cf457fa734ab149ffef225f80893eb38c6ff5cdc`) has 78 trunk
blocks, 75 MoE blocks, 256 routed experts per MoE block and eight experts selected per token. The trunk contains
724.776 billion routed-expert parameters and 18.600 billion other matrix parameters. These uniform-format
tensor-payload estimates exclude MTP blocks, GGUF metadata/alignment and mixed-quantization choices:

| Matrix format | Trunk file payload (GiB) | Resident weights (GiB) | Expert reads per cold decode token (GiB) | 3% expert cache (GiB) |
| :--- | ---: | ---: | ---: | ---: |
| BF16 / F16 | 1384.65 | 34.65 | 42.19 | 40.50 |
| Q8_0 | 735.60 | 18.41 | 22.41 | 21.52 |
| Q6_K | 567.93 | 14.21 | 17.30 | 16.61 |
| Q5_K | 475.98 | 11.91 | 14.50 | 13.92 |
| Q4_K | 389.44 | 9.75 | 11.87 | 11.39 |
| Q3_K | 297.49 | 7.45 | 9.06 | 8.70 |
| Q2_K | 227.17 | 5.69 | 6.92 | 6.64 |

Cold reads are `75 × 8 × 3 × 6144 × 2048` parameters per decode token; cache hits reduce them, speculative
prefetch can add reads. Main F16 latent keys plus F32 index keys currently reserve 129,792 bytes per position
across all 78 blocks, including shared-indexer blocks: 0.25 GiB at 2,048 positions and 15.84 GiB at 131,072.
Thus Q4_K with a 3% cache starts at about 21.4 GiB for weights/cache/a short context, or 37.0 GiB at 128K,
before activations, allocator overhead and the OS. A 15% Q4_K expert cache alone takes 56.95 GiB. These are
capacity estimates, not measured GLM-5.2 throughput or a full-model conformance result.

A short GLM-4.7-Flash Q4_K_M CPU streaming run on an M4 Pro (four threads, F16 KV, 512-token context) used a
435,191,808-byte expert cache, about 3% of the experts, and answered the capital-of-France prompt correctly.
It generated seven tokens at 4.278 tok/s. Across 18 prompt plus seven generated tokens it recorded 56.45% ready
cache hits, 16,606,912,512 bytes read (633.5 MiB per processed token) and 2,315.656 ms waiting for experts.
These are aggregate prompt/decode counters from one short smoke run, not decode-only bandwidth measurements;
the CLI labels the stall fraction as prefill plus decode. The file declares `deepseek2`; MTP blocks are skipped.

DeepSeek-V2-Lite-Chat Q8_0 also runs through the legacy combined-MLA layout. On the same host, four CPU threads,
F16 KV and no requantization, resident execution and two streamed-cache sizes gave byte-identical replies and
counts (46 prompt tokens, 64 generated tokens). The published 16,702,518,016-byte file has 15,294,529,536 bytes
of routed experts; the cache's minimum slot count raised the requested 256 MiB to 349,741,056 bytes (2.29%).

| Expert cache | Decode (tok/s) | Ready hits | Bytes read per processed token (MiB) | Expert stalls (ms) | Peak RSS (MiB) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Resident | 6.237 | — | 0 | 0 | 11,679.9 |
| 2.29% | 15.071 | 74.48% | 1,135.3 | 904.228 | 2,610.3 |
| 10.47% | 13.794 | 86.55% | 845.7 | 256.222 | 3,954.9 |

Reads and stalls aggregate prompt plus decode; ready hits include predictions that arrived before use.
These are individual observations under shared-host activity, with the file already read and no OS-cache
flush, not controlled speed comparisons or physical flash bandwidth. The smaller cache read 130,950,414,336
bytes and evicted 3,439 loaded experts unused; the larger read 97,550,143,488 bytes and evicted 1,442 unused.
Neither run skipped slow experts. The larger cache reduced read bytes by 25.5% and stalls by 71.7%; no
expert-store correctness failure was observed, and these measurements do not establish full GLM-5 throughput.

* **Optional lossy trade-offs**, all measured with `liyab-kl` (see [results](benchmarks.md#qwen36-35b-a3b-a-22-gb-moe-on-a-phone)):
  * `requant_bits` 4/5: fewer resident bytes (MoE), or fewer bytes read per token (dense models: see
    [the 14B measurement](benchmarks.md#fewer-bytes-per-token-requant-4-on-a-dense-model));
  * `moe_expert_mass`: run only the top experts covering a share of the router weight;
  * `moe_max_experts`: cap the experts per token;
  * `moe_skip_slow`: skip a light expert that is not in RAM yet instead of waiting for it (replies then depend on
    I/O timing).

---

## Context, memory and serving

A phone serves one user, but that user switches conversations, types, leaves the app and comes back. Liyab treats
each of those as a scheduling problem.

* **Context reuse across turns.** When a prompt continues the text the engine already processed (the previous prompt
  plus its reply, which is how a chat grows), that text keeps the tokens it was processed as. Only the rest is
  computed, recurrent DeltaNet states included. Re-encoding a reply need not reproduce the generated tokens (around
  `</think>`, for example), and that no longer forces a restart. The engine keeps knowing which text its tokens
  encode through work that is cancelled or cut short (a prefill typed ahead, a reply stopped while its prompt is
  processed), through rewinds and through `save_state()`. A prompt that leaves that text partway (a draft edited, a
  message prepared ahead then typed over) keeps its tokens up to where the texts part. When that text ends inside a
  character (a model may write one in two byte-level tokens), the tokens step back to the last whole character
  instead of the prompt being encoded again. Otherwise the longest common
  token prefix is kept: any length on attention-only models, and back to the nearest state snapshot on hybrid ones;
  the log says at which token a prompt left the context. On the 35B, follow-up turns start in ~1.1–1.6 s instead of
  reprocessing the conversation.
* **State snapshots for hybrid models.** Four are kept: the oldest one, the end of the last prompt (pinned, so
  background drafts cannot evict it), and the two newest others (a draft's words and the whole draft after them).
  They are set aside from the expert cache's budget (62.8 MiB each on the 35B).
* **Prefix cache across conversations** (`prefix_cache_mb`, `prefix_cache_dir`). A context that another prompt
  replaces (or an embedding call) is set aside in a content-addressed store of KV chunks. A prompt replaces it when
  it shares less than half of it, or leaves it before the end of its last prompt and rewinding would process at
  least 128 tokens again: another chat after a long shared system prompt, an edited message (on a hybrid model,
  rewinding goes back to the nearest state snapshot). A draft typed ahead and edited after the last reply keeps
  using the context's own snapshots instead. Whenever a prompt leaves the context, the store is asked for more of
  it than the context keeps. A chunk is a whole number of pages of the paged KV cache, about 2 MiB and at most
  256 tokens (each is a file on storage, and creating one costs about as much as writing a megabyte), keyed by a
  hash chained over every token before it and rooted in what its bytes depend on (the model file, cache layout and
  type, key scales, kernels and backends, a latent-attention model's prefill form, and the
  [LoRA adapter](inference-features.md#lora-adapters) selected with its scale, if any). A prefix that several
  conversations share (the system prompt, the tool block, the start of a conversation) is therefore stored once,
  and setting a conversation aside again writes only its new chunks. A later prompt is restored up to the longest
  prefix the store holds, chunk by chunk, then cut to where the two diverge, even when its earlier replies would
  encode to other tokens: the store keeps the text each context encodes. The rows the live context already holds
  for the same tokens stay where they are. Recurrent states cannot be cut into chunks, so a hybrid model keeps them
  as images at the positions the prefix cache always kept (the end of the context, the end of its last prompt and
  the oldest snapshot). Each image is attached to a chunk holding its prefix and moves to another such chunk when
  that one goes; a prompt restores up to the deepest one it reaches. Windowed blocks (Gemma 2 and 3, Phi-3) keep
  only their window, so their rows are not in the chunks (which hold the other blocks' rows): they go to the store
  page by page, each window page keyed by the prefix ending at its page's end (or at the context's end or a
  snapshot inside a page), so they are shared between contexts as chunks are. A context set aside keeps the window
  pages its cache still holds (the window before its end and before its snapshots), and a prompt restores to the
  deepest position whose own window page is stored along with every page of the window before it; the snapshots
  below it come back with their windows. The output equals that of a context never interrupted (tested on
  attention-only, hybrid and windowed models, in RAM, on storage and in a later engine).
  * **Tiers and caps.** RAM (`prefix_cache_mb`) holds the most recently used chunks. It comes out of the memory
    budget; on a MoE whose experts stream, only as far as the expert cache can spare above 10% of the experts, so
    it never causes a requantization. What RAM cannot hold goes to storage (`prefix_cache_dir`,
    `prefix_cache_disk_mb`), which costs no RAM at all. What storage cannot hold goes, least recently used first:
    a chunk with every chunk after it, a state image on its own. Room is made before anything is written, so
    neither tier is ever above its cap. A 2,000-token conversation on the 35B costs ~21 MiB of chunks plus
    62.8 MiB per state image.
  * **Across runs.** The storage tier outlives the engine. When destroyed, the engine sets its context aside
    there too, and a later engine or process with the same model and settings restores from it: a system prompt
    or a conversation comes back after a restart without being processed again. Other models' chunks stay and
    count against the cap. Files are written to a temporary name and renamed. When it opens the directory, the
    engine removes interrupted writes, damaged files, other format versions, broken chains and the files of
    builds before the store; a chunk whose payload hash does not match when read is removed and the prompt
    processed. One engine at a time uses a directory (an OS lock); another one runs without the storage tier.
  * **Compressed on storage, only where it pays.** What fills the storage cap is the KV rows, so a chunk can be
    written compressed (`src/core/kv_codec.cpp`) and decoded when read back; RAM keeps rows as they are, and every
    cap counts the bytes written. Measured on real pages of Qwen3-0.6B, Qwen3.5-0.8B and Llama 3.2 1B, KV bytes are
    close to random except the exponents of 16-bit floats, which stay near the largest one their channel takes in
    the page. The lossless encoding therefore stores each 16-bit float (F16 values, Q8_0 block scales) as its low
    byte plus a symbol for its high byte relative to its channel's largest exponent, and codes those symbols with
    our own static rANS coder (12-bit tables per plane, 4 interleaved states, branchless refills). Q8_0 quants are
    coded too, a key channel that changes less from one position to the next than it is large (the slowly turning
    RoPE pairs) as those differences. Pages are encoded one by one and decoded straight into the cache's pages,
    layers in parallel on the engine's threads. With every stream coded (`liyab-kl --store-codec`, 2,047
    positions, every logit identical; decode and encode on the Mac's chunk files, M4 Pro, best of 3):

    | Cache | Stored / raw | Contexts per 512 MB (2,000 tokens) | Decode, 1 thread / 10 | Encode, 10 threads |
    | :--- | ---: | ---: | ---: | ---: |
    | F16 (Qwen3-0.6B, Qwen3.5-0.8B, Llama 3.2 1B) | 1 / 1.168–1.177 | Qwen3-0.6B 2.34 → 2.74 | 0.65–0.89 / 0.11–0.16 ms per MiB | 0.34–0.55 ms per MiB |
    | Q8_0 (same) | 1 / 1.11–1.13 | Qwen3-0.6B 4.41 → 4.90 | 0.83–1.7 / 0.13–0.4 ms per MiB | 0.5–0.8 ms per MiB |

    Restores come first, so a stream is coded only when the read time its smaller size saves exceeds the time
    decoding it takes, on this device: once per storage directory, cache layout and thread count, the store
    writes an 8 MiB probe file next to its chunks, reads it back uncached (`O_DIRECT` / `F_NOCACHE`) and through
    its own buffered, hashed read (a read costs the slower of the two), and decodes a synthetic chunk of the
    layout on the pool. The result is kept in `cache_dir` (`kv-costs-*.bin`, 16 sets at most), so what is coded
    is fixed afterwards; where reads cannot be measured uncached, nothing is coded. A coded stream saves less than
    its own size, so where decoding a symbol costs at least as much as reading a byte nothing can pay: chunks are
    then written raw without trying, and set-asides and restores are exactly those of raw chunks. While a
    compressed chunk decodes, the next one's file is read on a thread of its own (`KvStore::read_ahead()`).
    On the Mac (M4 Pro: reads 8.4–8.8 GB/s, decoding 0.23–0.46 ns per symbol on 10 threads) nothing pays,
    and restores equal the raw store's (alternating runs, 5 each; "coded" forces every stream coded):

    | Context | Raw store | This build (raw chosen) | Coded | Storage, raw → coded |
    | :--- | ---: | ---: | ---: | ---: |
    | Qwen3.5-0.8B Q8_0, 27.9k tokens | 36.3–40.3 ms | 36.3–37.8 ms | 156–192 ms | 213 → 194 MiB |
    | Qwen3.5-0.8B F16, 27.9k tokens | 59.8–61.5 ms | 60.0–62.2 ms | 84.6–86.8 ms | 367 → 320 MiB |
    | Qwen3-0.6B Q8_0, 8.3k tokens | 73.2–75.1 ms | 73.1–75.1 ms | 155–161 ms | 480 → 430 MiB |
    | Qwen3-0.6B F16, 8.3k tokens | 137–141 ms | 134–140 ms | 192–201 ms | 903 → 772 MiB |

    Old raw chunk files stay readable; a build that does not know an encoding removes its files when it opens the
    directory. Recurrent state images are stored raw: their 32-bit floats would code ~13% smaller (Qwen3.5-0.8B),
    which on such storage would not pay either.
  * **Lossy, opt-in** (`prefix_cache_lossy`, `--prefix-cache-lossy`): chunks re-quantized to 4 bits on their way
    to storage. Keys channel by channel, each (layer, KV head, channel) with its own minimum and step over the
    chunk's 64 positions, so the few large, nearly constant key channels of some models keep their own range
    (key scaling, from the chunk itself); values in 32-channel blocks of one position (Q4_1). They come back in
    the cache's type. 3.37× smaller than F16 rows, 1.79× than Q8_0 (Qwen3-0.6B F16: 2.34 → 7.9 contexts of
    2,000 tokens per 512 MB). Cost, `liyab-kl` against the uncompressed restore (every page after the first
    attended to as decoded), 2,047 positions:

    | Model | Cache | KL mean / median | Top-1 agreement | Perplexity |
    | :--- | :--- | ---: | ---: | ---: |
    | Qwen3-0.6B | F16 | 0.0061 / 0.0023 | 96.34% | 14.475 → 14.518 |
    | Qwen3-0.6B | Q8_0 | 0.0063 / 0.0026 | 96.14% | 14.472 → 14.584 |
    | Qwen3.5-0.8B | F16 | 0.0017 / 0.0007 | 97.95% | 10.424 → 10.424 |
    | Qwen3.5-0.8B | Q8_0 | 0.0019 / 0.0007 | 97.85% | 10.428 → 10.414 |
    | Llama 3.2 1B | F16 | 0.0070 / 0.0032 | 95.51% | 13.981 → 14.224 |
    | Llama 3.2 1B | Q8_0 | 0.0078 / 0.0034 | 95.70% | 14.005 → 14.192 |

    A lossy engine keys its chunks apart (its store's seed differs): they are never restored by an engine that
    did not opt in, and a lossy chunk under a lossless engine's own key is removed as untrusted. Q4_0 / Q4_1
    caches are already 4-bit: they are stored lossless.
  * **Measured on the Mac** (8 threads; the app's configuration: storage only, 512 MiB) with a scripted workload: a
    1,392-token system and tool block, seven chats with 160-token replies, four of them resumed (two twice), three
    abandoned, then a restart that resumes the first. Every reply equals that of a context never interrupted, with
    this cache and with the earlier per-context one.
    * Qwen3-0.6B: 91.1% of prompt tokens reused, against 84.4%. The earlier cache set nothing aside (each chat was
      shorter than the system block), so a resumed chat processed its turns again: 1.8–3.8 s to the first token,
      against 0.19–0.25 s. A set-aside took 3–24 ms and wrote only the chat's own chunks, 10–13 MiB (91 MiB for the
      first context), where the earlier cache wrote the whole context each time; a restore took 1–4 ms, the system
      block's rows staying in place. Storage peaked at 208 MiB. After the restart the first chat came back from
      storage: 0.17 s to the first token instead of 14.5 s.
    * Qwen3.5-0.8B (hybrid): 44.5% reused, against nothing (rewinding went back to the start, the nearest state
      snapshot). Every resumed turn was restored: 0.19–0.30 s to the first token against 12–18 s. New chats still
      process the system block, whose end has no state snapshot when it is not prepared on its own. A set-aside
      wrote 1–2 chunks of 4 pages and 2 state images, ~40 MiB, in 9–11 ms; storage peaked at 312 MiB (352 MiB once
      the engine set its context aside when destroyed). After the restart: 0.17 s to the first token instead of
      18.4 s.
    * Throughput, Qwen3.5-0.8B with a 25.8k-token context and three state images (218.6 MiB), alternating runs:
      set aside in 43–49 ms (4.7–5.3 GB/s) and restored in 28–31 ms (7.5–8.3 GB/s), where the earlier cache wrote
      its one file in 23–25 ms and read it back in 41–47 ms. State images are read into the model's states and the
      snapshots' own buffers, and written from them, with no copy in between; each 1 MiB block is hashed while it
      is in cache (xxHash64 runs at ~25 GB/s, reading from the page cache at ~17).
    * Decoding is as fast as before (alternating runs, best of 3: 68.6, 67.9, 62.9 tok/s before; 68.9, 68.6,
      63.3 after).
  * On the phone (Snapdragon 8 Elite, Qwen3.6-35B-A3B, the app's 512 MiB on storage), a chat of 1,355 tokens
    that a local-API request replaced was set aside in 73 ms (2 of its 8 chunks new, its 2 state images: 127.7
    MiB written) and restored in 151 ms when the next message came, which then processed 17 new tokens and
    gave its first token 1.66 s after sending. The same chat set aside again one message later wrote 1 new
    chunk and its 2 state images (126.0 MiB) in 87 ms and came back in 127 ms. Most of what is written is the
    DeltaNet state images (62.8 MiB each), which change with every turn; the KV chunks are written once. The
    store's files outlived an update of the app: the next load kept 9 chunks and 2 contexts. (With the earlier
    per-context cache, a 4,907-token chat (114 MiB) was set aside in 81 ms and restored in 95 ms, where
    processing a 4,009-token one again had taken 163 s.)
* **Segment reuse: KV reused away from its prefix** (`segment_reuse`; opt-in and lossy). An agent's prompts carry
  documents, memory snippets and tool results in the middle, and a change before them forced everything after them
  to be processed again. A request can mark parts of its prompt (`GenerateOptions::segments`,
  `liyab_engine_generate_segmented`, `prefill()` with segments: byte ranges of a system block, tools, documents,
  tool results, turns). With segment reuse on, the KV of each marked part is kept, and a later prompt that holds the
  same part anywhere takes it from there instead of processing it again.
  * **Exact reuse first.** What the live context or the prefix cache holds of a prompt is kept or restored as
    before, losslessly; segments are reused only after it. A request that marks nothing never sees a reused row: a
    context holding some is processed again from the first one, the prefix cache keeps only the rows before it,
    and `save_state()` records where they start.
  * **Kept in the prefix cache's store.** A segment is kept once a prompt marking it has been processed, as a
    context of its own whose tokens sit between two ids no vocabulary has, the first naming the position it was
    computed at. So it gets the store's tiers, caps, LRU order and files, and no prompt's chain ever reaches it
    (its rows are exact only after what preceded it there). It is kept once, as first computed, and reused from
    there wherever it appears. Text a segment holds is encoded on its own, so its tokens are the same wherever
    it appears (with a SentencePiece vocabulary, wherever it follows the same kind of token), and segments
    shorter than 16 tokens are processed as usual.
  * **Turned to its new positions.** RoPE turns each key by an angle proportional to its position, so a stored key
    is turned back by its old angle and on by its new one, with each block's own table: both pairings, partial
    rotation, `rope_freqs`, linear scaling, LongRoPE, YaRN, the windowed blocks' own base, latent attention's RoPE
    key. Key scales and Q/K norms after RoPE are taken out first and put back after; only the quantized blocks
    that hold rotated channels are rounded again, and values, latents and keys moved by 0 keep their bits.
    Measured with `liyab-reuse --shift` (a context's rows moved 300 or 3,000 positions away and back, so rounded
    twice), against the unmoved context: KL 0.0003–0.0004 on Qwen3-0.6B and 0.0006–0.0007 on Llama 3.2 1B with
    a Q8_0 cache, under 0.000001 with F16, 0.02–0.03 with Q4_0. Segments are therefore kept in the cache's own
    format: an F16 copy would take nearly twice the memory of a Q8_0 one to remove a difference two orders of
    magnitude below what reuse itself changes, and an INT4 cache's extra rounding is part of what the
    calibration measures. A move takes ~4 µs per position across all the blocks of a 1B model on the Mac.
  * **Partly processed again.** Reused rows lack attention to what now precedes them. Every reused token runs the
    first two blocks again with the tokens around it; the share whose keys and values moved most there runs
    through the rest of the model, the others keep their reused rows. The whole span from the end of the exact
    prefix to the last reused token goes block by block, in passes in order, so each token sees every position
    before it. The share is not a constant: the engine measures it per model on prompts shaped like real ones,
    the documents of the corpus below in the model's own chat template: three prompts store them, four others
    arrange 3 to 5 of them in other orders, each followed in turn by three questions about them. For each share
    of 5, 10, 15, 20, 30 and 45% it takes the mean KL of the answers against processing the whole prompt, and a
    bound above it (1.645 standard errors of the 12 questions' own means); it keeps the smallest share whose
    bound, and every larger share's, is within `segment_reuse_max_kl` (default 0.03). With none, the model is
    refused. `Engine::segment_reuse()` gives the share and the calibration's KL, bound and top-1,
    `GenerationStats` and `PrefixCacheStats` what was reused.
  * **Calibrated in the background.** Loading does not wait for the measurement: it runs once the engine has
    been idle for 2 s, a stage (a few passes over one prompt's documents) at a time, on a context of its own
    swapped in for the stage, so the conversation's context stays as it was. A call arriving meanwhile stops
    the stage between two blocks, waiting a fraction of a second instead of getting `Busy`, and the stage runs
    again at the next idle time. Until the measurement ends segment reuse is off: marked prompts are processed as
    usual, and `Engine::segment_reuse()` reports `Busy` with its progress. The result is kept in `cache_dir`, so
    the next engine has it at once (without one, every load measures again).
  * **Measured on the Mac** (`liyab-reuse`; CPU, Q8_0 cache with key scaling; on a Mac other jobs loaded heavily,
    so times are compared within each run). 8 prompts per model: the system prompt of a small company's
    assistant, 3–5 of its 12 documents in random order (77–80% of ~555 tokens) and a question, through the
    model's chat template, after warm-up prompts had stored the documents in other orders. (The calibration
    draws on the same corpus, in other arrangements.) Quality against
    processing the whole prompt, at the first answer token and each next one up to the end of the greedy answer
    (at most 33); the first token's time against the lossless path, the part before the first document restored
    from the prefix cache and the rest processed:

    | Model | Processed again | Mean KL | Median KL | Top-1 | Same greedy answer | First token sooner |
    | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
    | Qwen3-0.6B | 0% | 0.049 | 0.0013 | 95.7% | 2 of 8 | 5.5× |
    | | 10% | 0.041 | 0.0012 | 96.2% | 3 of 8 | 3.7× |
    | | 20% | 0.030 | 0.0008 | 95.7% | 1 of 8 | 2.8× |
    | | 30% | 0.026 | 0.0006 | 98.3% | 4 of 8 | 2.5× |
    | Llama 3.2 1B | 0% | 0.027 | 0.0034 | 96.5% | 3 of 8 | 4.2× |
    | | 10% | 0.020 | 0.0025 | 97.2% | 4 of 8 | 3.2× |
    | | 20% | 0.013 | 0.0021 | 98.0% | 4 of 8 | 2.7× |
    | | 30% | 0.011 | 0.0015 | 98.4% | 4 of 8 | 2.2× |

    The first token comes as much sooner as the work left predicts: the selection runs 2 of Qwen3-0.6B's 28
    blocks (2 of Llama's 16) for every reused token, the share processed again all of them. The mean KL is a
    few answer tokens' doing: the median is 6 (Llama) to 48 (Qwen3) times lower, the largest single token's KL
    0.3–1.9. A reused document costs the store its rows, ~60 KiB per token on Qwen3-0.6B and ~17 KiB on Llama
    3.2 1B. Documents kept from prompts that had themselves reused others, as the engine keeps them
    (`--warm-up-reuse`, at 10% on Qwen3-0.6B and 5% on Llama), give the same KL within 0.0006: copies made after
    reused rows add no measurable error of their own.
  * **Which tokens are processed again matters**, at equal work too. Mean KL on Qwen3-0.6B at 10, 20 and 30%:
    the deviation criterion 0.041, 0.030, 0.026; random tokens 0.045, 0.043, 0.039; each document's first ones
    0.052, 0.041, 0.056; ranked by the attention the prompt's last tokens pay them (the question taken into the
    pass) 0.037, 0.038, 0.034, and by that attention times their deviation 0.042, 0.037, 0.026; the deviation
    with each document's first 8 tokens added 0.046 and 0.036 (10 and 20%), compared at block 3 instead of
    block 1 0.039 and 0.037. On Llama 3.2 1B random tokens give 0.025, 0.022, 0.018. Random needs no selection
    pass, yet with it counted the deviation still leads: at 20% it costs 26% of the documents' work on
    Qwen3-0.6B and 30% on Llama, and gives 0.030 and 0.013 where random tokens at 30% give 0.039 and 0.018.
    So the deviation at block 1 is the default (`liyab-reuse --criterion`, `--select-block` and `--first-tokens`
    measure the others).
  * **The calibration against the measurement.** The share the calibration keeps holds on the prompts measured
    above (`liyab-reuse --calibrate 0.03` runs both), or the model is refused:

    | Model | Kept at the default target (0.03) | Calibration: mean KL (bound) | Measured above at that share | First token sooner |
    | :--- | ---: | ---: | ---: | ---: |
    | Llama 3.2 1B | 15% | 0.0219 (0.0280) | 0.0162 | 3.0× |
    | Qwen3-0.6B | refused (45%, the largest) | 0.0364 (0.0543) | 0.0180 | |
    | Qwen3.5-0.8B | refused: hybrid | | | |

    Its prompts ask more than the measured ones. From 5 to 45% Llama's calibration gives 0.035, 0.027, 0.022,
    0.019, 0.019 and 0.012 (bounds 0.045 to 0.016), where the measured prompts give 0.020 at 10% and 0.013 at
    20%. Qwen3-0.6B's stays at 0.036–0.042 whatever the share, carried by a few of its 12 questions (at 45%,
    0.14 and 0.07 for two of them, under 0.022 for seven), while the measured prompts fall to 0.018 at 45%. So
    the share kept errs on the safe side: at a target of 0.02 Llama 3.2 1B kept 45% and got its first token
    1.8× sooner, where 20% (2.7×) would have stayed within 0.02 on the measured prompts; the default is 0.03,
    where it keeps 15% (3.0×) and measures 0.016. Earlier, with plain-text prompts, four
    documents and one or three questions, the calibration was the gentler one (Llama 3.2 1B: 0.0097 at 5%
    against 0.027 at 0% and 0.020 at 10% measured), which is why it now uses the model's chat template, the
    corpus's numbered documents and a bound. It took 205 s of background work on Llama 3.2 1B and 90 s on
    Qwen3-0.6B (BF16 weights on the Mac's CPU), about as much as processing 15,000 tokens, once per model and
    cache format.
  * **Hybrid models are refused**, with the reason in `Engine::segment_reuse()`: their recurrent blocks must run
    over the whole prompt in order, so every token still goes through every block, and reusing the attention
    blocks' keys and values would skip only their key and value projections (1.3% of a prompt token's matrix work
    on Qwen3.5-0.8B). Marked prompts are then processed as usual. Windowed blocks (Gemma) take part like global
    ones, each turned with its own RoPE table, a recomputed token seeing its window: a reused segment needs every
    block's rows wherever it lies, so with `segment_reuse` on the windowed blocks keep every position (full
    attention's KV memory, a shorter automatic context) instead of only their window. With the engine's own
    sliding window there is no prefix cache, so no segment reuse.
  * **Off, nothing changes.** The forward pass, split into the steps this reuse runs block by block at scattered
    positions, gives the same bits (golden logits, arm64 and x86) at the same speed: alternating runs on the Mac
    before and after, a 520-token prompt and 96 tokens, prefill median 5.59 and 5.61 s, decoding 72.2 and 72.1
    tok/s on Qwen3-0.6B, 9.49 and 9.53 s, 42.0 and 43.7 tok/s on Llama 3.2 1B, the same replies.
* **Cancellable, resumable prefill.** `cancel_prefill()` stops a background prefill between blocks, within a fraction
  of a second even inside a 1,024-token pass, and keeps the work done. The next prompt continues from there. A user
  request therefore never waits behind background work.
* **Several sequences decoded together** (`max_sequences`, `Engine::open_sequence()`). The chat, a request from
  another app through the local API and a background agent no longer queue behind each other: each one runs on a
  sequence of its own, with its own context (KV pages as it uses them, recurrent states and snapshots, the text its
  tokens encode, its LoRA adapter), sharing the weights, the threads, the expert cache and the prefix cache. Calls on
  different sequences run at once, each on its caller's thread, and their forward passes run together: whichever
  waiting thread leads takes every sequence's next tokens into one pass
  (`Transformer::forward_batch()`), whose matmuls go over all of their rows, so each weight is read once for all
  of them (a mixture of experts reads the union of the experts they choose), while each sequence attends to its own
  cache at its own positions, continues its own DeltaNet states, applies its own adapter and, with latent attention,
  attends in the form its pass alone would take. The rest stays on each caller's thread: sampling, grammars,
  callbacks, stopping.
  * **Each sequence's output is its own.** A row's matmul gives the same bits whatever rows share its batch, and
    attention depends on a token's own position only, so on the CPU backend every sequence's logits are the bits
    it gets alone: greedy output is token for token, sampling draws the same tokens from the same seed. Tested on
    dense, gated, windowed (Gemma 2/3, Phi-3), hybrid, MoE (in place and streamed), latent (both forms, mixed in one
    pass) and LoRA models (`tests/test_sequences.cpp`). The GPU kernels compute each row on its own too: on Metal,
    replies together equal replies alone (`liyab-seq-bench --backend metal`). With the experimental lossy hooks on,
    sequences take turns instead of sharing passes, so that holds there too.
  * **Alone, nothing changes.** A call that runs while no other sequence has one runs exactly as a single
    conversation always has: the same passes, speculative drafts, pacing per token. Alternating runs on the Mac
    against the engine before sequences (CPU, 8 threads, 128 tokens, medians) decoded at 57.6 and 57.0 tok/s on
    Qwen3-0.6B BF16 (63.7 and 64.2 with drafts), 26.8 and 26.3 on Llama 3.2 1B, 48.3 and 48.1 on Qwen3.5-0.8B,
    238.6 and 235.5 on Qwen2.5-0.5B Q8_0 (11 runs each, prefill 180 ms both), the same replies. While several
    run, their steps are plain (one token each, no drafts): the shared pass is the batch drafts would otherwise
    fill. Passes of a prompt are cut to `max_batch` tokens while others decode, since a pass's time is what their
    next token waits for (latent attention keeps its own pass sizes: each pass's form follows from its size).
  * **What it gains.** `liyab-seq-bench` on the Mac (M4 Pro, 8 threads, 128 tokens per sequence, greedy, medians of
    3 paired runs): every token generated together over the wall time, against the same generations one after
    another. Every reply together was its reply alone.

    | Model | 2 sequences | 3 | 4 | Each of 2, alone |
    |---|---|---|---|---|
    | Qwen2.5-0.5B Q8_0, CPU | 1.37x | 1.75x | 2.28x | 144 tok/s, 287 |
    | Qwen2.5-0.5B Q8_0, Metal | 1.90x | | 3.26x | 44.3 tok/s, 46.5 |
    | Qwen3.5-0.8B BF16 (hybrid), CPU | 1.23x | 1.29x | 1.27x | 23.1 tok/s, 39.2 |
    | Qwen2.5-0.5B F16, CPU | 1.14x | 1.20x | 1.24x | 38.1 tok/s, 69.9 |
    | Qwen3-0.6B BF16, CPU | 1.10x | 1.21x | 1.29x | 30.3 tok/s, 55.3 |
    | Llama 3.2 1B BF16, CPU | 1.11x | 1.18x | 1.18x | 14.3 tok/s, 25.6 |
    | Qwen3-30B-A3B Q4_K_M, experts streamed (2 GiB cache), CPU | 0.96x | | 1.01x | 7.7 tok/s, 17.9 |

    Quantized weights gain most: their kernels decode a block of weights once for four rows. F16 and BF16 weights
    are dotted row by row, so a row more costs almost a row's time ([next](development.md#roadmap)). On the GPU a pass's fixed
    cost dominates and is paid once for all. A MoE whose experts stream from storage reads the union of the
    experts its sequences choose, and with 8 of 128 per token they have few in common: a shared pass (64 tokens
    per sequence there) costs about as much as the passes it replaces, a draw. Each sequence decodes slower than alone, since a pass of more
    rows takes longer: the reason background steps join the user's passes only when that costs little. Drafts or a
    shared pass: with `lookup_drafts` on new text, the same generations in turns went no faster (Qwen2.5 Q8_0:
    222 tok/s with drafts, 223 without), and two sequences together gave 1.42x over the turns with drafts; drafts pay on repeated text
    (`liyab-spec-bench`), so a sequence alone keeps them, and sequences together share plain steps.
  * **The user first** (`SequencePriority`). Foreground sequences are the user's turn; background ones yield: their
    passes run when no foreground request waits, a pass of a background prompt is stopped between two blocks when
    foreground work arrives (its work kept, the pass run again later), and a background decoding step joins a
    foreground pass only while a pass of one more sequence measured at most 1.25 times as long: on a model in RAM
    a row more costs little, while a MoE whose experts stream from storage would make the user's step wait for the
    background step's expert reads. One step in 64 that would be kept out joins anyway, so the estimate follows
    the expert cache warming up and the device's load (at most 1/64 of the difference on the user's steps). A leader waits for the sequences about to submit (sampling, calling back) for
    at most half a pass, so a slow callback costs the others little.
  * **Per sequence.** Cancellation (`Sequence::cancel()`, `cancel_prefill()`), adapters (`set_adapter()`), the
    prefix cache's restores and set-asides (a context one sequence set aside, any other restores), and the Busy
    rule: one call at a time per sequence. `embed()`, `rerank()`, `save_state()`, `load_state()` and segment reuse
    stay on the default sequence; the others' passes wait while they run.
  * **Memory, only while a sequence is open.** `max_sequences` is a ceiling and reserves nothing: the model loads
    with the memory plan of a single conversation whatever it allows (the same expert cache, the same decision to
    requantize, the same bits out), so an engine with no sequence open is the engine of one sequence. Each
    sequence keeps its own KV pages as it uses them, up to `context_length()` positions. On hybrid models it also
    holds its recurrent states, rollback checkpoints and up to four snapshots (62.8 MiB of state each on the 35B,
    five of them): with streamed experts, `open_sequence()` takes that much out of the expert cache before
    allocating it (slots at the end of each size class, their experts evicted, their pages returned to the OS) and
    closing the sequence gives it back. Reserving it at load for every sequence allowed shrank the 35B's cache from
    2286 to about 1970 MiB with the app's API on, below the line where the resident Q8_0 matrices are requantized
    to 4 bits: other replies, 15.9 s to load instead of 3.7 s, an 11% higher peak, for a sequence that was never
    opened. A cache that cannot spare a sequence's memory without going below its minimum (a block's top-k and the
    reads in flight) refuses the open (`OutOfMemory`) instead of overcommitting; nothing is requantized after load.
    Not with a draft model, whose cache follows one sequence.
* **Automatic context length.** With `context_length = 0`, the context is what 256 MiB of KV holds: 4,096 to 32,768
  positions, at most the training context. That is ~24k on Qwen3.6-35B-A3B (only 10 of 40 blocks have a KV cache)
  and 4,096 on a dense 4B. Windowed blocks count for what they keep: their window plus the rollback margin is set
  aside once, and the rest of the budget goes to the blocks that keep every position, at their own cost per
  position ([Per-block KV](benchmarks.md#per-block-kv)).
* **Save and restore.** `save_state()` / `load_state()` write the context to a file (KV pages, recurrent states, and
  the tokens and text they encode) and restore it in a later process. This works only for the same model file and
  cache layout, which a fingerprint checks. A long system prompt is processed once, not after every start. On hybrid
  models the file also keeps two state snapshots, the oldest and the last prompt's end (62.8 MiB each on the 35B), so
  a restored context rewinds where the live one would. The next prompt re-sends the last reply as text, which may
  encode differently from the tokens generated; without that snapshot the whole conversation was processed again
  (a 1,436-token chat: 60 s to the first token). The file also records the key scales its KV pages were stored with;
  a context saved with other scales, or before key scaling existed, has its keys converted when loaded, so an
  upgrade does not process saved conversations again. With windowed blocks (format version 4) the pages hold the
  blocks that keep every position, and the windowed blocks' rows of the window before the end (and before each
  kept snapshot) follow; the file records the per-block layout, and a file of another layout (another context
  length, `segment_reuse` changed) is refused with that reason. Files written before the per-block layout (every
  block's rows of every position in one table) load converted: the windowed blocks keep the window before the end.
* **Giving memory back.** `trim_memory()` empties the expert cache while idle, and drops constrained decoding's
  caches (rebuilt on the next constrained generation); the model and its context stay loaded. Its hot list survives: `warm_memory()` refills the cache in one background pass of large reads (~0.7 s for
  2.4 GB), and `save_expert_profile()` / `load_expert_profile()` carry the list to a later process.
* **Structured output.** `generate()` takes an optional `ForceCallback`. After each piece it sees the output so far
  and may return text the output must continue with. The engine runs that text through the model in one batched
  pass, like a short prefill, and emits it as output. That pass also gives the logits of the token after the forced
  text, which is sampled from them, so forced text costs no decoding step of its own (the CPU backend computes a row
  the same bits in any batch, so the output is the same as with a separate step). Constrained decoding (below)
  forces a grammar's fixed text through the same pass: the syntax and names of tool calls. It works with speculative
  decoding too: the callback is asked after every token, and tokens a step verified past a forced point are rolled
  back. Forced text that ends inside a character is emitted up to it, the rest with the token that completes it.
  Forced tokens are counted apart from sampled ones, with the time their passes took, and
  `GenerationStats::decode_passes` counts the model's passes while decoding. On the 35B with streamed experts, a tool call's 9 forced tokens took 1.52 s against ~1.66 s to
  sample them: a batched pass reads the union of its tokens' experts, so what forcing buys there is a call that
  always parses, not speed.
* **Constrained decoding.** `generate()` also takes `GenerateOptions`: a grammar the whole output must match, or a
  `ContinuationCallback` that forces text and opens constrained spans where it chooses (a tool call's arguments,
  between forced syntax). While a grammar is active, every token whose bytes it cannot accept is masked before
  sampling, so the output always matches. Grammars come from a JSON Schema (the subset tool parameters and response
  formats use: objects, required and optional properties, enums, `$ref`, integer and number ranges, string lengths
  and formats; see `include/liyab/grammar.h`), any JSON, or grammar text in a GBNF-like syntax with `%json`,
  `%schema`, `%string` and `%until`. A schema keyword Liyab cannot enforce (`pattern`, `uniqueItems`...) is refused
  with an error naming it and where it is, never ignored.
  * **How.** Liyab's own grammar engine lowers rules to a byte-level pushdown automaton: UTF-8 classes become byte
    sequences, so a token may end inside a character, and a matched text is always valid UTF-8. The set of parse
    stacks the output allows is interned as a state, and byte transitions between states are cached: inside a
    string a transition is one table lookup. The vocabulary is a trie in depth-first order; a state's mask is one
    pass over it that skips every subtree whose prefix the state rejects, and is cached per state. Greedy and top-k
    sampling skip the mask altogether: the likeliest tokens are checked one by one until the k best allowed ones
    are found, which leaves exactly what masking then keeping the top k would.
  * **Forced, not sampled.** Text every continuation shares (keys, separators, brackets, the rest of the only enum
    value that fits) is forced in one batched pass, as forced continuations are. JSON is laid out as `json.dumps`
    writes it (`", "` and `": "`) by default, so the model samples only values and choices; any white space is an
    option. End-of-generation tokens are allowed only where the grammar accepts, and a whole-output grammar ends the
    generation as soon as its match is complete.
  * **Token healing.** Forced text ends where the grammar offers a choice, and its last token can be the start of a
    longer token the model writes there: Qwen writes ` {"` after `"arguments":` and `</` before `function>`. Forced
    alone, ` {` or `<` makes the model go on from a token boundary it never saw in training, which skews its choice.
    Through the app's tool calls, Qwen3-0.6B then mostly wrote `{}` and left out its optional arguments (a date in
    10 of 50 replies, against 30 when it wrote that token itself), and Qwen3.5-0.8B added a parameter it did not
    need. So when a longer token the grammar allows starts with the forced text's last token, that token is left
    to the model, which samples it or the longer one under the mask; the batched pass only gets shorter, so it
    costs no step. With it, Qwen3-0.6B gave a date in 30 of 50 replies again, and Qwen3.5-0.8B's greedy calls are
    the ones it writes when only their syntax and names are forced.
  * **Cost.** On an M4 Pro (Qwen3-0.6B, Qwen3.5-0.8B, Llama 3.2 1B, Qwen2.5-0.5B), constrained decoding cost
    0.14–0.24 ms per token greedy, 0.25–0.62 ms with top-k sampling, and 0.09–0.12 ms with whole masks once a
    state's mask is cached, against decode steps of 18–92 ms. A state's first whole mask takes 1.5–2.7 ms (inside a
    string, where nearly every token is allowed). The vocabulary trie is built on the first constrained token:
    333,000 nodes in ~30 ms for Qwen3's 151,000 tokens, 592,000 nodes in ~45 ms for Qwen3.5's 248,000. The engine
    keeps the states and masks of the 16 grammars used last (32 MiB per grammar, 64 MiB together), so a request with
    the same tools does not pay for them again. `GenerationStats::constrained_tokens` and `grammar_ms` report the
    cost of each generation, and `test_grammar` measures it on any GGUF (`LIYAB_GRAMMAR_MODEL`), together with a
    tool call made the way the local API makes one: Qwen3-0.6B (JSON calls) and Qwen3.5-0.8B (XML calls) wrote
    valid calls in every run.
  * **Speculative decoding** is off while a grammar is active (drafts would need a mask at every verified position);
    the unconstrained parts of the same generation use it.

---
