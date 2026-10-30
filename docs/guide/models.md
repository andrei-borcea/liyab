# Models, formats and reference tools

[← README](../../README.md) · [Documentation index](README.md)

Commands and source paths in examples are relative to the repository root.

## On this page

- [Supported models](#supported-models)
- [Architecture coverage and validation](#architecture-coverage-and-validation)
- [Reference tooling and additional model evidence](#reference-tooling-and-additional-model-evidence)
- [Gemma text and embedding details](#gemma-text-and-embedding-details)

---

## Supported models

### Architecture coverage and validation

The registry describes executable parts; a row alone does not prove every checkpoint in a family. Tiny gates use
F32 weights, F16 KV, identical prompt IDs, mean KL ≤ 0.0001 and top-1 ≥ 99%. Trained-model gates use mean KL ≤ 0.01
and top-1 ≥ 97%, plus source-tokenizer and template checks. Smoke replies and component tests are labelled separately.

| Family / GGUF architecture | Evidence and scope |
| :--- | :--- |
| Existing attention and hybrids: `llama`, `mistral`, `qwen2`, `qwen3`, `qwen35`, `qwen3moe`, `qwen35moe`, `hunyuan-dense`, `ernie4_5`, `gemma2`, `gemma3`, `phi3`, `deepseek2` | Trained-model and synthetic regression evidence in [Correctness](benchmarks.md#correctness); family/quantization limits still apply |
| Classic: `gpt2`, `gptneox`, `falcon`, `bloom`, `mpt`, `starcoder`, `starcoder2`, `phi2`, `stablelm`, `orion`, `olmo`, `command-r`, `nemotron`, `apertus`, `arcee`, `dbrx` | 20 tiny cases across 16 families; nine trained-checkpoint gates. Phi-2/StarCoder2 failed Q8 controls remain disclosed; F16 passes |
| Modern: `smollm3`, `olmo2` (OLMo 2/3), `exaone`, `exaone4`, `granite`, `minicpm`, `internlm2`, `gemma`, `mistral3`, `plamo3`, `seed_oss`, `glm4`, `deci` | 17 tiny recipes and ten trained gates across the section. Some source/export pairs require explicit metadata corrections; `seed_oss`, `glm4` and `deci` have tiny evidence |
| MoE: `gpt-oss`, Mixtral (`llama`), `qwen2moe`, `deepseek`, `glm4moe`, `bailingmoe`, `dots1`, `ernie4_5-moe`, `hy_v3` | Nine passing tiny recipes; GPT-OSS-20B MXFP4 has resident/streamed real-model smoke and timing evidence, rather than a trained numerical gate |
| State space: `mamba`, `mamba2` | Five tiny cases including Falcon-Mamba, grouped Mamba-2 and Codestral; trained Mamba-130M/370M and Mamba-2-130M gates |
| Convolution/delta: `lfm2`, `lfm2moe`, `qwen3next`, `kimi-linear` | Four tiny gates; trained LFM2-350M/700M/1.2B gates. Large MoE/linear checkpoints are not full-weight numerical validations |
| Gemma text/pooling: `gemma3n`, `gemma-embedding` | Gemma 3n E2B trained BF16 and tiny gates; EmbeddingGemma-300M checked on 39 inputs against independent pooling/sentence-transformers references |
| `gemma4` | Stage-1 uniform fixtures plus heterogeneous text and parallel MoE checkpoints pass; see [liyab-compat](../liyab-compat.md). Trained quality and multimodal inputs remain unverified |
| Sparse/grouped: `deepseek2` (V2/V3), `glm-dsa` (GLM-5/5.1/5.2), `deepseek32`, `bailingmoe2` | Canonical tiny gates, including genuinely sparse selections. V2-Lite has real streaming evidence; GLM-4.7-Flash has smoke evidence but fails the standard numerical gate |
| `deepseek4` | Full text composition passes a generated upstream checkpoint: hash/top-k MoE, clipped SwiGLU, sinks, CSA/HCA and mHC; [scope and limits](../liyab-compat.md). Production exports and trained quality remain unverified |
| `chatglm`, other proposed architectures | Partial/component coverage or failed conformance; registry entries and diagnostic successes do not establish model support |

InternLM 3 exports as `llama`; Nemotron-NAS as `deci`. MiniCPM3 and architectures absent from this table are not
implicitly covered by a related name. The detailed family tables, pinned revisions, failed controls and converter
compatibility notes are in [model descriptions](../model-descriptions.md) and the
[stage-1 validation report](../model-coverage-stage1.md).

CPU is the numerical reference. LayerNorm, activations, router/sink operations, AltUp/Laurel and recurrent scans
run on CPU; projections retain the existing CPU/Metal/Vulkan routing. Mamba scans, ShortConv/KDA and MXFP4 have
NEON, AVX2/FMA and scalar paths where described. BF16 matrix operations on the Metal route use CPU fallback;
F16 measurements identify actual Metal execution. Android compilation does not prove Vulkan runtime numerics.

GGUF metadata is part of correctness: OLMo's `dbrx`, tiny StarCoder's `refact`, EXAONE 3.5's normalizer, InternLM 3's
segmentation and Ministral's exported chat template have documented source/export conflicts. Correct copies
explicitly and retain source/tensor identities; a model-name heuristic never overrides the file silently.

Engine/CLI template rendering is broader than Flutter chat replay. Flutter uses a prefix-stable author template
or a recognized ChatML/Llama 3 fallback; an incompatible author template refuses chat precisely while raw
completion and pooling remain available. GPT-OSS Harmony currently has this Flutter replay limitation.

* **Format:** GGUF v2/v3, a single file or split with `gguf-split`. Open part 1; the other parts must sit next to it
  under their original names.
* **Architectures:**
  * `llama` (Llama 1/2/3.x, Mistral, TinyLlama…), `mistral`, `qwen2`, `qwen3`;
  * `qwen35`: Qwen3.5 / Qwen3.8 hybrids, with three Gated DeltaNet blocks per gated-attention block. The
    multi-token-prediction head is skipped;
  * mixture-of-experts: `qwen3moe` (Qwen3-30B-A3B…) and `qwen35moe` (Qwen3.5/3.6-35B-A3B…);
  * `hunyuan-dense` (Hunyuan-MT-7B, Hunyuan 0.5B–7B): Q/K norms after RoPE. Its dynamic-NTK "alpha" RoPE scaling
    is already folded into `rope.freq_base` by the GGUF converter. Its `hunyuan` / `hunyuan-dense` pre-tokenizers
    are table rows (below). A prompt must start with its BOS token, which the GGUF does not ask the tokenizer to
    add;
  * `gemma2` (Gemma 2): embeddings scaled by √d, GeGLU, norms after the attention and the FFN, every other block
    (the even ones) through the sliding window, and soft caps on the attention scores and the logits
    (cap · tanh(x / cap), from the file's `attn_logit_softcapping` and `final_logit_softcapping`; tanh is the
    engine's own (`core/portable_math.h`, within 1.32 ulp), because the C library's differs between Apple and
    Android in the last bits and the Mac and the phone gave Gemma 2 different logits; the logits' cap runs in
    chunks on the thread pool, the same bits for any
    thread count: 0.34 ms per token on one thread, 0.05 ms on eight, over a 256k vocabulary on an M-series Mac);
  * `gemma3` (Gemma 3 text models): embeddings scaled by √d, GeGLU, norms after the attention and the FFN, Q/K
    norms, five sliding-window blocks (RoPE base 10,000) per global block (the file's base and scaling). The GGUF
    stores its RMSNorm weights as 1 + w already, for Gemma 2 too;
  * `phi3` (Phi-3, Phi-3.5, Phi-4-mini): fused `attn_qkv` and `[gate | up]` projections, partial rotation, LongRoPE;
  * `smollm3` (SmolLM3-3B): every fourth block leaves Q/K unrotated; its `smaug-bpe` tokenizer and own chat
    template match transformers. The real F16 checkpoint passes `describe_model.py --gate` with identical
    498 prompt IDs, mean KL 0.00000 and 100% top-1 agreement; projections use the existing CPU/Metal/Vulkan routes;
  * `plamo3` (PLaMo-3-610m-fin-instruct): fused QKV and gate/up projections, Q/K and sandwich RMSNorms,
    seven local RoPE/windowed blocks per global YaRN block, raw unigram tokenizer. The real F16 checkpoint
    passes the gate with identical 512 prompt IDs, mean KL 0.00000 and 100% top-1. Norms, RoPE and tokenization
    run on the CPU; projections retain CPU/Metal/Vulkan routing;
  * `olmo2` (OLMo-2-0425-1B): RMSNorm after each residual branch, Q/K RMSNorm over the entire projection,
    and ordinary-merge Llama splitting. Its earlier real F16 gate passes with identical 512 prompt IDs,
    mean KL 0.00000, maximum 0.00003 and 100% top-1. Norms run on the CPU; projections retain CPU/Metal/Vulkan routing;
  * `exaone4` (EXAONE-4.0-1.2B): post-only branch norms, per-head Q/K RMSNorm and Llama 3 rotary frequency factors.
    Its real F16 checkpoint passes the gate with identical 512 prompt IDs, mean KL 0.00001, maximum 0.00018
    and 100% top-1. A separate four-block tiny case checks windowed rotary/global unrotated attention at a 64-token
    window (99.80% top-1). Norms and rotation run on the CPU; projections retain CPU/Metal/Vulkan routing;
  * `ernie4_5` (ERNIE 4.5 dense: ERNIE-4.5-0.3B): a pre-norm block with SwiGLU, RoPE on adjacent pairs; the row was
    proposed and gated by `tools/describe_model.py` ([docs/model-descriptions.md](../model-descriptions.md)).
    The published GGUFs (unsloth, bartowski and mradermacher were checked) ask for a BOS token
    (`tokenizer.ggml.add_bos_token` true, or absent, which for a SentencePiece vocabulary means true) and, by
    leaving `tokenizer.ggml.add_space_prefix` out, for a space before the text, while the model's own tokenizer
    adds neither. Liyab reads the file as written, so such a file encodes text unlike the model's tokenizer from the
    first token on (the forward pass itself matches the reference). Fix a copy once; the tensors are copied as
    they are:

    ```bash
    tools/gguf_set_keys.py ERNIE-4.5-0.3B-PT-F16.gguf ERNIE-4.5-0.3B-PT-F16-fixed.gguf \
        --bool tokenizer.ggml.add_bos_token=false --bool tokenizer.ggml.add_space_prefix=false
    ```
  * `deepseek2` (DeepSeek-V2-Lite, DeepSeek-V3-structured models with group-limited expert routing, and the models converted to
    their architecture, such as Youtu-LLM-2B): multi-head latent attention, queries from one projection or a
    low-rank pair with its norm, dense FFNs in the leading blocks and MoE with shared experts in the others (from
    the tensors), YaRN with its log multiplier. The cache holds one row per position and block, `[k_pe | latent]`
    (576 values on DeepSeek-V2-Lite instead of the 5,120 of per-head keys and values), whose latent channels are the
    values. Attention reads it in one of two forms, chosen per forward pass:
    * absorbed (always for decoding): each head's query is projected onto the latent once (`attn_k_b`), so all
      heads attend over the cached rows themselves; the weighted latent is then expanded per head (`attn_v_b`). Per
      position and head that scores 576 channels and accumulates 512;
    * expanded (prefill batches): per-head keys `[W_kb c | k_pe]` and values `W_vb c` are expanded from the cached
      latents of the positions the batch sees, a tile of at most 256 positions at a time (5 MiB on
      DeepSeek-V2-Lite, plus 8 MiB of the decoded `attn_k_b` and `attn_v_b` of the block being run; never the
      whole context: the cache keeps latents only), and the batch attends over them: 192 channels scored and 128
      accumulated per position and head, for the cost of expanding each position once per pass instead of
      projecting each token's query. Tiles have absolute bounds and are merged in order, so a token's result is the
      same however its prompt was split into passes and however many threads run.

    The form is chosen from measured cost, measured once: each form's work is counted per pass from its batch size
    and context, and the ratio of what a multiply-add costs in each form on this device sets the threshold, so it is
    the device's own instead of a constant. The engine times both forms of one block on a synthetic pass at load
    (256 tokens after 768 cached positions, best of 3 each; 0.18 s on Youtu-LLM-2B on the Mac, ratio 0.595), rounds
    the ratio to 1/8 octave and keeps it in `EngineConfig::cache_dir` next to the key scales (measured again when
    the file is missing or damaged; without a `cache_dir`, kept in memory for the process only, and two processes
    may measure a step apart: 0.595 and 0.648 in three runs there). It is never re-measured on
    the fly: from then on a pass's form follows from its shape alone, so the same prompt gives the same bits on that
    device in every run, whatever else loads it. Another device (or another kernel set) may set another threshold
    and round a long prompt differently. The engine prefills these models in passes of 512 tokens (64 for other models, 1,024 with streamed
    experts): a pass of 64 tokens deep in a context does not repay expanding every position it sees, and stays
    absorbed. Both forms compute the same function and round differently: outputs are not bit-identical across
    forms (each form on its own is, and `test_golden` pins both); against transformers both give a mean KL of
    0.00000 with an F16 cache, and the expanded form is the closer with a Q8_0 one ([Correctness](benchmarks.md#correctness)).
    `TransformerOptions::latent_form` / `EngineConfig::latent_form` pin one form. Queries are scaled for the
    per-head width the model was trained with (1/√192 there). Key scaling covers the latent too: the absorbed form
    multiplies the attention output back, the expanded one the latents before it expands them. Files converted
    before `attn_kv_b` was split into `attn_k_b` and `attn_v_b` are normalized at block acquisition: keys are decoded
    and transposed to F32, value rows retain their quantization, with one block of temporary memory.
    Group-limited routing scores groups by their maximum probability (V2), or by their two best biased
    probabilities (V3/R1), then selects experts inside those groups. Weights use unbiased probabilities,
    the declared renormalization and routed scale, identically for resident and streamed experts.
  * `gemma3n` (Gemma 3n text): AltUp residual streams, Laurel, per-layer token embeddings, Gaussian gate
    sparsity, normalized values and shared KV. The per-layer table is mapped and read by token row; only its
    current batch is materialized. The loader may warm the mapping, and the memory plan conservatively counts
    the whole file. Projections use the existing CPU kernels and GPU operation routing; AltUp's small router
    and corrections run on CPU.
  * `gemma-embedding` (EmbeddingGemma 300M): bidirectional attention, mean pooling and chained dense projections
    before L2 normalization. Each complete input runs alone through attention, including in a batch of inputs.
  * `bailingmoe2` (Ling 2.0): fused contiguous Q/K/V projections, per-head Q/K RMS norms before partial halves
    RoPE, dense leading FFNs and grouped sigmoid MoE with selection bias, normalized routed weights and shared
    experts. These reuse the generic attention and MoE paths; CPU is the reference and projections retain the
    existing Metal/Vulkan routing. Its NFC BPE pre-tokenizer preserves single digits and possessive symbol runs.

  Linear RoPE scaling, Llama 3.x `rope_freqs`, LongRoPE and YaRN are supported, and windowed blocks can rotate with
  a RoPE base of their own. LongRoPE picks its factors once, from the configured context: the short ones up to the
  model's original context (4,096 on Phi-3.5), where the result is the reference's, and the long ones beyond it from
  the first position on (the reference switches when the sequence grows past it, rotating earlier keys again).
  YaRN is read from the file's `rope.scaling.*` keys (factor, original context, `yarn_beta_fast` / `yarn_beta_slow`,
  `yarn_ext_factor`, `yarn_truncate`, `yarn_attn_factor`, `yarn_log_multiplier`): each rotary pair's frequency is
  blended between its own and the one divided by the factor along the reference's ramp, and queries and keys get
  its attention factor (0.1 ln factor + 1 unless the file states one). It does not depend on the context length,
  so one table serves every position. The q·k scale is the file's `attention.scale` when it has one, else 1/√head_dim.
  `liyab_supported_architectures()` lists what a build runs, so front ends can filter downloads.
* **Embedding models and rerankers** built on these architectures (Qwen3-Embedding, Qwen3-Reranker) run as their
  GGUF declares them: see [Embeddings and reranking](inference-features.md#embeddings-and-reranking). Encoder-only models (BERT, XLM-R)
  are not supported.
* **MoE details:**
  * softmax or sigmoid gating, optional selection bias, top-k, renormalization and scale from the metadata;
  * an optional shared expert with a sigmoid gate;
  * each expert's gate and up projections as two tensors or as one `[gate | up]` tensor (`ffn_gate_up_exps`,
    written by converters that fuse them); the fused rows are the split ones, so the logits are the same bits;
  * the tokens of a batch are grouped per expert, so each expert is read once.
* **State-space memory:** Mamba blocks keep convolution history and `inner_size × state_size` floats per block,
  independent of sequence length. Their state uses the existing recurrent snapshots, draft rollback, sequence
  contexts and prefix/KV-store persistence. Pure SSM stacks reserve a minimal, unused paged-KV placeholder.
* **Hybrid models:** recurrent blocks keep convolution history and, for DeltaNet and KDA, a matrix per value head
  sized from their metadata (128×128 on Qwen3.5 and Kimi-Linear). ShortConv keeps only its convolution history.
  These models can roll back only:
  * inside a window of state checkpoints (used by speculative decoding), or
  * to one of four state snapshots (the oldest, the end of the last prompt, and the two newest others).
* **Tensor types:** every format llama.cpp writes, mixed freely per tensor:
  * F32, F16, BF16;
  * Q4_0/Q4_1/Q5_0/Q5_1/Q8_0;
  * Q2_K…Q6_K (and the S/M/L mixes);
  * IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS;
  * Q1_0 (one sign bit per weight, 128-value blocks), TQ1_0/TQ2_0, MXFP4, NVFP4.

  Decoders are ported from llama.cpp's gguf-py reference, and `test_engine` checks the encoded quantized formats bit-exactly against
  `tests/data/quant_vectors.bin` (for Q1_0, which gguf-py cannot decode yet, the expected values come from llama.cpp's own
  libggml: `tools/quant_reference.py`). Repeated SSM projections use a separate
  `ssm_quant_vectors.bin` with uniformly conditioned scale fields; regenerate it with
  `python tools/gen_ssm_quant_vectors.py --out tests/data/ssm_quant_vectors.bin` (gguf 0.19.0, NumPy).
  The test writer refuses reader-only formats unless explicit encoded blocks are supplied.
* **Kernels:**
  * Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 have NEON + SDOT kernels that unpack bits in registers.
  * K-quants, TQ2_0 and I-quants use NEON kernels adapted from llama.cpp (`src/core/quant_lowbit.cpp`) against
    Q8_K activations, so a whole super-block accumulates in int32.
  * IQ4_NL and MXFP4 use table lookups.
  * Q1_0 expands each byte of sign bits to +1/−1 in registers (NEON table lookup, AVX2 shuffle) and multiplies them by
    the activations with the dot-product instructions.
  * With i8mm, resident Q4_K/Q5_K/Q6_K/Q8_0 matrices are rearranged losslessly at load into 8-row (4-row for Q8_0)
    interleaved layouts. One SMMLA then multiplies 8 weight rows by 4 activation rows.
  * Every kernel of a format ends each block the same way: exact integer sums, then the same fused multiply-adds in
    the same order. That holds on the file's layout or repacked, for one activation row or several. A row's result
    is then the same bits whichever kernel computed it, so logits depend neither on the batch size nor on the layout
    (tested for every format).
* **Tokenizers:** SentencePiece (`llama`: Llama 2, Mistral, Gemma, Phi-3), byte-level BPE (`gpt2`) and raw unigram
  (`plamo3`: integer-cost segmentation, longest-prefix ties and byte fallback).
  * **Pre-tokenizers are data.** Before merging, a byte-level BPE tokenizer cuts the text into pieces with regular
    expressions. Each scheme is one row of a table keyed by the GGUF's `tokenizer.ggml.pre`
    (`src/core/pre_tokenizer.cpp`): the patterns of the model's own `tokenizer.json`, NFC normalization where it
    asks for it (Qwen), and whole-piece vocabulary lookup before merging (Llama 3, Hunyuan, Tekken). Rows:
    `default` / `gpt-2` / `gpt2` / `phi-2`, `llama-bpe` / `llama3`, `qwen2` / `deepseek-r1-qwen`, `qwen35`,
    `bloom`, `starcoder` (individual digits), `refact` (plain GPT-2), `olmo` (NFC), `stablelm2`, `falcon`, `command-r` (NFC), `dbrx` (cl100k),
    `hunyuan`, `hunyuan-dense` / `deepseek-v3` / `joyai-llm`, `deepseek-llm`, `gpt-4o` (Phi-4-mini), `smollm` (SmolLM2: every
    digit alone, then GPT-2's split), `tekken` (Mistral-Nemo) and `youtu` (Youtu-LLM: Hangul, CJK punctuation,
    Bopomofo and Han/kana runs first; its `\p{Hangul}` is written out as the script's code-point ranges, since the
    regex engine has categories, not scripts). A GGUF without the key uses `default`. Any other value is an error
    naming it and the supported list; the model still loads and runs on token ids, with a quantized KV cache too
    (its keys then unscaled, see [key scaling](architecture.md#architecture)).
    Additional rows are `smaug-bpe`, `dbrx`, `refact`, `seed-coder`, `exaone4`, `glm4` and
    `gpt2-nfkc-digits` (Unicode 9 NFKC, individual numbers, then GPT-2 ByteLevel). The latter requires the
    corrected source-pipeline tag described in Correctness; EXAONE 4 keeps plain ByteLevel.
  * **Liyab's own regex engine** (`src/core/regex.cpp`) runs the patterns: Unicode general categories (`\p{L}`,
    `\p{Lu}`, `\P{N}`…), classes, alternation, greedy, lazy and possessive quantifiers, case-insensitive groups,
    lookahead and atomic groups. It decodes the text once into code points with a category key each, so a class
    test is a bit lookup. The Unicode data (categories, NFC and NFKC tables) is generated into `src/core/unicode_tables.inc`
    by `tools/gen_unicode_tables.py`.
  * **Special tokens** written in the text map to their ids, found in one pass over the text. Those of the Phi-3
    family (Phi-3/3.5, Phi-4-mini) swallow the white space after them, as their Hugging Face tokenizer does. A
    byte a vocabulary lacks (Hunyuan's dense models have no `\r`) is left out before merging, as there.
    Nemotron's Metaspace `first` policy adds a dummy space only at the start of plain text; real spaces after
    literal controls remain significant, including when matching saved contexts. Its absent BOS metadata
    defaults to no BOS; an explicit file setting always wins.
    SentencePiece compiled character maps run before BPE when the GGUF records one. Orion's converter omits
    its map and initial-space policy, so its architecture row supplies the exact registered map and defaults;
    this preserves its trained NMT-NFKC normalization, tabs and newlines without changing existing tokenizers.
    Byte fallback follows BYTE token types; ordinary `<0xXX>` vocabulary entries retain their literal text.
  * **Checked token for token** against each model's Hugging Face tokenizer, special tokens, contractions,
    digits, white-space runs, composed and decomposed accents, CJK, emoji and right-to-left text included
    ([Correctness](benchmarks.md#correctness)).
  * **Speed.** Tokenizing sits on the typing path (prompts are encoded while the user writes). Merges compare token
    ids through a heap, never strings, so a long piece costs O(n log n). On an Apple M4 Pro a 68 KB text encodes in
    2.4–3.6 ms (19–28 MB/s, Qwen3, Qwen3.5, Llama 3.2) and an 80 KB conversation full of special tokens in 2.5–3.2 ms.
* **Chat templates from the model.** When a GGUF carries the Jinja chat template of its Hugging Face tokenizer
  (`tokenizer.chat_template`), Liyab's own Jinja interpreter (`src/core/chat_template.cpp`) renders it the way
  transformers' `apply_chat_template` does: the same whitespace rules, Python's printing, `tojson`,
  `raise_exception`, `strftime_now`, loops with `loop.previtem` / `nextitem`, macros, namespaces, string methods and
  the usual filters and tests, and the sandbox's refusal of methods that change a list or dict.
  * `Engine::render_chat` (C: `liyab_engine_render_chat`) takes the request as JSON: `messages`, optional `tools`
    and `add_generation_prompt`, and any template variable such as `enable_thinking`. A text that starts with the
    BOS token's own text (Llama 3, Gemma) gets no second BOS when tokenized.
  * `Engine::chat_format` (C: `liyab_engine_chat_format`) renders probe conversations and reports how the template
    lays out turns: the system block, what opens and closes a user turn, the generation prompt and what ends a
    reply. It also says whether the template is **prefix-stable**: rendering *n* turns, then the reply, must be the
    start of rendering *n* + 1 turns, which context reuse across turns needs. Llama 3, Qwen2.5, Gemma 3, Phi-3.5,
    Phi-4-mini, Hunyuan-MT, DeepSeek V3.1, and Qwen3.6 with `preserve_thinking`, are. Qwen3 and Qwen3.5 are not:
    they drop the reasoning of earlier replies once the conversation goes on. The load log says which.
  * Checked byte for byte against transformers on 14 conversations per template (system, user, assistant and tool
    turns, tools, inline and separate reasoning, thinking on and off, no system message, non-ASCII and white space
    at the edges), for Qwen2.5, Qwen3, Qwen3.5, Qwen3.6, Llama 3.2, Gemma 3, Phi-3.5, Phi-4-mini, Hunyuan-MT,
    Hunyuan-0.5B, Hunyuan-A13B and DeepSeek V3.1: 165 renders identical, and the 3 conversations transformers
    refuses refused too (`tests/data/chat_templates.json`, written by `tools/chat_template_reference.py`). The
    templates the GGUFs themselves carry gave the same 168 results.
    Clock-reading fixtures pin a UTC instant and supply it to both renderers, so their expected date is reproducible.
    The C++ fixture helper uses the platform's thread-safe UTC conversion (`gmtime_s` on Windows).
* **Prompts are used as given** by `generate()` and `prefill()`: render them with the template first (the app does).

---


## Reference tooling and additional model evidence

SmolLM3 (`smaug-bpe`) uses Llama 3's split and whole-piece lookup; OLMo 2/3 needs the explicit
`llama-bpe-merges` tag for that split with ordinary merge ranks (b11819 exports a colliding `dbrx` label). Granite (`refact`) and EXAONE 4 (`exaone4`) use GPT-2's ByteLevel rule. Seed-OSS (`seed-coder`)
normalizes NFC and splits punctuation separately from following newlines. These tokenizers are checked against
their Hugging Face token IDs on prose, edge cases and chat text. SmolLM3's own chat template is pinned in the
template fixture, with 14 byte-identical reference renders.
GLM-4's `glm4` pre-tokenizer uses the Llama-3 split and whole-piece vocabulary lookup without normalization.
The tokenizer also reads the integer scores the converter writes for Tekken. The template fixture includes
GLM-4, Gemma 1, Granite, OLMo 3, EXAONE 4 and Seed-OSS; scalar dictionary keys retain their Python types,
including Seed-OSS's integer thinking-budget lookup. The reference tool disables the unused generation cache
for its single teacher-forced pass.
For old model classes with nonpersistent rotary buffers, `conform_reference.py --state-dict` constructs the local
model on CPU and loads its safetensors directly, checking every parameter (including tied weights). Reference
tracing can observe older models' native PyTorch SDPA calls when they predate transformers' attention interface.
`describe_model.py --gate --reference-state-dict` uses that loader for a local checkpoint's conformance gate.
Tiny-checkpoint JSON overrides pass through the configuration constructor, preserving nested block descriptions.
The description tool measures scalar offsets in RMSNorm weights and verifies that conversion folds them into
the stored weights. Native SDPA traces check causal/window masks and report additional score biases as unknown.
`chat_template_reference.py --trust-remote-code` can record templates from trusted official custom tokenizers.
The `chatglm-spm` tokenizer scheme adds its `[gMASK], sop` prefix when initial special tokens are requested,
recognizes a prefix already written by a chat template, and keeps automatic prefix tokens outside reusable segments.
Template fixtures also cover ChatGLM3, EXAONE 3.5, MiniCPM, Nemotron-NAS and IQuest-Coder (14 cases each).
The `plamo3` tokenizer uses integer-score unigram segmentation with byte fallback, preserving raw spaces and
Unicode. It keeps the authors' boundaries around special spellings, repeated characters and consecutive spaces.
Norm vectors accept the converter's bare names as well as names ending in `.weight`, including PLaMo3's post norms.
PLaMo3's two-block tiny case (local/global attention, an eight-token window) gives mean KL 0.00000 and 99.61% top-1.
The real 610m-fin-instruct F16 checkpoint passes the gate at mean KL 0.00000 and 100% top-1. The eight-block tiny
case gives 98.83% top-1 with F16 KV, below the tiny threshold.
Reference tools offer `--slow-tokenizer` when a new library's fast conversion changes the authors' SentencePiece
segmentation; `describe_model.py --reference-slow-tokenizer` applies that choice consistently throughout its gate.
Reference tracing also observes legacy eager attention's paired matrix products, accepting their scale and mask
only after recomputing the full softmax probabilities from the recorded queries and keys.
Packed GQA projections are matched to their observed per-group Q/K/V rows; the file may split those projections.
InternLM2's tiny checkpoint matches all three tokenizer texts with the original slow implementation and gives
mean KL 0.00000 and 100% top-1. The description tool reports that dynamic RoPE growth beyond the original
context is not implemented. For multimodal checkpoints such as Ministral 3, the reference tool selects
transformers' image/text model class and runs the same text-only prompt, without images or added image tokens.
InternLM2 and Ministral 3 each have 14 template fixtures. An explicit template file is rendered through
transformers' Jinja renderer, including for tokenizers whose own chat formatter does not use Jinja.
Reference tracing shortens a uniform configured attention window so its boundary occurs within the probe,
verifies the actual mask there and reports the original configured size. Full conformance uses the original size.
For a verified `scale_depth / sqrt(n_layers)` residual rule, it also reports the multiplier for the full layer
count; reducing the tracing stack must not make a correct MiniCPM file appear to have the wrong multiplier.
Both residual branches must be present and verified before this correction is applied.
`gpt2-nfkc-digits` normalizes Unicode 9 NFKC, isolates every numeric character and then applies GPT-2 ByteLevel.
It uses the existing canonical ordering/composition and additive Unicode 9 compatibility tables; existing NFC
tables are unchanged. A converter that labels this source pipeline as plain ByteLevel needs a metadata-corrected
copy. The description tool verifies the source JSON pipeline and refuses incompatible or unverified uses.
Normalization and tokenization run on the CPU for every backend.
Modern per-block geometry, scalar/rotation metadata, NoPE, post-only and whole-projection norm plans, and
corrected `spm-greedy` / `gpt2-nfkc-digits` files hash the complete first GGUF header for saved contexts,
key-scale/latent-form calibration and KV-store keys. Equal-sized metadata or tokenizer changes beyond the first MiB therefore invalidate incompatible contexts
and derived calibrations. Legacy plans without these features retain their prior identities and state compatibility.
If transformers reconstructs a fast tokenizer with a different normalizer or pre-tokenizer, reference tools
reload the author's `tokenizer.json` through the existing `tokenizer_object` API. The tokenizer class, special
configuration and chat template are retained; a class that discards the supplied pipeline is refused.
EXAONE 3.5's F32 tiny checkpoint passes the complete strict gate only after correcting the converter's
`tokenizer.ggml.pre` from `exaone4` to `gpt2-nfkc-digits`: mean/max KL 0.00000, top-1 100%, 512 conformance IDs
and all three tokenizer texts identical (615/917/243). All 21 tensor payloads are byte-identical in that copy;
its 14 template cases also match. The unmodified b11819 export fails the tokenizer gate. EXAONE 4 retains
its existing plain ByteLevel rule. The full corrected-file result is recorded below.
The explicit original slow-tokenizer path is unchanged.
`python3 tests/test_reference_tokenizer.py` checks matching, missing and discarded serialized pipelines without weights.

If a legacy reference omits its input-embedding accessor, the trace can use a unique embedding lookup observed
on the actual token IDs before the first block. Ambiguous lookups remain unknown; the measured embedding scale
must still explain the entire input to that block.
The same rule applies to a missing output accessor: one observed linear projection after the blocks must have
the logits' shape, and its input and full output must pass the norm and logit-transform checks.
Gate logs escape invalid UTF-8 in byte-token mismatch diagnostics and preserve the failed check's exit status.
SentencePiece files can select `tokenizer.ggml.pre=spm-greedy`: identity text normalization, escaped spaces and
longest-prefix matching with byte fallback. This represents a source vocabulary whose ordinary pieces are all
`USER_DEFINED`; the description tool verifies those flags through `sentencepiece` and `protobuf` before accepting
the mode. The converter can discard the flags and incorrectly declare ordinary BPE; its space-prefix declaration
can also disagree with the source model. `tools/gguf_set_keys.py` can correct those metadata keys in a copy.
InternLM3's F32 tiny checkpoint passes with `spm-greedy` and `add_space_prefix=false`: mean KL 0.00000,
top-1 100%, all three tokenizer texts identical (606/1,007/205 IDs), and 14 template renders identical.
The original slow tokenizer is the reference. All 21 tensor payloads remain byte-identical after the metadata
correction. This check stays within the declared 32,768-token context; dynamic-NTK extrapolation remains unsupported.
DeciLM's three-block F32 tiny checkpoint (2/1/4 KV heads) passes the strict gate with mean/max KL 0.00000,
top-1 100%, three tokenizer texts identical (629/1,032/279 IDs) and 14 template renders identical.
For DeciLM-7B, b11819's `LlamaHfVocab` drops SentencePiece scores; a local converter override must use its native
SentencePiece reader when `tokenizer.model` is present. All 30 tensor payloads are unchanged by that override.
The separate eight-block Nemotron-NAS tiny case covers varied query heads, absent attention and FFN widths.
The full F16 gate for `openbmb/MiniCPM-1B-sft-bf16` passes with mean KL 0.00000, top-1 100.00%,
identical tokenizer IDs on all three texts (631/1049/248), and 512 BOS-inclusive conformance IDs.
Its own template matches all 14 reference cases (14 rendered, 0 correctly refused).
The legacy MiniCPM reference needs two local transformers 5 API adaptations: recognize its normalized default
RoPE configuration as unscaled, and express its tied-weight declaration as a mapping. Its forward equations and
checkpoint tensors are unchanged; the b11819 GGUF conversion is unmodified.
The full F16 gate for `internlm/internlm2_5-1_8b-chat` passes with mean KL 0.00000, top-1 100.00%,
identical tokenizer IDs on all three texts (575/989/190), and 512 BOS-inclusive conformance IDs.
Its own template matches all 14 reference cases (14 rendered, 0 correctly refused).
This gate restores source EOS 2 and unknown token 0 in metadata, preserving the converter's chat terminator
92542 as EOT. The original converter replaces EOS and omits unknown; all 219 tensor payloads are unchanged.

The full F16 gate for `unsloth/gemma-1.1-2b-it` passes with mean KL 0.00000, top-1 100.00%,
identical tokenizer IDs on all three texts (513/786/211), and 512 BOS-inclusive conformance IDs.
Its own template matches all 14 reference cases (1 rendered, 13 correctly refused).
The full F16 gate for `ibm-granite/granite-3.3-2b-instruct` passes with mean KL 0.00000, top-1 100.00%,
identical tokenizer IDs on all three texts (625/971/247), and 512 BOS-inclusive conformance IDs.
Its own template matches all 14 reference cases (14 rendered, 0 correctly refused).
The full F16 gate for `mistralai/Ministral-3-3B-Instruct-2512-BF16` passes with mean KL 0.00000, top-1 100.00%,
identical tokenizer IDs on all three texts (517/879/235), and 512 BOS-inclusive conformance IDs.
Its own template matches all 14 reference cases (14 rendered, 0 correctly refused).
This corrected-file gate preserves the official authored template: b11819 otherwise selects a Devstral community fallback. The metadata-only correction leaves all 236 tensor descriptors and the entire tensor data section unchanged. Future conversion uses the native tekken vocabulary reader and the generic author-file precedence shim.
The tokenizer exposes no Jinja template attribute; its shipped `chat_template.jinja` is rendered explicitly through transformers, matching the actual GGUF rendered bytes. Comments/code whitespace may differ; the Jinja AST is equivalent. This does not assert equivalence to its separate native chat formatter.
The `ministral3` fixture tests the full official author template; `ministral3-real` tests its compact equivalent stored in the corrected GGUF.
The full F16 gate for `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` passes with mean KL 0.00000, top-1 100.00%,
identical tokenizer IDs on all three texts (615/917/243), and 512 BOS-inclusive conformance IDs.
Its own template matches all 14 reference cases (14 rendered, 0 correctly refused).
The b11819 tokenizer metadata needs the source-pipeline correction `gpt2-nfkc-digits`; this is a corrected-file gate, not a plug-and-play unmodified export. The metadata-only converter shim retains all tensor conversion code.
Under transformers 5, the validation converter explicitly permits the pinned official configuration class during tokenizer loading; the generic configuration fallback cannot initialize its legacy RoPE fields. The source configuration and tensor conversion code are unchanged.
Its 80-channel heads use the existing F16 KV fallback for Q8_0, Q4_0 and Q4_1 requests; all three produce the same full conformance result as the F16 request.
GraniteSWA remains a later integration case: its reference always includes learned per-head attention sinks,
in addition to interleaved windows, per-block RoPE choices and Granite's multipliers.

The generic block planner also handles blocks without rotary positions, norms applied only after a residual branch,
Q/K RMSNorm over a complete projection, and explicit embedding, residual and logit multipliers. YaRN can use the
file's whole-dimension magnitude convention, and queries can take a position-dependent attention temperature.
These operations run on the CPU on every backend; matrix projections retain the existing Metal/Vulkan routing.
Quantized-key calibration bounds query ranges with the largest attention temperature in the covered context.
Synthetic tests cover prefill, decode, rollback, sequences and KV reuse. The description tool checks these facts
against the reference and file; candidate registry rows alone do not establish model conformance.
A five-block Nemotron-NAS tiny case exercises two GQA ratios, a linear mixer, absent attention, a fully empty
block and different FFN widths. Its exact registry row passes the strict F32 gate: mean/max KL 0.00000,
top-1 99.80%, 499 conformance IDs and three tokenizer texts identical (498/797/202). The 14 template
cases match the existing fixture. Regeneration from the original configuration produces identical tensor bytes;
the official forward source is unchanged. The existing NAS golden case covers these numerical paths.
A linear mixer is established by recomputing its entire observed output from a unique linear projection,
including any bias. This also covers direct child `forward` calls that bypass module hooks. Ambiguous projections,
additional transformations and shape changes remain unknown; both one-branch and two-branch blocks use this rule.
`tests/test_reference_trace.py` checks these positive and negative cases with PyTorch and no model weights.
Per-block head counts and FFN widths can be scalar metadata or integer arrays. Zero KV heads select a single
linear mixer projection (positive query heads) or no mixer (zero query heads); a zero FFN width omits that branch.
Attention uses each block's own grouping. The existing KV image layout reserves the maximum KV-head count for
each attention block and zero-pads unused heads, preserving the same sequence, rollback and saved-context paths.
Latent attention retains uniform head geometry; files that vary its head counts are refused with a precise reason.
Saved states of models with per-block geometry or explicit modern scalars also check the model-file fingerprint,
so equal dimensions cannot mix contexts computed with different layouts or scales. Empty blocks create no streaming I/O items.
Per-block geometry is checked against an independent float reference on CPU and available GPU backends. Streamed
block projections use the existing CPU route because their reused slot addresses cannot identify persistent GPU weights.

The modern block changes pass all 14 host suites on native ARM64, Rosetta AVX2 and Rosetta scalar, plus the
experimental Android arm64 build. All 111 earlier golden pins remain unchanged; 12 new cases are pinned on
all three platforms. The three-model CPU/Metal comparison passes the symmetric ±3% bound after a separately
recorded AC confirmation of its one affected configuration. The original timing outlier and the accepted
[before/after measurements](../model-descriptions.md#modern-block-regression-check) are both retained.

---


### Gemma text and embedding details

The model planner also recognizes Gemma 3n's AltUp residual streams, Laurel low-rank branches, per-layer token
embeddings, Gaussian gate sparsity and shared keys/values. These compose with Gemma's existing GeGLU and sandwich
norms. Gemma 4 adds optional key-as-value projections and block output scales; files with different local/global
head geometry and per-block FFN widths now use explicit compatibility plans. Gemma 3n E2B's BF16 conversion
passes the real-model gate: mean KL 0.00000, top-1 100%, and identical token IDs on all three tokenizer texts.
Gemma 4 also has [heterogeneous dense/MoE synthetic checks](../liyab-compat.md#offline-gemma-4-check).
The per-layer table stays memory-mapped and is read by token row; scratch holds only the current batch's per-layer
inputs and residual streams. Dense projections use the existing CPU/Metal/Vulkan routing and CPU fallback; the
small AltUp router and corrections run on CPU. EmbeddingGemma runs bidirectional attention over each complete
input, mean pooling, the GGUF's `dense_2`/`dense_3` projections, then optional truncation and L2 normalization.
Bidirectional models reject incremental continuation, tree drafts and rolling caches; quantized-key calibration
also reads its complete text in one pass. Shared key-as-value projections include their projection bias.
The `gemma4` tokenizer uses SentencePiece space normalization with the file's explicit BPE merge priorities,
which can differ from vocabulary scores. Existing SentencePiece tokenizers keep their score-based merges.
Gemma 4's four-block test with uniform head geometry matches all 512 token IDs, mean KL 0.00000 and top-1 99.22%;
the newer compatibility-layer checks cover heterogeneous widths and parallel MoE, with trained checkpoints still unmeasured. Converter compatibility details are in
[model descriptions](../model-descriptions.md#gemma-residual-and-embedding-parts).
Text conformance of a multimodal checkpoint loads its registered text decoder alone; the reference tool infers
the decoder namespace from the checkpoint's embedding key, without requiring unused vision/audio packages.
Its `--config` overrides apply to the text configuration.
