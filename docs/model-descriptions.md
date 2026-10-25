# Describing a model automatically

Liyab runs a model when two things are known about it: the facts about its architecture that its tensors cannot
tell (one row of `ArchSpec` in `src/core/architecture.cpp`), and a plan for each block, which the engine makes from
the tensors present. `tools/describe_model.py` establishes the first from the model's Hugging Face implementation
and a GGUF of it, and verifies it:

```bash
tools/describe_model.py --model baidu/ERNIE-4.5-0.3B-PT --gguf ERNIE-4.5-0.3B-PT-F16.gguf --gate
```

It prints every `ArchSpec` field with the evidence for its value, the parts of the model the file holds (RoPE
encoding, windows, per-block plan, MoE routing, tokenizer), and then either the registry row or the precise list of
parts Liyab does not implement yet. A fact it cannot establish is reported as unknown and no row is proposed: it
never fills a field with a guess. `--json FILE` writes the same as data.

A description reads a few MB of the two checkpoints and runs a reduced copy of the model: Hunyuan-0.5B with a remote
GGUF took 16 s on an M4 Pro, 3 s of it CPU, the rest the network. The gate builds `liyab-kl` and `liyab-tok` once
(later runs recompile `architecture.cpp` when the row changes) and runs the model in transformers, in float32, and in
Liyab: the 4 GB of float32 weights of a 1B model, 13 GB for a 3B one.

## Contents

* [Where the facts come from](#where-the-facts-come-from)
* [The conformance gate](#the-conformance-gate)
* [Validation](#validation)
* [Limits](#limits)

## Where the facts come from

**The reference implementation, observed.** `tools/reference_trace.py` builds a reduced copy of the transformers
model from its configuration: the widths of the residual stream and of the attention heads are the model's, so
every scale, base and multiplier is its own; the blocks are fewer (the shortest prefix that holds every kind of
block the configuration lists), the vocabulary is 512 tokens, the FFNs 64 wide and the experts 16. Its weights are
random, with norm weights drawn away from 0 and 1 so that a norm multiplying by `w` and one multiplying by `1 + w`
differ. One pass over 13 tokens is recorded: every module's inputs and outputs (forward hooks), and the queries,
keys and values each attention receives with its arguments, through an attention function registered with
transformers. Each fact is then recomputed from the records, never read from a class or module name:

| Fact | How it is established |
| :--- | :--- |
| Block order | The norm whose input is the block input and whose output the mixer reads; the residual `input + mixer output` (or its normed form); the norm that reads that sum and feeds the FFN; the block output `residual + FFN output`. A norm reading the mixer's or the FFN's output is a norm after it. Anything else (a multiplier on the residual adds, an FFN reading the mixer's input, no norm before the mixer) is reported. |
| Norm formula | `x / rms(x) * w`, `x / rms(x) * (1 + w)` or LayerNorm, recomputed with the module's own epsilon. |
| RoPE pairing | The keys the attention received are matched against the key projection rotated by each frequency table the model holds, pairing dimensions `(i, i + r/2)` or `(2i, 2i + 1)`, over the first `r` dimensions. No match without rotation means the block has no position embedding. |
| Q/K norms | A norm reading the projection and feeding the rotation sits before RoPE; one reading the rotated projection and giving the keys sits after it. Its weight's size says whether it spans one head or all of them. |
| Attention | The `scaling`, `sliding_window`, `softcap` and sink arguments the attention receives; whether the output projection reads the attention output or `output * sigmoid(gate)` with the gate taken from `[q | gate]` heads. |
| FFN | Which of `silu`, `gelu` (tanh or erf), `relu`, `relu²`, quick GELU makes `act(gate) * up` equal the down projection's input, for separate or fused `[gate | up]` projections; `act(up)` is a non-gated FFN. |
| MoE routing | The router's logits and the indices and weights the experts receive: top-k over softmax or sigmoid probabilities, with or without a selection bias, renormalization and a scale; a shared expert, with or without a sigmoid gate, from how the block output is composed; the experts' activation from their weights. |
| Embeddings, logits | Block 0's input against the embedding table's rows (a scale of 1 or √d); the logits against the output projection's output (a scale or a soft-cap); whether the output matrix is the embedding table. |

**The GGUF.** Its metadata and tensor names and shapes give the hyperparameters, how the file encodes RoPE (base,
linear scaling, Llama 3 `rope_freqs`, LongRoPE factors, YaRN), the window, the MoE metadata, the tokenizer and each
block's plan, made the way `describe_model()` makes it. The RoPE frequencies and magnitude the engine computes from
the file are compared with the reference's table, the windowed blocks' own table included. Every tensor must be one
the engine binds (`src/core/transformer.cpp`); one it would ignore is a missing part. The file's BOS and
space-prefix rules (`tokenizer.ggml.add_bos_token`, `add_space_prefix`) are compared with what the model's own
tokenizer does with a short text. Where its `tokenizer_config.json` declares a BOS rule that transformers does not
apply (version 5 runs DeepSeek's `tokenizer.json` without the BOS its `LlamaTokenizerFast` added) and the GGUF
follows the declaration, that is a note, not a problem of the file.

What the engine takes from the file rather than from the row (an attention scale, logit soft-caps, YaRN) counts as
supported exactly when `describe_model()` reads the metadata key, or accepts the RoPE scaling type, that carries it;
the reference's value must then be the file's. Otherwise the reference's scale, soft-cap or scaling is a missing
part.

A disagreement between the GGUF and the reference (other weights, a window or BOS rule the file does not record) is
reported as a problem of the file: it does not change the row, and the gate fails on it where it matters.

**A few weights of both checkpoints.** Read with HTTP range requests when the files are remote: two heads of the
first attention block's key projection and the first block's norm weights (a few MB at most).

* The key rows say how the converter stored the heads: in the reference's order, or interleaved (the Llama
  converter turns pairs `(i, i + d/2)` into `(2i, 2i + 1)`). With the reference's pairing this gives
  `ArchSpec::rope`. When no row order matches, the GGUF holds other weights than the reference, and nothing else
  can be trusted.
* The norm weights say which GGUF tensor holds which norm, so the role found in the trace becomes a tensor name
  (`ffn_norm`, `post_mixer_norm`, `post_ffn_norm`), and whether a norm the reference applies as `x (1 + w)` is
  stored as `1 + w`. The Q/K norms and the final norm are checked the same way.

**The engine's sources.** The `ArchSpec` fields, their defaults and the enum values come from `architecture.h`, the
existing rows from `architecture.cpp` (printed back the way that file writes rows: `--self-test` checks the round
trip), the tensors the forward pass binds from `transformer.cpp`, the pre-tokenizer names and tokenizer kinds from
`pre_tokenizer.cpp` and `tokenizer.cpp`. An `ArchSpec` field the tool has no rule for makes it refuse to propose
rows until the rule is added.

## The conformance gate

`--gate` verifies the row on a build that has it:

1. The source tree (tracked files and untracked ones git does not ignore, without `app/` and `docs/`) is copied to
   `build/describe_model/src`; files whose contents did not change are left alone, so later runs rebuild only what
   changed. The row is added to `kArchitectures` there, or replaces the architecture's row; the repository itself
   is never modified.
2. `liyab-kl` and `liyab-tok` are built (CPU only), then `architecture.cpp` is restored in the scratch copy.
3. The tokenizer: `tools/tokenizer_reference.py` writes the model's own ids for `tests/data/conform_text.txt`,
   `tokenizer_edge.txt` and `tokenizer_chat.txt`, and `liyab-tok --ids` must give the same ids for each.
4. The forward pass: `tools/conform_reference.py` writes the transformers reference in float32, with the model's
   own special-token rule, and `liyab-kl --ref-tokens --kv f16 --max-kl 0.01 --min-top1 97` must pass (both
   thresholds are options). On the reference's ids the forward pass, which is what the row decides, is measured
   even when the file encodes the text differently; `liyab-kl` still reports whether its own encoding, BOS
   included, gives the same ids, and that must hold too. When the reference soft-caps attention scores and
   `conform_reference.py` takes `--attn-implementation`, the reference runs with transformers' eager attention:
   its default kernel leaves soft caps out.

The row is accepted only if every step passes. When only the encoding fails, the tool says so: the row is right and
the file's tokenizer settings are not. The GGUF must be a local file for the gate, BF16 or F16 (lower bit widths add
their own error), and of the weights the reference loads.

### Without the weights

An architecture can be checked before, or instead of, downloading a model of it: the largest ones (hundreds of GB)
never fit a development machine, and a small checkpoint isolates the forward pass from the weights.
`tools/tiny_checkpoint.py` builds the model from its configuration with random weights: the shortest prefix of
blocks that holds every kind of block, 4 attention heads of the model's own head size, FFNs and experts 128 wide, 16
routed experts; every scalar of the configuration and the model's own tokenizer files are kept, so the text is
encoded as the model encodes it. `--config` sets what the generic reduction cannot know (a latent rank, a state
size, an index top-k small enough to select within the text). The checkpoint is converted to GGUF with the converter
that writes the published files (llama.cpp's `convert_hf_to_gguf.py`), then checked like a real model:

```bash
tools/tiny_checkpoint.py --model Qwen/Qwen3-0.6B --out tiny-qwen3       # 2 of 28 blocks, 80M parameters
python3 convert_hf_to_gguf.py tiny-qwen3 --outtype f32 --outfile tiny-qwen3.gguf
tools/conform_reference.py --model tiny-qwen3 --text tests/data/conform_text.txt --out tiny-qwen3.ref
build/liyab-kl -m tiny-qwen3.gguf -f tests/data/conform_text.txt --compare tiny-qwen3.ref --ref-tokens --kv f16
```

Qwen3 this way: the same 512 token ids as the reference, mean KL 0.00000 (max 0.00000) and top-1 agreement 100.00%
over 511 positions. With F32 weights on both sides, only the order of float32 operations separates the two passes,
so a structural error (a norm, a pairing, a scale, a routing rule) shows as a large divergence rather than a small
one.

## Validation

### Every row in the registry, regenerated

Each supported architecture was described from a model of it; the tool's row equals the registry's row, field by
field and as text.

| Architecture | Model | GGUF | Notes |
| :--- | :--- | :--- | :--- |
| `llama` | unsloth/Llama-3.2-1B-Instruct | Llama-3.2-1B-Instruct-BF16 (local) | key rows interleaved by the converter; `rope_freqs` factors match the reference's Llama 3 frequencies |
| `mistral` | mistralai/Mistral-7B-Instruct-v0.3 | bartowski Q8_0, header renamed (see below) | the same facts as `llama` |
| `qwen2` | Qwen/Qwen2.5-0.5B-Instruct | bartowski f16 (local) | the official `qwen2.5-0.5b-instruct-fp16.gguf` is refused: its key projection matches the reference in no row order (other weights) |
| `qwen3` | Qwen/Qwen3-0.6B | Qwen3-0.6B-BF16 (local) | gate passed (below) |
| `qwen35` | Qwen/Qwen3.5-0.8B | Qwen3.5-0.8B-BF16 (local) | the norm before the FFN is the GGUF's `post_attention_norm`, from the weights; 64 of 256 dimensions rotated; gated attention output |
| `qwen3moe` | Qwen/Qwen3-30B-A3B | Qwen Q8_0 (remote) | routing renormalizes the top 8 |
| `qwen35moe` | Qwen/Qwen3.6-35B-A3B | unsloth BF16, two split parts (remote) | shared expert with a sigmoid gate |
| `hunyuan-dense` | tencent/Hunyuan-0.5B-Instruct | bartowski bf16 (remote) | Q/K norms after RoPE; the file's base (1.11588e7) is the reference's alpha-scaled one |
| `gemma3` | unsloth/gemma-3-270m-it | unsloth F16 (remote) | √640 embedding scale, GeGLU, both sandwich norms, every 6th block global, windowed blocks on their own base 10,000 |
| `phi3` | microsoft/Phi-3.5-mini-instruct | bartowski Q8_0 (remote) | fused QKV and gate/up; LongRoPE short factors and attention factor match; every block windowed (262,144) |
| `ernie4_5` | baidu/ERNIE-4.5-0.3B-PT | unsloth F16, tokenizer keys fixed (local) | adjacent pairs, key rows as the reference stores them; gate passed (below) |

No published GGUF uses `general.architecture = mistral`: llama.cpp writes Mistral 7B and Nemo as `llama`, newer
Mistral models as `mistral3` or `mistral4` (more than 2,400 GGUF repositories found by Mistral-related searches on
Hugging Face were checked). The `mistral` row was regenerated from Mistral-7B-Instruct-v0.3 with a copy of its GGUF's
header whose architecture name and metadata prefix were changed to `mistral`, holding the four tensors the tool
reads. Its original GGUF gives the `llama` row.

Two official Qwen GGUFs hold other weights than their Hugging Face repositories: `qwen2.5-0.5b-instruct-fp16.gguf`
and `qwen2.5-1.5b-instruct-fp16.gguf` (Qwen/Qwen2.5-*-Instruct-GGUF). Their key projections match the reference in no
row order and no block-0 tensor holds the reference's norms, so the tool establishes nothing from them; bartowski's
conversions of the same models match to rounding.

### A new architecture: ERNIE 4.5

`baidu/ERNIE-4.5-0.3B-PT` with `unsloth/ERNIE-4.5-0.3B-PT-GGUF` (F16), `general.architecture = ernie4_5`, which the
registry does not have. Its reference rotates adjacent pairs `(2i, 2i + 1)` and the converter keeps the rows as
they are, so the file pairs adjacent dimensions too; the rest is a plain pre-norm block with SwiGLU. The tool
proposes:

```
    {"ernie4_5", RopeLayout::Adjacent, "ffn_norm", false},
```

It also reports that this GGUF asks for a BOS and a space before the text (`tokenizer.ggml.add_bos_token` true,
`add_space_prefix` absent) while the model's tokenizer adds neither. The gate built the scratch copy with the row
(only `architecture.cpp` recompiled) and rejected it for the file: the forward pass on the reference's ids passed
(mean KL 0.00000, top-1 100.00%), the encodings did not (the first token differs on every text). With those two
keys set to the tokenizer's own rules in a copy of the file, the gate accepted the row.

The row is now in the registry, pinned by a synthetic `ernie4_5` case in `test_golden` (the same tensors as the
`llama` case, so the same hashes). bartowski's and mradermacher's ERNIE GGUFs have the same problem (no
`add_bos_token`, which a SentencePiece vocabulary reads as true, and no `add_space_prefix`). The engine does not
special-case the model: a file is fixed once with `tools/gguf_set_keys.py`, which copies it with the keys set and
the tensors byte for byte:

```bash
tools/gguf_set_keys.py ERNIE-4.5-0.3B-PT-F16.gguf ERNIE-4.5-0.3B-PT-F16-fixed.gguf \
    --bool tokenizer.ggml.add_bos_token=false --bool tokenizer.ggml.add_space_prefix=false
```

### The gate

| Model | Row | Tokenizer: conform / edge / chat | liyab-kl `--kv f16` | Verdict |
| :--- | :--- | ---: | :--- | :--- |
| Qwen3-0.6B (BF16) | registry's `qwen3` | 524 / 804 / 179, identical | 511 positions, mean KL 0.00000, top-1 99.80% | accepted |
| Qwen2.5-1.5B-Instruct (bartowski f16), not tested before | registry's `qwen2` | 524 / 804 / 181, identical | 511 positions, mean KL 0.00000, top-1 100.00% | accepted |
| gemma-3-1b-it (unsloth BF16), not tested before | registry's `gemma3` | 523 / 801 / 211, identical | 511 positions, mean KL 0.00000, top-1 100.00% | accepted |
| Llama-3.2-3B-Instruct (unsloth BF16), not tested before | registry's `llama` | 498 / 797 / 202, identical | 498 positions, mean KL 0.00000, top-1 99.80% | accepted |
| ERNIE-4.5-0.3B (F16, as published) | new `ernie4_5` | first token differs on each text | on the reference's ids: mean KL 0.00000, top-1 100.00% | rejected: the file's BOS and space-prefix rules |
| ERNIE-4.5-0.3B (F16, the two tokenizer keys set to the model's rules) | new `ernie4_5` | 575 / 920 / 240, identical | 511 positions, mean KL 0.00000, top-1 100.00% | accepted |
| ERNIE-4.5-0.3B (F16, fixed by `tools/gguf_set_keys.py`), again once the row was in the registry | registry's `ernie4_5` | 575 / 920 / 240, identical | 511 positions, mean KL 0.00000, top-1 100.00% | accepted |

### Models Liyab does not run

What the tool reports, in its words, for models whose architecture or tokenizer is not supported:

| Model (GGUF) | Not implemented in Liyab |
| :--- | :--- |
| OLMo 2 1B (`olmo2`) | no norm before the mixer or the FFN (only after them); Q/K norms over all heads (2048) rather than per head (128); the `dbrx` pre-tokenizer |
| Granite 3.1 2B (`granite`) | attention scale 0.015625 instead of 1/√64; residual multiplier 0.22; embeddings × 12 (not √d); logits × 0.125; the `refact` pre-tokenizer |
| SmolLM3 3B (`smollm3`) | blocks without RoPE (block 3 of the 4 traced; the configuration's `no_rope_layers` lists every 4th); the `smaug-bpe` pre-tokenizer |

Gemma 2 and DeepSeek-V2 were in this table until the engine gained what the tool listed for them: logit
soft-capping (Gemma 2), and multi-head latent attention, YaRN and the file's attention scale (DeepSeek-V2). The
tool now reproduces the registry's `gemma2` row exactly. Mistral-Nemo-Instruct-2407 and SmolLM2-360M-Instruct were
in it for their pre-tokenizers alone (`tekken`, `smollm`), which are table rows now: described again (Nemo from
bartowski's Q8_0 header, SmolLM2 from bartowski's f16), both give the registry's `llama` row with nothing missing
(Nemo's head size 128 from `key_length`). SmolLM2's f16 file then gave transformers' ids on the whole text and mean KL
0.00000, top-1 100.00% with `liyab-kl --kv f16` (511 positions). For latent attention it reports an unknown rather than a
row: the engine runs it from the file's tensors, but the tool does not compare its facts with the reference, so
such a model is checked with `liyab-kl` instead ([README, Correctness](../README.md#correctness)).

## Limits

* **Transformers must run the model.** The trace and the gate need the model's implementation in transformers
  (or its own code with `--trust-remote-code`). Modeling code that does not call transformers' attention functions
  leaves the attention facts unknown.
* **A prefix of the blocks is traced.** The shortest prefix holding every kind of block the configuration lists per
  block (`layer_types` and similar lists, the first MoE block, a full-attention interval), at least 2. Which blocks
  attend through a window comes from `layer_types` when the configuration has it; otherwise every traced block
  must agree, and the tool says it assumed the rest do too.
* **Experts' weights.** The experts' activation is recomputed from transformers' fused 3-D expert weights; other
  layouts leave it to the dense blocks' activation.
* **What the tool knows of the engine.** Row fields come from `architecture.h` and need a rule each in the tool (it
  refuses rows while one has none). Features the engine takes from the file are recognized by the metadata keys
  `describe_model()` reads: an attention scale (`attention.scale`), soft caps (`attn_logit_softcapping`,
  `final_logit_softcapping`) and the RoPE scaling types it accepts, YaRN's parameters included. A new kind of file
  feature (another multiplier, another scaling) is reported as missing until the tool learns to compare it with the
  reference.
* **Chat templates** are not checked (`liyab-tok --render` and `tests/data/chat_templates.json` do that).
* **The gate measures one text.** `tests/data/conform_text.txt` (512 tokens): long contexts past a model's sliding
  window or LongRoPE's original context are checked by hand (the README's correctness table has examples).

## Biased experts and learned attention sinks

Optional `ffn_gate_inp.bias` is added to the router logits; `exp_probs_b.bias` remains a separate selection-only
correction after softmax or sigmoid. The converter maps Flint's `.gate.expert_bias.bias` to that selection
correction. Expert projection biases have shape `[rows, experts]` in GGUF order and apply before activation or
router weighting. Shared-expert width and leading dense blocks already follow the tensors present; the metadata
counts do not cause an extra expert or a block to execute.

The reference trace recomputes fused expert outputs in either matrix orientation and with contiguous or
interleaved gate/up values, including projection biases. It recognizes `SwigluOai` only when the observed output
matches the activation and the reference's alpha and limit are 1.702 and 7. An unknown expert activation refuses
a proposed row. Learned sink logits are checked against `attn_sinks.weight` in the file and join the attention
softmax denominator once per query head, after merging its paged-cache spans.
MXFP4 batched matmuls share a weight decode across four input rows on NEON and AVX2; their tests compare every
result bit with single-row decode for partial row tiles and odd numbers of 32-value blocks. Decode dispatch and
existing quantized formats retain their kernels and accumulation order.
Per-block FFN width lists are reduced alongside scalar widths when constructing tiny checkpoints and reference
traces; attention sizes, routing parameters, and shared-expert counts keep their original contracts.
Learned sinks are covered by the multi-sequence scheduling tests and by KV relocation and segment-store reuse
checks, including quantized key scaling. They add no per-sequence state: their only persistent values are the
model's head logits. Router prefetch applies selection corrections in probability space, matching actual routing.
A dense checkpoint cannot establish its family's unused expert-routing default. The tool preserves that field
from a registered row and labels it unmeasurable from this checkpoint. Mixtral, which shares `llama` with dense
Llama, supplies the evidence for that family's renormalization default; dense Llama's logits are unaffected.
The trace randomizes declared router selection-correction buffers (`e_score_correction`,
`e_score_correction_bias`, `expert_bias`), which otherwise start at zero and hide their effect. It compares
candidate corrections anywhere inside the FFN with the experts' observed indices and weights; mathematical
buffers such as RoPE frequencies retain their configured values.
GLM-4 MoE binds `post_attention_norm` before the FFN. Its `glm4` BPE pre-tokenizer uses the same split expression
as Llama 3, has no normalization, and honors `ignore_merges` from the tokenizer contract. Its sparse block plan
skips trailing MTP blocks through `nextn_predict_layers`, as the existing loader already does.
A separate attention output-gate tensor is planned alongside the existing fused query/gate layout. Both forms
apply the sigmoid after attention and before the output projection; fused and separate gates together are
rejected as ambiguous. Golden cases and all-KV-type tests compare the two layouts bit for bit.
Legacy `rope_parameters.type` is renamed to `rope_parameters.rope_type` before constructing a transformers
reference (including nested per-layer parameter dictionaries). This is a spelling compatibility fix; factors,
bases, and scaling algorithms are unchanged.

The tiny checkpoints for Qwen1.5-MoE-A2.7B, Mixtral-8x7B, dots.llm1.inst, GLM-4.5 and ERNIE-4.5-21B-A3B exercise
routed and shared experts and leading dense blocks. The proposed Qwen2MoE, Mixtral and dots1 rows match the
registry. The gpt-oss trace proposes its complete row including SwigluOai and its alternating window. GLM-4 MoE
uses `post_attention_norm`; ERNIE uses adjacent-pair RoPE and renormalizes its selected probabilities. ERNIE's
reduced configuration sets `intermediate_size=256` with one KV head and two 128-wide shared experts, preserving
the converter's coupled shared-width metadata. Tiny numerical results are recorded in the README.
The `glm4` table entry is covered by an explicit split/flag test, the 498-token conformance comparison, and
fourteen reference chat-template renderings. Its normalizer is absent and `ignore_merges` is true, as in the
published tokenizer JSON.

`hy_v3` stores its expert selection correction as bare `exp_probs_b`; the loader accepts that spelling alongside
`exp_probs_b.bias` while rejecting a duplicate pair. The trace checks both spellings. The same attention-gate
part also accepts one scalar per head (broadcast over its value channels).

`swiglu_clamp_exp` and `swiglu_clamp_shexp` are per-block float32 arrays. Their positive finite limits clamp the
SiLU result, then multiply by the symmetrically clamped up projection. They differ from SwigluOai, whose clamp
precedes activation. Dense and shared experts use the shared array, routed experts the expert array. Negative,
nonfinite, short, or wrongly typed arrays are rejected; zero retains the original arithmetic. The reference
trace recomputes these limits and compares them with the file.

AFMoE's decimal splitter is equivalent to its tokenizer's bounded regex sequence: isolate each decimal run,
then split in groups of three aligned at the right. A direct linear scan avoids repeated lookaheads on long
numbers. Script splits and word splits follow it. EXAONE-MoE's own regex and NFC rule have a separate table row.
AFMoE and EXAONE-MoE still need modern's support for their global blocks without RoPE before an architecture row
can be admitted.
The description tool proposes `{"hy_v3", RopeLayout::Halves, "ffn_norm", true}` for `tencent/Hy3-preview`.
Its tiny checkpoint passes the standard reference gate (498 positions, KL 0.00000, top-1 100.00%), including the
bare selection-bias tensor. Scalar-head and per-channel attention gates are checked against equivalent fused
query/gate tensors for F16, Q8_0, Q4_0 and Q4_1 KV. An independent float reference checks the post-SiLU clamp;
MXFP4 biased experts with attention wider than the residual stream match block-streamed execution exactly.
