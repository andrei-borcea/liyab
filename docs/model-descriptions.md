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

## Classic transformer parts

The generic planner also recognizes `gpt2`, `gptneox`, `falcon`, `bloom`, `mpt`, `starcoder`, `starcoder2`, `phi2`,
`stablelm`, `orion`, `olmo`, `command-r`, `nemotron`, `apertus`, `arcee` and `dbrx`. Registry presence alone is not
conformance evidence: the family gates and their measured results must accompany a support claim.

`ArchSpec::norm` selects LayerNorm or RMSNorm. Each LayerNorm binds its own optional weight and bias; omitting
both implements OLMo 1's non-parametric normalization. The converter stores Nemotron LayerNorm1P weights as
`1 + w`, so the runtime applies ordinary LayerNorm. Norm arithmetic uses double precision for mean and variance,
then a scalar, NEON or AVX2 affine step. The AVX2 path uses the existing CPU feature dispatch.

A dense block without a gate and with an `n_ff`-row up projection plans an ungated FFN: `down(act(up(x) + b)) + b`.
Supported activations include exact GELU (erf), GELU (tanh), ReLU, ReLU squared and Apertus xIELU. The latter reads
its per-block parameters from the four `xielu.*` GGUF arrays. Split and fused gate/up projections apply their
optional biases after projection; attention output and LM-head biases follow the same rule. `logit_scale`
multiplies biased logits before any final soft-cap.

Parallel residual blocks preserve the normalized block input for the FFN while attention runs. The FFN uses its
own norm when present, otherwise the attention norm's input. Learned position rows are added at each sequence's
absolute positions and limit the addressable context. Their residuals cannot be moved for segment reuse; the
engine refuses that operation. Prefix reuse and context save/restore keep the original positions.

ALiBi adds each query head's slope times the logical key-to-query distance before softmax, independently of page
slots and speculative tree row slots. Q/K/V clamping follows projection and bias, before normalization and RoPE.
Per-head Q/K norms accept shared head weights or one row per head. StarCoder2's declared sliding window uses the
existing per-block paged-cache layout.

These elementwise operations execute on the CPU with Metal/Vulkan projection routing unchanged. The CPU is the
reference; the new norms have scalar, NEON and x86 AVX2 affine paths. Synthetic golden cases cover every added
part; batching, rollback, speculative trees, sequences, prefix caching, context files and segment reuse exercise
how those parts compose with existing caches.

The published b11819 StarCoder2 converter omits `attention.sliding_window`. All three official StarCoder2
configurations (3B, 7B and 15B) declare 4096. The transformers 5.19 configuration class defaults to `None`, so
a family-wide fallback cannot be inferred from that constructor or from a single customized checkpoint. The registry records this
as `sliding_window_default`, a generic fallback used only when the file omits the key. Explicit metadata wins,
including zero for full attention. A custom checkpoint whose window differs must record it: the description
tool reports a file error when an omitted key would choose a different window. The tool checks a declared
registry fallback against the observed reference window when the key is absent, and refuses an omitted-window
file with no established fallback. The 4223-position unpatched-converter check crosses position 4096: KL mean
0.00000, top-1 99.98%; an explicit 32-position override gives KL 0.00000, top-1 100%.

Some older MoE converters, including DBRX, store an expert's width as `feed_forward_length`. The planner accepts
that key when `expert_feed_forward_length` is absent; the latter always takes precedence. Expert tensor shapes
are still checked when weights are bound.

The description tool also observes eager attention implementations that bypass transformers' attention registry:
score products, softmax inputs and value products must agree numerically. Scaled QK and linear per-head position
slopes are established from those values. Non-parametric and LayerNorm1P formulas, both parallel residual forms,
per-head LayerNorm modules, learned positions, embedding norms, projection clamps and xIELU scalars have matching
rules. Nested attention and expert configuration objects retain their types when tiny checkpoints are reduced or
overridden with `--config`.

Falcon's pre-tokenizer orders punctuation splitting, ByteLevel, then digit grouping. The steps following
ByteLevel see its printable byte alphabet, so they can split within an original UTF-8 character. The pre-tokenizer
returns raw byte views in that case; the BPE encoder uses the shared alphabet to merge them. Existing profiles
continue to split Unicode text before byte encoding. Command-R and OLMo profiles normalize to NFC before their
splits; StarCoder/Refact and StableLM 2 keep the input's normalization.
DBRX follows its declared GPT-4/cl100k tiktoken encoding, with three-digit groups, possessive word matching,
and whole-piece vocabulary lookup before BPE merges. Its native IDs, including special-token gaps, are retained.

SentencePiece's dummy space after a literal control token is an architecture fact omitted by GGUF.
`spm_prefix_after_control` preserves the usual behavior by default; Nemotron's Metaspace `first` scheme
adds it only at the beginning of plain text. Real spaces after controls remain significant. The description
tool probes the reference tokenizer after literal BOS/EOS controls before proposing the row. Prefix matching
distinguishes an implicitly added BOS from an explicit control so saved contexts reuse the correct text.
`spm_add_bos_default` records an absent BOS key's policy independently: Nemotron adds no BOS, while existing
SentencePiece rows retain their default. Explicit `tokenizer.ggml.add_bos_token` always wins. The tool derives
this field from encoding with and without special tokens.

The Command-R and Arcee templates are checked byte for byte against their own transformers tokenizers,
including generation prompts, tool requests, Unicode text and rejected conversations.

Orion's published converter omits its SentencePiece normalization map and dummy-space policy. The generic
SentencePiece path reads `tokenizer.ggml.precompiled_charsmap` when present; otherwise `ArchSpec::spm_charsmap`
selects an exact registered map by SHA-256. `spm_add_space_prefix_default` supplies an omitted prefix key.
The Orion row records the map from official `OrionStarAI/Orion-14B-Base` revision
`b988a511e9c880a465e704c57e3f297a3e7ef282`: NMT-NFKC with tabs/newlines retained, no dummy prefix and no whitespace
collapse. This is the trained map, including its Unicode-version choices, rather than a substitute contemporary
NFKC transform. `tools/spm_normalizer.py` reads the model's protobuf fields without external protobuf packages
and emits the map data reproducibly. The description tool compares the native reference's exact map digest
and refuses a missing unregistered map. Character-map offsets and replacement terminators are validated at load;
normalization uses longest-prefix matches on every CPU backend before SentencePiece BPE.
The same converter also drops SentencePiece's user-defined token types. `spm_user_defined` identifies the exact
307-symbol set from that tokenizer, used only for the legacy omitted-map layout. Those symbols bypass character
normalization and remain indivisible during BPE; this is what preserves tabs, newlines and two-space pieces.
Files carrying an explicit character map use their own token types instead of the fallback.
SentencePiece byte fallback follows BYTE token types, not token spelling. Without them, consecutive unknown
characters form one unknown token; normal vocabulary entries spelled `<0xXX>` decode literally.
The trace registers an explicit eager causal mask while evaluating captured attention with SDPA, allowing
implementations that require a materialized mask (such as MPT) to expose their score operations too.
LayerNorm epsilon is taken from the observed function call when the reference supplies it there, including
parameterless OLMo norms whose modules have no epsilon attribute.
Scalar checkpoint tensors are read as scalars too, so per-block xIELU parameters are verified against the file.

`ArchSpec::attn_norm` names the mixer's norm independently of `ffn_norm`. A parallel block whose mixer norm
is absent uses the FFN norm as its shared input norm; otherwise each branch normalizes the block input with its
own tensor. This reads the published Falcon layouts: `ln_attn` is stored as `attn_norm_2`, and `ln_mlp` as
`attn_norm`, while Falcon-7B stores one shared `attn_norm`. The description tool matches each tensor to its
reference norm's weights and verifies both roles. The unused alternate name of a shared-norm checkpoint remains
the registered default until a two-norm checkpoint establishes it.
When eager attention expands grouped keys/values before the score product, the trace recovers logical KV
groups only after both repeated keys and values agree bit for bit, then matches them to their projections.
An observed GELU function's mode is authoritative when its input and output link exactly to the FFN projections;
small activations can otherwise make erf and tanh approximations indistinguishable at the trace's tolerance.
The xIELU path is the ungated FFN described above; files requesting a gated xIELU FFN are refused explicitly.
Synthetic F16 GPT-2, Falcon, BLOOM, Apertus and biased latent-attention models compare CPU execution with GPU projections plus CPU
operations through prefill, decode and rollback; available Metal/Vulkan backends use the existing operation routing.
ReLU and ReLU² have NEON and runtime-dispatched AVX2 kernels with scalar tails, for ungated and gated FFNs.
Comparison/select retains the scalar signed-zero and NaN behavior. Their arithmetic order stays unchanged;
GELU and xIELU keep scalar libm for the transcendental functions so pinned rounded results remain portable.
Odd FFN widths check ReLU² scalar tails against different batch boundaries and compare fused/split biased
gated projections on every KV type.
The reference trace reduces FFN widths and expert counts in typed child configurations too. Original residual
and head dimensions remain intact for matching file geometry; nested DBRX FFNs no longer allocate the full
expert widths while describing an architecture without its weights.
Learned position embeddings combined with an embedding norm are refused because their relative order is not
established; either a scale or a bias tensor establishes the norm's presence, including bias-only LayerNorm.
The optional `attn_output.bias` is bound for ordinary and latent attention alike. It is added after the output
projection in both absorbed and expanded latent forms, before any post-mixer norm or residual add.
Local PyTorch `.bin` checkpoints and their indexed shards can supply the reference tensors too. The reader uses
`torch.load(weights_only=True, mmap=True)`, maps each shard only when requested, and materializes only the rows
needed for matching. Remote checkpoint inspection still requires safetensors. Run
`python tests/test_describe_model.py` in the reference environment to compare both readers, scalar and partial
rows, saved gradient flags, invalid row ranges, missing/malformed checkpoints and lazy loading of PyTorch shards.

### Classic conformance

F32 checkpoints with reduced dimensions and random weights exercise the real transformers implementations,
using F16 KV and the same 512-token conformance text. Every row below has identical token IDs on all three
tokenizer texts and an exactly matching derived registry row. Printed KL values are rounded to five decimals.

| Architecture | Source configuration | Tiny mean KL | Tiny top-1 |
| :--- | :--- | ---: | ---: |
| `gpt2` | `openai-community/gpt2` | 0.00000 | 100.00% |
| `gptneox` | `EleutherAI/pythia-410m-deduped` | 0.00000 | 100.00% |
| `falcon` | `tiiuae/falcon-7b`, shared and separate branch norms | 0.00000 | 100.00% |
| `bloom` | `bigscience/bloomz-560m`, four and three heads | 0.00000 | 100.00% |
| `mpt` | `anas-awadalla/mpt-7b` | 0.00000 | 100.00% |
| `starcoder` | `bigcode/tiny_starcoder_py` | 0.00000 | 100.00% |
| `starcoder2` | `bigcode/starcoder2-3b`, explicit 32-position window | 0.00000 | 100.00% |
| `phi2` | `microsoft/phi-2` | 0.00000 | 100.00% |
| `stablelm` | `stabilityai/stablelm-2-1_6b`, with and without per-head Q/K norms | 0.00000 | 100.00% |
| `orion` | official `OrionStarAI/Orion-14B-Base` code | 0.00000 | 100.00% |
| `olmo` | `allenai/OLMo-1B-hf` | 0.00000 | 100.00% |
| `command-r` | `mlx-community/c4ai-command-r-v01-4bit` configuration, with and without Q/K norms | 0.00000 | 100.00% |
| `nemotron` | `nvidia/Minitron-4B-Base` | 0.00000 | 99.80% |
| `apertus` | `swiss-ai/Apertus-8B-2509` | 0.00000 | 99.80% |
| `arcee` | `arcee-ai/AFM-4.5B` | 0.00000 | 100.00% |
| `dbrx` | `alpindale/dbrx-instruct`, native GPT-4 tokenizer export | 0.00000 | 100.00% |

Real checkpoints measured so far use the full weights and pass the description tool's gate, including the
registry row and all three tokenizer texts. These results do not stand in for pending family gates.

| Checkpoint | GGUF | Mean KL | Top-1 |
| :--- | :--- | ---: | ---: |
| `openai-community/gpt2` | F16 | 0.00001 | 99.61% |
| `EleutherAI/pythia-410m-deduped` | F16 | 0.00065 | 97.46% |
| `bigscience/bloomz-560m` | F16 | 0.00000 | 99.80% |
| `bigcode/tiny_starcoder_py` | F16 | 0.00000 | 99.80% |
| `stabilityai/stablelm-2-1_6b` | Q8_0 | 0.00130 | 97.46% |
| `allenai/OLMo-1B-hf` | Q8_0 | 0.00089 | 97.26% |
| `microsoft/phi-2` | F16 | 0.00009 | 98.83% |
| `bigcode/starcoder2-3b` | F16 | 0.00000 | 100.00% |

The same Phi-2 checkpoint converted to Q8_0 gives mean KL 0.00573 and top-1 agreement 94.91%, below the
97% gate. Its registry row and token IDs match; the passing Phi-2 result above establishes F16 conformance.
StarCoder2-3B in Q8_0 gives mean KL 0.00091 and top-1 agreement 96.87%, also below the gate, with matching
registry row and token IDs. Its F16 result above uses the same full checkpoint and conformance text.

The unmodified b11819 converter rejects a reduced official `tiiuae/falcon-rw-1b` checkpoint at
`transformer.h.0.post_attention_layernorm.bias` (unmapped tensor). Its full weights were not downloaded;
Falcon evidence above covers the 7B configuration and its separate-norm variant.

The Minitron-4B real gate remains pending. The larger configurations above have
tiny conformance only. Command-R and Arcee chat templates are checked byte for byte against transformers;
the other selected base tokenizers do not provide a chat template.
