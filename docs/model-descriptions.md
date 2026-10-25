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
transformers. Older remote implementations dispatching directly through PyTorch SDPA or eager matmuls/softmax are
observed at those operations instead; only verified scaled dot products with full causal masks are accepted.
Direct router linear operations owned by a module are recorded with that module's actual inputs and outputs.
Each fact is then recomputed from the
records, never read from a class or module name:

| Fact | How it is established |
| :--- | :--- |
| Block order | The norm whose input is the block input and whose output the mixer reads; the residual `input + mixer output` (or its normed form); the norm that reads that sum and feeds the FFN; the block output `residual + FFN output`. A norm reading the mixer's or the FFN's output is a norm after it. Anything else (a multiplier on the residual adds, an FFN reading the mixer's input, no norm before the mixer) is reported. |
| Norm formula | `x / rms(x) * w`, `x / rms(x) * (1 + w)` or LayerNorm, recomputed with the module's own epsilon. |
| RoPE pairing | The keys the attention received are matched against the key projection rotated by each frequency table the model holds, pairing dimensions `(i, i + r/2)` or `(2i, 2i + 1)`, over the first `r` dimensions. No match without rotation means the block has no position embedding. |
| Q/K norms | A norm reading the projection and feeding the rotation sits before RoPE; one reading the rotated projection and giving the keys sits after it. Its weight's size says whether it spans one head or all of them. |
| Attention | The `scaling`, `sliding_window`, `softcap` and sink arguments the attention receives; whether the output projection reads the attention output or `output * sigmoid(gate)` with the gate taken from `[q | gate]` heads. |
| FFN | Which of `silu`, `gelu` (tanh or erf), `relu`, `relu²`, quick GELU makes `act(gate) * up` equal the down projection's input, for separate or fused `[gate | up]` projections; `act(up)` is a non-gated FFN. |
| MoE routing | The router's logits and the indices and weights the experts receive: top-k over softmax or sigmoid probabilities, with or without a selection bias, group-limited selection (group maximum or sum of the best two selection scores, recomputed against the observed expert indices), renormalization and a scale; a shared expert, with or without a sigmoid gate, from how the block output is composed; the experts' activation from their weights. |
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
0.00000, top-1 100.00% with `liyab-kl --kv f16` (511 positions). For latent attention the tool reconstructs expanded
keys and values from the observed normalized shared latent, the query's low-rank path and rotary suffix, and
checks the absorbed GGUF geometry separately from the expanded reference geometry. It compares the combined
KV projection's rows with the checkpoint before proposing the RoPE pairing. Numerical conformance still uses
`liyab-kl` ([README, Correctness](../README.md#correctness)).

## Limits

* **Transformers must run the model.** The trace and the gate need the model's implementation in transformers
  (or its own code with `--trust-remote-code`). Modeling code that does not call transformers' attention functions
  leaves the attention facts unknown unless its direct PyTorch SDPA or eager operations can be observed and its
  scores and causal mask verified.
* **A prefix of the blocks is traced.** The shortest prefix holding every kind of block the configuration lists per
  block (`layer_types` and similar lists, the first MoE block, a full-attention interval), at least 2. Which blocks
  attend through a window comes from `layer_types` when the configuration has it; otherwise every traced block
  must agree, and the tool says it assumed the rest do too.
* **Experts' weights.** The experts' activation is recomputed from transformers' fused 3-D expert weights or
  individually observed ModuleList experts whose routed inputs and weighted sum can be verified; unknown layouts
  remain unknown.
* **What the tool knows of the engine.** Row fields come from `architecture.h` and need a rule each in the tool (it
  refuses rows while one has none). Features the engine takes from the file are recognized by the metadata keys
  `describe_model()` reads: an attention scale (`attention.scale`), soft caps (`attn_logit_softcapping`,
  `final_logit_softcapping`) and the RoPE scaling types it accepts, YaRN's parameters included. A new kind of file
  feature (another multiplier, another scaling) is reported as missing until the tool learns to compare it with the
  reference.
* **Chat templates** are not checked (`liyab-tok --render` and `tests/data/chat_templates.json` do that).
* **The gate measures one text.** `tests/data/conform_text.txt` (512 tokens): long contexts past a model's sliding
  window or LongRoPE's original context are checked by hand (the README's correctness table has examples).

### Group-limited expert routing

The engine reads `expert_group_count` and `expert_group_used_count`, validates equal group sizes and enough
eligible experts for top-k, then limits expert selection to those groups. With a selection bias, groups score the
sum of their best two biased probabilities (one for a one-expert group); without it, the maximum probability.
Biases affect selection only. `reference_trace.py` tests both group-score formulas against the indices supplied
to the reference's experts; `describe_model.py` checks the observed geometry and formula against the GGUF.
A rule it cannot establish still prevents proposing a registry row. No new `ArchSpec` field is needed.
When `expert_weights_norm` explicitly specifies the behavior, the verified file value wins and an existing
registry's unused fallback stays unchanged. Such a file cannot establish the default of older files omitting it.
`tiny_checkpoint.py` removes a published checkpoint's quantization configuration before saving its random F32
weights, since those tensors are not quantized.
The group-routing golden cases are pinned on ARM64 NEON dotprod, x86 AVX2 and x86 scalar; streamed and resident
biased routing agree exactly on each platform.

Pinned official DeepSeek-V2-Lite-Chat (`85864749cd611b4353ce1decdb286193298f64c7`) and DeepSeek-V3
(`e815299b0bcbac849fa540c768ef21845365c9eb`) configurations produce the unchanged row:

```cpp
    {"deepseek2", RopeLayout::Adjacent, "ffn_norm", false},
```

At residual width 256, V2-Lite keeps its original 512-wide latent, single query projection and dense/MoE
blocks; its F32 checkpoint gives 512 identical IDs, 511 positions, mean KL 0.00000 and top-1 99.80% with F16 KV.
V3 keeps its original 1,536-wide query latent and 512-wide KV latent, three dense blocks and one MoE block with
sixteen experts in eight groups, four selected groups and eight selected experts. It gives 485 identical IDs,
484 positions, mean KL 0.00000 and top-1 99.38%. Both tokenizers match all three texts (617/1,013/241 IDs for V2,
485/783/217 for V3). The b11819 converter needs a local exact-fingerprint registration for the older V2 tokenizer;
the existing `deepseek-llm` pre-tokenizer then reproduces its IDs. No reference forward implementation is patched.
These are random-weight checks, without a real V2-Lite regression result.

Files predating split MLA projections describe the expanded head with `attention.key_length`, `value_length`
and one KV head per query head. `describe_model()` normalizes those dimensions to the cached latent and plans
`BlockPlan::latent_combined` from `attn_kv_b.weight`; ambiguous combined/split tensors are refused. The description
tool compares the corresponding expanded dimensions to the observed reference geometry. At block acquisition,
the key rows are dequantized and transposed to F32 and value rows retain their original bytes, then both use
the existing absorbed or expanded forward pass. These copies are bounded by one block, work after rebasing a
streaming slot, and count toward the expert-cache planning margin.

### Auxiliary index-key storage

`KvCacheConfig::auxiliary_dim` adds F32 channels after each cached latent key row. These channels are owned by
the paged cache and copied, saved and released with the main row. They are not attention values and do not take
part in per-channel main-key scaling. `KvCache::bytes_per_token()` and the image layout include their bytes;
the layout fingerprint separates these contexts from ordinary KV images. Mixed-format rows remain raw in the
prefix store so a codec cannot round an index key and change a later position selection. The cache tests cover
all four main KV types, page images, position copies, scaling and rollback.

### Sparse latent attention

`attention.indexer.head_count`, `key_length` and `top_k` describe the lightning indexer; tensors bind its query,
key and signed head-weight projections plus the key LayerNorm. `attention.indexer.types` distinguishes full
indexers from blocks sharing the preceding selection. The engine validates these flags against tensor presence
and refuses a shared first block. `attention.indexer.kpool > 1` is explicitly unsupported.

`reference_trace.py` recomputes the indexer's rectified, weighted causal scores and tests both rotary pairings
against its observed integer position selections. `ArchSpec::indexer_rope` records that pairing independently of
the main attention's, and `latent_norm_eps` records a latent RMS epsilon differing from the block norms.
The epsilon comparison permits GGUF's F32 metadata rounding, as the other scalar comparisons do.
Shared layers must pass the preceding layer's indices unchanged. The trace reconstructs query and KV latent
normalization and expanded MLA keys/values; the description compares both their geometry and attention scale
with the GGUF's split absorbed/expanded metadata. On the tiny GLM-5 checkpoint it proposes exactly:

```cpp
{.name = "glm-dsa", .rope = RopeLayout::Adjacent, .ffn_norm = "ffn_norm", .moe_norm_weights = true,
 .indexer_rope = RopeLayout::Adjacent, .latent_norm_eps = 1e-06f}
```

The standard F32 reference gate uses the original latent/indexer geometry, four blocks and `index_topk=256`:
493 scored positions, identical 494 token IDs, mean KL 0.00000, maximum 0.00001, top-1 99.80%. A separate
`index_topk=8` first-layer comparison checks all 494 causal selections exactly. The smaller full-layer top-k 8/32
diagnostics fail top-1 and remain numerical limitations, as recorded in the README; a rounded-KV diagnostic is
kept separate from the standard reference gate. No registry row is proposed when the latent decomposition or
indexer pairing cannot be established.

DeepSeek-V3.2 (`a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`) is checked with four blocks, residual width 256,
original query/KV/indexer widths and `index_topk=384`: 485 identical IDs, 484 scored positions, mean KL 0.00000,
maximum 0.00074, top-1 99.59%. At top-k 8 all 485 first-layer selections match the reference. The description
proposes exactly `{"deepseek32", RopeLayout::Adjacent, "ffn_norm", true}`. The converter's BOS assertion had to
be removed in a local copy because this official tokenizer explicitly sets `add_bos_token=false`; that rule
is preserved in both the reference and GGUF. This revision supplies no Jinja chat template. Top-k 256 diagnostics
at widths 256 and 512 give top-1 98.76% and remain below the required gate.

GLM-5.2 at revision `cf457fa734ab149ffef225f80893eb38c6ff5cdc` proposes the same `glm-dsa` row. Its four-block
tiny checkpoint keeps the original latent/index widths and top-k 256, with three full indexers and a shared
fourth block: 494 identical IDs, 493 positions, mean KL 0.00000, maximum 0.00008, top-1 99.80%. Its own template
passes 14 reference renders and its tokenizer matches the three tokenizer texts.

GLM-5.1 (`26e1bd6e011feb778d25ae34b09b07074139d92d`) also proposes the same row. Its four-block checkpoint at
top-k 256 gives 494 identical IDs, mean KL 0.00000, maximum 0.00001, top-1 99.80%, plus all three tokenizer texts
and 14 template renders matching.

Kimi K2 (`moonshotai/Kimi-K2-Instruct`, revision `fd1984e2b7a3350dbf7305fe73a4ede25c14de50`) uses the existing
`deepseek2` parts. Its older code's eager attention and direct `F.linear` router are observed and verified. A
two-block, width-256 checkpoint keeping the original latent widths gives mean KL 0.00000, maximum 0.00016 and
top-1 99.80% on 502 positions using reference IDs; the reference disables its obsolete cache API. This numerical
result is not a tokenizer/template gate: the `kimi-k2` pre-tokenizer and tokenizer-side tool sorting are an
integration dependency. The description correctly refuses support while that pre-tokenizer is absent.

Ling-mini-2.0 (`inclusionAI/Ling-mini-2.0`, revision `a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab`) proposes
`{"bailingmoe2", RopeLayout::Halves, "ffn_norm", true}`. Individual ModuleList expert calls are verified against
the router's token ordering and their weighted outputs reconstructed; each used expert's activation must agree.
The two-block tiny checkpoint uses width 512, four query/two KV heads of 128 channels, 64 rotary channels,
16 experts/top-8 and four of eight groups: F32 reference, F16 KV, 512 identical IDs, mean KL 0.00000, maximum
0.00040, top-1 99.61%. Fourteen template renders and all three tokenizer texts match. Its transformers-4-era
remote code needs compatibility for removed helpers, RoPE initialization and tied-weight metadata. Transformers 5
also materializes its nonpersistent RoPE buffer without calling the old constructor rule; the reference driver
restores that exact rule after loading. A reference with the resulting zero frequencies failed (KL 0.00470,
top-1 5.87%) and is retained separately from the corrected reference evidence. No engine numerical patch was
needed. Golden cases cover resident and streamed fused-QKV, Q/K-norm and grouped-routing execution.

The README's GLM-5.2 capacity estimates use the official configuration revision
`cf457fa734ab149ffef225f80893eb38c6ff5cdc`, without downloading its weights. There are 21 full indexers and
57 shared-indexer blocks. Matrix counts include untied embedding/output matrices, MLA projections, dense FFNs,
shared experts, routers and full indexer projections. Norms and router selection biases remain F32 in the
estimate. Routed expert parameters are `75 × 256 × 3 × 6144 × 2048`; a cold decode token reads eight of the
256 experts per MoE block. Format byte rates are F16/BF16 2, Q8_0 34/32, and Q2_K/Q3_K/Q4_K/Q5_K/Q6_K
84/110/144/176/210 bytes per 256 parameters. Actual GGUF sizes depend on their mixed tensor formats and whether
they include unexecuted MTP tensors. The current uniform cache layout reserves auxiliary channels in shared
blocks too; the estimate includes that allocation rather than assuming only full indexers consume key memory.
