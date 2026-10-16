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
linear scaling, Llama 3 `rope_freqs`, LongRoPE factors), the window, the MoE metadata, the tokenizer and each
block's plan, made the way `describe_model()` makes it. The RoPE frequencies the engine computes from the file are
compared with the reference's table, the windowed blocks' own table included. Every tensor must be one the engine
binds (`src/core/transformer.cpp`); one it would ignore is a missing part.

**A few weights of both checkpoints.** Read with HTTP range requests when the files are remote: two heads of the
first attention block's key projection and the first block's norm weights (a few MB at most).

* The key rows say how the converter stored the heads: in the reference's order, or interleaved (the Llama
  converter turns pairs `(i, i + d/2)` into `(2i, 2i + 1)`). With the reference's pairing this gives
  `ArchSpec::rope`. When no row order matches, the GGUF holds other weights than the reference, and nothing else
  can be trusted.
* The norm weights say which GGUF tensor holds which norm, so the role found in the trace becomes a tensor name
  (`ffn_norm`, `post_mixer_norm`, `post_ffn_norm`), and whether a norm the reference applies as `x (1 + w)` is
  stored as `1 + w`.

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
4. The forward pass: `tools/conform_reference.py` writes the transformers reference in float32 and `liyab-kl
   --kv f16 --max-kl 0.01 --min-top1 97` must pass (both thresholds are options).

The row is accepted only if every step passes. The GGUF must be a local file for the gate, BF16 or F16 (lower bit
widths add their own error), and of the weights the reference loads.

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

No published GGUF uses `general.architecture = mistral`: llama.cpp writes Mistral 7B and Nemo as `llama`, newer
Mistral models as `mistral3` or `mistral4` (more than 2,400 GGUF repositories found by Mistral-related searches on
Hugging Face were checked). The
`mistral` row was regenerated from Mistral-7B-Instruct-v0.3 with a copy of its GGUF's header whose architecture name
and metadata prefix were changed to `mistral`, holding the four tensors the tool reads. Its original GGUF gives the
`llama` row.

### The gate

| Model | Row | Tokenizer: conform / edge / chat | liyab-kl `--kv f16` | Verdict |
| :--- | :--- | ---: | :--- | :--- |
| Qwen3-0.6B (BF16) | registry's `qwen3` | 524 / 804 / 179, identical | 511 positions, mean KL 0.00000, top-1 99.80% | accepted |

### Models Liyab does not run

What the tool reports, in its words, for models whose architecture or tokenizer is not supported:

| Model (GGUF) | Not implemented in Liyab |
| :--- | :--- |
| Gemma 2 2B (`gemma2`) | attention logit soft-capping at 50; final logit soft-capping at 30. Everything else is expressible: GeGLU, √d embeddings, both sandwich norms, every 2nd block global (`swa_period` 2, window 4096) |
| DeepSeek-V2-Lite-Chat (`deepseek2`) | multi-head latent attention (`attn_kv_a_mqa`, `attn_kv_a_norm`, `attn_kv_b` unused; value heads of 128 with key heads of 192); YaRN RoPE scaling; attention scale 0.1147 instead of 1/√192. Its MoE is expressible: softmax top 6 of 64, no renormalization, an ungated shared expert, a dense first block |
| Mistral-Nemo-Instruct-2407 (`llama`) | the `tekken` BPE pre-tokenizer. The architecture is the `llama` row (head size 128 from `key_length`) |
| OLMo 2 1B (`olmo2`) | no norm before the mixer or the FFN (only after them); Q/K norms over all heads (2048) rather than per head (128); the `dbrx` pre-tokenizer |
| Granite 3.1 2B (`granite`) | attention scale 0.015625 instead of 1/√64; residual multiplier 0.22; embeddings × 12 (not √d); logits × 0.125; the `refact` pre-tokenizer |
| SmolLM3 3B (`smollm3`) | blocks without RoPE (block 3 of the 4 traced; the configuration's `no_rope_layers` lists every 4th); the `smaug-bpe` pre-tokenizer |
| SmolLM2 360M (`llama`) | the `smollm` pre-tokenizer. The architecture is the `llama` row |

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
* **Chat templates** are not checked (`liyab-tok --render` and `tests/data/chat_templates.json` do that).
* **The gate measures one text.** `tests/data/conform_text.txt` (512 tokens): long contexts past a model's sliding
  window or LongRoPE's original context are checked by hand (the README's correctness table has examples).
