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

The tool also checks blocks without rotation (the file's `attention.rope_pattern`, a periodic pattern, or global
blocks beside windowed ones), post-only branch norms, whole-projection Q/K RMSNorm, and embedding/residual/logit
scales against the reference. A file's logit scale is tested as both a multiplier and a divisor. YaRN's two
log-multiplier conventions are distinguished by the reference's rotation magnitude and attention scale. Windowed
blocks' unscaled rotary table can use the global base. A switch whose activation condition is absent cannot be
established from that checkpoint: the tool explicitly reports it as inactive and retains its registered value
(the engine default for a new row). It never uses a registered value to decide an active behavior.

Per-block `attention.head_count`, `attention.head_count_kv` and `feed_forward_length` arrays describe NAS geometry.
The planner distinguishes softmax attention, a single linear residual projection, and an absent mixer; zero-width
FFNs are absent. Only softmax-attention blocks own KV slots. Scratch buffers and cache images use the maximum
dimensions, with inactive KV heads stored as zeros; projections, normalization, grouped-query attention, key
calibration and relocation use the active dimensions of the block.
The independent float-reference check also runs through available GPU backends. Streamed block slots retain the
CPU projection route; its quantized activation arithmetic is the reference for the streaming check.
Latent attention still uses uniform head geometry; a file varying latent head counts is refused before loading
the uniform projection path.
Saved states include the file fingerprint when these arrays or modern scalar/rotation-pattern keys are present;
models with different geometry or scales cannot exchange states just because their dimensions and file sizes match.
For quantized KV caches, key calibration includes a conservative query-temperature gain over its covered context;
the measured channel bounds remain valid beyond the first temperature boundary.

A five-block Nemotron-NAS tiny case exercises two GQA ratios, a linear mixer, absent attention, a fully empty
block and different FFN widths. Its exact registry row passes the strict F32 gate: mean/max KL 0.00000,
top-1 99.80%, 499 conformance IDs and three tokenizer texts identical (498/797/202). The 14 template
cases match the existing fixture. Regeneration from the original configuration produces identical tensor bytes;
the official forward source is unchanged. The existing NAS golden case covers these numerical paths.
A linear mixer is established by recomputing its entire observed output from a unique linear projection,
including any bias. This also covers direct child `forward` calls that bypass module hooks. Ambiguous projections,
additional transformations and shape changes remain unknown; both one-branch and two-branch blocks use this rule.
`tests/test_reference_trace.py` checks these positive and negative cases with PyTorch and no model weights.
The modern model tokenizer rules (`smaug-bpe`, `dbrx`, `refact`, `seed-coder`, `exaone4`) are validated on all
three gate texts. SmolLM3's chat template is also recorded in `tests/data/chat_templates.json`; its 14 cases render
exactly as transformers does. An OLMo 2 base checkpoint has no chat template to render.
Clock-reading template fixtures store a fixed `clock_utc` instant; both reference generation and the C++ test
override `strftime_now` through their existing template context, independently of the current day and time zone.
The fixture helper uses `gmtime_s` on Windows and `gmtime_r` elsewhere; neither changes the runtime clock.
GLM-4's `glm4` name selects the Llama-3 split and whole-piece lookup, without Unicode normalization.
The tokenizer accepts the converter's integer Tekken scores as numeric values. Template fixtures cover GLM-4,
Gemma 1, Granite, OLMo 3, EXAONE 4 and Seed-OSS, including Seed-OSS's integer dictionary keys and thinking-budget
boundaries. `conform_reference.py` disables the unused generation cache during its single teacher-forced pass;
this also avoids generation-cache API differences in older official model implementations.
Its `--state-dict` option constructs a local model on CPU before loading safetensors and checks all parameters,
including tied aliases. This retains the authors' nonpersistent rotary buffers when newer transformers pretrained
loading does not. The trace observes native SDPA for classes that predate `AttentionInterface`, verifies repeated
KV heads before collapsing them for analysis, and leaves the attention computation unchanged. Tiny-checkpoint JSON
overrides run through the configuration constructor so nested block descriptions retain their required types.
`describe_model.py --gate --reference-state-dict` passes the local CPU loader through to the gate's reference run.
For RMSNorm weights with a scalar offset, the trace measures that offset against the full output and the tool
checks the converter's folded weights. Native SDPA observation also verifies the actual causal/window mask;
additional additive score biases remain unknown.
`chat_template_reference.py --trust-remote-code` enables official custom tokenizer code when recording templates;
the default still loads tokenizers without executing remote code.
For `chatglm-spm`, the initial-special-token rule is `[gMASK], sop`, independently of the ordinary BOS flag.
The tokenizer resolves these control tokens from the vocabulary, refuses an incomplete prefix, and inserts it
once; segmented encoding keeps an automatic prefix before the reusable text. The description tool compares the
scheme's prefix with the official tokenizer's actual special-token output.
ChatGLM3, EXAONE 3.5, MiniCPM, Nemotron-NAS and IQuest-Coder templates have 14 reference cases each.
ChatGLM3's tiny forward pass matches after adding its prefix, but the published converter also renames the
SentencePiece BOS/EOS strings and marks more literal spellings as control tokens than its official tokenizer
recognizes by default. Its three-text tokenizer gate therefore remains incomplete.
The `plamo3` tokenizer scheme reads the converter's raw vocabulary and scores, rounds scores to the reference's
integer units and finds the minimum-cost segmentation, preferring the longest prefix on ties. A Unicode character
without a piece uses byte fallback. Its scheme fixes boundaries around `<|plamo:...|>` spellings, runs of two spaces
and four repeated non-newline characters; raw whitespace is preserved. Missing byte tokens and invalid scores are
refused when loading. PLaMo3's block row combines existing sandwich norms, fused projections and local/global RoPE.
The converter does not record alternate boundary thresholds: the description tool refuses a reference using
values other than the `plamo3` scheme's two spaces and four repeated characters.
Norm names resolve both `.weight` and bare vector tensors through one helper shared by planning and loading. The
description tool verifies either spelling against the reference's offset-folded weights.
For `pfnet/plamo-3-610m-fin-instruct`, a tiny checkpoint with two blocks, window 8 and period 2 exercises both block
kinds: identical 512 reference IDs, mean KL 0.00000 and top-1 99.61% with F16 KV. Keeping the original eight-block
period instead gives mean KL 0.00000 (maximum 0.00005) and top-1 98.83%, below the tiny acceptance threshold. Its
real F16 checkpoint passes the gate with mean KL 0.00000, maximum 0.00002 and 100% top-1. The original-period file
proposes the registered `plamo3` row exactly; the two-block
file's explicit metadata overrides its period, and the tool proposes period 2 for that reduced configuration.
`conform_reference.py`, `tokenizer_reference.py --transformers` and `chat_template_reference.py` accept
`--slow-tokenizer` to use the model authors' original implementation when conversion to a fast tokenizer changes
its segmentation. `describe_model.py --reference-slow-tokenizer` uses it for settings, the three tokenizer texts
and the teacher-forced reference, so every part of a gate uses the same tokenizer.
For legacy eager attention, tracing observes the native `Q Kᵀ` and `probabilities V` matrix products. It retains
the actual Q/K/V tensors and accepts a standard scale and causal/window mask only after recomputing every softmax
probability; other score transformations remain unknown. The authors' forward calculation is unchanged.
Packed GQA source projections are checked against both contiguous and per-group Q/K/V views, and only an exact
numerical match establishes the layout. Weight comparison reads those measured rows even when the converter
splits the projection into separate GGUF tensors. A requested remote slow tokenizer is loaded from its declared
Python class, with explicit trust required; a library silently selecting a fast class is an error.
InternLM2's original slow tokenizer matches 575, 989 and 190 IDs on the three gate texts. Its tiny checkpoint gives
identical 512 reference IDs, mean KL 0.00000 and 100% top-1, and proposes the registered row exactly. Its real-model
gate remains pending; dynamic RoPE growth beyond the original context remains a reported limitation.
The teacher-forced reference selects `AutoModelForImageTextToText` when a multimodal checkpoint has no
`AutoModelForCausalLM` class. Its tokenizer IDs are passed directly to the model without images or an image
processor, so this measures the text-only route of the authors' original implementation.
`chat_template_reference.py --model NAME=TOKENIZER:TEMPLATE` renders that explicit file with transformers'
Jinja renderer. This also covers Mistral's backend, whose native chat formatter ignores a Jinja template
argument. InternLM2 and Ministral 3 each have 14 new fixtures, all byte-identical in the engine.
To expose a window boundary within the 13-token trace, the tracer reduces a uniform scalar `sliding_window` /
`window_size` through the configuration's constructor, verifies the resulting mask, then maps that observed size
back to the original configured size. This keeps local/global blocks distinguishable even when the full window
is thousands of tokens. It changes only the tracing configuration; the teacher-forced reference and engine gate
run the checkpoint's original window. An OLMo 3 regression verifies three local blocks and a global block; the
real PLaMo 3 description now derives its exact local-base/period row instead of conflating short causal masks.
For configurations with `scale_depth`, both residual branches must numerically match
`scale_depth / sqrt(traced_layer_count)` before the tool maps that multiplier to the original layer count.
An unmatched rule remains unknown. This preserves MiniCPM's full-stack multiplier when tracing only two blocks;
the teacher-forced reference continues to run the original checkpoint and configuration.
Blocks without either the mixer or the FFN cannot establish a two-branch depth rule and leave it unknown.
The `gpt2-nfkc-digits` scheme applies Unicode 9 NFKC before individual-digit splitting and GPT-2 ByteLevel.
Compatibility decompositions and Unicode 9 assignment ranges are appended to the existing Unicode tables;
canonical decomposition, ordering and composition are shared with NFC. Newer characters remain inert, matching
the reference normalization dependency. Unicode 16 normalization differs on newer characters and is not used
for this tokenizer mode. The description tool verifies this exact source JSON pipeline and reports a
wrong file tag, incompatible source, or unavailable source proof. Active corrected tokenizer modes also remain
unknown when the reference tokenizer itself cannot load. Neither rule depends on a model name.
Normalization preserves malformed bytes while still composing valid characters after them.
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
its existing plain ByteLevel rule. The full EXAONE 3.5 checkpoint gate remains pending.
The explicit original slow-tokenizer path is unchanged.
`python3 tests/test_reference_tokenizer.py` checks matching, missing and discarded serialized pipelines without weights.

`tools/gen_text_tests.py --nfkc-only` generates the additive NFKC expectations and source pipeline splits in
`tests/data/text_cases.txt`; the default generator includes these after all existing records.
When a legacy model does not implement its input-embedding accessor, the trace identifies a unique observed
`Embedding` call on the actual token IDs, before any first-block child runs and with the first block's input shape.
Repeated calls to the same table count once; different candidate tables remain unknown. The existing full-tensor
embedding-scale check still applies, so positional additions or other input transformations cannot be hidden.
An unavailable output accessor similarly requires exactly one observed linear projection after the last block,
with the logits' shape. Its norm input, possible input multiplier, bias and entire logit transform are still
verified numerically. Multiple candidate calls remain unknown, including repeated calls to the same head.
Byte-token mismatch diagnostics may contain individual UTF-8 bytes. Gate subprocess output escapes those bytes
when decoding its log, preserving the actual failed check instead of aborting with a Unicode decoding error.
For SentencePiece references, the tool reads the source model's piece flags with `sentencepiece` and `protobuf`.
If every ordinary BPE piece is `USER_DEFINED`, SentencePiece freezes the longest match rather than applying merge
ranks. The corresponding file mode is `tokenizer.ggml.pre=spm-greedy`, with all byte fallback tokens, identity
normalization, escaped spaces and preserved whitespace. The tool rejects an incompatible reference or leaves the
rule unknown if its source flags cannot be read. A converter that drops those flags needs a corrected metadata
copy; the tensor data does not change. Tokenization runs on the CPU with every backend.
InternLM3's two-block F32 checkpoint uses the existing `llama` row. With `spm-greedy` and
`tokenizer.ggml.add_space_prefix=false`, the strict tiny gate gives mean/max KL 0.00000 and top-1 100% over
511 positions, with 512 BOS-inclusive IDs identical. The three full tokenizer texts match 606/1,007/205 IDs
through the author's original slow tokenizer; all 14 chat-template cases render identically. The metadata copy
preserves all 21 tensor payloads byte for byte. This validates the declared 32,768-position context only:
the reference's dynamic-NTK recomputation beyond that limit is reported separately and is not implemented.
The `deci` row also passes a three-block DeciLM-7B F32 tiny gate with 2/1/4 KV heads: mean/max KL 0.00000,
top-1 100%, 512 BOS-inclusive IDs identical, and all three full tokenizer texts identical (629/1,032/279 IDs).
Its 14 template cases match transformers. The b11819 Python converter needs a local vocabulary override:
use `_set_vocab_sentencepiece()` when the source includes `tokenizer.model`, retaining its actual scores instead
of `LlamaHfVocab`'s constant scores. The tensor conversion remains unchanged (all 30 payload hashes identical).
Transformers 5 also needs the concrete remote causal class's `config_class = DeciLMConfig`; this registration
compatibility change does not alter its forward pass. The strict gate remains within the declared context.
The full F16 gate for `openbmb/MiniCPM-1B-sft-bf16` (revision `85252935bfc9acb3ee93dcea4b793fb87a0b43e8`) proposes the exact `minicpm` row.
It gives mean/max KL 0.00000/0.00002, top-1 100.00%, 512 BOS-inclusive IDs identical,
and three full tokenizer texts identical (631/1049/248). The source and GGUF chat templates and
special strings match; all 14 fixture cases agree (14 rendered, 0 correctly refused).
For this reference run under transformers 5, a local copy of the legacy Python class treats normalized
`rope_type=default` as the original unscaled branch and declares
`_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`. These two API adaptations leave its forward
equations unchanged. The source's 470 tensors retain their exact dtypes and values when transcoded to safetensors;
the unpatched b11819 converter then writes the F16 GGUF. The full 52-layer residual multiplier is verified as
`1.4 / sqrt(52)`, with embedding multiplier 12 and logit multiplier 1/6.
The full F16 gate for `internlm/internlm2_5-1_8b-chat` (revision `57dcbc5455bf050d27a80f0ebd7808387caa714d`) proposes the exact `internlm2` row.
It gives mean/max KL 0.00000/0.00001, top-1 100.00%, 512 BOS-inclusive IDs identical,
and three full tokenizer texts identical (575/989/190). The source and GGUF chat templates and
special strings match; all 14 fixture cases agree (14 rendered, 0 correctly refused).
A metadata-only copy restores source EOS 2 and unknown token 0 and preserves chat EOS 92542 as EOT, matching
the source generation stop set [2, 92542]. b11819 otherwise replaces EOS 2 with 92542 and omits unknown token 0.
The unmodified file passes numerical and three-text checks but differs in template context and misses the
original EOS stop. All 219 tensor payload hashes are unchanged.
The full F16 gate for `unsloth/gemma-1.1-2b-it` (revision `619e546640669a280738627cc623f4bd74a7b069`) proposes the exact `gemma` row.
It gives mean/max KL 0.00000/0.00000, top-1 100.00%, 512 BOS-inclusive IDs identical,
and three full tokenizer texts identical (513/786/211). The source and GGUF chat templates and
special strings match; all 14 fixture cases agree (1 rendered, 13 correctly refused).
GraniteSWA is deferred to integration with learned attention sinks. The current `GraniteSWAAttention` always
allocates a per-head `sinks` parameter, uses it as an extra softmax-denominator logit and passes it as `s_aux` to
registered attention. Its configuration also describes interleaved windows and per-layer `layer_rope_theta`
(zero means NoPE); the converter writes a RoPE mask from its older `no_rope_layers` field. The exact checkpoint's
metadata must therefore be checked at that integration. No GraniteSWA conformance result is claimed here.

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
| SmolLM3-3B (F16, b11819 conversion of revision `a07cc9a`) | registry's `smollm3`, exact | 498 / 797 / 165, identical | 497 positions, mean KL 0.00000, maximum 0.00001, top-1 100.00% | accepted |
| PLaMo-3-610m-fin-instruct (F16, b11819 conversion of revision `a000b59`) | registry's `plamo3`, exact | 521 / 950 / 251, identical | 511 positions, mean KL 0.00000, maximum 0.00002, top-1 100.00% | accepted |
| OLMo-2-0425-1B (F16, b11819 conversion of revision `a1847df`) | registry's `olmo2`, exact | 544 / 882 / 187, identical | 511 positions, mean KL 0.00000, maximum 0.00003, top-1 100.00% | accepted |
| EXAONE-4.0-1.2B (F16, b11819 conversion of revision `3abf281`) | registry's `exaone4`, exact | 618 / 993 / 254, identical | 511 positions, mean KL 0.00001, maximum 0.00018, top-1 100.00% | accepted |
| openbmb/MiniCPM-1B-sft-bf16 (F16, b11819 conversion of revision `8525293`) | registry's `minicpm`, exact | 631 / 1049 / 248, identical | 511 positions, mean KL 0.00000, maximum 0.00002, top-1 100.00% | accepted |
| internlm/internlm2_5-1_8b-chat (F16, b11819 with EOS/EOT metadata correction, revision `57dcbc5`) | registry's `internlm2`, exact | 575 / 989 / 190, identical | 511 positions, mean KL 0.00000, maximum 0.00001, top-1 100.00% | accepted corrected metadata |
| unsloth/gemma-1.1-2b-it (F16, b11819 conversion of revision `619e546`) | registry's `gemma`, exact | 513 / 786 / 211, identical | 511 positions, mean KL 0.00000, maximum 0.00000, top-1 100.00% | accepted |

EXAONE-4.0-1.2B uses full rotary attention throughout. Its row also describes hybrid configurations with windowed
rotary blocks and global unrotated blocks; the four-block/window-64 tiny case exercises both kinds with mean
KL 0.00000 and 99.80% top-1. The real gate and tiny hybrid check the same post-only norms and per-head Q/K norms.

### Models Liyab does not run

What the tool reports, in its words, for models whose architecture or tokenizer is not supported:

| Model (GGUF) | Not implemented in Liyab |
| :--- | :--- |
| Granite 3.1 2B (`granite`) | attention scale 0.015625 instead of 1/√64; residual multiplier 0.22; embeddings × 12 (not √d); logits × 0.125; the `refact` pre-tokenizer |

SmolLM3 moved out of this table after adding its unrotated blocks and `smaug-bpe` pre-tokenizer. The tool now
proposes its registered row exactly and its full F16 checkpoint passes the gate above; its 14 chat-template
fixtures also render byte for byte like transformers.

OLMo 2 moved out after adding post-only branch norms, whole-projection Q/K norms and the `dbrx` pre-tokenizer.
The real OLMo-2-0425-1B F16 checkpoint reproduces the registered row exactly and passes the gate above. This base
checkpoint has no chat template. OLMo 3's local/global window pattern is covered separately by tiny conformance.

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
