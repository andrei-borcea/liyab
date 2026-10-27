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
* [State-space mixers](#state-space-mixers)

## State-space mixers

The `mamba` and `mamba2` rows describe stacks without an FFN. A block with
`ssm_conv1d` and `ssm_d` is Mamba-1 when it also has `ssm_x`, otherwise
Mamba-2. `ssm_conv1d` without `ssm_d` remains Gated DeltaNet. Attention
projections select attention; an FFN without mixer tensors is an FFN-only
block. The loader rejects empty blocks and mixed SSM geometries.
Any attention projection identifies attention, including K/V/output: an
incomplete set fails binding rather than silently becoming an FFN-only block.
Likewise, a partial FFN or SSM projection set remains that part and fails
validation/binding; expert projections require a router. Parallel attention
and Mamba tensors in one block are explicitly refused until that topology
is implemented.

Mamba-1 uses the file's convolution, input, selective, time-step and output
projections; `ssm_a` already holds `-exp(A_log)`, as converter b11819 writes
it. Mamba-2 splits the inner stream into heads, with groups sharing B/C,
per-head decay and D skip, then gated RMS normalization. Both use existing
matmuls for every projection format. Scans and convolution run on CPU
(NEON, AVX2/FMA or scalar); GPU routes can still project on Metal/Vulkan.

Falcon-Mamba's `ssm.dt_b_c_rms` enables weightless dt/B/C RMS norms. The
epsilon is `ArchSpec::ssm_dt_b_c_eps` (1e-6); b11819 does not write
`mixer_rms_eps`. Current transformers' saved norm weights are dummy tensors
and do not affect these norms. Other weighted dt/B/C norms use the file's
RMS epsilon. Nondefault dt clipping is not expressible by b11819 and the
description tool refuses to propose a row for it. `ssm_norm_grouped`
distinguishes a gated norm over each group from transformers Mamba-2's norm
over all inner channels. The trace recomputes the gated norm to establish
this fact; the converter reshapes both forms identically.

The reference trace recognizes the projections and state tensors, observes
the mixer-only residual, and compares SSM geometry, activation and norm
settings with metadata. A pure SSM trace reduces its residual width while
preserving expansion and head/group divisibility, as does
`tiny_checkpoint.py --width`. `test_ssm` checks scans
against independent scalar equations and compares prefill, decode and
rollback for all 26 projection formats and four KV cache types. Reader-only
formats use `ssm_quant_vectors.bin`, derived from the existing independent
`quant_vectors.bin` blocks through an opt-in test-writer replacement. Every
block uses the same RMS target, `0.5 / sqrt(256)`, before scale rounding;
integer codes/grids are preserved. This bounds repeated correlated rows
without changing tolerances or selecting a rule by format or platform.
Normal synthetic fixtures, original quant vectors and golden hashes remain
unchanged. Regenerate with
`python tools/gen_ssm_quant_vectors.py --out tests/data/ssm_quant_vectors.bin`
using gguf 0.19.0 and NumPy. The generator
checks the source reference values and independently decodes the scaled
blocks with gguf-py. The ordinary writer still uses its eight implemented
encoders and aborts with a named diagnostic when a reader-only format lacks
an explicit encoded payload; production quantization behavior is unchanged.
Tests require nonzero decoded weights,
projection outputs, convolution history and scan state, and verify that
ablating each projection or the carried scan state changes the logits. Existing
context, tree-draft and prefix-cache tests also exercise Mamba states.
Synthetic golden cases combine SSM, attention and FFN-only blocks, including
MoE FFNs with in-place and streamed experts.
GPU prefill and decode projection checks use F32 throughout, keeping projection roundoff
separate from discrete Q8 activation rounding in the output head. Each GPU
backend is destroyed before the model mappings whose buffers it wraps.
The CLI labels their phase `recurrent`. The C API field `delta_net_ms` and
JSON key `phases_ms.delta_net` keep their names and include all recurrent
mixers; neither the ABI layout nor existing JSON keys change.

Rows established against transformers 5.19.0 and converter b11819:

```cpp
{"mamba", RopeLayout::Halves, "", false},
{.name = "mamba2", .rope = RopeLayout::Halves, .ffn_norm = "", .moe_norm_weights = false,
 .ssm_norm_grouped = false},
```

The following comparisons use 511 prediction positions and F16 KV. Every
case has identical token IDs. The real checkpoints also pass
`describe_model.py --gate`, including its three tokenizer texts and the
exact registry-row comparison.

| Checkpoint / tiny configuration | GGUF | Mean KL | Top-1 |
| :--- | :--- | ---: | ---: |
| Tiny `state-spaces/mamba-130m-hf` | F32 | 0.00000 | 100% |
| Tiny `tiiuae/falcon-mamba-7b`, weightless mixer norms | F32 | 0.00000 | 100% |
| Tiny `AntonV/mamba2-130m-hf` | F32 | 0.00000 | 100% |
| Tiny Mamba-2, two B/C groups and projection biases | F32 | 0.00000 | 100% |
| Tiny `mistralai/Mamba-Codestral-7B-v0.1`, two groups | F32 | 0.00000 | 100% |
| `state-spaces/mamba-130m-hf` | F16 | 0.00001 | 99.61% |
| `state-spaces/mamba-370m-hf` | F16 | 0.00000 | 100% |
| `AntonV/mamba2-130m-hf` | F16 | 0.00000 | 100% |

The AntonV configuration omits `architectures`; the validation copy adds
`["Mamba2ForCausalLM"]` before conversion. Its weights and forward settings
are unchanged. No converter code is patched. Tiny widths are 64, inner
widths 128, and two blocks; Mamba-1 has dt rank 8, Mamba-2 has four 32-wide
heads, state size 16 and reference chunk size 16. SSM goldens cover both
norm groupings, weighted and weightless dt/B/C norms, convolution without
history, biases and mixed topology on ARM NEON, x86 AVX2 and x86 scalar.
Nine SSM golden cases are pinned on each platform; all 111 pre-existing
hashes remain unchanged. The README correctness table includes the three
real F16 checkpoints above; its testing table lists the focused `test_ssm`
suite. The final experimental release passes all 15 test modules on native
ARM64 and Rosetta x86-64 with AVX2 enabled and disabled, with actual Metal
access. The Android ARM64 experimental build also passes; Vulkan execution
remains an on-device integration check.

The converter calls the GPT-NeoX/Mamba ByteLevel tokenizer `olmo` (the
existing GPT-2 regex after NFC). Falcon-Mamba's `falcon` pipeline isolates contiguous
punctuation, applies ByteLevel, isolates number runs, then ASCII triples.
The last two steps classify ByteLevel's encoded bytes, so the table marks
them as byte-alphabet steps. They may split inside an original UTF-8
character; the BPE encoder receives the exact resulting byte ranges.
`tools/gen_text_tests.py` checks both against the Rust tokenizer pipelines;
the resulting split records are appended to `tests/data/text_cases.txt`.
These base checkpoints supply no chat template; no template is invented.
The gate escapes non-UTF-8 byte-token diagnostics instead of failing while
decoding a tokenizer tool's output; token IDs are still compared exactly.

The next hybrid families remain separate validation work. This section does
not register them merely because they contain an SSM:

| Family | Remaining parts and ownership |
| :--- | :--- |
| Jamba | Per-block KV-head arrays and attention without RoPE (modern); architecture facts and conformance of weighted dt/B/C norms and dense/MoE combinations (ssm/moe). |
| Falcon-H1 | Attention and SSM in parallel inside one block, partial SSM widths and D layouts (ssm); verify the converter's baked multipliers against the reference. |
| Granite 4 hybrid | Per-block head arrays and NoPE (modern); embedding/residual/logit multipliers (classic/modern), then shared/routed-expert conformance (moe). |
| Nemotron-H | Ungated ReLU-squared FFN (classic), per-block head metadata (modern); grouped SSM norm is implemented here. |
| Nemotron-H-MoE | Nemotron-H parts plus its expert/routing variants (moe; group-limited selection deepseek-dsa). |
| Plamo2 | Per-block head arrays (modern); its distinct headwise B/C, dt/B/C normalization and SSM projection/state layout (ssm). |
| Bamba | Its converter uses `granitehybrid` with per-block KV-head arrays (modern); register only after its full hybrid facts and conformance are established. |

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
Seed-OSS and GLM-4-9B-0414 have F32 tiny-checkpoint conformance: mean KL 0.00000, top-1 100.00% and 99.80%,
respectively, with 512 and 498 identical conformance IDs and all three tokenizer texts matching. Their template
fixtures cover 19 and 14 requests. No full-weight checkpoint was measured for these families or for Deci/Nemotron-NAS.
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
gate also passes with the EOS/EOT metadata correction recorded below; dynamic RoPE growth beyond the original
context remains a reported limitation.
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
its existing plain ByteLevel rule. The full corrected-file result is recorded below.
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
The full F16 gate for `ibm-granite/granite-3.3-2b-instruct` (revision `707f574c62054322f6b5b04b6d075f0a8f05e0f0`) proposes the exact `granite` row.
It gives mean/max KL 0.00000/0.00004, top-1 100.00%, 512 BOS-inclusive IDs identical,
and three full tokenizer texts identical (625/971/247). The source and GGUF chat templates and
special strings match; all 14 fixture cases agree (14 rendered, 0 correctly refused).
The full F16 gate for `mistralai/Ministral-3-3B-Instruct-2512-BF16` (revision `b6d637bef2393152b3da2b2fde72eecdee30557e`) proposes the exact `mistral3` row.
It gives mean/max KL 0.00000/0.00003, top-1 100.00%, 512 BOS-inclusive IDs identical,
and three full tokenizer texts identical (517/879/235). Actual GGUF template renders and
special strings match the author reference; all 14 fixture cases agree (14 rendered, 0 correctly refused).
The unmodified b11819 HF-format path selects a different Devstral community template and fails author-output comparisons. Its native tekken vocabulary reader is selected by temporarily hiding the own tokenizer.json and restoring it before references. This gate changes only the template metadata slot to the official author Jinja, lexer-minified with the same AST and all 14 rendered outputs; all 236 tensor descriptors and the entire tensor data section are unchanged (SHA256 0ee6f18db9635f745dcc81d6904048ebcc66fc004d17247df1ea54f17e0c72c8). The private generic converter shim instead prefers a local author Jinja file after the existing vocabulary method, with no tensor changes. The existing `ministral3` fixture holds the full 11912-byte official author template (SHA256 0701cfbdc2b7d44fdbad104dff604faee4b0543e8247624568777fe465746f9b). The `ministral3-real` fixture tests the compact equivalent stored in the corrected GGUF. The Devstral fallback remains a separately recorded failed-export comparison.
The native tokenizer exposes no Jinja attribute. Template proof uses the checkpoint's own `chat_template.jinja` explicitly with transformers Jinja, rather than asserting equivalence to the separate native tokenizer chat formatter.
The full F16 gate for `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` (revision `ccce25bd39c141fe053e0bc75818a8f5fe962802`) proposes the exact `exaone` row.
It gives mean/max KL 0.00000/0.00003, top-1 100.00%, 512 BOS-inclusive IDs identical,
and three full tokenizer texts identical (615/917/243). Actual GGUF template renders and
special strings match the author reference; all 14 fixture cases agree (14 rendered, 0 correctly refused).
The standard b11819 tag `exaone4` loses the source NFKC/individual-digit pipeline. This real gate uses the metadata-only converter override `modern-nfkc-converter.py`, writing `gpt2-nfkc-digits` directly to avoid a duplicate full GGUF. The tiny direct override is byte-identical to the independently corrected unmodified export; all21 tensor payloads are unchanged. The unmodified real export was not separately gated and is not claimed to pass.
For transformers 5 API compatibility, the private converter keeps the original AutoTokenizer factory but explicitly trusts the pinned official LGAI-EXAONE configuration. b11819 otherwise falls back to a generic configuration that cannot initialize legacy rope_scaling before max_position_embeddings exists. No source configuration, forward function or tensor conversion code is patched.
The model has 80 channels per head. The existing Transformer loader selects F16 KV when a requested block-quantized KV row is not a multiple of 32. Q8_0, Q4_0 and Q4_1 requests each reproduce the F16 gate metrics and 512 source IDs; this is F16 fallback coverage, not native 80-channel quantized-KV support.
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
| ibm-granite/granite-3.3-2b-instruct (F16, b11819 conversion of revision `707f574`) | registry's `granite`, exact | 625 / 971 / 247, identical | 511 positions, mean KL 0.00000, maximum 0.00004, top-1 100.00% | accepted |
| mistralai/Ministral-3-3B-Instruct-2512-BF16 (F16, b11819 with author-template metadata correction, revision `b6d637b`) | registry's `mistral3`, exact | 517 / 879 / 235, identical | 511 positions, mean KL 0.00000, maximum 0.00003, top-1 100.00% | accepted corrected template; original community template differs |
| LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct (F16, b11819 with NFKC metadata correction, revision `ccce25b`) | registry's `exaone`, exact | 615 / 917 / 243, identical | 511 positions, mean KL 0.00000, maximum 0.00003, top-1 100.00% | accepted corrected file; original tag incompatible |

EXAONE-4.0-1.2B uses full rotary attention throughout. Its row also describes hybrid configurations with windowed
rotary blocks and global unrotated blocks; the four-block/window-64 tiny case exercises both kinds with mean
KL 0.00000 and 99.80% top-1. The real gate and tiny hybrid check the same post-only norms and per-head Q/K norms.
| Gemma 3n E2B it (unsloth, converted BF16) | registry's `gemma3n` | 523 / 801 / 211, identical | 511 positions, mean KL 0.00000, maximum 0.00001, top-1 100.00% | accepted |

### Models Liyab does not run

What the tool reports, in its words, for models whose architecture or tokenizer is not supported:

| Model (GGUF) | Not implemented in Liyab |
| :--- | :--- |
| ChatGLM3 (b11819 export) | Raw control spellings and recognition differ from the original tokenizer; its three-text gate remains incomplete. |
| IQuest-Coder (b11819 export) | The source added-token `single_word` boundaries are absent in the file; its tokenizer gate remains incomplete. |

Granite's former entry required metadata attention scaling, embedding/residual/logit multipliers and the `refact` pre-tokenizer. These generic parts now pass the full Granite 3.3-2B gate above; Granite 3.1 itself was not separately gated.

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
0.00000, top-1 100.00% with `liyab-kl --kv f16` (511 positions). For latent attention the tool reconstructs expanded
keys and values from the observed normalized shared latent, the query's low-rank path and rotary suffix, and
checks the absorbed GGUF geometry separately from the expanded reference geometry. It compares the combined
KV projection's rows with the checkpoint before proposing the RoPE pairing. Numerical conformance still uses
`liyab-kl` ([README, Correctness](../README.md#correctness)).

### Modern block regression check

The modern block, tokenizer and saved-context changes pass 14/14 test suites on native ARM64 with Metal
available (135.71 s), Rosetta AVX2 (343.65 s) and Rosetta scalar (236.38 s). The experimental Android arm64
build passes with four build jobs. There are 49 golden cases per platform: all 111 earlier pins are unchanged,
and 12 new cases add 36 pins. The saved-main Qwen3-0.6B BF16 reference still gives identical 512 IDs, mean/max
KL 0.00000, top-1 100% and perplexity 27.119 with F16 KV.

Performance compares the frozen main and branch binaries back to back on an M4 Pro. Each configuration has
a separate warmup pair and six balanced AB/BA pairs, using four threads. CPU `liyab-bench` measures 512-token
prefill with `--repeat 1`; CLI measurements use an ordinary preselected request per model, a 128-token target,
at least 64 naturally generated tokens, temperature 0 and seed 42. Main and branch replies and all six work
counts match: prompt, cached-prefix, generated, forced, proposed-draft and accepted-draft tokens.

Throughputs below are separate main/branch medians in tokens/s. Changes are medians of the six paired
percentage changes, so they need not equal the percentage change of those throughput medians.
BF16 Metal-route measurements include the existing CPU matrix fallback; F16 rows verify actual Metal
projections. No Vulkan runtime measurement was available on this Mac.

| Model | Route | Prefill main → branch | Paired change | Decode main → branch | Paired change |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | CPU CLI | 91.29 → 91.32 | -0.112% | 57.06 → 57.11 | +0.170% |
| Qwen3-0.6B | CPU bench, 512 tokens | 87.95 → 87.90 | -0.114% | — | — |
| Qwen3-0.6B | BF16 Metal route / CPU matrix fallback | 91.33 → 91.23 | -0.113% | 57.43 → 57.18 | -0.318% |
| Llama-3.2-1B-Instruct | CPU CLI | 37.61 → 37.57 | -0.139% | 28.19 → 28.19 | -0.027% |
| Llama-3.2-1B-Instruct | CPU bench, 512 tokens | 36.75 → 36.80 | +0.272% | — | — |
| Llama-3.2-1B-Instruct | BF16 Metal route / CPU matrix fallback | 37.65 → 37.62 | +0.068% | 28.29 → 28.20 | -0.266% |
| Qwen3.5-0.8B | CPU CLI | 84.73 → 84.23 | -0.338% | 53.43 → 53.29 | -0.444% |
| Qwen3.5-0.8B | CPU bench, 512 tokens | 83.90 → 83.85 | -0.059% | — | — |
| Qwen3.5-0.8B | BF16 Metal route / CPU matrix fallback | 84.25 → 84.11 | -0.178% | 53.50 → 53.36 | -0.434% |
| Qwen3-0.6B | F16 Metal projections | 277.96 → 272.59 | -0.668% | 27.06 → 27.30 | +2.200% |
| Llama-3.2-1B-Instruct | F16 Metal projections | 152.72 → 152.47 | -0.219% | 32.65 → 32.48 | -0.771% |
| Qwen3.5-0.8B | F16 Metal projections, AC confirmation | 272.60 → 272.35 | -0.085% | 33.13 → 33.02 | -0.231% |

The original 72-pair matrix had 20 of 21 metrics within the symmetric ±3% bound. Its Qwen3.5 F16 Metal
decode result was faster by +6.2486% (23.265 → 25.259 tokens/s), outside that band; this result is preserved.
Its prefill medians were 228.58 → 258.05 tokens/s, with a +0.851% paired median change and large individual
outliers. The original matrix has no per-sample power telemetry, so a specific cause is not assigned to it.
After a host battery/power-mode confound was identified, only that configuration received a separately
authorized AC confirmation: the same frozen binaries, request (317 prompt IDs), context 1024 and flags,
one warmup pair and six measured pairs. All 28 read-only power observations before/after those 14 runs
show AC power; no system power settings changed. The last row reports that confirmation. The other 20
original metrics remain accepted, and all 21 effective metrics are within ±3% (-0.7714% to +2.1996%).
All 60 measured CLI reply pairs and six work counts agree across the original and confirmation runs.

## Limits

### Gemma residual and embedding parts

`gemma3n` plans AltUp from `altup_proj` and its stream-count/active-index metadata, Laurel from each block's
`laurel_l`, per-layer token inputs from `embedding_length_per_layer_input` and `per_layer_token_embd`, and gate
sparsity from the converter's per-block Gaussian standard-deviation multipliers. `attention.shared_kv_layers`
maps each final block to the last non-sharing block with the same window kind. Only source blocks allocate KV
slots; sharing blocks contribute their query ranges to those slots during quantized-cache calibration.
Per-layer tables remain in the file mapping; each pass materializes only its token rows and projected inputs.

`gemma4` also plans absent value projections as values from the key projection, before normalization and RoPE,
and binds optional `layer_output_scale`. Different local/global head geometry and different per-block FFN widths
are explicitly rejected pending their generic implementation; the row does not assert support for such files.
Its tokenizer applies the explicit `tokenizer.ggml.merges` ranks after SentencePiece space normalization, rather
than selecting merges by vocabulary score.
With uniform local/global geometry, the four-block Gemma 4 tiny check gives mean KL 0.00000, maximum KL 0.00006,
top-1 agreement 99.22%, and 512 identical token IDs; the conformance, edge and chat tokenizer texts also match.
For this check only, a copy of converter b11819 accepts transformers 5.19's homogeneous configuration serialization
(`head_dim` without `global_head_dim` or explicit per-layer overrides) and records the tokenizer's actual BOS rule.
The stock converter forces BOS where that tokenizer export has no BOS postprocessor. Liyab follows the GGUF's BOS
flag; it does not override a published file's metadata. The shared converter is unchanged.
For text conformance of multimodal checkpoints, `conform_reference.py` uses the text configuration's registered
causal LM and matches its input-embedding parameter suffix to the checkpoint keys to establish the namespace.
An ambiguous match is refused. Vision/audio towers and their optional dependencies are not instantiated;
`--config` changes the selected text configuration.

`gemma-embedding` uses bidirectional attention. Pooling runs an entire input in one pass and shares no prefix
because every hidden state depends on later tokens. Windowed blocks see half the stored window on each side.
Incremental continuation, causal tree drafts and rolling caches are refused; calibration of quantized keys reads
the complete calibration input in one pass too.
The pooling layer validates and applies optional `dense_2.weight` and `dense_3.weight` in order before truncation
and normalization; `Engine::embedding_size()` reports the projected width. Biases on these dense layers are
explicitly refused. Its BF16 real-model check covers 39 inputs: identical tokenizations and minimum cosine
0.99992651 against the independent float32 reference on CPU and Metal.

The description tool establishes direction from the traced attention contract, unscaled scores from the actual
attention scale, and value normalization by matching projected values with and without unweighted RMS norm.
The trace reconstructs AltUp predictions from the coefficient projections, checks the corrected stream outputs
and per-layer residual additions, identifies Laurel in the mixer residual, and matches shared KV to an earlier
attention call. A topology it cannot establish is reported as unknown and no row is proposed.

The four-block Gemma 3n tiny check exercises all these paths: 512 identical token IDs, mean KL 0.00000,
maximum KL 0.00001 and top-1 agreement 99.41% with F32 weights/F16 KV. EmbeddingGemma's BF16 conversion matches
all 39 pooling-case tokenizations on CPU and Metal (minimum cosine 0.99992651 against float32); the independent
reference matches sentence-transformers at minimum cosine 0.9999998. Gemma 3n E2B it, converted from unsloth
revision `c4291af6285b4f90d88bdf6f9a634a8d2f262cc8` to BF16, also passes the real-model gate: the tool proposes
exactly the registered row, all three tokenizer texts match, and the 512-token float32-reference comparison gives
mean KL 0.00000, maximum 0.00001 and top-1 100.00%. Gemma 4's heterogeneous geometry remains pending in the
generic per-block geometry work. The separate K-equals-V/no-per-layer-input Gemma 4 tiny case gives mean KL
0.00000, maximum 0.00002, top-1 99.02%, and 512 identical IDs.
The additional Gemma 3n BF16-reference comparison gives mean KL 0.00168, maximum 0.02824 and top-1 98.24%.
CPU and Metal routes both greedily answer the checked chat prompt with "The capital of France is Paris."
The deeper ten-block random diagnostic is retained as a failure: mean KL 0.00001, maximum 0.00064 and top-1
91.78%. Rounding its reference keys and values to F16 improves top-1 to 98.83% (maximum KL 0.00006), which
accounts for most, but not all, of its precision sensitivity. It is not counted as passing conformance.

Existing Qwen3-0.6B and Llama-3.2-1B BF16 outputs remain unchanged in 36 paired CLI comparisons. In the quiet
performance check, six warmed pairs per configuration with alternating main/branch order kept every median
prefill/decode change within ±3% (largest measured median change: +1.07% decode time on Qwen's Metal route).
The CPU prefill tool processed 512 tokens; CLI prefill used 380/407-token prompts, and decode used 64-token replies.
Llama ended the first request after seven tokens, so its decode check also used an ordinary short question;
both sets of measurements are retained. BF16 matrix operations on the Metal route use the CPU fallback.

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
| `nvidia/Minitron-4B-Base` | F16 | 0.00000 | 100.00% |

The same Phi-2 checkpoint converted to Q8_0 gives mean KL 0.00573 and top-1 agreement 94.91%, below the
97% gate. Its registry row and token IDs match; the passing Phi-2 result above establishes F16 conformance.
StarCoder2-3B in Q8_0 gives mean KL 0.00091 and top-1 agreement 96.87%, also below the gate, with matching
registry row and token IDs. Its F16 result above uses the same full checkpoint and conformance text.

The unmodified b11819 converter rejects a reduced official `tiiuae/falcon-rw-1b` checkpoint at
`transformer.h.0.post_attention_layernorm.bias` (unmapped tensor). Its full weights were not downloaded;
Falcon evidence above covers the 7B configuration and its separate-norm variant.

The configurations above without a real-checkpoint row have tiny conformance only.
Command-R and Arcee chat templates are checked byte for byte against transformers;
the other selected base tokenizers do not provide a chat template.

### Classic performance regression check

On the M4 Pro, the controlled main/branch comparison passes all 14 metrics below within ±3%.
Each of eight configurations has one warmup pair followed by six balanced AB/BA measured pairs:
48 measured pairs in total. Rates are cohort medians; changes are medians of the six paired branch/main
ratios, so they need not equal the ratio of the displayed rates. Both cohorts use four threads and F16 KV.

| Model | Weights / route / workload | Main → branch prefill tok/s | Paired change | Main → branch decode tok/s | Paired change |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | BF16 / CPU / `liyab-bench` | 87.85 → 87.85 | −0.06% | N/A | N/A |
| Qwen3-0.6B | BF16 / CPU / CLI | 93.98 → 94.06 | +0.01% | 61.91 → 61.66 | −0.42% |
| Qwen3-0.6B | BF16 / Metal route, CPU matrix fallback / CLI | 95.66 → 94.85 | −0.23% | 61.60 → 62.00 | +0.96% |
| Qwen3-0.6B | F16 / actual Metal / CLI | 262.63 → 260.29 | −0.34% | 32.47 → 32.41 | −0.08% |
| Llama-3.2-1B-Instruct | BF16 / CPU / `liyab-bench` | 36.70 → 36.70 | +0.14% | N/A | N/A |
| Llama-3.2-1B-Instruct | BF16 / CPU / CLI | 38.12 → 38.08 | −0.24% | 28.46 → 28.15 | −1.08% |
| Llama-3.2-1B-Instruct | BF16 / Metal route, CPU matrix fallback / CLI | 37.61 → 37.56 | −0.17% | 27.96 → 27.85 | −0.24% |
| Llama-3.2-1B-Instruct | F16 / actual Metal / CLI | 126.02 → 126.36 | +0.37% | 29.60 → 29.63 | −0.17% |

The standalone CPU benchmark uses 512 tokens, batch 512 and three internal repeats. CLI timing uses one
fixed ordinary question per model and a maximum of 128 generated tokens; every run naturally generated
128 with zero forced tokens. Main/branch UTF-8 reply bytes and all four counts (`prompt_tokens`,
`cached_prefix_tokens`, `generated_tokens`, `forced_tokens`) match. Actual Metal attention is checked on
both Metal routes, and F16 additionally requires Metal FFNs. No cancellation, thermal rerouting or pacing
was accepted. Conformance text and tokenizer gates remain separate from timing.

The protocol was declared before measurement. Exact original executable/library bytes are copied before
each run to the same neutral tool paths for both cohorts, with identical arguments, environment and CWD.
Each CLI's loaded library path and source UUID are verified; standalone benchmarks load no shared Liyab
library. All six original SHA-256s match before and after the matrix. Both loader preflights and all 112
model processes have AC checks before and after: all 228 power stamps show AC, with no other model/build/test
jobs allowed. The AC confirmation reuses the original neutral runner; only its results directory changes.

Earlier results remain part of the evidence: the original separate-path matrix passed 13/14 metrics but
Qwen3 CPU 512-token prefill failed at −8.11%; its fully quiet control failed at −8.12%. An unchanged-byte
same-path diagnostic supported an execution-context confound without identifying its precise cause.
The first full neutral-path attempt passed 10/14 while the host discharged from 13% to 3% on battery;
that attempt remains a failure. The table reports the subsequent complete, prospectively approved AC
confirmation with unchanged binaries, runner, prompts, flags and ±3% threshold. This establishes the
controlled harness comparison; no engine or kernel source performance fix is claimed.


### Tokenizer export label collision

Integration found two distinct source pipelines sharing the converter's `dbrx` label.
DBRX uses its cl100k split and `ignore_merges=true`; the measured OLMo 2/3 source
uses a Llama split with ordinary merge ranks. Their whitespace splits differ,
for example on three spaces, two newlines, then two spaces. Liyab keeps `dbrx`
for the former and provides `llama-bpe-merges` for the latter. The description
tool verifies that explicit mode against the original serialized tokenizer and
reports the colliding uncorrected export. The earlier OLMo gates used three
ordinary texts and do not establish equivalence on this counterexample.
A corrected metadata tag does not alter model tensor payloads. Final integration
conformance for these files must record that correction explicitly.
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

The `ArchSpec::yarn_truncate` fact supplies the YaRN endpoint-rounding default when the file omits it. GPT-OSS
uses fractional endpoints (`false`), as its reference config states; the published ggml-org MXFP4 GGUF omits the
key. Explicit metadata overrides the row. The tool derives this default from the reference's YaRN parameters
and checks the resulting frequencies numerically; when inactive or overridden it preserves the registered
family default and labels it so. Every other existing row retains `true`, preserving its hashes.

The `bailingmoe` tokenizer uses the author’s possessive word pattern, including the `'m` contraction,
individual digits and NFC. It is checked independently against tokenizer.json splitting.

DeepSeek-MoE v1 and Ling-lite 1.5 add `{deepseek, Adjacent, ffn_norm, false}` and
`{bailingmoe, Adjacent, ffn_norm, true}` respectively. Their converters interleave the half-layout Q/K weights;
the router defaults differ in selected-weight renormalization. Leading dense blocks and shared experts are
planned from tensors. Each standard-reference tiny passes 511 positions (512 matching IDs), KL 0.00000,
top-1 99.61%, with fourteen template cases. Their legacy remote implementations index private eager/SDPA
class tables, rejecting the trace tool's `liyab_trace` attention function. Consequently the tool reports
that precise failure and proposes neither row; source and converter inspection plus numerical conformance
establish these rows. Local reference-only compatibility patches restore constructor RoPE buffers after
transformers 5's meta loading and adapt removed serialization APIs; the attention and FFN math is unchanged.
The DeepSeek tokenizer is loaded from its supplied tokenizer.json, avoiding transformers 5's incompatible
SentencePiece reconstruction. Reproduction recipes and compatibility diffs accompany the validation report.

`sarvam-moe` supplies a Unicode-symbol ranked-BPE mode: ASCII spaces become U+2581 without a prefix,
Unicode characters initialize the ranked-merge heap, and absent characters remain unmergeable until byte
fallback. Decoding reverses the space replacement and fallback. A synthetic vocabulary checks rank ordering,
multibyte tokens, repeated spaces, no implicit prefix and an absent emoji. Sarvam's generic attention and
sigmoid routing match a reference (diagnostic provisional row, KL 0.00000, top-1 100.00%), but the published
converter supplies BOS 6 while tokenizer.json inserts BOS 2. No `bailingmoe2` row is admitted: the required
same-token gate fails on the first ID. Its old weight-alias list and nonpersistent RoPE buffer also need
local transformers 5 reference compatibility repairs, preserved with the report. This is independent of
the stage-2 group-limited Ling 2.0 family.

Final validation includes the published GPT-OSS-20B MXFP4 file, using its official Harmony rendering: resident
F16 KV and forced triple-buffered Q8_0 KV return the same coherent greedy reply. Its numerical gate remains the
representative F32 tiny checkpoint; no real-model KL result is inferred from that smoke. Sixty golden cases
pass on NEON-dotprod, AVX2 and scalar without moving previous entries. All fourteen native and both x86 test
targets pass; the Android experimental build also passes. The saved reproduction recipe was rerun from pinned
upstream revisions for both GPT-OSS and DeepSeek-MoE v1, reproducing their stated conformance and tokenizer results.

Mixtral-8x7B-Instruct-v0.1's author template has a dedicated fourteen-case fixture generated from revision
`eba92302a2861cdc0098cc54bc9f17cb2c47eb61` with `tools/chat_template_reference.py`. Thirteen conversations match
transformers byte for byte, including the optional system message and BOS/INST/EOS framing; the tool-result
conversation is refused for its nonalternating roles. Native, AVX2 and scalar text suites check the same fixture.

Generation stops also follow the file's chat protocol. When a control token `<|return|>` is declared EOS,
`<|end|>` ends a message within that turn, and must allow the next channel to run; `<|call|>` hands off to a tool.
The tokenizer identifies this from vocabulary and EOS metadata rather than an architecture name; explicit
EOS/EOT/EOM overrides remain authoritative. A synthetic
vocabulary checks both channelized and ordinary stops, token encoding, and tool handoff. The earlier GPT-OSS
smokes stopped after coherent analysis because the generic `<|end|>` fallback was premature. Corrected resident
F16 and block-streamed Q8_0 runs now complete the final answer in 32 generated tokens. A token trace passes
through `<|end|>` into the final channel and stops at `<|return|>`; the export’s complete stop set equals the
author’s generation_config.json: return, end-of-text and tool handoff.

The final post-Harmony performance gate compares fixed main/branch binaries with a separate warmup followed by
six balanced AB/BA pairs per configuration, four CPU threads and greedy generation. All 84 measured pairs
(14 configurations, 26 metrics) pass ±3%; every CLI run produces 128 tokens with identical reply bytes,
prompt/generated/forced/prefix counts, and the same reply across trials. Raw output is kept as bytes, with
reversible UTF-8 decoding. Binary SHA256s and modification times are checked before and after the matrix.
The coordinated CPU/GPU slot permitted one background HTTP download capped at 819200 bytes/s, plus checksums;
these measurements do not claim zero concurrent downloads. No configuration required a rerun.
BF16 under the Metal route retains CPU matrix fallback; the shared F16 files confirm actual Metal attention
and FFN routing. CPU `liyab-bench` uses 512 synthetic tokens, Q8_0 KV and the best of three repeats; CLI uses
F16 KV, the fixed conformance text and the separately retained ordinary-question prompt. The table gives
medians of throughput, while each Δ is the median of the six paired percentage changes, not their ratio.

| Model | Weights / route / prompt | Prefill main → branch (tok/s) | Paired Δ | Decode main → branch (tok/s) | Paired Δ |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | BF16 / CPU / synthetic512, Q8_0 KV | 85.500 → 85.500 | +0.000% | — | — |
| Qwen3-0.6B | BF16 / CPU / conform, F16 KV | 88.017 → 88.000 | -0.086% | 52.414 → 52.251 | -0.011% |
| Qwen3-0.6B | BF16 / CPU matrix fallback under Metal route / conform, F16 KV | 88.066 → 87.952 | -0.178% | 52.439 → 51.859 | -0.472% |
| Qwen3-0.6B | F16 / true Metal / conform, F16 KV | 248.417 → 246.855 | -0.514% | 26.297 → 26.370 | -0.428% |
| Qwen3-0.6B | BF16 / CPU / question, F16 KV | 93.151 → 93.086 | -0.170% | 59.566 → 59.041 | -0.699% |
| Qwen3-0.6B | BF16 / CPU matrix fallback under Metal route / question, F16 KV | 93.035 → 92.801 | -0.039% | 59.181 → 59.522 | +0.814% |
| Qwen3-0.6B | F16 / true Metal / question, F16 KV | 289.386 → 288.349 | -0.157% | 31.505 → 31.410 | -0.446% |
| Llama-3.2-1B-Instruct | BF16 / CPU / synthetic512, Q8_0 KV | 36.800 → 36.750 | -0.272% | — | — |
| Llama-3.2-1B-Instruct | BF16 / CPU / conform, F16 KV | 37.521 → 37.416 | -0.169% | 27.774 → 27.679 | -0.191% |
| Llama-3.2-1B-Instruct | BF16 / CPU matrix fallback under Metal route / conform, F16 KV | 37.435 → 37.419 | -0.096% | 27.670 → 27.657 | -0.027% |
| Llama-3.2-1B-Instruct | F16 / true Metal / conform, F16 KV | 149.573 → 148.955 | -0.398% | 32.608 → 32.383 | -0.751% |
| Llama-3.2-1B-Instruct | BF16 / CPU / question, F16 KV | 37.971 → 37.919 | -0.093% | 28.483 → 28.404 | -0.276% |
| Llama-3.2-1B-Instruct | BF16 / CPU matrix fallback under Metal route / question, F16 KV | 37.973 → 37.897 | -0.194% | 28.463 → 28.453 | -0.035% |
| Llama-3.2-1B-Instruct | F16 / true Metal / question, F16 KV | 154.698 → 153.843 | -0.425% | 33.078 → 33.073 | -0.210% |

MXFP4 component measurements run sequentially after the model pairs, with a separate warmup and six balanced
pairs of 100000 iterations at width 4096 and four activation rows. Shared prefill preserves every output bit
of the four independent decode dots. NEON's paired median speedup is 1.652249× (47.222 → 28.575 ms), AVX2 under
Rosetta 1.196605× (683.064 → 570.839 ms), and scalar under Rosetta 1.001263× (1241.818 → 1239.361 ms). Scalar
retains its single-row implementation; no scalar speedup or whole-model GPT-OSS speedup is inferred. Earlier
noisy trials and pre-Harmony-fix measurements are retained separately and are not used for this final gate.
## Additional recurrent layouts

The registry adds `lfm2`, `lfm2moe`, `qwen3next` and `kimi-linear`. Blocks are planned from the file's tensors:
`shortconv.conv` selects LFM2's gated depthwise convolution; `ssm_conv1d_q` selects Kimi Delta Attention;
`ssm_conv1d` retains Gated DeltaNet. The metadata supplies `shortconv.l_cache`, `ssm.conv_kernel`, and `kda.head_dim`;
KDA heads come from its query projection. Per-block `attention.head_count_kv` arrays may contain zero in recurrent
blocks; attention blocks must share one cache layout. Kimi's expanded head count is normalized to the one shared
latent row in the absorbed cache.

New architecture facts are `output_norm` (the tensor holding the final RMSNorm), `ssm_grouped_heads` (whether
consecutive DeltaNet value heads share keys) and the shared NoPE plan (`nope_period = 1` for unrotated latent channels). The trace
recomputes the ShortConv formula and records delta-rule operands to establish head sharing. The description tool
compares the final norm's weights with global tensors and recognizes per-block KV counts and the new mixers.
It refuses a row if it cannot establish a fact: latent-attention decomposition and KDA's complete recurrence are
not yet numerically described by this tool. DeltaNet's converter head order is checked against the output projection's columns. Engine
conformance is measured separately; a registry entry alone is not a conformance result.

Fresh tiny F32 checkpoints regenerated from pinned public revisions measure mean KL 0.00000 and top-1 100.00%:
LFM2, LFM2-MoE and Qwen3-Next each run 512 identical reference token IDs; Kimi's full hybrid model runs 503.
The three tokenizer texts match at 535 / 937 / 198 IDs for LFM2 and LFM2-MoE, 524 / 804 / 179 for Qwen3-Next,
and 503 / 798 / 201 for Kimi. Kimi also matches 172 IDs on an additional rare-Han edge text.

The real LiquidAI/LFM2-350M checkpoint, converted with b11819 to BF16, passes the description gate: exactly the
registered row, identical IDs on all three texts (535 / 937 / 198), mean KL 0.00000 and top-1 100.00% on 512 tokens.
Its perplexity is 36.473 → 36.471 (reference → engine). LFM2-700M BF16 also passes: the same three tokenizer
counts, mean KL 0.00000, top-1 99.80%, perplexity 28.968 → 28.970. LFM2-1.2B BF16 passes the same gate with
mean KL 0.00000, top-1 99.80%, perplexity 28.375 → 28.379. All three gates propose exactly the registered row.
Their identical chat template (SHA-1 283ce60a51b5e134bf6abcea2b1d71c13244a45a) matches transformers on fourteen
conversations, as do the LFM2-MoE, Qwen3-Next and Kimi templates.

Qwen3-Next conversion uses b11819's `--no-mtp`. For Kimi with transformers 5.19, the tiny checkpoint preparation
renames the leading dense FFN's `.block_sparse_moe.{gate,up,down}_proj` keys to the official `.mlp.*` names,
without changing values. A local Python-only b11819 converter copy memoizes transformers' pure 256-byte
`bytes_to_unicode()` map: the unmodified converter exceeded 900 seconds repeatedly rebuilding that map.
The cached conversion preserves its tensor and vocabulary algorithms; the resulting F32 hybrid passes the
independent reference comparison above. Kimi uses the official tokenizer implementation for reference checks.

ShortConv state is `embedding_length * (l_cache - 1)` floats per convolution block. KDA state is
`3 * heads * head_dim * (conv_kernel - 1) + heads * head_dim * head_dim` floats per KDA block. Rollback rings and
snapshots multiply those sizes just as they do for DeltaNet; the memory plan and lazy sequence allocation take
these actual per-block sizes. On-disk state/prefix storage retains its existing bounds and eviction policy.

KDA prefill splits independent value rows across the active pool and scans token chunks in causal order. This
keeps decoding arithmetic (including rollback and tree paths) bit-identical across batch sizes. It is a chunked
recurrent scan, not an algebraically reassociated parallel scan. Decode holds one fixed-size matrix per head; its
work and state do not grow with context length. All reduction kernels have NEON, AVX2/FMA and scalar paths.

`Tokenizer::chat_sort_tools()` records tokenizer-side recursive ordering of tool-object keys, selected by the
`kimi-k2` tokenizer family for both Kimi K2 and Kimi-Linear rather than by architecture. Both description and chat
fixture tools probe `apply_chat_template` against the base renderer with unsorted and sorted tools, refusing
unrecognized preprocessing. The description tool checks this behavior against the engine's tokenizer-family
table. The engine and tokenizer CLI pass this fact when loading the model's unmodified template.

The synthetic KDA row-partition test runs the same prefill, rollback, branched verification and retained path
with one and eight workers, requiring identical logits. The prefix-image tests use prompts above the store's
existing 128-token minimum so that reuse is actually exercised.

ShortConv's empty matrix state, and a one-tap convolution's empty history, are omitted from the fragment views
used by prefix storage. Snapshot, rollback and file I/O skip empty buffers while retaining their zero lengths
in the existing file format. This avoids null-buffer copies without changing serialized bytes.

### Host throughput and validation

The real LFM2 BF16 files were measured on an M4 Pro with four CPU workers and F16 KV. Decode uses 64 greedy
tokens after the official chat template (the sky-color question); both replies coherently explain Rayleigh
scattering. Prefill is `liyab-bench --prefill`, 512 tokens, batch 512, best of three runs. Peak RSS is the CLI's
process peak, including the model and context.

| Model | CPU decode (tok/s) | CPU prefill (tok/s) | Peak RSS (MiB) |
| :--- | ---: | ---: | ---: |
| LiquidAI/LFM2-350M | 111.881 | 145.5 | 706.8 |
| LiquidAI/LFM2-1.2B | 30.917 | 35.3 | 2264.7 |

The final full suites pass 14/14 on native arm64, 14/14 under Rosetta with `ROSETTA_ADVERTISE_AVX=1`, and 14/14
under Rosetta without it. Their elapsed times were 145.88, 346.22 and 254.71 seconds, respectively. Metal tests
ran outside the workspace sandbox so they used the actual GPU. The experimental Android script build passes
with `CMAKE_BUILD_PARALLEL_LEVEL=4`; its dry run selects Ninja `-j 4`, while unset retains the generator's default.
No C ABI or app sources changed, so Flutter was not run. Phone validation is performed separately.

Each platform pins 50 golden paths, including 13 new recurrent cases (39 added platform/case hashes); all 111
previous hashes and their order are unchanged. Beyond logits, regressions check KDA's one/eight-worker results,
rollback and retained tree paths, and ShortConv's one/three-tap live/external state images, empty fields and
restored checkpoint rings. Earlier contended x86 calibration waits exceeded the existing five-minute deadline;
all three final suites pass without changing that deadline.

Quiet baseline-main/branch comparisons use four workers, F16 KV and context 1024. CPU prefill below is
`liyab-bench --prefill`, 512 seeded token IDs, batch 512, best of three runs; CPU decode uses CLI statistics.
Metal prefill and decode both use CLI statistics: the complete conformance text has 524 / 499 / 524 prompt
tokens for Qwen3 / Llama / Qwen3.5. Decode uses 64 greedy tokens, except the Qwen CPU followup's 256 tokens.
The initial matrix alternates main/branch order over three pairs. Longer, warmed, evenly counterbalanced
six-pair followups resolve the initial Qwen CPU decode and Llama direct-prefill run/order spread; their medians
are used below. All initial and followup raw runs are retained. All 33 measured CLI pairs have identical output
bytes, and every final prefill/decode comparison satisfies ±3%.

Metal initializes on the actual GPU outside the workspace sandbox. It has no BF16 matrix kernel in either
revision, so those projections fall back to CPU. The shared read-only F16 copies separately exercise actual
Metal projections; attention and recurrent operations continue using their existing CPU routes.

| Model | Route / weights | Prefill main → branch (tok/s) | Decode main → branch (tok/s) | Change prefill / decode |
| :--- | :--- | ---: | ---: | ---: |
| Qwen3-0.6B | CPU BF16 | 87.90 → 87.90 | 50.945 → 51.101 | +0.00% / +0.31% |
| Qwen3-0.6B | Metal BF16 (CPU matrix fallback) | 88.37 → 87.40 | 52.449 → 53.034 | -1.10% / +1.12% |
| Qwen3-0.6B | Metal F16 (GPU projections) | 249.84 → 249.53 | 27.516 → 27.479 | -0.12% / -0.13% |
| Llama-3.2-1B-Instruct | CPU BF16 | 36.70 → 36.75 | 27.711 → 27.738 | +0.14% / +0.10% |
| Llama-3.2-1B-Instruct | Metal BF16 (CPU matrix fallback) | 37.35 → 37.33 | 27.880 → 27.830 | -0.05% / -0.18% |
| Llama-3.2-1B-Instruct | Metal F16 (GPU projections) | 150.07 → 150.18 | 32.562 → 32.662 | +0.08% / +0.31% |
| Qwen3.5-0.8B | CPU BF16 | 84.00 → 83.90 | 53.329 → 53.348 | -0.12% / +0.04% |
| Qwen3.5-0.8B | Metal BF16 (CPU matrix fallback) | 84.09 → 84.10 | 53.114 → 53.288 | +0.00% / +0.33% |
| Qwen3.5-0.8B | Metal F16 (GPU projections) | 274.58 → 275.07 | 34.028 → 34.256 | +0.18% / +0.67% |

Shared-KV consumers must match their source block's query and KV head counts and rotation plan; incompatible files are refused before binding. Integration tests combine shared KV with per-block head and FFN widths across prefill, decode, rollback, tree drafts, serialized state and parked sequences for all four KV types.
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
These are random-weight numerical checks. The published `DeepSeek-V2-Lite-Chat.Q8_0.gguf` from
`mradermacher/DeepSeek-V2-Lite-Chat-GGUF` at revision `9ee9f1f8f39f673e1022e176d586aef4228f630d` separately
passed real execution and resident/streamed mode equivalence: 46 prompt tokens and 64 generated tokens, with
byte-identical replies and counts for resident, 2.29% and 10.47% expert caches, F16 KV and no requantization.
The 16,702,518,016-byte source verifies SHA256
`78a1a180db56416782f71cd7e019ea3d7227cf11a918de9ca1d50f038caba0ec`; its tokenizer matches 617/1,013/241 IDs
and its template all 14 reference renders. The saved earlier revision explicitly refuses the missing split-MLA
metadata, so it cannot provide an earlier real-weight comparison. No real HF logit reference or KL gate is
claimed for this file. README records the observed cache/read/stall counters and their timing limits; used
weights were removed after preserving the evidence.

The final unchanged-output performance check uses Qwen3-0.6B, Llama-3.2-1B-Instruct and the recurrent
Qwen3.5-0.8B on an M4 Pro: original BF16 CPU and Metal/CPU-matrix-fallback routes, F16 actual Metal routes,
and 512-token CPU `liyab-bench` prefill. One warmup precedes six balanced pairs per configuration. All 108
measured CLI turns generated 128 tokens, with byte-identical replies and all four token counts equal in
each pair. All 21 median paired speed ratios across 12 configurations and 72 pairs fall in
0.993247–1.011333, passing the unchanged ±3% gate with frozen executables/libraries and no competing jobs.
README records the three CPU prefill medians. These baseline-model timings are distinct from architecture
conformance and from the individual real V2 streaming observations above.

Files predating split MLA projections describe the expanded head with `attention.key_length`, `value_length`
and one KV head per query head. `describe_model()` normalizes those dimensions to the cached latent and plans
`BlockPlan::latent_combined` from `attn_kv_b.weight`; ambiguous combined/split tensors are refused. The description
tool compares the corresponding expanded dimensions to the observed reference geometry. At block acquisition,
the key rows are dequantized and transposed to F32 and value rows retain their original bytes, then both use
the existing absorbed or expanded forward pass. These copies are bounded by one block, work after rebasing a
streaming slot, and count toward the expert-cache planning margin.

### Auxiliary index-key storage

`KvCacheConfig::auxiliary_dim` adds F32 channels after each cached latent key row. `head_row_bytes()` includes
the main row in its cache format and those F32 channels. These channels are owned by
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

### Shared K=V compression components

The `deepseek4` component row records shared K=V, unweighted query-head RMSNorm, trailing adjacent-pair partial
RoPE and inverse output RoPE. `attention.q_lora_rank`, `output_group_count`, `output_lora_rank`,
`compress_rope_freq_base` and per-block `compress_ratios` describe geometry. Compressor tensor widths select
one series or two overlapping series; overlapping blocks require their own compressed-key indexer. No tensor
or metadata rule depends on a model's name. Learned sinks and the full V4 MoE variant are refused precisely
until their separately owned operations are integrated; this row is not a full V4 support claim.

Compression keeps channelwise stable-softmax maxima, denominators and weighted sums, not historical source
projections. Nonoverlapping state uses `3 * head_dim` floats; overlapping state uses `9 * head_dim` plus
`9 * index_dim` for its independent indexer. Existing recurrent snapshots/checkpoints carry these bounded
arrays through rollback, draft trees, sequences and context images. A closed window's normalized/rotated
entry follows its closing source token in the paged auxiliary-key image. Tree queries resolve that token
through their ancestry before reading the entry. Index head weights remain signed; only query/key dot scores
are rectified.
Public context and prefix-cache contracts include compression in their stateful snapshot/rollback rules.

This first layout retains raw shared-KV rows for the finite context, applying the model's sliding mask during
attention. It reserves `head_dim + index_dim` F32 auxiliary channels in each attention block at every position,
including unused nonclosing positions. Memory planning includes those bytes; this is not a compact strided
allocation or a physically bounded sliding cache. An engine-level sliding window and segment relocation are
refused because they would discard entries or change compressor phase. Key-only scaling is inapplicable to
shared K=V values and is not calibrated. Projections use the ordinary Metal/Vulkan/CPU routing; normalization,
attention, index scoring, residual mixing and grouped output slices use existing CPU kernels. The compressor
recurrence has NEON, runtime-dispatched AVX2 and scalar channelwise arithmetic.

The manifold-constrained residual path infers coefficients from normalized parallel streams, applies sigmoid
pre/post weights and the file's Sinkhorn iteration count, consumes the mixing matrix transposed and collapses
the final streams before output normalization. Expert lookahead is disabled for these streams: the collapsed
input is not a prediction of the next block's parallel residuals. Component fixtures keep dense FFNs and omit
sinks explicitly; they cannot establish full V4 conformance.
The description tool observes shared key/value tensors but refuses a row until it also reconstructs all four
shared-attention facts; it does not infer them from the family name. Its ordinary Q/K/V trace does not yet
decompose the compressors, grouped output or mHC. Consequently it cannot independently validate the new
`q_lora_rank`, output group/rank, compression ratio/base or hyper-connection count/iteration/epsilon metadata
against that trace. The component runner below supplies those dimensions directly to native modules and
does not substitute for that full-model description gate.

`test_compression` independently recomputes whole windows and dense attention (including F16 main-cache
rounding), exercises all 26 storage formats with attention and residual widths independent, and checks
decode splits, rollback and saved images in
all four KV formats. Engine tests also cover branches, suffix drafts, prefix-store restart and sequences.
The F32 whole-component reference also checks available Metal/Vulkan projection routes with and without
parallel residual streams, keeping the CPU fallback for other operations and the same logit error bound.
The shared test writer tiles encoded reference blocks for reader-only formats; their projection reference
multiplies decoded weights and Q8-rounded activations directly, without calling the backend's dot kernels.
Their attention norms balance the encoded blocks' measured RMS against the fixture's ordinary weight scale;
output/FFN norms, the eight writable-format controls and the logit error bound remain unchanged.
The Android regression runner includes this suite and supplies its encoded quantization fixtures.
LoRA targets for whole compression, indexer and residual-stream projections, including the final stream
collapse, use the normal adapter-aware matmul path. The grouped first output projection and legacy combined
latent expansion return `Unsupported` when selected as adapter targets, naming the tensor; their slices do
not share one input row. Tests compare supported F32/F16/BF16/Q8_0 adapters with merged weights through
prefill, decode, tree verification and rollback, and check explicit refusal on F32 and Q8_0 base matrices.
`tools/compression_reference.py` uses native transformers 5.19 modules without rounding their main keys:

| Component fixture | Tokens | Mean / max KL | Top-1 | CSA top-2 selections |
|---|---:|---:|---:|---:|
| Three kinds + mHC, head/index 32 | 83 | 0.000000219 / 0.00000573 | 100% | 79/83 |
| Three kinds + mHC + YaRN, trained head/index widths 512/128 | 512 | 0.000000647 / 0.00029766 | 99.80% | 511/512 |
| Isolated first CSA block, trained widths | 512 | 0.000000054 / 0.00000037 | 100% | 512/512 |

The isolated selection test excludes changes to the source hidden state caused by earlier layers' F16 main
cache. The complete component fixture retains sliding, CSA and HCA blocks; none is removed to claim a model
gate. Three synthetic golden cases pin compression with and without mHC and with YaRN on NEON, AVX2 and
scalar. Every previously pinned case remains unchanged.

### GLM-4.7-Flash export facts

The official `glm4_moe_lite` configuration has dense and sparse blocks, sigmoid routing, block RMS epsilon
1e-5 and separately constructed query/KV latent RMS epsilon 1e-6. The b11819 `Glm4MoeLiteModel` converter
writes `deepseek2` and the block epsilon, but omits `expert_gating_func`; the engine's documented file default
is softmax. The description tool reports that routing mismatch and proposes a latent-epsilon row different
from the existing `deepseek2` row. The file cannot select both latent normalization contracts unambiguously
from its architecture label and existing normalization metadata.

An official-configuration random F32 tiny with one dense and one sparse block (trained query/KV/head widths,
four attention heads, residual/FFN widths 256/128, 16 experts) gives 494 identical IDs / 493 positions, mean
KL 0.00019, max 0.00185 and top-1 58.22% on the branch. All three tokenizer texts match (494/783/204 IDs),
and fourteen chat-template cases render exactly. A diagnostic GGUF copy adding sigmoid gating plus the
tool's scratch-only latent-epsilon row gives mean/max KL 0.00000 and top-1 100%. It uses the same unmodified
native F32 reference. This isolates the missing export facts; it is not a standard converter gate, and no
name-based engine exception or global change to existing DeepSeek normalization is made.

The merged recurrent-format test reuses `ssm_quant_vectors.bin` for all five mixer layouts (26 formats × five mixers × four KV types). Test-only head geometry aligns projection rows to the formats' 256-value blocks; all ordinary fixture defaults and random streams are preserved. Nonzero projection and carried-state ablations prevent zero weights or unused bindings from counting as coverage.

Mapped PLE validation varies only the token table's dtype. Independent `gguf-py` fixture values form an F32 table in a second otherwise identical graph, and a zero-table control verifies that the table contributes. All four KV types and rollback are covered; no new format encoder is introduced.

Final norms selected by `ArchSpec::output_norm` bind their `.weight` and, for LayerNorm, optional `.bias` tensors. The description tool recognizes these dynamic names and proves recurrent-norm switches inactive when the corresponding mixer operation is absent; it does not mistake an affine output norm for an unused tensor.
