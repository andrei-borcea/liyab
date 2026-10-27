# Stage-1 model coverage: evidence and limitations

This report records the seven-section model-coverage wave and its integration. Section-level results below
are historical evidence from feature branches; they are not a claim that the final phone or performance gates
have passed. Recipes use random-weight tiny checkpoints unless explicitly labelled trained or real-model smoke.

## Integrated verification

| Check | Verified result |
| --- | --- |
| Native arm64 Release, experimental, actual Metal access | 16/16 tests, 144.95 s |
| Rosetta x86 scalar | 16/16 tests, 272.20 s |
| Rosetta x86 AVX2 | 16/16 tests, 355.74 s |
| Golden logits | 138 cases on each of three platforms, 414 pins; original 111 pins unchanged and ordered |
| Android Release experimental | Complete NDK 28.2/API 28 build; Vulkan/QNN/NeuroPilot enabled; 15 device test binaries built |
| Flutter with integrated native library | 53 passed, two optional real-model tests skipped; analyze reports no issues |
| Independent recurrent reader proofs | 520 cases: 26 formats × five mixers × four KV types; maximum logit/state error zero on all three host targets |
| Mapped per-layer embeddings | 104 cases: 26 formats × four KV types; maximum error versus independently decoded F32 control zero on all three host targets |
| Merged-tree tiny replay | Classic 20/20 pass, with source-verified StarCoder metadata correction; remaining sections in progress |
| Final paired performance versus main | Pending |
| Phone regression and four new trained CLI models | Pending; required serial `aca7d42d` was disconnected at the latest check |

Reader proofs use nonzero encoded rows, independent decoding and projection/state/table ablations, not merely
successful loading. The original golden pins are not regenerated to accommodate integration differences.

## Section conformance summary

| Section | Numerical evidence | Real-model scope / material limits |
| --- | --- | --- |
| Classic | 20 tiny cases, 16 families | Nine trained gates; failed Phi-2/StarCoder2 Q8 controls do not count as passes |
| Modern | 17 tiny recipes | Ten trained gates; corrected tokenizer/template metadata is explicit. ChatGLM 3 and IQuest remain failures |
| MoE | Nine passing tiny cases | GPT-OSS-20B smoke: 34.689 tok/s resident, 3.797 tok/s forced block streaming with Q8 KV; Harmony Flutter replay unsupported |
| SSM | Five tiny cases | Mamba-130M: KL 0.00001/top-1 99.61%; Mamba-370M and Mamba-2-130M: displayed KL zero/top-1 100% |
| Conv/KDA | Four tiny cases: displayed KL zero/top-1 100% | LFM2-350M/700M/1.2B: displayed KL zero/top-1 100/99.80/99.80% |
| Gemma | Gemma 3n tiny 99.41%; Gemma 4 uniform 99.22%, K=V 99.02% | Gemma 3n BF16: displayed KL zero/100%; separate BF16 reference 0.00168/98.24%. EmbeddingGemma: 39 identical ID sequences, minimum trained-vector cosine 0.99992651 |
| Sparse/group routing | V2 99.80%, V3 99.38%, GLM-5/5.1/5.2 99.80%, V3.2 99.59%, Ling 2.0 99.61%; displayed mean KL zero | Genuinely sparse top-k 256/493 and 384/484. GLM-4.7-Flash standard tiny gate fails at 58.22%; corrected diagnostic 100% is not support |

Each percentage measures teacher-forced argmax agreement, not task accuracy. Displayed KL zero is rounded, not
proof of exact equality. See [model descriptions](model-descriptions.md) for model IDs, revisions, recipes and controls.

## Stage 2 and later work

The first composition candidate is Flint: leading dense blocks, shared experts, router bias, sinks and NoPE/window
primitives exist, but their composition still needs its own source-tokenizer/template and numerical gates.
Llama 4, Cohere2/Command-A/R7B, Jamba, Falcon-H1, Granite hybrids/MoE, Nemotron-H/H-MoE, OLMoE, PhiMoE, MiniMax-M2,
Grok and encoder models each need independent composition/operation coverage. Existing Mamba or LayerNorm parts
alone do not establish those architectures. Full Gemma 4 still needs heterogeneous per-block head widths and a
trained gate; larger variants also need MoE-specific norms/scales. Full DeepSeek-V4 lacks hash/tid2eid routing,
sqrt-softplus/clipped SwiGLU and shared-attention sink integration despite passing compression/mHC components.
Sarvam has an unresolved converter/source BOS mismatch. MiniCPM3 and other unmeasured variants remain deferred.

Architectures without a supported converter require a later conversion project. Multimodal input is a separate
subsystem. Native Windows/Linux CI and release packaging are a separate task; no new CI or stage-2 work is
included in this wave.

## Detailed operation and section measurements

The test counts and throughput comparisons in the following section refer to each feature branch before integration.
Use the integrated-verification table above for current suite counts and pending acceptance gates.

Tokenizer export limitations, including the `dbrx` label collision between two
source pipelines, are recorded in [model descriptions](model-descriptions.md#tokenizer-export-label-collision).
Engine architecture coverage does not imply that every published GGUF has correct
chat or tokenizer metadata.
MoE projection variants: optional router logit bias and per-expert gate/up/down biases are applied before routing
and after their respective projections. `SwigluOai` clamps gate values above 7, clamps up values to [-7, 7], then
computes `gate * sigmoid(1.702 * gate) * (up + 1)`. Learned per-head `attn_sinks.weight` contributes one score only
to each attention softmax denominator, including windowed attention. These elementwise operations run on the CPU
beside the existing backend projection routing. MXFP4 projections use the existing NEON and AVX2/FMA codebook
kernels (scalar fallback), including expert slices loaded through the bounded expert cache. Mixtral's `llama`
files renormalize selected router weights and infer expert width from the down projection when metadata omits it.
MXFP4 prefill shares each nibble lookup and exponent conversion across up to four activation rows on NEON and
AVX2/FMA, preserving the corresponding decode kernel's float accumulation order. The CPU backend uses this path
for resident and streamed expert matrices; scalar targets retain the single-row kernel.
The tiny-checkpoint and reference-trace tools also reduce per-block lists of FFN widths, retaining each block's
original width when it is already smaller than the requested test width.
Tiny fixtures randomize declared persistent router selection-bias buffers as well as parameters, so those paths
are exercised even when the published configuration initializes the corrections to zero.
The `glm4` pre-tokenizer uses the model's three-digit grouping and contraction rules, without normalization and
with vocabulary-first merge lookup. Harmony and Qwen2MoE templates have byte-for-byte reference fixtures.
A gated attention block can also store its sigmoid gate in a separate `attn_gate.weight` projection (AFMoE),
instead of fusing it into each query head. It follows the same CPU/Metal/Vulkan projection routing.
For legacy configurations, the tiny-checkpoint and trace tools canonicalize RoPE's `type` spelling to
`rope_type`, preserving the scaling type and every numerical parameter.

Tiny MoE conformance against the models' transformers implementations (F32 GGUF, F16 KV, full fixed text):

| Architecture / configuration | Positions | Mean KL | Top-1 |
| --- | ---: | ---: | ---: |
| gpt-oss, two heads, top-4/16, both block kinds, window 128 | 483 | 0.00000 | 99.79% |
| Mixtral (`llama`), top-2 routing with inferred expert width | 511 | 0.00000 | 100.00% |
| qwen2moe, routed and gated shared experts | 511 | 0.00000 | 100.00% |
| dots1, leading dense block and nonzero selection correction | 511 | 0.00000 | 99.80% |
| glm4moe, leading dense blocks and shared experts | 497 | 0.00000 | 100.00% |
| ernie4_5-moe, leading dense block and two shared experts | 511 | 0.00000 | 99.80% |

The original four-head random gpt-oss checkpoint gave KL 0.00000 and top-1 98.76%, below the tiny gate. Seven of
959 expert selections changed when F32 reference keys/values were rounded to the F16 cache format; every routing
choice matched a reference rounded the same way. That rounded reference is diagnostic, and the table uses the
standard untouched reference. The smaller checkpoint keeps every block kind, crosses the sliding window, and
retains top-4 selection among sixteen experts.
GLM4 tokenization matches all 498 token IDs in its tiny conformance text; its fourteen author-template reference
conversations render byte for byte through the same chat interpreter.
MoE files may name a selection correction `exp_probs_b` or `exp_probs_b.bias`; supplying both is rejected.
Attention output gates may supply one scalar per head or one value per head channel. Optional per-block
`swiglu_clamp_exp` and `swiglu_clamp_shexp` arrays clamp the activated SiLU gate above their limit and the up
projection symmetrically; zero disables this separate clamp convention. AFMoE decimal groups align on the right
in a linear scan, and the `exaone-moe` pre-tokenizer preserves spaces inside words and normalizes NFC.
HY3 (`hy_v3`) tiny conformance uses its sigmoid router, nonzero selection correction, shared expert and dense
leading block: 498 positions, identical 499 token IDs, mean KL 0.00000 and top-1 100.00%. Its registry row is
proposed exactly by the description tool.
The gpt-oss row supplies fractional YaRN correction endpoints when older GGUFs omit `yarn_truncate` (the
published ggml-org MXFP4 file does). Explicit file metadata overrides that architecture default; existing
architectures keep rounded endpoints. Synthetic default/explicit cases must produce identical bits.

The `bailingmoe` pre-tokenizer preserves Ling’s contractions and individual digits, with NFC normalization.

DeepSeek-MoE v1 (`deepseek`) and Ling-lite 1.5 (`bailingmoe`) use ordinary attention, leading dense blocks and
shared experts. Their F32 tiny checkpoints pass the standard F32 reference with F16 KV: 511 positions each,
512 identical token IDs, mean KL 0.00000 and top-1 99.61%. Fourteen template cases per family match transformers.
Their old remote code needs local transformers 5 compatibility repairs for saved-weight aliases and unsaved
RoPE buffers. The automatic description tool refuses these legacy references because their hard-coded
attention-class tables do not expose its registered attention function; these two rows are verified by
converter tensor permutations, source inspection and end-to-end conformance instead.

The `sarvam-moe` tokenizer uses Unicode-symbol ranked BPE, replaces spaces with `▁` without adding a prefix,
and falls back to `<0xXX>` bytes for missing characters after merging. This shares the ranked merge engine with
byte-level BPE. Its parts are implemented, but Sarvam is not added to the architecture registry: b11819 writes
BOS ID 6 while the author's tokenizer.json postprocessor inserts ID 2. A diagnostic reference-token comparison
passes KL 0.00000/top-1 100.00%; it does not pass the required identical-token gate. The converter/tokenizer
metadata conflict must be resolved before claiming support.

The published ggml-org GPT-OSS-20B MXFP4 GGUF also produces the same coherent greedy reply through its own
Harmony template with resident F16 KV and forced block streaming with Q8_0 KV. This is a real-model smoke check;
the numerical conformance table above uses the random-weight F32 checkpoints. Sixty golden cases are pinned for
NEON-dotprod, x86 AVX2 and x86 scalar, preserving every earlier hash. The final native and both x86 test suites
pass all fourteen targets, including sequence and KV-store reuse, and the Android experimental build passes.

Mixtral's own chat template is checked against fourteen transformers reference cases: thirteen render byte for
byte, and a tool-result conversation is refused because the template requires alternating user/assistant turns.
The fixture covers its optional initial system message, BOS, `[INST]` framing and assistant EOS on native,
x86 AVX2 and x86 scalar builds.

Channelized chat vocabularies that declare `<|return|>` as EOS disable the generic `<|end|>` stop fallback,
so reasoning can continue into a final message. `<|return|>` completes the turn and `<|call|>` stops
for a tool handoff. This is selected from token vocabulary and EOS metadata; formats whose EOS is `<|end|>`
retain their existing stop behavior. Explicit EOS/EOT/EOM metadata remains authoritative.

The final MoE performance comparison uses six warmed, balanced main/branch pairs per configuration on the
M4 Pro: all 26 prefill/decode metrics across Qwen3-0.6B and Llama-3.2-1B are within ±3% (paired medians
−0.751% to +0.814%), with identical generated bytes and token counts. CPU 512-token `liyab-bench` uses Q8_0 KV;
CLI CPU, BF16 Metal-route/CPU-matrix-fallback and F16 true-Metal runs use F16 KV. A capped 800 KiB/s background
download was permitted during the coordinated CPU/GPU slot. MXFP4 four-row prefill is 1.65× faster on NEON and
1.20× on AVX2 than four separate decode dots, with identical bits; scalar fallback is unchanged (1.001×).
These are component speedups, not a measured whole-model GPT-OSS speedup. The full paired results and method
are recorded in [Model descriptions](model-descriptions.md).
### Additional recurrent mixers

LFM2's `ShortConv` projects each token to `[B | C | x]`, convolves `B * x` causally per channel, gates it with `C`,
and projects it back to the residual stream. Its state holds the last `L_cache - 1` convolution inputs per channel.
Kimi Delta Attention (`Kda`) uses a chunked recurrent prefill (independent value rows split across the active
thread pool, token chunks preserving decoding arithmetic), with context-independent decoding work. It convolves Q/K/V, L2-normalizes Q/K and applies a delta rule with decay per key channel,
then a sigmoid-gated per-head RMSNorm. Its state holds the three convolution histories and a key/value matrix
per head. Qwen3-Next uses the existing Gated DeltaNet with consecutive value heads sharing a key head and a fused
beta/alpha projection; Qwen3.5's existing head order stays unchanged. Kimi's latent-attention blocks keep their
positional channels unrotated. Per-block KV-head metadata may list zero for recurrent blocks.

These recurrent operations run on the CPU (KDA has NEON, AVX2/FMA and scalar kernels); Metal and Vulkan continue
routing supported projections to their backend. State sizing, independent sequences, snapshots, draft rollback,
tree branches and saved prefix images use the same `[convolution history | recurrent matrix]` ownership as DeltaNet.
The LFM2 pre-tokenizer uses Llama 3's splits with ordinary BPE merges. Kimi K2/Linear isolates Han runs before
its case-aware words. `chat_template_reference.py --trust-remote-code` loads official Python tokenizers when needed. The final norm tensor is an architecture fact:
LFM2 uses `token_embd_norm` after the last block. See the measured scope and tool limitations in
[model descriptions](model-descriptions.md#additional-recurrent-layouts).

Kimi's tokenizer sorts nested tool-object keys before chat rendering. `Tokenizer::chat_sort_tools()` selects this
behavior from `tokenizer.ggml.pre`, including Kimi K2 and Kimi-Linear independently of their architecture;
only the tools value is ordered, and the original template and other request values keep their order.
The reference-template tool probes this preprocessing separately from the Jinja renderer.

Fresh tiny F32 LFM2, LFM2-MoE, Qwen3-Next and Kimi-Linear checkpoints each measure mean KL 0.00000 and top-1
100% against transformers. Real LFM2-350M, 700M and 1.2B BF16 files pass the description gate with matching
tokenizer IDs, mean KL 0.00000 and top-1 100%, 99.80% and 99.80%, respectively. Their chat templates match
transformers; [model descriptions](model-descriptions.md#additional-recurrent-layouts) records the scope,
perplexities and converter compatibility details.

On an M4 Pro with four CPU workers, BF16 weights and F16 KV, LFM2-350M measures **111.9 tok/s decode** and
**145.5 tok/s prefill**; LFM2-1.2B measures **30.9 tok/s decode** and **35.3 tok/s prefill**. Decode uses 64 greedy
tokens after the model's own chat template; prefill uses 512 tokens in one batch, the best of three runs of
`liyab-bench --prefill`. Both models give coherent replies to the sky-color prompt. These are host CPU numbers;
phone throughput remains to be measured.

The recurrent regression cases exercise all four KV types, independent sequences, draft rollback, alternative
tree branches, bounded prefix images restored by another engine, and CPU/GPU projection routing. A KDA check
compares one worker with eight workers, including rollback and tree retention, to cover value-row partitioning.

Recurrent-state fragments omit empty components before storage copies. ShortConv therefore stores no matrix
fragment, and a one-tap convolution stores no history; state files retain the same zero-length fields and format.

The additional recurrent paths have 13 golden cases on each of `arm64-neon-dotprod`, `x86_64-avx2` and
`x86_64-scalar` (50 paths per platform). All 111 existing hashes retain their values and order. The full native,
Rosetta AVX2 and Rosetta scalar suites each pass 14/14 tests, including actual Metal access on the host; the
experimental Android build also passes with `CMAKE_BUILD_PARALLEL_LEVEL=4`.

Quiet main/branch measurements of Qwen3-0.6B, Llama-3.2-1B and Qwen3.5-0.8B pass the ±3% prefill/decode gate on
CPU and the Metal route; 33 paired CLI replies retain identical bytes. Metal's BF16 matrices fall back to the
CPU in both revisions, so separate F16 files establish actual GPU projection timings. The protocols, warmed
balanced followups and before/after numbers are recorded in [model descriptions](model-descriptions.md#host-throughput-and-validation).

Recurrent reader-format validation uses independent nonzero encoded blocks for Mamba, Mamba-2, ShortConv, KDA and Qwen3-Next projection bindings across all 26 weight formats and all four KV types. It checks decoded rows, projection and carried-state ablations, decode/prefill agreement and rollback, without changing pinned golden hashes.

Mapped per-layer embedding tables are also checked in all 26 reader formats and four KV types against independent decoded F32 tables in otherwise identical complete models. A zero-table control verifies their contribution; rollback must reproduce identical logits.

Flutter chat uses the author template when replay preserves the processed prompt, or a recognized ChatML/Llama 3 fallback. A present incompatible template without such a fallback now refuses chat explicitly instead of guessing Zephyr. Models remain loaded for raw completions and pooling. This includes the current Harmony replay limitation; engine/CLI template support is broader than Flutter chat support.

Android app installation requires an explicit `--serial` (or `ANDROID_SERIAL`) and uses that device for every check and installation. Native app builds respect `CMAKE_BUILD_PARALLEL_LEVEL`. Phone regression includes LoRA, SSM and compression tests and both independent quantization fixtures; an incomplete Android test build is refused instead of silently skipping binaries.

The description tool checks optional affine biases of registry-selected final LayerNorms and records inactive recurrent-norm switches explicitly when tracing attention-only models.

Reference-tokenizer validation compares pipelines after parsing with the installed tokenizer runtime, preserving implicit serialized defaults while still rejecting changed normalization or splitting.

The b11819 `refact` export label also collides: plain GPT-2 ByteLevel (Granite) and individual-Digits then ByteLevel (tiny StarCoder) require different splitting. Preserve plain `refact`; correct a source-verified digit pipeline to `tokenizer.ggml.pre=starcoder` on a copy. The description tool reports this conflict; no model-name override changes tokenization.
