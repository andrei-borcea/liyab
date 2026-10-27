# liyab-compat

`liyab-compat` owns load-time model description, format adaptation and execution planning.
The core runs the resulting plan with its existing CPU/GPU kernels.

| Responsibility | Owner |
| :--- | :--- |
| Architecture facts, metadata defaults and format conventions | `src/compat/architecture.{h,cpp}` |
| Tensor-driven mixer/FFN discovery, per-block geometry and validation | `liyab-compat` |
| `ModelConfig`, `BlockPlan` and required-operator decisions | `liyab-compat` |
| Source tracing and registry-row verification | `tools/describe_model.py`, reading the compatibility sources |
| Global/local RoPE, YaRN, LongRoPE and compression-table resolution | `src/compat/rotary.{h,cpp}` |
| Complete-header identity decisions for states and calibration | `liyab-compat` |
| Legacy combined-MLA layout normalization | `src/compat/weights.{h,cpp}` |
| Tensor binding, memory allocation, numerical operators and backends | Core runtime |
| Request scheduling, sequences and persistent contexts | Core engine |

## Load contract

1. Read the mapped file's metadata, tensor names and shapes.
2. Resolve the architecture facts and file-specific block plans.
3. Reject malformed or unsupported compositions with a precise reason.
4. Bind and execute the validated plan in the core.

Metadata planning reads no weight payloads. Rotary planning reads small frequency-factor tensors. The weight adapter decodes/transposes a combined MLA
key projection and preserves the quantized value rows. It owns only the active block's temporary
views, including during streaming. Neither path downloads files or rewrites the source GGUF.
The registry has static lifetime; each call produces its own configuration and plans. Concurrent
planning requires read-only access to the mapped file. The existing C ABI stays unchanged.

The CMake object target `liyab-compat` supplies the same compatibility implementation to the
library, CLI and tests. There is one registry and one planner. Source/export exceptions belong
here rather than in architecture-name branches in the forward pass.

## Validation without large model downloads

| Proof | Data needed |
| :--- | :--- |
| Registry and metadata planning | Local configs, tensor headers and generated GGUF fixtures |
| Full forward semantics | Small random checkpoints instantiated by the upstream model class |
| Tokenizer/template behavior | Configuration, tokenizer files and author templates; no trained weights |
| Quantized/recurrent/streamed paths | Nonzero encoded synthetic models and independent reference results |
| Trained-model quality and device capacity | A separate optional trained-weight/device check |

Keep head widths, block kinds, routing, KV sharing, windows and compression transitions in the
small configurations. Reduce layer counts, embedding width, vocabulary and expert counts where
the reference permits it. A generated checkpoint must exercise the feature before its conformance
result counts. Delete generated weights after recording results and source identities.

## Per-block adaptation

| Plan | Runtime invariant |
| :--- | :--- |
| Local/global head, value and rotary widths | Compact projections; fixed KV images use maximum strides and zero padding |
| KV sharing | Source and consumer must agree on head counts, logical widths and rotation |
| Quantized KV | Logical heads use whole quantization blocks; other widths select F16 |
| Calibration and relocated keys | Sample and rotate only active channels; padded scale factors stay 1 |
| Dense and routed FFNs in parallel | Independent branch norms, then the outer FFN norm and residual add |
| Learned router channel/expert scales | Normalize the residual independently of expert inputs; scale selected weights after top-k normalization |

The planner reads explicit sliding-window patterns even when the architecture has no default
window. A bare window key retains the existing architecture-specific behavior.

Gemma 4 text dense/MoE variants now pass small upstream forward comparisons with heterogeneous
heads, K=V, shared KV, per-layer embeddings and double-width shared-layer MLPs. These fixtures
establish execution semantics, not trained-model quality, multimodal support or device capacity.
DeepSeek-V4 now has the routed/shared-FFN and sink operations listed below. Both families use supported
operator plans; unknown operators still produce an explicit refusal. [Earlier coverage and pending work](model-coverage-stage1.md).

## Extraction checkpoint

The native Release experimental suite passes 16/16 tests after the move, including the existing
ARM64 golden logits. The description-tool tests pass 4/4 and reference-tokenizer tests 7/7.
No trained model weights or network downloads were needed for this checkpoint.

## Offline Gemma 4 check

Use a local Transformers installation that includes Gemma 4 and a local upstream converter.
`tools/compat_fixture.py` creates 332-token checkpoints while preserving Gemma 4's 256/512 local/global
head widths. The archived extraction check used smaller 32/64 heads (261,232 dense / 459,904 MoE parameters). Its routing probe requires at least three distinct experts per block.
The measured MoE fixture uses all four experts in each of its four blocks.

```bash
python tools/compat_fixture.py --out build/compat-fixtures/gemma4
python tools/compat_fixture.py --out build/compat-fixtures/gemma4-moe --moe
python /path/to/convert_hf_to_gguf.py build/compat-fixtures/gemma4-moe \
  --outtype f32 --outfile build/compat-fixtures/gemma4-moe.gguf
python tools/conform_reference.py --model build/compat-fixtures/gemma4-moe \
  --text tests/data/conform_text.txt --out build/compat-fixtures/gemma4-moe.ref \
  --attn-implementation eager --max-tokens 512
build/liyab-kl -m build/compat-fixtures/gemma4-moe.gguf -f tests/data/conform_text.txt \
  --compare build/compat-fixtures/gemma4-moe.ref --ref-tokens --max-tokens 512 \
  --kv f16 --max-kl 0.0001 --min-top1 99 --threads 4 --expert-cache 1
```

| Archived 32/64-head checkpoint | Positions | Mean KL (printed) | Maximum KL | Top-1 |
| :--- | ---: | ---: | ---: | ---: |
| Gemma 4 dense | 127 | 0.00000 | 0.00002 | 100% |
| Gemma 4 parallel MoE, streamed experts | 127 | 0.00000 | 0.00001 | 100% |

Both tokenizers produce the same 128 input IDs. Tests also compare heterogeneous attention
against an independent reference for F16/Q8_0 KV across four blocks and Q4_0/Q4_1 in isolated mixers.
The four-block Q4 cases check decode, rollback, saved state and zero padding: tiny differences in independent
F32 projections can cross Q4 bin boundaries and compound through a stack. The numerical bound remains 0.02
for the independent comparisons. The tests include F16 fallback for 24-wide heads. Key-range tests compare active factors
against an unpadded reference. Key relocation covers local partial rotation in three cache formats.
No existing golden hashes were changed. Generated checkpoints can be deleted after retaining
the config, routing audit, commands and comparison results.

The native experimental suite passes 16/16 (192.89 s). After extracting rotary and identity
planning, engine/golden/KV-reuse/registry pass 4/4 (83.20 s). Description tests pass 4/4,
and all 64 registry rows round-trip. [Checkpoint configs and results](validation/liyab-compat/gemma4.json).

## DeepSeek-V4 text composition

| Plan or adapter | Behavior |
| :--- | :--- |
| `ffn_gate_tid2eid.weight` | Validate I32 shape and expert IDs, then keep a resident token-to-expert table |
| Original token IDs | Preserve pass order across prefill, decode, multiple sequences and speculative trees |
| Router scores | Softmax, sigmoid or stable `sqrt(softplus(x))`; selection bias changes selection only |
| Routed/shared SwiGLU | The V4 profile clamps the raw gate before SiLU and clamps the up projection |
| Shared-attention sinks | Add each learned logit to the softmax denominator with no value contribution |
| Model identity | Hash route-table bytes as well as the extended header; equal-sized table edits invalidate saved contexts |

I32 is an auxiliary format. The compatibility planner rejects integer tensors outside validated route tables
before numerical binding. The runtime uses integer IDs directly, without a floating-point conversion.
The route adapter copies these small tables once; expert weights retain their existing streaming path.

```bash
python tools/compat_fixture.py --architecture deepseek4 \
  --out build/compat-fixtures/deepseek4 --gguf build/compat-fixtures/deepseek4.gguf
python tools/conform_reference.py --model build/compat-fixtures/deepseek4 \
  --text tests/data/conform_text.txt --out build/compat-fixtures/deepseek4.ref \
  --attn-implementation eager --max-tokens 512
build/liyab-kl -m build/compat-fixtures/deepseek4.gguf -f tests/data/conform_text.txt \
  --compare build/compat-fixtures/deepseek4.ref --ref-tokens --max-tokens 512 \
  --kv f16 --max-kl 0.0001 --min-top1 99 --threads 4
```

This fixture keeps 512-wide main heads, 64 rotary channels, and 64 index heads of width 128. Three blocks
exercise sliding attention, overlapping CSA with actual top-k selection, and nonoverlapping HCA. Four residual
streams use mHC; one block hashes original token IDs and two use learned top-k. Shared and routed experts
both clip active channels. The local byte-fallback tokenizer tests ID transport, not the production tokenizer.
The fixture-only GGUF writer exports this fixed synthetic configuration and rejects unknown tensor mappings;
use a production converter for trained checkpoints.

Two retained diagnostics qualify the gate:

- Reducing the indexer to two heads created 27 top-k cutoff ties on the 128-token control. Torch and Liyab
  selected different equal-scoring sets in 21 queries. The canonical fixture keeps the source head count and
  its probe rejects cutoff ties. Liyab resolves equal index scores by the lower source position.
- The canonical 128-token prefix gave mean KL 0.00013 / top-1 98.43% against F32 reference with F16 KV.
  The standard 512-token gate gives mean KL 0.00004 / top-1 99.61%, at unchanged 0.0001 / 99% thresholds.
  These random-weight comparisons do not measure trained-model quality.

The generated full-text check covers the decoder. Production checkpoint conversion, original tokenizer/chat
metadata, trained quality, multimodal inputs and MTP acceleration still need their own gates. Compression
continues to retain raw rows for the finite context and padded F32 auxiliary channels; this work does not
introduce a compact strided cache. The phone check remains waived.

## Current verification

The canonical fixtures preserve the source head widths. The C++ tests also exercise smaller heterogeneous
heads, quantized projections, four KV formats, streamed experts, saved contexts, interleaved sequences and
speculative trees. Route-table mutation tests reject states from an equal-sized table with different IDs.

| Generated text decoder | Positions | Mean KL | Maximum KL | Top-1 |
| :--- | ---: | ---: | ---: | ---: |
| Gemma 4 dense, 256/512 heads | 511 | 0.00001 | 0.00047 | 99.80% |
| Gemma 4 parallel MoE, streamed decode | 511 | 0.00000 | 0.00107 | 99.80% |
| DeepSeek-V4, 512 heads / 64 rotary channels | 511 | 0.00004 | 0.01330 | 99.61% |
| DeepSeek-V4, streamed decode | 511 | 0.00004 | 0.01330 | 99.61% |

All four checks use CPU execution, F32 weights, F16 KV and unchanged acceptance thresholds: mean KL ≤ 0.0001 and top-1 ≥ 99%.
The GGUF fixtures occupy 2.7, 3.5 and 5.2 MB. No trained weights were downloaded.

| Build / test | Result |
| :--- | :--- |
| Native Release experimental, Metal enabled | 16/16, 151.66 s |
| Rosetta AVX2 engine / golden | 2/2, 37.09 s |
| Rosetta scalar engine / golden | 2/2, 34.29 s |
| Android arm64 library / CLI | Cross-build passes; no phone run |

Existing golden pins remain unchanged. Trained-model throughput and quality remain unmeasured.
[Configs, source hashes, routing/clipping audits and failed diagnostic controls](validation/liyab-compat/text-conformance.json)
record the scope of these results. Generated weights and temporary reference tooling can be removed after
saving that evidence; the fixture scripts recreate them offline with the recorded dependencies.

Cleanup removed the generated checkpoints, temporary reference environment/converter and 610 stale test GGUF
files (1.09 GiB logical). The native build remains available. The final disk snapshot shows 88.9 GiB free;
owner downloads were left untouched.
