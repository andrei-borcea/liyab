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
Full DeepSeek-V4 remains incomplete. [Earlier coverage and pending work](model-coverage-stage1.md).

## Extraction checkpoint

The native Release experimental suite passes 16/16 tests after the move, including the existing
ARM64 golden logits. The description-tool tests pass 4/4 and reference-tokenizer tests 7/7.
No trained model weights or network downloads were needed for this checkpoint.

## Offline Gemma 4 check

Use a local Transformers installation that includes Gemma 4 and a local upstream converter.
`tools/compat_fixture.py` creates 332-token checkpoints: 261,232 parameters for dense,
459,904 for MoE. Its routing probe requires at least three distinct experts per block.
The measured MoE fixture uses all four experts in each of its four blocks.

```bash
python tools/compat_fixture.py --out build/compat-fixtures/gemma4
python tools/compat_fixture.py --out build/compat-fixtures/gemma4-moe --moe
python /path/to/convert_hf_to_gguf.py build/compat-fixtures/gemma4-moe \
  --outtype f32 --outfile build/compat-fixtures/gemma4-moe.gguf
python tools/conform_reference.py --model build/compat-fixtures/gemma4-moe \
  --text tests/data/conform_text.txt --out build/compat-fixtures/gemma4-moe.ref \
  --attn-implementation eager --max-tokens 128
build/liyab-kl -m build/compat-fixtures/gemma4-moe.gguf -f tests/data/conform_text.txt \
  --compare build/compat-fixtures/gemma4-moe.ref --ref-tokens --max-tokens 128 \
  --kv f16 --max-kl 0.0001 --min-top1 99 --threads 4 --expert-cache 1
```

| Upstream checkpoint | Positions | Mean KL (printed) | Maximum KL | Top-1 |
| :--- | ---: | ---: | ---: | ---: |
| Gemma 4 dense | 127 | 0.00000 | 0.00002 | 100% |
| Gemma 4 parallel MoE, streamed experts | 127 | 0.00000 | 0.00001 | 100% |

Both tokenizers produce the same 128 input IDs. Tests also compare heterogeneous attention
against an independent reference for F16/Q8_0/Q4_0/Q4_1 KV, including F16 fallback for 24-wide
heads, padded channels, decode, rollback and saved states. Key-range tests compare active factors
against an unpadded reference. Key relocation covers local partial rotation in three cache formats.
No existing golden hashes were changed. Generated checkpoints can be deleted after retaining
the config, routing audit, commands and comparison results.

The native experimental suite passes 16/16 (192.89 s). After extracting rotary and identity
planning, engine/golden/KV-reuse/registry pass 4/4 (83.20 s). Description tests pass 4/4,
and all 64 registry rows round-trip. [Checkpoint configs and results](validation/liyab-compat/gemma4.json).
