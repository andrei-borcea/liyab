# liyab-compat

`liyab-compat` owns load-time model description, format adaptation and execution planning.
The core runs the resulting plan with its existing CPU/GPU kernels.

| Responsibility | Owner |
| :--- | :--- |
| Architecture facts, metadata defaults and format conventions | `src/compat/architecture.{h,cpp}` |
| Tensor-driven mixer/FFN discovery, per-block geometry and validation | `liyab-compat` |
| `ModelConfig`, `BlockPlan` and required-operator decisions | `liyab-compat` |
| Source tracing and registry-row verification | `tools/describe_model.py`, reading the compatibility sources |
| Tensor binding, memory allocation, numerical operators and backends | Core runtime |
| Request scheduling, sequences and persistent contexts | Core engine |

## Load contract

1. Read the mapped file's metadata, tensor names and shapes.
2. Resolve the architecture facts and file-specific block plans.
3. Reject malformed or unsupported compositions with a precise reason.
4. Bind and execute the validated plan in the core.

Planning reads no weight payloads, performs no downloads and never rewrites the source GGUF.
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

Gemma 4 heterogeneous geometry, its MoE composition and full DeepSeek-V4 remain unverified at
this extraction checkpoint. Moving their descriptions into the layer does not establish support.
[Current coverage](model-coverage-stage1.md) records the earlier gates and outstanding operations.

## Extraction checkpoint

The native Release experimental suite passes 16/16 tests after the move, including the existing
ARM64 golden logits. The description-tool tests pass 4/4 and reference-tokenizer tests 7/7.
No trained model weights or network downloads were needed for this checkpoint.
