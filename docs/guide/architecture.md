# Engine architecture

[← README](../../README.md) · [Documentation index](README.md)

Commands and source paths in examples are relative to the repository root.

## On this page

- [Architecture](#architecture)

---

## Architecture

```
   Liyab app (Flutter, dart:ffi) · Kotlin (JNI) · Swift · C++ · other apps via the local API
                                     │
                         liyab_c_api.h  (stable C ABI)
                                     │
  ┌──────────────────────────────────▼──────────────────────────────────────────┐
  │ Engine                                                                      │
  │  ├─ Context manager ── turn-to-turn reuse, state snapshots, prefix cache,   │
  │  │                     cancellable chunked prefill, save/restore            │
  │  ├─ Generation ─────── sampler, grammars, forced text, speculative decoding │
  │  ├─ PowerManager ───── headroom forecast, pacing, performance hints, guards │
  │  ├─ DeviceDetect ───── SoC + accelerator probes → backend ranking           │
  │  └─ Transformer ────── per block: mixer (attention | Gated DeltaNet)        │
  │                        + FFN (dense | MoE), paged KV cache, recurrent state │
  └──────────┬───────────────────────────────────────────┬──────────────────────┘
             │ weights                                   │ matmuls (per-op route)
  ┌──────────▼─────────────────────────────┐   ┌─────────▼────────────────────────────┐
  │ MmapLoader      zero-copy GGUF, splits │   │ CPU   NEON · SDOT · i8mm · AVX2 ✅   │
  │ DirectFile      parallel O_DIRECT      │   │ GPU   Metal ✅ · Vulkan ✅            │
  │ TripleBuffer    streamed dense blocks  │   │ NPU   QNN · NeuroPilot (probe) 🟡    │
  │ ExpertStore     MoE expert cache:      │   │ fallback per matmul: NPU → GPU → CPU │
  │   LFU, prediction, size classes,       │   │ GPU-shared memory: the GPU reads     │
  │   repack on read, hot-list warmup      │   │ weights where storage wrote them     │
  └────────────────────────────────────────┘   └──────────────────────────────────────┘
```

* **Zero-copy weights.** `MmapLoader` maps the GGUF file read-only. Every tensor is a view into the mapping, and the
  CPU kernels read quantized blocks in place. On Apple Silicon each tensor is wrapped in an `MTLBuffer` with
  `newBufferWithBytesNoCopy`, so the GPU reads the page cache directly.
* **One forward pass for every architecture.** The transformer is not written per model. For each block the loader
  picks the token mixer from the tensors present (`ssm_conv1d` → Gated DeltaNet, `shortconv.conv` → ShortConv,
  `ssm_conv1d_q` → KDA, `attn_kv_a_mqa` → multi-head latent
  attention, otherwise softmax attention) and detects optional pieces (QKV biases, Q/K norms, an attention output
  gate, the pre-FFN norm, norms after the attention and the FFN). The facts the tensors cannot tell are one row per
  architecture: whether Q/K norms come before or after RoPE, the FFN's activation (SwiGLU or GeGLU), whether
  embeddings are scaled, which blocks attend through a sliding window and with which RoPE base. The FFN is a mixture
  of experts when the block has a router. Scalars a model states in its file (RoPE scaling, soft caps, the q·k
  scale) are read from it. See [Supported models](models.md#supported-models).
* **Per-operation routing.** Attention projections can go to the GPU and the FFN to the NPU. A backend that reports
  `Unsupported` for a matmul falls back to the next one for that matmul only.
* **Vulkan on phones.** Mobile drivers lack `VK_EXT_external_memory_host` (Adreno 830 included), so mapped weights
  cannot be wrapped in place. Liyab has two paths:
  * Streamed models: weights are placed in host-visible, device-local, coherent and cached memory. On Adreno that
    is the same RAM, and direct reads from storage land there. Native kernels (`shaders/matvec_native.comp`, 23
    formats) decode the GGUF blocks in place.
  * Models that fit: each tensor is copied once into GPU memory and repacked (`shaders/matvec.comp`).

  Kernels are compiled with `glslc` and embedded at build time. Matmuls that share an input go out in one submission,
  and a dispatch costs 0.055 ms.
* **Paged KV cache.** Fixed 64-token pages come from a pool on demand through page tables. Memory follows the
  positions actually used, and freed pages are reused exactly. Sliding windows keep the first `kv_sink_tokens`
  positions (attention sinks, default 8) plus the last *N*. Multi-head latent attention caches one row per position
  and block, its RoPE key and its latent, and reads the values from the latent channels of that row
  ([Supported models](models.md#supported-models)). Attention scores a KV head's whole query
  group at once, and each V row is dequantized once per group. On a 2,812-token context this cut attention from 67.6
  to 44.4 ms per token on the 35B. Each token's visible positions are cut into up to 84 equal spans of at least 32
  positions, merged in order with log-sum-exp. While decoding, the spans of the one token are the tasks every
  thread shares (168 on two KV heads, an even split for 1 to 4 and 6 to 8 threads). The cut follows the token's own
  position, never the batch or the threads, so its attention is the same bits however the prompt was split.
* **KV laid out per block.** Blocks the model itself windows (Gemma 2's and 3's local layers, Phi-3's) see only
  their last *W* positions, so they keep only those rows. Layers are grouped by window: the blocks that see every
  position form one group, the windowed ones another, and each group maps its own pages. A windowed group lets go
  of every page no query can see any more, after a rollback of up to 64 positions (speculative verification), so
  its memory stays at its window whatever the context; the other group grows with the context. Every page still
  has one size and comes from one pool: a pool page holds 64 positions of *U* layers, *U* dividing each group's
  layer count (all the layers when no block is windowed), so a page one group frees is reused by the other,
  without fragmentation. Pages are zeroed when mapped. A model without windowed blocks keeps the single-table
  layout, byte for byte, with the same memory, and every model's logits are the same bits as before (golden
  hashes unchanged, Gemma 2 and 3 included). A context returns to an earlier position only where the window before
  it is still kept: inside the rollback margin, and at the state snapshots the engine takes (the end of each
  prompt, the oldest one), whose window pages are pinned instead of copied (the live window usually holds them
  anyway). With `segment_reuse` on, windowed blocks keep every position as before (a reused segment needs every
  block's rows). [Measurements](benchmarks.md#per-block-kv).
* **Key scaling for quantized KV caches.** A quantized cache stores each key row in blocks of 32 values that share
  one scale, set by the block's largest value. Some model families carry a few key channels far larger than the
  rest and nearly constant across tokens: about −130 where the others are about 1 on Qwen2.5, up to ~300 on Qwen3.
  The small values sharing their block were then rounded with a step a hundred times too coarse, and almost all of
  a Q8_0 cache's error came from there. Attention only needs q·k, and q·k = (q∘t)·(k/t) for any positive
  per-channel t, so keys are stored divided by t and queries multiplied by it. Each channel then has the same range
  on both sides and the products are unchanged.
  * **Measured once per model**, on a short fixed text (about 45 tokens: prose, code, numbers, several scripts).
    For each channel, t takes the largest value the keys and queries reach at any position up to 32,768: RoPE turns
    each pair of channels by a known angle per position, so that range is exact even though the text covers only a
    few positions. Ranges are floored at a quarter of their head's median and t is bounded, so a channel the text
    barely reached cannot get an extreme factor. A model whose tokenizer cannot encode text (a pre-tokenizer
    Liyab does not have) cannot be measured: it loads with its keys as computed, and the log and `describe()`
    (a `Keys:` line) say so. On a model with large key channels that costs a quantized cache the accuracy above;
    an F16 cache loses nothing.
  * **Only where it helps.** A model is scaled when some head's largest key channel reaches 12 times its median
    channel: Qwen3-0.6B 107×, Qwen3-4B 87×, Qwen2.5-0.5B 52×. Below that (Qwen3.5-0.8B 10.5×, the Qwen3.6-35B
    MoE 7.4×, Qwen3.5-4B 7.5×, Llama 3.2 6.3×) scaling moved the KL against F16 only within noise, sometimes up,
    so those models keep their keys as computed, bit for bit. The first 12 tokens decide for most models; the
    rest of the text is processed only for those that need factors.
  * **Cost.** On the phone: 0.27 s for Qwen3.5-4B, 1.4 s for the 35B MoE (its experts are read from storage), 7.4
    s for a dense 27B larger than RAM, 1.4 s for Qwen3-4B, which is scaled. `EngineConfig::cache_dir` keeps the
    result (a few KB per model, the 16 most recently used models), so only a model's first load pays for it. The
    factors depend only on the model, the cache format and the kernel set, never on the backends or the configured
    context, so a prompt gets the same reply in every process. With the factors kept, speed is unchanged (A/B on
    the phone, 35B: decode 10.2 vs 10.0 tok/s, first token 5.8 vs 6.0 s).
  * **Saved contexts** carry the factors their keys were stored with; one saved with others (or before key scaling
    existed) is converted when loaded instead of being processed again.

  Measured against F16 ([Correctness](benchmarks.md#correctness)): Q8_0 keys went from a mean KL of 0.070 to 0.0018 on
  Qwen2.5, 0.051 to 0.0021 on Qwen3-0.6B and 0.015 to 0.0063 on Qwen3-4B, with the same memory and kernels.
* **x86-64 kernels.** One binary runs on any x86-64 CPU. The library is compiled for the x86-64 baseline; the AVX2
  kernels (`src/core/quant_x86.cpp`) carry a per-function `target("avx2,fma,f16c")` attribute and run only when
  CPUID reports AVX2, FMA and F16C and the OS saves the 256-bit registers (Intel Haswell and AMD Zen onwards).
  The choice is made once per process; `LIYAB_KERNELS=scalar` forces the scalar set and `quant::kernel_isa()`
  reports `avx2` or `scalar`. They cover Q8_0 (single and multi-row), Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL and MXFP4
  against Q8_0 activations; Q4_K, Q5_K, Q6_K (1 to 4 rows per weight decode) and IQ4_XS against Q8_K; F32, F16
  and BF16 dots; Q8_0 / Q8_K activation quantization; F16 encoding and the F16, BF16, Q8_0, Q4_0, Q4_1 decodes
  of the KV cache and embeddings. The int8 products use `VPMADDUBSW` (|w| times x with w's sign, which cannot
  saturate because activations stay in [-127, 127]) and `VPMADDWD` against the sub-block scale, so a K-quant
  super-block accumulates exactly in int32. Q8_0 and the K-quants then apply the same float epilogues as the NEON
  kernels in the same order, so a row's result does not depend on the batch, and activation quantization and
  decoding give the scalar code's bits.
* **Thread pool.** Workers wait for the next job in `WFE` on ARM64: the core sleeps until the job word changes, with
  no instructions issued, instead of spinning at full clock; on x86-64 they spin with `PAUSE`. They block on a
  condition variable after 300 µs idle. A job is split into 4 chunks per thread, claimed on demand, so a preempted
  core does not hold up a whole matmul. Matmuls that share an input run as one group, and MoE experts run in waves:
  two fork/joins per wave instead of two per expert.

---
