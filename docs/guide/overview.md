# Project overview

[← README](../../README.md) · [Documentation index](README.md)

Commands and source paths in examples are relative to the repository root.

## On this page

- [Why Liyab](#why-liyab)
- [Design principles](#design-principles)
- [What it does today](#what-it-does-today)

---

*Liyab* is Tagalog for **flame**.

Liyab is an inference engine for GGUF language models on phones and laptops, and a private assistant built on it. It
is written in C++20 and has no cloud dependency. Our aim is to be the reference engine for running open models on
whatever device is at hand. That means models larger than the device's RAM, on any phone, without heating it up, and
without a single byte of the user's data leaving the device.

Today a 22 GB mixture-of-experts model (**Qwen3.6-35B-A3B**) answers at **~9–10.5 tokens per second** from the
command line on a Snapdragon 8 Elite phone, and at **~7.5–9.4 tok/s in the app**. Its experts are streamed from flash
storage into the 6 GiB that Android lets one app use. A model that fits in RAM, **Qwen3.5-4B**, answers at **~12 tok/s**
from the command line and **~12–16 tok/s in the app**.

> **This README describes the code as it is.** Every feature listed here is implemented and tested; everything that
> is not lives in the [Roadmap](development.md#roadmap). Numbers are measured on real hardware. Where a number comes from a single
> run, a hot phone or a different setup, the text says so.

---


## Why Liyab

Cloud assistants are capable, but they cannot read your messages, your calendar or your files without sending them
somewhere else. A model running on the phone can. That only works if the device can run a *strong* model, and do it
in a way people will tolerate: no waiting minutes for an answer, no phone burning in the hand, no battery gone by
lunch.

Phones are not small servers. Their memory is shared with everything else and capped per app (6 GiB on HyperOS). The
fast storage reads at ~4.4 GB/s, but only when the reads are large and concurrent. Sustained load hits thermal limits
within minutes. Every SoC has a different mix of CPU cores, GPU and NPU, and every model architecture stresses them
differently.

Liyab is built around those constraints:

* **Models larger than RAM are the normal case.** Dense blocks and MoE experts stream from flash, with direct I/O
  sized to what the storage likes. The expert cache predicts the next block's experts from the router and overlaps
  reads with compute. Memory is planned against the budget the OS really allows.
* **Heat is part of scheduling.** The engine reads the OS's thermal-headroom forecast. As pressure builds, it paces
  tokens and asks the governor for lower clocks, so the same tokens cost fewer watts. It does not run flat out and
  then collapse.
* **Serving a single user well.** Context is reused across turns, including the recurrent state of hybrid models.
  A prefix cache of KV chunks brings back earlier conversations, keeps a shared system prompt once and outlives a
  restart. Prefill can be cancelled and resumed. Structured output covers
  tool calls. A local OpenAI, Anthropic Messages and gRPC API lets other apps on the device use the same model.
* **A private agent on top.** The Liyab app replaces the system assistant. It answers questions about your day from
  your own calendar, notifications, messages and calls, and nothing leaves the phone.

## Design principles

These rules decide what goes into the code:

1. **Generic and self-calibrating.** No optimization is tied to one SoC or one model. Thread counts, draft
   lengths, expert prefetch depth, context length, memory plans and thermal bands are measured or derived on the
   device the engine runs on.
2. **Kernels per instruction set, policy per platform.** Compute kernels target ISAs: ARM64 NEON with dot product
   and i8mm, x86-64 AVX2, Metal, Vulkan. Differences between operating systems (thermal APIs, performance hints, memory limits)
   live behind a thin adaptive layer, never in the kernels.
3. **Lossless by default.** Every speed-up that changes the output (fewer experts, requantization, skipping) is
   opt-in, labelled lossy, and measured with `liyab-kl` against the lossless path.
4. **Measured, not assumed.** Changes are judged by alternating A/B runs on a real phone. Ideas that did not pay off
   are documented, so nobody has to try them again blind.
5. **Private by construction.** The engine never touches the network. The app uses the network only to download
   models from Hugging Face.

---

## What it does today

| Area | Status |
| :--- | :--- |
| GGUF v2/v3 loader: `mmap`, zero-copy, validated against malformed files; split models (`-00001-of-0000N.gguf`) | ✅ |
| Text architecture coverage | Classic, modern dense, MoE, recurrent and Gemma text paths; [coverage and validation tiers](models.md#architecture-coverage-and-validation) distinguish trained checkpoints, tiny conformance and components |
| Every GGML tensor format (F32/F16/BF16, legacy Q4–Q8, K-quants, I-quants, TQ1_0/TQ2_0, MXFP4, NVFP4) | ✅ bit-exact against llama.cpp's reference decoder |
| Tokenizers: SentencePiece, byte-level BPE and raw unigram, pre-tokenizers as data (Qwen, Llama 3, GPT-2, DeepSeek, Hunyuan, Phi, Gemma, Mistral-Nemo, SmolLM2, Youtu-LLM, SmolLM3, OLMo 2/3, Granite, Seed-OSS, EXAONE 4, PLaMo 3) | ✅ token-identical to each model's Hugging Face tokenizer |
| State-space mixers: `mamba` (Mamba-1, Falcon-Mamba) and `mamba2` (Mamba-2); mixer-only and FFN-only blocks planned from tensors | CPU selective scans and depthwise convolution, with NEON, AVX2/FMA and scalar paths; Metal/Vulkan use CPU scans with their normal projection routing. Validation: [model descriptions](../model-descriptions.md#state-space-mixers). |
| Block-plan validation | Partial attention/FFN/SSM projections fail validation or binding instead of silently dropping a part; unsupported parallel attention/Mamba blocks are refused. |
| Mamba pre-tokenizers: GPT-NeoX (`olmo`, NFC then GPT-2 rules) and Falcon-Mamba (`falcon`, punctuation then ByteLevel and digit groups) | Data in the same pre-tokenizer table |
| SSM backend correctness | GPU prefill/decode projections are compared against CPU in an F32 graph; quantized projections and recurrent-state invariants are checked separately across all 26 formats using nonzero encoded fixtures and projection/state ablations (uniform RMS conditioning; [fixture recipe](../model-descriptions.md#state-space-mixers)). |
| Recurrent timing | CLI phase reports use `recurrent`; the existing C/JSON `delta_net` counters include DeltaNet and Mamba mixers. The ABI and JSON keys are unchanged. |
| Chat templates: the GGUF's Jinja template rendered by Liyab's own interpreter; layout of turns and prefix stability for context reuse | ✅ byte-identical to transformers' `apply_chat_template` |
| Models larger than RAM: dense blocks streamed by parallel `O_DIRECT` reads; MoE experts streamed into an LFU cache with router-predicted prefetch | ✅ automatic |
| Memory planning: per-app budget, automatic requantization only when the expert cache would be too small, recommended budget reported | ✅ |
| CPU backend: NEON + SDOT; i8mm (SMMLA) layouts for Q4_K/Q5_K/Q6_K/Q8_0; WFE-waiting thread pool with chunks claimed on demand | ✅ |
| CPU backend on x86-64: AVX2 + FMA + F16C kernels chosen at run time by CPUID, scalar fallback in the same binary | 🟡 tested under Rosetta 2 only |
| Windows x64 port of the engine, `liyab-cli`, the tests and tools: file mapping, unbuffered direct reads, processor groups and hybrid cores, UTF-8 paths and console; MSVC or clang-cl | 🟡 written, not yet compiled or run on Windows ([Windows](building.md#windows-x64)) |
| GPU backends: Metal (zero-copy on unified memory), Vulkan (Adreno/Mali; native kernels for 23 formats on GPU-shared memory) | ✅ |
| Paged KV cache (F16 / Q8_0 / Q4_0 / Q4_1) laid out per block (windowed blocks keep only their window), with per-channel key scaling, sliding window with attention sinks, grouped-query attention over spans, latent rows for multi-head latent attention, read in the absorbed form or, for long prefills, the expanded one (chosen by a threshold measured once per device) | ✅ |
| Context kept between turns, recurrent-state snapshots for DeltaNet and compressed attention, save/restore to disk | ✅ |
| Shared K=V attention components: sliding + overlapping/nonoverlapping gated compression, compressed-key selection, grouped output, manifold-constrained residual streams | 🧪 CPU component checks pass; full DeepSeek-V4 also requires the MoE routing, clipping and learned-sink integration |
| Several sequences decoded together: one pass over every sequence's rows, each sequence's own context, adapter and priority, each output what it gives alone | ✅ CPU and Metal backends |
| Automatic context length from the KV cache's cost; cancellable, resumable chunked prefill | ✅ |
| Prefix cache across conversations: a content-addressed store of KV chunks (shared prefixes kept once), in RAM and on storage across runs, capped, LRU | ✅ |
| Segment reuse: marked parts of a prompt (documents, tool results) reused wherever they appear, keys turned to their new positions, a share calibrated per model processed again; attention-only models | ✅ opt-in, lossy |
| Speculative decoding: draft model, n-gram lookup or suffix drafts from this and earlier conversations (their continuation tree verified in one pass), adaptive draft count, hybrid models included | ✅ |
| Structured output: grammar-constrained decoding (JSON Schema, grammar text; tool calls in the app and the local API) and forced continuations | ✅ |
| Embeddings and reranking: decoder embedding models and rerankers as their GGUF declares them (last-token or mean pooling, ranking heads, or LM-head labels), shared prefixes computed once | ✅ cross-checked against Hugging Face transformers |
| LoRA adapters: several GGUF adapters loaded next to one model within a memory cap, selected per generation, computed beside the base matmuls (never merged); dense, hybrid and MoE models (per-expert adapters); contexts kept apart by adapter | ✅ |
| Heat and power: OS headroom forecast, adaptive pacing, ADPF performance hints, emergency guards, power profiles | ✅ |
| SoC detection (Snapdragon, Dimensity, Tensor, Exynos, Apple) and backend ranking | ✅ |
| Qualcomm QNN (Hexagon NPU), MediaTek NeuroPilot | 🟡 detected at runtime; compute falls back to GPU/CPU |
| Liyab app (Flutter): chat, model library with Hugging Face downloads, activity meters, settings, system assistant sheet (phone) or command bar (computer) | ✅ Android · ✅ macOS (Apple silicon) · 🟡 iOS app not built yet (the engine builds for iOS) |
| Agent tools: calendar, notifications (chats, email previews), SMS, calls, contacts, clipboard on Android; calendar, Apple Mail, Messages, calls, contacts, clipboard on macOS; each source opt-in | ✅ |
| Local API for other apps on the device: OpenAI, Anthropic Messages, gRPC; loopback only, token-protected, off by default; tools and JSON answers by constrained decoding | ✅ text (no images) |
| Voice, actions (alarms, events, replies), local memory/RAG | 🔜 [Roadmap](development.md#roadmap) |
| Experimental: early exit, head pruning, EGLS, 2:4 sparsity, JIT unpacker, persistent KV prefix files, io_uring | 🧪 `LIYAB_ENABLE_EXPERIMENTAL` |
