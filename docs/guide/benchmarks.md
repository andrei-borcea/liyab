# Measurements and correctness

[← README](../../README.md) · [Documentation index](README.md)

Commands and source paths in examples are relative to the repository root.

## On this page

- [Measured results](#measured-results)
- [Qwen3.6-35B-A3B: a 22 GB MoE on a phone](#qwen36-35b-a3b-a-22-gb-moe-on-a-phone)
- [A dense 27B larger than RAM](#a-dense-27b-larger-than-ram)
- [Models that fit in RAM](#models-that-fit-in-ram)
- [Latent attention prefill (Mac)](#latent-attention-prefill-mac)
- [Per-block KV](#per-block-kv)
- [Storage and kernels](#storage-and-kernels)
- [Correctness](#correctness)
- [Charts](#charts)
- [What we found along the way](#what-we-found-along-the-way)
- [What did not pay off (so you don't have to try)](#what-did-not-pay-off-so-you-dont-have-to-try)

---

## Measured results

### M4 Pro dense 27B development check

On 28 October 2026, Liyab decoded 16 greedy tokens from `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf` on an
M4 Pro with 48 GiB RAM. Both runs used Metal, the Performance profile, a 512-token context and the same 10-token
prompt. These short runs show the cost of streaming; they do not establish sustained speed.

| Weight plan | Decode | Peak RSS | Storage wait |
| :--- | ---: | ---: | ---: |
| 64/64 blocks resident | 4.17 tok/s | 15.7 GiB | 0 ms |
| 13,000 MiB cap, block streaming | 2.54 tok/s | 12.0 GiB | 2,229 ms per run, prompt included |

With resident weights, the 4.17 tok/s run spent 145 ms/token in dense FFNs, 65 ms/token in recurrent mixers,
20 ms/token in attention and 9 ms/token in the LM head. The GGUF contains 15.41 GiB of weights. A 50 tok/s
ordinary decode would need to read roughly 770 GiB/s of weights; [Apple quotes 273 GB/s for M4 Pro](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/). Reaching that target needs fewer weight reads per accepted token, such as effective batched
verification, as well as faster kernels. No 50 tok/s result has been measured.

The next check used the same local GGUF and an 8-token prompt (`Explain why the sky is blue.`), with 16 greedy
decode tokens, Metal, Performance and a 512-token context. We alternated the existing two-command-buffer FFN
with a Metal path that keeps gate/up, SiLU and down in one command buffer. All weights stayed resident; both
paths produced the same 16-token output text.

| FFN path | Decode runs | Dense FFN time | Storage read |
| :--- | ---: | ---: | ---: |
| Separate gate/up, CPU SiLU, down | 4.438 / 4.433 tok/s | 137.1 / 137.8 ms/token | 0 bytes |
| Metal gate/up, SiLU, down | 4.717 / 4.689 tok/s | 126.0 / 126.0 ms/token | 0 bytes |

The new path applies only to bias-free, unclamped SiLU FFNs without LoRA or an experimental hook. Other FFNs
use their existing projections. These short runs establish a local 5.7–6.4% decode gain over the paired control;
they do not establish sustained speed or a cross-engine comparison.

#### Execution design from the llama.cpp and Colibri comparison

llama.cpp [tiles rows in its Metal matvec](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mv.metal),
[encodes graph operations into command buffers](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-context.m),
and [fuses norms, SSM convolution and recurrent-cache writes](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-fusion.cpp).
Colibri [keeps dense MoE weights in RAM and streams routed experts](https://github.com/uv-genai/colibri#the-idea);
its native MTP verification can save weight reads when drafts are accepted, but cold expert-cache misses can
erase that gain. Its GLM-5.2 MoE speed is not a baseline for this resident 27B dense model.

Liyab's next execution work follows the measured costs:

1. Tile Q4_K/Q5_K/Q6_K matvec across output rows. The first four-SIMD-group experiment kept one row per group
   and slowed decode from 4.416 to 4.297 tok/s, so row tiling and activation reuse need their own A/B proof.
2. Keep activations and recurrent state in GPU-owned segments across norm, projections, DeltaNet and residuals.
   The CPU commits KV and recurrent snapshots only after a segment completes; failed operations fall back at
   explicit boundaries. Adapters and speculative rollback must see the same state and output as the CPU route.
   `liyab-compat` continues to own model semantics and tensor mapping, not this execution schedule.
3. Add a multi-token matrix kernel that reuses each weight tile during verification. Current Metal matvec launches
   one row per token and rereads weights for every draft, so batching alone cannot meet the 50 tok/s target.
   The existing `DraftBudget` can select draft counts from measured accepted tokens per millisecond after that
   kernel exists; this noMTP GGUF uses suffix/context drafts, whose acceptance depends on the text.

Before enabling each path, check prefill and decode logits, greedy token IDs, batched sequences, LoRA updates,
recurrent snapshots and rollback, plus peak memory. Compare tok/s and bytes read on the same prompt and model;
keep the old path when the new one loses.

The reference device is a **Xiaomi 25010PN30G**:
* Snapdragon 8 Elite (SM8750), 15 GB RAM, UFS 4;
* Adreno 830 GPU, Hexagon v79 NPU;
* HyperOS, with a 6 GiB per-app memory limit.

Phones throttle as they heat up, so the comparisons below come from alternating A/B runs unless noted. The raw
numbers live in [`docs/benchmarks/results.json`](../benchmarks/results.json); after adding a measurement, run
`python3 tools/gen_bench_charts.py` to redraw the charts.

### Qwen3.6-35B-A3B: a 22 GB MoE on a phone

UD-Q4_K_M, 22.1 GB. 40 blocks, of which 30 are Gated DeltaNet. 256 experts, top-8. Run with `--memory-budget 5500`,
CPU backend, greedy decoding, starting from a cold expert cache.

| Engine version | Decode | ms/token | Expert reads/token |
| :--- | ---: | ---: | ---: |
| Condition-variable thread pool (start) | 3.06 tok/s | 327 | 596 MiB |
| + spinning thread pool | 3.24 tok/s | 308 | 480 MiB |
| + freshly loaded experts kept until used | 5.54 tok/s | 181 | 309 MiB |
| + a block's experts in one batched pass | 5.93 tok/s | 169 | 309 MiB |
| + Q4_K/Q5_K/Q6_K × Q8_K kernels | 6.15 tok/s | 163 | 308 MiB |
| + next block's experts predicted after the mixer | 6.44 tok/s | 155 | 283 MiB |
| + 7 threads, chunks claimed on demand | 7.83 tok/s | 128 | 283 MiB |
| + an expert's three matrices read in parallel | **8.89 tok/s** | 112 | 280–311 MiB |

Later work raised it further: WFE waits, parallel DeltaNet passes, prediction by rank, the shared expert overlapped
with reads, and the attention rewrite. The command line now reaches **~9–10.5 tok/s on a cool phone**, and the app
**~7.5–9.4 tok/s** (7.4 tok/s at a 1.9k-token context when first measured; the phone regression run tracks both,
see [Regression run on the phone](development.md#regression-run-on-the-phone)).

Per token, about 105 ms on a cool phone:

| Phase | ms |
| :--- | ---: |
| Routed experts | ~55 (about a third of the decode waiting for flash) |
| DeltaNet | ~25 |
| LM head | ~8 |
| Attention | ~7 |
| Routers and prediction | ~6 |
| Shared expert | ~4 |

The model reads ~2 GB of resident weights per token, plus ~0.27 GB of experts. RAM bandwidth (~55 GB/s measured)
therefore caps a single token near 20 tok/s.

**Lossy options, measured.** Quality is measured with `liyab-kl` over a 473-token mixed text (prose, code, Italian,
math) against the lossless path.

| Option | Decode | Expert reads/token | KL mean / median | Top-1 agreement |
| :--- | ---: | ---: | ---: | ---: |
| none (default) | 7.41–8.08 tok/s | 283 MiB | — | — |
| `--requant 5` | 7.88 tok/s | 255 MiB | 0.055 / 0.014 | 91.1% |
| `--requant 4` | 8.61 tok/s | 246 MiB | 0.082 / 0.023 | 89.6% |
| `--expert-mass 0.95` | 7.64 tok/s | 273 MiB | 0.027 / 0.007 | 94.9% |
| `--expert-mass 0.9` (~6.5 experts) | 8.19 tok/s | 234 MiB | 0.039 / 0.012 | 92.0% |
| `--expert-mass 0.8` (~5.5 experts) | 9.49 tok/s | 164 MiB | 0.092 / 0.034 | 90.1% |

A larger memory budget is free in quality. At `--memory-budget 7000` expert reads drop from 284 to 208 MiB per token
and decode goes from 8.29 to 9.50 tok/s. On an Apple M4 Pro (48 GB, CPU, 10 threads) the same file decodes at 48–50
tok/s fully in RAM, and at 27.5 tok/s within the phone's 5,500 MiB budget (single runs).

### A dense 27B larger than RAM

Qwen3.8-27B UD-IQ2_S, 8.4 GB, with 8.2 GB of free RAM:

| Engine | Prompt (18 tokens) | Decode |
| :--- | ---: | ---: |
| `mmap` streaming window, CPU | 71 s | 0.14 tok/s |
| Block streaming, CPU | 71 s | 0.17 tok/s |
| Block streaming + GPU-shared memory + native Vulkan kernels | 8.7 s | **0.94 tok/s** |

Dense 30B-class models are bandwidth-bound on phones. Every token reads every weight, so MoE models and speculative
decoding are the practical routes to interactive speeds.

### Models that fit in RAM

| Model | CPU (NEON + SDOT, 4 threads) | GPU (Vulkan) |
| :--- | ---: | ---: |
| TinyLlama-1.1B Q4_0 | 31.0 tok/s | **38.6 tok/s (+25%)** |
| TinyLlama-1.1B Q4_K_M | 24.7 tok/s | 20.7 tok/s |
| Qwen2.5-0.5B Q4_K_M | 35.2 tok/s | 27.9 tok/s |
| Qwen3.5-0.8B Q4_K_M (hybrid) | 22.6 tok/s | — |
| Qwen3.5-2B Q4_K_M (hybrid) | 11.8 tok/s | 12.3 tok/s |

Models that fit use the GPU's repacked-copy path, where some formats fall back to the CPU. Streamed models use the
native kernels instead.

### Latent attention prefill (Mac)

Youtu-LLM-2B (`deepseek2`, Q8_0 weights and KV cache) prefilling synthetic token ids on the Mac's CPU (10 threads,
`liyab-bench --prefill`), each setting run twice, alternated, while another job kept 8 cores busy, in tokens per
second:

| Prompt | Absorbed, passes of 64 (before) | Absorbed, passes of 512 | Expanded, passes of 512 | Automatic, passes of 512 |
| ---: | ---: | ---: | ---: | ---: |
| 512 tokens | 111 / 114 | 129 / 129 | 149 / 153 | 149 / 148 |
| 2,048 tokens | 80 / 80 | 90 / 92 | 120 / 120 | 119 / 123 |
| 8,192 tokens | 40 / 37 | 58 | 74 / 63 | 101 |

The automatic form chose the expanded one for every pass of 512 tokens. At 8,192 tokens the load moved the runs the
most; on a quieter machine the same prompt ran at 57.6 tok/s in passes of 64 (absorbed) and 121.3 in passes of 512
(expanded), 2.1 times faster. In passes of 64 the expanded form is no faster (2,048 tokens: 70 / 68 tok/s), which is
why the engine prefills latent-attention models in passes of 512. Decoding is absorbed in every case and unchanged.

### Per-block KV

Windowed blocks keeping only their window ([KV laid out per block](architecture.md#architecture)), Q8_0 KV cache, the Mac's CPU
(`liyab-cli --json`, greedy, a long prompt then 64 tokens, the build before and after alternated two or three
times; KV per position is what the blocks keeping every position cost, every block's before):

| Model | KV per position | KV after the prompt | Automatic context (256 MiB) | Prefill / decoding (median) |
| :--- | ---: | ---: | ---: | ---: |
| gemma-3-1b-it Q8_0 (4 of 26 blocks global, window 512), 4,185 tokens: before | 13.8 KiB | 57.8 MiB | 18,978 | 9,569 ms / 134.0 tok/s |
| per block | 2.1 KiB | 16.2 MiB | 32,768 (its training context) | 9,506 ms / 134.8 tok/s |
| gemma-3-4b-it Q4_K_M (5 of 34 global, window 1,024), 4,185 tokens (`--ctx 8192`): before | 72.2 KiB | 302.5 MiB | 4,096 | 37,872 ms / 53.5 tok/s |
| per block | 10.6 KiB | 113.8 MiB | 17,990 | 37,974 ms / 53.5 tok/s |
| gemma-2-2b-it Q8_0 (13 of 26 global, window 4,096), 6,157 tokens (`--ctx 8192`): before | 55.2 KiB | 338.4 MiB | 4,744 | 49,575 ms / 39.0 tok/s |
| per block | 27.6 KiB | 283.2 MiB | 5,265 | 49,386 ms / 39.2 tok/s |

The replies are the same tokens. The windows' share is fixed (Gemma 2's window is half its training context, so it
gains least); past it, a position costs only the global blocks' rows. Models without windowed blocks are
unchanged: Qwen3-0.6B (BF16) kept its 189.7 MiB of KV at 3,144 positions, its context (4,405) and its reply, at the
same speed (prefill median 24.5 / 23.7 s, decoding 59.4 / 62.6 tok/s, before / after, alternated), and so did
Llama 3.2 1B (51.0 MiB at 2,989 positions, context 15,420, 37.6 / 37.7 s, 53.3 / 53.6 tok/s). Attention alone at
24,000 positions (`LIYAB_BENCH_ATTENTION`, a row fetched per position and KV head): 1.50 ms per token and layer
before, 1.49 after (medians of 4, alternated).

On gemma-3-1b-it with a 2,094-token system prompt: a context saved by the build before (format 3, 29.9 MB) loads
converted and gives the same reply as one never saved, as does one saved now (format 4, 11.5 MB); the build before
refuses the new file. Through the prefix cache on storage, a 2,133-token context set aside at exit takes 12.8 MB (9
chunks and 11 window pages; its rows of every block would have taken 28.7 MB), and the next run, whose prompt leaves
it at token 2,094, restores 2,048 tokens and replies as a run without the cache.

### Storage and kernels

| Direct reads from UFS 4 | Throughput |
| :--- | ---: |
| `mmap` + page touch | 2.28 GB/s |
| `pread` + `O_DIRECT`, single requests | 2.80 GB/s |
| `pread` + `O_DIRECT`, 2–4 concurrent requests ≥ 256 KiB | **~4.4 GB/s** |
| `pread` + `O_DIRECT`, 16 KiB requests, 1–8 threads | 0.16–1.18 GB/s |

| Matvec on real tensors (GB/s) | CPU, generic (start) | CPU, NEON | GPU, native Vulkan |
| :--- | ---: | ---: | ---: |
| Q2_K | 0.24 | 12.6 | 15.5 |
| IQ2_XXS | 0.70 | 8.4 | 15.7 |
| IQ3_S | 2.07 | 6.7 | 21.1 |
| IQ4_XS | 0.99 | 21.7 | 24.1 |
| Q4_K | 16.8 | 16.8 | 17.4 |

With i8mm repacking, a 2048×8192 matmul over 4 rows on 8 threads takes:

| Format | Before | Now |
| :--- | ---: | ---: |
| Q8_0 | 500 µs | 380 µs |
| Q4_K | 623 µs | 300 µs |
| Q5_K | 601 µs | 408 µs |
| Q6_K | 728 µs | 387 µs |

KV memory per token for a Llama-3-8B-class shape:

| Format | Per token | vs F16 |
| :--- | ---: | ---: |
| F16 | 128 KiB | — |
| Q8_0 | 68 KiB | −47% |
| Q4_0 | 36 KiB | −72% |
| Q4_1 | 40 KiB | −69% |

### Correctness

The stage-1 integration passes 16/16 native, 16/16 Rosetta scalar and 16/16 Rosetta AVX2 tests, preserving
all 414 golden pins (138 cases per platform, including every original pin). All 66 merged-tree tiny recipes
pass, including one 39-input embedding check; Android Release experimental builds and Flutter has 53 passing
tests with two optional real-model skips. Independent reader proofs cover 520 recurrent and 104 mapped-table
cases across 26 formats and four KV types. The final 72-pair performance comparison passes all 21 metrics
(−2.088% to +0.617%) with identical CLI replies and token counts. Phone validation of the stage-1 changes is deferred; see the
[validation report](../model-coverage-stage1.md). These host results do not imply new on-device throughput
or compatibility figures.

All 26 supported reader formats decode bit-exactly like the GGUF reference decoder, and the tokenizers produce
the Hugging Face tokenizers' ids, special tokens, CJK and emoji included. Float models give logits within 0.0015 of the
reference implementation in the earlier float checks. Per-checkpoint KL and argmax agreement are reported below;
F16 KV rounding can change near-tied winners. On Qwen3.5 Q8_0 and Q4_K_M the top token matches
at 98.9% of positions, and the remaining differences are the reference's own second choice. Streamed and in-place
weights, the file's layouts and the repacked ones, batched and token-by-token prefill, any thread count, and
speculative and plain greedy decoding all give bit-identical logits (tested for every format and KV cache type). On
the phone, nine runs of the same chat on the 35B gave the same greedy replies, and the app gave the CLI's.

The classic transformer extension passes 20 tiny cases across 16 families and nine full-checkpoint gates.
Its controlled main/branch comparison on the M4 Pro passes all 14 prefill/decode metrics within ±3%, using
48 balanced pairs on AC power. BF16 Metal-route measurements include CPU matrix fallback; F16 measurements
verify actual Metal attention and FFNs. [Classic validation](../model-descriptions.md#classic-performance-regression-check)
records the method, measured rates and earlier failed attempts. No engine performance fix is claimed.

**Against Hugging Face transformers.** `liyab-kl` compared BF16/F16 GGUFs with transformers' own implementation in
float32 (`tools/conform_reference.py`) over `tests/data/conform_text.txt`, teacher-forced. Every supported tokenizer
gave the reference's ids for the whole text. The Hunyuan models were measured on the reference's ids
(`--ref-tokens`), before their pre-tokenizers were added; those now give the same ids on their own. With an F16 KV
cache the forward passes agree:

| Model (architecture) | Tokens | Mean KL | Top-1 agreement |
| :--- | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct (`llama`) | 499 | 0.00000 | 99.80% |
| Qwen2.5-0.5B-Instruct (`qwen2`) | 512 | 0.00001 | 99.80% |
| Qwen3-0.6B (`qwen3`) | 512 | 0.00000 | 99.80% |
| Qwen3-0.6B with YaRN (factor 4 over 32,768 positions, set on a copy of the GGUF and in transformers' config) | 512 | 0.00000 | 100.00% |
| Qwen3.5-0.8B (`qwen35`, hybrid) | 510 | 0.00000 | 100.00% |
| Hunyuan-0.5B-Instruct (`hunyuan-dense`, reference ids, text after its BOS) | 500 | 0.00000 | 100.00% |
| Hunyuan-MT-7B (`hunyuan-dense`, reference ids, text after its BOS) | 512 | 0.00000 | 100.00% |
| gemma-3-270m-it (`gemma3`) | 512 | 0.00000 | 100.00% |
| gemma-3-270m-it, a longer text: 4 times its 512-position window | 2,048 | 0.00000 | 99.95% |
| gemma-2-2b-it (`gemma2`, soft caps on the attention scores and the logits) | 512 | 0.00000 | 100.00% |
| gemma-2-2b-it, a longer text past its 4,096-position window | 4,400 | 0.00000 | 100.00% |
| Phi-3.5-mini-instruct (`phi3`, LongRoPE short factors) | 512 | 0.00000 | 99.80% |
| Phi-3.5-mini-instruct, a longer text past its 4,096-position original context (long factors) | 4,200 | 0.00000 | 99.95% |
| Youtu-LLM-2B (`deepseek2`: latent attention, low-rank queries; BF16 converted from the Hugging Face weights) | 512 | 0.00000 | 100.00% |
| SmolLM2-360M-Instruct (`llama`, `smollm` pre-tokenizer; bartowski f16) | 512 | 0.00000 | 100.00% |
| ERNIE-4.5-0.3B-PT (`ernie4_5`; unsloth F16, its tokenizer keys fixed) | 512 | 0.00000 | 100.00% |
| Youtu-LLM-2B, latent attention in the expanded form (passes of 512 tokens) | 512 | 0.00000 | 99.80% |
| Huihui-MoE-1.2B-A0.6B (`qwen3moe`, Qwen3-0.6B upcycled into 3 experts) | 512 | 0.00000 | 99.61% |
| Huihui-MoE-1.2B-A0.6B, its experts fused into `ffn_gate_up_exps` | 512 | 0.00000 | 99.61% |
| state-spaces/mamba-130m-hf (`mamba`, F16) | 512 | 0.00001 | 99.61% |
| state-spaces/mamba-370m-hf (`mamba`, F16) | 512 | 0.00000 | 100.00% |
| AntonV/mamba2-130m-hf (`mamba2`, F16) | 512 | 0.00000 | 100.00% |
| Gemma-3n-E2B-it (`gemma3n`; unsloth checkpoint converted to BF16) | 512 | 0.00000 | 100.00% |

Gemma 3n also passes against a BF16 transformers reference (mean KL 0.00168, top-1 98.24%); CPU and Metal routes
give the same greedy reply to the checked chat prompt, "The capital of France is Paris." BF16 matmuls retain the
existing per-operation CPU fallback. Its four-block F32 tiny checkpoint, including every new part and windows
within the text, gives mean KL 0.00000 and top-1 99.41%. A deeper ten-block random diagnostic gives mean KL
0.00001 but fails the 99% top-1 threshold (91.78%); adding F16 KV rounding to that reference improves it to
98.83%, so precision accounts for most of that difference. That diagnostic is not a passing conformance case.

Quiet baseline checks on Qwen3-0.6B and Llama-3.2-1B BF16 used six warmed pairs per configuration, alternating
main/branch order. Median paired CPU prefill throughput changed by −0.34% / +0.27%; 64-token decode time changed
by −0.74% / −0.42% on CPU and +1.07% / −0.17% on the Metal route. CLI prefill changes stayed within 0.11%.
All 36 paired CLI replies were byte-identical. These BF16 Metal-route measurements use the existing CPU fallback
for matrix operations; they do not measure native GPU matrix throughput.

Youtu-LLM-2B now encodes the text itself (its `youtu` pre-tokenizer), so its Q8_0 cache gets measured key scales:
mean KL 0.00065, top-1 97.65% (0.00115, 97.26% with keys as computed). Tencent's own `Youtu-LLM-2B-F16.gguf` holds
slightly other weights than the Hugging Face repository (16–36% of the values of the tensors compared are equal
after rounding, relative error about 1e-3), so it measures mean KL 0.0016 and top-1 97.46% against it, with the
same build and an F16 cache, where the BF16 conversion measures 0.00000.

The fused Huihui copy (written by `tools/gguf_fuse_experts.py`: the converter's `--fuse-gate-up-exps` does not
cover `qwen3moe`), its experts streamed through a 128 MB cache, gave the split file's distributions exactly (KL 0,
top-1 100%). Small random models built by transformers' own DeepSeek code with its structures, converted by
llama.cpp's converter to F32, were also compared (F16 KV, 511
positions; weights drawn wide, so attention and outputs are sharp and a wrong part shows):

* DeepSeek-V2-Lite's structure (latent attention without low-rank queries, YaRN with factor 40 and its mscale, a
  dense first block, softmax top-3 of 8 experts unnormalized and scaled by 1.5, two shared experts): mean KL
  0.00010, top-1 99.41% with the router's scores spread out; 0.00059, 99.61% with them as drawn, where a few
  positions sit at a near-tie between two experts and choose the other. Changing the file's expert scale or YaRN
  log multiplier moves the KL to 1.1 and 4.2. The same structure with dense blocks only: 0.00003, 100.00%.
* DeepSeek-V3's structure with one expert group (low-rank queries, sigmoid scores with a selection bias,
  renormalized top-3 scaled by 2.5, one shared expert): median KL 0.00001, top-1 99.80%; the mean, 0.0037, is one
  position whose expert choice flipped at a near-tie.

Group-limited routing has synthetic logit hashes pinned on ARM64 NEON dotprod, x86 AVX2 and x86 scalar for
softmax group maxima and sigmoid groups scored by their two best biased probabilities, both in place and
streamed. The existing dense and latent-attention hashes are unchanged.

Canonical tiny checkpoints from the pinned official configurations also exercise the current converter layout.
DeepSeek-V2-Lite-Chat (`85864749cd611b4353ce1decdb286193298f64c7`) preserves its original latent widths and
single query projection: 512 identical token IDs, 511 positions, mean KL 0.00000 and top-1 99.80% with F32
weights and F16 KV. DeepSeek-V3 (`e815299b0bcbac849fa540c768ef21845365c9eb`) includes three dense blocks and
one grouped MoE block, selecting four of eight groups before eight of sixteen experts: 485 identical IDs,
484 positions, mean KL 0.00000 and top-1 99.38%. Both propose the existing `deepseek2` row and match all three
tokenizer texts. These checks use random weights. Separately, the published
`mradermacher/DeepSeek-V2-Lite-Chat-GGUF` Q8_0 file at revision `9ee9f1f8f39f673e1022e176d586aef4228f630d`
passed a coherent 64-token real smoke run and exact resident/streamed reply and count checks at both cache
sizes reported above. Its tokenizer matches 617/1,013/241 IDs and its template matches all 14 reference
renders. An earlier revision explicitly refuses its legacy combined layout, so this supplies real execution
and mode-equivalence evidence without an earlier real-weight logit reference or a real KL claim.

Older latent-attention files with combined `attn_kv_b` tensors use the same forward paths. Their expanded-head
metadata is normalized to the latent cache layout. A block's key expansion is decoded and transposed to F32
while its quantized value rows are copied unchanged; these temporary buffers live only while that block runs,
including during layer streaming, and their peak bytes are reserved before sizing the expert cache.

Shared K=V compression keeps bounded channelwise softmax accumulators and stores completed entries with their
closing token in paged F32 auxiliary rows. Draft rollback, branch ancestry, sequences and saved prefixes carry
both together. CPU fixtures cover sliding, overlapping CSA and nonoverlapping HCA attention, inverse output
RoPE, grouped low-rank output and manifold-constrained residual streams. Against native transformers 5.19,
the 512-token component stack at the trained head/index widths measures mean KL 0.00000065 and top-1 99.80%;
an isolated first-layer CSA test matches all 512 top-2 selections. These fixtures explicitly use dense FFNs
and omit learned sinks; they do not establish full DeepSeek-V4 support. Projections use the normal backend
route; compression, index scoring, attention, residual mixing and grouped output slices run on CPU (NEON,
AVX2 and scalar). The F32 component fixture also routes projections through each available Metal/Vulkan
backend and compares with the independent dense reference using the same error bound.
Component projection tests cover all 26 storage formats and attention widths different
from the residual width, using amplitude-balanced encoded block fixtures for formats without a writer.
The Android regression runner includes `test_compression` when that executable is built.
Raw rows remain allocated for the finite context, and auxiliary channels are reserved at
every position in every block, including nonclosing positions; memory planning includes that cost.

The final M4 Pro regression check used four CPU threads, F16 KV, one warmup and six balanced, back-to-back
baseline/branch pairs per configuration. The 512-token CPU `liyab-bench` prefill medians were:

| Original BF16 model | Saved baseline (tok/s) | Branch (tok/s) |
| :--- | ---: | ---: |
| Qwen3-0.6B | 87.95 | 88.00 |
| Llama-3.2-1B-Instruct | 36.95 | 37.00 |
| Qwen3.5-0.8B | 83.40 | 83.65 |

CLI prefill/decode pairs also covered CPU, the BF16 Metal route with CPU matrix fallback, and actual Metal
routes using F16 derivatives. An ordinary request rendered through each model's own template produced 128
generated tokens in every measured turn; all 54 reply-byte/count pairs matched. Across 12 configurations,
72 balanced pairs and 21 metrics, median paired speed ratios ranged from 0.993247 to 1.011333, within the
unchanged ±3% gate. The six executable/library hashes stayed fixed, with no pacing, cancellation or thermal
rerouting. Other model, build, test and download jobs were stopped for these measurements.

GLM-4.7-Flash remains a real-model streaming smoke result, not a conformance claim. Its two-block F32 tiny
checkpoint converted with b11819 matches 494 token IDs but fails numerically (mean KL 0.00019, top-1 58.22%).
The export omits sigmoid `expert_gating_func`, and its `deepseek2` label does not distinguish the reference's
1e-6 latent normalization from its 1e-5 block normalization. A diagnostic copy with sigmoid metadata and the
derived latent-epsilon row reaches mean KL 0.00000 and top-1 100%; that changes the file and the scratch row,
so it is not the branch's support gate. The published layout needs these facts represented unambiguously.

The paged KV cache can carry auxiliary F32 index-key channels after each latent key row. They keep their original
precision independently of F16, Q8_0, Q4_0 or Q4_1 main rows; each key row's byte size includes both encodings.
Rounding an index score can change its discrete
top-k selection. Page copies, rollback, context images and the prefix store carry these channels with the main
row; key scaling leaves them unchanged. Their bytes count toward the memory budget and sequence context size.
Mixed-format images use the store's raw-image fallback, including when lossy prefix compression is requested,
so storage does not change index keys. Ordinary KV images retain their existing codecs and layout.

**Sparse latent attention.** Lightning indexers project queries from the normalized query latent and keys from
the block's normalized input, apply a separate key LayerNorm and RoPE, then combine rectified per-head dot
products with learned signed weights. Each query attends only to its highest-scoring causal positions. Shared
indexer blocks reuse the preceding full block's selected positions (`attention.indexer.types`). Main and indexer
RoPE pairings are independent registry facts; GLM's latent RMS norms use epsilon 1e-6 independently of its block
norm epsilon. With `index_topk` at least the visible context, attention is bit-identical to dense latent
attention. Equal index scores select earlier positions first, preserving batching and rollback determinism.
The description tool accepts F32 metadata rounding when comparing latent and block norm epsilons.
The `deepseek32` row supplies the alternative indexer pairing. Its four-block tiny checkpoint at `index_topk=384`
gives 485 identical token IDs, mean KL 0.00000 and top-1 99.59%; all 485 first-layer selections at top-k 8 match
transformers exactly. Top-k 256 diagnostics at residual widths 256 and 512 both miss top-1 (98.76%).
Pooled index keys are refused precisely, and sparse models currently refuse approximate segment relocation.
Indexer scoring and both sparse attention forms use CPU scalar/NEON/AVX paths; projections use the existing
operation routing, including Metal and Vulkan CPU fallbacks. The absorbed/expanded choice self-calibrates with
valid synthetic index keys and accounts for the restricted absorbed attention work.

A tiny GLM-5 checkpoint with all four blocks (three dense FFNs and one MoE), original latent/indexer widths and
`index_topk=256` gives 494 identical token IDs, mean KL 0.00000 (maximum 0.00001), top-1 99.80% against transformers
5.19 in F32, with F16 KV. At `index_topk=8`, all 494 first-layer position selections match transformers exactly.
The full four-layer F32-reference diagnostics with top-k 8 and 32 fail the 99% top-1 gate (98.78% and 92.90%);
top-k 32 compared against a diagnostic reference with the same F16 main-KV rounding reaches 99.59%. Those
diagnostics expose numerical sensitivity at discrete selection boundaries and are not the support gate.
The model's own chat template passes all 14 reference renders. Its `glm4` pre-tokenizer uses the Llama-3 split
expression and merge rules. No real GLM-5 weights were loaded.

GLM-5.2's four-block tiny checkpoint includes three full indexers and one shared-indexer/MoE block: 494 identical
IDs, 493 positions, mean KL 0.00000, maximum 0.00008 and top-1 99.80% with the original latent/index widths and
top-k 256. Its template passes 14 renders. Ling-mini-2.0's two-block F32 checkpoint covers fused QKV, partial RoPE,
Q/K norms, dense and grouped MoE FFNs plus a shared expert: 512 identical IDs, mean KL 0.00000, maximum 0.00040,
top-1 99.61% with F16 KV. Its older remote code requires transformers-5 compatibility shims, including restoring
the nonpersistent RoPE buffer from its constructor formula after loading. Its template passes 14 renders.
GLM-5.2 and Ling tokenizers match all three tokenizer texts. These are tiny-checkpoint results, without real
weights. GLM-5.1's four-block checkpoint at top-k 256 gives 494 identical IDs, mean KL 0.00000, maximum 0.00001,
top-1 99.80%, with all three tokenizer texts and 14 template renders matching.
The description tool verifies individually observed expert calls, direct router linear operations and causal
PyTorch SDPA or eager attention when an older implementation does not use transformers' attention registry.
Eager scoring must match a scaled dot product with a full causal mask; unknown transforms are refused. Explicit
`expert_weights_norm` metadata overrides the registry fallback, which the tool preserves for older files.
Kimi K2's tiny numerical check on reference IDs gives mean KL 0.00000 and top-1 99.80%. Its `kimi-k2`
pre-tokenizer and tokenizer-side tool sorting are integrated; complete cross-branch conformance remains to be established.

**Latent attention's two forms.** The absorbed and the expanded form ([Supported models](models.md#supported-models)) compute
the same function and round differently, so their outputs are not bit-identical; each is bit-identical on its own
for every split into passes and thread count. Measured with `liyab-kl` against transformers in float32 (Youtu-LLM-2B
on the reference's ids, and a new random model with DeepSeek-V2-Lite's structure built by transformers' DeepSeek-V2
code at V2-Lite's attention widths, 16 heads, a 512-wide latent, keys of 128 + 64 and values of 128, converted to
F32), 511 positions:

| Model, KV cache | Absorbed: mean KL / top-1 | Expanded: mean KL / top-1 |
| :--- | ---: | ---: |
| Youtu-LLM-2B (BF16), F16 | 0.00000 / 100.00% | 0.00000 / 99.80% |
| Youtu-LLM-2B (BF16), Q8_0 with keys unscaled (measured before its `youtu` pre-tokenizer was a row; with key scaling the absorbed form gives 0.00065 / 97.65%) | 0.00115 / 97.26% | 0.00098 / 98.43% |
| random V2-Lite structure (F32), F16 | 0.00000 / 99.02% | 0.00000 / 99.02% |
| random V2-Lite structure (F32), Q8_0 with key scaling | 0.00308 / 67.51% | 0.00186 / 74.76% |

With an F16 cache the forms are indistinguishable at this precision (on the random model, the expanded form against
the absorbed one: KL 0.00000, top-1 99.61%, the rest near-ties of a random model's flat distributions). With a Q8_0
cache the expanded form is the closer of the two: the absorbed form rounds each 576-wide absorbed query to Q8_0 to
score it against the cache, the expanded one scores in float against the decoded latents. Passes of 64 or 512 tokens
give the same expanded figures, as they must.

**Tokenizers against Hugging Face.** `liyab-tok` compared Liyab's ids with those of each model's own
`tokenizer.json`, run by the Hugging Face `tokenizers` library (`tools/tokenizer_reference.py`), on three texts:
`conform_text.txt`, `tokenizer_edge.txt` (contractions in both cases, digit runs, white-space runs, `\r\n`, CJK,
decomposed Hangul, emoji sequences, composed and decomposed accents, Devanagari, Thai, Hebrew and Arabic with marks,
code) and `tokenizer_chat.txt` (the chat special tokens of several families in one text). Vocabularies without a
local model came from the first megabytes of a published GGUF (`tools/gguf_vocab.py`). Every one gave the same ids:

| Tokenizer (`tokenizer.ggml.pre`) | Model | Tokens: conform / edge / chat |
| :--- | :--- | ---: |
| SentencePiece | Gemma-3-270m-it | 523 / 801 / 211 |
| SentencePiece | Phi-3.5-mini-instruct | 644 / 1,053 / 246 |
| `gpt-2` | GPT-2 | 655 / 1,077 / 276 |
| `default` (GPT-2 scheme) | Phi-2 | 641 / 1,056 / 276 |
| `llama-bpe` | Llama-3.2-1B-Instruct | 498 / 797 / 202 |
| `qwen2` | Qwen2.5-0.5B-Instruct | 524 / 804 / 181 |
| `qwen2` | Qwen3-0.6B | 524 / 804 / 179 |
| `qwen35` | Qwen3.5-0.8B | 510 / 790 / 186 |
| `gpt-4o` | Phi-4-mini-instruct | 484 / 725 / 198 |
| `deepseek-llm` | DeepSeek-LLM-7B-chat | 618 / 1,013 / 241 |
| `deepseek-v3` | DeepSeek-V3.1 | 485 / 783 / 215 |
| `hunyuan` | Hunyuan-MT-7B | 546 / 890 / 204 |
| `hunyuan` | Hunyuan-A13B-Instruct | 546 / 890 / 204 |
| `hunyuan-dense` | Hunyuan-0.5B-Instruct | 499 / 818 / 194 |
| `smollm` | SmolLM2-360M-Instruct | 676 / 1,093 / 229 |
| `tekken` | Mistral-Nemo-Instruct-2407 | 517 / 879 / 235 |
| `youtu` | Youtu-LLM-2B | 550 / 879 / 203 |

The quantized KV caches against the same references (Qwen3-4B Q4_K_M against its own F16-cache run), with keys
stored as computed and with [key scaling](architecture.md#architecture) (the default), as mean KL and top-1 agreement:

| Model (largest key channel / median) | Q8_0, keys as computed | Q8_0, default | Q4_1, keys as computed | Q4_1, default |
| :--- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct (52×, scaled) | 0.070, 86.9% | 0.0018, 97.7% | 2.51, 29.2% | 0.47, 63.2% |
| Qwen3-0.6B (107×, scaled) | 0.051, 88.7% | 0.0021, 97.9% | 3.23, 19.6% | 0.11, 81.2% |
| Qwen3-4B Q4_K_M (87×, scaled) | 0.015, 94.3% | 0.0063, 96.3% | 0.63, 69.5% | 0.10, 87.1% |
| Llama-3.2-1B-Instruct (6.3×, as computed) | 0.00090, 97.2% | same | 0.078, 79.7% | same |
| Qwen3.5-0.8B (10.5×, as computed) | 0.00007, 99.2% | same | 0.010, 92.5% | same |

What remains on Qwen3-4B is the 8-bit rounding itself: rounding only its values to Q8_0 costs as much as rounding
only its scaled keys. Measured on the phone the same way, Q8_0 costs Qwen3.5-4B 0.0019 (97.0%) and the Qwen3.6-35B
MoE 0.0056 (95.7%), whose keys need no scaling. `--kv f16` removes the remaining difference at twice the KV memory.

### Charts

![Qwen3.6-35B-A3B decode speed by engine version](../benchmarks/moe_35b_decode.svg)

![Qwen3.6-35B-A3B decode time per token by phase](../benchmarks/moe_35b_phases.svg)

![Qwen3.8-27B UD-IQ2_S decode speed by engine version](../benchmarks/dense_27b_decode.svg)

![Qwen3.8-27B UD-IQ2_S prompt time by engine version](../benchmarks/dense_27b_prompt.svg)

![Direct random read throughput by request size and thread count](../benchmarks/flash_reads.svg)

![Matrix-vector throughput by weight format](../benchmarks/matvec_formats.svg)

![Decode speed of models that fit in RAM, CPU vs GPU](../benchmarks/in_ram_decode.svg)

### What we found along the way

Facts measured while building Liyab that shaped the code, kept here because they are easy to get wrong.

* **Some model families need help to fit their keys in 8 bits.** Qwen2.5 and Qwen3 carry a few key channels 50–107
  times their head's median, nearly constant across tokens; a Q8_0 cache then rounded everything near them with a
  step a hundred times too coarse (KL 0.05–0.07 against F16). Scaling keys per channel, with the inverse on the
  queries, removes it (KL ~0.002) at no cost in memory or speed. Qwen3.5/3.6, the 27B and Llama 3.2 have no such
  channels (6–11×): scaling gained nothing there, so they keep their keys bit for bit
  ([key scaling](architecture.md#architecture)). What a Q8_0 cache still costs the 35B (KL 0.0056, 95.7% top-1) is 8-bit rounding
  itself; values alone cost as much as keys alone. Only an F16 cache removes it.
* **A published GGUF may not hold the weights of the model it names.** The official Qwen2.5-0.5B-Instruct GGUF
  matched its Hugging Face model only in the biases; a GGUF converted from the Hugging Face weights matched at KL
  0.00001. When conformance fails from the first token, compare tensors before suspecting the engine.
* **The Mac and the phone compute the same bits.** The same kernel set (NEON with dot product) gives identical
  logits on an Apple M4 Pro and a Snapdragon 8 Elite, through different compilers and operating systems
  (`test_golden`). Reproducibility across batch splits, thread counts and layouts holds on both.
* **A model may write one character in two tokens.** Byte-level vocabularies split rare characters and emoji, so a
  conversation's text can end inside a character; matching the next prompt must step back to a whole character,
  or a hybrid model processes the whole conversation again.
* **Tokenizing a prompt can be slow enough to matter.** Before tokenizers became data, an 81 KB Llama chat took
  1.35 s to encode and a 75 KB ChatML one 158 ms, on every prompt typed ahead; now 2.5–3.5 ms.
* **Not every chat template can be reused across turns.** Qwen3 and Qwen3.5 drop earlier reasoning from the
  rendered history, so turn n is not a prefix of turn n + 1; Liyab detects that at load and keeps its own format
  there. Llama 3's template writes today's date, so its system prompt changes once a day.
* **A stop inside a speculative step must cut the step.** Tokens verified past an end-of-generation token stayed
  in the cache, and the next turn diverged from plain decoding.
* **Experts are not low-rank.** Qwen3.6's routed experts need ~65% of their rank for 90% of their energy; the
  least used quarter of them carries only 5–8% of routings. A smaller tier for all experts in Q3_K costs about
  what `--expert-mass 0.9` costs ([study](../research/cold-experts.md)).
* **HyperOS starts nothing in the background without Autostart.** Android binds the notification listener after a
  reboot, but HyperOS rejects it unless Liyab's Autostart is allowed; with it, the local API answers a request
  after a reboot without Liyab ever being opened (first token 6.4 s, the 35B loaded from cold).
* **Swap distorts speed measurements.** With 7.5 GB of zram in use by other apps, the same build ran prefill
  ~30% slower; alternating A/B runs, not single runs, tell a change from the phone's state.
* **Rosetta 2 is not an x86 benchmark.** It translates `VPSIGNB` and `VPSHUFB` into long sequences, so AVX2
  kernels that are faster natively can run slower there than scalar code.

### What did not pay off (so you don't have to try)

* **Predicting experts two or three blocks ahead.** Late prefetches halved, but ~11% more bytes were read; it tied
  with one block ahead. Guessing 2 extra experts per block was slower too.
* **More or smaller I/O requests.** 8 I/O threads instead of 4, or each matrix split into 2–4 reads, were slower.
  Small reads cost the flash more than the concurrency saves.
* **The kernel page cache as a second expert cache.** Copies from it are fast, but buffered misses hold the I/O
  threads that urgent reads wait for. Decode got slower in every A/B run.
* **i8mm 2×2 tiles for the K-quants.** Two weight rows against two activation rows per SMMLA halves the
  multiply-adds, but the zips that build the tiles, the per-sub-block scale multiplies and the exact epilogue cost
  more: on the Snapdragon 8 Elite (2048×8192, 4 rows) Q4_K went 681 → 601–617 µs, Q5_K 683 → 863–895 µs and Q6_K
  756 → 953–1019 µs. The replies are bit-identical, so it stays as the opt-in `i8mm_tiles` (see
  [Experimental](development.md#experimental-modules)).
* **Speculation on streamed MoE.** A verification pass reads the union of its tokens' experts: 1.55× the cost of
  one token for 2 tokens, 2.5× for 3.
* **2:4 sparsity on phones.** There are no sparse tensor units, so index decoding costs more than the halved
  multiply-accumulates: −54% decode (see [Experimental](development.md#experimental-modules)).
* **Binary low-rank experts (LittleBit-style).** Qwen3.6's experts are nearly full-rank (90% of the energy needs
  ~65% of the rank), so binary factors at 1 bit/weight lose about half of each expert: with the least used quarter
  replaced, KL 0.069 against 0.0095 for Q3_K ([study](../research/cold-experts.md)).
* **Repacking every streamed expert on read.** It added ~0.1–0.2 ms to each decode miss (−6% decode), so only
  experts read for batched passes are repacked.

---
