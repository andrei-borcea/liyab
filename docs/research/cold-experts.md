# Smaller cold experts for the streamed 35B MoE: a measured study

Qwen3.6-35B-A3B decodes on the phone at the speed its routed experts arrive from flash: everything but the experts
stays in RAM, and each token reads the experts the RAM cache does not hold (283 MiB per token at
`--memory-budget 5500`, see the [README](../../README.md#qwen36-35b-a3b-a-22-gb-moe-on-a-phone)). Fewer bytes per
expert means fewer bytes per token. This study measures what smaller representations of the experts cost in
accuracy, first per matrix and then end to end with `liyab-kl`, and projects what they would give in speed.

Nothing in the engine changed for it. Every number below comes from a run described here; numbers labelled
**projection** are computed from a model, not measured on the phone.

## Contents

* [Summary](#summary)
* [Setup](#setup)
* [Are experts low-rank?](#are-experts-low-rank)
* [Representations, per matrix](#representations-per-matrix)
* [Which experts are cold](#which-experts-are-cold)
* [End to end: liyab-kl](#end-to-end-liyab-kl)
* [Bytes and speed (projection)](#bytes-and-speed-projection)
* [Conclusion](#conclusion)
* [Implementation plan for a runtime tier](#implementation-plan-for-a-runtime-tier)
* [Reproducing](#reproducing)

## Summary

* **Experts are not low-rank.** Effective rank 85–86% of full, 90% of the energy at 64–66% of the rank (a matrix
  of shuffled entries: 97% and 76%). A rank-25 SVD (1 bit/weight) leaves 89–91% of the matrix unexplained. Only
  block 0 is partly low-rank.
* **LittleBit-style binary factorization is not worth implementing for these experts.** With our local refinement
  (no training through the model) its output error is 0.52 at 1 bit/weight and 0.22 at 3 bits, worse than plain
  uniform 2-bit (0.21 at 2.63 bits). End to end, the least used 25% of experts at 1 bit cost KL 0.069 (top-1
  91.0%), seven times Q3_K's 0.0095 and over half of removing those experts outright (0.121).
* **A non-uniform 3-bit codebook is a small gain over a well-fitted uniform 3-bit.** At Q3_K's 3.44 bits/weight it
  lowers the output error by 8% (0.109 against 0.118) and the KL with the least used 50% replaced by 15% (0.0133
  against 0.0156), but it needs a new kernel. The earlier quick study's gap (0.169 against 0.237) came from a weak
  uniform baseline.
* **Q3_K (a format the engine already decodes), fitted with input importances, is the candidate.** Replacing the
  least used 25 / 50 / 75 / 100% of each block's experts costs KL 0.0095 / 0.0156 / 0.0253 / 0.0401 against the
  unmodified model (top-1 95.1 / 94.7 / 92.1 / 91.2%; the Q8_0 harness alone accounts for 0.0064 and 97.5%).
  All experts in Q3_K cost about what `--expert-mass 0.9` costs (KL 0.039, top-1 92.0%) and are 29% smaller.
* **IQ3_XXS** saves 11% more bytes than Q3_K but costs more quality (KL 0.0186 at 50%) and 2–2.7× the original's
  decode time per expert on the Mac, which cancels the I/O it saves.
* **"Cold" experts carry few bytes.** The least used 25% of experts carry 4.7% of the routings (7.6% on another
  text), so shrinking only them cuts the bytes read per token by 5–7% (projection). The bytes sit in the middle
  of the usage ranking: a tier pays off only at 50–100% of the experts.
* **Projected speed (not measured on the phone):** Q3_K for 50% of the experts, +4 to +8% at `--memory-budget 5500`;
  for all experts, +9 to +29%. The range is the unknown decode cost of Q3_K on the phone: on the Mac a Q3_K expert
  takes 1.18× (4 threads) to 1.48× (1 thread) the time of the original one. Measuring that on the phone
  (`liyab-bench` now has the shapes) is the first step of the plan below.

## Setup

* **Model.** `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22,134,528,992 bytes): 40 blocks × 256 experts, top-8, hidden size
  2048, expert width 512. Gate and up experts are Q4_K (4.5 bits/weight); down experts are Q5_K (5.5) in 37 blocks
  and Q6_K (6.56) in 3. One expert (gate + up + down, 3 × 1,048,576 weights) is 1,900,544 bytes (Q5_K down) or
  2,039,808 bytes (Q6_K down); the experts total 19.57 GB of the file.
* **Reference.** The errors are measured against the GGUF's own dequantized weights, which are what the phone runs.
  `expert_study.py` dequantizes them with the gguf package's decoders; `check-dequant` compared these with the
  engine's decoder (`liyab-experts dequant`) on Q4_K, Q5_K and Q6_K experts: 0 difference over 2,097,152 values each.
* **Calibration text** (8,192 tokens): the model's own greedy replies to 10 fixed prompts (a story in English, a
  letter in Italian, a short text in Chinese and Japanese, French and German paragraphs, a Python module, a C++
  class, worked math in LaTeX, a JSON configuration, a markdown table of financial figures, an algorithm
  comparison), followed by the start of this repository's README. It gives the routing counts that pick the
  cold experts and each expert's input second moments (E[x_j²] per input column), the only "calibration" any
  representation uses.
* **Evaluation text.** `tests/data/conform_text.txt` (510 tokens: prose, Italian, Python, numbers and URLs, CJK,
  Greek, Cyrillic, Hebrew, Arabic, emoji), never used to choose or fit anything. Its own capture provides the
  held-out inputs on which output errors are measured.
* **Capture.** `liyab-experts capture` runs a text one token at a time (each pass a decoding step) through a
  `Backend` that wraps the CPU backend: it sees each router matmul and each expert matmul the forward pass asks
  for, with the experts read in place from the mapping, so every matmul maps back to its (block, expert). The
  router's top-8 (softmax, renormalized) matched the experts then multiplied on every token of both texts
  (0 mismatches).
* **Sample for per-matrix errors.** Blocks 0, 10, 20, 30 and 39; in each, 16 experts spread over the usage ranking
  (hottest to coldest); their gate, up and down matrices (240 matrices). Relative Frobenius error
  ‖W − Ŵ‖/‖W‖, and relative output error ‖X(W − Ŵ)ᵀ‖/‖XWᵀ‖ on the inputs X those experts received in the
  evaluation text (up to 64 per expert; experts the text never routed to do not count).

## Are experts low-rank?

`expert_study.py spectra`: singular values of 120 dequantized matrices (8 experts per block, blocks 0, 10, 20, 30,
39, each of gate / up / down). Ranks are given as a share of the full rank (512). "Shuffled" is the same matrix
with its entries permuted at random, which keeps the value distribution and removes any structure.

| Matrix | Effective rank (entropy) | Rank for 90% of energy | for 99% | Largest singular value's energy | Rank-25 SVD error (1 bit/weight) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| gate | 0.85 | 0.64 | 0.87 | 3.1% | 0.89 |
| up | 0.86 | 0.66 | 0.87 | 1.4% | 0.91 |
| down | 0.86 | 0.66 | 0.87 | 1.5% | 0.91 |
| shuffled entries | 0.97 | 0.76 | | | |

* Blocks 10 to 39 are close to random matrices: effective rank 0.93–0.94, 90% of the energy needs 69–72% of the
  rank. Block 0 is the exception (effective rank 0.53, 90% of the energy at 41% of the rank; one expert at 0.22).
* Weighting the columns by the inputs' second moments (what the output error sees) lowers the rank for 90% of
  the energy only from 0.65 to 0.60 on average.
* Cold experts are not more compressible: the least used third of the sample has effective rank 0.92 and 90%
  energy at 0.70, the most used third 0.88 and 0.67.
* Nor do a block's experts share an input subspace: the stacked gate matrices of all 256 experts of a block
  (131,072 × 2048) need 75–83% of the 2048 input dimensions for 90% of their energy.

So the earlier finding holds on this model: the experts are nearly full-rank, and a low-rank approximation at
1 bit/weight leaves ~90% of the matrix unexplained.

## Representations, per matrix

All are our own implementations in `tools/expert_study.py`, fitted to each matrix from the GGUF weights alone; the
suffix **w** fits the importance-weighted error (columns weighted by the calibration text's E[x_j²], shrunk toward
the block's mean for rarely used experts), the way an importance matrix does.

* **Q3_K** (3.44 bits/weight): the standard format, which the engine already decodes. Our encoder searches 7 scales
  per 16 values. `check-encoders` decoded its blocks with gguf-py and with the engine: identical to the encoder's
  reconstruction.
* **Q2_K-like** (2.63): uniform 2-bit with 4-bit scale and minimum per 16 values.
* **IQ3_XXS** (3.06): the standard non-uniform 3-bit format (a 256-point grid of 4-value magnitudes plus sign
  patterns with even parity), which the engine already decodes with NEON kernels. Our encoder searches 6 scales
  per 32 values and flips the cheapest sign where the parity requires it; its blocks decode identically in
  gguf-py and in the engine.
* **Non-uniform 3-bit** (our tier candidate): one 8-level codebook per tensor (weighted Lloyd iterations), a signed
  scale per block found by search and refined by least squares. `nu3`: 8-bit scale per 32 values (3.31 bits/weight);
  `nu3s16`: 6-bit scale per 16 values (3.44, Q3_K's budget).
* **2-D vector quantization** (3.31): adjacent pairs from a 64-entry codebook per tensor, 8-bit scale per 32.
* **LittleBit-style binary factorization** (our take on arXiv 2506.13771, no code reused):
  W ≈ diag(h)·sign(U)·diag(ℓ)·sign(V)ᵀ·diag(g) plus a residual path of the same form, the rank of each path set by the
  bit budget (h, g, ℓ in fp16). It differs from the paper where the paper is weak for our use: no quantization-aware
  training through the model. Initialization from a truncated SVD of the importance-weighted matrix (sign of the
  factors, rank-1 fits of their magnitudes), then a local refinement on that weighted error only: alternating least
  squares on h, g, ℓ (closed forms) and exact coordinate descent on the signs, 4 rounds per path. `lbi1` is the
  initialization alone.
* **Truncated SVD** with fp16 factors at the same bit budgets: the low-rank reference.
* **zero**: the expert removed (its output dropped), the 0-bit floor.

Output errors on held-out inputs, root mean square over the sample (blocks 10–39, 192 matrices; block 0 separately):

| Representation | Bits/weight | Bytes/expert | Frobenius error | Output error | gate / up / down | Block 0 output error |
| :--- | ---: | ---: | ---: | ---: | :---: | ---: |
| original (Q4_K / Q5_K) | 4.83 | 1,900,544 | 0 | 0 | | 0 |
| Q3_K | 3.44 | 1,351,680 | 0.145 | 0.120 | 0.089 / 0.121 / 0.143 | 0.108 |
| Q3_K w | 3.44 | 1,351,680 | 0.147 | 0.118 | 0.086 / 0.118 / 0.143 | 0.101 |
| non-uniform, 16-value scales | 3.44 | 1,351,680 | 0.136 | 0.112 | 0.083 / 0.112 / 0.134 | 0.100 |
| non-uniform, 16-value scales, w | 3.44 | 1,351,680 | 0.141 | **0.109** | 0.078 / 0.107 / 0.134 | **0.090** |
| non-uniform, 32-value scales | 3.31 | 1,302,528 | 0.153 | 0.126 | 0.094 / 0.125 / 0.151 | 0.117 |
| non-uniform, 32-value scales, w | 3.31 | 1,302,528 | 0.158 | 0.123 | 0.089 / 0.120 / 0.152 | 0.104 |
| 2-D VQ, 64 entries | 3.31 | 1,302,528 | 0.151 | 0.124 | 0.092 / 0.124 / 0.149 | 0.111 |
| 2-D VQ, w | 3.31 | 1,302,528 | 0.157 | 0.122 | 0.088 / 0.119 / 0.150 | 0.096 |
| IQ3_XXS | 3.06 | 1,204,224 | 0.187 | 0.154 | 0.117 / 0.156 / 0.182 | 0.138 |
| IQ3_XXS w | 3.06 | 1,204,224 | 0.195 | 0.148 | 0.106 / 0.145 / 0.182 | 0.105 |
| Q2_K-like | 2.63 | 1,032,192 | 0.268 | 0.222 | 0.170 / 0.225 / 0.263 | 0.208 |
| Q2_K-like w | 2.63 | 1,032,192 | 0.275 | 0.214 | 0.156 / 0.211 / 0.262 | 0.168 |
| LittleBit 3 bits | 3.00 | 1,178,328 | 0.277 | 0.224 | 0.161 / 0.223 / 0.274 | 0.116 |
| LittleBit 2 bits w | 2.00 | 786,132 | 0.444 | 0.341 | 0.242 / 0.336 / 0.420 | 0.201 |
| LittleBit 2 bits | 2.00 | 786,132 | 0.430 | 0.353 | 0.262 / 0.354 / 0.425 | 0.219 |
| LittleBit 1 bit w | 1.00 | 392,004 | 0.667 | 0.520 | 0.376 / 0.522 / 0.630 | 0.379 |
| LittleBit 1 bit | 1.00 | 392,004 | 0.649 | 0.541 | 0.419 / 0.547 / 0.636 | 0.415 |
| LittleBit 1 bit, initialization only | 1.00 | 392,004 | 0.756 | 0.694 | 0.644 / 0.688 / 0.747 | 0.701 |
| LittleBit 0.5 bits | 0.50 | 194,940 | 0.806 | 0.680 | 0.539 / 0.692 / 0.786 | 0.576 |
| truncated SVD, 1 / 2 / 3 bits | 0.98 / 1.99 / 2.97 | | 0.917 / 0.857 / 0.804 | 0.760 / 0.697 / 0.648 | | 0.704 / 0.622 / 0.557 |
| zero | 0 | 0 | 1 | 1 | | 1 |

What this says:

* **A non-uniform 3-bit grid is a small gain over a well-fitted uniform one.** At Q3_K's budget the codebook
  lowers the output error from 0.118 to 0.109 (−8%). The earlier quick study's gap (relative error 0.169 against
  0.237 for uniform 3-bit) came from a weak uniform baseline: with a per-block scale search, uniform 3-bit is at
  0.145 Frobenius error. Spending the bits on finer scales matters as much as the grid: the same codebook with
  scales per 32 values instead of 16 is worse than Q3_K.
* **IQ3_XXS** saves 11% more bytes than Q3_K and costs 25% more output error (0.148 against 0.118).
* **Importance weighting** helps every representation on the output error (most in block 0, whose inputs have
  strong outlier channels) while raising the Frobenius error, as expected: it spends precision where inputs are.
* **LittleBit does not fit these experts.** At 1 bit/weight the output error is 0.52 (half of removing the expert)
  and at 0.5 bits 0.68. The local refinement matters (0.69 → 0.54 at 1 bit), but the experts are not low-rank, so
  binary factors at a low rank have little to capture. At 3 bits it is at 0.224, worse than uniform 2-bit at
  2.63 bits. Only in block 0, the one low-rank block, is LittleBit at 3 bits close to Q3_K (0.116 against 0.108).

## Which experts are cold

Routing on the calibration text (8,192 tokens × 40 blocks × 8 experts), per block, then averaged:

| Experts by use | Share of routings (calibration) | Share on the evaluation text |
| :--- | ---: | ---: |
| most used 5% (13) | 18.7% | |
| most used 25% (64) | 56.2% | |
| most used 50% (128) | 82.1% | |
| least used 10% | 1.0% | 2.0% |
| least used 25% | 4.7% | 7.6% |
| least used 50% | 17.9% | 24.6% |
| least used 75% | 43.8% | 54.9% |

0.8 experts per block were never routed in the 8,192 calibration tokens. The cold set chosen on one text
carries 1.4–2× its calibration share on another text: usage depends on the content, so a static cold set
leaks onto the cold path more often than its calibration counts promise.

## End to end: liyab-kl

Method: `liyab-experts to-q8` writes a copy of the model with every expert tensor converted to Q8_0 (36.8 GB;
Q8_0 adds about 1/16 of a 4-bit format's rounding). For each variant, `expert_study.py patch` clones that file
(APFS copy-on-write, so only the rewritten experts take space), replaces the least used fraction of each block's
experts (by the calibration counts) with their reconstruction, re-encoded as Q8_0, and `liyab-kl` compares it on
the evaluation text, with experts read in place (`--expert-cache 0`, no memory budget) on an Apple M4 Pro.

* **vs original**: against the reference saved from the unmodified GGUF (the number that matters).
* **vs Q8_0 copy**: against the reference saved from the Q8_0-expert copy, which isolates the replacement itself.

The Q8_0 copy alone measures: KL mean 0.00643 (median 0.00256), top-1 agreement 97.45%, perplexity 7.345 → 7.376.

Replaced: the least used fraction of every block's experts, by calibration counts (2,560 experts per 25%).
The Frobenius error is over all replaced matrices; KL, top-1 and perplexity over the 509 positions of the
evaluation text. The perplexity of a 510-token text moves by a few percent between runs of equal KL, so KL and
top-1 are the measures to read.

| Variant | Replaced | Frobenius error | KL mean / median vs original | KL p99 | Top-1 vs original | Perplexity (7.345) | KL / top-1 vs Q8_0 copy |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8_0 copy only | 0 | | 0.0064 / 0.0026 | 0.037 | 97.45% | 7.376 | — |
| Q3_K w | 25% | 0.147 | 0.0095 / 0.0037 | 0.060 | 95.09% | 7.388 | 0.0088 / 95.09% |
| non-uniform w (16-value scales) | 25% | 0.141 | **0.0091** / 0.0035 | 0.080 | 95.48% | 7.276 | 0.0092 / 96.07% |
| IQ3_XXS w | 25% | 0.194 | 0.0105 / 0.0041 | 0.083 | 95.48% | 7.370 | 0.0111 / 94.50% |
| LittleBit 1 bit w | 25% | 0.668 | 0.0694 / 0.0152 | 0.697 | 90.96% | 7.234 | 0.0673 / 90.37% |
| removed (zero) | 25% | 1 | 0.1209 / 0.0200 | 1.559 | 87.82% | 7.757 | 0.1185 / 87.43% |
| Q3_K w | 50% | 0.147 | 0.0156 / 0.0065 | 0.145 | 94.70% | 7.301 | 0.0132 / 94.70% |
| non-uniform w (16-value scales) | 50% | 0.142 | **0.0133** / 0.0053 | 0.108 | 95.09% | 7.234 | 0.0131 / 96.07% |
| IQ3_XXS w | 50% | 0.195 | 0.0186 / 0.0067 | 0.160 | 92.53% | 7.240 | 0.0169 / 92.14% |
| removed (zero) | 50% | 1 | 0.6921 / 0.1114 | 8.145 | 71.71% | 14.040 | 0.6988 / 70.73% |
| Q3_K w | 75% | 0.147 | 0.0253 / 0.0114 | 0.183 | 92.14% | 7.367 | 0.0233 / 92.73% |
| Q3_K w | 100% | 0.147 | 0.0401 / 0.0177 | 0.253 | 91.16% | 7.514 | 0.0389 / 91.94% |
| Q3_K, unweighted | 100% | 0.145 | 0.0482 / 0.0207 | 0.367 | 90.37% | 7.510 | 0.0464 / 90.57% |

* The ranking of the per-matrix study holds end to end: non-uniform < Q3_K < IQ3_XXS ≪ LittleBit 1 bit < removal.
* Importance weighting matters end to end: all experts in Q3_K cost KL 0.048 unweighted and 0.040 weighted (−17%),
  although the weighted fit has the larger Frobenius error.
* The cost grows faster than the replaced share: 25% of the experts carry ~5–8% of the routings, 100% carry all.

For scale, the existing lossy options on the same text (README): `--requant 5` KL 0.055 / top-1 91.1%,
`--requant 4` 0.082 / 89.6%, `--expert-mass 0.9` 0.039 / 92.0%, `--expert-mass 0.8` 0.092 / 90.1% (measured on the
473-token version of the text).

## Bytes and speed (projection)

**Projection, not a phone measurement.** `expert_study.py simulate` replays the calibration capture's routing trace
(tokens in decode order) through a byte-bounded LFU cache with the ExpertStore's policy (one use per routing,
counters halved every 16 tokens, evict the fewest uses, then the oldest). The cold set comes from the routing
counts of the trace's first 4,096 tokens; the replay runs on the other 4,096 (the first 1,024 of them warm the
cache and are not counted). Cold experts take their tier's size both when read and in the cache. The cache size is
the one the engine plans at `--memory-budget 5500` (`liyab-cli --json` on the Mac: 2,397,417,472 bytes, 12.25% of
the experts) and at 7000 (20.3%).

Bytes per token are then mapped to speed with the phone's measured pair (README): at budget 5500, 284 MiB/token
and 8.29 tok/s; at 7000, 208 MiB/token and 9.50 tok/s, that is 0.202 ms per MiB read. The simulated relative
change in bytes is applied to the measured 284 (or 208) MiB, and the time saved is 0.202 ms per MiB.

The replay has no prediction, so it misses less than the phone (200 simulated against 284 measured MiB per token at
the 5500 cache) and it overstates how much a larger cache saves: from the 5500 to the 7000 cache it predicts
−38% bytes where the phone measured −27%. The I/O-only figures below are therefore upper estimates.

**Decode cost of the tier formats** (measured on the Mac, Apple M4 Pro, not on the phone): `liyab-bench`, which
now includes these shapes, timed one 2048×512 expert matrix from RAM (µs, three runs):

| Format | 1 thread | 4 threads | Bits/weight |
| :--- | ---: | ---: | ---: |
| Q4_K (original gate / up) | 22–25 | 13–15 | 4.5 |
| Q5_K (original down) | 32–37 | 14–16 | 5.5 |
| Q3_K | 38–39 | 16–17 | 3.44 |
| IQ3_XXS | 66–75 | 24–34 | 3.06 |
| Q2_K | 31–32 | 14–22 | 2.63 |

Per expert (gate + up + down), Q3_K takes 1.18× (4 threads) to 1.48× (1 thread) the time of the original
formats, IQ3_XXS 2.07× to 2.65×. On the phone the routed experts take ~55 ms per token, of which about a third is
waiting for flash (README), so ~37 ms of compute. The "with decode cost" projection adds that compute, scaled by
the tier's share of the routings (measured on the evaluation text: 7.6 / 24.6 / 54.9 / 100% for 25 / 50 / 75 /
100% of the experts) and by the ratio above (4-thread ratio / 1-thread ratio). It assumes the extra compute does
not overlap with reads, so it is a lower estimate.

| Tier | Experts in the tier | Simulated MiB/token (5500 cache) | Projected MiB/token | tok/s at 5500 (8.29 measured), I/O only | with decode cost | tok/s at 7000 (9.50 measured), I/O only | with decode cost |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0 | 200 | 284 | 8.29 | | 9.50 | |
| Q3_K | 25% | 190 | 269 | 8.50 (+3%) | +2% / +1% | 9.78 (+3%) | +2% / +2% |
| Q3_K | 50% | 170 | 241 | 8.94 (+8%) | +6% / +4% | 10.34 (+9%) | +7% / +4% |
| Q3_K | 75% | 140 | 199 | 9.67 (+17%) | +13% / +7% | 11.20 (+18%) | +13% / +6% |
| Q3_K | 100% | 105 | 148 | 10.73 (+29%) | +21% / +9% | 12.21 (+29%) | +19% / +6% |
| IQ3_XXS | 25% | 187 | 266 | 8.56 (+3%) | +1% / −1% | 9.85 (+4%) | +1% / −1% |
| IQ3_XXS | 50% | 161 | 229 | 9.13 (+10%) | +1% / −3% | 10.58 (+11%) | +1% / −4% |
| IQ3_XXS | 75% | 125 | 177 | 10.10 (+22%) | 0% / −9% | 11.71 (+23%) | −2% / −12% |
| IQ3_XXS | 100% | 82 | 117 | 11.52 (+39%) | −5% / −18% | 12.97 (+37%) | −10% / −24% |
| 1 bit (LittleBit) | 25% | 172 | 244 | 8.88 (+7%) | no kernel | 10.29 (+8%) | no kernel |
| 1 bit (LittleBit) | 50% | 118 | 168 | 10.29 (+24%) | no kernel | 12.07 (+27%) | no kernel |

The non-uniform 3.44-bit tier reads the same bytes as Q3_K; its decode cost is unknown until a kernel exists.

The cold quarter of the experts carries only ~5% of the routings, so shrinking it barely changes the bytes read
(−5 to −7%): what is rarely used is also rarely read. The bytes are in the middle of the usage ranking.

## Conclusion

* **LittleBit: no.** These experts are nearly full-rank, so binary factors at 0.5–1 bit/weight lose half or more
  of each expert (output error 0.52–0.68), and end to end the least used quarter at 1 bit costs KL 0.069, seven
  times Q3_K. Its quality catches up with 3-bit quantization only at ~3 bits/weight, where it is worse than a
  2.6-bit uniform format. Initializing from the GGUF and refining locally helps (0.69 → 0.54 at 1 bit) but cannot
  make up for the missing low-rank structure.
* **Non-uniform 3-bit tier: not now.** At equal bits it beats Q3_K by 8% in output error and 3–15% in KL. That is
  less than the cost of a new kernel family (CPU NEON, Vulkan, Metal) whose decode cost is still unknown. It is
  worth revisiting if Q3_K's decode cost on the phone turns out to be the limit and a codebook kernel can be made
  cheaper than Q3_K's bit unpacking.
* **IQ3_XXS tier: no.** It reads 11% fewer bytes than Q3_K, costs more quality, and decodes at 2–2.7× the original's
  time on the Mac (on the phone the I-quant kernels were already measured slower per byte: IQ3_S 6.7 GB/s against
  16.8 for Q4_K), which eats the I/O it saves.
* **Q3_K tier, importance-weighted: worth implementing as an opt-in lossy option**, from 50% of the experts
  (KL 0.0156, top-1 94.7%, projected +4 to +8%, 2.8 GB less on flash) to all of them (KL 0.040, top-1 91.2%,
  projected +9 to +29%, 5.7 GB less). All experts in Q3_K cost what `--expert-mass 0.9` costs in quality, and the
  two cut different things (bytes per expert, experts per token). It needs no new kernel. Its speed gain hinges on
  Q3_K's decode cost on the Snapdragon, which must be measured before any engine work.
* **"Cold" is the wrong criterion for bytes.** With the cache of the 5500 MiB budget (12% of the experts), the least
  used experts are rarely read because they are rarely used; the reads come from the middle of the ranking. A
  tier chosen by usage pays off only once it covers half or more of the experts, and its quality cost is then set
  by how many routings land on it.

## Implementation plan for a runtime tier

Conditional on step 1.

1. **Measure Q3_K's decode cost on the phone.** Run `liyab-bench` on the Snapdragon (it now includes Q3_K, IQ3_XXS
   and Q2_K expert shapes) at 1 and 7 threads. Go on only if a Q3_K expert costs at most ~1.25× the original's,
   so that the projection's lower bound for all experts stays at about +9%. If the NEON Q3_K × Q8_K kernel is the
   bottleneck, speeding it up comes first.
2. **One-time conversion on the device, into a smaller file.** A GGUF tensor has one type, so a mixed model keeps
   each block's experts in two stacked tensors, `ffn_{gate,up,down}_exps.weight` (original types) and
   `ffn_{gate,up,down}_exps_q3.weight` (the tier, Q3_K), plus a per-block map from expert id to (tensor, index) in
   the metadata. The converter reads the original file, encodes the tier with this study's importance-weighted
   Q3_K encoder (ported to C++ next to `quantize_row_q4_K`), and writes the new file, which replaces the original
   once complete and verified. Flash use shrinks (−2.8 GB at 50%, −5.7 GB at 100%) instead of growing with a second
   copy. The importances come from a short calibration pass on a fixed text the engine ships (like the key-scaling
   calibration text), not from the user's data.
3. **Which experts.** Rank experts by routing counts on that calibration pass, or by the ExpertStore's persisted hot
   list, which follows the user's own use. The tier fraction is the user-facing setting (off, 50%, 100%), shown with
   its measured KL as the other lossy options are.
4. **ExpertStore.** Size classes per (block, layout) already exist: the tier adds a Q3_K class per block, and
   `segment()` resolves an expert id through the map. Batched passes repack Q4_K/Q5_K/Q6_K to `_R8`; Q3_K has no
   repacked layout, so prefill uses the file layout for tier experts (measure the prefill cost). Prediction, LFU and
   determinism are unchanged: an expert always has one format, so results do not depend on I/O timing.
5. **Kernels.** CPU: the existing Q3_K × Q8_K NEON kernel. Vulkan and Metal: check Q3_K coverage for experts and add
   it where missing.
6. **Validation.** `liyab-kl` against the original file at each tier fraction (gated on the KL measured here), the
   streamed-vs-in-place bit-identity test extended to a mixed model, a golden-logits model with a mixed expert
   layout, then decode speed on the phone in alternating A/B runs at budgets 5500 and 7000, and the phone
   regression run.

## Reproducing

Tools: `build/liyab-experts` (C++, built with the tests) and `tools/expert_study.py` (needs `numpy`, `torch`,
`gguf`). Paths below are placeholders.

```bash
# routing, input moments and sample inputs (calibration text and evaluation text)
build/liyab-experts capture -m qwen36.gguf -f calib.txt -o cap_cal --max-tokens 8192
build/liyab-experts capture -m qwen36.gguf -f tests/data/conform_text.txt -o cap_eval --max-tokens 512
# decoders and encoders checked against the engine
tools/expert_study.py check-dequant --model qwen36.gguf --experts-tool build/liyab-experts --work /tmp
tools/expert_study.py check-encoders --model qwen36.gguf --experts-tool build/liyab-experts --work /tmp
# spectra, per-matrix errors, routing shares, LFU replay
tools/expert_study.py spectra --model qwen36.gguf --capture cap_cal --out spectra.json
tools/expert_study.py methods --model qwen36.gguf --capture cap_cal --heldout cap_eval --methods q3k q3kw iq3xxsw nu3s16w lb1 --out methods.json
tools/expert_study.py overlap --capture cap_cal --heldout cap_eval
tools/expert_study.py simulate --model qwen36.gguf --capture cap_cal --cache-shares 0.1225 0.203 --fractions 0.25 0.5 0.75 1.0 --tier-bpw 3.0625 3.4375 1.0 --out sim.json
# end to end: Q8_0-expert copy, references, one variant
build/liyab-experts to-q8 -m qwen36.gguf -o base.gguf
build/liyab-kl -m qwen36.gguf -f tests/data/conform_text.txt --save ref_orig.bin --expert-cache 0
build/liyab-kl -m base.gguf -f tests/data/conform_text.txt --save ref_base.bin --expert-cache 0
tools/expert_study.py patch --model qwen36.gguf --base base.gguf --capture cap_cal --fraction 0.5 --method q3kw --out v.gguf
build/liyab-kl -m v.gguf -f tests/data/conform_text.txt --compare ref_orig.bin --expert-cache 0
```

The calibration text is generated, not committed: `liyab-cli -m qwen36.gguf -p PROMPT -n 700 --temp 0` for each of
the 10 prompts listed in [Setup](#setup), the replies concatenated, then `README.md` appended (the capture keeps the
first 8,192 tokens).
