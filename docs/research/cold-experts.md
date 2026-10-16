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

SUMMARY_PLACEHOLDER

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

E2E_TABLE_PLACEHOLDER

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
−38% bytes where the phone measured −27%. The speed figures below are therefore upper estimates. The model is
also linear in bytes and ignores any change in compute (a 3-bit expert multiplies at a different cost than a
Q4_K one).

| Cold tier | Cold share | Simulated MiB/token (5500 cache) | Projected MiB/token | Projected tok/s at 5500 (8.29 measured) | at 7000 (9.50 measured) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| none | 0 | 200 | 284 | 8.29 | 9.50 |
| Q3_K | 25% | 190 | 269 | 8.50 (+3%) | 9.78 (+3%) |
| Q3_K | 50% | 170 | 241 | 8.94 (+8%) | 10.34 (+9%) |
| Q3_K | 75% | 140 | 199 | 9.67 (+17%) | 11.20 (+18%) |
| Q3_K | 100% | 105 | 148 | 10.73 (+29%) | 12.21 (+29%) |
| IQ3_XXS | 25% | 187 | 266 | 8.56 (+3%) | 9.85 (+4%) |
| IQ3_XXS | 50% | 161 | 229 | 9.13 (+10%) | 10.58 (+11%) |
| IQ3_XXS | 75% | 125 | 177 | 10.10 (+22%) | 11.71 (+23%) |
| IQ3_XXS | 100% | 82 | 117 | 11.52 (+39%) | 12.97 (+37%) |
| 1 bit (LittleBit) | 25% | 172 | 244 | 8.88 (+7%) | 10.29 (+8%) |
| 1 bit (LittleBit) | 50% | 118 | 168 | 10.29 (+24%) | 12.07 (+27%) |

The cold quarter of the experts carries only ~5% of the routings, so shrinking it barely changes the bytes read
(−5 to −7%): what is rarely used is also rarely read. The bytes are in the middle of the usage ranking.

## Conclusion

CONCLUSION_PLACEHOLDER

## Implementation plan for a runtime tier

PLAN_PLACEHOLDER

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
