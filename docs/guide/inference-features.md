# Power, speculation, pooling and adapters

[← README](../../README.md) · [Documentation index](README.md)

Commands and source paths in examples are relative to the repository root.

## On this page

- [Heat and power](#heat-and-power)
- [Speculative decoding](#speculative-decoding)
- [Host measurements: suffix drafts and context lookup](#host-measurements-suffix-drafts-and-context-lookup)
- [Host measurements: tree verification against the best branch](#host-measurements-tree-verification-against-the-best-branch)
- [Embeddings and reranking](#embeddings-and-reranking)
- [LoRA adapters](#lora-adapters)

---

## Heat and power

A phone that gets hot slows down by itself, and gets slower still once the OS steps in. Liyab steers before that
happens.

* **Forecast, not reaction.** While the engine processes a prompt or generates, a polling thread reads the
  following once a second:
  * the OS thermal status (Android `AThermal_*`, Apple `NSProcessInfo.thermalState`);
  * Android's thermal-headroom forecast 10 s ahead (Android 12+, smoothed);
  * the sysfs skin and SoC zones.

  Idle, it reads nothing. One reading costs ~10 ms of CPU and ~25 wakeups on a Snapdragon 8 Elite, which a loaded
  but unused model would otherwise pay every second (about 1% of a core, the screen off included).

  The forecast becomes a pressure from 0 to 1 over a band set by the power profile: Performance 0.75–0.95, Balanced
  0.65–0.90, LowPower 0.35–0.60. A cooler profile reacts earlier.
* **Same tokens, fewer watts.** From a pressure of 0.1, the engine sheds cores (down to half). It also paces tokens
  at up to twice their full-speed work time, learned while running unslowed. On Android 13+ the decode threads report
  each token's work time to a performance-hint session (ADPF) with the token period as target, so the governor runs
  just fast enough instead of sprinting and sleeping. Unpaced decoding opens no session. Below 0.1 nothing is paced:
  a hint asking for the current pace had been holding the clocks there.
* **Guards for emergencies only.** These trigger a throttle (GPU work rerouted, half the threads, half the rate):
  * pressure 1;
  * the profile's OS-status limit;
  * the skin threshold. With a forecast this applies only 8 °C above the threshold, because on many phones the
    "skin" sensor is a board thermistor that reads 45–50 °C under load;
  * a smoothed SoC reading past 95 °C. Single samples spike by 25–30 °C, and 85–90 °C is normal under load.
* **Duty cycling.** Profiles can cap output at a reading-speed rate (Balanced 12 tok/s, LowPower 6 tok/s; `--tps` to
  override), so the SoC idles between tokens.
* **Live counters.** `Engine::counters()` reports GPU busy time, bytes streamed and tokens generated, and is safe to
  call while a generation runs. The app's Activity page adds battery watts and J/token on battery.

---

## Speculative decoding

A small draft model proposes *k* tokens, or, with `lookup_drafts`, the text itself does, with no extra weights and
for any model. `lookup_drafts` is **automatic** by default: the engine decides for each model and device whether it
pays.

* **How a model is recognized.** At load the engine times one plain decoding step and one verification pass of
  *k* + 1 tokens (with the recurrent checkpoints a rollback needs), best of three after a warm-up, and keeps their
  ratio. Drafting is on where a pass costs at most 2.5 steps. The ratio is the price of a wrong guess, so it
  separates models without a list of names: a dense 14B in RAM measured 1.84 steps in 0.54 s, and a MoE whose
  experts stream from storage is switched off without a timing, since a pass reads the union of its tokens' experts.
  The measurement is kept in `cache_dir` as `verify-<id>.bin` (16 files at most, least recently used out first), keyed
  by the model file's identity, the KV format, the kernel set, the route decoding takes, the threads and *k*; without
  a `cache_dir` it is kept for the process. `Engine::lookup_drafts()` reports the decision and the ratio.
  It also stays off while an experimental decoding feature (early exit, EGLS, head pruning, TDSS) is on, because a
  speculative step runs without those hooks. `EngineConfig::LookupDrafts::On` / `Off` (`--lookup`, `--no-lookup`)
  skip the measurement. Where it is on, the
  learned draft count (below) still backs off to plain steps on text that repeats nothing.

Two drafters:

* **Suffix drafts** (the default, `suffix_drafts`): the longest suffix of the conversation (2…32 tokens) that
  occurred before, in this conversation or in earlier ones of the same engine, and how its occurrences continue:
  their continuation tree, which branches where the occurrences part.
  Earlier conversations are kept with a suffix array each, so finding a match is a binary search. They are set
  aside when a prompt stops continuing them (a new conversation, a regenerated reply) and are bounded: 262,144
  tokens (~2 MiB) in at most 64 of them, least recently used out first. A conversation taken up again replaces its
  stored copy, so no text is counted twice. Each drafted token carries an acceptance probability, calibrated on
  the device from what verification accepted, by match length, by how many occurrences agree and by whether the
  conversation itself supports it; a token's probability is the product along its branch. A step verifies the
  tree's likeliest nodes, as many as give the most expected tokens per millisecond at the measured cost of a step
  per draft count. A confident draft is verified at full length (`draft_tokens` at most) and a doubtful one not at
  all.
* **Context lookup** (`suffix_drafts = false`): the tokens that followed the most recent earlier occurrence of the
  last 4…2 tokens of the conversation.

The target verifies the drafts in one batched pass, reading its weights once for *k* + 1 tokens.

**Tree verification** (suffix drafts, `draft_tree`, on by default). Where earlier occurrences of the matched text
continue differently (a list whose items differ, a JSON field whose values do), the drafter offers both
continuations, and one pass verifies them together:

* **Which drafts.** The *k* likeliest nodes of the continuation tree, taken best first: a node's children become
  candidates once it is taken, so any first *j* of them form a tree. Siblings are exclusive (the target picks one
  token), so their probabilities are scaled to sum to at most 1. The expected tokens of a step are 1 + the sum of
  the nodes' probabilities; the draft budget weighs them against the measured cost of a pass of *k* + 1 tokens,
  which is the same whether they form a branch or a tree (on a MoE that streams its experts, the union of the
  tokens' experts either way).
* **One pass.** The batch is the last token, then the drafts depth first with the likeliest child first, so the
  likeliest branch is the batch's first tokens. Every token is computed at its branch's position (the context's
  length plus its depth), attending to the context and its own ancestors only, through the same visible positions
  and spans as a linear pass over its branch. So its row of logits has that pass's bits. Each token's keys and
  values go to a slot of their own. Windowed blocks see their window over the branch; latent attention runs in the
  absorbed form, as decoding does; an active LoRA adapter applies as in any pass.
* **Hybrid models branch their recurrent state.** A token whose parent is not the token before it continues from
  the state after its parent, which the pass keeps for every token (in the rollback checkpoints, *k* + 1 of them
  as before). Every token's state is copied once, as a linear pass of *k* + 1 tokens copies it.
* **Kept branch.** From the last token down, each row is sampled once, as plain decoding samples it, and the child
  holding the sampled token is kept. The kept branch's rows move to their positions (none when it is the likeliest
  branch), its recurrent states become the rollback checkpoints, and the rest of the tree is dropped: the context
  is exactly the one a linear pass over that branch leaves. A tree that is a single branch runs as a linear pass.
  `draft_tree = false` verifies only the best branch.

* **Exact.** Acceptance follows Leviathan et al. (2023), so the output keeps the target's distribution. With greedy
  sampling it is token-for-token identical to plain decoding (tested). Suffix drafts are kept while they equal the
  token plain decoding samples from the same row, on a branch or in a tree. With temperature, their output is
  therefore plain decoding's own tokens for a seed, whatever was drafted (tested on the CPU backend). The one
  exception is the tokens after a forced continuation cut a step short.
* **Hybrid models included.** Both models keep copies of their recurrent states for the last *k* + 1 tokens, so
  rejected tokens roll back exactly.
* **Self-tuning.** How many drafts to verify is learned on the device. With a draft model or context lookup, it is
  the draft count with the most tokens per millisecond, and losing counts are retried after a doubling wait. With
  suffix drafts, it is chosen for each draft from its acceptance probabilities and each count's step time (the
  median of its last 8 steps). Counts are tried upwards, one more draft at a time.
* **When it pays.** It pays on dense models with i8mm tiles, where 4 rows cost ~1.7× one row. Qwen3-4B on the
  phone went from 11.2 to 12.6 tok/s on a code edit, with 65% of drafts accepted. It does not pay on a MoE whose
  experts stream from flash, because a pass reads the union of its tokens' experts. That is what the automatic
  mode measures, and why it leaves such models off.

### A 14B on the M4 Pro, with weights lowered and a draft model

Qwen2.5-Coder-14B Q5_K_M, CPU, 10 threads, Q8_0 KV, greedy, replies of 160–192 tokens, every reply identical to plain
decoding's. `liyab-spec-bench` for the drafts without a model (context lookup, 3 drafts), `--requant 4` for the
lowered weights ([the measurement](benchmarks.md#fewer-bytes-per-token-requant-4-on-a-dense-model)):

| | Code edit | Summary | Free chat | Tool chat again |
|---|---|---|---|---|
| plain decoding | 19.1 tok/s | 18.7 | 19.3 | 18.3 |
| context lookup | 35.5 (1.86×, 98% accepted) | 18.3 (0.98×) | 19.7 (1.02×) | 18.3 (1.00×) |
| suffix drafts | 33.7 (1.77×) | 18.9 (1.01×) | 20.4 (1.06×) | 22.1 (1.21×) |
| plain, weights lowered (`--requant 4`) | 21.8 | | 21.5 | |
| lookup, weights lowered | 42.9 (2.2× the plain file's rate) | | 23.2 | |

A draft *model* (Qwen2.5-Coder 0.5B Q8_0, 3 drafts, lowered 14B, `liyab-cli`) on free chat went from 24.1 to
28.4 tok/s (62% of drafts accepted), and from 18.4 to 18.2 on a summary (37% accepted): the draft's own cost cancels
the gain wherever acceptance is under about half. Text that repeats earlier text is what speculation speeds up; free
text gains at most a fifth.

The draft model's token table (`tokenizer.ggml.tokens`) must agree with the target's over the ids both have; the
output width may differ (the 0.5B pads to 151,936 ids, the 14B to 152,064). The draft then drafts among the ids it has,
and the target verifies every draft.

### Host measurements: suffix drafts and context lookup

Measured with `liyab-spec-bench` on an Apple M4 Pro with the CPU backend, 6 threads, greedy decoding and replies of up to
256 tokens. The default of 3 drafts per step was used, and the draft count was learned. Each cell gives the decode
speed-up over plain decoding (median of 2 runs) and, in parentheses, the tokens produced per verification pass. The
tasks are a code edit (a 50-line module rewritten with one rename), a summary of a pasted text, two turns that each
carry the same tool output (the second asks for it in full), free chat, and that tool conversation again as a new
conversation. Every reply was identical to plain decoding's.

| Model | Drafter | Code edit | Summary | Tool output | Free chat | Tool chat again |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 | lookup | 2.60× (3.44) | 1.42× (1.51) | 1.71× (1.92) | 1.06× (1.09) | 1.69× (1.81) |
| | suffix | 2.96× (3.61) | 1.40× (1.56) | 1.63× (1.94) | 1.15× (1.05) | 2.59× (3.83) |
| Qwen3-0.6B BF16 | lookup | 1.25× (2.69) | 0.98× (1.04) | 0.98× (1.06) | 1.04× (1.01) | 0.98× (1.06) |
| | suffix | 1.28× (3.26) | 0.99× (1.16) | 1.00× (1.24) | 1.01× (1.00) | 1.33× (3.62) |
| Qwen3.5-0.8B BF16 (hybrid) | lookup | 1.18× (2.28) | 0.93× (1.02) | 0.99× (1.02) | 0.96× (1.00) | 0.95× (1.01) |
| | suffix | 1.18× (2.19) | 0.97× (1.13) | 0.98× (1.18) | 0.97× (1.01) | 1.30× (3.91) |
| Llama-3.2-1B BF16 (best run) | lookup | 0.94× (2.33) | 0.89× (1.03) | 1.10× (0.99) | 0.96× (1.00) | 1.13× (1.02) |
| | suffix | 0.98× (2.60) | 0.93× (1.04) | 0.93× (1.00) | 1.00× (1.01) | 1.36× (3.01) |

* **Acceptance.** Of the drafts verified, suffix drafts had 92–98% accepted on the code edit, 42–62% on the summary,
  26–77% on the tool output, 0–35% in free chat and 100% on the conversation seen before. Lookup had 86–97%,
  24–55%, 0–64%, 0–29% and 29–60%.
* **The same drafts where the text occurs once.** With every draft verified (`suffix:K:fixed`), both drafters
  give nearly the same tokens per pass on the code edit and the summary (3.71 for 3 drafts, 7.11 for 8 on Qwen2.5).
  On the repeated conversation, suffix drafts give 3.90–3.96 tokens per pass for 3 drafts and 8.53–8.67 for 8.
  Lookup gives 1.15–2.01 and 1.16–2.42, because it sees only the conversation in progress.
* **Doubtful drafts are not verified.** Suffix drafts size each draft from its acceptance probabilities, so they
  stay within 7% of plain decoding where drafting cannot pay. With 8 drafts per step, lookup lost up to 20% on
  the same tasks (Llama summary 0.80×, Qwen3 summary 0.84×), and suffix drafts lost at most 5%.
  Suffix drafts reached 1.24–2.85× on the repeated conversation, with 6.1–7.8 tokens per pass.
* **Why the BF16 models gain less.** On this CPU a pass over 4 tokens costs about 3× one token on the BF16
  models, against ~1.5× on Q8_0 (from the tokens per pass and speed-ups above). Only near-certain drafts pay there.
* The Llama runs shared the host with other heavy work. Their speed-ups are the least disturbed run's; tokens per
  pass and acceptance do not depend on timing.

### Host measurements: tree verification against the best branch

The same tasks and host (CPU backend, 6 threads, greedy, replies of up to 256 tokens), each suffix-draft mode with
its tree (`suffix`, the default) and with its best branch only (`suffix:linear`). Every reply of every mode was
identical to plain decoding's, the hybrid model's included. Tokens per verification pass with every draft verified
(`suffix:8:fixed` against `suffix:8:fixed:linear`; these do not depend on timing):

| Model | Code edit | Summary | Tool output | Free chat | Tool chat again |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 | 7.11 / 7.11 | 1.82 / 1.82 | 2.53 / 2.50 | 1.12 / 1.11 | 8.53 / 8.53 |
| Qwen3-0.6B BF16 | 6.74 / 6.74 | 1.36 / 1.37 | 1.77 / 1.77 | 1.02 / 1.02 | 8.67 / 8.67 |
| Qwen3.5-0.8B BF16 (hybrid) | 4.20 / 4.20 | 1.34 / 1.34 | 1.46 / 1.45 | 1.02 / 1.02 | 8.56 / 8.56 |
| Llama-3.2-1B BF16 | 7.11 / 7.11 | 1.19 / 1.18 | 1.21 / 1.20 | 1.03 / 1.03 | 8.42 / 8.42 |

(tree / best branch.) With the learned draft count on Qwen2.5-0.5B Q8_0 (median of 3 runs), the tree and the best
branch gave 3.52 / 3.57, 1.55 / 1.53, 1.64 / 1.75, 1.03 / 1.03 and 3.89 / 3.85 tokens per pass for 3 drafts per
step, and speed-ups over plain decoding of 2.02× / 2.10×, 1.17× / 1.11×, 1.49× / 1.37×, 1.00× / 1.02× and
2.28× / 2.06×: equal within the run-to-run spread of this shared host.

* **On these tasks a tree verifies what the best branch does.** Where a draft is rejected, the target almost always
  writes a token that no earlier occurrence of the matched text continued with, so no sibling could have been
  right; where the occurrences part, the likeliest one is mostly the right one. A tree gains where earlier text
  continues the same words several ways and the target takes a less frequent one: the tests build such text and
  check that the sibling is kept.
* **It costs what the branch costs.** A tree pass of *k* + 1 tokens runs the same matmuls as a linear one, its
  tokens attend to fewer positions (their ancestors only), and building the tree takes microseconds (7 µs for 8
  drafts on a 6000-token text against 3 µs for the branch). Plain decoding is unchanged (the decode step at a
  4000-token context timed alternately against the previous build: equal within the host's noise).

---

## Embeddings and reranking

EmbeddingGemma (`gemma-embedding`, 300M) adds bidirectional attention and mean pooling with the model's two
bias-free dense projections (768 → 3072 → 768). On CPU and Metal routes, a BF16 conversion of `unsloth/embeddinggemma-300m`
matched all tokenizer IDs in the 39 pooling cases and reached cosine ≥ 0.99992651 against the float32 reference;
all 39 vectors were bit-identical alone and in the batch. The independent reference agreed with
sentence-transformers at cosine ≥ 0.9999998. Include the model card's task prefix in the input text.
`tools/pooling_reference.py` reads mean-pooling and identity Dense modules from `modules.json`;
`tools/tiny_checkpoint.py --base-model` saves embedding transformers without adding an LM head.

Local memory and retrieval need two more kinds of model next to the generative one: an embedding model, which turns
a text into a vector so similar texts lie close together, and a reranker, which scores how well a document answers a
query. Decoder models fine-tuned for these jobs (Qwen3-Embedding, Qwen3-Reranker) run on the architectures above with
the same forward pass; `Engine::embed()` and `Engine::rerank()` (C: `liyab_engine_embed` / `_rerank`) use them, and
so does the app's [local API](app.md#local-api-one-model-for-every-app-on-the-device) (`/v1/embeddings`, `/v1/rerank`).

* **What a model does is data from its GGUF**, never a per-model branch:
  * `<arch>.pooling_type` says how the decoder's final hidden states become one vector: the last token's (3,
    Qwen3-Embedding), the mean of all tokens' (1), or a ranking head over the last token's (4, a reranker).
    `Engine::pooling()` reports it.
  * `tokenizer.ggml.add_eos_token` ends every input with the EOS token, as the model's Hugging Face tokenizer
    appends it (Qwen3-Embedding's `<|endoftext|>`, whose state is the one pooled). A file converted without the flag
    needs the token's text at the end of each input.
  * A reranker carries its ranking head (`cls.output.weight`, one row per label of
    `<arch>.classifier.output_labels`: Qwen3-Reranker's are the rows of "yes" and "no" in its LM head, with an
    optional `cls.output.bias`) and the prompt of one pair (`tokenizer.chat_template.rerank`, with `{query}` and
    `{document}`). Some published Qwen3-Reranker GGUFs are plain causal models without either: `RerankOptions`
    gives the prompt and `labels` (`{"yes", "no"}`), and the pair is scored with the LM head's logits of those
    tokens, the way the model card does.
* **How.** `embed()` runs each text through the transformer without the LM head and takes the final hidden state
  after the output norm, the state the LM head would read: the last token's or the mean of every token's. Vectors are
  L2-normalized; `dimensions` keeps the first values for models trained for it (Qwen3-Embedding: 32 to 1,024).
  `rerank()` writes each pair into the prompt and scores the last token's state with the head: the softmax
  probability of the first label, which for Qwen3-Reranker is P(yes) over {yes, no}, its model card's
  normalization (a head with one output: its sigmoid). An instruction is part of the text: Qwen3-Embedding queries
  are `Instruct: {task}\nQuery:{query}`, its documents plain text; Qwen3-Reranker's prompt carries its instruction.
* **Shared prefixes are computed once.** The inputs of a call run one after another in the context, sorted by
  their tokens, and each keeps the KV of the prefix it shares with the one before: the queries' instruction, the
  reranker's system prompt and query. That is 17% of the tokens of the embedding check set below and 47% of the
  reranking one. The last token always runs. On the CPU backend a row's result does not depend on the pass it is
  computed in, so a vector or a score is the same bits alone, in any batch and in any order (tested; on Metal, each
  input of the check sets below gave the same bits alone as in its batch too). Hybrid models start every input over
  (their recurrent states cannot rewind to any position), and a mean needs every row, so mean pooling shares
  nothing.
* **The KV cache of a pooling model is F16** (`EngineConfig::pooling_kv_cache_type`): it holds one input at a time
  and no conversation, so a quantized cache saves little memory, while its rounding moves every result. With the
  Q8_0 default of generative models, Qwen3-Embedding-0.6B's vectors kept a cosine of 0.99949 with transformers' and
  Qwen3-Reranker-0.6B's scores moved by up to 0.0038; with F16, 0.9999997 and 0.00006.
* **Contract.** An input holds at most `context_length()` tokens, its special tokens included; a longer one is an
  error naming it, or is cut (`EmbedOptions::truncate`, keeping BOS and EOS) for embeddings. The calls use the
  engine's context and leave it empty: a conversation in progress is set aside in the prefix cache when one is
  configured, and restored when its next prompt comes, as when another conversation's prompt replaces it. An
  engine serves one call at a time (`Busy` while a generation, a prefill or another call runs), and `cancel()`
  stops a call within a block of the model. The public header has the whole contract.

**Against Hugging Face transformers.** `tools/pooling_reference.py` computes the reference the way the model cards
do, in float32 (for Qwen3-Embedding it reproduces its card's example scores to 1e-6; sentence-transformers 6.1,
run in float32, agrees with it to a cosine of 0.9999998 and, through the sigmoid of its CrossEncoder's score, to
2e-7 on every reranker score), over
`tests/data/pooling_cases.json`: 39 texts in 18 languages, code and emoji (Instruct-formatted queries, documents,
an empty text, a chat-formatted text, a 232-token paragraph that changes language four times) and 5 queries with 19
documents. `liyab-embed` checks that every input encodes to the reference's token ids, then compares. Every
input gave the reference's ids, and every vector and score was the same bits alone as in its batch:

| GGUF | Backend | Against the reference |
| :--- | :--- | :--- |
| Qwen3-Embedding-0.6B F16 (Qwen's), last token | CPU / Metal | cosine ≥ 0.99999965 / 0.99999967 per vector |
| the same, mean pooling | CPU | cosine ≥ 0.99999964 |
| the same, not normalized | CPU | relative L2 error ≤ 0.00084 |
| Qwen3-Reranker-0.6B BF16 (ranking head) | CPU / Metal | scores within 0.000063 |
| the same, scored with the LM head's "yes" and "no" (`labels`) | CPU | scores within 0.000063 |
| Qwen3-Embedding-0.6B Q8_0 (Qwen's) | CPU / Metal | cosine mean 0.9972 / 0.9983; ≥ 0.9983 / 0.9994 except the empty text (its EOS alone): 0.92 / 0.95 |
| Qwen3-Reranker-0.6B Q8_0 (ggml-org) | CPU / Metal | scores within 0.011 / 0.012 |

The 8-bit files lose what their weights lose (and, on the CPU, what 8-bit activations lose), most where a single
token's state is pooled: use F16 or BF16 files of these small models when the vectors must match the reference.

**Throughput** on an Apple M4 Pro (CPU backend with 10 threads, or Metal), over the same sets: 39 texts of 37
tokens on average in one `embed()` call, and the 5 queries' 19 pairs of 110 tokens on average, one `rerank()` call
per query. Medians of 7 runs (`liyab-embed --bench 7`), on a Mac other jobs shared (load average 10–14):

| GGUF | CPU | Metal |
| :--- | ---: | ---: |
| Qwen3-Embedding-0.6B Q8_0 | 30.2 texts/s (1,128 tokens/s) | 9.4 texts/s |
| Qwen3-Embedding-0.6B F16 | 6.6 texts/s | 8.6 texts/s |
| Qwen3-Reranker-0.6B Q8_0 | 18.4 pairs/s | 6.7 pairs/s |
| Qwen3-Reranker-0.6B BF16 | 3.5 pairs/s | 3.6 pairs/s |

The CPU's 8-bit kernels lead: Metal submits one command buffer per matrix product, which these small passes
cannot hide, and F16 / BF16 weights multiply row by row on the CPU (see [Known limitations](development.md#known-limitations)).
Sharing prefixes skipped 17% of the embedding tokens and 47% of the reranking ones. A single 525-token text ran at
~380 tokens/s on the Q8_0 file: per token, short inputs are cheaper, since attention grows with the input.

---

## LoRA adapters

A LoRA adapter specializes a model (a task, a style, a tool format) with a low-rank update of some of its matrices:
W' = W + s·B·A, where A has `rank` rows and B `rank` columns, rank 8 to 64 against matrices thousands wide. Liyab
never merges it into the weights. Each adapted matmul adds the update to the base product:

```
y = W·x + s·B·(A·x)
```

so several adapters stay loaded next to one model, the one in use is chosen per generation, and a generation
without one runs exactly as before.

* **The file** is the GGUF LoRA format the common converters write from a PEFT adapter: `general.type` `adapter`,
  `adapter.type` `lora`, `adapter.lora.alpha`, and for each adapted matrix `<name>.lora_a` (ne {cols, rank}) and
  `<name>.lora_b` (ne {rank, rows}), a mixture of experts' expert stacks with a third dimension (one A and B per
  expert). A is read in place from the mapping when F32, F16 or BF16 (other types are converted to F32 once at
  load); B is kept transposed (see below), a copy in its own type: an adapter costs about its file's size.
* **Scale**: s = scale · alpha / rank, with each matrix's own rank (alpha 0 or absent: s = scale); `scale` is chosen
  when the adapter is selected, 1 meaning as trained.
* **Where it applies**: every matrix the forward pass multiplies whole: attention, DeltaNet and FFN projections
  (separate or fused), a mixture of experts' router, shared expert and routed experts (each expert its own slice, for
  the tokens routed to it), latent attention's whole projections, compression/indexer and residual-stream
  projections, and the LM head. Refused when loaded, naming the
  tensor: adapters of the token embeddings (they change a lookup, not a matmul), of latent attention's per-head
  `attn_k_b` / `attn_v_b` or combined `attn_kv_b`, shared-KV's grouped `attn_output_a`, activated LoRA
  (`adapter.alora.*`), and any matrix whose name or shape the model does not
  have.
* **How it is computed**: t = A·x is a matmul of `rank` rows, run by the CPU backend's kernels in the same parallel
  pass as the base matmuls sharing its input (Q/K/V, gate/up, one wave of experts), so it costs no fork/join of its
  own. The update y += s·B·t is a kernel of its own over Bᵀ, transposed at load: `rank` scaled row additions per
  token, contiguous and vectorized, where B·t as a matmul would be one dot product of only `rank` values per output
  (mostly call overhead). Output columns go to threads in fixed chunks of 256 (small
  updates stay on the calling thread); each value sums its `rank` terms in order on one thread, so the result is the
  same bits for any thread count. With an accelerator, the base product comes from the GPU and A·x and the update
  from the CPU. Without an adapter selected, a matmul pays one branch.
* **Several at once**: `Engine::load_adapter(name, path)` maps an adapter, checked against the model (architecture,
  every matrix's name, shape and expert count), within `EngineConfig::adapter_memory_mb` (256 MiB by default, 0 turns
  adapters off). Past the cap the least recently selected adapters are released and mapped again from their files
  when selected. `Engine::set_adapter(name, scale)` selects the one later calls apply (generate, prefill, embed,
  rerank); `set_adapter("")` goes back to the base model; `Engine::adapters()` lists them.
* **Contexts never cross adapters.** The live context is continued only while the same adapter and scale stay
  selected. Selecting another sets it aside in the prefix cache under the adapter it was computed with, and the next
  prompt starts from what the cache keeps for the new one. The prefix cache's key seed covers the adapter (a hash of
  its file) and the scale, so its chunks, in RAM and on storage across runs, are only restored under the adapter that
  computed them. `save_state()` records the adapter; `load_state()` refuses a context saved under another one.
* **Speculative decoding** verifies every draft with the adapter applied, so the output is the adapter's. A draft
  model drafts without it, which may lower acceptance, not change the output.
* **Segment reuse** is off while an adapter is selected (its share was calibrated on the base model): marked prompts
  are processed as usual, exactly.

**Conformance.** Random PEFT adapters (rank 16, alpha 32, every attention, DeltaNet and FFN projection), converted
to GGUF with the common converter, against transformers + PEFT in float32 over `tests/data/conform_text.txt`
(`liyab-kl --lora`, `--kv f16`, CPU). Without the adapter, Liyab's distributions are 0.29–0.40 nats of KL away from
the adapted reference; with it, as close as the base model is to its own reference:

| Base GGUF | Adapter | Mean KL | Top-1 | Perplexity (reference → Liyab) |
| :--- | :--- | ---: | ---: | :--- |
| Qwen2.5-0.5B-Instruct F16 | none, base reference | 0.00001 | 99.80% | 25.615 → 25.615 |
| Qwen2.5-0.5B-Instruct F16 | none, against the adapted reference | 0.289 | 67.32% | 31.590 → 25.615 |
| Qwen2.5-0.5B-Instruct F16 | F16, 168 matrices | 0.00001 | 99.61% | 31.590 → 31.592 |
| Qwen2.5-0.5B-Instruct F16 | F32 | 0.00001 | 99.61% | 31.590 → 31.602 |
| Qwen2.5-0.5B-Instruct BF16 | F32 | 0.00001 | 99.41% | 31.590 → 31.602 |
| Qwen3.5-0.8B BF16 (hybrid DeltaNet) | none, against the adapted reference | 0.403 | 67.98% | 31.483 → 21.081 |
| Qwen3.5-0.8B BF16 (hybrid DeltaNet) | F16, 186 matrices | < 0.00001 | 100.00% | 31.483 → 31.489 |

Per-expert adapters of a mixture of experts, the router, the shared expert, fused projections and latent attention
are checked in `test_lora` against the same model with the adapter merged into its weights.

**Cost per token** on the Mac (Apple M4 Pro, CPU backend, greedy, 128 tokens after a 100-token prompt; medians of 7
runs, the configurations alternating round by round on a Mac other jobs shared, load average 12–20). Random F16
adapters on every attention and FFN projection of every block:

| Base GGUF | Decode, no adapter | rank 8 | rank 16 | rank 64 | Prefill, rank 8 / 16 / 64 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Qwen2.5-0.5B-Instruct F16 (8.4 / 16.8 / 67 MiB adapters) | 161 tok/s | +0.42 ms (+6.6%) | +0.39 ms (+6.2%) | +1.27 ms (+20%) | −4% / −3% / −11% |
| Llama-3.2-1B-Instruct BF16 (11 / 21 / 86 MiB adapters) | 63 tok/s | +0.68 ms (+4.3%) | +0.75 ms (+4.8%) | +1.81 ms (+11.5%) | 0% / 0% / −5% |

Without an adapter selected the engine runs as before: against the build before adapters, 161 vs 159 tok/s on the
0.5B and 63.0 vs 63.4 on the 1B (within the runs' spread). The update's cost is mostly fixed per matmul at low
rank (on these small models a decoding step has ~100 adapted matmuls, each paying its A·x rows, its update and a
fork/join for the larger ones); at rank 64 it grows with the adapter's bytes, which every token reads.

---
