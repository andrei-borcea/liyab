# Merged stage-1 conformance evidence

`conformance.json` records all 66 successful merged-tree recipes, their effective configurations,
source revisions when known, measured result lines, commands and generated-file identities.
It contains 65 generative tiny cases and one embedding case with 39 inputs. These are random-weight
checks, not 66 trained checkpoints. Earlier failed numerical controls remain documented in
[the stage-1 report](../../model-coverage-stage1.md) and [model descriptions](../../model-descriptions.md).

Commands retain explicit path aliases rather than the owner's machine paths:

- `$PY`: the reference Python environment (torch 2.14.1, transformers 5.19.0, gguf 0.19.0).
- `$LLAMA_CPP`: the b11819 Python converter checkout; no C/C++ inference code is used here.
- `$INTEGRATION`: the tested repository/build tree.
- `$SECTION_WORKTREES`: section-local source snapshots retained during the interrupted wave.
- `$EVIDENCE`: compatibility helpers and source/config snapshots used by the recorded command.

Reconstruct source snapshots from the recorded upstream revision, fetching only configuration,
tokenizer and official Python source files. Apply the relevant compatibility patches and use the
recorded complete configuration with `tools/tiny_checkpoint.py --config`. A null revision means the
original source revision was not recorded; do not replace it with an invented pin. Section-worktree
paths document the executed source, not a promise that those local paths exist on another machine.

Copy b11819 `convert_hf_to_gguf.py`, `conversion/` and `gguf-py/` before applying a `converter-*.patch`
with `patch -p1`. The shared converter was never edited. The Kimi converter helper memoizes a fixed
byte mapping; its checkpoint helper restores names without changing tensor values. The embedding
helper materializes deterministic dense modules from the checkpoint's declared module configuration;
its source/target path aliases must be mapped to reconstructed local directories before running.

The reference compatibility runners restore removed Transformers imports/bookkeeping and the
original constructors' nonpersistent RoPE buffers; they do not patch the reference forward math.
PLaMo3 additionally used the existing numba 0.68.0 / llvmlite 0.50.0 installation; legacy SentencePiece
references used sentencepiece 0.2.1. Do not infer tokenizer equivalence from class names: the Ministral
three-text check uses the serialized tokenizer JSON's explicit control-token policy, while its
MistralCommon class defaults to treating literal control spellings as ordinary text.

Metadata repairs are part of the recipe, not silent engine exceptions. OLMo uses `llama-bpe-merges`,
EXAONE 3.5 `gpt2-nfkc-digits`, InternLM 3 `spm-greedy` without a dummy space, and the source-verified
StarCoder digit pipeline `starcoder`. Preserve the original exported file/hash when correcting a copy.

Generated GGUF/checkpoint weights were deleted immediately after successful checks. This archive
contains no weights, credentials, full transcripts or generated reference-logit files.
