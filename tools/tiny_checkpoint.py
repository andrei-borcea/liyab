#!/usr/bin/env python3
"""Writes a small checkpoint of a model with random weights, for conformance checks of its architecture.

An architecture can be checked against its Hugging Face implementation without the model's weights: this tool
builds a copy of the model from its configuration with few blocks (every kind of block the configuration lists, as
tools/reference_trace.py keeps them), few attention heads, narrow FFNs and few experts, fills it with random weights
that keep activations of order 1 (reference_trace's), and saves it as a Hugging Face checkpoint next to the model's
own tokenizer files. The checkpoint converts to GGUF like the real model and goes through the same conformance check:

    tools/tiny_checkpoint.py --model Qwen/Qwen3-0.6B --out tiny-qwen3
    python3 convert_hf_to_gguf.py tiny-qwen3 --outtype f32 --outfile tiny-qwen3.gguf
    tools/conform_reference.py --model tiny-qwen3 --text tests/data/conform_text.txt --out tiny-qwen3.ref
    build/liyab-kl -m tiny-qwen3.gguf -f tests/data/conform_text.txt --compare tiny-qwen3.ref --ref-tokens \\
        --kv f16 --max-kl 0.0001 --min-top1 99

What is kept and why:

* The vocabulary and the tokenizer: the reference and Liyab encode the text with the model's own tokenizer, so the
  check covers the tokenizer and the chat-format tokens too. The embedding and output matrices are therefore the
  largest tensors (vocabulary x width).
* The size of a key/query head: RoPE tables, frequency factors (LongRoPE, Llama 3) and the default attention
  scale depend on it. The width of the residual stream becomes heads x head size (unless --width sets it).
* Every scalar of the configuration (bases, epsilons, multipliers, windows, soft caps, routing parameters).

What shrinks: the blocks (the shortest prefix holding every kind of block, at least 2; multi-token-prediction
blocks dropped), the attention heads (--heads, default 4; key/value heads in the model's ratio: as many as the
query heads without grouping, 1 with multi-query attention, else --kv-heads), every FFN and expert width (--ffn,
default 128) and the routed experts (--experts, default 16, at least top-k + 1 and a multiple of the expert
groups). A reduction this generic cannot know every architecture's coupled sizes (a latent rank, a state size, a
convolution's channels, an index's top-k small enough to select within a short text): --config sets fields after
it, as JSON, and the tool prints the configuration it saved.

Only the configuration and the tokenizer files are downloaded, plus the repository's Python files with
--trust-remote-code (modeling code that runs: only for repositories you trust). Needs torch, transformers and
huggingface_hub (the ref environment of the development caches).
"""

import argparse
import copy
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_trace import EXPERT_COUNT_KEYS, TOP_K_KEYS, _layer_count, _randomize, blocks_to_trace  # noqa: E402

HIDDEN_KEYS = ("hidden_size", "n_embd", "d_model", "model_dim")
HEAD_KEYS = ("num_attention_heads", "n_head", "num_heads", "n_heads")
KV_HEAD_KEYS = ("num_key_value_heads", "n_head_kv", "num_kv_heads", "multi_query_group_num", "n_kv_heads")
HEAD_DIM_KEYS = ("head_dim", "kv_channels")
FFN_KEYS = ("n_inner", "ffn_hidden_size", "ffn_dim", "d_ff")  # besides every "*intermediate_size*" field
TOKENIZER_FILES = ("tokenizer*", "vocab*", "merges.txt", "special_tokens_map.json", "added_tokens.json", "*.tiktoken",
                   "*.model", "chat_template*", "generation_config.json", "tekken.json")


def first(cfg, keys):
    for k in keys:
        v = getattr(cfg, k, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return k, v
    return None, None


def reduce(cfg, heads, kv_heads, width, ffn, experts):
    """Shrinks `cfg` (a text configuration) in place; returns (blocks kept, blocks of the model)."""
    key, n = _layer_count(cfg)
    keep = blocks_to_trace(cfg)
    for k, v in list(vars(cfg).items()):
        if isinstance(v, (list, tuple)) and len(v) == n and n > 1:
            setattr(cfg, k, type(v)(v[:keep]))
    setattr(cfg, key, keep)
    if isinstance(getattr(cfg, "mlp_only_layers", None), list):
        cfg.mlp_only_layers = [i for i in cfg.mlp_only_layers if i < keep]
    for k in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
        if isinstance(getattr(cfg, k, None), int):
            setattr(cfg, k, 0)

    head_key, n_head = first(cfg, HEAD_KEYS)
    hidden_key, hidden = first(cfg, HIDDEN_KEYS)
    if head_key and hidden_key:
        dim_key, head_dim = first(cfg, HEAD_DIM_KEYS)
        head_dim = head_dim or hidden // n_head
        new_heads = min(heads, n_head)
        kv_key, n_kv = first(cfg, KV_HEAD_KEYS)
        setattr(cfg, head_key, new_heads)
        if kv_key:
            if n_kv == n_head:
                new_kv = new_heads  # no grouping
            elif n_kv == 1:
                new_kv = 1  # multi-query
            else:
                new_kv = max(1, min(kv_heads, new_heads))
                while new_heads % new_kv:
                    new_kv -= 1
            setattr(cfg, kv_key, new_kv)
        # The head size stays the model's: explicit when the configuration names it, else implied by the width.
        setattr(cfg, hidden_key, width or new_heads * head_dim)

    for k, v in list(vars(cfg).items()):
        if isinstance(v, int) and not isinstance(v, bool) and v > ffn and ("intermediate_size" in k or k in FFN_KEYS):
            setattr(cfg, k, ffn)

    top_k = max([getattr(cfg, k) for k in TOP_K_KEYS if isinstance(getattr(cfg, k, None), int)] or [0])
    for k in EXPERT_COUNT_KEYS:
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > experts:
            groups = getattr(cfg, "n_group", None) or 1
            count = max(experts, top_k + 1)
            setattr(cfg, k, min(v, ((count + groups - 1) // groups) * groups))
    return keep, n


def tokenizer_files(model, revision, trust_remote_code, out):
    """Copies the tokenizer files (and the Python files with --trust-remote-code) of `model` into `out`."""
    patterns = list(TOKENIZER_FILES) + (["*.py"] if trust_remote_code else [])
    if os.path.isdir(model):
        src = model
    else:
        from huggingface_hub import snapshot_download

        src = snapshot_download(model, revision=revision, allow_patterns=patterns)
    import fnmatch

    copied = []
    for name in sorted(os.listdir(src)):
        path = os.path.join(src, name)
        if os.path.isfile(path) and any(fnmatch.fnmatch(name, p) for p in patterns):
            shutil.copy(path, os.path.join(out, name))
            copied.append(name)
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local directory")
    parser.add_argument("--out", required=True, help="directory to write (created; must not exist)")
    parser.add_argument("--revision", help="model revision (branch, tag or commit)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="run modeling code shipped with the model (only for repositories you trust)")
    parser.add_argument("--heads", type=int, default=4, help="attention heads (default 4)")
    parser.add_argument("--kv-heads", type=int, default=2,
                        help="key/value heads when the model groups them (default 2)")
    parser.add_argument("--width", type=int, help="residual width (default heads x the model's head size)")
    parser.add_argument("--ffn", type=int, default=128, help="FFN and expert widths (default 128)")
    parser.add_argument("--experts", type=int, default=16, help="routed experts (default 16)")
    parser.add_argument("--config", help="JSON object of configuration fields to set after the reduction")
    parser.add_argument("--seed", type=int, default=1234, help="seed of the random weights (default 1234)")
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists: give a new directory")
    config = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=args.trust_remote_code)
    cfg = copy.deepcopy(config.get_text_config() if hasattr(config, "get_text_config") else config)
    keep, n = reduce(cfg, args.heads, args.kv_heads, args.width, args.ffn, args.experts)
    for field, value in json.loads(args.config or "{}").items():
        setattr(cfg, field, value)

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=args.trust_remote_code, dtype=torch.float32)
    model.eval()
    _randomize(model)  # seeded: the same arguments give the same checkpoint
    os.makedirs(args.out)
    model.save_pretrained(args.out, safe_serialization=True)
    copied = tokenizer_files(args.model, args.revision, args.trust_remote_code, args.out)

    params = sum(p.numel() for p in model.parameters())
    print(f"{type(model).__name__}: {keep} of {n} blocks, {params / 1e6:.1f}M parameters (float32) in {args.out}")
    print(f"tokenizer files: {', '.join(copied) or 'none'}")
    print(json.dumps(model.config.to_diff_dict(), indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
