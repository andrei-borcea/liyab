#!/usr/bin/env python3
"""Generate small upstream checkpoints for liyab-compat conformance, offline.

Requires torch and transformers with Gemma 4 support. Writes random weights
and a local byte-fallback tokenizer; never fetches trained weights or configs.
Convert with the upstream HF-to-GGUF converter, then compare with
conform_reference.py and liyab-kl. This tests text semantics, not trained quality.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_tokenizer():
    """Local byte-fallback vocabulary shared by the numerical fixtures."""
    from transformers import GemmaTokenizer

    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4}
    for i in range(256):
        vocab[f"<0x{i:02X}>"] = len(vocab)
    for c in "▁abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?\n":
        if c not in vocab:
            vocab[c] = len(vocab)
    vocab["th"] = len(vocab)
    return GemmaTokenizer(vocab=vocab, merges=[("t", "h")], add_bos_token=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="directory for generated checkpoint and tokenizer",
    )
    parser.add_argument(
        "--moe",
        action="store_true",
        help="add independent dense/expert branches and router scales",
    )
    parser.add_argument(
        "--architecture", choices=("gemma4", "deepseek4"), default="gemma4"
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        help="DeepSeek-V4 fixture-only GGUF export (requires gguf/numpy)",
    )
    args = parser.parse_args()
    if args.architecture == "deepseek4":
        if args.moe:
            parser.error("DeepSeek-V4 already includes routed and shared experts")
        from compat_deepseek_fixture import create

        create(args.out, args.gguf)
        return
    if args.gguf:
        parser.error("Gemma 4 uses the upstream HF-to-GGUF converter")
    import json
    import torch
    from transformers import Gemma4TextConfig, Gemma4ForCausalLM
    from reference_trace import _randomize

    torch.set_num_threads(4)
    tokenizer = make_tokenizer()
    moe = args.moe
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(root)
    config = Gemma4TextConfig(
        vocab_size=len(tokenizer),
        vocab_size_per_layer_input=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=8,
        hidden_size_per_layer_input=16,
        num_kv_shared_layers=2,
        attention_k_eq_v=True,
        use_double_wide_mlp=True,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    if moe:
        config.enable_moe_block = True
        config.num_experts = 4
        config.top_k_experts = 2
        config.moe_intermediate_size = 64
    config._attn_implementation = "eager"
    model = Gemma4ForCausalLM(config)
    _randomize(model)
    # Count distinct routed experts so a conformance run cannot pass with a
    # dormant MoE branch. The checkpoint uses the same weights as this probe.
    audit = {}
    hooks = []
    if moe:

        def record(name):
            def hook(module, inputs, output):
                audit[name] = sorted(set(output[2].flatten().tolist()))

            return hook

        for name, module in model.named_modules():
            if name.endswith(".router"):
                hooks.append(module.register_forward_hook(record(name)))
        model.eval()
        with torch.no_grad():
            model(
                torch.arange(128).remainder(len(tokenizer)).unsqueeze(0),
                use_cache=False,
            )
        for hook in hooks:
            hook.remove()
        if len(audit) != config.num_hidden_layers or any(
            len(ids) < 3 for ids in audit.values()
        ):
            raise RuntimeError(f"inactive expert-routing fixture: {audit}")
    model.save_pretrained(root)
    (root / "compat-fixture.json").write_text(
        json.dumps({"seed": 1234, "routed_experts": audit}, indent=2) + "\n"
    )
    print(
        f"{root}: {sum(p.numel() for p in model.parameters())} parameters, {len(tokenizer)} tokens"
    )


if __name__ == "__main__":
    main()
