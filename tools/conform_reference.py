#!/usr/bin/env python3
"""Writes a liyab-kl reference from Hugging Face transformers, for conformance checks.

The reference holds the text's token ids as the model's own Hugging Face
tokenizer encodes them, then, for every position, the log-probabilities of the
32 most likely next tokens and of the text's actual next token, computed by
transformers' implementation of the architecture (teacher-forced, one pass).
liyab-kl compares a GGUF of the same model against it:

    tools/conform_reference.py --model Qwen/Qwen3-0.6B --text text.txt --out qwen3.ref
    liyab-kl -m Qwen3-0.6B-BF16.gguf -f text.txt --compare qwen3.ref --max-kl 0.01 --min-top1 97

liyab-kl first checks that both tokenizers give the same ids, then measures
the distance of the two forward passes. The GGUF's format sets how close they
can be: compare a BF16/F16/Q8_0 file to verify an architecture, since lower
bit widths add their own quantization error on top; pass --kv f16 to liyab-kl
to leave out the KV cache's rounding too.

A GGUF must hold the weights of the revision the reference loads: a file
converted from another checkpoint fails however correct the engine is. When
the distance is large from the first position on, compare a few tensors of the
two files before suspecting the forward pass (one published Qwen2.5-0.5B GGUF
holds other weights than its Hugging Face repository, biases aside).

Needs torch and transformers (pip install torch transformers). Run the model
in float32 (the default) for a reference free of half-precision rounding;
--dtype bfloat16 halves its memory for larger models.

--config sets configuration fields before the model is built, as JSON, for a
reference of a variant the repository does not ship: a GGUF copy given YaRN
metadata is compared with the model run with the same RoPE parameters, e.g.
--config '{"rope_parameters": {"rope_type": "yarn", "rope_theta": 1000000,
"factor": 4.0, "original_max_position_embeddings": 32768}}'.
--attn-implementation picks transformers' attention kernel: Gemma 2 needs
"eager", the default kernel leaves its attention soft cap out.
"""

import argparse
import json
import struct
import sys

KTOP = 32  # tools/liyab_kl.cpp: kTop
MAGIC = b"LYKL"  # tools/liyab_kl.cpp: kMagic
VERSION = 2  # tools/liyab_kl.cpp: kVersion


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local directory")
    parser.add_argument("--text", required=True, help="UTF-8 text file (the one given to liyab-kl -f)")
    parser.add_argument("--out", required=True, help="reference file to write")
    parser.add_argument("--max-tokens", type=int, default=512, help="keep the first N tokens (as liyab-kl, default 512)")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                        help="precision of the reference pass (default float32)")
    parser.add_argument("--revision", help="model revision (branch, tag or commit)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="run modeling code shipped with the model (only for repositories you trust)")
    parser.add_argument("--config", help="JSON object of configuration fields to set before the model is built")
    parser.add_argument("--attn-implementation", help='transformers attention kernel, e.g. "eager"')
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # newline="" keeps "\r\n" as written: liyab-kl encodes the file's exact bytes.
    with open(args.text, encoding="utf-8", newline="") as f:
        text = f.read()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              trust_remote_code=args.trust_remote_code)
    # add_special_tokens follows the model's own rule for a BOS, as liyab-kl
    # follows tokenizer.ggml.add_bos_token.
    ids = tokenizer(text, add_special_tokens=True)["input_ids"][: args.max_tokens]
    if len(ids) < 2:
        sys.exit("the text must encode to at least 2 tokens")

    config = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=args.trust_remote_code)
    for field, value in json.loads(args.config or "{}").items():
        setattr(config, field, value)
    extra = {"attn_implementation": args.attn_implementation} if args.attn_implementation else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, config=config,
                                                 dtype=getattr(torch, args.dtype),
                                                 trust_remote_code=args.trust_remote_code, **extra)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, :-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    top_logp, top_ids = torch.topk(logp, KTOP, dim=-1)
    target = logp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])]

    with open(args.out, "wb") as out:
        out.write(MAGIC + struct.pack("<IQ", VERSION, len(ids)))
        out.write(struct.pack(f"<{len(ids)}i", *ids))
        out.write(struct.pack("<Q", len(ids) - 1))
        # struct Position { int32 ids[32]; float logp[32]; float target_logp; }
        for t in range(len(ids) - 1):
            out.write(struct.pack(f"<{KTOP}i", *top_ids[t].tolist()))
            out.write(struct.pack(f"<{KTOP}f", *top_logp[t].tolist()))
            out.write(struct.pack("<f", target[t].item()))
    nll = -target.double().mean().item()
    print(f"saved {len(ids) - 1} positions to {args.out}; perplexity {torch.exp(torch.tensor(nll)).item():.3f}")


if __name__ == "__main__":
    main()
