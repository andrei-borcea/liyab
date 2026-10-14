#!/usr/bin/env python3
"""Writes the token ids a model's own Hugging Face tokenizer gives a text, for liyab-tok to compare against.

    tools/tokenizer_reference.py --model Qwen/Qwen3-0.6B --text tests/data/tokenizer_edge.txt --out qwen3.ids
    build/liyab-tok -m Qwen3-0.6B-BF16.gguf -f tests/data/tokenizer_edge.txt --ids qwen3.ids

Only the tokenizer files are read (tokenizer.json, tokenizer_config.json, ...), never weights, so checking a
tokenizer costs a few MB; liyab-tok accepts the full GGUF or a vocabulary-only one written by tools/gguf_vocab.py.
The text is encoded without BOS/EOS unless --special is given (then liyab-tok needs --bos). Special-token texts
written in the text (such as "<|im_end|>") map to their ids on both sides.

The model's tokenizer.json is run by the Hugging Face `tokenizers` library itself: it is what the model was
trained with, and transformers may rebuild a tokenizer from its class instead (transformers 5 turns DeepSeek
LLM's into a SentencePiece-style one with other ids). --transformers uses AutoTokenizer, for models without a
tokenizer.json.

Needs tokenizers and huggingface_hub (pip install tokenizers huggingface_hub), or transformers with --transformers.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local directory")
    parser.add_argument("--text", required=True, help="UTF-8 text file (the one given to liyab-tok -f)")
    parser.add_argument("--out", required=True, help="ids file to write")
    parser.add_argument("--special", action="store_true", help="add the model's BOS/EOS as its tokenizer does")
    parser.add_argument("--revision", help="model revision (branch, tag or commit)")
    parser.add_argument("--transformers", action="store_true", help="use transformers' AutoTokenizer")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="with --transformers: run tokenizer code shipped with the model (trusted repositories only)")
    args = parser.parse_args()

    # newline="" keeps "\r\n" as written: the tokenizer must see the file's exact text.
    with open(args.text, encoding="utf-8", newline="") as f:
        text = f.read()
    if args.transformers:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                  trust_remote_code=args.trust_remote_code)
        ids = tokenizer(text, add_special_tokens=args.special)["input_ids"]
    else:
        from tokenizers import Tokenizer
        path = os.path.join(args.model, "tokenizer.json")
        if not os.path.isfile(path):
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(args.model, "tokenizer.json", revision=args.revision)
        ids = Tokenizer.from_file(path).encode(text, add_special_tokens=args.special).ids
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(f"# {args.model} {args.text} special={int(args.special)} tokens={len(ids)}\n")
        for i in range(0, len(ids), 32):
            out.write(" ".join(str(t) for t in ids[i:i + 32]) + "\n")
    print(f"{len(ids)} tokens -> {args.out}")


if __name__ == "__main__":
    main()
