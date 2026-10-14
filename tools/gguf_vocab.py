#!/usr/bin/env python3
"""Writes a vocabulary-only GGUF from the header of a GGUF model, for tokenizer and chat-template checks.

A GGUF file starts with its metadata (architecture, tokenizer vocabulary, merges, chat template) and only then
holds the tensors. This tool reads that header, from a local file or straight from a Hugging Face repository with
HTTP range requests (a few MB instead of the whole model), and writes a GGUF with the same metadata and no
tensors. Liyab's tokenizer loads such a file like the full model, so tokenizers can be checked against Hugging
Face without downloading weights:

    tools/gguf_vocab.py --repo unsloth/gemma-3-270m-it-GGUF --filename gemma-3-270m-it-Q8_0.gguf --out gemma3.gguf
    tools/gguf_vocab.py --file Qwen3-0.6B-BF16.gguf --show
    tools/gguf_vocab.py --file model.gguf --template chat_template.jinja   # writes tokenizer.chat_template

--show prints the tokenizer's metadata (model, pre-tokenizer, vocabulary size, BOS rule, special ids). Split
bookkeeping (split.*) is dropped: the output is always a single file. Needs only the standard library, plus
huggingface_hub for --repo.
"""

import argparse
import struct
import sys
import urllib.request

GGUF_MAGIC = b"GGUF"
# GGUF value types (gguf.md): 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string, 9 array,
# 10 u64, 11 i64, 12 f64.
SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
SCALAR_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Source:
    """Sequential reader over a file or an HTTP URL, fetching ranges on demand."""

    CHUNK = 4 << 20

    def __init__(self, path=None, url=None, headers=None):
        self.path, self.url, self.headers = path, url, headers or {}
        self.buf = b""
        self.pos = 0
        self.file = open(path, "rb") if path else None

    def _fill(self, need):
        while len(self.buf) < need:
            want = max(self.CHUNK, need - len(self.buf))
            if self.file:
                data = self.file.read(want)
            else:
                start = len(self.buf)
                req = urllib.request.Request(self.url, headers={**self.headers,
                                                                 "Range": f"bytes={start}-{start + want - 1}"})
                with urllib.request.urlopen(req) as r:
                    data = r.read()
            if not data:
                sys.exit("unexpected end of the GGUF header")
            self.buf += data

    def read(self, n):
        self._fill(self.pos + n)
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.read(size))


def read_string(src):
    (n,) = src.unpack("Q")
    return src.read(n)


def read_value(src, vtype):
    """Returns the decoded value (strings as str, arrays as lists)."""
    if vtype == 8:
        return read_string(src).decode("utf-8", errors="replace")
    if vtype == 9:
        (etype, count) = src.unpack("IQ")
        if etype == 8:
            return [read_string(src).decode("utf-8", errors="replace") for _ in range(count)]
        if etype in SCALAR_SIZES:
            raw = src.read(SCALAR_SIZES[etype] * count)
            return list(struct.unpack(f"<{count}{SCALAR_FORMATS[etype]}", raw))
        return [read_value(src, etype) for _ in range(count)]
    if vtype in SCALAR_FORMATS:
        return src.unpack(SCALAR_FORMATS[vtype])[0]
    sys.exit(f"unknown GGUF value type {vtype}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", help="local GGUF file")
    parser.add_argument("--repo", help="Hugging Face repository holding the GGUF")
    parser.add_argument("--filename", help="GGUF file name in --repo")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", help="vocabulary-only GGUF to write")
    parser.add_argument("--template", help="write tokenizer.chat_template to this file")
    parser.add_argument("--show", action="store_true", help="print the tokenizer metadata")
    args = parser.parse_args()

    if args.file:
        src = Source(path=args.file)
    elif args.repo and args.filename:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        src = Source(url=hf_hub_url(args.repo, args.filename, revision=args.revision), headers=build_hf_headers())
    else:
        parser.error("give --file, or --repo and --filename")

    if src.read(4) != GGUF_MAGIC:
        sys.exit("not a GGUF file")
    (version, _n_tensors, n_kv) = src.unpack("IQQ")
    if version not in (2, 3):
        sys.exit(f"unsupported GGUF version {version}")

    kept = []  # raw bytes of every key/value pair written to the output
    meta = {}
    for _ in range(n_kv):
        start = src.pos
        key = read_string(src).decode("utf-8")
        (vtype,) = src.unpack("I")
        value = read_value(src, vtype)
        meta[key] = value
        if not key.startswith("split."):
            kept.append(src.buf[start:src.pos])

    if args.out:
        alignment = int(meta.get("general.alignment", 32))
        with open(args.out, "wb") as out:
            out.write(GGUF_MAGIC + struct.pack("<IQQ", 3, 0, len(kept)))
            for raw in kept:
                out.write(raw)
            pad = (-out.tell()) % alignment
            out.write(b"\0" * pad)
    if args.template:
        template = meta.get("tokenizer.chat_template")
        if template is None:
            sys.exit("the model has no tokenizer.chat_template")
        with open(args.template, "w", encoding="utf-8", newline="") as f:
            f.write(template)
    if args.show:
        for key in ("general.architecture", "general.name", "tokenizer.ggml.model", "tokenizer.ggml.pre",
                    "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token", "tokenizer.ggml.add_space_prefix",
                    "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                    "tokenizer.ggml.padding_token_id"):
            if key in meta:
                print(f"{key} = {meta[key]!r}")
        if "tokenizer.ggml.tokens" in meta:
            print(f"vocabulary: {len(meta['tokenizer.ggml.tokens'])} tokens, "
                  f"{len(meta.get('tokenizer.ggml.merges', []))} merges")
        template = meta.get("tokenizer.chat_template")
        print(f"chat template: {len(template)} characters" if template else "chat template: none")


if __name__ == "__main__":
    main()
