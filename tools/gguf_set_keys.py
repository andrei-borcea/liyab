#!/usr/bin/env python3
"""Copies a GGUF model with metadata keys set or added, to fix what a converter wrote wrong.

Some published GGUFs declare tokenizer rules their model's own tokenizer does not follow: the ERNIE 4.5 files ask
for a BOS token and, by leaving `tokenizer.ggml.add_space_prefix` out, for a space before the text, while the
Hugging Face tokenizer adds neither. Liyab reads those keys as written, so such a file encodes text differently
from the model's training. This tool writes a copy with the keys corrected; the tensors are copied byte for byte:

    tools/gguf_set_keys.py ERNIE-4.5-0.3B-PT-F16.gguf ERNIE-4.5-0.3B-PT-F16-fixed.gguf \\
        --bool tokenizer.ggml.add_bos_token=false --bool tokenizer.ggml.add_space_prefix=false

A key the file has is replaced in place (the other keys keep their order); a new one is appended. --bool, --u32,
--f32 and --str set a key of that GGUF type. The data section keeps the file's alignment (general.alignment, which
cannot be set here), so tensor offsets stay valid. Check the result with
`tools/gguf_vocab.py --file OUT --show` and `liyab-tok`. Needs only the standard library.
"""

import argparse
import os
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_vocab import GGUF_MAGIC, Source, read_string, read_value  # noqa: E402


def encode_string(s):
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def encode_pair(key, vtype, value):
    out = encode_string(key) + struct.pack("<I", vtype)
    if vtype == 7:
        return out + struct.pack("<?", value)
    if vtype == 4:
        return out + struct.pack("<I", value)
    if vtype == 6:
        return out + struct.pack("<f", value)
    return out + encode_string(value)


def parse_bool(text):
    lowered = text.lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"not a boolean: {text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="GGUF model to copy (a single file, or one part of a split model)")
    parser.add_argument("output", help="GGUF to write (must differ from the input)")
    for flag, help_text in (("--bool", "KEY=true|false"), ("--u32", "KEY=N"), ("--f32", "KEY=X"),
                            ("--str", "KEY=TEXT")):
        parser.add_argument(flag, action="append", default=[], metavar=help_text, help=f"set a {flag[2:]} key")
    args = parser.parse_args()
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        parser.error("write to another file: the input is read while the output is written")

    sets = {}
    for values, vtype, convert in ((args.bool, 7, parse_bool), (args.u32, 4, int), (args.f32, 6, float),
                                   (args.str, 8, str)):
        for item in values:
            key, sep, text = item.partition("=")
            if not sep or not key:
                parser.error(f"expected KEY=VALUE, got {item!r}")
            if key == "general.alignment":
                parser.error("general.alignment cannot be changed: tensor offsets depend on it")
            sets[key] = encode_pair(key, vtype, convert(text))
    if not sets:
        parser.error("nothing to set")

    src = Source(path=args.input)
    if src.read(4) != GGUF_MAGIC:
        sys.exit("not a GGUF file")
    (version, n_tensors, n_kv) = src.unpack("IQQ")
    if version not in (2, 3):
        sys.exit(f"unsupported GGUF version {version}")

    pairs, alignment, replaced = [], 32, set()
    for _ in range(n_kv):
        start = src.pos
        key = read_string(src).decode("utf-8")
        (vtype,) = src.unpack("I")
        value = read_value(src, vtype)
        if key == "general.alignment":
            alignment = int(value)
        if key in sets:
            pairs.append(sets[key])
            replaced.add(key)
            print(f"{key}: {value!r} -> set")
        else:
            pairs.append(src.buf[start:src.pos])
    for key, raw in sets.items():
        if key not in replaced:
            pairs.append(raw)
            print(f"{key}: added")

    infos_start = src.pos
    for _ in range(n_tensors):
        read_string(src)
        (n_dims,) = src.unpack("I")
        src.read(8 * n_dims + 4 + 8)  # dimensions, type, offset (relative to the data section: unchanged)
    infos = src.buf[infos_start:src.pos]
    data_start = src.pos + (-src.pos) % alignment

    with open(args.output, "wb") as out, open(args.input, "rb") as data:
        out.write(GGUF_MAGIC + struct.pack("<IQQ", version, n_tensors, len(pairs)))
        for raw in pairs:
            out.write(raw)
        out.write(infos)
        out.write(b"\0" * ((-out.tell()) % alignment))
        data.seek(data_start)
        shutil.copyfileobj(data, out, 16 << 20)
    print(f"wrote {args.output}: {len(pairs)} keys, {n_tensors} tensors")


if __name__ == "__main__":
    main()
