#!/usr/bin/env python3
"""Copies a GGUF with each block's ffn_gate_exps and ffn_up_exps fused into one ffn_gate_up_exps tensor.

    tools/gguf_fuse_experts.py model.gguf model-fused.gguf

The fused tensor holds, per expert, the gate rows and then the up rows: [n_embd, 2 * n_ff, n_expert], the layout
convert_hf_to_gguf.py --fuse-gate-up-exps writes (which covers only some architectures). Rows are copied byte for
byte, so it works for every quantization type with whole rows, and the copy must give the original's logits bit for
bit: a check of the engine's fused-expert path on a real model (compare the two files with liyab-kl --save /
--compare, in place and with --expert-cache to stream the experts). Everything else, metadata included, is copied
unchanged. Needs only the Python standard library; the whole output is written before the input is closed.
"""

import argparse
import re
import struct

GGUF_MAGIC = 0x46554747
T_STRING, T_ARRAY = 8, 9
SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
# GGML type id -> (bytes per block, values per block), for the types a model's experts come in.
TYPE_BLOCKS = {0: (4, 1), 1: (2, 1), 30: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32), 8: (34, 32),
               10: (84, 256), 11: (110, 256), 12: (144, 256), 13: (176, 256), 14: (210, 256), 16: (66, 256),
               17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32), 21: (110, 256), 22: (82, 256),
               23: (136, 256), 29: (56, 256), 34: (54, 256), 35: (66, 256), 39: (17, 32), 40: (36, 64)}


def read_string(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n)


def skip_value(f, kind):
    if kind == T_STRING:
        read_string(f)
    elif kind == T_ARRAY:
        element, count = struct.unpack("<IQ", f.read(12))
        if element == T_STRING:
            for _ in range(count):
                read_string(f)
        else:
            f.seek(SCALAR_SIZES[element] * count, 1)
    else:
        f.seek(SCALAR_SIZES[kind], 1)


def tensor_bytes(dims, kind):
    if kind not in TYPE_BLOCKS:
        raise SystemExit(f"tensor type {kind} is not handled by this script")
    block_bytes, block_values = TYPE_BLOCKS[kind]
    rows = 1
    for d in dims[1:]:
        rows *= d
    return dims[0] // block_values * block_bytes * rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    with open(args.input, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != GGUF_MAGIC:
            raise SystemExit("not a GGUF file")
        kv_start = f.tell()
        alignment = 32
        for _ in range(n_kv):
            key = read_string(f)
            (kind,) = struct.unpack("<I", f.read(4))
            if key == b"general.alignment":
                (alignment,) = struct.unpack("<I", f.read(4))
            else:
                skip_value(f, kind)
        kv_end = f.tell()
        tensors = []
        for _ in range(n_tensors):
            name = read_string(f).decode()
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            kind, offset = struct.unpack("<IQ", f.read(12))
            tensors.append((name, dims, kind, offset))
        data_start = (f.tell() + alignment - 1) // alignment * alignment
        by_name = {t[0]: t for t in tensors}

        def read_tensor(tensor):
            f.seek(data_start + tensor[3])
            return f.read(tensor_bytes(tensor[1], tensor[2]))

        # (name, dims, type, bytes producer), in the input's order; up tensors go with their gate.
        out = []
        fused = 0
        for name, dims, kind, _ in tensors:
            match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up)_exps\.weight", name)
            if match is None:
                out.append((name, dims, kind, lambda t=by_name[name]: read_tensor(t)))
                continue
            if match.group(2) == "up":
                continue
            gate = by_name[name]
            up = by_name.get(f"blk.{match.group(1)}.ffn_up_exps.weight")
            if up is None or up[1] != gate[1] or up[2] != gate[2]:
                raise SystemExit(f"block {match.group(1)}: no ffn_up_exps of the gate's shape and type")
            n_embd, n_ff, n_expert = gate[1]

            def fused_bytes(gate=gate, up=up, n_expert=n_expert):
                g, u = read_tensor(gate), read_tensor(up)
                per_expert = len(g) // n_expert
                return b"".join(g[e * per_expert:(e + 1) * per_expert] + u[e * per_expert:(e + 1) * per_expert]
                                for e in range(n_expert))

            out.append((f"blk.{match.group(1)}.ffn_gate_up_exps.weight", [n_embd, 2 * n_ff, n_expert], kind, fused_bytes))
            fused += 1

        with open(args.output, "wb") as w:
            w.write(struct.pack("<IIQQ", magic, version, len(out), n_kv))
            f.seek(kv_start)
            w.write(f.read(kv_end - kv_start))
            offsets = []
            offset = 0
            for name, dims, kind, _ in out:
                encoded = name.encode()
                w.write(struct.pack("<Q", len(encoded)) + encoded)
                w.write(struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims))
                w.write(struct.pack("<IQ", kind, offset))
                offsets.append(offset)
                offset = (offset + tensor_bytes(dims, kind) + alignment - 1) // alignment * alignment
            w.write(b"\0" * (-w.tell() % alignment))
            base = w.tell()
            for (_, _, _, produce), at in zip(out, offsets):
                w.write(b"\0" * (base + at - w.tell()))
                w.write(produce())
    print(f"{args.output}: {fused} blocks fused, {len(out)} tensors")


if __name__ == "__main__":
    main()
