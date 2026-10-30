#!/usr/bin/env python3
"""Condition existing GGML test blocks for repeated SSM projection matrices.

The original quant_vectors.bin is unchanged. Only scale fields change; all
integer codes/grids are preserved. Every block uses the same RMS target,
0.5 / sqrt(width), before rounding scales to the format's representation.
Independent decoding (gguf-py, or llama.cpp's libggml for Q1_0: tools/quant_reference.py) supplies the new
expected values.
"""
import argparse
import math
import struct
from pathlib import Path

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType as T, quants

from quant_reference import dequantize


def scale_blocks(blocks, kind, factor):
    # These are encoding layouts, not format-dependent numerical targets.
    half_offsets = {
        T.Q4_0: (0,), T.Q4_1: (0, 2), T.Q5_0: (0,), T.Q5_1: (0, 2), T.Q8_0: (0,),
        T.Q2_K: (80, 82), T.Q3_K: (108,), T.Q4_K: (0, 2), T.Q5_K: (0, 2), T.Q6_K: (208,),
        T.IQ2_XXS: (0,), T.IQ2_XS: (0,), T.IQ2_S: (0,), T.IQ3_XXS: (0,), T.IQ3_S: (0,),
        T.IQ1_S: (0,), T.IQ4_NL: (0,), T.IQ4_XS: (0,), T.TQ1_0: (52,), T.TQ2_0: (64,), T.Q1_0: (0,),
    }
    if kind in half_offsets:
        for at in half_offsets[kind]:
            scales = blocks[:, at:at + 2].copy().view(np.float16)
            blocks[:, at:at + 2] = (scales.astype(np.float32) * factor).astype(np.float16).view(np.uint8)
    elif kind == T.BF16:
        bits = blocks.copy().view(np.uint16).astype(np.uint32) << 16
        values = (bits.view(np.float32) * factor).astype(np.float32)
        bits = values.view(np.uint32)
        blocks[:] = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16).view(np.uint8)
    elif kind == T.IQ1_M:
        scales = blocks[:, 48:56].copy().view(np.uint16)
        bits = np.zeros((len(blocks), 1), dtype=np.uint16)
        for i in range(4):
            bits[:, 0] |= ((scales[:, i] >> 12) & 15) << (4 * i)
        bits = (bits.view(np.float16).astype(np.float32) * factor).astype(np.float16).view(np.uint16)
        for i in range(4):
            scales[:, i] = (scales[:, i] & 0x0FFF) | (((bits[:, 0] >> (4 * i)) & 15) << 12)
        blocks[:, 48:56] = scales.view(np.uint8)
    elif kind == T.MXFP4:
        exponent = blocks[:, 0].astype(np.int32) + np.rint(np.log2(factor[:, 0])).astype(np.int32)
        assert np.all((exponent >= 2) & (exponent < 255))
        blocks[:, 0] = exponent.astype(np.uint8)
    elif kind == T.NVFP4:
        scales = quants.NVFP4.ue4m3_to_fp32(blocks[:, :4]) * 2
        blocks[:, :4] = quants.NVFP4.fp32_to_ue4m3(scales * factor)
    else:
        raise ValueError(f"no scale layout for {kind.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tests/data/quant_vectors.bin"),
                        help="original LYQV fixtures (never modified)")
    parser.add_argument("--out", type=Path, required=True, help="conditioned LYQV fixture output")
    parser.add_argument("--width", type=int, default=256, help="projection width for RMS target (default: 256)")
    args = parser.parse_args()
    if args.width <= 0 or args.input.resolve() == args.out.resolve():
        parser.error("width must be positive and output must differ from input")
    data = memoryview(args.input.read_bytes())
    magic, version, count = struct.unpack_from("<4sII", data)
    assert magic == b"LYQV" and version == 1
    output = bytearray(data[:12])
    pos = 12
    target = 0.5 / math.sqrt(args.width)
    for _ in range(count):
        type_id, n_blocks = struct.unpack_from("<II", data, pos)
        pos += 8
        kind = T(type_id)
        size, block_bytes = GGML_QUANT_SIZES[kind]
        encoded = np.frombuffer(data[pos:pos + n_blocks * block_bytes], dtype=np.uint8).copy().reshape(n_blocks, block_bytes)
        pos += n_blocks * block_bytes
        expected = np.frombuffer(data[pos:pos + n_blocks * size * 4], dtype="<f4")
        pos += n_blocks * size * 4
        # Check the source against its independent decoder before deriving
        # one factor per block, shared by every projection role. Per-block
        # conditioning prevents tiny source blocks underflowing when the
        # same fixture also contains much larger blocks (notably NVFP4).
        decoded = dequantize(encoded.reshape(-1), kind).reshape(-1)
        np.testing.assert_array_equal(decoded, expected)
        rms = np.sqrt(np.mean(expected.reshape(n_blocks, size).astype(np.float64) ** 2, axis=1, keepdims=True))
        before = float(np.sqrt(np.mean(rms ** 2)))
        assert np.all(np.isfinite(rms)) and np.all(rms > 0)
        scale_blocks(encoded, kind, target / rms)
        values = dequantize(encoded.reshape(-1), kind).astype("<f4").reshape(-1)
        after = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
        assert np.all(np.isfinite(values)) and after > 0
        output += struct.pack("<II", type_id, n_blocks) + encoded.tobytes() + values.tobytes()
        print(f"{kind.name}: RMS {before:.9g} -> {after:.9g}; target {target:.9g}")
    assert pos == len(data)
    args.out.write_bytes(output)


if __name__ == "__main__":
    main()
