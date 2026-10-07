#!/usr/bin/env python3
"""Generates tests/data/quant_vectors.bin: random blocks of every GGML format
Liyab decodes, with the values produced by llama.cpp's reference decoder
(gguf-py). tests/test_engine.cpp checks Liyab's decoders against them.

    pip install gguf==0.19.0
    python3 tools/gen_quant_vectors.py tests/data/quant_vectors.bin

Layout: "LYQV" u32 version, u32 count, then per entry:
    u32 ggml_type, u32 n_blocks, block bytes, f32 expected[n_blocks * block_size]
"""
import struct
import sys

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType as T, quants

rng = np.random.default_rng(1234)
N_BLOCKS = 4


def f16(v):
    return np.array([v], dtype=np.float16).view(np.uint8)


def blocks_for(t):
    block_size, type_size = GGML_QUANT_SIZES[t]
    b = rng.integers(0, 256, size=(N_BLOCKS, type_size), dtype=np.uint8)
    for blk in b:
        d = f16(rng.uniform(0.002, 0.05))
        m = f16(rng.uniform(0.001, 0.02))
        if t in (T.Q4_0, T.Q8_0, T.Q5_0, T.IQ2_XXS, T.IQ2_XS, T.IQ2_S, T.IQ3_XXS, T.IQ3_S, T.IQ1_S, T.IQ4_NL, T.IQ4_XS):
            blk[0:2] = d
        elif t in (T.Q4_1, T.Q5_1, T.Q4_K, T.Q5_K):
            blk[0:2], blk[2:4] = d, m
        elif t == T.Q2_K:
            blk[80:82], blk[82:84] = d, m
        elif t == T.Q3_K:
            blk[108:110] = d
        elif t == T.Q6_K:
            blk[208:210] = d
        elif t == T.TQ1_0:
            blk[52:54] = d
        elif t == T.TQ2_0:
            blk[64:66] = d
        elif t == T.MXFP4:
            blk[0] = rng.integers(110, 136)
        elif t == T.IQ1_M:
            # fp16 d is spread over the top nibble of the four u16 scales.
            dh = int(f16(rng.uniform(0.002, 0.05)).view(np.uint16)[0])
            sc = blk[48:56].view(np.uint16)
            for i in range(4):
                sc[i] = (sc[i] & 0x0FFF) | (((dh >> (4 * i)) & 0xF) << 12)
        elif t == T.BF16:
            vals = rng.normal(0, 1, size=type_size // 2).astype(np.float32)
            blk[:] = (vals.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
    return b


TYPES = [T.BF16, T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0, T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K,
         T.IQ2_XXS, T.IQ2_XS, T.IQ2_S, T.IQ3_XXS, T.IQ3_S, T.IQ1_S, T.IQ1_M, T.IQ4_NL, T.IQ4_XS,
         T.TQ1_0, T.TQ2_0, T.MXFP4, T.NVFP4]

with open(sys.argv[1], "wb") as out:
    out.write(b"LYQV" + struct.pack("<II", 1, len(TYPES)))
    for t in TYPES:
        b = blocks_for(t)
        expected = quants.dequantize(b.reshape(-1), t).astype(np.float32).reshape(-1)
        out.write(struct.pack("<II", int(t), N_BLOCKS))
        out.write(b.tobytes())
        out.write(expected.tobytes())
        assert np.all(np.isfinite(expected)), t.name
print(f"wrote {len(TYPES)} formats x {N_BLOCKS} blocks to {sys.argv[1]}")
