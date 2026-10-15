#!/usr/bin/env python3
"""Cold-expert compression study for streamed MoE models (docs/research/cold-experts.md).

A streamed MoE decodes at the speed its experts arrive from flash, so fewer bytes per expert means faster tokens.
This tool measures what smaller representations of the routed experts cost in accuracy:

  spectra   singular-value spectra and effective ranks of dequantized expert matrices (are experts low-rank?)
  methods   every candidate representation at several bit budgets, on a sample of experts: relative Frobenius
            error, and relative output error on real inputs captured by `liyab-experts capture`
  patch     writes a model variant for liyab-kl: an APFS clone of the Q8_0-expert base (`liyab-experts to-q8`)
            with chosen experts (the least used fraction, by routing counts measured on a calibration text)
            overwritten by their reconstruction, re-encoded as Q8_0
  simulate  replays captured routing traces through an LFU expert cache (the ExpertStore's policy) to project
            the bytes read per token with a smaller cold tier (a projection, not a phone measurement)

Representations (all our own implementations, built from the GGUF's dequantized weights; column importances
come from the calibration capture's input second moments, the only "calibration" used):

  q3k       the standard Q3_K format (the engine already decodes it), our own encoder: uniform 3-bit,
            16-value sub-blocks, signed levels -4..3, 6-bit scales                                 3.44 bpw
  q2k       uniform 2-bit, Q2_K-like: 16-value sub-blocks, 4-bit scale and min                    2.63 bpw
  nu3       non-uniform 3-bit: an 8-level codebook per tensor (weighted Lloyd), 8-bit scale / 32   3.31 bpw
  nu3s16    the same with 6-bit scales per 16 values (Q3_K's bit budget)                          3.44 bpw
  vq2       2-D vector quantization: 64 pairs per tensor, 6 bits per pair, 8-bit scale / 32       3.31 bpw
  iq3xxs    the standard IQ3_XXS format (the engine already decodes it), our own encoder          3.06 bpw
  lbN       LittleBit-style binary factorization at N bits/weight, two paths (main + residual):
            W ~ diag(h) sign(U) diag(l) sign(V)^T diag(g), initialized from a truncated SVD of the
            importance-weighted matrix, then refined locally by alternating least squares on h, g, l
            and coordinate descent on the signs (no training through the model)
  lbiN      the same, its SVD-based initialization only (scales by least squares, no sign search)
  svdN      truncated SVD with fp16 factors at N bits/weight (reference for "low rank")
  zero      the expert removed (0 bits): its output is dropped
  Suffix "w": the representation is fitted to the importance-weighted error (input second moments).

Dequantization uses the gguf package's numpy decoders (gguf-py); `check-dequant` compares them with the engine's
own decoder (`liyab-experts dequant`), and `check-encoders` decodes our IQ3_XXS and Q3_K blocks with both. Requires numpy, torch and gguf (pip install gguf).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    import gguf
    from gguf import quants as gq
except ImportError:  # pragma: no cover
    sys.exit("expert_study.py needs the gguf package (pip install gguf)")

MATRICES = ("gate", "up", "down")
TENSOR = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
QK = 256

# ---------------------------------------------------------------------------------------------------------------
# Model access


class Model:
    """The routed-expert tensors of a GGUF, dequantized on demand with gguf-py."""

    def __init__(self, path: str):
        self.path = path
        self.reader = gguf.GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        arch = self.reader.fields["general.architecture"]
        self.arch = bytes(arch.parts[arch.data[0]]).decode()
        self.n_layers = self._int("block_count")
        self.n_expert = self._int("expert_count")

    def _int(self, key: str) -> int:
        f = self.reader.fields[f"{self.arch}.{key}"]
        return int(f.parts[f.data[0]][0])

    def tensor(self, layer: int, matrix: str):
        return self.tensors[f"blk.{layer}.{TENSOR[matrix]}.weight"]

    def experts(self, layer: int, matrix: str, ids) -> np.ndarray:
        """float32 [len(ids), rows, cols] of the given experts (rows = outputs, cols = inputs)."""
        t = self.tensor(layer, matrix)
        raw = np.asarray(t.data)[np.asarray(ids)]
        return gq.dequantize(raw, t.tensor_type).astype(np.float32, copy=False)

    def bpw(self, layer: int, matrix: str) -> float:
        t = self.tensor(layer, matrix)
        return 8.0 * int(t.n_bytes) / int(t.n_elements)


@dataclass
class Capture:
    """Routing and activation statistics written by `liyab-experts capture`."""

    dir: Path
    layers: int
    experts: int
    top_k: int
    n_embd: int
    n_ff: int
    tokens: int
    samples: int
    sample_layers: list

    @staticmethod
    def load(path: str) -> "Capture":
        d = Path(path)
        meta = {}
        for line in (d / "meta.txt").read_text().splitlines():
            k, *v = line.split()
            meta[k] = v
        return Capture(d, int(meta["layers"][0]), int(meta["experts"][0]), int(meta["top_k"][0]),
                       int(meta["n_embd"][0]), int(meta["n_ff"][0]), int(meta["tokens"][0]), int(meta["samples"][0]),
                       [int(x) for x in meta.get("sample_layers", [])])

    def counts(self) -> np.ndarray:
        return np.fromfile(self.dir / "counts.u32", np.uint32).reshape(self.layers, self.experts)

    def trace(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.fromfile(self.dir / "trace_ids.u16", np.uint16).reshape(-1, self.layers, self.top_k)
        w = np.fromfile(self.dir / "trace_w.f32", np.float32).reshape(-1, self.layers, self.top_k)
        return ids, w

    def importance(self, layer: int, matrix: str, ids, prior: float = 8.0) -> np.ndarray:
        """Per input column E[x_j^2] of each expert, shrunk toward the block's mean for rarely used experts."""
        width = self.n_ff if matrix == "down" else self.n_embd
        name = "mom_mid.f64" if matrix == "down" else "mom_in.f64"
        mom = np.memmap(self.dir / name, np.float64, "r", shape=(self.layers, self.experts, width))[layer]
        cnt = self.counts()[layer].astype(np.float64)
        layer_mean = mom.sum(0) / max(cnt.sum(), 1.0)
        out = (mom[np.asarray(ids)] + prior * layer_mean) / (cnt[np.asarray(ids)][:, None] + prior)
        return out.astype(np.float32)

    def inputs(self, layer: int, matrix: str, expert: int) -> np.ndarray:
        width = self.n_ff if matrix == "down" else self.n_embd
        kind = "mid" if matrix == "down" else "in"
        n = np.fromfile(self.dir / f"samples_n{'mid' if kind == 'mid' else ''}_L{layer}.u32", np.uint32)[expert]
        data = np.memmap(self.dir / f"samples_{kind}_L{layer}.f32", np.float32, "r",
                         shape=(self.experts, self.samples, width))
        return np.array(data[expert, :n])


def device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------------------------------------------
# Representations. Each takes W [B, m, n] (float32 torch, rows = outputs, cols = inputs, blocks run along the
# inputs as in GGUF) and column importances cw [B, n] (ones when unweighted), and returns the reconstruction.


def _f16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


def q3k_encode(W: torch.Tensor, cw: torch.Tensor):
    """Encodes W [B, m, n] as Q3_K: x = d * sc * q, q in -4..3 per value, sc a signed 6-bit scale per 16 values,
    d fp16 per 256. Per 16 values the scale is the best of a few candidates (largest value mapped near level
    -4, by weighted squared error). Returns (reconstruction, raw blocks uint8 [B, m, n/256, 110])."""
    B, m, n = W.shape
    x = W.reshape(B, m, n // 16, 16)
    w = cw.reshape(B, 1, n // 16, 16).expand_as(x)
    amax_idx = x.abs().argmax(-1, keepdim=True)
    vmax = torch.gather(x, -1, amax_idx)  # signed value of largest magnitude
    best_err, best_s = None, None
    for t in (-4.5, -4.25, -4.0, -3.75, -3.5, -3.25, -3.0):
        s = vmax / t
        s = torch.where(s == 0, torch.full_like(s, 1e-12), s)
        q = torch.clamp(torch.round(x / s), -4, 3)
        err = (w * (x - q * s) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_s = err, s
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_s = torch.where(better, s, best_s)
    # 6-bit signed sub-block scales relative to a per-256 fp16 scale.
    s = best_s.reshape(B, m, n // QK, QK // 16)
    d = _f16(s.abs().amax(-1, keepdim=True) / 31.0)
    d = torch.where(d == 0, torch.full_like(d, 1e-12), d)
    sc = torch.clamp(torch.round(s / d), -32, 31)
    sq = (sc * d).reshape(B, m, n // 16, 1)
    q = torch.clamp(torch.round(x / torch.where(sq == 0, torch.ones_like(sq), sq)), -4, 3)
    q = torch.where(sq == 0, torch.zeros_like(q), q)
    recon = (q * sq).reshape(B, m, n)

    # raw blocks, GGML's block_q3_K: hmask[32] | qs[64] | scales[12] | d (fp16)
    L = (q + 4).to(torch.int64).reshape(-1, QK).cpu().numpy()            # 0..7
    nb = L.shape[0]
    hi = (L > 3).reshape(nb, 8, 32)                                     # element j: bit j / 32 of hmask[j % 32]
    hmask = (hi.astype(np.uint16) << np.arange(8, dtype=np.uint16)[None, :, None]).sum(1).astype(np.uint8)
    low = (L & 3).reshape(nb, 2, 4, 32)                                 # j = 128 h + 32 k + l
    qs = (low << (2 * np.arange(4))[None, None, :, None]).sum(2).astype(np.uint8).reshape(nb, 64)
    ls = (sc.reshape(nb, 16).to(torch.int64) + 32).cpu().numpy()       # 0..63
    scales = np.zeros((nb, 12), dtype=np.uint8)
    for j in range(16):
        lo4, hi2 = ls[:, j] & 0xF, ls[:, j] >> 4
        if j < 8:
            scales[:, j] |= lo4.astype(np.uint8)
        else:
            scales[:, j - 8] |= (lo4 << 4).astype(np.uint8)
        scales[:, j % 4 + 8] |= (hi2 << (2 * (j // 4))).astype(np.uint8)
    out = np.concatenate([hmask, qs, scales, d.reshape(-1).cpu().numpy().astype(np.float16).view(np.uint8)
                          .reshape(nb, 2)], axis=1)
    return recon, out.reshape(B, m, n // QK, 110)


def q3k(W: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    return q3k_encode(W, cw)[0]


def q2k(W: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    """Uniform 2-bit like Q2_K: x = d*sc*q - dmin*mn, q in 0..3, sc and mn 4-bit per 16, d and dmin fp16 per 256."""
    B, m, n = W.shape
    x = W.reshape(B, m, n // 16, 16)
    w = cw.reshape(B, 1, n // 16, 16).expand_as(x)
    lo = torch.clamp(x.amin(-1, keepdim=True), max=0.0)
    hi = x.amax(-1, keepdim=True)
    best_err, best_s, best_lo = None, None, None
    for shrink in (1.0, 0.9, 0.8, 0.7):
        for shift in (0.0, 0.1, 0.2):
            l = lo * (1 - shift * 0.5)
            s = (hi - l) * shrink / 3.0
            s = torch.clamp(s, min=1e-12)
            q = torch.clamp(torch.round((x - l) / s), 0, 3)
            # least-squares refit of scale and offset for these levels
            sw = w.sum(-1, keepdim=True)
            sq_ = (w * q).sum(-1, keepdim=True)
            sqq = (w * q * q).sum(-1, keepdim=True)
            sx = (w * x).sum(-1, keepdim=True)
            sqx = (w * q * x).sum(-1, keepdim=True)
            det = sw * sqq - sq_ * sq_
            ok = det.abs() > 1e-20
            s2 = torch.where(ok, (sw * sqx - sq_ * sx) / torch.where(ok, det, torch.ones_like(det)), s)
            l2 = torch.where(ok, (sx - s2 * sq_) / sw.clamp(min=1e-20), l)
            l2 = torch.clamp(l2, max=0.0)
            s2 = torch.clamp(s2, min=1e-12)
            err = (w * (x - (q * s2 + l2)) ** 2).sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_s, best_lo = err, s2, l2
            else:
                b = err < best_err
                best_err = torch.where(b, err, best_err)
                best_s = torch.where(b, s2, best_s)
                best_lo = torch.where(b, l2, best_lo)
    s = best_s.reshape(B, m, n // QK, QK // 16)
    mn = (-best_lo).reshape(B, m, n // QK, QK // 16)
    d = _f16(s.amax(-1, keepdim=True) / 15.0).clamp(min=1e-12)
    dm = _f16(mn.amax(-1, keepdim=True) / 15.0).clamp(min=1e-12)
    sq = (torch.clamp(torch.round(s / d), 0, 15) * d).reshape(B, m, n // 16, 1).clamp(min=1e-12)
    mq = (torch.clamp(torch.round(mn / dm), 0, 15) * dm).reshape(B, m, n // 16, 1)
    q = torch.clamp(torch.round((x + mq) / sq), 0, 3)
    return (q * sq - mq).reshape(B, m, n)


def _quantize_scales(s: torch.Tensor, B: int, m: int, n: int, sub: int, bits: int) -> torch.Tensor:
    """Signed sub-block scales [B, m, n/sub, 1]: a sign bit and `bits` - 1 bits of magnitude relative to a
    per-256 fp16 maximum (the sign flips the codebook, a free second, mirrored codebook per block)."""
    levels = (1 << (bits - 1)) - 1
    sb = s.reshape(B, m, n // QK, QK // sub)
    d = _f16(sb.abs().amax(-1, keepdim=True) / levels).clamp(min=1e-12)
    mag = torch.clamp(torch.round(sb.abs() / d), 1, levels) * d
    return (torch.where(sb < 0, -mag, mag)).reshape(B, m, n // sub, 1)


_FACTORS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8)


def _search_scale(x, w, rms, quant, dims):
    """Per block, the best of a few scales (both signs) around the block's RMS for the reconstruction
    quant(x / s) * s, by weighted squared error. dims: the block's value axes."""
    best_err = best_s = None
    for sign in (1.0, -1.0):
        for f in _FACTORS:
            s = rms * (sign * f)
            err = (w * (x - quant(x / s) * s) ** 2).sum(dims, keepdim=True)
            if best_err is None:
                best_err, best_s = err, s
            else:
                b = err < best_err
                best_err = torch.where(b, err, best_err)
                best_s = torch.where(b, s, best_s)
    return best_s


def nonuniform(W: torch.Tensor, cw: torch.Tensor, sub: int = 32, scale_bits: int = 8, levels: int = 8,
               iters: int = 6) -> torch.Tensor:
    """Scalar non-uniform quantization: x = s_b * c[q], one codebook c (levels entries) for the whole batch
    (a tensor's experts), a signed s_b per `sub` values. Alternates: per-block scale search (both signs, so
    each block may use the codebook mirrored) with a least-squares refinement, then a weighted Lloyd update
    of the codebook."""
    B, m, n = W.shape
    x = W.reshape(B, m, n // sub, sub)
    w = cw.reshape(B, 1, n // sub, sub).expand_as(x)
    rms = torch.sqrt((w * x * x).sum(-1, keepdim=True) / w.sum(-1, keepdim=True)).clamp(min=1e-12)
    # init: Lloyd-Max levels of a unit Gaussian (8 levels), else quantiles
    if levels == 8:
        c = torch.tensor([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152], device=W.device)
    else:
        c = torch.distributions.Normal(0, 1).icdf(torch.linspace(0.5 / levels, 1 - 0.5 / levels, levels)).to(W.device)

    def assign(xs: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
        mid = (cc[1:] + cc[:-1]) / 2
        return torch.bucketize(xs.contiguous(), mid)

    for it in range(iters):
        s = _search_scale(x, w, rms, lambda z: c[assign(z, c)], (-1,))
        q = assign(x / s, c)
        cq = c[q]
        s_ls = (w * x * cq).sum(-1, keepdim=True) / (w * cq * cq).sum(-1, keepdim=True).clamp(min=1e-20)
        s = torch.where(s_ls.abs() > 1e-12, s_ls, s)
        q = assign(x / s, c)
        if it == iters - 1:
            break
        num = torch.zeros(levels, device=W.device).index_add_(0, q.flatten(), (w * s * x).flatten())
        den = torch.zeros(levels, device=W.device).index_add_(0, q.flatten(), (w * s * s).flatten())
        c = torch.where(den > 0, num / den.clamp(min=1e-30), c)
        c, _ = torch.sort(c)
    c = _f16(c)
    sq = _quantize_scales(s, B, m, n, sub, scale_bits)
    q = assign(x / sq, c)
    return (c[q] * sq).reshape(B, m, n)


def vq2(W: torch.Tensor, cw: torch.Tensor, sub: int = 32, scale_bits: int = 8, entries: int = 64,
        iters: int = 6) -> torch.Tensor:
    """2-D vector quantization of adjacent pairs: x_pair = s_b * C[q], one codebook C (entries x 2) for the
    batch, a signed s_b per `sub` values; scale search + least squares, then a weighted Lloyd update."""
    B, m, n = W.shape
    x = W.reshape(B, m, n // sub, sub // 2, 2)
    w = cw.reshape(B, 1, n // sub, sub // 2, 2).expand_as(x)
    rms = torch.sqrt((w * x * x).sum((-1, -2), keepdim=True) / w.sum((-1, -2), keepdim=True)).clamp(min=1e-12)
    # init: a product of the 8-level Gaussian Lloyd-Max grid (64 points)
    g1 = torch.tensor([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152], device=W.device)
    C = torch.stack(torch.meshgrid(g1, g1, indexing="ij"), -1).reshape(-1, 2)[:entries].clone()

    def assign(xs: torch.Tensor, ws: torch.Tensor, CC: torch.Tensor) -> torch.Tensor:
        flat, wf = xs.reshape(-1, 2), ws.reshape(-1, 2)
        out = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
        step = 1 << 22
        for i in range(0, flat.shape[0], step):
            a, b = flat[i:i + step], wf[i:i + step]
            # weighted distance: sum_c w_c (a_c - C_kc)^2 = const - 2 (w*a).C + w.C^2
            dist = -2 * (b * a) @ CC.T + b @ (CC * CC).T
            out[i:i + step] = dist.argmin(-1)
        return out.reshape(xs.shape[:-1])

    for it in range(iters):
        s = _search_scale(x, w, rms, lambda z: C[assign(z, w, C)], (-1, -2))
        Cq = C[assign(x / s, w, C)]
        s_ls = (w * x * Cq).sum((-1, -2), keepdim=True) / (w * Cq * Cq).sum((-1, -2), keepdim=True).clamp(min=1e-20)
        s = torch.where(s_ls.abs() > 1e-12, s_ls, s)
        if it == iters - 1:
            break
        q = assign(x / s, w, C).flatten()
        num = torch.zeros(entries, 2, device=W.device).index_add_(0, q, (w * s * x).reshape(-1, 2))
        den = torch.zeros(entries, 2, device=W.device).index_add_(0, q, (w * s * s).expand_as(x).reshape(-1, 2))
        C = torch.where(den > 0, num / den.clamp(min=1e-30), C)
    C = _f16(C)
    sq = _quantize_scales(s.reshape(B, m, n // sub, 1), B, m, n, sub, scale_bits).reshape(B, m, n // sub, 1, 1)
    q = assign(x / sq, w, C)
    return (C[q] * sq).reshape(B, m, n)


# --- IQ3_XXS ---------------------------------------------------------------------------------------------------

_IQ3 = {}


def _iq3_tables(dev: torch.device):
    if dev not in _IQ3:
        gq.IQ3_XXS.init_grid()
        grid = np.asarray(gq.IQ3_XXS.grid, dtype=np.float32).reshape(256, 4)
        ksigns = np.frombuffer(gq.IQ2_XXS.ksigns, dtype=np.uint8)
        _IQ3[dev] = (torch.tensor(grid, device=dev), ksigns)
    return _IQ3[dev]


def _iq3_assign(a: torch.Tensor, w: torch.Tensor, db: torch.Tensor, grid: torch.Tensor):
    """Nearest grid point (weighted) of each group of 4 magnitudes a [N, 8, 4] at block scale db [N, 1, 1]."""
    N = a.shape[0]
    idx = torch.empty(N, 8, dtype=torch.long, device=a.device)
    g2 = grid * grid
    step = 1 << 17
    for i in range(0, N, step):
        aa, ww, dd = a[i:i + step], w[i:i + step], db[i:i + step]
        cross = (ww * aa) @ grid.T            # [n, 8, 256]
        quad = ww @ g2.T
        dist = -2 * dd * cross + dd * dd * quad
        idx[i:i + step] = dist.argmin(-1)
    return idx


def iq3xxs_encode(W: torch.Tensor, cw: torch.Tensor):
    """Encodes W [B, m, n] as IQ3_XXS blocks. Returns (reconstruction, raw blocks uint8 [B, m, n/256, 98]).

    Per 32 values: one 4-bit scale (db = d * (0.5 + s) / 2), eight 4-value groups of magnitudes from the format's
    256-point grid, and four 7-bit sign patterns (an 8th sign is implied: each group of 8 has an even number of
    negatives, so where the signs of x have odd parity the cheapest value to flip is flipped)."""
    grid, ksigns = _iq3_tables(W.device)
    B, m, n = W.shape
    x = W.reshape(-1, 8, 4)                        # [N, 8 groups, 4] per 32-value block
    w = cw.reshape(B, 1, n // 32, 8, 4).expand(B, m, n // 32, 8, 4).reshape(-1, 8, 4)
    a = x.abs()
    gmax = float(grid.max())
    amax = a.amax((-1, -2), keepdim=True).clamp(min=1e-12)
    best_err, best_db = None, None
    for f in (0.70, 0.78, 0.86, 0.94, 1.02, 1.10):
        db = amax / gmax * f
        idx = _iq3_assign(a, w, db, grid)
        err = (w * (a - db * grid[idx]) ** 2).sum((-1, -2), keepdim=True)
        if best_err is None:
            best_err, best_db = err, db
        else:
            b = err < best_err
            best_err = torch.where(b, err, best_err)
            best_db = torch.where(b, db, best_db)
    # 4-bit block scales against an fp16 super-block scale (8 blocks of 32 per 256)
    dbs = best_db.reshape(-1, 8)
    d = _f16(dbs.amax(-1, keepdim=True) / 7.75).clamp(min=1e-12)
    sc = torch.clamp(torch.round(2 * dbs / d - 0.5), 0, 15)
    dbq = (d * (0.5 + sc) * 0.5).reshape(-1, 1, 1)
    idx = _iq3_assign(a, w, dbq, grid)
    mag = dbq * grid[idx]                          # [N, 8, 4]
    neg = (x < 0).reshape(-1, 4, 8)                # sign groups of 8 values
    odd = (neg.sum(-1) % 2) == 1
    cost = (w * x.abs() * mag).reshape(-1, 4, 8)   # flipping costs 4 * w * |x| * mag
    cost = torch.where(mag.reshape(-1, 4, 8) == 0, torch.zeros_like(cost), cost)
    flip = torch.nn.functional.one_hot(cost.argmin(-1), 8).bool() & odd.unsqueeze(-1)
    neg = neg ^ flip
    sign = torch.where(neg, -1.0, 1.0).reshape(-1, 8, 4)
    recon = (mag * sign).reshape(B, m, n)

    # raw blocks: d (fp16) | 64 grid indices | 8 x uint32 (4 x 7 sign bits | scale << 28)
    nb = d.shape[0]
    bits = (neg.to(torch.int64) << torch.arange(8, device=x.device)).sum(-1) & 127   # [N, 4]
    aux = (bits << (torch.arange(4, device=x.device) * 7)).sum(-1) | (sc.reshape(-1).to(torch.int64) << 28)
    out = np.zeros((nb, 98), dtype=np.uint8)
    out[:, 0:2] = d.reshape(-1).cpu().numpy().astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:66] = idx.reshape(nb, 64).cpu().numpy().astype(np.uint8)
    out[:, 66:98] = aux.reshape(nb, 8).cpu().numpy().astype(np.uint32).view(np.uint8).reshape(nb, 32)
    return recon, out.reshape(B, m, n // QK, 98)


def iq3xxs(W: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    return iq3xxs_encode(W, cw)[0]


# --- LittleBit-style binary factorization -----------------------------------------------------------------------


def lb_rank(m: int, n: int, bpw: float, paths: int = 2) -> int:
    """Rank per path such that paths x (r(m+n) sign bits + 16(m+n+r) scale bits) fits in bpw * m * n bits."""
    per_path = bpw * m * n / paths
    return max(1, int((per_path - 16 * (m + n)) // (m + n + 16)))


def lb_bits(m: int, n: int, r: int, paths: int = 2) -> float:
    return paths * (r * (m + n) + 16 * (m + n + r))


def _lb_path(R: torch.Tensor, cw: torch.Tensor, r: int, rounds: int = 4) -> torch.Tensor:
    """One binary path fitted to R [B, m, n] under column weights cw [B, n]:
    R ~ diag(h) S diag(l) T^T diag(g), S [m, r] and T [n, r] in {-1, +1}."""
    B, m, n = R.shape
    sw = cw.sqrt().unsqueeze(1)                     # [B, 1, n]
    U_, Sv, Vh = torch.linalg.svd((R * sw).cpu(), full_matrices=False)
    U_, Sv, Vh = U_.to(R.device), Sv.to(R.device), Vh.to(R.device)
    rr = min(r, Sv.shape[-1])
    U = U_[..., :rr] * Sv[..., None, :rr].sqrt()                       # [B, m, rr]
    V = Vh[..., :rr, :].transpose(-1, -2) * Sv[..., None, :rr].sqrt()  # [B, n, rr] (weighted space)
    V = V / sw.transpose(-1, -2).clamp(min=1e-12)                      # back to the original columns
    if rr < r:  # rank beyond min(m, n): extra random sign columns, refined below
        g = torch.Generator(device="cpu").manual_seed(r)
        U = torch.cat([U, 1e-3 * torch.randn(B, m, r - rr, generator=g).to(R.device)], -1)
        V = torch.cat([V, 1e-3 * torch.randn(B, n, r - rr, generator=g).to(R.device)], -1)

    def rank1(A: torch.Tensor):  # |A| ~ a b^T (top singular pair of a non-negative matrix: power iteration)
        b = torch.ones(A.shape[0], A.shape[2], device=A.device)
        for _ in range(8):
            a = (A @ b.unsqueeze(-1)).squeeze(-1)
            a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-20)
            b = (A.transpose(-1, -2) @ a.unsqueeze(-1)).squeeze(-1)
        return a, b

    hu, lu = rank1(U.abs())
    gv, lv = rank1(V.abs())
    S, T = torch.sign(U), torch.sign(V)
    S[S == 0] = 1
    T[T == 0] = 1
    h, g, l = hu, gv, lu * lv
    W2 = cw.unsqueeze(1)                                               # column weights for least squares

    def recon():
        return h.unsqueeze(-1) * ((S * l.unsqueeze(1)) @ T.transpose(-1, -2)) * g.unsqueeze(1)

    for _ in range(rounds):
        # l: normal equations G l = b with G = (S^T diag(h^2) S) o (T^T diag(g^2 w) T)
        GS = S.transpose(-1, -2) @ (S * (h * h).unsqueeze(-1))
        GT = T.transpose(-1, -2) @ (T * (g * g * cw).unsqueeze(-1))
        rhs = ((S * h.unsqueeze(-1)).transpose(-1, -2) @ (R * W2) @ (T * g.unsqueeze(-1))).diagonal(dim1=-2, dim2=-1)
        G = GS * GT + 1e-6 * torch.eye(r, device=R.device)
        l = torch.linalg.solve(G.cpu(), rhs.cpu().unsqueeze(-1)).squeeze(-1).to(R.device)
        # h: per row, closed form
        P = (S * l.unsqueeze(1)) @ T.transpose(-1, -2) * g.unsqueeze(1)   # [B, m, n]
        h = (R * P * W2).sum(-1) / (P * P * W2).sum(-1).clamp(min=1e-20)
        # g: per column
        Q = h.unsqueeze(-1) * ((S * l.unsqueeze(1)) @ T.transpose(-1, -2))
        g = (R * Q).sum(-2) / (Q * Q).sum(-2).clamp(min=1e-20)
        # signs of S, one coordinate at a time: row i minimizes ||w^1/2 (R_i - h_i A s)||^2, A = (g o T) diag(l)
        A = (T * g.unsqueeze(-1)) * l.unsqueeze(1)                  # [B, n, r]
        GA = A.transpose(-1, -2) @ (A * cw.unsqueeze(-1))           # [B, r, r]
        c = (R * W2) @ A                                            # [B, m, r]
        hh = h.unsqueeze(-1)
        for k in range(r):
            # gradient-free exact coordinate step: s_k = sign(c_k - h * sum_{j != k} G_kj s_j)
            inner = (S @ GA[:, :, k:k + 1]).squeeze(-1) - S[..., k] * GA[:, k, k].unsqueeze(-1)
            val = hh.squeeze(-1) * c[..., k] - hh.squeeze(-1) ** 2 * inner
            S[..., k] = torch.where(val >= 0, 1.0, -1.0)
        # signs of T likewise, per column j: minimize w_j ||R_:j - g_j Bm t||^2, Bm = (h o S) diag(l)
        Bm = (S * h.unsqueeze(-1)) * l.unsqueeze(1)                 # [B, m, r]
        GB = Bm.transpose(-1, -2) @ Bm
        cT = R.transpose(-1, -2) @ Bm                               # [B, n, r]
        gg = g
        for k in range(r):
            inner = (T @ GB[:, :, k:k + 1]).squeeze(-1) - T[..., k] * GB[:, k, k].unsqueeze(-1)
            val = gg * cT[..., k] - gg * gg * inner
            T[..., k] = torch.where(val >= 0, 1.0, -1.0)
    # final scales after the last sign update
    GS = S.transpose(-1, -2) @ (S * (h * h).unsqueeze(-1))
    GT = T.transpose(-1, -2) @ (T * (g * g * cw).unsqueeze(-1))
    rhs = ((S * h.unsqueeze(-1)).transpose(-1, -2) @ (R * W2) @ (T * g.unsqueeze(-1))).diagonal(dim1=-2, dim2=-1)
    l = torch.linalg.solve((GS * GT + 1e-6 * torch.eye(r, device=R.device)).cpu(),
                           rhs.cpu().unsqueeze(-1)).squeeze(-1).to(R.device)
    h, g, l = _f16(h), _f16(g), _f16(l)
    return recon()


def littlebit(W: torch.Tensor, cw: torch.Tensor, bpw: float, paths: int = 2, rounds: int = 4) -> torch.Tensor:
    B, m, n = W.shape
    r = lb_rank(m, n, bpw, paths)
    out = torch.zeros_like(W)
    for _ in range(paths):
        out = out + _lb_path(W - out, cw, r, rounds)
    return out


def svd_lowrank(W: torch.Tensor, cw: torch.Tensor, bpw: float) -> torch.Tensor:
    B, m, n = W.shape
    r = max(1, int(bpw * m * n // (16 * (m + n))))
    sw = cw.sqrt().unsqueeze(1)
    U, S, Vh = torch.linalg.svd((W * sw).cpu(), full_matrices=False)
    r = min(r, S.shape[-1])
    low = (_f16(U[..., :r] * S[..., None, :r]) @ _f16(Vh[..., :r, :])).to(W.device)
    return low / sw


def bits_per_weight(method: str, m: int, n: int) -> float:
    base = method[:-1] if method.endswith("w") and not method.startswith("zero") else method
    fixed = {"q3k": 3 + 6 / 16 + 16 / 256, "q2k": 2 + 8 / 16 + 32 / 256, "nu3": 3 + 8 / 32 + 16 / 256,
             "nu3s16": 3 + 6 / 16 + 16 / 256, "vq2": 3 + 8 / 32 + 16 / 256, "iq3xxs": 98 * 8 / 256, "zero": 0.0}
    if base in fixed:
        return fixed[base]
    if base.startswith("lb"):
        return lb_bits(m, n, lb_rank(m, n, float(base.lstrip("lbi")))) / (m * n)
    if base.startswith("svd"):
        r = max(1, int(float(base[3:]) * m * n // (16 * (m + n))))
        return 16 * r * (m + n) / (m * n)
    raise ValueError(method)


def reconstruct(method: str, W: torch.Tensor, cw: torch.Tensor | None) -> torch.Tensor:
    """The representation `method` of W; the suffix 'w' weights the fit by column importances cw."""
    weighted = method.endswith("w") and method != "zero"
    base = method[:-1] if weighted else method
    c = cw if (weighted and cw is not None) else torch.ones(W.shape[0], W.shape[2], device=W.device)
    c = c / c.mean(-1, keepdim=True).clamp(min=1e-30)
    if base == "zero":
        return torch.zeros_like(W)
    if base == "q3k":
        return q3k(W, c)
    if base == "q2k":
        return q2k(W, c)
    if base == "nu3":
        return nonuniform(W, c, sub=32, scale_bits=8)
    if base == "nu3s16":
        return nonuniform(W, c, sub=16, scale_bits=6)
    if base == "vq2":
        return vq2(W, c)
    if base == "iq3xxs":
        return iq3xxs(W, c)
    if base.startswith("lbi"):  # the SVD-based initialization only (scales by least squares, no sign search)
        return littlebit(W, c, float(base[3:]), rounds=0)
    if base.startswith("lb"):
        return littlebit(W, c, float(base[2:]))
    if base.startswith("svd"):
        return svd_lowrank(W, c, float(base[3:]))
    raise ValueError(method)


# ---------------------------------------------------------------------------------------------------------------
# Commands


def pick_experts(counts: np.ndarray, per_layer: int) -> list[int]:
    """Experts spread over the usage ranking of a block: hottest to coldest."""
    order = np.argsort(-counts.astype(np.int64), kind="stable")
    pos = np.linspace(0, len(order) - 1, per_layer).round().astype(int)
    return sorted(set(int(order[p]) for p in pos))


def cmd_check_dequant(args):
    model = Model(args.model)
    for layer, matrix in ((0, "gate"), (0, "down"), (args.q6_layer, "down")):
        name = f"blk.{layer}.{TENSOR[matrix]}.weight"
        out = Path(args.work) / "dq_check.f32"
        subprocess.run([args.experts_tool, "dequant", "-m", args.model, "-t", name, "--experts", "3:2", "-o", str(out)],
                       check=True, capture_output=True)
        ours = np.fromfile(out, np.float32)
        ref = model.experts(layer, matrix, [3, 4]).reshape(-1)
        out.unlink()
        print(f"{name} ({model.tensor(layer, matrix).tensor_type.name}): max |gguf-py - engine| = "
              f"{np.abs(ours - ref).max():.3g} over {ref.size} values")


def cmd_check_encoders(args):
    """Encodes a few experts as IQ3_XXS and Q3_K and decodes the blocks with gguf-py and with the engine."""
    model = Model(args.model)
    dev = device()
    W = torch.tensor(model.experts(args.layer, "gate", [0, 1]), device=dev)
    ones = torch.ones(W.shape[0], W.shape[2], device=dev)
    for name, ggml_type, encode, size in (("IQ3_XXS", 18, iq3xxs_encode, 98), ("Q3_K", 11, q3k_encode, 110)):
        recon, blocks = encode(W, ones)
        raw = Path(args.work) / "blocks.bin"
        blocks.tofile(raw)
        ref = gq.dequantize(blocks.reshape(-1, size), gguf.GGMLQuantizationType(ggml_type)).reshape(-1)
        out = Path(args.work) / "blocks.f32"
        subprocess.run([args.experts_tool, "dequant-raw", "--type", str(ggml_type), "--cols", str(W.shape[2]),
                        "-i", str(raw), "-o", str(out)], check=True)
        eng = np.fromfile(out, np.float32)
        r = recon.reshape(-1).cpu().numpy()
        raw.unlink()
        out.unlink()
        print(f"{name}: max |encoder - gguf-py| = {np.abs(r - ref).max():.3g}, max |gguf-py - engine| = "
              f"{np.abs(ref - eng).max():.3g}; relative error vs the weights "
              f"{np.linalg.norm(r - W.reshape(-1).cpu().numpy()) / np.linalg.norm(W.cpu().numpy()):.4f}")


def effective_rank(s: np.ndarray) -> float:
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def rank_for_energy(s: np.ndarray, frac: float) -> int:
    e = np.cumsum(s * s) / (s * s).sum()
    return int(np.searchsorted(e, frac) + 1)


def cmd_spectra(args):
    model = Model(args.model)
    cap = Capture.load(args.capture)
    counts = cap.counts()
    rows = []
    rng = np.random.default_rng(0)
    for layer in args.layers:
        ids = pick_experts(counts[layer], args.per_layer)
        for matrix in MATRICES:
            W = model.experts(layer, matrix, ids)
            imp = cap.importance(layer, matrix, ids)
            for i, e in enumerate(ids):
                w = W[i].astype(np.float64)
                s = np.linalg.svd(w, compute_uv=False)
                sw = np.linalg.svd(w * np.sqrt(imp[i])[None, :], compute_uv=False)
                # the same entries, shuffled: keeps the value distribution, removes any structure
                sh = np.linalg.svd(rng.permutation(w.reshape(-1)).reshape(w.shape), compute_uv=False)
                k = len(s)
                rows.append(dict(layer=layer, matrix=matrix, expert=e, uses=int(counts[layer, e]), k=k,
                                 eff=effective_rank(s) / k, eff_w=effective_rank(sw) / k, eff_shuf=effective_rank(sh) / k,
                                 r50=rank_for_energy(s, .5) / k, r90=rank_for_energy(s, .9) / k,
                                 r99=rank_for_energy(s, .99) / k, r90_w=rank_for_energy(sw, .9) / k,
                                 r90_shuf=rank_for_energy(sh, .9) / k, top1=float(s[0] ** 2 / (s * s).sum()),
                                 lowrank_err_1bpw=float(np.sqrt((s[int(w.size // (16 * sum(w.shape))):] ** 2).sum() / (s * s).sum()))))
                print(json.dumps(rows[-1]))
    # shared input subspace across a block's experts (gate): energy of the stacked matrix's Gram spectrum
    shared = []
    for layer in args.layers:
        W = model.experts(layer, "gate", list(range(model.n_expert))).reshape(-1, cap.n_embd).astype(np.float64)
        ev = np.clip(np.linalg.eigvalsh(W.T @ W)[::-1], 0, None)
        sv = np.sqrt(ev)
        shared.append(dict(layer=layer, stacked_r90=rank_for_energy(sv, .9) / len(sv),
                           stacked_eff=effective_rank(sv) / len(sv)))
        print(json.dumps(shared[-1]))
    Path(args.out).write_text(json.dumps(dict(per_matrix=rows, stacked_gate=shared), indent=1))


def rel_errors(W: torch.Tensor, R: torch.Tensor, X: list) -> tuple[np.ndarray, list]:
    """Per-matrix relative Frobenius error; per-matrix (num, den) of the output error on inputs X[i] [s, n]."""
    fro = ((W - R).norm(dim=(-1, -2)) / W.norm(dim=(-1, -2))).cpu().numpy()
    out = []
    for i, x in enumerate(X):
        if x is None or len(x) == 0:
            out.append((0.0, 0.0))
            continue
        xt = torch.tensor(x, device=W.device)
        y = xt @ W[i].T
        dy = xt @ (W[i] - R[i]).T
        out.append((float((dy * dy).sum()), float((y * y).sum())))
    return fro, out


def cmd_methods(args):
    model = Model(args.model)
    cal = Capture.load(args.capture)
    held = Capture.load(args.heldout) if args.heldout else None
    counts = cal.counts()
    dev = device()
    results = []
    for layer in args.layers:
        ids = pick_experts(counts[layer], args.per_layer)
        for matrix in MATRICES:
            W = torch.tensor(model.experts(layer, matrix, ids), device=dev)
            imp = torch.tensor(cal.importance(layer, matrix, ids), device=dev)
            X = [held.inputs(layer, matrix, e) if held and layer in held.sample_layers else None for e in ids]
            m, n = W.shape[1], W.shape[2]
            for method in args.methods:
                t0 = time.time()
                R = reconstruct(method, W, imp)
                fro, out = rel_errors(W, R, X)
                num = sum(o[0] for o in out)
                den = sum(o[1] for o in out)
                row = dict(layer=layer, matrix=matrix, method=method, bpw=bits_per_weight(method, m, n),
                           orig_bpw=model.bpw(layer, matrix), rel_fro=float(np.sqrt((fro ** 2).mean())),
                           rel_out=float(np.sqrt(num / den)) if den > 0 else None, n_experts=len(ids),
                           seconds=round(time.time() - t0, 2))
                results.append(row)
                print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


def q8_0_encode(x: np.ndarray) -> np.ndarray:
    """float32 [..., n] -> Q8_0 bytes [..., n/32 * 34] (fp16 scale, 32 int8)."""
    shp = x.shape
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(-1, keepdims=True)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0)
    q = np.clip(np.round(b * inv), -127, 127).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(*shp[:-1], shp[-1] // 32 * 34)


def cold_set(counts: np.ndarray, fraction: float) -> np.ndarray:
    """[layers, experts] mask of the least used `fraction` of each block's experts (ties by expert id)."""
    mask = np.zeros_like(counts, dtype=bool)
    k = int(round(fraction * counts.shape[1]))
    for l in range(counts.shape[0]):
        order = np.lexsort((np.arange(counts.shape[1]), counts[l]))
        mask[l, order[:k]] = True
    return mask


def cmd_patch(args):
    model = Model(args.model)
    base = Model(args.base)
    cal = Capture.load(args.capture)
    mask = cold_set(cal.counts(), args.fraction)
    out = Path(args.out)
    if out.exists():
        out.unlink()
    subprocess.run(["cp", "-c", args.base, str(out)], check=True)  # APFS clone: only rewritten blocks use space
    dev = device()
    stats = dict(method=args.method, fraction=args.fraction, experts=int(mask.sum()), sq_err={}, sq_norm={})
    fd = os.open(out, os.O_RDWR)
    try:
        t0 = time.time()
        for layer in range(model.n_layers):
            ids = np.flatnonzero(mask[layer])
            if len(ids) == 0:
                continue
            for matrix in MATRICES:
                bt = base.tensor(layer, matrix)
                assert bt.tensor_type == gguf.GGMLQuantizationType.Q8_0
                expert_bytes = int(bt.n_bytes) // model.n_expert
                imp = torch.tensor(cal.importance(layer, matrix, ids), device=dev)
                for chunk in np.array_split(ids, max(1, len(ids) // args.batch)):
                    W = torch.tensor(model.experts(layer, matrix, chunk), device=dev)
                    R = reconstruct(args.method, W, imp[torch.as_tensor(np.searchsorted(ids, chunk), device=dev)])
                    stats["sq_err"][matrix] = stats["sq_err"].get(matrix, 0.0) + float(((W - R) ** 2).sum())
                    stats["sq_norm"][matrix] = stats["sq_norm"].get(matrix, 0.0) + float((W * W).sum())
                    enc = q8_0_encode(R.cpu().numpy())
                    for j, e in enumerate(chunk):
                        os.pwrite(fd, enc[j].tobytes(), int(bt.data_offset) + int(e) * expert_bytes)
            print(f"\rblock {layer + 1}/{model.n_layers} ({time.time() - t0:.0f} s)", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    finally:
        os.close(fd)
    stats["rel_fro"] = {k: math.sqrt(stats["sq_err"][k] / stats["sq_norm"][k]) for k in stats["sq_err"]}
    print(json.dumps(stats))
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats, indent=1))


def cmd_overlap(args):
    """How well a calibration text's cold set predicts another text's usage."""
    cal = Capture.load(args.capture).counts()
    ev = Capture.load(args.heldout).counts()
    for frac in args.fractions:
        mask = cold_set(cal, frac)
        share = ev[mask].sum() / ev.sum()
        cal_share = cal[mask].sum() / cal.sum()
        print(json.dumps(dict(fraction=frac, calibration_routing_share=float(cal_share),
                              heldout_routing_share=float(share))))


def simulate_lfu(ids: np.ndarray, sizes: np.ndarray, capacity: float, age_every: int = 16, warm: int = 0) -> dict:
    """Replays a routing trace [tokens, layers, k] through a byte-bounded LFU cache with the ExpertStore's
    policy (uses += 1 per use, all counters halved every `age_every` tokens, evict the lowest uses, then the
    oldest). sizes [layers, experts]: bytes of each expert. The first `warm` tokens fill the cache and are not
    counted. Returns misses and bytes read per token.

    Halving every counter keeps their order, so counters are stored scaled by 2^ages (a use adds 2^ages) and
    the eviction order lives in a lazy min-heap: no pass over the whole cache per miss."""
    import heapq
    T, L, K = ids.shape
    E = sizes.shape[1]
    flat_sizes = sizes.reshape(-1)
    uses = np.zeros(L * E)
    version = np.zeros(L * E, dtype=np.int64)
    cached = np.zeros(L * E, dtype=bool)
    heap: list = []
    used_bytes = 0.0
    clock = 0
    inc = 1.0
    misses = 0
    read = 0.0
    for t in range(T):
        for l in range(L):
            for e in ids[t, l]:
                key = l * E + int(e)
                clock += 1
                if not cached[key]:
                    need = flat_sizes[key]
                    while used_bytes + need > capacity and heap:
                        u, last, victim, ver = heapq.heappop(heap)
                        if not cached[victim] or ver != version[victim]:
                            continue
                        cached[victim] = False
                        used_bytes -= flat_sizes[victim]
                    cached[key] = True
                    used_bytes += need
                    if t >= warm:
                        misses += 1
                        read += need
                uses[key] += inc
                version[key] += 1
                heapq.heappush(heap, (uses[key], clock, key, int(version[key])))
        if (t + 1) % age_every == 0:
            inc *= 2.0
            if inc > 2.0 ** 900:  # renormalize before float64 overflows
                uses /= inc
                inc = 1.0
                heap = [(uses[k], c, k, v) for (_, c, k, v) in heap if cached[k] and v == version[k]]
                heapq.heapify(heap)
        if len(heap) > 8 * L * E:  # drop stale entries now and then
            heap = [h for h in heap if cached[h[2]] and h[3] == version[h[2]]]
            heapq.heapify(heap)
    n = max(1, T - warm)
    return dict(misses_per_token=misses / n, mib_per_token=read / n / 2**20)


def trace_counts(ids: np.ndarray, experts: int) -> np.ndarray:
    L = ids.shape[1]
    counts = np.zeros((L, experts), dtype=np.int64)
    for l in range(L):
        counts[l] = np.bincount(ids[:, l].reshape(-1), minlength=experts)
    return counts


def cmd_simulate(args):
    """Cold set from the routing counts of the first half of the trace, replayed on the second half (the cache
    warms on its first quarter), so the cold set is never chosen on the tokens it is measured on."""
    cap = Capture.load(args.capture)
    ids, _ = cap.trace()
    half = ids.shape[0] // 2
    pick, replay = ids[:half], ids[half:]
    counts = trace_counts(pick, cap.experts)
    model = Model(args.model)
    full = np.array([[sum(int(model.tensor(l, m).n_bytes) for m in MATRICES) / model.n_expert] * model.n_expert
                     for l in range(model.n_layers)])
    total = full.sum()
    weights = 3 * cap.n_ff * cap.n_embd
    warm = replay.shape[0] // 4
    results = []
    for share in args.cache_shares:
        capacity = share * total
        base = simulate_lfu(replay, full, capacity, warm=warm)
        results.append(dict(cache_share=share, tier="none", fraction=0.0, **base))
        print(json.dumps(results[-1]), flush=True)
        for frac in args.fractions:
            mask = cold_set(counts, frac)
            for tier_bpw in args.tier_bpw:
                sizes = np.where(mask, weights * tier_bpw / 8, full)
                r = simulate_lfu(replay, sizes, capacity, warm=warm)
                results.append(dict(cache_share=share, tier=tier_bpw, fraction=frac, **r))
                print(json.dumps(results[-1]), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-dequant", help="gguf-py's dequantizer against the engine's (liyab-experts dequant)")
    p.add_argument("--model", required=True)
    p.add_argument("--experts-tool", required=True, help="path to liyab-experts")
    p.add_argument("--work", required=True, help="scratch directory")
    p.add_argument("--q6-layer", type=int, default=39, help="a block whose down experts are Q6_K")
    p.set_defaults(fn=cmd_check_dequant)

    p = sub.add_parser("check-encoders", help="our IQ3_XXS and Q3_K encoders, decoded by gguf-py and the engine")
    p.add_argument("--model", required=True)
    p.add_argument("--experts-tool", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--layer", type=int, default=20)
    p.set_defaults(fn=cmd_check_encoders)

    p = sub.add_parser("spectra", help="singular-value spectra and effective ranks")
    p.add_argument("--model", required=True)
    p.add_argument("--capture", required=True, help="calibration capture (routing counts, input moments)")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 10, 20, 30, 39])
    p.add_argument("--per-layer", type=int, default=8)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_spectra)

    p = sub.add_parser("methods", help="reconstruction errors of every representation on sampled experts")
    p.add_argument("--model", required=True)
    p.add_argument("--capture", required=True, help="calibration capture (importances)")
    p.add_argument("--heldout", help="capture of another text: its inputs measure the output error")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 10, 20, 30, 39])
    p.add_argument("--per-layer", type=int, default=16)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_methods)

    p = sub.add_parser("patch", help="a variant GGUF with the coldest experts replaced by a representation")
    p.add_argument("--model", required=True, help="the original GGUF (weights to compress)")
    p.add_argument("--base", required=True, help="the Q8_0-expert copy (liyab-experts to-q8)")
    p.add_argument("--capture", required=True, help="calibration capture: routing counts pick the experts")
    p.add_argument("--fraction", type=float, required=True, help="least used share of each block's experts")
    p.add_argument("--method", required=True)
    p.add_argument("--batch", type=int, default=128, help="experts per reconstruction batch")
    p.add_argument("--out", required=True)
    p.add_argument("--stats")
    p.set_defaults(fn=cmd_patch)

    p = sub.add_parser("overlap", help="routing share of the cold set on a held-out text")
    p.add_argument("--capture", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.set_defaults(fn=cmd_overlap)

    p = sub.add_parser("simulate", help="LFU cache replay: bytes read per token with a cold tier (projection)")
    p.add_argument("--model", required=True)
    p.add_argument("--capture", required=True, help="capture with a routing trace (decode order)")
    p.add_argument("--cache-shares", type=float, nargs="+", default=[0.10, 0.125, 0.15])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5])
    p.add_argument("--tier-bpw", type=float, nargs="+", default=[3.0625])
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_simulate)

    args = ap.parse_args()
    torch.manual_seed(0)
    args.fn(args)


if __name__ == "__main__":
    main()
