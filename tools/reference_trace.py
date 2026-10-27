#!/usr/bin/env python3
"""Observes what a model's Hugging Face transformers implementation computes, for tools/describe_model.py.

A reduced copy of the model is built from its configuration with random weights: fewer blocks (enough to hold
every kind of block the configuration lists), a small vocabulary, narrow FFNs and few experts. The widths of the
residual stream and of the attention heads are kept, so every scale, base and multiplier is the model's own. One
pass over a short sequence is recorded: what every module receives and returns, and, through an attention function
registered with transformers (or by observing native SDPA in older classes), the queries, keys and values each
attention gets with its arguments (scale, window, soft-cap). The facts the engine needs are then established by
recomputing them, never from class or module names:

* the order of norms, token mixer and FFN in a block and how the residual adds combine them;
* each norm's formula (RMSNorm multiplying by w plus a measured scalar offset, LayerNorm);
* how keys and queries are rotated (which dimensions are paired, how many, with which frequencies), whether their
  norms come before or after the rotation, and over how many dimensions those norms reach;
* the attention scale, window and soft-cap, and whether the attention output is gated;
* the FFN's form and activation, and a mixture of experts' routing (probabilities, selection bias, top-k,
  renormalization, scale, shared expert and its gate);
* how the embeddings enter the first block and how the logits leave the last one.

What cannot be recomputed is reported as unknown, never guessed. Needs torch and transformers.

    tools/reference_trace.py --model Qwen/Qwen3-0.6B          # prints the facts as JSON
"""

import argparse
import copy
import json
import functools
import inspect
import math
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

SEQ = 13  # trace length: prime, so it is never mistaken for a head count when shapes are matched
VOCAB = 512  # reduced vocabulary (the trace's token ids are below it)
FFN = 64  # reduced FFN and expert widths
EXPERTS = 16  # reduced routed expert count (kept at least top-k + 1)
ATTN_IMPL = "liyab_trace"

# Recomputed values must match the reference to within float32 rounding of a few operations.
RTOL = 2e-4
ATOL = 2e-5


@dataclass
class Call:
    """One module invocation, in execution order."""
    order: int
    path: str
    module: nn.Module
    args: list  # positional arguments (tensors kept, others as given)
    kwargs: dict
    outputs: list  # every tensor in the output, in order

    @property
    def input(self):
        """The main input: first positional tensor, else hidden_states / x / the first tensor keyword."""
        for a in self.args:
            if torch.is_tensor(a):
                return a
        for key in ("hidden_states", "x", "input", "inputs_embeds"):
            if torch.is_tensor(self.kwargs.get(key)):
                return self.kwargs[key]
        for a in self.args:
            nested = _tensors(a)
            if nested:
                return nested[0]
        for v in self.kwargs.values():
            if torch.is_tensor(v):
                return v
        return None

    @property
    def output(self):
        return self.outputs[0] if self.outputs else None


@dataclass
class AttnCall:
    """One call of the attention function: what the attention itself receives."""
    order: int
    module: nn.Module
    query: torch.Tensor  # [batch, heads, seq, head_dim]
    key: torch.Tensor
    value: torch.Tensor
    kwargs: dict  # non-tensor arguments (scaling, sliding_window, softcap, ...) and tensor ones by name
    output: torch.Tensor = None  # [batch, seq, heads, head_dim] as returned


@dataclass
class Recorder:
    calls: list = field(default_factory=list)
    attn: list = field(default_factory=list)
    recurrences: list = field(default_factory=list)
    counter: int = 0
    clamps: list = field(default_factory=list)
    norm_eps: dict = field(default_factory=dict)
    gelu_modes: list = field(default_factory=list)

    def next(self):
        self.counter += 1
        return self.counter


RECORDER = Recorder()


def _tensors(obj):
    if torch.is_tensor(obj):
        return [obj.detach().clone()]
    if isinstance(obj, (list, tuple)):
        out = []
        for o in obj:
            out.extend(_tensors(o))
        return out
    if isinstance(obj, dict):
        out = []
        for o in obj.values():
            out.extend(_tensors(o))
        return out
    if hasattr(obj, "to_tuple"):  # ModelOutput
        return _tensors(obj.to_tuple())
    return []


def _register_attention():
    """Registers ATTN_IMPL: records the attention's inputs and arguments, then runs transformers' SDPA path."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

    def traced(module, query, key, value, attention_mask, **kwargs):
        rec = AttnCall(RECORDER.next(), module, query.detach().clone(), key.detach().clone(),
                       value.detach().clone(), {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()})
        RECORDER.attn.append(rec)
        # SDPA ignores what it does not know (a soft-cap, sinks): the trace only needs the inputs, and the blocks
        # after this one stay self-consistent.
        passed = {k: v for k, v in kwargs.items() if k in ("dropout", "scaling", "is_causal", "sliding_window")}
        out, weights = sdpa_attention_forward(module, query, key, value, attention_mask, **passed)
        rec.output = out.detach().clone()
        return out, weights

    AttentionInterface.register(ATTN_IMPL, traced)
    # An explicit mask also works for eager implementations that cast it to
    # bool. SDPA's optional None fast path would make those fail before the
    # observed score operations; the trace itself still runs SDPA.
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTN_IMPL, eager_mask)


# ----------------------------------------------------------------------------------------------------------------
# Configuration: a reduced copy that keeps every kind of block
# ----------------------------------------------------------------------------------------------------------------

LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
TOP_K_KEYS = ("num_experts_per_tok", "num_experts_per_token", "moe_k", "moe_topk", "top_k", "moe_top_k")


def canonical_rope_config(cfg):
    """Spell a legacy RoPE `type` key as `rope_type`, without changing its value or other parameters."""
    def visit(value):
        if isinstance(value, dict):
            if "type" in value and "rope_type" not in value:
                value["rope_type"] = value.pop("type")
            for child in value.values():
                visit(child)
    visit(getattr(cfg, "rope_parameters", None))


def _layer_count(cfg):
    for k in LAYER_COUNT_KEYS:
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > 0:
            return k, v
    raise ValueError("the configuration names no layer count")


def blocks_to_trace(cfg):
    """The shortest prefix of blocks that holds every kind of block the configuration lists per block, at least 2."""
    _, n = _layer_count(cfg)
    want = min(n, 2)
    fields = dict(vars(cfg))
    for key in ("layer_types", "interleaved_sliding_window"):
        value = getattr(cfg, key, None)
        if isinstance(value, (list, tuple)):
            fields[key] = value
    for key, value in fields.items():
        if isinstance(value, (list, tuple)) and len(value) == n and n > 1:
            seen = set()
            for i, v in enumerate(value):
                h = json.dumps(v, sort_keys=True, default=str)
                if h not in seen:
                    seen.add(h)
                    want = max(want, i + 1)
    for key in ("first_k_dense_replace", "decoder_sparse_step", "moe_layer_freq", "full_attention_interval",
                "num_dense_layers", "moe_layer_start_index"):
        v = getattr(cfg, key, None)
        if isinstance(v, int) and v > 0:
            want = max(want, v + 1)
    mlp_only = getattr(cfg, "mlp_only_layers", None)
    if mlp_only:
        dense = [i for i in range(n) if i in mlp_only]
        sparse = [i for i in range(n) if i not in mlp_only]
        if dense and sparse:
            want = max(want, max(dense[0], sparse[0]) + 1)
    # A shared-KV suffix needs a source block of every attention kind before
    # it starts. Keep the declared suffix and that shortest source prefix.
    shared = getattr(cfg, "num_kv_shared_layers", 0) or 0
    kinds = getattr(cfg, "layer_types", None)
    if shared and kinds:
        first_of_kind = max(kinds.index(t) + 1 for t in set(kinds))
        want = max(want, shared + first_of_kind)
    return min(n, max(want, 2))


def reduced_config(config):
    """A copy of the text configuration with fewer blocks, a small vocabulary, narrow FFNs and few experts."""
    cfg = copy.deepcopy(config.get_text_config() if hasattr(config, "get_text_config") else config)
    # A window longer than the trace looks exactly like global causal
    # attention. Shorten a uniform configured window so its boundary is
    # exercised, just as FFN widths are reduced below. Reconstruct through
    # the configuration owner so aliases and computed layer types agree.
    windows = {k: getattr(cfg, k, None) for k in ("sliding_window", "window_size")}
    windows = {k: v for k, v in windows.items() if isinstance(v, int) and not isinstance(v, bool) and v > 0}
    if len(set(windows.values())) == 1 and next(iter(windows.values())) > SEQ // 2:
        cfg = type(cfg).from_dict({**cfg.to_dict(), **{k: SEQ // 2 for k in windows}})
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
    if getattr(cfg, "vocab_size", 0) > VOCAB:
        cfg.vocab_size = VOCAB
    if getattr(cfg, "vocab_size_per_layer_input", 0) > VOCAB:
        cfg.vocab_size_per_layer_input = VOCAB
    for k, v in list(vars(cfg).items()):
        if k.endswith("token_id") and isinstance(v, int) and v >= VOCAB:
            setattr(cfg, k, None)
        if k.endswith("token_id") and isinstance(v, list):
            setattr(cfg, k, [t for t in v if isinstance(t, int) and t < VOCAB] or None)
    for k, v in list(vars(cfg).items()):
        if "intermediate_size" in k and isinstance(v, int) and v > FFN:
            setattr(cfg, k, FFN)
        elif "intermediate_size" in k and isinstance(v, (list, tuple)):
            setattr(cfg, k, type(v)(min(x, FFN) if isinstance(x, int) and x > 0 else x for x in v))
    # Pure SSM stacks have no attention head geometry to preserve. Keep the
    # expansion and head/group divisibility, with a small residual stream.
    if hasattr(cfg, "state_size") and hasattr(cfg, "conv_kernel") and not hasattr(cfg, "num_attention_heads"):
        cfg.hidden_size = 128
        cfg.intermediate_size = int(getattr(cfg, "expand", 2)) * cfg.hidden_size
        if hasattr(cfg, "num_heads"):
            groups = int(getattr(cfg, "n_groups", 1))
            cfg.num_heads = max(groups, cfg.intermediate_size // cfg.head_dim)
            cfg.intermediate_size = cfg.num_heads * cfg.head_dim
            cfg.hidden_size = cfg.intermediate_size // int(getattr(cfg, "expand", 2))
    top_k = max([getattr(cfg, k) for k in TOP_K_KEYS if isinstance(getattr(cfg, k, None), int)] or [0])
    for k in EXPERT_COUNT_KEYS:
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > EXPERTS:
            groups = getattr(cfg, "n_group", None) or 1
            count = max(EXPERTS, top_k + 1)
            count = ((count + groups - 1) // groups) * groups
            setattr(cfg, k, min(v, count))
    # Typed child configurations can own expert widths/counts too (DBRX).
    # Keep the residual width for matching the original file, but bound the
    # randomly constructed FFNs just as the top-level ones are bounded.
    for child in vars(cfg).values():
        if not hasattr(child, "to_dict"):
            continue
        child_top_k = max([getattr(child, key) for key in TOP_K_KEYS
                           if isinstance(getattr(child, key, None), int)] or [top_k])
        for key, value in list(vars(child).items()):
            if not isinstance(value, int) or isinstance(value, bool):
                continue
            if value > FFN and ("intermediate_size" in key or key in ("n_inner", "ffn_hidden_size", "ffn_dim", "d_ff")):
                setattr(child, key, FFN)
            elif key in EXPERT_COUNT_KEYS and value > EXPERTS:
                groups = getattr(child, "n_group", None) or 1
                count = max(EXPERTS, child_top_k + 1)
                setattr(child, key, min(value, ((count + groups - 1) // groups) * groups))
    return cfg, keep, n


# ----------------------------------------------------------------------------------------------------------------
# Numerics: the formulas the facts are tested with
# ----------------------------------------------------------------------------------------------------------------


def close(a, b, rtol=RTOL, atol=ATOL):
    if a is None or b is None or a.numel() != b.numel():
        return False
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return bool(torch.allclose(a, b, rtol=rtol, atol=atol))


def _as_heads(x, heads, head_dim):
    """x as [batch, heads, seq, head_dim], from [b, s, h*d], [b, s, h, d] or [b, h, s, d]."""
    if x is None or x.numel() != SEQ * heads * head_dim * x.shape[0]:
        return None
    b = x.shape[0]
    if x.dim() == 4 and x.shape[1] == heads and x.shape[2] == SEQ:
        return x
    return x.reshape(b, SEQ, heads, head_dim).transpose(1, 2)


def rope(x, inv_freq, layout, magnitude=1.0):
    """Rotates the first 2 * len(inv_freq) dimensions of x [b, h, seq, d] at positions 0.. seq-1.

    layout "halves" pairs (i, i + r/2), "adjacent" pairs (2i, 2i + 1), as the engine's RopeLayout."""
    half = inv_freq.numel()
    pos = torch.arange(x.shape[2], dtype=torch.float32)
    ang = pos[:, None] * inv_freq.float()[None, :]
    cos = (ang.cos() * magnitude)[None, None]
    sin = (ang.sin() * magnitude)[None, None]
    out = x.clone().float()
    if layout == "halves":
        a, b = x[..., :half].float(), x[..., half:2 * half].float()
        out[..., :half] = a * cos - b * sin
        out[..., half:2 * half] = a * sin + b * cos
    else:
        a, b = x[..., 0:2 * half:2].float(), x[..., 1:2 * half:2].float()
        out[..., 0:2 * half:2] = a * cos - b * sin
        out[..., 1:2 * half:2] = a * sin + b * cos
    return out


ACTIVATIONS = {
    "silu": lambda x: x * torch.sigmoid(x),
    "gelu_tanh": lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3))),
    "gelu": lambda x: 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0))),
    "gelu_quick": lambda x: x * torch.sigmoid(1.702 * x),
    "relu": torch.relu,
    "relu2": lambda x: torch.relu(x) ** 2,
}


def _eps_of(module, fallback):
    if module in RECORDER.norm_eps:
        return RECORDER.norm_eps[module]
    for k in ("variance_epsilon", "eps", "epsilon"):
        v = getattr(module, k, None)
        if isinstance(v, float):
            return v
    return fallback


def rms_offset(call, eps):
    """A measured scalar offset to the weight; accepted only if the entire RMSNorm output agrees."""
    x, y = call.input.float(), call.output.float()
    rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    valid = rms.abs() > 1e-4
    if not valid.any():
        return None
    offset = float((y / torch.where(valid, rms, torch.ones_like(rms)) - call.module.weight.detach().float())[valid].median())
    return offset if math.isfinite(offset) and close(rms * (call.module.weight.detach().float() + offset), y) else None


def norm_formula(call, eps_fallback):
    """How a norm maps input to output: (rms | rms_1p | rms_offset | layer | rms_noweight, eps), or None."""
    x, y = call.input, call.output
    if x is None or y is None or x.shape != y.shape:
        return None
    w = getattr(call.module, "weight", None)
    bias = getattr(call.module, "bias", None)
    if w is not None and tuple(w.shape) != tuple(x.shape[-w.dim():]):
        return None
    eps = _eps_of(call.module, eps_fallback)
    xf = x.float()
    rms = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if w is not None and w.dim() == 1 and w.numel() == x.shape[-1]:
        wf = w.detach().float()
        if close(rms * wf, y):
            return "rms", eps
        if close(rms * (1.0 + wf), y):
            return "rms_1p", eps
        if rms_offset(call, eps) is not None:
            return "rms_offset", eps
    elif w is None and close(rms, y):
        return "rms_noweight", eps
    mean = xf.mean(-1, keepdim=True)
    var = (xf - mean).pow(2).mean(-1, keepdim=True)
    ln = (xf - mean) * torch.rsqrt(var + eps)
    for formula, gain in (("layer", w.detach().float() if w is not None else 1.0),
                           ("layer_1p", 1.0 + w.detach().float() if w is not None else 1.0)):
        result = ln * gain
        if bias is not None and torch.is_tensor(bias):
            result = result + bias.detach().float()
        if close(result, y):
            return formula, eps
    return None


def norm_fact(call, eps):
    form = norm_formula(call, eps)
    fact = {"module": call.path, "formula": form[0] if form else None, "eps": form[1] if form else None,
            "weighted": getattr(call.module, "weight", None) is not None}
    if form and form[0] in ("rms", "rms_1p", "rms_offset"):
        fact["weight_offset"] = {"rms": 0.0, "rms_1p": 1.0}.get(form[0])
        if fact["weight_offset"] is None:
            fact["weight_offset"] = rms_offset(call, form[1])
    return fact


# ----------------------------------------------------------------------------------------------------------------
# Running the reduced model
# ----------------------------------------------------------------------------------------------------------------


def _randomize(model):
    """Random weights that keep activations of order 1, with norm weights away from 0 and 1 so that a norm
    multiplying by w and one multiplying by 1 + w give different outputs."""
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:
                if "norm" in name.lower() or "ln" in name.lower().split("."):
                    p.copy_(torch.rand(p.shape, generator=g) + 0.5)
                else:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.1)
            else:
                fan_in = p.shape[-1]
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(max(1, fan_in)))
        # Persistent router corrections are often buffers initialized to zero.
        # Give these declared selection biases a signal so the trace and tiny
        # checkpoint exercise their selection-only effect. RoPE and other
        # mathematical buffers keep the model's values.
        for name, b in model.named_buffers():
            if b.is_floating_point() and b.dim() == 1 and name.rsplit(".", 1)[-1] in (
                    "e_score_correction", "e_score_correction_bias", "expert_bias"):
                b.copy_(torch.randn(b.shape, generator=g) * 0.1)


def _decoder_layers(model, keep):
    best = None
    for path, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == keep and len(mod) > 0:
            if len({type(m) for m in mod}) > 2:
                continue
            size = sum(p.numel() for p in mod.parameters())
            if best is None or size > best[2]:
                best = (path, mod, size)
    if best is None:
        raise RuntimeError("no list of decoder blocks found in the model")
    return best[0], list(best[1])


@dataclass
class Trace:
    config: object  # full text configuration
    reduced: object  # the configuration that was traced
    n_layers: int  # blocks of the full model
    keep: int  # blocks traced
    model: nn.Module
    layers_path: str
    layers: list
    calls: list
    attn: list
    ids: torch.Tensor
    logits: torch.Tensor
    class_name: str
    clamps: list = field(default_factory=list)
    recurrences: list = field(default_factory=list)


class NativeAttentionTrace(TorchFunctionMode):
    """Observe eager matmul/softmax attention in implementations without AttentionInterface.

    No model classes are selected: a score product, its softmax and its value
    product must agree numerically. Scores are fitted to scaled QK plus a
    per-head linear position bias and a softmax-invariant constant per query.
    """
    def __init__(self, stack):
        super().__init__()
        self.stack, self.dots, self.probs = stack, {}, {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if not self.stack or not torch.is_tensor(out):
            return out
        module = self.stack[-1][1]
        if any(rec.module is module for rec in RECORDER.attn):
            return out  # AttentionInterface already recorded this invocation.
        name = func.__name__
        if name == "gelu":
            approximate = kwargs.get("approximate", args[1] if len(args) > 1 else "none")
            if approximate in ("none", "tanh"):
                source = args[0] if args else kwargs["input"]
                RECORDER.gelu_modes.append((module, "gelu" if approximate == "none" else "gelu_tanh",
                                           source.detach().clone(), out.detach().clone()))
        elif name == "layer_norm":
            RECORDER.norm_eps[module] = float(kwargs.get("eps", args[4] if len(args) > 4 else 1e-5))
        elif name in ("clamp", "clamp_"):
            lo = kwargs.get("min", args[1] if len(args) > 1 else None)
            hi = kwargs.get("max", args[2] if len(args) > 2 else None)
            if isinstance(lo, (float, int)) and isinstance(hi, (float, int)) and hi > 0 and lo == -hi:
                RECORDER.clamps.append((module, float(hi)))
        elif name in ("scaled_dot_product_attention",):
            q, k, v = args[:3]
            rec = AttnCall(RECORDER.next(), module, q.detach().clone(), k.detach().clone(), v.detach().clone(),
                           {"scaling": kwargs.get("scale") or q.shape[-1] ** -0.5,
                            "observed_sdpa": True,
                            "is_causal": kwargs.get("is_causal", args[5] if len(args) > 5 else False),
                            "attention_mask": (kwargs.get("attn_mask", args[3] if len(args) > 3 else None))})
            rec.output = out.transpose(1, 2).detach().clone()
            RECORDER.attn.append(rec)
        elif name in ("matmul", "bmm", "baddbmm") and (len(args) >= 2 or "batch1" in kwargs):
            a = kwargs["batch1"] if "batch1" in kwargs else args[-2]
            b = kwargs["batch2"] if "batch2" in kwargs else args[-1]
            if not torch.is_tensor(a) or not torch.is_tensor(b) or a.dim() < 3 or b.dim() < 3:
                return out
            pending = self.probs.get(module)
            if pending and a.shape[-2:] == (SEQ, SEQ) and b.shape[-2] == SEQ and close(a, pending[0]):
                rec = pending[1]
                rec.value = b.reshape(1, -1, SEQ, b.shape[-1]).detach().clone()
                rec.output = out.reshape(1, -1, SEQ, b.shape[-1]).transpose(1, 2).detach().clone()
                RECORDER.attn.append(rec)
                del self.probs[module]
            elif out.shape[-2:] == (SEQ, SEQ) and a.shape[-2] == SEQ and b.shape[-1] == SEQ:
                q = a.reshape(1, -1, SEQ, a.shape[-1]).detach().clone()
                k = b.transpose(-1, -2).reshape(1, -1, SEQ, b.shape[-2]).detach().clone()
                self.dots[module] = (q, k)
        elif name == "softmax" and out.shape[-2:] == (SEQ, SEQ) and module in self.dots:
            q, k = self.dots.pop(module)
            score = args[0].reshape(1, -1, SEQ, SEQ).double()
            dots = (q.double() @ k.double().transpose(-1, -2))
            scales, slopes = [], []
            valid = True
            for h in range(score.shape[1]):
                # Differences to column zero remove the constant per query.
                ds, dq, positions = [], [], []
                for t in range(1, SEQ):
                    for j in range(1, t + 1):
                        ds.append(score[0, h, t, j] - score[0, h, t, 0])
                        dq.append(dots[0, h, t, j] - dots[0, h, t, 0])
                        positions.append(j)
                design = torch.stack((torch.stack(dq), torch.tensor(positions, dtype=torch.float64)), dim=1)
                target = torch.stack(ds)
                fit = torch.linalg.lstsq(design, target).solution
                valid = valid and close(design @ fit, target, rtol=2e-4, atol=2e-5)
                scales.append(float(fit[0]))
                slopes.append(float(fit[1]))
            if valid and max(scales) - min(scales) < 1e-5:
                rec = AttnCall(RECORDER.next(), module, q, k, None,
                               {"scaling": sum(scales) / len(scales), "alibi_slopes": slopes})
                self.probs[module] = (out.detach().clone(), rec)
        return out


def run_trace(model_id, revision=None, trust_remote_code=False):
    """Builds the reduced model, runs it once and returns the records."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # configuration and kernel notices are not facts about the model
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    reduced, keep, n = reduced_config(config)
    canonical_rope_config(reduced)
    _register_attention()
    torch.manual_seed(0)
    try:
        model = AutoModelForCausalLM.from_config(reduced, trust_remote_code=trust_remote_code, dtype=torch.float32,
                                                 attn_implementation=ATTN_IMPL)
    except (ValueError, KeyError) as error:
        if ATTN_IMPL not in str(error):
            raise
        # Older official classes predate AttentionInterface. Observe their
        # native SDPA call without replacing any forward computation.
        reduced._attn_implementation = "sdpa"
        torch.manual_seed(0)
        try:
            model = AutoModelForCausalLM.from_config(reduced, trust_remote_code=trust_remote_code, dtype=torch.float32,
                                                     attn_implementation="sdpa")
        except (ValueError, KeyError) as unsupported:
            if not any(name in str(unsupported) for name in ("sdpa", "scaled_dot_product_attention")):
                raise
            # Some classes call functional SDPA directly without declaring
            # transformers' backend capability. Their eager path is observed
            # too, including paired manual QK and probabilities/V products.
            reduced._attn_implementation = "eager"
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_config(reduced, trust_remote_code=trust_remote_code, dtype=torch.float32,
                                                     attn_implementation="eager")
    model.eval()
    _randomize(model)
    layers_path, layers = _decoder_layers(model, keep)

    # Record algorithm operands as well as module boundaries. These functions
    # are operations (shared by several architectures), not model-name rules.
    originals = []
    namespaces = {id(type(m).forward.__globals__): type(m).forward.__globals__
                  for m in model.modules() if hasattr(type(m).forward, "__globals__")}
    for namespace in namespaces.values():
        for name, fn in list(namespace.items()):
            if not callable(fn) or "chunk" not in name or not any(t in name for t in ("delta_rule", "delta_attention")):
                continue
            signature = inspect.signature(fn)
            if not {"query", "key", "value"}.issubset(signature.parameters):
                continue
            @functools.wraps(fn)
            def recorded(*args, _fn=fn, _sig=signature, **kwargs):
                bound = _sig.bind(*args, **kwargs).arguments
                values = {k: v.detach().clone() for k, v in bound.items() if torch.is_tensor(v)}
                RECORDER.recurrences.append(values)
                return _fn(*args, **kwargs)
            originals.append((namespace, name, fn))
            namespace[name] = recorded
    RECORDER.recurrences.clear()
    RECORDER.calls.clear()
    RECORDER.attn.clear()
    RECORDER.clamps.clear()
    RECORDER.norm_eps.clear()
    RECORDER.gelu_modes.clear()
    handles = []
    stack = []
    for path, mod in model.named_modules():
        if mod is model:
            continue

        def hook(module, args, kwargs, output, path=path):
            RECORDER.calls.append(Call(RECORDER.next(), path, module,
                                       [a.detach().clone() if torch.is_tensor(a) else a for a in args],
                                       {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()},
                                       _tensors(output)))
            stack.pop()

        def enter(module, args, path=path):
            stack.append((path, module))

        handles.append(mod.register_forward_pre_hook(enter))
        handles.append(mod.register_forward_hook(hook, with_kwargs=True))
    ids = torch.tensor([[(7 + 37 * i) % VOCAB for i in range(SEQ)]])
    try:
        try:
            with torch.no_grad(), NativeAttentionTrace(stack):
                out = model(input_ids=ids, use_cache=False)
        except KeyError as error:
            if error.args != (ATTN_IMPL,):
                raise
            # Some implementations expose only their local eager/SDPA map.
            # Observe their eager operations through the same numerical trace.
            model.set_attn_implementation("eager")
            stack.clear()
            RECORDER.calls.clear(); RECORDER.attn.clear(); RECORDER.clamps.clear(); RECORDER.norm_eps.clear(); RECORDER.gelu_modes.clear()
            with torch.no_grad(), NativeAttentionTrace(stack):
                out = model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
        for namespace, name, fn in originals:
            namespace[name] = fn
    logits = out.logits.detach() if hasattr(out, "logits") else _tensors(out)[0]
    return Trace(text, reduced, n, keep, model, layers_path, layers, list(RECORDER.calls), list(RECORDER.attn), ids,
                 logits, type(model).__name__, list(RECORDER.clamps), list(RECORDER.recurrences))


# ----------------------------------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------------------------------


class Facts(dict):
    """Plain JSON-able facts; "unknown" lists what could not be established, "absent" what the reference lacks
    that the engine requires (both with reasons)."""


def _is_descendant(path, parent):
    return path.startswith(parent + ".")


def _direct_children(calls, parent):
    depth = parent.count(".") + 1
    return [c for c in calls if _is_descendant(c.path, parent) and c.path.count(".") == depth]


def _linears(calls, parent):
    from transformers.pytorch_utils import Conv1D
    return [c for c in calls if _is_descendant(c.path, parent) and isinstance(c.module, (nn.Linear, Conv1D))]


def analyze_linear_mixer(call):
    """A unique Linear must explain the entire mixer output, including its bias.

    Explicit child.forward(x) calls bypass module hooks. Enumerating parameters
    and recomputing the full observed output also covers those implementations;
    unused duplicate candidates, extra transformations and shape changes remain
    unknown. No class name or parameter spelling decides the computation.
    """
    if call.input is None or call.output is None or call.input.dim() == 0:
        return None
    matches = []
    for name, module in call.module.named_modules():
        if not isinstance(module, nn.Linear) or module.in_features != call.input.shape[-1]:
            continue
        bias = module.bias.detach().float() if module.bias is not None else None
        expected = torch.nn.functional.linear(call.input.float(), module.weight.detach().float(), bias)
        if expected.shape == call.output.shape and close(expected, call.output):
            matches.append(call.path + ("." + name if name else ""))
    if len(matches) == 1:
        return Facts(kind="linear", projection=matches[0], unknown=[])
    return None


def _norm_like(call):
    w = getattr(call.module, "weight", None)
    return ((w is None or (torch.is_tensor(w) and w.dim() in (1, 2))) and call.input is not None
            and call.output is not None and call.input.shape == call.output.shape
            and not isinstance(call.module, (nn.Linear, nn.Embedding))
            and ((w is not None and w.dim() == 1) or norm_formula(call, 1e-5) is not None))


def _inv_freqs(model):
    """Every RoPE frequency table the model holds: (name, inv_freq, attention_scaling)."""
    out = []
    for path, mod in model.named_modules():
        for bname, buf in mod.named_buffers(recurse=False):
            if bname.endswith("inv_freq") and "original" not in bname:
                prefix = bname[: -len("inv_freq")]
                scaling = getattr(mod, prefix + "attention_scaling", getattr(mod, "concentration", None))
                if not isinstance(scaling, (int, float)):
                    scaling = 1.0
                out.append((f"{path}.{bname}" if path else bname, buf.detach().float().clone(), float(scaling)))
    return out


def _rotations(x, freqs):
    """Every rotation of x [b, h, s, d] the engine could perform: (description, tensor)."""
    yield {"rope": None}, x.float()
    for name, inv, mag in freqs:
        if 2 * inv.numel() > x.shape[-1]:
            continue
        for layout in ("halves", "adjacent"):
            yield {"rope": {"table": name, "layout": layout, "dims": 2 * inv.numel(), "magnitude": mag}}, rope(
                x, inv, layout, mag)


def _projection_views(linear_calls, heads, head_dim, total_heads):
    """Per-head views [b, heads, s, d] of linear outputs that can hold `heads` heads of `head_dim`:
    a plain projection, a slice of a fused [q | k | v] one, or the first half of each head of [q | gate]."""
    views = []
    for c in linear_calls:
        y = c.output
        if y is None or y.dim() != 3 or y.shape[1] != SEQ:
            continue
        b, _, f = y.shape
        if f == heads * head_dim:
            views.append((c, "plain", _as_heads(y, heads, head_dim)))
        if f == 2 * heads * head_dim:
            per = y.reshape(b, SEQ, heads, 2 * head_dim)
            views.append((c, "per-head [x | gate]", per[..., :head_dim].transpose(1, 2)))
        if total_heads and f == total_heads * head_dim and f != heads * head_dim:
            views.append((c, "fused", y))  # sliced by the caller, which knows the offsets
    return views


def _match_rotated(target, raw_views, norm_calls, freqs, heads, head_dim, eps):
    """How target [b, heads, s, d] (what the attention received) is made from a projection: returns a dict with
    the projection, its layout, the rotation, and where a norm sits (with its formula), or None."""
    for call, how, raw, rows in raw_views:
        if raw is None:
            continue
        for rot, r in _rotations(raw, freqs):
            if close(r, target):
                return {"projection": call.path, "projection_layout": how, "projection_rows": rows,
                        **rot, "norm": None}
        for n in norm_calls:
            nin = n.input
            nout = n.output
            if nin is None or nout is None or nin.numel() != raw.numel():
                continue
            w = getattr(n.module, "weight", None)
            span = int(n.input.shape[-1])
            norm = {**norm_fact(n, eps), "span": span}
            # Norm before the rotation: the norm reads the projection, the rotation reads the norm.
            if close(_as_heads(nin, heads, head_dim), raw):
                normed = _as_heads(nout, heads, head_dim)
                for rot, r in _rotations(normed, freqs):
                    if close(r, target):
                        return {"projection": call.path, "projection_layout": how, "projection_rows": rows, **rot,
                                "norm": {**norm, "order": "before_rope"}}
            # Norm after the rotation: the norm reads the rotated projection and gives the target.
            if close(_as_heads(nout, heads, head_dim), target):
                for rot, r in _rotations(raw, freqs):
                    if close(r, _as_heads(nin, heads, head_dim)):
                        return {"projection": call.path, "projection_layout": how, "projection_rows": rows, **rot,
                                "norm": {**norm, "order": "after_rope"}}
        # One LayerNorm module per head (rather than one shared weight).
        per_head = []
        for h in range(heads):
            match = next((n for n in norm_calls if n.input is not None and
                          n.input.numel() == raw[:, h:h + 1].numel() and
                          close(_as_heads(n.input, 1, head_dim), raw[:, h:h + 1])), None)
            if match is None:
                break
            per_head.append(match)
        if len(per_head) == heads:
            forms = [norm_formula(n, eps) for n in per_head]
            if forms[0] is not None and all(form == forms[0] for form in forms):
                normed = torch.cat([_as_heads(n.output, 1, head_dim) for n in per_head], dim=1)
                for rot, r in _rotations(normed, freqs):
                    if close(r, target):
                        return {"projection": call.path, "projection_layout": how, "projection_rows": rows, **rot,
                                "norm": {"module": per_head[0].path, "modules": [n.path for n in per_head],
                                         "span": head_dim, "formula": forms[0][0], "eps": forms[0][1],
                                         "order": "before_rope"}}
    return None


def analyze_attention(trace, mixer_path, block_calls, freqs, eps):
    att = [a for a in trace.attn if any(c.module is a.module for c in block_calls if c.path == mixer_path
                                         or _is_descendant(c.path, mixer_path))]
    if len(att) != 1:
        return None
    a = att[0]
    f = Facts(kind="attention")
    _, n_q, s, hd = a.query.shape
    n_kv = a.key.shape[1]
    key_states, value_states = a.key, a.value
    # Eager implementations may expand GQA keys/values before the observed
    # score product. Recover their logical groups only when every repeated
    # key AND value agrees bit for bit; projection matching then proves it.
    if a.value.shape[1] == n_kv:
        for groups in range(1, n_kv):
            if n_kv % groups:
                continue
            keys = a.key.reshape(a.key.shape[0], groups, n_kv // groups, s, hd)
            values = a.value.reshape(a.value.shape[0], groups, n_kv // groups, s, a.value.shape[-1])
            if torch.equal(keys, keys[:, :, :1].expand_as(keys)) and torch.equal(values, values[:, :, :1].expand_as(values)):
                key_states, value_states = keys[:, :, 0], values[:, :, 0]
                n_kv = groups
                break
    hd_v = a.value.shape[-1]
    f.update(heads=n_q, kv_heads=n_kv, head_dim=hd, value_head_dim=hd_v, unknown=[])
    linears = _linears(block_calls, mixer_path)
    norms = [c for c in block_calls if _is_descendant(c.path, mixer_path) and _norm_like(c)]
    fused_total = n_q + 2 * n_kv
    caps = {cap for module, cap in trace.clamps if any(c.module is module and
            (c.path == mixer_path or _is_descendant(c.path, mixer_path)) for c in block_calls)}
    cap = next(iter(caps)) if len(caps) == 1 else None
    f["clamp_kqv"] = cap or 0.0

    def views_for(heads, offset_heads):
        views = []
        for call, how, v in _projection_views(linears, heads, hd, fused_total):
            if how == "fused":
                y = v.clamp(-cap, cap) if cap else v
                rows = list(range(offset_heads * hd, (offset_heads + heads) * hd))
                views.append((call, f"fused [q | k | v] at head {offset_heads}",
                              _as_heads(y[..., rows], heads, hd), rows))
                if n_q % n_kv == 0:
                    # Packed GQA stores each KV group's queries, key and value
                    # together. Verify this view against the actual attention
                    # tensors, just as for the contiguous Q/K/V candidate.
                    group = n_q // n_kv
                    rows = []
                    for g in range(n_kv):
                        start = g * (group + 2)
                        slots = range(start, start + group) if offset_heads == 0 else [start + group + (offset_heads != n_q)]
                        rows.extend(row for slot in slots for row in range(slot * hd, (slot + 1) * hd))
                    views.append((call, "fused groups [queries | key | value]",
                                  _as_heads(y[..., rows], heads, hd), rows))
            else:
                views.append((call, how, v.clamp(-cap, cap) if cap else v, list(range(heads * hd))))
        return views

    key = _match_rotated(key_states, views_for(n_kv, n_q), norms, freqs, n_kv, hd, eps)
    query_target = a.query
    params = getattr(trace.reduced, "rope_parameters", None) or {}
    beta, length = params.get("llama_4_scaling_beta"), params.get("original_max_position_embeddings")
    if beta is not None and length:
        # A position-dependent query temperature is separate from the rotary transform. Remove precisely
        # the configuration's factor before matching the projection/norm/rotation, so a mismatch still fails.
        positions = torch.arange(s, device=query_target.device)
        temperature = 1.0 + float(beta) * torch.log1p(torch.floor(positions / int(length)))
        query_target = query_target / temperature[None, None, :, None]
        f["temperature"] = {"beta": float(beta), "length": int(length)}
    query = _match_rotated(query_target, views_for(n_q, 0), norms, freqs, n_q, hd, eps)
    value = None
    for call, how, v, rows in views_for(n_kv, n_q + n_kv) if hd_v == hd else []:
        if close(v, value_states):
            value = {"projection": call.path, "projection_layout": how, "projection_rows": rows, "rms_norm": False}
            break
        # Some architectures normalize values per head without a learned
        # weight. Establish that formula from the projection and observed
        # values, rather than from a family or module name.
        normalized = v * torch.rsqrt(v.float().square().mean(-1, keepdim=True) + eps)
        if close(normalized, value_states):
            value = {"projection": call.path, "projection_layout": how, "projection_rows": rows, "rms_norm": True}
            break
    if key is None and value is None:
        # Shared KV is observed as the exact earlier attention's K/V, with
        # no local projections. Reuse that source's established formula.
        for previous in trace.attn:
            if previous is a:
                break
            if close(previous.key, a.key) and close(previous.value, a.value):
                source = next((c.path for c in trace.calls if c.module is previous.module), None)
                if source:
                    prior = analyze_attention(trace, source, trace.calls, freqs, eps)
                    if prior and prior.get("key") and prior.get("value"):
                        key, value = prior["key"], prior["value"]
                        f["shared_kv_source"] = source
                        break
    f["key"] = key
    f["query"] = query
    f["value"] = value
    if key is None:
        f["unknown"].append("keys are not a rotation of a projection of the block input (with or without a norm)")
    if query is None:
        f["unknown"].append("queries are not a rotation of a projection of the block input (with or without a norm)")
    if value is None:
        f["unknown"].append("values are not a projection of the block input")
    scaling = a.kwargs.get("scaling")
    f["scale"] = float(scaling) if isinstance(scaling, (int, float)) else hd ** -0.5
    f["window"] = a.kwargs.get("sliding_window") if isinstance(a.kwargs.get("sliding_window"), int) else None
    softcap = a.kwargs.get("softcap")
    f["softcap"] = float(softcap) if isinstance(softcap, (int, float)) and softcap else None
    f["sinks"] = torch.is_tensor(a.kwargs.get("s_aux")) or torch.is_tensor(getattr(a.module, "sinks", None))
    f["causal"] = bool(getattr(a.module, "is_causal", True))
    slopes = a.kwargs.get("alibi_slopes")
    f["alibi_slopes"] = slopes if slopes and max(map(abs, slopes)) > 1e-6 else None
    if a.kwargs.get("observed_sdpa"):
        if a.kwargs.get("manual_scores_verified") is False:
            f["unknown"].append("native matmul attention probabilities are not the verified scaled Q/K softmax")
        mask = a.kwargs.get("attention_mask")
        f["causal"] = bool(a.kwargs["is_causal"])
        if mask is not None:
            if mask.shape[-2:] != (s, s):
                f["unknown"].append("native SDPA mask does not span the traced sequence")
            else:
                if mask.dtype == torch.bool:
                    allowed = mask
                else:
                    blocked = torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min)
                    if not torch.all((mask == 0) | blocked):
                        f["unknown"].append("native SDPA uses an additive score bias beyond a causal mask")
                    allowed = ~blocked
                masks = allowed.reshape(-1, s, s)
                positions = torch.arange(s, device=mask.device)
                causal = positions[None, :] <= positions[:, None]
                if a.kwargs["is_causal"]:
                    masks = masks & causal
                window = int(masks[0, -1].count_nonzero())
                expected = causal & (positions[None, :] > positions[:, None] - window)
                if window > 0 and torch.all(masks == expected):
                    f["causal"] = True
                    f["window"] = window if window < s else None
                else:
                    f["unknown"].append("native SDPA mask is not causal or a fixed causal window")
    # Output: o_proj reads the attention output, possibly gated by sigmoid of the second half of [q | gate].
    att_out = a.output.reshape(1, SEQ, n_q * hd_v) if a.output is not None else None
    later = [c for c in linears if c.order > a.order]
    f["output_gate"] = None
    f["output"] = None
    for c in later:
        if close(c.input, att_out):
            f["output"] = c.path
            break
        if query is not None and query.get("projection_layout") == "per-head [x | gate]":
            qcall = next(x for x in linears if x.path == query["projection"])
            gate = qcall.output.reshape(1, SEQ, n_q, 2 * hd)[..., hd:].reshape(1, SEQ, n_q * hd)
            if close(c.input, att_out * torch.sigmoid(gate)):
                f["output"] = c.path
                f["output_gate"] = "sigmoid of the second half of each query head's projection"
                break
    if f["output"] is None and att_out is not None:
        for c in later:
            for gate in linears:
                values = gate.output
                if values.numel() == SEQ * n_q and values.shape[-1] == n_q:
                    values = values.reshape(1, SEQ, n_q, 1).expand(1, SEQ, n_q, hd_v).reshape(att_out.shape)
                if gate is not c and values.numel() == att_out.numel() and close(
                        c.input, att_out * torch.sigmoid(values.reshape(att_out.shape))):
                    f["output"] = c.path
                    f["output_gate"] = "sigmoid of a separate gate projection"
                    break
            if f["output"]:
                break
    if f["output"] is None:
        f["unknown"].append("the output projection does not read the attention output (or its gated form)")
    return f


def analyze_dense_ffn(ffn_call, block_calls):
    """A dense FFN: down(act(gate) * up), down(act(up)), or a fused [gate | up] projection."""
    lin = [c for c in _linears(block_calls, ffn_call.path)]
    x, y = ffn_call.input, ffn_call.output
    residuals = [a for a in list(ffn_call.args[1:]) + list(ffn_call.kwargs.values())
                 if torch.is_tensor(a) and a.shape == y.shape]
    down = next((c for c in lin if c.output.shape == y.shape and
                 (close(c.output, y) or any(close(c.output + r, y) for r in residuals))), None)
    if down is None:
        return Facts(kind="dense", unknown=["the FFN's output is not one projection's output"])
    first = [c for c in lin if c is not down and close(c.input, x)]
    d_in = down.input.float()
    limit = float(getattr(ffn_call.module, "limit", 0.0) or 0.0)
    limit = limit if math.isfinite(limit) and limit > 0 else 0.0
    def activate(act, gate, up):
        return act(gate).clamp(max=limit) * up.clamp(-limit, limit) if limit else act(gate) * up
    modules = {ffn_call.module} | {c.module for c in block_calls if _is_descendant(c.path, ffn_call.path)}

    def activation_allowed(name, source, multiply=None):
        # Tiny norm scales can make erf/tanh GELU numerically indistinguishable.
        # An observed GELU call linked exactly to the down input establishes
        # its mode independently of that approximation tolerance.
        modes = set()
        for module, mode, x, y in RECORDER.gelu_modes:
            if module not in modules or not torch.equal(x.float(), source.float()):
                continue
            linked = y.float() if multiply is None else y.float() * multiply.float()
            if torch.equal(linked, d_in):
                modes.add(mode)
        return not modes or name in modes
    for c in first:
        for activation in block_calls:
            m = activation.module
            if not all(hasattr(m, attr) for attr in ("alpha_p", "alpha_n", "beta", "eps")):
                continue
            if not close(c.output, activation.input) or not close(activation.output, d_in):
                continue
            x = activation.input.float()
            ap, an = nn.functional.softplus(m.alpha_p.float()), m.beta.float() + nn.functional.softplus(m.alpha_n.float())
            expected = torch.where(x > 0, ap * x * x + m.beta * x,
                                   (torch.expm1(torch.minimum(x, m.eps)) - x) * an + m.beta * x)
            if close(expected, d_in):
                return Facts(kind="dense", gated=False, activation="xielu", up=c.path, down=down.path,
                             activation_module=activation.path,
                             scalars={k: float(getattr(m, k).detach().reshape(-1)[0]) for k in
                                      ("alpha_p", "alpha_n", "beta", "eps")}, unknown=[])
    for act_name, act in ACTIVATIONS.items():
        for g in first:
            for u in first:
                gate = g.output.float()
                sparsity = getattr(ffn_call.module, "activation_sparsity", 0.0)
                if sparsity > 0.0:
                    multiplier = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(sparsity))
                    gate = torch.relu(gate - (gate.mean(-1, keepdim=True) +
                                             gate.std(-1, keepdim=True, correction=0) * multiplier))
                if g is not u and activation_allowed(act_name, gate, u.output) and close(activate(act, gate, u.output.float()), d_in):
                    return Facts(kind="dense", gated=True, fused_gate_up=False, activation=act_name,
                                 gate=g.path, up=u.path, down=down.path,
                                 activation_sparsity=float(sparsity), clamp=limit,
                                 biases=any(getattr(c.module, "bias", None) is not None for c in (g, u, down)),
                                 unknown=[])
        for c in first:
            o = c.output.float()
            if o.shape[-1] == 2 * d_in.shape[-1]:
                h = d_in.shape[-1]
                for order, (gp, up) in (("gate_first", (o[..., :h], o[..., h:])), ("up_first", (o[..., h:], o[..., :h]))):
                    if activation_allowed(act_name, gp, up) and close(activate(act, gp, up), d_in):
                        return Facts(kind="dense", gated=True, fused_gate_up=True, fused_order=order,
                                     activation=act_name, gate=c.path, up=c.path, down=down.path, clamp=limit,
                                     biases=any(getattr(m.module, "bias", None) is not None for m in (c, down)),
                                     unknown=[])
            if activation_allowed(act_name, o) and close(act(o), d_in):
                return Facts(kind="dense", gated=False, activation=act_name, up=c.path, down=down.path, unknown=[])
    return Facts(kind="dense", unknown=["the FFN is not down(act(gate) * up) or down(act(up)) for a known activation"])


def analyze_moe(ffn_call, block_calls, cfg):
    """Routing of a mixture of experts, recomputed from the router's logits and what the experts were given."""
    f = Facts(kind="moe", unknown=[])
    inside = [c for c in block_calls if _is_descendant(c.path, ffn_call.path)]
    n_tokens = SEQ
    experts_call = None
    for c in inside:
        ints = [a for a in list(c.args) + list(c.kwargs.values())
                if torch.is_tensor(a) and not a.is_floating_point() and a.dim() == 2 and a.shape[0] == n_tokens]
        floats = [a for a in list(c.args) + list(c.kwargs.values())
                  if torch.is_tensor(a) and a.is_floating_point() and a.dim() == 2 and a.shape[0] == n_tokens
                  and ints and a.shape == ints[0].shape]
        if ints and floats:
            experts_call = (c, ints[0], floats[0])
            break
    if experts_call is None:
        f["unknown"].append("no experts call with routed indices and weights was observed")
        return f
    ecall, idx, wts = experts_call
    k = idx.shape[1]
    # The expert count: the leading dimension of the experts' 3-D weights, else the configuration's.
    n_exp = next((p.shape[0] for p in ecall.module.parameters() if p.dim() == 3), None)
    if n_exp is None:
        configs = [cfg] + [child for child in vars(cfg).values() if hasattr(child, "to_dict")]
        n_exp = max([getattr(config, key) for config in configs for key in EXPERT_COUNT_KEYS
                     if isinstance(getattr(config, key, None), int)] or [0])
    # The router: what produced one logit per expert and token from the block input, before the experts ran.
    router = None
    for c in inside:
        if c.order >= ecall.order:
            break
        if not close(c.input.reshape(n_tokens, -1) if c.input is not None and c.input.numel() % n_tokens == 0
                     else None, ffn_call.input.reshape(n_tokens, -1)):
            continue
        for t in c.outputs:
            if t.is_floating_point() and t.dim() >= 2 and t.shape[-1] == n_exp and t.numel() // n_exp == n_tokens:
                router = (c, t.reshape(n_tokens, -1).float())
                break
        if router is not None:
            break
    if router is None:
        f["unknown"].append("no router logits were observed")
        return f
    rcall, logits = router
    f.update(experts=n_exp, top_k=k, router=rcall.path, experts_module=ecall.path)
    f["router_bias"] = torch.is_tensor(getattr(rcall.module, "bias", None))
    biases = []
    for name, t in list(ffn_call.module.named_parameters()) + list(ffn_call.module.named_buffers()):
        if t.numel() == n_exp and t.squeeze().dim() == 1:
            biases.append((name, t.detach().float().reshape(-1)))
    found = None
    for gating in ("softmax", "sigmoid"):
        probs = torch.softmax(logits, -1) if gating == "softmax" else torch.sigmoid(logits)
        for bias_name, bias in [(None, None)] + biases:
            sel = probs + (bias if bias is not None else 0.0)
            top = torch.topk(sel, k, dim=-1).indices
            if not torch.equal(torch.sort(top, -1).values, torch.sort(idx.long(), -1).values):
                continue
            chosen = torch.gather(probs, -1, idx.long())
            for norm in (False, True):
                w = chosen / chosen.sum(-1, keepdim=True) if norm else chosen
                ratio = (wts.float() / w)
                scale = float(ratio.mean())
                if close(w * scale, wts.float()):
                    found = {"gating": gating, "selection_bias": bias_name, "norm_topk": norm, "scale": scale}
                    break
            if found:
                break
        if found:
            break
    if found is None:
        f["unknown"].append("routing is not top-k over softmax or sigmoid probabilities (with an optional "
                            "selection bias, renormalization and scale), e.g. group-limited routing")
        return f
    f.update(found)
    # Shared expert: a sub-module reading the block input whose output is added (gated or not) to the experts'.
    y = ffn_call.output
    e_out = ecall.output.reshape(y.shape) if ecall.output is not None and ecall.output.numel() == y.numel() else None
    f["shared_expert"] = None
    if e_out is not None and not close(e_out, y):
        children = _direct_children(block_calls, ffn_call.path)
        for c in children:
            if c is ecall or c.path == rcall.path or c.output is None or c.output.numel() != y.numel():
                continue
            s = c.output.reshape(y.shape)
            if close(e_out + s, y):
                f["shared_expert"] = {"module": c.path, "gate": None}
                break
            for g in children:
                if g.output is not None and g.output.shape[-1] == 1 and close(g.input, ffn_call.input):
                    if close(e_out + torch.sigmoid(g.output.reshape(*y.shape[:-1], 1)) * s, y):
                        f["shared_expert"] = {"module": c.path, "gate": g.path + " (sigmoid)"}
                        break
            if f["shared_expert"]:
                break
        if f["shared_expert"] is None:
            f["unknown"].append("the block output is not the experts' output plus a (gated) shared expert")
    # Recompute fused experts numerically. Both projection orientations and
    # contiguous/interleaved gate-up layouts occur in transformers. Biases are
    # part of each projection, before activation and before routing weights.
    params = dict(ecall.module.named_parameters(recurse=False))
    gu, dn = params.get("gate_up_proj"), params.get("down_proj")
    gb, db = params.get("gate_up_proj_bias"), params.get("down_proj_bias")
    f["expert_activation"] = None
    if gu is not None and dn is not None and gu.dim() == 3 and dn.dim() == 3:
        x = ecall.input.reshape(n_tokens, -1).float()
        target = ecall.output.reshape(n_tokens, -1).float()
        limit = float(getattr(ecall.module, "limit", 0.0) or 0.0)
        limit = limit if math.isfinite(limit) and limit > 0 and not hasattr(ecall.module, "alpha") else 0.0
        acts = {name: (lambda g, u, act=act: act(g).clamp(max=limit) * u.clamp(-limit, limit)
                       if limit else act(g) * u) for name, act in ACTIVATIONS.items()}
        # These constants define SwigluOai; another clamp/alpha needs a new
        # engine contract, so never silently describe it as this activation.
        if getattr(ecall.module, "alpha", None) == 1.702 and getattr(ecall.module, "limit", None) == 7.0:
            acts["swiglu_oai"] = lambda g, u: (u.clamp(-7, 7) + 1) * (g.clamp(max=7) *
                                                                           torch.sigmoid(1.702 * g.clamp(max=7)))
        for transposed in (False, True):
            gate_up = gu.float().transpose(-1, -2) if transposed else gu.float()
            down = dn.float().transpose(-1, -2) if transposed else dn.float()
            if gate_up.shape[-1] != x.shape[-1] or down.shape[1] != target.shape[-1] or gate_up.shape[1] != 2 * down.shape[2]:
                continue
            ff = down.shape[2]
            for interleaved in (False, True):
                for name, act in acts.items():
                    out = torch.zeros_like(target)
                    for t in range(n_tokens):
                        for j in range(k):
                            e = int(idx[t, j])
                            h = gate_up[e] @ x[t]
                            if gb is not None:
                                h = h + gb[e].float()
                            gate, up = (h[::2], h[1::2]) if interleaved else (h[:ff], h[ff:])
                            y = down[e] @ act(gate, up)
                            if db is not None:
                                y = y + db[e].float()
                            out[t] += wts[t, j].float() * y
                    if close(out, target):
                        f["expert_activation"] = name
                        f["expert_biases"] = gb is not None or db is not None
                        f["expert_clamp"] = limit
                        break
                if f["expert_activation"]:
                    break
            if f["expert_activation"]:
                break
    if f["expert_activation"] is None:
        # Eager experts can pass their three weight slices as arguments to a
        # shared module instead of holding fused 3-D parameters on the caller.
        expert_calls = [c for c in inside if _is_descendant(c.path, ecall.path) and len(c.args) == 4
                        and all(torch.is_tensor(w) and w.dim() == 2 for w in c.args)]
        for name, act in ACTIVATIONS.items():
            matched = bool(expert_calls)
            for c in expert_calls:
                x, gate, up, down = (w.float() for w in c.args)
                if gate.shape != up.shape or x.shape[-1] != gate.shape[-1]:
                    matched = False
                    break
                h = act(x @ gate.T) * (x @ up.T)
                result = h @ down if h.shape[-1] == down.shape[0] else h @ down.T
                if not close(result, c.output):
                    matched = False
                    break
            if matched:
                f["expert_activation"] = name
                break
    if f["expert_activation"] is None:
        f["unknown"].append("the experts' activation could not be recomputed from their weights and inputs")
    if f.get("shared_expert"):
        shared = next((c for c in inside if c.path == f["shared_expert"]["module"]), None)
        limit = float(getattr(shared.module, "limit", 0.0) or 0.0) if shared else 0.0
        f["shared_expert"]["clamp"] = limit if math.isfinite(limit) and limit > 0 else 0.0
    return f


def _input_embedding(trace, first_block):
    """Use the public accessor, or one unambiguous token lookup observed before the first block."""
    try:
        embedding = trace.model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        embedding = None
    if embedding is not None:
        return embedding
    boundary = min(c.order for c in trace.calls if c.path == first_block.path or
                   c.path.startswith(first_block.path + "."))
    candidates = {id(c.module): c.module for c in trace.calls
                  if c.order < boundary and isinstance(c.module, nn.Embedding) and
                  c.input is not None and torch.equal(c.input, trace.ids) and c.output is not None and
                  first_block.input is not None and c.output.shape == first_block.input.shape}
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _output_projection(trace, last_block):
    """Use the public accessor, or a unique linear call after the blocks with the logits' shape."""
    try:
        projection = trace.model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        projection = None
    if projection is not None:
        return next((c for c in trace.calls if c.module is projection), None)
    candidates = [c for c in trace.calls if c.order > last_block.order and isinstance(c.module, nn.Linear) and
                  c.output is not None and c.output.shape == trace.logits.shape]
    return candidates[0] if len(candidates) == 1 else None


def analyze(trace):
    """All facts from a trace (see the module docstring)."""
    calls = trace.calls
    eps = float(getattr(trace.config, "rms_norm_eps", None) or getattr(trace.config, "layer_norm_eps", 1e-6) or 1e-6)
    freqs = _inv_freqs(trace.model)
    facts = Facts(model_class=trace.class_name, blocks=trace.n_layers, traced_blocks=trace.keep, unknown=[],
                  rope_tables=[{"name": n, "dims": 2 * inv.numel(), "inv_freq": inv.tolist(), "magnitude": m}
                               for n, inv, m in freqs])
    layer_paths = [f"{trace.layers_path}.{i}" for i in range(trace.keep)]
    layer_calls = []
    for p in layer_paths:
        own = [c for c in calls if c.path == p]
        if len(own) != 1:
            facts["unknown"].append(f"block {p} ran {len(own)} times")
            return facts
        layer_calls.append(own[0])

    # Embeddings into block 0, against the rows of the embedding table (what a GGUF stores as token_embd): an
    # embedding module may scale its own output.
    emb_mod = _input_embedding(trace, layer_calls[0])
    h0 = layer_calls[0].input
    streams = h0 is not None and h0.dim() == 4
    if streams:
        h0 = h0[0]  # the embedding stream; projections make the others
    facts["embedding_scale"] = None
    if emb_mod is None:
        facts["unknown"].append("the reference's input embedding table cannot be identified uniquely from its calls")
    if emb_mod is not None and h0 is not None:
        e = emb_mod.weight.detach()[trace.ids[0]].float().reshape(h0.shape)
        if close(e, h0):
            facts["embedding_scale"] = 1.0
        else:
            ratio = h0.float() / torch.where(e.abs() > 1e-6, e, torch.ones_like(e))
            c = float(ratio.flatten()[e.abs().flatten() > 1e-3].median())
            facts["embedding_scale"] = c if close(e * c, h0) else None
            if facts["embedding_scale"] is None:
                for call in calls:
                    if call.order >= layer_calls[0].order:
                        continue
                    if isinstance(call.module, nn.Embedding) and call.module is not emb_mod and call.output.shape == e.shape and close(e + call.output, h0):
                        facts["position_embeddings"] = {"module": call.path, "rows": call.module.weight.shape[0]}
                        facts["embedding_scale"] = 1.0
                        break
                    if _norm_like(call) and close(call.input, e) and close(call.output, h0):
                        form = norm_formula(call, eps)
                        facts["embedding_norm"] = {"module": call.path, "formula": form[0] if form else None,
                                                   "eps": form[1] if form else None}
                        facts["embedding_scale"] = 1.0
                        break
                if facts["embedding_scale"] is None:
                    facts["unknown"].append("block 0 does not receive token embeddings, scaled embeddings, "
                                            "learned positions or a norm of the embeddings")
    # Output: final norm, then the output projection, then the logits.
    last = layer_calls[-1]
    head = _output_projection(trace, last)
    head_mod = head.module if head is not None else None
    after = [c for c in calls if c.order > last.order and not _is_descendant(c.path, trace.layers_path)]
    final_norm = next((c for c in after if _norm_like(c) and close(c.input, last.output)), None)
    if streams and head is not None:
        # AltUp merges its final streams through the unembedding projections,
        # matching their magnitudes to stream zero, then takes their mean.
        candidates = [c for c in after if _norm_like(c) and close(c.output, head.input)]
        for candidate in candidates:
            st = last.output
            merged = st[0].float().clone()
            magnitude = st[0].float().square().mean(-1, keepdim=True).sqrt()
            for i in range(1, st.shape[0]):
                projection = next((c for c in after if isinstance(c.module, nn.Linear) and close(c.input, st[i])), None)
                if projection is None:
                    break
                x = projection.output.float()
                merged += x * magnitude / x.square().mean(-1, keepdim=True).clamp(min=1e-5).sqrt()
            else:
                if close(merged / st.shape[0], candidate.input):
                    final_norm = candidate
                    break
    facts["final_norm"] = None
    if final_norm is not None:
        facts["final_norm"] = norm_fact(final_norm, eps)
    else:
        facts["unknown"].append("no norm between the last block and the output projection")
    facts["logits"] = None
    if head is None:
        facts["unknown"].append("the reference's output projection cannot be identified uniquely from its calls")
    if head is not None:
        input_scale = 1.0
        if final_norm is not None and not close(head.input, final_norm.output):
            source = final_norm.output.float()
            nonzero = source.abs() > 1e-6
            input_scale = float((head.input.float()[nonzero] / source[nonzero]).median())
            if (not math.isfinite(input_scale) or input_scale <= 0 or
                    not close(head.input, source * input_scale) or getattr(head_mod, "bias", None) is not None):
                facts["unknown"].append("the output projection does not read the final norm's output or a "
                                        "constant multiple of it without a bias")
                input_scale = 1.0
        # Scaling a bias-free projection's input is the same linear scale
        # as scaling its output. Canonicalize both conventions for the row.
        lm = head.output.float() / input_scale
        lg = trace.logits.float()
        if close(lm, lg):
            facts["logits"] = {"transform": None}
        else:
            ratio = float((lg.flatten() / torch.where(lm.abs() > 1e-6, lm, torch.ones_like(lm)).flatten()).median())
            if close(lm * ratio, lg):
                facts["logits"] = {"transform": "scale", "value": ratio}
            else:
                cap = getattr(trace.config, "final_logit_softcapping", None)
                if cap and close(cap * torch.tanh(lm / cap), lg):
                    facts["logits"] = {"transform": "softcap", "value": float(cap)}
                else:
                    facts["unknown"].append("the logits are not the output projection's output or a scaled form of it")
        facts["tied_output"] = bool(head_mod is not None and emb_mod is not None
                                    and head_mod.weight.data_ptr() == emb_mod.weight.data_ptr())

    blocks = []
    for i, (lp, lc) in enumerate(zip(layer_paths, layer_calls)):
        bc = [c for c in calls if _is_descendant(c.path, lp)]
        blocks.append(analyze_block(trace, i, lp, lc, bc, freqs, eps))
    facts["block_facts"] = blocks
    return facts


def analyze_recurrent(trace, mixer, block_calls):
    """Prove short convolution and the query/key head sharing of delta-rule operations."""
    calls = [c for c in block_calls if _is_descendant(c.path, mixer.path)]
    result = Facts(kind="recurrent", unknown=[], classes=sorted({type(c.module).__name__ for c in calls}))
    d = mixer.input.shape[-1]
    linears = [c for c in calls if isinstance(c.module, nn.Linear)]
    convs = [c.module for c in calls if isinstance(c.module, nn.Conv1d)]
    # Functional convolution kernels may bypass nn.Conv1d.forward; its weights
    # remain owned by the mixer and are inspected without invoking that kernel.
    convs += [m for m in mixer.module.modules() if isinstance(m, nn.Conv1d) and m not in convs]
    for projected in linears:
        if projected.output.shape[-1] != 3 * d:
            continue
        B, C, x = projected.output.chunk(3, dim=-1)
        for conv in convs:
            if conv.in_channels != d or conv.groups != d:
                continue
            y = torch.nn.functional.conv1d((B * x).transpose(1, 2), conv.weight, conv.bias,
                                           padding=conv.kernel_size[0] - 1, groups=d)[..., :SEQ].transpose(1, 2) * C
            if any(close(c.input, y) and close(c.output, mixer.output) for c in linears):
                result.update(operation="shortconv", taps=conv.kernel_size[0],
                              formula="out_proj(C * causal_depthwise_conv(B * x)); in_proj = [B | C | x]")
                return result
    # A convolution's SiLU output is what the delta rule receives before
    # repeating query/key heads. Match both sharing orders numerically.
    for conv in convs:
        if conv.bias is not None or conv.groups != conv.in_channels:
            continue
        for projected in linears:
            raw = projected.output
            if raw.ndim != 3:
                continue
            candidates = [raw]
            # A projection can group [q | k | v | gate] by key head. Its
            # ordering method is exercised with recorded inputs, never named
            # by architecture or trusted from a class-name assertion.
            ordering = getattr(mixer.module, "fix_query_key_value_ordering", None)
            if callable(ordering):
                for other in linears:
                    try:
                        q, k, v, *_ = ordering(raw, other.output)
                        candidates.append(torch.cat([t.reshape(1, SEQ, -1) for t in (q, k, v)], dim=-1))
                    except (RuntimeError, ValueError):
                        pass
            for raw_qkv in candidates:
                if raw_qkv.shape[-1] != conv.in_channels:
                    continue
                y = torch.nn.functional.silu(torch.nn.functional.conv1d(raw_qkv.transpose(1, 2), conv.weight,
                    padding=conv.kernel_size[0] - 1, groups=conv.groups)[..., :SEQ].transpose(1, 2))
                for rec in trace.recurrences:
                    q, k, v = (rec[t] for t in ("query", "key", "value"))
                    hv, hd = v.shape[-2:]
                    key_dim = (conv.in_channels - hv * v.shape[-1]) // 2
                    if key_dim <= 0 or key_dim % hd or hv % (key_dim // hd):
                        continue
                    hk = key_dim // hd
                    q0, k0, v0 = y.split((key_dim, key_dim, hv * v.shape[-1]), dim=-1)
                    if not close(v0, v):
                        continue
                    q0, k0 = (t.reshape(1, SEQ, hk, hd) for t in (q0, k0))
                    if close(q0.repeat_interleave(hv // hk, dim=2), q) and close(k0.repeat_interleave(hv // hk, dim=2), k):
                        out_proj = next((c.path for c in linears if close(c.output, mixer.output)), None)
                        result.update(operation="deltanet", grouped_heads=True, key_heads=hk, value_heads=hv,
                                      head_dim=hd, output_projection=out_proj)
                        return result
    return result


def analyze_block(trace, index, path, layer_call, block_calls, freqs, eps):
    b = Facts(index=index, unknown=[], absent=[])
    h_in, h_out = layer_call.input, layer_call.output
    # Sub-modules that pass their input through unchanged (dropout in evaluation mode) take no part.
    direct = [c for c in _direct_children(block_calls, path)
              if not (not any(True for _ in c.module.parameters()) and close(c.input, c.output))]
    norms = [c for c in block_calls if c.input is not None and c.input.shape == h_in.shape and _norm_like(c)]
    others = [c for c in direct if not _norm_like(c)]
    if len(others) == 1:
        mixer = others[0]
        m = mixer.module
        # Identify the SSM by its projections and state tensors, independent
        # of model/class names. Unknown one-part blocks still refuse a row.
        if all(hasattr(m, k) for k in ("in_proj", "conv1d", "A_log", "D", "out_proj")):
            pre = next((c for c in norms if close(c.input, h_in) and close(c.output, mixer.input)), None)
            form = norm_formula(pre, eps) if pre else None
            b["norm_before_mixer"] = ({"module": pre.path, "formula": form[0] if form else None,
                                       "eps": form[1] if form else None, "size": pre.module.weight.numel()}
                                      if pre else None)
            if pre is None:
                b["unknown"].append("SSM mixer has no identified input norm")
            if not close(h_in.float() + mixer.output.float(), h_out):
                b["unknown"].append("SSM block output is not its input plus the mixer output")
            v2 = not hasattr(m, "x_proj")
            b["mixer"] = Facts(kind="mamba2" if v2 else "mamba", unknown=[],
                inner=m.intermediate_size, state=m.ssm_state_size, kernel=m.conv1d.kernel_size[0],
                rank=m.num_heads if v2 else m.time_step_rank, groups=m.n_groups if v2 else 1,
                activation=m.activation, dt_b_c_rms=hasattr(m, "dt_layernorm"),
                dt_b_c_eps=float(getattr(m, "rms_eps", 1e-6)),
                dt_limit=list(getattr(m, "time_step_limit", (0.0, float("inf")))))
            if v2:
                nc = next((c for c in block_calls if c.module is m.norm), None)
                group_size = int(getattr(m.norm, "group_size", m.intermediate_size))
                b["mixer"]["norm_grouped"] = group_size < m.intermediate_size
                # Verify the observed norm, including gating order. A module
                # that does anything else leaves the SSM fact unknown.
                if nc is None or len(nc.args) < 2:
                    b["unknown"].append("SSM gated norm invocation was not recorded")
                else:
                    z = nc.input.float() * torch.nn.functional.silu(nc.args[1].float())
                    z = z.reshape(*z.shape[:-1], -1, group_size)
                    e = float(getattr(m.norm, "variance_epsilon", eps))
                    want = (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + e)).reshape(nc.input.shape)
                    want = want * m.norm.weight.detach()
                    if not close(want, nc.output):
                        b["unknown"].append("SSM gated norm is not gate-then-RMSNorm with the declared grouping")
            b["ffn"] = Facts(kind="none")
            b["residual"] = "sequential"
            return b

    def branch_norm(c):
        return {**norm_fact(c, eps), "size": int(c.input.shape[-1]),
                "weighted": getattr(c.module, "weight", None) is not None}

    if not others and not norms and close(h_in, h_out):
        return Facts(index=index, unknown=[], absent=[], mixer={"kind": "none"}, ffn={"kind": "none"},
                     residual="sequential", norm_before_mixer=None, norm_after_mixer=None,
                     norm_before_ffn=None, norm_after_ffn=None)
    if len(others) == 1:
        branch = others[0]
        ffn_facts = analyze_dense_ffn(branch, block_calls)
        mixer_facts = analyze_attention(trace, branch.path, block_calls, freqs, eps)
        if mixer_facts is None:
            mixer_facts = analyze_linear_mixer(branch)
        is_ffn = not ffn_facts.get("unknown")
        if is_ffn or mixer_facts is not None:
            before = next((c for c in norms if close(c.input, h_in) and close(c.output, branch.input)), None)
            after = next((c for c in norms if c.order > branch.order and close(c.input, branch.output)), None)
            if before is None and not close(branch.input, h_in):
                b["unknown"].append("the only branch does not read the residual or its norm")
            if any(c is not before and c is not after for c in norms):
                b["unknown"].append("an extra norm outside the only residual branch")
            value = after.output if after is not None else branch.output
            delta = h_out.float() - h_in.float()
            b["residual"] = "sequential"
            if not close(h_in.float() + value.float(), h_out):
                nonzero = value.abs() > 1e-6
                r = float((delta[nonzero] / value.float()[nonzero]).median()) if nonzero.any() else 1.0
                if close(h_in.float() + r * value.float(), h_out):
                    b["residual"] = {"multiplier": r}
                else:
                    b["unknown"].append("the only branch does not add to the input residual")
            b.update(mixer={"kind": "none"} if is_ffn else mixer_facts,
                     ffn=ffn_facts if is_ffn else {"kind": "none"},
                     norm_before_mixer=None if is_ffn or before is None else branch_norm(before),
                     norm_after_mixer=None if is_ffn or after is None else branch_norm(after),
                     norm_before_ffn=branch_norm(before) if is_ffn and before else None,
                     norm_after_ffn=branch_norm(after) if is_ffn and after else None)
            return b
    predicted = None
    laurel = None
    if h_in is not None and h_in.dim() == 4:
        k = h_in.shape[0]
        coefficients = next((c for c in _linears(block_calls, path)
                             if c.output.shape[-1] == k * k and c.input.shape[-1] == k), None)
        if coefficients is not None:
            coef = coefficients.output.reshape(1, SEQ, k, k).transpose(-1, -2)
            predicted = torch.matmul(h_in.permute(1, 2, 3, 0), coef).permute(3, 0, 1, 2) + h_in
            mixer = next((c for c in direct if any(a.module is c.module for a in trace.attn)), None)
            ffn = next((c for c in direct if len(_linears(block_calls, c.path)) == 3 and
                        analyze_dense_ffn(c, block_calls).get("activation") is not None), None)
            pre = next((n for n in norms if mixer is not None and close(n.output, mixer.input)), None)
            active = next((i for i in range(k) if pre is not None and close(predicted[i], pre.input)), None)
            if active is not None and mixer is not None and ffn is not None:
                b["altup"] = {"streams": k, "active": active}
                h_in = predicted[active]
                others = [mixer, ffn]
                laurel = next((c for c in direct if c not in others and
                               close(c.input, pre.output) and c.output is not None and
                               len(_linears(block_calls, c.path)) == 2), None)
    elif len(others) > 2:
        # A per-layer token-input branch adds two direct projections beside
        # the mixer and gated FFN. Identify the latter by their observed
        # attention and activation equations, leaving the branch for the
        # output check below.
        mixer = next((c for c in direct if any(a.module is c.module for a in trace.attn)), None)
        ffn = next((c for c in direct if len(_linears(block_calls, c.path)) == 3 and
                    analyze_dense_ffn(c, block_calls).get("activation") is not None), None)
        if mixer is not None and ffn is not None:
            others = [mixer, ffn]
    if len(others) != 2:
        b["unknown"].append(f"the block runs {len(others)} sub-modules besides its norms "
                            f"({', '.join(c.path.rsplit('.', 1)[-1] for c in others)}), not a mixer and an FFN")
        return b
    mixer, ffn = others
    nested_att = [c for c in block_calls if (c.path == mixer.path or _is_descendant(c.path, mixer.path))
                  and any(c.module is a.module for a in trace.attn)]
    if len(nested_att) == 1:
        mixer = nested_att[0]  # norms may wrap the attention and its residual
    b["mixer_module"] = mixer.path
    b["ffn_module"] = ffn.path

    pre_mixer = next((n for n in norms if close(n.input, h_in) and close(n.output, mixer.input)), None)
    b["norm_before_mixer"] = branch_norm(pre_mixer) if pre_mixer else None
    if pre_mixer is None:
        if close(mixer.input, h_in):
            b["absent"].append("no norm before the token mixer (it reads the residual stream directly)")
        else:
            b["unknown"].append("the token mixer does not read the block input or a norm of it")
    pre_ffn = next((n for n in norms if n.order < ffn.order and n is not pre_mixer and close(n.output, ffn.input)),
                   None)
    post_mixer = next((n for n in norms if n is not pre_ffn and n.order > mixer.order and close(n.input, mixer.output)), None)
    b["norm_after_mixer"] = branch_norm(post_mixer) if post_mixer else None
    m_out = post_mixer.output if post_mixer else mixer.output
    # Some eager implementations return a residual from the mixer or FFN.
    # Prove it against the inner projection rather than its module name.
    if pre_ffn is not None and close(pre_ffn.input, m_out) and any(
            c.output.shape == h_in.shape and close(c.output + h_in, m_out) for c in _linears(block_calls, mixer.path)):
        m_out = m_out.float() - h_in.float()
    h_mid = h_in.float() + m_out.float()
    if predicted is not None and laurel is not None:
        h_mid = (h_mid + laurel.output.float()) / math.sqrt(2)
        b["laurel"] = True
    b["parallel"] = False
    b["residual"] = "sequential"
    if pre_mixer is not None and close(ffn.input, pre_mixer.output):
        b["parallel"] = True
        b["residual"] = "parallel"
        b["absent"].append("the FFN reads the same normed input as the mixer (parallel attention and FFN)")
    elif pre_ffn is not None and close(pre_ffn.input, h_in):
        b["parallel"] = True
        b["residual"] = "parallel"
    elif pre_ffn is not None and not close(pre_ffn.input, h_mid):
        # A multiplier on the mixer's output before the residual add.
        delta = pre_ffn.input.float() - h_in.float()
        r = float((delta.flatten() / torch.where(m_out.abs() > 1e-6, m_out.float(), torch.ones_like(m_out.float()))
                   .flatten()).median())
        if close(h_in.float() + r * m_out.float(), pre_ffn.input):
            b["residual"] = {"multiplier": r}
            h_mid = h_in.float() + r * m_out.float()
        else:
            b["unknown"].append("the FFN's norm does not read the block input plus the mixer's output")
    b["norm_before_ffn"] = branch_norm(pre_ffn) if pre_ffn else None
    if pre_ffn is None and not b["parallel"]:
        if close(ffn.input, h_mid):
            b["absent"].append("no norm before the FFN (it reads the residual stream directly)")
        else:
            b["unknown"].append("the FFN does not read the residual stream or a norm of it")
    post_ffn = next((n for n in norms if n.order > ffn.order and close(n.input, ffn.output)), None)
    b["norm_after_ffn"] = branch_norm(post_ffn) if post_ffn else None
    f_out = (post_ffn.output if post_ffn else ffn.output).float()
    if any(c.output.shape == h_mid.shape and close(c.output + h_mid, f_out) for c in _linears(block_calls, ffn.path)):
        f_out = f_out - h_mid
    if b.get("altup"):
        # Verify the correction and per-layer residual against the complete
        # observed stream output; the ordinary block below ends before them.
        core_out = h_mid + f_out
        k, active = b["altup"]["streams"], b["altup"]["active"]
        coefficients = next((c for c in _linears(block_calls, path)
                             if c.output.shape[-1] == k and c.input.shape[-1] == k), None)
        corrected = None
        if coefficients is not None:
            coefs = (coefficients.output + 1).permute(2, 0, 1).unsqueeze(-1)
            corrected = predicted + (core_out - predicted[active]).unsqueeze(0) * coefs
        per_layer_norms = [n for n in norms if post_ffn is not None and n.order > post_ffn.order]
        if corrected is None or len(per_layer_norms) != 1:
            b["unknown"].append("could not establish the multi-stream correction and per-layer residual")
        else:
            corrected[1:] += per_layer_norms[0].output
            if not close(corrected, h_out):
                b["unknown"].append("the stream output does not match prediction, correction and per-layer residual")
        h_out = core_out
        norms = [n for n in norms if n not in per_layer_norms]
    elif post_ffn is not None:
        per_layer_norms = [n for n in norms if n.order > post_ffn.order]
        if len(per_layer_norms) == 1:
            core_out = h_mid + f_out
            branch = per_layer_norms[0].output
            combined = core_out + branch
            scale = float((h_out.float() / torch.where(combined.abs() > 1e-6, combined,
                                                      torch.ones_like(combined))).median())
            if close(combined * scale, h_out):
                b["per_layer_input"] = True
                b["output_scale"] = scale
                h_out = core_out
                norms = [n for n in norms if n not in per_layer_norms]
    extra = [n for n in norms if n not in (pre_mixer, post_mixer, pre_ffn, post_ffn)]
    if extra:
        b["unknown"].append("norms with no place in the block: " + ", ".join(n.path for n in extra))
    if b["parallel"]:
        if not close(h_in.float() + m_out.float() + f_out, h_out):
            b["unknown"].append("the block output is not its input plus the mixer's and the FFN's outputs")
    elif isinstance(b["residual"], dict):
        r = b["residual"]["multiplier"]
        if not close(h_mid + r * f_out, h_out):
            b["unknown"].append("the block output is not the residual plus the FFN's scaled output")
    elif not close(h_mid + f_out, h_out):
        delta = h_out.float() - h_mid
        r = float((delta.flatten() / torch.where(f_out.abs() > 1e-6, f_out, torch.ones_like(f_out)).flatten()).median())
        if close(h_mid + r * f_out, h_out):
            b["residual"] = {"multiplier": r, "only_ffn": True}
        else:
            b["unknown"].append("the block output is not the residual plus the FFN's output")

    mixer_facts = analyze_attention(trace, mixer.path, block_calls, freqs, eps)
    if mixer_facts is None:
        mixer_facts = analyze_linear_mixer(mixer)
        if mixer_facts is None:
            mixer_facts = analyze_recurrent(trace, mixer, block_calls)
    b["mixer"] = mixer_facts
    is_moe = any(not a.is_floating_point() and a.dim() == 2
                 for c in block_calls if _is_descendant(c.path, ffn.path)
                 for a in list(c.args) + list(c.kwargs.values()) if torch.is_tensor(a))
    b["ffn"] = analyze_moe(ffn, block_calls, trace.reduced) if is_moe else analyze_dense_ffn(ffn, block_calls)
    return b


def restore_depth_scale(facts, config, traced_layers, full_layers):
    """Map a verified scale_depth/sqrt(layer_count) branch scale to the full configuration."""
    depth = getattr(config, "scale_depth", None)
    if depth is None or traced_layers == full_layers:
        return
    blocks = facts["block_facts"]
    if not isinstance(depth, (int, float)) or isinstance(depth, bool) or not math.isfinite(depth) or depth <= 0:
        facts["unknown"].append("scale_depth is not a finite positive scalar")
        return
    expected = float(depth) / math.sqrt(traced_layers)
    if (not blocks or any(b.get("mixer", {}).get("kind") in (None, "none")
                                 or b.get("ffn", {}).get("kind") in (None, "none")
                                 or not isinstance(b.get("residual"), dict)
                                 or b["residual"].get("only_ffn")
                                 or not math.isclose(b["residual"]["multiplier"], expected, rel_tol=1e-5, abs_tol=1e-6)
                                 for b in blocks)):
        facts["unknown"].append("scale_depth is not the verified scale_depth/sqrt(traced layer count) "
                                "on both residual branches")
        return
    original = float(depth) / math.sqrt(full_layers)
    for block in blocks:
        block["residual"]["traced_multiplier"] = block["residual"]["multiplier"]
        block["residual"]["multiplier"] = original
    facts["depth_scale_probe"] = {"scale_depth": depth, "traced_layers": traced_layers,
                                  "original_layers": full_layers, "traced_multiplier": expected,
                                  "original_multiplier": original}


def describe_reference(model_id, revision=None, trust_remote_code=False):
    """Traces the model's reference implementation and returns its facts (a JSON-able dict)."""
    trace = run_trace(model_id, revision, trust_remote_code)
    facts = analyze(trace)
    cfg = trace.config
    restore_depth_scale(facts, cfg, trace.keep, trace.n_layers)
    original_windows = {getattr(cfg, k, None) for k in ("sliding_window", "window_size")
                        if isinstance(getattr(cfg, k, None), int) and getattr(cfg, k) > 0}
    reduced_windows = {getattr(trace.reduced, k, None) for k in ("sliding_window", "window_size")
                       if isinstance(getattr(trace.reduced, k, None), int) and getattr(trace.reduced, k) > 0}
    if len(original_windows) == len(reduced_windows) == 1 and original_windows != reduced_windows:
        original, reduced = next(iter(original_windows)), next(iter(reduced_windows))
        facts["window_probe"] = {"configured": original, "traced": reduced}
        for block in facts["block_facts"]:
            mixer = block.get("mixer", {})
            if mixer.get("window") == reduced:
                mixer["observed_window"] = reduced
                mixer["window"] = original
    facts["config"] = {k: v for k, v in cfg.to_dict().items() if not k.startswith("_") or k == "_sliding_window_pattern"}
    for key in ("layer_types", "interleaved_sliding_window"):
        value = getattr(cfg, key, None)
        if isinstance(value, (list, tuple)):
            facts["config"][key] = list(value)
    facts["model_type"] = getattr(cfg, "model_type", None)
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local directory")
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="run modeling code shipped with the model (only for repositories you trust)")
    args = parser.parse_args()
    facts = describe_reference(args.model, args.revision, args.trust_remote_code)
    facts.pop("config", None)
    json.dump(facts, sys.stdout, indent=1, default=str)
    print()


if __name__ == "__main__":
    main()
