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
import math
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn

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
    counter: int = 0

    def next(self):
        self.counter += 1
        return self.counter


RECORDER = Recorder()


def _tensors(obj):
    if torch.is_tensor(obj):
        return [obj.detach()]
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
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

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
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTN_IMPL, sdpa_mask)


# ----------------------------------------------------------------------------------------------------------------
# Configuration: a reduced copy that keeps every kind of block
# ----------------------------------------------------------------------------------------------------------------

LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
TOP_K_KEYS = ("num_experts_per_tok", "num_experts_per_token", "moe_k", "moe_topk", "top_k")


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
    for key in ("first_k_dense_replace", "decoder_sparse_step", "moe_layer_freq", "full_attention_interval"):
        v = getattr(cfg, key, None)
        if isinstance(v, int) and v > 0:
            want = max(want, v + 1)
    mlp_only = getattr(cfg, "mlp_only_layers", None)
    if mlp_only:
        dense = [i for i in range(n) if i in mlp_only]
        sparse = [i for i in range(n) if i not in mlp_only]
        if dense and sparse:
            want = max(want, max(dense[0], sparse[0]) + 1)
    return min(n, max(want, 2))


def reduced_config(config):
    """A copy of the text configuration with fewer blocks, a small vocabulary, narrow FFNs and few experts."""
    cfg = copy.deepcopy(config.get_text_config() if hasattr(config, "get_text_config") else config)
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
    for k, v in list(vars(cfg).items()):
        if k.endswith("token_id") and isinstance(v, int) and v >= VOCAB:
            setattr(cfg, k, None)
        if k.endswith("token_id") and isinstance(v, list):
            setattr(cfg, k, [t for t in v if isinstance(t, int) and t < VOCAB] or None)
    for k, v in list(vars(cfg).items()):
        if "intermediate_size" in k and isinstance(v, int) and v > FFN:
            setattr(cfg, k, FFN)
    top_k = max([getattr(cfg, k) for k in TOP_K_KEYS if isinstance(getattr(cfg, k, None), int)] or [0])
    for k in EXPERT_COUNT_KEYS:
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > EXPERTS:
            groups = getattr(cfg, "n_group", None) or 1
            count = max(EXPERTS, top_k + 1)
            count = ((count + groups - 1) // groups) * groups
            setattr(cfg, k, min(v, count))
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
        mean = xf.mean(-1, keepdim=True)
        var = (xf - mean).pow(2).mean(-1, keepdim=True)
        ln = (xf - mean) * torch.rsqrt(var + eps) * wf
        if bias is not None and torch.is_tensor(bias):
            ln = ln + bias.detach().float()
        if close(ln, y):
            return "layer", eps
    elif w is None and close(rms, y):
        return "rms_noweight", eps
    return None


def norm_fact(call, eps):
    form = norm_formula(call, eps)
    fact = {"module": call.path, "formula": form[0] if form else None, "eps": form[1] if form else None}
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


def run_trace(model_id, revision=None, trust_remote_code=False):
    """Builds the reduced model, runs it once and returns the records."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # configuration and kernel notices are not facts about the model
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    reduced, keep, n = reduced_config(config)
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

    RECORDER.calls.clear()
    RECORDER.attn.clear()
    handles = []
    active = []
    for path, mod in model.named_modules():
        if mod is model:
            continue

        def hook(module, args, kwargs, output, path=path):
            RECORDER.calls.append(Call(RECORDER.next(), path, module,
                                       [a.detach() if torch.is_tensor(a) else a for a in args],
                                       {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()},
                                       _tensors(output)))
            active.pop()

        handles.append(mod.register_forward_hook(hook, with_kwargs=True))
        handles.append(mod.register_forward_pre_hook(lambda module, args: active.append(module)))
    ids = torch.tensor([[(7 + 37 * i) % VOCAB for i in range(SEQ)]])
    sdpa = torch.nn.functional.scaled_dot_product_attention
    matmul = torch.matmul
    pending_attention = {}
    def distinct_kv(module, key, value):
        heads = getattr(module, "num_key_value_heads", key.shape[1])
        if isinstance(heads, int) and heads > 0 and key.shape[1] % heads == 0:
            repeats = key.shape[1] // heads
            k, v = key[:, ::repeats], value[:, ::repeats]
            if torch.equal(k.repeat_interleave(repeats, dim=1), key) and torch.equal(v.repeat_interleave(repeats, dim=1), value):
                return k, v
        return key, value

    def observed_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None,
                      enable_gqa=False):
        module = active[-1]
        # Some custom classes accept an attention backend without using its
        # interface. Observe native calls too, but keep the richer registered
        # record when that interface already captured this module's attention.
        if any(record.module is module and record.output is None for record in RECORDER.attn):
            return sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
        keys, values = distinct_kv(module, key, value)
        rec = AttnCall(RECORDER.next(), module, query.detach().clone(), keys.detach().clone(),
                       values.detach().clone(), {"scaling": scale if scale is not None else query.shape[-1] ** -0.5,
                                                 "is_causal": is_causal, "observed_sdpa": True,
                                                 "attention_mask": attn_mask.detach().clone() if attn_mask is not None else None})
        RECORDER.attn.append(rec)
        result = sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
        rec.output = result.transpose(1, 2).detach().clone()
        return result

    def observed_matmul(left, right, *args, **kwargs):
        result = matmul(left, right, *args, **kwargs)
        if not active or not (torch.is_tensor(left) and torch.is_tensor(right) and left.dim() == right.dim() == 4):
            return result
        module = active[-1]
        if any(record.module is module and record.output is None for record in RECORDER.attn):
            return result  # a registered attention call already records this computation
        pending = pending_attention.get(module)
        if pending is not None and left.shape[-2:] == (SEQ, SEQ) and right.shape[-2] == SEQ:
            order, query, key = pending_attention.pop(module)
            scale = getattr(module, "scaling", query.shape[-1] ** -0.5)
            scale = float(scale) if isinstance(scale, (int, float)) else query.shape[-1] ** -0.5
            probabilities = left.detach().float()
            allowed = probabilities > 0
            scores = matmul(query.float(), key.float().transpose(-1, -2)) * scale
            expected = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
            keys, values = distinct_kv(module, key, right)
            rec = AttnCall(order, module, query, keys.detach().clone(), values.detach().clone(),
                           {"scaling": scale, "is_causal": False, "observed_sdpa": True,
                            "attention_mask": allowed, "manual_scores_verified": close(expected, probabilities)})
            rec.output = result.transpose(1, 2).detach().clone()
            RECORDER.attn.append(rec)
        elif left.shape[-2] == SEQ and right.shape[-1] == SEQ and left.shape[-1] == right.shape[-2]:
            # Legacy eager attention uses Q K^T followed by probabilities V.
            # Keep the actual tensors and verify the entire softmax before
            # accepting a scale or mask; unrelated matmuls remain unknown.
            pending_attention[module] = (RECORDER.next(), left.detach().clone(), right.transpose(-1, -2).detach().clone())
        return result
    try:
        torch.nn.functional.scaled_dot_product_attention = observed_sdpa
        torch.matmul = observed_matmul
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
    finally:
        torch.nn.functional.scaled_dot_product_attention = sdpa
        torch.matmul = matmul
        for h in handles:
            h.remove()
    logits = out.logits.detach() if hasattr(out, "logits") else _tensors(out)[0]
    return Trace(text, reduced, n, keep, model, layers_path, layers, list(RECORDER.calls), list(RECORDER.attn), ids,
                 logits, type(model).__name__)


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
    return [c for c in calls if _is_descendant(c.path, parent) and isinstance(c.module, nn.Linear)]


def _norm_like(call):
    w = getattr(call.module, "weight", None)
    return (w is not None and torch.is_tensor(w) and w.dim() == 1 and call.input is not None
            and call.output is not None and call.input.shape == call.output.shape
            and not isinstance(call.module, (nn.Linear, nn.Embedding)))


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
            span = int(w.numel()) if w is not None else None
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
    hd_v = a.value.shape[-1]
    f.update(heads=n_q, kv_heads=n_kv, head_dim=hd, value_head_dim=hd_v, unknown=[])
    linears = _linears(block_calls, mixer_path)
    norms = [c for c in block_calls if _is_descendant(c.path, mixer_path) and _norm_like(c)]
    fused_total = n_q + 2 * n_kv

    def views_for(heads, offset_heads):
        views = []
        for call, how, v in _projection_views(linears, heads, hd, fused_total):
            if how == "fused":
                y = v
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
                views.append((call, how, v, list(range(heads * hd))))
        return views

    key = _match_rotated(a.key, views_for(n_kv, n_q), norms, freqs, n_kv, hd, eps)
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
        if close(v, a.value):
            value = {"projection": call.path, "projection_layout": how, "projection_rows": rows}
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
    if f["output"] is None:
        f["unknown"].append("the output projection does not read the attention output (or its gated form)")
    return f


def analyze_dense_ffn(ffn_call, block_calls):
    """A dense FFN: down(act(gate) * up), down(act(up)), or a fused [gate | up] projection."""
    lin = [c for c in _linears(block_calls, ffn_call.path)]
    x, y = ffn_call.input, ffn_call.output
    down = next((c for c in lin if close(c.output, y)), None)
    if down is None:
        return Facts(kind="dense", unknown=["the FFN's output is not one projection's output"])
    first = [c for c in lin if c is not down and close(c.input, x)]
    d_in = down.input.float()
    for act_name, act in ACTIVATIONS.items():
        for g in first:
            for u in first:
                if g is not u and close(act(g.output.float()) * u.output.float(), d_in):
                    return Facts(kind="dense", gated=True, fused_gate_up=False, activation=act_name,
                                 gate=g.path, up=u.path, down=down.path,
                                 biases=any(getattr(c.module, "bias", None) is not None for c in (g, u, down)),
                                 unknown=[])
        for c in first:
            o = c.output.float()
            if o.shape[-1] == 2 * d_in.shape[-1]:
                h = d_in.shape[-1]
                for order, (gp, up) in (("gate_first", (o[..., :h], o[..., h:])), ("up_first", (o[..., h:], o[..., :h]))):
                    if close(act(gp) * up, d_in):
                        return Facts(kind="dense", gated=True, fused_gate_up=True, fused_order=order,
                                     activation=act_name, gate=c.path, up=c.path, down=down.path,
                                     biases=any(getattr(m.module, "bias", None) is not None for m in (c, down)),
                                     unknown=[])
            if close(act(o), d_in):
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
        n_exp = max([getattr(cfg, key) for key in EXPERT_COUNT_KEYS if isinstance(getattr(cfg, key, None), int)]
                    or [0])
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
    biases = []
    for name, t in list(rcall.module.named_parameters()) + list(rcall.module.named_buffers()):
        if t.dim() == 1 and t.numel() == n_exp:
            biases.append((name, t.detach().float()))
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
    # Experts' activation: recomputed from 3-D [experts, 2 * ff, d] / [experts, d, ff] weights when the experts
    # module holds them in that layout (transformers' fused experts).
    params = dict(ecall.module.named_parameters(recurse=False))
    gu, dn = params.get("gate_up_proj"), params.get("down_proj")
    f["expert_activation"] = None
    if gu is not None and dn is not None and gu.dim() == 3 and dn.dim() == 3:
        x = ecall.input.reshape(n_tokens, -1).float()
        for name, act in ACTIVATIONS.items():
            out = torch.zeros(n_tokens, dn.shape[1])
            ok = True
            for t in range(n_tokens):
                for j in range(k):
                    e = int(idx[t, j])
                    w = gu[e].float()
                    if w.shape[0] != 2 * dn.shape[2]:
                        ok = False
                        break
                    h = w @ x[t]
                    g_, u_ = h[: dn.shape[2]], h[dn.shape[2]:]
                    out[t] += float(wts[t, j]) * (dn[e].float() @ (act(g_) * u_))
                if not ok:
                    break
            if ok and close(out, ecall.output.reshape(n_tokens, -1)):
                f["expert_activation"] = name
                break
    return f


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
    emb_mod = trace.model.get_input_embeddings()
    h0 = layer_calls[0].input
    facts["embedding_scale"] = None
    if emb_mod is not None and h0 is not None:
        e = emb_mod.weight.detach()[trace.ids[0]].float().reshape(h0.shape)
        if close(e, h0):
            facts["embedding_scale"] = 1.0
        else:
            ratio = h0.float() / torch.where(e.abs() > 1e-6, e, torch.ones_like(e))
            c = float(ratio.flatten()[e.abs().flatten() > 1e-3].median())
            facts["embedding_scale"] = c if close(e * c, h0) else None
            if facts["embedding_scale"] is None:
                facts["unknown"].append("block 0 does not receive the token embeddings times a constant "
                                        "(position embeddings or another input are added)")
    # Output: final norm, then the output projection, then the logits.
    last = layer_calls[-1]
    head_mod = trace.model.get_output_embeddings()
    head = next((c for c in calls if c.module is head_mod), None)
    after = [c for c in calls if c.order > last.order and not _is_descendant(c.path, trace.layers_path)]
    final_norm = next((c for c in after if _norm_like(c) and close(c.input, last.output)), None)
    facts["final_norm"] = None
    if final_norm is not None:
        facts["final_norm"] = norm_fact(final_norm, eps)
    else:
        facts["unknown"].append("no norm between the last block and the output projection")
    facts["logits"] = None
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


def analyze_block(trace, index, path, layer_call, block_calls, freqs, eps):
    b = Facts(index=index, unknown=[], absent=[])
    h_in, h_out = layer_call.input, layer_call.output
    # Sub-modules that pass their input through unchanged (dropout in evaluation mode) take no part.
    direct = [c for c in _direct_children(block_calls, path)
              if not (not any(True for _ in c.module.parameters()) and close(c.input, c.output))]
    norms = [c for c in direct if _norm_like(c)]
    others = [c for c in direct if not _norm_like(c)]
    def branch_norm(c):
        return {**norm_fact(c, eps), "size": int(c.module.weight.numel())}

    if not others and not norms and close(h_in, h_out):
        return Facts(index=index, unknown=[], absent=[], mixer={"kind": "none"}, ffn={"kind": "none"},
                     residual="sequential", norm_before_mixer=None, norm_after_mixer=None,
                     norm_before_ffn=None, norm_after_ffn=None)
    if len(others) == 1:
        branch = others[0]
        ffn_facts = analyze_dense_ffn(branch, block_calls)
        mixer_facts = analyze_attention(trace, branch.path, block_calls, freqs, eps)
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
    if len(others) != 2:
        b["unknown"].append(f"the block runs {len(others)} sub-modules besides its norms "
                            f"({', '.join(c.path.rsplit('.', 1)[-1] for c in others)}), not a mixer and an FFN")
        return b
    mixer, ffn = others
    b["mixer_module"] = mixer.path
    b["ffn_module"] = ffn.path

    pre_mixer = next((n for n in norms if close(n.input, h_in) and close(n.output, mixer.input)), None)
    b["norm_before_mixer"] = branch_norm(pre_mixer) if pre_mixer else None
    if pre_mixer is None:
        if close(mixer.input, h_in):
            b["absent"].append("no norm before the token mixer (it reads the residual stream directly)")
        else:
            b["unknown"].append("the token mixer does not read the block input or a norm of it")
    post_mixer = next((n for n in norms if n.order > mixer.order and close(n.input, mixer.output)), None)
    b["norm_after_mixer"] = branch_norm(post_mixer) if post_mixer else None
    m_out = post_mixer.output if post_mixer else mixer.output
    h_mid = h_in.float() + m_out.float()
    pre_ffn = next((n for n in norms if n.order < ffn.order and n is not pre_mixer and close(n.output, ffn.input)),
                   None)
    b["parallel"] = False
    b["residual"] = "sequential"
    if pre_mixer is not None and close(ffn.input, pre_mixer.output):
        b["parallel"] = True
        b["residual"] = "parallel"
        b["absent"].append("the FFN reads the same normed input as the mixer (parallel attention and FFN)")
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
        projections = [c for c in _linears(block_calls, mixer.path)
                       if close(c.input, mixer.input) and close(c.output, mixer.output)]
        if len(projections) == 1:
            mixer_facts = Facts(kind="linear", projection=projections[0].path, unknown=[])
        else:
            mixer_facts = Facts(kind="recurrent", unknown=[],
                                classes=sorted({type(c.module).__name__ for c in block_calls
                                                if _is_descendant(c.path, mixer.path) or c.path == mixer.path}))
    b["mixer"] = mixer_facts
    is_moe = any(not a.is_floating_point() and a.dim() == 2
                 for c in block_calls if _is_descendant(c.path, ffn.path)
                 for a in list(c.args) + list(c.kwargs.values()) if torch.is_tensor(a))
    b["ffn"] = analyze_moe(ffn, block_calls, trace.reduced) if is_moe else analyze_dense_ffn(ffn, block_calls)
    return b


def describe_reference(model_id, revision=None, trust_remote_code=False):
    """Traces the model's reference implementation and returns its facts (a JSON-able dict)."""
    trace = run_trace(model_id, revision, trust_remote_code)
    facts = analyze(trace)
    cfg = trace.config
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
