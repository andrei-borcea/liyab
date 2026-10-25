#!/usr/bin/env python3
"""Observes what a model's Hugging Face transformers implementation computes, for tools/describe_model.py.

A reduced copy of the model is built from its configuration with random weights: fewer blocks (enough to hold
every kind of block the configuration lists), a small vocabulary, narrow FFNs and few experts. The widths of the
residual stream and of the attention heads are kept, so every scale, base and multiplier is the model's own. One
pass over a short sequence is recorded: what every module receives and returns, and, through an attention function
registered with transformers (or observed direct PyTorch SDPA / eager operations), the queries, keys and values each attention
gets with its arguments (scale, window, soft-cap). The facts the engine needs are then established by recomputing
them, never from class or module names:

* the order of norms, token mixer and FFN in a block and how the residual adds combine them;
* each norm's formula (RMSNorm multiplying by w or by 1 + w, LayerNorm);
* how keys and queries are rotated (which dimensions are paired, how many, with which frequencies), whether their
  norms come before or after the rotation, and over how many dimensions those norms reach;
* the attention scale, window and soft-cap, and whether the attention output is gated;
* the FFN's form and activation, and a mixture of experts' routing (fused or individually observed experts,
  logits from a module or a direct linear operation, probabilities, selection bias, top-k,
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
import torch.nn.functional as F

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
    for key, value in vars(cfg).items():
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
    if isinstance(getattr(cfg, "index_topk", None), int):
        cfg.index_topk = min(cfg.index_topk, SEQ // 2)  # select within the short trace
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


def norm_formula(call, eps_fallback):
    """How a norm module maps its input to its output: ("rms" | "rms_1p" | "layer" | "rms_noweight", eps), or None."""
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
    reduced._attn_implementation = ATTN_IMPL
    torch.manual_seed(0)
    direct_attention = None
    try:
        model = AutoModelForCausalLM.from_config(reduced, trust_remote_code=trust_remote_code, dtype=torch.float32,
                                                 attn_implementation=ATTN_IMPL)
    except KeyError as error:
        if error.args != (ATTN_IMPL,):
            raise
        # Older remote implementations dispatch through a local table instead
        # of AttentionInterface. Observe their standard PyTorch operation.
        direct_attention = "sdpa"
        reduced._attn_implementation = "sdpa"
        torch.manual_seed(0)
        try:
            model = AutoModelForCausalLM.from_config(reduced, trust_remote_code=trust_remote_code, dtype=torch.float32,
                                                     attn_implementation="sdpa")
        except (KeyError, ValueError) as error:
            if not ((isinstance(error, KeyError) and error.args == ("sdpa",)) or
                    (isinstance(error, ValueError) and "does not support an attention implementation through "
                     "torch.nn.functional.scaled_dot_product_attention" in str(error))):
                raise
            direct_attention = "eager"
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
    stack = []
    paths = {}
    for path, mod in model.named_modules():
        if mod is model:
            continue
        paths[mod] = path

        if direct_attention:
            handles.append(mod.register_forward_pre_hook(lambda module, args: stack.append(module)))

        def hook(module, args, kwargs, output, path=path):
            RECORDER.calls.append(Call(RECORDER.next(), path, module,
                                       [a.detach() if torch.is_tensor(a) else a for a in args],
                                       {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()},
                                       _tensors(output)))
            if direct_attention:
                stack.pop()

        handles.append(mod.register_forward_hook(hook, with_kwargs=True))
    ids = torch.tensor([[(7 + 37 * i) % VOCAB for i in range(SEQ)]])
    original_sdpa = F.scaled_dot_product_attention
    original_matmul, original_softmax = torch.matmul, F.softmax
    original_linear = F.linear
    pending = {}

    def traced_linear(input, weight, bias=None):
        result = original_linear(input, weight, bias)
        owner = stack[-1] if stack else None
        own_weight = getattr(owner, "weight", None)
        if (owner is not None and not isinstance(owner, nn.Linear) and torch.is_tensor(own_weight) and
                own_weight.dim() == 2 and torch.equal(weight, own_weight)):
            # Some routers own a Parameter and call F.linear directly. Record
            # the actual operation's logits, not an inferred module formula.
            RECORDER.calls.append(Call(RECORDER.next(), paths[owner], owner, [input.detach()], {},
                                       [result.detach()]))
        return result

    def collapse_kv(key, value):
        kv_heads = getattr(reduced, "num_key_value_heads", key.shape[1])
        if key.shape[1] > kv_heads and key.shape[1] % kv_heads == 0:
            groups = key.shape[1] // kv_heads
            if torch.equal(key, key[:, ::groups].repeat_interleave(groups, dim=1)) and torch.equal(
                    value, value[:, ::groups].repeat_interleave(groups, dim=1)):
                return key[:, ::groups], value[:, ::groups]
        return key, value

    def traced_matmul(left, right, *args, **kwargs):
        result = original_matmul(left, right, *args, **kwargs)
        if not stack or left.dim() != 4 or right.dim() != 4:
            return result
        owner = stack[-1]
        item = pending.get(owner)
        if item is not None and "weights" in item and torch.equal(left, item["weights"]):
            k, v = collapse_kv(item["key"], right)
            rec = AttnCall(item["order"], owner, item["query"], k.detach().clone(), v.detach().clone(),
                           {"scaling": item["scale"], "is_causal": True, "dropout": 0.0})
            rec.output = result.transpose(1, 2).detach().clone()
            RECORDER.attn.append(rec)
            del pending[owner]
        elif (left.shape[-2] == SEQ and right.shape[-1] == SEQ and
              left.shape[-1] == right.shape[-2] and left.shape[:-2] == right.shape[:-2]):
            pending[owner] = {"query": left.detach().clone(), "key": right.transpose(-1, -2).detach().clone(),
                              "raw": result.detach().clone(), "order": RECORDER.next()}
        return result

    def traced_softmax(input, *args, **kwargs):
        result = original_softmax(input, *args, **kwargs)
        item = pending.get(stack[-1]) if stack else None
        if item is not None and input.shape == item["raw"].shape:
            # The trace can describe plain scaled causal attention from its
            # operations, but must refuse unknown score transforms or masks.
            visible = torch.ones_like(input, dtype=torch.bool).tril()
            raw, scores = item["raw"][visible].double(), input[visible].double()
            scale = (raw * scores).sum() / raw.square().sum()
            if not (torch.isfinite(scale) and scale > 0 and
                    close(scores, raw * scale) and torch.all(input[~visible] < -1e20)):
                raise ValueError("direct eager trace needs verified scaled scores and a full causal mask")
            item["scale"], item["weights"] = scale.item(), result.detach().clone()
        return result

    def traced_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        # Remote GQA implementations often repeat KV before calling SDPA.
        # Collapse only exactly repeated heads, preserving the tensors the
        # projections and per-head norms actually produced.
        k, v = collapse_kv(key, value)
        causal = is_causal
        if attn_mask is not None:
            expected = torch.ones_like(attn_mask, dtype=torch.bool).tril()
            visible = attn_mask if attn_mask.dtype == torch.bool else attn_mask == 0
            if not torch.equal(visible, expected) or (attn_mask.dtype != torch.bool and
                    not torch.all(attn_mask[~visible] < -1e20)):
                raise ValueError("direct SDPA trace needs a full causal mask; its mask semantics are unknown")
            causal = True
        rec = AttnCall(RECORDER.next(), stack[-1], query.detach().clone(), k.detach().clone(),
                       v.detach().clone(), {"scaling": kwargs.get("scale", query.shape[-1] ** -0.5),
                                            "is_causal": causal, "dropout": dropout_p})
        RECORDER.attn.append(rec)
        result = original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)
        rec.output = result.transpose(1, 2).detach().clone()
        return result

    try:
        if direct_attention:
            F.linear = traced_linear
        if direct_attention == "sdpa":
            F.scaled_dot_product_attention = traced_sdpa
        elif direct_attention == "eager":
            torch.matmul, F.softmax = traced_matmul, traced_softmax
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
    finally:
        F.scaled_dot_product_attention = original_sdpa
        torch.matmul, F.softmax = original_matmul, original_softmax
        F.linear = original_linear
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
                scaling = getattr(mod, prefix + "attention_scaling", None)
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
    for call, how, raw in raw_views:
        if raw is None:
            continue
        for rot, r in _rotations(raw, freqs):
            if close(r, target):
                return {"projection": call.path, "projection_layout": how, **rot, "norm": None}
        for n in norm_calls:
            nin = n.input
            nout = n.output
            if nin is None or nout is None or nin.numel() != raw.numel():
                continue
            w = getattr(n.module, "weight", None)
            span = int(w.numel()) if w is not None else None
            form = norm_formula(n, eps)
            norm = {"module": n.path, "span": span, "formula": form[0] if form else None,
                    "eps": form[1] if form else None}
            # Norm before the rotation: the norm reads the projection, the rotation reads the norm.
            if close(_as_heads(nin, heads, head_dim), raw):
                normed = _as_heads(nout, heads, head_dim)
                for rot, r in _rotations(normed, freqs):
                    if close(r, target):
                        return {"projection": call.path, "projection_layout": how, **rot,
                                "norm": {**norm, "order": "before_rope"}}
            # Norm after the rotation: the norm reads the rotated projection and gives the target.
            if close(_as_heads(nout, heads, head_dim), target):
                for rot, r in _rotations(raw, freqs):
                    if close(r, _as_heads(nin, heads, head_dim)):
                        return {"projection": call.path, "projection_layout": how, **rot,
                                "norm": {**norm, "order": "after_rope"}}
    return None


def analyze_indexer(trace, mixer_path, block_calls, freqs):
    """Recompute the observed integer position selection from the indexer's own projections."""
    cfg = trace.reduced
    heads, dim = getattr(cfg, "index_n_heads", 0), getattr(cfg, "index_head_dim", 0)
    if not heads or not dim:
        return None
    for call in block_calls:
        if not _is_descendant(call.path, mixer_path):
            continue
        outputs = [t for t in call.outputs if not t.is_floating_point() and t.dim() == 3 and t.shape[1] == SEQ]
        if not outputs:
            continue
        args = [t for t in list(call.args) + list(call.kwargs.values())
                if torch.is_tensor(t) and t.is_floating_point() and t.dim() == 3]
        if len(args) < 2:
            continue
        hidden, residual = args[:2]
        linears = _linears(block_calls, call.path)
        q = next((c for c in linears if close(c.input, residual) and c.output.shape[-1] == heads * dim), None)
        key = next((c for c in linears if close(c.input, hidden) and c.output.shape[-1] == dim), None)
        weight = next((c for c in linears if close(c.input, hidden) and c.output.shape[-1] == heads), None)
        norm = next((c for c in block_calls if _is_descendant(c.path, call.path) and _norm_like(c)
                     and key is not None and close(c.input, key.output)), None)
        if q is None or key is None or weight is None or norm is None:
            return Facts(unknown=["indexer projections and key norm could not be established"])
        form = norm_formula(norm, 1e-6)
        if not form or form[0] != "layer" or not math.isclose(form[1], 1e-6):
            return Facts(unknown=[f"indexer key norm is {form}, not LayerNorm with epsilon 1e-6"])
        observed = outputs[0][0]
        raw_q, raw_k = _as_heads(q.output, heads, dim), _as_heads(norm.output, 1, dim)
        matched = []
        for name, inv, magnitude in freqs:
            if 2 * inv.numel() > dim:
                continue
            for layout in ("adjacent", "halves"):
                qr, kr = rope(raw_q, inv, layout, magnitude), rope(raw_k, inv, layout, magnitude)
                scores = torch.relu(qr @ kr.transpose(-1, -2) / math.sqrt(dim))
                scores = (scores * weight.output.transpose(1, 2).unsqueeze(-1) / math.sqrt(heads)).sum(1)
                scores = scores.masked_fill(torch.ones(SEQ, SEQ, dtype=torch.bool).triu(1), -torch.inf)
                ok = True
                for t in range(SEQ):
                    count = min(observed.shape[-1], t + 1)
                    expected = scores[0, t].topk(count).indices.sort().values
                    actual = observed[t][observed[t] <= t].long().sort().values
                    if not torch.equal(expected, actual):
                        ok = False
                        break
                if ok:
                    matched.append((name, layout, int(inv.numel() * 2)))
        layouts = {m[1] for m in matched}
        if len(layouts) != 1:
            return Facts(unknown=["indexer RoPE pairing is not uniquely established by position selection"])
        return Facts(unknown=[], rope=matched[0][1], table=matched[0][0], rope_dim=matched[0][2],
                     heads=heads, head_dim=dim, topk=int(getattr(trace.config, "index_topk", 0)),
                     query=q.path, key=key.path, weights=weight.path, key_norm=norm.path)
    mixer = next((c for c in block_calls if c.path == mixer_path), None)
    attention = next((a for a in trace.attn if mixer is not None and a.module is mixer.module), None)
    indices = attention.kwargs.get("indices") if attention is not None else None
    if torch.is_tensor(indices):
        earlier = [a for a in trace.attn if a.order < attention.order]
        if earlier and torch.is_tensor(earlier[-1].kwargs.get("indices")) and torch.equal(
                indices, earlier[-1].kwargs["indices"]):
            return Facts(unknown=[], shared=True)
    return Facts(unknown=["no indexer's integer position selection was observed"])


def analyze_latent_attention(att, linears, norms, freqs, eps, cfg):
    """Establish MLA by reconstructing both expanded tensors from the observed shared latent.

    The rotary result may be returned in halves even when its input pairs are adjacent;
    that simultaneous output permutation changes neither query/key dots nor the GGUF row contract.
    """
    rank = getattr(cfg, "kv_lora_rank", 0)
    rd = getattr(cfg, "qk_rope_head_dim", 0)
    if not rank or not rd:
        return None
    heads, hd, vd = att.query.shape[1], att.query.shape[-1], att.value.shape[-1]
    nope = hd - rd
    for kv in linears:
        if kv.output.shape[-1] != rank + rd:
            continue
        norm = next((n for n in norms if close(n.input, kv.output[..., :rank])), None)
        form = norm_formula(norm, eps) if norm is not None else None
        if not form or form[0] != "rms":
            continue
        expanded = next((p for p in linears if close(p.input, norm.output)
                         and p.output.shape[-1] == heads * (nope + vd)), None)
        if expanded is None:
            continue
        parts = _as_heads(expanded.output, heads, nope + vd)
        if not close(parts[..., nope:], att.value):
            continue
        for query in linears:
            if query.output.shape[-1] != heads * hd:
                continue
            qrank = 0
            if not close(query.input, kv.input):
                qnorm = next((n for n in norms if close(n.output, query.input)), None)
                qform = norm_formula(qnorm, eps) if qnorm is not None else None
                qa = next((p for p in linears if qnorm is not None and close(p.output, qnorm.input)
                           and close(p.input, kv.input)), None)
                if qa is None or qform != form:
                    continue
                qrank = query.input.shape[-1]
            raw_q = _as_heads(query.output, heads, hd)
            raw_k = kv.output[..., rank:].unsqueeze(1)
            for name, inv, magnitude in freqs:
                if 2 * inv.numel() != rd:
                    continue
                for layout in ("adjacent", "halves"):
                    qr = rope(raw_q[..., nope:], inv, layout, magnitude)
                    kr = rope(raw_k, inv, layout, magnitude)
                    for permuted in (False, True) if layout == "adjacent" else (False,):
                        qrot, krot = qr, kr
                        if permuted:
                            qrot = torch.cat([qr[..., ::2], qr[..., 1::2]], -1)
                            krot = torch.cat([kr[..., ::2], kr[..., 1::2]], -1)
                        q = torch.cat([raw_q[..., :nope], qrot], -1)
                        k = torch.cat([parts[..., :nope], krot.expand(-1, heads, -1, -1)], -1)
                        if close(q, att.query) and close(k, att.key):
                            r = {"table": name, "layout": layout, "dims": rd, "magnitude": magnitude}
                            return {"kv_rank": rank, "q_rank": qrank, "nope_dim": nope, "v_dim": vd,
                                    "norm_eps": form[1],
                                    "kv_projection": kv.path, "kv_norm": norm.path,
                                    "key": {"projection": kv.path, "projection_layout": "latent", "rope": r,
                                            "norm": None},
                                    "query": {"projection": query.path, "projection_layout": "latent", "rope": r,
                                              "norm": None},
                                    "value": {"projection": expanded.path, "projection_layout": "latent"}}
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
                part = y[..., offset_heads * hd:(offset_heads + heads) * hd]
                views.append((call, f"fused [q | k | v] at head {offset_heads}", _as_heads(part, heads, hd)))
            else:
                views.append((call, how, v))
        return views

    key = _match_rotated(a.key, views_for(n_kv, n_q), norms, freqs, n_kv, hd, eps)
    query = _match_rotated(a.query, views_for(n_q, 0), norms, freqs, n_q, hd, eps)
    value = None
    for call, how, v in views_for(n_kv, n_q + n_kv) if hd_v == hd else []:
        if close(v, a.value):
            value = {"projection": call.path, "projection_layout": how}
            break
    latent = analyze_latent_attention(a, linears, norms, freqs, eps, trace.reduced)
    if latent is not None:
        key, query, value = latent["key"], latent["query"], latent["value"]
        f["latent"] = latent
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
        # Older modules return routing from a gate and call ModuleList
        # experts individually. Reconstruct their weighted sum from observed
        # inputs/outputs, checking the router's token ordering for every call.
        for gate in inside:
            indices = next((a for a in gate.outputs if not a.is_floating_point() and a.dim() == 2
                            and a.shape[0] == n_tokens), None)
            weights = next((a for a in gate.outputs if a.is_floating_point() and indices is not None
                            and a.shape == indices.shape), None)
            if indices is None or weights is None:
                continue
            count = max([getattr(cfg, key) for key in EXPERT_COUNT_KEYS
                         if isinstance(getattr(cfg, key, None), int)] or [0])
            for name, module in ffn_call.module.named_children():
                if not isinstance(module, nn.ModuleList) or len(module) != count:
                    continue
                order = indices.flatten().argsort()
                flat_ids = indices.flatten()[order]
                x = ffn_call.input.reshape(n_tokens, -1)
                flat_out = torch.empty(indices.numel(), x.shape[-1], dtype=x.dtype)
                activation = set()
                observed = []
                for expert, item in enumerate(module):
                    slots = order[flat_ids == expert]
                    if slots.numel() == 0:
                        continue
                    calls = [call for call in inside if call.module is item]
                    if len(calls) != 1 or not close(calls[0].input, x[slots // indices.shape[1]]):
                        break
                    call = calls[0]
                    facts = analyze_dense_ffn(call, block_calls)
                    if facts.get("unknown") or not facts.get("gated"):
                        break
                    activation.add(facts["activation"])
                    flat_out[slots] = call.output
                    observed.append(call)
                else:
                    if len(activation) != 1 or not observed:
                        continue
                    total = (flat_out.reshape(*indices.shape, -1) * weights.unsqueeze(-1)).sum(1)
                    aggregate = Call(max(c.order for c in observed), ffn_call.path + "." + name, module,
                                     [x, indices, weights], {}, [total.reshape(ffn_call.output.shape)])
                    experts_call = (aggregate, indices, weights)
                    f["expert_activation"] = activation.pop()
                    break
            if experts_call is not None:
                break
        if experts_call is None:
            f["unknown"].append("no experts call or verified per-expert routed outputs were observed")
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
        for bias_name, bias in biases + [(None, None)]:
            sel = probs + (bias if bias is not None else 0.0)
            groups = getattr(cfg, "n_group", 1) or 1
            used = getattr(cfg, "topk_group", groups) or groups
            # Test the grouping rule against the expert indices actually
            # observed. Configuration supplies geometry, never the rule.
            candidates = [(1, 1, None, sel)]
            if 1 <= used < groups and n_exp % groups == 0:
                grouped = sel.reshape(n_tokens, groups, n_exp // groups)
                for rule in ("max", "top2"):
                    scores = (grouped.max(-1).values if rule == "max" else
                              grouped.topk(min(2, n_exp // groups), dim=-1).values.sum(-1))
                    keep = scores.topk(used, dim=-1).indices
                    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, keep, True)
                    candidates.append((groups, used, rule,
                                       sel.masked_fill(~mask.repeat_interleave(n_exp // groups, -1), -torch.inf)))
            for groups_, used_, rule, selection in candidates:
                top = torch.topk(selection, k, dim=-1).indices
                if not torch.equal(torch.sort(top, -1).values, torch.sort(idx.long(), -1).values):
                    continue
                chosen = torch.gather(probs, -1, idx.long())
                for norm in (False, True):
                    w = chosen / chosen.sum(-1, keepdim=True) if norm else chosen
                    ratio = (wts.float() / w)
                    scale = float(ratio.mean())
                    if close(w * scale, wts.float()):
                        found = {"gating": gating, "selection_bias": bias_name, "norm_topk": norm, "scale": scale,
                                 "groups": groups_, "groups_used": used_, "group_score": rule}
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    if found is None:
        f["unknown"].append("routing is not top-k over softmax or sigmoid probabilities (with an optional "
                            "selection bias, group maximum or top-two sum, renormalization and scale)")
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
    f.setdefault("expert_activation", None)
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
        form = norm_formula(final_norm, eps)
        facts["final_norm"] = {"module": final_norm.path, "formula": form[0] if form else None,
                               "eps": form[1] if form else None}
    else:
        facts["unknown"].append("no norm between the last block and the output projection")
    facts["logits"] = None
    if head is not None:
        if final_norm is not None and not close(head.input, final_norm.output):
            facts["unknown"].append("the output projection does not read the final norm's output")
        lm = head.output.float()
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
    if len(others) != 2:
        b["unknown"].append(f"the block runs {len(others)} sub-modules besides its norms "
                            f"({', '.join(c.path.rsplit('.', 1)[-1] for c in others)}), not a mixer and an FFN")
        return b
    mixer, ffn = others
    b["mixer_module"] = mixer.path
    b["ffn_module"] = ffn.path

    def norm_fact(c):
        form = norm_formula(c, eps)
        return {"module": c.path, "formula": form[0] if form else None, "eps": form[1] if form else None,
                "size": int(c.module.weight.numel())}

    pre_mixer = next((n for n in norms if close(n.input, h_in) and close(n.output, mixer.input)), None)
    b["norm_before_mixer"] = norm_fact(pre_mixer) if pre_mixer else None
    if pre_mixer is None:
        if close(mixer.input, h_in):
            b["absent"].append("no norm before the token mixer (it reads the residual stream directly)")
        else:
            b["unknown"].append("the token mixer does not read the block input or a norm of it")
    post_mixer = next((n for n in norms if n.order > mixer.order and close(n.input, mixer.output)), None)
    b["norm_after_mixer"] = norm_fact(post_mixer) if post_mixer else None
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
    b["norm_before_ffn"] = norm_fact(pre_ffn) if pre_ffn else None
    if pre_ffn is None and not b["parallel"]:
        if close(ffn.input, h_mid):
            b["absent"].append("no norm before the FFN (it reads the residual stream directly)")
        else:
            b["unknown"].append("the FFN does not read the residual stream or a norm of it")
    post_ffn = next((n for n in norms if n.order > ffn.order and close(n.input, ffn.output)), None)
    b["norm_after_ffn"] = norm_fact(post_ffn) if post_ffn else None
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
        mixer_facts = Facts(kind="recurrent", unknown=[],
                            classes=sorted({type(c.module).__name__ for c in block_calls
                                            if _is_descendant(c.path, mixer.path) or c.path == mixer.path}))
    mixer_facts["indexer"] = analyze_indexer(trace, mixer.path, block_calls, freqs)
    b["mixer"] = mixer_facts
    is_moe = any(not a.is_floating_point() and a.dim() == 2
                 for c in block_calls if _is_descendant(c.path, ffn.path)
                 for a in list(c.args) + list(c.kwargs.values()) + c.outputs if torch.is_tensor(a))
    b["ffn"] = analyze_moe(ffn, block_calls, trace.reduced) if is_moe else analyze_dense_ffn(ffn, block_calls)
    return b


def describe_reference(model_id, revision=None, trust_remote_code=False):
    """Traces the model's reference implementation and returns its facts (a JSON-able dict)."""
    trace = run_trace(model_id, revision, trust_remote_code)
    facts = analyze(trace)
    cfg = trace.config
    facts["config"] = {k: v for k, v in cfg.to_dict().items() if not k.startswith("_") or k == "_sliding_window_pattern"}
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
