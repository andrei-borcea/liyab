#!/usr/bin/env python3
"""Compare explicitly isolated shared-KV/mHC components with native transformers.

This is not a DeepSeek-V4 model gate: fixtures have dense FFNs and sinks=-inf.
Reference math is native float32, including the main keys (never rounded to the
engine's F16 cache). All sliding, CSA and HCA kinds remain in the component run.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("prefix", help="PREFIX emitted by test_compression --export-fixture")
parser.add_argument("--out", required=True, help="JSON measurements and limitations")
args = parser.parse_args()

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention, DeepseekV4HyperConnection, DeepseekV4HyperHead,
    DeepseekV4MLP, DeepseekV4RMSNorm, DeepseekV4RotaryEmbedding,
)

torch.set_num_threads(4)
reader = gguf.GGUFReader(args.prefix + ".gguf")
tensors = {t.name: torch.from_numpy(np.array(t.data, copy=True)).float() for t in reader.tensors}

def scalar(suffix):
    field = reader.fields["deepseek4." + suffix]
    return field.parts[field.data[0]].reshape(-1)[0].item()

def copy(parameter, name):
    parameter.copy_(tensors[name].reshape(parameter.shape))

rates_field = reader.fields["deepseek4.attention.compress_ratios"]
rates = [int(rates_field.parts[i].reshape(-1)[0]) for i in rates_field.data]
dim = int(scalar("attention.key_length"))
hidden = int(scalar("embedding_length"))
layer_types = []
for layer, rate in enumerate(rates):
    if rate == 0:
        layer_types.append("sliding_attention")
    elif tensors[f"blk.{layer}.attn_compressor_kv.weight"].shape[0] == 2 * dim:
        layer_types.append("compressed_sparse_attention")
    else:
        layer_types.append("heavily_compressed_attention")
compress_rates = {kind: rate for kind, rate in zip(layer_types, rates) if rate}
partial = int(scalar("rope.dimension_count")) / dim
compressed_rope = {"rope_type": "default", "rope_theta": float(scalar("attention.compress_rope_freq_base")),
                   "partial_rotary_factor": partial}
if "deepseek4.rope.scaling.factor" in reader.fields:
    compressed_rope.update(rope_type="yarn", factor=float(scalar("rope.scaling.factor")), attention_factor=1.0,
                           original_max_position_embeddings=int(scalar("rope.scaling.original_context_length")),
                           beta_fast=float(scalar("rope.scaling.yarn_beta_fast")),
                           beta_slow=float(scalar("rope.scaling.yarn_beta_slow")))
cfg = DeepseekV4Config(
    vocab_size=tensors["token_embd.weight"].shape[0], hidden_size=hidden,
    num_hidden_layers=len(rates), num_attention_heads=int(scalar("attention.head_count")),
    num_key_value_heads=1, head_dim=dim, q_lora_rank=int(scalar("attention.q_lora_rank")),
    partial_rotary_factor=int(scalar("rope.dimension_count")) / dim,
    o_groups=int(scalar("attention.output_group_count")), o_lora_rank=int(scalar("attention.output_lora_rank")),
    index_head_dim=int(scalar("attention.indexer.key_length")), index_n_heads=int(scalar("attention.indexer.head_count")),
    index_topk=int(scalar("attention.indexer.top_k")), sliding_window=int(scalar("attention.sliding_window")),
    layer_types=layer_types, compress_rates=compress_rates,
    compress_rope_theta=float(scalar("attention.compress_rope_freq_base")),
    rope_parameters={"main": {"rope_type": "default", "rope_theta": float(scalar("rope.freq_base")), "partial_rotary_factor": partial},
                     "compress": compressed_rope},
    hc_mult=int(scalar("hyper_connection.count")), hc_sinkhorn_iters=int(scalar("hyper_connection.sinkhorn_iterations")),
    hc_eps=float(scalar("hyper_connection.epsilon")), rms_norm_eps=float(scalar("attention.layer_norm_rms_epsilon")),
    moe_intermediate_size=int(scalar("feed_forward_length")), mlp_layer_types=["moe"] * len(rates),
    num_nextn_predict_layers=0, n_routed_experts=8, num_experts_per_tok=2,
)
cfg._attn_implementation = "eager"
engine = json.loads(Path(args.prefix + ".json").read_text())
ids = torch.tensor([engine["tokens"]], dtype=torch.long)
pos = torch.arange(ids.shape[1]).unsqueeze(0)
rope = DeepseekV4RotaryEmbedding(cfg)
mask = torch.full((ids.shape[1], ids.shape[1]), float("-inf"))
for t in range(ids.shape[1]):
    mask[t, max(0, t + 1 - cfg.sliding_window):t + 1] = 0
mask = mask[None, None]
captured = []
with torch.no_grad():
    streams = F.embedding(ids, tensors["token_embd.weight"]).unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    for layer in range(len(rates)):
        prefix = f"blk.{layer}."
        attention = DeepseekV4Attention(cfg, layer).eval()
        for attr, suffix in (("q_a_proj.weight", "attn_q_a.weight"), ("q_a_norm.weight", "attn_q_a_norm.weight"),
                             ("q_b_proj.weight", "attn_q_b.weight"), ("kv_proj.weight", "attn_kv.weight"),
                             ("kv_norm.weight", "attn_kv_a_norm.weight"), ("o_a_proj.weight", "attn_output_a.weight"),
                             ("o_b_proj.weight", "attn_output_b.weight")):
            owner, parameter = attr.rsplit(".", 1)
            copy(getattr(attention.get_submodule(owner), parameter), prefix + suffix)
        attention.sinks.fill_(float("-inf"))
        if rates[layer]:
            comp = attention.compressor
            for parameter, suffix in ((comp.kv_proj.weight, "kv"), (comp.gate_proj.weight, "gate"),
                                      (comp.position_bias, "ape"), (comp.kv_norm.weight, "norm")):
                copy(parameter, prefix + "attn_compressor_" + suffix + ".weight")
            if hasattr(comp, "indexer"):
                indexer = comp.indexer
                for parameter, suffix in ((indexer.kv_proj.weight, "kv"), (indexer.gate_proj.weight, "gate"),
                                          (indexer.position_bias, "ape"), (indexer.kv_norm.weight, "norm")):
                    copy(parameter, prefix + "indexer_compressor_" + suffix + ".weight")
                copy(indexer.q_b_proj.weight, prefix + "indexer.attn_q_b.weight")
                copy(indexer.scorer.weights_proj.weight, prefix + "indexer.proj.weight")
                indexer.register_forward_hook(lambda _m, _a, output: captured.append(output.detach().clone()))
        hc_attention, hc_ffn = DeepseekV4HyperConnection(cfg), DeepseekV4HyperConnection(cfg)
        for hc, name in ((hc_attention, "hc_attn"), (hc_ffn, "hc_ffn")):
            copy(hc.fn, prefix + name + "_fn.weight")
            copy(hc.base, prefix + name + "_base.weight")
            copy(hc.scale, prefix + name + "_scale.weight")
        anorm, fnorm = DeepseekV4RMSNorm(hidden, eps=cfg.rms_norm_eps), DeepseekV4RMSNorm(hidden, eps=cfg.rms_norm_eps)
        copy(anorm.weight, prefix + "attn_norm.weight"); copy(fnorm.weight, prefix + "ffn_norm.weight")
        post, comb, collapsed = hc_attention(streams)
        normalized = anorm(collapsed)
        rotations = {kind: rope(normalized, pos, layer_type=kind) for kind in ("main", "compress")}
        output = attention(normalized, rotations, pos, mask, past_key_values=None)[0]
        streams = post.unsqueeze(-1) * output.unsqueeze(-2) + comb.transpose(-1, -2) @ streams
        post, comb, collapsed = hc_ffn(streams)
        mlp = DeepseekV4MLP(cfg)
        copy(mlp.gate_proj.weight, prefix + "ffn_gate.weight")
        copy(mlp.up_proj.weight, prefix + "ffn_up.weight")
        copy(mlp.down_proj.weight, prefix + "ffn_down.weight")
        output = mlp(fnorm(collapsed))
        streams = post.unsqueeze(-1) * output.unsqueeze(-2) + comb.transpose(-1, -2) @ streams
    head = DeepseekV4HyperHead(cfg)
    copy(head.hc_fn, "output_hc_fn.weight"); copy(head.hc_base, "output_hc_base.weight"); copy(head.hc_scale, "output_hc_scale.weight")
    output_norm = DeepseekV4RMSNorm(hidden, eps=cfg.rms_norm_eps)
    copy(output_norm.weight, "output_norm.weight")
    reference = F.linear(output_norm(head(streams)), tensors["output.weight"])[0].float()
    candidate = torch.tensor(engine["logits"]).reshape(reference.shape)
    logp, logq = reference.log_softmax(-1), candidate.log_softmax(-1)
    kl = (logp.exp() * (logp - logq)).sum(-1).clamp(min=0)
    correct = int((reference.argmax(-1) == candidate.argmax(-1)).sum())
    expected_selections = [sorted(int(i) for i in row if i >= 0) for row in captured[-1][0].tolist()]
    actual_selections = [sorted((int(i) + 1) // rates[-1] - 1 for i in row) for row in engine["index_positions"]]
    selection_matches = sum(a == b for a, b in zip(actual_selections, expected_selections))
    report = {"kind": engine["kind"], "reference": "native transformers5.19 DeepseekV4Attention/HyperConnection/HyperHead/MLP",
              "reference_dtype": "float32; main keys not rounded", "engine_kv": "f16", "tokens": len(engine["tokens"]),
              "layer_types": layer_types, "compress_rates": rates, "head_dim": dim, "index_dim": cfg.index_head_dim,
              "q_rank": cfg.q_lora_rank, "output_rank": cfg.o_lora_rank, "index_topk": cfg.index_topk,
              "mean_kl": kl.mean().item(), "max_kl": kl.max().item(), "top1_correct": correct,
              "top1_percent": correct / len(engine["tokens"]) * 100, "max_abs_logit": (reference - candidate).abs().max().item(),
              "selection_matches": selection_matches, "selection_rows": len(expected_selections),
              "limitations": ["component fixture, not an official full-model tiny gate", "dense FFNs; no hash or learned MoE routing",
                              "learned attention sinks explicitly set to -inf; sink operation owned by MoE", "no tokenizer/template support claim"]}
Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
