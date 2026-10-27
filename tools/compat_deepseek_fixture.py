"""DeepSeek-V4 numerical fixture, using the installed upstream text model.

The GGUF writer below exports only this fixed synthetic configuration. It is
not a general checkpoint converter. Explicit mappings fail on unknown tensors;
no payload is silently dropped. The local tokenizer tests byte/ID transport,
not the production DeepSeek tokenizer or chat template. No network access.
"""

import json


def create(root, gguf_path=None):
    import torch
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
    from compat_fixture import make_tokenizer
    from reference_trace import _randomize

    torch.set_num_threads(4)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = make_tokenizer()
    tokenizer.save_pretrained(root)
    c = DeepseekV4Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=512,
        q_lora_rank=32,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        max_position_embeddings=512,
        o_groups=2,
        o_lora_rank=32,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2,
        sliding_window=8,
        partial_rotary_factor=0.125,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        mlp_layer_types=["hash_moe", "moe", "moe"],
        swiglu_limit=0.3,
        hc_mult=4,
        hc_sinkhorn_iters=6,
        num_nextn_predict_layers=0,
    )
    c._attn_implementation = "eager"
    m = DeepseekV4ForCausalLM(c)
    _randomize(m)
    with torch.no_grad():
        for name, table in m.named_buffers():
            if name.endswith(".tid2eid"):
                table.copy_(
                    (
                        torch.arange(c.vocab_size).unsqueeze(1) * 3
                        + torch.arange(c.num_experts_per_tok)
                    )
                    % c.num_local_experts
                )

    # Keep the production index-head count: too few ReLU heads create artificial
    # ties at the top-k cutoff. Frameworks do not specify which equal-scoring
    # indices topk returns. The probe records margins instead of changing scores.
    audit = {
        "seed": 1234,
        "routed_experts": {},
        "hash_differs_from_learned_topk": 0,
        "gate_clipped": 0,
        "up_clipped": 0,
        "index_cutoff_ties": 0,
        "index_min_cutoff_margin": None,
    }
    hooks = []

    def routing_hook(name):
        def hook(module, inputs, outputs):
            logits, weights, indices = outputs
            audit["routed_experts"][name] = sorted(set(indices.flatten().tolist()))
            if name.endswith(".0.mlp.gate"):
                learned = torch.topk(
                    torch.nn.functional.softplus(logits).sqrt(),
                    c.num_experts_per_tok,
                    dim=-1,
                ).indices
                audit["hash_differs_from_learned_topk"] += int(
                    (learned.sort(dim=-1).values != indices.sort(dim=-1).values)
                    .any(dim=-1)
                    .sum()
                )

        return hook

    def clipping_hook(kind):
        def hook(module, inputs, output):
            clipped = (
                output > c.swiglu_limit
                if kind == "gate"
                else output.abs() > c.swiglu_limit
            )
            audit[kind + "_clipped"] += int(clipped.sum())

        return hook

    def index_hook(module, inputs, scores):
        for t in range(scores.shape[1]):
            ready = min(
                (t + 1) // c.compress_rates["compressed_sparse_attention"],
                scores.shape[-1],
            )
            if ready <= c.index_topk:
                continue
            ordered = scores[0, t, :ready].topk(c.index_topk + 1).values
            margin = float(ordered[-2] - ordered[-1])
            audit["index_cutoff_ties"] += int(margin == 0)
            old = audit["index_min_cutoff_margin"]
            audit["index_min_cutoff_margin"] = (
                margin if old is None else min(old, margin)
            )

    for name, module in m.named_modules():
        if name.endswith(".mlp.gate"):
            hooks.append(module.register_forward_hook(routing_hook(name)))
        elif name.endswith(".mlp.shared_experts.gate_proj"):
            hooks.append(module.register_forward_hook(clipping_hook("gate")))
        elif name.endswith(".mlp.shared_experts.up_proj"):
            hooks.append(module.register_forward_hook(clipping_hook("up")))
        elif name.endswith(".compressor.indexer.scorer"):
            hooks.append(module.register_forward_hook(index_hook))
    m.eval()
    with torch.no_grad():
        m(torch.arange(128).remainder(c.vocab_size).unsqueeze(0), use_cache=False)
    for hook in hooks:
        hook.remove()
    if (
        len(audit["routed_experts"]) != c.num_hidden_layers
        or any(len(experts) < 3 for experts in audit["routed_experts"].values())
        or not audit["hash_differs_from_learned_topk"]
        or not audit["gate_clipped"]
        or not audit["up_clipped"]
        or audit["index_cutoff_ties"]
    ):
        raise RuntimeError(f"inactive or ambiguous compatibility fixture: {audit}")
    m.save_pretrained(root)
    (root / "compat-fixture.json").write_text(json.dumps(audit, indent=2) + "\n")
    if gguf_path:
        write_gguf(m, tokenizer, gguf_path)
    print(
        f"{root}: {sum(p.numel() for p in m.parameters())} parameters, {len(tokenizer)} tokens"
    )


def write_gguf(model, tokenizer, path):
    import gguf
    import numpy as np

    config = model.config
    path.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(path), "deepseek4")
    for key, val in {
        "embedding_length": config.hidden_size,
        "block_count": config.num_hidden_layers,
        "feed_forward_length": config.moe_intermediate_size,
        "context_length": config.max_position_embeddings,
        "attention.head_count": config.num_attention_heads,
        "attention.head_count_kv": 1,
        "attention.key_length": config.head_dim,
        "attention.value_length": config.head_dim,
        "attention.q_lora_rank": config.q_lora_rank,
        "attention.output_group_count": config.o_groups,
        "attention.output_lora_rank": config.o_lora_rank,
        "attention.sliding_window": config.sliding_window,
        "attention.indexer.head_count": config.index_n_heads,
        "attention.indexer.key_length": config.index_head_dim,
        "attention.indexer.top_k": config.index_topk,
        "rope.dimension_count": config.qk_rope_head_dim,
        "expert_count": config.num_local_experts,
        "expert_used_count": config.num_experts_per_tok,
        "expert_feed_forward_length": config.moe_intermediate_size,
        "expert_shared_feed_forward_length": config.moe_intermediate_size,
        "expert_gating_func": 4,
        "hash_layer_count": 1,
        "hyper_connection.count": config.hc_mult,
        "hyper_connection.sinkhorn_iterations": config.hc_sinkhorn_iters,
    }.items():
        w.add_uint32("deepseek4." + key, val)
    for key, val in {
        "attention.layer_norm_rms_epsilon": config.rms_norm_eps,
        "rope.freq_base": config.rope_theta,
        "attention.compress_rope_freq_base": config.compress_rope_theta,
        "hyper_connection.epsilon": config.hc_eps,
        "expert_weights_scale": config.routed_scaling_factor,
    }.items():
        w.add_float32("deepseek4." + key, float(val))
    w.add_bool("deepseek4.expert_weights_norm", True)
    w.add_array("deepseek4.attention.compress_ratios", [0, 4, 8])
    w.add_array(
        "deepseek4.swiglu_clamp_exp",
        [float(config.swiglu_limit)] * config.num_hidden_layers,
    )
    w.add_array(
        "deepseek4.swiglu_clamp_shexp",
        [float(config.swiglu_limit)] * config.num_hidden_layers,
    )
    vocab = tokenizer.get_vocab()
    ordered = sorted(vocab, key=vocab.get)
    w.add_tokenizer_model("gemma4")
    w.add_token_list(ordered)
    w.add_token_scores([0.0] * len(ordered))
    w.add_token_merges(["t h"])
    w.add_token_types(
        [
            int(gguf.TokenType.CONTROL)
            if i < 5
            else int(gguf.TokenType.BYTE)
            if t.startswith("<0x")
            else int(gguf.TokenType.NORMAL)
            for i, t in enumerate(ordered)
        ]
    )
    w.add_bos_token_id(tokenizer.bos_token_id)
    w.add_eos_token_id(tokenizer.eos_token_id)
    w.add_unk_token_id(tokenizer.unk_token_id)
    w.add_add_bos_token(False)
    w.add_add_space_prefix(False)
    roots = {
        "model.embed_tokens.weight": "token_embd.weight",
        "lm_head.weight": "output.weight",
        "model.norm.weight": "output_norm.weight",
        "model.hc_head.hc_fn": "output_hc_fn.weight",
        "model.hc_head.hc_base": "output_hc_base.weight",
        "model.hc_head.hc_scale": "output_hc_scale.weight",
    }
    parts = {
        "self_attn.sinks": "attn_sinks.weight",
        "self_attn.q_a_proj.weight": "attn_q_a.weight",
        "self_attn.q_a_norm.weight": "attn_q_a_norm.weight",
        "self_attn.q_b_proj.weight": "attn_q_b.weight",
        "self_attn.kv_proj.weight": "attn_kv.weight",
        "self_attn.kv_norm.weight": "attn_kv_a_norm.weight",
        "self_attn.o_a_proj.weight": "attn_output_a.weight",
        "self_attn.o_b_proj.weight": "attn_output_b.weight",
        "mlp.gate.weight": "ffn_gate_inp.weight",
        "mlp.gate.tid2eid": "ffn_gate_tid2eid.weight",
        "mlp.gate.e_score_correction_bias": "exp_probs_b.bias",
        "mlp.experts.gate_up_proj": "ffn_gate_up_exps.weight",
        "mlp.experts.down_proj": "ffn_down_exps.weight",
        "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight",
        "mlp.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
        "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.compressor.position_bias": "attn_compressor_ape.weight",
        "self_attn.compressor.kv_proj.weight": "attn_compressor_kv.weight",
        "self_attn.compressor.gate_proj.weight": "attn_compressor_gate.weight",
        "self_attn.compressor.kv_norm.weight": "attn_compressor_norm.weight",
        "self_attn.compressor.indexer.position_bias": "indexer_compressor_ape.weight",
        "self_attn.compressor.indexer.kv_proj.weight": "indexer_compressor_kv.weight",
        "self_attn.compressor.indexer.gate_proj.weight": "indexer_compressor_gate.weight",
        "self_attn.compressor.indexer.kv_norm.weight": "indexer_compressor_norm.weight",
        "self_attn.compressor.indexer.q_b_proj.weight": "indexer.attn_q_b.weight",
        "self_attn.compressor.indexer.scorer.weights_proj.weight": "indexer.proj.weight",
    }
    for kind in ["attn", "ffn"]:
        for field in ["fn", "base", "scale"]:
            parts[f"{kind}_hc.{field}"] = f"hc_{kind}_{field}.weight"
    for name, t in model.state_dict().items():
        if name in roots:
            target = roots[name]
        else:
            _, _, layer, suffix = name.split(".", 3)
            target = f"blk.{layer}." + parts[suffix]
        data = t.detach().cpu().numpy()
        if data.dtype == np.int64:
            data = data.astype(np.int32)
        w.add_tensor(target, np.ascontiguousarray(data))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
