"""Run an official legacy HF implementation with transformers 5.19 API compatibility.

Only import/type bookkeeping and checkpoint materialization change. Model
forward functions and tensor values are not patched. Keep this runner in
the tiny-conformance recipe for cases that require it.
"""
import runpy
import sys
import os
from pathlib import Path
from typing import TypedDict

os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / "build/modern-hf-modules"))
import torch
import transformers.modeling_utils as modeling
import transformers.utils as utils
import transformers.utils.import_utils as imports

if not hasattr(imports, "is_torch_fx_available"):
    imports.is_torch_fx_available = lambda: False
if not hasattr(utils, "LossKwargs"):
    class LossKwargs(TypedDict, total=False):
        pass
    utils.LossKwargs = LossKwargs

# Older remote classes declare tied-key names as a list; 5.19's exporter
# expects a mapping. Its values are unused when collecting those names.
def tied_keys(module):
    names = []
    for name, child in module.named_modules():
        tied = getattr(child, "_tied_weights_keys", {}) or {}
        names.extend(f"{name}.{key}" if name else key for key in tied)
    return names
modeling._get_tied_weight_keys = tied_keys

# Old remote RoPE classes cache nonpersistent sin/cos buffers in __init__.
# Meta initialization loses those values when the checkpoint lacks them.
# Construct on CPU so the authors' buffer constructors run normally.
original_context = modeling.PreTrainedModel.get_init_context.__func__
def cpu_context(cls, *args, **kwargs):
    return [ctx for ctx in original_context(cls, *args, **kwargs)
            if not (isinstance(ctx, torch.device) and ctx.type == "meta")]
modeling.PreTrainedModel.get_init_context = classmethod(cpu_context)

# Legacy official code uses the singular spelling from transformers 4.
import transformers.masking_utils as masking
import inspect
for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
    function = getattr(masking, name, None)
    if function is not None and "inputs_embeds" in inspect.signature(function).parameters:
        def compatible_mask(*args, _function=function, **kwargs):
            if "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            if "cache_position" not in inspect.signature(_function).parameters:
                kwargs.pop("cache_position", None)
            return _function(*args, **kwargs)
        setattr(masking, name, compatible_mask)

# transformers 5 moved default RoPE initialization out of the public registry.
# This is the transformers 4.44 formula retained by the official NAS source.
import transformers.modeling_rope_utils as ropes
if "default" not in ropes.ROPE_INIT_FUNCTIONS:
    def default_rope(config=None, device=None, seq_len=None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("config and rope kwargs are mutually exclusive")
        if kwargs:
            base, dim = kwargs["base"], kwargs["dim"]
        else:
            base = config.rope_theta
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)), 1.0
    ropes.ROPE_INIT_FUNCTIONS["default"] = default_rope

# Legacy NAS generation registration; teacher-forced conformance never uses it.
import transformers.generation.utils as generation
if not hasattr(generation, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
    generation.NEED_SETUP_CACHE_CLASSES_MAPPING = {}

sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
runpy.run_path(sys.argv[0], run_name="__main__")
