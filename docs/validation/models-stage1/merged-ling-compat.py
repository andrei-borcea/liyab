# Compatibility for an import helper removed after transformers 4; no model math changes.
import runpy, sys
import transformers.utils.import_utils as imports
imports.is_torch_fx_available = lambda: True
import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
def old_default(config,device=None,seq_len=None,**kwargs):
    dim=int((config.head_dim or config.hidden_size//config.num_attention_heads)*getattr(config,"partial_rotary_factor",1.0))
    return 1.0/(config.rope_theta**(torch.arange(0,dim,2,dtype=torch.float32,device=device)/dim)),1.0
ROPE_INIT_FUNCTIONS.setdefault("default",old_default)
import transformers.modeling_utils as modeling
def old_tied(module):
    result=[]
    for name,submodule in module.named_modules():
        tied=getattr(submodule,"_tied_weights_keys",{}) or {}
        keys=tied.keys() if isinstance(tied,dict) else tied
        result.extend(f"{name}.{key}" if name else key for key in keys)
    return result
modeling._get_tied_weight_keys=old_tied
# Transformers 5 materializes nonpersistent buffers after meta loading; the old
# model's _init_weights does not rebuild RoPE. Restore its own constructor rule.
from transformers import PreTrainedModel
original_from_pretrained = PreTrainedModel.from_pretrained.__func__
def old_from_pretrained(cls, *args, **kwargs):
    model = original_from_pretrained(cls, *args, **kwargs)
    for module in model.modules():
        if "inv_freq" in module._non_persistent_buffers_set and hasattr(module, "rope_init_fn"):
            module.inv_freq, module.attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
            module.original_inv_freq = module.inv_freq
    return model
PreTrainedModel.from_pretrained = classmethod(old_from_pretrained)
script = sys.argv.pop(1)
from pathlib import Path
sys.path.insert(0, str(Path(script).resolve().parent))
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
