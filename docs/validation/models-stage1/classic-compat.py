"""Legacy reference compatibility only; leaves Orion's tensor arithmetic unchanged."""
import runpy, sys
import transformers.utils
import transformers.modeling_utils as modeling_utils
from transformers import AutoConfig, AutoModelForCausalLM
# Renamed optional CUDA capability check; flash_attn is absent here.
transformers.utils.is_flash_attn_available = lambda: False
load_config=AutoConfig.from_pretrained
def compatible_config(*args,**kwargs):
 config=load_config(*args,**kwargs)
 # transformers 5 rewrites the legacy None to a default rope_parameters
 # object. The author's old implementation expects None for no scaling.
 rope=getattr(config,'rope_scaling',None)
 if isinstance(rope,dict) and rope.get('rope_type')=='default' and 'type' not in rope:
  config.rope_scaling=None
 return config
AutoConfig.from_pretrained=staticmethod(compatible_config)
build_model=AutoModelForCausalLM.from_config
def compatible_model(config,*args,**kwargs):
 rope=getattr(config,'rope_scaling',None)
 if isinstance(rope,dict) and rope.get('rope_type')=='default' and 'type' not in rope:
  config.rope_scaling=None
 return build_model(config,*args,**kwargs)
AutoModelForCausalLM.from_config=staticmethod(compatible_model)
get_tied=modeling_utils._get_tied_weight_keys
def compatible_ties(model):
 for module in model.modules():
  if isinstance(getattr(module,'_tied_weights_keys',None),list):
   seen={}; aliases={}
   for name,param in module.named_parameters(remove_duplicate=False):
    if id(param) in seen:aliases[name]=seen[id(param)]
    else:seen[id(param)]=name
   module._tied_weights_keys=aliases
 return get_tied(model)
modeling_utils._get_tied_weight_keys=compatible_ties
# Transformers5 meta loading leaves this legacy implementation's nonpersistent
# RoPE buffers uninitialized. Its original constructor plus strict state loading
# retains the author's own cache initialization, with no arithmetic changes.
import os, torch
from safetensors.torch import load_file
load_model=AutoModelForCausalLM.from_pretrained
def compatible_pretrained(path,*args,**kwargs):
 if kwargs.get('trust_remote_code') and os.path.isfile(os.path.join(path,'model.safetensors')):
  config=kwargs.get('config') or compatible_config(path,trust_remote_code=True)
  model=compatible_model(config,trust_remote_code=True,dtype=kwargs.get('dtype',torch.float32))
  model.load_state_dict(load_file(os.path.join(path,'model.safetensors')),strict=True)
  return model.eval()
 return load_model(path,*args,**kwargs)
AutoModelForCausalLM.from_pretrained=staticmethod(compatible_pretrained)
entry=sys.argv.pop(1)
from pathlib import Path
sys.path.insert(0, str(Path(entry).resolve().parent))
runpy.run_path(entry,run_name='__main__')
