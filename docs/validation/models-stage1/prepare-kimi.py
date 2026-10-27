"""Preserve values while restoring the official checkpoint's dense-MLP names for b11819."""
import json,pathlib,shutil,sys
from safetensors.torch import load_file,save_file
p=pathlib.Path(sys.argv[1])
for f in p.glob('*.safetensors'):
    data=load_file(str(f));out={}
    for key,value in data.items():
        for part in ('gate_proj','up_proj','down_proj'):
            key=key.replace(f'.block_sparse_moe.{part}.',f'.mlp.{part}.')
        out[key]=value
    save_file(out,str(f),metadata={'format':'pt'})
config=json.loads((p/'config.json').read_text());config.pop('auto_map',None)
(p/'config.json').write_text(json.dumps(config,indent=1)+'\n')
shutil.copy(pathlib.Path(__file__).with_name('tokenization_kimi.py'),p/'tokenization_kimi.py')
