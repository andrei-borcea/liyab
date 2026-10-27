"""Populate a tiny embedding checkpoint's declared identity pooling projections."""
import json, math, shutil
from pathlib import Path
import torch
from safetensors.torch import save_file
source=Path('$SECTION_WORKTREES/agent-a043cbd90df96540d/build/models/embeddinggemma'); target=Path('$INTEGRATION/build/stage1-conformance/gemma-next/embeddinggemma')
mods=json.loads((source/'modules.json').read_text()); (target/'modules.json').write_text(json.dumps(mods))
generator=torch.Generator().manual_seed(2345)
width=json.loads((target/'config.json').read_text())['hidden_size']
for mod in mods:
 kind=mod['type'].rsplit('.',1)[-1]
 if kind not in ['Pooling','Dense']:continue
 dest=target/mod['path'];dest.mkdir(exist_ok=True)
 config=json.loads((source/mod['path']/'config.json').read_text())
 if kind=='Pooling':config['word_embedding_dimension']=width
 else:
  config['in_features']=width
  out=config['out_features']
  weights=torch.randn((out,width),generator=generator)/math.sqrt(width)
  save_file({'linear.weight':weights},str(dest/'model.safetensors'))
  width=out
 (dest/'config.json').write_text(json.dumps(config))
shutil.copyfile(source/'config_sentence_transformers.json',target/'config_sentence_transformers.json')
