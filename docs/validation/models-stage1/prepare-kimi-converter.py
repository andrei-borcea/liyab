import pathlib,shutil,sys
src=pathlib.Path.home()/'Library/Caches/liyab-dev/llama.cpp'
dst=pathlib.Path(sys.argv[1]);dst.mkdir(exist_ok=True)
shutil.copy(src/'convert_hf_to_gguf.py',dst/'convert_hf_to_gguf.py')
for name in ('conversion','gguf-py'):
    shutil.copytree(src/name,dst/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
p=dst/'conversion/qwen.py';s=p.read_text()
needle='        from transformers.convert_slow_tokenizer import bytes_to_unicode\n'
assert needle in s
s=s.replace('import json\n', 'import json\nfrom functools import lru_cache\nfrom transformers.convert_slow_tokenizer import bytes_to_unicode\n\n# Cache the pure fixed mapping; no vocabulary or tensor changes.\nbytes_to_unicode = lru_cache(maxsize=1)(bytes_to_unicode)\n',1).replace(needle,'',1)
p.write_text(s)
