"""b11819 converter shim: retain native SentencePiece scores when its model is present.

Only the Python vocabulary source changes. The tensor conversion is unmodified.
The source DeciLM-7B has tokenizer.model, but LlamaHfVocab currently drops its scores.
"""
import pathlib,sys
converter=pathlib.Path.home()/'Library/Caches/liyab-dev/llama.cpp'
sys.path.insert(0,str(converter))
from conversion.deci import DeciModel
from conversion.base import TextModel
original=DeciModel.set_vocab
def native_sentencepiece(self):
    if self.hparams.get('vocab_size',128256)!=128256 and (self.dir_model/'tokenizer.model').is_file():
        return TextModel._set_vocab_sentencepiece(self)
    return original(self)
DeciModel.set_vocab=native_sentencepiece
import convert_hf_to_gguf
convert_hf_to_gguf.main()
