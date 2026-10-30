"""Reference decoding for the test-vector generators (gen_quant_vectors.py, gen_ssm_quant_vectors.py).

dequantize() is gguf-py's decoder. For a type gguf-py names but cannot decode yet (Q1_0 in gguf 0.19.0) it calls
llama.cpp's own dequantize_row_<type> from libggml, found through the GGML_LIB environment variable or the system
library path (a llama.cpp install, e.g. Homebrew's). The expected values in tests/data are thereby always
produced by an implementation other than Liyab's.
"""
import ctypes
import ctypes.util
import os

import numpy as np
from gguf import GGML_QUANT_SIZES, quants

_ggml = None


def _load_ggml():
    global _ggml
    if _ggml is None:
        candidates = [os.environ.get("GGML_LIB"), ctypes.util.find_library("ggml-base"),
                      "/opt/homebrew/lib/libggml-base.dylib", "/usr/local/lib/libggml-base.dylib",
                      "/usr/local/lib/libggml-base.so", "/usr/lib/libggml-base.so"]
        for path in filter(None, candidates):
            try:
                _ggml = ctypes.CDLL(path)
                break
            except OSError:
                continue
        else:
            raise SystemExit("this type needs llama.cpp's libggml-base to decode: install llama.cpp or set GGML_LIB")
    return _ggml


def dequantize(data, qtype):
    """Float32 values of the blocks in `data` (flat uint8 array) of GGML type `qtype`."""
    try:
        return quants.dequantize(data, qtype)
    except NotImplementedError:
        pass
    block_size, block_bytes = GGML_QUANT_SIZES[qtype]
    function = getattr(_load_ggml(), "dequantize_row_" + qtype.name.lower(), None)
    if function is None:
        raise SystemExit(f"neither gguf-py nor libggml can decode {qtype.name}")
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    function.restype = None
    blocks = np.ascontiguousarray(data, dtype=np.uint8)
    assert blocks.size % block_bytes == 0
    out = np.empty(blocks.size // block_bytes * block_size, dtype=np.float32)
    function(blocks.ctypes.data, out.ctypes.data, out.size)
    return out
