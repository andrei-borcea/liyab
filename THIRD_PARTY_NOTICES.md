# Third-party notices

Liyab includes code adapted from the following projects.

## llama.cpp / ggml

* Source: <https://github.com/ggml-org/llama.cpp>
* Used in: `src/core/quant_lowbit.cpp` (ARM NEON dot-product kernels for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, TQ2_0, IQ1_S, IQ1_M,
  IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL and MXFP4, and the Q8_K activation quantization,
  adapted from `ggml/src/ggml-cpu/arch/arm/quants.c` and `ggml/src/ggml-quants.c`). The I-quant lookup grids in
  `src/core/quant_tables.inc` are generated from gguf-py, llama.cpp's Python package, by
  `tools/gen_quant_tables.py`.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
