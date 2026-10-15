// Liyab — GGML block layouts of the low-bit formats (ggml-common.h), shared
// by the NEON kernels (quant_lowbit.cpp) and the AVX2 ones (quant_x86.cpp).
// Internal. Rows of these blocks are read straight from the mapped GGUF;
// every block size is even, so the uint16 members are always 2-byte aligned.
#ifndef LIYAB_CORE_QUANT_BLOCKS_H
#define LIYAB_CORE_QUANT_BLOCKS_H

#include <cstdint>

namespace liyab::quant {

struct BlockQ2K {
    uint8_t scales[16];  // low nibble: scale, high nibble: min, per 16 values
    uint8_t qs[64];
    uint16_t d, dmin;
};
struct BlockQ3K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
};
struct BlockTQ2_0 {
    uint8_t qs[64];
    uint16_t d;
};
struct BlockIq2Xxs {
    uint16_t d;
    uint16_t qs[32];
};
struct BlockIq2Xs {
    uint16_t d;
    uint16_t qs[32];
    uint8_t scales[8];
};
struct BlockIq2S {
    uint16_t d;
    uint8_t qs[64];  // 32 grid indices (low 8 bits), then 32 sign bytes
    uint8_t qh[8];
    uint8_t scales[8];
};
struct BlockIq3Xxs {
    uint16_t d;
    uint8_t qs[96];  // 64 grid indices, then 8 x uint32 scales-and-signs
};
struct BlockIq3S {
    uint16_t d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t signs[32];
    uint8_t scales[4];
};
struct BlockIq1S {
    uint16_t d;
    uint8_t qs[32];
    uint16_t qh[8];
};
struct BlockIq1M {
    uint8_t qs[32];
    uint8_t qh[16];
    uint8_t scales[8];  // 4 x uint16; the fp16 super-block scale is spread over their top nibbles
};
struct BlockIq4Nl {
    uint16_t d;
    uint8_t qs[16];
};
struct BlockIq4Xs {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct BlockMxfp4 {
    uint8_t e;
    uint8_t qs[16];
};
static_assert(sizeof(BlockQ2K) == 84 && sizeof(BlockQ3K) == 110 && sizeof(BlockTQ2_0) == 66);
static_assert(sizeof(BlockIq2Xxs) == 66 && sizeof(BlockIq2Xs) == 74 && sizeof(BlockIq2S) == 82);
static_assert(sizeof(BlockIq3Xxs) == 98 && sizeof(BlockIq3S) == 110);
static_assert(sizeof(BlockIq1S) == 50 && sizeof(BlockIq1M) == 56);
static_assert(sizeof(BlockIq4Nl) == 18 && sizeof(BlockIq4Xs) == 136 && sizeof(BlockMxfp4) == 17);

}  // namespace liyab::quant

#endif  // LIYAB_CORE_QUANT_BLOCKS_H
