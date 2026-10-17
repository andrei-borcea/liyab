// Liyab — encodings of KV chunks on storage (internal).
//
// What storage limits is how many contexts fit under the prefix cache's cap
// (EngineConfig::prefix_cache_disk_mb), not how fast they come back: so the
// KV store (core/kv_store.h) writes a chunk's rows encoded, and decodes them
// when the chunk is read back. Callers of the store only ever see the rows as
// KvCache::read_rows() writes them.
//
// Lossless (KvEncoding::Lossless, the default). Measured on real chunks of
// Qwen3-0.6B, Qwen3.5-0.8B and Llama-3.2-1B, the bytes of a KV page are close
// to random except for one part: the exponents of the 16-bit floats. The low
// byte of an F16 value (its low 8 mantissa bits) has 8.0 bits of entropy per
// byte; the high byte (sign, 5-bit exponent, 2 mantissa bits) 6.2, and a
// channel's exponents stay close to the largest one it takes in the chunk. A
// Q8_0 block's 32 quants carry 7.5 bits per byte, its F16 scale behaves like
// the F16 values above. So, per run (one layer's K or V rows of the chunk):
//  * every 16-bit float (F16 values, block scales and minimums) is split
//    into its low byte, stored as it is, and one symbol for its high byte:
//    how far its exponent lies below the largest exponent of its channel in
//    the run (stored once per channel), with its sign and top 2 mantissa
//    bits;
//  * those symbols, and the quant bytes of block formats, are entropy-coded
//    by static rANS (range asymmetric numeral systems: one 12-bit frequency
//    table per kind of symbol and plane, kept in the chunk; 4 interleaved
//    32-bit states, 16-bit renormalization, branchless refills), our own
//    implementation;
//  * with one value per quant byte (Q8_0), a channel that changes less from
//    one position to the next than it is large (the slowly turning RoPE
//    pairs of keys: 7.4 -> 7.0 bits per byte on Qwen3-0.6B, 7.5 -> 6.7 on
//    Llama 3.2 1B) is coded as those differences, with a table of its own;
//  * a stream is coded only where it pays on the device: when the read time
//    its smaller size saves exceeds the time decoding it takes, with read and
//    decode speeds the store measures once (KvCodecCosts); elsewhere it is
//    kept raw, and a chunk where nothing pays is stored raw;
//  * runs are independent, so a thread pool encodes and decodes them in
//    parallel.
// Order-0 coding with the per-channel exponent reference reached the measured
// entropy within a percent; conditioning quants on their channel's magnitude
// gained about 1% more and was left out. F16 rows come out ~1.17x smaller,
// Q8_0 rows 1.04-1.09x.
//
// Lossy (KvEncoding::Lossy4, opt-in: KvStore::Config::lossy). Rows are
// re-quantized to 4 bits on their way to storage:
//  * keys channel by channel: each (layer, KV head, channel) gets its own
//    minimum and step over the chunk's positions (key scaling: the few large,
//    nearly constant key channels of some models keep their own range instead
//    of coarsening the small channels sharing a row block), 4-bit codes;
//  * values row block by row block, as Q4_1 (scale and minimum per 32
//    channels of one position);
//  * rows come back in the cache's type (F16 converted, Q8_0 quantized
//    again). Q4_0 / Q4_1 caches are already 4-bit: they are stored lossless.
// It is never applied unless the engine asks for it, and the store's seed
// then differs, so a lossy chunk is never served to an engine that did not
// opt in (core/kv_store.h). Its cost is measured with liyab-kl
// (--store-codec) and documented in the README.
//
// An encoding returns false when it does not apply or does not pay (the
// encoded chunk would not be smaller): the store then keeps the rows raw.
// Decoding checks every size and offset against the layout and the rANS
// states against their initial value, so a malformed payload fails cleanly
// (the store also checks the payload's hash before decoding).
//
// Thread-compatible: separate calls may run concurrently on different
// buffers; a call uses `pool` (when given) for its own parallel work, so one
// call at a time per pool.
#ifndef LIYAB_CORE_KV_CODEC_H
#define LIYAB_CORE_KV_CODEC_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "liyab/kv_cache.h"
#include "liyab/types.h"

namespace liyab {

class ThreadPool;

// How a chunk's payload is stored on storage (FileHeader::encoding in
// core/kv_store.cpp). Values are part of the file format: never renumber.
enum class KvEncoding : uint32_t {
    Raw = 0,       // the rows as KvCache::read_rows() writes them
    Lossless = 1,  // exponent-referenced byte planes and rANS (see above)
    Lossy4 = 2,    // 4-bit re-quantization with per-channel key ranges (see above)
};

// The geometry of a chunk image (KvCache::rows_bytes): [plane][layer][slot]
// [kv_head][row], a row being head_dim values of `type`.
struct KvRowLayout {
    KvCacheType type = KvCacheType::F16;
    int32_t planes = 0;  // 2 (K, then V); 1 when values live in the keys' rows
    int32_t layers = 0;
    int32_t heads = 0;   // KV heads
    int32_t head_dim = 0;

    // Planes, layers, heads and head_dim positive, head_dim a multiple of 32
    // for block types.
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] size_t row_bytes() const noexcept;
    // Bytes of an image of `slots` positions.
    [[nodiscard]] size_t image_bytes(int32_t slots) const noexcept;
};

// The layout of a cache's chunk images (KvCache::read_rows).
KvRowLayout kv_row_layout(const KvCacheConfig& config) noexcept;

// What reading and decoding cost on this device, for the lossless
// encoding's choice of what to code (see kv_encode()): measured once per
// device and layout by the KV store (core/kv_store.h).
struct KvCodecCosts {
    double read_ns_per_byte = 0;      // reading a chunk file from storage
    double decode_ns_per_symbol = 0;  // decoding a coded symbol, wall clock on the pool used
};

// Encodes `image` (an image of `slots` positions laid out as `layout` says)
// with `encoding` into `out` (replaced). False when the encoding does not
// apply to the layout (Raw, an invalid layout, Lossy4 on a 4-bit cache) or
// would not make the chunk smaller; `out` is then unspecified. Lossless:
// with `costs`, a stream is coded only when the read time its smaller size
// saves exceeds its decode time (else kept raw, so a chunk where nothing
// pays is not encoded at all); without, whenever coding makes it smaller.
bool kv_encode(KvEncoding encoding, const KvRowLayout& layout, int32_t slots, std::span<const uint8_t> image,
               std::vector<uint8_t>& out, ThreadPool* pool = nullptr, const KvCodecCosts* costs = nullptr);

// A store chunk of several KV pages (core/kv_store.h): `pages` are the page
// images in order (KvCache::read_rows, `page_slots` positions each but a
// partial last one). Each page is encoded on its own as kv_encode() does,
// or kept raw where that does not apply or pay: u32 page count, then per
// page a u8 (1: encoded, 0: raw) and its u32 size, then the pages. False
// when no page is encoded (the chunk is then stored raw).
bool kv_encode_chunk(KvEncoding encoding, const KvRowLayout& layout, int32_t page_slots,
                     std::span<const std::span<const uint8_t>> pages, std::vector<uint8_t>& out,
                     ThreadPool* pool = nullptr, const KvCodecCosts* costs = nullptr);
// Decodes a chunk of `slots` positions written by kv_encode_chunk() into
// `out`, its rows as parts in any partition (the cache's pages in place, or
// scratch). False when it is malformed.
bool kv_decode_chunk(KvEncoding encoding, const KvRowLayout& layout, int32_t page_slots, int32_t slots,
                     std::span<const uint8_t> in, std::span<const std::span<uint8_t>> out, ThreadPool* pool = nullptr);

// Wall-clock nanoseconds the lossless decoder takes per coded symbol for
// chunks of `layout` and `slots` positions on `pool` (best of 3 on a
// synthetic chunk); 0 when the layout is invalid.
double kv_calibrate_decode(const KvRowLayout& layout, int32_t slots, ThreadPool* pool);

// Decodes `in`, written by kv_encode() with the same encoding, layout and
// slots, into `out` (image_bytes(slots) bytes). False when it is malformed;
// `out` is then unspecified.
bool kv_decode(KvEncoding encoding, const KvRowLayout& layout, int32_t slots, std::span<const uint8_t> in,
               std::span<uint8_t> out, ThreadPool* pool = nullptr);

}  // namespace liyab

#endif  // LIYAB_CORE_KV_CODEC_H
