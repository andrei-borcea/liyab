#include "core/kv_codec.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>

#include "core/quant.h"
#include "core/thread_pool.h"

namespace liyab {

namespace {

// --- Geometry ------------------------------------------------------------------

// A row of a cache type as blocks of `values` values: `halves` 16-bit floats
// (F16 values; block scales and minimums) then `qbytes` quant bytes.
struct Blocks {
    int32_t values = 1;
    int32_t halves = 1;
    int32_t qbytes = 0;
    [[nodiscard]] size_t bytes() const noexcept { return 2 * static_cast<size_t>(halves) + static_cast<size_t>(qbytes); }
};

Blocks blocks_of(KvCacheType type) noexcept {
    switch (type) {
        case KvCacheType::F16: return {1, 1, 0};
        case KvCacheType::Q8_0: return {quant::kBlock, 1, quant::kBlock};
        case KvCacheType::Q4_0: return {quant::kBlock, 1, quant::kBlock / 2};
        case KvCacheType::Q4_1: return {quant::kBlock, 2, quant::kBlock / 2};
    }
    return {};
}

DType dtype_of(KvCacheType type) noexcept {
    switch (type) {
        case KvCacheType::F16: return DType::F16;
        case KvCacheType::Q8_0: return DType::Q8_0;
        case KvCacheType::Q4_0: return DType::Q4_0;
        case KvCacheType::Q4_1: return DType::Q4_1;
    }
    return DType::F16;
}

// One run: one layer's K (or V) rows of a chunk, contiguous in the image.
struct Runs {
    Blocks b;
    int32_t count = 0;          // planes * layers
    int32_t slots = 0;
    size_t blocks = 0;          // per run: slots * heads * blocks per row
    size_t channels = 0;        // 16-bit floats per position and run: heads * blocks per row * halves
    size_t bytes = 0;           // per run
};

bool runs_of(const KvRowLayout& layout, int32_t slots, Runs& r) {
    if (!layout.valid() || slots <= 0) return false;
    r.b = blocks_of(layout.type);
    r.count = layout.planes * layout.layers;
    r.slots = slots;
    const size_t per_row = static_cast<size_t>(layout.head_dim / r.b.values);
    r.blocks = static_cast<size_t>(slots) * static_cast<size_t>(layout.heads) * per_row;
    r.channels = static_cast<size_t>(layout.heads) * per_row * static_cast<size_t>(r.b.halves);
    r.bytes = r.blocks * r.b.bytes();
    return true;
}

// Runs [0, n) through `pool` when there is one, else in order.
void for_each(ThreadPool* pool, int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    if (pool != nullptr && n > 1) {
        pool->parallel_for(n, fn);
    } else if (n > 0) {
        fn(0, n);
    }
}

uint16_t load16(const uint8_t* p) noexcept {
    uint16_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

void store16(uint8_t* p, uint16_t v) noexcept { std::memcpy(p, &v, sizeof v); }

template <typename T>
void put(std::vector<uint8_t>& out, T v) {
    const size_t at = out.size();
    out.resize(at + sizeof v);
    std::memcpy(out.data() + at, &v, sizeof v);
}

// Reads a T at `at` of `in`, advancing `at`; false past the end.
template <typename T>
bool get(std::span<const uint8_t> in, size_t& at, T& v) {
    if (at > in.size() || in.size() - at < sizeof v) return false;
    std::memcpy(&v, in.data() + at, sizeof v);
    at += sizeof v;
    return true;
}

// --- Static rANS -------------------------------------------------------------------
// Byte symbols, frequencies normalized to kScale; 32-bit states kept in
// [kLow, kLow << 16), renormalized 16 bits at a time; kStates states take
// symbols in turn (symbol i: state i % kStates), so a decoder's steps are
// independent of each other's latency. The encoder runs backwards from the
// initial state kLow; the decoder must end there, which checks the stream.
// Symbol i of a stream uses table i % period of a periodic pattern (the
// tables of a position's channels).

constexpr int kScaleBits = 12;
constexpr uint32_t kScale = 1u << kScaleBits;
constexpr uint32_t kLow = 1u << 16;
constexpr int kStates = 4;

struct Freqs {
    uint16_t freq[256] = {};
    uint16_t cum[256] = {};
};

// Frequencies summing to kScale, every symbol that occurs keeping at least 1.
void normalize(const uint64_t* counts, Freqs& f) {
    uint64_t total = 0;
    for (int s = 0; s < 256; ++s) total += counts[s];
    if (total == 0) return;
    int32_t sum = 0;
    for (int s = 0; s < 256; ++s) {
        if (counts[s] == 0) continue;
        const double share = static_cast<double>(counts[s]) * kScale / static_cast<double>(total);
        f.freq[s] = static_cast<uint16_t>(std::max<int64_t>(1, std::llround(share)));
        sum += f.freq[s];
    }
    // Rounding leaves the sum off by a little: the most frequent symbols absorb it.
    int order[256];
    for (int s = 0; s < 256; ++s) order[s] = s;
    std::sort(order, order + 256, [&](int a, int b) { return f.freq[a] > f.freq[b]; });
    while (sum != static_cast<int32_t>(kScale)) {
        for (int i = 0; i < 256 && sum != static_cast<int32_t>(kScale); ++i) {
            uint16_t& q = f.freq[order[i]];
            if (sum < static_cast<int32_t>(kScale)) {
                if (q == 0) break;
                ++q;
                ++sum;
            } else if (q > 1) {
                --q;
                --sum;
            }
        }
    }
    uint16_t c = 0;
    for (int s = 0; s < 256; ++s) {
        f.cum[s] = c;
        c = static_cast<uint16_t>(c + f.freq[s]);
    }
}

// Table: a 32-byte mask of the symbols present, then each one's frequency (u16).
void write_freqs(std::vector<uint8_t>& out, const Freqs& f) {
    uint8_t mask[32] = {};
    for (int s = 0; s < 256; ++s) {
        if (f.freq[s] != 0) mask[s / 8] = static_cast<uint8_t>(mask[s / 8] | (1u << (s % 8)));
    }
    out.insert(out.end(), mask, mask + sizeof mask);
    for (int s = 0; s < 256; ++s) {
        if (f.freq[s] != 0) put<uint16_t>(out, f.freq[s]);
    }
}

// Decoding table, by the state's low kScaleBits (its slot): step[slot] =
// freq | (slot - cum) << 16, so a step is one multiply-add, and the symbol
// apart. Zero-initialized: a table no stream should use still decodes within
// bounds (to garbage the final state check rejects).
struct DecodeTable {
    uint32_t step[kScale] = {};
    uint8_t symbol[kScale] = {};
};

// Reads a table: empty (no symbol) or summing to kScale.
bool read_freqs(std::span<const uint8_t> in, size_t& at, DecodeTable& t) {
    uint8_t mask[32];
    for (uint8_t& m : mask) {
        if (!get(in, at, m)) return false;
    }
    uint32_t cum = 0;
    for (int s = 0; s < 256; ++s) {
        if ((mask[s / 8] >> (s % 8) & 1) == 0) continue;
        uint16_t f = 0;
        if (!get(in, at, f) || f == 0 || cum + f > kScale) return false;
        for (uint32_t slot = cum; slot < cum + f; ++slot) {
            t.step[slot] = static_cast<uint32_t>(f) | (slot - cum) << 16;
            t.symbol[slot] = static_cast<uint8_t>(s);
        }
        cum += f;
    }
    return cum == kScale || cum == 0;
}

// Encodes syms[0, n), symbol i with tables[i % period], into 16-bit words.
void rans_encode(const Freqs* const* tables, size_t period, const uint8_t* syms, size_t n,
                 std::vector<uint16_t>& words) {
    words.clear();
    if (n == 0) return;
    // At most one word per symbol, then the final states.
    words.resize(n + 2 * kStates);
    uint16_t* w = words.data();
    uint32_t x[kStates];
    for (uint32_t& s : x) s = kLow;
    size_t c = (n - 1) % period;
    auto put_symbol = [&](uint32_t s, size_t i) -> uint32_t {
        const Freqs& f = *tables[c];
        c = c == 0 ? period - 1 : c - 1;
        const uint8_t v = syms[i];
        const uint32_t freq = f.freq[v];
        // Branchless (whether a state sheds a word is close to a coin
        // toss): the word is always written, kept only when shed. The bound
        // is the largest state that stays below kLow << 16 once v is encoded.
        const uint32_t shed = s >= ((uint64_t{kLow} >> kScaleBits) << 16) * freq ? 1u : 0u;
        *w = static_cast<uint16_t>(s);
        w += shed;
        s = shed != 0 ? s >> 16 : s;
        const uint32_t q = s / freq;
        return (q << kScaleBits) + (s - q * freq) + f.cum[v];
    };
    // Backwards: the symbols past the last whole group of kStates first.
    size_t i = n;
    while (i % kStates != 0) {
        --i;
        x[i % kStates] = put_symbol(x[i % kStates], i);
    }
    while (i > 0) {
        i -= kStates;
        for (int j = kStates; j-- > 0;) x[j] = put_symbol(x[j], i + static_cast<size_t>(j));
    }
    // Read first by the decoder, state 0 first, high half first.
    for (int j = kStates; j-- > 0;) {
        *w++ = static_cast<uint16_t>(x[j]);
        *w++ = static_cast<uint16_t>(x[j] >> 16);
    }
    words.resize(static_cast<size_t>(w - words.data()));
    std::reverse(words.begin(), words.end());
}

// Decodes n symbols, symbol i with tables[i % period], from `words` (count
// 16-bit words, unaligned) into out. False when the stream is malformed: too
// short, too long, or not ending in the encoder's initial states.
bool rans_decode(const DecodeTable* const* tables, size_t period, const uint8_t* words, size_t count, size_t n,
                 uint8_t* __restrict out) {
    if (n == 0) return count == 0;
    if (count < 2 * kStates) return false;
    const uint8_t* p = words;
    const uint8_t* const end = words + 2 * count;
    uint32_t x[kStates];
    for (uint32_t& s : x) {
        s = static_cast<uint32_t>(load16(p)) << 16 | load16(p + 2);
        p += 4;
    }
    auto advance = [](const DecodeTable& t, uint32_t s) {
        const uint32_t e = t.step[s & (kScale - 1)];
        return (e & 0xFFFF) * (s >> kScaleBits) + (e >> 16);
    };
    size_t c = 0;
    const DecodeTable* t[kStates];
    size_t i = 0;
    // The states advance independently; only the word pointer is shared.
    // Groups of 4 states refill from one 64-bit load (at most 4 words), each
    // state taking the next word it needs, without branches (whether a state
    // renormalizes is close to a coin toss): no load waits on another state's
    // choice. A group of kStates symbols reads at most kStates words.
    for (; i + kStates <= n && end - p >= 2 * kStates; i += kStates) {
        if (period == 1) {
            for (int j = 0; j < kStates; ++j) t[j] = tables[0];
        } else {
            for (int j = 0; j < kStates; ++j) {
                t[j] = tables[c];
                c = c + 1 == period ? 0 : c + 1;
            }
        }
        for (int j = 0; j < kStates; ++j) out[i + static_cast<size_t>(j)] = t[j]->symbol[x[j] & (kScale - 1)];
        for (int j = 0; j < kStates; ++j) x[j] = advance(*t[j], x[j]);
        for (int g = 0; g < kStates; g += 4) {
            uint64_t next;
            std::memcpy(&next, p, sizeof next);  // little-endian: word k at bits 16 k
            uint32_t taken = 0;
            for (int j = g; j < g + 4; ++j) {
                const uint32_t more = x[j] < kLow ? 1u : 0u;
                const auto word = static_cast<uint32_t>(next >> (16 * taken) & 0xFFFF);
                taken += more;
                x[j] = more != 0 ? (x[j] << 16 | word) : x[j];
            }
            p += 2 * taken;
        }
    }
    // The rest, checking every read.
    bool ok = true;
    for (; i < n; ++i) {
        uint32_t& s = x[i % kStates];
        const DecodeTable& tab = *tables[c];
        c = c + 1 == period ? 0 : c + 1;
        out[i] = tab.symbol[s & (kScale - 1)];
        s = advance(tab, s);
        if (s < kLow) {
            if (end - p < 2) {
                ok = false;
                break;
            }
            s = s << 16 | load16(p);
            p += 2;
        }
    }
    for (const uint32_t s : x) ok = ok && s == kLow;
    return ok && p == end;
}

// --- Lossless ---------------------------------------------------------------------
// Per run, the positions' 16-bit floats and quants form two symbol streams:
//  * float symbols: a float's high byte relative to its channel's largest
//    exponent in the run (high_symbol); its low byte is stored as it is;
//  * quants (block types): the quant bytes; with one value per byte (Q8_0),
//    a channel whose values change less from one position to the next than
//    they are large (the slowly turning RoPE pairs of keys) is coded as those
//    differences instead (mod 256), with a table of its own.
// Payload: u32 runs; per plane the float-symbol table, then (block types)
// the quant table and (Q8_0) the difference table; per run the u32 word
// counts of its two streams; then each run: its channels' largest exponents,
// (Q8_0) a bit per quant channel set when it is coded as differences, the
// floats' low bytes, the float-symbol stream, the quant stream.

// A 16-bit float's high byte (sign, exponent, top 2 mantissa bits) as a symbol
// relative to its channel's largest exponent `top`, and back.
inline uint8_t high_symbol(uint8_t high, uint8_t top) noexcept {
    const uint32_t e = high >> 2 & 31;
    return static_cast<uint8_t>((top - e) << 3 | (high >> 7) << 2 | (high & 3));
}
inline uint8_t high_byte(uint8_t symbol, uint8_t top) noexcept {
    const uint32_t e = (top - (symbol >> 3)) & 31;
    return static_cast<uint8_t>((symbol >> 2 & 1) << 7 | e << 2 | (symbol & 3));
}

// A stream as stored: rANS-coded, or its symbols as they are. With `costs`
// (the device's, KvCodecCosts), coded only when the read time its smaller
// size saves exceeds the time decoding it takes, so compression never makes
// a restore slower; without, whenever it is smaller. `field`: the 16-bit
// word count of a coded stream, kRawStream for a raw one.
constexpr uint32_t kRawStream = 0xFFFFFFFFu;

struct Stream {
    uint32_t field = 0;
    std::vector<uint8_t> bytes;
};

bool pays(size_t coded_bytes, size_t n, const KvCodecCosts* costs) {
    if (coded_bytes >= n) return false;
    if (costs == nullptr) return true;
    return static_cast<double>(n - coded_bytes) * costs->read_ns_per_byte > static_cast<double>(n) * costs->decode_ns_per_symbol;
}

Stream make_stream(const std::vector<uint16_t>& words, const uint8_t* syms, size_t n, const KvCodecCosts* costs) {
    Stream s;
    if (!words.empty() && pays(2 * words.size(), n, costs)) {
        s.field = static_cast<uint32_t>(words.size());
        s.bytes.resize(2 * words.size());
        if (!words.empty()) std::memcpy(s.bytes.data(), words.data(), s.bytes.size());
    } else {
        s.field = kRawStream;
        s.bytes.assign(syms, syms + n);
    }
    return s;
}

size_t stream_bytes(uint32_t field, size_t n) { return field == kRawStream ? n : 2 * static_cast<size_t>(field); }

bool decode_stream(const DecodeTable* const* tables, size_t period, const uint8_t* bytes, uint32_t field, size_t n,
                   uint8_t* out) {
    if (field == kRawStream) {
        if (n > 0) std::memcpy(out, bytes, n);
        return true;
    }
    return rans_decode(tables, period, bytes, field, n, out);
}

// The sizes of one run's parts.
struct RunParts {
    size_t floats = 0;    // 16-bit floats (and float symbols)
    size_t channels = 0;  // float channels: floats per position
    size_t quants = 0;    // quant bytes
    size_t qchannels = 0; // quant bytes per position
    size_t mode_bytes = 0;  // difference bits of the quant channels (Q8_0), packed
    bool differences = false;
};

RunParts parts_of(const Runs& r) {
    RunParts p;
    p.floats = r.blocks * static_cast<size_t>(r.b.halves);
    p.channels = r.channels;
    p.quants = r.blocks * static_cast<size_t>(r.b.qbytes);
    p.qchannels = p.quants / static_cast<size_t>(r.slots);
    p.differences = r.b.qbytes == r.b.values;  // one value per byte
    p.mode_bytes = p.differences ? (p.qchannels + 7) / 8 : 0;
    return p;
}

bool encode_lossless(const KvRowLayout& layout, const Runs& r, std::span<const uint8_t> image,
                     std::vector<uint8_t>& out, ThreadPool* pool, const KvCodecCosts* costs) {
    const Blocks& b = r.b;
    const auto runs = static_cast<size_t>(r.count);
    const RunParts rp = parts_of(r);
    const auto slots = static_cast<size_t>(r.slots);
    const size_t row_blocks = r.channels / static_cast<size_t>(b.halves);  // blocks per position
    const auto halves = static_cast<size_t>(b.halves);
    const auto qbytes = static_cast<size_t>(b.qbytes);
    // Pass 1, per run: exponents, difference choices, symbols and their counts.
    struct Run {
        std::vector<uint8_t> top, mode, low, fsym, qsym;
        uint64_t counts[3][256] = {};  // float symbols, quants, differences
    };
    std::vector<Run> work(runs);
    for_each(pool, static_cast<int64_t>(runs), [&](int64_t first, int64_t last) {
        for (auto run = static_cast<size_t>(first); run < static_cast<size_t>(last); ++run) {
            const uint8_t* base = image.data() + run * r.bytes;
            Run& w = work[run];
            w.top.assign(rp.channels, 0);
            w.low.resize(rp.floats);
            w.fsym.resize(rp.floats);
            w.qsym.resize(rp.quants);
            for (size_t s = 0, k = 0; s < slots; ++s) {
                for (size_t j = 0; j < row_blocks; ++j, ++k) {
                    const uint8_t* blk = base + k * b.bytes();
                    for (size_t h = 0; h < halves; ++h) {
                        uint8_t& t = w.top[j * halves + h];
                        t = std::max<uint8_t>(t, static_cast<uint8_t>(blk[2 * h + 1] >> 2 & 31));
                    }
                }
            }
            for (size_t s = 0, k = 0; s < slots; ++s) {
                for (size_t j = 0; j < row_blocks; ++j, ++k) {
                    const uint8_t* blk = base + k * b.bytes();
                    for (size_t h = 0; h < halves; ++h) {
                        const uint8_t sym = high_symbol(blk[2 * h + 1], w.top[j * halves + h]);
                        w.low[k * halves + h] = blk[2 * h];
                        w.fsym[k * halves + h] = sym;
                        ++w.counts[0][sym];
                    }
                    std::memcpy(w.qsym.data() + k * qbytes, blk + 2 * halves, qbytes);
                }
            }
            if (qbytes == 0) continue;
            // Differences where they are smaller than the values, channel by channel.
            w.mode.assign(rp.qchannels, 0);
            if (rp.differences) {
                std::vector<int64_t> plain(rp.qchannels, 0), diff(rp.qchannels, 0);
                for (size_t s = 0; s < slots; ++s) {
                    const auto* q = reinterpret_cast<const int8_t*>(w.qsym.data() + s * rp.qchannels);
                    for (size_t c = 0; c < rp.qchannels; ++c) {
                        plain[c] += std::abs(static_cast<int32_t>(q[c]));
                        if (s > 0) diff[c] += std::abs(static_cast<int32_t>(q[c]) - q[static_cast<std::ptrdiff_t>(c) - static_cast<std::ptrdiff_t>(rp.qchannels)]);
                    }
                }
                for (size_t c = 0; c < rp.qchannels; ++c) w.mode[c] = diff[c] < plain[c] ? 1 : 0;
                for (size_t s = slots; s-- > 1;) {  // backwards: each difference from the original values
                    uint8_t* q = w.qsym.data() + s * rp.qchannels;
                    for (size_t c = 0; c < rp.qchannels; ++c) {
                        if (w.mode[c] != 0) q[c] = static_cast<uint8_t>(q[c] - q[static_cast<std::ptrdiff_t>(c) - static_cast<std::ptrdiff_t>(rp.qchannels)]);
                    }
                }
            }
            for (size_t i = 0; i < rp.quants; ++i) ++w.counts[1 + w.mode[i % rp.qchannels]][w.qsym[i]];
        }
    });
    // Tables per plane and kind.
    const auto planes = static_cast<size_t>(layout.planes);
    const auto layers = static_cast<size_t>(layout.layers);
    std::vector<Freqs> tables(planes * 3);
    for (size_t p = 0; p < planes; ++p) {
        for (size_t kind = 0; kind < 3; ++kind) {
            uint64_t sum[256] = {};
            for (size_t run = p * layers; run < (p + 1) * layers; ++run) {
                for (int s = 0; s < 256; ++s) sum[s] += work[run].counts[kind][s];
            }
            normalize(sum, tables[p * 3 + kind]);
        }
    }
    // Pass 2, per run: the two streams, each coded or kept as it is.
    std::vector<Stream> streams(runs * 2);
    for_each(pool, static_cast<int64_t>(runs), [&](int64_t first, int64_t last) {
        std::vector<const Freqs*> pattern(std::max<size_t>(rp.qchannels, 1));
        std::vector<uint16_t> words;
        for (auto run = static_cast<size_t>(first); run < static_cast<size_t>(last); ++run) {
            const Run& w = work[run];
            const Freqs* t = &tables[run / layers * 3];
            rans_encode(&t, 1, w.fsym.data(), rp.floats, words);
            streams[run * 2] = make_stream(words, w.fsym.data(), rp.floats, costs);
            if (qbytes == 0) continue;
            for (size_t c = 0; c < rp.qchannels; ++c) pattern[c] = t + 1 + w.mode[c];
            rans_encode(pattern.data(), rp.qchannels, w.qsym.data(), rp.quants, words);
            streams[run * 2 + 1] = make_stream(words, w.qsym.data(), rp.quants, costs);
        }
    });
    out.clear();
    put<uint32_t>(out, static_cast<uint32_t>(runs));
    for (size_t p = 0; p < planes; ++p) {
        write_freqs(out, tables[p * 3]);
        if (qbytes > 0) write_freqs(out, tables[p * 3 + 1]);
        if (rp.differences) write_freqs(out, tables[p * 3 + 2]);
    }
    size_t total = out.size();
    for (size_t run = 0; run < runs; ++run) {
        put<uint32_t>(out, streams[run * 2].field);
        put<uint32_t>(out, streams[run * 2 + 1].field);
        total += 8 + rp.channels + rp.mode_bytes + rp.floats + streams[run * 2].bytes.size() + streams[run * 2 + 1].bytes.size();
    }
    if (total >= image.size()) return false;  // does not pay
    out.reserve(total);
    for (size_t run = 0; run < runs; ++run) {
        const Run& w = work[run];
        out.insert(out.end(), w.top.begin(), w.top.end());
        for (size_t i = 0; i < rp.mode_bytes; ++i) {
            uint8_t bits = 0;
            for (size_t j = 0; j < 8 && i * 8 + j < rp.qchannels; ++j) bits = static_cast<uint8_t>(bits | w.mode[i * 8 + j] << j);
            out.push_back(bits);
        }
        out.insert(out.end(), w.low.begin(), w.low.end());
        for (size_t s = 0; s < 2; ++s) out.insert(out.end(), streams[run * 2 + s].bytes.begin(), streams[run * 2 + s].bytes.end());
    }
    return true;
}

bool decode_lossless(const KvRowLayout& layout, const Runs& r, std::span<const uint8_t> in, std::span<uint8_t> out,
                     ThreadPool* pool) {
    const Blocks& b = r.b;
    const auto runs = static_cast<size_t>(r.count);
    const RunParts rp = parts_of(r);
    const auto slots = static_cast<size_t>(r.slots);
    const size_t row_blocks = r.channels / static_cast<size_t>(b.halves);
    const auto halves = static_cast<size_t>(b.halves);
    const auto qbytes = static_cast<size_t>(b.qbytes);
    size_t at = 0;
    uint32_t stored_runs = 0;
    if (!get(in, at, stored_runs) || stored_runs != runs) return false;
    const auto planes = static_cast<size_t>(layout.planes);
    const auto layers = static_cast<size_t>(layout.layers);
    std::vector<DecodeTable> tables(planes * 3);
    for (size_t p = 0; p < planes; ++p) {
        if (!read_freqs(in, at, tables[p * 3])) return false;
        if (qbytes > 0 && !read_freqs(in, at, tables[p * 3 + 1])) return false;
        if (rp.differences && !read_freqs(in, at, tables[p * 3 + 2])) return false;
    }
    std::vector<size_t> offset(runs);
    std::vector<uint32_t> words(runs * 2);
    size_t data = at + 8 * runs;
    for (size_t run = 0; run < runs; ++run) {
        if (!get(in, at, words[run * 2]) || !get(in, at, words[run * 2 + 1])) return false;
        offset[run] = data;
        data += rp.channels + rp.mode_bytes + rp.floats + stream_bytes(words[run * 2], rp.floats) +
                stream_bytes(words[run * 2 + 1], rp.quants);
        if (data > in.size()) return false;
    }
    if (data != in.size()) return false;
    std::atomic<bool> ok{true};
    for_each(pool, static_cast<int64_t>(runs), [&](int64_t first, int64_t last) {
        std::vector<uint8_t> syms(std::max(rp.floats, rp.quants));
        std::vector<const DecodeTable*> pattern(std::max<size_t>(rp.qchannels, 1));
        std::vector<uint8_t> mask(rp.qchannels);
        std::vector<uint16_t> row16;
        for (auto run = static_cast<size_t>(first); run < static_cast<size_t>(last); ++run) {
            const uint8_t* top = in.data() + offset[run];
            const uint8_t* modes = top + rp.channels;
            const uint8_t* low = modes + rp.mode_bytes;
            const uint8_t* stream = low + rp.floats;
            uint8_t* base = out.data() + run * r.bytes;
            const DecodeTable* t = &tables[run / layers * 3];
            if (!decode_stream(&t, 1, stream, words[run * 2], rp.floats, syms.data())) {
                ok = false;
                return;
            }
            if (halves == 1 && qbytes == 0) {  // F16: one float per block
                // One position at a time through a row buffer: plain loops
                // over non-aliasing arrays, which the compiler vectorizes.
                std::vector<uint16_t>& row = row16;
                row.resize(row_blocks);
                for (size_t s = 0; s < slots; ++s) {
                    const uint8_t* __restrict sy = syms.data() + s * row_blocks;
                    const uint8_t* __restrict lo = low + s * row_blocks;
                    uint16_t* __restrict o = row.data();
                    for (size_t j = 0; j < row_blocks; ++j) {
                        const uint32_t e = (static_cast<uint32_t>(top[j]) - (sy[j] >> 3)) & 31;
                        const uint32_t hi = (sy[j] >> 2 & 1) << 7 | e << 2 | (sy[j] & 3);
                        o[j] = static_cast<uint16_t>(lo[j] | hi << 8);
                    }
                    std::memcpy(base + 2 * s * row_blocks, row.data(), 2 * row_blocks);
                }
                continue;
            }
            for (size_t s = 0, k = 0; s < slots; ++s) {
                for (size_t j = 0; j < row_blocks; ++j, ++k) {
                    uint8_t* blk = base + k * b.bytes();
                    for (size_t h = 0; h < halves; ++h) {
                        blk[2 * h] = low[k * halves + h];
                        blk[2 * h + 1] = high_byte(syms[k * halves + h], top[j * halves + h]);
                    }
                }
            }
            for (size_t c = 0; c < rp.qchannels; ++c) {
                const bool diff = rp.differences && (modes[c / 8] >> (c % 8) & 1) != 0;
                pattern[c] = t + 1 + (diff ? 1 : 0);
                mask[c] = diff ? 0xFF : 0;
            }
            if (!decode_stream(pattern.data(), rp.qchannels, stream + stream_bytes(words[run * 2], rp.floats),
                               words[run * 2 + 1], rp.quants, syms.data())) {
                ok = false;
                return;
            }
            if (rp.differences) {  // running sums of the channels coded as differences
                for (size_t s = 1; s < slots; ++s) {
                    uint8_t* q = syms.data() + s * rp.qchannels;
                    const uint8_t* prev = q - rp.qchannels;
                    for (size_t c = 0; c < rp.qchannels; ++c) q[c] = static_cast<uint8_t>(q[c] + (prev[c] & mask[c]));
                }
            }
            for (size_t k = 0; k < r.blocks; ++k) std::memcpy(base + k * b.bytes() + 2 * halves, syms.data() + k * qbytes, qbytes);
        }
    });
    return ok;
}

// --- Lossy: 4 bits ------------------------------------------------------------------
// Payload, per run: key runs (plane 0, and every run when values live in the
// keys' rows): per channel (heads * head_dim) its step and minimum (F16
// each), then per position the 4-bit codes of its row, two per byte, low
// nibble first; value runs: per position, head and 32 channels a Q4_1 block.

bool key_run(const KvRowLayout& layout, size_t run) noexcept {
    return layout.planes == 1 || run < static_cast<size_t>(layout.layers);
}

size_t lossy_run_bytes(const KvRowLayout& layout, int32_t slots, bool key) noexcept {
    const size_t channels = static_cast<size_t>(layout.heads) * static_cast<size_t>(layout.head_dim);
    if (key) return channels * 4 + static_cast<size_t>(slots) * channels / 2;
    return static_cast<size_t>(slots) * channels / quant::kBlock * sizeof(quant::BlockQ4_1);
}

bool encode_lossy4(const KvRowLayout& layout, const Runs& r, std::span<const uint8_t> image, std::vector<uint8_t>& out,
                   ThreadPool* pool) {
    if (layout.type == KvCacheType::Q4_0 || layout.type == KvCacheType::Q4_1) return false;  // 4-bit already
    if (layout.head_dim % quant::kBlock != 0) return false;  // value rows are Q4_1 blocks
    const auto runs = static_cast<size_t>(r.count);
    const DType dtype = dtype_of(layout.type);
    const size_t row_bytes = layout.row_bytes();
    const auto dim = static_cast<size_t>(layout.head_dim);
    const size_t channels = static_cast<size_t>(layout.heads) * dim;
    const auto slots = static_cast<size_t>(r.slots);
    std::vector<size_t> offset(runs + 1, 0);
    for (size_t run = 0; run < runs; ++run) offset[run + 1] = offset[run] + lossy_run_bytes(layout, r.slots, key_run(layout, run));
    if (offset[runs] >= image.size()) return false;
    out.assign(offset[runs], 0);
    for_each(pool, static_cast<int64_t>(runs), [&](int64_t first, int64_t last) {
        std::vector<float> values(slots * channels);
        for (auto run = static_cast<size_t>(first); run < static_cast<size_t>(last); ++run) {
            const uint8_t* base = image.data() + run * r.bytes;
            for (size_t row = 0; row < slots * static_cast<size_t>(layout.heads); ++row) {
                quant::dequantize_row(dtype, base + row * row_bytes, values.data() + row * dim, static_cast<int64_t>(dim));
            }
            uint8_t* dst = out.data() + offset[run];
            if (!key_run(layout, run)) {
                quant::quantize_row_q4_1(values.data(), reinterpret_cast<quant::BlockQ4_1*>(dst),
                                         static_cast<int64_t>(slots * channels));
                continue;
            }
            // Keys: each channel's own range over the chunk's positions.
            std::vector<float> lo(channels), step(channels);
            for (size_t c = 0; c < channels; ++c) {
                float mn = values[c], mx = values[c];
                for (size_t s = 1; s < slots; ++s) {
                    mn = std::min(mn, values[s * channels + c]);
                    mx = std::max(mx, values[s * channels + c]);
                }
                const uint16_t mn16 = quant::fp32_to_fp16(mn);
                const uint16_t d16 = quant::fp32_to_fp16((mx - mn) / 15.0f);
                store16(dst + 4 * c, d16);
                store16(dst + 4 * c + 2, mn16);
                lo[c] = quant::fp16_to_fp32(mn16);
                step[c] = quant::fp16_to_fp32(d16);
            }
            uint8_t* codes = dst + 4 * channels;
            for (size_t s = 0; s < slots; ++s) {
                for (size_t c = 0; c < channels; c += 2) {
                    uint8_t pair = 0;
                    for (size_t j = 0; j < 2; ++j) {
                        const float d = step[c + j];
                        const float q = d > 0 ? std::nearbyint((values[s * channels + c + j] - lo[c + j]) / d) : 0.0f;
                        const auto code = static_cast<uint8_t>(std::clamp(q, 0.0f, 15.0f));
                        pair = static_cast<uint8_t>(pair | code << (4 * j));
                    }
                    codes[(s * channels + c) / 2] = pair;
                }
            }
        }
    });
    return true;
}

bool decode_lossy4(const KvRowLayout& layout, const Runs& r, std::span<const uint8_t> in, std::span<uint8_t> out,
                   ThreadPool* pool) {
    if (layout.type == KvCacheType::Q4_0 || layout.type == KvCacheType::Q4_1) return false;
    if (layout.head_dim % quant::kBlock != 0) return false;
    const auto runs = static_cast<size_t>(r.count);
    const DType dtype = dtype_of(layout.type);
    const size_t row_bytes = layout.row_bytes();
    const auto dim = static_cast<size_t>(layout.head_dim);
    const size_t channels = static_cast<size_t>(layout.heads) * dim;
    const auto slots = static_cast<size_t>(r.slots);
    std::vector<size_t> offset(runs + 1, 0);
    for (size_t run = 0; run < runs; ++run) offset[run + 1] = offset[run] + lossy_run_bytes(layout, r.slots, key_run(layout, run));
    if (offset[runs] != in.size()) return false;
    for_each(pool, static_cast<int64_t>(runs), [&](int64_t first, int64_t last) {
        std::vector<float> row(channels);
        std::vector<float> lo(channels), step(channels);
        for (auto run = static_cast<size_t>(first); run < static_cast<size_t>(last); ++run) {
            const uint8_t* src = in.data() + offset[run];
            uint8_t* base = out.data() + run * r.bytes;
            const bool key = key_run(layout, run);
            if (key) {
                for (size_t c = 0; c < channels; ++c) {
                    step[c] = quant::fp16_to_fp32(load16(src + 4 * c));
                    lo[c] = quant::fp16_to_fp32(load16(src + 4 * c + 2));
                }
            }
            for (size_t s = 0; s < slots; ++s) {
                if (key) {
                    const uint8_t* codes = src + 4 * channels + s * channels / 2;
                    for (size_t c = 0; c < channels; c += 2) {
                        row[c] = lo[c] + step[c] * static_cast<float>(codes[c / 2] & 15);
                        row[c + 1] = lo[c + 1] + step[c + 1] * static_cast<float>(codes[c / 2] >> 4);
                    }
                } else {
                    quant::dequantize_row(DType::Q4_1, src + s * channels / quant::kBlock * sizeof(quant::BlockQ4_1), row.data(),
                                          static_cast<int64_t>(channels));
                }
                for (size_t h = 0; h < static_cast<size_t>(layout.heads); ++h) {
                    quant::quantize_row(dtype, row.data() + h * dim, base + (s * static_cast<size_t>(layout.heads) + h) * row_bytes,
                                        static_cast<int64_t>(dim));
                }
            }
        }
    });
    return true;
}

}  // namespace

KvRowLayout kv_row_layout(const KvCacheConfig& config) noexcept {
    KvRowLayout l;
    l.type = config.type;
    l.planes = config.values_in_keys ? 1 : 2;
    l.layers = config.n_layers;
    l.heads = config.n_head_kv;
    l.head_dim = config.head_dim;
    return l;
}

bool KvRowLayout::valid() const noexcept {
    if (planes < 1 || planes > 2 || layers <= 0 || heads <= 0 || head_dim <= 0) return false;
    const Blocks b = blocks_of(type);
    return head_dim % b.values == 0;
}

size_t KvRowLayout::row_bytes() const noexcept {
    const Blocks b = blocks_of(type);
    return static_cast<size_t>(head_dim / b.values) * b.bytes();
}

size_t KvRowLayout::image_bytes(int32_t slots) const noexcept {
    return static_cast<size_t>(planes) * static_cast<size_t>(layers) * static_cast<size_t>(std::max(slots, 0)) *
           static_cast<size_t>(heads) * row_bytes();
}

bool kv_encode(KvEncoding encoding, const KvRowLayout& layout, int32_t slots, std::span<const uint8_t> image,
               std::vector<uint8_t>& out, ThreadPool* pool, const KvCodecCosts* costs) {
    Runs r;
    if (!runs_of(layout, slots, r) || image.size() != layout.image_bytes(slots)) return false;
    switch (encoding) {
        case KvEncoding::Raw: return false;
        case KvEncoding::Lossless: return encode_lossless(layout, r, image, out, pool, costs);
        case KvEncoding::Lossy4: return encode_lossy4(layout, r, image, out, pool);
    }
    return false;
}

bool kv_decode(KvEncoding encoding, const KvRowLayout& layout, int32_t slots, std::span<const uint8_t> in,
               std::span<uint8_t> out, ThreadPool* pool) {
    Runs r;
    if (!runs_of(layout, slots, r) || out.size() != layout.image_bytes(slots)) return false;
    switch (encoding) {
        case KvEncoding::Raw: return false;
        case KvEncoding::Lossless: return decode_lossless(layout, r, in, out, pool);
        case KvEncoding::Lossy4: return decode_lossy4(layout, r, in, out, pool);
    }
    return false;
}

double kv_calibrate_decode(const KvRowLayout& layout, int32_t slots, ThreadPool* pool) {
    Runs r;
    if (!runs_of(layout, slots, r)) return 0;
    // Rows shaped like keys and values: channels of their own magnitude, so
    // every stream codes smaller and is decoded as a real chunk's would be.
    std::vector<uint8_t> image(layout.image_bytes(slots));
    const DType dtype = dtype_of(layout.type);
    std::vector<float> row(static_cast<size_t>(layout.head_dim));
    const size_t row_bytes = layout.row_bytes();
    uint32_t seed = 12345;
    for (size_t at = 0, k = 0; at < image.size(); at += row_bytes, ++k) {
        for (size_t c = 0; c < row.size(); ++c) {
            seed = seed * 1664525u + 1013904223u;
            const float noise = static_cast<float>(seed >> 8) / 16777216.0f - 0.5f;
            row[c] = (0.2f + static_cast<float>(c % 9) * 0.4f) * (noise + 0.3f * std::sin(0.05f * static_cast<float>(k)));
        }
        quant::quantize_row(dtype, row.data(), image.data() + at, layout.head_dim);
    }
    std::vector<uint8_t> enc;
    if (!encode_lossless(layout, r, image, enc, pool, nullptr)) return 0;
    const double symbols = static_cast<double>(r.count) * static_cast<double>(r.blocks) *
                           static_cast<double>(r.b.halves + r.b.qbytes);
    std::vector<uint8_t> back(image.size());
    double best = 0;
    for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!decode_lossless(layout, r, enc, back, pool)) return 0;
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
        best = rep == 0 ? ns : std::min(best, ns);
    }
    return best / symbols;
}

}  // namespace liyab
