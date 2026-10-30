#include "core/kernel_conformance.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>

#include "core/os.h"
#include "core/quant.h"

namespace liyab::conformance {

namespace {

using Clock = std::chrono::steady_clock;

// splitmix64: a few lines, the same sequence on every platform (std::mt19937 distributions differ between
// standard libraries), so a failure reproduces from its seed anywhere.
class Rng {
public:
    explicit Rng(uint64_t seed) : state_(seed) {}
    uint64_t next() noexcept {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t below(uint32_t n) noexcept { return static_cast<uint32_t>((next() >> 32) % n); }
    float unit() noexcept { return static_cast<float>(next() >> 40) / 16777216.0f; }  // [0, 1)

private:
    uint64_t state_;
};

struct Shape {
    int64_t rows;
    int64_t cols;
    int32_t n;  // activation rows
};

// Row counts that are not multiples of any tile (37, 5), one and several blocks per row (256, 512, 768 are whole
// blocks of every format: 32, 64, 128 and 256 values), and activation counts around the usual batch edges.
constexpr Shape kDecodeShapes[] = {{37, 256, 1}, {37, 768, 1}};
constexpr Shape kBatchShapes[] = {{37, 768, 2}, {5, 256, 5}, {8, 512, 33}};

// A block of a quantized format is accepted when it decodes to finite values whose largest magnitude is neither
// negligible nor large: random bytes in a scale field would otherwise give NaN, infinities or values 1e9 times
// the others, and the check would measure float rounding instead of the kernel.
constexpr float kMinPeak = 0.05f;
constexpr float kMaxPeak = 8.0f;
constexpr int kMaxDraws = 20000;

// The activations' step: multiples of 2^-6 up to 127 * 2^-6 are exact in float, in fp16 and as the int8 grid.
constexpr float kStep = 1.0f / 64.0f;
// Relative tolerance on sum|w*x|. Correct float32 kernels land at ~5e-8 of it (measured on the CPU and Metal), so
// 1e-5 allows another summation order (GPU lanes, other SIMD widths) and fails a half-precision accumulator.
constexpr double kTolerance = 1e-5;

// Floats of a log-uniform magnitude in [2^-6, 2^3) with a random sign: every exponent and mantissa pattern of
// the float formats shows up, and the sum of a row stays far from overflow.
float random_float(Rng& rng) {
    const float magnitude = std::ldexp(1.0f + rng.unit(), static_cast<int>(rng.below(9)) - 6);
    return rng.below(2) != 0 ? -magnitude : magnitude;
}

// Fills `rows` rows of `type` (cols values each) at `dst` with random valid data. False, with `why`, when no valid
// block was found in kMaxDraws draws (never seen with real formats).
bool fill_weights(DType type, int64_t rows, int64_t cols, uint8_t* dst, Rng& rng, std::string& why) {
    const DTypeTraits traits = dtype_traits(type);
    if (traits.block_size == 1) {  // F32, F16, BF16: one value per "block"
        for (int64_t i = 0; i < rows * cols; ++i) {
            const float v = random_float(rng);
            if (type == DType::F32) {
                std::memcpy(dst + static_cast<size_t>(i) * 4, &v, 4);
            } else if (type == DType::F16) {
                const uint16_t h = quant::fp32_to_fp16(v);
                std::memcpy(dst + static_cast<size_t>(i) * 2, &h, 2);
            } else if (type == DType::BF16) {
                uint32_t bits;
                std::memcpy(&bits, &v, 4);
                const auto h = static_cast<uint16_t>(bits >> 16);  // the reference decodes what is stored: truncation is fine
                std::memcpy(dst + static_cast<size_t>(i) * 2, &h, 2);
            } else {
                why = "no generator for single-value format";
                return false;
            }
        }
        return true;
    }
    std::vector<float> decoded(static_cast<size_t>(traits.block_size));
    const int64_t blocks = rows * (cols / traits.block_size);
    const auto block_bytes = static_cast<size_t>(traits.block_bytes);
    for (int64_t b = 0; b < blocks; ++b) {
        uint8_t* block = dst + static_cast<size_t>(b) * block_bytes;
        bool accepted = false;
        for (int draw = 0; draw < kMaxDraws && !accepted; ++draw) {
            for (size_t i = 0; i < block_bytes; i += 8) {
                const uint64_t bits = rng.next();
                std::memcpy(block + i, &bits, std::min<size_t>(8, block_bytes - i));
            }
            quant::dequantize_row(type, block, decoded.data(), traits.block_size);
            float peak = 0.0f;
            bool finite = true;
            for (const float v : decoded) {
                finite = finite && std::isfinite(v);
                peak = std::max(peak, std::fabs(v));
            }
            accepted = finite && peak >= kMinPeak && peak <= kMaxPeak;
        }
        if (!accepted) {
            why = "no valid " + std::string(traits.name) + " block in " + std::to_string(kMaxDraws) + " draws";
            return false;
        }
    }
    return true;
}

// Activations on the int8 grid the CPU backend quantizes to (see the header): in every run of 32 values the
// multiples of kStep up to +-127 * kStep, one of them pinned at the extreme so each block's scale is exactly kStep.
void fill_activations(float* x, int32_t n, int64_t cols, Rng& rng) {
    for (int32_t t = 0; t < n; ++t) {
        for (int64_t b = 0; b < cols / 32; ++b) {
            float* block = x + static_cast<size_t>(t) * static_cast<size_t>(cols) + static_cast<size_t>(b) * 32;
            for (int i = 0; i < 32; ++i) block[i] = static_cast<float>(static_cast<int>(rng.below(255)) - 127) * kStep;
            block[rng.below(32)] = (rng.below(2) != 0 ? 127.0f : -127.0f) * kStep;
        }
    }
}

// Ordinary-memory weights that stay allocated for the life of the process (see the header).
std::vector<std::unique_ptr<uint8_t[]>>& heap_arena() {
    static std::vector<std::unique_ptr<uint8_t[]>> arena;
    return arena;
}

// A tensor's bytes in one residency, 32 bytes into a page like a GGUF tensor's data.
class Placement {
public:
    Placement(Backend& backend, Residency where, size_t bytes, bool keep_heap) {
        constexpr size_t kOffset = 32;
        if (where == Residency::Shared) {
            region_ = backend.allocate_shared(bytes + 2 * kOffset);
            if (region_ == nullptr) return;
            owner_ = &backend;
            data_ = region_ + kOffset;
            return;
        }
        const size_t page = os::page_size();
        auto heap = std::make_unique<uint8_t[]>(bytes + 2 * kOffset + page);
        const auto address = reinterpret_cast<uintptr_t>(heap.get());
        data_ = reinterpret_cast<uint8_t*>((address + page - 1) / page * page) + kOffset;
        if (keep_heap) {
            heap_arena().push_back(std::move(heap));
        } else {
            heap_ = std::move(heap);
        }
    }
    ~Placement() {
        if (owner_ != nullptr) owner_->free_shared(region_);
    }
    Placement(const Placement&) = delete;
    Placement& operator=(const Placement&) = delete;

    [[nodiscard]] uint8_t* data() const noexcept { return data_; }

private:
    Backend* owner_ = nullptr;
    uint8_t* region_ = nullptr;
    uint8_t* data_ = nullptr;
    std::unique_ptr<uint8_t[]> heap_;
};

TensorView view_of(DType type, int64_t rows, int64_t cols, uint8_t* data) {
    TensorView w;
    w.name = "conformance.weight";
    w.type = type;
    w.n_dims = 2;
    w.ne = {cols, rows, 1, 1};
    w.data = data;
    w.nbytes = w.row_bytes() * static_cast<size_t>(rows);
    return w;
}

// One shape, one placement: weights filled (and, in shared memory, refilled in place), one matmul compared with
// the reference, and the batching bits.
struct Probe {
    Verdict verdict = Verdict::Ok;  // Ok, Refused, Wrong or Failed
    double worst = 0.0;
    bool exact = true;
    std::string detail;
};

std::string shape_text(const Shape& s) {
    char text[64];
    std::snprintf(text, sizeof text, "%lldx%lld n=%d", static_cast<long long>(s.rows), static_cast<long long>(s.cols), s.n);
    return text;
}

bool same_bits(const float* a, const float* b, size_t count) { return std::memcmp(a, b, count * sizeof(float)) == 0; }

Probe probe(Backend& backend, DType type, Residency where, const Shape& shape, uint64_t seed) {
    Probe out;
    const size_t row_bytes = dtype_row_bytes(type, shape.cols);
    const size_t bytes = row_bytes * static_cast<size_t>(shape.rows);
    Placement placement(backend, where, bytes, /*keep_heap=*/true);
    if (placement.data() == nullptr) {
        out.verdict = Verdict::Failed;
        out.detail = "cannot allocate " + std::to_string(bytes) + " bytes of " + residency_name(where) + " memory";
        return out;
    }
    const TensorView w = view_of(type, shape.rows, shape.cols, placement.data());
    const auto rows = static_cast<size_t>(shape.rows);
    const auto cols = static_cast<size_t>(shape.cols);
    const auto n = static_cast<size_t>(shape.n);
    // Shared regions may be refilled with other tensors at any time (streaming slots, expert caches): a second
    // fill in place must not show the first one's data.
    const int fills = where == Residency::Shared ? 2 : 1;
    for (int fill = 0; fill < fills; ++fill) {
        Rng rng(seed + 7919ull * static_cast<uint64_t>(fill));
        std::string why;
        if (!fill_weights(type, shape.rows, shape.cols, placement.data(), rng, why)) {
            out.verdict = Verdict::Failed;
            out.detail = why;
            return out;
        }
        std::vector<float> decoded(rows * cols);
        for (size_t r = 0; r < rows; ++r) {
            quant::dequantize_row(type, placement.data() + r * row_bytes, decoded.data() + r * cols, shape.cols);
        }
        std::vector<float> x(n * cols);
        fill_activations(x.data(), shape.n, shape.cols, rng);
        // NaN everywhere first: a backend that answers Ok without writing an output is caught as wrong.
        std::vector<float> y(n * rows, std::numeric_limits<float>::quiet_NaN());
        const Status status = backend.matmul(w, x.data(), y.data(), shape.n);
        if (status.code() == ErrorCode::Unsupported) {
            out.verdict = Verdict::Refused;
            out.detail = status.message();
            return out;
        }
        if (!status.is_ok()) {
            out.verdict = Verdict::Failed;
            out.detail = shape_text(shape) + ": " + status.message();
            return out;
        }
        double worst = 0.0;
        for (size_t t = 0; t < n; ++t) {
            for (size_t r = 0; r < rows; ++r) {
                double reference = 0.0, magnitude = 0.0;
                for (size_t c = 0; c < cols; ++c) {
                    const double product = static_cast<double>(decoded[r * cols + c]) * static_cast<double>(x[t * cols + c]);
                    reference += product;
                    magnitude += std::fabs(product);
                }
                const double error = std::fabs(static_cast<double>(y[t * rows + r]) - reference);
                double ratio = error / (kTolerance * magnitude + 1e-6);
                if (std::isnan(ratio)) ratio = std::numeric_limits<double>::infinity();
                worst = std::max(worst, ratio);
            }
        }
        out.worst = std::max(out.worst, worst);
        if (worst > 1.0) {
            char text[96];
            std::snprintf(text, sizeof text, "%s: %.3g x the tolerance%s", shape_text(shape).c_str(), worst,
                          fill > 0 ? " after a refill" : "");
            out.verdict = Verdict::Wrong;
            out.detail = text;
            return out;
        }
        // Batching: the same bits from n one-row calls and from a grouped call.
        bool exact = true;
        if (shape.n >= 2) {
            std::vector<float> single(rows);
            for (size_t t = 0; t < n && exact; ++t) {
                exact = backend.matmul(w, x.data() + t * cols, single.data(), 1).is_ok() &&
                        same_bits(single.data(), y.data() + t * rows, rows);
            }
        }
        if (exact) {
            std::vector<float> a(n * rows), b(n * rows);
            const TensorView* together[] = {&w, &w};
            float* outputs[] = {a.data(), b.data()};
            exact = backend.matmul_group(together, x.data(), outputs, shape.n).is_ok() &&
                    same_bits(a.data(), y.data(), n * rows) && same_bits(b.data(), y.data(), n * rows);
        }
        if (!exact) {
            out.exact = false;
            if (out.detail.empty()) out.detail = shape_text(shape) + ": batched and single results differ in bits";
        }
    }
    return out;
}

// Every shape of one kind of call, folded into a cell.
Cell run_kind(Backend& backend, DType type, Residency where, const Shape* shapes, size_t count, uint64_t salt) {
    Cell cell;
    size_t ok = 0;
    bool wrong = false, failed = false;
    for (size_t i = 0; i < count; ++i) {
        const uint64_t seed = (static_cast<uint64_t>(type) * 1000003ull + salt) * 131ull + i * 17ull +
                              (where == Residency::Shared ? 5ull : 0ull);
        const Probe p = probe(backend, type, where, shapes[i], seed);
        switch (p.verdict) {
            case Verdict::Ok:
                ++ok;
                cell.worst = std::max(cell.worst, p.worst);
                cell.exact = cell.exact && p.exact;
                if (!p.exact && cell.detail.empty()) cell.detail = p.detail;
                break;
            case Verdict::Refused:
                if (cell.detail.empty()) cell.detail = p.detail;
                break;
            case Verdict::Wrong:
                wrong = true;
                cell.worst = std::max(cell.worst, p.worst);
                cell.detail = p.detail;
                break;
            default:
                failed = true;
                cell.detail = p.detail;
                break;
        }
    }
    cell.verdict = failed ? Verdict::Failed
                   : wrong ? Verdict::Wrong
                   : ok == count ? Verdict::Ok
                   : ok > 0 ? Verdict::Partial
                            : Verdict::Refused;
    return cell;
}

// The smallest time of `call` over a short run: the minimum is steady where means and medians move with whatever
// else the host is doing.
template <typename F>
double best_seconds(F&& call) {
    constexpr double kBudget = 0.08;
    constexpr int kMinCalls = 3, kMaxCalls = 200;
    double best = std::numeric_limits<double>::infinity();
    const auto start = Clock::now();
    for (int i = 0; i < kMaxCalls; ++i) {
        const auto t0 = Clock::now();
        call();
        best = std::min(best, std::chrono::duration<double>(Clock::now() - t0).count());
        if (i + 1 >= kMinCalls && std::chrono::duration<double>(Clock::now() - start).count() > kBudget) break;
    }
    return best;
}

// Throughput of a one-row call and per-row cost of a 16-row call on a matrix of about `timing_bytes`. The matrix
// repeats a tile of 32 random valid rows: the kernels cost the same on any bytes (liyab-bench's assumption too).
bool measure(Backend& backend, DType type, Residency where, size_t timing_bytes, double& gb_per_s, double& batch_cost) {
    constexpr int64_t kCols = 4096, kTile = 32;
    constexpr int32_t kBatch = 16;
    const size_t row_bytes = dtype_row_bytes(type, kCols);
    const int64_t rows = std::max<int64_t>(kTile, static_cast<int64_t>(timing_bytes / row_bytes));
    Placement placement(backend, where, row_bytes * static_cast<size_t>(rows), /*keep_heap=*/false);
    if (placement.data() == nullptr) return false;
    Rng rng(0xC0FFEEull + static_cast<uint64_t>(type));
    std::string why;
    if (!fill_weights(type, kTile, kCols, placement.data(), rng, why)) return false;
    for (int64_t r = kTile; r < rows; ++r) {
        std::memcpy(placement.data() + static_cast<size_t>(r) * row_bytes,
                    placement.data() + static_cast<size_t>(r % kTile) * row_bytes, row_bytes);
    }
    const TensorView w = view_of(type, rows, kCols, placement.data());
    std::vector<float> x(static_cast<size_t>(kBatch * kCols)), y(static_cast<size_t>(kBatch * rows));
    fill_activations(x.data(), kBatch, kCols, rng);
    if (!backend.matmul(w, x.data(), y.data(), 1).is_ok() || !backend.matmul(w, x.data(), y.data(), kBatch).is_ok()) {
        return false;
    }
    const double one = best_seconds([&] { (void)backend.matmul(w, x.data(), y.data(), 1); });
    const double many = best_seconds([&] { (void)backend.matmul(w, x.data(), y.data(), kBatch); });
    if (!(one > 0.0) || !std::isfinite(one) || !std::isfinite(many)) return false;
    gb_per_s = static_cast<double>(w.nbytes) / one / 1e9;
    batch_cost = many / kBatch / one;
    return true;
}

bool ran(const Cell& cell) noexcept { return cell.verdict == Verdict::Ok || cell.verdict == Verdict::Partial; }

}  // namespace

std::vector<DType> weight_formats() {
    std::vector<DType> out;
    for (uint32_t id = 0; id < 256; ++id) {
        DType type;
        if (dtype_from_ggml(id, type) && type != DType::I32) out.push_back(type);
    }
    return out;
}

bool ran_anywhere(const FormatReport& format) noexcept {
    return std::any_of(format.placements.begin(), format.placements.end(),
                       [](const PlacementReport& p) { return ran(p.decode) || ran(p.batch); });
}

const char* verdict_name(Verdict verdict) noexcept {
    switch (verdict) {
        case Verdict::Ok: return "ok";
        case Verdict::Partial: return "partial";
        case Verdict::Refused: return "refused";
        case Verdict::Wrong: return "WRONG";
        case Verdict::Failed: return "FAILED";
    }
    return "?";
}

const char* residency_name(Residency where) noexcept { return where == Residency::Heap ? "heap" : "shared"; }

BackendReport check_backend(Backend& backend, const Options& options) {
    BackendReport report;
    report.kind = backend.kind();
    report.name = backend.description();
    std::vector<Residency> placements = {Residency::Heap};
    if (uint8_t* probe_region = backend.allocate_shared(os::page_size())) {
        backend.free_shared(probe_region);
        placements.push_back(Residency::Shared);
    }
    // Timings go where the backend's fast path is: shared memory when it has any (the zero-copy path of streamed
    // experts and placed weights), else ordinary memory. A backend that caches per address (Vulkan's upload of
    // heap weights) would keep every timing matrix.
    const Residency timed = placements.back();
    const std::vector<DType> formats = options.only.empty() ? weight_formats() : options.only;
    for (const DType type : formats) {
        FormatReport format;
        format.type = type;
        format.claimed = backend.supports(type);
        for (const Residency where : placements) {
            PlacementReport placement;
            placement.where = where;
            placement.decode = run_kind(backend, type, where, kDecodeShapes, std::size(kDecodeShapes), 1);
            placement.batch = run_kind(backend, type, where, kBatchShapes, std::size(kBatchShapes), 2);
            if (options.timing && where == timed && ran(placement.decode) && ran(placement.batch)) {
                measure(backend, type, where, options.timing_bytes, placement.decode.gb_per_s,
                        placement.batch.batch_cost);
            }
            format.placements.push_back(std::move(placement));
        }
        report.formats.push_back(std::move(format));
    }
    return report;
}

std::vector<std::string> problems(const BackendReport& report) {
    std::vector<std::string> out;
    for (const FormatReport& format : report.formats) {
        const std::string name(dtype_traits(format.type).name);
        for (const PlacementReport& placement : format.placements) {
            const Cell* cells[] = {&placement.decode, &placement.batch};
            const char* kinds[] = {"decode", "batch"};
            for (int k = 0; k < 2; ++k) {
                const Cell& cell = *cells[k];
                const std::string label = name + " " + residency_name(placement.where) + " " + kinds[k];
                if (cell.verdict == Verdict::Wrong) {
                    out.push_back(label + ": wrong result (" + cell.detail + ")");
                } else if (cell.verdict == Verdict::Failed) {
                    out.push_back(label + ": failed (" + cell.detail + ")");
                } else if (report.kind == BackendKind::Cpu && cell.verdict != Verdict::Ok) {
                    out.push_back(label + ": the CPU backend must run every format, and it answered " +
                                  verdict_name(cell.verdict) + " (" + cell.detail + ")");
                } else if (report.kind == BackendKind::Cpu && !cell.exact) {
                    out.push_back(label + ": the CPU backend's results must not depend on the batch (" + cell.detail + ")");
                }
            }
        }
        if (format.claimed != ran_anywhere(format)) {
            out.push_back(format.claimed ? name + ": supports() says yes, but no placement or shape ran"
                                         : name + ": ran, but supports() says no");
        }
    }
    return out;
}

}  // namespace liyab::conformance
