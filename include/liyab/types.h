// Liyab — core value types shared by every module.
#ifndef LIYAB_TYPES_H
#define LIYAB_TYPES_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#if defined(_WIN32)
#define LIYAB_API __declspec(dllexport)
#else
#define LIYAB_API __attribute__((visibility("default")))
#endif

namespace liyab {

// ---------------------------------------------------------------------------
// Error handling: no exceptions cross module boundaries; fallible calls
// return Status or Result<T>.
// ---------------------------------------------------------------------------
enum class ErrorCode : int32_t {
    Ok = 0,
    InvalidArgument = 1,
    IoError = 2,
    InvalidModel = 3,
    Unsupported = 4,
    OutOfMemory = 5,
    ContextFull = 6,
    Cancelled = 7,
    BackendError = 8,
    Busy = 9,
    Internal = 10,
};

LIYAB_API const char* error_code_name(ErrorCode code) noexcept;

class [[nodiscard]] Status {
public:
    Status() = default;
    Status(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

    static Status ok() { return {}; }

    [[nodiscard]] bool is_ok() const noexcept { return code_ == ErrorCode::Ok; }
    explicit operator bool() const noexcept { return is_ok(); }
    [[nodiscard]] ErrorCode code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }
    [[nodiscard]] std::string to_string() const;

private:
    ErrorCode code_ = ErrorCode::Ok;
    std::string message_;
};

template <typename T>
class [[nodiscard]] Result {
public:
    Result(T value) : data_(std::move(value)) {}              // NOLINT(google-explicit-constructor)
    Result(Status status) : data_(std::move(status)) {         // NOLINT(google-explicit-constructor)
        if (std::get<Status>(data_).is_ok()) {
            data_ = Status(ErrorCode::Internal, "Result constructed from an OK status without a value");
        }
    }

    [[nodiscard]] bool has_value() const noexcept { return std::holds_alternative<T>(data_); }
    explicit operator bool() const noexcept { return has_value(); }

    T& value() & { return std::get<T>(data_); }
    const T& value() const& { return std::get<T>(data_); }
    T&& value() && { return std::get<T>(std::move(data_)); }
    T* operator->() { return &value(); }
    const T* operator->() const { return &value(); }
    T& operator*() & { return value(); }

    [[nodiscard]] Status status() const {
        return has_value() ? Status::ok() : std::get<Status>(data_);
    }

private:
    std::variant<T, Status> data_;
};

#define LIYAB_RETURN_IF_ERROR(expr)                 \
    do {                                            \
        ::liyab::Status liyab_status_ = (expr);     \
        if (!liyab_status_.is_ok()) return liyab_status_; \
    } while (0)

// ---------------------------------------------------------------------------
// Tensor element types. Numeric values match the GGML/GGUF type ids so the
// loader can map file tensors without translation tables.
// ---------------------------------------------------------------------------
enum class DType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,   // 32-element blocks: fp16 scale + 16 bytes of 4-bit values (symmetric)
    Q4_1 = 3,   // 32-element blocks: fp16 scale + fp16 min + 16 bytes (asymmetric)
    Q5_0 = 6,   // 32-element blocks: fp16 scale + 5-bit values (high bits packed), symmetric
    Q5_1 = 7,   // 32-element blocks: fp16 scale + fp16 min + 5-bit values
    Q8_0 = 8,   // 32-element blocks: fp16 scale + 32 int8 values
    // K-quants: 256-element super-blocks with 6/8-bit sub-block scales.
    Q4_K = 12,  // 8 x 32: 4-bit values, scale + min per sub-block (4.5 bits/weight)
    Q5_K = 13,  // 8 x 32: 5-bit values, scale + min per sub-block (5.5 bits/weight)
    Q6_K = 14,  // 16 x 16: 6-bit signed values, int8 scale per sub-block (6.56 bits/weight)
    Q2_K = 10,  // 2-bit values, 4-bit scale + min per 16
    Q3_K = 11,  // 3-bit values, 6-bit scale per 16
    // I-quants: lattice codebooks (grids) + signs, 1.6 to 4.25 bits/weight.
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,  // 32-value blocks, non-linear 4-bit codebook
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    IQ1_M = 29,
    BF16 = 30,    // bfloat16
    TQ1_0 = 34,   // ternary {-1, 0, 1}, 1.69 bits/weight
    TQ2_0 = 35,   // ternary, 2.06 bits/weight
    MXFP4 = 39,   // OCP microscaling FP4 (e2m1) with e8m0 scale per 32
    NVFP4 = 40,   // FP4 (e2m1) with ue4m3 scale per 16
};

struct DTypeTraits {
    std::string_view name;
    int32_t block_size;    // elements per block
    int32_t block_bytes;   // bytes per block
};

// Returns false for type ids Liyab cannot execute (e.g. K-quants, IQ-quants).
LIYAB_API bool dtype_from_ggml(uint32_t ggml_type, DType& out) noexcept;
LIYAB_API DTypeTraits dtype_traits(DType type) noexcept;
LIYAB_API std::string_view ggml_type_name(uint32_t ggml_type) noexcept;
// Bytes needed to store `n_elements` (must be a multiple of the block size).
LIYAB_API size_t dtype_row_bytes(DType type, int64_t n_elements) noexcept;

// Non-owning view of a tensor that lives in the memory-mapped model file.
// Shape follows GGUF order: ne[0] is the innermost (contiguous) dimension,
// so a weight matrix has ne[0] = input features (cols), ne[1] = outputs (rows).
struct TensorView {
    std::string_view name;
    DType type = DType::F32;
    int32_t n_dims = 0;
    std::array<int64_t, 4> ne{1, 1, 1, 1};
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    uint64_t file_offset = 0;

    [[nodiscard]] int64_t cols() const noexcept { return ne[0]; }
    [[nodiscard]] int64_t rows() const noexcept { return ne[1] * ne[2] * ne[3]; }
    [[nodiscard]] int64_t elements() const noexcept { return ne[0] * ne[1] * ne[2] * ne[3]; }
    [[nodiscard]] size_t row_bytes() const noexcept { return dtype_row_bytes(type, ne[0]); }
    [[nodiscard]] const uint8_t* row(int64_t r) const noexcept { return data + static_cast<size_t>(r) * row_bytes(); }
};

// ---------------------------------------------------------------------------
// Execution and policy enums.
// ---------------------------------------------------------------------------
enum class BackendKind : int32_t {
    Cpu = 0,
    Metal = 1,
    Vulkan = 2,
    Qnn = 3,
    NeuroPilot = 4,
};

enum class ProcessorClass : int32_t { Cpu, Gpu, Npu };

LIYAB_API const char* backend_kind_name(BackendKind kind) noexcept;
LIYAB_API ProcessorClass processor_class(BackendKind kind) noexcept;

enum class PowerProfile : int32_t {
    Performance = 0,  // no pacing, throttle only at the hard thermal limit
    Balanced = 1,     // paced to ~12 tok/s, throttle at the skin threshold
    LowPower = 2,     // paced to ~6 tok/s, reduced threads, early throttling
};

enum class KvCacheType : int32_t {
    F16 = 0,
    Q8_0 = 1,  // 8.5 bits/value: 53% of F16
    Q4_0 = 2,  // symmetric INT4, 4.5 bits/value: 28% of F16 (-72%)
    Q4_1 = 3,  // asymmetric INT4 (scale + min), 5 bits/value: 31% of F16 (-69%)
};

struct SamplingParams {
    float temperature = 0.8f;   // <= 0 selects greedy decoding
    int32_t top_k = 40;         // <= 0 disables
    float top_p = 0.95f;        // >= 1 disables
    uint64_t seed = 0;          // 0 picks a random seed
    int32_t max_tokens = 256;
    bool add_bos = true;
};

struct GenerationStats {
    int32_t prompt_tokens = 0;
    int32_t generated_tokens = 0;
    double prefill_ms = 0.0;
    double ttft_ms = 0.0;           // time to first token (prefill + first decode step)
    int32_t cached_prefix_tokens = 0;  // experimental (KV dedup): prompt tokens restored from a snapshot
    double decode_ms = 0.0;
    double tokens_per_second = 0.0;
    int32_t draft_tokens_proposed = 0;
    int32_t draft_tokens_accepted = 0;
    int32_t thermal_reroutes = 0;   // decode steps executed on the throttled route
    double paced_idle_ms = 0.0;     // time spent idling for duty-cycle pacing
    int32_t early_exits = 0;        // experimental: decode steps that exited early
    int32_t early_exit_layers_skipped = 0;
    int32_t head_pruned_steps = 0;  // experimental: decode steps with pruned heads
    int32_t ffn_blocks_skipped = 0; // experimental (EGLS): FFN blocks bypassed during decode
    int32_t sparse_ffn_steps = 0;   // experimental (TDSS): decode steps on 2:4 sparse FFN weights
    int32_t weight_stalls = 0;      // triple-buffer: block fetches the executor had to wait for
    double weight_wait_ms = 0.0;
    uint64_t kv_cache_bytes = 0;    // KV pages in use at the end of generation
    bool cancelled = false;
};

}  // namespace liyab

#endif  // LIYAB_TYPES_H
