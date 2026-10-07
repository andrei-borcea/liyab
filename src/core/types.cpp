#include "liyab/types.h"

namespace liyab {

const char* error_code_name(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::Ok: return "ok";
        case ErrorCode::InvalidArgument: return "invalid_argument";
        case ErrorCode::IoError: return "io_error";
        case ErrorCode::InvalidModel: return "invalid_model";
        case ErrorCode::Unsupported: return "unsupported";
        case ErrorCode::OutOfMemory: return "out_of_memory";
        case ErrorCode::ContextFull: return "context_full";
        case ErrorCode::Cancelled: return "cancelled";
        case ErrorCode::BackendError: return "backend_error";
        case ErrorCode::Busy: return "busy";
        case ErrorCode::Internal: return "internal";
    }
    return "unknown";
}

std::string Status::to_string() const {
    if (is_ok()) return "ok";
    return std::string(error_code_name(code_)) + ": " + message_;
}

bool dtype_from_ggml(uint32_t ggml_type, DType& out) noexcept {
    switch (ggml_type) {
        case 0: out = DType::F32; return true;
        case 1: out = DType::F16; return true;
        case 2: out = DType::Q4_0; return true;
        case 3: out = DType::Q4_1; return true;
        case 8: out = DType::Q8_0; return true;
        default: return false;
    }
}

DTypeTraits dtype_traits(DType type) noexcept {
    switch (type) {
        case DType::F32: return {"f32", 1, 4};
        case DType::F16: return {"f16", 1, 2};
        case DType::Q4_0: return {"q4_0", 32, 18};
        case DType::Q4_1: return {"q4_1", 32, 20};
        case DType::Q8_0: return {"q8_0", 32, 34};
    }
    return {"invalid", 1, 0};
}

std::string_view ggml_type_name(uint32_t ggml_type) noexcept {
    // Names for diagnostics when a model uses a type Liyab cannot run yet.
    static constexpr std::string_view kNames[] = {
        "f32", "f16", "q4_0", "q4_1", "q4_2", "q4_3", "q5_0", "q5_1", "q8_0", "q8_1",
        "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k", "iq2_xxs", "iq2_xs", "iq3_xxs",
        "iq1_s", "iq4_nl", "iq3_s", "iq2_s", "iq4_xs", "i8", "i16", "i32", "i64", "f64",
        "iq1_m", "bf16", "q4_0_4_4", "q4_0_4_8", "q4_0_8_8", "tq1_0", "tq2_0",
    };
    if (ggml_type == 39) return "mxfp4";
    return ggml_type < std::size(kNames) ? kNames[ggml_type] : "unknown";
}

size_t dtype_row_bytes(DType type, int64_t n_elements) noexcept {
    const DTypeTraits t = dtype_traits(type);
    return static_cast<size_t>(n_elements / t.block_size) * static_cast<size_t>(t.block_bytes);
}

const char* backend_kind_name(BackendKind kind) noexcept {
    switch (kind) {
        case BackendKind::Cpu: return "cpu";
        case BackendKind::Metal: return "metal";
        case BackendKind::Vulkan: return "vulkan";
        case BackendKind::Qnn: return "qnn";
        case BackendKind::NeuroPilot: return "neuropilot";
    }
    return "unknown";
}

ProcessorClass processor_class(BackendKind kind) noexcept {
    switch (kind) {
        case BackendKind::Metal:
        case BackendKind::Vulkan: return ProcessorClass::Gpu;
        case BackendKind::Qnn:
        case BackendKind::NeuroPilot: return ProcessorClass::Npu;
        case BackendKind::Cpu: break;
    }
    return ProcessorClass::Cpu;
}

}  // namespace liyab
