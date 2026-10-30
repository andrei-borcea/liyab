// Liyab — kernel conformance (internal): what a backend does with each weight format on this host.
//
// The engine sends a matrix product to a backend, and when the backend answers ErrorCode::Unsupported it runs that
// one product on the CPU instead. Which formats a backend really runs, and whether it runs them right, used to
// live in switch statements and a few hand-picked tests. This module answers by running: for every weight format
// the model loader recognizes it builds random but valid weights, multiplies them by activations on the backend,
// and compares the result with a reference computed from the weights' own decoded values.
//
// What is checked, per backend and format:
//   * Placement. The weights sit in ordinary memory (what a memory-mapped model file looks like, 32 bytes into a
//     page like a GGUF tensor) and, when the backend offers it, in memory the backend hands out for reading in
//     place (Backend::allocate_shared), which is then refilled and checked again: a backend must not keep the old
//     contents of a region it was told may be reused.
//   * Shapes. One activation row (decoding) and several (verification, prefill), on row counts that are not
//     multiples of any tile, and on one and several blocks per row.
//   * The answer. Each output must be within 1e-5 of sum|w*x| of the reference, which is computed in double
//     precision from quant::dequantize_row of the same bytes (the project's decoder, which test_engine pins to
//     llama.cpp's output: this check compares a kernel with the decoder, not the decoder with the format). The
//     activations are chosen on the int8 grid the CPU quantizes them to (every 32 values: multiples of 2^-6, one
//     of them +-127 * 2^-6), so that quantizing them loses nothing and the tolerance can be as tight as float
//     rounding allows: correct float32 kernels deviate by about 5e-8, so 1e-5 leaves 200 times for another order
//     of summation, yet sees a wrong scale, sign or bit in any block (hundreds of times over) and an accumulator
//     in half precision (about 2e-5). Backends that read float activations are held to the same bound.
//   * The claim. Backend::supports(type) must say whether the backend runs the format: true exactly when some
//     placement and shape ran. A wrong claim sends work to the wrong place or hides a gap.
//   * Batching. Whether an n-row call, n one-row calls and a grouped call (matmul_group) give the same bits. The
//     CPU backend must (speculative verification and concurrent sequences rely on it); other backends are only
//     reported.
//
// With Options::timing it also measures, on a matrix of Options::timing_bytes, the weight bytes read per second
// by a one-row call and the cost per activation row of a 16-row call relative to it (below 1 means the kernel
// reuses each weight tile across rows; about 1 means it reads the weights again for every row). The figures are
// indicative (a few calls each, the minimum of a short run): liyab-bench is the instrument for stable CPU numbers.
//
// The weights are random bytes, kept valid format by format with one rule that needs no knowledge of any layout:
// a block is redrawn until its decoded values are finite and its largest magnitude lies in [0.05, 8]. A format
// added to dtype_from_ggml is therefore checked without touching this file.
//
// Not thread-safe: one check_backend() at a time per backend. Weights in ordinary memory stay allocated until the
// process ends, because a backend may cache a copy or a wrapper keyed by address (Vulkan, Metal), so no address may
// ever hold two different tensors; shared regions and timing matrices are released.
#ifndef LIYAB_CORE_KERNEL_CONFORMANCE_H
#define LIYAB_CORE_KERNEL_CONFORMANCE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "liyab/backend.h"
#include "liyab/types.h"

namespace liyab::conformance {

// The weight formats of GGUF files, in file-type order: every type dtype_from_ggml() recognizes except I32, which
// holds indices. Q8_1 and Q8_K are activation formats and the repacked layouts exist only in memory: none is listed.
std::vector<DType> weight_formats();

// Where the weights live during a call.
enum class Residency : uint8_t {
    Heap,    // ordinary memory, like a memory-mapped model file
    Shared,  // memory from Backend::allocate_shared, filled in place and refilled
};

enum class Verdict : uint8_t {
    Ok,       // every shape ran and matches the reference
    Partial,  // some shapes ran and match; the others were refused (the engine runs those on the CPU)
    Refused,  // every shape was refused with ErrorCode::Unsupported: the engine runs the operation on the CPU
    Wrong,    // a shape ran and differs from the reference beyond the tolerance, or returned NaN
    Failed,   // a call returned an error other than Unsupported, or no valid weights could be built
};

// What one backend did with one format, one placement and one kind of call (decode: one activation row; batch:
// several).
struct Cell {
    Verdict verdict = Verdict::Refused;
    double worst = 0.0;      // the largest |result - reference| / tolerance over the outputs that ran (<= 1: within)
    bool exact = true;       // batch cells: n-row call, n one-row calls and a grouped call gave the same bits
    double gb_per_s = 0.0;   // decode cells, when timed: weight bytes read per second, in GB/s
    double batch_cost = 0.0; // batch cells, when timed: time per activation row of a 16-row call / time of a 1-row call
    std::string detail;      // the first refusal or failure message, or the first shape that disagreed
};

struct PlacementReport {
    Residency where = Residency::Heap;
    Cell decode;
    Cell batch;
};

struct FormatReport {
    DType type = DType::F32;
    bool claimed = false;  // Backend::supports(type)
    std::vector<PlacementReport> placements;  // Heap, then Shared when the backend offers shared memory
};

struct BackendReport {
    BackendKind kind = BackendKind::Cpu;
    std::string name;  // Backend::description()
    std::vector<FormatReport> formats;
};

struct Options {
    // Measure GB/s and batch cost. Off in unit tests: timings are noisy and cost time.
    bool timing = false;
    // Weight bytes of the matrix a timing runs on: larger than the last-level cache of most hosts, so a format is
    // measured reading memory and not cache.
    size_t timing_bytes = size_t{128} << 20;
    // Check only these formats; empty: all of weight_formats().
    std::vector<DType> only;
};

// Runs every check above on `backend`. Never throws; a call that crashes the backend crashes the process, which
// is itself a finding.
BackendReport check_backend(Backend& backend, const Options& options = {});

// What is wrong with a report, one line each; empty when it passes. A report fails when
//   * any cell is Wrong or Failed;
//   * a format's claim differs from whether it ran anywhere (claims a format it never ran, or runs one it does
//     not claim);
//   * for the CPU backend, any cell is not Ok or not exact: the CPU is the fallback of last resort and the
//     numerical reference, so it must run every format, right, whatever the batch.
std::vector<std::string> problems(const BackendReport& report);

// Whether any cell of the format ran (Ok or Partial).
bool ran_anywhere(const FormatReport& format) noexcept;

const char* verdict_name(Verdict verdict) noexcept;
const char* residency_name(Residency where) noexcept;

}  // namespace liyab::conformance

#endif  // LIYAB_CORE_KERNEL_CONFORMANCE_H
