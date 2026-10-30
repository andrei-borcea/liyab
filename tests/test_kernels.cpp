// Kernel conformance: every backend this host offers, against every weight format the loader recognizes
// (src/core/kernel_conformance.h), and the check itself against backends that are wrong on purpose.
//
//   ./test_kernels
//
// liyab-kernels prints the same check as a table, with throughput.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/kernel_conformance.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "liyab/backend.h"
#include "test_util.h"

using namespace liyab;
using namespace liyab::conformance;

namespace {

const FormatReport* find(const BackendReport& report, DType type) {
    for (const FormatReport& f : report.formats) {
        if (f.type == type) return &f;
    }
    return nullptr;
}

void print_problems(const std::vector<std::string>& found) {
    for (const std::string& line : found) std::printf("  %s\n", line.c_str());
}

// Wraps the CPU backend and misbehaves in one way per format, to show that the check notices each.
class Faulty : public Backend {
public:
    Faulty(ThreadPool& pool, BackendKind kind) : cpu_(make_cpu_backend(pool)), kind_(kind) {}

    [[nodiscard]] BackendKind kind() const noexcept override { return kind_; }
    [[nodiscard]] std::string description() const override { return "faulty test backend"; }
    // Claims everything except IQ4_XS (which it runs fine) and claims Q2_K (which it refuses).
    [[nodiscard]] bool supports(DType type) const noexcept override { return type != DType::IQ4_XS; }

    Status matmul(const TensorView& w, const float* x, float* y, int32_t n) override {
        switch (w.type) {
            case DType::Q2_K: return Status(ErrorCode::Unsupported, "no kernel");
            case DType::Q6_K: return Status(ErrorCode::BackendError, "device lost");
            case DType::IQ4_NL: return Status::ok();  // answers without writing the output
            default: break;
        }
        Status s = cpu_->matmul(w, x, y, n);
        if (!s.is_ok()) return s;
        const size_t count = static_cast<size_t>(n) * static_cast<size_t>(w.rows());
        if (w.type == DType::Q5_K) {  // one percent too high
            for (size_t i = 0; i < count; ++i) y[i] *= 1.01f;
        } else if (w.type == DType::Q4_1) {  // right to half precision: what an fp16 accumulator would give
            for (size_t i = 0; i < count; ++i) y[i] = quant::fp16_to_fp32(quant::fp32_to_fp16(y[i]));
        } else if (w.type == DType::Q8_0 && n > 1) {  // within the tolerance, but not the single-row bits
            for (size_t i = 0; i < count; ++i) y[i] = std::nextafter(y[i], 1e30f);
        }
        return Status::ok();
    }

private:
    std::unique_ptr<Backend> cpu_;
    BackendKind kind_;
};

// A backend that offers shared memory and copies the weights of a region the first time it sees it, then keeps
// using the copy: the bug the refill check exists for.
class StaleCache : public Backend {
public:
    explicit StaleCache(ThreadPool& pool) : cpu_(make_cpu_backend(pool)) {}

    [[nodiscard]] BackendKind kind() const noexcept override { return BackendKind::Vulkan; }
    [[nodiscard]] std::string description() const override { return "stale-cache test backend"; }
    [[nodiscard]] bool supports(DType) const noexcept override { return true; }

    Status matmul(const TensorView& w, const float* x, float* y, int32_t n) override {
        auto it = copies_.find(w.data);
        if (it == copies_.end()) it = copies_.emplace(w.data, std::vector<uint8_t>(w.data, w.data + w.nbytes)).first;
        TensorView copy = w;
        copy.data = it->second.data();
        return cpu_->matmul(copy, x, y, n);
    }

    uint8_t* allocate_shared(size_t bytes) override {
        regions_.push_back(std::make_unique<uint8_t[]>(bytes + 4096));
        return reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(regions_.back().get()) + 4095) / 4096 * 4096);
    }
    void free_shared(uint8_t*) noexcept override {}  // regions are reused by address in this fake

private:
    std::unique_ptr<Backend> cpu_;
    std::map<const uint8_t*, std::vector<uint8_t>> copies_;
    std::vector<std::unique_ptr<uint8_t[]>> regions_;
};

}  // namespace

TEST_CASE("The weight formats checked are the ones the loader recognizes") {
    const std::vector<DType> formats = weight_formats();
    std::vector<DType> sorted = formats;
    std::sort(sorted.begin(), sorted.end());
    CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
    size_t recognized = 0;
    for (uint32_t id = 0; id < 256; ++id) {
        DType type;
        if (!dtype_from_ggml(id, type) || type == DType::I32) continue;
        ++recognized;
        CHECK(std::find(formats.begin(), formats.end(), type) != formats.end());
    }
    CHECK(formats.size() == recognized);
    CHECK(recognized >= 26);
    for (const DType repacked : {DType::Q4_K_R8, DType::Q5_K_R8, DType::Q6_K_R8, DType::Q8_0_R4, DType::I32}) {
        CHECK(std::find(formats.begin(), formats.end(), repacked) == formats.end());
    }
}

TEST_CASE("The CPU backend runs every weight format right, whatever the batch") {
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const BackendReport report = check_backend(*cpu);
    CHECK(report.formats.size() == weight_formats().size());
    const std::vector<std::string> found = problems(report);
    print_problems(found);
    CHECK(found.empty());
}

TEST_CASE("Metal and Vulkan do what Backend::supports() says, and right") {
    for (const char* name : {"metal", "vulkan"}) {
        auto made = std::string(name) == "metal" ? make_metal_backend() : make_vulkan_backend();
        if (!made) {
            std::printf("  %s skipped: %s\n", name, made.status().to_string().c_str());
            continue;
        }
        const BackendReport report = check_backend(*made.value());
        const std::vector<std::string> found = problems(report);
        std::printf("  %s: %zu formats claimed, %zu problems\n", name,
                    static_cast<size_t>(std::count_if(report.formats.begin(), report.formats.end(),
                                                      [](const FormatReport& f) { return f.claimed; })),
                    found.size());
        print_problems(found);
        CHECK(found.empty());
    }
}

TEST_CASE("The check notices a wrong answer, a refusal, an error, a missing write and a false claim") {
    ThreadPool pool(2);
    Faulty faulty(pool, BackendKind::Metal);
    Options only;
    only.only = {DType::Q4_0, DType::Q5_K, DType::Q4_1, DType::Q2_K, DType::Q6_K, DType::IQ4_NL, DType::IQ4_XS};
    const BackendReport report = check_backend(faulty, only);
    auto verdicts = [&](DType type) { return find(report, type)->placements.front(); };

    CHECK(verdicts(DType::Q4_0).decode.verdict == Verdict::Ok);       // untouched: passes
    CHECK(verdicts(DType::Q5_K).decode.verdict == Verdict::Wrong);    // 1% too high
    CHECK(verdicts(DType::Q5_K).decode.worst > 1.0);
    CHECK(verdicts(DType::Q4_1).decode.verdict == Verdict::Wrong);    // rounded through half precision
    CHECK(verdicts(DType::Q2_K).decode.verdict == Verdict::Refused);  // Unsupported
    CHECK(verdicts(DType::Q6_K).decode.verdict == Verdict::Failed);   // another error
    CHECK(verdicts(DType::IQ4_NL).decode.verdict == Verdict::Wrong);  // Ok without writing: NaN
    CHECK(verdicts(DType::IQ4_XS).decode.verdict == Verdict::Ok);     // runs, but is not claimed

    const std::vector<std::string> found = problems(report);
    auto mentions = [&](const std::string& what) {
        return std::any_of(found.begin(), found.end(), [&](const std::string& l) { return l.find(what) != std::string::npos; });
    };
    CHECK(mentions("q5_k heap decode: wrong result"));
    CHECK(mentions("q6_k heap decode: failed"));
    CHECK(mentions("iq4_nl heap decode: wrong result"));
    CHECK(mentions("q2_k: supports() says yes, but no placement or shape ran"));
    CHECK(mentions("iq4_xs: ran, but supports() says no"));
    CHECK(!mentions("q4_0"));
}

TEST_CASE("The check holds the CPU to every format, and to results that do not depend on the batch") {
    ThreadPool pool(2);
    Faulty pretender(pool, BackendKind::Cpu);
    Options only;
    only.only = {DType::Q2_K, DType::Q8_0};
    const BackendReport report = check_backend(pretender, only);
    CHECK(!find(report, DType::Q8_0)->placements.front().batch.exact);
    const std::vector<std::string> found = problems(report);
    auto mentions = [&](const std::string& what) {
        return std::any_of(found.begin(), found.end(), [&](const std::string& l) { return l.find(what) != std::string::npos; });
    };
    CHECK(mentions("q2_k heap decode: the CPU backend must run every format"));
    CHECK(mentions("q8_0 heap batch: the CPU backend's results must not depend on the batch"));

    // The same drift on another backend is reported, not a problem.
    Faulty gpu(pool, BackendKind::Metal);
    const BackendReport other = check_backend(gpu, only);
    CHECK(!find(other, DType::Q8_0)->placements.front().batch.exact);
    const std::vector<std::string> other_found = problems(other);
    CHECK(std::none_of(other_found.begin(), other_found.end(),
                       [](const std::string& l) { return l.find("q8_0") != std::string::npos; }));
}

TEST_CASE("The check refills shared memory and notices a backend that kept the old contents") {
    ThreadPool pool(2);
    StaleCache stale(pool);
    Options only;
    only.only = {DType::Q4_0, DType::Q8_0};
    const BackendReport report = check_backend(stale, only);
    for (const FormatReport& f : report.formats) {
        REQUIRE(f.placements.size() == 2);
        CHECK(f.placements[0].where == Residency::Heap);
        CHECK(f.placements[0].decode.verdict == Verdict::Ok);  // filled once: the copy is right
        CHECK(f.placements[1].where == Residency::Shared);
        CHECK(f.placements[1].decode.verdict == Verdict::Wrong);  // refilled in place: the copy is stale
        CHECK(f.placements[1].decode.detail.find("after a refill") != std::string::npos);
    }
}

TEST_CASE("Timing reports a throughput and a batch cost, and stays off by default") {
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    Options options;
    options.only = {DType::Q4_K};
    BackendReport untimed = check_backend(*cpu, options);
    CHECK(untimed.formats.front().placements.front().decode.gb_per_s == 0.0);
    options.timing = true;
    options.timing_bytes = size_t{2} << 20;
    BackendReport timed = check_backend(*cpu, options);
    const PlacementReport& placement = timed.formats.front().placements.front();
    CHECK(placement.decode.gb_per_s > 0.0);
    CHECK(placement.batch.batch_cost > 0.0);
}

TEST_MAIN()
