// Kernel conformance: every backend this host offers, against every weight format the loader recognizes
// (src/core/kernel_conformance.h), and the check itself against backends that are wrong on purpose.
//
//   ./test_kernels
//
// liyab-kernels prints the same check as a table, with throughput.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/host_profile.h"
#include "core/kernel_conformance.h"
#include "core/model_bytes.h"
#include "core/model_fit.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "liyab/backend.h"
#include "liyab/mmap_loader.h"
#include "test_model.h"
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

TEST_CASE("A token's read bytes follow the tensor table: no embedding rows when a head exists, the router's share of the experts") {
    const std::string path = test::temp_dir() + "/liyab_model_bytes_" + std::to_string(test::process_id()) + ".gguf";
    auto bytes_of = [&](const test::TinyModelSpec& spec, ModelBytes* out, double* embd, double* experts) {
        test::write_tiny_model(path, spec);
        auto loader = MmapLoader::open(path);
        REQUIRE(loader.has_value());
        *out = read_model_bytes(*loader.value());
        *embd = *experts = 0.0;
        for (const TensorView& t : loader.value()->tensors()) {
            if (t.name == "token_embd.weight") *embd = static_cast<double>(t.nbytes);
            if (t.name.find("_exps") != std::string_view::npos) *experts += static_cast<double>(t.nbytes);
        }
    };
    auto total = [](const ModelBytes& m) {
        double sum = 0.0;
        for (const auto& entry : m.per_token) sum += entry.second;
        return sum;
    };
    ModelBytes dense;
    double embd = 0.0, experts = 0.0;
    test::TinyModelSpec spec;
    bytes_of(spec, &dense, &embd, &experts);
    CHECK(dense.architecture == "llama");
    CHECK(dense.blocks == spec.n_layers);
    CHECK(embd > 0.0);
    CHECK(total(dense) == dense.weight_bytes - embd);  // a separate head: a token reads one embedding row, not the table
    CHECK(dense.per_token.size() >= 2);
    CHECK(std::is_sorted(dense.per_token.begin(), dense.per_token.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; }));

    spec.tied_output = true;  // the embedding table is the head: all of it is read
    ModelBytes tied;
    bytes_of(spec, &tied, &embd, &experts);
    CHECK(total(tied) == tied.weight_bytes);

    test::TinyModelSpec moe;
    moe.arch = "qwen3moe";
    moe.n_expert = 4;
    moe.n_expert_used = 2;
    ModelBytes some;
    bytes_of(moe, &some, &embd, &experts);
    CHECK(experts > 0.0);
    CHECK(total(some) == some.weight_bytes - embd - experts * 0.5);  // 2 of 4 experts per token
    moe.n_expert_used = 4;
    ModelBytes all;
    bytes_of(moe, &all, &embd, &experts);
    CHECK(total(all) == all.weight_bytes - embd);
    std::remove(path.c_str());
}

TEST_CASE("The decode ceiling follows the bytes a token reads and the speeds measured; a refused format runs at the CPU's") {
    ModelBytes model;
    model.architecture = "llama";
    model.file_bytes = uint64_t{4} << 30;
    model.blocks = 8;
    model.weight_bytes = 4.0e9;
    model.per_token = {{DType::Q4_K, 3.0e9}, {DType::Q6_K, 1.0e9}};  // 4 GB per token
    DeviceInfo device;
    device.available_memory = uint64_t{16} << 30;
    const std::vector<DType> timed = {DType::Q4_K, DType::Q6_K};
    const BackendSpeed cpu{BackendKind::Cpu, "cpu", [](DType type) { return type == DType::Q4_K ? 200.0 : 100.0; }};
    const BackendSpeed gpu{BackendKind::Metal, "gpu", [](DType type) { return type == DType::Q4_K ? 400.0 : 0.0; }};
    const BackendSpeed blind{BackendKind::Vulkan, "blind", [](DType) { return 0.0; }};
    ModelFit fit = describe_model_fit("m.gguf", model, device, {cpu, gpu, blind}, timed);
    REQUIRE(fit.ceiling.size() == 3);
    CHECK(std::fabs(fit.ceiling[0] - 1.0 / (3.0e9 / 200e9 + 1.0e9 / 100e9)) < 1e-9);  // 1 / (15 ms + 10 ms) = 40
    CHECK(std::fabs(fit.ceiling[1] - 1.0 / (3.0e9 / 400e9 + 1.0e9 / 100e9)) < 1e-9);  // Q6_K refused: at the CPU's speed
    CHECK(fit.ceiling[2] == fit.ceiling[0]);  // a backend that runs nothing: all of it at the CPU's speed
    CHECK(fit.text.find("memory: fits") != std::string::npos);
    CHECK(fit.text.find("cpu 100.0") != std::string::npos);  // the refusal is shown as the CPU's figure

    // A format with no timing leaves the ceiling unknown; a format left out of `timed` is said to be left out.
    const BackendSpeed no_cpu_q6{BackendKind::Cpu, "cpu", [](DType type) { return type == DType::Q4_K ? 200.0 : 0.0; }};
    fit = describe_model_fit("m.gguf", model, device, {no_cpu_q6}, timed);
    CHECK(fit.ceiling[0] == 0.0);
    CHECK(fit.text.find("unknown") != std::string::npos);
    fit = describe_model_fit("m.gguf", model, device, {cpu}, {DType::Q4_K});
    CHECK(fit.text.find("left out of the ceilings") != std::string::npos);
    CHECK(std::fabs(fit.ceiling[0] - 200e9 / 3.0e9) < 1e-9);

    device.available_memory = uint64_t{2} << 30;  // 4 GiB of weights do not fit
    fit = describe_model_fit("m.gguf", model, device, {cpu}, timed);
    CHECK(fit.text.find("streams layers from storage") != std::string::npos);
}

TEST_CASE("The formats worth timing are the ones a token reads in numbers; the host profile keeps what was timed") {
    ModelBytes model;
    model.per_token = {{DType::Q4_K, 9.0e9}, {DType::Q6_K, 0.97e9}, {DType::F32, 0.003e9}};
    const std::vector<DType> formats = formats_to_time(model);
    CHECK(formats == (std::vector<DType>{DType::Q4_K, DType::Q6_K}));  // 0.03% of the bytes: not worth a timing

    const std::string dir = test::temp_dir() + "/liyab_host_profile_" + std::to_string(test::process_id());
    std::filesystem::remove_all(dir);
    DeviceInfo device;
    device.soc.name = "test soc";
    device.total_memory = uint64_t{8} << 30;
    device.backend_order = {BackendKind::Cpu};
    {
        HostProfile profile(device, 4, dir);
        CHECK(!profile.speed(BackendKind::Cpu, DType::Q4_K).has_value());
        CHECK(!profile.knows(BackendKind::Cpu));
        profile.set(BackendKind::Cpu, DType::Q4_K, 120.0f);
        profile.set(BackendKind::Cpu, DType::Q6_K, 0.0f);  // timed, not run
        profile.set(BackendKind::Cpu, DType::Q4_K, 130.0f);  // timed again: replaced
        profile.save();
    }
    {
        HostProfile again(device, 4, dir);
        CHECK(again.knows(BackendKind::Cpu));
        CHECK(!again.knows(BackendKind::Metal));
        REQUIRE(again.speed(BackendKind::Cpu, DType::Q4_K).has_value());
        CHECK(*again.speed(BackendKind::Cpu, DType::Q4_K) == 130.0f);
        REQUIRE(again.speed(BackendKind::Cpu, DType::Q6_K).has_value());
        CHECK(*again.speed(BackendKind::Cpu, DType::Q6_K) == 0.0f);
        // Another thread count or memory size is another host: its speeds are not these.
        CHECK(!HostProfile(device, 8, dir).speed(BackendKind::Cpu, DType::Q4_K).has_value());
        DeviceInfo bigger = device;
        bigger.total_memory = uint64_t{16} << 30;
        CHECK(!HostProfile(bigger, 4, dir).speed(BackendKind::Cpu, DType::Q4_K).has_value());
    }
    std::filesystem::remove_all(dir);
}

TEST_MAIN()
