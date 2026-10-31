// liyab-kernels — which weight formats each backend runs on this host, and whether it runs them right.
//
// For every weight format the model loader recognizes and every backend this host offers (the CPU, plus Metal or
// Vulkan when compiled in and a device exists), it multiplies random valid weights by activations, compares the
// result with a reference decoded from the same bytes, and prints what happened: ran and matched (ok), refused
// (the engine runs that product on the CPU instead), or gave a wrong answer. See src/core/kernel_conformance.h for
// what exactly is checked, and why the tolerance can be that tight.
//
//   liyab-kernels [--backend cpu|metal|vulkan|all] [--only q4_k,iq2_xs,...] [--threads N] [--no-timing]
//                 [--model MODEL.gguf]
//
// With --model the tool also answers "how will this model run here": it checks the formats the model holds (and the
// CPU, which runs what a backend refuses), then prints the model's weights by format, whether they fit in memory,
// and for each backend the decode rate its measured read speeds allow at most (see print_model).
//
// Per format and placement of the weights (heap: ordinary memory, like a memory-mapped model; shared: memory the
// backend reads in place):
//   decode  one activation row. "ok 61.3 GB/s": weight bytes read per second on a matrix of ~128 MiB.
//   batch   several rows. "ok x0.31 =": the cost per row of a 16-row call relative to a 1-row call (about 1: the
//           weights are read again for every row; lower: the kernel reuses them), then "=" if an n-row call, n
//           one-row calls and a grouped call gave the same bits, "~" if they only agree within the tolerance.
// The throughput figures are indicative (caches warm for small formats, a few calls each); liyab-bench is the
// instrument for stable CPU numbers.
//
// Exit status 0 when nothing is wrong: no wrong or failed result, every claim of Backend::supports() true, and the
// CPU backend running every format identically for every batch. 1 otherwise, with the findings listed.
//
// Uses internal headers: built with the tests (LIYAB_BUILD_TESTS), also for Android (adb push it next to
// liyab-cli to read a phone's coverage).
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/kernel_conformance.h"
#include "core/model_bytes.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "liyab/backend.h"
#include "liyab/device_detect.h"
#include "liyab/mmap_loader.h"

namespace {

using namespace liyab;
using namespace liyab::conformance;

std::string number(const char* format, double value) {
    char text[32];
    std::snprintf(text, sizeof text, format, value);
    return text;
}

const char* backend_name(BackendKind kind) {
    switch (kind) {
        case BackendKind::Cpu: return "cpu";
        case BackendKind::Metal: return "metal";
        case BackendKind::Vulkan: return "vulkan";
        case BackendKind::Qnn: return "qnn";
        case BackendKind::NeuroPilot: return "neuropilot";
    }
}

std::string decode_text(const Cell& cell) {
    switch (cell.verdict) {
        case Verdict::Ok:
        case Verdict::Partial:
            return std::string(verdict_name(cell.verdict)) + (cell.gb_per_s > 0 ? " " + number("%.1f", cell.gb_per_s) + " GB/s" : "");
        case Verdict::Wrong: return "WRONG " + number("%.3g", cell.worst) + "x";
        default: return verdict_name(cell.verdict);
    }
}

std::string batch_text(const Cell& cell) {
    switch (cell.verdict) {
        case Verdict::Ok:
        case Verdict::Partial:
            return std::string(verdict_name(cell.verdict)) + (cell.batch_cost > 0 ? " x" + number("%.2f", cell.batch_cost) : "") +
                   (cell.exact ? " =" : " ~") + (cell.tflops > 0 ? " " + number("%.1f", cell.tflops) + "T" : "");
        case Verdict::Wrong: return "WRONG " + number("%.3g", cell.worst) + "x";
        default: return verdict_name(cell.verdict);
    }
}

void print_report(const BackendReport& report) {
    std::printf("\n== %s ==\n", report.name.c_str());
    const size_t claimed = static_cast<size_t>(std::count_if(report.formats.begin(), report.formats.end(),
                                                             [](const FormatReport& f) { return f.claimed; }));
    std::printf("claims %zu of %zu formats\n\n", claimed, report.formats.size());
    std::printf("%-8s %6s", "format", "bits/w");
    for (const PlacementReport& p : report.formats.front().placements) {
        std::printf("  %-18s %-18s", (std::string(residency_name(p.where)) + " decode").c_str(),
                    (std::string(residency_name(p.where)) + " batch").c_str());
    }
    std::printf("\n");
    for (const FormatReport& f : report.formats) {
        const DTypeTraits traits = dtype_traits(f.type);
        std::printf("%-8s %6.2f", std::string(traits.name).c_str(),
                    8.0 * traits.block_bytes / traits.block_size);
        for (const PlacementReport& p : f.placements) {
            std::printf("  %-18s %-18s", decode_text(p.decode).c_str(), batch_text(p.batch).c_str());
        }
        std::printf("%s\n", f.claimed == ran_anywhere(f) ? "" : f.claimed ? "  <- claimed, but never ran" : "  <- ran, but not claimed");
    }
    std::vector<std::string> on_cpu;
    double margin = 0.0;
    std::string margin_at;
    for (const FormatReport& f : report.formats) {
        if (!ran_anywhere(f)) on_cpu.emplace_back(dtype_traits(f.type).name);
        for (const PlacementReport& p : f.placements) {
            for (const Cell* cell : {&p.decode, &p.batch}) {
                if (cell->worst > margin) {
                    margin = cell->worst;
                    margin_at = std::string(dtype_traits(f.type).name) + " " + residency_name(p.where);
                }
            }
        }
    }
    std::printf("\nlargest deviation from the reference: %.3g of the tolerance (%s)\n", margin, margin_at.c_str());
    if (report.kind != BackendKind::Cpu && !on_cpu.empty()) {
        std::printf("%zu formats run on the CPU instead:", on_cpu.size());
        for (const std::string& name : on_cpu) std::printf(" %s", name.c_str());
        std::printf("\n");
    }
}

// The best decode read speed (GB/s) the report measured for `type`, 0 when the backend did not run it.
double decode_speed(const BackendReport& report, DType type) {
    double best = 0.0;
    for (const FormatReport& f : report.formats) {
        if (f.type != type) continue;
        for (const PlacementReport& p : f.placements) {
            if (p.decode.verdict == Verdict::Ok || p.decode.verdict == Verdict::Partial) best = std::max(best, p.decode.gb_per_s);
        }
    }
    return best;
}

// The model's weights by format, whether they fit, and per backend the most tokens per second its measured read
// speeds allow: each token reads its bytes once, a format the backend refuses at the speed the CPU reads it. The
// figure is an upper bound (it leaves out attention over the cache, recurrent state, norms and launch costs) built
// from short timings; the engine's automatic routing decides by timing real passes.
void print_model(const std::string& path, const ModelBytes& model, const std::vector<BackendReport>& reports,
                 const DeviceInfo& device, const std::vector<DType>& timed) {
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    std::printf("\n== model %s ==\n", path.c_str());
    std::printf("architecture %s, %lld blocks, %.2f GiB in the file\n", model.architecture.c_str(),
                static_cast<long long>(model.blocks), static_cast<double>(model.file_bytes) / kGiB);
    const uint64_t capacity = resident_weight_capacity_bytes(device.available_memory);
    if (capacity >= model.file_bytes) {
        std::printf("memory: fits (%.2f GiB available for weights, %.2f GiB needed)\n",
                    static_cast<double>(capacity) / kGiB, static_cast<double>(model.file_bytes) / kGiB);
    } else {
        std::printf("memory: %.2f GiB available for weights, %.2f GiB needed: the engine streams layers from storage\n",
                    static_cast<double>(capacity) / kGiB, static_cast<double>(model.file_bytes) / kGiB);
    }
    double per_token = 0.0;
    for (const auto& entry : model.per_token) per_token += entry.second;
    std::printf("read per decoded token: %.2f GiB\n\n", per_token / kGiB);
    const BackendReport* cpu = nullptr;
    for (const BackendReport& r : reports) {
        if (r.kind == BackendKind::Cpu) cpu = &r;
    }
    std::printf("%-8s %9s %6s", "format", "GiB/token", "share");
    for (const BackendReport& r : reports) std::printf("  %-10s", (std::string(backend_name(r.kind)) + " GB/s").c_str());
    std::printf("\n");
    double untimed = 0.0;
    std::vector<double> seconds(reports.size(), 0.0);
    std::vector<bool> known(reports.size(), true);
    for (const auto& [type, bytes] : model.per_token) {
        if (std::find(timed.begin(), timed.end(), type) == timed.end()) {
            untimed += bytes;
            continue;
        }
        std::printf("%-8s %9.3f %5.1f%%", std::string(dtype_traits(type).name).c_str(), bytes / kGiB, 100.0 * bytes / per_token);
        for (size_t i = 0; i < reports.size(); ++i) {
            double speed = decode_speed(reports[i], type);
            std::string text = number("%.1f", speed);
            if (speed <= 0.0 && cpu != nullptr) {
                speed = decode_speed(*cpu, type);
                text = speed > 0.0 ? "cpu " + number("%.1f", speed) : "none";
            }
            if (speed > 0.0) seconds[i] += bytes / (speed * 1e9);
            else known[i] = false;
            std::printf("  %-10s", text.c_str());
        }
        std::printf("\n");
    }
    if (untimed > 0.0) std::printf("(%.1f%% of the bytes are in formats too small to time, left out of the ceilings)\n", 100.0 * untimed / per_token);
    std::printf("\n");
    for (size_t i = 0; i < reports.size(); ++i) {
        if (known[i] && seconds[i] > 0.0) {
            std::printf("decode ceiling, %s: %.1f tokens/s\n", reports[i].name.c_str(), 1.0 / seconds[i]);
        } else {
            std::printf("decode ceiling, %s: unknown (a format of the model has no timing)\n", reports[i].name.c_str());
        }
    }
}

std::vector<DType> parse_formats(const std::string& list) {
    const std::vector<DType> known = weight_formats();
    std::vector<DType> out;
    size_t begin = 0;
    while (begin <= list.size()) {
        const size_t end = std::min(list.find(',', begin), list.size());
        std::string name = list.substr(begin, end - begin);
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
        const auto match = std::find_if(known.begin(), known.end(), [&](DType t) { return dtype_traits(t).name == name; });
        if (match == known.end()) {
            std::fprintf(stderr, "unknown format '%s'\n", name.c_str());
            std::exit(2);
        }
        out.push_back(*match);
        begin = end + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string which = "all";
    Options options;
    options.timing = true;
    int threads = 0;
    std::string model_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", arg.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--backend") which = value();
        else if (arg == "--only") options.only = parse_formats(value());
        else if (arg == "--threads") threads = std::atoi(value().c_str());
        else if (arg == "--no-timing") options.timing = false;
        else if (arg == "--model") model_path = value();
        else {
            std::fprintf(stderr,
                         "usage: liyab-kernels [--backend cpu|metal|vulkan|all] [--only q4_k,iq2_xs,...] "
                         "[--threads N] [--no-timing] [--model MODEL.gguf]\n");
            return arg == "--help" || arg == "-h" ? 0 : 2;
        }
    }
    if (which != "all" && which != "cpu" && which != "metal" && which != "vulkan") {
        std::fprintf(stderr, "unknown backend '%s'\n", which.c_str());
        return 2;
    }

    const DeviceInfo device = detect_device();
    std::printf("%s", describe_device(device).c_str());
    std::printf("CPU kernels: %s\n", quant::kernel_isa());

    // The model's formats are the ones worth timing: those that make up at least 0.5% of what a token reads (the
    // norms' F32 would only add noise), and the CPU is always checked, since it runs what a backend refuses.
    ModelBytes model;
    if (!model_path.empty()) {
        LoaderOptions loading;
        loading.streaming = false;
        auto loader = MmapLoader::open(model_path, loading);
        if (!loader) {
            std::fprintf(stderr, "cannot open %s: %s\n", model_path.c_str(), loader.status().to_string().c_str());
            return 2;
        }
        model = read_model_bytes(*loader.value());
        double total = 0.0;
        for (const auto& entry : model.per_token) total += entry.second;
        const std::vector<DType> known = weight_formats();
        for (const auto& [type, bytes] : model.per_token) {
            if (bytes >= 0.005 * total && std::find(known.begin(), known.end(), type) != known.end()) options.only.push_back(type);
        }
    }

    ThreadPool pool(threads);
    std::vector<std::unique_ptr<Backend>> backends;
    if (which == "all" || which == "cpu" || !model_path.empty()) backends.push_back(make_cpu_backend(pool));
    if (which == "all" || which == "metal") {
        auto made = make_metal_backend();
        if (made) backends.push_back(std::move(made).value());
        else if (which == "metal") std::printf("metal: %s\n", made.status().to_string().c_str());
    }
    if (which == "all" || which == "vulkan") {
        auto made = make_vulkan_backend();
        if (made) backends.push_back(std::move(made).value());
        else if (which == "vulkan") std::printf("vulkan: %s\n", made.status().to_string().c_str());
    }
    if (backends.empty()) {
        std::fprintf(stderr, "no backend to check\n");
        return 2;
    }

    std::vector<std::string> findings;
    std::vector<BackendReport> reports;
    for (const auto& backend : backends) {
        reports.push_back(check_backend(*backend, options));
        print_report(reports.back());
        for (const std::string& problem : problems(reports.back())) findings.push_back(reports.back().name + ": " + problem);
    }
    if (!model_path.empty()) print_model(model_path, model, reports, device, options.only);
    std::printf("\n");
    if (findings.empty()) {
        std::printf("no problems: every claim holds, no wrong result, and the CPU runs every format the same for any batch\n");
        return 0;
    }
    std::printf("%zu problems:\n", findings.size());
    for (const std::string& finding : findings) std::printf("  %s\n", finding.c_str());
    return 1;
}
