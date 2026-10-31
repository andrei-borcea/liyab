// liyab-seq-bench — several sequences decoded together (Engine::open_sequence):
// tokens per second per sequence and in total for 1..N sequences decoding at
// once, against the same generations taking turns on one sequence (each
// alone, with speculative drafts when --lookup is given), greedy.
//
//   liyab-seq-bench -m model.gguf [--sequences 1,2,3,4] [-n 128] [--repeat R]
//                   [--threads N] [--backend cpu|metal|auto] [--lookup]
//                   [--kv q8_0|f16] [--expert-cache MB] [--memory-budget MB]
//
// Each sequence gets a task of its own (a long text to write, through the
// model's chat template, thinking off where the template has the switch).
// Its prompt is processed first (prefill() up to its last token, not
// timed); then two runs:
//   together  every sequence's generation starts at once, one thread each:
//             the engine decodes them in shared passes; "seq tok/s" is the
//             median sequence's decode rate, "slowest" the slowest one's, and
//             "together" every token over the wall time from the start to the
//             last one's end;
//   in turns  the same generations one after another (each alone: the engine
//             runs them as a single conversation, drafts included with
//             --lookup): every token over the sum of their times.
// "gain" is together / in turns. A repeat runs both back to back, in
// alternating order (on a busy host the speed drifts; the gain compares runs
// that saw the same conditions); --repeat R prints the median of R repeats.
// Every together reply is checked against its reply alone ("same": greedy
// decoding, and the CPU and Metal kernels compute a row the same whatever
// rows share its pass). The last together run's scheduler counters follow:
// passes shared, time in a pass, time a leader waited for sequences about to
// submit. Every run on a fresh engine of max(sequences) sequences (suffix
// drafts keep earlier conversations: a run never drafts from an earlier
// one's replies), unpaced (Performance profile), warmed up on an unrelated
// text.
//
// Uses only the public C++ API; built with the tests (LIYAB_BUILD_TESTS).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "liyab/engine.h"

namespace {

using liyab::Engine;
using liyab::EngineConfig;
using liyab::GenerationStats;
using Clock = std::chrono::steady_clock;

constexpr const char* kTasks[] = {
    "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle. Describe the sea, the "
    "weather and the keeper's thoughts in many paragraphs.",
    "Explain in detail, step by step, how a refrigerator works: the compressor, the condenser, the expansion valve and "
    "the evaporator, and why each one matters. Write at least ten paragraphs.",
    "Describe a full week of meals for a family of four: breakfast, lunch and dinner for each day, with the "
    "ingredients and how to cook each dish.",
    "Write a long travel guide to a coastal town of your choice: its neighbourhoods, food, museums, walks and day "
    "trips, in many paragraphs.",
};
constexpr const char* kWarmUp = "Hello! Say something short.";

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream in(s);
    std::string item;
    while (std::getline(in, item, ',')) out.push_back(std::atoi(item.c_str()));
    return out;
}

struct Run {
    std::string reply;
    GenerationStats stats;
    bool ok = false;
};

Run generate(Engine& engine, Engine::Sequence* seq, const std::string& prompt, const liyab::SamplingParams& params) {
    Run run;
    auto on_token = [&](std::string_view piece, int32_t) {
        run.reply += piece;
        return true;
    };
    auto r = seq != nullptr ? seq->generate(prompt, params, on_token) : engine.generate(prompt, params, on_token);
    if (!r) {
        std::fprintf(stderr, "generation failed: %s\n", r.status().to_string().c_str());
        return run;
    }
    run.stats = *r;
    run.ok = true;
    return run;
}

// An engine and its sequences for one run.
struct Bench {
    std::unique_ptr<Engine> engine;
    std::vector<std::unique_ptr<Engine::Sequence>> sequences;
    Engine::Sequence* sequence(int i) { return i == 0 ? nullptr : sequences[static_cast<size_t>(i - 1)].get(); }
    ~Bench() { sequences.clear(); }  // before the engine
};

// A prompt processed up to its last byte (the generation prompt's final
// newline, its last token), so the timed generation decodes only: its prompt
// continues the context, whose last token seeds decoding (on a hybrid model a
// context holding the whole prompt would be processed again from a snapshot).
void prepare(Engine& engine, Engine::Sequence* seq, const std::string& prompt) {
    const std::string head = prompt.substr(0, prompt.size() - 1);
    if (seq != nullptr) {
        (void)seq->prefill(head);
    } else {
        (void)engine.prefill(head);
    }
}

}  // namespace

int main(int argc, char** argv) {
    EngineConfig config;
    config.power.profile = liyab::PowerProfile::Performance;
    config.thermal_polling = false;
    std::vector<int> counts = {1, 2, 3, 4};
    int32_t tokens = 128;
    int32_t repeat = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m") {
            config.model_path = next();
        } else if (a == "--sequences") {
            counts = parse_list(next());
        } else if (a == "-n") {
            tokens = std::atoi(next().c_str());
        } else if (a == "--repeat") {
            repeat = std::max(1, std::atoi(next().c_str()));
        } else if (a == "--threads") {
            config.n_threads = std::atoi(next().c_str());
        } else if (a == "--backend") {
            const std::string b = next();
            if (b == "cpu") config.backend = liyab::BackendKind::Cpu;
            if (b == "metal") config.backend = liyab::BackendKind::Metal;
        } else if (a == "--lookup") {
            config.lookup_drafts = EngineConfig::LookupDrafts::On;
        } else if (a == "--kv") {
            const std::string k = next();
            config.kv_cache_type = k == "f16" ? liyab::KvCacheType::F16 : liyab::KvCacheType::Q8_0;
        } else if (a == "--expert-cache") {
            config.expert_cache_mb = std::atoll(next().c_str());
        } else if (a == "--memory-budget") {
            config.memory_budget_mb = std::atoll(next().c_str());
        } else {
            std::fprintf(stderr,
                         "usage: liyab-seq-bench -m MODEL [--sequences 1,2,3,4] [-n 128] [--repeat R] [--threads N]\n"
                         "                       [--backend cpu|metal|auto] [--lookup] [--kv q8_0|f16]\n"
                         "                       [--expert-cache MB] [--memory-budget MB]\n");
            return 2;
        }
    }
    if (config.model_path.empty() || counts.empty()) {
        std::fprintf(stderr, "a model (-m) and sequence counts are required\n");
        return 2;
    }
    const int most = *std::max_element(counts.begin(), counts.end());
    if (*std::min_element(counts.begin(), counts.end()) < 1 || most > 4) {
        std::fprintf(stderr, "1 to 4 sequences (one task each)\n");
        return 2;
    }
    config.max_sequences = most;
    std::vector<std::string> prompts;
    std::string description;
    // A fresh engine for every run: suffix drafts keep earlier
    // conversations, and a run must not draft from an earlier one's replies.
    auto make = [&](int k) -> std::unique_ptr<Bench> {
        auto created = Engine::create(config);
        if (!created) {
            std::fprintf(stderr, "%s\n", created.status().to_string().c_str());
            return nullptr;
        }
        auto bench = std::make_unique<Bench>();
        bench->engine = std::move(created).value();
        for (int i = 1; i < k; ++i) {
            auto seq = bench->engine->open_sequence();
            if (!seq) {
                std::fprintf(stderr, "open_sequence: %s\n", seq.status().to_string().c_str());
                return nullptr;
            }
            bench->sequences.push_back(std::move(seq).value());
        }
        if (prompts.empty()) {  // the tasks through the model's chat template (plain text without one)
            for (const char* task : kTasks) {
                const std::string json = R"({"messages":[{"role":"user","content":")" + std::string(task) +
                                         R"("}],"add_generation_prompt":true,"enable_thinking":false})";
                auto rendered = bench->engine->render_chat(json);
                prompts.push_back(rendered ? *rendered : std::string(task) + "\n");
            }
            description = bench->engine->describe();
        }
        liyab::SamplingParams warm;
        warm.temperature = 0.0f;
        warm.max_tokens = 8;
        (void)generate(*bench->engine, nullptr, kWarmUp, warm);
        bench->engine->reset_context();
        return bench;
    };
    liyab::SamplingParams params;
    params.temperature = 0.0f;
    params.max_tokens = tokens;
    bool header = false;
    // The same generations one after another, each alone; returns every
    // token over the sum of their times.
    auto run_turns = [&](int k, std::vector<Run>& turns) -> double {
        auto bench = make(k);
        if (!bench) return -1.0;
        if (!header) {
            std::printf("%s\n%-4s %10s %10s %12s %12s %8s %6s   %s\n", description.c_str(), "seqs", "seq tok/s",
                        "slowest", "together", "in turns", "gain", "same", "scheduler");
            header = true;
        }
        Engine& engine = *bench->engine;
        double ms = 0.0;
        int64_t generated = 0;
        for (int i = 0; i < k; ++i) {
            engine.reset_context();
            prepare(engine, nullptr, prompts[static_cast<size_t>(i)]);
            const auto t0 = Clock::now();
            turns[static_cast<size_t>(i)] = generate(engine, nullptr, prompts[static_cast<size_t>(i)], params);
            ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            generated += turns[static_cast<size_t>(i)].stats.generated_tokens;
        }
        return generated * 1000.0 / ms;
    };
    // Every sequence at once, one thread each; returns every token over the
    // wall time, and each sequence's run.
    auto run_together = [&](int k, std::vector<Run>& runs, Engine::SchedulerStats& stats) -> double {
        auto bench = make(k);
        if (!bench) return -1.0;
        Engine& shared = *bench->engine;
        for (int i = 0; i < k; ++i) prepare(shared, bench->sequence(i), prompts[static_cast<size_t>(i)]);
        std::atomic<int> ready{0};
        std::vector<std::thread> threads;
        const auto t0 = Clock::now();
        for (int i = 0; i < k; ++i) {
            threads.emplace_back([&, i] {
                ready.fetch_add(1);
                while (ready.load() < k) std::this_thread::yield();
                runs[static_cast<size_t>(i)] = generate(shared, bench->sequence(i), prompts[static_cast<size_t>(i)], params);
            });
        }
        for (std::thread& t : threads) t.join();
        const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        int64_t generated = 0;
        for (const Run& run : runs) generated += run.stats.generated_tokens;
        stats = shared.scheduler_stats();
        return generated * 1000.0 / wall;
    };
    for (const int k : counts) {
        std::vector<double> together_total, together_seq, together_min, turns_total, gains;
        bool same = true;
        Engine::SchedulerStats last{};
        for (int r = 0; r < repeat; ++r) {
            // A repeat's two runs back to back, in alternating order: on a busy
            // host the speed drifts, and the gain compares runs that saw the
            // same conditions.
            std::vector<Run> turns(static_cast<size_t>(k)), runs(static_cast<size_t>(k));
            double in_turns = 0.0, together = 0.0;
            for (int half = 0; half < 2; ++half) {
                if ((half == 0) == (r % 2 == 0)) {
                    in_turns = run_turns(k, turns);
                } else {
                    together = run_together(k, runs, last);
                }
            }
            if (in_turns < 0.0 || together < 0.0) return 1;
            double slowest = 1e30;
            std::vector<double> rates;
            for (int i = 0; i < k; ++i) {
                const Run& run = runs[static_cast<size_t>(i)];
                rates.push_back(run.stats.tokens_per_second);
                slowest = std::min(slowest, run.stats.tokens_per_second);
                if (!run.ok || run.reply != turns[static_cast<size_t>(i)].reply) {
                    same = false;
                    std::fprintf(stderr, "sequence %d: the reply together differs from its reply alone\n", i);
                }
            }
            together_total.push_back(together);
            turns_total.push_back(in_turns);
            gains.push_back(together / in_turns);
            together_seq.push_back(median(rates));
            together_min.push_back(slowest);
        }
        const double passes = static_cast<double>(std::max<uint64_t>(last.passes, 1));
        std::printf("%-4d %10.1f %10.1f %12.1f %12.1f %8.2f %6s   %llu of %llu passes shared, %.2f ms a pass, %.2f ms "
                    "gathering\n",
                    k, median(together_seq), median(together_min), median(together_total), median(turns_total),
                    median(gains), same ? "yes" : "NO", static_cast<unsigned long long>(last.shared_passes),
                    static_cast<unsigned long long>(last.passes), last.pass_ms / passes, last.gather_ms / passes);
    }
    return 0;
}
