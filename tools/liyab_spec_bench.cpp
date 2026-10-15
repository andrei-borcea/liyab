// liyab-spec-bench — speculative decoding without a draft model, measured on
// the texts a phone assistant writes: plain decoding against context lookup
// and suffix drafts, greedy, on one model.
//
//   liyab-spec-bench -m model.gguf [--modes plain,lookup,suffix,suffix:8]
//                    [--tasks code,summary,tool,chat,again] [-n 512] [--repeat R]
//                    [--threads N] [--backend cpu|metal|auto]
//
// A mode is plain, lookup or suffix, optionally with its most drafts per step
// (":K", default EngineConfig::draft_tokens) and ":fixed" to verify every
// draft on offer instead of a learned count (acceptance and tokens per pass
// then do not depend on timing); the first mode is the reference (plain). Every mode has an engine of its own, unpaced, and the tasks run in
// this order, each on all modes back to back (so the modes share the load of
// the host), each a new conversation formatted by the model's chat template
// (thinking off where the template has the switch):
//   code     the model rewrites a code block with one change;
//   summary  a summary of a pasted text;
//   tool     two turns, each with the same tool output, the second asking
//            for it in full;
//   chat     free chat;
//   again    the tool conversation once more, as a new conversation (what an
//            engine keeps from earlier conversations, suffix drafts only).
// For each task and mode it prints the reply's tokens, decode tok/s, the
// speed-up over plain decoding, the share of drafts accepted, tokens per
// verification pass and whether the reply is the plain one (it must be:
// greedy speculative decoding is exact). --repeat R runs it all R times on
// fresh engines and prints the median tok/s and speed-up (taken within each
// run) next to the best run's (the least disturbed on a shared host).
//
// Uses only the public C++ API; built with the tests (LIYAB_BUILD_TESTS).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "liyab/engine.h"

namespace {

using liyab::Engine;
using liyab::EngineConfig;
using liyab::GenerationStats;

constexpr const char* kCode = R"(Rename the function `parse_record` to `parse_entry` everywhere in this module and output the complete module, nothing else.

```python
import csv
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Record:
    name: str
    amount: float
    date: datetime
    tags: list[str]


def parse_record(row: dict[str, str]) -> Record:
    """Builds a Record from one CSV row."""
    amount = float(row["amount"].replace(",", "."))
    date = datetime.strptime(row["date"], "%Y-%m-%d")
    tags = [t.strip() for t in row.get("tags", "").split(";") if t.strip()]
    return Record(name=row["name"].strip(), amount=amount, date=date, tags=tags)


def load(path: str) -> list[Record]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                records.append(parse_record(row))
            except (KeyError, ValueError) as error:
                print(f"skipping row {row}: {error}")
    return records


def total_by_tag(records: list[Record]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for record in records:
        for tag in record.tags:
            totals[tag] = totals.get(tag, 0.0) + record.amount
    return totals


def main() -> None:
    records = load("expenses.csv")
    for tag, total in sorted(total_by_tag(records).items()):
        print(f"{tag:>12}: {total:10.2f}")


if __name__ == "__main__":
    main()
```)";

constexpr const char* kSummary = R"(Summarize the following text in one paragraph of about 120 words.

The town of Valdera sits where two rivers meet at the foot of the hills, and for most of its history it lived from the water. Mills along the eastern river ground grain for the farms of the valley, and in the nineteenth century a paper factory was built next to the old bridge, using the clean water that came down from the springs above the town. At its peak the factory employed almost a third of the inhabitants, and the town grew around it: a school, a hospital, a railway station that connected Valdera to the coast in less than three hours.

The decline began in the nineteen-seventies. Larger factories abroad produced paper more cheaply, the railway line was reduced to two trains a day, and young people left for the cities. By the end of the century the population had fallen from eleven thousand to fewer than six thousand, and the factory closed its doors in 1998. For years the building stood empty, its windows broken, and many considered it the symbol of a town without a future.

Things changed slowly. In 2006 a group of former workers and local teachers founded an association to save the factory. They obtained a lease from the municipality, repaired the roof with donations and opened a small museum about the history of paper. A few years later, the regional government financed the restoration of the main hall, which now hosts concerts, markets and a school of restoration where students learn to repair old books and documents. The springs above the town, once used by the factory, now supply a small hydroelectric plant that covers most of the electricity needs of the public buildings.

Today Valdera has about six thousand five hundred inhabitants, the first increase in fifty years. Tourism is still modest, but the restoration school attracts students from several countries, and two small companies producing handmade paper have opened in the old warehouses. The mayor likes to say that the town did not find a new industry, but rediscovered the one it already had.)";

constexpr const char* kTool = R"(Tool `calendar.list_events` returned:
```json
{"date": "2026-10-14", "events": [
  {"time": "08:30", "title": "Gym session", "place": "City Fitness, Via Roma 12", "notes": "bring towel"},
  {"time": "10:00", "title": "Weekly team meeting", "place": "Office, room B2", "notes": "present Q3 numbers"},
  {"time": "12:45", "title": "Lunch with Marta", "place": "Trattoria Da Gino, Piazza Garibaldi 3", "notes": ""},
  {"time": "15:30", "title": "Dentist appointment", "place": "Studio Dentistico Bianchi, Corso Italia 48", "notes": "check-up"},
  {"time": "18:00", "title": "Pick up the kids", "place": "Scuola Primaria Manzoni", "notes": ""},
  {"time": "20:30", "title": "Call with parents", "place": "Home", "notes": "video call"}
]}
```
)";

constexpr const char* kToolAsk1 = "What do I have this afternoon?";
constexpr const char* kToolAsk2 =
    "List every event of today again, one per line, with its time, title, place and notes exactly as given.";
constexpr const char* kChat =
    "What are some good habits for learning a new language as an adult? Answer in a few paragraphs.";

std::string json_string(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out + "\"";
}

struct Message {
    std::string role, content;
};

struct Task {
    std::string name;
    std::vector<std::string> user_turns;  // a conversation: each turn sees the replies before it
};

struct Result {
    std::vector<std::string> replies;
    int32_t tokens = 0, passes = 0, proposed = 0, accepted = 0;
    double decode_ms = 0.0;
};

struct Mode {
    std::string name;
    bool lookup = false, suffix = false;
    int32_t draft_tokens = 0;  // 0: the default
    bool fixed = false;        // always the most drafts on offer (EngineConfig::adaptive_drafts off)
};

// Runs one conversation; returns the replies and the decode statistics of all its turns.
bool run(Engine& engine, const Task& task, int32_t max_tokens, Result& result) {
    std::vector<Message> messages;
    for (const std::string& turn : task.user_turns) {
        messages.push_back({"user", turn});
        std::string request = "{\"messages\":[";
        for (size_t i = 0; i < messages.size(); ++i) {
            request += (i ? "," : "") + std::string("{\"role\":") + json_string(messages[i].role) +
                       ",\"content\":" + json_string(messages[i].content) + "}";
        }
        request += "],\"add_generation_prompt\":true,\"enable_thinking\":false}";
        auto prompt = engine.render_chat(request);
        if (!prompt) {
            std::fprintf(stderr, "chat template: %s\n", prompt.status().to_string().c_str());
            return false;
        }
        liyab::SamplingParams params;
        params.temperature = 0.0f;
        params.max_tokens = max_tokens;
        std::string reply;
        auto stats = engine.generate(*prompt, params, [&](std::string_view piece, int32_t) {
            reply += piece;
            return true;
        });
        if (!stats) {
            std::fprintf(stderr, "generate: %s\n", stats.status().to_string().c_str());
            return false;
        }
        result.tokens += stats->generated_tokens;
        result.passes += stats->decode_passes;
        result.proposed += stats->draft_tokens_proposed;
        result.accepted += stats->draft_tokens_accepted;
        result.decode_ms += stats->decode_ms;
        result.replies.push_back(reply);
        messages.push_back({"assistant", reply});
    }
    return true;
}

void usage() {
    std::fprintf(stderr,
                 "usage: liyab-spec-bench -m MODEL.gguf [--modes plain,lookup,suffix,suffix:8] [-n 512]\n"
                 "                        [--tasks code,summary,tool,chat,again] [--repeat R] [--threads N]\n"
                 "                        [--backend cpu|metal|auto]\n");
}

}  // namespace

int main(int argc, char** argv) {
    EngineConfig base;
    base.backend = liyab::BackendKind::Cpu;
    base.power.profile = liyab::PowerProfile::Performance;  // unpaced: measure the engine, not the duty cycle
    base.thermal_polling = false;
    std::string modes_arg = "plain,lookup,suffix";
    std::string tasks_arg;  // empty: all
    int32_t max_tokens = 512;
    int32_t repeats = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "-m") base.model_path = next();
        else if (arg == "--modes") modes_arg = next();
        else if (arg == "--tasks") tasks_arg = next();
        else if (arg == "-n") max_tokens = std::atoi(next().c_str());
        else if (arg == "--repeat") repeats = std::max(1, std::atoi(next().c_str()));
        else if (arg == "--threads") base.n_threads = std::atoi(next().c_str());
        else if (arg == "--backend") {
            const std::string b = next();
            if (b == "auto") base.backend.reset();
            else if (b == "metal") base.backend = liyab::BackendKind::Metal;
            else base.backend = liyab::BackendKind::Cpu;
        } else {
            usage();
            return 2;
        }
    }
    if (base.model_path.empty()) {
        usage();
        return 2;
    }
    std::vector<Mode> modes;
    for (size_t start = 0; start <= modes_arg.size();) {
        size_t end = modes_arg.find(',', start);
        if (end == std::string::npos) end = modes_arg.size();
        Mode m;
        m.name = modes_arg.substr(start, end - start);
        const size_t colon = m.name.find(':');
        const std::string kind = m.name.substr(0, colon);
        if (colon != std::string::npos) m.draft_tokens = std::atoi(m.name.c_str() + colon + 1);
        m.lookup = kind == "lookup";
        m.suffix = kind == "suffix";
        m.fixed = m.name.ends_with(":fixed");
        if (!m.lookup && !m.suffix && kind != "plain") {
            usage();
            return 2;
        }
        modes.push_back(m);
        start = end + 1;
    }

    std::vector<Task> tasks = {
        {"code", {kCode}},
        {"summary", {kSummary}},
        {"tool", {std::string(kTool) + kToolAsk1, std::string(kTool) + kToolAsk2}},
        {"chat", {kChat}},
        {"again", {std::string(kTool) + kToolAsk1, std::string(kTool) + kToolAsk2}},
    };
    if (!tasks_arg.empty()) {
        std::erase_if(tasks, [&](const Task& t) { return ("," + tasks_arg + ",").find("," + t.name + ",") == std::string::npos; });
    }
    // results[rep][mode][task]. Every mode has its engine for the whole
    // repetition (each keeps what its conversations left), and each task runs
    // on all modes back to back, so the modes see the same load on the host.
    std::vector<std::vector<std::vector<Result>>> results(
        static_cast<size_t>(repeats), std::vector<std::vector<Result>>(modes.size(), std::vector<Result>(tasks.size())));
    for (int32_t rep = 0; rep < repeats; ++rep) {
        std::vector<std::unique_ptr<Engine>> engines;
        for (const Mode& mode : modes) {
            EngineConfig config = base;
            config.lookup_drafts = mode.lookup;
            config.suffix_drafts = mode.suffix;
            if (mode.draft_tokens > 0) config.draft_tokens = mode.draft_tokens;
            config.adaptive_drafts = !mode.fixed;
            auto engine = Engine::create(config);
            if (!engine) {
                std::fprintf(stderr, "%s\n", engine.status().to_string().c_str());
                return 1;
            }
            engines.push_back(std::move(engine).value());
        }
        for (size_t t = 0; t < tasks.size(); ++t) {
            for (size_t m = 0; m < modes.size(); ++m) {
                Result& r = results[static_cast<size_t>(rep)][m][t];
                if (!run(*engines[m], tasks[t], max_tokens, r)) return 1;
                std::fprintf(stderr, "run %d, %s / %s: %d tokens, %.1f tok/s\n", rep + 1, tasks[t].name.c_str(),
                             modes[m].name.c_str(), r.tokens, r.tokens * 1000.0 / r.decode_ms);
            }
        }
    }

    // Medians over the repetitions; the speed-up is taken within each one.
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    };
    std::printf("%-8s %-10s %7s %8s %8s %9s %9s %8s %10s %5s\n", "task", "mode", "tokens", "tok/s", "best", "speedup",
                "best-x", "accept", "tok/pass", "same");
    auto best_rate = [&](size_t m, size_t t) {
        double best = 0.0;
        for (const auto& run_results : results) {
            best = std::max(best, run_results[m][t].tokens * 1000.0 / run_results[m][t].decode_ms);
        }
        return best;
    };
    for (size_t t = 0; t < tasks.size(); ++t) {
        for (size_t m = 0; m < modes.size(); ++m) {
            std::vector<double> rates, speedups;
            int32_t tokens = 0, passes = 0, proposed = 0, accepted = 0;
            bool same = true;
            for (const auto& run_results : results) {
                const Result& r = run_results[m][t];
                const Result& plain = run_results[0][t];
                rates.push_back(r.tokens * 1000.0 / r.decode_ms);
                speedups.push_back(rates.back() / (plain.tokens * 1000.0 / plain.decode_ms));
                tokens += r.tokens;
                passes += r.passes;
                proposed += r.proposed;
                accepted += r.accepted;
                same = same && r.replies == results[0][0][t].replies;
            }
            std::printf("%-8s %-10s %7d %8.1f %8.1f %8.2fx %8.2fx %7.0f%% %10.2f %5s\n", tasks[t].name.c_str(),
                        modes[m].name.c_str(), tokens / repeats, median(rates), best_rate(m, t), median(speedups),
                        best_rate(m, t) / best_rate(0, t),
                        proposed > 0 ? 100.0 * accepted / proposed : 0.0,
                        passes > 0 ? static_cast<double>(tokens) / passes : 0.0, same ? "yes" : "NO");
        }
    }
    return 0;
}
