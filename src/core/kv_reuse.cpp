#include "core/kv_reuse.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>

#include "core/log.h"
#include "core/quant.h"
#include "core/thread_pool.h"

namespace liyab {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Squared size of a row and of its difference from another, dequantized.
void row_change(DType type, const uint8_t* now, const uint8_t* before, int32_t n, std::vector<float>& a,
                std::vector<float>& b, double& size, double& change) {
    a.resize(static_cast<size_t>(n));
    b.resize(static_cast<size_t>(n));
    quant::dequantize_row(type, now, a.data(), n);
    quant::dequantize_row(type, before, b.data(), n);
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        change += d * d;
        size += static_cast<double>(a[i]) * a[i];
    }
}

}  // namespace

Status check_segments(std::span<const PromptSegment> segments, size_t size, std::string_view text) {
    auto mid_character = [&](size_t at) {
        return at < text.size() && (static_cast<unsigned char>(text[at]) & 0xC0) == 0x80;
    };
    size_t last_end = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        const PromptSegment& s = segments[i];
        if (s.begin >= s.end || s.end > size || (i > 0 && s.begin < last_end) || mid_character(s.begin) ||
            mid_character(s.end)) {
            return Status(ErrorCode::InvalidArgument,
                          "prompt segment " + std::to_string(i) +
                              ": segments must be non-empty, ascending, apart, inside the prompt and on whole "
                              "characters");
        }
        last_end = s.end;
    }
    return Status::ok();
}

Status encode_segmented(const Tokenizer& tokenizer, std::string_view text, size_t from,
                        std::span<const PromptSegment> segments, bool bos, std::vector<int32_t>& tokens,
                        std::vector<TokenSegment>& out) {
    LIYAB_RETURN_IF_ERROR(check_segments(segments, text.size(), text));
    auto piece = [&](std::string_view part) -> Status {
        if (part.empty()) return Status::ok();
        auto encoded = tokens.empty() ? tokenizer.encode(part, bos) : tokenizer.encode_continuation(part, tokens.back());
        if (!encoded) return encoded.status();
        tokens.insert(tokens.end(), encoded->begin(), encoded->end());
        return Status::ok();
    };
    size_t at = from;
    for (const PromptSegment& s : segments) {
        if (s.begin < at) continue;  // in the known text, or straddling it: plain text from `at` on
        LIYAB_RETURN_IF_ERROR(piece(text.substr(at, s.begin - at)));
        if (tokens.empty() && bos && tokenizer.bos() >= 0 && !text.starts_with(tokenizer.token_text(tokenizer.bos()))) {
            tokens.push_back(tokenizer.bos());  // BOS before the segment, not in it
        }
        const auto begin = static_cast<int32_t>(tokens.size());
        if (tokens.empty()) {
            auto encoded = tokenizer.encode(text.substr(s.begin, s.end - s.begin), false);
            if (!encoded) return encoded.status();
            tokens = std::move(encoded).value();
        } else {
            LIYAB_RETURN_IF_ERROR(piece(text.substr(s.begin, s.end - s.begin)));
        }
        if (static_cast<int32_t>(tokens.size()) > begin) out.push_back({begin, static_cast<int32_t>(tokens.size())});
        at = s.end;
    }
    return piece(text.substr(at));
}

Result<ReuseResult> reuse_segments(Transformer& model, KvStore& store, std::span<const int32_t> tokens,
                                   std::span<const TokenSegment> segments, int32_t exact, int32_t limit,
                                   const ReuseSettings& settings, const Route& route, ThreadPool& pool,
                                   int32_t chunk, const std::atomic<bool>* stop) {
    LIYAB_RETURN_IF_ERROR(model.reuse_support());
    ReuseResult result;
    result.end = exact;
    limit = std::min<int32_t>(limit, static_cast<int32_t>(tokens.size()));
    if (model.n_past() != exact || exact < 0 || chunk < 1) {
        return Status(ErrorCode::InvalidArgument, "reuse_segments: the context must hold the exact prefix");
    }
    const auto t_rows = Clock::now();
    // The copy each segment reaching past the prefix reuses: one computed at
    // the same position needs no turning, else the most recently used.
    struct Use {
        TokenSegment segment;
        int32_t lo = 0, hi = 0;  // the positions it fills
        KvStore::SegmentCopy copy;
    };
    std::vector<Use> uses;
    int32_t last_end = 0;
    for (const TokenSegment& s : segments) {
        if (s.begin < last_end || s.begin >= s.end || s.end > static_cast<int32_t>(tokens.size())) {
            return Status(ErrorCode::InvalidArgument, "reuse_segments: segments must ascend inside the prompt");
        }
        last_end = s.end;
        const int32_t lo = std::max(s.begin, exact);
        const int32_t hi = std::min(s.end, limit);
        if (s.end - s.begin < settings.min_tokens || hi - lo < 1) continue;
        std::vector<KvStore::SegmentCopy> copies = store.segments(tokens.subspan(static_cast<size_t>(s.begin),
                                                                                   static_cast<size_t>(s.end - s.begin)));
        if (copies.empty()) continue;
        auto same = std::find_if(copies.begin(), copies.end(),
                                 [&](const KvStore::SegmentCopy& c) { return c.base == s.begin; });
        uses.push_back({s, lo, hi, std::move(same != copies.end() ? *same : copies.front())});
    }
    if (uses.empty()) return result;

    // Their rows, turned to where they now are. A copy whose file turns out
    // damaged is dropped (its positions are then processed like the others).
    KvCache& kv = model.mutable_kv_cache();
    const int32_t c = store.chunk_tokens();
    int32_t end = exact;
    for (const Use& u : uses) end = std::max(end, u.hi);
    auto fail = [&](Status s) {
        (void)model.truncate(std::min(model.n_past(), exact));
        return s;
    };
    if (Status s = kv.reserve(end); !s.is_ok()) return fail(s);
    std::vector<uint8_t> image;
    std::vector<uint8_t> filled;  // per position of [exact, end): its rows came from a copy
    filled.assign(static_cast<size_t>(end - exact), 0);
    for (Use& u : uses) {
        // Chain position p of the copy holds segment token p - 1 (position 0
        // is the marker), computed at base + p - 1, now at begin + p - 1.
        const int32_t first = 1 + (u.lo - u.segment.begin), last = 1 + (u.hi - u.segment.begin);
        bool ok = true;
        for (size_t i = 0; i < u.copy.chunks.size() && ok; ++i) {
            const int32_t start = static_cast<int32_t>(i) * c;
            const int32_t n = u.copy.chunk_sizes[i];
            const int32_t s0 = std::max(first, start) - start, s1 = std::min(last, start + n) - start;
            if (s0 >= s1) continue;
            image.resize(kv.rows_bytes(n));
            // A damaged file is removed with every chunk after it, which may
            // take a chunk of another copy listed here (copies that start
            // alike share chunks): either way that copy is dropped.
            Status s = store.read_chunk(u.copy.chunks[i], image);
            if (!s.is_ok()) {
                ok = false;
                break;
            }
            const int32_t from = u.copy.base + start + s0 - 1, to = u.segment.begin + start + s0 - 1;
            if (s = model.move_keys(image.data(), n, s0, s1 - s0, from, to, &pool); !s.is_ok()) return fail(s);
            if (s = kv.write_rows_at(to, image.data(), n, s0, s1 - s0); !s.is_ok()) return fail(s);
        }
        if (!ok) {
            LIYAB_LOG_WARN("segment reuse: a stored segment could not be read; its tokens are processed");
            u.hi = u.lo;  // nothing of it is used
            continue;
        }
        for (int32_t p = u.lo; p < u.hi; ++p) filled[static_cast<size_t>(p - exact)] = 1;
        KvStore::Plan used;
        used.chunks = u.copy.chunks;
        store.used(used);
        ++result.segments;
    }
    std::erase_if(uses, [](const Use& u) { return u.hi <= u.lo; });
    if (uses.empty()) return result;
    end = exact;
    for (const Use& u : uses) end = std::max(end, u.hi);
    // The question-guided criteria need the tail's queries: it runs with the span.
    const bool guided = settings.criterion == ReuseCriterion::Attention || settings.criterion == ReuseCriterion::Impact;
    const int32_t tail_begin = end;
    if (guided && limit > end) {
        if (Status s = kv.reserve(limit); !s.is_ok()) return fail(s);
        end = limit;
    }
    filled.resize(static_cast<size_t>(end - exact));
    if (Status s = model.adopt_context(end, {}, {}, false); !s.is_ok()) return fail(s);
    result.rows_ms = ms_since(t_rows);

    // The span [exact, end) block by block: every position not reused, and
    // the reused ones still selected.
    const auto t_blend = Clock::now();
    const ModelConfig& cfg = model.config();
    const auto d = static_cast<size_t>(cfg.n_embd);
    std::vector<int32_t> active;
    std::vector<uint8_t> reused;  // per active position
    for (int32_t p = exact; p < end; ++p) {
        active.push_back(p);
        reused.push_back(filled[static_cast<size_t>(p - exact)]);
        result.reused += filled[static_cast<size_t>(p - exact)];
    }
    result.computed = static_cast<int32_t>(active.size()) - result.reused;
    // Reused positions kept at a selection: `fraction` of them in the end,
    // twice as many at each refining block before (Deviation).
    auto kept_at = [&](int32_t step) {
        const double share = std::min(1.0, static_cast<double>(settings.fraction) * std::ldexp(1.0, step));
        return static_cast<int32_t>(std::lround(share * static_cast<double>(result.reused)));
    };
    int32_t keep = kept_at(0);
    // Positions recomputed whatever the criterion: the first tokens of each segment.
    std::vector<uint8_t> forced(active.size(), 0);
    for (const Use& u : uses) {
        for (int32_t p = u.lo; p < std::min(u.hi, u.segment.begin + settings.first_tokens); ++p) {
            forced[static_cast<size_t>(p - exact)] = 1;
        }
    }
    // Keeps the active positions `chosen` says (and every one not reused).
    std::vector<float> x(active.size() * d);
    auto compact = [&](const std::vector<uint8_t>& chosen) {
        size_t to = 0;
        for (size_t i = 0; i < active.size(); ++i) {
            if (reused[i] && !chosen[i]) continue;
            if (to != i) {
                active[to] = active[i];
                reused[to] = reused[i];
                forced[to] = forced[i];
                std::copy_n(x.data() + i * d, d, x.data() + to * d);
            }
            ++to;
        }
        active.resize(to);
        reused.resize(to);
        forced.resize(to);
        x.resize(to * d);
    };
    // The `keep` reused positions with the highest `score` (forced ones first).
    auto choose = [&](const std::vector<double>& score) {
        std::vector<size_t> order;
        for (size_t i = 0; i < active.size(); ++i) {
            if (reused[i]) order.push_back(i);
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (forced[a] != forced[b]) return forced[a] > forced[b];
            return score[a] > score[b];
        });
        std::vector<uint8_t> chosen(active.size(), 0);
        size_t forced_count = 0;
        for (const size_t i : order) forced_count += forced[i];
        const size_t take = std::max<size_t>(forced_count, static_cast<size_t>(std::max(keep, 0)));
        for (size_t k = 0; k < std::min(take, order.size()); ++k) chosen[order[k]] = 1;
        return chosen;
    };
    const int32_t select_block = std::clamp(settings.select_block, 0, cfg.n_layers - 1);
    const bool measured = settings.criterion == ReuseCriterion::Deviation || guided;
    if (!measured) {
        // Chosen before block 0: the others never run.
        std::vector<double> score(active.size(), 0.0);
        if (settings.criterion == ReuseCriterion::First) {
            for (size_t i = 0; i < active.size(); ++i) {
                const auto u = std::find_if(uses.begin(), uses.end(), [&](const Use& use) {
                    return active[i] >= use.lo && active[i] < use.hi;
                });
                if (u != uses.end()) {
                    score[i] = 1.0 - static_cast<double>(active[i] - u->segment.begin) /
                                         static_cast<double>(u->segment.end - u->segment.begin);
                }
            }
        } else {
            std::mt19937_64 rng(settings.seed);
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            for (double& s : score) s = uniform(rng);
        }
        x.resize(active.size() * d);
        compact(choose(score));
    }
    {
        std::vector<int32_t> ids(active.size());
        for (size_t i = 0; i < active.size(); ++i) ids[i] = tokens[static_cast<size_t>(active[i])];
        x.resize(active.size() * d);
        if (Status s = model.embed(ids, x); !s.is_ok()) return fail(s);
    }
    const KvCache& cache = kv;
    const size_t row = cache.head_row_bytes();
    const int32_t heads = cache.config().n_head_kv;
    const bool values = !cache.config().values_in_keys;
    std::vector<uint8_t> before;  // reused rows at the selection block, per reused active position
    std::vector<float> a, b;
    for (int32_t layer = 0; layer < cfg.n_layers; ++layer) {
        if (stop != nullptr && stop->load(std::memory_order_relaxed)) {
            return fail(Status(ErrorCode::Cancelled, "segment reuse cancelled"));
        }
        const int32_t slot = model.kv_slot_of(layer);
        const int32_t refine = std::clamp(settings.refine, 0, cfg.n_layers - 1 - select_block);
        const bool select = measured && layer >= select_block && layer <= select_block + refine && slot >= 0;
        if (select) keep = kept_at(select_block + refine - layer);
        if (select) {
            before.assign(active.size() * static_cast<size_t>(heads) * row * 2, 0);
            for (size_t i = 0; i < active.size(); ++i) {
                if (!reused[i]) continue;
                for (int32_t g = 0; g < heads; ++g) {
                    uint8_t* at = before.data() + (i * static_cast<size_t>(heads) + static_cast<size_t>(g)) * row * 2;
                    std::memcpy(at, cache.k_row(slot, active[i], g), row);
                    if (values) std::memcpy(at + row, cache.v_row(slot, active[i], g), row);
                }
            }
        }
        if (Status s = model.run_block_at(layer, active, x, chunk, route, pool); !s.is_ok()) return fail(s);
        if (!select) continue;
        // How far each reused position's rows moved at this block.
        std::vector<double> score(active.size(), 0.0);
        for (size_t i = 0; i < active.size(); ++i) {
            if (!reused[i]) continue;
            double size = 0.0, change = 0.0, v_size = 0.0, v_change = 0.0;
            for (int32_t g = 0; g < heads; ++g) {
                const uint8_t* at = before.data() + (i * static_cast<size_t>(heads) + static_cast<size_t>(g)) * row * 2;
                row_change(cache.dtype(), cache.k_row(slot, active[i], g), at, cache.config().head_dim, a, b, size,
                           change);
                if (values) {
                    row_change(cache.dtype(), cache.v_row(slot, active[i], g), at + row, cache.config().head_dim, a,
                               b, v_size, v_change);
                }
            }
            score[i] = settings.relative ? change / std::max(size, 1e-30) + (values ? v_change / std::max(v_size, 1e-30) : 0.0)
                                         : change + v_change;
        }
        if (guided) {
            // The attention the tail's last tokens (the pass's last rows: their
            // queries at this block) pay each position, summed over their heads.
            const std::span<const float> q = model.last_queries();
            const int32_t hd = cache.config().head_dim;
            const auto q_row = static_cast<size_t>(cfg.n_head) * static_cast<size_t>(hd);
            const auto m = static_cast<int32_t>(q.size() / q_row);
            const int32_t group = cfg.n_head / heads;
            const int32_t window = cfg.blocks[static_cast<size_t>(layer)].windowed ? cfg.sliding_window : 0;
            std::vector<double> mass(static_cast<size_t>(end), 0.0);
            std::vector<float> key(static_cast<size_t>(hd)), logits;
            // Keys dequantized once per position and KV head.
            std::vector<float> keys(static_cast<size_t>(end) * static_cast<size_t>(heads) * static_cast<size_t>(hd));
            for (int32_t p = 0; p < end; ++p) {
                for (int32_t g = 0; g < heads; ++g) {
                    quant::dequantize_row(cache.dtype(), cache.k_row(slot, p, g),
                                          keys.data() + (static_cast<size_t>(p) * static_cast<size_t>(heads) + static_cast<size_t>(g)) * static_cast<size_t>(hd), hd);
                }
            }
            for (int32_t r = 0; r < m; ++r) {
                const size_t i = active.size() - static_cast<size_t>(m) + static_cast<size_t>(r);
                const int32_t pos = active[i];
                if (pos < tail_begin) continue;
                const int32_t first = window > 0 ? std::max(0, pos - window + 1) : 0;
                logits.resize(static_cast<size_t>(pos - first + 1));
                for (int32_t h = 0; h < cfg.n_head; ++h) {
                    const float* qh = q.data() + static_cast<size_t>(r) * q_row + static_cast<size_t>(h) * static_cast<size_t>(hd);
                    const int32_t g = h / group;
                    float top = -INFINITY;
                    for (int32_t p = first; p <= pos; ++p) {
                        const float* k = keys.data() + (static_cast<size_t>(p) * static_cast<size_t>(heads) + static_cast<size_t>(g)) * static_cast<size_t>(hd);
                        float dot = 0.0f;
                        for (int32_t c = 0; c < hd; ++c) dot += qh[c] * k[c];
                        dot *= cfg.attn_scale;
                        if (cfg.attn_softcap > 0.0f) dot = cfg.attn_softcap * std::tanh(dot / cfg.attn_softcap);
                        logits[static_cast<size_t>(p - first)] = dot;
                        top = std::max(top, dot);
                    }
                    double sum = 0.0;
                    for (float& v : logits) sum += v = std::exp(v - top);
                    for (int32_t p = first; p <= pos; ++p) mass[static_cast<size_t>(p)] += logits[static_cast<size_t>(p - first)] / sum;
                }
            }
            for (size_t i = 0; i < active.size(); ++i) {
                if (!reused[i]) continue;
                const double a = mass[static_cast<size_t>(active[i])];
                score[i] = settings.criterion == ReuseCriterion::Impact ? a * score[i] : a;
            }
        }
        const std::vector<uint8_t> chosen = choose(score);
        double total = 0.0, kept = 0.0;
        for (size_t i = 0; i < active.size(); ++i) {
            total += score[i];
            kept += chosen[i] ? score[i] : 0.0;
        }
        result.kept_deviation = total > 0.0 ? kept / total : 0.0;
        compact(chosen);
    }
    for (size_t i = 0; i < active.size(); ++i) result.recomputed += reused[i];
    result.end = end;
    for (const Use& u : uses) {
        if (result.first_reused < 0 || u.lo < result.first_reused) result.first_reused = u.lo;
    }
    result.blend_ms = ms_since(t_blend);
    return result;
}

namespace {

// The calibration's text: an assistant's documents about a regional railway.
// Plain text (no chat template), so every model is measured the same way.
constexpr std::string_view kCalibrationSystem =
    "You answer travellers' questions for the Valley Line railway. Use only the documents below and quote times "
    "and prices exactly.";
constexpr std::string_view kCalibrationDocs[] = {
    "Timetable, weekdays. Trains leave Northgate for Ashford at 06:10, 06:40 and then every half hour until 21:40, "
    "with a last train at 23:05. The journey takes 52 minutes and calls at Millbrook, Fernhill and Old Quarry. On "
    "Saturdays the first train leaves at 07:10; on Sundays trains run every hour from 08:10 to 22:10. Engineering "
    "works close the line between Fernhill and Ashford on the second Sunday of each month, when buses replace "
    "trains and the journey takes about 75 minutes.",
    "Fares and passes. A single ticket from Northgate to Ashford costs 9.40, a return 15.80 and a day return after "
    "09:30 only 11.20. Children under 5 travel free; children from 5 to 15 and holders of the senior card pay half. "
    "A monthly pass costs 182 and can be shared within one household on weekends. Tickets bought on board cost 2.00 "
    "more, unless the ticket machine at the departure station was out of order.",
    "Bicycles and luggage. Each train carries up to six bicycles in the marked area of the middle carriage; they "
    "travel free outside the peak hours of 07:00 to 09:00 and 16:30 to 18:30, when a bicycle ticket costs 3.00. "
    "Folded bicycles count as luggage. Pushchairs and wheelchairs always have priority in the multi-purpose area. "
    "Luggage over 25 kg, or longer than two metres, must be booked two days ahead.",
    "Delays and refunds. If a train arrives at the passenger's destination 20 minutes or more behind schedule, 25% "
    "of the ticket price is refunded; from 60 minutes, 50%. Pass holders receive a credit of one day per three "
    "delays in a month. Claims are made online within 28 days, with the ticket number and the train's departure "
    "time. Delays caused by severe weather warnings are refunded at the same rates.",
};
constexpr std::string_view kCalibrationQuestion =
    "I bought a day return after ten and my train back reached Northgate 70 minutes late. What do I get back, and "
    "how do I claim it?";

// One calibration prompt: the documents in `order`, and their byte ranges.
std::string calibration_prompt(std::span<const int> order, std::vector<PromptSegment>& documents) {
    std::string text(kCalibrationSystem);
    documents.clear();
    for (const int d : order) {
        text += "\n\n";
        documents.push_back({text.size(), text.size() + kCalibrationDocs[d].size()});
        text += kCalibrationDocs[d];
    }
    text += "\n\nQuestion: ";
    text += kCalibrationQuestion;
    text += "\nAnswer:";
    return text;
}

std::vector<double> log_probs(std::span<const float> logits) {
    const float mx = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (const float v : logits) sum += std::exp(static_cast<double>(v - mx));
    const double lse = std::log(sum) + mx;
    std::vector<double> out(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) out[i] = logits[i] - lse;
    return out;
}

}  // namespace

Result<ReuseCalibration> calibrate_reuse(Transformer& model, const Tokenizer& tokenizer, const ReuseSettings& settings,
                                         double max_kl, const Route& route, ThreadPool& pool, int32_t chunk) {
    LIYAB_RETURN_IF_ERROR(model.reuse_support());
    const auto t0 = Clock::now();
    ReuseCalibration out;
    KvStore::Config sc;
    const uint64_t parts[] = {model.context_fingerprint(), kReuseCalibrationVersion};
    sc.seed = kv_seed_key(parts);
    sc.chunk_tokens = model.kv_cache().config().page_tokens;
    sc.chunk_bytes = model.kv_cache().page_bytes();
    sc.ram_bytes = model.kv_cache().rows_bytes(4096);  // room for every document's copy several times over
    auto store = KvStore::open(sc);
    if (!store) return Status(ErrorCode::Internal, "calibration: no store");
    const bool bos = tokenizer.add_bos_default();
    auto prefill = [&](std::span<const int32_t> tokens) -> Status {
        for (size_t i = 0; i < tokens.size(); i += static_cast<size_t>(chunk)) {
            const auto part = tokens.subspan(i, std::min(static_cast<size_t>(chunk), tokens.size() - i));
            auto r = model.forward(part, Transformer::Logits::None, route, pool);
            if (!r) return r.status();
        }
        return Status::ok();
    };
    auto step = [&](int32_t token) -> Result<std::vector<double>> {
        auto r = model.forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, route, pool);
        if (!r) return r.status();
        return log_probs(*r);
    };
    auto encode = [&](std::span<const int> order, std::vector<int32_t>& tokens, std::vector<TokenSegment>& segments) {
        std::vector<PromptSegment> documents;
        const std::string text = calibration_prompt(order, documents);
        tokens.clear();
        segments.clear();
        return encode_segmented(tokenizer, text, 0, documents, bos, tokens, segments);
    };
    struct Case {
        std::vector<int32_t> tokens;
        std::vector<TokenSegment> segments;
        std::vector<std::vector<double>> reference;  // the full prefill's distributions
        std::vector<int32_t> answer;                 // its greedy answer, fed back
    };
    auto finish = [&](Status s) -> Result<ReuseCalibration> {
        model.reset();
        if (!s.is_ok()) return s;
        out.ms = ms_since(t0);
        return out;
    };
    // The documents stored from one order...
    {
        const int order[] = {0, 1, 2, 3};
        std::vector<int32_t> tokens;
        std::vector<TokenSegment> segments;
        if (Status s = encode(order, tokens, segments); !s.is_ok()) return finish(s);
        model.reset();
        if (Status s = prefill(tokens); !s.is_ok()) return finish(s);
        keep_segments(model, *store, tokens, segments, settings.min_tokens);
    }
    // ...answered in two others, first from a full prefill.
    std::vector<Case> cases;
    for (const auto& order : {std::array<int, 4>{2, 0, 3, 1}, std::array<int, 4>{1, 3, 0, 2}}) {
        Case c;
        if (Status s = encode(order, c.tokens, c.segments); !s.is_ok() || c.segments.empty()) {
            return finish(s.is_ok() ? Status(ErrorCode::Internal, "calibration: no documents") : s);
        }
        model.reset();
        if (Status s = prefill(std::span(c.tokens).first(c.tokens.size() - 1)); !s.is_ok()) return finish(s);
        auto dist = step(c.tokens.back());
        if (!dist) return finish(dist.status());
        // Up to the answer's end of generation, included: where it ends is part
        // of what reuse may change; what a model says past it is not an answer.
        for (int32_t a = 0; a <= kCalibrationAnswer; ++a) {
            const auto best = static_cast<int32_t>(std::max_element(dist->begin(), dist->end()) - dist->begin());
            c.reference.push_back(std::move(dist).value());
            if (a == kCalibrationAnswer || tokenizer.is_end_of_generation(best)) break;
            c.answer.push_back(best);
            dist = step(best);
            if (!dist) return finish(dist.status());
        }
        cases.push_back(std::move(c));
    }
    // The smallest fraction whose answers stay within max_kl.
    const auto vocab = static_cast<size_t>(model.config().n_vocab);
    for (const float fraction : kReuseFractions) {
        ReuseSettings s = settings;
        s.fraction = fraction;
        double kl_sum = 0.0;
        int64_t agree = 0, positions = 0;
        for (const Case& c : cases) {
            const int32_t exact = c.segments.front().begin;
            const auto n = static_cast<int32_t>(c.tokens.size());
            model.reset();
            if (Status st = prefill(std::span(c.tokens).first(static_cast<size_t>(exact))); !st.is_ok()) return finish(st);
            auto reused = reuse_segments(model, *store, c.tokens, c.segments, exact, n - 1, s, route, pool, chunk);
            if (!reused) return finish(reused.status());
            if (Status st = prefill(std::span(c.tokens).subspan(static_cast<size_t>(reused->end),
                                                                 static_cast<size_t>(n - 1 - reused->end)));
                !st.is_ok()) {
                return finish(st);
            }
            auto first = step(c.tokens.back());
            if (!first) return finish(first.status());
            std::vector<std::vector<double>> mine = {std::move(first).value()};
            if (!c.answer.empty()) {
                auto rows = model.forward(c.answer, Transformer::Logits::All, route, pool);
                if (!rows) return finish(rows.status());
                for (size_t a = 0; a < c.answer.size(); ++a) {
                    mine.push_back(log_probs(std::span<const float>(rows->data() + a * vocab, vocab)));
                }
            }
            for (size_t p = 0; p < mine.size(); ++p) {
                const std::vector<double>& ref = c.reference[p];
                double k = 0.0;
                for (size_t i = 0; i < vocab; ++i) k += std::exp(ref[i]) * (ref[i] - mine[p][i]);
                kl_sum += std::max(k, 0.0);
                agree += std::max_element(ref.begin(), ref.end()) - ref.begin() ==
                         std::max_element(mine[p].begin(), mine[p].end()) - mine[p].begin();
                ++positions;
            }
        }
        out.fraction = fraction;
        out.kl = kl_sum / static_cast<double>(positions);
        out.top1 = static_cast<double>(agree) / static_cast<double>(positions);
        out.kls.push_back(out.kl);
        if (out.kl <= max_kl) {
            out.met = true;
            break;
        }
    }
    return finish(Status::ok());
}

int32_t keep_segments(const Transformer& model, KvStore& store, std::span<const int32_t> tokens,
                      std::span<const TokenSegment> segments, int32_t min_tokens) {
    const KvCache& kv = model.kv_cache();
    const int32_t c = store.chunk_tokens();
    const int32_t held = std::min<int32_t>(model.n_past(), static_cast<int32_t>(tokens.size()));
    int32_t kept = 0;
    for (const TokenSegment& s : segments) {
        if (s.begin < 0 || s.end > held || s.end - s.begin < std::max(min_tokens, 1)) continue;
        const auto span = tokens.subspan(static_cast<size_t>(s.begin), static_cast<size_t>(s.end - s.begin));
        if (!store.segments(span).empty()) continue;
        const std::vector<int32_t> stored = kv_segment_tokens(span, s.begin);
        const int32_t n = s.end - s.begin;
        KvStore::ContextSource source;
        source.tokens = stored;
        // Chain position p holds segment token p - 1: the context's position
        // begin + p - 1. The two markers get zero rows (never read).
        source.rows = [&](int32_t chunk, int32_t count, uint8_t* out) {
            std::memset(out, 0, kv.rows_bytes(count));
            const int32_t start = chunk * c;
            const int32_t p0 = std::max(start, 1), p1 = std::min(start + count, n + 1);
            if (p0 >= p1) return Status::ok();
            return kv.read_rows_at(s.begin + p0 - 1, p1 - p0, out, count, p0 - start);
        };
        if (store.put(source).whole) ++kept;
    }
    return kept;
}

}  // namespace liyab
