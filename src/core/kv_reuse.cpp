#include "core/kv_reuse.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <numeric>
#include <random>

#include "core/log.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "liyab/experimental/forward_hooks.h"

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
    const int32_t page_tokens = kv.config().page_tokens;  // a chunk is whole pages, its image theirs in order
    int32_t end = exact;
    for (const Use& u : uses) end = std::max(end, u.hi);
    auto fail = [&](Status s) {
        (void)model.truncate(std::min(model.n_past(), exact));
        return s;
    };
    if (Status s = kv.reserve(end); !s.is_ok()) return fail(s);
    std::vector<std::vector<uint8_t>> images;  // one per page of a chunk
    std::vector<std::span<uint8_t>> parts;
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
            if (std::max(first, start) >= std::min(last, start + n)) continue;
            parts.clear();
            for (int32_t k = 0; k * page_tokens < n; ++k) {
                if (images.size() <= static_cast<size_t>(k)) images.emplace_back();
                images[static_cast<size_t>(k)].resize(kv.rows_bytes(std::min(page_tokens, n - k * page_tokens)));
                parts.emplace_back(images[static_cast<size_t>(k)]);
            }
            // A damaged file is removed with every chunk after it, which may
            // take a chunk of another copy listed here (copies that start
            // alike share chunks): either way that copy is dropped.
            Status s = store.read_chunk(u.copy.chunks[i], parts);
            if (!s.is_ok()) {
                ok = false;
                break;
            }
            // Each page's image: the slots it holds of the positions used.
            for (int32_t k = 0; k * page_tokens < n; ++k) {
                const int32_t at = start + k * page_tokens, rows = std::min(page_tokens, n - k * page_tokens);
                const int32_t s0 = std::max(first, at) - at, s1 = std::min(last, at + rows) - at;
                if (s0 >= s1) continue;
                uint8_t* image = images[static_cast<size_t>(k)].data();
                const int32_t from = u.copy.base + at + s0 - 1, to = u.segment.begin + at + s0 - 1;
                if (s = model.move_keys(image, rows, s0, s1 - s0, from, to, &pool); !s.is_ok()) return fail(s);
                if (s = kv.write_rows_at(to, image, rows, s0, s1 - s0); !s.is_ok()) return fail(s);
            }
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
    if (Status s = model.adopt_context(end, {}, false); !s.is_ok()) return fail(s);
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
            std::vector<float> logits;
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
                const double attention = mass[static_cast<size_t>(active[i])];
                score[i] = settings.criterion == ReuseCriterion::Impact ? attention * score[i] : attention;
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

// The corpus (kReuseDocuments): the knowledge base of a company that builds
// electric cargo bikes, and one question per document.
constexpr std::string_view kCorpusSystem =
    "You are the assistant of Harbor & Pine, a small company that builds and services electric cargo bikes. Answer "
    "the question using only the documents below. Quote numbers exactly and say which document they come from. If "
    "the documents do not answer the question, say so in one sentence.";

constexpr std::string_view kCorpusDocuments[kReuseDocuments] = {
    "Warranty policy (revised March). Frames are covered for seven years against defects in materials and "
    "workmanship. Motors and controllers are covered for two years or 10,000 km, whichever comes first. Batteries "
    "are covered for two years and must keep at least 70% of their rated capacity during that time; a battery below "
    "that level is replaced, not repaired. Wear parts such as brake pads, chains, tyres and cassettes are not "
    "covered. Damage from crashes, from cleaning with a pressure washer or from motors tuned past 25 km/h voids the "
    "warranty of the affected parts. Claims need the frame number and the original invoice.",
    "Battery care sheet. Store the battery between 10 and 25 degrees Celsius and at 40 to 60 percent charge when the "
    "bike is not used for more than two weeks. Never charge below 0 degrees: let a cold battery warm up indoors for "
    "an hour first. A full charge of the 720 Wh pack takes about 5 hours with the standard 4 A charger and 2.5 hours "
    "with the fast charger. Expect about 800 full cycles before the capacity falls to 80 percent. If the battery "
    "case is cracked, swollen or smells of solvent, stop using it and bring it to a service point in its box.",
    "Service intervals. The first inspection is due after 300 km or three months. After that, a standard service "
    "every 2,000 km or once a year covers brakes, drivetrain, wheel truing, bolt torques and a firmware update. "
    "Heavy cargo users (over 100 kg of load most days) should halve the interval. Hydraulic brake fluid is replaced "
    "every two years. A standard service costs 129 euros; parts are extra. The service book must be stamped at "
    "every visit to keep the frame warranty valid beyond the third year.",
    "Model line-up for the coming season. The Courier has a 120 cm front box, carries 100 kg in the box and 180 kg "
    "in total, and weighs 38 kg. The Courier Long adds 30 cm of box and a second battery slot. The Nimbus is a "
    "longtail for two children with a 65 kg rear rack and a total load of 200 kg. All three use a 90 Nm mid-drive "
    "motor and a hub gear with a belt. Prices start at 5,490 euros for the Courier, 6,290 for the Courier Long and "
    "4,990 for the Nimbus.",
    "Delivery and returns. Bikes ordered online ship within 10 working days, fully assembled, on a pallet. Delivery "
    "inside the country costs 89 euros; collection from the workshop in Leeuwarden is free. Customers have 14 days "
    "after delivery to return a bike that has been ridden less than 50 km, in its original packaging; the return "
    "transport is paid by the customer. Accessories can be returned within 30 days unless they were fitted to a "
    "bike. Refunds are paid within 14 days of receiving the return.",
    "Workshop notes, belt drives. A belt stretched by more than 2% or with cracked teeth must be replaced. Belt "
    "tension is set with the phone app: the target frequency is 45 to 50 Hz for the Courier and 50 to 55 Hz for the "
    "Nimbus. A squeaking belt is usually misaligned, not dry: check the rear wheel alignment before adding any "
    "silicone spray, and never use chain oil. A belt lasts about 15,000 km, roughly three times as long as a chain "
    "on the same bike.",
    "Error codes of the display. E01: the motor sensor lost contact, check the plug under the bottom bracket. E07: "
    "battery temperature outside the allowed range, let it cool down or warm up. E12: the speed sensor sees no "
    "magnet, check the gap of 2 to 5 mm at the rear wheel. E21: controller overload after a long climb with full "
    "load; the motor resumes after five minutes. E30: firmware mismatch between display and motor, run an update "
    "from the app. Any other code means the bike must go to a service point.",
    "Accessory compatibility. The rain cover fits every Courier box but not the Nimbus. The child seat adapter is "
    "for the Nimbus only and holds seats up to 22 kg. The second battery slot exists only on the Courier Long, and "
    "both batteries must be the same 720 Wh type. Front lights draw power from the main battery; the rear light "
    "runs on it too since frame number HP-2300. Trailers are allowed behind the Nimbus only, up to 40 kg, with the "
    "reinforced axle nut kit.",
    "Leasing for companies. Businesses can lease any model for 36 or 48 months. The monthly rate includes the yearly "
    "standard service, theft insurance with a 10% deductible and a replacement bike within 48 hours when a repair "
    "takes longer than three days. At the end of the term the bike can be bought for 15% of its list price or "
    "returned. Leasing contracts need a company registration number and at least two bikes per contract.",
    "Theft and insurance. Every bike leaves the workshop with a frame number engraved under the bottom bracket and a "
    "GPS tracker inside the down tube. The tracker's subscription is free for the first year and costs 4 euros per "
    "month afterwards. Insurance through our partner covers theft and accidental damage for 9% of the bike's price "
    "per year. A theft must be reported to the police within 24 hours, and the report number is needed for the "
    "claim.",
    "Riding with children. Children under four years ride in the box with the five-point harness of the bench "
    "fastened. The bench holds two children of up to 25 kg each on the Courier. The Nimbus carries two children on "
    "the rear rack with the side bars and foot plates mounted. A helmet is mandatory for every child on board. Do "
    "not leave children in a parked cargo bike without the parking stand fully down and the brake lock engaged.",
    "Firmware release notes, version 4.2. Adds a walk assist mode for pushing the loaded bike uphill at up to 6 km/h. "
    "The range estimate now learns from the last 50 km instead of the last 10 km. Fixes a bug that showed E12 after "
    "the wheel had been removed and refitted. Lowers the motor's peak power by 5% when the battery is below 15% so "
    "that the last kilometres are not lost. Updating takes about 12 minutes; keep the bike switched on and the "
    "phone within 2 metres.",
};

constexpr std::string_view kCorpusQuestions[kReuseDocuments] = {
    "How long is the motor covered, and what voids the warranty?",
    "My bike stood in the garage all winter at full charge. Was that right, and how should I charge it now?",
    "I carry about 150 kg most days. How often should my bike be serviced and what does it cost?",
    "Which model can carry two children, and how much does it cost?",
    "Can I return a bike I rode for 30 km, and who pays the transport?",
    "The belt squeaks. What should I check first and what must I avoid?",
    "The display shows E12 after I refitted the rear wheel. What does it mean and how do I fix it?",
    "Can I mount a second battery on my Courier, and which battery types can be combined?",
    "What happens at the end of a company lease?",
    "My bike was stolen yesterday. What do I need for the insurance claim?",
    "How many children can ride on the Courier, and what is required for them?",
    "What changes in firmware 4.2, and how long does the update take?",
};

std::string json_string(std::string_view s) {
    std::string out = "\"";
    for (const char ch : s) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default: out += ch;
        }
    }
    return out + "\"";
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

// The calibration's prompts. Three store the 12 documents, four documents
// each; four arrange 3 to 5 of them in other orders, and after each the
// three questions are asked in turn: every document is stored, and every
// question asked, once.
constexpr size_t kStoreOrders[][4] = {{3, 0, 2, 1}, {6, 4, 7, 5}, {9, 11, 8, 10}};
struct Arrangement {
    std::array<size_t, 5> documents;
    size_t count;
    std::array<size_t, 3> questions;
};
constexpr Arrangement kArrangements[] = {
    {{5, 1, 9}, 3, {1, 9, 5}},
    {{2, 10, 7, 0}, 4, {10, 0, 7}},
    {{11, 3, 6, 8, 4}, 5, {6, 11, 3}},
    {{8, 2, 4, 9}, 4, {4, 2, 8}},
};
constexpr size_t kStorePrompts = std::size(kStoreOrders);
constexpr size_t kShares = std::size(kReuseFractions);

}  // namespace

Result<ReusePrompt> reuse_corpus_prompt(const ReusePromptStyle& style, std::span<const size_t> documents,
                                        size_t question) {
    if (question >= kReuseDocuments) return Status(ErrorCode::InvalidArgument, "no such question in the corpus");
    std::string system(kCorpusSystem);
    for (size_t i = 0; i < documents.size(); ++i) {
        if (documents[i] >= kReuseDocuments) return Status(ErrorCode::InvalidArgument, "no such document in the corpus");
        system += "\n\n";
        if (style.numbered) system += "Document " + std::to_string(i + 1) + ":\n";
        system += kCorpusDocuments[documents[i]];
    }
    ReusePrompt p;
    if (style.chat != nullptr) {
        const std::string request = "{\"messages\": [{\"role\": \"system\", \"content\": " + json_string(system) +
                                    "}, {\"role\": \"user\", \"content\": " + json_string(kCorpusQuestions[question]) +
                                    "}], \"add_generation_prompt\": true" +
                                    (style.thinking_off ? ", \"enable_thinking\": false" : "") + "}";
        auto rendered = style.chat->render_chat(request, style.special);
        if (!rendered) return rendered.status();
        p.text = std::move(rendered).value();
    } else {
        p.text = system + "\n\nQuestion: " + std::string(kCorpusQuestions[question]) + "\nAnswer:";
    }
    size_t from = 0;
    for (const size_t d : documents) {
        const size_t at = p.text.find(kCorpusDocuments[d], from);
        if (at == std::string::npos) return Status(ErrorCode::Internal, "a document is not in the rendered prompt");
        p.documents.push_back({at, at + kCorpusDocuments[d].size()});
        from = at + kCorpusDocuments[d].size();
    }
    return p;
}

struct ReuseCalibrator::State {
    const Tokenizer* tokenizer = nullptr;
    ReuseSettings settings;
    double max_kl = 0.0;
    bool guided = false;  // the criterion reads the question: a reuse pass per question
    int32_t vocab = 0;
    int32_t positions = 0;
    std::unique_ptr<KvStore> store;  // the documents' copies

    struct Prompt {
        std::vector<int32_t> tokens;
        std::vector<TokenSegment> segments;  // the documents
        int32_t documents_end = 0;           // the end of the last one
    };
    struct Question {
        Prompt prompt;
        std::vector<std::vector<float>> reference;  // the full prefill's log-probabilities, per scored position
        std::vector<int32_t> answer;                // its greedy answer, fed back
    };
    struct Arranged {
        std::vector<Question> questions;
        bool shared = true;  // the documents are the same tokens before every question
    };
    std::vector<Prompt> store_prompts;
    std::vector<Arranged> arranged;

    // Per share and question: KL summed over the scored positions, how many,
    // and how many agreed on the top token.
    struct Tally {
        double kl = 0.0;
        int64_t n = 0;
        int64_t agree = 0;
    };
    std::vector<std::vector<Tally>> tallies;  // [share][question]

    size_t stage = 0;
    size_t stages = 0;
    double ms = 0.0;
    ReuseCalibration result;
};

ReuseCalibrator::ReuseCalibrator(std::unique_ptr<State> state) : s_(std::move(state)) {}
ReuseCalibrator::~ReuseCalibrator() = default;

Result<std::unique_ptr<ReuseCalibrator>> ReuseCalibrator::create(const Transformer& model, const Tokenizer& tokenizer,
                                                                 const ReusePromptStyle& style,
                                                                 const ReuseSettings& settings, double max_kl) {
    LIYAB_RETURN_IF_ERROR(model.reuse_support());
    auto st = std::make_unique<State>();
    State& s = *st;
    s.tokenizer = &tokenizer;
    s.settings = settings;
    s.max_kl = max_kl;
    s.guided = settings.criterion == ReuseCriterion::Attention || settings.criterion == ReuseCriterion::Impact;
    s.vocab = model.config().n_vocab;
    const bool bos = tokenizer.add_bos_default();
    auto encode = [&](std::span<const size_t> documents, size_t question, State::Prompt& out) -> Status {
        auto prompt = reuse_corpus_prompt(style, documents, question);
        if (!prompt) return prompt.status();
        LIYAB_RETURN_IF_ERROR(encode_segmented(tokenizer, prompt->text, 0, prompt->documents, bos, out.tokens,
                                               out.segments));
        if (out.segments.size() != documents.size() || out.tokens.size() <= static_cast<size_t>(out.segments.back().end) + 1) {
            return Status(ErrorCode::Internal, "calibration: a document or the question encodes to nothing");
        }
        out.documents_end = out.segments.back().end;
        s.positions = std::max(s.positions, static_cast<int32_t>(out.tokens.size()) + kCalibrationAnswer + 1);
        return Status::ok();
    };
    int64_t document_tokens = 0;
    for (const auto& order : kStoreOrders) {
        State::Prompt p;
        LIYAB_RETURN_IF_ERROR(encode(order, order[0], p));
        for (const TokenSegment& d : p.segments) document_tokens += d.end - d.begin;
        s.store_prompts.push_back(std::move(p));
    }
    for (const Arrangement& a : kArrangements) {
        State::Arranged arranged;
        const std::span<const size_t> documents(a.documents.data(), a.count);
        for (const size_t q : a.questions) {
            State::Question question;
            LIYAB_RETURN_IF_ERROR(encode(documents, q, question.prompt));
            if (!arranged.questions.empty()) {
                // Encoded piece by piece, the documents' tokens do not depend
                // on the question after them, unless the template writes
                // something before them that does: then each question gets
                // a pass of its own.
                const State::Prompt& first = arranged.questions.front().prompt;
                const auto end = static_cast<size_t>(first.documents_end);
                arranged.shared = arranged.shared && question.prompt.documents_end == first.documents_end &&
                                  std::equal(first.tokens.begin(), first.tokens.begin() + static_cast<std::ptrdiff_t>(end),
                                             question.prompt.tokens.begin());
            }
            arranged.questions.push_back(std::move(question));
        }
        s.arranged.push_back(std::move(arranged));
    }
    KvStore::Config sc;
    const uint64_t parts[] = {model.context_fingerprint(), kReuseCalibrationVersion};
    sc.seed = kv_seed_key(parts);
    sc.chunk_tokens = model.kv_cache().config().page_tokens;
    sc.chunk_bytes = model.kv_cache().page_bytes();
    // Every document once, with a page of slack per document.
    sc.ram_bytes = model.kv_cache().rows_bytes(static_cast<int32_t>(document_tokens)) +
                   model.kv_cache().page_bytes() * static_cast<size_t>(kReuseDocuments + 4);
    s.store = KvStore::open(sc);
    if (!s.store) return Status(ErrorCode::Internal, "calibration: no store");
    size_t questions = 0;
    for (const State::Arranged& a : s.arranged) questions += a.questions.size();
    s.tallies.assign(kShares, std::vector<State::Tally>(questions));
    s.stages = kStorePrompts + s.arranged.size() * (1 + kShares);
    return std::unique_ptr<ReuseCalibrator>(new ReuseCalibrator(std::move(st)));
}

int32_t ReuseCalibrator::positions() const noexcept { return s_->positions; }

float ReuseCalibrator::progress() const noexcept {
    return s_->stages == 0 ? 1.0f : static_cast<float>(s_->stage) / static_cast<float>(s_->stages);
}

bool ReuseCalibrator::done() const noexcept { return s_->stage >= s_->stages; }

const ReuseCalibration& ReuseCalibrator::result() const noexcept { return s_->result; }

Result<bool> ReuseCalibrator::step(Transformer& model, const Route& route, ThreadPool& pool, int32_t chunk,
                                   const std::atomic<bool>* stop) {
    State& s = *s_;
    if (s.stage >= s.stages) return true;
    if (model.config().n_vocab != s.vocab || chunk < 1) {
        return Status(ErrorCode::InvalidArgument, "calibration: another model, or no pass size");
    }
    const auto t0 = Clock::now();
    ForwardHooks hooks;
    hooks.cancel = stop;
    const Status cancelled(ErrorCode::Cancelled, "calibration stopped");
    auto stopped = [&] { return stop != nullptr && stop->load(std::memory_order_relaxed); };
    // A pass of the context's next tokens; a stopped pass is abandoned.
    auto run = [&](std::span<const int32_t> tokens, Transformer::Logits logits) -> Result<std::span<const float>> {
        auto r = model.forward(tokens, logits, route, pool, &hooks);
        if (!r && r.status().code() == ErrorCode::Cancelled) {
            (void)model.abandon_pass();
            return cancelled;
        }
        return r;
    };
    auto prefill = [&](std::span<const int32_t> tokens) -> Status {
        for (size_t i = 0; i < tokens.size(); i += static_cast<size_t>(chunk)) {
            auto r = run(tokens.subspan(i, std::min(static_cast<size_t>(chunk), tokens.size() - i)),
                         Transformer::Logits::None);
            if (!r) return r.status();
        }
        return Status::ok();
    };
    // The prompt up to position `to`, from where the context is.
    auto prefill_to = [&](const State::Prompt& p, int32_t to) -> Status {
        const int32_t from = model.n_past();
        if (from >= to) return Status::ok();
        return prefill(std::span(p.tokens).subspan(static_cast<size_t>(from), static_cast<size_t>(to - from)));
    };
    auto finish = [&](Status st) -> Result<bool> {
        s.ms += ms_since(t0);
        if (!st.is_ok()) return st;
        ++s.stage;
        if (s.stage == s.stages) s.result = result_for(s.max_kl);
        return s.stage >= s.stages;
    };

    if (s.stage < kStorePrompts) {
        // Stored from an earlier prompt: the documents processed in another order.
        const State::Prompt& p = s.store_prompts[s.stage];
        model.reset();
        if (Status st = prefill_to(p, p.documents_end); !st.is_ok()) return finish(st);
        keep_segments(model, *s.store, p.tokens, p.segments, s.settings.min_tokens);
        return finish(Status::ok());
    }
    const size_t t = s.stage - kStorePrompts;
    State::Arranged& a = s.arranged[t / (1 + kShares)];
    const size_t share = t % (1 + kShares);
    size_t first_question = 0;  // index of the arrangement's first question among all
    for (size_t i = 0; i < t / (1 + kShares); ++i) first_question += s.arranged[i].questions.size();

    // The context holds a question's prompt from its documents' end on, at
    // most: the rest of it processed, then the first answer token's
    // distribution and the reference answer's next ones (teacher-forced).
    auto answer = [&](const State::Question& q) -> Result<std::vector<std::vector<double>>> {
        const auto n = static_cast<int32_t>(q.prompt.tokens.size());
        LIYAB_RETURN_IF_ERROR(prefill_to(q.prompt, n - 1));
        auto first = run(std::span(q.prompt.tokens).last(1), Transformer::Logits::Last);
        if (!first) return first.status();
        std::vector<std::vector<double>> mine = {log_probs(*first)};
        if (!q.answer.empty()) {
            auto rows = run(q.answer, Transformer::Logits::All);
            if (!rows) return rows.status();
            const auto vocab = static_cast<size_t>(s.vocab);
            for (size_t i = 0; i < q.answer.size(); ++i) {
                mine.push_back(log_probs(std::span<const float>(rows->data() + i * vocab, vocab)));
            }
        }
        return mine;
    };

    if (share == 0) {
        // The reference: each question answered from a full prefill, greedy,
        // up to its end of generation (included: where it ends is part of
        // what reuse may change; what a model says past it is not an answer).
        for (State::Question& q : a.questions) {
            q.reference.clear();
            q.answer.clear();
        }
        model.reset();
        for (State::Question& q : a.questions) {
            const State::Prompt& p = q.prompt;
            if (a.shared && model.n_past() >= p.documents_end) {
                if (Status st = model.truncate(p.documents_end); !st.is_ok()) return finish(st);
            } else {
                model.reset();
            }
            if (Status st = prefill_to(p, static_cast<int32_t>(p.tokens.size()) - 1); !st.is_ok()) return finish(st);
            auto dist = run(std::span(p.tokens).last(1), Transformer::Logits::Last);
            for (int32_t i = 0;; ++i) {
                if (!dist) return finish(dist.status());
                const auto best = static_cast<int32_t>(std::max_element(dist->begin(), dist->end()) - dist->begin());
                const std::vector<double> lp = log_probs(*dist);
                q.reference.emplace_back(lp.begin(), lp.end());
                if (i == kCalibrationAnswer || s.tokenizer->is_end_of_generation(best)) break;
                q.answer.push_back(best);
                dist = run(std::span<const int32_t>(&q.answer.back(), 1), Transformer::Logits::Last);
            }
            if (stopped()) return finish(cancelled);
        }
        return finish(Status::ok());
    }

    // One share: the documents reused (once for every question when they are
    // shared and the criterion does not read the question), each question
    // answered, its positions scored against the reference.
    ReuseSettings rs = s.settings;
    rs.fraction = kReuseFractions[share - 1];
    auto reuse_documents = [&](const State::Prompt& p, int32_t limit) -> Status {
        const int32_t exact = p.segments.front().begin;
        model.reset();
        LIYAB_RETURN_IF_ERROR(prefill_to(p, exact));
        auto reused = reuse_segments(model, *s.store, p.tokens, p.segments, exact, limit, rs, route, pool, chunk, stop);
        if (!reused) return reused.status();
        return prefill_to(p, p.documents_end);
    };
    const bool shared = a.shared && !s.guided;
    std::vector<State::Tally> got(a.questions.size());
    if (shared) {
        if (Status st = reuse_documents(a.questions.front().prompt, a.questions.front().prompt.documents_end); !st.is_ok()) {
            return finish(st);
        }
    }
    for (size_t i = 0; i < a.questions.size(); ++i) {
        const State::Question& q = a.questions[i];
        const auto n = static_cast<int32_t>(q.prompt.tokens.size());
        Status st = shared ? model.truncate(q.prompt.documents_end)
                           : reuse_documents(q.prompt, s.guided ? n - 1 : q.prompt.documents_end);
        if (!st.is_ok()) return finish(st);
        auto mine = answer(q);
        if (!mine) return finish(mine.status());
        for (size_t p = 0; p < mine->size() && p < q.reference.size(); ++p) {
            const std::vector<float>& ref = q.reference[p];
            const std::vector<double>& m = (*mine)[p];
            double k = 0.0;
            size_t ref_best = 0, best = 0;
            for (size_t v = 0; v < ref.size(); ++v) {
                k += std::exp(static_cast<double>(ref[v])) * (ref[v] - m[v]);
                if (ref[v] > ref[ref_best]) ref_best = v;
                if (m[v] > m[best]) best = v;
            }
            got[i].kl += std::max(k, 0.0);
            got[i].n += 1;
            got[i].agree += ref_best == best ? 1 : 0;
        }
        if (stopped()) return finish(cancelled);
    }
    for (size_t i = 0; i < got.size(); ++i) s.tallies[share - 1][first_question + i] = got[i];
    if (share == kShares) {
        for (State::Question& q : a.questions) {
            q.reference = {};  // this arrangement is measured: its distributions are not needed again
            q.reference.shrink_to_fit();
        }
    }
    return finish(Status::ok());
}

ReuseCalibration ReuseCalibrator::result_for(double max_kl) const {
    const State& s = *s_;
    ReuseCalibration out;
    out.ms = s.ms;
    std::vector<double> top1(kShares, 0.0);
    for (size_t k = 0; k < kShares; ++k) {
        double kl = 0.0;
        int64_t n = 0, agree = 0;
        std::vector<double> means;
        for (const State::Tally& t : s.tallies[k]) {
            kl += t.kl;
            n += t.n;
            agree += t.agree;
            if (t.n > 0) means.push_back(t.kl / static_cast<double>(t.n));
        }
        out.question_kls.push_back(means);
        const double mean = n > 0 ? kl / static_cast<double>(n) : 0.0;
        // The questions' own means vary more than positions within one answer
        // do (an answer's tokens move together): their spread sets the margin.
        double sd = 0.0;
        if (means.size() > 1) {
            const double m = std::accumulate(means.begin(), means.end(), 0.0) / static_cast<double>(means.size());
            for (const double v : means) sd += (v - m) * (v - m);
            sd = std::sqrt(sd / static_cast<double>(means.size() - 1));
        }
        const double se = means.empty() ? 0.0 : sd / std::sqrt(static_cast<double>(means.size()));
        out.kls.push_back(mean);
        out.bounds.push_back(mean + kCalibrationBoundZ * se);
        top1[k] = n > 0 ? static_cast<double>(agree) / static_cast<double>(n) : 0.0;
    }
    // The smallest share whose bound, and every larger share's, is within
    // the target.
    int64_t chosen = -1;
    for (auto k = static_cast<int64_t>(kShares) - 1; k >= 0 && out.bounds[static_cast<size_t>(k)] <= max_kl; --k) {
        chosen = k;
    }
    out.met = chosen >= 0;
    const auto at = static_cast<size_t>(out.met ? chosen : static_cast<int64_t>(kShares) - 1);
    out.fraction = kReuseFractions[at];
    out.kl = out.kls[at];
    out.bound = out.bounds[at];
    out.top1 = top1[at];
    return out;
}

Result<ReuseCalibration> calibrate_reuse(Transformer& model, const Tokenizer& tokenizer, const ReusePromptStyle& style,
                                         const ReuseSettings& settings, double max_kl, const Route& route,
                                         ThreadPool& pool, int32_t chunk) {
    auto calibrator = ReuseCalibrator::create(model, tokenizer, style, settings, max_kl);
    if (!calibrator) return calibrator.status();
    for (;;) {
        auto done = (*calibrator)->step(model, route, pool, chunk);
        if (!done) {
            model.reset();
            return done.status();
        }
        if (*done) break;
    }
    model.reset();
    return (*calibrator)->result();
}

int32_t keep_segments(const Transformer& model, KvStore& store, std::span<const int32_t> tokens,
                      std::span<const TokenSegment> segments, int32_t min_tokens) {
    const KvCache& kv = model.kv_cache();
    const int32_t c = store.chunk_tokens();
    const int32_t page_tokens = kv.config().page_tokens;
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
        // A chunk's image is its pages' images in order. Chain position p
        // holds segment token p - 1: the context's position begin + p - 1,
        // rarely on a page boundary, so every page image is packed (into
        // `packed`, alive while put() runs). The two markers get zero rows
        // (never read).
        std::deque<std::vector<uint8_t>> packed;
        source.rows = [&](int32_t chunk, int32_t count) {
            std::vector<std::span<const uint8_t>> parts;
            for (int32_t k = 0; k * page_tokens < count; ++k) {
                const int32_t start = chunk * c + k * page_tokens, rows = std::min(page_tokens, count - k * page_tokens);
                packed.emplace_back(kv.rows_bytes(rows), uint8_t{0});
                const int32_t p0 = std::max(start, 1), p1 = std::min(start + rows, n + 1);
                if (p0 < p1 && !kv.read_rows_at(s.begin + p0 - 1, p1 - p0, packed.back().data(), rows, p0 - start).is_ok()) {
                    return std::vector<std::span<const uint8_t>>();
                }
                parts.emplace_back(packed.back());
            }
            return parts;
        };
        if (store.put(source).whole) ++kept;
    }
    return kept;
}

}  // namespace liyab
