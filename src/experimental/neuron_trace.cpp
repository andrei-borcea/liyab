#include "liyab/experimental/neuron_trace.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "core/log.h"

namespace liyab::experimental {

namespace {

// The neurons whose squares add up to at least `share` of the total, as a 0/1 byte per neuron: the largest ones, down
// to the threshold the sort finds. Equal values at the threshold are all in (the set can exceed the minimum by ties).
size_t energy_set(const float* h, size_t n, double total, double share, std::vector<float>& scratch,
                  std::vector<uint8_t>& set) {
    scratch.resize(n);
    for (size_t i = 0; i < n; ++i) scratch[i] = h[i] * h[i];
    std::sort(scratch.begin(), scratch.end(), std::greater<float>());
    double acc = 0.0;
    float threshold = scratch[0];
    for (size_t i = 0; i < n; ++i) {
        acc += scratch[i];
        threshold = scratch[i];
        if (acc >= share * total) break;
    }
    set.assign(n, 0);
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (h[i] * h[i] >= threshold) {
            set[i] = 1;
            ++count;
        }
    }
    return count;
}

double energy_in(const float* h, size_t n, const std::vector<uint8_t>& set) {
    double e = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (set[i]) e += static_cast<double>(h[i]) * h[i];
    }
    return e;
}

}  // namespace

void NeuronTrace::begin(int32_t n_layers) {
    n_layers_ = n_layers;
    last_layer_ = -1;
    tokens_ = 0;
    blocks_.assign(static_cast<size_t>(std::max(n_layers, 0)), {});
    sum_ = {};
}

void NeuronTrace::end_token() {
    ++tokens_;
    sum_.tokens = tokens_;
    if (tokens_ == train_tokens_) {
        for (Block& b : blocks_) {
            if (b.frequency.empty()) continue;
            b.hot_rank.resize(b.frequency.size());
            std::iota(b.hot_rank.begin(), b.hot_rank.end(), 0u);
            std::stable_sort(b.hot_rank.begin(), b.hot_rank.end(),
                             [&](uint32_t a, uint32_t c) { return b.frequency[a] > b.frequency[c]; });
        }
    }
    if (report_interval_ > 0 && tokens_ % report_interval_ == 0) LIYAB_LOG_INFO("%s", report().c_str());
}

void NeuronTrace::activations(int32_t layer, std::span<const float> hidden) {
    if (layer < 0 || layer >= n_layers_ || hidden.empty()) return;
    if (layer <= last_layer_) end_token();
    last_layer_ = layer;
    Block& block = blocks_[static_cast<size_t>(layer)];
    const size_t n = hidden.size();
    double total = 0.0;
    for (const float v : hidden) total += static_cast<double>(v) * v;
    if (!(total > 0.0) || !std::isfinite(total)) return;

    std::vector<float> scratch;
    std::array<std::vector<uint8_t>, 2> sets;
    const bool had_previous = block.previous[0].size() == n;
    ++sum_.observations;
    for (size_t k = 0; k < kCoverage.size(); ++k) {
        const size_t count = energy_set(hidden.data(), n, total, kCoverage[k], scratch, sets[k]);
        sum_.density[k] += static_cast<double>(count) / static_cast<double>(n);
        if (had_previous) {
            size_t both = 0, either = 0;
            for (size_t i = 0; i < n; ++i) {
                both += sets[k][i] && block.previous[k][i];
                either += sets[k][i] || block.previous[k][i];
            }
            sum_.jaccard[k] += either > 0 ? static_cast<double>(both) / static_cast<double>(either) : 1.0;
            sum_.recall[k] += energy_in(hidden.data(), n, block.previous[k]) / total;
        }
    }
    if (had_previous) ++sum_.persistence_pairs;

    if (tokens_ < train_tokens_) {
        if (block.frequency.size() != n) block.frequency.assign(n, 0);
        for (size_t i = 0; i < n; ++i) block.frequency[i] += sets[0][i];
    } else if (block.hot_rank.size() == n) {
        ++sum_.hot_observations;
        std::vector<uint8_t> hot(n, 0);
        size_t taken = 0;
        std::array<double, 3> recalls{};
        for (size_t k = 0; k < kHotShare.size(); ++k) {
            const auto want = static_cast<size_t>(kHotShare[k] * static_cast<double>(n));
            for (; taken < want; ++taken) hot[block.hot_rank[taken]] = 1;
            recalls[k] = energy_in(hidden.data(), n, hot) / total;
            sum_.hot_recall[k] += recalls[k];
            if (k == 1) {  // the top 25% plus the previous token's set
                std::vector<uint8_t> both = hot;
                size_t size = 0;
                if (had_previous) {
                    for (size_t i = 0; i < n; ++i) both[i] = both[i] || block.previous[0][i];
                }
                for (size_t i = 0; i < n; ++i) size += both[i];
                sum_.union_recall += energy_in(hidden.data(), n, both) / total;
                sum_.union_density += static_cast<double>(size) / static_cast<double>(n);
            }
        }
    }
    block.previous = sets;
}

NeuronTrace::Summary NeuronTrace::summary() const {
    Summary out = sum_;
    const auto mean = [](double sum, int64_t count) { return count > 0 ? sum / static_cast<double>(count) : 0.0; };
    for (size_t k = 0; k < kCoverage.size(); ++k) {
        out.density[k] = mean(sum_.density[k], sum_.observations);
        out.jaccard[k] = mean(sum_.jaccard[k], sum_.persistence_pairs);
        out.recall[k] = mean(sum_.recall[k], sum_.persistence_pairs);
    }
    for (size_t k = 0; k < kHotShare.size(); ++k) out.hot_recall[k] = mean(sum_.hot_recall[k], sum_.hot_observations);
    out.union_recall = mean(sum_.union_recall, sum_.hot_observations);
    out.union_density = mean(sum_.union_density, sum_.hot_observations);
    return out;
}

std::string NeuronTrace::report() const {
    const Summary s = summary();
    char text[640];
    std::snprintf(text, sizeof text,
                  "neuron trace: %lld tokens | 90%% of the energy: density %.3f, jaccard %.3f, previous set recall %.3f | "
                  "99%%: density %.3f, jaccard %.3f, recall %.3f | hot (ranked on %d tokens) recall: top 10%% %.3f, "
                  "25%% %.3f, 50%% %.3f; 25%% + previous: recall %.3f at density %.3f",
                  static_cast<long long>(s.tokens), s.density[0], s.jaccard[0], s.recall[0], s.density[1], s.jaccard[1],
                  s.recall[1], train_tokens_, s.hot_recall[0], s.hot_recall[1], s.hot_recall[2], s.union_recall,
                  s.union_density);
    return text;
}

}  // namespace liyab::experimental
