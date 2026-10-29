#include "liyab/experimental/prerouter_trace.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "core/log.h"

namespace liyab::experimental {

const std::array<PrerouterTrace::Candidate, PrerouterTrace::kCandidates>& PrerouterTrace::candidates() noexcept {
    static const std::array<Candidate, kCandidates> list = {{
        {"A router", 0.0f, 0.0f},
        {"B b=.5", 0.5f, 0.0f},
        {"B b=1", 1.0f, 0.0f},
        {"B b=2", 2.0f, 0.0f},
        {"C g=1", 0.0f, 1.0f},
        {"C g=2", 0.0f, 2.0f},
        {"B1+C1", 1.0f, 1.0f},
    }};
    return list;
}

void PrerouterTrace::begin(int32_t n_layers, int32_t n_experts, int32_t experts_per_token) {
    n_layers_ = n_layers;
    n_experts_ = n_experts;
    k_ = experts_per_token;
    last_layer_ = -1;
    tokens_ = guesses_ = 0;
    hits_.fill(0);
    const auto layers = static_cast<size_t>(std::max(n_layers, 0));
    logits_.assign(layers, {});
    // The pair counts (n_experts^2 floats a block) are allocated when a block first learns, so dense blocks cost nothing.
    pairs_.assign(layers, {});
    times_.assign(layers, {});
    previous_.assign(layers, {});
    current_.assign(layers, {});
}

void PrerouterTrace::predicted(int32_t layer, std::span<const float> router_logits) {
    if (layer < 0 || layer >= n_layers_ || router_logits.size() != static_cast<size_t>(n_experts_)) return;
    logits_[static_cast<size_t>(layer)].assign(router_logits.begin(), router_logits.end());
}

void PrerouterTrace::end_token() {
    ++tokens_;
    for (auto& experts : current_) experts.clear();  // a block that does not run this token has no prior for the next
    if (report_interval_ > 0 && tokens_ % report_interval_ == 0) LIYAB_LOG_INFO("%s", report().c_str());
}

void PrerouterTrace::chosen(int32_t layer, std::span<const int32_t> experts) {
    if (layer < 0 || layer >= n_layers_) return;
    if (layer <= last_layer_) end_token();
    last_layer_ = layer;
    const auto l = static_cast<size_t>(layer);
    const auto E = static_cast<size_t>(n_experts_);
    const auto K = static_cast<size_t>(k_);
    std::vector<float>& z = logits_[l];

    if (z.size() == E && K > 0 && K <= E) {
        // The prior: P(b | the experts block l - 1 ran for this token), smoothed by 0.1 per expert.
        const bool have_prior = layer > 0 && !current_[l - 1].empty();
        std::vector<float> prior(E, 0.0f);
        if (have_prior) {
            const std::vector<float>& before_times = times_[l - 1];
            const std::vector<float>& before_pairs = pairs_[l - 1];
            float denominator = 0.1f * static_cast<float>(E);
            if (!before_times.empty()) {
                for (const int32_t a : current_[l - 1]) denominator += before_times[static_cast<size_t>(a)];
            }
            for (size_t e = 0; e < E; ++e) {
                float numerator = 0.1f;
                if (!before_pairs.empty()) {
                    for (const int32_t a : current_[l - 1]) numerator += before_pairs[static_cast<size_t>(a) * E + e];
                }
                prior[e] = std::log(numerator / denominator);
            }
        }
        std::vector<char> ran_last_token(E, 0);
        for (const int32_t e : previous_[l]) ran_last_token[static_cast<size_t>(e)] = 1;

        std::vector<int32_t> order(E);
        for (int i = 0; i < kCandidates; ++i) {
            const Candidate& c = candidates()[static_cast<size_t>(i)];
            const auto score = [&](size_t e) {
                return z[e] + (have_prior ? c.prior_weight * prior[e] : 0.0f) + c.repeat_weight * static_cast<float>(ran_last_token[e]);
            };
            for (size_t e = 0; e < E; ++e) order[e] = static_cast<int32_t>(e);
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(K), order.end(),
                              [&](int32_t a, int32_t b) { return score(static_cast<size_t>(a)) > score(static_cast<size_t>(b)); });
            for (size_t k = 0; k < K; ++k) hits_[static_cast<size_t>(i)] += std::binary_search(experts.begin(), experts.end(), order[k]);
        }
        guesses_ += static_cast<int64_t>(K);
    }
    z.clear();

    // Learn from this token, then remember it.
    if (layer > 0 && !current_[l - 1].empty() && !experts.empty()) {
        std::vector<float>& pairs = pairs_[l - 1];
        if (pairs.empty()) pairs.assign(E * E, 0.0f);
        for (const int32_t a : current_[l - 1]) {
            for (const int32_t b : experts) pairs[static_cast<size_t>(a) * E + static_cast<size_t>(b)] += 1.0f;
        }
    }
    if (times_[l].empty()) times_[l].assign(E, 0.0f);
    for (const int32_t e : experts) times_[l][static_cast<size_t>(e)] += 1.0f;
    current_[l].assign(experts.begin(), experts.end());
    previous_[l].assign(experts.begin(), experts.end());
}

double PrerouterTrace::precision(int index) const noexcept {
    if (index < 0 || index >= kCandidates || guesses_ == 0) return 0.0;
    return static_cast<double>(hits_[static_cast<size_t>(index)]) / static_cast<double>(guesses_);
}

std::string PrerouterTrace::report() const {
    char head[96];
    std::snprintf(head, sizeof head, "prerouter: %lld tokens, precision@%d:", static_cast<long long>(tokens_), k_);
    std::string text = head;
    for (int i = 0; i < kCandidates; ++i) {
        char item[64];
        std::snprintf(item, sizeof item, "%s %s %.1f%%", i == 0 ? "" : " |", candidates()[static_cast<size_t>(i)].name,
                      100.0 * precision(i));
        text += item;
    }
    return text;
}

}  // namespace liyab::experimental
