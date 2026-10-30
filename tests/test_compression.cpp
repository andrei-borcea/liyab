// Independent dense recomputation of shared-KV compression components. These
// fixtures deliberately have dense FFNs and no learned sinks; full V4 MoE
// conformance remains a separate integration gate.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

#include "core/compression.h"
#include "core/kernel_conformance.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "core/transformer.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;

namespace {
using Row = std::vector<float>;
using Rows = std::vector<Row>;

std::string path() {
    return test::temp_dir() + "/liyab_compression_" + std::to_string(test::process_id()) + ".gguf";
}

Row norm(const Row& x, const Row& weight = {}) {
    double sum = 0;
    for (float value : x) sum += double{value} * value;
    const double scale = 1 / std::sqrt(sum / x.size() + 1e-5);
    Row out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = static_cast<float>(x[i] * scale * (weight.empty() ? 1 : weight[i]));
    return out;
}

Row linear(const Row& weight, const Row& input) {
    Row output(weight.size() / input.size());
    for (size_t i = 0; i < output.size(); ++i) {
        double sum = 0;
        for (size_t j = 0; j < input.size(); ++j) sum += double{weight[i * input.size() + j]} * input[j];
        output[i] = static_cast<float>(sum);
    }
    return output;
}

void rotate(Row& row, int heads, int rd, int pos, float base) {
    const size_t dim = row.size() / static_cast<size_t>(heads);
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < rd / 2; ++i) {
            const double theta = pos * std::pow(double{base}, -2.0 * i / rd);
            const size_t at = static_cast<size_t>(h) * dim + dim - static_cast<size_t>(rd) + static_cast<size_t>(2 * i);
            const float a = row[at], b = row[at + 1];
            row[at] = static_cast<float>(a * std::cos(theta) - b * std::sin(theta));
            row[at + 1] = static_cast<float>(a * std::sin(theta) + b * std::cos(theta));
        }
    }
}

// Recompute an entire window, including the previous window's Ca half when
// requested. The engine instead updates bounded online accumulators.
Row compress(const Rows& kv, const Rows& gate, const Row& bias, int end, int stride, int dim, bool overlap) {
    Row out(static_cast<size_t>(dim));
    const int begin = end - stride, width = dim * (overlap ? 2 : 1);
    for (int channel = 0; channel < dim; ++channel) {
        std::vector<std::pair<double, double>> terms;
        const int first = overlap ? std::max(0, begin - stride) : begin;
        for (int pos = first; pos < end; ++pos) {
            const int offset = overlap && pos >= begin ? dim : 0;
            const size_t at = static_cast<size_t>(offset + channel);
            const double g = gate[static_cast<size_t>(pos)][at] + bias[static_cast<size_t>(pos % stride * width) + at];
            terms.emplace_back(g, kv[static_cast<size_t>(pos)][at]);
        }
        double maximum = -INFINITY;
        for (const auto& term : terms) maximum = std::max(maximum, term.first);
        double denominator = 0, numerator = 0;
        for (const auto& term : terms) {
            const double probability = std::exp(term.first - maximum);
            denominator += probability;
            numerator += probability * term.second;
        }
        out[static_cast<size_t>(channel)] = static_cast<float>(numerator / denominator);
    }
    return out;
}

class Reference {
public:
    Reference(const MmapLoader& file, test::TinyModelSpec spec, bool quantized_inputs = false)
        : file_(file), spec_(std::move(spec)), quantized_inputs_(quantized_inputs) {}
    Rows logits(const std::vector<int32_t>& tokens) const {
        const auto d = static_cast<size_t>(spec_.n_embd), heads = static_cast<size_t>(spec_.n_head);
        const size_t hd = spec_.shared_head_dim > 0 ? static_cast<size_t>(spec_.shared_head_dim) : d / heads;
        const auto hc = static_cast<size_t>(spec_.hc_count);
        const Row embedding = weights("token_embd.weight");
        Rows x(tokens.size()), streams(tokens.size());
        for (size_t t = 0; t < tokens.size(); ++t) {
            x[t] = Row(embedding.begin() + tokens[t] * static_cast<int64_t>(d), embedding.begin() + (tokens[t] + 1) * static_cast<int64_t>(d));
            for (size_t h = 0; h < hc; ++h) streams[t].insert(streams[t].end(), x[t].begin(), x[t].end());
        }
        for (int layer = 0; layer < spec_.n_layers; ++layer) {
            const std::string p = "blk." + std::to_string(layer) + ".";
            Rows post(tokens.size()), comb(tokens.size());
            if (hc > 0) collapse(streams, p + "hc_attn", x, post, comb);
            Rows input(tokens.size()), q(tokens.size()), k(tokens.size()), latent(tokens.size());
            const int stride = spec_.compression_strides[static_cast<size_t>(layer)];
            const bool overlap = spec_.compression_overlap[static_cast<size_t>(layer)];
            const float base = stride > 0 ? spec_.compress_rope_base : spec_.rope_base;
            for (size_t t = 0; t < tokens.size(); ++t) {
                input[t] = norm(x[t], weights(p + "attn_norm.weight"));
                latent[t] = norm(project(p + "attn_q_a.weight", input[t]), weights(p + "attn_q_a_norm.weight"));
                q[t] = project(p + "attn_q_b.weight", latent[t]);
                for (size_t h = 0; h < heads; ++h) {
                    Row head(q[t].begin() + static_cast<int64_t>(h * hd), q[t].begin() + static_cast<int64_t>((h + 1) * hd));
                    head = norm(head);
                    std::copy(head.begin(), head.end(), q[t].begin() + static_cast<int64_t>(h * hd));
                }
                rotate(q[t], spec_.n_head, spec_.rope_dim, static_cast<int>(t), base);
                k[t] = norm(project(p + "attn_kv.weight", input[t]), weights(p + "attn_kv_a_norm.weight"));
                rotate(k[t], 1, spec_.rope_dim, static_cast<int>(t), base);
                for (float& value : k[t]) value = quant::fp16_to_fp32(quant::fp32_to_fp16(value));
            }
            Rows compressed, index_keys, index_queries(tokens.size()), index_weights(tokens.size());
            if (stride > 0) {
                Rows kv(tokens.size()), gate(tokens.size()), ikv(tokens.size()), igate(tokens.size());
                for (size_t t = 0; t < tokens.size(); ++t) {
                    kv[t] = project(p + "attn_compressor_kv.weight", input[t]);
                    gate[t] = project(p + "attn_compressor_gate.weight", input[t]);
                    if (overlap) {
                        ikv[t] = project(p + "indexer_compressor_kv.weight", input[t]);
                        igate[t] = project(p + "indexer_compressor_gate.weight", input[t]);
                        index_queries[t] = project(p + "indexer.attn_q_b.weight", latent[t]);
                        rotate(index_queries[t], spec_.index_heads, spec_.rope_dim, static_cast<int>(t), base);
                        index_weights[t] = project(p + "indexer.proj.weight", input[t]);
                    }
                }
                for (int end = stride; end <= static_cast<int>(tokens.size()); end += stride) {
                    Row key = norm(compress(kv, gate, weights(p + "attn_compressor_ape.weight"), end, stride, static_cast<int>(hd), overlap),
                                   weights(p + "attn_compressor_norm.weight"));
                    rotate(key, 1, spec_.rope_dim, end - stride, base);
                    compressed.push_back(std::move(key));
                    if (overlap) {
                        key = norm(compress(ikv, igate, weights(p + "indexer_compressor_ape.weight"), end, stride, spec_.index_dim, true),
                                   weights(p + "indexer_compressor_norm.weight"));
                        rotate(key, 1, spec_.rope_dim, end - stride, base);
                        index_keys.push_back(std::move(key));
                    }
                }
            }
            Rows output(tokens.size());
            for (size_t t = 0; t < tokens.size(); ++t) {
                std::vector<std::pair<double, int>> ranked;
                const size_t ready = stride > 0 ? (t + 1) / static_cast<size_t>(stride) : 0;
                for (size_t entry = 0; entry < ready; ++entry) {
                    double score = 0;
                    if (overlap) {
                        for (int h = 0; h < spec_.index_heads; ++h) {
                            double dot = 0;
                            for (int j = 0; j < spec_.index_dim; ++j) {
                                dot += double{index_queries[t][static_cast<size_t>(h * spec_.index_dim + j)]} * index_keys[entry][static_cast<size_t>(j)];
                            }
                            score += std::max(dot, 0.0) / std::sqrt(static_cast<double>(spec_.index_dim)) *
                                     index_weights[t][static_cast<size_t>(h)] / std::sqrt(static_cast<double>(spec_.index_heads));
                        }
                    }
                    ranked.emplace_back(score, static_cast<int>(entry));
                }
                const size_t selected = overlap ? std::min(ready, static_cast<size_t>(spec_.index_topk)) : ready;
                std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
                Row attended(heads * hd, 0.0f);
                for (size_t h = 0; h < heads; ++h) {
                    std::vector<const Row*> visible;
                    const size_t first = t + 1 > static_cast<size_t>(spec_.sliding_window) ? t + 1 - static_cast<size_t>(spec_.sliding_window) : 0;
                    for (size_t pos = first; pos <= t; ++pos) visible.push_back(&k[pos]);
                    for (size_t i = 0; i < selected; ++i) visible.push_back(&compressed[static_cast<size_t>(ranked[i].second)]);
                    std::vector<double> scores;
                    double maximum = -INFINITY;
                    for (const Row* key : visible) {
                        double dot = 0;
                        for (size_t j = 0; j < hd; ++j) dot += double{q[t][h * hd + j]} * (*key)[j];
                        scores.push_back(dot / std::sqrt(static_cast<double>(hd)));
                        maximum = std::max(maximum, scores.back());
                    }
                    double sum = 0;
                    for (double& score : scores) sum += (score = std::exp(score - maximum));
                    for (size_t j = 0; j < hd; ++j) {
                        double value = 0;
                        for (size_t i = 0; i < scores.size(); ++i) value += scores[i] / sum * (*visible[i])[j];
                        attended[h * hd + j] = static_cast<float>(value);
                    }
                }
                rotate(attended, spec_.n_head, spec_.rope_dim, -static_cast<int>(t), base);
                Row grouped;
                const size_t groups = static_cast<size_t>(spec_.shared_output_groups), cols = heads * hd / groups;
                const auto rank = static_cast<size_t>(spec_.shared_output_rank);
                const Row wa = weights(p + "attn_output_a.weight");
                for (size_t group = 0; group < groups; ++group) {
                    Row part = linear(Row(wa.begin() + static_cast<int64_t>(group * rank * cols), wa.begin() + static_cast<int64_t>((group + 1) * rank * cols)),
                                      projection_input(file_.tensor(p + "attn_output_a.weight")->type,
                                          Row(attended.begin() + static_cast<int64_t>(group * cols), attended.begin() + static_cast<int64_t>((group + 1) * cols))));
                    grouped.insert(grouped.end(), part.begin(), part.end());
                }
                output[t] = project(p + "attn_output_b.weight", grouped);
            }
            if (hc > 0) {
                mix(streams, output, post, comb);
                collapse(streams, p + "hc_ffn", x, post, comb);
            } else {
                for (size_t t = 0; t < x.size(); ++t) for (size_t j = 0; j < d; ++j) x[t][j] += output[t][j];
            }
            for (size_t t = 0; t < x.size(); ++t) {
                Row normalized = norm(x[t], weights(p + "ffn_norm.weight"));
                Row gate = project(p + "ffn_gate.weight", normalized), up = project(p + "ffn_up.weight", normalized);
                for (size_t j = 0; j < gate.size(); ++j) gate[j] = gate[j] / (1 + std::exp(-gate[j])) * up[j];
                output[t] = project(p + "ffn_down.weight", gate);
            }
            if (hc > 0) mix(streams, output, post, comb);
            else for (size_t t = 0; t < x.size(); ++t) for (size_t j = 0; j < d; ++j) x[t][j] += output[t][j];
        }
        if (hc > 0) {
            const Row base = weights("output_hc_base.weight"), scale = weights("output_hc_scale.weight");
            for (size_t t = 0; t < x.size(); ++t) {
                const Row coefficient = linear(weights("output_hc_fn.weight"), norm(streams[t]));
                std::fill(x[t].begin(), x[t].end(), 0.0f);
                for (size_t h = 0; h < hc; ++h) {
                    const float pre = 1 / (1 + std::exp(-(coefficient[h] * scale[0] + base[h]))) + 1e-6f;
                    for (size_t j = 0; j < d; ++j) x[t][j] += pre * streams[t][h * d + j];
                }
            }
        }
        for (Row& row : x) row = project("output.weight", norm(row, weights("output_norm.weight")));
        return x;
    }

private:
    // Independently multiply decoded weights and the actual activation
    // values. Raw encoded fixtures can have a much wider range than trained
    // weights; include the backend's documented Q8 activation rounding, not
    // its dot kernels, so the bound measures projection/index/compression
    // correctness rather than quantization error between different inputs.
    Row projection_input(DType type, const Row& input) const {
        if (!quantized_inputs_ || !quant::is_block_quantized(type)) return input;
        Row decoded(input.size());
        const auto n = static_cast<int64_t>(input.size());
        if (quant::uses_q8_K(type) && n % quant::kSuperBlock == 0) {
            std::vector<quant::BlockQ8_K> blocks(input.size() / quant::kSuperBlock);
            quant::quantize_row_q8_K(input.data(), blocks.data(), n);
            for (size_t b = 0; b < blocks.size(); ++b) for (size_t j = 0; j < quant::kSuperBlock; ++j) {
                decoded[b * quant::kSuperBlock + j] = blocks[b].d * blocks[b].qs[j];
            }
        } else {
            std::vector<quant::BlockQ8_0> blocks(input.size() / quant::kBlock);
            quant::quantize_row_q8_0(input.data(), blocks.data(), n);
            quant::dequantize_row(DType::Q8_0, blocks.data(), decoded.data(), n);
        }
        return decoded;
    }
    Row project(const std::string& name, const Row& input) const {
        return linear(weights(name), projection_input(file_.tensor(name)->type, input));
    }
    Row weights(const std::string& name) const {
        const TensorView* tensor = file_.tensor(name);
        Row out(static_cast<size_t>(tensor->elements()));
        for (int64_t i = 0; i < tensor->rows(); ++i) quant::dequantize_row(tensor->type, tensor->row(i), out.data() + i * tensor->cols(), tensor->cols());
        return out;
    }
    void collapse(const Rows& streams, const std::string& name, Rows& x, Rows& post, Rows& comb) const {
        const auto hc = static_cast<size_t>(spec_.hc_count), d = static_cast<size_t>(spec_.n_embd);
        const Row base = weights(name + "_base.weight"), scale = weights(name + "_scale.weight"), projection = weights(name + "_fn.weight");
        for (size_t t = 0; t < streams.size(); ++t) {
            const Row coefficients = linear(projection, norm(streams[t]));
            x[t].assign(d, 0.0f); post[t].resize(hc); comb[t].resize(hc * hc);
            for (size_t h = 0; h < hc; ++h) {
                const float pre = 1 / (1 + std::exp(-(coefficients[h] * scale[0] + base[h]))) + 1e-6f;
                post[t][h] = 2 / (1 + std::exp(-(coefficients[hc + h] * scale[1] + base[hc + h])));
                for (size_t j = 0; j < d; ++j) x[t][j] += pre * streams[t][h * d + j];
                double sum = 0;
                for (size_t k = 0; k < hc; ++k) sum += (comb[t][h * hc + k] = std::exp(coefficients[2 * hc + h * hc + k] * scale[2] + base[2 * hc + h * hc + k]));
                for (size_t k = 0; k < hc; ++k) comb[t][h * hc + k] = static_cast<float>(comb[t][h * hc + k] / sum + 1e-6);
            }
            for (int iteration = 0; iteration < 20; ++iteration) {
                if (iteration > 0) for (size_t h = 0; h < hc; ++h) {
                    double sum = 1e-6;
                    for (size_t k = 0; k < hc; ++k) sum += comb[t][h * hc + k];
                    for (size_t k = 0; k < hc; ++k) comb[t][h * hc + k] = static_cast<float>(comb[t][h * hc + k] / sum);
                }
                for (size_t k = 0; k < hc; ++k) {
                    double sum = 1e-6;
                    for (size_t h = 0; h < hc; ++h) sum += comb[t][h * hc + k];
                    for (size_t h = 0; h < hc; ++h) comb[t][h * hc + k] = static_cast<float>(comb[t][h * hc + k] / sum);
                }
            }
        }
    }
    void mix(Rows& streams, const Rows& output, const Rows& post, const Rows& comb) const {
        const auto hc = static_cast<size_t>(spec_.hc_count), d = static_cast<size_t>(spec_.n_embd);
        for (size_t t = 0; t < streams.size(); ++t) {
            Row next(hc * d);
            for (size_t h = 0; h < hc; ++h) for (size_t j = 0; j < d; ++j) {
                double residual = 0;
                for (size_t k = 0; k < hc; ++k) residual += double{comb[t][k * hc + h]} * streams[t][k * d + j];
                next[h * d + j] = static_cast<float>(post[t][h] * output[t][j] + residual);
            }
            streams[t] = std::move(next);
        }
    }
    const MmapLoader& file_;
    test::TinyModelSpec spec_;
    bool quantized_inputs_ = false;
};
}  // namespace

TEST_CASE("Gated compression matches whole-window softmax, with and without overlap") {
    for (bool overlap : {false, true}) for (int stride : {1, 4, 128}) {
        constexpr int dim = 37;  // SIMD remainder as well as whole lanes
        const CompressionPlan plan{dim, stride, overlap};
        const int width = dim * (overlap ? 2 : 1), count = 3 * stride;
        Rows kv(static_cast<size_t>(count), Row(static_cast<size_t>(width))), gate = kv;
        Row bias(static_cast<size_t>(stride * width));
        for (size_t i = 0; i < bias.size(); ++i) bias[i] = std::sin(static_cast<float>(i)) * 5;
        for (int t = 0; t < count; ++t) for (int j = 0; j < width; ++j) {
            kv[static_cast<size_t>(t)][static_cast<size_t>(j)] = std::cos(static_cast<float>(t * width + j));
            gate[static_cast<size_t>(t)][static_cast<size_t>(j)] = std::sin(static_cast<float>(t + j)) * 1000;
        }
        Row state(plan.state_floats()), scratch(2 * dim), out(dim);
        for (int t = 0; t < count; ++t) {
            const bool closed = plan.step(state, kv[static_cast<size_t>(t)].data(), gate[static_cast<size_t>(t)].data(),
                                          bias.data() + t % stride * width, t, out.data(), scratch);
            CHECK(closed == ((t + 1) % stride == 0));
            if (closed) {
                const Row expected = compress(kv, gate, bias, t + 1, stride, dim, overlap);
                for (size_t i = 0; i < out.size(); ++i) CHECK_NEAR(out[i], expected[i], 2e-6);
            }
        }
    }
}

TEST_CASE("Shared-KV compression and residual streams match independent dense recomputation") {
    for (int hc : {0, 4}) {
        auto spec = test::compressed_spec(); spec.hc_count = hc;
        test::write_tiny_model(path(), spec);
        auto file = MmapLoader::open(path()); REQUIRE(file);
        auto reference_file = MmapLoader::open(path()); REQUIRE(reference_file);
        const Reference reference(*reference_file.value(), spec);
        std::vector<int32_t> tokens(41);
        for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = 3 + static_cast<int32_t>(i * 13 % 256);
        const Rows expected = reference.logits(tokens);
        TransformerOptions options; options.kv_type = KvCacheType::F16;
        auto model = Transformer::load(std::move(file).value(), options); REQUIRE(model);
        ThreadPool pool(4); auto backend = make_cpu_backend(pool); REQUIRE(backend);
        auto check_route = [&](Backend* projections) {
            model.value()->reset();
            const Route route{projections, projections, backend.get()};
            auto got = model.value()->forward(tokens, Transformer::Logits::All, route, pool); REQUIRE(got);
            double maximum = 0;
            for (size_t t = 0; t < expected.size(); ++t) for (size_t j = 0; j < expected[t].size(); ++j) {
                maximum = std::max(maximum, std::fabs(double{(*got)[t * expected[t].size() + j]} - expected[t][j]));
            }
            std::printf("  %s hc=%d max logit error %.9g\n", projections->description().c_str(), hc, maximum);
            CHECK(maximum < 2e-4);
        };
        check_route(backend.get());
        for (auto make : {make_metal_backend, make_vulkan_backend}) {
            auto device = make();
            if (!device) {
                std::printf("  skipped: %s\n", device.status().to_string().c_str());
                continue;
            }
            const TensorView& projection = *reference_file.value()->tensor("blk.1.attn_compressor_kv.weight");
            Row input(static_cast<size_t>(projection.cols()), 0.125f), output(static_cast<size_t>(projection.rows()));
            REQUIRE(device.value()->matmul(projection, input.data(), output.data(), 1).is_ok());
            check_route(device.value().get());
        }
        std::remove(path().c_str());
    }
}

TEST_CASE("Compressed windows survive decode splits, rollback and context images in every KV format") {
    test::write_tiny_model(path(), test::compressed_spec());
    for (const KvCacheType type : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
        TransformerOptions options; options.kv_type = type; options.max_batch = 128;
        auto open = [&] { auto file = MmapLoader::open(path()); return Transformer::load(std::move(file).value(), options); };
        auto batch = open(), steps = open(); REQUIRE(batch); REQUIRE(steps);
        ThreadPool pool(4); auto backend = make_cpu_backend(pool); REQUIRE(backend);
        const Route route{backend.get(), backend.get(), backend.get()};
        std::vector<int32_t> tokens(83);
        for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = 3 + static_cast<int32_t>(i * 19 % 256);
        batch.value()->set_rollback_window(128); steps.value()->set_rollback_window(128);
        auto all = batch.value()->forward(tokens, Transformer::Logits::All, route, pool); REQUIRE(all);
        Row expected(all->begin(), all->end());
        const size_t vocab = static_cast<size_t>(batch.value()->config().n_vocab);
        for (size_t t = 0; t < tokens.size(); ++t) {
            auto got = steps.value()->forward(std::span<const int32_t>(tokens).subspan(t, 1), Transformer::Logits::All, route, pool); REQUIRE(got);
            CHECK(std::equal(got->begin(), got->end(), expected.begin() + static_cast<int64_t>(t * vocab)));
        }
        REQUIRE(steps.value()->truncate(67).is_ok());
        std::stringstream image(std::ios::in | std::ios::out | std::ios::binary);
        REQUIRE(steps.value()->write_state(image).is_ok());
        auto restored = open(); REQUIRE(restored);
        REQUIRE(restored.value()->read_state(image).is_ok());
        auto got = restored.value()->forward(std::span<const int32_t>(tokens).subspan(67), Transformer::Logits::All, route, pool); REQUIRE(got);
        CHECK(std::equal(got->begin(), got->end(), expected.begin() + 67 * static_cast<int64_t>(vocab)));
    }
    std::remove(path().c_str());
}

TEST_CASE("Shared-KV quantized projections match independent dequantized computation") {
    std::vector<test::QuantVector> formats;
    REQUIRE(test::load_quant_vectors(formats));
    for (const DType type : {DType::F32, DType::F16}) {
        test::QuantVector fixture;
        fixture.type = type;
        formats.push_back(std::move(fixture));
    }
    size_t checked = 0;
    std::vector<DType> seen;
    for (const auto& fixture : formats) {
        const DType type = fixture.type;
        REQUIRE(std::find(seen.begin(), seen.end(), type) == seen.end());
        seen.push_back(type);
        auto spec = test::compressed_spec();
        spec.n_embd = 512;
        spec.shared_head_dim = 256;  // attention width 1024 differs from residual width 512
        spec.n_ff = 256;
        spec.mla_q_rank = spec.shared_output_rank = 256;
        spec.index_dim = 128;
        const bool writable = type == DType::F32 || type == DType::F16 || type == DType::BF16 ||
            type == DType::Q8_0 || type == DType::Q4_0 || type == DType::Q4_1 ||
            type == DType::Q4_K || type == DType::Q5_K;
        spec.attn_type = type;
        if (writable) {
            spec.ffn_type = spec.embd_type = spec.output_type = type;
        } else {
            // Existing independent decoder fixtures supply valid payloads for
            // reader-only formats. Keep unrelated FFN/head/embedding weights
            // F32; every attention/compressor/indexer/grouped-output matrix
            // runs in the format being checked.
            spec.attention_block_pattern = fixture.blocks;
            // Decoder stress blocks have order-one weights, unlike this
            // fixture's 0.02-scale projections. Balance attention's weighted
            // norms by the measured block RMS, keeping output/FFN norms and
            // the logit error bound unchanged. Otherwise float summation
            // differences cascade through Q8 rounding into large changes.
            double variance = 0;
            for (float value : fixture.expected) variance += double{value} * value;
            REQUIRE(variance > 0 && std::isfinite(variance));
            spec.attention_norm_scale = spec.weight_scale /
                static_cast<float>(std::sqrt(variance / fixture.expected.size()));
            spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
        }
        test::write_tiny_model(path(), spec);
        auto file = MmapLoader::open(path()), reference_file = MmapLoader::open(path());
        REQUIRE(file); REQUIRE(reference_file);
        const std::vector<int32_t> tokens{3, 23, 47, 67, 89, 101, 123, 149, 157, 181, 199, 211, 229};
        const Rows expected = Reference(*reference_file.value(), spec, !writable).logits(tokens);
        TransformerOptions options; options.kv_type = KvCacheType::F16;
        options.repack_cpu = quant::repack_kernels_available();
        auto model = Transformer::load(std::move(file).value(), options); REQUIRE(model);
        ThreadPool pool(4); auto cpu = make_cpu_backend(pool); const Route route{cpu.get(), cpu.get(), cpu.get()};
        auto actual = model.value()->forward(tokens, Transformer::Logits::All, route, pool); REQUIRE(actual);
        double error = 0;
        for (size_t t = 0; t < expected.size(); ++t) for (size_t j = 0; j < expected[t].size(); ++j) {
            REQUIRE(std::isfinite(expected[t][j]));
            REQUIRE(std::isfinite((*actual)[t * expected[t].size() + j]));
            error = std::max(error, std::fabs(double{(*actual)[t * expected[t].size() + j]} - expected[t][j]));
        }
        std::printf("  shared-KV %s max logit error %.9g\n", std::string(dtype_traits(type).name).c_str(), error);
        CHECK(error < 0.03);
        ++checked;
    }
    CHECK(checked == conformance::weight_formats().size());
    std::remove(path().c_str());
}

TEST_CASE("Shared-KV planning refuses incompatible geometry and engine windows") {
    for (int invalid = 0; invalid < 4; ++invalid) {
        auto spec = test::compressed_spec();
        if (invalid == 0) spec.n_head_kv = 2;
        if (invalid == 1) spec.shared_output_groups = 3;
        if (invalid == 2) spec.sliding_window = 0;
        if (invalid == 3) spec.compress_rope_base = 0;
        test::write_tiny_model(path(), spec);
        auto file = MmapLoader::open(path()); REQUIRE(file);
        auto model = Transformer::load(std::move(file).value(), {});
        CHECK(!model);
        if (!model) CHECK(model.status().code() == ErrorCode::InvalidModel);
    }
    test::write_tiny_model(path(), test::compressed_spec());
    auto file = MmapLoader::open(path()); REQUIRE(file);
    TransformerOptions options; options.sliding_window = 32;
    auto model = Transformer::load(std::move(file).value(), options);
    CHECK(!model);
    if (!model) CHECK(model.status().code() == ErrorCode::Unsupported);
    std::remove(path().c_str());
}

int main(int argc, char** argv) {
    // A reproducible component fixture for an external native-transformers
    // module comparison. It is explicitly not a full-model support gate.
    if (argc > 1) {
        bool original = false, indexer_only = false;
        bool valid = argc >= 3 && argc <= 5 && std::string(argv[1]) == "--export-fixture";
        for (int i = 3; i < argc; ++i) {
            if (std::string(argv[i]) == "--original-geometry") original = true;
            else if (std::string(argv[i]) == "--indexer-only") indexer_only = true;
            else valid = false;
        }
        if (!valid) {
            std::fprintf(stderr, "Usage: test_compression [--export-fixture PREFIX [--original-geometry] [--indexer-only]]\n");
            return 1;
        }
        auto spec = test::compressed_spec();
        spec.compression_strides = {0, 8, 4};
        spec.compression_overlap = {false, false, true};
        int count = 83;
        if (original) {
            spec.n_head = 8;
            spec.n_embd = 4096;  // eight heads of the trained 512 channels
            spec.n_ff = 128;
            spec.mla_q_rank = 1024;
            spec.rope_dim = 64;
            spec.shared_output_rank = 1024;
            spec.shared_output_groups = 8;
            spec.index_dim = 128;
            spec.index_heads = 64;
            spec.sliding_window = 128;
            spec.rms_eps = 1e-6f;
            spec.compress_rope_base = 160000;
            spec.compression_strides = {0, 128, 4};
            spec.yarn_factor = 16;
            spec.yarn_ctx = 65536;
            spec.yarn_beta_fast = 32;
            spec.yarn_beta_slow = 1;
            count = 512;
        }
        if (indexer_only) {
            // Isolate selection before any earlier layer's F16 main cache
            // can perturb its source hidden state. This is a diagnostic,
            // separate from the fixture retaining all attention kinds.
            spec.n_layers = 1;
            spec.compression_strides = {4};
            spec.compression_overlap = {true};
        }
        const std::string prefix = argv[2];
        test::write_tiny_model(prefix + ".gguf", spec);
        auto file = MmapLoader::open(prefix + ".gguf");
        if (!file) { std::fprintf(stderr, "%s\n", file.status().to_string().c_str()); return 1; }
        TransformerOptions options; options.kv_type = KvCacheType::F16; options.max_batch = count;
        auto model = Transformer::load(std::move(file).value(), options);
        if (!model) { std::fprintf(stderr, "%s\n", model.status().to_string().c_str()); return 1; }
        ThreadPool pool(4); auto cpu = make_cpu_backend(pool); const Route route{cpu.get(), cpu.get(), cpu.get()};
        std::vector<int32_t> tokens(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) tokens[static_cast<size_t>(i)] = 3 + i * 19 % 256;
        auto result = model.value()->forward(tokens, Transformer::Logits::All, route, pool);
        if (!result) { std::fprintf(stderr, "%s\n", result.status().to_string().c_str()); return 1; }
        std::ofstream out(prefix + ".json");
        out << std::setprecision(std::numeric_limits<float>::max_digits10);
        out << "{\"kind\":\"shared-kv-components-without-moe-or-sinks\",\"tokens\":[";
        for (size_t i = 0; i < tokens.size(); ++i) out << (i ? "," : "") << tokens[i];
        out << "],\"logits\":[";
        for (size_t i = 0; i < result->size(); ++i) out << (i ? "," : "") << (*result)[i];
        out << "],\"index_positions\":[";
        const auto selections = model.value()->last_index_positions();
        for (size_t t = 0; t < selections.size(); ++t) {
            out << (t ? ",[" : "[");
            for (size_t i = 0; i < selections[t].size(); ++i) out << (i ? "," : "") << selections[t][i];
            out << "]";
        }
        out << "]}\n";
        if (!out) { std::fprintf(stderr, "cannot write component fixture\n"); return 1; }
        return 0;
    }
    return test::run_all();
}
