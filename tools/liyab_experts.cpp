// liyab-experts — measurement helpers for the routed experts of a MoE model,
// used by the cold-expert compression study (docs/research/cold-experts.md,
// driven by tools/expert_study.py). Nothing here changes how the engine runs.
//
//   liyab-experts capture -m model.gguf -f text.txt -o DIR [--max-tokens N] [--threads N]
//                         [--sample-layers 0,10,20,30,39] [--samples K]
//       Runs the text through the model one token at a time (each pass is a
//       decoding step, as on the phone) and records, through a Backend that
//       wraps the CPU backend and watches the matmuls the forward pass asks
//       for (the engine itself is not modified):
//         meta.txt          layers, experts, top-k, widths, tokens
//         counts.u32        [layer][expert] tokens routed to each expert
//         trace_ids.u16     [token][layer][top-k] experts chosen, by router rank
//         trace_w.f32       [token][layer][top-k] their normalized router weights
//         mom_in.f64        [layer][expert][n_embd] sum of x_j^2 of the expert's input
//         mom_mid.f64       [layer][expert][n_ff]   same for the down projection's input
//         samples_in_L<l>.f32 / samples_mid_L<l>.f32  [expert][K][width]: the first K
//                           inputs each expert of block l saw (zero-padded),
//                           samples_n_L<l>.u32 [expert]: how many were stored
//       The routing is read from the router's own matmul (softmax, top-k,
//       renormalized as the model does) and checked against the experts the
//       forward pass then multiplies; mismatches are reported.
//
//   liyab-experts to-q8 -m model.gguf -o out.gguf [--threads N]
//       Copies the GGUF with every routed-expert tensor (*_exps.weight)
//       converted to Q8_0 by the engine's own dequantizer and quantizer; all
//       other tensors and the metadata are copied byte for byte. The study
//       then overwrites chosen experts of a copy of this file with their
//       compressed reconstruction: Q8_0 adds ~1/16 of a 4-bit format's
//       rounding error, so what liyab-kl measures is the representation
//       under test, not a second 4-bit rounding.
//
//   liyab-experts dequant -m model.gguf -t TENSOR [--experts FIRST:COUNT] -o out.f32
//       Dequantizes a (stacked) tensor, or a range of its experts, to float32
//       rows with the engine's dequantizer.
//
//   liyab-experts dequant-raw --type GGML_TYPE --cols N -i blocks.bin -o out.f32
//       Dequantizes raw blocks of any type the engine reads: checks that an
//       encoder written elsewhere (e.g. tools/expert_study.py's IQ3_XXS
//       encoder) produces blocks the engine decodes as intended.
//
// Uses internal headers: built with the tests (LIYAB_BUILD_TESTS). Loads the
// model resident (no expert streaming), so experts are read in place from the
// mapping and every expert matmul can be traced back to its (block, expert).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/quant.h"
#include "core/thread_pool.h"
#include "core/tokenizer.h"
#include "core/transformer.h"
#include "liyab/backend.h"
#include "liyab/mmap_loader.h"

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: liyab-experts capture -m model.gguf -f text.txt -o DIR [--max-tokens N] [--threads N]\n"
                 "                             [--sample-layers L,L,...] [--samples K]\n"
                 "       liyab-experts to-q8 -m model.gguf -o out.gguf [--threads N]\n"
                 "       liyab-experts dequant -m model.gguf -t TENSOR [--experts FIRST:COUNT] -o out.f32\n"
                 "       liyab-experts dequant-raw --type GGML_TYPE --cols N -i blocks.bin -o out.f32\n");
    return 2;
}

template <typename T>
bool write_file(const std::string& path, const std::vector<T>& v) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    return static_cast<bool>(out);
}

// ---------------------------------------------------------------------------
// capture

enum Matrix { kGate = 0, kUp = 1, kDown = 2 };

// Where an expert matmul's weights live: block, matrix and the stacked tensor.
struct Stacked {
    int32_t layer = 0;
    int32_t matrix = 0;
    const uint8_t* data = nullptr;
    size_t expert_bytes = 0;
    size_t total_bytes = 0;
};

class CaptureBackend final : public liyab::Backend {
public:
    CaptureBackend(liyab::Backend& inner, const liyab::MmapLoader& file, int32_t n_layers, int32_t n_expert,
                   int32_t top_k, int32_t n_embd, int32_t n_ff, std::vector<int32_t> sample_layers, int32_t samples)
        : inner_(inner), L_(n_layers), E_(n_expert), K_(top_k), d_(n_embd), ff_(n_ff),
          sample_layers_(std::move(sample_layers)), samples_(samples) {
        static const char* kNames[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
        for (int32_t l = 0; l < L_; ++l) {
            for (int32_t m = 0; m < 3; ++m) {
                const std::string name = "blk." + std::to_string(l) + "." + kNames[m] + ".weight";
                if (const liyab::TensorView* t = file.tensor(name)) {
                    stacked_[t->data] = {l, m, t->data, t->nbytes / static_cast<size_t>(E_), t->nbytes};
                }
            }
            const std::string router = "blk." + std::to_string(l) + ".ffn_gate_inp.weight";
            if (const liyab::TensorView* t = file.tensor(router)) routers_[t->data] = l;
        }
        counts_.assign(static_cast<size_t>(L_) * E_, 0);
        mom_in_.assign(static_cast<size_t>(L_) * E_ * d_, 0.0);
        mom_mid_.assign(static_cast<size_t>(L_) * E_ * ff_, 0.0);
        for (const int32_t l : sample_layers_) {
            Samples& s = samples_by_layer_[l];
            s.n.assign(static_cast<size_t>(E_), 0);
            s.n_mid.assign(static_cast<size_t>(E_), 0);
            s.in.assign(static_cast<size_t>(E_) * samples_ * d_, 0.0f);
            s.mid.assign(static_cast<size_t>(E_) * samples_ * ff_, 0.0f);
        }
    }

    // Starts a token: its router decisions and observed experts are recorded per block.
    void begin_token() {
        ids_.resize(ids_.size() + static_cast<size_t>(L_) * K_, 0xFFFF);
        weights_.resize(weights_.size() + static_cast<size_t>(L_) * K_, 0.0f);
        seen_.assign(static_cast<size_t>(L_), {});
    }
    // Compares the router's top-k with the experts the forward pass multiplied.
    void end_token() {
        const size_t base = ids_.size() - static_cast<size_t>(L_) * K_;
        for (int32_t l = 0; l < L_; ++l) {
            std::vector<int32_t> routed;
            for (int32_t k = 0; k < K_; ++k) {
                const uint16_t id = ids_[base + static_cast<size_t>(l) * K_ + k];
                if (id != 0xFFFF) routed.push_back(id);
            }
            if (routed.empty()) continue;  // a dense block
            std::sort(routed.begin(), routed.end());
            std::vector<int32_t>& seen = seen_[static_cast<size_t>(l)];
            std::sort(seen.begin(), seen.end());
            if (routed != seen) ++mismatches_;
        }
        ++tokens_;
    }

    [[nodiscard]] liyab::BackendKind kind() const noexcept override { return inner_.kind(); }
    [[nodiscard]] std::string description() const override { return "capture(" + inner_.description() + ")"; }
    [[nodiscard]] bool supports(liyab::DType t) const noexcept override { return inner_.supports(t); }
    uint8_t* allocate_shared(size_t bytes) override { return inner_.allocate_shared(bytes); }
    void free_shared(uint8_t* data) noexcept override { inner_.free_shared(data); }
    [[nodiscard]] double busy_ms() const noexcept override { return inner_.busy_ms(); }

    liyab::Status matmul(const liyab::TensorView& w, const float* x, float* y, int32_t n) override {
        LIYAB_RETURN_IF_ERROR(inner_.matmul(w, x, y, n));
        if (auto it = routers_.find(w.data); it != routers_.end() && n == 1) record_router(it->second, y);
        else note_expert(w, x, n);
        return liyab::Status::ok();
    }
    liyab::Status matmul_group(std::span<const liyab::TensorView* const> ws, const float* x,
                               std::span<float* const> ys, int32_t n) override {
        for (const liyab::TensorView* w : ws) note_expert(*w, x, n);
        return inner_.matmul_group(ws, x, ys, n);
    }
    liyab::Status matmul_batch(std::span<const MatmulItem> items) override {
        for (const MatmulItem& it : items) note_expert(*it.w, it.x, it.n);
        return inner_.matmul_batch(items);
    }

    bool write(const std::string& dir) const {
        bool ok = write_file(dir + "/counts.u32", counts_) && write_file(dir + "/trace_ids.u16", ids_) &&
                  write_file(dir + "/trace_w.f32", weights_) && write_file(dir + "/mom_in.f64", mom_in_) &&
                  write_file(dir + "/mom_mid.f64", mom_mid_);
        for (const auto& [l, s] : samples_by_layer_) {
            const std::string suffix = "_L" + std::to_string(l);
            ok = ok && write_file(dir + "/samples_in" + suffix + ".f32", s.in) &&
                 write_file(dir + "/samples_mid" + suffix + ".f32", s.mid) &&
                 write_file(dir + "/samples_n" + suffix + ".u32", s.n) &&
                 write_file(dir + "/samples_nmid" + suffix + ".u32", s.n_mid);
        }
        std::ofstream meta(dir + "/meta.txt");
        meta << "layers " << L_ << "\nexperts " << E_ << "\ntop_k " << K_ << "\nn_embd " << d_ << "\nn_ff " << ff_
             << "\ntokens " << tokens_ << "\nsamples " << samples_ << "\nsample_layers";
        for (const int32_t l : sample_layers_) meta << ' ' << l;
        meta << "\nrouting_mismatches " << mismatches_ << "\n";
        return ok && static_cast<bool>(meta);
    }
    [[nodiscard]] uint64_t mismatches() const noexcept { return mismatches_; }

private:
    struct Samples {
        std::vector<uint32_t> n, n_mid;
        std::vector<float> in, mid;
    };

    // Qwen3.6's routing: softmax over the experts, top-k, weights renormalized
    // over the top-k (the engine's moe_ffn with moe_norm_weights).
    void record_router(int32_t layer, const float* logits) {
        std::vector<float> p(logits, logits + E_);
        const float mx = *std::max_element(p.begin(), p.end());
        double sum = 0;
        for (float& v : p) sum += v = std::exp(v - mx);
        for (float& v : p) v = static_cast<float>(v / sum);
        std::vector<int32_t> order(static_cast<size_t>(E_));
        for (int32_t e = 0; e < E_; ++e) order[static_cast<size_t>(e)] = e;
        std::partial_sort(order.begin(), order.begin() + K_, order.end(), [&](int32_t a, int32_t b) {
            return p[static_cast<size_t>(a)] > p[static_cast<size_t>(b)] ||
                   (p[static_cast<size_t>(a)] == p[static_cast<size_t>(b)] && a < b);
        });
        float top = 0;
        for (int32_t k = 0; k < K_; ++k) top += p[static_cast<size_t>(order[static_cast<size_t>(k)])];
        const size_t base = ids_.size() - static_cast<size_t>(L_) * K_ + static_cast<size_t>(layer) * K_;
        for (int32_t k = 0; k < K_; ++k) {
            const int32_t e = order[static_cast<size_t>(k)];
            ids_[base + k] = static_cast<uint16_t>(e);
            weights_[base + k] = p[static_cast<size_t>(e)] / top;
        }
    }

    void note_expert(const liyab::TensorView& w, const float* x, int32_t n) {
        auto it = stacked_.upper_bound(w.data);
        if (it == stacked_.begin()) return;
        --it;
        const Stacked& s = it->second;
        if (w.data >= s.data + s.total_bytes) return;
        const auto e = static_cast<int32_t>(static_cast<size_t>(w.data - s.data) / s.expert_bytes);
        if (s.matrix == kUp) return;  // same input as gate
        const size_t key = static_cast<size_t>(s.layer) * E_ + static_cast<size_t>(e);
        const int32_t width = s.matrix == kGate ? d_ : ff_;
        double* mom = (s.matrix == kGate ? mom_in_.data() + key * d_ : mom_mid_.data() + key * ff_);
        auto sit = samples_by_layer_.find(s.layer);
        for (int32_t r = 0; r < n; ++r) {
            const float* row = x + static_cast<size_t>(r) * width;
            for (int32_t j = 0; j < width; ++j) mom[j] += static_cast<double>(row[j]) * row[j];
            if (sit != samples_by_layer_.end()) {
                Samples& smp = sit->second;
                uint32_t& count = s.matrix == kGate ? smp.n[static_cast<size_t>(e)] : smp.n_mid[static_cast<size_t>(e)];
                if (count < static_cast<uint32_t>(samples_)) {
                    float* dst = (s.matrix == kGate ? smp.in.data() : smp.mid.data()) +
                                 (static_cast<size_t>(e) * samples_ + count) * width;
                    std::copy_n(row, width, dst);
                    ++count;
                }
            }
        }
        if (s.matrix == kGate) {
            counts_[key] += static_cast<uint32_t>(n);
            if (!seen_.empty()) seen_[static_cast<size_t>(s.layer)].push_back(e);
        }
    }

    liyab::Backend& inner_;
    int32_t L_, E_, K_, d_, ff_;
    std::vector<int32_t> sample_layers_;
    int32_t samples_;
    std::map<const uint8_t*, Stacked> stacked_;
    std::map<const uint8_t*, int32_t> routers_;
    std::vector<uint32_t> counts_;
    std::vector<uint16_t> ids_;
    std::vector<float> weights_;
    std::vector<double> mom_in_, mom_mid_;
    std::map<int32_t, Samples> samples_by_layer_;
    std::vector<std::vector<int32_t>> seen_;
    uint64_t tokens_ = 0;
    uint64_t mismatches_ = 0;
};

int capture(int argc, char** argv) {
    std::string model_path, text_path, out_dir;
    int32_t max_tokens = 8192, threads = 0, samples = 64;
    std::vector<int32_t> sample_layers = {0, 10, 20, 30, 39};
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "-m") model_path = next();
        else if (arg == "-f") text_path = next();
        else if (arg == "-o") out_dir = next();
        else if (arg == "--max-tokens") max_tokens = std::atoi(next());
        else if (arg == "--threads") threads = std::atoi(next());
        else if (arg == "--samples") samples = std::atoi(next());
        else if (arg == "--sample-layers") {
            sample_layers.clear();
            std::stringstream ss(next());
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) sample_layers.push_back(std::atoi(item.c_str()));
            }
        } else return usage();
    }
    if (model_path.empty() || text_path.empty() || out_dir.empty()) return usage();
    std::ifstream text_file(text_path);
    std::stringstream text;
    text << text_file.rdbuf();

    liyab::LoaderOptions loader_options;
    loader_options.streaming = false;  // experts in place: their matmuls point into the mapping
    auto file = liyab::MmapLoader::open(model_path, loader_options);
    if (!file) {
        std::fprintf(stderr, "%s\n", file.status().to_string().c_str());
        return 1;
    }
    liyab::TransformerOptions options;
    options.context_length = max_tokens + 16;
    options.max_batch = 1;
    options.expert_cache_bytes = 0;  // no ExpertStore: experts read in place
    auto loaded = liyab::Transformer::load(std::move(file).value(), options);
    if (!loaded) {
        std::fprintf(stderr, "%s\n", loaded.status().to_string().c_str());
        return 1;
    }
    auto model = std::move(loaded).value();
    if (model->expert_store() != nullptr) {
        std::fprintf(stderr, "experts are streamed: capture needs them in place\n");
        return 1;
    }
    const liyab::ModelConfig& c = model->config();
    auto tokenizer = liyab::Tokenizer::load(model->file());
    if (!tokenizer) {
        std::fprintf(stderr, "%s\n", tokenizer.status().to_string().c_str());
        return 1;
    }
    auto encoded = tokenizer->encode(text.str(), tokenizer->add_bos_default());
    if (!encoded || encoded->empty()) {
        std::fprintf(stderr, "cannot encode %s\n", text_path.c_str());
        return 1;
    }
    std::vector<int32_t> tokens = std::move(encoded).value();
    if (tokens.size() > static_cast<size_t>(max_tokens)) tokens.resize(static_cast<size_t>(max_tokens));

    liyab::ThreadPool pool(threads);
    auto cpu = liyab::make_cpu_backend(pool);
    CaptureBackend capture(*cpu, model->file(), c.n_layers, c.n_expert, c.n_expert_used, c.n_embd, c.n_ff_expert,
                           sample_layers, samples);
    const liyab::Route route{&capture, &capture, &capture};
    for (size_t t = 0; t < tokens.size(); ++t) {
        capture.begin_token();
        auto r = model->forward(std::span<const int32_t>(tokens.data() + t, 1), liyab::Transformer::Logits::None,
                                route, pool);
        if (!r) {
            std::fprintf(stderr, "%s\n", r.status().to_string().c_str());
            return 1;
        }
        capture.end_token();
        if (t % 64 == 0 || t + 1 == tokens.size()) std::fprintf(stderr, "\r%zu / %zu tokens", t + 1, tokens.size());
    }
    std::fprintf(stderr, "\n");
    if (!capture.write(out_dir)) {
        std::fprintf(stderr, "cannot write to %s\n", out_dir.c_str());
        return 1;
    }
    std::printf("captured %zu tokens to %s; routing mismatches: %llu\n", tokens.size(), out_dir.c_str(),
                static_cast<unsigned long long>(capture.mismatches()));
    return 0;
}

// ---------------------------------------------------------------------------
// GGUF copy with the experts in Q8_0

// Minimal GGUF header reader: enough to copy the metadata verbatim and
// rewrite the tensor table.
struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    template <typename T>
    T get() {
        T v{};
        if (p + sizeof(T) > end) throw std::runtime_error("truncated GGUF header");
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const auto n = get<uint64_t>();
        if (p + n > end) throw std::runtime_error("truncated GGUF string");
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
    void skip_value(uint32_t type) {
        static const size_t kSize[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (type == 8) {
            str();
        } else if (type == 9) {
            const auto elem = get<uint32_t>();
            const auto n = get<uint64_t>();
            for (uint64_t i = 0; i < n; ++i) skip_value(elem);
        } else if (type < 13) {
            p += kSize[type];
        } else {
            throw std::runtime_error("unknown GGUF value type");
        }
    }
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> ne;
    uint32_t type = 0;
    uint64_t offset = 0;
};

void put_bytes(std::ofstream& out, const void* p, size_t n) { out.write(static_cast<const char*>(p), static_cast<std::streamsize>(n)); }
template <typename T>
void put(std::ofstream& out, T v) { put_bytes(out, &v, sizeof v); }

int to_q8(int argc, char** argv) {
    std::string model_path, out_path;
    int32_t threads = static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency()));
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "-m") model_path = next();
        else if (arg == "-o") out_path = next();
        else if (arg == "--threads") threads = std::max(1, std::atoi(next()));
        else return usage();
    }
    if (model_path.empty() || out_path.empty()) return usage();
    liyab::LoaderOptions loader_options;
    loader_options.streaming = false;
    auto opened = liyab::MmapLoader::open(model_path, loader_options);
    if (!opened) {
        std::fprintf(stderr, "%s\n", opened.status().to_string().c_str());
        return 1;
    }
    const liyab::MmapLoader& file = **opened;
    if (file.shard_count() != 1) {
        std::fprintf(stderr, "split models are not supported\n");
        return 1;
    }
    const uint8_t* base = file.file().data();
    Reader r{base, base + file.file().size()};
    uint32_t alignment = 32;
    std::vector<TensorInfo> infos;
    size_t kv_begin = 0, kv_end = 0;
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    try {
        if (std::memcmp(r.p, "GGUF", 4) != 0) throw std::runtime_error("not a GGUF file");
        r.p += 4;
        version = r.get<uint32_t>();
        n_tensors = r.get<uint64_t>();
        n_kv = r.get<uint64_t>();
        kv_begin = static_cast<size_t>(r.p - base);
        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = r.str();
            const auto type = r.get<uint32_t>();
            if (key == "general.alignment" && type == 4) {
                alignment = r.get<uint32_t>();
            } else {
                r.skip_value(type);
            }
        }
        kv_end = static_cast<size_t>(r.p - base);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            t.name = r.str();
            const auto dims = r.get<uint32_t>();
            for (uint32_t d = 0; d < dims; ++d) t.ne.push_back(r.get<uint64_t>());
            t.type = r.get<uint32_t>();
            t.offset = r.get<uint64_t>();
            infos.push_back(std::move(t));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    const size_t header_end = static_cast<size_t>(r.p - base);
    const size_t data_begin = (header_end + alignment - 1) / alignment * alignment;
    (void)data_begin;

    auto is_expert = [](const std::string& name) { return name.find("_exps.weight") != std::string::npos; };
    // New tensor table: experts become Q8_0, offsets are recomputed in order.
    std::vector<uint32_t> new_type(infos.size());
    std::vector<uint64_t> new_offset(infos.size()), new_bytes(infos.size());
    uint64_t offset = 0;
    for (size_t i = 0; i < infos.size(); ++i) {
        const TensorInfo& t = infos[i];
        const liyab::TensorView* v = file.tensor(t.name);
        if (v == nullptr) {
            std::fprintf(stderr, "tensor %s not found by the loader\n", t.name.c_str());
            return 1;
        }
        new_type[i] = is_expert(t.name) ? static_cast<uint32_t>(liyab::DType::Q8_0) : t.type;
        new_bytes[i] = is_expert(t.name) ? liyab::dtype_row_bytes(liyab::DType::Q8_0, v->cols()) *
                                               static_cast<size_t>(v->rows())
                                         : v->nbytes;
        new_offset[i] = offset;
        offset += (new_bytes[i] + alignment - 1) / alignment * alignment;
    }

    std::ofstream out(out_path, std::ios::binary);
    out.write("GGUF", 4);
    put(out, version);
    put(out, n_tensors);
    put(out, n_kv);
    put_bytes(out, base + kv_begin, kv_end - kv_begin);
    for (size_t i = 0; i < infos.size(); ++i) {
        const TensorInfo& t = infos[i];
        put(out, static_cast<uint64_t>(t.name.size()));
        put_bytes(out, t.name.data(), t.name.size());
        put(out, static_cast<uint32_t>(t.ne.size()));
        for (const uint64_t d : t.ne) put(out, d);
        put(out, new_type[i]);
        put(out, new_offset[i]);
    }
    const std::vector<char> zeros(alignment, 0);
    auto pad = [&](size_t n) { put_bytes(out, zeros.data(), (alignment - n % alignment) % alignment); };
    pad(static_cast<size_t>(out.tellp()));

    for (size_t i = 0; i < infos.size(); ++i) {
        const liyab::TensorView* v = file.tensor(infos[i].name);
        if (!is_expert(infos[i].name)) {
            put_bytes(out, v->data, v->nbytes);
        } else {
            const int64_t rows = v->rows(), cols = v->cols();
            const size_t out_row = liyab::dtype_row_bytes(liyab::DType::Q8_0, cols);
            std::vector<uint8_t> converted(out_row * static_cast<size_t>(rows));
            std::atomic<int64_t> next{0};
            auto work = [&] {
                std::vector<float> f(static_cast<size_t>(cols));
                for (int64_t row; (row = next.fetch_add(1)) < rows;) {
                    liyab::quant::dequantize_row(v->type, v->row(row), f.data(), cols);
                    liyab::quant::quantize_row_q8_0(
                        f.data(), reinterpret_cast<liyab::quant::BlockQ8_0*>(converted.data() + row * out_row), cols);
                }
            };
            std::vector<std::thread> pool;
            for (int32_t k = 0; k < threads; ++k) pool.emplace_back(work);
            for (auto& th : pool) th.join();
            put_bytes(out, converted.data(), converted.size());
        }
        pad(new_bytes[i]);
        std::fprintf(stderr, "\r%zu / %zu tensors", i + 1, infos.size());
    }
    std::fprintf(stderr, "\n");
    if (!out) {
        std::fprintf(stderr, "write failed: %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%.2f GB)\n", out_path.c_str(), static_cast<double>(out.tellp()) / 1e9);
    return 0;
}

// ---------------------------------------------------------------------------
// dequant / dequant-raw

int dequant(int argc, char** argv) {
    std::string model_path, tensor_name, out_path;
    int64_t first = 0, count = -1;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "-m") model_path = next();
        else if (arg == "-t") tensor_name = next();
        else if (arg == "-o") out_path = next();
        else if (arg == "--experts") {
            const std::string v = next();
            const size_t colon = v.find(':');
            if (colon == std::string::npos) return usage();
            first = std::atoll(v.substr(0, colon).c_str());
            count = std::atoll(v.substr(colon + 1).c_str());
        } else return usage();
    }
    if (model_path.empty() || tensor_name.empty() || out_path.empty()) return usage();
    liyab::LoaderOptions loader_options;
    loader_options.streaming = false;
    auto opened = liyab::MmapLoader::open(model_path, loader_options);
    if (!opened) {
        std::fprintf(stderr, "%s\n", opened.status().to_string().c_str());
        return 1;
    }
    const liyab::TensorView* v = (*opened)->tensor(tensor_name);
    if (v == nullptr) {
        std::fprintf(stderr, "no tensor %s\n", tensor_name.c_str());
        return 1;
    }
    const int64_t per_expert = v->ne[1];  // rows of one expert in a stacked [cols, rows, experts] tensor
    const int64_t experts = v->ne[2];
    if (count < 0) count = experts - first;
    if (first < 0 || first + count > experts) return usage();
    std::ofstream out(out_path, std::ios::binary);
    std::vector<float> f(static_cast<size_t>(v->cols()));
    for (int64_t row = first * per_expert; row < (first + count) * per_expert; ++row) {
        liyab::quant::dequantize_row(v->type, v->row(row), f.data(), v->cols());
        out.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size() * sizeof(float)));
    }
    std::printf("%s: type %u, %lld x %lld per expert, experts %lld..%lld -> %s\n", tensor_name.c_str(),
                static_cast<unsigned>(v->type), static_cast<long long>(per_expert), static_cast<long long>(v->cols()),
                static_cast<long long>(first), static_cast<long long>(first + count - 1), out_path.c_str());
    return out ? 0 : 1;
}

int dequant_raw(int argc, char** argv) {
    std::string in_path, out_path;
    int32_t type = -1;
    int64_t cols = 0;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "--type") type = std::atoi(next());
        else if (arg == "--cols") cols = std::atoll(next());
        else if (arg == "-i") in_path = next();
        else if (arg == "-o") out_path = next();
        else return usage();
    }
    if (type < 0 || cols <= 0 || in_path.empty() || out_path.empty()) return usage();
    const auto dtype = static_cast<liyab::DType>(type);
    const size_t row_bytes = liyab::dtype_row_bytes(dtype, cols);
    std::ifstream in(in_path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (row_bytes == 0 || bytes.size() % row_bytes != 0) {
        std::fprintf(stderr, "%zu bytes are not whole rows of %zu bytes\n", bytes.size(), row_bytes);
        return 1;
    }
    const size_t rows = bytes.size() / row_bytes;
    std::vector<float> f(rows * static_cast<size_t>(cols));
    for (size_t r = 0; r < rows; ++r) {
        liyab::quant::dequantize_row(dtype, bytes.data() + r * row_bytes, f.data() + r * cols, cols);
    }
    return write_file(out_path, f) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    if (cmd == "capture") return capture(argc, argv);
    if (cmd == "to-q8") return to_q8(argc, argv);
    if (cmd == "dequant") return dequant(argc, argv);
    if (cmd == "dequant-raw") return dequant_raw(argc, argv);
    return usage();
}
