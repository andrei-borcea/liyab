// Liyab — every supported architecture's forward pass, pinned bit for bit.
//
// Each case writes a small synthetic model that takes one combination of the
// paths the forward pass can follow (architecture, token mixer, dense or MoE
// FFN, optional tensors, RoPE variants, KV cache type, key scaling, sliding
// window, streamed experts), runs a fixed token sequence through it (a batched
// prefill returning every row's logits, then single-token decoding steps) and
// hashes the bits of every logit it returned. The hashes are pinned in
// tests/data/golden_logits.txt per platform, because the kernels a build
// compiles in (scalar, NEON, dot product) are free to round differently
// from one another but never from one run to the next.
//
// Why: the model-support layer moves how architectures are described out of
// the forward pass; this proves the move changes no bit of any architecture's
// output, and keeps proving it for every later change. A change that is meant
// to move outputs rewrites the platform's hashes (LIYAB_GOLDEN_UPDATE=1) and
// says why in its commit. A platform with no pinned hashes is skipped with a
// note, so a new CPU target never fails before its hashes are recorded.
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "core/quant.h"
#include "core/thread_pool.h"
#include "core/transformer.h"
#include "liyab/backend.h"
#include "liyab/mmap_loader.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;

namespace {

// The kernel set the library was compiled with (its flags, not this test's):
// the hashes of one platform hold for every machine that runs those kernels,
// whatever its thread count. The repacked i8mm kernels give the same bits
// (test_engine), so they need no hashes of their own.
std::string platform() {
#if defined(__aarch64__)
    std::string s = "arm64-";
#elif defined(__x86_64__)
    std::string s = "x86_64-";
#else
    std::string s = "other-";
#endif
    s += quant::kernel_isa();
    std::replace(s.begin(), s.end(), '+', '-');
    return s;
}

std::string golden_path() {
    std::string dir = LIYAB_TEST_DATA_DIR;
    if (const char* env = std::getenv("LIYAB_TEST_DATA")) dir = env;
    return dir + "/golden_logits.txt";
}

struct GoldenCase {
    const char* name;
    test::TinyModelSpec spec;
    TransformerOptions options;
    bool scale_keys = false;  // quantized KV cache: per-channel key scales, measured first (calibrate_keys)
};

test::TinyModelSpec moe_spec(const char* arch) {
    test::TinyModelSpec s;
    s.arch = arch;
    s.n_expert = 4;
    s.n_expert_used = 2;
    s.identical_experts = false;
    return s;
}

// One case per path of the forward pass; together they cover every tensor
// and metadata key the loader reads for the supported architectures.
std::vector<GoldenCase> cases() {
    std::vector<GoldenCase> out;
    auto add = [&](const char* name, test::TinyModelSpec spec, TransformerOptions options = {}, bool scale_keys = false) {
        out.push_back({name, std::move(spec), options, scale_keys});
    };
    test::TinyModelSpec s;

    add("llama", {});
    s = {};
    s.tied_output = true;
    s.rope_freqs = true;
    s.rope_scale = 2.0f;
    add("llama-tied-ropefreqs-linear", s);
    TransformerOptions f16_kv;
    f16_kv.kv_type = KvCacheType::F16;
    add("llama-f16kv", {}, f16_kv);
    TransformerOptions window;
    window.sliding_window = 12;
    window.sink_tokens = 2;
    add("llama-sliding", {}, window);
    s = {};
    s.arch = "mistral";
    add("mistral", s);
    s = {};
    s.arch = "qwen2";
    add("qwen2", s);
    s.key_outlier = 60.0f;  // large constant key channels, scaled before the Q8_0 cache
    add("qwen2-keyscaled", s, {}, true);
    s = {};
    s.arch = "qwen3";
    s.qk_norm = true;
    add("qwen3", s);
    s.rope_dim = 16;  // half of the 32-wide heads
    add("qwen3-partial-rope", s);

    s = moe_spec("qwen3moe");
    s.qk_norm = true;
    s.expert_weights_norm = -1;  // the architecture's default
    add("qwen3moe", s);
    TransformerOptions streamed;
    streamed.expert_cache_bytes = int64_t{1} << 20;  // experts read through the ExpertStore
    add("qwen3moe-streamed", s, streamed);
    s = moe_spec("qwen3moe");
    s.sigmoid_gating = true;
    s.expert_bias = true;
    s.expert_weights_norm = 0;
    s.n_ff_shared = 96;
    s.shared_gate = true;
    add("qwen3moe-sigmoid-bias-shared", s);

    s = {};
    s.arch = "qwen35";
    s.n_layers = 4;
    s.delta_net_interval = 2;
    s.qk_norm = true;
    s.attn_gate = true;
    add("qwen35", s);
    s = moe_spec("qwen35moe");
    s.n_layers = 4;
    s.delta_net_interval = 2;
    s.qk_norm = true;
    s.attn_gate = true;
    s.n_ff_shared = 96;
    s.shared_gate = true;
    add("qwen35moe", s);

    s = {};
    s.arch = "hunyuan-dense";  // Q/K norms after RoPE
    s.qk_norm = true;
    add("hunyuan-dense", s);
    s.k_norm_outlier = 60.0f;  // large key channels from the norm, measured through it (key scaling)
    add("hunyuan-dense-keyscaled", s, {}, true);

    s = {};
    s.arch = "gemma3";  // scaled embeddings, GeGLU, norms after the attention and the FFN
    s.n_layers = 6;     // five windowed blocks (RoPE base 10000), then a global one
    s.sliding_window = 8;
    s.rope_base = 1e6f;
    s.qk_norm = true;
    s.post_norms = true;
    s.tied_output = true;
    add("gemma3", s);
    TransformerOptions engine_window;  // the engine's own window and sinks over the model's windowed blocks
    engine_window.sliding_window = 12;
    engine_window.sink_tokens = 2;
    add("gemma3-sliding", s, engine_window);
    s.k_norm_outlier = 60.0f;  // key scaling with two RoPE tables
    add("gemma3-keyscaled", s, {}, true);
    return out;
}

// FNV-1a over the bits of every float.
void hash_floats(uint64_t& h, std::span<const float> values) {
    for (const float v : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof bits);
        for (int i = 0; i < 4; ++i) {
            h ^= (bits >> (8 * i)) & 0xffu;
            h *= 0x100000001b3ull;
        }
    }
}

// Runs the case's model over a fixed sequence: 20 tokens prefilled in one
// batch with every row's logits, then 6 decoding steps. Returns false (and
// reports why) when the model does not load or a pass fails.
bool run_case(const GoldenCase& c, uint64_t& hash) {
    const std::string path = test::temp_dir() + "/liyab_" + std::to_string(getpid()) + "_golden.gguf";
    test::write_tiny_model(path, c.spec);
    // Removed once the model is gone: streamed experts are read through the path.
    struct Remove {
        const std::string& path;
        ~Remove() { std::remove(path.c_str()); }
    } remove{path};
    auto file = MmapLoader::open(path);
    if (!file) {
        std::printf("  %s: %s\n", c.name, file.status().to_string().c_str());
        return false;
    }
    TransformerOptions options = c.options;
    options.context_length = 64;
    options.max_batch = 32;
    auto loaded = Transformer::load(std::move(file).value(), options);
    if (!loaded) {
        std::printf("  %s: %s\n", c.name, loaded.status().to_string().c_str());
        return false;
    }
    auto model = std::move(loaded).value();
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};

    const int32_t vocab = model->config().n_vocab;
    if (c.scale_keys) {
        std::vector<int32_t> calibration = {1};
        for (int32_t i = 1; i < 48; ++i) calibration.push_back(3 + (i * 29) % (vocab - 3));
        if (Status s = model->calibrate_keys(calibration, route, pool); !s.is_ok() || model->key_scales().empty()) {
            std::printf("  %s: key calibration failed: %s\n", c.name, s.to_string().c_str());
            return false;
        }
    }
    std::vector<int32_t> tokens = {1};  // BOS
    for (int32_t i = 1; i < 26; ++i) tokens.push_back(3 + (i * 37) % (vocab - 3));
    hash = 0xcbf29ce484222325ull;
    auto prefill = model->forward(std::span(tokens).first(20), Transformer::Logits::All, route, pool);
    if (!prefill) {
        std::printf("  %s: %s\n", c.name, prefill.status().to_string().c_str());
        return false;
    }
    hash_floats(hash, *prefill);
    for (size_t t = 20; t < tokens.size(); ++t) {
        auto step = model->forward(std::span(tokens).subspan(t, 1), Transformer::Logits::Last, route, pool);
        if (!step) {
            std::printf("  %s: %s\n", c.name, step.status().to_string().c_str());
            return false;
        }
        hash_floats(hash, *step);
    }
    return true;
}

std::string hex(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

}  // namespace

TEST_CASE("Every architecture's logits match the hashes pinned for this platform (tests/data/golden_logits.txt)") {
    const std::string me = platform();
    // platform -> case -> hash, in file order for rewriting.
    std::map<std::string, std::map<std::string, std::string>> pinned;
    std::ifstream in(golden_path());
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        std::string plat, name, hash;
        if (fields >> plat >> name >> hash) pinned[plat][name] = hash;
    }

    std::map<std::string, std::string> computed;
    bool all_ran = true;
    for (const GoldenCase& c : cases()) {
        uint64_t h = 0;
        const bool ran = run_case(c, h);
        CHECK(ran);
        all_ran = all_ran && ran;
        computed[c.name] = hex(h);
    }

    const char* update = std::getenv("LIYAB_GOLDEN_UPDATE");
    if (update != nullptr && std::strcmp(update, "1") == 0) {
        REQUIRE(all_ran);  // never pin the hash of a pass that failed
        pinned[me] = computed;
        std::ofstream out(golden_path());
        out << "# Logit hashes pinned by tests/test_golden.cpp: <platform> <case> <FNV-1a of the logits' bits>.\n"
               "# Rewritten with LIYAB_GOLDEN_UPDATE=1 when a change is meant to move outputs (say why in the commit).\n";
        for (const auto& [plat, by_case] : pinned) {
            for (const auto& [name, hash] : by_case) out << plat << ' ' << name << ' ' << hash << '\n';
        }
        CHECK(static_cast<bool>(out));
        std::printf("  pinned %zu hashes for %s in %s\n", computed.size(), me.c_str(), golden_path().c_str());
        return;
    }
    const auto mine = pinned.find(me);
    if (mine == pinned.end()) {
        std::printf("  skipped: no hashes pinned for %s in %s (record them with LIYAB_GOLDEN_UPDATE=1)\n", me.c_str(),
                    golden_path().c_str());
        return;
    }
    for (const auto& [name, hash] : computed) {
        const auto it = mine->second.find(name);
        if (it == mine->second.end()) {
            std::printf("  %s: not pinned for %s (record it with LIYAB_GOLDEN_UPDATE=1)\n", name.c_str(), me.c_str());
            CHECK(false);
        } else if (it->second != hash) {
            std::printf("  %s: logits changed on %s: %s, pinned %s\n", name.c_str(), me.c_str(), hash.c_str(),
                        it->second.c_str());
            CHECK(false);
        }
    }
}

TEST_MAIN()
