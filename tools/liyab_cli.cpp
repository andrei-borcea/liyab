// liyab-cli — command-line front end over the C ABI (the same surface that
// JNI and Swift use). Useful for on-device testing over `adb shell`.
//
//   liyab-cli --device
//   liyab-cli -m model.gguf -p "Once upon a time" [-n 128] [--temp 0.8]
//             [--draft draft.gguf] [--profile performance|balanced|low_power]
//             [--backend auto|cpu|metal|vulkan] [--kv f16|q8_0|q4_0|q4_1] [--ctx N]
//             [--window N] [--threads N] [--seed N]
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "liyab/liyab_c_api.h"

namespace {

liyab_engine* g_engine = nullptr;

void on_sigint(int) {
    if (g_engine != nullptr) liyab_engine_cancel(g_engine);  // async-signal-safe: sets an atomic flag
}

int32_t print_piece(const char* piece, size_t len, int32_t, void*) {
    std::fwrite(piece, 1, len, stdout);
    std::fflush(stdout);
    return 1;
}

void usage() {
    std::fprintf(stderr,
                 "usage: liyab-cli --device\n"
                 "       liyab-cli -m MODEL.gguf -p PROMPT [options]\n"
                 "options:\n"
                 "  --tokenize         print the prompt's token ids and exit\n"
                 "  -n N               max new tokens (default 128)\n"
                 "  --temp T           temperature, 0 = greedy (default 0.8)\n"
                 "  --top-k K          (default 40)    --top-p P (default 0.95)\n"
                 "  --seed N           RNG seed (default random)\n"
                 "  --draft PATH       draft model for speculative decoding\n"
                 "  --draft-tokens K   tokens per speculative step (default 4)\n"
                 "  --profile P        performance | balanced | low_power (default balanced)\n"
                 "  --tps R            override the paced token rate\n"
                 "  --skin-threshold C thermal throttle threshold in °C (default 40)\n"
                 "  --backend B        auto | cpu | metal | vulkan (default auto)\n"
                 "  --kv T             f16 | q8_0 | q4_0 | q4_1 (default q8_0)\n"
                 "  --ctx N            context length    --window N  sliding window\n"
                 "  --sinks N          attention-sink tokens kept with --window (default 8)\n"
                 "  --triple-buffer    stream blocks through 3 rotating buffers (bounded memory)\n"
                 "  --expert-cache MB  MoE: RAM for streamed experts (-1 auto, 0 off; default auto)\n"
                 "  --memory-budget MB cap on resident memory (weights + caches); default: free RAM\n"
                 "  --threads N        worker threads (default: performance cores, minus one\n"
                 "                     when every core is a performance core)\n"
                 "experimental (LIYAB_ENABLE_EXPERIMENTAL=ON builds):\n"
                 "  --early-exit P     exit early when token confidence > P (e.g. 0.98)\n"
                 "  --prune-heads R    keep ratio R of attention heads when hot / low power\n"
                 "  --egls T           skip a block's FFN when its entropy delta < T (e.g. 0.002)\n"
                 "  --tdss hot|always  2:4 sparse FFN weights while throttled, or always\n"
                 "  --kv-dedup DIR     persistent prefix KV cache (reuse system-prompt KV across runs)\n");
}

}  // namespace

int main(int argc, char** argv) {
    liyab_engine_config config;
    liyab_engine_config_default(&config);
    liyab_sampling_params params;
    liyab_sampling_params_default(&params);
    params.max_tokens = 128;
    std::string prompt;
    bool device_only = false;
    bool tokenize_only = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--device") device_only = true;
        else if (arg == "--tokenize") tokenize_only = true;
        else if (arg == "-m" || arg == "--model") config.model_path = next();
        else if (arg == "-p" || arg == "--prompt") prompt = next();
        else if (arg == "-n") params.max_tokens = std::atoi(next());
        else if (arg == "--temp") params.temperature = static_cast<float>(std::atof(next()));
        else if (arg == "--top-k") params.top_k = std::atoi(next());
        else if (arg == "--top-p") params.top_p = static_cast<float>(std::atof(next()));
        else if (arg == "--seed") params.seed = std::strtoull(next(), nullptr, 10);
        else if (arg == "--draft") config.draft_model_path = next();
        else if (arg == "--draft-tokens") config.draft_tokens = std::atoi(next());
        else if (arg == "--tps") config.target_tps = std::atof(next());
        else if (arg == "--skin-threshold") config.skin_threshold_c = static_cast<float>(std::atof(next()));
        else if (arg == "--ctx") config.context_length = std::atoi(next());
        else if (arg == "--window") config.sliding_window = std::atoi(next());
        else if (arg == "--sinks") config.kv_sink_tokens = std::atoi(next());
        else if (arg == "--triple-buffer") config.triple_buffer_loading = 1;
        else if (arg == "--expert-cache") config.expert_cache_mb = std::atoll(next());
        else if (arg == "--memory-budget") config.memory_budget_mb = std::atoll(next());
        else if (arg == "--threads") config.n_threads = std::atoi(next());
        else if (arg == "--early-exit") {
            config.early_exit = 1;
            config.early_exit_threshold = static_cast<float>(std::atof(next()));
        } else if (arg == "--egls") {
            config.egls = 1;
            config.egls_threshold = static_cast<float>(std::atof(next()));
        } else if (arg == "--kv-dedup") {
            config.kv_dedup_dir = next();
        } else if (arg == "--tdss") {
            const std::string v = next();
            config.tdss = v == "always" ? 2 : 1;
        } else if (arg == "--prune-heads") {
            config.head_pruning = 1;
            config.head_keep_ratio = static_cast<float>(std::atof(next()));
        }
        else if (arg == "--profile") {
            const std::string v = next();
            if (v == "performance") config.power_profile = LIYAB_POWER_PERFORMANCE;
            else if (v == "balanced") config.power_profile = LIYAB_POWER_BALANCED;
            else if (v == "low_power") config.power_profile = LIYAB_POWER_LOW_POWER;
            else return usage(), 2;
        } else if (arg == "--backend") {
            const std::string v = next();
            if (v == "auto") config.backend = LIYAB_BACKEND_AUTO;
            else if (v == "cpu") config.backend = LIYAB_BACKEND_CPU;
            else if (v == "metal") config.backend = LIYAB_BACKEND_METAL;
            else if (v == "vulkan") config.backend = LIYAB_BACKEND_VULKAN;
            else return usage(), 2;
        } else if (arg == "--kv") {
            const std::string v = next();
            if (v == "f16") config.kv_cache_type = LIYAB_KV_F16;
            else if (v == "q8_0") config.kv_cache_type = LIYAB_KV_Q8_0;
            else if (v == "q4_0") config.kv_cache_type = LIYAB_KV_Q4_0;
            else if (v == "q4_1") config.kv_cache_type = LIYAB_KV_Q4_1;
            else return usage(), 2;
        } else if (arg == "-h" || arg == "--help") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage();
            return 2;
        }
    }

    std::printf("liyab %s\n", liyab_version());
    if (device_only) {
        std::string text(liyab_describe_device(nullptr, 0) + 1, '\0');
        liyab_describe_device(text.data(), text.size());
        std::printf("%s", text.c_str());
        return 0;
    }
    if (config.model_path == nullptr || prompt.empty()) {
        usage();
        return 2;
    }

    if (liyab_engine_create(&config, &g_engine) != LIYAB_OK) {
        std::fprintf(stderr, "error: %s\n", liyab_last_error());
        return 1;
    }
    if (tokenize_only) {
        const int32_t n = liyab_engine_tokenize(g_engine, prompt.c_str(), params.add_bos, nullptr, 0);
        std::vector<int32_t> ids(static_cast<size_t>(std::max(0, n)));
        liyab_engine_tokenize(g_engine, prompt.c_str(), params.add_bos, ids.data(), n);
        for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? " " : "", ids[i]);
        std::printf("\n");
        liyab_engine_destroy(g_engine);
        return n < 0 ? 1 : 0;
    }
    std::string info(liyab_engine_describe(g_engine, nullptr, 0) + 1, '\0');
    liyab_engine_describe(g_engine, info.data(), info.size());
    std::fprintf(stderr, "%s\n", info.c_str());

    std::signal(SIGINT, on_sigint);
    std::printf("%s", prompt.c_str());
    liyab_generation_stats stats{};
    const liyab_status status = liyab_engine_generate(g_engine, prompt.c_str(), &params, print_piece, nullptr, &stats);
    std::printf("\n");
    if (status != LIYAB_OK) {
        std::fprintf(stderr, "error: %s\n", liyab_last_error());
    } else {
        std::fprintf(stderr,
                     "\nprompt %d tok in %.1f ms (TTFT %.1f ms) | %d tok in %.1f ms = %.2f tok/s | paced idle %.1f ms%s\n",
                     stats.prompt_tokens, stats.prefill_ms, stats.ttft_ms, stats.generated_tokens, stats.decode_ms,
                     stats.tokens_per_second, stats.paced_idle_ms, stats.cancelled ? " | cancelled" : "");
        if (stats.draft_tokens_proposed > 0) {
            std::fprintf(stderr, "speculative: %d/%d draft tokens accepted (%.0f%%)\n", stats.draft_tokens_accepted,
                         stats.draft_tokens_proposed,
                         100.0 * stats.draft_tokens_accepted / stats.draft_tokens_proposed);
        }
        if (stats.thermal_reroutes > 0) std::fprintf(stderr, "thermal reroutes: %d steps\n", stats.thermal_reroutes);
        if (stats.early_exits > 0) {
            std::fprintf(stderr, "early exits: %d steps, %d blocks skipped\n", stats.early_exits,
                         stats.early_exit_layers_skipped);
        }
        std::fprintf(stderr, "KV cache: %.1f MiB in use\n", static_cast<double>(stats.kv_cache_bytes) / (1024.0 * 1024.0));
        if (stats.weight_stalls > 0) {
            std::fprintf(stderr, "weight stalls: %d (%.1f ms waiting for storage)\n", stats.weight_stalls,
                         stats.weight_wait_ms);
        }
        if (stats.ffn_blocks_skipped > 0) {
            std::fprintf(stderr, "EGLS: %d FFN blocks skipped (%.1f per token)\n", stats.ffn_blocks_skipped,
                         static_cast<double>(stats.ffn_blocks_skipped) / std::max(1, stats.generated_tokens));
        }
        if (const int32_t used = stats.expert_hits + stats.expert_late + stats.expert_misses; used > 0) {
            std::fprintf(stderr,
                         "experts: %d used, %.0f%% cached, %.0f%% prefetched late, %.0f%% missed; %.1f MiB/token read, "
                         "%.0f ms waiting (%.0f%% of decode)\n",
                         used, 100.0 * stats.expert_hits / used, 100.0 * stats.expert_late / used,
                         100.0 * stats.expert_misses / used,
                         static_cast<double>(stats.expert_bytes_read) / (1024.0 * 1024.0) /
                             std::max(1, stats.prompt_tokens + stats.generated_tokens),
                         stats.expert_stall_ms, 100.0 * stats.expert_stall_ms / std::max(1.0, stats.prefill_ms + stats.decode_ms));
        }
        if (stats.expert_predicted > 0) {
            std::fprintf(stderr, "experts: prediction precision %.0f%% (%d of %d guesses chosen)\n",
                         100.0 * stats.expert_predicted_used / stats.expert_predicted, stats.expert_predicted_used,
                         stats.expert_predicted);
        }
        if (stats.expert_unused > 0) {
            std::fprintf(stderr, "experts: %d prefetched and evicted unused (%.1f per token)\n", stats.expert_unused,
                         static_cast<double>(stats.expert_unused) /
                             std::max(1, stats.prompt_tokens + stats.generated_tokens));
        }
        if (stats.generated_tokens > 0 && stats.decode_ms > 0.0) {
            // Decode time per token by phase; "other" is norms, residuals, sampling and callbacks.
            const double per = 1.0 / stats.generated_tokens;
            const double phases[] = {stats.attention_ms, stats.delta_net_ms, stats.router_ms, stats.experts_ms,
                                     stats.shared_expert_ms, stats.dense_ffn_ms, stats.lm_head_ms};
            const char* names[] = {"attention", "DeltaNet", "router", "experts", "shared expert", "dense FFN", "LM head"};
            double timed = 0.0;
            std::fprintf(stderr, "decode ms/token:");
            for (size_t i = 0; i < sizeof phases / sizeof phases[0]; ++i) {
                timed += phases[i];
                if (phases[i] > 0.0) std::fprintf(stderr, " %s %.1f |", names[i], phases[i] * per);
            }
            std::fprintf(stderr, " other %.1f | total %.1f\n", (stats.decode_ms - timed) * per, stats.decode_ms * per);
        }
        if (stats.cached_prefix_tokens > 0) {
            std::fprintf(stderr, "KV dedup: %d prompt tokens restored from snapshot\n", stats.cached_prefix_tokens);
        }
        if (stats.sparse_ffn_steps > 0) std::fprintf(stderr, "TDSS: %d steps on 2:4 sparse FFN\n", stats.sparse_ffn_steps);
        if (stats.head_pruned_steps > 0) std::fprintf(stderr, "head pruning: %d steps\n", stats.head_pruned_steps);
    }
    liyab_engine_destroy(g_engine);
    g_engine = nullptr;
    return status == LIYAB_OK ? 0 : 1;
}
