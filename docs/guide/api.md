# Engine API and CLI examples

[← README](../../README.md) · [Documentation index](README.md)

Commands and source paths in examples are relative to the repository root.

## On this page

- [Using the engine](#using-the-engine)
- [C++](#c)
- [C ABI (`liyab_c_api.h`)](#c-abi-liyab_c_apih)
- [Android (Kotlin)](#android-kotlin)
- [iOS (Swift)](#ios-swift)
- [Command line](#command-line)

---

## Using the engine

### C++

```cpp
#include "liyab/liyab.h"
#include <iostream>

int main() {
    liyab::EngineConfig config;
    config.model_path = "/data/local/tmp/model.gguf";
    config.power.profile = liyab::PowerProfile::Balanced;  // paces near reading speed, reacts to the thermal forecast
    config.memory_budget_mb = 5500;                         // stay under the platform's per-app limit

    auto engine = liyab::Engine::create(config);
    if (!engine) { std::cerr << engine.status().to_string() << "\n"; return 1; }
    std::cerr << engine.value()->describe();

    liyab::SamplingParams params;
    params.max_tokens = 128;
    auto stats = engine.value()->generate("Explain unified memory in one paragraph.", params,
                                          [](std::string_view piece, int32_t /*token*/) {
                                              std::cout << piece << std::flush;
                                              return true;  // false stops generation
                                          });
    if (stats) std::cerr << "\n" << stats->tokens_per_second << " tok/s\n";
}
```

Constrained decoding: the reply is a JSON value valid against a schema.

```cpp
auto schema = liyab::Grammar::from_json_schema(R"({"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer", "minimum": 1, "maximum": 14}},
    "required": ["city"]})");
liyab::GenerateOptions options;
options.grammar = *schema;  // or options.continuation: forced text and constrained spans where you choose
auto stats = engine.value()->generate(prompt, params, on_piece, options);
```

Embeddings and reranking, with an embedding model and a reranker (each its own engine):

```cpp
const std::string_view texts[] = {
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?",
    "The capital of China is Beijing."};
auto vectors = embedder->embed(texts);  // unit vectors of embedding_size() values; dot product = cosine
const std::string_view documents[] = {"The capital of China is Beijing.", "Gravity is a force..."};
auto scores = reranker->rerank("What is the capital of China?", documents);  // P(relevant) per document
```

Segment reuse (opt-in, lossy; needs the prefix cache): mark the documents of a prompt, and a later prompt that holds
any of them anywhere reuses their KV instead of processing them again.

```cpp
config.prefix_cache_mb = 256;
config.segment_reuse = true;       // calibrated once per model in the background (off until then), kept in config.cache_dir
liyab::GenerateOptions options;
for (const std::string& doc : documents) {  // byte ranges of the documents in the rendered prompt
    const size_t at = prompt.find(doc);
    options.segments.push_back({at, at + doc.size()});
}
auto stats = engine.value()->generate(prompt, params, on_piece, options);
// stats->segment_reused_tokens, segment_recomputed_tokens; engine.value()->segment_reuse() says why it is off
```

Several conversations at once (`max_sequences`): each on a sequence of its own, called from a thread of its own;
the engine decodes them together, and each reply is what it would be alone.

```cpp
config.max_sequences = 2;  // the default sequence (the engine's own calls) and one more
auto engine = liyab::Engine::create(config);
auto agent = engine.value()->open_sequence(liyab::SequencePriority::Background);  // yields to the user's turn
std::thread background([&] { agent.value()->generate(agent_prompt, params, on_agent_piece); });
engine.value()->generate(chat_prompt, params, on_chat_piece);  // the user's, foreground
background.join();
agent.value().reset();  // closed: its context set aside in the prefix cache, its memory back to the expert cache
```

`Engine::cancel()` is thread-safe and stops at the next token. `prefill(text)` processes a prefix ahead of time,
such as a system prompt right after loading. `reset_context()` forgets everything. The public headers in
`include/liyab/` document each function's contract: preconditions, ownership, thread safety and error cases.

### C ABI (`liyab_c_api.h`)

```c
liyab_engine_config config;
liyab_engine_config_default(&config);
config.model_path = path;

liyab_engine* engine = NULL;
if (liyab_engine_create(&config, &engine) != LIYAB_OK) { fprintf(stderr, "%s\n", liyab_last_error()); }

liyab_sampling_params params;
liyab_sampling_params_default(&params);
liyab_generation_stats stats;
liyab_engine_generate(engine, "Hello", &params, on_piece /* (piece, len, token, user) -> continue? */, user, &stats);
liyab_engine_destroy(engine);
```

No C++ exception crosses the ABI. Errors return a `liyab_status`, with the message in `liyab_last_error()`
(thread-local). Fields are only ever appended to the C structs.

| Function | Purpose |
| :--- | :--- |
| `liyab_engine_generate` / `_generate_tokens` | Blocking generation from text or token ids; pieces are whole UTF-8 |
| `liyab_engine_generate_forced` | Generation with forced continuations (structured output) |
| `liyab_grammar_from_json_schema` / `_parse` / `_matches` / `_destroy` | Grammars for constrained decoding: from a JSON Schema or grammar text |
| `liyab_engine_generate_structured` | Generation constrained by a grammar, or forced text and constrained spans from a callback |
| `liyab_engine_cancel` / `_cancel_prefill` | Stop a generation / a background prefill (thread-safe) |
| `liyab_engine_prefill` / `_reset_context` | Process a prefix ahead of time / forget the context |
| `liyab_engine_save_state` / `_load_state` | Persist and restore the context across processes |
| `liyab_engine_save_state_of` | Persist the context a text continues, also one the prefix cache set aside |
| `liyab_engine_prefix_cache_stats` | What the prefix cache holds (RAM, storage, chunks, contexts, segments) and how it was used (thread-safe) |
| `liyab_engine_generate_segmented` / `_prefill_segmented` / `_segment_reuse` | Segment reuse (opt-in, lossy): generation and prefill with marked parts of the prompt; whether it is on and its calibrated share, or how far its background calibration got (`LIYAB_ERR_BUSY` until it ends) |
| `liyab_engine_trim_memory` / `_warm_memory` | Give the expert cache back while idle / refill it from its hot list |
| `liyab_engine_save_expert_profile` / `_load_expert_profile` | Carry the expert hot list to a later process |
| `liyab_engine_memory_plan` / `_context_length` | How the model uses memory; the resolved context length |
| `liyab_engine_set_power_profile` | Change the power profile at once, even mid-generation |
| `liyab_engine_get_counters` | Live counters for monitoring UIs (thread-safe) |
| `liyab_engine_metadata` / `_describe` / `liyab_describe_device` | GGUF metadata (e.g. recommended sampling), engine and device descriptions |
| `liyab_describe_model_fit` | How a model file will run on this device without loading it: memory fit and the decode rate each backend's measured read speeds allow at most (timed once per format, kept in `cache_dir`) |
| `liyab_supported_architectures` | Architectures this build runs |
| `liyab_engine_render_chat` / `_chat_format` | The model's chat template applied to a conversation (JSON); its layout of turns and whether it is prefix-stable |
| `liyab_engine_embed` / `_rerank` | Pooling models: one vector per text; one relevance score per document of a query |
| `liyab_engine_pooling` / `_embedding_size` | What the model pools (none, mean, last token, rank); the values per vector |
| `liyab_engine_load_adapter` / `_unload_adapter` / `_set_adapter` / `_adapter_count` / `_adapter_info` | [LoRA adapters](inference-features.md#lora-adapters): load several next to the model, select the one later calls apply (and its scale), list them |
| `liyab_sequence_open` / `_close` / `_generate` / `_prefill` / `_cancel` / `_cancel_prefill` / `_set_priority` / `_set_adapter` / `_reset_context` / `_id`; `liyab_engine_scheduler_stats` | [Several sequences](memory-and-serving.md#context-memory-and-serving) (`max_sequences`): conversations of their own whose calls run at once from different threads, decoded together, each output what it gives alone; foreground or background priority; per-sequence cancellation and adapter; how passes were shared and the time in them |
| `liyab_set_log_callback` / `liyab_log_buffer_enable` / `_take` | Logs to a callback, or to a buffer for polling UIs |

`liyab_generation_stats` reports where time went:
* prompt, cached-prefix, generated and forced tokens, and the time of the forced passes; time to first token, the
  time before it spent matching or restoring the context, and the part of it spent waiting for expert reads;
* decode time per phase: attention, DeltaNet, routers, experts, shared expert, dense FFN, LM head;
* the expert cache's behaviour: hits, late prefetches, misses, bytes read, read-but-unused, prediction precision,
  dropped guesses, skipped experts, time waiting;
* segment reuse: prompt positions taken from stored segments, how many of them were processed again, the time it
  took.

`liyab_engine_config` exposes:
* the backend, threads and power profile;
* the memory budget, expert cache, `requant_bits`, context length, and the prefix cache in RAM (`prefix_cache_mb`)
  and on storage (`prefix_cache_dir`, `prefix_cache_disk_mb`);
* `cache_dir`, where what is measured once per model is kept (the KV cache's key scales, latent attention's
  form ratio, segment reuse's calibration), and, with `converted_weights_mb`, the weights converted at load (see
  [memory and serving](memory-and-serving.md#converted-weights-cache));
* segment reuse (`segment_reuse`, its target `segment_reuse_max_kl`);
* the memory cap of loaded LoRA adapters (`adapter_memory_mb`);
* the sequences kept at once (`max_sequences`, 1 by default; a ceiling: a sequence takes its memory when opened);
* KV type, sliding window and sinks, and the KV type of pooling models (`pooling_kv_cache_type`, F16 by default);
* speculative decoding (draft model, lookup or suffix drafts);
* the MoE trade-offs;
* the experimental flags.

### Android (Kotlin)

The library exposes the C ABI. The JNI glue lives in your app, because its symbol names depend on your package:

```cpp
// app/src/main/cpp/liyab_jni.cpp
#include <jni.h>
#include "liyab/liyab_c_api.h"

struct Ctx { JNIEnv* env; jobject cb; jmethodID onToken; };

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_ai_Liyab_nativeCreate(JNIEnv* env, jclass, jstring path) {
    liyab_engine_config c; liyab_engine_config_default(&c);
    const char* p = env->GetStringUTFChars(path, nullptr);
    c.model_path = p;
    liyab_engine* e = nullptr;
    liyab_engine_create(&c, &e);
    env->ReleaseStringUTFChars(path, p);
    return reinterpret_cast<jlong>(e);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_ai_Liyab_nativeGenerate(JNIEnv* env, jclass, jlong h, jstring prompt, jobject cb) {
    Ctx ctx{env, cb, env->GetMethodID(env->GetObjectClass(cb), "onToken", "([B)Z")};
    liyab_sampling_params sp; liyab_sampling_params_default(&sp);
    const char* p = env->GetStringUTFChars(prompt, nullptr);
    liyab_engine_generate(reinterpret_cast<liyab_engine*>(h), p, &sp,
        [](const char* s, size_t n, int32_t, void* u) -> int32_t {
            auto* c = static_cast<Ctx*>(u);
            jbyteArray bytes = c->env->NewByteArray(static_cast<jsize>(n));
            c->env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(n), reinterpret_cast<const jbyte*>(s));
            const jboolean go = c->env->CallBooleanMethod(c->cb, c->onToken, bytes);
            c->env->DeleteLocalRef(bytes);
            return go ? 1 : 0;
        }, &ctx, nullptr);
    env->ReleaseStringUTFChars(prompt, p);
}
```

```kotlin
object Liyab {
    init { System.loadLibrary("liyab"); System.loadLibrary("app_jni") }
    external fun nativeCreate(modelPath: String): Long
    external fun nativeGenerate(handle: Long, prompt: String, cb: TokenCallback)
    fun interface TokenCallback { fun onToken(utf8: ByteArray): Boolean }
}
// Pieces are complete UTF-8, so String(bytes, Charsets.UTF_8) is safe. Generate off the main thread.
```

### iOS (Swift)

```swift
import Liyab

final class LiyabEngine {
    private var engine: OpaquePointer?

    init(modelPath: String) throws {
        var config = liyab_engine_config()
        liyab_engine_config_default(&config)
        let status = modelPath.withCString { path -> liyab_status in
            config.model_path = path
            return liyab_engine_create(&config, &engine)
        }
        guard status == LIYAB_OK else { throw NSError(domain: String(cString: liyab_last_error()), code: Int(status.rawValue)) }
    }

    func generate(_ prompt: String, onPiece: @escaping (String) -> Bool) {
        var params = liyab_sampling_params()
        liyab_sampling_params_default(&params)
        let box = Unmanaged.passRetained(onPiece as AnyObject)
        defer { box.release() }
        liyab_engine_generate(engine, prompt, &params, { piece, len, _, user in
            let cb = Unmanaged<AnyObject>.fromOpaque(user!).takeUnretainedValue() as! (String) -> Bool
            return cb(String(decoding: Data(bytes: piece!, count: len), as: UTF8.self)) ? 1 : 0
        }, box.toOpaque(), nil)
    }

    deinit { liyab_engine_destroy(engine) }
}
```

### Command line

```bash
liyab-cli --device                                          # SoC, accelerators, backend ranking
liyab-cli --doctor -m model.gguf --cache-dir DIR            # memory fit and decode ceiling per backend for that model
liyab-cli -m model.gguf -p "Once upon a time" -n 128 --temp 0.8
liyab-cli -m moe.gguf -p "..." --memory-budget 5500         # stay within a per-app memory limit
liyab-cli -m moe.gguf -p "..." --expert-cache 3000          # MoE: experts through a 3000 MiB cache (-1 auto)
liyab-cli -m moe.gguf -p "..." --expert-mass 0.9            # lossy: top experts covering 90% of router weight
liyab-cli -m model.gguf -p "..." --lookup                   # always draft from the text (the default is automatic: only where a timed verification pass is cheap)
liyab-cli -m model.gguf -p "..." --no-lookup                # never draft from the text
liyab-cli -m model.gguf -p "..." --context-lookup           # ... by context lookup instead
liyab-cli -m model.gguf -p "..." --lookup --linear-drafts   # suffix drafts' best branch only, not their tree
liyab-cli -m model.gguf -p "..." --draft draft.gguf         # speculative decoding with a draft model
liyab-cli -m model.gguf -p "..." --prefix-cache 256         # keep replaced contexts as KV chunks, 256 MiB of RAM
liyab-cli -m model.gguf -p "..." --prefix-cache-disk 1024 ~/.cache/liyab/ctx  # ... or in files, kept across runs
liyab-cli -m model.gguf -p "$SYSTEM$Q2" --prefix-cache-disk 1024 ~/.cache/liyab/ctx --prefix-stats
                                                            # a later run gets the shared $SYSTEM back from them
liyab-cli -m model.gguf -p "..." --prefix-cache-disk 1024 ~/.cache/liyab/ctx --prefix-cache-lossy
                                                            # lossy: chunks on storage in 4 bits, ~3.4x smaller (F16)
liyab-cli -m model.gguf -p "..." --profile low_power        # performance | balanced | low_power
liyab-cli -m model.gguf -p "..." --window 1024 --sinks 8 --kv q4_1
liyab-cli -m model.gguf -p "..." --cache-dir ~/.cache/liyab   # keep the key scales: measured on the first run only
liyab-cli -m model.gguf -p "..." --cache-dir ~/.cache/liyab --converted-cache 20000   # also keep the weights converted at load (20 GB at most): later loads map them
liyab-cli -m model.gguf -p "$TURN1" --then "$TURN2"           # a follow-up turn that reuses the context, as a chat does
liyab-cli -m model.gguf -p "..." --json                     # load line: memory/weight layout; turn lines: stats, GPU/I/O counters, peak RSS, reply
liyab-cli -m model.gguf -p "..." --lora adapter.gguf --lora-scale 0.5  # a LoRA adapter applied to every turn
liyab-cli --help                                            # every option
```

For each generation, `liyab-cli` prints the prompt tokens it reused, decode time per phase, the expert cache's
behaviour and the process's peak resident memory. `--then TEXT` (repeatable) adds a turn whose prompt is the previous
prompt, the reply to it and `TEXT`, so it measures what a follow-up message costs.

---
