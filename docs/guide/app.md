# App and local API

[← README](../../README.md) · [Documentation index](README.md)

Commands and source paths in examples are relative to the repository root.

## On this page

- [The Liyab app](#the-liyab-app)
- [An assistant that replaces the cloud one](#an-assistant-that-replaces-the-cloud-one)
- [Fast where users feel it](#fast-where-users-feel-it)
- [Local API: one model for every app on the device](#local-api-one-model-for-every-app-on-the-device)
- [Everything else](#everything-else)

---

## The Liyab app

`app/` is a Flutter app (Android and macOS today, iOS next) that calls the C ABI directly through `dart:ffi`. The
bindings are hand-written in `app/lib/engine/liyab_ffi.dart`. On macOS the engine is linked statically into the app
(`app/macos/Runner/Configs/Engine.xcconfig`) and looked up in the process, as on iOS. Loads, prefills and generations run on a worker isolate, while
cancellation and live counters go straight to the C API, which makes them thread-safe.

### An assistant that replaces the cloud one

* **System assistant.** Liyab can be the default digital assistant. The assist gesture opens a sheet over the
  current app, with the living flame, a text field and the latest answer. Pulling the sheet up turns it into the full
  app, as Gemini's overlay does. Both windows share one engine, so the model is loaded once.
* **On a Mac, a command bar.** **Option-Space** opens Liyab from any app, in the manner of Spotlight: a large input
  with the living flame, the conversation under it, and a footer with the model and the keys (Return asks, ⌘N
  starts a new chat, ⌘Return opens the app, Esc stops an answer or closes the bar). The bar grows downward as the
  answer streams, over the system's blurred material; an aura of the flame's colours turns around it with the
  assistant's state. It is the app's own window and engine in another mode, so the model is loaded once; closing it
  returns to the app you were in. The app keeps running with its window closed, so the shortcut always works.
* **Agent tools.** The assistant reads your own data to answer about your day:
  * on Android: calendar events; notifications from every app (chat messages, email previews; kept in memory
    only); SMS, the call log, contacts and the clipboard;
  * on macOS: calendar events and contacts (after the system's prompt); email from every account in Apple Mail,
    iMessage and SMS, and the Phone/FaceTime call history (these three read Apple's own databases, which need Full
    Disk Access, granted in System Settings); the clipboard. macOS has no API for other apps' notifications.

  Each source is off until you turn it on, and only enabled sources are offered to the model.
* **How tools are called.** Tools are described and called in the model's own format, read from its GGUF chat
  template: Qwen3.5/3.6 XML calls or Qwen2.5/3 JSON calls. A reply can make up to 4 calls, each shown above the
  answer ("Read your calendar: 3 events").
  * Calls are **constrained decoding**, the path the local API's calls take: the model chooses to write
    `<tool_call>`, and from there the call is one constrained span (see
    [constrained decoding](memory-and-serving.md#context-memory-and-serving)) that admits only the names of the enabled tools and arguments
    valid against the chosen tool's JSON Schema, the one its description in the system prompt shows. Names, keys and
    separators are fixed by the grammar (forced, or sampled under its mask where the model's own token covers
    them, see token healing), so the model chooses only whether to call, which tool and the values. Every call
    parses, names only declared parameters and has the required ones.
  * Times are local: a date-time in the shape the prompt shows by example (`2026-10-09T08:00`, seconds optional), a
    day alone for its midnight, or a time alone for today (a "since" still to come today means yesterday's). The
    engine's JSON Schema subset cannot say this (`date-time` requires a UTC offset, `pattern` is refused), so these
    values follow a grammar of their own, and a time the model writes always parses (a free value such as "today"
    would leave the tool to its default range). The time alone is there because small models write one when the
    day is today, and years are 20xx so that a first digit tells a time from a date: with dates only, Qwen2.5-0.5B's
    `16:30:00` was bent into the year 1606.
  * Text outside a call is sampled with no mask, token for token as in an unconstrained reply. Each step's log line
    gives the tokens a call's grammar constrained and the grammar's time ("9 constrained, grammar 4.3 ms").
  * **Measured** on a shared M4 Pro's CPU through the app's own prompt and path, with Android's six tools and ten
    questions asked five times each at the app's settings: Qwen2.5-0.5B made 45 calls, all valid. With only their
    syntax and names forced, 15 of its 45 were not: times that do not parse (`16 October 2026, 10:30 UTC+03:00`,
    `2026-10-10T08:00; default 24 h ago`), parameters the tool does not have, calls that do not parse. Qwen3-0.6B (25
    calls) and Qwen3.5-0.8B (28) wrote valid calls either way, also at temperature 1.0 on harder questions with
    relative times. A call's grammar cost 0.6–0.9 ms per constrained token against decode steps of about 80 ms
    in the same runs (a call has 15–35 constrained tokens), and asking the structure after each token of a reply
    without a call cost 20–60 µs a token. Greedy replies without a call came out the same, token for token.
  * Results are compact text with ISO dates. On Qwen3.6's tokenizer, 6 events take 190 tokens instead of 428 as
    JSON, which saves ~18 s of prefill on the 35B.
  * The system prompt tells the model to treat tool results as data, not instructions.
* **Prompts in the model's own format.** At load the app asks the engine how the model's chat template lays out
  turns (`chat_format`), in each thinking mode it uses and with `preserve_thinking`, since past replies are replayed
  with their reasoning. When the template is prefix-stable, prompts are built from its pieces, so each turn's prompt
  continues the previous one and only the new message is processed. Otherwise (Qwen3, Qwen3.5), or without a
  template, the built-in format of the model's family is used (ChatML, Llama 3, Zephyr) and the log says why. For
  Qwen3.6 the template's pieces are exactly the built-in ChatML ones, so its prompts and saved contexts are
  unchanged. Llama 3's template writes today's date into the system block, so its context is prepared again
  once a day. Tools are offered when the format is ChatML's, whose tool-call formats the app knows.

### Fast where users feel it

* **Context-aware chat.** History is cut by the token budget, and past replies are replayed exactly as generated, so
  each message reuses the engine's context. The system block is restored from a state saved the first time it was
  processed.
* **Typing ahead.** While you write, the draft's words are processed in the background as each one ends. When you
  pause, so is the whole message as it would be sent, time line and reply opening included, all but its last token.
  Text that arrives all at once (a paste, dictation, text shared into the composer: 12 or more characters in one
  edit) is prepared whole right away, without waiting for the pause.
  On the 35B a message sent after a pause started its answer in 0.38 s (1 new prompt token), and short messages
  typed and sent at once in 0.68–0.91 s (9–10 new tokens), where they had taken 3.5–5.6 s (~48).
* **The time, when it changed.** A message carries the current date and time (`[Now: …]`, ~20 tokens) only when the
  history holds none from the last half hour of the same day; in a quick exchange the last one still holds.
* **A priority scheduler.** Your requests (loading, generating) go before background work (typed-ahead prefills,
  preparation, saves, cache warming). A request that arrives during a background prefill cancels it and continues
  from the work already done.
* **Long conversations.** When the history no longer fits, the oldest turns leave the context in one step, until
  the history fills at most half of it. A divider marks the cut and the messages stay on screen. The following
  messages extend the same prompt, so the cost comes once every many messages, not every time.
* **Leaving the screen.** HyperOS stops background apps that hold several GB. When Liyab is hidden, it parks:
  1. it saves the expert hot list;
  2. it empties the expert cache (2.4 GB on the 35B);
  3. it saves the conversation and its context: the engine's, or, when another app's request replaced it, the copy
     the prefix cache set aside (written as kept, nothing processed).

  After 5 minutes hidden (configurable) the model is unloaded. A request from another app through the
  [local API](#local-api-one-model-for-every-app-on-the-device) loads it again, and it is unloaded again 5 minutes
  after the last request. Coming back reloads it, restores the context and warms the cache; with the model still
  loaded (back from copying something in another app, say), it warms the cache at once. While the model wakes up,
  the flame burns in the middle of the conversation.
* **Finishing work in the background (Android).** Android leaves a hidden app little: on HyperOS four of the eight
  cores (a reply drops from ~7 to 1.3 tok/s on the 35B while another app is in use), and soon none, because its
  cgroup freezer stops hidden apps. Leave Liyab while it works (a reply, a load, a model being moved, a request of
  another app) and it holds a foreground service until that work and the park after it are done. The process keeps
  all eight cores and is not frozen, and a partial wake lock lets the reply finish with the screen off. The service
  computes nothing itself and starts only with work in hand, so leaving an idle Liyab costs nothing. On the
  reference phone, Qwen3.6-35B-A3B:

  | A reply while Liyab is hidden | Without the service | With it |
  |-------------------------------|--------------------:|--------:|
  | Another app in use            | 1.3 tok/s           | 8.2 tok/s |
  | Phone locked                  | 7.4 tok/s (four cores) | 8.1–8.6 tok/s |

  In Liyab itself the same chat ran at 7.5–7.6 tok/s. The service stopped within a second of the conversation being
  saved, and HyperOS froze the idle process two seconds later. Its notification asks for a Live Update (Android
  16): systems that support it show a chip in the status bar or their island (HyperOS 3 shows it in its Super
  Island) and a card on the lock screen. It alerts once, without sound or vibration, on a channel with a
  notification light; HyperOS plays no edge light for it. Android starts the service only while or just after
  Liyab is on screen, or at any time when Liyab's battery use is unrestricted.

  With the [local API](#local-api-one-model-for-every-app-on-the-device) on, the service also stays while Liyab is
  hidden and idle, only listening: a frozen process never accepts a connection, and on HyperOS a request to a
  hidden Liyab hung for 10 minutes without an answer. Listening holds no wake lock and runs no code (0 CPU ticks
  in 2 minutes; 382 MB resident once the idle model is released) and shows a notification of the lowest
  importance. A request updates the running service in place, which Android allows at any time: the wake lock and
  the Live Update come with it, and it goes back to listening when the request ends. On the reference phone a
  request to a hidden Liyab whose 35B had been released loaded it in 2.3–3.4 s and answered 200 tokens at
  10.9–11.1 tok/s. Releasing the model again a minute later took about 1 s of CPU, and the 2 minutes after it
  none. A model loaded that way warms its expert cache only once you write in Liyab again.
* **A new chat** restores the system prompt from its saved state (47 ms on the 35B), so its first message costs
  only its own tokens.
* **A first-token budget.** A reply's first token is due within 2 s of sending. A slower one is logged
  ("Latency budget: …") with where the time went: the model waking up, waiting in line, restoring or rewinding the
  context, the new prompt tokens and the first step, and expert reads. Every step's log line also gives the time
  from sending to its first token.

### Local API: one model for every app on the device

Settings → Apps on this phone → **Local API** (off by default) lets other apps on the device use the loaded model as
they would a server:

| Protocol | Endpoint | Calls |
| :--- | :--- | :--- |
| OpenAI | `http://127.0.0.1:8642/v1` | `POST /chat/completions`, `POST /completions`, `POST /embeddings`, `GET /models` |
| Rerank (Jina / Cohere shape) | `http://127.0.0.1:8642/v1` | `POST /rerank` (also at `/rerank` and `/v2/rerank`) |
| Anthropic Messages | `http://127.0.0.1:8642/v1` | `POST /messages`, `POST /messages/count_tokens`, `GET /models` |
| gRPC | `127.0.0.1:8643` | `liyab.v1.Liyab`: `ListModels`, `Generate` (server streaming), `CountTokens` ([`app/proto/liyab.proto`](../../app/proto/liyab.proto)) |

The model lists name the model requests are served with, also while it is not in memory (released while idle, or
not loaded yet in a process Android started without a window): the first request loads it.

```bash
curl http://127.0.0.1:8642/v1/chat/completions \
  -H "Authorization: Bearer $LIYAB_TOKEN" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Summarize my day in one line."}],"stream":true}'
```

* **Security.** Both servers listen on the loopback interface only, so nothing outside the device can connect.
  Every call needs the token shown in Settings: `Authorization: Bearer`, `x-api-key` for Anthropic clients, or gRPC
  metadata. **New token** revokes the old one.
* **Compatibility.**
  * Prompts are built with the loaded model's own chat template when it is prefix-stable, else with the built-in
    format of its family (see [the app](#an-assistant-that-replaces-the-cloud-one)).
  * Streaming (server-sent events), sampling, `max_tokens`, stop sequences and usage are supported. Usage includes
    the reused prompt tokens.
  * Thinking is off unless requested, through `reasoning_effort`, `chat_template_kwargs.enable_thinking` or
    Anthropic's `thinking`. Reasoning comes back separately: `reasoning_content` for OpenAI clients, `thinking`
    blocks for Anthropic ones.
  * A final assistant message is continued, like Anthropic's prefill.
* **Alongside your chat, after it.** With the API on, the model loads with room for a second engine sequence for
  other apps' requests (`max_sequences` 2, a lane with its own worker isolate): a request runs at the same time as
  your chat, the engine decoding the two in shared passes, each reply what it would be alone. The lane's sequence
  is opened when a request arrives and closed once the lane has been idle for a few seconds: the API is stateless
  across requests (each carries its whole history, and the prefix cache keeps what the closed sequence set aside),
  so nothing is lost, and while no request runs the model and your chat use memory exactly as with the API off.
  Should the engine not have the memory to open it, the request runs on the chat's sequence, as below. Requests
  are background work: your chat's steps go first, and a request's prompt is stopped mid-pass when you send a
  message. Requests queue behind each other on their sequence; embeddings and reranking run on the chat's.
  Switching the API on sets the lane up from the next load (until then requests run on the chat's sequence, as
  below). A client that
  disconnects stops its generation, even while it is still queued or its prompt is being processed, and its
  embedding or reranking call, queued or running.
* **Your chat keeps its place.** On their own sequence, requests leave the chat's context alone. Without one, a
  request's prompt replaces the chat's context in the engine, which sets the chat's aside on storage (the prefix
  cache: at most 512 MB, no RAM) and restores it on your next message: 151 ms for a 1,355-token chat on the 35B
  (127 ms the next time), its first token 1.66 s after sending, where processing a 4,009-token chat again had taken
  163 s.
* **Listening without Liyab open.** With notification access granted, Android binds Liyab's notification listener
  on its own (at boot, after it stopped the process), which starts Liyab's process without a window. With the local
  API on, that process starts the app's Dart side and the API listens at once, under its listening service; the
  model loads for the first request (and unloads again after the idle delay). Android allows that service although
  Liyab was never on screen: the system binds the listener as a foreground binding, which lets the process start
  foreground services. With the API off, such a process starts nothing. HyperOS / MIUI refuse that binding unless
  Liyab's Autostart is allowed (after a reboot the log shows "AutoStartManagerService: Reject service …
  LiyabNotificationListener" and no process starts). No API can allow it, so with the local API on, Settings shows
  **Allow Autostart** while the switch reads off, and opens the system's Autostart screen.
* **Tools and JSON answers.** OpenAI clients pass `tools`, `tool_choice` (`auto`, `none`, `required`, a named
  function) and `parallel_tool_calls`, and get `tool_calls` back, streamed as deltas; tool results come back in
  `tool` messages. Anthropic clients pass `tools` and `tool_choice` (`auto`, `any`, `tool`, `none`,
  `disable_parallel_tool_use`) and get `tool_use` blocks, streamed as `input_json_delta` events; results come back
  in `tool_result` blocks. Calls are written in the loaded model's own format (Hermes JSON for Qwen2.5 / Qwen3,
  XML for Qwen3.5 / Qwen3.6, from its chat template), and past calls and results are replayed the way its template
  renders them.
  * **Every call is valid.** The model chooses to write `<tool_call>`; from there the call is one constrained span
    (see [constrained decoding](memory-and-serving.md#context-memory-and-serving)): only the names of the request's tools, and only
    arguments valid against the chosen tool's schema. Names, keys and separators are fixed by the grammar (forced,
    or sampled under its mask where the model's own token covers them), so the model chooses only the tool and the
    values. With `tool_choice` `required` or a named tool, the call is forced right away
    (after the reasoning, when the model reasons). In XML calls, string parameters are written raw, as Qwen's
    parser reads them, and every other type as JSON. A raw string of free characters (with a `minLength`, a
    `maxLength` long enough to hold the 13 characters of `\n</parameter>`, or a `format` the engine does not check,
    such as `email` or `uri`) is written without `<`: such a value could otherwise take in the closing tag and go
    on, and the call would never end. The first `\n</parameter>` closes every value.
  * **A tool whose schema uses a keyword the engine cannot enforce** (`pattern`, `uniqueItems`...) keeps working:
    its arguments are then constrained to any JSON object (any parameters in XML), not to its schema.
  * **`response_format`** `json_object` or `json_schema` constrains the whole answer to a JSON object or to the
    schema (after the reasoning, when the model reasons); a schema the engine cannot enforce is refused with an
    error naming the keyword. `response_format` and `tools` together are refused.
  * Tools need a ChatML model whose template uses one of these formats; other models refuse them with an error.
    gRPC carries text only.
* **Embeddings and reranking.** With an [embedding model](inference-features.md#embeddings-and-reranking) loaded (Qwen3-Embedding),
  `/v1/embeddings` returns its vectors: `input` is a string or a list (one engine call for the whole list, shared
  prefixes computed once), `encoding_format` `float` or `base64` (the official OpenAI clients' default), and
  `dimensions` keeps the first values for models trained for it. A query carries its instruction in its text
  (`Instruct: {task}\nQuery:{query}` for Qwen3-Embedding). With a reranker loaded (a GGUF with its ranking head,
  such as Qwen3-Reranker's from ggml-org), `/v1/rerank` takes `query`, `documents` (strings or `{"text": …}`
  objects), `top_n` and `return_documents` (default true), and returns `results` sorted by `relevance_score`
  (P(relevant), highest first) with each document's `index`. Usage counts the tokens processed. A pooling model
  writes no text, so generation requests are refused while one is loaded, and so are embeddings or reranking with
  a model that does not do them.
* **LoRA adapters.** GGUF [LoRA adapters](inference-features.md#lora-adapters) in an `adapters` folder, under the app's models folder or
  the shared app folder (`/sdcard/Android/data/<package>/files/adapters`, where `adb push` puts files), load with
  every model, each named after its file without `.gguf` (the engine refuses, in the log, those of another
  architecture or that do not fit the model). The model lists name them after the model. A request selects one by
  naming it as its `model` (OpenAI, Anthropic), or by an `adapter` field (OpenAI, Anthropic; gRPC
  `GenerateRequest.adapter`), which must name a loaded adapter (else 400) and wins over `model`; any other model
  name is the model itself. Only that generation applies it: your own chat, its prepared prompts, saved
  conversations and embeddings run on the model itself, and the engine keeps each adapter's context apart (set
  aside in the prefix cache under its adapter when another request comes).
* **Not yet.** Images, `n` > 1, logprobs and token arrays as embedding input are refused with an error, never
  silently ignored.

### Everything else

* **Models.** Search Hugging Face and pick a quantization. Models that the engine cannot run are flagged before
  download. Downloads use 4 parallel range requests per file, can be paused and resumed (they survive app restarts),
  and every part is checked against the published SHA-256. GGUFs can also be imported through the system file
  picker. Models stored in shared storage (FUSE, where direct reads are unusable) are marked, with a one-tap move to
  app storage.
* **Activity.** Live tok/s, battery W, CPU %, storage MB/s, thermal status with the 10 s headroom forecast, and
  J/token on battery, plus the app's and the engine's log.
* **Settings.**
  * Per model: sampling, context length (Auto by default) and system prompt.
  * Per device: CPU or GPU, the power profile (Fastest / Balanced / Coolest), the memory budget with its
    recommendation, and the idle release time. On a computer the memory budget is 5 GiB at most, whatever its RAM,
    so the assistant stays a light app next to your work; larger models stream from the SSD. On an M4 Pro,
    Qwen3.6-35B-A3B answers at 9.7–10.2 tok/s within that budget (4.8 GB resident), against 11.1–11.5 tok/s with
    all 22 GB in RAM. The model stays loaded by default, so the command bar answers at once.
  * The permissions Liyab uses, and why.
  * An **Experimental** section exposes the engine's lossy or measurement-only options.
* **The living flame.** The Liyab mark is drawn live as the assistant's presence: see
  [Look and feel](design.md#look-and-feel).

```bash
scripts/build_flutter_app.sh --install --serial PHONE_SERIAL   # libliyab + release APK; never interrupts the app while it is in use
scripts/build_flutter_app.sh --macos     # libliyab (static) + Liyab.app; then `flutter run -d macos` works too
cd app && flutter analyze && flutter test
```

The application id is `com.liyab.chat`. Models downloaded in the app live in app-private storage (plain f2fs),
where direct reads work at full speed.

---
