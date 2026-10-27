# Liyab app

The Flutter client for Liyab, a private assistant backed by the repository’s C++20 GGUF engine. Models run on the device; the app manages downloads, conversations, device tools and an authenticated local API.

See the [project README](../README.md#the-liyab-app) for app behavior, the [model coverage](../README.md#supported-models) for compatibility, and [model descriptions](../docs/model-descriptions.md) for numerical validation and variant limits. The app obtains supported architecture names from the linked engine through `liyab_supported_architectures`; there is no separate app architecture allowlist. The loader validates each model’s actual tensors, metadata and quantization.

## Chat compatibility

The app replays previous replies exactly to preserve the engine's context. It uses the model's author template when each new prompt continues the preceding prompt and reply. Recognized ChatML and Llama 3 vocabulary fallbacks preserve existing Qwen and Llama conversations. Zephyr is the legacy fallback only when a model has no author template.

A present incompatible template without a recognized fallback disables chat with an explicit error. The model stays loaded for raw-completion API requests or pooling. In particular, engine/CLI Harmony rendering does not yet establish Flutter Harmony replay support. Embedding models and rerankers use their pooling APIs and do not need a chat format.

## Build

Run the build scripts from the repository root. Use a Flutter SDK whose bundled Dart version satisfies [pubspec.yaml](pubspec.yaml). Android also needs the Android SDK, NDK, JDK 17 and CMake; macOS needs Xcode and CMake. The project README documents native toolchain setup and CMake options.

```sh
# Android arm64: build the experimental engine, package it, then build a release APK.
CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build_flutter_app.sh

# macOS Apple silicon: link the native engine statically into the release app.
CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build_flutter_app.sh --macos
```

Outputs:

- Android: `app/build/app/outputs/flutter-apk/app-release.apk`.
- macOS: `app/build/macos/Build/Products/Release/Liyab.app`.

Android updates must use the same `build/liyab-debug.keystore` as the installed app. Restore the existing key before building an update; a newly generated key cannot update an installation signed with another key. The installation mode waits while Liyab is on screen or running a foreground work service. See `scripts/build_flutter_app.sh --help` for installation options.

Android opens `libliyab.so` from the APK. macOS and iOS resolve C ABI symbols from the application process. The repository includes an iOS project; native framework instructions are in the [project quick start](../README.md#quick-start).

## Test

From `app/`:

```sh
flutter pub get
flutter test
```

The suite checks chat templates, tool-call grammars and parsing, conversation replay, typing ahead, widgets, and the OpenAI, Anthropic and gRPC APIs over local sockets. Tests that need a native engine or real GGUF report skips when their prerequisites are absent.

To include the host engine, first build the shared library from the repository root:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLIYAB_ENABLE_EXPERIMENTAL=ON
cmake --build build -j 4
cd app
LIYAB_LIBRARY="$PWD/../build/libliyab.dylib" flutter test
```

`LIYAB_LIBRARY` selects the native shared library for host tests; use the corresponding library path for the host platform. The macOS path above also enables native grammar checks. Optional real-model tests use `LIYAB_TEST_MODEL` for engine lanes and `LIYAB_TOOL_MODEL` for comma-separated GGUF paths in tool-call checks. See the [project correctness guide](../README.md#correctness) for their full commands and validation scope.

Phone regression runs are managed from the repository root by `scripts/phone_regression.py`. They compare device unit tests and CLI/app replies against recorded runs. The app part requires the Local API token through `LIYAB_API_TOKEN`, the matching app/CLI model, and the CPU backend. Follow the [phone regression instructions](../README.md#regression-run-on-the-phone); installation, screen state and device selection matter for valid measurements.

## Source layout

- `lib/engine/`: C ABI bindings and engine service.
- `lib/chat/`: chat widgets and template integration; `lib/state/` owns conversation state and model settings.
- `lib/models/` and `lib/state/`: model catalog, downloads, persistence and app state.
- `lib/agent/`: device tools and their constrained call formats.
- `lib/api/`: authenticated local API and protocol adapters.
- `lib/assist/` and `lib/activity/`: assistant entry points and activity UI.
- `test/`: UI, parser, API and optional native-engine tests.

Engine contracts and architecture implementation live outside this directory, in `include/liyab/` and `src/`. Keep app documentation linked to those contracts when engine capabilities change.
