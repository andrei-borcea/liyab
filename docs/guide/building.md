# Build and run

[← README](../../README.md) · [Documentation index](README.md)

Commands and source paths in examples are relative to the repository root.

## On this page

- [Quick start](#quick-start)
- [Windows (x64)](#windows-x64)
- [CMake options](#cmake-options)

---

## Quick start

Requirements: CMake ≥ 3.22, a C++20 compiler (Clang 17+ / Apple Clang 15+), Ninja recommended. Android: NDK r26+.
iOS: Xcode 15+. Windows: Visual Studio 2022 with its C++ tools (MSVC, or its clang-cl). The app: Flutter 3.x,
Android SDK platform 36, JDK 17.

```bash
# Build and test on the host (macOS / Linux)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # add -DLIYAB_ENABLE_EXPERIMENTAL=ON for experimental modules
cmake --build build
ctest --test-dir build --output-on-failure

# x86-64 build on an Apple-silicon Mac, run under Rosetta 2. Rosetta executes AVX2 but reports it in CPUID only
# with ROSETTA_ADVERTISE_AVX=1; without it the scalar kernels run
cmake -S . -B build-x86 -G Ninja -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_BUILD_TYPE=Release
cmake --build build-x86
ROSETTA_ADVERTISE_AVX=1 ctest --test-dir build-x86 --output-on-failure

# Run a model
./build/liyab-cli -m model.gguf -p "Explain unified memory in one paragraph." -n 128
./build/liyab-cli --device                                 # what Liyab sees: SoC, accelerators, backend ranking
./build/liyab-cli --doctor -m model.gguf                   # will the model fit here, and how fast can it decode at most
```

```bash
# Android (arm64-v8a): libliyab.so, liyab-cli, liyab-kernels and the tests in build/android/arm64-v8a
scripts/build_android.sh                     # --experimental, --abi, --api, --ndk, --no-vulkan, ... (see --help)

# iOS: build/ios/Liyab.xcframework (device arm64 + simulator arm64/x86_64), importable from Swift as `Liyab`
scripts/build_ios.sh

# The Liyab app: builds libliyab first, then the release APK; --install waits while the app is in use
scripts/build_flutter_app.sh --install --serial PHONE_SERIAL

# The Liyab app for macOS (Apple silicon): app/build/macos/Build/Products/Release/Liyab.app
scripts/build_flutter_app.sh --macos
```

The Android NDK is located from `--ndk`, `$ANDROID_NDK_HOME`, or the newest NDK in the usual SDK locations.
The Android build honors `CMAKE_BUILD_PARALLEL_LEVEL`; set it to limit concurrent jobs, for example
`CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build_android.sh --experimental`. Unset uses the build generator's default.
The `.so` is linked with 16 KB page alignment, which Android 15+ devices with 16 KB pages require. If `xcodebuild`
reports missing components, run `sudo xcodebuild -runFirstLaunch` once.

### Windows (x64)

The engine library (`liyab.dll`), `liyab-cli`, the tests and the tools are ported to Windows x64. The port has not
been compiled or run on Windows yet (see [Known limitations](development.md#known-limitations)). In an "x64 Native Tools Command
Prompt for VS 2022":

```bat
rem clang-cl (Visual Studio's "C++ Clang tools"); leave out the two compiler options to build with cl.exe
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLIYAB_ENABLE_EXPERIMENTAL=ON -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl
cmake --build build
ctest --test-dir build --output-on-failure
build\liyab-cli.exe --device
```

What the port does differently from the POSIX platforms:

* **Mapping.** The GGUF is mapped with `CreateFileMappingW` / `MapViewOfFile` as one view. `PrefetchVirtualMemory`
  and `VirtualUnlock` (which takes unlocked pages out of the working set) stand in for `madvise`'s `WILLNEED` and
  `DONTNEED`; a mapped view has no `SEQUENTIAL` or `RANDOM` hint. The file stays deletable while it is mapped.
* **Direct reads.** `DirectFile` opens with `FILE_FLAG_NO_BUFFERING` (offsets, lengths and buffers in 4 KiB
  multiples, as with `O_DIRECT`) and reads with overlapped `ReadFile`, one event per request: requests made through
  a synchronous handle would run one after another. A volume that refuses an unbuffered read of the first block gets
  buffered reads.
* **Threads.** The pool's default size counts the physical cores of the highest efficiency class (an Intel hybrid's
  P-cores; every core of a homogeneous CPU), SMT siblings excluded. On machines with more than 64 logical
  processors, which Windows splits into processor groups, each worker is moved to a group so that the pool spreads
  over them. Workers wait with `PAUSE` and are named with `SetThreadDescription`.
* **Device.** The CPU name comes from the registry, memory from `GlobalMemoryStatusEx`, AVX2/FMA/F16C from
  `__cpuidex` / `_xgetbv` under MSVC and clang-cl.
* **Text.** Every path is UTF-8, as on the other platforms: the library converts paths to UTF-16 for Windows.
  `liyab-cli` reads its UTF-16 command line and switches the console to UTF-8 while it runs; the executables
  declare UTF-8 as their code page in their manifest (Windows 10 1903 and later).
* **Linking.** `liyab.dll` exports the C and the C++ API; CMake's `liyab` target defines `LIYAB_DLL_IMPORT` for the
  code that links it. An app that loads the DLL at run time (FFI) only needs the C API's exported names.
* **Not available on Windows:** the GPU and NPU backends (`LIYAB_USE_VULKAN`, `LIYAB_USE_QNN`,
  `LIYAB_USE_NEUROPILOT` stop the configuration), thermal readings (Windows has no thermal status, headroom forecast
  or sensor zones Liyab reads, so the steering of [Heat and power](inference-features.md#heat-and-power) stays idle while the profiles'
  rate caps and core shares still apply), and locking the expert cache in RAM (`VirtualLock` is capped by the
  working-set minimum).

### CMake options

| Option | Default | Effect |
| :--- | :--- | :--- |
| `LIYAB_USE_NEON` | ON on arm64 | NEON intrinsics kernels |
| `LIYAB_ARM_DOTPROD` | ON on arm64 | `-march=armv8.2-a+dotprod+fp16` on Android/iOS/Linux (SDOT) |
| `LIYAB_USE_AVX2` | ON on x86-64 | AVX2 + FMA + F16C kernels, used when CPUID reports them (the library itself keeps the x86-64 baseline) |
| `LIYAB_USE_METAL` | ON on Apple | Metal GPU backend |
| `LIYAB_USE_VULKAN` | ON on Android | Vulkan GPU backend and probe (loader opened with `dlopen`; shaders need `glslc` from the NDK or the Vulkan SDK); not available on Windows |
| `LIYAB_USE_QNN` / `LIYAB_USE_NEUROPILOT` | ON on Android | NPU runtime probes (no SDK needed); not available on Windows |
| `LIYAB_ENABLE_EXPERIMENTAL` | OFF | [Experimental modules](development.md#experimental-modules) and `test_experimental` |
| `LIYAB_BUILD_SHARED` | ON | Shared (`.so`/`.dylib`/`.dll`) or static library |
| `LIYAB_BUILD_TESTS` / `LIYAB_BUILD_CLI` | ON | Unit tests, benchmarks and `liyab-cli` |

---
