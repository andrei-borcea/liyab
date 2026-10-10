# Contributing to Liyab

Thank you for helping make large open models run well on the devices people already own. This guide explains how
the project works and what a change needs before it can be merged.

## What Liyab cares about

* **Any device, any model.** Optimizations are generic and self-calibrating: no code path tuned for one SoC or one
  model. Kernels are written per instruction set (ARM64 NEON, dot product, i8mm; Metal; Vulkan); platform
  differences (thermal APIs, performance hints, memory limits) live behind a thin adaptive layer.
* **Lossless by default.** Anything that changes the output is opt-in, labelled lossy, and measured with `liyab-kl`
  against the lossless path.
* **Measured, not assumed.** Speed claims come from alternating A/B runs on a real device. Ideas that did not pay
  off are worth documenting too.
* **Private by construction.** The engine never touches the network. The app uses it only to download models.

## Getting started

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLIYAB_ENABLE_EXPERIMENTAL=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The tests generate small GGUF models on the fly, so nothing needs downloading. For Android, `scripts/build_android.sh
--experimental` builds the library, `liyab-cli` and the tests for `arm64-v8a` (see the README's Testing section for
running them over `adb`). The app builds with `scripts/build_flutter_app.sh` (Android) or
`scripts/build_flutter_app.sh --macos`; check it with
`cd app && flutter analyze && flutter test`.

## Good places to start

* **Model support.** An architecture made of the existing pieces (softmax attention or Gated DeltaNet; dense or MoE
  FFN) needs only a traits row; the rest is detected from the tensors. Open a *Model support* issue with the GGUF's
  `general.architecture` and tensor names first.
* **Kernels.** Faster NEON / i8mm paths for formats that still use the generic decode (TQ1_0, NVFP4, lattice
  I-quants), x86 SIMD kernels, Vulkan and Metal kernels. Every kernel is checked against the reference decoder.
* **Measurements on your device.** `liyab-cli --device` and a run of `liyab-cli` with the phase breakdown on a phone
  we have not tested are valuable on their own: open an issue with the output.
* **The app.** Accessibility, localization, and the items in the README's roadmap.

## Before you open a pull request

1. **Tests pass** on the host (`ctest`), and on an ARM64 device for kernel or engine changes.
2. **New behaviour has a test.** Engine features are tested for exactness where possible (for example, a cached or
   streamed path must give the same output as the plain one).
3. **Docs change in the same commit as the code**: `README.md`, the documentation in `include/liyab/` (purpose,
   preconditions, ownership, thread safety, error cases), and any `--help` text. Never document what is not
   implemented.
4. **Performance changes include numbers**: device, model, setup, and before/after from alternating runs.
5. **English** for code, identifiers, comments and documentation.
6. **C ABI stability**: fields are only appended to the C structs, never reordered or removed.

Keep pull requests focused: one topic per branch (`feat/<topic>`), with a description of what changed, why, and how
it was tested.

## Reporting bugs and security issues

Use the issue templates for bugs, features and model support. Security problems (for example in GGUF parsing of
untrusted files or in the app's local API) should be reported privately, as described in [SECURITY.md](SECURITY.md).

## Code of conduct

Everyone taking part in the project is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
Code adapted from other projects must be compatible with it and listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
