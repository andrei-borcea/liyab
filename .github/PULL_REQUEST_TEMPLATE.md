## What and why

<!-- What this changes, and the problem it solves. -->

## How it was tested

<!-- Host ctest, device tests, app checks. For performance changes: device, model, setup, and before/after from
     alternating A/B runs. For lossy changes: liyab-kl results. -->

## Checklist

- [ ] `ctest --test-dir build --output-on-failure` passes (and the device suites for engine or kernel changes)
- [ ] New behaviour is covered by a test
- [ ] README, `include/liyab/` documentation and `--help` texts describe the code as it is
- [ ] Lossy behaviour is opt-in and measured with `liyab-kl`
- [ ] C structs only gained fields at the end
