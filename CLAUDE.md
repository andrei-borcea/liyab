# Liyab — working rules

## Documentation stays in sync with code (mandatory)

Every code change must update `README.md` and any related documentation
(`docs/`, header comments of public APIs in `include/liyab/`, build-script
`--help` text) in the same change whenever the change affects what they
describe: public API, C ABI, build flags/options, supported models/formats,
backends, scripts, test commands, or behavior. A change is not complete until
the docs describe the code as it is. Never document features that are not
implemented.

## Commits and pushes (fundamental rule)

The repository owner is the only author. Never add Claude (or any AI) as an
author or co-author: no `Co-Authored-By: Claude ...` trailers, no
"Generated with Claude Code" lines in commit messages or PR descriptions.

## Branches and identity (mandatory)

- Branch names follow `feat/<short-topic>` (e.g. `feat/vulkan-backend`); never
  tool-generated prefixes such as `emdash/...`.
- This is a personal repository: commit as
  `Andrei Borcea <72470237+andrei-borcea@users.noreply.github.com>` (set in the
  repo-local git config) and push through the personal SSH alias
  `git@github.com-andrei:andrei-borcea/liyab.git`. Never use the work
  (Smartpricing) email or key here.

## Language and in-code documentation (mandatory)

- Code, identifiers, comments and documentation are always written in English.
- Code is documented: every public header in `include/liyab/` describes the
  purpose, contract (preconditions, ownership, thread-safety, error cases) of
  each type and function; non-obvious implementation choices (memory layout,
  invariants, platform quirks, performance trade-offs) carry a comment
  explaining *why*.

## Build & test

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release            # add -DLIYAB_ENABLE_EXPERIMENTAL=ON for experimental modules
cmake --build build
ctest --test-dir build --output-on-failure
scripts/build_android.sh --experimental                             # then push build/android/arm64-v8a/test_* via adb
```
