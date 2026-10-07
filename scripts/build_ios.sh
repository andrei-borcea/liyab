#!/usr/bin/env bash
# Builds Liyab.xcframework (static library + headers + Swift module map) for
# iOS devices (arm64) and the iOS Simulator (arm64 + x86_64).
#
#   scripts/build_ios.sh [--min-ios 16.0] [--build-type Release] [--no-metal]
#                        [--no-simulator] [--out DIR]
#
# Output: <out>/Liyab.xcframework (default out: build/ios)
# In Swift:  import Liyab   // exposes the C API from liyab_c_api.h
# The module map auto-links Metal, Foundation and libc++.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIN_IOS="16.0"
BUILD_TYPE="Release"
METAL=ON
SIMULATOR=ON
OUT="$ROOT/build/ios"

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-ios) MIN_IOS="$2"; shift 2 ;;
    --build-type) BUILD_TYPE="$2"; shift 2 ;;
    --no-metal) METAL=OFF; shift ;;
    --no-simulator) SIMULATOR=OFF; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ "$(uname)" == "Darwin" ]] || { echo "error: iOS builds require macOS with Xcode" >&2; exit 1; }
command -v xcodebuild >/dev/null || { echo "error: xcodebuild not found (install Xcode)" >&2; exit 1; }
command -v cmake >/dev/null || { echo "error: cmake not found" >&2; exit 1; }
GENERATOR=()
command -v ninja >/dev/null && GENERATOR=(-G Ninja)

# build <name> <sdk> <arch>: one static libliyab.a per SDK/arch slice.
build() {
  local name="$1" sdk="$2" arch="$3"
  local dir="$OUT/$name"
  echo "==> $name ($sdk, $arch)"
  cmake -S "$ROOT" -B "$dir" "${GENERATOR[@]}" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="$sdk" \
    -DCMAKE_OSX_ARCHITECTURES="$arch" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MIN_IOS" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DLIYAB_BUILD_SHARED=OFF \
    -DLIYAB_BUILD_TESTS=OFF \
    -DLIYAB_BUILD_CLI=OFF \
    -DLIYAB_USE_METAL="$METAL" >/dev/null
  cmake --build "$dir" --parallel
}

build device-arm64 iphoneos arm64
LIBS=(-library "$OUT/device-arm64/libliyab.a" -headers "$OUT/headers")

if [[ "$SIMULATOR" == ON ]]; then
  # Separate builds: the arm64 slice uses NEON flags that x86_64 cannot take.
  build sim-arm64 iphonesimulator arm64
  build sim-x86_64 iphonesimulator x86_64
  mkdir -p "$OUT/sim-universal"
  lipo -create "$OUT/sim-arm64/libliyab.a" "$OUT/sim-x86_64/libliyab.a" -output "$OUT/sim-universal/libliyab.a"
  LIBS+=(-library "$OUT/sim-universal/libliyab.a" -headers "$OUT/headers")
fi

# Public headers + a module map so Swift can `import Liyab`.
rm -rf "$OUT/headers"
mkdir -p "$OUT/headers"
cp -R "$ROOT/include/liyab" "$OUT/headers/"
FRAMEWORK_LINKS='  link framework "Foundation"'
[[ "$METAL" == ON ]] && FRAMEWORK_LINKS+=$'\n  link framework "Metal"'
cat > "$OUT/headers/module.modulemap" <<EOF
module Liyab {
  header "liyab/liyab_c_api.h"
  export *
  link "c++"
$FRAMEWORK_LINKS
}
EOF

rm -rf "$OUT/Liyab.xcframework"
if ! xcodebuild -create-xcframework "${LIBS[@]}" -output "$OUT/Liyab.xcframework" >"$OUT/xcframework.log" 2>&1; then
  echo "error: xcodebuild -create-xcframework failed (log: $OUT/xcframework.log)" >&2
  echo "       the static libraries are built in $OUT/*/libliyab.a" >&2
  if grep -q "runFirstLaunch" "$OUT/xcframework.log"; then
    echo "       Xcode reports missing components: run 'sudo xcodebuild -runFirstLaunch' and retry" >&2
  fi
  exit 1
fi
echo "==> Built $OUT/Liyab.xcframework"
find "$OUT/Liyab.xcframework" -name "*.a" -exec ls -lh {} \; | awk '{print "    " $5 "  " $9}'
