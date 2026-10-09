#!/usr/bin/env bash
# Builds libliyab.so (+ liyab-cli and the unit tests) for Android with the NDK.
#
#   scripts/build_android.sh [--abi arm64-v8a] [--api 28] [--build-type Release]
#                            [--ndk PATH] [--no-vulkan] [--no-qnn] [--no-neuropilot]
#                            [--no-tests] [--experimental] [--out DIR]
#
# Output: <out>/<abi>/libliyab.so, liyab-cli, liyab-bench, liyab-kl, test_* (default out: build/android)
# NDK lookup order: --ndk, $ANDROID_NDK_HOME, $ANDROID_NDK_ROOT, newest NDK under
# $ANDROID_HOME/ndk, $ANDROID_SDK_ROOT/ndk, ~/Library/Android/sdk/ndk, ~/Android/Sdk/ndk.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABI="arm64-v8a"
API="28"
BUILD_TYPE="Release"
OUT="$ROOT/build/android"
NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
VULKAN=ON
QNN=ON
NEUROPILOT=ON
TESTS=ON
EXPERIMENTAL=OFF

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --abi) ABI="$2"; shift 2 ;;
    --api) API="$2"; shift 2 ;;
    --build-type) BUILD_TYPE="$2"; shift 2 ;;
    --ndk) NDK="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-vulkan) VULKAN=OFF; shift ;;
    --no-qnn) QNN=OFF; shift ;;
    --no-neuropilot) NEUROPILOT=OFF; shift ;;
    --no-tests) TESTS=OFF; shift ;;
    --experimental) EXPERIMENTAL=ON; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$NDK" ]]; then
  for sdk in "${ANDROID_HOME:-}" "${ANDROID_SDK_ROOT:-}" "$HOME/Library/Android/sdk" "$HOME/Android/Sdk" "$HOME/android-sdk"; do
    if [[ -n "$sdk" && -d "$sdk/ndk" ]]; then
      NDK="$(ls -d "$sdk"/ndk/*/ 2>/dev/null | sort -V | tail -n 1)"
      NDK="${NDK%/}"
      [[ -n "$NDK" ]] && break
    fi
  done
fi
TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"
if [[ -z "$NDK" || ! -f "$TOOLCHAIN" ]]; then
  echo "error: Android NDK not found; pass --ndk or set ANDROID_NDK_HOME" >&2
  exit 1
fi
if [[ "$ABI" != "arm64-v8a" ]]; then
  echo "warning: Liyab's SIMD kernels target arm64-v8a; $ABI builds use scalar code" >&2
fi

command -v cmake >/dev/null || { echo "error: cmake not found" >&2; exit 1; }
GENERATOR=()
command -v ninja >/dev/null && GENERATOR=(-G Ninja)

BUILD_DIR="$OUT/$ABI"
echo "==> NDK:   $NDK"
echo "==> ABI:   $ABI (API $API, $BUILD_TYPE)"
echo "==> Flags: vulkan=$VULKAN qnn=$QNN neuropilot=$NEUROPILOT tests=$TESTS experimental=$EXPERIMENTAL"

cmake -S "$ROOT" -B "$BUILD_DIR" ${GENERATOR[@]+"${GENERATOR[@]}"} \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI="$ABI" \
  -DANDROID_PLATFORM="android-$API" \
  -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DLIYAB_USE_VULKAN="$VULKAN" \
  -DLIYAB_USE_QNN="$QNN" \
  -DLIYAB_USE_NEUROPILOT="$NEUROPILOT" \
  -DLIYAB_BUILD_TESTS="$TESTS" \
  -DLIYAB_ENABLE_EXPERIMENTAL="$EXPERIMENTAL"
cmake --build "$BUILD_DIR" --parallel

LIB="$BUILD_DIR/libliyab.so"
STRIP="$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip 2>/dev/null | head -n 1)"
if [[ "$BUILD_TYPE" == "Release" && -x "$STRIP" ]]; then
  "$STRIP" --strip-unneeded "$LIB"
fi

echo "==> Built:"
ls -lh "$LIB" "$BUILD_DIR/liyab-cli" 2>/dev/null | awk '{print "    " $5 "  " $9}'
cat <<EOF
==> Run on a device (USB debugging enabled):
    adb push $BUILD_DIR/liyab-cli $BUILD_DIR/test_* /data/local/tmp/
    adb shell 'cd /data/local/tmp && for t in test_*; do ./\$t || exit 1; done'
    adb shell /data/local/tmp/liyab-cli --device
EOF
