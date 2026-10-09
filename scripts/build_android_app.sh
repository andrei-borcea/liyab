#!/usr/bin/env bash
# Builds the Liyab Chat demo APK (android/chat) without Gradle, using the
# Android SDK build-tools directly, and optionally installs it.
#
#   scripts/build_android_app.sh [--install] [--push-model FILE.gguf]
#
# Requires: Android SDK (platforms/android-35, build-tools 35+), NDK, JDK 17,
# cmake. SDK lookup: $ANDROID_HOME, $ANDROID_SDK_ROOT, ~/Library/Android/sdk,
# ~/Android/Sdk, ~/android-sdk. Output: build/android-app/liyab-chat.apk
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ROOT/android/chat"
OUT="$ROOT/build/android-app"
INSTALL=0
MODEL=""

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install) INSTALL=1; shift ;;
    --push-model) MODEL="$2"; INSTALL=1; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

SDK=""
for c in "${ANDROID_HOME:-}" "${ANDROID_SDK_ROOT:-}" "$HOME/Library/Android/sdk" "$HOME/Android/Sdk" "$HOME/android-sdk"; do
  if [[ -n "$c" && -d "$c/platforms" ]]; then SDK="$c"; break; fi
done
[[ -n "$SDK" ]] || { echo "error: Android SDK not found (set ANDROID_HOME)" >&2; exit 1; }
PLATFORM_JAR="$(ls -d "$SDK"/platforms/android-3[5-9]/android.jar 2>/dev/null | sort -V | tail -n 1)"
BUILD_TOOLS="$(ls -d "$SDK"/build-tools/3[5-9]*/ 2>/dev/null | sort -V | tail -n 1)"
BUILD_TOOLS="${BUILD_TOOLS%/}"
[[ -f "$PLATFORM_JAR" && -x "$BUILD_TOOLS/aapt2" ]] || { echo "error: need platforms/android-35+ and build-tools 35+" >&2; exit 1; }
NDK="$(ls -d "$SDK"/ndk/*/ 2>/dev/null | sort -V | tail -n 1)"; NDK="${NDK%/}"
NDK="${ANDROID_NDK_HOME:-$NDK}"
CXX="$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android28-clang++ | head -n 1)"

echo "==> libliyab (experimental build: KV dedup for multi-turn chat)"
"$ROOT/scripts/build_android.sh" --experimental --no-tests --ndk "$NDK" >/dev/null
LIB_DIR="$ROOT/build/android/arm64-v8a"

rm -rf "$OUT" && mkdir -p "$OUT/lib/arm64-v8a" "$OUT/classes" "$OUT/dex"
echo "==> JNI bridge"
"$CXX" -std=c++20 -O2 -shared -fPIC -fvisibility=hidden -I"$ROOT/include" "$APP/jni/liyab_jni.cpp" \
  -L"$LIB_DIR" -lliyab -static-libstdc++ -Wl,-z,max-page-size=16384 -Wl,--gc-sections \
  -o "$OUT/lib/arm64-v8a/libliyab_chat.so"
cp "$LIB_DIR/libliyab.so" "$OUT/lib/arm64-v8a/"

echo "==> Java -> dex"
javac -source 11 -target 11 -Xlint:-options -encoding UTF-8 -classpath "$PLATFORM_JAR" -d "$OUT/classes" \
  $(find "$APP/java" -name '*.java')
"$BUILD_TOOLS/d8" --release --min-api 28 --lib "$PLATFORM_JAR" --output "$OUT/dex" $(find "$OUT/classes" -name '*.class')

echo "==> APK"
# Resources: the launcher icon (adaptive vector icon, res/).
rm -rf "$OUT/res.zip"
"$BUILD_TOOLS/aapt2" compile --dir "$APP/res" -o "$OUT/res.zip"
"$BUILD_TOOLS/aapt2" link -o "$OUT/unsigned.apk" --manifest "$APP/AndroidManifest.xml" -I "$PLATFORM_JAR" \
  --min-sdk-version 28 --target-sdk-version 35 "$OUT/res.zip"
(cd "$OUT/dex" && zip -q -j "$OUT/unsigned.apk" classes.dex)
(cd "$OUT" && zip -q -r unsigned.apk lib)
"$BUILD_TOOLS/zipalign" -f -P 16 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"

KEYSTORE="$OUT/../liyab-debug.keystore"   # build/liyab-debug.keystore, reused across builds
if [[ ! -f "$KEYSTORE" ]]; then
  keytool -genkeypair -keystore "$KEYSTORE" -storepass android -keypass android -alias liyab \
    -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Liyab Debug" >/dev/null 2>&1
fi
"$BUILD_TOOLS/apksigner" sign --ks "$KEYSTORE" --ks-pass pass:android --key-pass pass:android \
  --out "$OUT/liyab-chat.apk" "$OUT/aligned.apk"
rm -f "$OUT/unsigned.apk" "$OUT/aligned.apk"
echo "==> Built $OUT/liyab-chat.apk ($(du -h "$OUT/liyab-chat.apk" | cut -f1))"

if [[ "$INSTALL" == 1 ]]; then
  adb install -r "$OUT/liyab-chat.apk"
  if [[ -n "$MODEL" ]]; then
    DEST=/sdcard/Android/data/com.liyab.chat/files
    adb shell mkdir -p "$DEST"
    adb push "$MODEL" "$DEST/"
  fi
  adb shell am start -n com.liyab.chat/.HomeActivity >/dev/null
  echo "==> Installed and launched com.liyab.chat"
fi
