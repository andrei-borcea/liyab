#!/usr/bin/env bash
# Builds the Liyab Flutter app (app/) for Android: libliyab with the NDK, then
# the release APK, and optionally installs it.
#
#   scripts/build_flutter_app.sh [--install]
#
# Requires: Flutter, Android SDK (platform 36), NDK, JDK 17, cmake. The APK is
# signed with build/liyab-debug.keystore (created here if missing): keep it, an
# APK signed with another key cannot update the installed app.
# --install never interrupts the app: it waits while Liyab is on screen or runs
# a foreground service (a model download), then installs, which restarts it.
# Output: app/build/app/outputs/flutter-apk/app-release.apk
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL=0
usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install) INSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

echo "==> libliyab"
"$ROOT/scripts/build_android.sh" --no-tests >/dev/null
JNI="$ROOT/app/android/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$JNI"
cp "$ROOT/build/android/arm64-v8a/libliyab.so" "$JNI/"

KEYSTORE="$ROOT/build/liyab-debug.keystore"
if [[ ! -f "$KEYSTORE" ]]; then
  mkdir -p "$ROOT/build"
  keytool -genkeypair -keystore "$KEYSTORE" -storepass android -keypass android -alias liyab \
    -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Liyab Debug" >/dev/null 2>&1
fi

echo "==> APK"
(cd "$ROOT/app" && flutter build apk --release --target-platform android-arm64)
APK="$ROOT/app/build/app/outputs/flutter-apk/app-release.apk"
echo "==> Built $APK ($(du -h "$APK" | cut -f1))"

if [[ "$INSTALL" == 1 ]]; then
  # The process alone is no sign of use: the notification listener and a
  # backgrounded chat keep it alive indefinitely.
  in_use() {
    adb shell dumpsys activity activities | grep -E 'ResumedActivity' | grep -q com.liyab.chat ||
      adb shell dumpsys activity services com.liyab.chat | grep -q 'isForeground=true'
  }
  while in_use; do
    echo "    Liyab is in use on the phone; waiting for it to go to the background…"
    sleep 10
  done
  adb install -r "$APK"
  adb shell monkey -p com.liyab.chat -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
  echo "==> Installed and launched com.liyab.chat"
fi
