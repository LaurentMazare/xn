#!/usr/bin/env bash
# Build the pocket-tts Android demo:
#   1. Cross-compile pocket-tts-android to aarch64 Android
#   2. Run Gradle to assemble the APK
#
# Prerequisites:
#   - rustup target add aarch64-linux-android
#   - EITHER  cargo install cargo-ndk
#     OR      export ANDROID_NDK_HOME=/path/to/ndk (direct cross-compile path)
#   - Android SDK (ANDROID_HOME)
#
# Env vars:
#   BUILD_TYPE   debug (default) | release
#   MIN_SDK      minimum Android API level for the .so (default 26)
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_TYPE="${BUILD_TYPE:-debug}"
MIN_SDK="${MIN_SDK:-26}"
JNILIBS="android/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$JNILIBS"

if [[ "$BUILD_TYPE" == "release" ]]; then
    CARGO_PROFILE="android-release"
    PROFILE_DIR="android-release"
    CARGO_FLAGS=("--profile" "$CARGO_PROFILE")
    GRADLE_TASK="assembleRelease"
else
    PROFILE_DIR="debug"
    CARGO_FLAGS=()
    GRADLE_TASK="assembleDebug"
fi

echo "==> Cross-compiling pocket-tts-android (profile: $PROFILE_DIR)..."
if command -v cargo-ndk >/dev/null 2>&1; then
    cargo ndk \
        --target arm64-v8a \
        --platform "$MIN_SDK" \
        -o android/app/src/main/jniLibs \
        build -p ptts-android "${CARGO_FLAGS[@]}"
elif [[ -n "${ANDROID_NDK_HOME:-}" ]]; then
    TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64"
    export PATH="$TOOLCHAIN/bin:$PATH"
    export CC_aarch64_linux_android="$TOOLCHAIN/bin/aarch64-linux-android${MIN_SDK}-clang"
    export AR_aarch64_linux_android="$TOOLCHAIN/bin/llvm-ar"
    export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC_aarch64_linux_android"
    cargo build -p ptts-android --target aarch64-linux-android "${CARGO_FLAGS[@]}"
    cp "target/aarch64-linux-android/$PROFILE_DIR/libptts_android.so" "$JNILIBS/"
else
    echo "ERROR: need either cargo-ndk (cargo install cargo-ndk) or ANDROID_NDK_HOME." >&2
    exit 1
fi

echo "==> Running gradle $GRADLE_TASK..."
cd android
./gradlew "$GRADLE_TASK"

echo "==> Build complete. APK(s):"
find app/build/outputs/apk -name '*.apk'
