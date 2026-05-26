#!/usr/bin/env bash
# Build the xn-moshi ASR Android demo:
#   1. Cross-compile xn-moshi-android to aarch64 Android (libxn_moshi_android.so)
#   2. Assemble the APK with Gradle
#
# Prerequisites:
#   - rustup target add aarch64-linux-android
#   - EITHER  cargo install cargo-ndk
#     OR      export ANDROID_NDK_HOME=/path/to/ndk
#   - Android SDK (ANDROID_HOME)
#   - app/src/main/assets/{mimi.safetensors,lm.safetensors,tokenizer.model}
#
# Env vars:
#   BUILD_TYPE   debug (default) | release
#   MIN_SDK      minimum Android API level for the .so (default 26)
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_TYPE="${BUILD_TYPE:-debug}"
MIN_SDK="${MIN_SDK:-26}"
JNILIBS="xn-moshi-android/android/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$JNILIBS"

if [[ "$BUILD_TYPE" == "release" ]]; then
    PROFILE_DIR="android-release"
    CARGO_FLAGS=("--profile" "android-release")
    GRADLE_TASK="assembleRelease"
else
    PROFILE_DIR="debug"
    CARGO_FLAGS=()
    GRADLE_TASK="assembleDebug"
fi

echo "==> Cross-compiling xn-moshi-android (profile: $PROFILE_DIR)..."
if command -v cargo-ndk >/dev/null 2>&1; then
    cargo ndk \
        --target arm64-v8a \
        --platform "$MIN_SDK" \
        -o xn-moshi-android/android/app/src/main/jniLibs \
        build -p xn-moshi-android "${CARGO_FLAGS[@]}"
elif [[ -n "${ANDROID_NDK_HOME:-}" ]]; then
    TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64"
    export PATH="$TOOLCHAIN/bin:$PATH"
    export CC_aarch64_linux_android="$TOOLCHAIN/bin/aarch64-linux-android${MIN_SDK}-clang"
    export AR_aarch64_linux_android="$TOOLCHAIN/bin/llvm-ar"
    export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC_aarch64_linux_android"
    cargo build -p xn-moshi-android --target aarch64-linux-android "${CARGO_FLAGS[@]}"
    cp "target/aarch64-linux-android/$PROFILE_DIR/libxn_moshi_android.so" "$JNILIBS/"
else
    echo "ERROR: need either cargo-ndk (cargo install cargo-ndk) or ANDROID_NDK_HOME." >&2
    exit 1
fi

echo "==> Checking for model assets..."
for f in mimi.safetensors model.safetensors tokenizer.model config.json; do
    if [[ ! -s "xn-moshi-android/android/app/src/main/assets/$f" ]]; then
        echo "WARN: xn-moshi-android/android/app/src/main/assets/$f is missing or empty." >&2
        echo "      Place the four files from huggingface.co/..." >&2
        echo "      under xn-moshi-android/android/app/src/main/assets/." >&2
    fi
done

echo "==> Running gradle $GRADLE_TASK..."
cd xn-moshi-android/android
./gradlew "$GRADLE_TASK"

echo "==> Build complete. APK(s):"
find app/build/outputs/apk -name '*.apk' 2>/dev/null || true
