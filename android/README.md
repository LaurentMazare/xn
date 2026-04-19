# Pocket TTS — Android demo

Minimal Android app that runs the `pocket-tts` Rust inference library
on-device and reports the same stats as the CLI example (RTF, average
backbone step time, total elapsed, first-audio latency, peak RSS).

The native core is built from [`../pocket-tts-android/`](../pocket-tts-android)
as `libptts_android.so` and loaded via JNI from `sh.gradium.ptts.Ptts`.

## One-time setup

1. Install the Android SDK + NDK (r26 or newer). Export `ANDROID_HOME` and
   `ANDROID_NDK_HOME`.
2. `rustup target add aarch64-linux-android`
3. `cargo install cargo-ndk`

## Build & install

```bash
./build.sh                 # debug build
BUILD_TYPE=release ./build.sh
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Runtime

First launch streams the following from Hugging Face
`kyutai/pocket-tts` into the app's private `files/models/` dir
(~240 MB, one-time). Subsequent launches skip the download.

- `tts_b6369a24.safetensors` — model weights
- `tokenizer.model` — SentencePiece Unigram vocabulary
- `embeddings/{voice}.safetensors` — 8 voice embeddings

Tap **Generate** to run inference on CPU f32. Audio is streamed to
`AudioTrack` chunk-by-chunk; the stats card fills in once generation
finishes.

## Stats parity with the CLI

The session-level stats exposed by `Ptts.stats()` map 1:1 to the CLI
example (`pocket-tts/examples/pocket_tts.rs`):

| Kotlin field       | CLI counterpart                     |
| ------------------ | ----------------------------------- |
| `rtf`              | `generated Ds in Ts (RTF=…)`        |
| `avgStepMs`        | `average backbone step time: …ms`   |
| `durationS`        | audio duration (PCM samples / SR)   |
| `totalElapsedS`    | wall clock from start to last chunk |
| `firstAudioS`      | (added) latency to first PCM chunk  |
| `peakRssMb`        | `peak RSS: … MB`                    |
