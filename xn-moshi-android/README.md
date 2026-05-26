# xn-moshi ASR — Android demo

On-device streaming ASR powered by the
[`xn-moshi`](../xn-moshi) crate. The mic captures 24 kHz mono PCM, frames
are fed to the LM 1920 samples at a time, and recognized words stream
to a `TextView` as they come out.

```
┌────────────────────────────────────────────────────┐
│  Compute dtype  [ q8_0 ▾ ]                         │
│                                                    │
│  [   Start   ]  [   Clear   ]                      │
│                                                    │
│  00:12.3                                           │
│  ┌──────────────────────────────────────────────┐  │
│  │ Hello world this is on device speech recogni-│  │
│  │ tion running entirely offline.               │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

## Layout

- `Cargo.toml` / `src/`  — the Rust JNI crate (builds as
  `libxn_moshi_android.so`). Exports `Java_sh_gradium_xnmoshi_Asr_*`
  symbols and erases the `BackendQ` dtype behind a `Box<dyn Session>`
  so Kotlin only deals with a `Long` handle.
- `android/`             — Gradle/Kotlin app project. Single activity,
  XML layout, JNI loader, mic capture.
- `build.sh`             — one-shot script: cross-compile the .so, then
  run Gradle.

## One-time setup

```bash
rustup target add aarch64-linux-android
cargo install cargo-ndk      # easiest path; the script also handles raw NDK
# Android SDK + NDK r26 or newer, ANDROID_HOME and ANDROID_NDK_HOME set
```

## Model files

Four files live in `android/app/src/main/assets/` (gitignored — bring
your own).

| In `assets/`        | Purpose                                       |
| ------------------- | --------------------------------------------- |
| `config.json`       | LM architecture + `asr_delay_in_tokens`       |
| `model.safetensors` | LM weights                                    |
| `mimi.safetensors`  | Mimi audio tokenizer weights                  |
| `tokenizer.model`   | SentencePiece pieces table                    |

Bundled size for this checkpoint is ~920 MB so the APK is large —
that's the price of fully on-device, no-download inference. The asset
extractor (`ModelAssets.kt`) copies them into `filesDir/models/` on
first launch and the Rust safetensors loader mmaps from there.

## Build & install

```bash
./build.sh                              # debug
BUILD_TYPE=release ./build.sh           # release (thin LTO, ~1.3 MB .so)
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

## Runtime

1. Pick a compute dtype from the spinner. `q8_0` is the default — good
   quality/speed tradeoff on flagship phones. `q4_0` is the fastest;
   `f32` the slowest and most accurate.
2. Tap **Start** — grants `RECORD_AUDIO` on first run. The timer starts
   ticking; recognized words stream into the transcript view as the
   model emits them.
3. **Stop** ends mic capture but keeps the accumulated transcript.
4. **Clear** wipes the transcript and resets the Rust-side token
   buffer so the next utterance starts fresh.

Changing the dtype while idle reloads the model from scratch
(~10–20 s on a Pixel 8a). The dtype spinner is disabled while
listening.

## How it maps to `xn-moshi/examples/moshi.rs`

The Rust JNI layer is a streaming version of the `Command::Asr` path:

| `examples/moshi.rs`                          | `xn-moshi-android/src/lib.rs`         |
| -------------------------------------------- | ------------------------------------- |
| `xn::Runner::new().dtype(dtype).run(asr, 0)` | `make_session(dtype, …)` switch       |
| `Asr::new(asr_delay_in_tokens, …)`           | `Asr::load(…)` via `AsrSession::new`  |
| `state.step_pcm(pcm, mask, …)` loop          | `Session::step` per audio frame       |
| `sp.decode_piece_ids(&tokens)`               | pure-Rust `SpDecoder::decode_piece_ids` |
| stdout `print!(new chars)`                   | returned `String` → JNI → `TextView`  |

The reference loop pads 2 frames of silence before audio and 2.5 s
after to flush the model's 2.5 s ASR delay. The streaming Android path
doesn't pre-pad (the mic is silent before you talk anyway) but the
trailing flush isn't there either — words emitted in the last ~2.5 s
of speech will only appear after you continue making sound, or after a
short period of silence captured by the mic.

## CPU dtype support

The Rust workspace's CPU backend supports the GGML quantization grid
(`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`, `q2k…q8k`) plus `f32`.
`bf16` / `f16` compile but aren't wired up on CPU yet; `fp8*` is CUDA
only. Those three are filtered out of the dropdown.
