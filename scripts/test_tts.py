#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.9"
# dependencies = ["sphn"]
# ///

import argparse
from pathlib import Path

import ptts
import sphn


def main():
    parser = argparse.ArgumentParser(description="Test pocket-tts generation")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument(
        "-o", "--output", default="output.wav", help="Output WAV file path"
    )
    parser.add_argument("-v", "--voice", default="alba", help="Voice to use")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--pad-to", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--eos-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=4242424242424242)
    parser.add_argument(
        "--config", type=str, default=None, help="Path to JSON config file"
    )
    args = parser.parse_args()

    if args.config is None:
        print("Loading model...")
        model = ptts.load_model(eos_threshold=args.eos_threshold)
        print(
            f"Model loaded, sample_rate={model.sample_rate()}, voices={model.voices()}"
        )
    else:
        model = ptts.load_model(config=args.config, eos_threshold=args.eos_threshold)

    if Path(args.voice).exists():
        print(f"Loading audio from {args.voice}...")
        audio, _ = sphn.read(args.voice, sample_rate=24000)
        print(f"Audio loaded, {audio.shape} {audio.dtype}")
        audio = audio[0, : 24000 * 10]
        state = model.get_state_for_audio(audio, cfg_coef=args.cfg)
    else:
        print(f"Creating state for voice '{args.voice}'...")
        state = model.get_state_for_voice(args.voice)

    print(f"Generating audio for: {args.text!r}")
    audio = state.generate_audio(
        args.text, temperature=args.temperature, seed=args.seed, pad_to=args.pad_to
    )
    print(f"Generated {len(audio) / model.sample_rate():.2f}s of audio")

    sphn.write_wav(args.output, audio, model.sample_rate())
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
