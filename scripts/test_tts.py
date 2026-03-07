#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.9"
# dependencies = ["sphn"]
# ///

import argparse

import ptts
import sphn


def main():
    parser = argparse.ArgumentParser(description="Test pocket-tts generation")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output.wav", help="Output WAV file path")
    parser.add_argument("-v", "--voice", default="alba", help="Voice to use")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=4242424242424242)
    args = parser.parse_args()

    print("Loading model...")
    model = ptts.load_model(temperature=args.temperature)
    print(f"Model loaded, sample_rate={model.sample_rate()}, voices={model.voices()}")

    print(f"Creating state for voice '{args.voice}'...")
    state = model.get_state_for_voice(args.voice)

    print(f"Generating audio for: {args.text!r}")
    audio = state.generate_audio(args.text, temperature=args.temperature, seed=args.seed)
    print(f"Generated {len(audio) / model.sample_rate():.2f}s of audio")

    sphn.write_wav(args.output, audio, model.sample_rate())
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
