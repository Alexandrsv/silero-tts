#!/usr/bin/env python3
"""Launch Silero TTS Gradio app."""
import argparse
from silero_tts.app import create_app, preload_model


def parse_args():
    parser = argparse.ArgumentParser(description="Silero TTS")
    parser.add_argument("--server-port", type=int, default=7860, help="Port for Gradio server")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Starting Silero TTS app...")
    print(f"Model: v5_ru, Sample rate: 48000, Device: auto, Port: {args.server_port}")
    preload_model()
    app = create_app()
    app.launch(
        server_port=args.server_port,
        share=False,
        server_name="0.0.0.0",
        show_error=True
    )
