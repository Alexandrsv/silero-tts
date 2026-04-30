#!/usr/bin/env python3
"""Launch Silero TTS Gradio app."""
from silero_tts.app import create_app, preload_model
import sys

if __name__ == "__main__":
    print("Starting Silero TTS app...")
    print("Model: v5_ru, Sample rate: 48000, Device: auto")
    preload_model()
    app = create_app()
    app.launch(
        server_port=7860,
        share=False,
        server_name="0.0.0.0",
        show_error=True
    )
