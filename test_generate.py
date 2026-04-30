#!/usr/bin/env python3
"""
Test script for Silero TTS generation.
Uses Whisper for transcription check if available.
"""
import sys
import os
import tempfile
import time

def test_model_load():
    print("Testing model load...")
    from silero_tts.model_loader import load_model
    start = time.time()
    model, example_text = load_model()
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f}s")
    print(f"Example text: {example_text[:100]}...")
    return model, example_text

def test_generate_audio(model, text="Привет, мир!"):
    print(f"Generating audio for: {text}")
    from silero_tts.audio_utils import generate_audio, save_audio
    from silero_tts.config import TTSConfig
    config = TTSConfig()
    start = time.time()
    audio = generate_audio(model, text, config.default_speaker, config.sample_rate, config.device)
    elapsed = time.time() - start
    print(f"Generated {len(audio)} samples in {elapsed:.2f}s")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    save_audio(audio, config.sample_rate, path)
    size = os.path.getsize(path)
    print(f"Saved to {path} ({size} bytes)")
    return path

def test_with_whisper(audio_path, expected_text):
    try:
        import whisper
    except ImportError:
        print("Whisper not installed, skipping transcription test")
        return
    print("Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="ru")
    transcribed = result["text"].strip()
    print(f"Transcribed: {transcribed}")
    print(f"Original: {expected_text}")
    # simple check: if key words appear
    keywords = expected_text.lower().split()
    found = sum(1 for kw in keywords if kw in transcribed.lower())
    print(f"Keyword match: {found}/{len(keywords)}")

def main():
    print("Starting Silero TTS tests")
    model, example_text = test_model_load()
    audio_path = test_generate_audio(model, example_text[:200])
    test_with_whisper(audio_path, example_text[:200])
    print("Tests completed")

if __name__ == "__main__":
    main()
