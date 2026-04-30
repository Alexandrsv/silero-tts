import torch
import os
import numpy as np
from .text_chunker import split_text

def save_audio(audio: torch.Tensor, sample_rate: int, path: str) -> str:
    """
    Save audio tensor to WAV file using scipy.
    audio: 1D tensor
    """
    from scipy.io import wavfile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    audio_np = audio.cpu().numpy()
    wavfile.write(path, sample_rate, (audio_np * 32767).astype("int16"))
    return path

def generate_audio(model, text: str, speaker: str, sample_rate: int, device: str = "cpu", put_accent: bool = True, put_yo: bool = True, speed: float = 1.0):
    """
    Generate audio using Silero model.
    Returns audio tensor.
    speed: 0.5 to 4.0 (1.0 = normal)
    """
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate, put_accent=put_accent, put_yo=put_yo, speed=speed)
    return audio

def generate_long_text(
    model, text: str, speaker: str, sample_rate: int, device: str = "cpu", max_chars: int = 140, put_accent: bool = True, put_yo: bool = True, speed: float = 1.0
) -> torch.Tensor:
    """
    Generate audio for long text by splitting into chunks.
    Returns concatenated audio tensor with small silence between chunks.
    """
    chunks = split_text(text, max_chars=max_chars)
    
    if len(chunks) == 1:
        return generate_audio(model, chunks[0], speaker, sample_rate, device, put_accent, put_yo, speed)
    
    audio_parts = []
    silence_samples = int(0.12 * sample_rate)  # 120ms silence
    silence = torch.zeros(silence_samples)
    
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        try:
            audio = generate_audio(model, chunk, speaker, sample_rate, device, put_accent, put_yo, speed)
            audio_parts.append(audio)
            if i < len(chunks) - 1:
                audio_parts.append(silence)
        except Exception as e:
            print(f"Failed to generate chunk {i}: {e}")
            continue
    
    if not audio_parts:
        raise ValueError("No audio generated")
    
    return torch.cat(audio_parts)


def apply_stress(text: str, method: str = "model"):
    """
    Apply stress marks to text.
    Methods: 'model' (put_accent param), 'silero-stress', 'manual' (no change), 'none'.
    Returns processed text ready for TTS.
    """
    if method == "silero-stress":
        try:
            from silero_stress import load_accentor
            accentor = load_accentor()
            return accentor(text)
        except Exception as e:
            print(f"Silero Stress failed: {e}, falling back to manual")
            return text
    # 'model' and 'manual' and 'none' - return as is, model handles it
    return text
