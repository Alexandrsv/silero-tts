import torch
import os
import numpy as np
from .text_chunker import split_text

def save_audio(audio: torch.Tensor, sample_rate: int, path: str) -> str:
    from scipy.io import wavfile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    audio_np = audio.cpu().numpy()
    wavfile.write(path, sample_rate, (audio_np * 32767).astype("int16"))
    return path

def change_speed(audio: torch.Tensor, speed: float) -> torch.Tensor:
    if speed == 1.0:
        return audio
    orig_len = audio.shape[0]
    new_len = int(orig_len / speed)
    return torch.nn.functional.interpolate(
        audio.unsqueeze(0).unsqueeze(0),
        size=new_len,
        mode='linear',
        align_corners=False
    ).squeeze()

def generate_audio(model, text: str, speaker: str, sample_rate: int, device: str = "cpu",
                 put_accent: bool = True, put_yo: bool = True):
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate,
                           put_accent=put_accent, put_yo=put_yo)
    return audio

def generate_long_text(
    model, text: str, speaker: str, sample_rate: int, device: str = "cpu",
    max_chars: int = 140, put_accent: bool = True, put_yo: bool = True
) -> torch.Tensor:
    chunks = split_text(text, max_chars=max_chars)
    if len(chunks) == 1:
        return generate_audio(model, chunks[0], speaker, sample_rate, device,
                            put_accent, put_yo)
    audio_parts = []
    silence_len = int(0.12 * sample_rate)
    silence = torch.zeros(silence_len)
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        try:
            audio = generate_audio(model, chunk, speaker, sample_rate, device,
                                  put_accent, put_yo)
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
    if method == "silero-stress":
        try:
            from silero_stress import load_accentor
            accentor = load_accentor()
            return accentor(text)
        except Exception as e:
            print(f"Silero Stress failed: {e}, falling back to manual")
            return text
    return text
    # 'model' and 'manual' and 'none' - return as is, model handles it
    return text
