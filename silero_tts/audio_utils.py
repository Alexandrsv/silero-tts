import torch
import os

def save_audio(audio: torch.Tensor, sample_rate: int, path: str) -> str:
    """
    Save audio tensor to WAV file using scipy.
    audio: 1D tensor
    """
    import numpy as np
    from scipy.io import wavfile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    audio_np = audio.cpu().numpy()
    wavfile.write(path, sample_rate, (audio_np * 32767).astype("int16"))
    return path

def generate_audio(model, text: str, speaker: str, sample_rate: int, device: str = "cpu"):
    """
    Generate audio using Silero model.
    Returns audio tensor.
    """
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
    return audio

def generate_and_save(model, text: str, speaker: str, sample_rate: int, path: str) -> str:
    """
    Generate and save audio using model's built-in method.
    Returns path to saved file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    audio_paths = model.save_wav(text=text, speaker=speaker, sample_rate=sample_rate, audio_path=path)
    return audio_paths if isinstance(audio_paths, str) else path
