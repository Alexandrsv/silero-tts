import torch
import os
import re
import numpy as np
from scipy.io import wavfile
from .text_chunker import split_text


def clean_text_for_tts(text: str) -> str:
    """Clean text for Silero TTS - keep only supported characters."""
    # Replace non-breaking spaces and other special spaces
    text = text.replace('\xa0', ' ').replace('\u2009', ' ').replace('\t', ' ')
    # Keep Cyrillic, Latin (for loanwords), digits, punctuation, spaces
    # Silero actually supports some Latin characters in Russian model
    text = re.sub(r'[^\u0400-\u04FF\u0300-\u036F\u00C0-\u024F\u1E00-\u1EFFa-zA-Z0-9\s.,!?;:"\'()\-–—…]', '', text)
    # Remove BOM if present
    text = text.replace('\ufeff', '').replace('￼', '')
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_valid_for_tts(text: str) -> bool:
    """Check if text has enough Cyrillic characters to be processed by Russian TTS."""
    if not text:
        return False
    cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    return cyrillic_chars >= 3

def save_audio(audio: torch.Tensor, sample_rate: int, path: str, fmt: str = "wav") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    audio_np = audio.cpu().numpy()
    audio_int16 = (audio_np * 32767).astype("int16")
    
    if fmt == "mp3":
        try:
            save_audio_as_mp3(audio_int16, sample_rate, path)
        except Exception as e:
            print(f"MP3 export failed: {e}")
            raise
    else:
        wavfile.write(path, sample_rate, audio_int16)
    return path


def save_audio_as_mp3(pcm_data: np.ndarray, sample_rate: int, mp3_path: str) -> None:
    """Save raw PCM data as MP3 using ffmpeg via pipe (no temp files)."""
    import subprocess
    import shutil
    
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg for MP3 export.")
    
    # FFmpeg command: read raw s16le PCM from stdin, encode to MP3
    cmd = [
        ffmpeg_path,
        '-y',  # overwrite output
        '-f', 's16le',  # input format: signed 16-bit little-endian
        '-ar', str(sample_rate),
        '-ac', '1',  # mono
        '-i', 'pipe:0',  # input from stdin
        '-codec:a', 'libmp3lame',
        '-qscale:a', '2',  # VBR quality (0-9, 2 is high quality)
        mp3_path
    ]
    
    # Convert numpy array to bytes and send to ffmpeg stdin
    pcm_bytes = pcm_data.tobytes()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate(input=pcm_bytes)
    
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")

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
                 put_accent: bool = True, put_yo: bool = True, progress_callback=None):
    text = clean_text_for_tts(text)
    if not text:
        return torch.tensor([])
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate,
                           put_accent=put_accent, put_yo=put_yo)
    return audio

def generate_long_text(
    model, text: str, speaker: str, sample_rate: int, device: str = "cpu",
    max_chars: int = 140, put_accent: bool = True, put_yo: bool = True,
    progress_callback=None
) -> torch.Tensor:
    text = clean_text_for_tts(text)
    chunks = split_text(text, max_chars=max_chars)
    if len(chunks) == 1:
        return generate_audio(model, chunks[0], speaker, sample_rate, device,
                             put_accent, put_yo)
    audio_parts = []
    silence_len = int(0.12 * sample_rate)
    silence = torch.zeros(silence_len)
    skipped = 0
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        if not chunk.strip() or not is_valid_for_tts(chunk):
            skipped += 1
            if progress_callback:
                progress_callback(i + 1, total_chunks)
            continue
        try:
            audio = generate_audio(model, chunk, speaker, sample_rate, device,
                                  put_accent, put_yo)
            if len(audio) > 0:
                audio_parts.append(audio)
                if i < len(chunks) - 1:
                    audio_parts.append(silence)
        except Exception as e:
            print(f"Failed to generate chunk {i}: {e}")
            if progress_callback:
                progress_callback(i + 1, total_chunks)
            continue
        if progress_callback:
            progress_callback(i + 1, total_chunks)
    if skipped:
        print(f"Skipped {skipped} chunks (non-Russian or empty)")
    if not audio_parts:
        raise ValueError("No audio generated")
    return torch.cat(audio_parts)

def _split_by_tokens(text: str, max_tokens: int = 1800) -> list:
    """Split text into chunks ≤ max_tokens using BERT tokenizer."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        words = text.split()
        chunks = []
        current = []
        current_tokens = 0
        for word in words:
            token_count = len(tokenizer.tokenize(word))
            if current_tokens + token_count > max_tokens and current:
                chunks.append(" ".join(current))
                current = [word]
                current_tokens = token_count
            else:
                current.append(word)
                current_tokens += token_count
        if current:
            chunks.append(" ".join(current))
        return chunks
    except Exception:
        # fallback: rough estimate ~1.5 tokens per word for Russian
        words = text.split()
        max_words = max_tokens // 2
        return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def apply_stress(text: str, method: str = "model"):
    if method == "silero-stress":
        try:
            from silero_stress import load_accentor
            accentor = load_accentor()
            chunks = _split_by_tokens(text)
            return " ".join(accentor(c) for c in chunks)
        except Exception as e:
            print(f"Silero Stress failed: {e}, falling back to manual")
            return text
    return text
