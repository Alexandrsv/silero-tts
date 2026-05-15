import torch
import numpy as np


def _window_rms(audio_np: np.ndarray, window_samples: int):
    num_windows = len(audio_np) // window_samples
    if num_windows == 0:
        return np.array([]), 0
    windows = audio_np[:num_windows * window_samples].reshape(num_windows, window_samples)
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    return rms, num_windows


def trim_silence(
    audio: torch.Tensor,
    sample_rate: int,
    threshold_db: float = -50.0,
    min_leading_dur: float = 0.1,
    min_trailing_dur: float = 0.2,
    min_gap_dur: float = 0.3,
    keep_gap_dur: float = 0.1,
) -> torch.Tensor:
    audio_np = audio.cpu().numpy().squeeze()
    if audio_np.ndim != 1 or len(audio_np) == 0:
        return audio

    threshold_amp = 10 ** (threshold_db / 20.0)
    window_ms = 10
    window_samples = int(window_ms * sample_rate / 1000)
    if len(audio_np) <= window_samples * 2:
        return audio

    rms, num_windows = _window_rms(audio_np, window_samples)
    is_speech = rms > threshold_amp

    leading_windows = max(1, int(min_leading_dur * 1000 / window_ms))
    start_idx = 0
    speech_count = 0
    for i in range(num_windows):
        if is_speech[i]:
            speech_count += 1
            if speech_count >= leading_windows:
                start_idx = max(0, i - leading_windows + 1)
                break
        else:
            speech_count = 0
    else:
        return audio

    trailing_windows = max(1, int(min_trailing_dur * 1000 / window_ms))
    last_speech = -1
    for i in range(num_windows - 1, -1, -1):
        if is_speech[i]:
            last_speech = i
            break

    if 0 <= last_speech < num_windows - 1:
        trailing_len = num_windows - 1 - last_speech
        if trailing_len >= trailing_windows:
            end_idx = last_speech
        else:
            end_idx = num_windows - 1
    else:
        end_idx = num_windows - 1

    start_sample = start_idx * window_samples
    end_sample = min((end_idx + 1) * window_samples, len(audio_np))
    if start_sample >= end_sample:
        return audio

    trimmed = audio_np[start_sample:end_sample]

    min_gap_windows = int(min_gap_dur * 1000 / window_ms)
    keep_gap_windows = int(keep_gap_dur * 1000 / window_ms)

    if min_gap_windows < 2:
        return torch.from_numpy(trimmed)

    rms2, num_w2 = _window_rms(trimmed, window_samples)
    if num_w2 == 0:
        return torch.from_numpy(trimmed)
    is_speech2 = rms2 > threshold_amp

    output_parts = []
    i = 0
    while i < num_w2:
        if is_speech2[i]:
            seg_start = i * window_samples
            while i < num_w2 and is_speech2[i]:
                i += 1
            seg_end = i * window_samples
            output_parts.append(trimmed[seg_start:seg_end])
        else:
            gap_start = i
            while i < num_w2 and not is_speech2[i]:
                i += 1
            gap_len = i - gap_start
            if gap_len >= min_gap_windows:
                keep = min(keep_gap_windows, gap_len) * window_samples
                output_parts.append(np.zeros(keep, dtype=trimmed.dtype))
            else:
                output_parts.append(trimmed[gap_start * window_samples:i * window_samples])

    if not output_parts:
        return torch.from_numpy(trimmed)

    return torch.from_numpy(np.concatenate(output_parts))
