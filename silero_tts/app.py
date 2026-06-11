import gradio as gr
import logging
from .config import TTSConfig, AVAILABLE_SAMPLE_RATES, DEFAULT_TEXT
from .model_loader import load_model
from .audio_utils import generate_audio, generate_long_text, apply_stress
from .silence_trimmer import trim_silence
import os
import resource
import re

# Increase file descriptor limit
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(65536, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    logging.getLogger(__name__).info(f"Set file descriptor limit to {new_soft}")
except Exception as e:
    logging.getLogger(__name__).warning(f"Could not set fd limit: {e}")

import sys


# Patch Gradio's compiled JS to extend the audio playback speed range
# from [0.5, 1, 1.5, 2] to [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4].
# This modifies the vendored frontend asset at runtime so that the native
# speed button in gr.Audio cycles through 8 speeds instead of 4.
# The patch is re-applied on every startup to survive Gradio reinstalls.
_EXTENDED_SPEEDS = "[.5,1,1.5,2,2.5,3,3.5,4]"
_ORIGINAL_SPEEDS_PAT = re.compile(r"\[\.5,1,1\.5,2\]")


def _patch_gradio_playback_speeds():
    assets_dir = os.path.join(os.path.dirname(gr.__file__), "templates", "frontend", "assets")
    if not os.path.isdir(assets_dir):
        logger.warning("Gradio assets directory not found, skipping playback speed patch")
        return False

    patched = False
    for fname in os.listdir(assets_dir):
        if not fname.endswith(".js"):
            continue
        fpath = os.path.join(assets_dir, fname)
        try:
            data = open(fpath).read()
        except OSError:
            continue
        if _ORIGINAL_SPEEDS_PAT.search(data):
            new_data = _ORIGINAL_SPEEDS_PAT.sub(_EXTENDED_SPEEDS, data)
            try:
                open(fpath, "w").write(new_data)
                logger.info(f"Patched playback speeds in {fname}: [0.5,1,1.5,2] -> [0.5,1,1.5,2,2.5,3,3.5,4]")
                patched = True
            except OSError as e:
                logger.warning(f"Cannot write {fpath}: {e}")
        elif _EXTENDED_SPEEDS in data:
            logger.info(f"Playback speeds already patched in {fname}")
            patched = True

    if not patched:
        logger.warning("Could not patch Gradio playback speeds — pattern not found in any JS asset")
    return patched


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_color = getattr(sys.stderr, "isatty", lambda: False)()

    def format(self, record):
        if self.force_color or hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(levelname)s:%(name)s:%(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])
for logger_name in ["httpx", "gradio"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

config = TTSConfig()

MAX_CHARS = 140
ACCENT_METHODS = ["model", "silero-stress", "manual", "none"]

# Check available normalizers
NORMALIZER_METHODS = ["none"]
_ru_normalizr = None
try:
    from ru_normalizr import Normalizer as RUNormalizr, NormalizeOptions
    NORMALIZER_METHODS.append("ru-normalizr")
    _ru_normalizr = RUNormalizr(NormalizeOptions.tts())
    logger.info("ru-normalizr available")
except Exception as e:
    _ru_normalizr = None
    logger.info(f"ru-normalizr not available: {e}")

AVAILABLE_DEVICES = ["cpu"]
if __import__('torch').cuda.is_available():
    AVAILABLE_DEVICES.append("cuda")

def normalize_text(text, normalizer_method):
    if normalizer_method == "none" or not text:
        return text
    try:
        if normalizer_method == "ru-normalizr" and _ru_normalizr:
            return _ru_normalizr.normalize(text)
    except Exception as e:
        logger.warning(f"Normalization failed ({normalizer_method}): {e}")
    return text


def preview_stress(text, accent_method, normalizer_method):
    if not text or not text.strip():
        return "", "Empty text"
    try:
        text = normalize_text(text, normalizer_method)
        if accent_method == "silero-stress":
            processed = apply_stress(text, method="silero-stress")
            return processed, f"Preview: {len(processed)} chars"
        elif accent_method == "model":
            return text, "Model will apply stress automatically (+ marks)"
        elif accent_method == "manual":
            return text, "Manual mode - use + for stress, ё for yo"
        else:
            return text, "No stress marks will be added"
    except Exception as e:
        return text, f"Error: {str(e)}"

def tts_generate(text, speaker, sample_rate, accent_method, device, normalizer_method, remove_silence, silence_threshold, min_leading_dur, min_trailing_dur, min_gap_dur, keep_gap_dur, progress=gr.Progress()):
    if not text or not text.strip():
        return None, "Empty text", text, "", None, ""

    text = normalize_text(text, normalizer_method)

    logger.info(f"Generating: {len(text)} chars, speaker={speaker}, sr={sample_rate}, accent={accent_method}, device={device}")
    
    try:
        model, _ = load_model(
            model_id=config.model_id,
            language=config.language,
            device=device
        )
        
        processed_text = text
        put_accent = True
        if accent_method == "silero-stress":
            logger.info("Applying silero-stress")
            processed_text = apply_stress(text, method="silero-stress")
            put_accent = False
        elif accent_method == "none":
            put_accent = False
        elif accent_method == "manual":
            put_accent = False
        
        # Progress callback function
        def progress_callback(current, total):
            progress(current / total, desc=f"Processing chunk {current}/{total}")
        
        if len(processed_text) > MAX_CHARS:
            logger.info("Using long text generation")
            audio = generate_long_text(model, processed_text, speaker, int(sample_rate), device, put_accent=put_accent, progress_callback=progress_callback)
        else:
            logger.info("Using single text generation")
            audio = generate_audio(model, processed_text, speaker, int(sample_rate), device, put_accent=put_accent)
        
        logger.info(f"Generated {len(audio)} samples")
        
        # Compute duration
        duration_seconds = len(audio) / int(sample_rate)
        duration_str = f"{int(duration_seconds // 3600):02d}:{int((duration_seconds % 3600) // 60):02d}:{int(duration_seconds % 60):02d}"

        if remove_silence:
            progress(0.93, desc="Trimming silence...")
            trimmed_audio = trim_silence(audio, int(sample_rate), threshold_db=silence_threshold, min_leading_dur=min_leading_dur, min_trailing_dur=min_trailing_dur, min_gap_dur=min_gap_dur, keep_gap_dur=keep_gap_dur)
            trimmed_duration_seconds = len(trimmed_audio) / int(sample_rate)
            trimmed_duration_str = f"{int(trimmed_duration_seconds // 3600):02d}:{int((trimmed_duration_seconds % 3600) // 60):02d}:{int(trimmed_duration_seconds % 60):02d}"
            progress(1.0, desc="Complete!")
            return (
                (int(sample_rate), audio.cpu().numpy()),
                f"OK: {len(audio)} samples",
                processed_text,
                duration_str,
                (int(sample_rate), trimmed_audio.cpu().numpy()),
                trimmed_duration_str,
            )
        else:
            progress(1.0, desc="Complete!")
            return (
                (int(sample_rate), audio.cpu().numpy()),
                f"OK: {len(audio)} samples",
                processed_text,
                duration_str,
                None,
                "",
            )
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, error_msg, text, "", None, ""

def create_app():
    with gr.Blocks(title="Silero TTS") as demo:
        gr.Markdown("# Silero TTS - Russian")
        gr.Markdown("Playback speed button cycles: 0.5x → 1x → 1.5x → 2x → 2.5x → 3x → 3.5x → 4x")

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(value=DEFAULT_TEXT, label="Text", lines=5)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Voice")
                        speaker_dropdown = gr.Dropdown(choices=config.speakers, value=config.default_speaker, label="Speaker", container=False)
                        sample_rate_dropdown = gr.Dropdown(choices=AVAILABLE_SAMPLE_RATES, value=config.sample_rate, label="Sample Rate", container=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### Processing")
                        accent_dropdown = gr.Dropdown(choices=ACCENT_METHODS, value="model", label="Accent Method", container=False)
                        normalizer_dropdown = gr.Dropdown(choices=NORMALIZER_METHODS, value=NORMALIZER_METHODS[-1] if len(NORMALIZER_METHODS) > 1 else "none", label="Text Normalizer", container=False)
                        device_dropdown = gr.Dropdown(choices=AVAILABLE_DEVICES, value=config.device, label="Device", container=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### Output")
                        gr.Markdown("Format: MP3")

                with gr.Row():
                    preview_btn = gr.Button("Preview Stress", variant="secondary", scale=1)
                    generate_btn = gr.Button("Generate", variant="primary", scale=2)

                processed_text_output = gr.Textbox(label="Processed Text (with stress marks)", lines=3, interactive=False)
                status = gr.Textbox(label="Status", interactive=False, lines=2)

            with gr.Column(scale=2):
                remove_silence_cb = gr.Checkbox(value=True, label="Remove silence pauses")

                with gr.Accordion("Trim parameters", open=False):
                    silence_threshold = gr.Slider(-80, -20, value=-50, step=1, label="Silence threshold (dB)")
                    with gr.Row():
                        min_leading_dur = gr.Slider(0, 1.0, value=0.1, step=0.05, label="Min leading speech (s)")
                        min_trailing_dur = gr.Slider(0, 2.0, value=0.2, step=0.05, label="Min trailing silence (s)")
                    with gr.Row():
                        min_gap_dur = gr.Slider(0, 3.0, value=0.3, step=0.05, label="Min internal gap (s)")
                        keep_gap_dur = gr.Slider(0, 1.0, value=0.1, step=0.05, label="Keep gap after trim (s)")

                gr.Markdown("### Original")
                audio_output = gr.Audio(label="Generated Audio", type="numpy", format="mp3")
                duration_output = gr.Textbox(label="Duration", interactive=False)

                trimmed_audio_output = gr.Audio(label="Trimmed Audio (silence removed)", type="numpy", format="mp3")
                trimmed_duration_output = gr.Textbox(label="Duration (Trimmed)", interactive=False)

        preview_btn.click(
            fn=preview_stress,
            inputs=[text_input, accent_dropdown, normalizer_dropdown],
            outputs=[processed_text_output, status]
        )

        generate_btn.click(
            fn=tts_generate,
            inputs=[text_input, speaker_dropdown, sample_rate_dropdown, accent_dropdown, device_dropdown, normalizer_dropdown, remove_silence_cb, silence_threshold, min_leading_dur, min_trailing_dur, min_gap_dur, keep_gap_dur],
            outputs=[audio_output, status, processed_text_output, duration_output, trimmed_audio_output, trimmed_duration_output]
        )
    return demo

def preload_model():
    """Preload model at startup to avoid loading during first request."""
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    try:
        load_model(model_id=config.model_id, language=config.language, device=device)
        logger.info(f"Preloaded model on {device}")
    except Exception as e:
        logger.error(f"Failed to preload model: {e}")

if __name__ == "__main__":
    _patch_gradio_playback_speeds()
    preload_model()
    logger.info(f"Starting Silero TTS app - Model: {config.model_id}, Device: {config.device}")
    app = create_app()
    app.launch(server_port=7860, share=False, server_name="0.0.0.0")
