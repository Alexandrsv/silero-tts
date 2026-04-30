import gradio as gr
import logging
from .config import TTSConfig, AVAILABLE_SAMPLE_RATES, DEFAULT_TEXT
from .model_loader import load_model
from .audio_utils import generate_audio, generate_long_text, save_audio, apply_stress
import tempfile
import os
import atexit
import glob
import threading
import time
import resource

# Increase file descriptor limit
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(65536, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    logging.getLogger(__name__).info(f"Set file descriptor limit to {new_soft}")
except Exception as e:
    logging.getLogger(__name__).warning(f"Could not set fd limit: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_temp_files = []

def _cleanup_temp_files():
    """Clean up tracked temp files."""
    for f in list(_temp_files):
        try:
            os.unlink(f)
        except OSError:
            pass
    _temp_files.clear()

def _delayed_cleanup(path, delay=30):
    """Delete file after delay (enough time for Gradio to read it)."""
    def _clean():
        time.sleep(delay)
        try:
            os.unlink(path)
        except OSError:
            pass
        if path in _temp_files:
            _temp_files.remove(path)
    threading.Thread(target=_clean, daemon=True).start()

atexit.register(_cleanup_temp_files)

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

def tts_generate(text, speaker, sample_rate, accent_method, device, audio_format, normalizer_method, progress=gr.Progress()):
    if not text or not text.strip():
        return None, "Empty text", text

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
        
        # Progress for saving
        progress(0.9, desc="Saving audio...")
        suffix = ".mp3" if audio_format == "mp3" else ".wav"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        _temp_files.append(path)
        save_audio(audio, int(sample_rate), path, fmt=audio_format)
        logger.info(f"Saved to {path}")
        _delayed_cleanup(path)
        progress(1.0, desc="Complete!")
        return path, f"OK: {len(audio)} samples", processed_text
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, error_msg, text

def create_app():
    with gr.Blocks(title="Silero TTS") as demo:
        gr.Markdown("# Silero TTS - Russian")
        gr.Markdown("Speed slider controls audio playback rate (0.5x = slow, 4x = fast)")

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
                        format_dropdown = gr.Dropdown(choices=["wav", "mp3"], value="mp3", label="Audio Format", container=False)

                with gr.Row():
                    preview_btn = gr.Button("Preview Stress", variant="secondary", scale=1)
                    generate_btn = gr.Button("Generate", variant="primary", scale=2)

                processed_text_output = gr.Textbox(label="Processed Text (with stress marks)", lines=3, interactive=False)
                status = gr.Textbox(label="Status", interactive=False, lines=2)

            with gr.Column(scale=2):
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
        
        preview_btn.click(
            fn=preview_stress,
            inputs=[text_input, accent_dropdown, normalizer_dropdown],
            outputs=[processed_text_output, status]
        )
        
        generate_btn.click(
            fn=tts_generate,
            inputs=[text_input, speaker_dropdown, sample_rate_dropdown, accent_dropdown, device_dropdown, format_dropdown, normalizer_dropdown],
            outputs=[audio_output, status, processed_text_output]
        )
    return demo

def _cleanup_old_temp_files():
    """Clean up old temp files from previous runs."""
    patterns = ['/tmp/tmp*.mp3', '/tmp/tmp*.wav']
    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                if time.time() - os.path.getmtime(f) > 3600:
                    os.unlink(f)
            except OSError:
                pass

def preload_model():
    """Preload model at startup to avoid loading during first request."""
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    try:
        load_model(model_id=config.model_id, language=config.language, device=device)
        logger.info(f"Preloaded model on {device}")
    except Exception as e:
        logger.error(f"Failed to preload model: {e}")

if __name__ == "__main__":
    _cleanup_old_temp_files()
    preload_model()
    logger.info(f"Starting Silero TTS app - Model: {config.model_id}, Device: {config.device}")
    app = create_app()
    app.launch(server_port=7860, share=False, server_name="0.0.0.0")
