import gradio as gr
import logging
from .config import TTSConfig, AVAILABLE_SAMPLE_RATES, DEFAULT_TEXT
from .model_loader import load_model
from .audio_utils import generate_audio, generate_long_text, save_audio
import tempfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = TTSConfig()
model, _ = load_model(
    model_id=config.model_id,
    language=config.language,
    device=config.device
)

MAX_CHARS = 140

def tts_generate(text, speaker, sample_rate):
    if not text or not text.strip():
        return None, "Empty text"
    
    logger.info(f"Generating: {len(text)} chars, speaker={speaker}, sr={sample_rate}")
    
    try:
        if len(text) > MAX_CHARS:
            logger.info("Using long text generation")
            audio = generate_long_text(model, text, speaker, int(sample_rate), config.device)
        else:
            logger.info("Using single text generation")
            audio = generate_audio(model, text, speaker, int(sample_rate), config.device)
        
        logger.info(f"Generated {len(audio)} samples")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        save_audio(audio, int(sample_rate), path)
        logger.info(f"Saved to {path}")
        return path, f"OK: {len(audio)} samples"
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, error_msg

def create_app():
    with gr.Blocks(title="Silero TTS") as demo:
        gr.Markdown("# Silero TTS - Russian")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(value=DEFAULT_TEXT, label="Text", lines=3)
                speaker_dropdown = gr.Dropdown(choices=config.speakers, value=config.default_speaker, label="Speaker")
                sample_rate_dropdown = gr.Dropdown(choices=AVAILABLE_SAMPLE_RATES, value=config.sample_rate, label="Sample Rate")
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
                status = gr.Textbox(label="Status", interactive=False, lines=2)
        
        generate_btn.click(
            fn=tts_generate,
            inputs=[text_input, speaker_dropdown, sample_rate_dropdown],
            outputs=[audio_output, status]
        )
    return demo

if __name__ == "__main__":
    logger.info(f"Starting Silero TTS app - Model: {config.model_id}, Device: {config.device}")
    app = create_app()
    app.launch(server_port=7860, share=False, server_name="0.0.0.0")
