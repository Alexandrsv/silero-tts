import gradio as gr
import torch
from .config import TTSConfig, AVAILABLE_SAMPLE_RATES, DEFAULT_TEXT
from .model_loader import load_model
from .audio_utils import generate_audio, save_audio
import tempfile
import os

config = TTSConfig()
model, _ = load_model(
    model_id=config.model_id,
    language=config.language,
    device=config.device
)

def tts_generate(text, speaker, sample_rate):
    if not text.strip():
        return None, "Empty text"
    try:
        audio = generate_audio(model, text, speaker, int(sample_rate), config.device)
        # save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        save_audio(audio, int(sample_rate), path)
        return path, f"Generated OK: {len(audio)} samples"
    except Exception as e:
        return None, str(e)

def create_app():
    with gr.Blocks(title="Silero TTS") as demo:
        gr.Markdown("# Silero TTS - Russian")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(value=DEFAULT_TEXT, label="Text", lines=3)
                speaker_dropdown = gr.Dropdown(choices=config.speakers, value=config.default_speaker, label="Speaker")
                sample_rate_dropdown = gr.Dropdown(choices=AVAILABLE_SAMPLE_RATES, value=config.sample_rate, label="Sample Rate")
                generate_btn = gr.Button("Generate")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
                status = gr.Textbox(label="Status", interactive=False)
        
        generate_btn.click(
            fn=tts_generate,
            inputs=[text_input, speaker_dropdown, sample_rate_dropdown],
            outputs=[audio_output, status]
        )
    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch(server_port=7860, share=False)
