import gradio as gr
import logging
from .config import TTSConfig, AVAILABLE_SAMPLE_RATES, DEFAULT_TEXT
from .model_loader import load_model
from .audio_utils import generate_audio, generate_long_text, save_audio, apply_stress
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
ACCENT_METHODS = ["model", "silero-stress", "manual", "none"]

def preview_stress(text, accent_method):
    if not text or not text.strip():
        return "", "Empty text"
    try:
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

def tts_generate(text, speaker, sample_rate, accent_method):
    if not text or not text.strip():
        return None, "Empty text", text
    
    logger.info(f"Generating: {len(text)} chars, speaker={speaker}, sr={sample_rate}, accent={accent_method}")
    
    try:
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
        
        if len(processed_text) > MAX_CHARS:
            logger.info("Using long text generation")
            audio = generate_long_text(model, processed_text, speaker, int(sample_rate), config.device, put_accent=put_accent)
        else:
            logger.info("Using single text generation")
            audio = generate_audio(model, processed_text, speaker, int(sample_rate), config.device, put_accent=put_accent)
        
        logger.info(f"Generated {len(audio)} samples")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        save_audio(audio, int(sample_rate), path)
        logger.info(f"Saved to {path}")
        return path, f"OK: {len(audio)} samples", processed_text
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, error_msg, text

def create_app():
    with gr.Blocks(title="Silero TTS", js=AUDIO_JS) as demo:
        gr.Markdown("# Silero TTS - Russian")
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(value=DEFAULT_TEXT, label="Text", lines=3)
                with gr.Row():
                    speaker_dropdown = gr.Dropdown(choices=config.speakers, value=config.default_speaker, label="Speaker")
                    sample_rate_dropdown = gr.Dropdown(choices=AVAILABLE_SAMPLE_RATES, value=config.sample_rate, label="Sample Rate")
                with gr.Row():
                    speed_slider = gr.Slider(minimum=0.5, maximum=4.0, value=1.0, step=0.1, label="Playback Speed (x)")
                    accent_dropdown = gr.Dropdown(choices=ACCENT_METHODS, value="model", label="Accent Method")
                with gr.Row():
                    preview_btn = gr.Button("Preview Stress", variant="secondary")
                    generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column(scale=1):
                processed_text_output = gr.Textbox(label="Processed Text (with stress marks)", lines=3, interactive=False)
                audio_output = gr.Audio(label="Generated Audio", type="filepath", elem_id="audio-player")
                speed_html = gr.HTML(SPEED_CONTROL_HTML)
                status = gr.Textbox(label="Status", interactive=False, lines=2)
        
        preview_btn.click(
            fn=preview_stress,
            inputs=[text_input, accent_dropdown],
            outputs=[processed_text_output, status]
        )
        
        generate_btn.click(
            fn=tts_generate,
            inputs=[text_input, speaker_dropdown, sample_rate_dropdown, accent_dropdown],
            outputs=[audio_output, status, processed_text_output]
        )
    return demo

# JavaScript to control audio playback speed
AUDIO_JS = """
function() {
    // Wait for audio element and speed slider
    function setupSpeedControl() {
        const audio = document.querySelector('#audio-player audio');
        const slider = document.querySelector('input[aria-label="Playback Speed (x)"]');
        if (!audio || !slider) return;
        
        slider.addEventListener('input', (e) => {
            audio.playbackRate = parseFloat(e.target.value);
        });
        
        // Update slider when audio source changes
        const observer = new MutationObserver(() => {
            audio.playbackRate = parseFloat(slider.value);
        });
        observer.observe(audio, { attributes: true, attributeFilter: ['src'] });
    }
    
    // Run on load and periodically check
    setupSpeedControl();
    setInterval(setupSpeedControl, 1000);
}
"""

SPEED_CONTROL_HTML = """
<script>
// Backup: direct control if JS in Blocks doesn't work
document.addEventListener('input', function(e) {
    if (e.target.getAttribute('aria-label') === 'Playback Speed (x)') {
        const audio = document.querySelector('#audio-player audio');
        if (audio) audio.playbackRate = parseFloat(e.target.value);
    }
});
</script>
"""

if __name__ == "__main__":
    logger.info(f"Starting Silero TTS app - Model: {config.model_id}, Device: {config.device}")
    app = create_app()
    app.launch(server_port=7860, share=False, server_name="0.0.0.0")
