from .app import create_app, preload_model, _patch_gradio_playback_speeds

if __name__ == "__main__":
    _patch_gradio_playback_speeds()
    preload_model()
    app = create_app()
    app.launch(server_port=7860, share=False, server_name="0.0.0.0", show_error=True)
