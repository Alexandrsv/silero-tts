import torch
import os
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_model_cache = {}
_model_cache_lock = threading.Lock()

MODEL_URLS = {
    "v5_ru": "https://models.silero.ai/models/tts/ru/v5_ru.pt",
    "v5_5_ru": "https://models.silero.ai/models/tts/ru/v5_5_ru.pt",
    "v5_4_ru": "https://models.silero.ai/models/tts/ru/v5_4_ru.pt",
}

def _download_model(model_id: str, models_dir: str) -> str:
    """Download model to local models dir if not present."""
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_id}.pt")
    if not os.path.isfile(model_path):
        url = MODEL_URLS.get(model_id, MODEL_URLS["v5_ru"])
        logger.info(f"Downloading model from {url}")
        torch.hub.download_url_to_file(url, model_path, progress=True)
    return model_path

def load_model(
    model_id: str = "v5_ru",
    language: str = "ru",
    device: str = None,
    models_dir: str = None,
):
    """
    Load Silero TTS model from local file or download if missing.
    Returns (model, example_text).
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    models_dir = os.path.abspath(models_dir)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cache_key = f"{model_id}_{language}_{device}"
    
    with _model_cache_lock:
        if cache_key in _model_cache:
            logger.info(f"Using cached model {cache_key}")
            return _model_cache[cache_key]
        
        model_path = _download_model(model_id, models_dir)
        logger.info(f"Loading model from {model_path} onto {device}")
        
        try:
            importer = torch.package.PackageImporter(model_path)
            model = importer.load_pickle("tts_models", "model")
            model.to(device)
            # Example texts from Silero docs
            example_text = "В недрах тундры выдры в гетрах тырят в вёдра ядра кедров."
            _model_cache[cache_key] = (model, example_text)
            return model, example_text
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
