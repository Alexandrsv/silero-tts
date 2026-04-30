from .config import TTSConfig, AVAILABLE_SAMPLE_RATES, DEFAULT_TEXT
from .model_loader import load_model
from .audio_utils import generate_audio, save_audio
from .app import create_app

__version__ = "0.1.0"
