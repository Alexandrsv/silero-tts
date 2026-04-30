from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TTSConfig:
    model_id: str = "v5_ru"
    language: str = "ru"
    sample_rate: int = 48000
    device: Optional[str] = None
    speakers: List[str] = None
    default_speaker: str = "xenia"
    
    def __post_init__(self):
        if self.speakers is None:
            self.speakers = ["aidar", "baya", "kseniya", "xenia", "eugene"]
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

AVAILABLE_SAMPLE_RATES = [8000, 24000, 48000]
DEFAULT_TEXT = "Привет! Это тестовая озвучка с помощью силеро моделей."
