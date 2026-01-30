from pydantic import BaseModel
from pathlib import Path

# Path(__file__) es .../app/config/settings.py
# .parent es .../app/config/
# .parent.parent es .../app/
# .parent.parent.parent es .../Proyecto IA/ (RAÍZ)
BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent 

class Settings(BaseModel):
    # =========================
    # RUTAS (Corregidas para buscar fuera de app/)
    # =========================
    DATASET_PATH: Path = ROOT_DIR / "data"
    AUDIO_PATH: Path = ROOT_DIR / "data"  # Los audios están en data/train y data/validation
    MODEL_PATH: Path = BASE_DIR / "models" / "whisper_finetuned"
    METRICS_FILE: Path = BASE_DIR / "models" / "metrics.json"
    
    # =========================
    # MODELO
    # =========================
    BASE_MODEL_NAME: str = "openai/whisper-tiny"
    LANGUAGE: str = "spanish"
    TASK: str = "transcribe"

    # =========================
    # CONFIG AUDIO
    # =========================
    SAMPLE_RATE: int = 16000

settings = Settings()

def get_settings():
    return settings

def update_settings(new_settings: dict):
    global settings
    settings = settings.model_copy(update=new_settings)
    return settings