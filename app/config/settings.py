from pydantic import BaseModel
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    # =========================
    # RUTAS
    # =========================
    DATASET_PATH: Path = BASE_DIR / "data"
    AUDIO_PATH: Path = BASE_DIR / "data" / "audio"
    MODEL_PATH: Path = BASE_DIR / "models"
    TRAINED_MODEL_PATH: Path = BASE_DIR / "models" / "trained_model"

    # =========================
    # MODELO
    # =========================
    BASE_MODEL_NAME: str = "openai/whisper-small"
    LANGUAGE: str = "spanish"
    TASK: str = "transcribe"

    # =========================
    # ENTRENAMIENTO
    # =========================
    BATCH_SIZE: int = 8
    EPOCHS: int = 3
    LEARNING_RATE: float = 1e-5
    WARMUP_STEPS: int = 500

    # =========================
    # AUDIO
    # =========================
    SAMPLE_RATE: int = 16000
    MAX_AUDIO_LENGTH: int = 30


# Instancia global
settings = Settings()


# =========================
# FUNCIONES PARA LA API
# =========================

def get_settings():
    """
    Devuelve la configuración actual del sistema
    """
    return settings


def update_settings(new_settings: dict):
    """
    Actualiza parámetros del sistema en tiempo de ejecución
    """
    global settings
    settings = settings.model_copy(update=new_settings)
    return settings
