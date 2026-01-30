from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor
)
import torch
import os

# Modelo base (ligero para proyectos académicos)
MODEL_NAME = "openai/whisper-small"

# Ruta donde se guardará el modelo ajustado
MODEL_PATH = "app/models/whisper_finetuned"


def get_device():
    """
    Devuelve el dispositivo disponible (GPU o CPU)
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """
    Carga el modelo Whisper (base o ajustado si existe)
    """
    device = get_device()

    if os.path.exists(MODEL_PATH):
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        print("✔ Modelo ajustado cargado")
    else:
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        print("✔ Modelo base Whisper cargado")

    model.to(device)

    return model, processor, device