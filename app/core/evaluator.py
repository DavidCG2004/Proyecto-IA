import json
import os
from jiwer import wer, cer
from app.config import settings

def save_metrics(references, hypotheses, loss=None):
    """
    Calcula y guarda las métricas del modelo.
    """
    # Asegurar que la carpeta exista
    os.makedirs(settings.METRICS_FILE.parent, exist_ok=True)

    metrics = {
        "wer": wer(references, hypotheses),
        "cer": cer(references, hypotheses),
        "loss": loss
    }

    with open(settings.METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return metrics

def get_metrics():
    """
    Devuelve las métricas del último entrenamiento.
    """
    if not os.path.exists(settings.METRICS_FILE):
        return {
            "message": f"No existen métricas aún. El archivo no se encontró en {settings.METRICS_FILE}"
        }

    with open(settings.METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics