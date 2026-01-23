import json
import os
from jiwer import wer, cer


# Ruta donde se guardarán las métricas
METRICS_PATH = "app/models/metrics.json"

def save_metrics(references, hypotheses, loss=None):
    """
    Calcula y guarda las metricas del model
    """

    metrics = {
        "wer": wer(references, hypotheses),
        "cer": cer(references, hypotheses),
        "loss": loss
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def get_metrics():
    """
    Devuelve las metricas del ultimo entrenmiento
    """
    if not os.path.exists(METRICS_PATH):
        return {
            "message": "No existen métricas aún. Entrene el modelo primero."
        }

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)


    return metrics
