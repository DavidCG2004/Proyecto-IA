from fastapi import APIRouter
from app.core.evaluator import get_metrics

router = APIRouter(tags=["Metrics"])


@router.get("/")
def metrics_endpoint():
    """
    Devuelve las métricas del último entrenamiento.
    """
    metrics = get_metrics()
    return metrics
