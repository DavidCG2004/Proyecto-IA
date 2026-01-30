from fastapi import APIRouter
from app.schemas.train_schema import TrainSchema
from app.core.trainer import train_model

router = APIRouter(tags=["Training"])


@router.post("/")
def train_endpoint(params: TrainSchema):
    """
    Entrena o ajusta el modelo Speech-to-Text con los parámetros enviados.
    """
    result = train_model(params)
    return {
        "message": "Entrenamiento finalizado",
        "details": result
    }
