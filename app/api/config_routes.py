from fastapi import APIRouter
from app.schemas.config_schema import ConfigSchema
from app.config.settings import update_settings, get_settings

router = APIRouter(prefix="/config", tags=["Configuration"])


@router.get("/")
def get_config():
    """
    Obtiene la configuración actual del sistema.
    """
    return get_settings()


@router.post("/")
def update_config(config: ConfigSchema):
    """
    Actualiza los parámetros de entrenamiento del modelo.
    """
    update_settings(config.model_dump())
    return {
        "message": "Configuración actualizada correctamente"
    }
