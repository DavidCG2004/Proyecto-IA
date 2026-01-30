from pydantic import BaseModel, Field


class ConfigSchema(BaseModel):
    language: str = Field(
        default="es",
        description="Idioma del modelo"
    )
    sample_rate: int = Field(
        default=16000,
        description="Frecuencia de muestreo del audio"
    )
    max_audio_length: int = Field(
        default=30,
        description="Duración máxima del audio en segundos"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "language": "es",
                "sample_rate": 16000,
                "max_audio_length": 30
            }
        }
    }