from pydantic import BaseModel, Field


class PredictResponseSchema(BaseModel):
    transcription: str = Field(
        ...,
        description="Texto transcrito desde el audio"
    )
    language: str = Field(
        default="es",
        description="Idioma detectado o configurado"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "transcription": "hola cómo estás",
                "language": "es"
            }
        }
    }