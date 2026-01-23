from pydantic import BaseModel, Field


class TrainSchema(BaseModel):
    epochs: int = Field(
        ...,
        gt=0,
        description="Número de épocas de entrenamiento"
    )
    batch_size: int = Field(
        ...,
        gt=0,
        description="Tamaño del batch"
    )
    learning_rate: float = Field(
        ...,
        gt=0,
        description="Tasa de aprendizaje"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "epochs": 5,
                "batch_size": 8,
                "learning_rate": 0.0001
            }
        }
    }