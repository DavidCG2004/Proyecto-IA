from fastapi import APIRouter, UploadFile, File
from app.schemas.predict_schema import PredictResponseSchema
from app.core.inference import transcribe_audio

router = APIRouter(tags=["Prediction"])


@router.post("/", response_model=PredictResponseSchema)
def predict_endpoint(file: UploadFile = File(...)):
    transcription, language = transcribe_audio(file)

    return {
        "transcription": transcription,
        "language": language
    }
