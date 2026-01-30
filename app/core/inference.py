import torch
from fastapi import UploadFile
from app.services.audio_service import load_audio
from app.services.file_service import save_uploaded_file
from app.core.model import load_model
from app.config import settings


def transcribe_audio(file: UploadFile):
    file_path = save_uploaded_file(file)

    model, processor, device = load_model()

    audio = load_audio(file_path)

    inputs = processor(
        audio,
        sampling_rate=settings.SAMPLE_RATE,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription, settings.LANGUAGE