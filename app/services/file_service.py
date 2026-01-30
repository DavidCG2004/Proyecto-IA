import os
from fastapi import UploadFile
from app.config import settings


def save_uploaded_file(file: UploadFile) -> str:
    """
    Guarda un archivo subido y devuelve su ruta.
    """

    os.makedirs(settings.AUDIO_PATH, exist_ok=True)

    file_path = os.path.join(settings.AUDIO_PATH, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return file_path
