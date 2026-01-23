import librosa
from app.config import settings


def load_audio(file_path: str):
    """
    Carga un archivo de audio y lo convierte al formato para ser usado
    """

    audio, sample_rate = librosa.load(
        file_path,
        sr=settings.SAMPLE_RATE
    )


    return audio
