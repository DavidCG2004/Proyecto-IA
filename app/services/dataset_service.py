from datasets import load_dataset, Audio
from app.config import settings


def load_training_dataset():
    dataset = load_dataset(
        "mozilla-foundation/common_voice_13_0",
        "es",
        split={
            "train": "train[:1%]",
            "validation": "validation[:1%]"
        }
    )

    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=settings.SAMPLE_RATE)
    )

    return dataset