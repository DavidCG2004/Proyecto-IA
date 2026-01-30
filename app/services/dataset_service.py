import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio
from app.config import settings

def load_training_dataset():
    """
    Carga el dataset local desde data/metadata.csv y las carpetas de audio.
    """
    metadata_path = os.path.join(settings.DATASET_PATH, "metadata.csv")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No se encontró el archivo de metadatos en {metadata_path}")

    # 1. Cargar el CSV con pandas
    df = pd.read_csv(metadata_path)

    # 2. Crear rutas absolutas para que la librería datasets encuentre los audios
    # Esto une la ruta base del dataset con el nombre del archivo en el CSV
    df['audio'] = df['file_name'].apply(
        lambda x: os.path.join(settings.DATASET_PATH, x)
    )

    # 3. Separar en Train y Validation (si no están separados en el CSV)
    # Si tu CSV ya tiene una columna 'split', puedes usarla. 
    # Si no, aquí lo dividimos 90% entrenamiento, 10% validación.
    train_df = df[df['file_name'].str.contains('train/')].reset_index(drop=True)
    val_df = df[df['file_name'].str.contains('validation/')].reset_index(drop=True)

    # Convertir a objetos Dataset de Hugging Face
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    # 4. Castear la columna de audio al tipo Audio de datasets
    # Esto es crucial para que Whisper pueda procesar el array numérico
    dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=settings.SAMPLE_RATE))

    return dataset_dict