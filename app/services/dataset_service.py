import os
import pandas as pd
from datasets import Dataset, DatasetDict
from app.config import settings

def load_training_dataset():
    """
    Carga el dataset local desde la carpeta raíz /data/
    """
    metadata_path = settings.DATASET_PATH / "metadata.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de metadatos en {metadata_path}")

    # 1. Leer CSV
    df = pd.read_csv(metadata_path)

    # 2. Crear la columna con la RUTA ABSOLUTA del audio
    # Importante: No usamos el nombre "audio" para que la librería no intente decodificarlo
    df['audio_path'] = df['file_name'].apply(
        lambda x: str(settings.DATASET_PATH / x)
    )

    # 3. Separar en Train y Validation
    train_df = df[df['file_name'].str.startswith('train/')].reset_index(drop=True)
    val_df = df[df['file_name'].str.startswith('validation/')].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("El CSV no contiene suficientes datos de 'train/' o 'validation/'")

    # 4. Crear Dataset de Hugging Face (sin cast_column)
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })

    return dataset_dict