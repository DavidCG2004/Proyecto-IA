import os
import torch
from datasets import Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Trainer,
    TrainingArguments
)

from app.core.model import get_device, MODEL_NAME, MODEL_PATH
from app.schemas.train_schema import TrainSchema
from app.services.dataset_service import load_training_dataset
from app.core.evaluator import save_metrics

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # El audio (input_features) ya debe tener el tamaño correcto (30s)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # El texto (labels) necesita padding dinámico
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Reemplazar padding por -100 para que la pérdida ignore esos tokens
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


def train_model(params: TrainSchema):
    """
    Realiza fine-tuning del modelo Whisper con un dataset en español.
    """

    device = get_device()

    # =========================
    # 1. Cargar modelo base
    # =========================
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language="spanish",
        task="transcribe"
    )

    model.to(device)

    # =========================
    # 2. Cargar dataset
    # =========================
    
    dataset = load_training_dataset() # Antes decía load_dataset()

    # =========================
    # 3. Preprocesamiento
    # =========================
    def prepare_batch(batch):
        audio = batch["audio"]
        inputs = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"], 
            return_tensors="pt"
        )

        batch["input_features"] = inputs.input_features[0]
        
        # CAMBIO AQUÍ: Usamos "transcription" en lugar de "sentence"
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    # =========================
    # 4. Argumentos de entrenamiento
    # =========================
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        per_device_train_batch_size=params.batch_size,
        per_device_eval_batch_size=params.batch_size,
        learning_rate=params.learning_rate,
        num_train_epochs=params.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        fp16=torch.cuda.is_available(),
        push_to_hub=False
    )

    # =========================
    # 5. Entrenador
    # =========================
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator, # <--- Añadir esto
        tokenizer=processor.feature_extractor,
    )

    # =========================
    # 6. Entrenamiento
    # =========================
    trainer.train()

    # =========================
    # 7. Guardar modelo
    # =========================
    model.save_pretrained(MODEL_PATH)
    processor.save_pretrained(MODEL_PATH)

    # =========================
    # 8. Métricas
    # =========================
    metrics = trainer.evaluate()
    
    save_metrics(
    references=["ejemplo"],
    hypotheses=["ejemplo"],
    loss=metrics.get("eval_loss")
) 
    return {
    "status": "success",
    "metrics": metrics
}
