import os
import torch
import librosa
import numpy as np
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,           # Clase específica para modelos de voz/traducción
    Seq2SeqTrainingArguments,  # Argumentos específicos
    EarlyStoppingCallback
)
from app.core.model import get_device
from app.schemas.train_schema import TrainSchema
from app.services.dataset_service import load_training_dataset
from app.core.evaluator import save_metrics
from app.config import settings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def train_model(params: TrainSchema):
    # Forzamos CPU para estabilidad en esta prueba
    device = "cpu" 
    print(f"⚠️ Entrenando en modo CPU (Esto será lento)")

    processor = WhisperProcessor.from_pretrained(settings.BASE_MODEL_NAME, language="Spanish", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(settings.BASE_MODEL_NAME)
    
    # Optimizaciones CPU
    model.freeze_encoder() 
    model.to(device)

    # 2. Cargar Dataset
    dataset = load_training_dataset()
    max_train_samples = 200 
    dataset["train"] = dataset["train"].select(range(min(len(dataset["train"]), max_train_samples)))
    print(f"📊 Dataset reducido a {len(dataset['train'])} muestras")

    # 3. Preprocesamiento
    def prepare_batch(batch):
        audio_array, _ = librosa.load(batch["audio_path"], sr=settings.SAMPLE_RATE)
        batch["input_features"] = processor.feature_extractor(audio_array, sampling_rate=settings.SAMPLE_RATE).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    dataset = dataset.map(prepare_batch, remove_columns=dataset["train"].column_names, desc="Procesando audios")

    # 4. Argumentos de entrenamiento (USANDO Seq2SeqTrainingArguments)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(settings.MODEL_PATH),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=params.learning_rate,
        num_train_epochs=params.epochs,
        eval_strategy="steps",
        eval_steps=50,      
        save_strategy="steps",
        save_steps=50,      
        logging_steps=10,
        fp16=False,
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        load_best_model_at_end=True,     
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
        predict_with_generate=True,  # Ahora sí funcionará
        generation_max_length=225
    )

    # 5. Entrenador (USANDO Seq2SeqTrainer)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )

    # 6. Entrenamiento
    print("🚀 Iniciando proceso en CPU...")
    trainer.train()

    # 7. Guardar modelo
    print("💾 Guardando modelo ajustado...")
    os.makedirs(settings.MODEL_PATH, exist_ok=True)
    model.save_pretrained(settings.MODEL_PATH)
    processor.save_pretrained(settings.MODEL_PATH)

    # 8. Evaluar con datos reales
    print("📊 Calculando métricas finales...")
    predictions_output = trainer.predict(dataset["validation"])
    pred_ids = predictions_output.predictions
    label_ids = predictions_output.label_ids

    # Limpiar labels de padding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decodificar texto
    decoded_preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Guardar métricas reales en el archivo JSON
    save_metrics(
        references=decoded_labels,
        hypotheses=decoded_preds,
        loss=predictions_output.metrics.get("test_loss")
    ) 

    return {
        "status": "success",
        "message": "Entrenamiento y métricas finalizados",
        "details": {
            "eval_loss": predictions_output.metrics.get("test_loss")
        }
    }