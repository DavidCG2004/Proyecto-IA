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
    dataset = load_training_dataset(
        "mozilla-foundation/common_voice_13_0",
        "es",
        split={
            "train": "train[:1%]",
            "validation": "validation[:1%]"
        }
    )

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # =========================
    # 3. Preprocesamiento
    # =========================
    def prepare_batch(batch):
        audio = batch["audio"]["array"]
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        with processor.as_target_processor():
            labels = processor(batch["sentence"]).input_ids

        batch["input_features"] = inputs.input_features[0]
        batch["labels"] = labels
        return batch

    dataset = dataset.map(
        prepare_batch,
        remove_columns=dataset["train"].column_names
    )

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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
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
