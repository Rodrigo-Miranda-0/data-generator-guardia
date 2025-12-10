"""
Fine-tune DistilBERT for phishing detection in Spanish emails.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from model.utils import compute_classification_metrics, load_model_and_tokenizer, readable_confusion, set_seed

DEFAULT_MODEL_NAME = "distilbert-base-multilingual-cased"
SEED = 42


class EmailDataset(Dataset):
    """
    Simple torch Dataset wrapping tokenized emails.
    """

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load dataset from CSV or generate a small mock dataset for demonstration.
    Expected columns: email_text, label (0 legit, 1 phishing).
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        data = {
            "email_text": [
                "Estimado usuario, su cuenta necesita verificación urgente. Haga clic en el enlace.",
                "Recordatorio de la reunión mensual mañana a las 10 AM en la sala principal.",
                "Se detectó actividad sospechosa. Restablezca su contraseña inmediatamente.",
                "Gracias por su compra, adjuntamos la factura.",
                "Su correo excede el límite. Valide su cuenta ahora para evitar suspensión.",
                "Invitación al almuerzo del equipo este viernes.",
            ],
            "label": [1, 0, 1, 0, 1, 0],
        }
        df = pd.DataFrame(data)
    return df


def train_val_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministic train/validation split.
    """
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = max(1, int(len(df_shuffled) * val_ratio))
    val_df = df_shuffled.iloc[:val_size]
    train_df = df_shuffled.iloc[val_size:]
    return train_df, val_df


def tokenize_dataset(tokenizer: AutoTokenizer, texts: List[str], max_length: int = 256) -> Dict[str, torch.Tensor]:
    """
    Tokenize a list of texts.
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    metrics = compute_classification_metrics(np.array(logits), np.array(labels))
    return metrics.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for phishing detection (Spanish).")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to CSV with email_text,label columns.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="HF model name or local path.")
    parser.add_argument("--output_dir", type=str, default="artifacts/phishing-model", help="Where to save the model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Train and eval batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision if available.")
    args = parser.parse_args()

    set_seed(SEED)

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_dataset(args.csv_path)
    train_df, val_df = train_val_split(df, val_ratio=args.val_ratio, seed=SEED)

    model, tokenizer = load_model_and_tokenizer(args.model_name, num_labels=2)

    train_enc = tokenize_dataset(tokenizer, train_df["email_text"].tolist(), max_length=args.max_length)
    val_enc = tokenize_dataset(tokenizer, val_df["email_text"].tolist(), max_length=args.max_length)

    train_dataset = EmailDataset(train_enc, train_df["label"].tolist())
    val_dataset = EmailDataset(val_enc, val_df["label"].tolist())

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_recall",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        fp16=args.fp16 and torch.cuda.is_available(),
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    # Compute confusion matrix with final model predictions on validation set
    preds = trainer.predict(val_dataset)
    metrics = compute_classification_metrics(preds.predictions, preds.label_ids)
    conf_table = readable_confusion(metrics.confusion)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Validation metrics:")
    print(tabulate([metrics.as_dict()], headers="keys", floatfmt=".4f"))
    print("Confusion matrix [[tn, fp], [fn, tp]]:")
    print(tabulate(conf_table, tablefmt="grid"))


if __name__ == "__main__":
    main()




