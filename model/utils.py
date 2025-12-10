import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Pick GPU if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    model_name: str, num_labels: int = 2, cache_dir: Optional[str] = None
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load a pretrained DistilBERT model and tokenizer for sequence classification.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
    )
    return model, tokenizer


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def compute_classification_metrics(preds: np.ndarray, labels: np.ndarray) -> Metrics:
    """
    Compute classification metrics for binary labels.
    """
    pred_labels = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, zero_division=0)
    recall = recall_score(labels, pred_labels, zero_division=0)
    f1 = f1_score(labels, pred_labels, zero_division=0)
    conf = confusion_matrix(labels, pred_labels, labels=[0, 1])
    return Metrics(accuracy, precision, recall, f1, conf)


def readable_confusion(conf: np.ndarray) -> List[List[int]]:
    """
    Convert confusion matrix to list for logging/reporting.
    """
    return conf.tolist()




