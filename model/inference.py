"""
Inference utilities for phishing detection.
"""

from typing import Dict, List, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model.utils import get_device

LABELS = {0: "legitimate", 1: "phishing"}
DEFAULT_MODEL_DIR = "artifacts/phishing-model"


class PhishingDetector:
    """
    Wrapper for loading a fine-tuned model and running predictions.
    """

    def __init__(self, model_dir: str, device: torch.device | None = None) -> None:
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_email(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict a single email.
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)
        outputs = self.model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        score, idx = torch.max(probs, dim=-1)
        return {"label": LABELS[idx.item()], "score": float(score.item())}

    @torch.inference_mode()
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict a batch of emails.
        """
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)
        outputs = self.model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)
        scores, idxs = torch.max(probs, dim=-1)
        return [
            {"label": LABELS[idx.item()], "score": float(score.item())} for score, idx in zip(scores, idxs)
        ]


# Convenience function matching requested signature
def predict_email(text: str, model_dir: str = DEFAULT_MODEL_DIR) -> Dict[str, Union[str, float]]:
    """
    Convenience prediction with default model location.
    """
    detector = PhishingDetector(model_dir=model_dir)
    return detector.predict_email(text)

