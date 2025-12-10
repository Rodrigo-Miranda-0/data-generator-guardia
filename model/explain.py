"""
Lightweight explainability using attention weights.
Returns top influential tokens for a given email.
"""

from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model.utils import get_device


class AttentionExplainer:
    """
    Produces token-level importances from the model's last-layer attention.
    """

    def __init__(self, model_dir: str, device: torch.device | None = None) -> None:
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def explain_email(self, text: str, top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Return top influential tokens by attention from [CLS]/[SOS].
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        outputs = self.model(**encoded, output_attentions=True)
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("Model did not return attentions. Ensure output_attentions=True.")

        last_attn = attentions[-1]  # shape: (batch, heads, seq, seq)
        cls_attn = last_attn[0, :, 0, :]  # attention from CLS to others
        token_importance = cls_attn.mean(dim=0)  # average over heads

        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        scores = token_importance.tolist()

        token_scores = []
        for tok, score in zip(tokens, scores):
            if tok in self.tokenizer.all_special_tokens:
                continue
            token_scores.append((tok, float(score)))

        # Sort by importance descending
        token_scores = sorted(token_scores, key=lambda x: x[1], reverse=True)[:top_k]
        return {"top_tokens": token_scores}




