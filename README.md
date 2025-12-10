# Guard-IA Phishing Detector (MVP)

Minimal DistilBERT-based phishing detector for Spanish corporate emails. Uses Hugging Face `transformers` and PyTorch with a focus on recall to minimize false negatives.

## Project Structure
```
model/
  train.py         # Fine-tune DistilBERT
  inference.py     # Prediction utilities
  explain.py       # Attention-based token importance
  utils.py         # Shared helpers
run/
  example_inference.py
requirements.txt
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Training
Fine-tune with your CSV (`email_text,label`):
```bash
python model/train.py --csv_path path/to/data.csv --output_dir artifacts/phishing-model
```
Defaults to `distilbert-base-multilingual-cased`. If no CSV is provided, a small mock dataset is used.

Key flags: `--epochs`, `--batch_size`, `--learning_rate`, `--max_length`, `--fp16`.

Outputs: accuracy, precision, recall, F1, confusion matrix (prioritizing recall).

## Inference
```bash
python run/example_inference.py  # uses artifacts/phishing-model
```
Or programmatically:
```python
from model.inference import PhishingDetector

detector = PhishingDetector("artifacts/phishing-model")
detector.predict_email("Texto de correo en español")
detector.predict_batch(["uno", "dos"])
```

## Explainability
Attention-based token importance:
```python
from model.explain import AttentionExplainer
explainer = AttentionExplainer("artifacts/phishing-model")
explainer.explain_email("Texto del correo", top_k=5)
```

## Notes
- Uses seed for reproducibility (CPU/GPU).
- GPU optional; FP16 supported when available.
- Suitable for integration into SMTP/Google Workspace pipelines. Keep the model directory accessible to the inference service.




