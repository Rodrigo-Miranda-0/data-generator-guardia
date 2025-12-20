# Memory Bank - Guard-IA Phishing Detector

> Last Updated: December 19, 2025

---

## 📖 Project History & Development Journey

### The Problem
The goal was to build a phishing detector for Spanish corporate emails as part of a thesis project. The challenge: **there are almost no labeled Spanish phishing datasets available**. Most existing datasets are English-only, and real corporate email data is confidential.

### Phase 1: The Synthetic Data Approach (Failed)

**The Idea**: "Let's use an LLM to generate both phishing and legitimate emails, then train a classifier on them."

**What We Built**:
1. Created `data/gen_prompts.py` to generate parameterized prompts covering:
   - 10 phishing archetypes (account suspension, password reset, CEO fraud, etc.)
   - 8 legitimate archetypes (meeting invites, IT notices, etc.)
   - Variations in tone, urgency, links, attachments, language mix

2. Generated 540+ prompts that could be fed to LLMs

3. Built three generation scripts:
   - `generate_synthetic_ollama.py` - Local Ollama (recommended)
   - `generate_synthetic_kobold.py` - KoboldCpp
   - `generate_synthetic.py` - HuggingFace Transformers

**First Roadblock**: Llama 3.1 (aligned model) **refused to generate phishing emails** ~60% of the time:
> "Lo siento, pero no puedo cumplir con esa solicitud."

**Solution**: Switched to **Dolphin-Mistral** (uncensored model) which had 0% refusal rate.

**The Training Results**:
```
Accuracy:  100%
Precision: 100%
Recall:    100%
F1:        100%
```

**🚨 Red Flag**: 100% accuracy is almost always a sign of data leakage or the model learning shortcuts.

### Phase 2: The Investigation

We tested the "perfect" model with **8 manually written emails** (not from the synthetic dataset):

| Category | Correct | Accuracy |
|----------|---------|----------|
| Phishing | 0/4 | 0% |
| Legitimate | 4/4 | 100% |
| **Total** | **4/8** | **50%** |

**The model was useless on real data.** It predicted everything as legitimate.

**Root Cause Analysis**:

1. **The model learned to detect "LLM style" vs "human style"** - All phishing was synthetic, all legitimate was synthetic. The model just learned which LLM wrote what.

2. **Data Leakage in .eml files**: We also had ~1,000 real phishing .eml files from phishingpot.com. Analysis revealed 31% (313/1000) contained the word "phishing" in metadata (email addresses like `phishing@pot.com`).

3. **Synthetic patterns**: 74% of LLM-generated legitimate emails had detectable patterns:
   - Formulaic greetings: "Dear [Name],"
   - Predictable closings: "Best regards", "Thanks", "Cheers"
   - Repetitive names: Michael, Sarah, Robert

**Conclusion**: `P_synthetic(X,Y) ≠ P_real(X,Y)` - The distribution of synthetic data doesn't match real-world data.

### Phase 3: Finding Real Data

**Attempt: HuggingFace Dataset**

Downloaded `cybersectony/PhishingEmailDetectionv2.0`:
- 6,809 legitimate emails (Enron corpus - real corporate emails!)
- 6,684 "phishing" emails

**Problem**: The "phishing" samples were **generic spam**, not corporate phishing:
```
Dear ricardo1, COST EFFECTIVE Direct Email Advertising
Promote Your Business For As Low As $50 Per 1 Million Email Addresses...
```

This is spam marketing, not the credential-stealing phishing we needed.

**What We Kept**: The Enron legitimate emails were perfect - real corporate communication from 2001-2002.

### Phase 4: The Final Dataset

**Data Cleaning Steps**:

1. **Cleaned the .eml phishing files**:
   ```python
   # Remove emails with "phishing" in text (data leakage)
   clean_phishing = phishing[~phishing['email_text'].str.lower().str.contains('phishing')]
   # Result: 687 clean phishing emails (from 1,000)
   ```

2. **Discarded all synthetic legitimate emails** (100% removal)

3. **Sampled from Enron corpus** to balance classes:
   ```python
   enron_sample = enron_legit.sample(n=687, random_state=42)
   ```

**Final Dataset Composition**:

| Class | Source | Count |
|-------|--------|-------|
| Legitimate (0) | Enron Corpus (real) | 687 |
| Phishing (1) | .eml files (cleaned) | 687 |
| **Total** | | **1,374** |

### Phase 5: Final Training

**Configuration**:
- Model: `distilbert-base-uncased`
- Epochs: 3
- Batch size: 8
- Learning rate: 5e-5
- Hardware: Apple M-series (MPS)
- Time: ~3 minutes

**Results**:

| Metric | Value |
|--------|-------|
| Accuracy | 99.64% |
| Precision | 99.22% |
| Recall | 100.00% |
| F1 Score | 99.61% |

**Confusion Matrix** (Validation Set - 275 samples):
```
              Predicted
              Legit  Phishing
Actual Legit   145      1      (1 false positive)
       Phish     0    128      (0 false negatives)
```

### What the Model Learned (and Didn't)

**✅ Detects Well**:
- Package delivery phishing
- Fake invoice scams
- Account suspension threats
- Generic credential harvesting

**❌ Doesn't Detect**:
- BEC (Business Email Compromise / CEO fraud)
- Spear phishing (targeted attacks)
- Context-dependent threats

**Why**: The .eml dataset is consumer phishing (mass-market attacks). BEC requires understanding business context and sender relationships - this is handled by other layers in the full Guard-IA system.

### Key Lessons Learned

1. **Synthetic data is treacherous**: Models learn shortcuts, not patterns. Validation on synthetic data means nothing.

2. **Always test on truly held-out data**: Manual examples exposed the 100% model as useless.

3. **Data leakage is subtle**: "phishing@pot.com" in email headers was enough to give away the label.

4. **Aligned LLMs won't help generate "bad" content**: Safety filters block malicious content generation.

5. **Real data beats synthetic data**: Even old data (Enron 2001-2002) generalized better than fresh synthetic data.

6. **Domain matters**: Consumer phishing ≠ corporate phishing. The model needs data matching the deployment context.

---

## Project Overview

**Guard-IA** is a Spanish corporate email phishing detection system built as part of a thesis project. It uses a fine-tuned DistilBERT model to classify emails as legitimate (0) or phishing (1), with a focus on high recall to minimize false negatives.

### Repository
- **Name**: data-generator-guardia
- **GitHub**: git@github.com:Rodrigo-Miranda-0/data-generator-guardia.git
- **Language**: Python 3.x
- **Framework**: PyTorch + Hugging Face Transformers

---

## Project Architecture

```
data-generator-guardia/
├── model/                    # ML model code
│   ├── train.py             # Fine-tune DistilBERT
│   ├── inference.py         # PhishingDetector class
│   ├── explain.py           # Attention-based explainability
│   └── utils.py             # Shared utilities (metrics, seed, device)
├── data/                     # Data processing scripts
│   ├── gen_prompts.py       # Generate prompts (no LLM)
│   ├── generate_synthetic_ollama.py   # Generate emails via Ollama
│   ├── generate_synthetic_kobold.py   # Generate emails via KoboldCpp
│   ├── generate_synthetic.py          # Generate emails via HuggingFace
│   ├── parse_eml_dataset.py           # Parse .eml files
│   ├── clean_synthetic_data.py        # Data cleaning
│   └── prepare_training_data.py       # Prepare final dataset
├── artifacts/               # Saved models
├── model_cache/             # Cached tokenizer/config
├── run/
│   └── example_inference.py # Example usage
├── requirements.txt
└── Documentation files (.md)
```

---

## Key Components

### 1. Model Training (`model/train.py`)

**Base Model**: `distilbert-base-multilingual-cased` (default)

**Key Features**:
- Custom `EmailDataset` class wrapping tokenized emails
- 80/20 train/val split (configurable)
- Optimizes for recall (phishing detection priority)
- Supports FP16 mixed precision

**Training Command**:
```bash
python model/train.py \
    --csv_path data/corporate_phishing_dataset.csv \
    --output_dir artifacts/phishing-model \
    --epochs 3 \
    --batch_size 8 \
    --learning_rate 5e-5
```

**Current Best Model Metrics**:
| Metric | Value |
|--------|-------|
| Accuracy | 99.64% |
| Precision | 99.22% |
| Recall | 100.00% |
| F1 Score | 99.61% |

### 2. Inference (`model/inference.py`)

**Main Class**: `PhishingDetector`

```python
from model.inference import PhishingDetector

detector = PhishingDetector("artifacts/phishing-model")
result = detector.predict_email("Email text here")
# Returns: {"label": "phishing" or "legitimate", "score": float}

# Batch prediction
results = detector.predict_batch(["email1", "email2"])
```

### 3. Explainability (`model/explain.py`)

**Main Class**: `AttentionExplainer`

Uses attention weights from the last transformer layer to identify influential tokens:

```python
from model.explain import AttentionExplainer

explainer = AttentionExplainer("artifacts/phishing-model")
result = explainer.explain_email("Email text", top_k=5)
# Returns: {"top_tokens": [("token", score), ...]}
```

### 4. Data Generation Pipeline

**Step 1**: Generate prompts (`data/gen_prompts.py`)
- Creates parameterized prompts without calling LLM
- 10 phishing archetypes + 8 legitimate archetypes
- Varies: tone, urgency, links, attachments, language mix, length
- Output: `prompts.txt` (540 prompts with default settings)

**Step 2**: Generate emails (`data/generate_synthetic_ollama.py`)
- Uses Ollama API (localhost:11434)
- Default model: `llama3.1:8b`
- Incremental saving (resumable)
- Labels inferred from prompt text

```bash
# Generate prompts
python data/gen_prompts.py --output prompts.txt --samples_per_prompt 30

# Generate emails with Ollama
python data/generate_synthetic_ollama.py \
    --prompts_file prompts.txt \
    --output_csv synthetic_emails.csv \
    --model llama3.1:8b
```

---

## Dataset Evolution & Lessons Learned

### Version 1: Synthetic + .eml (Failed)
- **Problem**: 100% accuracy - too good to be true
- **Root Cause**: 
  - 31% of phishing emails contained "phishing" in metadata (data leakage)
  - 74% of synthetic legitimate emails had detectable LLM patterns
- **Conclusion**: Model learned artifacts, not phishing patterns

### Version 2: HuggingFace Dataset (Unsuitable)
- **Dataset**: `cybersectony/PhishingEmailDetectionv2.0`
- **Problem**: "Phishing" samples were generic spam, not corporate phishing
- **Enron legitimate emails**: Good quality but in English

### Version 3: Final Dataset (Current)
- **Phishing (687)**: Cleaned .eml files (removed "phishing" metadata leakage)
- **Legitimate (687)**: Enron corpus (real corporate emails)
- **Total**: 1,374 samples, 80/20 split

**Key Insight**: The model detects consumer phishing patterns well but struggles with:
- BEC (Business Email Compromise / CEO fraud)
- Spear phishing (targeted corporate attacks)

These gaps are addressed by other layers in the full system (rule-based + LLM warning).

---

## Important Files

| File | Purpose |
|------|---------|
| `model/train.py` | Main training script |
| `model/inference.py` | `PhishingDetector` class |
| `model/explain.py` | Attention-based explainability |
| `model/utils.py` | Metrics, seed, device utilities |
| `data/gen_prompts.py` | Prompt generation (no LLM) |
| `data/generate_synthetic_ollama.py` | Email generation via Ollama |
| `artifacts/phishing-model/` | Saved trained model |

---

## Dependencies

```
torch>=2.1.0
transformers>=4.36.0
scikit-learn>=1.3.0
pandas>=2.1.0
numpy>=1.24.0
tabulate>=0.9.0
```

---

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Training (with existing dataset)
python model/train.py --csv_path data/corporate_phishing_dataset.csv

# Inference
python run/example_inference.py

# Or in Python:
from model.inference import PhishingDetector
detector = PhishingDetector("artifacts/phishing-model")
detector.predict_email("Su cuenta será suspendida. Haga clic aquí.")
```

---

## Known Limitations

1. **BEC Detection**: Model doesn't detect CEO fraud / wire transfer requests well
2. **Language**: Trained primarily on English (Enron) + Spanish phishing mix
3. **Synthetic Data Issues**: Pure synthetic data caused overfitting to LLM patterns
4. **Domain Shift**: Real corporate traffic may differ from training distribution

---

## Architecture Decisions

1. **DistilBERT over BERT**: Faster inference, smaller model, sufficient for task
2. **Multilingual Cased**: Supports Spanish + English mixed content
3. **Recall Priority**: Business requirement - minimize false negatives
4. **Threshold Tuning**: Decision threshold can be adjusted for precision/recall tradeoff

---

## Future Improvements

1. Add real corporate email data for training
2. Implement few-shot learning for BEC detection
3. Add rule-based layer for URL/header analysis
4. Integrate with SMTP/Google Workspace pipeline
5. Add confidence calibration

---

## Related Documentation

- `README.md` - Project overview
- `ABOUT_THIS_PROJECT.md` - Synthetic data pipeline details
- `DOCUMENTACION_MODELO.md` - Complete model documentation (Spanish)
- `DATA_GENERATION.md` - Synthetic dataset planning
- `DATASET.md` - Training data format guide
- `SYNTHETIC_DATA_ANALYSIS.md` - Analysis of synthetic data issues
