# Training Data Guide (Spanish Email Phishing)

## 1) Format
- CSV columns: `email_text` (Spanish email text), `label` (0 = legitimate, 1 = phishing).
- UTF-8. Keep subject + body together if possible.
- Light HTML is fine; strip scripts/boilerplate if they drown the text.

## 2) Splits
- Train/Val/Test (e.g., 70/15/15), stratified to keep class balance.
- Avoid leakage: keep similar emails (same thread/sender/campaign) in the same split.

## 3) Size (good enough vs. better)
- Minimum workable MVP: 2k–5k labeled emails with reasonable balance.
- Better: 10k–20k total, with at least several hundred phishing samples (thousands ideal).
- If phishing is rare, consider class weights or light oversampling—avoid heavy duplication.

## 4) Labeling basics
- Clear rubric: what is phishing vs. legitimate; how to treat marketing spam; what to do with uncertain cases (mark as “ignore”).
- Double-check tricky samples; deduplicate near-identical emails.

## 5) Content to include
- Phishing cues: urgency (“urgente”, “verificación”, “suspensión”), credentials, payments/invoices, links/attachments, brand impersonation, mixed ES/EN, typos.
- Legitimate lookalikes: real invoices, real password resets, meeting reminders—helps reduce false positives.
- Keep URLs/attachment mentions; they’re strong signals.

## 6) Preprocessing (light)
- Keep casing (use cased DistilBERT). Max length ~256–320 tokens; truncate long signatures/threads.
- Remove repetitive footers/banners if they dominate; keep meaningful text.

## 7) Business focus (recall)
- Tune threshold on validation to minimize false negatives; then inspect precision trade-offs.
- Keep a small “recent attacks” validation slice to watch new tactics.

## 8) Privacy
- Anonymize PII if required (replace with placeholders like `[EMAIL]`), but keep structure markers.

## 9) Monitoring
- Freeze a test set for comparison.
- Collect production false positives/negatives and retrain periodically.

## Example rows (CSV)
```
email_text,label
"Estimado usuario, su cuenta necesita verificación urgente. Haga clic en el enlace.",1
"Recordatorio de la reunión mensual mañana a las 10 AM en la sala principal.",0
"Se detectó actividad sospechosa. Restablezca su contraseña inmediatamente.",1
```

