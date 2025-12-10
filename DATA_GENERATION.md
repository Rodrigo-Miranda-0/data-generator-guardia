# Synthetic Dataset Plan (Phishing Detector POC)

## Purpose
Generate a small synthetic dataset to bootstrap a Spanish corporate phishing detector. Use it as a starter set; replace/augment with real labeled emails as soon as possible.

## What to generate
- Phishing variants: credential reset/harvest, urgent suspension, invoice/payment issues, shipment problems, CEO/CFO wire requests, attachment malware, shortened/obfuscated links, mixed ES/EN, typos/noise.
- Legitimate lookalikes: real password resets, real invoices/receipts, calendar invites, IT notices, newsletters. These “hard negatives” reduce false positives.

## Prompt patterns (examples)
- “Write a short Spanish corporate phishing email about urgent account suspension. Include a login link. Tone: formal. Under 150 words.”
- “Spanish phishing email requesting payment of an overdue invoice with a link to view the invoice. Include a sense of urgency. Under 120 words.”
- “Spanish CEO fraud asking finance to wire funds today; no link, but include a PDF attachment mention.”
- “Legitimate Spanish meeting invite for a recurring team sync with a calendar link.”
- “Legitimate Spanish invoice notification with a support contact; no threat/urgency language.”
Tips: create ~25–40 prompts covering phishing archetypes and legit lookalikes; request 20–50 variants per prompt.

## LLM options (cost vs. quality)
- Local/small (cheapest): Llama 3 8B Instruct, Mistral 7B/8B Instruct. Good for bulk; expect to filter odd outputs.
- Hosted/cheap: mid-tier endpoints (e.g., GPT-4o-mini class, small Mistral/Anthropic tiers). Better fluency for a small cost.
- Higher fidelity (pay a bit more): use a mid-tier hosted model for the initial batch, then stop once you have a few thousand samples.

## Volume targets (POC)
- Synthetic-only POC: ~2k–5k emails total, roughly balanced or slightly phishing-heavy to favor recall.
- Per prompt: 20–50 variants × ~25–40 prompts to cover diversity quickly.
- Keep synthetic a minority once real labeled data arrives (<20–30% of final mix).

## Quality checks (lightweight)
- Auto-filter: language mostly Spanish, length within 50–320 tokens, presence/absence of links/attachments per spec, reject malformed HTML.
- Deduplicate near-identical messages.
- Spot-check a small random subset (e.g., 100–200) for realism and correct labels.

## Minimal workflow
1) Define prompt set (phishing + legit) and small dictionaries for brands/roles/amounts/links to add variety.
2) Generate in batches from the LLM; collect raw texts.
3) Auto-filter by language/length/markers; drop off-spec items.
4) Deduplicate; save to CSV `email_text,label`.
5) Quick human spot-check; fix labels if needed.
6) Train the model; tune decision threshold for high recall.

## Caveats
- Synthetic data ≠ real corporate traffic; expect to retrain once real data is available.
- Avoid over-regular templates; inject variation (tone, typos, mixed ES/EN) to reduce brittleness.
- Keep legit lookalikes strong; they are key to avoiding false positives.


