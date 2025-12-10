"""
Generate synthetic Spanish emails (phishing + legitimate) using a local LLM.

Prereqs:
    - Download a local instruction-tuned model (e.g., Llama-3-8B-Instruct, Mistral-7B-Instruct).
    - Enough VRAM for the model (8–12GB for 7–8B with 4/8-bit loading if desired).

Usage:
    python data/generate_synthetic.py \
        --model_name_or_path /path/to/local/model \
        --prompts_file prompts.txt \
        --output_csv synthetic_emails.csv \
        --max_new_tokens 180 \
        --temperature 0.9 \
        --top_p 0.9

Notes:
    - Prompts are assumed to contain the words "phishing" or "legitimate" to derive labels.
    - This script is intended for small/medium local models; adjust batch_size if you hit OOM.
"""

import argparse
import csv
import random
import re
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

random.seed(42)
torch.manual_seed(42)


def read_prompts(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_label_from_prompt(prompt: str) -> int:
    p = prompt.lower()
    if "phishing" in p:
        return 1
    if "legitimate" in p:
        return 0
    # default to phishing if unclear, but logically you should ensure prompts are explicit
    return 1


def clean_text(text: str) -> str:
    # Remove the prompt echo if model includes it; keep only the generated part after the last newline
    return text.strip()


def generate_samples(
    gen_pipe,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    for prompt in prompts:
        label = infer_label_from_prompt(prompt)
        outputs = gen_pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=gen_pipe.tokenizer.eos_token_id,
        )
        for out in outputs:
            text = out["generated_text"]
            email_text = clean_text(text)
            rows.append((email_text, label))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic emails with a local LLM.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Local path or HF model name.")
    parser.add_argument("--prompts_file", type=str, required=True, help="File with one prompt per line.")
    parser.add_argument("--output_csv", type=str, default="synthetic_emails.csv", help="Output CSV path.")
    parser.add_argument("--max_new_tokens", type=int, default=180, help="Max tokens to generate per sample.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Samples per prompt.")
    parser.add_argument("--batch_size", type=int, default=1, help="Generation batch size.")
    args = parser.parse_args()

    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        batch_size=args.batch_size,
    )

    prompts = read_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts")

    rows = generate_samples(
        gen_pipe,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences,
    )

    out_path = Path(args.output_csv)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["email_text", "label"])
        for email_text, label in rows:
            writer.writerow([email_text, label])

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

