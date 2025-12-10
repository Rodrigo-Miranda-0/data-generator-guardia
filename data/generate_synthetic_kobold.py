"""
Generate synthetic Spanish emails (phishing + legitimate) using a running KoboldCpp server (llama.cpp API).

Prereqs:
- Start KoboldCpp with API enabled, e.g.:
    .\\koboldcpp.exe --model C:\\path\\to\\llama-3-8b-instruct.Q4_K_M.gguf --host 0.0.0.0 --port 5001 --api --threads 8
- Ensure the endpoint http://localhost:5001/api/v1/generate is reachable.
- Run data/gen_prompts.py first to create prompts.txt.

Usage:
    python data/generate_synthetic_kobold.py --prompts_file prompts.txt --output_csv synthetic_emails.csv --api_url http://localhost:5001/api/v1/generate
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

import requests

random.seed(42)


def read_prompts(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_label_from_prompt(prompt: str) -> int:
    p = prompt.lower()
    if "phishing" in p:
        return 1
    if "legitimate" in p:
        return 0
    return 1


def generate_for_prompt(api_url: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    payload = {
        "prompt": prompt,
        "max_length": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop_sequence": [],
    }
    resp = requests.post(api_url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Kobold/llama.cpp style: "results" -> [{"text": "..."}]
    return data.get("results", [{}])[0].get("text", "").strip()


def generate_samples(
    api_url: str,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    for prompt in prompts:
        label = infer_label_from_prompt(prompt)
        text = generate_for_prompt(api_url, prompt, max_new_tokens, temperature, top_p)
        rows.append((text, label))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic emails via KoboldCpp API.")
    parser.add_argument("--prompts_file", type=str, required=True, help="File with one prompt per line.")
    parser.add_argument("--output_csv", type=str, default="synthetic_emails.csv", help="Where to save CSV.")
    parser.add_argument("--api_url", type=str, default="http://localhost:5001/api/v1/generate", help="Kobold API URL.")
    parser.add_argument("--max_new_tokens", type=int, default=180, help="Max tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling.")
    args = parser.parse_args()

    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    prompts = read_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts")

    rows = generate_samples(
        api_url=args.api_url,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
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

