"""
Generate synthetic Spanish emails (phishing + legitimate) using Ollama.

Prereqs:
    - Ollama installed and running (ollama serve)
    - Model pulled: ollama pull llama3.1:8b

Usage:
    python data/generate_synthetic_ollama.py \
        --prompts_file prompts.txt \
        --output_csv synthetic_emails.csv \
        --model llama3.1:8b \
        --num_samples_per_prompt 1

Notes:
    - Prompts containing "phishing" get label=1, "legitimate" get label=0
    - Progress is saved incrementally to avoid data loss on interruption
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Tuple
import urllib.request
import urllib.error

OLLAMA_API_URL = "http://localhost:11434/api/generate"


def read_prompts(path: Path) -> List[str]:
    """Read prompts from file, one per line."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_label_from_prompt(prompt: str) -> int:
    """Infer label from prompt text: 1=phishing, 0=legitimate."""
    p = prompt.lower()
    if "phishing" in p:
        return 1
    if "legitimate" in p:
        return 0
    # Default to phishing if unclear
    return 1


def clean_generated_text(text: str, prompt: str) -> str:
    """Clean the generated text, removing any prompt echo."""
    # Remove the prompt if it was echoed back
    if text.startswith(prompt):
        text = text[len(prompt):]
    
    # Clean up whitespace
    text = text.strip()
    
    # Remove common prefixes like "Here's the email:" or "Subject:"
    lines = text.split('\n')
    cleaned_lines = []
    started = False
    
    for line in lines:
        line_lower = line.lower().strip()
        # Skip meta-commentary lines
        if any(skip in line_lower for skip in [
            "here's", "here is", "below is", "i'll write", "i will write",
            "sure,", "certainly", "of course"
        ]) and not started:
            continue
        started = True
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def generate_with_ollama(
    prompt: str,
    model: str = "llama3.1:8b",
    temperature: float = 0.9,
    max_tokens: int = 300,
) -> str:
    """Generate text using Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running (ollama serve)")
        raise


def generate_samples(
    prompts: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
    num_samples_per_prompt: int,
    output_path: Path,
    resume: bool = True,
) -> List[Tuple[str, int]]:
    """Generate samples from all prompts and save incrementally."""
    
    rows: List[Tuple[str, int]] = []
    start_index = 0
    
    # Resume from existing file if it exists
    if resume and output_path.exists():
        with output_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_rows = list(reader)
            if len(existing_rows) > 1:  # Has header + data
                rows = [(row[0], int(row[1])) for row in existing_rows[1:]]
                # Calculate which prompt to resume from
                start_index = len(rows) // num_samples_per_prompt
                print(f"Resuming from prompt {start_index + 1}/{len(prompts)} ({len(rows)} samples already generated)")
    
    total_prompts = len(prompts)
    
    # Open file in append mode if resuming, otherwise write mode
    mode = "a" if rows else "w"
    
    with output_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        
        # Write header only if starting fresh
        if not rows:
            writer.writerow(["email_text", "label"])
        
        for i, prompt in enumerate(prompts[start_index:], start=start_index + 1):
            label = infer_label_from_prompt(prompt)
            label_str = "phishing" if label == 1 else "legitimate"
            
            for sample_num in range(num_samples_per_prompt):
                print(f"[{i}/{total_prompts}] Generating {label_str} sample {sample_num + 1}/{num_samples_per_prompt}...", end=" ", flush=True)
                
                start_time = time.time()
                try:
                    generated = generate_with_ollama(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    
                    email_text = clean_generated_text(generated, prompt)
                    
                    if email_text:  # Only save non-empty results
                        writer.writerow([email_text, label])
                        f.flush()  # Ensure it's written to disk
                        rows.append((email_text, label))
                        elapsed = time.time() - start_time
                        print(f"done ({elapsed:.1f}s, {len(email_text)} chars)")
                    else:
                        print("skipped (empty output)")
                        
                except Exception as e:
                    print(f"error: {e}")
                    continue
    
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic emails with Ollama.")
    parser.add_argument("--prompts_file", type=str, default="prompts.txt", help="File with one prompt per line.")
    parser.add_argument("--output_csv", type=str, default="synthetic_emails.csv", help="Output CSV path.")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Ollama model name.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=300, help="Max tokens to generate per sample.")
    parser.add_argument("--num_samples_per_prompt", type=int, default=1, help="Number of samples per prompt.")
    parser.add_argument("--no_resume", action="store_true", help="Start fresh instead of resuming.")
    args = parser.parse_args()
    
    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    output_path = Path(args.output_csv)
    
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Samples per prompt: {args.num_samples_per_prompt}")
    print(f"Output: {output_path}")
    print()
    
    # Test connection to Ollama
    print("Testing connection to Ollama...", end=" ")
    try:
        generate_with_ollama("Say 'OK' in one word.", args.model, 0.1, 10)
        print("OK")
    except Exception as e:
        print(f"\nFailed to connect to Ollama: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. Open a new terminal")
        print("  2. Run: ollama serve")
        print("  3. Then run this script again")
        return
    
    prompts = read_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts")
    
    phishing_count = sum(1 for p in prompts if "phishing" in p.lower())
    legit_count = len(prompts) - phishing_count
    print(f"  - Phishing prompts: {phishing_count}")
    print(f"  - Legitimate prompts: {legit_count}")
    print()
    
    rows = generate_samples(
        prompts=prompts,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_samples_per_prompt=args.num_samples_per_prompt,
        output_path=output_path,
        resume=not args.no_resume,
    )
    
    # Final stats
    phishing_generated = sum(1 for _, label in rows if label == 1)
    legit_generated = sum(1 for _, label in rows if label == 0)
    
    print()
    print("=" * 50)
    print(f"Generation complete!")
    print(f"Total samples: {len(rows)}")
    print(f"  - Phishing (label=1): {phishing_generated}")
    print(f"  - Legitimate (label=0): {legit_generated}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

