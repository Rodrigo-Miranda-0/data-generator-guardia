"""
Clean synthetic email dataset by removing:
1. Model refusals (e.g., "Lo siento, pero no puedo cumplir...")
2. Duplicate emails
3. Emails that are too short or too long
4. Emails not primarily in Spanish

Usage:
    python data/clean_synthetic_data.py --input synthetic_emails.csv --output synthetic_emails_clean.csv

Options:
    --input         Input CSV file (default: synthetic_emails.csv)
    --output        Output CSV file (default: synthetic_emails_clean.csv)
    --min_length    Minimum character length (default: 150)
    --max_length    Maximum character length (default: 2000)
    --dry_run       Show what would be removed without saving
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple, Set
from collections import Counter


# Patterns that indicate model refused to generate phishing
REFUSAL_PATTERNS = [
    r"lo siento.*no puedo",
    r"i can'?t (fulfill|complete|assist|help|write)",
    r"i cannot (fulfill|complete|assist|help|write)",
    r"i'?m not able to",
    r"i will not",
    r"i won'?t",
    r"is there something else i can help",
    r"no puedo cumplir",
    r"no me es posible",
    r"no estoy en posición de",
    r"cannot create.*phishing",
    r"can'?t create.*phishing",
    r"won'?t write.*phishing",
]

# Patterns that indicate the text is meta-commentary, not an actual email
META_PATTERNS = [
    r"^(here'?s?|aquí está|este es) (the|un|el|a) (correo|email)",
    r"^(sure|certainly|of course|claro|por supuesto)",
    r"^i'?d be happy to",
    r"^i'?ll (write|create|draft)",
]

# High confidence Spanish indicators (expanded)
SPANISH_INDICATORS = [
    # Greetings/closings
    "estimado", "estimada", "atentamente", "saludos", "cordiales",
    "querido", "querida", "hola", "buenos días", "buenas tardes",
    # Common words
    "por favor", "gracias", "adjunto", "factura", "cuenta",
    "empresa", "señor", "señora", "urgente", "importante",
    # Verbs/phrases common in Spanish emails
    "informar", "solicitar", "necesario", "seguridad", "contraseña",
    "acceso", "enlace", "verificar", "actualizar", "confirmar",
    "inmediatamente", "departamento", "correo", "mensaje", "envío",
    "pago", "plataforma", "sistema", "datos", "información",
]

# English indicators (if too many, it's not Spanish)
ENGLISH_INDICATORS = [
    "dear", "sincerely", "regards", "please find", "attached",
    "kindly", "request", "would like", "i hope this", "best regards",
]


def is_refusal(text: str) -> bool:
    """Check if text is a model refusal to generate content."""
    text_lower = text.lower().strip()
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def is_meta_commentary(text: str) -> bool:
    """Check if text starts with meta-commentary instead of actual email."""
    text_lower = text.lower().strip()
    for pattern in META_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def is_mostly_spanish(text: str, threshold: float = 0.3) -> bool:
    """
    Check if text is primarily Spanish.
    Uses heuristic: presence of Spanish indicators vs English indicators.
    """
    text_lower = text.lower()
    
    # Count Spanish indicators
    spanish_count = sum(1 for ind in SPANISH_INDICATORS if ind in text_lower)
    
    # Count English indicators
    english_count = sum(1 for ind in ENGLISH_INDICATORS if ind in text_lower)
    
    # If more English than Spanish, reject
    if english_count > spanish_count:
        return False
    
    # Need at least 2 Spanish indicators
    return spanish_count >= 2


def is_valid_length(text: str, min_len: int, max_len: int) -> bool:
    """Check if text length is within acceptable range."""
    return min_len <= len(text) <= max_len


def clean_email_text(text: str) -> str:
    """Clean up the email text."""
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove excessive newlines (more than 2 in a row)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove markdown artifacts that might have been generated
    text = re.sub(r'^\*\*', '', text)
    text = re.sub(r'\*\*$', '', text)
    
    return text


def analyze_removals(
    rows: List[Tuple[str, int]],
    min_length: int,
    max_length: int,
) -> dict:
    """Analyze what would be removed and why."""
    stats = {
        'total': len(rows),
        'refusals': 0,
        'meta_commentary': 0,
        'too_short': 0,
        'too_long': 0,
        'not_spanish': 0,
        'duplicates': 0,
        'valid': 0,
    }
    
    seen_texts: Set[str] = set()
    refusal_examples = []
    
    for text, label in rows:
        text_clean = clean_email_text(text)
        
        if is_refusal(text_clean):
            stats['refusals'] += 1
            if len(refusal_examples) < 5:
                refusal_examples.append(text_clean[:60])
            continue
            
        if is_meta_commentary(text_clean):
            stats['meta_commentary'] += 1
            continue
            
        if len(text_clean) < min_length:
            stats['too_short'] += 1
            continue
            
        if len(text_clean) > max_length:
            stats['too_long'] += 1
            continue
            
        if not is_mostly_spanish(text_clean):
            stats['not_spanish'] += 1
            continue
            
        # Check for duplicates
        text_normalized = text_clean.lower().strip()
        if text_normalized in seen_texts:
            stats['duplicates'] += 1
            continue
        seen_texts.add(text_normalized)
        
        stats['valid'] += 1
    
    stats['refusal_examples'] = refusal_examples
    return stats


def clean_dataset(
    rows: List[Tuple[str, int]],
    min_length: int,
    max_length: int,
) -> Tuple[List[Tuple[str, int]], dict]:
    """Clean the dataset and return valid rows with stats."""
    
    clean_rows: List[Tuple[str, int]] = []
    seen_texts: Set[str] = set()
    
    stats = {
        'refusals': 0,
        'meta_commentary': 0,
        'too_short': 0,
        'too_long': 0,
        'not_spanish': 0,
        'duplicates': 0,
    }
    
    for text, label in rows:
        text_clean = clean_email_text(text)
        
        # Check each filter
        if is_refusal(text_clean):
            stats['refusals'] += 1
            continue
            
        if is_meta_commentary(text_clean):
            stats['meta_commentary'] += 1
            continue
            
        if len(text_clean) < min_length:
            stats['too_short'] += 1
            continue
            
        if len(text_clean) > max_length:
            stats['too_long'] += 1
            continue
            
        if not is_mostly_spanish(text_clean):
            stats['not_spanish'] += 1
            continue
        
        # Check for duplicates (case-insensitive)
        text_normalized = text_clean.lower().strip()
        if text_normalized in seen_texts:
            stats['duplicates'] += 1
            continue
        seen_texts.add(text_normalized)
        
        clean_rows.append((text_clean, label))
    
    return clean_rows, stats


def read_csv(path: Path) -> List[Tuple[str, int]]:
    """Read CSV and return list of (text, label) tuples."""
    rows = []
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row['email_text'], int(row['label'])))
    return rows


def write_csv(path: Path, rows: List[Tuple[str, int]]) -> None:
    """Write cleaned data to CSV."""
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['email_text', 'label'])
        for text, label in rows:
            writer.writerow([text, label])


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean synthetic email dataset.")
    parser.add_argument("--input", type=str, default="synthetic_emails.csv", help="Input CSV file.")
    parser.add_argument("--output", type=str, default="synthetic_emails_clean.csv", help="Output CSV file.")
    parser.add_argument("--min_length", type=int, default=150, help="Minimum text length in characters.")
    parser.add_argument("--max_length", type=int, default=2000, help="Maximum text length in characters.")
    parser.add_argument("--dry_run", action="store_true", help="Show stats without saving.")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Reading {input_path}...")
    rows = read_csv(input_path)
    print(f"Loaded {len(rows)} rows")
    
    # Count original label distribution
    original_labels = Counter(label for _, label in rows)
    print(f"  - Phishing (1): {original_labels[1]}")
    print(f"  - Legitimate (0): {original_labels[0]}")
    print()
    
    if args.dry_run:
        print("=== DRY RUN - Analyzing what would be removed ===")
        stats = analyze_removals(rows, args.min_length, args.max_length)
        print(f"\nRemoval breakdown:")
        print(f"  - Refusals:        {stats['refusals']}")
        print(f"  - Meta-commentary: {stats['meta_commentary']}")
        print(f"  - Too short (<{args.min_length}): {stats['too_short']}")
        print(f"  - Too long (>{args.max_length}):  {stats['too_long']}")
        print(f"  - Not Spanish:     {stats['not_spanish']}")
        print(f"  - Duplicates:      {stats['duplicates']}")
        print(f"\nWould keep: {stats['valid']} / {stats['total']} rows")
        
        if stats['refusal_examples']:
            print(f"\nExample refusals:")
            for ex in stats['refusal_examples']:
                print(f"  - \"{ex}...\"")
        return
    
    # Clean the dataset
    print("Cleaning dataset...")
    clean_rows, stats = clean_dataset(rows, args.min_length, args.max_length)
    
    total_removed = sum(stats.values())
    print(f"\nRemoved {total_removed} rows:")
    print(f"  - Refusals:        {stats['refusals']}")
    print(f"  - Meta-commentary: {stats['meta_commentary']}")
    print(f"  - Too short (<{args.min_length}): {stats['too_short']}")
    print(f"  - Too long (>{args.max_length}):  {stats['too_long']}")
    print(f"  - Not Spanish:     {stats['not_spanish']}")
    print(f"  - Duplicates:      {stats['duplicates']}")
    
    # Count final label distribution
    final_labels = Counter(label for _, label in clean_rows)
    print(f"\nFinal dataset:")
    print(f"  - Total rows: {len(clean_rows)}")
    print(f"  - Phishing (1): {final_labels[1]}")
    print(f"  - Legitimate (0): {final_labels[0]}")
    
    # Calculate balance
    if final_labels[0] > 0 and final_labels[1] > 0:
        ratio = final_labels[1] / final_labels[0]
        print(f"  - Ratio (phishing/legit): {ratio:.2f}")
    
    # Save
    write_csv(output_path, clean_rows)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

