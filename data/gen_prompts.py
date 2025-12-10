"""
Utility to generate a list of synthetic email-generation prompts for Spanish corporate phishing and legitimate lookalikes.

Run:
    python data/gen_prompts.py --output prompts.txt --samples_per_prompt 30

This does NOT call an LLM. It creates prompt strings you can feed to your chosen LLM
to generate actual emails. Adjust archetypes/tones/parameters as needed.
"""

import argparse
import itertools
import random
from pathlib import Path
from typing import List

random.seed(42)

# Archetypes for phishing and legitimate messages
PHISHING_ARCHETYPES = [
    "urgent account suspension",
    "password reset required",
    "invoice overdue payment",
    "CEO wire transfer request",
    "shipment held for customs payment",
    "security alert unusual login",
    "benefits or payroll update",
    "software update attachment",
    "shared drive access request",
    "tax document download",
]

LEGIT_ARCHETYPES = [
    "meeting invitation recurring",
    "legitimate invoice notification",
    "password reset confirmation",
    "IT maintenance notice",
    "travel itinerary update",
    "benefits enrollment reminder",
    "project status update",
    "shipping confirmation",
]

TONES = ["formal", "neutral", "slightly informal"]
URGENCY = ["no urgency", "mild urgency", "high urgency"]
LINK_PRESENCE = ["include a login link", "include a payment link", "no links"]
ATTACH_PRESENCE = ["mention a PDF attachment", "mention no attachments"]
LANGUAGE_MIX = ["pure Spanish", "mostly Spanish with a few English words"]
LENGTH = ["under 120 words", "under 180 words"]


def build_prompt(archetype: str, phishing: bool) -> str:
    tone = random.choice(TONES)
    urgency = random.choice(URGENCY)
    link = random.choice(LINK_PRESENCE)
    attach = random.choice(ATTACH_PRESENCE)
    lang = random.choice(LANGUAGE_MIX)
    length = random.choice(LENGTH)
    label_text = "phishing" if phishing else "legitimate"

    return (
        f"Write a {label_text} Spanish corporate email about {archetype}. "
        f"Tone: {tone}. Urgency: {urgency}. {link}. {attach}. "
        f"Language: {lang}. Length: {length}. Avoid placeholders; make it realistic."
    )


def expand_prompts(archetypes: List[str], phishing: bool, samples_per_prompt: int) -> List[str]:
    prompts = []
    for archetype in archetypes:
        for _ in range(samples_per_prompt):
            prompts.append(build_prompt(archetype, phishing))
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic prompts for phishing/legit emails.")
    parser.add_argument("--output", type=str, default="prompts.txt", help="Where to save the prompts.")
    parser.add_argument("--samples_per_prompt", type=int, default=20, help="Variants per archetype.")
    args = parser.parse_args()

    phishing_prompts = expand_prompts(PHISHING_ARCHETYPES, phishing=True, samples_per_prompt=args.samples_per_prompt)
    legit_prompts = expand_prompts(LEGIT_ARCHETYPES, phishing=False, samples_per_prompt=args.samples_per_prompt)

    all_prompts = phishing_prompts + legit_prompts
    random.shuffle(all_prompts)

    out_path = Path(args.output)
    out_path.write_text("\n".join(all_prompts), encoding="utf-8")
    print(f"Wrote {len(all_prompts)} prompts to {out_path}")


if __name__ == "__main__":
    main()

