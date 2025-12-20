#!/usr/bin/env python3
"""
Complete data preparation pipeline for phishing detection model.
Parses .eml files, cleans content, and prepares balanced training data.

Usage:
    python data/prepare_training_data.py \
        --phishing_dir /path/to/eml/files \
        --output_csv data/training_data.csv \
        --max_samples 2000
"""

import argparse
import email
import os
import re
import csv
import random
import hashlib
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Optional, Tuple, List
import html
from html.parser import HTMLParser
from collections import Counter


class HTMLTextExtractor(HTMLParser):
    """Extract clean text from HTML content."""
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_data = False
        self.skip_tags = {'script', 'style', 'head', 'meta', 'link', 'noscript'}
        
    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.skip_data = True
            
    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.skip_data = False
        # Add newlines for block elements
        if tag in ('p', 'div', 'br', 'li', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'td', 'th'):
            self.text_parts.append('\n')
            
    def handle_data(self, data):
        if not self.skip_data:
            self.text_parts.append(data)
            
    def get_text(self) -> str:
        return ''.join(self.text_parts)


def html_to_text(html_content: str) -> str:
    """Convert HTML to plain text, removing all markup."""
    try:
        # Decode HTML entities
        html_content = html.unescape(html_content)
        
        # Remove common obfuscation patterns
        html_content = re.sub(r'=\r?\n', '', html_content)  # Quoted-printable line continuations
        html_content = re.sub(r'=3D', '=', html_content)  # Quoted-printable equals
        html_content = re.sub(r'=20', ' ', html_content)  # Quoted-printable space
        
        # Parse HTML and extract text
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        text = parser.get_text()
        
        return text
    except Exception:
        # Fallback: simple regex-based removal
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = html.unescape(text)
        return text


def extract_email_body(msg: email.message.EmailMessage) -> str:
    """Extract the body text from an email message."""
    body_parts = []
    plain_text_found = False
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
                
            # Prefer plain text
            if content_type == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            text = payload.decode(charset, errors='replace')
                        except (UnicodeDecodeError, LookupError):
                            text = payload.decode('utf-8', errors='replace')
                        if len(text.strip()) > 20:  # Only count substantial plain text
                            body_parts.append(text)
                            plain_text_found = True
                except Exception:
                    continue
                    
            # Use HTML if no plain text
            elif content_type == "text/html" and not plain_text_found:
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            html_content = payload.decode(charset, errors='replace')
                        except (UnicodeDecodeError, LookupError):
                            html_content = payload.decode('utf-8', errors='replace')
                        text = html_to_text(html_content)
                        if len(text.strip()) > 20:
                            body_parts.append(text)
                except Exception:
                    continue
    else:
        # Non-multipart message
        content_type = msg.get_content_type()
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                try:
                    text = payload.decode(charset, errors='replace')
                except (UnicodeDecodeError, LookupError):
                    text = payload.decode('utf-8', errors='replace')
                    
                if content_type == "text/html":
                    text = html_to_text(text)
                body_parts.append(text)
        except Exception:
            pass
    
    return '\n'.join(body_parts).strip()


def decode_header(header_value) -> str:
    """Decode email header value."""
    if header_value is None:
        return ""
    decoded_parts = []
    try:
        for part, charset in email.header.decode_header(header_value):
            if isinstance(part, bytes):
                try:
                    decoded_parts.append(part.decode(charset or 'utf-8', errors='replace'))
                except (UnicodeDecodeError, LookupError):
                    decoded_parts.append(part.decode('utf-8', errors='replace'))
            else:
                decoded_parts.append(str(part))
    except Exception:
        return str(header_value)
    return ' '.join(decoded_parts)


def clean_text(text: str, max_length: int = 2000) -> str:
    """Clean and normalize email text for training."""
    if not text:
        return ""
    
    # Remove URLs but keep a marker
    text = re.sub(r'https?://[^\s]+', '[URL]', text)
    
    # Remove email addresses but keep a marker
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
    
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove very long words (likely encoded/obfuscated content)
    text = re.sub(r'\b\w{50,}\b', '[ENCODED]', text)
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()


def parse_eml_file(file_path: Path) -> Optional[Tuple[str, str]]:
    """Parse a single .eml file and return (subject, body) if valid."""
    try:
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        
        subject = decode_header(msg.get('Subject', ''))
        body = extract_email_body(msg)
        
        # Clean the content
        subject = clean_text(subject, max_length=200)
        body = clean_text(body, max_length=2000)
        
        # Validate
        if len(body) < 30:  # Too short
            return None
            
        return subject, body
    except Exception as e:
        return None


def calculate_content_hash(text: str) -> str:
    """Calculate hash of content for deduplication."""
    # Normalize for hashing
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def process_phishing_emails(
    input_dir: str,
    max_samples: int = 2000,
    min_length: int = 50
) -> List[dict]:
    """Process phishing .eml files and return cleaned data."""
    
    input_path = Path(input_dir)
    eml_files = list(input_path.glob('**/*.eml'))
    
    print(f"Found {len(eml_files)} .eml files in {input_dir}")
    
    # Shuffle for random sampling
    random.shuffle(eml_files)
    
    processed = []
    seen_hashes = set()
    skipped_short = 0
    skipped_duplicate = 0
    skipped_error = 0
    
    for i, eml_file in enumerate(eml_files):
        if len(processed) >= max_samples:
            break
            
        if (i + 1) % 500 == 0:
            print(f"Processing {i + 1}/{len(eml_files)}... ({len(processed)} collected)")
        
        result = parse_eml_file(eml_file)
        
        if result is None:
            skipped_error += 1
            continue
            
        subject, body = result
        
        # Combine for training
        if subject:
            email_text = f"Subject: {subject}\n\n{body}"
        else:
            email_text = body
        
        # Skip if too short
        if len(email_text) < min_length:
            skipped_short += 1
            continue
        
        # Deduplicate
        content_hash = calculate_content_hash(email_text)
        if content_hash in seen_hashes:
            skipped_duplicate += 1
            continue
        seen_hashes.add(content_hash)
        
        processed.append({
            'email_text': email_text,
            'label': 1,  # Phishing
            'source': 'phishing_pot'
        })
    
    print(f"\nPhishing emails processed:")
    print(f"  - Collected: {len(processed)}")
    print(f"  - Skipped (too short): {skipped_short}")
    print(f"  - Skipped (duplicate): {skipped_duplicate}")
    print(f"  - Skipped (parse error): {skipped_error}")
    
    return processed


# Legitimate email templates for synthetic generation
LEGITIMATE_TEMPLATES = [
    # Business/Work
    "Subject: {subject}\n\nHi {name},\n\n{body}\n\nBest regards,\n{sender}",
    "Subject: {subject}\n\nDear {name},\n\n{body}\n\nThanks,\n{sender}",
    "Subject: {subject}\n\nTeam,\n\n{body}\n\nBest,\n{sender}",
    
    # Newsletters
    "Subject: {subject}\n\n{body}\n\nTo unsubscribe, click here: [URL]\n\n{company}",
    
    # Notifications
    "Subject: {subject}\n\n{body}\n\nThis is an automated message from {company}.",
    
    # Personal
    "Subject: {subject}\n\nHey {name}!\n\n{body}\n\nCheers,\n{sender}",
]

LEGITIMATE_SUBJECTS = [
    # Meetings
    "Meeting tomorrow at {time}", "Team sync - {day}", "Quick catch up?",
    "Reschedule our call", "Conference room booking", "Agenda for Monday",
    
    # Work
    "Project update", "Q{quarter} report review", "Weekly status",
    "Document for your review", "Feedback needed", "Quick question",
    "Re: Your proposal", "Follow up on our discussion", "Action items from today",
    
    # Newsletters
    "Your weekly digest", "{month} newsletter", "What's new this week",
    "Industry news roundup", "Tips and updates", "New features available",
    
    # E-commerce (legitimate)
    "Your order has shipped", "Order confirmation #{number}",
    "Your receipt from {store}", "Delivery update", "Thank you for your purchase",
    
    # Social
    "Photos from {event}", "Happy birthday!", "Thanks for last night",
    "See you Saturday?", "Dinner plans", "Weekend plans?",
    
    # Services
    "Your appointment confirmation", "Reminder: {event} tomorrow",
    "Your monthly statement is ready", "Account summary for {month}",
    "Your subscription has been renewed",
]

LEGITIMATE_BODIES = [
    # Meetings
    "Just wanted to confirm our meeting for {day}. Please let me know if the time still works for you.",
    "Can we reschedule our call to {time}? Something came up on my end.",
    "Here's the agenda for our upcoming meeting:\n1. Project updates\n2. Budget review\n3. Next steps",
    
    # Work
    "I've attached the document you requested. Please review and let me know if you have any questions.",
    "The project is progressing well. We're on track to deliver by {date}.",
    "Thanks for your feedback on the proposal. I've incorporated your suggestions.",
    "Could you please review the attached and share your thoughts by end of day?",
    "Here are the action items from our meeting today:\n- Item 1\n- Item 2\n- Item 3",
    
    # E-commerce
    "Your order #{number} has been shipped. You can track it using the link below.",
    "Thank you for your purchase! Your order will arrive by {date}.",
    "Your payment of ${amount} has been processed successfully.",
    
    # Services
    "This is a reminder that your appointment is scheduled for {date} at {time}.",
    "Your monthly statement for {month} is now available in your account.",
    "Your subscription has been renewed. Next billing date: {date}.",
    
    # Personal
    "It was great catching up yesterday! Let's do it again soon.",
    "Are you free this weekend? Thought we could grab coffee.",
    "Here are the photos from the {event}. Hope you enjoy them!",
]

NAMES = ["John", "Sarah", "Michael", "Emily", "David", "Lisa", "James", "Anna", "Robert", "Maria"]
COMPANIES = ["Acme Corp", "Tech Solutions", "Digital Services", "Cloud Systems", "Data Pro"]
STORES = ["Amazon", "Target", "Best Buy", "Walmart", "Apple Store"]


def generate_legitimate_email() -> str:
    """Generate a synthetic legitimate email."""
    template = random.choice(LEGITIMATE_TEMPLATES)
    subject_template = random.choice(LEGITIMATE_SUBJECTS)
    body_template = random.choice(LEGITIMATE_BODIES)
    
    # Fill in placeholders
    replacements = {
        'subject': subject_template,
        'name': random.choice(NAMES),
        'sender': random.choice(NAMES),
        'company': random.choice(COMPANIES),
        'store': random.choice(STORES),
        'body': body_template,
        'time': f"{random.randint(9, 17)}:00",
        'day': random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']),
        'month': random.choice(['January', 'February', 'March', 'April', 'May', 'June']),
        'quarter': random.choice(['1', '2', '3', '4']),
        'date': f"{random.choice(['Monday', 'Tuesday', 'Wednesday'])}, {random.randint(1, 28)} {random.choice(['Jan', 'Feb', 'Mar'])}",
        'number': str(random.randint(100000, 999999)),
        'amount': f"{random.randint(10, 500)}.{random.randint(0, 99):02d}",
        'event': random.choice(['the conference', 'last weekend', 'the party', 'our trip']),
    }
    
    email_text = template
    for key, value in replacements.items():
        email_text = email_text.replace('{' + key + '}', value)
        # Also replace in nested templates
        for nested_key, nested_value in replacements.items():
            email_text = email_text.replace('{' + nested_key + '}', nested_value)
    
    return email_text


def generate_legitimate_emails(count: int) -> List[dict]:
    """Generate synthetic legitimate emails."""
    print(f"\nGenerating {count} synthetic legitimate emails...")
    
    emails = []
    seen_hashes = set()
    
    while len(emails) < count:
        email_text = generate_legitimate_email()
        content_hash = calculate_content_hash(email_text)
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            emails.append({
                'email_text': email_text,
                'label': 0,  # Legitimate
                'source': 'synthetic'
            })
    
    print(f"Generated {len(emails)} unique legitimate emails")
    return emails


def prepare_training_data(
    phishing_dir: str,
    output_csv: str,
    max_samples: int = 2000,
    balance_ratio: float = 1.0
):
    """Prepare balanced training data from phishing emails and synthetic legitimate emails."""
    
    # Process phishing emails
    phishing_emails = process_phishing_emails(
        phishing_dir,
        max_samples=max_samples // 2 if balance_ratio == 1.0 else max_samples
    )
    
    if not phishing_emails:
        print("Error: No phishing emails could be processed!")
        return
    
    # Generate legitimate emails to balance
    num_legitimate = int(len(phishing_emails) * balance_ratio)
    legitimate_emails = generate_legitimate_emails(num_legitimate)
    
    # Combine and shuffle
    all_emails = phishing_emails + legitimate_emails
    random.shuffle(all_emails)
    
    # Save to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['email_text', 'label'])
        writer.writeheader()
        for item in all_emails:
            writer.writerow({
                'email_text': item['email_text'],
                'label': item['label']
            })
    
    # Statistics
    phishing_count = sum(1 for e in all_emails if e['label'] == 1)
    legitimate_count = sum(1 for e in all_emails if e['label'] == 0)
    
    print(f"\n{'='*50}")
    print(f"TRAINING DATA PREPARED")
    print(f"{'='*50}")
    print(f"Total samples: {len(all_emails)}")
    print(f"  - Phishing (label=1): {phishing_count}")
    print(f"  - Legitimate (label=0): {legitimate_count}")
    print(f"Output saved to: {output_csv}")
    print(f"\nEstimated training time: ~{len(all_emails) // 200} minutes")
    print(f"{'='*50}")
    
    # Sample preview
    print("\n--- Sample Phishing Email ---")
    sample_phishing = next(e for e in all_emails if e['label'] == 1)
    print(sample_phishing['email_text'][:500] + "...")
    
    print("\n--- Sample Legitimate Email ---")
    sample_legitimate = next(e for e in all_emails if e['label'] == 0)
    print(sample_legitimate['email_text'][:500] + "...")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data from phishing .eml files'
    )
    parser.add_argument(
        '--phishing_dir',
        type=str,
        required=True,
        help='Directory containing phishing .eml files'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='data/training_data.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=2000,
        help='Maximum total samples (default: 2000 for ~10min training)'
    )
    parser.add_argument(
        '--balance_ratio',
        type=float,
        default=1.0,
        help='Ratio of legitimate to phishing emails (default: 1.0 = balanced)'
    )
    
    args = parser.parse_args()
    
    prepare_training_data(
        args.phishing_dir,
        args.output_csv,
        args.max_samples,
        args.balance_ratio
    )


if __name__ == '__main__':
    main()
