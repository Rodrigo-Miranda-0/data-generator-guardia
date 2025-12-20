#!/usr/bin/env python3
"""
Parse .eml files from a phishing dataset and prepare for model training.

Usage:
    python data/parse_eml_dataset.py --input_dir /path/to/eml/files --output_csv phishing_data.csv --label 1
"""

import argparse
import email
import os
import re
import csv
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Optional, Tuple
import html
from html.parser import HTMLParser


class HTMLTextExtractor(HTMLParser):
    """Extract text from HTML content."""
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_data = False
        
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'head', 'meta', 'link'):
            self.skip_data = True
            
    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'head', 'meta', 'link'):
            self.skip_data = False
        if tag in ('p', 'div', 'br', 'li', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.text_parts.append('\n')
            
    def handle_data(self, data):
        if not self.skip_data:
            self.text_parts.append(data)
            
    def get_text(self) -> str:
        return ''.join(self.text_parts)


def html_to_text(html_content: str) -> str:
    """Convert HTML to plain text."""
    try:
        # Decode HTML entities
        html_content = html.unescape(html_content)
        
        # Parse HTML and extract text
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        text = parser.get_text()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    except Exception:
        # Fallback: simple regex-based removal
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def extract_email_body(msg: email.message.EmailMessage) -> str:
    """Extract the body text from an email message."""
    body_parts = []
    
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
                        body_parts.append(text)
                except Exception:
                    continue
                    
            # Fallback to HTML
            elif content_type == "text/html" and not body_parts:
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            html_content = payload.decode(charset, errors='replace')
                        except (UnicodeDecodeError, LookupError):
                            html_content = payload.decode('utf-8', errors='replace')
                        text = html_to_text(html_content)
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


def extract_email_metadata(msg: email.message.EmailMessage) -> dict:
    """Extract useful metadata from email headers."""
    def decode_header(header_value):
        if header_value is None:
            return ""
        decoded_parts = []
        for part, charset in email.header.decode_header(header_value):
            if isinstance(part, bytes):
                try:
                    decoded_parts.append(part.decode(charset or 'utf-8', errors='replace'))
                except (UnicodeDecodeError, LookupError):
                    decoded_parts.append(part.decode('utf-8', errors='replace'))
            else:
                decoded_parts.append(part)
        return ' '.join(decoded_parts)
    
    return {
        'subject': decode_header(msg.get('Subject', '')),
        'from': decode_header(msg.get('From', '')),
        'to': decode_header(msg.get('To', '')),
        'date': msg.get('Date', ''),
        'return_path': msg.get('Return-Path', ''),
    }


def parse_eml_file(file_path: Path) -> Optional[Tuple[str, dict]]:
    """Parse a single .eml file and return body text and metadata."""
    try:
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        
        body = extract_email_body(msg)
        metadata = extract_email_metadata(msg)
        
        if not body or len(body) < 10:
            return None
            
        return body, metadata
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def clean_text(text: str) -> str:
    """Clean and normalize email text for training."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters except newlines
    text = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Limit length
    if len(text) > 5000:
        text = text[:5000]
    
    return text.strip()


def process_eml_directory(
    input_dir: str,
    output_csv: str,
    label: int,
    include_subject: bool = True,
    min_length: int = 50
) -> int:
    """Process all .eml files in a directory and create a CSV for training."""
    
    input_path = Path(input_dir)
    eml_files = list(input_path.glob('**/*.eml'))
    
    print(f"Found {len(eml_files)} .eml files in {input_dir}")
    
    processed = 0
    skipped = 0
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['email_text', 'label'])
        
        for i, eml_file in enumerate(eml_files):
            if (i + 1) % 100 == 0:
                print(f"Processing {i + 1}/{len(eml_files)}...")
            
            result = parse_eml_file(eml_file)
            
            if result is None:
                skipped += 1
                continue
                
            body, metadata = result
            
            # Combine subject with body if requested
            if include_subject and metadata['subject']:
                email_text = f"Subject: {metadata['subject']}\n\n{body}"
            else:
                email_text = body
            
            # Clean the text
            email_text = clean_text(email_text)
            
            # Skip if too short
            if len(email_text) < min_length:
                skipped += 1
                continue
            
            writer.writerow([email_text, label])
            processed += 1
    
    print(f"\nProcessed: {processed} emails")
    print(f"Skipped: {skipped} emails")
    print(f"Output saved to: {output_csv}")
    
    return processed


def main():
    parser = argparse.ArgumentParser(
        description='Parse .eml files and prepare for model training'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing .eml files'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--label',
        type=int,
        required=True,
        choices=[0, 1],
        help='Label for all emails: 1=phishing, 0=legitimate'
    )
    parser.add_argument(
        '--include_subject',
        action='store_true',
        default=True,
        help='Include email subject in training text'
    )
    parser.add_argument(
        '--min_length',
        type=int,
        default=50,
        help='Minimum text length to include (default: 50)'
    )
    
    args = parser.parse_args()
    
    process_eml_directory(
        args.input_dir,
        args.output_csv,
        args.label,
        args.include_subject,
        args.min_length
    )


if __name__ == '__main__':
    main()
