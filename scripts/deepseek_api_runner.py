#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deepseek API Runner for SimpleQuestions Evaluation
Uses Deepseek's OpenAI-compatible API for zero-shot question answering
"""

import os
import json
import time
import requests
from tqdm import tqdm

# Get API key from environment variable (NO hardcoded key!)
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


def call_deepseek_api(question, max_retries=3):
    """
    Call Deepseek API for question answering
    
    Args:
        question: The question string
        max_retries: Maximum number of retry attempts
        
    Returns:
        Predicted answer string or None if failed
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set!")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions directly and concisely. Provide only the answer without explanation."
            },
            {
                "role": "user",
                "content": f"Question: {question}\nAnswer:"
            }
        ],
        "temperature": 0.0,
        "max_tokens": 64
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            return answer
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"\nAPI call failed (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\nFailed after {max_retries} attempts: {e}")
                return None
    
    return None


def run(input_file, output_file, limit=None):
    """
    Run Deepseek API evaluation on dataset
    
    Args:
        input_file: Path to input JSON file (test.json)
        output_file: Path to save predictions
        limit: Maximum number of samples to process (None for all)
    """
    # Load dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
        print(f"Processing first {limit} samples...")
    else:
        print(f"Processing all {len(data)} samples...")
    
    # Run predictions
    results = []
    for i, item in enumerate(tqdm(data, desc="Calling Deepseek API")):
        question = item['question']
        answers = item['answers']
        
        # Call API
        pred = call_deepseek_api(question)
        
        results.append({
            'id': i,
            'question': question,
            'pred': pred if pred else "",
            'answers': answers
        })
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Processed {len(results)} samples.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Deepseek API evaluation')
    parser.add_argument('--input', default='test.json', help='Input dataset file')
    parser.add_argument('--output', default='results/deepseek_api_test.json', help='Output predictions file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples (default: all)')
    
    args = parser.parse_args()
    
    # Check API key before starting
    if not DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable is not set!")
        print("\nPlease set it using:")
        print("  PowerShell: $env:DEEPSEEK_API_KEY='your-api-key'")
        print("  CMD: set DEEPSEEK_API_KEY=your-api-key")
        exit(1)
    
    run(args.input, args.output, args.limit)
