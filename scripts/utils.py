import json
import re
import string
from collections import Counter
import random

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_answer(s):
    if s is None:
        return ""
    s = s.lower().strip()
    # remove punctuation
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def exact_match(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)

def f1_score(pred, gold):
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0
    pred_counts = Counter(p)
    gold_counts = Counter(g)
    common = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts)
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def sample_k_examples(data, k=8, seed=42):
    random.seed(seed)
    if k >= len(data):
        return [(d['question'], d.get('answers',[""])[0] if d.get('answers') else "") for d in data]
    sample = random.sample(data, k)
    return [(d['question'], d.get('answers',[""])[0] if d.get('answers') else "") for d in sample]
