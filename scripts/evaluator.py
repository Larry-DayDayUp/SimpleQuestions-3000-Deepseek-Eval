import argparse
import json
from utils import normalize_answer, f1_score
import numpy as np


def eval_chatgpt(preds_path):
    with open(preds_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    ems = []
    f1s = []
    for item in preds:
        golds = item.get('answers', [])
        pred = item.get('pred', '')
        em = max(1 if normalize_answer(pred) == normalize_answer(g) else 0 for g in golds)
        f1 = max(f1_score(pred, g) for g in golds) if golds else 0.0
        ems.append(em)
        f1s.append(f1)
    return {'EM': float(np.mean(ems)), 'F1': float(np.mean(f1s)), 'n': len(ems)}


def eval_deepseek(preds_path, k=5):
    with open(preds_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    hits = []
    rr = []
    for item in preds:
        golds = item.get('answers', [])
        results = item.get('results', [])
        found_rank = None
        for idx, res in enumerate(results[:k]):
            txt = res.get('text', '')
            if any(normalize_answer(txt) == normalize_answer(g) for g in golds):
                found_rank = idx + 1
                break
        hits.append(1 if found_rank else 0)
        rr.append(1.0 / found_rank if found_rank else 0.0)
    import numpy as np
    return {'Hits@%d' % k: float(np.mean(hits)), 'MRR': float(np.mean(rr)), 'n': len(hits)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['chatgpt', 'deepseek'], required=True)
    p.add_argument('--preds', required=True)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--out', required=False)
    args = p.parse_args()
    if args.mode == 'chatgpt':
        report = eval_chatgpt(args.preds)
    else:
        report = eval_deepseek(args.preds, k=args.k)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    else:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
