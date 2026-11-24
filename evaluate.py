"""Recalculate class errors, confusion matrices and metrics from saved predictions.

Run from the repository root: python evaluate.py
No model training or API requests are performed. Outputs: results/evaluation/.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

LABELS = ['negative', 'neutral', 'positive']


def evaluate(root, output):
    summaries, classes, matrices = [], [], {}
    for language in ('english', 'turkish'):
        source = pd.read_csv(root / 'data' / f'{language}.csv', keep_default_na=False)
        if source.id.duplicated().any() or set(source.label) != set(LABELS):
            raise ValueError(f'Invalid input IDs or labels: {language}')
        methods = ['vader', 'bert', 'llm']
        if language == 'turkish':
            methods.append('vader_translated')
        for method in methods:
            name = f'{method}_{language}' if method != 'vader_translated' else 'vader_turkish_translated'
            frame = pd.read_csv(root / 'results' / f'{name}.csv', keep_default_na=False)
            if frame.id.duplicated().any() or set(frame.id) != set(source.id):
                raise ValueError(f'IDs differ: {name}')
            expected = source.set_index('id').sort_index()
            frame = frame.set_index('id').sort_index()
            if not frame[['text', 'label']].equals(expected[['text', 'label']]):
                raise ValueError(f'Text or reference labels differ: {name}')
            if not set(frame.prediction) <= set(LABELS):
                raise ValueError(f'Unknown prediction label: {name}')
            correct = frame.label == frame.prediction
            stored = frame.correct.astype(str).str.lower().map({'true': True, 'false': False})
            if stored.isna().any() or not (stored == correct).all():
                raise ValueError(f'Incorrect correct column: {name}')
            cm = confusion_matrix(frame.label, frame.prediction, labels=LABELS)
            p, r, f, support = precision_recall_fscore_support(
                frame.label, frame.prediction, labels=LABELS, zero_division=0)
            matrices[name] = {'labels': LABELS, 'rows': 'true', 'columns': 'prediction', 'matrix': cm.tolist()}
            row = {'method': method, 'language': language, 'n': len(frame),
                   'accuracy': accuracy_score(frame.label, frame.prediction),
                   'macro_precision': float(p.mean()), 'macro_recall': float(r.mean()),
                   'macro_f1': float(f.mean()), 'total_errors': int((~correct).sum())}
            for i, label in enumerate(LABELS):
                errors = int(support[i] - cm[i, i])
                row[f'{label}_errors'] = errors
                classes.append({'method': method, 'language': language, 'class': label,
                                'support': int(support[i]), 'errors': errors,
                                'precision': p[i], 'recall': r[i], 'f1': f[i]})
            summaries.append(row)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(classes).to_csv(output / 'class_metrics.csv', index=False, encoding='utf-8-sig')
    (output / 'confusion_matrices.json').write_text(json.dumps(matrices, indent=2), encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    evaluate(args.root, args.output or args.root / 'results' / 'evaluation')
