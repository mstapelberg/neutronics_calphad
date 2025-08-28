"""Classifier-style evaluation utilities with guard bands.

Defines pass/gray/fail rules based on margins to limits and computes
precision/recall for fast surrogate vs reference (full depletion) labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


@dataclass
class GuardBands:
    lower: float = 0.8  # pass if value <= lower * limit
    upper: float = 1.2  # fail if value >= upper * limit


def label_with_bands(values: Dict[str, float], limits: Dict[str, float], bands: GuardBands) -> str:
    """Return 'pass', 'fail', or 'gray' for a dict of metric->value against limits.

    values should contain metrics like 'dose_30d', 'dose_365d', 'He_appm', 'H_appm'.
    limits should map those to limit values in same units.
    """
    states = []
    for k, limit in limits.items():
        v = float(values.get(k, 0.0))
        if v <= bands.lower * limit:
            states.append('pass')
        elif v >= bands.upper * limit:
            states.append('fail')
        else:
            states.append('gray')
    if 'fail' in states:
        return 'fail'
    if 'gray' in states:
        return 'gray'
    return 'pass'


def precision_recall_with_gray(
    pred_labels: List[str],
    true_labels: List[str]
) -> Dict[str, float]:
    """Compute precision/recall ignoring gray in truth and/or prediction.

    - Precision: TP / (TP + FP), only counting predictions marked 'pass'
    - Recall: TP / (TP + FN), only counting truths marked 'pass'
    Gray predictions or truths are excluded from denominator appropriately.
    """
    tp = fp = fn = 0
    for p, t in zip(pred_labels, true_labels):
        if t == 'gray':
            continue
        if p == 'gray':
            if t == 'pass':
                fn += 1
            continue
        if p == 'pass' and t == 'pass':
            tp += 1
        elif p == 'pass' and t != 'pass':
            fp += 1
        elif p != 'pass' and t == 'pass':
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn}

