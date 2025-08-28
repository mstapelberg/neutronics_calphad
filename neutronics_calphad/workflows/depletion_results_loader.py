"""Load existing batch depletion results into arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np  # type: ignore


def load_existing_depletion_results(
    depl_dir: Path,
    dose_times_h: List[float],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load X_raw and Y_nat from ``depletion_result.json`` files.

    Parameters
    ----------
    depl_dir : Path
        Root directory that contains per-composition run subfolders produced by
        the batch subprocess runner (each with a ``depletion_result.json``).
    dose_times_h : List[float]
        Dose times in hours. These define the order of dose outputs in Y.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray]]
        ``(X_raw, Y_nat)`` where ``X_raw`` has shape (n, 4) with columns
        [Cr, Ti, W, Zr] and ``Y_nat`` has shape (n, M) where
        ``M = len(dose_times_h) + 2`` for [dose_t..., He_2y, H_2y]. Returns
        ``None`` if no result files are found.
    """
    if not depl_dir.exists():
        return None

    json_files = list(depl_dir.rglob("depletion_result.json"))
    if not json_files:
        return None

    X_rows: List[List[float]] = []
    Y_rows: List[List[float]] = []
    dose_days = [int(h / 24) for h in dose_times_h]

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data: Dict[str, Any] = json.load(f)

            comp: Dict[str, float] = data.get("composition", {})
            cr = float(comp.get("Cr", 0.0))
            ti = float(comp.get("Ti", 0.0))
            w = float(comp.get("W", 0.0))
            zr = float(comp.get("Zr", 0.0))
            X_rows.append([cr, ti, w, zr])

            dose_dict: Dict[Any, float] = data.get("dose_at_cooling_times", {}) or {}
            # Accept both int and str keys
            row: List[float] = []
            for d in dose_days:
                v = dose_dict.get(d)
                if v is None:
                    v = dose_dict.get(str(d), 0.0)
                row.append(float(v))
            gas = data.get("gas_production", {}) or {}
            row.append(float(gas.get("He_appm", 0.0)))
            row.append(float(gas.get("H_appm", 0.0)))
            Y_rows.append(row)

        except Exception:
            # Skip unreadable entries and continue
            continue

    if not X_rows:
        return None

    X = np.asarray(X_rows, dtype=float)
    Y = np.asarray(Y_rows, dtype=float)
    return X, Y


