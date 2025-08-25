"""Tests for composition sampling and filters.

These are lightweight checks using pytest only.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest



from neutronics_calphad.workflows.composition_sampling import SamplingConstraints, sample_compositions
from neutronics_calphad.workflows.filters import ActivationLimits, make_activation_filter, make_ductility_filter, apply_filters


def test_sample_compositions_basic() -> None:
    """Sampler returns normalized compositions obeying per-element caps and total cap."""
    caps: Dict[str, float] = {"Cr": 0.2, "Ti": 0.2, "W": 0.2, "Zr": 0.2}
    cons = SamplingConstraints(per_element_max=caps, total_alloy_max=0.2)
    df = sample_compositions(cons, n_samples=128, random_state=0)
    assert set(df.columns) == {"V", "Cr", "Ti", "W", "Zr"}
    totals = df.sum(axis=1).to_numpy()
    assert np.allclose(totals, 1.0, atol=1e-9)
    for el, cap in caps.items():
        assert (df[el] <= cap + 1e-12).all()
    assert ((1.0 - df["V"]) <= 0.2 + 1e-12).all()


def test_filters_activation_and_ductility() -> None:
    """Activation filter + ductility filter accept mid-point comp and reject extremes."""
    # synthetic pure results: V is benign; others scale linearly to exceed limits near caps
    pure = {
        "V": {"gas_production": {"He_appm": 100.0, "H_appm": 100.0}, "dose_at_cooling_times": {30: 0.1}},
        "Cr": {"gas_production": {"He_appm": 500.0, "H_appm": 600.0}, "dose_at_cooling_times": {30: 100.0}},
        "Ti": {"gas_production": {"He_appm": 400.0, "H_appm": 500.0}, "dose_at_cooling_times": {30: 50.0}},
        "W":  {"gas_production": {"He_appm": 800.0, "H_appm": 700.0}, "dose_at_cooling_times": {30: 80.0}},
        "Zr": {"gas_production": {"He_appm": 300.0, "H_appm": 400.0}, "dose_at_cooling_times": {30: 40.0}},
    }
    limits = ActivationLimits(gas_appm={"He_appm": 586.1, "H_appm": 1200.0}, dose_at_days={30: 1e3})
    f_act = make_activation_filter(pure_results=pure, limits=limits)
    f_duc = make_ductility_filter(max_total_alloy=0.2)

    mid = {"V": 0.8, "Cr": 0.05, "Ti": 0.05, "W": 0.05, "Zr": 0.05}
    hi = {"V": 0.6, "Cr": 0.2, "Ti": 0.1, "W": 0.05, "Zr": 0.05}

    assert f_duc(mid) is True
    assert f_act(mid) is True
    # ductility fails for hi
    assert f_duc(hi) is False

    df = pd.DataFrame([mid, hi])
    mask = apply_filters(df, filters=(f_act, f_duc))
    assert mask.values.tolist() == [True, False]


