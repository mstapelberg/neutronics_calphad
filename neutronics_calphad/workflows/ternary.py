"""Ternary slice plotting utilities with minimal dependencies.

Uses `python-ternary` if available; otherwise performs a simple barycentric
transform to plot on a triangular simplex in Matplotlib.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


try:
    import ternary as pyternary  # type: ignore
    _HAS_TERNARY = True
except Exception:
    _HAS_TERNARY = False


def _barycentric_to_cart(points: np.ndarray) -> np.ndarray:
    """Map ternary (a,b,c) with a+b+c=1 to 2D Cartesian for plotting."""
    # Equilateral triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    return points[:, 0:1] * v0 + points[:, 1:2] * v1 + points[:, 2:3] * v2


def plot_ternary_slice(
    grid: pd.DataFrame,
    varying: Tuple[str, str, str],
    value: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    cmap: str = "viridis",
) -> plt.Axes:
    """Plot a ternary slice with optional scalar coloring.

    Args:
        grid: DataFrame with columns ["V","Cr","Ti","W","Zr"]. One element is fixed.
        varying: The three elements forming the ternary axes in order (A, B, C).
        value: Optional scalar array aligned with grid rows for colormap.
        ax: Optional Matplotlib axes.
        title: Plot title.
        cmap: Colormap name.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))

    A, B, C = varying
    abc = grid[[A, B, C]].values.astype(float)
    # Normalize within the ternary plane
    sums = abc.sum(axis=1, keepdims=True)
    sums[sums == 0.0] = 1.0
    abc = abc / sums

    if _HAS_TERNARY:
        scale = 100
        fig, tax = pyternary.figure(scale=scale)
        pts = (abc * scale).round().astype(int)
        if value is not None:
            sc = tax.scatter(pts, marker='o', color=None, colormap=cmap, c=value, s=10)
            fig.colorbar(sc)
        else:
            tax.scatter(pts, marker='o', color='k', s=5)
        tax.boundary()
        tax.ticks(axis='lbr', multiple=20, linewidth=1, offset=0.02)
        tax.set_title(title)
        return tax.get_axes()

    # Fallback: barycentric transform in Matplotlib
    xy = _barycentric_to_cart(abc)
    if value is not None:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=value, s=8, cmap=cmap)
        plt.colorbar(sc, ax=ax)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=8, c='k')
    # draw triangle
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0], [0.0, 0.0]])
    ax.plot(tri[:, 0], tri[:, 1], 'k-', lw=1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return ax


