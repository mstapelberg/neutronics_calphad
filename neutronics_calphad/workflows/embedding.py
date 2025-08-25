"""Embedding and clustering utilities with graceful fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

try:
    import hdbscan  # type: ignore
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

try:
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.cluster import DBSCAN, KMeans  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def embed_2d(data: pd.DataFrame, n_neighbors: int = 15, min_dist: float = 0.1, random_state: Optional[int] = None) -> np.ndarray:
    """Reduce to 2D with UMAP if available, else PCA.

    Args:
        data: Numeric DataFrame to embed.
        n_neighbors: UMAP neighbors.
        min_dist: UMAP min_dist.
        random_state: Optional seed.

    Returns:
        (n_samples, 2) array embedding.
    """
    X = data.values.astype(float)
    if _HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=random_state)
        return reducer.fit_transform(X)
    if _HAS_SKLEARN:
        return PCA(n_components=2, random_state=random_state).fit_transform(X)
    # naive fallback: first two columns
    if X.shape[1] < 2:
        X = np.pad(X, ((0, 0), (0, 2 - X.shape[1])), mode="constant")
    return X[:, :2]


def cluster_labels(embedding_2d: np.ndarray, min_cluster_size: int = 20, eps: float = 0.05, random_state: Optional[int] = None) -> np.ndarray:
    """Cluster 2D points using HDBSCAN if available, else DBSCAN, else KMeans.

    Returns an array of integer labels; -1 denotes noise if HDBSCAN/DBSCAN are used.
    """
    if _HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        return clusterer.fit_predict(embedding_2d)
    if _HAS_SKLEARN:
        # Try DBSCAN
        db = DBSCAN(eps=eps, min_samples=max(5, min_cluster_size // 2))
        labels = db.fit_predict(embedding_2d)
        if len(set(labels)) <= 1:
            # fallback to KMeans with 2 clusters
            km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
            return km.fit_predict(embedding_2d)
        return labels
    # Fallback: single cluster
    return np.zeros(embedding_2d.shape[0], dtype=int)


def plot_embedding(
    embedding_2d: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Composition manifold",
) -> plt.Axes:
    """Scatter plot of 2D embedding with optional validity and clusters."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 5))
    x, y = embedding_2d[:, 0], embedding_2d[:, 1]
    if labels is not None:
        # plot by labels; invalid points faded
        unique = np.unique(labels)
        for lab in unique:
            sel = labels == lab
            alpha = 0.9
            color = None
            ax.scatter(x[sel], y[sel], s=18, alpha=alpha, label=f"Cluster {lab}")
    else:
        ax.scatter(x, y, s=14, alpha=0.85)
    if valid_mask is not None:
        # overlay invalid in red hollow
        inv = ~valid_mask
        ax.scatter(x[inv], y[inv], s=34, facecolors="none", edgecolors="r", linewidths=1.0, label="Filtered out")
    ax.set_xlabel("Comp-1")
    ax.set_ylabel("Comp-2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if labels is not None or valid_mask is not None:
        ax.legend()
    return ax


