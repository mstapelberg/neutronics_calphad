# fusion_opt/manifold.py
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from .evaluate import evaluate

def sample_simplex(n, seed=1):
    """Generates random compositions on a 5-dimensional simplex.

    This function creates `n` random alloy compositions, where each composition
    is a vector of 5 atomic fractions that sum to 1. This is used to sample
    the design space for the V-Cr-Ti-W-Zr alloy system.

    Args:
        n (int): The number of random compositions to generate.
        seed (int, optional): A seed for the random number generator to ensure
            reproducibility. Defaults to 1.

    Returns:
        numpy.ndarray: A 2D array of shape (n, 5) where each row is a valid
            atomic composition.
    """
    rng=np.random.default_rng(seed)
    x = rng.random((n,5))
    return x/ x.sum(axis=1)[:,None]

def build_manifold(n=15000, workers=8):
    """Builds a dataset of alloy performance by sampling the design space.

    This function orchestrates the evaluation of a large number of random alloy
    compositions. It first generates the compositions using `sample_simplex`,
    then uses a process pool to evaluate the performance of each composition
    in parallel using the `evaluate` function. The results are compiled into a
    Pandas DataFrame and saved to a Parquet file.

    Args:
        n (int, optional): The total number of alloy compositions to sample and
            evaluate. Defaults to 15000.
        workers (int, optional): The number of parallel worker processes to use
            for evaluation. Defaults to 8.

    Returns:
        pandas.DataFrame: A DataFrame containing the evaluation results for
            each sampled composition.
    """
    comp = sample_simplex(n)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(evaluate, comp))
    df = pd.DataFrame(rows)
    df.to_parquet("manifold.parquet", compression="zstd")
    return df
