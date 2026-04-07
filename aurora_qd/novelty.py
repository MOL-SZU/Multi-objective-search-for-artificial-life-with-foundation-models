import numpy as np

def score_novelty_batch(
    bd_batch: np.ndarray,
    archive_bds: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """
    Compute novelty as mean distance to k nearest neighbours in archive.

    Parameters
    ----------
    bd_batch : np.ndarray, shape (pop_size, D)
    archive_bds : np.ndarray, shape (N, D)
    k : int

    Returns
    -------
    novelty : np.ndarray, shape (pop_size,)
        Higher is more novel.
    """
    bd_batch = np.asarray(bd_batch, dtype=np.float64)
    archive_bds = np.asarray(archive_bds, dtype=np.float64)

    if bd_batch.ndim != 2:
        raise ValueError(f"Expected bd_batch shape (pop_size, D), got {bd_batch.shape}")

    if archive_bds.size == 0:
        return np.zeros((bd_batch.shape[0],), dtype=np.float64)

    if archive_bds.ndim != 2:
        raise ValueError(f"Expected archive_bds shape (N, D), got {archive_bds.shape}")

    novelty = np.zeros((bd_batch.shape[0],), dtype=np.float64)

    for i, bd in enumerate(bd_batch):
        dists = np.linalg.norm(archive_bds - bd[None, :], axis=1)
        kk = min(k, len(dists))
        novelty[i] = np.mean(np.partition(dists, kk - 1)[:kk])

    return novelty