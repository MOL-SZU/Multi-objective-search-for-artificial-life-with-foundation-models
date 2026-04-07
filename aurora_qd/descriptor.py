import numpy as np

def compute_bd_batch(z_batch: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute behavioural descriptors from rollout sensory data.

    Parameters
    ----------
    z_batch : np.ndarray, shape (pop_size, T, D) or (pop_size, D)
        Sensory / latent sequence returned by rollout.
    normalize : bool
        Whether to L2-normalize each descriptor.

    Returns
    -------
    bd_batch : np.ndarray, shape (pop_size, D)
        Behavioural descriptors.
    """
    z_batch = np.asarray(z_batch, dtype=np.float64)

    if z_batch.ndim == 3:
        # Mean over time: (pop_size, T, D) -> (pop_size, D)
        bd_batch = z_batch.mean(axis=1)
    elif z_batch.ndim == 2:
        bd_batch = z_batch
    else:
        raise ValueError(f"Expected z_batch ndim in {{2, 3}}, got {z_batch.ndim}")

    if normalize:
        norms = np.linalg.norm(bd_batch, axis=1, keepdims=True) + 1e-8
        bd_batch = bd_batch / norms

    return bd_batch