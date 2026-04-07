import numpy as np

from .descriptor import compute_bd_batch
from .novelty import score_novelty_batch


class AuroraManager:
    """
    Minimal AURORA-style novelty manager.

    Responsibilities
    ----------------
    - Convert sensory rollout outputs into behavioural descriptors (BDs)
    - Score novelty against an external archive
    - Update the archive after each generation (typically from Pareto elites)
    """

    def __init__(self, archive, k: int = 10, normalize_bd: bool = True):
        self.archive = archive
        self.k = k
        self.normalize_bd = normalize_bd

    def compute_bd(self, z_batch: np.ndarray) -> np.ndarray:
        return compute_bd_batch(z_batch, normalize=self.normalize_bd)

    def score_from_z(self, z_batch: np.ndarray) -> np.ndarray:
        bd_batch = self.compute_bd(z_batch)
        archive_bds = self.archive.as_array()
        return score_novelty_batch(bd_batch, archive_bds, k=self.k)

    def score_from_bd(self, bd_batch: np.ndarray) -> np.ndarray:
        archive_bds = self.archive.as_array()
        return score_novelty_batch(bd_batch, archive_bds, k=self.k)

    def update_from_z(self, z_batch: np.ndarray):
        bd_batch = self.compute_bd(z_batch)
        self.archive.extend(bd_batch)

    def update_from_bd(self, bd_batch: np.ndarray):
        self.archive.extend(bd_batch)