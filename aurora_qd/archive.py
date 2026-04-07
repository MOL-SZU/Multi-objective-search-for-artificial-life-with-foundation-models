import numpy as np

class AuroraArchive:
    """
    Minimal archive storing behavioural descriptors for novelty scoring.
    """

    def __init__(self, max_size: int | None = None):
        self.max_size = max_size
        self._bds = []

    def __len__(self):
        return len(self._bds)

    def is_empty(self) -> bool:
        return len(self._bds) == 0

    def add(self, bd: np.ndarray):
        bd = np.asarray(bd, dtype=np.float64)
        if bd.ndim != 1:
            raise ValueError(f"Expected 1D descriptor, got shape {bd.shape}")
        self._bds.append(bd)

        if self.max_size is not None and len(self._bds) > self.max_size:
            # FIFO truncation
            self._bds.pop(0)

    def extend(self, bd_batch: np.ndarray):
        bd_batch = np.asarray(bd_batch, dtype=np.float64)
        if bd_batch.ndim != 2:
            raise ValueError(f"Expected 2D descriptor batch, got shape {bd_batch.shape}")
        for bd in bd_batch:
            self.add(bd)

    def as_array(self) -> np.ndarray:
        if self.is_empty():
            return np.empty((0, 0), dtype=np.float64)
        return np.stack(self._bds, axis=0)