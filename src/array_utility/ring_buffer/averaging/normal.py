from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseAveragingRingBuffer


@dataclass
class AveragingRingBuffer(BaseAveragingRingBuffer):
    def _sum_chunk(self, values: np.ndarray) -> np.ndarray:
        """
        Sum vectors over a chunk.

        Parameters
        ----------
        values : np.ndarray
            Values with shape (m, *feature_shape).

        Returns
        -------
        np.ndarray
            Element-wise sum with shape (*feature_shape).
        """
        return np.sum(values, axis=0)

    @property
    def mean(self) -> np.ndarray:
        """
        Get the mean of the buffer.

        Returns
        -------
        np.ndarray
            The arithmetic mean of the buffer with shape (*feature_shape).
        """
        return self._accumulator / self.n
