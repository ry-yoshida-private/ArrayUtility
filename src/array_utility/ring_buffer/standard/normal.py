from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseStandardRingBuffer


@dataclass
class RingBuffer(BaseStandardRingBuffer):
    @property
    def mean(self) -> np.ndarray:
        """
        Get the mean of the buffer.

        Returns
        -------
        np.ndarray
            The arithmetic mean of the buffer with shape (*feature_shape).
        """
        return np.mean(self.value, axis=0)
