from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .base import BaseRingBuffer


@dataclass
class RingBuffer(BaseRingBuffer):
    def update(self, value: np.ndarray) -> None:
        """
        Update the buffer with a new vector.

        Parameters
        ----------
        value : np.ndarray
            The new vector with shape (*feature_shape) to be stored.
        """
        self.value[self._index] = value
        self._update_index()

    def extend(self, values: np.ndarray) -> None:
        """
        Extend the buffer with new vectors.

        Parameters
        ----------
        values : np.ndarray
            The new vectors with shape (n, *feature_shape) to be stored.
        """
        m = len(values)
        if m > self.n:
            warnings.warn(
                f"values length ({m}) exceeds buffer size ({self.n}); truncating to the last {self.n} vectors.",
                stacklevel=2,
            )
            values = values[-self.n :]
            m = self.n
        end_space = self.n - self._index
        if m <= end_space:
            self.value[self._index : self._index + m] = values
        else:
            self.value[self._index :] = values[:end_space]
            self.value[: m - end_space] = values[end_space:]
        self._update_index(m)

    @property
    def mean(self) -> np.ndarray:
        """
        Get the mean of the buffer.

        Returns
        -------
        np.ndarray
            The mean of the buffer.
        """

        return np.mean(self.value, axis=0)
