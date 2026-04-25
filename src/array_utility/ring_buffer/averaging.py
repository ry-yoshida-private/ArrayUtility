from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np

from .base import BaseRingBuffer


@dataclass
class AveragingRingBuffer(BaseRingBuffer):
    _sum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._sum = np.sum(self.value, axis=0)

    def update(self, value: np.ndarray) -> None:
        """
        Update the buffer with a new vector.

        Parameters
        ----------
        value : np.ndarray
            The new vector with shape (*feature_shape) to be stored.
        """
        old_value = self.value[self._index]
        self._sum += value
        self._sum -= old_value
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
            old_chunk = self.value[self._index : self._index + m]
            self._sum -= np.sum(old_chunk, axis=0)
            self._sum += np.sum(values, axis=0)
            self.value[self._index : self._index + m] = values
        else:
            first_values = values[:end_space]
            second_values = values[end_space:]
            old_first = self.value[self._index :]
            old_second = self.value[: m - end_space]
            self._sum -= np.sum(old_first, axis=0)
            self._sum -= np.sum(old_second, axis=0)
            self._sum += np.sum(first_values, axis=0)
            self._sum += np.sum(second_values, axis=0)
            self.value[self._index :] = first_values
            self.value[: m - end_space] = second_values
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

        return self._sum / self.n

