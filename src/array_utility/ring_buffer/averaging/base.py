from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
import warnings

import numpy as np

from ..base import BaseRingBuffer


@dataclass
class BaseAveragingRingBuffer(BaseRingBuffer):
    _accumulator: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """
        Run validation and initialize cached accumulator from current buffer values.
        """
        super().__post_init__()
        self._validate_dtype()
        self._accumulator = self._sum_chunk(self.value)

    def _validate_dtype(self) -> None:
        """
        Optional dtype validation hook for subclasses.
        """

    @abstractmethod
    def _sum_chunk(self, values: np.ndarray) -> np.ndarray:
        """
        Convert one or more vectors into accumulated statistics.

        Parameters
        ----------
        values : np.ndarray
            Values with shape (m, *feature_shape).

        Returns
        -------
        np.ndarray
            Accumulated statistics with shape (*accumulator_shape).
        """

    def update(self, value: np.ndarray) -> None:
        """
        Update the buffer with a new vector.

        Parameters
        ----------
        value : np.ndarray
            The new vector with shape (*feature_shape) to be stored.
        """
        old_value = self.value[self._index]
        self._accumulator += self._sum_chunk(value[None, ...])
        self._accumulator -= self._sum_chunk(old_value[None, ...])
        self.value[self._index] = value
        self._update_index()

    def extend(self, values: np.ndarray) -> None:
        """
        Extend the buffer with new vectors.

        Parameters
        ----------
        values : np.ndarray
            The new vectors with shape (m, *feature_shape) to be stored.
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
            self._accumulator -= self._sum_chunk(old_chunk)
            self._accumulator += self._sum_chunk(values)
            self.value[self._index : self._index + m] = values
        else:
            first_values = values[:end_space]
            second_values = values[end_space:]
            old_first = self.value[self._index :]
            old_second = self.value[: m - end_space]
            self._accumulator -= self._sum_chunk(old_first)
            self._accumulator -= self._sum_chunk(old_second)
            self._accumulator += self._sum_chunk(first_values)
            self._accumulator += self._sum_chunk(second_values)
            self.value[self._index :] = first_values
            self.value[: m - end_space] = second_values
        self._update_index(m)

    @property
    @abstractmethod
    def mean(self) -> np.ndarray:
        """
        Get the representative value of the buffer.

        Returns
        -------
        np.ndarray
            Representative value with shape (*feature_shape).
        """
