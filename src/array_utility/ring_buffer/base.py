from __future__ import annotations
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar


BaseRingBufferT = TypeVar("BaseRingBufferT", bound="BaseRingBuffer")


@dataclass
class BaseRingBuffer(ABC):
    """
    Fixed-length ring buffer for storing feature vectors.

    Attributes
    ----------
    n : int
        The maximum number of frames stored in the buffer.
    value : np.ndarray
        Buffer array of shape (n, *feature_shape).
    _index : int
        Internal pointer to the next write position.
    """

    n: int
    value: np.ndarray
    _index: int = field(init=False, default=0)

    def __post_init__(self):
        if self.value.ndim == 1:
            raise ValueError(f"value must be 2D or higher, but got {self.value.ndim}")
        if self.value.shape[0] != self.n:
            raise ValueError(f"value first axis length ({self.value.shape[0]}) must match n ({self.n})")

    @abstractmethod
    def update(self, value: np.ndarray) -> None:
        """
        Update the buffer with a new vector.

        Parameters
        ----------
        value : np.ndarray
            The new vector with shape (*feature_shape) to be stored.
        """

    @abstractmethod
    def extend(self, values: np.ndarray) -> None:
        """
        Extend the buffer with new vectors.

        Parameters
        ----------
        values : np.ndarray
            The new vectors with shape (n, *feature_shape) to be stored.
        """

    def get_last_k(self, k: int) -> np.ndarray:
        """
        Get the last k vectors from the buffer.

        Parameters
        ----------
        k : int
            The number of vectors to get.
        """
        if k > self.n:
            raise ValueError(f"k ({k}) cannot be greater than buffer size ({self.n})")
        start = self._index - k
        if start >= 0:
            return self.value[start : self._index]
        return np.concatenate([self.value[start:], self.value[: self._index]], axis=0)

    @property
    def latest(self) -> np.ndarray:
        """
        Get the most recently appended vector.

        Returns
        -------
        np.ndarray
            The most recently stored vector with shape (*feature_shape).
        """
        return self.value[self._index - 1]

    @property
    def ordered_value(self) -> np.ndarray:
        """
        Get the value of the buffer in the order of the oldest to latest.

        Returns
        -------
        np.ndarray
            The ordered value of the buffer with shape (n, *feature_shape).
        """
        return np.roll(self.value, -self._index, axis=0)

    @property
    @abstractmethod
    def mean(self) -> np.ndarray:
        """
        Get the mean of the buffer.

        Returns
        -------
        np.ndarray
            The mean of the buffer with shape (*feature_shape).
        """

    @classmethod
    def build(
        cls: type[BaseRingBufferT],
        n: int,
        init_value: np.ndarray,
    ) -> BaseRingBufferT:
        """
        Create a buffer instance pre-filled with an initial tensor of shape.

        Parameters
        ----------
        n : int
            The maximum number of frames stored in the buffer.
        init_value : np.ndarray
            Initial value with shape (*feature_shape) used to fill all frames.

        Returns
        -------
        BaseRingBufferT
            The buffer instance.
        """

        init_buffer = np.empty((n, *init_value.shape), dtype=init_value.dtype)
        init_buffer[:] = init_value
        return cls(n=n, value=init_buffer)

    def __len__(self) -> int:
        """
        Get the number of frames stored in the buffer.
        """

        return self.n

    def _update_index(self, step: int = 1) -> None:
        """
        Update write index with wrap-around.
        """

        self._index = (self._index + step) % self.n

