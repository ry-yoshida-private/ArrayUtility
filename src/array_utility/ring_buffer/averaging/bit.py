from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseAveragingRingBuffer


@dataclass
class BitAveragingRingBuffer(BaseAveragingRingBuffer):
    """
    Ring buffer for bit-encoded vectors with cached bit-count updates.
    """

    def _validate_dtype(self) -> None:
        """
        Validate that stored vectors are uint8 byte arrays.

        Raises
        ------
        TypeError
            If `value.dtype` is not `np.uint8`.
        """
        if self.value.dtype != np.uint8:
            raise TypeError(f"value dtype must be uint8, but got {self.value.dtype}")

    def update(self, value: np.ndarray) -> None:
        if value.dtype != np.uint8:
            raise TypeError(f"value dtype must be uint8, but got {value.dtype}")
        super().update(value)

    def extend(self, values: np.ndarray) -> None:
        if values.dtype != np.uint8:
            raise TypeError(f"values dtype must be uint8, but got {values.dtype}")
        super().extend(values)

    def _sum_chunk(self, values: np.ndarray) -> np.ndarray:
        """
        Sum unpacked bits over a chunk.

        Parameters
        ----------
        values : np.ndarray
            Bit-encoded values with shape (m, *feature_shape).

        Returns
        -------
        np.ndarray
            Bit-count array with shape (*feature_shape[:-1], feature_shape[-1] * 8).
        """
        return np.sum(np.unpackbits(values, axis=-1), axis=0)

    @property
    def mean(self) -> np.ndarray:
        """
        Get bitwise majority value over the whole buffer.

        Returns
        -------
        np.ndarray
            Bitwise-majority representative with shape (*feature_shape),
            where each bit is selected by majority vote across n entries.
        """
        majority_bits = (self._accumulator >= (self.n / 2)).astype(np.uint8)
        return np.packbits(majority_bits, axis=-1)
