from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseStandardRingBuffer


@dataclass
class BitRingBuffer(BaseStandardRingBuffer):
    """
    Ring buffer for bit-encoded vectors (for example, pHash bytes).
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
        bit_sum = np.sum(np.unpackbits(self.value, axis=-1), axis=0)
        majority_bits = (bit_sum >= (self.n / 2)).astype(np.uint8)
        return np.packbits(majority_bits, axis=-1)
