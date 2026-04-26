from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import numpy.typing as npt


@dataclass
class BitEMACalculator:
    """
    Bit-wise Exponential Moving Average calculator for packed uint8 inputs.

    Note
    ----
    Input is not a raw binary (0/1) array. This class expects packed uint8 bytes
    (for example, pHash output like `[30, 6, 193, ...]`), unpacks them into bits
    internally, and then applies EMA per bit.

    Internally keeps EMA as float values in [0, 1] for each unpacked bit.

    Attributes
    ----------
    _ema_value : np.ndarray
        Internal per-bit EMA buffer. Its shape is the same as
        `np.unpackbits(input, axis=-1)` (the last axis is expanded into bits).
    alpha : float, default=0.5
        Smoothing factor in [0, 1]. Larger values react faster to new samples.
    """

    _ema_value: npt.NDArray[np.floating[Any]]
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], but got {self.alpha}")

    def update(self, value: npt.NDArray[np.uint8]) -> None:
        """
        Update the bit-wise EMA with a packed uint8 vector.

        Parameters
        ----------
        value : np.ndarray
            New packed uint8 vector (not a raw 0/1 bit array) with
            shape (*feature_shape).
        """
        if value.dtype != np.uint8:
            raise TypeError("BitEMACalculator only supports uint8 arrays.")
        bits = np.unpackbits(value, axis=-1)
        if bits.shape != self._ema_value.shape:
            raise ValueError(f"unpacked value shape {bits.shape} must match {self._ema_value.shape}")
        self._ema_value += self.alpha * (bits - self._ema_value)

    @property
    def value(self) -> npt.NDArray[np.uint8]:
        """
        Get thresholded packed bits from the internal EMA state.

        Returns
        -------
        np.ndarray
            Packed uint8 bits where each bit is 1 when EMA >= 0.5.
        """
        thresholded = (self._ema_value >= 0.5).astype(np.uint8)
        return np.packbits(thresholded, axis=-1)

    @classmethod
    def build(cls, value: npt.NDArray[np.uint8], alpha: float = 0.5) -> BitEMACalculator:
        """
        Create calculator from an initial packed uint8 bit vector.

        Parameters
        ----------
        value : np.ndarray
            Initial packed uint8 vector (not a raw 0/1 bit array) with
            shape (*feature_shape).
        alpha : float, default=0.5
            Smoothing factor in [0, 1].

        Returns
        -------
        BitEMACalculator
            Bit EMA calculator initialized from the given value.
        """
        if value.dtype != np.uint8:
            raise TypeError("BitEMACalculator only supports uint8 arrays.")
        bit_float = np.unpackbits(value, axis=-1).astype(np.float32)
        return cls(alpha=alpha, _ema_value=bit_float)
