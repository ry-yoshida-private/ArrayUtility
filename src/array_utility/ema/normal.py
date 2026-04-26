from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import numpy.typing as npt


@dataclass
class EMACalculator:
    """
    Exponential Moving Average calculator.

    Attributes
    ----------
    value : np.ndarray
        Current EMA value with shape (*feature_shape).
    alpha : float, default=0.5
        Smoothing factor in [0, 1]. Larger values react faster to new samples.
    """

    value: npt.NDArray[np.floating[Any]]
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], but got {self.alpha}")

    def update(self, value: npt.NDArray[np.floating[Any]]) -> None:
        """
        Update the EMA with a new value.

        Parameters
        ----------
        value : np.ndarray
            New sample with shape (*feature_shape).
        """
        self.value += self.alpha * (value - self.value)
