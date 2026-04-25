from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class EMACalculator:
    """
    Exponential Moving Average calculator.
    """
    alpha: float
    value: np.ndarray

    def update(self, value: np.ndarray):
        """
        Update the EMA with a new value.

        Parameters
        ----------
        value : np.ndarray
            The new value to be added to the EMA.
        """
        self.value = self.alpha * value + (1 - self.alpha) * self.value
    

