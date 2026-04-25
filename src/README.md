# src

## Overview

This directory contains the `array_utility` source package.

See [array_utility/README.md](array_utility/README.md) for package details.

## Example

```python
import numpy as np
from array_utility import EMACalculator

ema = EMACalculator(alpha=0.1, value=np.array([10.0], dtype=np.float32))
ema.update(np.array([20.0], dtype=np.float32))
print(ema.value)  # [11.0]
```
