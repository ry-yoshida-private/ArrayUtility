# ema

## Overview

This directory contains EMA calculators for numeric arrays and bit-encoded arrays.
It provides a standard numeric EMA and a bit-oriented EMA for packed `uint8` values.

## Components

| File/Directory | Responsibility |
| --- | --- |
| [`normal.py`](normal.py) | `EMACalculator` for numeric vectors with standard EMA updates. |
| [`bit.py`](bit.py) | `BitEMACalculator` for `uint8` packed bits with per-bit EMA and thresholded output. |

## Usage

### Numeric EMA

```python
import numpy as np
from array_utility import EMACalculator

ema = EMACalculator(alpha=0.2, value=np.array([0.0, 10.0], dtype=np.float32))
ema.update(np.array([10.0, 0.0], dtype=np.float32))
print(ema.value)
```

### Bit-Encoded EMA

```python
import numpy as np
from array_utility import BitEMACalculator

bit_ema = BitEMACalculator.build(alpha=0.4, value=np.array([0b00000000], dtype=np.uint8))
bit_ema.update(np.array([0b11110000], dtype=np.uint8))
print(bit_ema.value)
```
