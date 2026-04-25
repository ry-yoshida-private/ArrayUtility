# ArrayUtility

## Overview

`ArrayUtility` is a Python repository that provides utility classes for array-based computation.
It supports both Ring Buffer operations and Exponential Moving Average (EMA) updates.

## Available utilities

| Utility | Purpose | Notes |
| ------- | ------- | ----- |
| `EMACalculator` | Exponential moving average update | Good for lightweight smoothing in streams |
| `BaseRingBuffer` | Base API for fixed-length history storage | Defines common buffer operations |
| `AveragingRingBuffer` | Ring buffer with running mean | Efficient for frequent mean queries |

For module-level details, see [src/README.md](src/README.md).

## Requirements

- `numpy`

## Setup

```bash
pip install -r requirements.txt
```

## Example

```python
import numpy as np
from array_utility import EMACalculator, AveragingRingBuffer

# EMA
ema = EMACalculator(alpha=0.2, value=np.array([0.0, 0.0], dtype=np.float32))
ema.update(np.array([1.0, 3.0], dtype=np.float32))
print(ema.value)

# Ring buffer + running mean
buf = AveragingRingBuffer.build(n=3, init_value=np.array([0.0, 0.0], dtype=np.float32))
buf.update(np.array([1.0, 2.0], dtype=np.float32))
buf.update(np.array([2.0, 4.0], dtype=np.float32))
print(buf.latest)  # [2.0, 4.0]
print(buf.mean)    # mean over fixed window size (n=3)
```
