# ArrayUtility

## Overview

`ArrayUtility` provides typed, NumPy-based utilities for streaming array operations.
The package focuses on two domains: exponential moving average (EMA) and fixed-size ring buffers, including bit-oriented variants for `uint8` data.

## Requirements

- Python 3.10+
- `numpy`

## Usage

```bash
pip install -r requirements.txt
```

For module-level details, see [src/README.md](src/README.md).

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
