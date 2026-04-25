# array_utility

## Overview

`array_utility` provides utilities for streaming numeric arrays.

## API at a glance

| Symbol | Kind | Purpose |
| ------ | ---- | ------- |
| `EMACalculator` | Class | Exponential moving average update |
| `BaseRingBuffer` | Class | Abstract ring buffer interface |
| `AveragingRingBuffer` | Class | Ring buffer with efficient running mean |
| `BitRingBuffer` | Class | Ring buffer for bit-encoded vectors |
| `BitAveragingRingBuffer` | Class | Bit ring buffer with cached bit counts |

## Components

| Component | Description |
| --------- | ----------- |
| [ema.py](ema.py) | Exponential moving average helper |
| [ring_buffer](ring_buffer) | Ring buffer interfaces and implementations |

## Example

```python
import numpy as np
from array_utility import EMACalculator, AveragingRingBuffer

# EMA
ema = EMACalculator(alpha=0.2, value=np.array([0.0, 0.0], dtype=np.float32))
ema.update(np.array([1.0, 3.0], dtype=np.float32))
print(ema.value)

# Ring buffer
buffer = AveragingRingBuffer.build(n=4, init_value=np.array([0.0, 0.0], dtype=np.float32))
buffer.extend(np.array([[1.0, 1.0], [2.0, 3.0]], dtype=np.float32))

print(buffer.latest)        # [2.0, 3.0]
print(buffer.get_last_k(2)) # last 2 inserted vectors
print(buffer.ordered_value) # oldest -> latest order
print(buffer.mean)          # running average over 4 slots
```
