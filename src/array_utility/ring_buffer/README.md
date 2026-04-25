# ring_buffer

## Overview

This directory contains fixed-length ring buffer utilities for array features.

## Which class to use

| Class | Use this when |
| ----- | ------------- |
| `RingBuffer` | Keep the latest N data points |
| `AveragingRingBuffer` | Keep the latest N data points and query `mean` often |
| `BitRingBuffer` | Keep the latest N bit-encoded vectors (for example pHash bytes) |
| `BitAveragingRingBuffer` | Keep bit-encoded vectors and query bitwise majority often |

If `mean` is queried often, use `AveragingRingBuffer`.

## Components

| Component | Description |
| --------- | ----------- |
| [base.py](base.py) | Abstract base class with common buffer operations |
| [averaging](averaging) | Averaging ring buffers (base + numeric + bit) |
| [standard](standard) | Standard ring buffers (base + numeric + bit) |

## Quick Usage

```python
import numpy as np
from array_utility import AveragingRingBuffer

buf = AveragingRingBuffer.build(n=3, init_value=np.array([0.0, 0.0], dtype=np.float32))
buf.update(np.array([1.0, 2.0], dtype=np.float32))
buf.extend(np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))

print(buf.latest)        # most recent vector
print(buf.get_last_k(2)) # last k vectors
print(buf.mean)          # mean over all buffer slots
```
