# standard

## Overview

This directory contains standard ring buffer implementations.
These classes focus on storage and retrieval behavior, and compute representative values on demand.

## Components

| File/Directory | Responsibility |
| --- | --- |
| [`base.py`](base.py) | Base implementation for update and batch-append operations in a fixed-size circular buffer. |
| [`normal.py`](normal.py) | `RingBuffer` for numeric arrays with arithmetic-mean access via `mean`. |
| [`bit.py`](bit.py) | `BitRingBuffer` for `uint8` bit-packed arrays with bitwise-majority `mean`. |

## Usage

### Numeric Ring Buffer

```python
import numpy as np
from array_utility import RingBuffer

buf = RingBuffer.build(n=3, init_value=np.array([0.0, 0.0], dtype=np.float32))
buf.update(np.array([1.0, 2.0], dtype=np.float32))
print(buf.mean)
```

### Bit-Encoded Ring Buffer

```python
import numpy as np
from array_utility import BitRingBuffer

bit_buf = BitRingBuffer.build(n=3, init_value=np.array([0b00000000], dtype=np.uint8))
bit_buf.extend(np.array([[0b11110000], [0b11000011]], dtype=np.uint8))
print(bit_buf.mean)
```
