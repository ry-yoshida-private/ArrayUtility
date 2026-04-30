# averaging

## Overview

This directory contains ring buffer implementations with cached accumulation.
These classes optimize frequent `mean` queries by updating internal statistics during `update` and `extend`.

## Components

| File/Directory | Responsibility |
| --- | --- |
| [`base.py`](base.py) | Base averaging ring buffer with incremental accumulator updates for single and batch writes. |
| [`normal.py`](normal.py) | `AveragingRingBuffer` for numeric arrays with arithmetic mean from cached sums. |
| [`bit.py`](bit.py) | `BitAveragingRingBuffer` for `uint8` bit-packed arrays with cached bit counts and majority output. |

## Usage

### Numeric Averaging Ring Buffer

```python
import numpy as np
from array_utility import AveragingRingBuffer

buf = AveragingRingBuffer.build(n=3, init_value=np.array([0.0, 0.0], dtype=np.float32))
buf.extend(np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32))
print(buf.mean)
```

### Bit-Encoded Averaging Ring Buffer

```python
import numpy as np
from array_utility import BitAveragingRingBuffer

bit_avg_buf = BitAveragingRingBuffer.build(n=3, init_value=np.array([0b00000000], dtype=np.uint8))
bit_avg_buf.extend(np.array([[0b11110000], [0b11110000], [0b00001111]], dtype=np.uint8))
print(bit_avg_buf.mean)
```
