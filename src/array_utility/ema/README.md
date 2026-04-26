# ema

EMA-related calculators for numeric arrays and bit-encoded arrays.

## Classes

- `EMACalculator`
  - Standard EMA for numeric vectors.
  - Updates with: `ema = alpha * x + (1 - alpha) * ema`.
- `BitEMACalculator`
  - Accepts packed `uint8` bit arrays.
  - Internally tracks per-bit EMA as floating-point values in `[0, 1]`.
  - Exposes `value` as packed `uint8` bits using threshold `0.5`.

## Example

```python
import numpy as np
from array_utility.ema import BitEMACalculator, EMACalculator

ema = EMACalculator(alpha=0.2, value=np.array([0.0, 10.0], dtype=np.float32))
ema.update(np.array([10.0, 0.0], dtype=np.float32))
print(ema.value)

bit_ema = BitEMACalculator.build(alpha=0.4, value=np.array([0b00000000], dtype=np.uint8))
bit_ema.update(np.array([0b11110000], dtype=np.uint8))
print(bit_ema.value)
```
