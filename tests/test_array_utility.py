# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from array_utility import (  # noqa: E402
    AveragingRingBuffer,
    BitEMACalculator,
    BitAveragingRingBuffer,
    BitRingBuffer,
    EMACalculator,
)
from array_utility.ring_buffer.standard import RingBuffer  # noqa: E402


class TestEMACalculator(unittest.TestCase):
    def test_update(self) -> None:
        ema = EMACalculator(alpha=0.2, value=np.array([0.0, 10.0], dtype=np.float32))
        ema.update(np.array([10.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(ema.value, np.array([2.0, 8.0], dtype=np.float32))


class TestBitEMACalculator(unittest.TestCase):
    def test_value_thresholds_bits_with_ema(self) -> None:
        ema = BitEMACalculator.build(alpha=0.4, value=np.array([0b00000000], dtype=np.uint8))
        ema.update(np.array([0b11110000], dtype=np.uint8))
        np.testing.assert_array_equal(ema.value, np.array([0b00000000], dtype=np.uint8))

        ema.update(np.array([0b11110000], dtype=np.uint8))
        np.testing.assert_array_equal(ema.value, np.array([0b11110000], dtype=np.uint8))

    def test_requires_uint8_input(self) -> None:
        with self.assertRaises(TypeError):
            BitEMACalculator.build(alpha=0.2, value=np.array([1.0], dtype=np.float32))

        ema = BitEMACalculator.build(alpha=0.2, value=np.array([0b00000000], dtype=np.uint8))
        with self.assertRaises(TypeError):
            ema.update(np.array([1.0], dtype=np.float32))


class TestRingBuffer(unittest.TestCase):
    def test_update_extend_and_accessors(self) -> None:
        buf = RingBuffer.build(n=3, init_value=np.array([0.0, 0.0], dtype=np.float32))
        buf.update(np.array([1.0, 1.0], dtype=np.float32))
        buf.extend(np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))

        np.testing.assert_array_equal(buf.latest, np.array([3.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(
            buf.get_last_k(2),
            np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            buf.ordered_value,
            np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        )

    def test_extend_warns_and_truncates_when_input_exceeds_capacity(self) -> None:
        buf = RingBuffer.build(n=3, init_value=np.array([0.0], dtype=np.float32))
        values = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            buf.extend(values)
        self.assertEqual(len(caught), 1)
        self.assertIn("truncating", str(caught[0].message))
        np.testing.assert_array_equal(
            buf.ordered_value,
            np.array([[2.0], [3.0], [4.0]], dtype=np.float32),
        )


class TestAveragingRingBuffer(unittest.TestCase):
    def test_mean_uses_cached_accumulator_after_updates(self) -> None:
        buf = AveragingRingBuffer.build(n=3, init_value=np.array([0.0, 0.0], dtype=np.float32))
        buf.update(np.array([1.0, 2.0], dtype=np.float32))
        buf.update(np.array([2.0, 4.0], dtype=np.float32))
        buf.update(np.array([3.0, 6.0], dtype=np.float32))
        np.testing.assert_allclose(buf.mean, np.array([2.0, 4.0], dtype=np.float32))

        buf.extend(np.array([[4.0, 8.0], [5.0, 10.0]], dtype=np.float32))
        np.testing.assert_allclose(buf.mean, np.array([4.0, 8.0], dtype=np.float32))


class TestBitRingBuffers(unittest.TestCase):
    def test_bit_ring_buffer_mean_majority(self) -> None:
        # Majority for each bit across the 3 rows should be:
        # [1, 1, 0, 0, 1, 0, 1, 0] == 0b11001010
        value = np.array([[0b11001010], [0b11110000], [0b10001011]], dtype=np.uint8)
        buf = BitRingBuffer(n=3, value=value.copy())
        np.testing.assert_array_equal(buf.mean, np.array([0b11001010], dtype=np.uint8))

    def test_bit_averaging_ring_buffer_mean_and_updates(self) -> None:
        init = np.array([0b00000000], dtype=np.uint8)
        buf = BitAveragingRingBuffer.build(n=3, init_value=init)
        buf.extend(np.array([[0b11110000], [0b11110000], [0b00001111]], dtype=np.uint8))
        np.testing.assert_array_equal(buf.mean, np.array([0b11110000], dtype=np.uint8))

        buf.update(np.array([0b00001111], dtype=np.uint8))
        np.testing.assert_array_equal(buf.mean, np.array([0b00001111], dtype=np.uint8))

    def test_bit_buffers_require_uint8(self) -> None:
        with self.assertRaises(TypeError):
            BitRingBuffer.build(n=2, init_value=np.array([1.0], dtype=np.float32))
        with self.assertRaises(TypeError):
            BitAveragingRingBuffer.build(n=2, init_value=np.array([1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
