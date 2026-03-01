from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EncodedBits:
    payload: bytes
    bit_length: int


class BitOutput:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current_byte = 0
        self._bits_filled = 0
        self.bit_length = 0

    def write(self, bit: int) -> None:
        if bit not in (0, 1):
            raise ValueError(f"Bit must be 0 or 1, got {bit}")
        self._current_byte = (self._current_byte << 1) | bit
        self._bits_filled += 1
        self.bit_length += 1
        if self._bits_filled == 8:
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._bits_filled = 0

    def finish(self) -> EncodedBits:
        if self._bits_filled:
            self._current_byte <<= 8 - self._bits_filled
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._bits_filled = 0
        return EncodedBits(payload=bytes(self._buffer), bit_length=self.bit_length)


class BitInput:
    def __init__(self, payload: bytes, bit_length: int) -> None:
        if bit_length < 0:
            raise ValueError("bit_length must be non-negative")
        self._payload = payload
        self._bit_length = bit_length
        self._bit_offset = 0

    def read(self) -> int:
        if self._bit_offset >= self._bit_length:
            return 0
        byte_index = self._bit_offset // 8
        bit_index = 7 - (self._bit_offset % 8)
        bit = (self._payload[byte_index] >> bit_index) & 1
        self._bit_offset += 1
        return bit


class ArithmeticEncoder:
    def __init__(self, state_bits: int = 32) -> None:
        if state_bits < 16:
            raise ValueError("state_bits must be >= 16")
        self.state_bits = state_bits
        self._full_range = 1 << state_bits
        self._half_range = self._full_range >> 1
        self._quarter_range = self._half_range >> 1
        self._three_quarter_range = self._quarter_range * 3

        self.low = 0
        self.high = self._full_range - 1
        self._pending_bits = 0
        self._output = BitOutput()

    def encode_symbol(self, cum_low: int, cum_high: int, total: int) -> None:
        if not (0 <= cum_low < cum_high <= total):
            raise ValueError("Invalid cumulative bounds")

        interval = self.high - self.low + 1
        self.high = self.low + (interval * cum_high // total) - 1
        self.low = self.low + (interval * cum_low // total)

        while True:
            if self.high < self._half_range:
                self._write_bit_with_pending(0)
            elif self.low >= self._half_range:
                self._write_bit_with_pending(1)
                self.low -= self._half_range
                self.high -= self._half_range
            elif self.low >= self._quarter_range and self.high < self._three_quarter_range:
                self._pending_bits += 1
                self.low -= self._quarter_range
                self.high -= self._quarter_range
            else:
                break

            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> EncodedBits:
        self._pending_bits += 1
        if self.low < self._quarter_range:
            self._write_bit_with_pending(0)
        else:
            self._write_bit_with_pending(1)
        return self._output.finish()

    def _write_bit_with_pending(self, bit: int) -> None:
        self._output.write(bit)
        complement = 1 - bit
        for _ in range(self._pending_bits):
            self._output.write(complement)
        self._pending_bits = 0


class ArithmeticDecoder:
    def __init__(self, bit_input: BitInput, state_bits: int = 32) -> None:
        if state_bits < 16:
            raise ValueError("state_bits must be >= 16")
        self.state_bits = state_bits
        self._full_range = 1 << state_bits
        self._half_range = self._full_range >> 1
        self._quarter_range = self._half_range >> 1
        self._three_quarter_range = self._quarter_range * 3

        self.low = 0
        self.high = self._full_range - 1
        self._input = bit_input
        self.code = 0
        for _ in range(state_bits):
            self.code = (self.code << 1) | self._input.read()

    def decode_symbol(self, cumulative: np.ndarray, total: int) -> int:
        if cumulative.ndim != 1:
            raise ValueError("Cumulative table must be a 1D array")

        interval = self.high - self.low + 1
        value = ((self.code - self.low + 1) * total - 1) // interval
        symbol = int(np.searchsorted(cumulative, value, side="right") - 1)
        cum_low = int(cumulative[symbol])
        cum_high = int(cumulative[symbol + 1])

        self.high = self.low + (interval * cum_high // total) - 1
        self.low = self.low + (interval * cum_low // total)

        while True:
            if self.high < self._half_range:
                pass
            elif self.low >= self._half_range:
                self.low -= self._half_range
                self.high -= self._half_range
                self.code -= self._half_range
            elif self.low >= self._quarter_range and self.high < self._three_quarter_range:
                self.low -= self._quarter_range
                self.high -= self._quarter_range
                self.code -= self._quarter_range
            else:
                break

            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self._input.read()

        return symbol
