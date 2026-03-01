from __future__ import annotations

import numpy as np

from minicrunch.arithmetic import ArithmeticDecoder, ArithmeticEncoder, BitInput


def test_roundtrip_fixed_distribution() -> None:
    frequencies = np.array([2, 1, 4, 3], dtype=np.int64)
    cumulative = np.empty(len(frequencies) + 1, dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(frequencies, out=cumulative[1:])
    total = int(cumulative[-1])

    sequence = [0, 3, 2, 2, 1, 0, 3, 2, 0, 3, 1, 2]

    encoder = ArithmeticEncoder()
    for symbol in sequence:
        encoder.encode_symbol(
            cum_low=int(cumulative[symbol]),
            cum_high=int(cumulative[symbol + 1]),
            total=total,
        )
    encoded = encoder.finish()

    decoder = ArithmeticDecoder(BitInput(encoded.payload, encoded.bit_length))
    decoded = [decoder.decode_symbol(cumulative, total) for _ in sequence]

    assert decoded == sequence


def test_roundtrip_dynamic_distributions() -> None:
    rng = np.random.default_rng(7)

    distributions: list[tuple[np.ndarray, int]] = []
    sequence: list[int] = []

    for _ in range(120):
        frequencies = rng.integers(1, 12, size=9, dtype=np.int64)
        cumulative = np.empty(len(frequencies) + 1, dtype=np.int64)
        cumulative[0] = 0
        np.cumsum(frequencies, out=cumulative[1:])
        total = int(cumulative[-1])

        symbol = int(rng.integers(0, len(frequencies)))

        distributions.append((cumulative, total))
        sequence.append(symbol)

    encoder = ArithmeticEncoder()
    for symbol, (cumulative, total) in zip(sequence, distributions):
        encoder.encode_symbol(
            cum_low=int(cumulative[symbol]),
            cum_high=int(cumulative[symbol + 1]),
            total=total,
        )
    encoded = encoder.finish()

    decoder = ArithmeticDecoder(BitInput(encoded.payload, encoded.bit_length))
    decoded = [decoder.decode_symbol(cumulative, total) for cumulative, total in distributions]

    assert decoded == sequence
