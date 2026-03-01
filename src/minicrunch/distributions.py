from __future__ import annotations

import numpy as np
import torch


def logits_to_cumulative(logits: torch.Tensor, total_freq: int) -> np.ndarray:
    """Convert logits to an integer cumulative distribution for arithmetic coding.

    Every token gets at least frequency 1 to avoid zero-probability events.
    """
    if logits.ndim != 1:
        raise ValueError("Expected a 1D logits tensor")

    vocab_size = logits.shape[0]
    if total_freq <= vocab_size:
        raise ValueError(
            f"total_freq ({total_freq}) must be > vocab_size ({vocab_size})"
        )

    probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
    extra_total = total_freq - vocab_size

    allocation = probs * extra_total
    extras = np.floor(allocation).astype(np.int64)
    frequencies = extras + 1

    remainder = int(extra_total - int(extras.sum()))
    if remainder > 0:
        fractions = allocation - extras
        top_indices = np.argpartition(fractions, -remainder)[-remainder:]
        frequencies[top_indices] += 1

    cumulative = np.empty(vocab_size + 1, dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(frequencies, out=cumulative[1:])

    if int(cumulative[-1]) != total_freq:
        raise RuntimeError(
            f"Internal error: cumulative total {int(cumulative[-1])} != {total_freq}"
        )

    return cumulative
