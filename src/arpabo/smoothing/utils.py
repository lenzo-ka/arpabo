"""Utility functions for smoothing algorithms"""

from collections.abc import Callable
from typing import Any


def katz_discount_ratios(freq_of_freq: dict[int, int], cutoff: int = 7) -> Callable[[int], float]:
    """Return a function c -> d_c giving the Katz-Good-Turing discount ratio.

    Katz (1987) discounts only counts at or below ``cutoff`` (large counts are
    reliable and kept undiscounted, d_c = 1), using the Good-Turing estimate
    ``c* = (c+1) N(c+1) / N(c)`` renormalized so that

        0 < d_c <= 1     and     sum of discounted mass == N(1)/N.

    The renormalization is only valid when N(1) and N(k+1) are present and the
    resulting ratios stay in (0, 1]; for the small/gappy frequency tables that
    are common on tiny corpora we shrink the cutoff and, failing that, fall back
    to a plain absolute-discount ratio D = N1/(N1 + 2 N2). Either way the
    returned ratio never inflates a count, which is what keeps every context's
    seen probability mass at or below 1.
    """
    n = freq_of_freq

    def absolute_fallback() -> Callable[[int], float]:
        n1 = n.get(1, 0)
        n2 = n.get(2, 0)
        d = n1 / (n1 + 2 * n2) if n2 > 0 else 0.5
        # Keep strictly inside (0, 1) so singleton counts retain positive mass
        # (seen probability) while still reserving mass for backoff.
        d = min(max(d, 0.05), 0.95)

        def ratio(c: int) -> float:
            if c <= 0:
                return 1.0
            return max(c - d, 0.0) / c

        return ratio

    n1 = n.get(1, 0)
    if n1 <= 0:
        return absolute_fallback()

    # Shrink the cutoff until every frequency it depends on is present.
    k = cutoff
    while k >= 1 and (n.get(k + 1, 0) <= 0 or any(n.get(c, 0) <= 0 for c in range(1, k + 1))):
        k -= 1
    if k < 1:
        return absolute_fallback()

    common = (k + 1) * n[k + 1] / n1
    if not 0.0 < common < 1.0:
        return absolute_fallback()

    ratios: dict[int, float] = {}
    for c in range(1, k + 1):
        c_star = (c + 1) * n[c + 1] / n[c]
        d_c = (c_star / c - common) / (1.0 - common)
        if not 0.0 < d_c <= 1.0:
            return absolute_fallback()
        ratios[c] = d_c

    def ratio(c: int) -> float:
        if c <= 0:
            return 1.0
        return ratios.get(c, 1.0)

    return ratio


def set_ngram_prob(prob_dict: Any, ngram_words: list[str], prob: float) -> None:
    """Store a probability value in nested dictionary structure.

    Args:
        prob_dict: Nested dictionary to store probability in
        ngram_words: List of words forming the n-gram
        prob: Probability value to store
    """
    current = prob_dict
    for word in ngram_words[:-1]:
        current = current[word]
    current[ngram_words[-1]] = prob


def compute_simple_backoff_weights(
    gram_dict: Any, order: int, probs: list[Any], alpha_dict: Any, set_prob_fn: callable
) -> None:
    """Compute simple backoff weights (always 1.0).

    Used by Good-Turing and Kneser-Ney when not using Katz backoff.

    Args:
        gram_dict: N-gram dictionary for this order
        order: Current n-gram order
        probs: Probability dictionaries (unused, kept for interface compatibility)
        alpha_dict: Dictionary to store backoff weights
        set_prob_fn: Function to set probabilities in nested dict
    """

    def process_alphas(gd: Any, parent_words: list[str], current_order: int) -> None:
        if current_order == 0:
            if parent_words:
                set_prob_fn(alpha_dict, parent_words, 1.0)
        else:
            for word, value in gd.items():
                process_alphas(value, parent_words + [word], current_order - 1)

    process_alphas(gram_dict, [], order)
