"""Utility functions for smoothing algorithms"""

from typing import Any


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
