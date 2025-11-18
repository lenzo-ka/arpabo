"""Kneser-Ney smoothing"""

from collections import defaultdict
from typing import Any

from arpabo.smoothing.base import SmoothingMethod
from arpabo.smoothing.utils import compute_simple_backoff_weights, set_ngram_prob


class KneserNeySmoother(SmoothingMethod):
    """Kneser-Ney smoothing with absolute discounting.

    Kneser-Ney uses:
    - Absolute discounting: subtract D from all counts
    - Continuation counts: for lower orders, use unique context counts
    - Interpolation: blend with lower-order probabilities

    Often performs better than Good-Turing for well-represented data.
    """

    DEFAULT_DISCOUNT = 0.75

    def __init__(self, max_order: int, verbose: bool = False, discount: float = DEFAULT_DISCOUNT):
        super().__init__(max_order, verbose)
        self.discount = discount
        self.grams = None  # Will be set during compute

    def needs_backoff_weights(self) -> bool:
        return False

    def compute_probabilities(self, grams: list[Any], sum_1: int, probs: list[Any], alphas: list[Any]) -> None:
        """Compute Kneser-Ney smoothed probabilities."""
        self.grams = grams  # Store for continuation count computation

        for order in range(self.max_order):
            if order == 0:
                self._compute_unigram_probabilities(grams, sum_1, probs[0])
            else:
                self._compute_order_probabilities(grams[order], order, probs[order])

            # Compute backoff weights
            if order < self.max_order - 1:
                self._compute_backoff_weights(grams[order], order, probs, alphas[order])

    def _compute_continuation_counts(self, order: int) -> Any:
        """Compute continuation counts: how many different contexts each word appears in."""
        if order == 0:
            continuation_counts = defaultdict(int)
            if self.max_order > 1:
                for _w1, bigrams in self.grams[1].items():
                    for w2 in bigrams:
                        continuation_counts[w2] += 1
            return continuation_counts

        from arpabo.lm import ArpaBoLM

        continuation_counts = ArpaBoLM._make_nested_defaultdict(order - 1)

        def count_contexts(gd: Any, parent_words: list[str], current_order: int) -> None:
            if current_order == 0:
                for word in gd:
                    ngram_words = parent_words + [word]
                    suffix = ngram_words[1:]
                    if suffix:
                        current = continuation_counts
                        for w in suffix[:-1]:
                            current = current[w]
                        current[suffix[-1]] += 1
            else:
                for word, value in gd.items():
                    count_contexts(value, parent_words + [word], current_order - 1)

        count_contexts(self.grams[order], [], order)
        return continuation_counts

    def _compute_unigram_probabilities(self, grams: list[Any], sum_1: int, probs_0: Any) -> None:
        """Compute Kneser-Ney unigram probabilities using continuation counts."""
        continuation_counts = self._compute_continuation_counts(0)
        total_continuations = sum(continuation_counts.values()) if continuation_counts else sum_1

        for word in grams[0]:
            if continuation_counts and word in continuation_counts:
                prob = continuation_counts[word] / total_continuations
            else:
                prob = grams[0][word] / sum_1
            probs_0[word] = prob

    def _compute_order_probabilities(self, gram_dict: Any, order: int, prob_dict: Any) -> None:
        """Compute Kneser-Ney probabilities with absolute discounting."""

        def process_ngrams(gd: Any, parent_words: list[str], current_order: int) -> None:
            if current_order == 0:
                total_count = sum(gd.values())
                len(gd)

                for word, count in gd.items():
                    ngram_words = parent_words + [word]
                    discounted_count = max(count - self.discount, 0.0)

                    prob = (discounted_count / total_count) if total_count > 0 else 0.0
                    set_ngram_prob(prob_dict, ngram_words, prob)
            else:
                for word, value in gd.items():
                    process_ngrams(value, parent_words + [word], current_order - 1)

        process_ngrams(gram_dict, [], order)

    def _compute_backoff_weights(self, gram_dict: Any, order: int, probs: list[Any], alpha_dict: Any) -> None:
        """Compute backoff weights (always 1.0 for interpolated Kneser-Ney)."""
        compute_simple_backoff_weights(gram_dict, order, probs, alpha_dict, set_ngram_prob)
