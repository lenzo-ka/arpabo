"""Good-Turing smoothing"""

from collections import defaultdict
from typing import Any

from arpabo.smoothing.base import SmoothingMethod
from arpabo.smoothing.utils import katz_discount_ratios, set_ngram_prob


class GoodTuringSmoother(SmoothingMethod):
    """Good-Turing smoothing using frequency of frequencies.

    Good-Turing re-estimates counts using frequency-of-frequencies:
    c* = (c+1) * N(c+1) / N(c)
    where N(c) is the number of n-grams that appear exactly c times.

    Best for sparse data with many singleton n-grams.
    """

    def needs_backoff_weights(self) -> bool:
        return False

    def compute_probabilities(self, grams: list[Any], sum_1: int, probs: list[Any], alphas: list[Any]) -> None:
        """Compute Good-Turing smoothed probabilities."""
        for order in range(self.max_order):
            if order == 0:
                self._compute_unigram_probabilities(grams[0], sum_1, probs[0])
            else:
                self._compute_order_probabilities(grams[order], order, probs[order])

    def _compute_frequency_of_frequencies(self, gram_dict: Any, order: int) -> dict[int, int]:
        """Compute frequency of frequencies: how many n-grams appear c times."""
        freq_of_freq = defaultdict(int)

        def count_frequencies(d: Any, current_order: int) -> None:
            if current_order == 0:
                for _word, count in d.items():
                    freq_of_freq[count] += 1
            else:
                for _word, value in d.items():
                    count_frequencies(value, current_order - 1)

        count_frequencies(gram_dict, order)
        return dict(freq_of_freq)

    def _compute_unigram_probabilities(self, grams_0: Any, sum_1: int, probs_0: Any) -> None:
        """Compute Good-Turing smoothed unigram probabilities.

        The unigram distribution is renormalized downstream, so we only need
        proper (never-inflating) discounted counts here.
        """
        freq_of_freq = self._compute_frequency_of_frequencies(grams_0, 0)
        discount = katz_discount_ratios(freq_of_freq)
        total_count = sum(grams_0.values())

        for word, count in grams_0.items():
            adjusted_count = discount(count) * count
            prob = adjusted_count / total_count if total_count > 0 else 0.0
            probs_0[word] = min(prob, 1.0)

    def _compute_order_probabilities(self, gram_dict: Any, order: int, prob_dict: Any) -> None:
        """Compute Good-Turing probabilities for n-grams of a given order."""
        freq_of_freq = self._compute_frequency_of_frequencies(gram_dict, order)
        discount = katz_discount_ratios(freq_of_freq)

        def process_ngrams(gd: Any, parent_words: list[str], current_order: int) -> None:
            if current_order == 0:
                total_count = sum(gd.values())

                for word, count in gd.items():
                    ngram_words = parent_words + [word]
                    adjusted_count = discount(count) * count
                    prob = adjusted_count / total_count if total_count > 0 else 0.0
                    set_ngram_prob(prob_dict, ngram_words, min(prob, 1.0))
            else:
                for word, value in gd.items():
                    process_ngrams(value, parent_words + [word], current_order - 1)

        process_ngrams(gram_dict, [], order)
