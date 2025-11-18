"""Good-Turing smoothing"""

from collections import defaultdict
from typing import Any

from arpalm.smoothing.base import SmoothingMethod
from arpalm.smoothing.utils import compute_simple_backoff_weights, set_ngram_prob


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

            # Compute backoff weights for all but last order
            if order < self.max_order - 1:
                self._compute_backoff_weights(grams[order], order, probs, alphas[order])

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
        """Compute Good-Turing smoothed unigram probabilities."""
        freq_of_freq = self._compute_frequency_of_frequencies(grams_0, 0)
        total_count = sum(grams_0.values())

        n1 = freq_of_freq.get(1, 0)
        total_adjusted = total_count

        if n1 > 0:
            total_adjusted = total_count - n1 / 2.0

        for word, count in grams_0.items():
            if count + 1 in freq_of_freq and count in freq_of_freq and freq_of_freq[count] > 0:
                adjusted_count = (count + 1) * freq_of_freq[count + 1] / freq_of_freq[count]
            else:
                adjusted_count = count

            prob = adjusted_count / total_adjusted if total_adjusted > 0 else 0.0
            prob = min(prob, 1.0)
            probs_0[word] = prob

    def _compute_order_probabilities(self, gram_dict: Any, order: int, prob_dict: Any) -> None:
        """Compute Good-Turing probabilities for n-grams of a given order."""
        freq_of_freq = self._compute_frequency_of_frequencies(gram_dict, order)

        def process_ngrams(gd: Any, parent_words: list[str], current_order: int) -> None:
            if current_order == 0:
                total_count = sum(gd.values())

                for word, count in gd.items():
                    ngram_words = parent_words + [word]

                    if count + 1 in freq_of_freq and count in freq_of_freq and freq_of_freq[count] > 0:
                        adjusted_count = (count + 1) * freq_of_freq[count + 1] / freq_of_freq[count]
                    else:
                        adjusted_count = count

                    prob = adjusted_count / total_count if total_count > 0 else 0.0
                    prob = min(prob, 1.0)
                    set_ngram_prob(prob_dict, ngram_words, prob)
            else:
                for word, value in gd.items():
                    process_ngrams(value, parent_words + [word], current_order - 1)

        process_ngrams(gram_dict, [], order)

    def _compute_backoff_weights(self, gram_dict: Any, order: int, probs: list[Any], alpha_dict: Any) -> None:
        """Compute backoff weights (always 1.0 for Good-Turing without Katz)."""
        compute_simple_backoff_weights(gram_dict, order, probs, alpha_dict, set_ngram_prob)
