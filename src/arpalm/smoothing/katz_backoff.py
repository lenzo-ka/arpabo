"""Katz backoff smoothing"""

from typing import Any

from arpalm.smoothing.base import SmoothingMethod
from arpalm.smoothing.utils import set_ngram_prob


class KatzBackoffSmoother(SmoothingMethod):
    """Katz backoff smoothing with discount mass.

    Uses a discount factor to reserve probability mass for unseen n-grams,
    then backs off to lower-order n-grams with backoff weights (alphas).

    Can be used with fixed discount mass or auto-optimized.
    """

    DEFAULT_DISCOUNT_MASS = 0.3
    MAX_DISCOUNT_MASS = 1.0

    def __init__(
        self,
        max_order: int,
        verbose: bool = False,
        discount_mass: float = DEFAULT_DISCOUNT_MASS,
        auto_optimize: bool = False,
        discount_step: float = 0.05,
    ):
        super().__init__(max_order, verbose)
        self.discount_mass = discount_mass
        self.auto_optimize = auto_optimize
        self.discount_step = discount_step
        self.deflator = 1.0 - discount_mass
        self.grams = None
        self.sum_1 = None

    def needs_backoff_weights(self) -> bool:
        return True

    def compute_probabilities(self, grams: list[Any], sum_1: int, probs: list[Any], alphas: list[Any]) -> None:
        """Compute Katz backoff smoothed probabilities."""
        self.grams = grams
        self.sum_1 = sum_1

        if self.auto_optimize:
            self._optimize_discount_mass()

        for order in range(self.max_order):
            if order == 0:
                for word, count in grams[0].items():
                    probs[0][word] = count / sum_1
            else:
                self._compute_order_probabilities(grams[order], order, probs[order])
                self._compute_order_alphas(grams[order], order, probs, alphas[order])

    def _optimize_discount_mass(self) -> None:
        """Find the best discount mass by testing different values."""
        if self.verbose:
            import sys

            print("Optimizing discount mass...", file=sys.stderr)

        candidates = [
            round(i * self.discount_step, 3) for i in range(1, int(self.MAX_DISCOUNT_MASS / self.discount_step))
        ]
        best_mass = self.DEFAULT_DISCOUNT_MASS
        best_likelihood = float("-inf")

        for mass in candidates:
            self.discount_mass = mass
            self.deflator = 1.0 - mass

            likelihood = self._compute_likelihood()

            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_mass = mass

        self.discount_mass = best_mass
        self.deflator = 1.0 - best_mass

        if self.verbose:
            import sys

            print(f"Optimal discount mass: {best_mass:.3f} ({len(candidates)} candidates tested)", file=sys.stderr)

    def _compute_likelihood(self) -> float:
        """Calculate how well the model predicts words in the corpus."""
        if self.sum_1 == 0:
            return 0.0

        from math import log

        likelihood = 0.0
        for _word, count in self.grams[0].items():
            if count > 0:
                prob = count / self.sum_1
                likelihood += count * log(prob)

        return likelihood

    def _compute_order_probabilities(self, gram_dict: Any, order: int, prob_dict: Any) -> None:
        """Calculate probabilities for n-grams with discounting."""

        def process_ngrams(gd: Any, parent_words: list[str], current_order: int) -> None:
            if current_order == 0:
                for word, count in gd.items():
                    ngram_words = parent_words + [word]
                    parent_count = self._get_parent_count(ngram_words, order)
                    if parent_count > 0:
                        prob = count * self.deflator / parent_count
                        set_ngram_prob(prob_dict, ngram_words, prob)
            else:
                for word, value in gd.items():
                    process_ngrams(value, parent_words + [word], current_order - 1)

        process_ngrams(gram_dict, [], order)

    def _compute_order_alphas(self, gram_dict: Any, order: int, probs: list[Any], alpha_dict: Any) -> None:
        """Calculate backoff weights (alphas)."""

        def process_alphas(gd: Any, parent_words: list[str], current_order: int) -> None:
            if current_order == 0:
                if not parent_words:
                    return

                sum_denom = 0.0
                for word in gd:
                    ngram_words = parent_words + [word]
                    if order + 1 < self.max_order:
                        next_prob = self._get_next_word_prob(ngram_words, order + 1)
                        sum_denom += next_prob

                alpha = 1.0 if sum_denom >= 1.0 else self.discount_mass / (1.0 - sum_denom)

                set_ngram_prob(alpha_dict, parent_words, alpha)
            else:
                for word, value in gd.items():
                    process_alphas(value, parent_words + [word], current_order - 1)

        process_alphas(gram_dict, [], order)

    def _get_parent_count(self, ngram_words: list[str], order: int) -> int:
        """Get the count of the parent n-gram."""
        if len(ngram_words) == 1:
            return self.sum_1
        parent_words = ngram_words[:-1]
        return self._get_ngram_count(self.grams[len(parent_words) - 1], parent_words)

    def _get_ngram_count(self, gram_dict: Any, ngram_words: list[str]) -> int:
        """Get the count of a specific n-gram."""
        current = gram_dict
        for word in ngram_words[:-1]:
            if word not in current:
                return 0
            current = current[word]
        return current.get(ngram_words[-1], 0)

    def _get_next_word_prob(self, context_words: list[str], order: int) -> float:
        """Calculate the probability of any next word given context."""
        if order >= self.max_order:
            return 0.0

        gram_dict = self.grams[order]
        current = gram_dict
        for word in context_words:
            if word not in current:
                return 0.0
            current = current[word]

        total_prob = 0.0
        for word, count in current.items():
            ngram_words = context_words + [word]
            parent_count = self._get_parent_count(ngram_words, order)
            if parent_count > 0:
                total_prob += count / parent_count
        return total_prob
