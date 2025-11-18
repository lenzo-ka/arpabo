"""Base class for smoothing methods"""

from abc import ABC, abstractmethod
from typing import Any


class SmoothingMethod(ABC):
    """Base class for language model smoothing methods.

    Smoothing handles the problem of zero probabilities for unseen n-grams.
    Different methods use different strategies to redistribute probability mass.
    """

    def __init__(self, max_order: int, verbose: bool = False):
        self.max_order = max_order
        self.verbose = verbose

    @abstractmethod
    def compute_probabilities(self, grams: list[Any], sum_1: int, probs: list[Any], alphas: list[Any]) -> None:
        """Compute probabilities for all n-gram orders.

        Args:
            grams: List of n-gram count dictionaries (one per order)
            sum_1: Total unigram count
            probs: List of probability dictionaries to populate (one per order)
            alphas: List of backoff weight dictionaries to populate (one per order)
        """
        pass

    @abstractmethod
    def needs_backoff_weights(self) -> bool:
        """Whether this smoothing method uses backoff weights (alphas)."""
        pass
