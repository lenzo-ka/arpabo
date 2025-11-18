"""Smoothing methods for language models

This module provides various smoothing techniques for n-gram language models:
- Good-Turing: Uses frequency-of-frequencies
- Kneser-Ney: Uses continuation counts with absolute discounting
- Katz Backoff: Uses discount mass with backoff weights
"""

from arpabo.smoothing.base import SmoothingMethod
from arpabo.smoothing.good_turing import GoodTuringSmoother
from arpabo.smoothing.katz_backoff import KatzBackoffSmoother
from arpabo.smoothing.kneser_ney import KneserNeySmoother

__all__ = [
    "SmoothingMethod",
    "GoodTuringSmoother",
    "KneserNeySmoother",
    "KatzBackoffSmoother",
    "create_smoother",
]


def create_smoother(
    method: str, max_order: int, verbose: bool = False, discount_mass: float = None, discount_step: float = 0.05
) -> SmoothingMethod:
    """Factory function to create appropriate smoother.

    Args:
        method: Smoothing method name ('good_turing', 'kneser_ney', 'auto', 'fixed', 'mle')
        max_order: Maximum n-gram order
        verbose: Verbose output
        discount_mass: Discount mass for Katz backoff (None for auto)
        discount_step: Step size for auto optimization

    Returns:
        Appropriate SmoothingMethod instance

    Examples:
        >>> smoother = create_smoother("good_turing", max_order=3)
        >>> smoother = create_smoother("kneser_ney", max_order=4, verbose=True)
        >>> smoother = create_smoother("auto", max_order=3, discount_step=0.01)
    """
    if method == "good_turing":
        return GoodTuringSmoother(max_order, verbose)

    elif method == "kneser_ney":
        return KneserNeySmoother(max_order, verbose)

    elif method == "auto":
        if discount_mass is None:
            discount_mass = min(0.1 * max_order, 0.5)
        return KatzBackoffSmoother(
            max_order, verbose, discount_mass=discount_mass, auto_optimize=True, discount_step=discount_step
        )

    elif method in ["fixed", "mle"]:
        if discount_mass is None:
            discount_mass = min(0.1 * max_order, 0.5) if method == "fixed" else 0.0
        return KatzBackoffSmoother(max_order, verbose, discount_mass=discount_mass, auto_optimize=False)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")
