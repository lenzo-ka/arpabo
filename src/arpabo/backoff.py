"""Finalize a computed model into a proper ARPA backoff distribution.

Smoothers produce discounted conditional probabilities P(w | h) that leave some
mass unassigned for unseen events. Turning that into a normalized model is a
smoother-independent step:

  1. The unigram distribution is rescaled to sum to 1 over the predictable
     vocabulary (everything except ``<s>``), so the recursion has a proper base.
  2. Every context h below the top order gets the backoff weight

         alpha(h) = (1 - sum_{w seen after h} P(w | h))
                    / (1 - sum_{w seen after h} P(w | h'))

     where h' is h with its oldest word dropped. With these weights the full
     backoff distribution sums to exactly 1 for every context.

Keeping this out of the individual smoothers is what makes Good-Turing,
Kneser-Ney, Katz and MLE all emit valid probability models.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

START = "<s>"

# log10 sentinel for events that must never be predicted (the <s> unigram) and
# for a context that reserves no backoff mass at all.
NEG_LOG10 = -99.0

_EPS = 1e-12


def finalize_model(lm) -> None:
    """Normalize unigrams and compute correct backoff weights in place."""
    normalize_unigrams(lm)
    compute_backoff_weights(lm)


def normalize_unigrams(lm) -> None:
    """Rescale unigram probabilities to sum to 1 over predictable words.

    ``<s>`` is excluded from the mass (it is never predicted); its stored value
    is left untouched since the writer emits the -99 sentinel for it.
    """
    if not lm.probs or not lm.probs[0]:
        return

    total = sum(p for w, p in lm.probs[0].items() if w != START)
    if total <= _EPS:
        return

    for word in lm.probs[0]:
        if word != START:
            lm.probs[0][word] /= total


def compute_backoff_weights(lm) -> None:
    """Compute alpha(h) for every context below the top order, keyed by the
    full n-gram (the context itself), and store it in ``lm.alphas``.

    Backoff weights are rebuilt from scratch here so that any (differently
    keyed) values a smoother may have left behind cannot corrupt the lookup.
    """
    for order in range(lm.max_order):
        # alphas[order] is keyed by the full (order+1)-word context.
        lm.alphas[order] = {} if order == 0 else _nested_defaultdict(order)

    for order in range(lm.max_order - 1):
        alpha_dict = lm.alphas[order]
        for context in _iter_ngrams(lm.probs[order], order + 1):
            alpha = _alpha_for_context(lm, context, order)
            _set_nested(alpha_dict, context, alpha)


def _nested_defaultdict(depth: int) -> Any:
    if depth == 0:
        return defaultdict(float)
    return defaultdict(lambda: _nested_defaultdict(depth - 1))


def _alpha_for_context(lm, context: tuple[str, ...], order: int) -> float:
    """Backoff weight for ``context`` (an (order+1)-gram) predicting order+2 grams."""
    extensions = _child_dict(lm.probs[order + 1], context)
    if not extensions:
        # No observed continuations: no mass is discounted, backoff weight is 1.
        return 1.0

    lower_ctx = context[1:]  # h' : drop the oldest word
    seen_high = 0.0
    seen_low = 0.0
    for word, high_prob in extensions.items():
        if isinstance(high_prob, dict):
            # Deeper structure than a leaf; skip defensively (should not happen).
            continue
        seen_high += high_prob
        seen_low += _lower_prob(lm, lower_ctx, word, order)

    numerator = 1.0 - seen_high
    denominator = 1.0 - seen_low
    if numerator <= _EPS or denominator <= _EPS:
        # Saturated context: essentially no backoff mass to distribute.
        return 0.0
    return numerator / denominator


def _lower_prob(lm, lower_ctx: tuple[str, ...], word: str, order: int) -> float:
    """P(word | lower_ctx) read from the order-below distribution."""
    if not lower_ctx:
        return lm.probs[0].get(word, 0.0)
    node = _child_dict(lm.probs[order], lower_ctx)
    if not node:
        return 0.0
    value = node.get(word, 0.0)
    return value if not isinstance(value, dict) else 0.0


def _child_dict(prob_root: Any, words: tuple[str, ...]) -> Any:
    """Navigate a nested prob dict by ``words``; return the child dict or None."""
    current = prob_root
    for w in words:
        if not isinstance(current, dict) or w not in current:
            return None
        current = current[w]
    return current if isinstance(current, dict) else None


def _iter_ngrams(prob_root: Any, length: int):
    """Yield every full n-gram tuple of the given length from a nested dict."""
    if length == 1:
        for word in prob_root:
            yield (word,)
        return

    def walk(node: Any, prefix: tuple[str, ...], depth: int):
        if depth == 1:
            for word in node:
                yield prefix + (word,)
        else:
            for word, child in node.items():
                yield from walk(child, prefix + (word,), depth - 1)

    yield from walk(prob_root, (), length)


def _set_nested(root: Any, words: tuple[str, ...], value: float) -> None:
    current = root
    for w in words[:-1]:
        current = current[w]
    current[words[-1]] = value
