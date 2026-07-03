"""Kneser-Ney smoothing (backoff form)."""

from typing import Any

from arpabo.smoothing.base import SmoothingMethod
from arpabo.smoothing.utils import set_ngram_prob


class KneserNeySmoother(SmoothingMethod):
    """Kneser-Ney smoothing with absolute discounting and continuation counts.

    Kneser-Ney's defining idea is that lower-order distributions should be
    estimated from *continuation counts* -- how many distinct contexts a word
    completes -- rather than raw frequencies. This implementation produces a
    backoff model (the natural ARPA representation):

    - Top order:  P(w|h) = max(c(h,w) - D, 0) / c(h)          (raw counts)
    - Lower orders: P(w|h) = max(N1+(*,h,w) - D, 0) / N1+(*,h,*)  (continuation
      counts, where N1+(*,h,w) is the number of distinct words that precede the
      n-gram (h,w))
    - Unigrams: continuation counts over the bigram inventory.

    Backoff weights are added afterward by ``arpabo.backoff`` so the whole model
    normalizes; each context here only needs to leave mass for backoff, which
    absolute discounting guarantees.
    """

    DEFAULT_DISCOUNT = 0.75

    def __init__(self, max_order: int, verbose: bool = False, discount: float = DEFAULT_DISCOUNT):
        super().__init__(max_order, verbose)
        self.discount = discount
        self.grams: list[Any] = []

    def needs_backoff_weights(self) -> bool:
        # Backoff weights are computed centrally in arpabo.backoff.
        return False

    def compute_probabilities(self, grams: list[Any], sum_1: int, probs: list[Any], alphas: list[Any]) -> None:
        self.grams = grams
        top = self.max_order - 1

        for order in range(self.max_order):
            if order == 0:
                self._compute_unigram_probabilities(grams, sum_1, probs[0])
            elif order == top:
                self._compute_raw_probabilities(order, probs[order])
            else:
                self._compute_continuation_probabilities(order, probs[order])

    # ------------------------------------------------------------------
    # Continuation counts
    # ------------------------------------------------------------------

    def _continuation_counts(self, m: int) -> dict[tuple[str, ...], int]:
        """Continuation counts for m-length n-grams.

        N1+(* g) = number of distinct words that immediately precede g, derived
        from the (m+1)-grams in ``grams[m]`` by dropping their first word.
        """
        counts: dict[tuple[str, ...], int] = {}
        for ngram in self._iter_ngrams(self.grams[m], m + 1):
            suffix = ngram[1:]
            counts[suffix] = counts.get(suffix, 0) + 1
        return counts

    def _iter_ngrams(self, root: Any, length: int):
        if length == 1:
            yield from ((w,) for w in root)
            return

        def walk(node: Any, prefix: tuple[str, ...], depth: int):
            if depth == 1:
                for w in node:
                    yield prefix + (w,)
            else:
                for w, child in node.items():
                    yield from walk(child, prefix + (w,), depth - 1)

        yield from walk(root, (), length)

    # ------------------------------------------------------------------
    # Per-order probability estimators
    # ------------------------------------------------------------------

    def _compute_unigram_probabilities(self, grams: list[Any], sum_1: int, probs_0: Any) -> None:
        cont = self._continuation_counts(1) if self.max_order > 1 else {}
        total = sum(cont.values())

        for word in grams[0]:
            c = cont.get((word,), 0)
            if total > 0 and c > 0:
                probs_0[word] = c / total
            else:
                # Words that never appear as a continuation (e.g. <s>) keep a
                # raw ML estimate; the unigram row is renormalized downstream.
                probs_0[word] = grams[0][word] / sum_1

    def _compute_raw_probabilities(self, order: int, prob_dict: Any) -> None:
        """Top-order absolute discounting on raw counts."""
        d = self.discount
        for context, children in self._contexts(self.grams[order], order):
            total = sum(children.values())
            if total <= 0:
                continue
            for word, count in children.items():
                prob = max(count - d, 0.0) / total
                set_ngram_prob(prob_dict, list(context) + [word], prob)

    def _compute_continuation_probabilities(self, order: int, prob_dict: Any) -> None:
        """Lower-order absolute discounting on continuation counts."""
        d = self.discount
        cont = self._continuation_counts(order + 1)

        # Group continuation counts by their (order)-length context.
        by_context: dict[tuple[str, ...], dict[str, int]] = {}
        for ngram, c in cont.items():
            ctx, word = ngram[:-1], ngram[-1]
            by_context.setdefault(ctx, {})[word] = c

        for context, children in self._contexts(self.grams[order], order):
            cont_children = by_context.get(context)
            if cont_children:
                total = sum(cont_children.values())
                for word in children:
                    c = cont_children.get(word, 0)
                    prob = max(c - d, 0.0) / total if total > 0 else 0.0
                    set_ngram_prob(prob_dict, list(context) + [word], prob)
            else:
                # No continuation evidence for this context: fall back to raw
                # absolute discounting so the context still leaves backoff mass.
                total = sum(children.values())
                for word, count in children.items():
                    prob = max(count - d, 0.0) / total if total > 0 else 0.0
                    set_ngram_prob(prob_dict, list(context) + [word], prob)

    def _contexts(self, root: Any, order: int):
        """Yield (context_tuple, {word: count}) pairs for every context at this order.

        ``grams[order]`` is nested order+1 deep; descending ``order`` levels lands
        on the innermost {word: count} dict, with the length-``order`` context as
        the accumulated prefix.
        """

        def walk(node: Any, prefix: tuple[str, ...], depth: int):
            if depth == 0:
                yield prefix, node
            else:
                for w, child in node.items():
                    yield from walk(child, prefix + (w,), depth - 1)

        yield from walk(root, (), order)
