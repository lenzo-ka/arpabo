"""Independent ARPA reader and backoff scorer for tests.

Deliberately does NOT reuse arpabo's own I/O so it can serve as an oracle:
it parses an ARPA file positionally by declared order and evaluates
P(w | history) with the standard backoff recursion

    P(w | h) = p(h, w)                        if (h, w) is explicit
             = bow(h) * P(w | h[1:])          otherwise

so that a correctly normalized model satisfies

    sum_{w in vocab, w != <s>} P(w | h) == 1   for every context h.
"""

from __future__ import annotations


class ArpaModel:
    def __init__(self) -> None:
        self.order = 0
        # tuple(words) -> log10 prob ; tuple(context words) -> log10 backoff
        self.logprob: dict[tuple[str, ...], float] = {}
        self.logbow: dict[tuple[str, ...], float] = {}
        self.vocab: list[str] = []

    def prob(self, history: tuple[str, ...], word: str) -> float:
        """Linear P(word | history) via ARPA backoff recursion."""
        ngram = history + (word,)
        if ngram in self.logprob:
            return 10.0 ** self.logprob[ngram]
        if not history:
            # No unigram for this word -> unseen (closed vocab: probability 0).
            return 0.0
        bow = 10.0 ** self.logbow.get(history, 0.0)
        return bow * self.prob(history[1:], word)

    def context_sum(self, history: tuple[str, ...]) -> float:
        """Sum of P(w | history) over the predictable vocabulary (excludes <s>)."""
        return sum(self.prob(history, w) for w in self.vocab if w != "<s>")


def parse_arpa(path: str) -> ArpaModel:
    model = ArpaModel()
    current_order = 0
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("Corpus:") or line == "\\data\\" or line == "\\end\\":
                continue
            if line.startswith("ngram ") and "=" in line:
                o = int(line.split("=")[0].split()[1])
                model.order = max(model.order, o)
                continue
            if line.startswith("\\") and line.endswith("-grams:"):
                current_order = int(line[1:].split("-")[0])
                continue
            if current_order == 0:
                continue
            parts = line.split()
            # Positional parse: logprob, then exactly `current_order` words,
            # then an optional backoff weight.
            logp = float(parts[0])
            words = tuple(parts[1 : 1 + current_order])
            model.logprob[words] = logp
            rest = parts[1 + current_order :]
            if rest:
                model.logbow[words] = float(rest[0])
            if current_order == 1:
                model.vocab.append(words[0])
    return model
