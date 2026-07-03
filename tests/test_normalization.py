"""Truth harness: every emitted ARPA model must be a proper probability model.

For a valid backoff LM, the conditional distribution for every context sums to
1 over the predictable vocabulary. Before the normalization fix, backoff weights
were hardcoded to 1.0 and these sums were 1.12 (GT), 1.25 (KN), 1.48 (Katz).
"""

import pytest

from arpabo import ArpaBoLM

from .arpa_eval import parse_arpa

CORPUS = """the cat sat on the mat
the dog sat on the log
a cat sat on a mat
the cat ran to the dog
the dog ran to the cat
a dog sat on the mat
the cat and the dog sat
the mat and the log
""".strip()

SMOOTHERS = ["good_turing", "kneser_ney", "fixed", "auto", "mle"]


def _build(tmp_path, smoothing, order):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS + "\n", encoding="utf-8")
    out = tmp_path / "model.arpa"
    lm = ArpaBoLM(max_order=order, smoothing_method=smoothing)
    with open(corpus, encoding="utf-8") as f:
        lm.read_corpus(f)
    lm.compute()
    lm.write_file(str(out))
    return parse_arpa(str(out))


def _contexts(model, order):
    """A representative set of contexts of length 0..order-1 drawn from the model."""
    seen = {(): True}
    contexts = [()]
    for ngram in model.logprob:
        for k in range(1, order):
            if len(ngram) >= k:
                ctx = ngram[:k]
                if ctx not in seen:
                    seen[ctx] = True
                    contexts.append(ctx)
    return contexts


# ARPA writes log10 probabilities with 4 decimals, so a re-parsed file can only
# be normalized to text precision; ~1e-3 still catches the pre-fix errors, which
# ranged from 0.02 to 0.48, with a 20x margin.
FILE_TOL = 1e-3


@pytest.mark.parametrize("smoothing", SMOOTHERS)
@pytest.mark.parametrize("order", [2, 3])
def test_context_distributions_sum_to_one(tmp_path, smoothing, order):
    model = _build(tmp_path, smoothing, order)
    contexts = _contexts(model, order)
    assert contexts, "model produced no contexts"
    for ctx in contexts:
        total = model.context_sum(ctx)
        assert total == pytest.approx(
            1.0, abs=FILE_TOL
        ), f"{smoothing} order={order} context={ctx!r} sums to {total:.6f}, not 1.0"


@pytest.mark.parametrize("smoothing", SMOOTHERS)
def test_unigram_distribution_sums_to_one(tmp_path, smoothing):
    model = _build(tmp_path, smoothing, 3)
    total = model.context_sum(())
    assert total == pytest.approx(1.0, abs=FILE_TOL), f"{smoothing} unigrams sum to {total:.6f}"


@pytest.mark.parametrize("smoothing", SMOOTHERS)
@pytest.mark.parametrize("order", [2, 3])
def test_internal_distributions_sum_to_one(tmp_path, smoothing, order):
    """The in-memory distribution must be exactly normalized (pre-rounding)."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS + "\n", encoding="utf-8")
    lm = ArpaBoLM(max_order=order, smoothing_method=smoothing)
    with open(corpus, encoding="utf-8") as f:
        lm.read_corpus(f)
    lm.compute()

    vocab = [w for w in lm.probs[0] if w != "<s>"]
    unigram_sum = sum(lm.probs[0][w] for w in vocab)
    assert unigram_sum == pytest.approx(1.0, abs=1e-9), f"{smoothing} internal unigrams sum to {unigram_sum}"


@pytest.mark.parametrize("smoothing", SMOOTHERS)
def test_start_symbol_not_predicted(tmp_path, smoothing):
    """<s> must never be a predictable event (ARPA convention: prob -99)."""
    model = _build(tmp_path, smoothing, 3)
    if ("<s>",) in model.logprob:
        assert model.logprob[("<s>",)] <= -90.0, "<s> unigram should carry the -99 sentinel"
