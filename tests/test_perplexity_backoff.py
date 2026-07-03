"""Perplexity must score the same model the package writes to disk.

Before the fix, perplexity() did longest-match on raw conditional probabilities
and never consulted backoff weights, so it scored neither the ARPA model nor a
normalized distribution. These tests pin it to the emitted file via the
independent backoff oracle.
"""

import math

import pytest

from arpabo import ArpaBoLM

from .arpa_eval import parse_arpa

CORPUS = """the cat sat on the mat
the dog sat on the log
a cat sat on a mat
the cat ran to the dog
a dog sat on the mat
""".strip()

TEST_SENTENCE = "the cat sat on the log"


def _oracle_perplexity(model, sentence, order):
    """Perplexity of a sentence under the written ARPA file, excluding <s>."""
    words = ["<s>", *sentence.split(), "</s>"]
    total_log = 0.0
    n = 0
    for i, w in enumerate(words):
        if w == "<s>":
            continue
        start = max(0, i - (order - 1))
        history = tuple(words[start:i])
        p = model.prob(history, w)
        total_log += math.log10(p)
        n += 1
    return math.pow(10, -total_log / n)


@pytest.mark.parametrize("smoothing", ["good_turing", "kneser_ney", "fixed"])
@pytest.mark.parametrize("order", [2, 3])
def test_perplexity_matches_emitted_model(tmp_path, smoothing, order):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS + "\n", encoding="utf-8")
    test = tmp_path / "test.txt"
    test.write_text(TEST_SENTENCE + "\n", encoding="utf-8")
    arpa = tmp_path / "model.arpa"

    lm = ArpaBoLM(max_order=order, smoothing_method=smoothing)
    with open(corpus, encoding="utf-8") as f:
        lm.read_corpus(f)
    lm.compute()
    lm.write_file(str(arpa))

    with open(test, encoding="utf-8") as f:
        internal = lm.perplexity(f, oov_handling="unk")["perplexity"]

    oracle = _oracle_perplexity(parse_arpa(str(arpa)), TEST_SENTENCE, order)

    # Agreement to ARPA text precision confirms perplexity uses backoff weights.
    assert internal == pytest.approx(oracle, rel=1e-2)


def test_start_symbol_excluded_from_word_count(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS + "\n", encoding="utf-8")
    test = tmp_path / "test.txt"
    test.write_text("the cat sat\n", encoding="utf-8")

    lm = ArpaBoLM(max_order=3, smoothing_method="kneser_ney")
    with open(corpus, encoding="utf-8") as f:
        lm.read_corpus(f)
    lm.compute()
    with open(test, encoding="utf-8") as f:
        results = lm.perplexity(f)

    # "the cat sat" -> scored words are the, cat, sat, </s> (not <s>).
    assert results["num_words"] == 4
