"""Property tests for the Kneser-Ney continuation-count estimator."""

from io import StringIO

from arpabo import ArpaBoLM

# "x" completes four distinct contexts; "z" only ever follows "y". Both words
# occur exactly four times, so any frequency-based estimator rates them equal --
# only continuation counts separate them.
CORPUS = "a x\nb x\nc x\nd x\ny z\ny z\ny z\ny z\n"


def _unigrams(smoothing):
    lm = ArpaBoLM(max_order=2, smoothing_method=smoothing, add_start=False)
    lm.read_corpus(StringIO(CORPUS))
    lm.compute()
    return lm


def test_continuation_counts_beat_raw_frequency():
    kn = _unigrams("kneser_ney")
    assert kn.grams[0]["x"] == kn.grams[0]["z"] == 4
    # Diverse-context word wins despite identical raw frequency.
    assert kn.probs[0]["x"] > 2 * kn.probs[0]["z"]


def test_mle_rates_equal_frequency_words_equally():
    mle = _unigrams("mle")
    assert mle.probs[0]["x"] == mle.probs[0]["z"]


def test_kneser_ney_leaves_backoff_mass():
    """Absolute discounting must leave every non-degenerate context some mass."""
    lm = ArpaBoLM(max_order=2, smoothing_method="kneser_ney")
    lm.read_corpus(StringIO("the cat sat\nthe dog ran\na cat ran\n"))
    lm.compute()
    # Bigram context "the" has two continuations (cat, dog); discounting them
    # must leave positive backoff mass, i.e. sum of seen probs < 1.
    ctx = lm.probs[1]["the"]
    assert 0 < sum(ctx.values()) < 1.0
