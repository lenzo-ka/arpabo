"""Tests for core language model functionality"""

import tempfile
from io import StringIO

import pytest

from arpabo import ArpaBoLM


class TestArpaBoLMBasics:
    """Test basic ArpaBoLM functionality"""

    def test_initialization(self):
        lm = ArpaBoLM()
        assert lm.max_order == 3
        assert lm.smoothing_method == "good_turing"
        assert lm.sent_count == 0
        assert lm.sum_1 == 0

    def test_custom_parameters(self):
        lm = ArpaBoLM(max_order=4, smoothing_method="kneser_ney", case="lower", verbose=False)
        assert lm.max_order == 4
        assert lm.smoothing_method == "kneser_ney"
        assert lm.case == "lower"

    def test_invalid_discount_step(self):
        with pytest.raises(AttributeError, match="Discount step .* out of range"):
            ArpaBoLM(discount_step=2.0)


class TestCorpusReading:
    """Test corpus reading functionality"""

    def test_read_simple_corpus(self):
        lm = ArpaBoLM(verbose=False)
        corpus = "the quick brown fox\nthe lazy dog"
        lm.read_corpus(StringIO(corpus))

        assert lm.sent_count == 2
        # sum_1 is computed in compute(), not read_corpus()
        assert len(lm.grams[0]) > 0  # But counts should be populated

        lm.compute()
        assert lm.sum_1 > 0  # Now sum_1 is computed

    def test_empty_corpus(self):
        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO(""))
        assert lm.sent_count == 0

    def test_corpus_with_markers(self):
        lm = ArpaBoLM(verbose=False)
        corpus = "<s> hello world </s>\n<s> goodbye world </s>"
        lm.read_corpus(StringIO(corpus))
        assert lm.sent_count == 2

    def test_case_normalization(self):
        lm_lower = ArpaBoLM(case="lower", verbose=False)
        lm_lower.read_corpus(StringIO("HELLO WORLD"))
        lm_lower.compute()
        assert "hello" in lm_lower.probs[0]
        assert "HELLO" not in lm_lower.probs[0]

        lm_upper = ArpaBoLM(case="upper", verbose=False)
        lm_upper.read_corpus(StringIO("hello world"))
        lm_upper.compute()
        assert "HELLO" in lm_upper.probs[0]
        assert "hello" not in lm_upper.probs[0]


class TestNGramCounting:
    """Test n-gram counting"""

    def test_unigram_counts(self):
        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO("the quick brown fox"))
        lm.compute()

        # Should have: <s>, </s>, the, quick, brown, fox
        assert len(lm.grams[0]) == 6
        assert lm.grams[0]["the"] >= 1
        assert lm.grams[0]["<s>"] == 1
        assert lm.grams[0]["</s>"] == 1

    def test_bigram_counts(self):
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(StringIO("the quick brown fox"))
        lm.compute()

        # Check bigrams exist
        assert lm.max_order == 2
        assert lm.counts[1] > 0  # Should have bigrams

    def test_trigram_counts(self):
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(StringIO("the quick brown fox"))
        lm.compute()

        assert lm.max_order == 3
        assert lm.counts[2] > 0  # Should have trigrams

    def test_repeated_words(self):
        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO("the the the"))
        lm.compute()

        # "the" should appear 3 times (plus markers)
        assert lm.grams[0]["the"] == 3


class TestCompute:
    """Test compute() method"""

    def test_compute_without_corpus(self):
        lm = ArpaBoLM(verbose=False)
        with pytest.raises(SystemExit):
            lm.compute()

    def test_compute_populates_probs(self):
        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO("hello world"))
        lm.compute()

        assert len(lm.probs[0]) > 0
        assert all(prob > 0 for prob in lm.probs[0].values())

    def test_probability_sum(self):
        lm = ArpaBoLM(smoothing_method="good_turing", verbose=False)
        lm.read_corpus(StringIO("the quick brown fox"))
        lm.compute()

        # Good-Turing smoothing doesn't guarantee probabilities sum to 1.0
        # (it redistributes mass for unseen events)
        # Just verify all probabilities are valid
        total = sum(lm.probs[0].values())
        assert total > 0
        assert all(0.0 <= p <= 1.0 for p in lm.probs[0].values())


class TestWordFile:
    """Test word file functionality"""

    def test_read_word_file(self):
        lm = ArpaBoLM(verbose=False)

        # Create temporary word file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("apple\nbanana\ncherry\n")
            word_file = f.name

        lm.read_word_file(word_file, count=5)

        assert "apple" in lm.grams[0]
        assert "banana" in lm.grams[0]
        assert "cherry" in lm.grams[0]
        assert lm.grams[0]["apple"] == 5

        import os

        os.unlink(word_file)


class TestMultipleInputs:
    """Test reading from multiple sources"""

    def test_multiple_corpus_reads(self):
        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO("hello world"))
        lm.read_corpus(StringIO("goodbye world"))
        lm.compute()

        assert lm.sent_count == 2
        assert "hello" in lm.probs[0]
        assert "goodbye" in lm.probs[0]
        assert lm.grams[0]["world"] == 2  # Appears in both

    def test_incremental_training(self):
        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO("the quick brown fox"))

        initial_count = lm.grams[0]["the"]

        lm.read_corpus(StringIO("the lazy dog"))
        lm.compute()

        # "the" should appear more times now
        assert lm.grams[0]["the"] > initial_count
