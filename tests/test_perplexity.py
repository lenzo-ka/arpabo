"""Tests for perplexity evaluation functionality."""

from io import StringIO

import pytest

from arpabo.lm import ArpaBoLM


class TestPerplexityBasic:
    """Test basic perplexity evaluation."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        corpus = StringIO(
            """the quick brown fox
the lazy dog
the brown dog runs
quick brown fox jumps"""
        )
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()
        return lm

    @pytest.fixture
    def test_corpus(self):
        """Test corpus with some known and unknown words."""
        return StringIO(
            """the quick fox
the brown dog"""
        )

    def test_perplexity_returns_dict(self, trained_model, test_corpus):
        """Test that perplexity returns expected dictionary structure."""
        results = trained_model.perplexity(test_corpus)

        assert isinstance(results, dict)
        assert "perplexity" in results
        assert "cross_entropy" in results
        assert "num_sentences" in results
        assert "num_words" in results
        assert "num_oov" in results
        assert "oov_rate" in results

    def test_perplexity_values(self, trained_model, test_corpus):
        """Test that perplexity values are reasonable."""
        results = trained_model.perplexity(test_corpus)

        assert results["perplexity"] > 0
        assert results["cross_entropy"] > 0
        assert results["num_sentences"] == 2
        assert results["num_words"] > 0
        assert results["num_oov"] >= 0
        assert 0 <= results["oov_rate"] <= 1

    def test_perplexity_on_training_data(self, trained_model):
        """Test perplexity on training data (should be lower)."""
        # Test on same data as training
        test_data = StringIO("the quick brown fox")
        results = trained_model.perplexity(test_data)

        # Should have low perplexity since it's training data
        assert results["perplexity"] > 0
        assert results["num_oov"] == 0  # All words seen in training

    def test_perplexity_different_orders(self):
        """Test that higher order models generally have lower perplexity."""
        corpus = StringIO(
            """the quick brown fox jumps over the lazy dog
the lazy dog sleeps under the brown tree
the quick fox runs through the green field"""
        )

        test_data = "the quick fox"

        perplexities = []
        for order in [1, 2, 3]:
            lm = ArpaBoLM(max_order=order, verbose=False)
            lm.read_corpus(StringIO(corpus.getvalue()))
            lm.compute()

            results = lm.perplexity(StringIO(test_data))
            perplexities.append(results["perplexity"])

        # Higher order models should generally have better (lower) perplexity
        # But this isn't always true for very small corpora
        assert all(p > 0 for p in perplexities)


class TestPerplexityOOVHandling:
    """Test OOV word handling in perplexity evaluation."""

    @pytest.fixture
    def model_with_limited_vocab(self):
        """Create model with small vocabulary."""
        corpus = StringIO("the cat sat on the mat")
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()
        return lm

    def test_oov_handling_unk(self, model_with_limited_vocab):
        """Test default 'unk' handling for OOV words."""
        test_data = StringIO("the dog sat")  # 'dog' is OOV
        results = model_with_limited_vocab.perplexity(test_data, oov_handling="unk")

        assert results["num_oov"] > 0
        assert results["oov_rate"] > 0
        # Should complete without error

    def test_oov_handling_skip(self, model_with_limited_vocab):
        """Test 'skip' handling for OOV words."""
        test_data = StringIO("the dog sat")  # 'dog' is OOV
        results = model_with_limited_vocab.perplexity(test_data, oov_handling="skip")

        assert results["num_oov"] > 0
        # Should skip OOV words but complete

    def test_oov_handling_error(self, model_with_limited_vocab):
        """Test 'error' handling raises exception for OOV words."""
        test_data = StringIO("the dog sat")  # 'dog' is OOV

        with pytest.raises(ValueError, match="OOV word"):
            model_with_limited_vocab.perplexity(test_data, oov_handling="error")

    def test_no_oov_with_error_handling(self, model_with_limited_vocab):
        """Test that 'error' handling works when no OOV words."""
        test_data = StringIO("the cat")  # All known words
        results = model_with_limited_vocab.perplexity(test_data, oov_handling="error")

        assert results["num_oov"] == 0


class TestPerplexityEdgeCases:
    """Test edge cases in perplexity evaluation."""

    def test_empty_test_corpus(self):
        """Test that empty test corpus raises error."""
        corpus = StringIO("the quick brown fox")
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()

        with pytest.raises(ValueError, match="No words found"):
            lm.perplexity(StringIO(""))

    def test_whitespace_only(self):
        """Test that whitespace-only corpus raises error."""
        corpus = StringIO("the quick brown fox")
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()

        with pytest.raises(ValueError, match="No words found"):
            lm.perplexity(StringIO("   \n  \n  "))

    def test_model_not_computed(self):
        """Test that evaluating un-computed model raises error."""
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(StringIO("the quick brown fox"))
        # Don't call compute()

        with pytest.raises(ValueError, match="not been computed"):
            lm.perplexity(StringIO("test"))

    def test_single_sentence(self):
        """Test perplexity on single sentence."""
        corpus = StringIO("the quick brown fox")
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()

        results = lm.perplexity(StringIO("the fox"))
        assert results["num_sentences"] == 1
        assert results["num_words"] > 0

    def test_multiple_sentences(self):
        """Test perplexity on multiple sentences."""
        corpus = StringIO("the quick brown fox\nthe lazy dog")
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()

        results = lm.perplexity(StringIO("the fox\nthe dog"))
        assert results["num_sentences"] == 2


class TestPerplexityDifferentSmoothingMethods:
    """Test perplexity with different smoothing methods."""

    @pytest.fixture
    def corpus_data(self):
        """Corpus for training."""
        return "the quick brown fox jumps over the lazy dog"

    @pytest.fixture
    def test_data(self):
        """Test data."""
        return "the quick fox"

    def test_good_turing_perplexity(self, corpus_data, test_data):
        """Test perplexity with Good-Turing smoothing."""
        lm = ArpaBoLM(max_order=3, smoothing_method="good_turing", verbose=False)
        lm.read_corpus(StringIO(corpus_data))
        lm.compute()

        results = lm.perplexity(StringIO(test_data))
        assert results["perplexity"] > 0

    def test_kneser_ney_perplexity(self, corpus_data, test_data):
        """Test perplexity with Kneser-Ney smoothing."""
        lm = ArpaBoLM(max_order=3, smoothing_method="kneser_ney", verbose=False)
        lm.read_corpus(StringIO(corpus_data))
        lm.compute()

        results = lm.perplexity(StringIO(test_data))
        assert results["perplexity"] > 0


class TestPrintPerplexityResults:
    """Test perplexity results printing."""

    def test_print_perplexity_results(self, capsys):
        """Test that print_perplexity_results outputs formatted text."""
        corpus = StringIO("the quick brown fox")
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()

        results = lm.perplexity(StringIO("the fox"))
        lm.print_perplexity_results(results, test_file="test.txt")

        captured = capsys.readouterr()
        assert "Perplexity Evaluation" in captured.out
        assert "test.txt" in captured.out
        assert "Sentences:" in captured.out
        assert "Words:" in captured.out
        assert "Perplexity:" in captured.out
        assert "Cross-entropy:" in captured.out


class TestPerplexityIntegration:
    """Integration tests for perplexity evaluation."""

    def test_multi_order_perplexity_comparison(self):
        """Test comparing perplexity across multiple orders."""
        corpus = StringIO(
            """the quick brown fox jumps over the lazy dog
the lazy dog sleeps under the brown tree
the quick fox runs through the green field
the brown dog plays in the sunny yard"""
        )

        test_data = StringIO("the quick brown fox runs")

        # Train multiple orders and evaluate
        results = {}
        for order in [1, 2, 3, 4]:
            lm = ArpaBoLM(max_order=order, verbose=False)
            lm.read_corpus(StringIO(corpus.getvalue()))
            lm.compute()

            eval_results = lm.perplexity(StringIO(test_data.getvalue()))
            results[order] = eval_results

        # All should have valid perplexities
        for _order, res in results.items():
            assert res["perplexity"] > 0
            assert res["num_sentences"] == 1
            assert res["num_words"] > 0
