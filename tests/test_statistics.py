"""Tests for model statistics functionality."""

from io import StringIO

import pytest

from arpabo.lm import ArpaBoLM


class TestGetStatistics:
    """Test get_statistics method."""

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

    def test_statistics_structure(self, trained_model):
        """Test that statistics returns expected structure."""
        stats = trained_model.get_statistics()

        assert isinstance(stats, dict)
        assert "order" in stats
        assert "smoothing" in stats
        assert "vocab_size" in stats
        assert "ngram_counts" in stats
        assert "training_corpus" in stats

    def test_statistics_values(self, trained_model):
        """Test that statistics values are reasonable."""
        stats = trained_model.get_statistics()

        assert stats["order"] == 3
        assert stats["smoothing"] == "good_turing"
        assert stats["vocab_size"] > 0
        assert isinstance(stats["ngram_counts"], dict)
        assert isinstance(stats["training_corpus"], dict)

    def test_ngram_counts(self, trained_model):
        """Test that n-gram counts are present for all orders."""
        stats = trained_model.get_statistics()

        # Should have counts for 1-gram, 2-gram, 3-gram
        assert 1 in stats["ngram_counts"]
        assert 2 in stats["ngram_counts"]
        assert 3 in stats["ngram_counts"]

        # Counts should decrease with order (generally)
        assert stats["ngram_counts"][1] > 0
        assert stats["ngram_counts"][2] > 0
        assert stats["ngram_counts"][3] > 0

    def test_training_corpus_stats(self, trained_model):
        """Test training corpus statistics."""
        stats = trained_model.get_statistics()

        assert "sentences" in stats["training_corpus"]
        assert "tokens" in stats["training_corpus"]
        assert stats["training_corpus"]["sentences"] == 4
        assert stats["training_corpus"]["tokens"] > 0

    def test_model_not_computed(self):
        """Test that statistics fails if model not computed."""
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(StringIO("the quick brown fox"))
        # Don't call compute()

        with pytest.raises(ValueError, match="not been computed"):
            lm.get_statistics()

    def test_different_orders(self):
        """Test statistics for different model orders."""
        corpus = StringIO("the quick brown fox jumps over the lazy dog")

        for order in [1, 2, 3, 4]:
            lm = ArpaBoLM(max_order=order, verbose=False)
            lm.read_corpus(StringIO(corpus.getvalue()))
            lm.compute()

            stats = lm.get_statistics()
            assert stats["order"] == order
            assert len(stats["ngram_counts"]) == order


class TestBackoffRate:
    """Test backoff_rate method."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        corpus = StringIO(
            """the quick brown fox jumps over the lazy dog
the lazy dog sleeps under the brown tree
the quick fox runs through the green field"""
        )
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()
        return lm

    def test_backoff_structure(self, trained_model):
        """Test that backoff_rate returns expected structure."""
        test_data = StringIO("the quick fox")
        results = trained_model.backoff_rate(test_data)

        assert isinstance(results, dict)
        assert "overall_backoff_rate" in results
        assert "order_usage" in results
        assert "total_queries" in results

    def test_backoff_values(self, trained_model):
        """Test that backoff values are reasonable."""
        test_data = StringIO("the quick brown fox")
        results = trained_model.backoff_rate(test_data)

        assert 0 <= results["overall_backoff_rate"] <= 1
        assert results["total_queries"] > 0
        assert isinstance(results["order_usage"], dict)

    def test_order_usage_sums_to_one(self, trained_model):
        """Test that order usage fractions sum to 1."""
        test_data = StringIO("the quick brown fox")
        results = trained_model.backoff_rate(test_data)

        total_usage = sum(results["order_usage"].values())
        assert abs(total_usage - 1.0) < 0.01  # Allow small floating point error

    def test_order_usage_keys(self, trained_model):
        """Test that order_usage has correct keys."""
        test_data = StringIO("the quick brown fox")
        results = trained_model.backoff_rate(test_data)

        # Should have keys for 1, 2, 3 (model is 3-gram)
        assert 1 in results["order_usage"]
        assert 2 in results["order_usage"]
        assert 3 in results["order_usage"]

    def test_empty_test_corpus(self, trained_model):
        """Test backoff_rate with empty corpus."""
        results = trained_model.backoff_rate(StringIO(""))

        assert results["overall_backoff_rate"] == 0.0
        assert results["total_queries"] == 0

    def test_oov_words_skipped(self, trained_model):
        """Test that OOV words are skipped in backoff analysis."""
        # Include some OOV words
        test_data = StringIO("the unknown word")
        results = trained_model.backoff_rate(test_data)

        # Should only count "the", not "unknown" or "word"
        assert results["total_queries"] > 0

    def test_model_not_computed(self):
        """Test that backoff_rate fails if model not computed."""
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(StringIO("the quick brown fox"))
        # Don't call compute()

        with pytest.raises(ValueError, match="not been computed"):
            lm.backoff_rate(StringIO("test"))

    def test_backoff_different_orders(self):
        """Test backoff analysis for different model orders."""
        corpus_text = "the quick brown fox jumps over the lazy dog"
        test_data = "the quick fox"

        for order in [2, 3, 4]:
            lm = ArpaBoLM(max_order=order, verbose=False)
            lm.read_corpus(StringIO(corpus_text))
            lm.compute()

            results = lm.backoff_rate(StringIO(test_data))

            # Should have usage stats for all orders up to max_order
            assert len(results["order_usage"]) == order
            assert all(1 <= k <= order for k in results["order_usage"])


class TestPrintStatistics:
    """Test print_statistics method."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        corpus = StringIO("the quick brown fox\nthe lazy dog")
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()
        return lm

    def test_print_basic_statistics(self, trained_model, capsys):
        """Test printing basic statistics without backoff."""
        trained_model.print_statistics()

        captured = capsys.readouterr()
        assert "Model Statistics" in captured.out
        assert "Order:" in captured.out
        assert "Smoothing:" in captured.out
        assert "Vocabulary:" in captured.out
        assert "N-gram counts:" in captured.out
        assert "Training corpus:" in captured.out

    def test_print_with_backoff(self, trained_model, capsys, tmp_path):
        """Test printing statistics with backoff analysis."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("the quick fox")

        trained_model.print_statistics(test_file=str(test_file))

        captured = capsys.readouterr()
        assert "Model Statistics" in captured.out
        assert "Backoff analysis" in captured.out
        assert "Overall backoff rate:" in captured.out
        assert "Query resolution:" in captured.out


class TestGetOrderUsed:
    """Test _get_order_used helper method."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        corpus = StringIO("the quick brown fox")
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()
        return lm

    def test_get_order_used(self, trained_model):
        """Test that _get_order_used returns valid orders."""
        words = ["the", "quick", "brown", "fox"]

        for i, word in enumerate(words):
            order = trained_model._get_order_used(word, words, i)
            assert 1 <= order <= trained_model.max_order

    def test_unigram_fallback(self, trained_model):
        """Test that unknown n-grams fall back to unigram."""
        # These words exist as unigrams but not in this combination
        words = ["fox", "quick", "the"]

        for i, word in enumerate(words):
            order = trained_model._get_order_used(word, words, i)
            # Should return valid order (may be 1 if n-gram not found)
            assert 1 <= order <= trained_model.max_order


class TestIntegration:
    """Integration tests for statistics functionality."""

    def test_complete_workflow(self, tmp_path):
        """Test complete statistics workflow."""
        # Create corpus and test files
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("the quick brown fox\nthe lazy dog")

        test_file = tmp_path / "test.txt"
        test_file.write_text("the quick fox")

        # Train model
        lm = ArpaBoLM(max_order=3, verbose=False)
        with open(corpus_file) as f:
            lm.read_corpus(f)
        lm.compute()

        # Get statistics
        stats = lm.get_statistics()
        assert stats["order"] == 3
        assert stats["vocab_size"] > 0

        # Get backoff rate
        with open(test_file) as f:
            backoff = lm.backoff_rate(f)
        assert 0 <= backoff["overall_backoff_rate"] <= 1

    def test_multi_order_statistics(self):
        """Test statistics for multiple orders."""
        corpus = StringIO("the quick brown fox jumps over the lazy dog")

        lm = ArpaBoLM(max_order=4, verbose=False)
        lm.read_corpus(corpus)
        models = lm.compute_multiple_orders([1, 2, 3, 4])

        # Check statistics for each order
        for order, model in models.items():
            stats = model.get_statistics()
            assert stats["order"] == order
            assert len(stats["ngram_counts"]) == order
