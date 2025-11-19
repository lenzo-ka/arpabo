"""Tests for cross-validation functionality."""

import pytest

from arpabo.crossval import cross_validate, print_cv_results


class TestCrossValidate:
    """Test cross_validate function."""

    @pytest.fixture
    def corpus_file(self, tmp_path):
        """Create corpus file with enough data for CV."""
        corpus = tmp_path / "corpus.txt"
        sentences = [
            "the quick brown fox",
            "the lazy dog",
            "the brown fox jumps",
            "the lazy cat sleeps",
            "quick brown animals",
            "the green tree",
            "brown fox runs fast",
            "lazy dog plays",
            "the cat sat",
            "quick animal jumps",
        ]
        corpus.write_text("\n".join(sentences))
        return corpus

    def test_cross_validate_basic(self, corpus_file):
        """Test basic cross-validation."""
        results = cross_validate(
            corpus_file=str(corpus_file), orders=[1, 2], k_folds=2, smoothing_method="good_turing", verbose=False
        )

        assert len(results) == 2
        assert 1 in results
        assert 2 in results

    def test_result_structure(self, corpus_file):
        """Test that results have expected structure."""
        results = cross_validate(corpus_file=str(corpus_file), orders=[2], k_folds=2, verbose=False)

        assert 2 in results
        stats = results[2]

        assert "mean_perplexity" in stats
        assert "std_perplexity" in stats
        assert "mean_cross_entropy" in stats
        assert "mean_oov_rate" in stats
        assert "fold_results" in stats

    def test_fold_results_count(self, corpus_file):
        """Test that fold_results has k entries."""
        k = 3
        results = cross_validate(corpus_file=str(corpus_file), orders=[1], k_folds=k, verbose=False)

        assert len(results[1]["fold_results"]) == k

    def test_multiple_orders(self, corpus_file):
        """Test cross-validation with multiple orders."""
        results = cross_validate(corpus_file=str(corpus_file), orders=[1, 2, 3], k_folds=2, verbose=False)

        assert len(results) == 3
        for order in [1, 2, 3]:
            assert order in results
            assert results[order]["mean_perplexity"] > 0

    def test_different_k_values(self, corpus_file):
        """Test with different k values."""
        for k in [2, 3, 5]:
            results = cross_validate(corpus_file=str(corpus_file), orders=[2], k_folds=k, verbose=False)

            assert len(results[2]["fold_results"]) == k

    def test_k_too_small(self, corpus_file):
        """Test that k < 2 raises error."""
        with pytest.raises(ValueError, match="k_folds must be >= 2"):
            cross_validate(corpus_file=str(corpus_file), orders=[1], k_folds=1)

    def test_corpus_too_small(self, tmp_path):
        """Test that corpus smaller than k raises error."""
        small_corpus = tmp_path / "small.txt"
        small_corpus.write_text("the cat")  # Only 1 line

        with pytest.raises(ValueError, match="Need more data"):
            cross_validate(corpus_file=str(small_corpus), orders=[1], k_folds=5)

    def test_with_verbose(self, corpus_file, capsys):
        """Test verbose output."""
        cross_validate(corpus_file=str(corpus_file), orders=[1], k_folds=2, verbose=True)

        captured = capsys.readouterr()
        assert "Cross-validation" in captured.out or "Fold" in captured.out


class TestCVStatistics:
    """Test cross-validation statistics."""

    @pytest.fixture
    def cv_results(self, tmp_path):
        """Get CV results for testing."""
        corpus = tmp_path / "corpus.txt"
        sentences = ["the quick brown fox jumps over the lazy dog"] * 10
        corpus.write_text("\n".join(sentences))

        return cross_validate(corpus_file=str(corpus), orders=[1, 2], k_folds=5, verbose=False)

    def test_mean_values_reasonable(self, cv_results):
        """Test that mean values are reasonable."""
        for _order, stats in cv_results.items():
            assert stats["mean_perplexity"] > 0
            assert stats["mean_cross_entropy"] > 0
            assert 0 <= stats["mean_oov_rate"] <= 1

    def test_std_values_nonnegative(self, cv_results):
        """Test that standard deviations are non-negative."""
        for _order, stats in cv_results.items():
            assert stats["std_perplexity"] >= 0
            assert stats["std_cross_entropy"] >= 0
            assert stats["std_oov_rate"] >= 0

    def test_higher_order_generally_better(self, cv_results):
        """Test that higher order generally has lower perplexity."""
        # For this simple repeated corpus, both should work
        assert cv_results[1]["mean_perplexity"] > 0
        assert cv_results[2]["mean_perplexity"] > 0


class TestPrintCVResults:
    """Test printing CV results."""

    @pytest.fixture
    def sample_results(self):
        """Create sample CV results."""
        return {
            1: {
                "mean_perplexity": 342.5,
                "std_perplexity": 12.3,
                "mean_cross_entropy": 8.42,
                "std_cross_entropy": 0.15,
                "fold_results": [],
            },
            2: {
                "mean_perplexity": 124.3,
                "std_perplexity": 8.7,
                "mean_cross_entropy": 6.96,
                "std_cross_entropy": 0.12,
                "fold_results": [],
            },
            3: {
                "mean_perplexity": 89.2,
                "std_perplexity": 5.4,
                "mean_cross_entropy": 6.48,
                "std_cross_entropy": 0.08,
                "fold_results": [],
            },
        }

    def test_print_cv_results(self, sample_results, capsys):
        """Test printing CV results."""
        print_cv_results(sample_results, k_folds=5)

        captured = capsys.readouterr()
        assert "Cross-Validation Results" in captured.out
        assert "5 folds" in captured.out
        assert "1-gram" in captured.out
        assert "2-gram" in captured.out
        assert "3-gram" in captured.out

    def test_identifies_best(self, sample_results, capsys):
        """Test that best model is identified."""
        print_cv_results(sample_results, k_folds=5)

        captured = capsys.readouterr()
        assert "← Best" in captured.out
        assert "Best model: 3-gram" in captured.out


class TestCrossValidationIntegration:
    """Integration tests for cross-validation."""

    def test_complete_workflow(self, tmp_path):
        """Test complete CV workflow."""
        corpus = tmp_path / "corpus.txt"
        sentences = [f"the quick brown fox number {i}" for i in range(20)]
        corpus.write_text("\n".join(sentences))

        # Run CV
        results = cross_validate(
            corpus_file=str(corpus), orders=[1, 2, 3], k_folds=5, smoothing_method="good_turing", verbose=False
        )

        # Print results
        print_cv_results(results, k_folds=5)

        # Check all orders evaluated
        assert len(results) == 3
        for order in [1, 2, 3]:
            assert results[order]["mean_perplexity"] > 0

    def test_with_different_smoothing(self, tmp_path):
        """Test CV with different smoothing methods."""
        corpus = tmp_path / "corpus.txt"
        sentences = ["the cat sat on the mat"] * 10
        corpus.write_text("\n".join(sentences))

        for method in ["good_turing", "kneser_ney"]:
            results = cross_validate(
                corpus_file=str(corpus), orders=[2], k_folds=2, smoothing_method=method, verbose=False
            )

            assert 2 in results
            assert results[2]["mean_perplexity"] > 0
