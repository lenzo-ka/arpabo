"""Tests for smoothing method comparison functionality."""

import pytest

from arpabo.comparison import compare_smoothing_methods, print_smoothing_comparison


class TestCompareSmoothingMethods:
    """Test compare_smoothing_methods function."""

    @pytest.fixture
    def data_files(self, tmp_path):
        """Create corpus and test files."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            """the quick brown fox jumps over the lazy dog
the lazy dog sleeps under the brown tree
the quick fox runs through the green field"""
        )

        test = tmp_path / "test.txt"
        test.write_text("the quick brown fox")

        return corpus, test

    def test_compare_default_methods(self, data_files):
        """Test comparing default smoothing methods."""
        corpus, test = data_files

        results = compare_smoothing_methods(corpus_file=str(corpus), test_file=str(test), methods=None, max_order=2)

        # Should compare default methods (good_turing, kneser_ney, auto)
        assert len(results) == 3
        assert "good_turing" in results
        assert "kneser_ney" in results
        assert "auto" in results

    def test_compare_specific_methods(self, data_files):
        """Test comparing specific methods."""
        corpus, test = data_files

        results = compare_smoothing_methods(
            corpus_file=str(corpus), test_file=str(test), methods=["good_turing", "kneser_ney"], max_order=2
        )

        assert len(results) == 2
        assert "good_turing" in results
        assert "kneser_ney" in results

    def test_result_structure(self, data_files):
        """Test that results have expected structure."""
        corpus, test = data_files

        results = compare_smoothing_methods(
            corpus_file=str(corpus), test_file=str(test), methods=["good_turing"], max_order=2
        )

        assert "good_turing" in results
        metrics = results["good_turing"]

        # Check required keys
        assert "perplexity" in metrics
        assert "cross_entropy" in metrics
        assert "num_words" in metrics
        assert "oov_rate" in metrics
        assert "training_time_seconds" in metrics

    def test_perplexity_values(self, data_files):
        """Test that perplexity values are reasonable."""
        corpus, test = data_files

        results = compare_smoothing_methods(
            corpus_file=str(corpus), test_file=str(test), methods=["good_turing", "kneser_ney"], max_order=2
        )

        for _method, metrics in results.items():
            assert metrics["perplexity"] > 0
            assert metrics["cross_entropy"] > 0
            assert metrics["training_time_seconds"] > 0

    def test_different_orders(self, data_files):
        """Test comparing at different n-gram orders."""
        corpus, test = data_files

        for order in [1, 2, 3]:
            results = compare_smoothing_methods(
                corpus_file=str(corpus), test_file=str(test), methods=["good_turing"], max_order=order
            )

            assert "good_turing" in results
            assert results["good_turing"]["perplexity"] > 0

    def test_with_verbose(self, data_files, capsys):
        """Test verbose output."""
        corpus, test = data_files

        compare_smoothing_methods(
            corpus_file=str(corpus), test_file=str(test), methods=["good_turing"], max_order=2, verbose=True
        )

        captured = capsys.readouterr()
        assert "Training" in captured.out or "Comparing" in captured.out

    def test_single_method(self, data_files):
        """Test comparing single method (useful for timing)."""
        corpus, test = data_files

        results = compare_smoothing_methods(
            corpus_file=str(corpus), test_file=str(test), methods=["kneser_ney"], max_order=3
        )

        assert len(results) == 1
        assert "kneser_ney" in results


class TestPrintSmoothingComparison:
    """Test print_smoothing_comparison function."""

    @pytest.fixture
    def sample_results(self):
        """Create sample comparison results."""
        return {
            "good_turing": {
                "perplexity": 92.4,
                "cross_entropy": 6.53,
                "num_words": 1000,
                "oov_rate": 0.012,
                "training_time_seconds": 1.2,
            },
            "kneser_ney": {
                "perplexity": 89.2,
                "cross_entropy": 6.48,
                "num_words": 1000,
                "oov_rate": 0.012,
                "training_time_seconds": 1.8,
            },
            "auto": {
                "perplexity": 95.1,
                "cross_entropy": 6.57,
                "num_words": 1000,
                "oov_rate": 0.012,
                "training_time_seconds": 1.4,
            },
        }

    def test_print_comparison(self, sample_results, capsys):
        """Test printing smoothing comparison."""
        print_smoothing_comparison(sample_results, test_file="test.txt", max_order=3)

        captured = capsys.readouterr()
        assert "Smoothing Method Comparison" in captured.out
        assert "Order: 3" in captured.out
        assert "test.txt" in captured.out
        assert "good_turing" in captured.out
        assert "kneser_ney" in captured.out
        assert "auto" in captured.out

    def test_identifies_best_method(self, sample_results, capsys):
        """Test that best method is identified."""
        print_smoothing_comparison(sample_results, test_file="test.txt", max_order=3)

        captured = capsys.readouterr()
        # kneser_ney has best (lowest) perplexity
        assert "Best method: kneser_ney" in captured.out
        assert "← Best" in captured.out

    def test_includes_timing(self, sample_results, capsys):
        """Test that timing information is included."""
        print_smoothing_comparison(sample_results, test_file="test.txt", max_order=3)

        captured = capsys.readouterr()
        assert "Time(s)" in captured.out
        # Should show timing values
        assert "1.2" in captured.out or "1.8" in captured.out


class TestSmoothingComparisonIntegration:
    """Integration tests for smoothing comparison."""

    def test_compare_all_available_methods(self, tmp_path):
        """Test comparing all available smoothing methods."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the quick brown fox\nthe lazy dog")

        test = tmp_path / "test.txt"
        test.write_text("the fox")

        results = compare_smoothing_methods(corpus_file=str(corpus), test_file=str(test), methods=None, max_order=2)

        # Should have results for multiple methods
        assert len(results) >= 2

        # All should have valid metrics
        for _method, metrics in results.items():
            assert metrics["perplexity"] > 0
            assert 0 <= metrics["oov_rate"] <= 1
            assert metrics["training_time_seconds"] >= 0

    def test_kneser_ney_vs_good_turing(self, tmp_path):
        """Test comparing Kneser-Ney vs Good-Turing."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            """the cat sat on the mat
the dog sat on the log
the bird sat on the wire"""
        )

        test = tmp_path / "test.txt"
        test.write_text("the cat sat")

        results = compare_smoothing_methods(
            corpus_file=str(corpus), test_file=str(test), methods=["good_turing", "kneser_ney"], max_order=3
        )

        # Both should complete successfully
        assert "good_turing" in results
        assert "kneser_ney" in results

        # Should have different perplexities (methods differ)
        gt_ppl = results["good_turing"]["perplexity"]
        kn_ppl = results["kneser_ney"]["perplexity"]

        # Both should be positive and reasonable
        assert gt_ppl > 0
        assert kn_ppl > 0

    def test_with_larger_corpus(self, tmp_path):
        """Test smoothing comparison with larger corpus."""
        # Create larger corpus
        corpus = tmp_path / "corpus.txt"
        sentences = []
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        for i in range(50):
            sentence = " ".join(words[i % len(words) :] + words[: i % len(words)])
            sentences.append(sentence)

        corpus.write_text("\n".join(sentences))

        test = tmp_path / "test.txt"
        test.write_text("the quick fox")

        results = compare_smoothing_methods(
            corpus_file=str(corpus), test_file=str(test), methods=["good_turing", "kneser_ney"], max_order=3
        )

        # Both should work
        assert len(results) == 2

    def test_complete_workflow(self, tmp_path, capsys):
        """Test complete workflow: compare → print → analyze."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the quick brown fox\nthe lazy dog")

        test = tmp_path / "test.txt"
        test.write_text("the quick fox")

        # Compare
        results = compare_smoothing_methods(
            corpus_file=str(corpus), test_file=str(test), methods=["good_turing", "kneser_ney"], max_order=2
        )

        # Print
        print_smoothing_comparison(results, test_file=str(test), max_order=2)

        # Verify output
        captured = capsys.readouterr()
        assert "Smoothing Method Comparison" in captured.out

        # Find best method
        best_method = min(results.keys(), key=lambda m: results[m]["perplexity"])
        assert best_method in ["good_turing", "kneser_ney"]
