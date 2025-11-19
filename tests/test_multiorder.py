"""Tests for multi-order training functionality."""

from io import StringIO

import pytest

from arpabo.lm import ArpaBoLM


class TestComputeMultipleOrders:
    """Test multi-order training."""

    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for testing."""
        return StringIO(
            """the quick brown fox
the lazy dog
the brown dog runs
quick brown fox jumps"""
        )

    def test_basic_multi_order(self, sample_corpus):
        """Test basic multi-order training."""
        lm = ArpaBoLM(max_order=4, verbose=False)
        lm.read_corpus(sample_corpus)

        models = lm.compute_multiple_orders([1, 2, 3])

        assert len(models) == 3
        assert 1 in models
        assert 2 in models
        assert 3 in models

        assert models[1].max_order == 1
        assert models[2].max_order == 2
        assert models[3].max_order == 3

    def test_single_order(self, sample_corpus):
        """Test single order returns one model."""
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(sample_corpus)

        models = lm.compute_multiple_orders([2])

        assert len(models) == 1
        assert 2 in models
        assert models[2].max_order == 2

    def test_non_sequential_orders(self, sample_corpus):
        """Test non-sequential order list."""
        lm = ArpaBoLM(max_order=5, verbose=False)
        lm.read_corpus(sample_corpus)

        models = lm.compute_multiple_orders([1, 3, 5])

        assert len(models) == 3
        assert models[1].max_order == 1
        assert models[3].max_order == 3
        assert models[5].max_order == 5

    def test_order_exceeds_max(self, sample_corpus):
        """Test requesting order higher than max_order raises error."""
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(sample_corpus)

        with pytest.raises(ValueError, match="exceeds corpus max_order"):
            lm.compute_multiple_orders([1, 2, 4])

    def test_no_corpus_raises_error(self):
        """Test calling compute_multiple_orders without corpus raises error."""
        lm = ArpaBoLM(max_order=3, verbose=False)

        with pytest.raises(SystemExit):
            lm.compute_multiple_orders([1, 2, 3])

    def test_models_can_write(self, sample_corpus, tmp_path):
        """Test that generated models can be written to files."""
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(sample_corpus)

        models = lm.compute_multiple_orders([1, 2, 3])

        # Write each model
        for order, model in models.items():
            output_file = tmp_path / f"{order}gram.arpa"
            with open(output_file, "w") as f:
                model.write(f)

            # Verify file was created and has content
            assert output_file.exists()
            assert output_file.stat().st_size > 0

            # Verify it's valid ARPA format
            content = output_file.read_text()
            assert "\\data\\" in content
            assert f"ngram {order}=" in content
            assert "\\end\\" in content

    def test_different_smoothing_methods(self, sample_corpus):
        """Test multi-order training with different smoothing methods."""
        for method in ["good_turing", "kneser_ney"]:
            lm = ArpaBoLM(max_order=3, smoothing_method=method, verbose=False)
            lm.read_corpus(StringIO(sample_corpus.getvalue()))

            models = lm.compute_multiple_orders([1, 2, 3])

            assert len(models) == 3
            for model in models.values():
                assert model.smoothing_method == method

    def test_models_share_vocab(self, sample_corpus):
        """Test that models share the same vocabulary."""
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(sample_corpus)

        models = lm.compute_multiple_orders([1, 2, 3])

        # All models should have the same unigram counts
        vocab_1 = set(models[1].grams[0].keys())
        vocab_2 = set(models[2].grams[0].keys())
        vocab_3 = set(models[3].grams[0].keys())

        assert vocab_1 == vocab_2 == vocab_3

    def test_empty_orders_list(self, sample_corpus):
        """Test empty orders list raises error."""
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(sample_corpus)

        with pytest.raises(ValueError):
            lm.compute_multiple_orders([])

    def test_models_have_correct_sentence_counts(self, sample_corpus):
        """Test that all models have the same sentence count."""
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(sample_corpus)

        models = lm.compute_multiple_orders([1, 2, 3])

        # All models should have same sentence count
        sent_count = lm.sent_count
        for model in models.values():
            assert model.sent_count == sent_count
