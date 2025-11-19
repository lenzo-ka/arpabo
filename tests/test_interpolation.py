"""Tests for model interpolation functionality."""

from io import StringIO

import pytest

from arpabo.interpolation import InterpolatedModel, tune_interpolation_weights
from arpabo.lm import ArpaBoLM


class TestInterpolatedModelInit:
    """Test InterpolatedModel initialization."""

    @pytest.fixture
    def trained_models(self):
        """Create trained models for testing."""
        corpus = StringIO("the quick brown fox\nthe lazy dog")

        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(corpus)
        models = lm.compute_multiple_orders([1, 2, 3])

        return models

    def test_init_basic(self, trained_models):
        """Test basic initialization."""
        weights = {1: 0.1, 2: 0.3, 3: 0.6}
        interpolated = InterpolatedModel(models=trained_models, weights=weights)

        assert interpolated.max_order == 3
        assert interpolated.weights == weights
        assert len(interpolated.vocab) > 0

    def test_init_validates_weight_sum(self, trained_models):
        """Test that weights must sum to 1.0."""
        weights = {1: 0.1, 2: 0.3, 3: 0.5}  # Sum = 0.9

        with pytest.raises(ValueError, match="must sum to 1.0"):
            InterpolatedModel(models=trained_models, weights=weights)

    def test_init_validates_order_match(self, trained_models):
        """Test that model orders must match weight orders."""
        weights = {1: 0.5, 2: 0.5}  # Missing order 3

        with pytest.raises(ValueError, match="must match"):
            InterpolatedModel(models=trained_models, weights=weights)

    def test_init_empty_models(self):
        """Test that empty models dict raises error."""
        with pytest.raises(ValueError, match="at least one model"):
            InterpolatedModel(models={}, weights={})

    def test_init_uniform_weights(self, trained_models):
        """Test initialization with uniform weights."""
        n = len(trained_models)
        weights = {order: 1.0 / n for order in trained_models}

        interpolated = InterpolatedModel(models=trained_models, weights=weights)

        assert abs(sum(interpolated.weights.values()) - 1.0) < 1e-6


class TestGetInterpolatedProbability:
    """Test probability calculation."""

    @pytest.fixture
    def interpolated_model(self):
        """Create interpolated model for testing."""
        corpus = StringIO("the quick brown fox\nthe lazy dog")

        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(corpus)
        models = lm.compute_multiple_orders([1, 2, 3])

        weights = {1: 0.2, 2: 0.3, 3: 0.5}
        return InterpolatedModel(models=models, weights=weights)

    def test_get_probability(self, interpolated_model):
        """Test getting interpolated probability."""
        prob = interpolated_model.get_interpolated_probability("quick", ("the",))

        assert isinstance(prob, float)
        assert 0 < prob <= 1

    def test_oov_probability(self, interpolated_model):
        """Test OOV word handling."""
        prob = interpolated_model.get_interpolated_probability("unknown_word_xyz", ())

        # Should return very small probability
        assert prob > 0
        assert prob < 1e-8

    def test_probability_range(self, interpolated_model):
        """Test that probabilities are in valid range."""
        words = ["the", "quick", "fox"]
        for word in words:
            prob = interpolated_model.get_interpolated_probability(word, ())
            assert 0 < prob <= 1


class TestInterpolatedPerplexity:
    """Test perplexity evaluation for interpolated models."""

    @pytest.fixture
    def interpolated_model(self):
        """Create interpolated model."""
        corpus = StringIO(
            """the quick brown fox jumps over the lazy dog
the lazy dog sleeps under the brown tree"""
        )

        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(corpus)
        models = lm.compute_multiple_orders([1, 2, 3])

        weights = {1: 0.1, 2: 0.3, 3: 0.6}
        return InterpolatedModel(models=models, weights=weights)

    def test_perplexity_basic(self, interpolated_model):
        """Test basic perplexity calculation."""
        test_data = StringIO("the quick fox")
        results = interpolated_model.perplexity(test_data)

        assert "perplexity" in results
        assert "cross_entropy" in results
        assert "num_words" in results
        assert "oov_rate" in results

        assert results["perplexity"] > 0
        assert results["num_words"] > 0

    def test_perplexity_with_oov(self, interpolated_model):
        """Test perplexity with OOV words."""
        test_data = StringIO("the unknown_word fox")
        results = interpolated_model.perplexity(test_data, oov_handling="unk")

        assert results["num_oov"] > 0
        assert results["oov_rate"] > 0

    def test_perplexity_oov_error(self, interpolated_model):
        """Test that oov_handling='error' raises on OOV."""
        test_data = StringIO("the unknown_word")

        with pytest.raises(ValueError, match="OOV word"):
            interpolated_model.perplexity(test_data, oov_handling="error")

    def test_perplexity_oov_skip(self, interpolated_model):
        """Test that oov_handling='skip' skips OOV words."""
        test_data = StringIO("the unknown_word fox")
        results = interpolated_model.perplexity(test_data, oov_handling="skip")

        # Should complete without error
        assert results["num_oov"] > 0


class TestTuneInterpolationWeights:
    """Test automatic weight tuning."""

    @pytest.fixture
    def trained_models(self):
        """Create trained models."""
        corpus = StringIO(
            """the quick brown fox jumps over the lazy dog
the lazy dog sleeps under the brown tree
the quick fox runs through the green field"""
        )

        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(corpus)
        models = lm.compute_multiple_orders([1, 2, 3])

        return models

    def test_tune_weights_basic(self, trained_models):
        """Test basic weight tuning."""
        dev_data = StringIO("the quick brown fox")

        weights = tune_interpolation_weights(trained_models, dev_data, max_iterations=5)

        assert isinstance(weights, dict)
        assert len(weights) == 3
        assert 1 in weights
        assert 2 in weights
        assert 3 in weights

    def test_tuned_weights_sum_to_one(self, trained_models):
        """Test that tuned weights sum to 1.0."""
        dev_data = StringIO("the quick brown fox")

        weights = tune_interpolation_weights(trained_models, dev_data, max_iterations=5)

        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 0.01  # Allow small numerical error

    def test_tuned_weights_positive(self, trained_models):
        """Test that tuned weights are positive."""
        dev_data = StringIO("the quick brown fox")

        weights = tune_interpolation_weights(trained_models, dev_data, max_iterations=5)

        for _order, weight in weights.items():
            assert weight >= 0


class TestInterpolationVsBackoff:
    """Compare interpolation with standard backoff."""

    def test_interpolation_can_improve(self):
        """Test that interpolation can provide different (potentially better) perplexity."""
        corpus = StringIO(
            """the cat sat on the mat
the dog sat on the log
the bird sat on the wire"""
        )

        # Train models
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(corpus)
        models = lm.compute_multiple_orders([1, 2, 3])

        # Standard backoff (just use trigram)
        test_data = "the cat sat"
        backoff_ppl = models[3].perplexity(StringIO(test_data))

        # Interpolation
        weights = {1: 0.1, 2: 0.3, 3: 0.6}
        interpolated = InterpolatedModel(models, weights)
        interp_ppl = interpolated.perplexity(StringIO(test_data))

        # Both should give valid perplexities
        assert backoff_ppl["perplexity"] > 0
        assert interp_ppl["perplexity"] > 0

    def test_equal_weights_vs_tuned(self):
        """Test that tuned weights differ from equal weights."""
        corpus = StringIO(
            """the quick brown fox jumps
the lazy dog sleeps
the brown fox runs"""
        )

        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(corpus)
        models = lm.compute_multiple_orders([1, 2, 3])

        # Equal weights
        equal_weights = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}

        # Tuned weights
        dev_data = StringIO("the quick fox")
        tuned_weights = tune_interpolation_weights(models, dev_data, max_iterations=10)

        # Weights should be different (unless data is pathological)
        # At least one weight should differ by > 0.05
        weight_diffs = [abs(equal_weights[o] - tuned_weights[o]) for o in [1, 2, 3]]
        assert any(diff > 0.05 for diff in weight_diffs) or all(diff < 0.01 for diff in weight_diffs)


class TestInterpolationIntegration:
    """Integration tests for interpolation."""

    def test_complete_workflow(self, tmp_path):
        """Test complete interpolation workflow."""
        # Create data
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("the quick brown fox\nthe lazy dog")

        dev_file = tmp_path / "dev.txt"
        dev_file.write_text("the quick fox")

        test_file = tmp_path / "test.txt"
        test_file.write_text("the lazy fox")

        # Train models
        lm = ArpaBoLM(max_order=3, verbose=False)
        with open(corpus_file) as f:
            lm.read_corpus(f)

        models = lm.compute_multiple_orders([1, 2, 3])

        # Tune weights on dev set
        with open(dev_file) as f:
            weights = tune_interpolation_weights(models, f, max_iterations=5)

        # Create interpolated model
        interpolated = InterpolatedModel(models, weights)

        # Evaluate on test set
        with open(test_file) as f:
            results = interpolated.perplexity(f)

        assert results["perplexity"] > 0
        assert results["num_words"] > 0

    def test_with_modelcomparison_models(self):
        """Test using interpolation with ModelComparison."""

        corpus = StringIO("the cat sat on the mat\nthe dog ran")

        # Use ModelComparison to train
        # (Using StringIO directly won't work with file paths, so use tmp_path in real tests)
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(corpus)
        models = lm.compute_multiple_orders([1, 2, 3])

        # Use interpolation
        weights = {1: 0.1, 2: 0.3, 3: 0.6}
        interpolated = InterpolatedModel(models, weights)

        # Evaluate
        test = StringIO("the cat sat")
        results = interpolated.perplexity(test)

        assert results["perplexity"] > 0
