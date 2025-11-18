"""Tests for smoothing methods"""

from io import StringIO

import pytest

from arpabo import ArpaBoLM, GoodTuringSmoother, KatzBackoffSmoother, KneserNeySmoother, create_smoother


class TestSmoothingFactory:
    """Test smoother factory function"""

    def test_create_good_turing(self):
        smoother = create_smoother("good_turing", max_order=3)
        assert isinstance(smoother, GoodTuringSmoother)
        assert smoother.max_order == 3

    def test_create_kneser_ney(self):
        smoother = create_smoother("kneser_ney", max_order=4)
        assert isinstance(smoother, KneserNeySmoother)
        assert smoother.max_order == 4

    def test_create_katz_fixed(self):
        smoother = create_smoother("fixed", max_order=3, discount_mass=0.5)
        assert isinstance(smoother, KatzBackoffSmoother)
        assert smoother.discount_mass == 0.5
        assert not smoother.auto_optimize

    def test_create_katz_auto(self):
        smoother = create_smoother("auto", max_order=3)
        assert isinstance(smoother, KatzBackoffSmoother)
        assert smoother.auto_optimize

    def test_create_mle(self):
        smoother = create_smoother("mle", max_order=3)
        assert isinstance(smoother, KatzBackoffSmoother)
        assert smoother.discount_mass == 0.0

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            create_smoother("invalid", max_order=3)


class TestGoodTuringSmoother:
    """Test Good-Turing smoothing"""

    def test_basic_functionality(self):
        lm = ArpaBoLM(smoothing_method="good_turing", verbose=False)
        lm.read_corpus(StringIO("the quick brown fox\nthe lazy dog"))
        lm.compute()

        assert len(lm.probs[0]) == 8  # <s>, </s>, the, quick, brown, fox, lazy, dog
        assert all(prob > 0 for prob in lm.probs[0].values())
        assert isinstance(lm.smoother, GoodTuringSmoother)

    def test_needs_backoff_weights(self):
        smoother = GoodTuringSmoother(max_order=3)
        assert not smoother.needs_backoff_weights()

    def test_different_orders(self):
        for order in [2, 3, 4, 5]:
            lm = ArpaBoLM(max_order=order, smoothing_method="good_turing", verbose=False)
            lm.read_corpus(StringIO("the quick brown fox jumps over the lazy dog"))
            lm.compute()
            assert lm.max_order == order
            assert len(lm.counts) == order


class TestKneserNeySmoother:
    """Test Kneser-Ney smoothing"""

    def test_basic_functionality(self):
        lm = ArpaBoLM(smoothing_method="kneser_ney", verbose=False)
        lm.read_corpus(StringIO("the quick brown fox\nthe lazy dog"))
        lm.compute()

        assert len(lm.probs[0]) == 8
        assert all(prob > 0 for prob in lm.probs[0].values())
        assert isinstance(lm.smoother, KneserNeySmoother)

    def test_custom_discount(self):
        lm = ArpaBoLM(smoothing_method="kneser_ney", verbose=False)
        assert lm.smoother.discount == 0.75  # DEFAULT_DISCOUNT

    def test_needs_backoff_weights(self):
        smoother = KneserNeySmoother(max_order=3)
        assert not smoother.needs_backoff_weights()


class TestKatzBackoffSmoother:
    """Test Katz backoff smoothing"""

    def test_fixed_discount(self):
        lm = ArpaBoLM(smoothing_method="fixed", discount_mass=0.5, verbose=False)
        lm.read_corpus(StringIO("the quick brown fox\nthe lazy dog"))
        lm.compute()

        assert len(lm.probs[0]) == 8
        assert lm.discount_mass == 0.5
        assert isinstance(lm.smoother, KatzBackoffSmoother)
        assert not lm.smoother.auto_optimize

    def test_auto_optimization(self):
        lm = ArpaBoLM(smoothing_method="auto", verbose=False)
        lm.read_corpus(StringIO("the quick brown fox\nthe lazy dog"))
        lm.compute()

        assert isinstance(lm.smoother, KatzBackoffSmoother)
        assert lm.smoother.auto_optimize
        # Discount mass should be optimized (not default)
        assert 0.0 < lm.discount_mass < 1.0

    def test_mle_zero_discount(self):
        lm = ArpaBoLM(smoothing_method="fixed", discount_mass=0.0, verbose=False)
        lm.read_corpus(StringIO("the quick brown fox\nthe lazy dog"))
        lm.compute()

        assert lm.discount_mass == 0.0

    def test_needs_backoff_weights(self):
        smoother = KatzBackoffSmoother(max_order=3)
        assert smoother.needs_backoff_weights()


class TestSmoothingComparison:
    """Compare different smoothing methods"""

    def test_all_methods_produce_valid_probabilities(self):
        corpus = "the quick brown fox\nthe lazy dog\nthe brown dog"

        for method in ["good_turing", "kneser_ney", "fixed"]:
            lm = ArpaBoLM(smoothing_method=method, verbose=False)
            lm.read_corpus(StringIO(corpus))
            lm.compute()

            # All methods should produce valid probability distributions
            assert len(lm.probs[0]) > 0
            assert all(0.0 <= prob <= 1.0 for prob in lm.probs[0].values())

            # Verify probabilities are reasonable (but don't enforce sum=1.0
            # as smoothing methods redistribute mass differently)
            total_prob = sum(lm.probs[0].values())
            assert total_prob > 0, f"Method {method} produced zero total probability"
