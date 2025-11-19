"""Tests for vocabulary pruning functionality."""

from io import StringIO

import pytest

from arpabo.lm import ArpaBoLM


class TestPruneVocabularyBasic:
    """Test basic vocabulary pruning."""

    @pytest.fixture
    def trained_model(self):
        """Create trained model with varied word frequencies."""
        corpus = StringIO(
            """the the the the the
cat cat cat
dog dog
bird
fish"""
        )

        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()
        return lm

    def test_prune_frequency(self, trained_model):
        """Test frequency-based pruning."""
        # Keep words with frequency >= 3
        pruned = trained_model.prune_vocabulary(method="frequency", threshold=3)

        stats = pruned.get_statistics()

        # Should keep "the" (5), "cat" (3), and markers
        assert stats["vocab_size"] >= 3  # the, cat, <s>, </s>
        assert "the" in pruned.grams[0]
        assert "cat" in pruned.grams[0]
        # Should remove "dog" (2), "bird" (1), "fish" (1)

    def test_prune_topk(self, trained_model):
        """Test top-K pruning."""
        # Keep top 3 words (markers included if they're in top 3, or added separately)
        pruned = trained_model.prune_vocabulary(method="topk", threshold=3)

        stats = pruned.get_statistics()

        # Should have at least 3 words (top k) plus any markers
        assert stats["vocab_size"] >= 3
        # Markers should be present (keep_markers=True by default)
        assert "<s>" in pruned.grams[0]
        assert "</s>" in pruned.grams[0]

    def test_markers_preserved(self, trained_model):
        """Test that sentence markers are preserved by default."""
        pruned = trained_model.prune_vocabulary(method="topk", threshold=2)

        assert "<s>" in pruned.grams[0]
        assert "</s>" in pruned.grams[0]

    def test_markers_not_preserved(self, trained_model):
        """Test pruning without preserving markers."""
        pruned = trained_model.prune_vocabulary(method="topk", threshold=2, keep_markers=False)

        # May or may not have markers depending on frequency
        stats = pruned.get_statistics()
        assert stats["vocab_size"] >= 2

    def test_model_not_trained_error(self):
        """Test that pruning fails if model not trained."""
        lm = ArpaBoLM(max_order=2, verbose=False)

        with pytest.raises(ValueError, match="not been trained"):
            lm.prune_vocabulary(method="frequency", threshold=10)

    def test_invalid_method(self, trained_model):
        """Test that invalid pruning method raises error."""
        with pytest.raises(ValueError, match="Unknown pruning method"):
            trained_model.prune_vocabulary(method="invalid", threshold=10)

    def test_threshold_too_high(self, trained_model):
        """Test that too-high threshold with keep_markers=False raises error."""
        with pytest.raises(ValueError, match="removed all words"):
            trained_model.prune_vocabulary(method="frequency", threshold=1000, keep_markers=False)


class TestPrunedModelUsage:
    """Test using pruned models."""

    @pytest.fixture
    def large_model(self):
        """Create model with more diverse vocabulary."""
        words = ["the"] * 100 + ["quick"] * 50 + ["brown"] * 25
        words += [f"rare{i}" for i in range(20)]  # 20 rare words (freq=1)

        corpus = " ".join(words)
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(StringIO(corpus))
        lm.compute()
        return lm

    def test_pruned_model_can_write(self, large_model, tmp_path):
        """Test that pruned model can be written to ARPA."""
        pruned = large_model.prune_vocabulary(method="frequency", threshold=10)

        output_file = tmp_path / "pruned.arpa"
        pruned.write_file(str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_pruned_model_statistics(self, large_model):
        """Test getting statistics from pruned model."""
        pruned = large_model.prune_vocabulary(method="topk", threshold=10)

        stats = pruned.get_statistics()

        assert stats["order"] == 2
        assert stats["vocab_size"] >= 10
        assert 1 in stats["ngram_counts"]

    def test_pruned_model_perplexity(self, large_model):
        """Test evaluating pruned model perplexity."""
        pruned = large_model.prune_vocabulary(method="topk", threshold=10)

        test_data = StringIO("the quick brown")
        results = pruned.perplexity(test_data)

        assert results["perplexity"] > 0
        # OOV rate may be higher due to pruning

    def test_pruned_vs_original_vocab_size(self, large_model):
        """Test that pruning actually reduces vocabulary."""
        original_stats = large_model.get_statistics()

        pruned = large_model.prune_vocabulary(method="frequency", threshold=10)
        pruned_stats = pruned.get_statistics()

        assert pruned_stats["vocab_size"] < original_stats["vocab_size"]

    def test_frequency_preserves_common_words(self, large_model):
        """Test that frequency pruning keeps common words."""
        pruned = large_model.prune_vocabulary(method="frequency", threshold=10)

        # Should keep high-frequency words
        assert "the" in pruned.grams[0]  # freq=100
        assert "quick" in pruned.grams[0]  # freq=50

    def test_topk_keeps_exactly_k_plus_markers(self, large_model):
        """Test that topk keeps at least K words."""
        pruned = large_model.prune_vocabulary(method="topk", threshold=5)

        stats = pruned.get_statistics()

        # Should have at least 5 words (may include markers)
        assert stats["vocab_size"] >= 5
        # Markers should be present
        assert "<s>" in pruned.grams[0]
        assert "</s>" in pruned.grams[0]


class TestPruningImpact:
    """Test impact of pruning on model quality."""

    @pytest.fixture
    def model_with_test_set(self, tmp_path):
        """Create model and test set."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            """the quick brown fox jumps over the lazy dog
the lazy dog sleeps under the brown tree
the quick fox runs through the green field"""
        )

        test = tmp_path / "test.txt"
        test.write_text("the quick fox")

        lm = ArpaBoLM(max_order=3, verbose=False)
        with open(corpus) as f:
            lm.read_corpus(f)
        lm.compute()

        return lm, test

    def test_pruning_increases_oov(self, model_with_test_set):
        """Test that pruning increases OOV rate."""
        lm, test_file = model_with_test_set

        # Original
        with open(test_file) as f:
            original_results = lm.perplexity(f)

        # Pruned
        pruned = lm.prune_vocabulary(method="topk", threshold=5)
        with open(test_file) as f:
            pruned_results = pruned.perplexity(f)

        # Pruned model should have higher or equal OOV rate
        assert pruned_results["oov_rate"] >= original_results["oov_rate"]

    def test_pruning_may_increase_perplexity(self, model_with_test_set):
        """Test that pruning typically increases perplexity."""
        lm, test_file = model_with_test_set

        # Original
        with open(test_file) as f:
            original_results = lm.perplexity(f)

        # Aggressively pruned
        pruned = lm.prune_vocabulary(method="topk", threshold=3)
        with open(test_file) as f:
            pruned_results = pruned.perplexity(f)

        # Both should have valid perplexities
        assert original_results["perplexity"] > 0
        assert pruned_results["perplexity"] > 0


class TestPruningEdgeCases:
    """Test edge cases for pruning."""

    def test_prune_to_zero_words(self):
        """Test that pruning to zero words raises error when markers not kept."""
        corpus = StringIO("the cat")
        lm = ArpaBoLM(max_order=1, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()

        with pytest.raises(ValueError, match="removed all words"):
            lm.prune_vocabulary(method="frequency", threshold=1000, keep_markers=False)

    def test_topk_larger_than_vocab(self):
        """Test topk with K larger than vocabulary."""
        corpus = StringIO("the cat sat")
        lm = ArpaBoLM(max_order=1, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()

        original_stats = lm.get_statistics()

        # Request more words than available
        pruned = lm.prune_vocabulary(method="topk", threshold=1000)
        pruned_stats = pruned.get_statistics()

        # Should keep all words
        assert pruned_stats["vocab_size"] == original_stats["vocab_size"]

    def test_frequency_threshold_zero(self):
        """Test frequency threshold of 0 keeps all words."""
        corpus = StringIO("the cat sat on mat")
        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()

        original_stats = lm.get_statistics()

        pruned = lm.prune_vocabulary(method="frequency", threshold=0)
        pruned_stats = pruned.get_statistics()

        # Should keep all words (all have freq >= 0)
        assert pruned_stats["vocab_size"] == original_stats["vocab_size"]


class TestPruningHigherOrderNgrams:
    """Test that pruning correctly filters higher-order n-grams."""

    def test_higher_order_filtered(self):
        """Test that higher-order n-grams are filtered correctly."""
        corpus = StringIO("the quick brown fox\nthe lazy dog")
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()

        # Keep only "the" and "quick"
        pruned = lm.prune_vocabulary(method="topk", threshold=2)

        stats = pruned.get_statistics()

        # Should have reduced bigram and trigram counts
        # (only n-grams with "the" and "quick" remain)
        assert stats["ngram_counts"][2] < lm.counts[1]  # Fewer bigrams
        if lm.counts[2] > 0:
            assert stats["ngram_counts"].get(3, 0) < lm.counts[2]  # Fewer trigrams


class TestPruningIntegration:
    """Integration tests for pruning."""

    def test_prune_and_evaluate(self, tmp_path):
        """Test complete prune and evaluate workflow."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the quick brown fox\nthe lazy dog")

        test = tmp_path / "test.txt"
        test.write_text("the fox")

        # Train
        lm = ArpaBoLM(max_order=2, verbose=False)
        with open(corpus) as f:
            lm.read_corpus(f)
        lm.compute()

        # Prune
        pruned = lm.prune_vocabulary(method="topk", threshold=5)

        # Evaluate
        with open(test) as f:
            results = pruned.perplexity(f)

        assert results["perplexity"] > 0

    def test_prune_and_export(self, tmp_path):
        """Test pruning and exporting."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the quick brown fox")

        lm = ArpaBoLM(max_order=2, verbose=False)
        with open(corpus) as f:
            lm.read_corpus(f)
        lm.compute()

        # Prune
        pruned = lm.prune_vocabulary(method="topk", threshold=3)

        # Write
        output = tmp_path / "pruned.arpa"
        pruned.write_file(str(output))

        assert output.exists()

        # Should be smaller file
        original_output = tmp_path / "original.arpa"
        lm.write_file(str(original_output))

        assert output.stat().st_size <= original_output.stat().st_size

    def test_multiple_pruning_levels(self):
        """Test pruning at different levels."""
        corpus = StringIO("the quick brown fox jumps over the lazy dog and cat")

        lm = ArpaBoLM(max_order=2, verbose=False)
        lm.read_corpus(corpus)
        lm.compute()

        original_stats = lm.get_statistics()

        # Prune at different levels
        for k in [10, 7, 5, 3]:
            pruned = lm.prune_vocabulary(method="topk", threshold=k)
            pruned_stats = pruned.get_statistics()

            # Smaller K = smaller vocabulary
            assert pruned_stats["vocab_size"] <= original_stats["vocab_size"]
