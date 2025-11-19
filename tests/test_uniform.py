"""Tests for uniform language model functionality."""

from io import StringIO

import pytest

from arpabo.lm import ArpaBoLM


class TestCreateUniform:
    """Test create_uniform classmethod."""

    def test_from_word_list(self):
        """Test creating uniform LM from word list."""
        words = ["the", "cat", "sat", "on", "mat"]
        lm = ArpaBoLM.create_uniform(words)

        assert lm.max_order == 1
        assert lm.smoothing_method == "uniform"
        # Should include start/end markers by default
        assert "<s>" in lm.probs[0]
        assert "</s>" in lm.probs[0]

    def test_from_word_list_no_markers(self):
        """Test creating uniform LM without sentence markers."""
        words = ["the", "cat", "sat"]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        stats = lm.get_statistics()
        assert stats["vocab_size"] == 3
        assert "<s>" not in lm.probs[0]
        assert "</s>" not in lm.probs[0]

    def test_from_file(self, tmp_path):
        """Test creating uniform LM from vocabulary file."""
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("the\ncat\nsat\non\nmat\n")

        lm = ArpaBoLM.create_uniform(str(vocab_file))

        assert lm.max_order == 1
        assert "the" in lm.probs[0]
        assert "cat" in lm.probs[0]
        assert "mat" in lm.probs[0]

    def test_equal_probabilities(self):
        """Test that all words have equal probability."""
        words = ["word1", "word2", "word3", "word4", "word5"]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        # Get all probabilities
        probs = list(lm.probs[0].values())

        # All should be equal
        assert all(abs(p - probs[0]) < 1e-10 for p in probs)

        # Should be 1/5 = 0.2
        expected_prob = 1.0 / 5
        assert abs(probs[0] - expected_prob) < 1e-10

    def test_probability_value(self):
        """Test that probability value is correct."""
        words = ["a", "b", "c"]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        # With 3 words, each should have prob 1/3
        expected = 1.0 / 3

        for word in words:
            assert abs(lm.probs[0][word] - expected) < 1e-6

    def test_with_sentence_markers(self):
        """Test that sentence markers are handled correctly."""
        words = ["the", "cat"]
        lm = ArpaBoLM.create_uniform(words, add_start=True)

        # Should have 4 words: the, cat, <s>, </s>
        stats = lm.get_statistics()
        assert stats["vocab_size"] == 4

        # All should have equal probability: 1/4 = 0.25
        expected = 1.0 / 4
        for word in ["the", "cat", "<s>", "</s>"]:
            assert abs(lm.probs[0][word] - expected) < 1e-6

    def test_markers_not_duplicated(self):
        """Test that markers aren't added if already present."""
        words = ["<s>", "the", "cat", "</s>"]
        lm = ArpaBoLM.create_uniform(words, add_start=True)

        # Should still have 4 words (not 6)
        stats = lm.get_statistics()
        assert stats["vocab_size"] == 4

    def test_empty_vocab_error(self):
        """Test that empty vocabulary raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ArpaBoLM.create_uniform([])

    def test_file_with_blank_lines(self, tmp_path):
        """Test that blank lines in vocab file are skipped."""
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("word1\n\nword2\n  \nword3\n")

        lm = ArpaBoLM.create_uniform(str(vocab_file), add_start=False)

        stats = lm.get_statistics()
        # Should have 3 words (blank lines skipped)
        assert stats["vocab_size"] == 3


class TestUniformModelUsage:
    """Test using uniform language models."""

    def test_can_write_arpa(self, tmp_path):
        """Test that uniform model can be written to ARPA format."""
        words = ["the", "cat", "sat"]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        output_file = tmp_path / "uniform.arpa"
        lm.write_file(str(output_file))

        assert output_file.exists()

        # Verify ARPA format
        content = output_file.read_text()
        assert "\\data\\" in content
        assert "ngram 1=" in content
        assert "\\1-grams:" in content
        assert "\\end\\" in content

        # Check that words are in output
        assert "the" in content
        assert "cat" in content
        assert "sat" in content

    def test_statistics(self):
        """Test that get_statistics works with uniform model."""
        words = ["a", "b", "c", "d", "e"]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        stats = lm.get_statistics()

        assert stats["order"] == 1
        assert stats["smoothing"] == "uniform"
        assert stats["vocab_size"] == 5
        assert 1 in stats["ngram_counts"]
        assert stats["ngram_counts"][1] == 5

    def test_perplexity_equals_vocab_size(self):
        """Test that perplexity of uniform model equals vocab size."""
        words = ["the", "cat", "sat", "on", "mat"]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        # Test on the same vocabulary
        test_data = StringIO("the cat sat on mat")

        results = lm.perplexity(test_data)

        # For uniform model, perplexity should equal vocabulary size
        # (when all test words are in vocabulary)
        assert abs(results["perplexity"] - 5.0) < 0.1

    def test_backoff_rate(self):
        """Test backoff_rate with uniform model."""
        words = ["the", "cat", "sat"]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        test_data = StringIO("the cat")
        backoff = lm.backoff_rate(test_data)

        # Uniform model is unigram-only, so backoff_rate should be 0
        # (always uses unigram = full order for unigram model)
        assert backoff["overall_backoff_rate"] == 0.0
        assert backoff["order_usage"][1] == 1.0


class TestUniformComparison:
    """Test comparing uniform models with trained models."""

    def test_uniform_vs_trained(self):
        """Test comparing uniform model with trained model."""
        corpus = StringIO("the cat sat on the mat\nthe dog sat on the log")

        # Train regular model
        trained_lm = ArpaBoLM(max_order=2, verbose=False)
        trained_lm.read_corpus(corpus)
        trained_lm.compute()

        # Create uniform model from same vocabulary
        vocab = list(trained_lm.grams[0].keys())
        uniform_lm = ArpaBoLM.create_uniform(vocab, add_start=False)

        # Test data
        test_data = "the cat sat"

        # Evaluate both
        trained_ppl = trained_lm.perplexity(StringIO(test_data))
        uniform_ppl = uniform_lm.perplexity(StringIO(test_data))

        # Trained model should have better (lower) perplexity
        assert trained_ppl["perplexity"] < uniform_ppl["perplexity"]

    def test_large_vocabulary(self):
        """Test uniform model with larger vocabulary."""
        # Generate 100 words
        words = [f"word{i}" for i in range(100)]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        stats = lm.get_statistics()
        assert stats["vocab_size"] == 100

        # Each word should have probability 1/100 = 0.01
        expected = 1.0 / 100

        for i in range(10):  # Check first 10
            word = f"word{i}"
            assert abs(lm.probs[0][word] - expected) < 1e-6


class TestUniformEdgeCases:
    """Test edge cases for uniform models."""

    def test_single_word(self):
        """Test uniform model with single word."""
        lm = ArpaBoLM.create_uniform(["only"], add_start=False)

        stats = lm.get_statistics()
        assert stats["vocab_size"] == 1

        # Probability should be 1/1 = 1.0
        assert abs(lm.probs[0]["only"] - 1.0) < 1e-6

    def test_unicode_words(self):
        """Test uniform model with unicode characters."""
        words = ["café", "naïve", "résumé", "日本語"]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        stats = lm.get_statistics()
        assert stats["vocab_size"] == 4

        # All should have equal probability: 1/4 = 0.25
        expected = 1.0 / 4
        for word in words:
            assert abs(lm.probs[0][word] - expected) < 1e-6

    def test_special_tokens(self):
        """Test uniform model with special tokens."""
        words = ["<unk>", "<pad>", "</s>", "<s>", "word"]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        stats = lm.get_statistics()
        assert stats["vocab_size"] == 5

        # Special tokens should work like regular words
        for word in words:
            assert word in lm.probs[0]


class TestUniformIntegration:
    """Integration tests for uniform models."""

    def test_uniform_with_binary_conversion(self, tmp_path):
        """Test that uniform model can be converted to binary."""
        words = ["the", "cat", "sat"]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        arpa_file = tmp_path / "uniform.arpa"
        lm.write_file(str(arpa_file))

        assert arpa_file.exists()
        # Binary conversion would happen via external tool

    def test_uniform_can_be_loaded(self, tmp_path):
        """Test that uniform model can be written and loaded."""
        words = ["alpha", "beta", "gamma", "delta"]
        lm = ArpaBoLM.create_uniform(words, add_start=False)

        # Write
        arpa_file = tmp_path / "uniform.arpa"
        lm.write_file(str(arpa_file))

        # Load
        loaded_lm = ArpaBoLM.from_arpa_file(str(arpa_file))

        # Should have same vocabulary
        assert loaded_lm.max_order == 1
        for word in words:
            assert word in loaded_lm.probs[0]

    def test_uniform_in_comparison(self):
        """Test using uniform model as baseline in comparison."""
        # Larger corpus with repeated patterns
        corpus = StringIO("""the quick brown fox jumps over the lazy dog
the quick brown fox runs fast
the lazy dog sleeps all day
the brown fox jumps high
quick brown animals run fast""")

        # Train models of different orders
        models = {}
        for order in [1, 2, 3]:
            lm = ArpaBoLM(max_order=order, verbose=False)
            lm.read_corpus(StringIO(corpus.getvalue()))
            lm.compute()
            models[order] = lm

        # Create uniform baseline
        vocab = list(models[1].grams[0].keys())
        uniform = ArpaBoLM.create_uniform(vocab, add_start=False)

        # Test data
        test_data = "the quick brown fox"

        # Evaluate all
        results = {}
        for order, model in models.items():
            ppl = model.perplexity(StringIO(test_data))
            results[order] = ppl["perplexity"]

        uniform_ppl = uniform.perplexity(StringIO(test_data))
        results["uniform"] = uniform_ppl["perplexity"]

        # Higher-order trained models should beat uniform baseline
        # (Unigram might not depending on distribution)
        assert results[2] < results["uniform"]
        assert results[3] < results["uniform"]
