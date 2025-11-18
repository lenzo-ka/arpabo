"""Tests for example data management"""

import pytest

from arpabo import get_example_corpus, list_example_corpora


class TestExampleData:
    """Test example corpus functionality"""

    def test_list_corpora(self):
        corpora = list_example_corpora()
        assert isinstance(corpora, list)
        assert "alice.txt" in corpora

    def test_get_example_corpus(self):
        corpus_path = get_example_corpus()
        assert corpus_path.endswith("alice.txt")

        # Verify file exists
        import os

        assert os.path.exists(corpus_path)

    def test_get_specific_corpus(self):
        corpus_path = get_example_corpus("alice.txt")
        assert corpus_path.endswith("alice.txt")

    def test_nonexistent_corpus(self):
        with pytest.raises(FileNotFoundError):
            get_example_corpus("nonexistent.txt")

    def test_alice_corpus_usable(self):
        from arpabo import ArpaBoLM

        corpus_path = get_example_corpus()
        lm = ArpaBoLM(verbose=False)

        with open(corpus_path) as f:
            lm.read_corpus(f)

        lm.compute()

        # Should have processed Alice corpus successfully
        assert lm.sent_count > 0
        assert len(lm.probs[0]) > 100  # Alice has reasonable vocabulary
