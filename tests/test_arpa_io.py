"""Tests for ARPA format I/O"""

import tempfile
from io import StringIO

from arpabo import ArpaBoLM


class TestArpaWrite:
    """Test ARPA format writing"""

    def test_write_file(self):
        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO("the quick brown fox"))
        lm.compute()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".arpa", delete=False) as f:
            arpa_path = f.name

        success = lm.write_file(arpa_path)
        assert success

        # Verify file exists and has content
        with open(arpa_path) as f:
            content = f.read()
            assert "\\data\\" in content
            assert "\\1-grams:" in content
            assert "\\2-grams:" in content
            assert "\\end\\" in content

        import os

        os.unlink(arpa_path)

    def test_write_to_handle(self):
        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO("hello world"))
        lm.compute()

        output = StringIO()
        lm.write(output)
        content = output.getvalue()

        assert "\\data\\" in content
        assert "\\1-grams:" in content
        assert "\\end\\" in content


class TestArpaRead:
    """Test ARPA format reading"""

    def test_read_write_roundtrip(self):
        # Create a model
        lm1 = ArpaBoLM(verbose=False)
        lm1.read_corpus(StringIO("the quick brown fox\nthe lazy dog"))
        lm1.compute()

        # Write it
        with tempfile.NamedTemporaryFile(mode="w", suffix=".arpa", delete=False) as f:
            arpa_path = f.name
        lm1.write_file(arpa_path)

        # Read it back
        lm2 = ArpaBoLM.from_arpa_file(arpa_path, verbose=False)

        # Verify same vocabulary
        assert len(lm2.probs[0]) == len(lm1.probs[0])
        assert set(lm2.probs[0].keys()) == set(lm1.probs[0].keys())

        # Verify same max order
        assert lm2.max_order == lm1.max_order

        import os

        os.unlink(arpa_path)

    def test_load_and_update(self):
        # Create initial model
        lm1 = ArpaBoLM(verbose=False)
        lm1.read_corpus(StringIO("hello world"))
        lm1.compute()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".arpa", delete=False) as f:
            arpa_path = f.name
        lm1.write_file(arpa_path)

        # Load and add more data
        lm2 = ArpaBoLM.from_arpa_file(arpa_path, verbose=False)
        initial_vocab_size = len(lm2.probs[0])

        lm2.read_corpus(StringIO("goodbye world"))
        lm2.compute()

        # Should have new word
        assert "goodbye" in lm2.probs[0]
        assert len(lm2.probs[0]) > initial_vocab_size

        import os

        os.unlink(arpa_path)


class TestArpaFormat:
    """Test ARPA format compliance"""

    def test_required_sections(self):
        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO("hello world"))
        lm.compute()

        output = StringIO()
        lm.write(output)
        content = output.getvalue()

        # Required ARPA sections
        assert "\\data\\" in content
        assert "\\1-grams:" in content
        assert "\\end\\" in content

    def test_ngram_counts(self):
        lm = ArpaBoLM(max_order=3, verbose=False)
        lm.read_corpus(StringIO("the quick brown fox"))
        lm.compute()

        output = StringIO()
        lm.write(output)
        content = output.getvalue()

        # Should declare ngram counts
        assert "ngram 1=" in content
        assert "ngram 2=" in content
        assert "ngram 3=" in content

    def test_log_probabilities_format(self):
        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO("hello world"))
        lm.compute()

        output = StringIO()
        lm.write(output)
        content = output.getvalue()

        # Should have log probabilities (negative numbers)
        lines = content.split("\n")
        ngram_lines = [
            line
            for line in lines
            if line and not line.startswith("\\") and not line.startswith("Corpus") and not line.startswith("ngram")
        ]

        # At least some lines should have negative log probs
        has_negative = any("-" in line.split()[0] if line.split() else False for line in ngram_lines if line)
        assert has_negative
