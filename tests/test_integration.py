"""Integration tests for end-to-end workflows"""

import subprocess
import sys
import tempfile
from io import StringIO

from arpabo import ArpaBoLM


class TestEndToEnd:
    """Test complete workflows"""

    def test_build_and_load_model(self):
        # Build a model
        lm1 = ArpaBoLM(max_order=3, smoothing_method="good_turing", verbose=False)
        corpus = """
        the quick brown fox jumps over the lazy dog
        the lazy dog sleeps in the sun
        the quick fox runs fast
        """
        lm1.read_corpus(StringIO(corpus))
        lm1.compute()

        # Save it
        with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as f:
            arpa_path = f.name
        lm1.write_file(arpa_path)

        # Load it
        lm2 = ArpaBoLM.from_arpa_file(arpa_path)

        # Verify equivalence
        assert len(lm2.probs[0]) == len(lm1.probs[0])
        assert set(lm2.probs[0].keys()) == set(lm1.probs[0].keys())

        import os

        os.unlink(arpa_path)

    def test_demo_to_pocketsphinx(self):
        """Test demo corpus → ARPA → PocketSphinx conversion"""
        with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as f:
            arpa_path = f.name

        # Generate ARPA file
        result = subprocess.run(
            [sys.executable, "-m", "arpabo.cli", "--demo", "-o", arpa_path],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Try to convert to binary (if pocketsphinx_lm_convert available)
        bin_path = arpa_path.replace(".arpa", ".lm.bin")

        import os
        import shutil

        ps_convert = shutil.which("pocketsphinx_lm_convert")
        if not ps_convert:
            ps_convert = os.path.expanduser("~/dev/cmu/pocketsphinx/build/pocketsphinx_lm_convert")

        if os.path.exists(ps_convert):
            convert_result = subprocess.run(
                [ps_convert, "-i", arpa_path, "-o", bin_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if convert_result.returncode == 0:
                assert os.path.exists(bin_path)
                os.unlink(bin_path)

        os.unlink(arpa_path)

    def test_all_smoothing_methods(self):
        """Test all smoothing methods produce valid output"""
        corpus = "the quick brown fox\nthe lazy dog"

        for method in ["good_turing", "kneser_ney", "fixed", "auto"]:
            lm = ArpaBoLM(smoothing_method=method, verbose=False)
            lm.read_corpus(StringIO(corpus))
            lm.compute()

            with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as f:
                arpa_path = f.name

            success = lm.write_file(arpa_path)
            assert success, f"Failed to write with {method}"

            # Verify can read it back
            lm2 = ArpaBoLM.from_arpa_file(arpa_path)
            assert len(lm2.probs[0]) > 0

            import os

            os.unlink(arpa_path)

    def test_normalization_pipeline(self):
        """Test text normalization end-to-end"""
        lm = ArpaBoLM(
            case="lower",
            token_norm=True,
            unicode_norm=True,
            verbose=False,
        )

        corpus = "Hello, WORLD! How are you?"
        lm.read_corpus(StringIO(corpus))
        lm.compute()

        # Should be lowercase and punctuation-stripped
        assert "hello" in lm.probs[0]
        assert "world" in lm.probs[0]
        assert "HELLO" not in lm.probs[0]
        assert "WORLD" not in lm.probs[0]


class TestLargeCorpus:
    """Test with larger corpus (Alice)"""

    def test_alice_corpus_all_methods(self):
        """Test Alice corpus with all smoothing methods"""
        from arpabo import get_example_corpus

        corpus_path = get_example_corpus()

        for method in ["good_turing", "kneser_ney", "fixed"]:
            lm = ArpaBoLM(max_order=3, smoothing_method=method, verbose=False)

            with open(corpus_path) as f:
                lm.read_corpus(f)

            lm.compute()

            # Verify reasonable model size
            assert lm.sent_count > 100
            assert len(lm.probs[0]) > 500
            assert lm.sum_1 > 2000

    def test_alice_different_orders(self):
        """Test Alice corpus with different n-gram orders"""
        from arpabo import get_example_corpus

        corpus_path = get_example_corpus()

        for order in [2, 3, 4]:
            lm = ArpaBoLM(max_order=order, verbose=False)

            with open(corpus_path) as f:
                lm.read_corpus(f)

            lm.compute()

            assert lm.max_order == order
            # Higher orders should have more n-grams (up to a point)
            assert lm.counts[0] > 0  # Always have unigrams
