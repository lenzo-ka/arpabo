"""Tests for CLI functionality"""

import subprocess
import sys
import tempfile


class TestCLI:
    """Test command-line interface"""

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "arpalm.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ARPA language model" in result.stdout

    def test_demo_mode(self):
        with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as f:
            arpa_path = f.name

        result = subprocess.run(
            [sys.executable, "-m", "arpalm.cli", "--demo", "-o", arpa_path],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Verify file was created
        import os

        assert os.path.exists(arpa_path)

        # Verify it's valid ARPA format
        with open(arpa_path) as f:
            content = f.read()
            assert "\\data\\" in content
            assert "\\1-grams:" in content

        os.unlink(arpa_path)

    def test_text_input(self):
        with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as f:
            arpa_path = f.name

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "arpalm.cli",
                "-t",
                "hello world",
                "-o",
                arpa_path,
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        import os

        assert os.path.exists(arpa_path)
        os.unlink(arpa_path)

    def test_smoothing_methods(self):
        for method in ["good_turing", "kneser_ney", "fixed"]:
            with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as f:
                arpa_path = f.name

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "arpalm.cli",
                    "-t",
                    "hello world",
                    "-o",
                    arpa_path,
                    "-s",
                    method,
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Failed for method: {method}"

            import os

            os.unlink(arpa_path)

    def test_max_order(self):
        for order in [2, 3, 4]:
            with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as f:
                arpa_path = f.name

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "arpalm.cli",
                    "-t",
                    "the quick brown fox",
                    "-o",
                    arpa_path,
                    "-m",
                    str(order),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Failed for order: {order}"

            # Verify correct order in output
            with open(arpa_path) as f:
                content = f.read()
                assert f"ngram {order}=" in content

            import os

            os.unlink(arpa_path)

    def test_no_input_error(self):
        result = subprocess.run(
            [sys.executable, "-m", "arpalm.cli"],
            capture_output=True,
            text=True,
        )

        # Should fail when no input provided
        assert result.returncode != 0
