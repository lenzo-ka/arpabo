"""Tests for CLI functionality"""

import subprocess
import sys
import tempfile


class TestCLI:
    """Test command-line interface"""

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "arpabo.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ARPA language model" in result.stdout

    def test_demo_mode(self):
        with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as f:
            arpa_path = f.name

        result = subprocess.run(
            [sys.executable, "-m", "arpabo.cli", "--demo", "-o", arpa_path],
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
                "arpabo.cli",
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
                    "arpabo.cli",
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
                    "arpabo.cli",
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
            [sys.executable, "-m", "arpabo.cli"],
            capture_output=True,
            text=True,
        )

        # Should fail when no input provided
        assert result.returncode != 0

    def test_multi_order_training(self):
        """Test --orders flag for multi-order training."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "arpabo.cli",
                    "--demo",
                    "-o",
                    tmpdir,
                    "--orders",
                    "1,2,3",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

            # Verify all files were created
            for order in [1, 2, 3]:
                arpa_file = os.path.join(tmpdir, f"{order}gram.arpa")
                assert os.path.exists(arpa_file), f"Missing {order}gram.arpa"

                # Verify valid ARPA format
                with open(arpa_file) as f:
                    content = f.read()
                    assert "\\data\\" in content
                    assert f"ngram {order}=" in content
                    assert "\\end\\" in content

    def test_multi_order_range_syntax(self):
        """Test --orders with range syntax."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "arpabo.cli",
                    "--demo",
                    "-o",
                    tmpdir,
                    "--orders",
                    "1-3",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

            # Verify all files were created
            for order in [1, 2, 3]:
                arpa_file = os.path.join(tmpdir, f"{order}gram.arpa")
                assert os.path.exists(arpa_file)

    def test_multi_order_mixed_syntax(self):
        """Test --orders with mixed syntax."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "arpabo.cli",
                    "--demo",
                    "-o",
                    tmpdir,
                    "--orders",
                    "1-2,4",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

            # Verify correct files were created
            for order in [1, 2, 4]:
                arpa_file = os.path.join(tmpdir, f"{order}gram.arpa")
                assert os.path.exists(arpa_file)

            # Order 3 should not exist
            assert not os.path.exists(os.path.join(tmpdir, "3gram.arpa"))

    def test_multi_order_requires_output_dir(self):
        """Test --orders requires -o flag."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "arpabo.cli",
                "--demo",
                "--orders",
                "1,2",
            ],
            capture_output=True,
            text=True,
        )

        # Should fail without -o
        assert result.returncode != 0
        assert "requires -o" in result.stderr
