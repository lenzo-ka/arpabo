"""Tests for normalization CLI"""

import subprocess
import sys
import tempfile


class TestNormalizeCLI:
    """Test arpalm-normalize command"""

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "arpabo.cli_normalize", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Normalize text corpus" in result.stdout

    def test_stdin_to_stdout(self):
        result = subprocess.run(
            [sys.executable, "-m", "arpabo.cli_normalize"],
            input="Hello World\nGoodbye World\n",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Hello" in result.stdout
        assert "World" in result.stdout

    def test_lowercase_normalization(self):
        result = subprocess.run(
            [sys.executable, "-m", "arpabo.cli_normalize", "-c", "lower"],
            input="HELLO WORLD\n",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "hello world" in result.stdout
        assert "HELLO" not in result.stdout

    def test_token_normalization(self):
        result = subprocess.run(
            [sys.executable, "-m", "arpabo.cli_normalize", "-n"],
            input="Hello, World!\n",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        lines = result.stdout.strip().split("\n")
        assert "Hello World" in lines[0] or "Hello" in lines[0]

    def test_add_markers(self):
        result = subprocess.run(
            [sys.executable, "-m", "arpabo.cli_normalize", "--add-markers"],
            input="hello world\n",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "<s>" in result.stdout
        assert "</s>" in result.stdout

    def test_file_input(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello World\nGoodbye World\n")
            input_file = f.name

        result = subprocess.run(
            [sys.executable, "-m", "arpabo.cli_normalize", input_file, "-c", "lower"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "hello world" in result.stdout
        assert "goodbye world" in result.stdout

        import os

        os.unlink(input_file)

    def test_output_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            input_file = f.name
            f.write("Hello World\n")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_file = f.name

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "arpabo.cli_normalize",
                input_file,
                "-o",
                output_file,
                "-c",
                "lower",
                "-n",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        with open(output_file) as f:
            content = f.read()
            assert "hello world" in content

        import os

        os.unlink(input_file)
        os.unlink(output_file)

    def test_pipeline_workflow(self):
        """Test normalize → arpalm pipeline"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            input_file = f.name
            f.write("The QUICK brown FOX!\nThe LAZY dog.\n")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            normalized_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".arpa", delete=False) as f:
            arpa_file = f.name

        # Step 1: Normalize
        result1 = subprocess.run(
            [
                sys.executable,
                "-m",
                "arpabo.cli_normalize",
                input_file,
                "-o",
                normalized_file,
                "-c",
                "lower",
                "-n",
                "--add-markers",
            ],
            capture_output=True,
            text=True,
        )
        assert result1.returncode == 0

        # Step 2: Build LM
        result2 = subprocess.run(
            [sys.executable, "-m", "arpabo.cli", normalized_file, "-o", arpa_file],
            capture_output=True,
            text=True,
        )
        assert result2.returncode == 0

        # Verify ARPA file created
        import os

        assert os.path.exists(arpa_file)

        with open(arpa_file) as f:
            content = f.read()
            assert "\\data\\" in content

        os.unlink(input_file)
        os.unlink(normalized_file)
        os.unlink(arpa_file)
