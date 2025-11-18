"""Tests for binary format conversion"""

import os
import tempfile
from io import StringIO

import pytest

from arpabo import ArpaBoLM, ConversionError, check_conversion_tools, to_pocketsphinx_binary


class TestConversionTools:
    """Test conversion tool detection"""

    def test_check_tools(self):
        tools = check_conversion_tools()
        assert isinstance(tools, dict)
        assert "pocketsphinx_lm_convert" in tools
        assert "arpa2fst" in tools
        assert isinstance(tools["pocketsphinx_lm_convert"], bool)


class TestPocketSphinxConversion:
    """Test PocketSphinx binary conversion"""

    def test_to_binary_if_available(self):
        """Test binary conversion if PocketSphinx is available"""
        tools = check_conversion_tools()

        if not tools["pocketsphinx_lm_convert"]:
            pytest.skip("pocketsphinx_lm_convert not available")

        # Create ARPA model
        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO("the quick brown fox\nthe lazy dog"))
        lm.compute()

        with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as f:
            arpa_path = f.name

        lm.write_file(arpa_path)

        # Convert to binary
        bin_path = to_pocketsphinx_binary(arpa_path, verbose=False)

        assert os.path.exists(bin_path)
        assert bin_path.endswith(".lm.bin")
        assert os.path.getsize(bin_path) > 0

        # Cleanup
        os.unlink(arpa_path)
        os.unlink(bin_path)

    def test_custom_output_path(self):
        """Test custom binary output path"""
        tools = check_conversion_tools()

        if not tools["pocketsphinx_lm_convert"]:
            pytest.skip("pocketsphinx_lm_convert not available")

        lm = ArpaBoLM(verbose=False)
        lm.read_corpus(StringIO("hello world"))
        lm.compute()

        with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as f:
            arpa_path = f.name

        lm.write_file(arpa_path)

        custom_bin = "/tmp/custom_test.lm.bin"
        bin_path = to_pocketsphinx_binary(arpa_path, custom_bin, verbose=False)

        assert bin_path == custom_bin
        assert os.path.exists(custom_bin)

        # Cleanup
        os.unlink(arpa_path)
        os.unlink(custom_bin)

    def test_conversion_error_handling(self):
        """Test error handling for invalid input"""
        tools = check_conversion_tools()

        if not tools["pocketsphinx_lm_convert"]:
            pytest.skip("pocketsphinx_lm_convert not available")

        # Try to convert non-existent file
        with pytest.raises(ConversionError):
            to_pocketsphinx_binary("/nonexistent/file.arpa")


class TestCLIConversion:
    """Test CLI conversion flags"""

    def test_cli_to_bin_flag(self):
        """Test --to-bin flag"""
        import subprocess
        import sys

        tools = check_conversion_tools()
        if not tools["pocketsphinx_lm_convert"]:
            pytest.skip("pocketsphinx_lm_convert not available")

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
                "--to-bin",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        bin_path = arpa_path.replace(".arpa", ".lm.bin")
        assert os.path.exists(bin_path)

        # Cleanup
        os.unlink(arpa_path)
        os.unlink(bin_path)
