"""Binary format conversion utilities

This module provides helper functions to convert ARPA models to binary formats
using external tools (PocketSphinx, Kaldi, etc.)
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional


class ConversionError(Exception):
    """Raised when conversion to binary format fails"""

    pass


def find_pocketsphinx_converter() -> Optional[str]:
    """Find pocketsphinx_lm_convert executable.

    Returns:
        Path to executable, or None if not found
    """
    # Try system PATH first
    path = shutil.which("pocketsphinx_lm_convert")
    if path:
        return path

    # Try common build locations
    common_paths = [
        "~/dev/cmu/pocketsphinx/build/pocketsphinx_lm_convert",
        "~/pocketsphinx/build/pocketsphinx_lm_convert",
        "/usr/local/bin/pocketsphinx_lm_convert",
    ]

    for path_str in common_paths:
        path = Path(path_str).expanduser()
        if path.exists() and path.is_file():
            return str(path)

    return None


def to_pocketsphinx_binary(arpa_path: str, bin_path: Optional[str] = None, verbose: bool = False) -> str:
    """Convert ARPA model to PocketSphinx binary format.

    Args:
        arpa_path: Path to ARPA format file
        bin_path: Output path (default: replace .arpa with .lm.bin)
        verbose: Print conversion progress

    Returns:
        Path to generated binary file

    Raises:
        ConversionError: If conversion fails
        FileNotFoundError: If pocketsphinx_lm_convert not found
    """
    converter = find_pocketsphinx_converter()
    if not converter:
        raise FileNotFoundError("pocketsphinx_lm_convert not found. Install PocketSphinx or add to PATH.")

    if bin_path is None:
        bin_path = arpa_path.replace(".arpa", ".lm.bin")

    if verbose:
        print(f"Converting {arpa_path} to {bin_path}...")

    try:
        result = subprocess.run(
            [converter, "-i", arpa_path, "-o", bin_path],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise ConversionError(f"Conversion failed: {result.stderr or result.stdout}")

        if verbose:
            print(f"Successfully created {bin_path}")

        return bin_path

    except subprocess.TimeoutExpired as e:
        raise ConversionError("Conversion timed out after 5 minutes") from e
    except Exception as e:
        raise ConversionError(f"Conversion failed: {e}") from e


def find_kaldi_arpa2fst() -> Optional[str]:
    """Find Kaldi's arpa2fst executable.

    Returns:
        Path to executable, or None if not found
    """
    path = shutil.which("arpa2fst")
    if path:
        return path

    # Try common Kaldi locations
    common_paths = [
        "~/kaldi/src/lmbin/arpa2fst",
        "~/dev/kaldi/src/lmbin/arpa2fst",
    ]

    for path_str in common_paths:
        path = Path(path_str).expanduser()
        if path.exists() and path.is_file():
            return str(path)

    return None


def to_kaldi_fst(arpa_path: str, fst_path: Optional[str] = None, verbose: bool = False) -> str:
    """Convert ARPA model to Kaldi FST format.

    Args:
        arpa_path: Path to ARPA format file
        fst_path: Output path (default: replace .arpa with .fst)
        verbose: Print conversion progress

    Returns:
        Path to generated FST file

    Raises:
        ConversionError: If conversion fails
        FileNotFoundError: If arpa2fst not found
    """
    converter = find_kaldi_arpa2fst()
    if not converter:
        raise FileNotFoundError("arpa2fst not found. Install Kaldi or add to PATH.")

    if fst_path is None:
        fst_path = arpa_path.replace(".arpa", ".fst")

    if verbose:
        print(f"Converting {arpa_path} to Kaldi FST {fst_path}...")

    try:
        result = subprocess.run(
            [converter, arpa_path, fst_path],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise ConversionError(f"Conversion failed: {result.stderr or result.stdout}")

        if verbose:
            print(f"Successfully created {fst_path}")

        return fst_path

    except subprocess.TimeoutExpired as e:
        raise ConversionError("Conversion timed out after 5 minutes") from e
    except Exception as e:
        raise ConversionError(f"Conversion failed: {e}") from e


def check_conversion_tools() -> dict[str, bool]:
    """Check which conversion tools are available.

    Returns:
        Dict mapping tool name to availability
    """
    return {
        "pocketsphinx_lm_convert": find_pocketsphinx_converter() is not None,
        "arpa2fst": find_kaldi_arpa2fst() is not None,
    }
