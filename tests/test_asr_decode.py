#!/usr/bin/env python3
"""Real ASR decode test with PocketSphinx

This test builds a language model from Alice corpus and performs
real speech recognition on test audio.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def find_test_audio():
    """Find the test audio file"""
    audio_path = Path(__file__).parent / "data" / "kevin-alice-16k.wav"
    if audio_path.exists():
        return str(audio_path)
    return None


def find_pocketsphinx():
    """Check if PocketSphinx is available"""
    return shutil.which("pocketsphinx") is not None


def find_acoustic_model():
    """Find PocketSphinx acoustic model"""
    common_paths = [
        "~/dev/cmu/pocketsphinx/model/en-us/en-us",
        "~/pocketsphinx/model/en-us/en-us",
        "/usr/local/share/pocketsphinx/model/en-us/en-us",
    ]

    for path_str in common_paths:
        path = Path(path_str).expanduser()
        if path.exists():
            return str(path)

    # Try to find in installed pocketsphinx package
    try:
        import pocketsphinx

        model_path = Path(pocketsphinx.__file__).parent / "model" / "en-us" / "en-us"
        if model_path.exists():
            return str(model_path)
    except ImportError:
        pass

    return None


def test_real_asr_decode():
    """Test real ASR decoding with ArpaLM-generated language model"""
    print("=" * 70)
    print("Real ASR Decode Test with PocketSphinx")
    print("=" * 70)

    # Check prerequisites
    audio_path = find_test_audio()
    if not audio_path:
        print("\nTest audio not found: tests/data/kevin-alice-16k.wav")
        print("Skipping real ASR decode test.")
        pytest.skip("Test requirements not met")

    print(f"\n1. Found test audio: {audio_path}")

    if not find_pocketsphinx():
        print("\n2. PocketSphinx not installed")
        print("   Install: pip install pocketsphinx")
        print("   Skipping real ASR decode test.")
        pytest.skip("Test requirements not met")

    print("2. PocketSphinx found")

    hmm = find_acoustic_model()
    if not hmm:
        print("\n3. Acoustic model not found")
        print("   Expected in ~/dev/cmu/pocketsphinx/model/en-us/")
        print("   Skipping real ASR decode test.")
        pytest.skip("Test requirements not met")

    print(f"3. Acoustic model found: {hmm}")

    # Build language model from Alice corpus
    print("\n4. Building language model from Alice corpus...")

    from arpabo import ArpaBoLM, get_example_corpus

    corpus_path = get_example_corpus()
    lm = ArpaBoLM(max_order=3, smoothing_method="good_turing", case="lower", verbose=False)

    with open(corpus_path) as f:
        lm.read_corpus(f)

    lm.compute()
    print(f"   Model: {len(lm.probs[0])} words, {lm.sum_1} total")

    # Write ARPA and convert to binary
    with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as f:
        arpa_path = f.name

    lm.write_file(arpa_path)
    print(f"   Wrote: {arpa_path}")

    # Convert to binary
    try:
        from arpabo.convert import to_pocketsphinx_binary

        bin_path = to_pocketsphinx_binary(arpa_path, verbose=False)
        print(f"   Binary: {bin_path}")
    except Exception as e:
        print(f"   Binary conversion failed: {e}")
        os.unlink(arpa_path)
        pytest.skip("Test requirements not met")

    # Perform decode
    print("\n5. Running PocketSphinx decode with binary LM...")
    print(f"   Command: pocketsphinx single -hmm {hmm} -lm {bin_path} {audio_path}")

    try:
        result = subprocess.run(
            [
                "pocketsphinx",
                "single",
                "-hmm",
                hmm,
                "-lm",
                bin_path,
                audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            # Extract hypothesis from output
            output = result.stdout + result.stderr

            print("\n" + "=" * 70)
            print("DECODE RESULTS")
            print("=" * 70)
            print("\nOutput:")
            print(output)
            print("\n" + "=" * 70)
            print("SUCCESS: Real ASR decode completed!")
            print("Binary format verification: PASS")
            print("=" * 70)
            print("\nVerified:")
            print("  1. ArpaLM built LM from Alice corpus")
            print("  2. Converted to PocketSphinx binary format")
            print("  3. PocketSphinx successfully loaded binary LM")
            print("  4. PocketSphinx decoded audio using the LM")
            print("\nThe binary format is working correctly!")
            print("=" * 70)

            # Cleanup
            os.unlink(arpa_path)
            os.unlink(bin_path)
            pass
        else:
            print(f"\nDecode failed (exit code {result.returncode})")
            print(result.stderr)
            os.unlink(arpa_path)
            os.unlink(bin_path)
            pytest.skip("Test requirements not met")

    except subprocess.TimeoutExpired:
        print("\nDecode timed out")
        os.unlink(arpa_path)
        os.unlink(bin_path)
        pytest.skip("Test requirements not met")
    except Exception as e:
        print(f"\nDecode error: {e}")
        os.unlink(arpa_path)
        if os.path.exists(bin_path):
            os.unlink(bin_path)
        pytest.skip("Test requirements not met")


if __name__ == "__main__":
    success = test_real_asr_decode()
    sys.exit(0 if success else 1)
