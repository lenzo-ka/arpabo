# Test Data

This directory contains test data files that are not packaged with the module.

## Files

- **kevin-alice-16k.wav** - Audio recording for ASR testing
  - Used to test real decoding with Alice language model
  - 16kHz sampling rate (standard for ASR)
  - Not included in PyPI package (too large)

## Usage

### Real ASR Decode Test

```bash
# From project root
python tests/test_asr_decode.py
```

This will:
1. Build language model from Alice corpus
2. Convert to PocketSphinx binary format
3. Perform real speech recognition on the audio
4. Display recognition results

### Manual Decode

```bash
# Build Alice LM
arpalm --demo -o alice.arpa --to-bin

# Decode with PocketSphinx (requires PocketSphinx installation)
pocketsphinx_continuous \
    -lm alice.lm.bin \
    -infile tests/data/kevin-alice-16k.wav \
    -hmm <path-to-acoustic-model>
```

## Adding Test Data

To add new test audio:
1. Place files in this directory
2. Use 16kHz sampling rate (standard)
3. Update `.gitignore` pattern if needed
4. Keep files under 1MB if possible

## Note

Test data files are excluded from the PyPI package to keep it lightweight.
Only the Alice text corpus is packaged for demos.
