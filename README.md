# arpabo

Build ARPA format statistical language models with multiple smoothing methods.

[![Tests](https://github.com/lenzo-ka/arpabo/workflows/Tests/badge.svg)](https://github.com/lenzo-ka/arpabo/actions)
[![Lint](https://github.com/lenzo-ka/arpabo/workflows/Lint/badge.svg)](https://github.com/lenzo-ka/arpabo/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- Multiple smoothing methods (Good-Turing, Kneser-Ney, Katz backoff)
- Support for arbitrary n-gram orders
- Standard ARPA format output
- Binary format conversion (PocketSphinx, Kaldi)
- Corpus normalization tool
- Interactive debug mode
- Zero runtime dependencies (pure Python)

## Installation

```bash
pip install arpabo
```

This installs two commands:
- `arpabo` - Build language models
- `arpabo-normalize` - Normalize text corpora

## Quick Start

```bash
# Quick demo
arpabo --demo -o model.arpa

# Build from your corpus
arpabo corpus.txt -o model.arpa

# With binary conversion
arpabo corpus.txt -o model.arpa --to-bin

# Two-stage: normalize then build
arpabo-normalize corpus.txt -o normalized.txt -c lower -n
arpabo normalized.txt -o model.arpa
```

## Python API

```python
from arpalm import ArpaBoLM

# Build a language model
lm = ArpaBoLM(max_order=3, smoothing_method="good_turing")
with open("corpus.txt") as f:
    lm.read_corpus(f)
lm.compute()
lm.write_file("model.arpa")
```

## Smoothing Methods

- `good_turing` (default) - Best for sparse data
- `kneser_ney` - Best for larger corpora
- `auto` - Automatically optimizes discount mass
- `fixed` - Fixed discount mass (use `-d 0.0` for MLE)

## Common Workflows

### Basic Usage

```bash
arpabo corpus.txt -o model.arpa
```

### With Options

```bash
# 4-gram with Kneser-Ney smoothing
arpabo corpus.txt -o model.arpa -m 4 -s kneser_ney

# Lowercase normalization
arpabo corpus.txt -o model.arpa -c lower -v

# Token normalization (strip punctuation)
arpabo corpus.txt -o model.arpa -n
```

### Corpus Preprocessing

```bash
# Normalize separately
arpabo-normalize corpus.txt -o clean.txt -c lower -n

# Build model
arpabo clean.txt -o model.arpa

# Or pipeline
cat corpus.txt | arpabo-normalize -c lower -n | arpabo -o model.arpa
```

### Binary Conversion

```bash
# Automatic PocketSphinx binary
arpabo corpus.txt -o model.arpa --to-bin

# Kaldi FST format
arpabo corpus.txt -o model.arpa --to-fst

# Manual conversion
pocketsphinx_lm_convert -i model.arpa -o model.lm.bin
```

## Compatibility

ArpaLM produces standard ARPA format models compatible with:

- **Kaldi** - Convert with `arpa2fst`
- **PocketSphinx** - Convert with `pocketsphinx_lm_convert`
- **SphinxTrain** - Use ARPA directly
- **NVIDIA Riva** - ARPA format supported
- **Julius**, **HTK** - ARPA compatible

## Development

```bash
git clone https://github.com/lenzo-ka/arpabo.git
cd arpabo
make venv
source venv/bin/activate
make test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT
