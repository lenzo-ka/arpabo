# arpa-lm

Build ARPA format statistical language models with multiple smoothing methods.

[![Tests](https://github.com/lenzo-ka/arpa-lm/workflows/Tests/badge.svg)](https://github.com/lenzo-ka/arpa-lm/actions)
[![Lint](https://github.com/lenzo-ka/arpa-lm/workflows/Lint/badge.svg)](https://github.com/lenzo-ka/arpa-lm/actions)
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
pip install arpa-lm
```

This installs two commands:
- `arpa-lm` - Build language models
- `arpa-lm-normalize` - Normalize text corpora

## Quick Start

```bash
# Quick demo
arpa-lm --demo -o model.arpa

# Build from your corpus
arpa-lm corpus.txt -o model.arpa

# With binary conversion
arpa-lm corpus.txt -o model.arpa --to-bin

# Two-stage: normalize then build
arpa-lm-normalize corpus.txt -o normalized.txt -c lower -n
arpa-lm normalized.txt -o model.arpa
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
arpa-lm corpus.txt -o model.arpa
```

### With Options

```bash
# 4-gram with Kneser-Ney smoothing
arpa-lm corpus.txt -o model.arpa -m 4 -s kneser_ney

# Lowercase normalization
arpa-lm corpus.txt -o model.arpa -c lower -v

# Token normalization (strip punctuation)
arpa-lm corpus.txt -o model.arpa -n
```

### Corpus Preprocessing

```bash
# Normalize separately
arpa-lm-normalize corpus.txt -o clean.txt -c lower -n

# Build model
arpa-lm clean.txt -o model.arpa

# Or pipeline
cat corpus.txt | arpa-lm-normalize -c lower -n | arpa-lm -o model.arpa
```

### Binary Conversion

```bash
# Automatic PocketSphinx binary
arpa-lm corpus.txt -o model.arpa --to-bin

# Kaldi FST format
arpa-lm corpus.txt -o model.arpa --to-fst

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
git clone https://github.com/lenzo-ka/arpa-lm.git
cd arpa-lm
make venv
source venv/bin/activate
make test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT
