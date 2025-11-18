# ArpaLM

Build ARPA format statistical language models with multiple smoothing methods.

[![Tests](https://github.com/lenzo-ka/arpalm/workflows/Tests/badge.svg)](https://github.com/lenzo-ka/arpalm/actions)
[![Lint](https://github.com/lenzo-ka/arpalm/workflows/Lint/badge.svg)](https://github.com/lenzo-ka/arpalm/actions)
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
pip install arpalm
```

This installs two commands:
- `arpalm` - Build language models
- `arpalm-normalize` - Normalize text corpora

## Quick Start

```bash
# Quick demo
arpalm --demo -o model.arpa

# Build from your corpus
arpalm corpus.txt -o model.arpa

# With binary conversion
arpalm corpus.txt -o model.arpa --to-bin

# Two-stage: normalize then build
arpalm-normalize corpus.txt -o normalized.txt -c lower -n
arpalm normalized.txt -o model.arpa
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
arpalm corpus.txt -o model.arpa
```

### With Options

```bash
# 4-gram with Kneser-Ney smoothing
arpalm corpus.txt -o model.arpa -m 4 -s kneser_ney

# Lowercase normalization
arpalm corpus.txt -o model.arpa -c lower -v

# Token normalization (strip punctuation)
arpalm corpus.txt -o model.arpa -n
```

### Corpus Preprocessing

```bash
# Normalize separately
arpalm-normalize corpus.txt -o clean.txt -c lower -n

# Build model
arpalm clean.txt -o model.arpa

# Or pipeline
cat corpus.txt | arpalm-normalize -c lower -n | arpalm -o model.arpa
```

### Binary Conversion

```bash
# Automatic PocketSphinx binary
arpalm corpus.txt -o model.arpa --to-bin

# Kaldi FST format
arpalm corpus.txt -o model.arpa --to-fst

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
git clone https://github.com/lenzo-ka/arpalm.git
cd arpalm
make venv
source venv/bin/activate
make test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT
