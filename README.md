# arpabo

Build ARPA format statistical language models with multiple smoothing methods.

[![Tests](https://github.com/lenzo-ka/arpabo/workflows/Tests/badge.svg)](https://github.com/lenzo-ka/arpabo/actions)
[![Lint](https://github.com/lenzo-ka/arpabo/workflows/Lint/badge.svg)](https://github.com/lenzo-ka/arpabo/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

### Core Features
- Multiple smoothing methods (Good-Turing, Kneser-Ney, Katz backoff)
- Support for arbitrary n-gram orders
- Standard ARPA format output
- Binary format conversion (PocketSphinx, Kaldi)
- Corpus normalization tool
- Interactive debug mode
- Zero runtime dependencies (pure Python)

### New: First-Pass ASR Optimization
- **Multi-order training** - Train multiple n-gram orders efficiently (`--orders 1-4`)
- **Perplexity evaluation** - Evaluate model quality on test data (`--eval test.txt`)
- **Model statistics** - Analyze backoff rates and model behavior (`--stats --backoff test.txt`)
- **Presets** - Pre-configured settings for common use cases (`--preset first-pass`)
- **Smoothing comparison** - Automatically compare smoothing methods (`--compare-smoothing`)
- **Vocabulary pruning** - Reduce model size for mobile (`--prune-vocab topk:10000`)
- **ModelComparison API** - High-level Python API for complete workflows
- **Uniform baseline** - Maximum entropy models for comparison
- **Cross-validation** - K-fold CV for robust model selection
- **Model interpolation** - Alternative probability mixing strategy

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

### Basic Usage

```python
from arpabo import ArpaBoLM

# Build a language model
lm = ArpaBoLM(max_order=3, smoothing_method="good_turing")
with open("corpus.txt") as f:
    lm.read_corpus(f)
lm.compute()
lm.write_file("model.arpa")
```

### Use Presets (New!)

```python
# No need to pick parameters - use a preset!
lm = ArpaBoLM.from_preset("first-pass")  # For first-pass ASR
lm.read_corpus(open("corpus.txt"))
lm.compute()
lm.write_file("model.arpa")
```

### High-Level API (New!)

```python
from arpabo import ModelComparison

# Complete optimization workflow
comparison = ModelComparison(corpus_file="train.txt")
comparison.train_orders([1, 2, 3, 4])
comparison.add_uniform_baseline()
comparison.evaluate(test_file="test.txt")
comparison.print_comparison()

# Get recommendation
best = comparison.recommend(goal="first-pass")
print(f"Best model: {best}-gram")

# Export for deployment
comparison.export_for_optimization("experiments/", convert_to_binary=True)
```

### Evaluation & Analysis (New!)

```python
# Evaluate model quality
with open("test.txt") as f:
    results = lm.perplexity(f)
print(f"Perplexity: {results['perplexity']:.1f}")

# Analyze backoff behavior
with open("test.txt") as f:
    backoff = lm.backoff_rate(f)
print(f"Backoff rate: {backoff['overall_backoff_rate']*100:.1f}%")

# Get model statistics
stats = lm.get_statistics()
print(f"Vocabulary: {stats['vocab_size']:,} words")
```

## Smoothing Methods

- `good_turing` (default) - Best for sparse data
- `kneser_ney` - Best for larger corpora
- `auto` - Automatically optimizes discount mass
- `fixed` - Fixed discount mass (use `-d 0.0` for MLE)

## Common Workflows

### Basic Usage

```bash
# Simple model
arpabo corpus.txt -o model.arpa

# Use a preset (easiest!)
arpabo corpus.txt -o model.arpa --preset balanced

# List available presets
arpabo --list-presets
```

### Multi-Order Training (New!)

```bash
# Train multiple orders efficiently
arpabo corpus.txt -o models/ --orders 1-4 --to-bin

# Creates: 1gram.arpa, 2gram.arpa, 3gram.arpa, 4gram.arpa (+ .lm.bin files)
```

### Model Evaluation (New!)

```bash
# Train and evaluate
arpabo corpus.txt -o model.arpa --eval test.txt

# Evaluate existing model
arpabo --eval-only model.arpa test.txt

# With statistics and backoff analysis
arpabo corpus.txt -o model.arpa --stats --backoff test.txt
```

### Advanced: Compare & Optimize (New!)

```bash
# Compare smoothing methods
arpabo corpus.txt --compare-smoothing --eval test.txt

# Prune for mobile deployment
arpabo corpus.txt -o mobile.arpa --prune-vocab topk:10000 --to-bin
```

### Traditional Options

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

### Binary Conversion (Optional)

ARPA files work directly with PocketSphinx. Binary conversion is optional for better performance:

```bash
# Use ARPA directly (works as-is)
arpabo corpus.txt -o model.arpa

# Optional: Convert to binary for faster loading
arpabo corpus.txt -o model.arpa --to-bin

# Optional: Kaldi FST format
arpabo corpus.txt -o model.arpa --to-fst

# Or convert manually later
pocketsphinx_lm_convert -i model.arpa -o model.lm.bin
```

## Compatibility

Produces standard ARPA format models that work directly with:

- **PocketSphinx** - Use ARPA directly (optional binary conversion for speed)
- **Kaldi** - Use ARPA directly or convert to FST
- **SphinxTrain** - Use ARPA directly
- **NVIDIA Riva** - ARPA format supported
- **Julius**, **HTK** - ARPA compatible

Binary conversion is optional and only improves loading speed.

## Documentation

### Guides
- [Multi-Order Training](docs/multi_order_training.md) - Train multiple models efficiently
- [Perplexity Evaluation](docs/perplexity_evaluation.md) - Evaluate model quality
- [ModelComparison API](docs/model_comparison.md) - High-level workflow API

### Examples
- [examples/model_comparison_example.py](examples/model_comparison_example.py) - Complete workflow example

### Feature Summaries
See `PHASE_1_COMPLETE.md`, `PHASE_2_COMPLETE.md`, and `PHASE_3_COMPLETE.md` for detailed feature documentation.

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
