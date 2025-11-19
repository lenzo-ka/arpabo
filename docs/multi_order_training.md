# Multi-Order Training

Train multiple n-gram orders from the same corpus in a single command.

## Overview

When experimenting with different n-gram orders for first-pass ASR decoding or other applications, you often need to train models of different orders (e.g., unigram, bigram, trigram) from the same corpus. The `--orders` flag makes this efficient by reading and counting the corpus only once, then generating multiple models with different order configurations.

## Usage

### Basic Syntax

```bash
arpabo corpus.txt -o output_dir/ --orders ORDER_SPEC
```

Where `ORDER_SPEC` can be:
- **Single order**: `3` → train only a 3-gram
- **Comma-separated**: `1,2,4` → train 1-gram, 2-gram, and 4-gram
- **Range**: `1-4` → train 1-gram through 4-gram (expands to 1,2,3,4)
- **Mixed**: `1-3,5,7-10` → train 1,2,3,5,7,8,9,10-grams

### Examples

#### Train 1-gram through 4-gram

```bash
arpabo corpus.txt -o models/ --orders 1-4
```

Output:
```
models/
├── 1gram.arpa
├── 2gram.arpa
├── 3gram.arpa
└── 4gram.arpa
```

#### Train specific orders

```bash
arpabo corpus.txt -o models/ --orders 1,3,5
```

Output:
```
models/
├── 1gram.arpa
├── 3gram.arpa
└── 5gram.arpa
```

#### Mixed syntax

```bash
arpabo corpus.txt -o models/ --orders 1-3,5,7-10
```

Output:
```
models/
├── 1gram.arpa
├── 2gram.arpa
├── 3gram.arpa
├── 5gram.arpa
├── 7gram.arpa
├── 8gram.arpa
├── 9gram.arpa
└── 10gram.arpa
```

### With Demo Corpus

```bash
arpabo --demo -o models/ --orders 1-3 --verbose
```

### With Smoothing Method

```bash
arpabo corpus.txt -o models/ --orders 1-4 --smoothing-method kneser_ney
```

### With Binary Conversion

```bash
arpabo corpus.txt -o models/ --orders 1-3 --to-bin
```

Output:
```
models/
├── 1gram.arpa
├── 1gram.lm.bin
├── 2gram.arpa
├── 2gram.lm.bin
├── 3gram.arpa
└── 3gram.lm.bin
```

## Python API

### Basic Usage

```python
from arpabo import ArpaBoLM

# Create model with max order needed
lm = ArpaBoLM(max_order=4)

# Read corpus
with open("corpus.txt") as f:
    lm.read_corpus(f)

# Train multiple orders
models = lm.compute_multiple_orders([1, 2, 3, 4])

# Write each model
for order, model in models.items():
    with open(f"{order}gram.arpa", "w") as f:
        model.write(f)
```

### With Different Smoothing

```python
from arpabo import ArpaBoLM

lm = ArpaBoLM(max_order=4, smoothing_method="kneser_ney")
with open("corpus.txt") as f:
    lm.read_corpus(f)

models = lm.compute_multiple_orders([1, 2, 3, 4])

for order, model in models.items():
    model.write_file(f"{order}gram.arpa")
```

### Using Order Specification Parser

```python
from arpabo.utils import parse_order_spec
from arpabo import ArpaBoLM

# Parse order specification string
orders = parse_order_spec("1-3,5,7-10")
# Returns: [1, 2, 3, 5, 7, 8, 9, 10]

# Train those orders
lm = ArpaBoLM(max_order=max(orders))
with open("corpus.txt") as f:
    lm.read_corpus(f)

models = lm.compute_multiple_orders(orders)
```

## Benefits

1. **Efficiency**: Corpus is read and counted only once
2. **Consistency**: All models trained from identical corpus statistics
3. **Convenience**: Single command to generate multiple models
4. **Flexibility**: Supports all smoothing methods and conversion formats

## Use Cases

### First-Pass ASR Optimization

When optimizing PocketSphinx for first-pass decoding before LLM rescoring:

```bash
# Train multiple orders to evaluate oracle WER
arpabo corpus.txt -o experiments/ --orders 1-4 --to-bin

# Test each in decoder to find optimal configuration
```

### Model Size vs Quality Trade-off

```bash
# Compare model sizes and quality metrics
arpabo corpus.txt -o comparison/ --orders 1,2,3,4,5 --verbose

# Models increase in size with order
ls -lh comparison/
```

### Domain-Specific Tuning

```bash
# Train with different smoothing for comparison
arpabo domain_corpus.txt -o domain_models/ --orders 2-4 -s kneser_ney
```

## Requirements

- The `-o` flag must specify a directory (will be created if it doesn't exist)
- The `max_order` of the base model must be >= the highest requested order
- All orders must be positive integers (>= 1)

## Error Handling

```bash
# Missing output directory
arpabo corpus.txt --orders 1,2
# Error: --orders requires -o to specify output directory

# Order exceeds max_order
# Python: ArpaBoLM(max_order=3).compute_multiple_orders([1, 2, 5])
# Error: Requested order 5 exceeds corpus max_order 3

# Invalid syntax
arpabo corpus.txt -o models/ --orders 5-3
# Error: Invalid range: 5-3 (start must be <= end)
```

## Performance

Training multiple orders is significantly faster than running arpabo multiple times:

```bash
# Slow: Run 4 times, read corpus 4 times
arpabo corpus.txt -o 1gram.arpa -m 1
arpabo corpus.txt -o 2gram.arpa -m 2
arpabo corpus.txt -o 3gram.arpa -m 3
arpabo corpus.txt -o 4gram.arpa -m 4

# Fast: Read corpus once
arpabo corpus.txt -o models/ --orders 1-4
```

For a typical corpus (100k sentences), this can save 75% of the total runtime.
