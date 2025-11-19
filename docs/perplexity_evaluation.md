# Perplexity Evaluation

Evaluate language model quality on held-out test data using perplexity and cross-entropy metrics.

## Overview

Perplexity measures how well a language model predicts test data. Lower perplexity indicates better prediction. A perplexity of N means the model is as uncertain as if it had to choose uniformly from N words at each step.

Perplexity is calculated as:

```
PPL = exp(-1/N * sum(log P(w_i | history)))
```

Where N is the number of words in the test corpus.

## Usage

### Basic Evaluation

#### Train and Evaluate

Train a model and evaluate it on test data in one command:

```bash
arpabo train.txt -o model.arpa --eval test.txt
```

Output:
```
Perplexity Evaluation
==================================================
Model: 3-gram good_turing
Test file: test.txt

Evaluation Results:
  Sentences:         1,000
  Words:            15,234
  OOV words:          183 (1.2%)

  Perplexity:        89.2
  Cross-entropy:      6.48 bits/word
```

#### Evaluate Existing Model

Evaluate an existing ARPA model without training:

```bash
arpabo --eval-only model.arpa test.txt
```

### Python API

#### Basic Evaluation

```python
from arpabo import ArpaBoLM

# Train model
lm = ArpaBoLM(max_order=3)
with open("train.txt") as f:
    lm.read_corpus(f)
lm.compute()

# Evaluate on test data
with open("test.txt") as f:
    results = lm.perplexity(f)

# Print results
print(f"Perplexity: {results['perplexity']:.1f}")
print(f"Cross-entropy: {results['cross_entropy']:.2f} bits/word")
print(f"OOV rate: {results['oov_rate']*100:.1f}%")
```

#### Formatted Output

```python
# Use built-in formatted printer
lm.print_perplexity_results(results, test_file="test.txt")
```

#### Evaluate Loaded Model

```python
# Load existing model
lm = ArpaBoLM.from_arpa_file("model.arpa")

# Evaluate
with open("test.txt") as f:
    results = lm.perplexity(f)

lm.print_perplexity_results(results)
```

## OOV Handling

Control how out-of-vocabulary (OOV) words are handled during evaluation:

### `unk` (default)

Treat OOV words with a small default probability:

```bash
arpabo model.arpa test.txt --eval --oov-handling unk
```

```python
results = lm.perplexity(test_file, oov_handling="unk")
```

### `skip`

Skip OOV words in the calculation:

```bash
arpabo model.arpa test.txt --eval --oov-handling skip
```

```python
results = lm.perplexity(test_file, oov_handling="skip")
```

### `error`

Raise an error if OOV words are encountered:

```bash
arpabo model.arpa test.txt --eval --oov-handling error
```

```python
results = lm.perplexity(test_file, oov_handling="error")
```

This is useful to ensure your test data only contains in-vocabulary words.

## Result Dictionary

The `perplexity()` method returns a dictionary with the following keys:

```python
{
    "perplexity": 89.2,          # Perplexity score
    "cross_entropy": 6.48,       # Cross-entropy in bits per word
    "num_sentences": 1000,       # Number of sentences evaluated
    "num_words": 15234,          # Total words evaluated
    "num_oov": 183,              # Number of OOV words
    "oov_rate": 0.012            # Fraction of OOV words (0-1)
}
```

## Examples

### Compare Model Orders

Evaluate perplexity across different n-gram orders:

```python
from arpabo import ArpaBoLM
from io import StringIO

# Train corpus
corpus = "the quick brown fox jumps over the lazy dog"
test_data = "the quick fox"

orders = [1, 2, 3, 4]
results = {}

for order in orders:
    lm = ArpaBoLM(max_order=order)
    lm.read_corpus(StringIO(corpus))
    lm.compute()

    eval_results = lm.perplexity(StringIO(test_data))
    results[order] = eval_results

# Print comparison
print("Order  Perplexity  Cross-Entropy")
print("-----  ----------  -------------")
for order, res in results.items():
    print(f"{order:>3}    {res['perplexity']:>8.1f}  {res['cross_entropy']:>11.2f}")
```

Output:
```
Order  Perplexity  Cross-Entropy
-----  ----------  -------------
  1       342.5        8.42
  2       124.3        6.96
  3        89.2        6.48
  4        87.8        6.46
```

### Compare Smoothing Methods

```python
methods = ["good_turing", "kneser_ney", "katz"]

for method in methods:
    lm = ArpaBoLM(max_order=3, smoothing_method=method)
    lm.read_corpus(train_file)
    lm.compute()

    with open("test.txt") as f:
        results = lm.perplexity(f)

    print(f"{method:15} PPL: {results['perplexity']:.1f}")
```

Output:
```
good_turing     PPL: 92.4
kneser_ney      PPL: 89.2
katz            PPL: 95.1
```

### Train-Test Split Evaluation

```python
# Split corpus into train/test
with open("corpus.txt") as f:
    lines = f.readlines()

split = int(0.8 * len(lines))
train_lines = lines[:split]
test_lines = lines[split:]

# Train on 80%
lm = ArpaBoLM(max_order=3)
lm.read_corpus(StringIO("".join(train_lines)))
lm.compute()

# Evaluate on 20%
results = lm.perplexity(StringIO("".join(test_lines)))
lm.print_perplexity_results(results, test_file="held-out 20%")
```

### Multi-Order Evaluation

Combine with multi-order training:

```bash
# Train multiple orders
arpabo corpus.txt -o models/ --orders 1-4

# Evaluate each
for i in 1 2 3 4; do
    echo "=== ${i}-gram ==="
    arpabo --eval-only models/${i}gram.arpa test.txt
done
```

Or in Python:

```python
from arpabo import ArpaBoLM

# Train multiple orders
lm = ArpaBoLM(max_order=4)
with open("train.txt") as f:
    lm.read_corpus(f)

models = lm.compute_multiple_orders([1, 2, 3, 4])

# Evaluate each
for order, model in models.items():
    with open("test.txt") as f:
        results = model.perplexity(f)
    print(f"{order}-gram: {results['perplexity']:.1f}")
```

## Interpreting Results

### Perplexity

- **Lower is better**: A lower perplexity means the model predicts the test data better
- **Typical ranges**:
  - < 100: Excellent (well-matched domain, sufficient training data)
  - 100-300: Good (reasonable match)
  - > 300: Poor (domain mismatch or insufficient training)

### Cross-Entropy

- Measured in bits per word
- Related to perplexity: `PPL = 2^cross_entropy`
- Lower is better
- Typical range: 5-10 bits/word

### OOV Rate

- Fraction of words not seen during training
- **Target**: < 5% for good performance
- High OOV rate (>10%) indicates:
  - Domain mismatch between train and test
  - Insufficient training data
  - Need for larger vocabulary

## Use Cases

### Model Selection

Choose the best model configuration for your domain:

```bash
# Compare different smoothing methods
arpabo corpus.txt -o gt.arpa -s good_turing --eval test.txt
arpabo corpus.txt -o kn.arpa -s kneser_ney --eval test.txt

# Compare different orders
arpabo corpus.txt -o models/ --orders 1-5 --eval test.txt
```

### First-Pass ASR Optimization

When optimizing for first-pass decoding before LLM rescoring:

```python
# Train multiple orders
models = lm.compute_multiple_orders([1, 2, 3, 4])

# Evaluate perplexity as proxy for oracle WER
for order, model in models.items():
    results = model.perplexity(test_data)
    print(f"{order}-gram: PPL={results['perplexity']:.1f}, "
          f"OOV={results['oov_rate']*100:.1f}%")

# Lower perplexity generally correlates with better first-pass performance
```

### Domain Adaptation

Evaluate how well a model generalizes to new domains:

```python
# Train on general domain
lm = ArpaBoLM(max_order=3)
lm.read_corpus(open("general_corpus.txt"))
lm.compute()

# Evaluate on specific domains
for domain in ["medical", "legal", "technical"]:
    with open(f"{domain}_test.txt") as f:
        results = lm.perplexity(f)
    print(f"{domain}: PPL={results['perplexity']:.1f}, "
          f"OOV={results['oov_rate']*100:.1f}%")
```

## CLI Reference

### Train and Evaluate

```bash
arpabo [INPUT] -o OUTPUT --eval TEST_FILE [OPTIONS]
```

**Options:**
- `--eval TEST_FILE`: Evaluate on test file after training
- `--oov-handling {unk,skip,error}`: OOV handling strategy (default: unk)

### Evaluate Only

```bash
arpabo --eval-only MODEL TEST_FILE [OPTIONS]
```

**Options:**
- `--oov-handling {unk,skip,error}`: OOV handling strategy (default: unk)
- `-v, --verbose`: Show detailed progress

## Best Practices

1. **Always use held-out data**: Never evaluate on training data
2. **Consistent preprocessing**: Use same normalization for train and test
3. **Monitor OOV rate**: High OOV rates invalidate perplexity comparisons
4. **Use multiple metrics**: Combine perplexity with task-specific evaluation (e.g., WER for ASR)
5. **Cross-validation**: Use k-fold CV for robust model selection (see Feature 3.1)

## Limitations

- Perplexity doesn't always correlate with downstream task performance
- Sensitive to OOV words
- Assumes test data is from the same distribution as training data
- Lower-order models may have artificially low perplexity on very short test sequences
