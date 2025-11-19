# ModelComparison API

High-level API for comparing multiple n-gram configurations and finding optimal models for first-pass ASR optimization.

## Overview

The `ModelComparison` class provides a complete workflow API that orchestrates all Phase 1 features:
- Multi-order training (Feature 1.1)
- Perplexity evaluation (Feature 1.2)
- Statistics and backoff analysis (Feature 1.3)
- Uniform baseline comparison (Feature 1.4)

## Quick Start

```python
from arpabo import ModelComparison

# Create comparison
comparison = ModelComparison(corpus_file="train.txt")

# Train multiple orders
comparison.train_orders([1, 2, 3, 4])

# Add uniform baseline
comparison.add_uniform_baseline()

# Evaluate on test data
comparison.evaluate(test_file="test.txt")

# Print results
comparison.print_comparison()

# Get recommendation
best = comparison.recommend(goal="first-pass")
print(f"Best model: {best}-gram")

# Export everything
comparison.export_for_optimization(
    output_dir="experiments/",
    convert_to_binary=True
)
```

## API Reference

### Initialization

```python
ModelComparison(
    corpus_file: str,
    smoothing_method: str = "kneser_ney",
    verbose: bool = False
)
```

**Args:**
- `corpus_file`: Path to training corpus
- `smoothing_method`: Smoothing method for all models (default: kneser_ney)
- `verbose`: Enable verbose output

### Methods

#### `train_orders(orders: list[int]) -> dict[int, ArpaBoLM]`

Train multiple n-gram orders from the corpus.

```python
models = comparison.train_orders([1, 2, 3, 4])
# Returns: {1: model, 2: model, 3: model, 4: model}
```

#### `add_uniform_baseline() -> ArpaBoLM`

Add uniform language model as baseline for comparison.

```python
uniform = comparison.add_uniform_baseline()
# Automatically extracts vocabulary from trained models
```

Must be called after `train_orders()`.

#### `evaluate(test_file: str, include_backoff: bool = True) -> dict[int, dict]`

Evaluate all models on test data.

```python
results = comparison.evaluate(
    test_file="test.txt",
    include_backoff=True  # Include backoff analysis (slower)
)
# Returns: {order: {perplexity, cross_entropy, backoff_rate, ...}}
```

Must be called after `train_orders()`.

#### `recommend(goal: str = "first-pass", max_perplexity_increase: float = 0.05) -> int`

Recommend best n-gram order for specified goal.

```python
# For first-pass decoding (balance PPL vs diversity)
best = comparison.recommend(
    goal="first-pass",
    max_perplexity_increase=0.05  # Allow 5% worse PPL for simpler model
)

# For rescoring (minimize perplexity)
best = comparison.recommend(goal="rescoring")
```

**Goals:**
- `"first-pass"`: Balance perplexity vs simplicity, prefer good backoff rate
- `"rescoring"`: Minimize perplexity only

Must be called after `evaluate()`.

#### `export_for_optimization(output_dir: str, convert_to_binary: bool = True) -> str`

Export models in format ready for optimization frameworks.

```python
manifest_path = comparison.export_for_optimization(
    output_dir="ngram_experiments/",
    convert_to_binary=True  # Create .lm.bin files
)
```

Creates:
- `{order}gram.arpa` files
- `{order}gram.lm.bin` files (if convert_to_binary=True)
- `uniform.arpa` (if uniform baseline added)
- `manifest.json` with all metadata

Must be called after `train_orders()`.

#### `print_comparison() -> None`

Print formatted comparison table.

```python
comparison.print_comparison()
```

Output:
```
Model Comparison
======================================================================
Corpus: train.txt

Model           PPL  Entropy   OOV%  Backoff
----------------------------------------------------------------------
uniform    1335.0    10.38    1.2%     0.0%
1-gram      342.5     8.42    1.2%     0.0%
2-gram      124.3     6.96    1.2%    31.2%
3-gram       89.2     6.48    1.2%    42.3%
4-gram       87.8     6.46    1.2%    48.1%
```

#### `print_recommendation(goal: str = "first-pass", max_perplexity_increase: float = 0.05) -> None`

Print recommendation with explanation.

```python
comparison.print_recommendation(goal="first-pass")
```

Output:
```
Recommendation (first-pass): 3-gram
--------------------------------------------------
  Perplexity:        89.2
  Cross-entropy:      6.48 bits/word
  OOV rate:           1.2%
  Backoff rate:      42.3%

  → 1.6% higher PPL than best, but simpler model
  → Good backoff rate (42.3%) for diverse N-best lists
```

#### `get_model(order: int) -> ArpaBoLM`

Get trained model by order.

```python
model = comparison.get_model(2)  # Get bigram model
model = comparison.get_model(0)  # Get uniform baseline
```

#### `get_evaluation(order: int) -> dict`

Get evaluation results for specific order.

```python
eval_data = comparison.get_evaluation(2)
print(f"Bigram PPL: {eval_data['perplexity']:.1f}")
```

#### `list_models() -> list[int]`

List available model orders.

```python
orders = comparison.list_models()  # [0, 1, 2, 3, 4]
```

#### `summary() -> dict`

Get summary of all models and evaluations.

```python
summary = comparison.summary()
print(json.dumps(summary, indent=2))
```

## Complete Examples

### Basic Workflow

```python
from arpabo import ModelComparison

# Initialize
comparison = ModelComparison(corpus_file="train.txt")

# Train
comparison.train_orders([1, 2, 3, 4])

# Evaluate
comparison.evaluate(test_file="test.txt")

# Show results
comparison.print_comparison()

# Get recommendation
best = comparison.recommend(goal="first-pass")
print(f"Use {best}-gram for first-pass decoding")
```

### With Uniform Baseline

```python
comparison = ModelComparison(corpus_file="train.txt")

# Train models
comparison.train_orders([1, 2, 3, 4])

# Add uniform baseline for comparison
comparison.add_uniform_baseline()

# Evaluate everything
comparison.evaluate(test_file="test.txt")

# Compare all
comparison.print_comparison()

# Uniform should have highest (worst) perplexity
```

### Export for Grid Search

```python
from arpabo import ModelComparison

comparison = ModelComparison(corpus_file="corpus.txt", verbose=True)

# Train orders to test
comparison.train_orders([1, 2, 3, 4, 5])

# Evaluate
comparison.evaluate(test_file="test.txt")

# Export with binary conversion
comparison.export_for_optimization(
    output_dir="ngram_experiments/",
    convert_to_binary=True
)

# Creates:
# ngram_experiments/
# ├── 1gram.arpa
# ├── 1gram.lm.bin
# ├── 2gram.arpa
# ├── 2gram.lm.bin
# ├── ...
# └── manifest.json
```

### Compare Smoothing Methods

```python
methods = ["good_turing", "kneser_ney", "katz"]

for method in methods:
    comparison = ModelComparison(
        corpus_file="corpus.txt",
        smoothing_method=method
    )

    comparison.train_orders([3])  # Just trigrams
    comparison.evaluate(test_file="test.txt")

    results = comparison.get_evaluation(3)
    print(f"{method:15} PPL: {results['perplexity']:.1f}")
```

### Detailed Analysis

```python
from arpabo import ModelComparison

comparison = ModelComparison(corpus_file="train.txt")
comparison.train_orders([1, 2, 3, 4])
comparison.add_uniform_baseline()
comparison.evaluate(test_file="test.txt")

# Print comparison table
comparison.print_comparison()

# Get detailed recommendations
comparison.print_recommendation(goal="first-pass")
comparison.print_recommendation(goal="rescoring")

# Access individual model results
for order in comparison.list_models():
    if order == 0:
        continue  # Skip uniform for this

    eval_data = comparison.get_evaluation(order)
    model = comparison.get_model(order)
    stats = model.get_statistics()

    print(f"\n{order}-gram Details:")
    print(f"  Vocab: {stats['vocab_size']:,}")
    print(f"  N-grams: {stats['ngram_counts'][order]:,}")
    print(f"  PPL: {eval_data['perplexity']:.1f}")
    print(f"  Backoff: {eval_data['overall_backoff_rate']*100:.1f}%")
```

## Manifest Format

The `manifest.json` file created by `export_for_optimization()`:

```json
{
  "corpus": "train.txt",
  "smoothing": "kneser_ney",
  "vocab_size": 50000,
  "models": [
    {
      "order": 0,
      "file": "uniform.arpa",
      "size_mb": 2.1,
      "smoothing": "uniform",
      "perplexity": 50000.0,
      "cross_entropy": 15.61,
      "oov_rate": 0.012,
      "backoff_rate": 0.0,
      "binary": "uniform.lm.bin"
    },
    {
      "order": 1,
      "file": "1gram.arpa",
      "size_mb": 2.1,
      "smoothing": "kneser_ney",
      "perplexity": 342.5,
      "cross_entropy": 8.42,
      "oov_rate": 0.012,
      "backoff_rate": 0.0,
      "binary": "1gram.lm.bin"
    },
    {
      "order": 2,
      "file": "2gram.arpa",
      "size_mb": 15.4,
      "smoothing": "kneser_ney",
      "perplexity": 124.3,
      "cross_entropy": 6.96,
      "oov_rate": 0.012,
      "backoff_rate": 0.312,
      "binary": "2gram.lm.bin"
    },
    ...
  ]
}
```

## Use Cases

### First-Pass ASR Optimization

Find optimal n-gram order for first-pass decoding before LLM rescoring:

```python
comparison = ModelComparison(corpus_file="asr_corpus.txt")

# Train candidate orders
comparison.train_orders([1, 2, 3, 4, 5])

# Evaluate
comparison.evaluate(test_file="test_audio_transcripts.txt")

# Get recommendation for first-pass
best = comparison.recommend(goal="first-pass", max_perplexity_increase=0.10)

# Export for PocketSphinx grid search
comparison.export_for_optimization(
    output_dir="pocketsphinx_experiments/",
    convert_to_binary=True
)

print(f"Use {best}-gram in decoder configuration")
```

### Model Selection with Baseline

Compare trained models against uniform baseline:

```python
comparison = ModelComparison(corpus_file="corpus.txt")
comparison.train_orders([2, 3, 4])
comparison.add_uniform_baseline()
comparison.evaluate(test_file="test.txt")

# Print table showing improvement over baseline
comparison.print_comparison()

# Uniform should have highest perplexity
# Trained models show actual learning from corpus
```

### Quick Comparison

Rapid comparison during development:

```python
from arpabo import ModelComparison

comparison = ModelComparison(corpus_file="dev_corpus.txt", verbose=True)
comparison.train_orders([2, 3])
comparison.evaluate(test_file="dev_test.txt")
comparison.print_comparison()
comparison.print_recommendation(goal="first-pass")
```

### Production Export

Export models ready for deployment:

```python
comparison = ModelComparison(
    corpus_file="production_corpus.txt",
    smoothing_method="kneser_ney"
)

# Train production candidates
comparison.train_orders([2, 3, 4])

# Evaluate on held-out test set
comparison.evaluate(test_file="held_out_test.txt")

# Export with binary conversion
comparison.export_for_optimization(
    output_dir="production_models/",
    convert_to_binary=True
)

# Use manifest.json in deployment pipeline
```

## Recommendation Logic

### First-Pass Goal

Balances perplexity vs model simplicity:

1. Find best (lowest) perplexity
2. Calculate threshold: `best_ppl * (1 + max_perplexity_increase)`
3. Find all models within threshold
4. Among those, pick simplest (lowest order)

**Rationale**: Simpler models with acceptable perplexity:
- Faster decoding
- Lower memory usage
- Often better backoff rates (more diversity)

### Rescoring Goal

Simply picks model with lowest perplexity:

**Rationale**: For rescoring/reranking, you want best discrimination, regardless of complexity.

## Integration with Phase 1 Features

ModelComparison uses all Phase 1 features internally:

```python
# Uses Feature 1.1 internally
models = comparison.train_orders([1,2,3,4])
# → Calls lm.compute_multiple_orders()

# Uses Feature 1.2 internally
results = comparison.evaluate(test_file="test.txt")
# → Calls lm.perplexity() for each model

# Uses Feature 1.3 internally
# → Calls lm.backoff_rate() for each model
# → Calls lm.get_statistics() during export

# Uses Feature 1.4 internally
comparison.add_uniform_baseline()
# → Calls ArpaBoLM.create_uniform()
```

## Error Handling

```python
# Must train before evaluate
comparison = ModelComparison("corpus.txt")
comparison.evaluate("test.txt")  # ❌ ValueError

# Must evaluate before recommend
comparison.train_orders([1,2,3])
comparison.recommend()  # ❌ ValueError

# Must train before export
comparison = ModelComparison("corpus.txt")
comparison.export_for_optimization("output/")  # ❌ ValueError

# Invalid goal
comparison.recommend(goal="invalid")  # ❌ ValueError
```

## Performance

| Operation | Time (100k words) | Notes |
|-----------|-------------------|-------|
| train_orders([1,2,3,4]) | ~2-3s | Corpus read once |
| add_uniform_baseline() | <1ms | Uses existing vocab |
| evaluate() | ~0.5s | Per test file |
| export_for_optimization() | ~1s | Includes file I/O |

**Total workflow**: ~3-5s for complete comparison of 4 orders.

## Best Practices

1. **Use held-out test data**: Don't evaluate on training data
2. **Include uniform baseline**: Sanity check that models are learning
3. **Evaluate with backoff**: Important for first-pass optimization
4. **Export with binary**: Ready for PocketSphinx deployment
5. **Save manifest.json**: Track experiments and metadata

## Example Output

### Comparison Table

```
Model Comparison
======================================================================
Corpus: train.txt

Model           PPL  Entropy   OOV%  Backoff
----------------------------------------------------------------------
uniform    1335.0    10.38    1.2%     0.0%
1-gram      342.5     8.42    1.2%     0.0%
2-gram      124.3     6.96    1.2%    31.2%
3-gram       89.2     6.48    1.2%    42.3%
4-gram       87.8     6.46    1.2%    48.1%
```

### Recommendation Output

```
Recommendation (first-pass): 3-gram
--------------------------------------------------
  Perplexity:        89.2
  Cross-entropy:      6.48 bits/word
  OOV rate:           1.2%
  Backoff rate:      42.3%

  → 1.6% higher PPL than best, but simpler model
  → Good backoff rate (42.3%) for diverse N-best lists

Recommendation (rescoring): 4-gram
--------------------------------------------------
  Perplexity:        87.8
  Cross-entropy:      6.46 bits/word
  OOV rate:           1.2%
  Backoff rate:      48.1%

  → Lowest perplexity for best discrimination
```

## Comparison with Manual Approach

**Before (manual):**
```python
# Train each model separately
models = {}
for order in [1, 2, 3, 4]:
    lm = ArpaBoLM(max_order=order)
    lm.read_corpus(open("train.txt"))
    lm.compute()
    models[order] = lm

# Evaluate each
results = {}
for order, model in models.items():
    with open("test.txt") as f:
        results[order] = model.perplexity(f)

# Manually compare...
```

**After (ModelComparison):**
```python
comparison = ModelComparison("train.txt")
comparison.train_orders([1, 2, 3, 4])
comparison.evaluate("test.txt")
comparison.print_comparison()
best = comparison.recommend(goal="first-pass")
```

**Benefits:**
- ✅ Cleaner code
- ✅ Less boilerplate
- ✅ Built-in recommendations
- ✅ Export functionality
- ✅ Formatted output

## Advanced Usage

### Custom Analysis

Access raw data for custom analysis:

```python
comparison.evaluate(test_file="test.txt")

# Get raw evaluation data
for order in comparison.list_models():
    eval_data = comparison.get_evaluation(order)

    # Custom metric: perplexity per OOV rate
    if eval_data["oov_rate"] > 0:
        metric = eval_data["perplexity"] / eval_data["oov_rate"]
        print(f"{order}-gram: {metric:.1f}")
```

### Multiple Test Sets

Evaluate on multiple test sets:

```python
test_files = ["test1.txt", "test2.txt", "test3.txt"]

for test_file in test_files:
    print(f"\n=== {test_file} ===")
    comparison.evaluate(test_file=test_file)
    comparison.print_comparison()
```

### Programmatic Export

Use exported models in automation:

```python
import json

# Export
manifest_path = comparison.export_for_optimization("output/")

# Load manifest
with open(manifest_path) as f:
    manifest = json.load(f)

# Use in pipeline
for model_info in manifest["models"]:
    order = model_info["order"]
    binary_file = model_info.get("binary")

    if binary_file and order in [2, 3, 4]:
        # Run PocketSphinx with this model
        run_decoder(model_file=f"output/{binary_file}")
```

## See Also

- Feature 1.1: Multi-Order Training
- Feature 1.2: Perplexity Evaluation
- Feature 1.3: Model Statistics & Backoff
- Feature 1.4: Uniform Language Model
- Example script: `examples/model_comparison_example.py`
