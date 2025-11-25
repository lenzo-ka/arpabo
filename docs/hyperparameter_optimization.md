# Hyperparameter Optimization

This guide explains how to find the optimal n-gram order, smoothing method, and parameters for your language model using the `optimize_hyperparameters()` function.

## Overview

The hyperparameter optimization feature performs grid search across multiple configurations and provides interpretable comparison results at each step. It supports three evaluation strategies:

1. **Holdout Validation**: Fast, splits corpus into train/dev sets
2. **External Test Set**: Uses a separate test file
3. **Cross-Validation**: Robust, uses k-fold CV on source data

## Quick Start

```python
from arpabo import optimize_hyperparameters

# Basic optimization with holdout validation
results = optimize_hyperparameters(
    corpus_file="train.txt",
    evaluation_mode="holdout",
    orders=[1, 2, 3, 4],
    smoothing_methods=["good_turing", "kneser_ney"],
)

# Use the best configuration
best = results["best_config"]
print(f"Best: {best['order']}-gram {best['smoothing_method']}")
print(f"Perplexity: {best['perplexity']:.1f}")
```

## Evaluation Modes

### 1. Holdout Validation (Recommended)

Fast and effective for most use cases. Randomly splits your corpus into training and development sets.

```python
results = optimize_hyperparameters(
    corpus_file="corpus.txt",
    evaluation_mode="holdout",
    holdout_ratio=0.1,  # Use 10% for validation
)
```

**Pros:**
- Fast execution
- Simple to use
- Good for large datasets

**Cons:**
- Single evaluation (no variance estimate)
- Results depend on random split

### 2. External Test Set

Train on full corpus, evaluate on separate test file. Use when you have a dedicated test set.

```python
results = optimize_hyperparameters(
    corpus_file="train.txt",
    evaluation_mode="external",
    test_file="test.txt",
)
```

**Pros:**
- Uses full training corpus
- Reproducible results
- Realistic evaluation scenario

**Cons:**
- Requires separate test set
- May overfit to test set if evaluated repeatedly

### 3. Cross-Validation

Most robust but slowest. Uses k-fold cross-validation on source data.

```python
results = optimize_hyperparameters(
    corpus_file="corpus.txt",
    evaluation_mode="source",
    k_folds=5,
)
```

**Pros:**
- Robust variance estimates
- No separate test set needed
- Less sensitive to data splits

**Cons:**
- Slower (trains k models per configuration)
- May be overkill for large datasets

## Search Space Configuration

### N-gram Orders

Specify which n-gram orders to evaluate:

```python
# Common configurations
orders=[1, 2, 3, 4]        # Unigram through 4-gram
orders=[2, 3]              # Just bigram and trigram
orders=[1, 2, 3, 4, 5]     # Include 5-gram
```

### Smoothing Methods

Choose which smoothing methods to compare:

```python
# All major methods
smoothing_methods=["good_turing", "kneser_ney", "auto"]

# Kneser-Ney only
smoothing_methods=["kneser_ney"]

# Compare Good-Turing vs Katz backoff
smoothing_methods=["good_turing", "auto", "fixed"]
```

Available methods:
- `"good_turing"`: Good-Turing smoothing (no parameters)
- `"kneser_ney"`: Kneser-Ney smoothing (no parameters)
- `"auto"`: Katz backoff with optimized discount (needs discount_mass parameter)
- `"fixed"`: Katz backoff with fixed discount (needs discount_mass parameter)

### Discount Mass Parameters

For methods that support discount mass (`"auto"` and `"fixed"`), specify values to try:

```python
results = optimize_hyperparameters(
    corpus_file="corpus.txt",
    smoothing_methods=["auto"],
    discount_masses=[0.3, 0.5, 0.7, 0.9],
)
```

## Interpretable Results

The function provides detailed comparisons at multiple levels:

### 1. Real-time Progress

With `verbose=True` and `show_comparisons=True`, see results as they're computed:

```python
results = optimize_hyperparameters(
    corpus_file="corpus.txt",
    verbose=True,
    show_comparisons=True,
)
```

Output includes:
- Current configuration being tested
- Perplexity and training time
- Comparison vs current best
- Parameter comparisons for each method
- Summary after completing each order

### 2. Final Summary Tables

Comprehensive comparison views:

**Top Configurations:**
```
Rank   Order   Method          Discount   PPL        Entropy
------------------------------------------------------------
1      3       kneser_ney      N/A        156.2      7.29 ✓
2      4       kneser_ney      N/A        158.4      7.31
3      2       kneser_ney      N/A        165.8      7.37
```

**By N-gram Order:**
```
Order    Best Method          Best PPL     Avg PPL      # Configs
-----------------------------------------------------------------
1        good_turing          425.3        425.3        2
2        kneser_ney           165.8        172.1        2
3        kneser_ney           156.2        159.5        2
4        kneser_ney           158.4        162.7        2
```

**By Smoothing Method:**
```
Method           Best Order   Best PPL     Avg PPL      # Configs
-----------------------------------------------------------------
good_turing      3            162.7        224.1        4
kneser_ney       3            156.2        175.1        4
```

**Trade-off Analysis:**
```
Config                              PPL        Time(s)    PPL/sec
-----------------------------------------------------------------
2-gram good_turing                  172.3      1.2        143.6
3-gram kneser_ney                   156.2      2.8        55.8
```

### 3. Key Insights

Automatic analysis of results:

```
Key Insights:
----------------------------------------------------------------------
• Best order: 3-gram (PPL=156.2)
  23.4% better than 2-gram

• Best smoothing: kneser_ney (PPL=156.2)
  4.1% better than good_turing

• Training time range: 1.2s - 5.4s
  Fastest: 2-gram good_turing (PPL=172.3)
  Slowest: 4-gram kneser_ney (PPL=158.4)
```

## Result Structure

The function returns a comprehensive dictionary:

```python
{
    "best_config": {
        "order": 3,
        "smoothing_method": "kneser_ney",
        "discount_mass": None,
        "perplexity": 156.2,
        "cross_entropy": 7.29,
        "training_time": 2.8
    },
    "all_results": [
        # List of all evaluated configurations
        {"order": 1, "smoothing_method": "good_turing", ...},
        {"order": 2, "smoothing_method": "good_turing", ...},
        # ...
    ],
    "results_by_order": {
        1: [...],  # All 1-gram results
        2: [...],  # All 2-gram results
        # ...
    },
    "results_by_method": {
        "good_turing": [...],  # All Good-Turing results
        "kneser_ney": [...],   # All Kneser-Ney results
        # ...
    },
    "evaluation_mode": "holdout",
    "corpus_file": "corpus.txt",
    "test_file": None,
    "search_space": {
        "orders": [1, 2, 3, 4],
        "smoothing_methods": ["good_turing", "kneser_ney"],
        "discount_masses": None
    },
    "timestamp": "2025-11-21 10:30:45"
}
```

## Exporting Results

Save results to JSON for later analysis:

```python
results = optimize_hyperparameters(
    corpus_file="corpus.txt",
    export_results="optimization_results.json",
)
```

Load and visualize later:

```python
import json
from arpabo import print_optimization_results

with open("optimization_results.json") as f:
    results = json.load(f)

print_optimization_results(results, detailed=True)
```

## Training the Best Model

After finding the optimal configuration:

```python
from arpabo import ArpaBoLM, optimize_hyperparameters

# Find best configuration
results = optimize_hyperparameters(
    corpus_file="train.txt",
    evaluation_mode="holdout",
)

best = results["best_config"]

# Train final model on full corpus
lm = ArpaBoLM(
    max_order=best["order"],
    smoothing_method=best["smoothing_method"],
    verbose=True
)

# Apply discount_mass if needed
if best["discount_mass"] is not None:
    lm.discount_mass = best["discount_mass"]

with open("train.txt") as f:
    lm.read_corpus(f)

lm.compute()
lm.write_file("optimized_model.arpa")

print(f"Trained {best['order']}-gram {best['smoothing_method']}")
print(f"Expected perplexity: {best['perplexity']:.1f}")
```

## Complete Example

```python
from arpabo import optimize_hyperparameters, ArpaBoLM

# Step 1: Find optimal configuration
print("Optimizing hyperparameters...")
results = optimize_hyperparameters(
    corpus_file="train.txt",
    orders=[1, 2, 3, 4],
    smoothing_methods=["good_turing", "kneser_ney", "auto"],
    evaluation_mode="holdout",
    holdout_ratio=0.1,
    discount_masses=[0.5, 0.7, 0.9],
    verbose=True,
    show_comparisons=True,
    export_results="optimization_results.json",
)

# Step 2: Review results
best = results["best_config"]
print(f"\nBest configuration:")
print(f"  {best['order']}-gram {best['smoothing_method']}")
print(f"  Perplexity: {best['perplexity']:.1f}")

# Step 3: Train final model
print("\nTraining final model on full corpus...")
lm = ArpaBoLM(
    max_order=best["order"],
    smoothing_method=best["smoothing_method"],
    verbose=True
)

if best["discount_mass"] is not None:
    lm.discount_mass = best["discount_mass"]

with open("train.txt") as f:
    lm.read_corpus(f)

lm.compute()
lm.write_file("final_model.arpa")

# Step 4: Evaluate on test set
print("\nEvaluating on test set...")
with open("test.txt") as f:
    test_results = lm.perplexity(f)

print(f"Test perplexity: {test_results['perplexity']:.1f}")
```

## Best Practices

1. **Start with holdout validation** for initial exploration (fast)
2. **Use cross-validation** for final evaluation (robust)
3. **Limit search space** for quick iterations
4. **Export results** for reproducibility
5. **Train on full corpus** after finding best config
6. **Validate on true test set** to confirm performance

## Performance Tips

### For Large Corpora

```python
# Use smaller holdout ratio
results = optimize_hyperparameters(
    corpus_file="large_corpus.txt",
    evaluation_mode="holdout",
    holdout_ratio=0.05,  # Only 5% for dev
)
```

### For Quick Exploration

```python
# Limit search space
results = optimize_hyperparameters(
    corpus_file="corpus.txt",
    orders=[2, 3],  # Skip unigram and higher orders
    smoothing_methods=["kneser_ney"],  # Single method
    verbose=False,
    show_comparisons=False,
)
```

### For Thorough Search

```python
# Comprehensive grid search
results = optimize_hyperparameters(
    corpus_file="corpus.txt",
    orders=[1, 2, 3, 4, 5],
    smoothing_methods=["good_turing", "kneser_ney", "auto", "fixed"],
    discount_masses=[0.3, 0.5, 0.7, 0.9],
    evaluation_mode="source",
    k_folds=10,
)
```

## See Also

- [Model Comparison](model_comparison.md) - High-level comparison workflows
- [Perplexity Evaluation](perplexity_evaluation.md) - Understanding perplexity
- [Multi-Order Training](multi_order_training.md) - Training multiple orders efficiently
