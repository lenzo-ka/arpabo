#!/usr/bin/env python
"""
Example: Complete model comparison workflow using ModelComparison class.

This demonstrates how to use the high-level ModelComparison API to:
1. Train multiple n-gram orders
2. Add uniform baseline
3. Evaluate on test data
4. Get recommendations
5. Export for optimization
"""

import sys

from arpabo import ModelComparison

# Example data files
# Use provided file or default to built-in Alice corpus
if len(sys.argv) > 1:
    CORPUS_FILE = sys.argv[1]
    TEST_FILE = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
else:
    # Use built-in Alice corpus
    from arpabo import get_example_corpus

    CORPUS_FILE = get_example_corpus()
    TEST_FILE = CORPUS_FILE  # Same file for demo (should be different in practice!)


def main():
    print("=" * 70)
    print("Model Comparison Example")
    print("=" * 70)

    # Step 1: Create comparison instance
    comparison = ModelComparison(corpus_file=CORPUS_FILE, smoothing_method="kneser_ney", verbose=True)

    # Step 2: Train multiple orders
    print("\n>>> Training models...")
    models = comparison.train_orders([1, 2, 3, 4])
    print(f"Trained {len(models)} models")

    # Step 3: Add uniform baseline
    print("\n>>> Adding uniform baseline...")
    uniform = comparison.add_uniform_baseline()
    print(f"Uniform baseline: {len(uniform.grams[0])} words")

    # Step 4: Evaluate all models
    print("\n>>> Evaluating models...")
    results = comparison.evaluate(test_file=TEST_FILE, include_backoff=True)
    print(f"Evaluated {len(results)} models")

    # Step 5: Print comparison table
    print()
    comparison.print_comparison()

    # Step 6: Get recommendations
    print("\n>>> Recommendations:")

    # For first-pass decoding
    comparison.print_recommendation(goal="first-pass", max_perplexity_increase=0.05)

    # For rescoring
    comparison.print_recommendation(goal="rescoring")

    # Step 7: Export for optimization
    print("\n>>> Exporting models...")
    manifest_path = comparison.export_for_optimization(output_dir="ngram_experiments/", convert_to_binary=True)
    print("Exported to: ngram_experiments/")
    print(f"Manifest: {manifest_path}")

    # Step 8: Show summary
    print("\n>>> Summary:")
    summary = comparison.summary()
    print(f"  Corpus: {summary['corpus_file']}")
    print(f"  Smoothing: {summary['smoothing_method']}")
    print(f"  Models trained: {summary['num_models']}")
    print(f"  Orders: {summary['orders']}")

    print("\n" + "=" * 70)
    print("Complete! Check ngram_experiments/ for output files.")
    print("=" * 70)


if __name__ == "__main__":
    main()
