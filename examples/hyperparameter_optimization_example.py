#!/usr/bin/env python
"""
Example: Hyperparameter Optimization for Language Models

This example demonstrates how to find the optimal n-gram order, smoothing method,
and parameters for a language model using different evaluation strategies.
"""

import os
import sys

from arpabo.comparison import optimize_hyperparameters, plot_optimization_results, print_optimization_results
from arpabo.lm import ArpaBoLM


def example_1_holdout_validation():
    """
    Example 1: Optimize using holdout validation.

    Split the corpus into train/dev sets and find the best configuration.
    This is fast and suitable for large datasets.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Holdout Validation")
    print("=" * 70)

    corpus_file = "src/arpabo/data/alice.txt"

    if not os.path.exists(corpus_file):
        print(f"Error: {corpus_file} not found")
        return

    results = optimize_hyperparameters(
        corpus_file=corpus_file,
        orders=[1, 2, 3, 4],
        smoothing_methods=["good_turing", "kneser_ney"],
        evaluation_mode="holdout",
        holdout_ratio=0.1,  # Use 10% for validation
        verbose=True,
        show_comparisons=True,
        export_results="optimization_results_holdout.json",
    )

    print("\n" + "=" * 70)
    print("Training best model on full corpus...")
    print("=" * 70)

    # Train the best model on full corpus
    best = results["best_config"]
    lm = ArpaBoLM(max_order=best["order"], smoothing_method=best["smoothing_method"], verbose=True)

    if best["discount_mass"] is not None:
        lm.discount_mass = best["discount_mass"]

    with open(corpus_file) as f:
        lm.read_corpus(f)

    lm.compute()
    lm.write_file("optimized_model.arpa")

    print("\nOptimized model saved to: optimized_model.arpa")
    print(f"Configuration: {best['order']}-gram {best['smoothing_method']}")


def example_2_external_test_set():
    """
    Example 2: Optimize using external test set.

    Train on full corpus and evaluate on separate test file.
    Use this when you have a dedicated test set.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: External Test Set")
    print("=" * 70)

    corpus_file = "src/arpabo/data/alice.txt"

    if not os.path.exists(corpus_file):
        print(f"Error: {corpus_file} not found")
        return

    # For this example, we'll create a small test set from the corpus
    print("Creating train/test split for demonstration...")

    with open(corpus_file) as f:
        lines = f.readlines()

    # Use last 10% as test
    split_idx = int(len(lines) * 0.9)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    # Write temporary files
    with open("temp_train.txt", "w") as f:
        f.writelines(train_lines)

    with open("temp_test.txt", "w") as f:
        f.writelines(test_lines)

    try:
        optimize_hyperparameters(
            corpus_file="temp_train.txt",
            orders=[1, 2, 3],
            smoothing_methods=["good_turing", "kneser_ney", "auto"],
            evaluation_mode="external",
            test_file="temp_test.txt",
            discount_masses=[0.5, 0.7, 0.9],  # Try different discount values for "auto"
            verbose=True,
            show_comparisons=True,
            export_results="optimization_results_external.json",
        )

        print("\n→ Optimization results saved to: optimization_results_external.json")

    finally:
        # Clean up temp files
        if os.path.exists("temp_train.txt"):
            os.unlink("temp_train.txt")
        if os.path.exists("temp_test.txt"):
            os.unlink("temp_test.txt")


def example_3_cross_validation():
    """
    Example 3: Optimize using cross-validation.

    Use k-fold cross-validation for robust evaluation when you don't have
    a separate test set. This is slower but more reliable.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Cross-Validation")
    print("=" * 70)

    corpus_file = "src/arpabo/data/alice.txt"

    if not os.path.exists(corpus_file):
        print(f"Error: {corpus_file} not found")
        return

    results = optimize_hyperparameters(
        corpus_file=corpus_file,
        orders=[1, 2, 3],
        smoothing_methods=["good_turing", "kneser_ney"],
        evaluation_mode="source",
        k_folds=5,  # 5-fold cross-validation
        verbose=True,
        show_comparisons=True,
        export_results="optimization_results_cv.json",
    )

    print("\n" + "=" * 70)
    print("Cross-validation provides robust estimates with std deviations")
    print("=" * 70)

    best = results["best_config"]
    print(f"\nBest: {best['order']}-gram {best['smoothing_method']}")
    print(f"Mean PPL: {best['perplexity']:.1f} ± {best['std_perplexity']:.1f}")


def example_4_load_and_visualize_results():
    """
    Example 4: Load previously saved results and visualize.

    Load optimization results from JSON and display comparisons.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Load and Visualize Saved Results")
    print("=" * 70)

    import json

    results_file = "optimization_results_holdout.json"

    if not os.path.exists(results_file):
        print("No saved results found. Run Example 1 first.")
        return

    print(f"Loading results from: {results_file}")

    with open(results_file) as f:
        results = json.load(f)

    # Display comprehensive comparison
    print_optimization_results(results, detailed=True)


def example_5_quick_optimization():
    """
    Example 5: Quick optimization with limited search space.

    For rapid iteration, search only a few key configurations.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Quick Optimization (Limited Search)")
    print("=" * 70)

    corpus_file = "src/arpabo/data/alice.txt"

    if not os.path.exists(corpus_file):
        print(f"Error: {corpus_file} not found")
        return

    results = optimize_hyperparameters(
        corpus_file=corpus_file,
        orders=[2, 3],  # Only bigram and trigram
        smoothing_methods=["kneser_ney"],  # Only one method
        evaluation_mode="holdout",
        holdout_ratio=0.15,
        verbose=True,
        show_comparisons=True,
    )

    print("\n→ Quick optimization complete!")
    print(f"Best: {results['best_config']['order']}-gram (PPL={results['best_config']['perplexity']:.1f})")


def example_6_comprehensive_search_with_visualization():
    """
    Example 6: Comprehensive search with visualization.

    Demonstrates the full search space including automatic parameter optimization,
    shows what discount mass the 'auto' method finds, and generates a matplotlib plot.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Comprehensive Search with Visualization")
    print("=" * 70)

    corpus_file = "src/arpabo/data/alice.txt"

    if not os.path.exists(corpus_file):
        print(f"Error: {corpus_file} not found")
        return

    print("\nRunning comprehensive search (4 methods, 4 orders, parameter tuning)...")
    print("This tests ~60 configurations...")

    # Run optimization with all methods
    results = optimize_hyperparameters(
        corpus_file=corpus_file,
        orders=[1, 2, 3, 4],
        smoothing_methods=["good_turing", "kneser_ney", "auto", "fixed"],
        discount_masses=[0.3, 0.5, 0.7, 0.9],  # For auto and fixed methods - wider range
        evaluation_mode="holdout",
        holdout_ratio=0.15,
        include_uniform=True,  # Include 0-gram baseline
        verbose=False,
        show_comparisons=False,
    )

    print("\n→ Optimization complete!\n")

    # Display text summary
    plot_optimization_results(results, use_matplotlib=False)

    # Generate matplotlib plot if available (gracefully falls back if not installed)
    print()
    plot_optimization_results(results, use_matplotlib=True, output_file="alice_optimization.png")

    # Show auto method optimization results
    print()
    print("=" * 70)
    print("AUTO METHOD: Optimized Discount Mass")
    print("=" * 70)
    print("\nThe 'auto' method automatically finds the optimal discount:")
    print()
    print(f"{'Initial':<10} {'Optimized To':<15} {'Order':<8} {'PPL':>10}")
    print("-" * 50)

    auto_results = [r for r in results["all_results"] if r["smoothing_method"] == "auto"]
    for r in sorted(auto_results, key=lambda x: x["perplexity"])[:8]:  # Show top 8
        initial = r.get("discount_mass", "N/A")
        optimized = r.get("optimized_discount_mass", "N/A")
        order = r["order"]
        ppl = r["perplexity"]
        print(f"{initial:<10} {optimized:<15} {order}-gram   {ppl:>10.1f}")

    print()
    print("→ All initial values converge to the same optimized discount!")
    print("→ The 'auto' method finds this value through likelihood optimization.")


def main():
    """Run examples based on command-line argument."""
    if len(sys.argv) < 2:
        print("Usage: python hyperparameter_optimization_example.py <example_number>")
        print("\nAvailable examples:")
        print("  1 - Holdout validation (fast, recommended)")
        print("  2 - External test set")
        print("  3 - Cross-validation (slow but robust)")
        print("  4 - Load and visualize saved results")
        print("  5 - Quick optimization (limited search)")
        print("  6 - Comprehensive search with visualization")
        print("\nOr run 'all' to run all examples sequentially")
        sys.exit(1)

    example = sys.argv[1]

    if example == "1":
        example_1_holdout_validation()
    elif example == "2":
        example_2_external_test_set()
    elif example == "3":
        example_3_cross_validation()
    elif example == "4":
        example_4_load_and_visualize_results()
    elif example == "5":
        example_5_quick_optimization()
    elif example == "6":
        example_6_comprehensive_search_with_visualization()
    elif example == "all":
        example_1_holdout_validation()
        example_2_external_test_set()
        example_3_cross_validation()
        example_4_load_and_visualize_results()
        example_5_quick_optimization()
        example_6_comprehensive_search_with_visualization()
    else:
        print(f"Unknown example: {example}")
        sys.exit(1)


if __name__ == "__main__":
    main()
