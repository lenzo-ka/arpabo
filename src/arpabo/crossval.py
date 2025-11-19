"""Cross-validation utilities for robust model selection."""

import math

from arpabo.lm import ArpaBoLM


def cross_validate(
    corpus_file: str, orders: list[int], k_folds: int = 5, smoothing_method: str = "kneser_ney", verbose: bool = False
) -> dict[int, dict]:
    """
    Perform k-fold cross-validation to evaluate models robustly.

    Splits corpus into k folds, trains on k-1 folds, evaluates on held-out fold.
    Repeats k times and aggregates results with mean and standard deviation.

    Args:
        corpus_file: Path to corpus file
        orders: List of n-gram orders to evaluate
        k_folds: Number of folds (default: 5)
        smoothing_method: Smoothing method to use
        verbose: Enable verbose output

    Returns:
        Dictionary mapping order → statistics:
        {
            order: {
                "mean_perplexity": float,
                "std_perplexity": float,
                "mean_cross_entropy": float,
                "std_cross_entropy": float,
                "mean_oov_rate": float,
                "fold_results": [...]
            }
        }

    Example:
        results = cross_validate(
            corpus_file="corpus.txt",
            orders=[1, 2, 3, 4],
            k_folds=5,
            smoothing_method="kneser_ney"
        )

        for order, stats in results.items():
            print(f"{order}-gram: PPL={stats['mean_perplexity']:.1f} "
                  f"±{stats['std_perplexity']:.1f}")
    """
    if k_folds < 2:
        raise ValueError(f"k_folds must be >= 2, got {k_folds}")

    # Read corpus
    with open(corpus_file) as f:
        lines = [line for line in f if line.strip()]

    if len(lines) < k_folds:
        raise ValueError(f"Corpus has {len(lines)} lines but k_folds={k_folds}. Need more data.")

    # Calculate fold size
    fold_size = len(lines) // k_folds

    if verbose:
        print(f"Cross-validation: {k_folds} folds, {len(lines)} sentences")
        print(f"Fold size: ~{fold_size} sentences")
        print()

    results = {order: {"fold_results": []} for order in orders}

    # Perform k-fold CV
    for fold_idx in range(k_folds):
        if verbose:
            print(f"Fold {fold_idx + 1}/{k_folds}:")

        # Split data
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < k_folds - 1 else len(lines)

        test_lines = lines[test_start:test_end]
        train_lines = lines[:test_start] + lines[test_end:]

        # Train models for each order
        fold_models = {}

        for order in orders:
            if verbose:
                print(f"  Training {order}-gram...", end=" ", flush=True)

            lm = ArpaBoLM(max_order=order, smoothing_method=smoothing_method, verbose=False)

            # Read training portion
            for line in train_lines:
                lm.read_corpus_line(line)

            lm.compute()
            fold_models[order] = lm

            if verbose:
                print("done")

        # Evaluate each model on test fold
        for order, model in fold_models.items():
            if verbose:
                print(f"  Evaluating {order}-gram...", end=" ", flush=True)

            # Create test corpus from test lines
            from io import StringIO

            test_corpus = StringIO("\n".join(test_lines))

            fold_results = model.perplexity(test_corpus)

            results[order]["fold_results"].append(
                {
                    "perplexity": fold_results["perplexity"],
                    "cross_entropy": fold_results["cross_entropy"],
                    "oov_rate": fold_results["oov_rate"],
                }
            )

            if verbose:
                print(f"PPL={fold_results['perplexity']:.1f}")

    # Aggregate results
    for order in orders:
        fold_results = results[order]["fold_results"]

        # Calculate mean and std
        perplexities = [fr["perplexity"] for fr in fold_results]
        entropies = [fr["cross_entropy"] for fr in fold_results]
        oov_rates = [fr["oov_rate"] for fr in fold_results]

        results[order]["mean_perplexity"] = sum(perplexities) / len(perplexities)
        results[order]["std_perplexity"] = _std_dev(perplexities)

        results[order]["mean_cross_entropy"] = sum(entropies) / len(entropies)
        results[order]["std_cross_entropy"] = _std_dev(entropies)

        results[order]["mean_oov_rate"] = sum(oov_rates) / len(oov_rates)
        results[order]["std_oov_rate"] = _std_dev(oov_rates)

    return results


def _std_dev(values: list[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def print_cv_results(results: dict[int, dict], k_folds: int = 5) -> None:
    """
    Print formatted cross-validation results.

    Args:
        results: Results from cross_validate()
        k_folds: Number of folds used

    Example:
        results = cross_validate("corpus.txt", [1,2,3,4])
        print_cv_results(results)
    """
    print(f"\nCross-Validation Results ({k_folds} folds)")
    print("=" * 70)
    print(f"{'Order':<8} {'Mean PPL':>10} {'Std PPL':>10} {'Mean Entropy':>12} {'Recommendation':<20}")
    print("-" * 70)

    # Sort by mean perplexity
    sorted_orders = sorted(results.keys(), key=lambda o: results[o]["mean_perplexity"])

    best_order = sorted_orders[0]

    for order in sorted(results.keys()):
        stats = results[order]

        mean_ppl = stats["mean_perplexity"]
        std_ppl = stats["std_perplexity"]
        mean_entropy = stats["mean_cross_entropy"]

        # Recommendation
        if order == best_order:
            rec = "← Best"
        elif order > 0:
            # Check if significantly different from best
            best_ppl = results[best_order]["mean_perplexity"]
            ppl_increase = (mean_ppl / best_ppl - 1) * 100

            rec = f"(+{ppl_increase:.1f}% PPL)" if ppl_increase < 2 else ""
        else:
            rec = ""

        print(f"{order}-gram  {mean_ppl:>10.1f} ±{std_ppl:>8.1f} {mean_entropy:>12.2f} {rec:<20}")

    print()
    print(f"Best model: {best_order}-gram")
    print(f"  Mean perplexity: {results[best_order]['mean_perplexity']:.1f}")
    print(f"  Std perplexity:  {results[best_order]['std_perplexity']:.1f}")
