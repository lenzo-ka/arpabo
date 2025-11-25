"""Model comparison and optimization utilities."""

import json
import math
import os
import random
import time
from io import StringIO
from typing import Optional

from arpabo.convert import to_pocketsphinx_binary
from arpabo.lm import ArpaBoLM

# Constants
MARGINAL_IMPROVEMENT_THRESHOLD = 0.02  # 2% threshold for "marginal" PPL improvement


class ModelComparison:
    """
    Compare multiple n-gram configurations for first-pass ASR optimization.

    This class provides a high-level API for training, evaluating, and comparing
    multiple n-gram models to find the optimal configuration for your use case.

    Example:
        comparison = ModelComparison(corpus_file="train.txt")
        comparison.train_orders([1, 2, 3, 4])
        comparison.evaluate(test_file="test.txt")

        best = comparison.recommend(goal="first-pass")
        print(f"Best model: {best}-gram")

        comparison.export_for_optimization(
            output_dir="experiments/",
            convert_to_binary=True
        )
    """

    def __init__(self, corpus_file: str, smoothing_method: str = "kneser_ney", verbose: bool = False):
        """
        Initialize model comparison.

        Args:
            corpus_file: Path to training corpus
            smoothing_method: Smoothing method to use for all models
            verbose: Enable verbose output
        """
        self.corpus_file = corpus_file
        self.smoothing_method = smoothing_method
        self.verbose = verbose
        self.models = {}
        self.evaluations = {}
        self._corpus_lm = None

    def train_orders(self, orders: list[int]) -> dict[int, ArpaBoLM]:
        """
        Train multiple n-gram orders from the corpus.

        Args:
            orders: List of n-gram orders to train (e.g., [1, 2, 3, 4])

        Returns:
            Dictionary mapping order → trained model

        Example:
            models = comparison.train_orders([1, 2, 3, 4])
        """
        if self.verbose:
            print(f"Training models from {self.corpus_file}...")

        # Create base model with highest order
        max_order = max(orders)
        lm = ArpaBoLM(max_order=max_order, smoothing_method=self.smoothing_method, verbose=self.verbose)

        with open(self.corpus_file) as f:
            lm.read_corpus(f)

        # Train all orders
        self.models = lm.compute_multiple_orders(orders)
        self._corpus_lm = lm  # Save for vocabulary extraction

        if self.verbose:
            print(f"Trained {len(orders)} models: {orders}")

        return self.models

    def add_uniform_baseline(self) -> ArpaBoLM:
        """
        Add uniform language model as baseline for comparison.

        Returns:
            The uniform model

        Raises:
            ValueError: If train_orders() hasn't been called yet

        Example:
            comparison.train_orders([1, 2, 3, 4])
            comparison.add_uniform_baseline()
        """
        if not self.models:
            raise ValueError("Must call train_orders() before adding uniform baseline")

        # Extract vocabulary from trained models
        first_model = list(self.models.values())[0]
        vocab = list(first_model.grams[0].keys())

        if self.verbose:
            print(f"Creating uniform baseline with {len(vocab)} words...")

        uniform = ArpaBoLM.create_uniform(vocab, add_start=True)
        self.models[0] = uniform  # Use order 0 for uniform

        return uniform

    def evaluate(self, test_file: str, include_backoff: bool = True) -> dict[int, dict]:
        """
        Evaluate all models on test data.

        Args:
            test_file: Path to test corpus
            include_backoff: Whether to include backoff analysis (slower)

        Returns:
            Dictionary mapping order → evaluation metrics

        Example:
            results = comparison.evaluate(test_file="test.txt")
        """
        if not self.models:
            raise ValueError("Must call train_orders() before evaluate()")

        if self.verbose:
            print(f"Evaluating on {test_file}...")

        for order, model in self.models.items():
            if self.verbose:
                name = "uniform" if order == 0 else f"{order}-gram"
                print(f"  Evaluating {name}...")

            # Perplexity
            with open(test_file) as f:
                ppl = model.perplexity(f)

            eval_data = {**ppl}

            # Backoff analysis (skip for uniform unigram)
            if include_backoff and order > 0:
                with open(test_file) as f:
                    backoff = model.backoff_rate(f)
                eval_data.update(backoff)
            elif order == 0:
                # Uniform model: manually set backoff to 0
                eval_data["overall_backoff_rate"] = 0.0
                eval_data["order_usage"] = {1: 1.0}
                eval_data["total_queries"] = ppl["num_words"]

            self.evaluations[order] = eval_data

        return self.evaluations

    def recommend(self, goal: str = "first-pass", max_perplexity_increase: float = 0.05) -> int:
        """
        Recommend best n-gram order for specified goal.

        Args:
            goal: Optimization goal:
                - "first-pass": Balance PPL vs diversity (good backoff)
                - "rescoring": Minimize perplexity (best accuracy)
            max_perplexity_increase: For first-pass, acceptable PPL degradation
                                     to prefer simpler model (default: 5%)

        Returns:
            Recommended n-gram order (excludes uniform baseline if present)

        Raises:
            ValueError: If evaluate() hasn't been called yet

        Example:
            best = comparison.recommend(goal="first-pass", max_perplexity_increase=0.05)
            print(f"Recommended: {best}-gram")
        """
        if not self.evaluations:
            raise ValueError("Must call evaluate() before recommend()")

        # Exclude uniform baseline (order 0) from recommendations
        candidates = {k: v for k, v in self.evaluations.items() if k > 0}

        if not candidates:
            raise ValueError("No trained models available for recommendation")

        if goal == "rescoring":
            # Simply minimize perplexity
            return min(candidates.keys(), key=lambda o: candidates[o]["perplexity"])

        elif goal == "first-pass":
            # Find simplest model within perplexity tolerance
            best_ppl = min(e["perplexity"] for e in candidates.values())
            threshold = best_ppl * (1 + max_perplexity_increase)

            acceptable = [order for order, eval_data in candidates.items() if eval_data["perplexity"] <= threshold]

            if not acceptable:
                # Fallback to best perplexity if none within threshold
                return min(candidates.keys(), key=lambda o: candidates[o]["perplexity"])

            # Among acceptable, prefer simpler model (lower order)
            return min(acceptable)

        else:
            raise ValueError(f"Unknown goal: {goal}. Use 'first-pass' or 'rescoring'")

    def export_for_optimization(self, output_dir: str, convert_to_binary: bool = True) -> str:
        """
        Export models in format ready for optimization frameworks.

        Creates directory with:
        - {order}gram.arpa files
        - {order}gram.lm.bin files (if convert_to_binary)
        - manifest.json with metadata and evaluation results

        Args:
            output_dir: Output directory path (will be created if needed)
            convert_to_binary: Whether to convert to PocketSphinx binary format

        Returns:
            Path to manifest.json file

        Example:
            comparison.export_for_optimization(
                output_dir="ngram_experiments/",
                convert_to_binary=True
            )
        """
        if not self.models:
            raise ValueError("Must call train_orders() before export")

        os.makedirs(output_dir, exist_ok=True)

        if self.verbose:
            print(f"Exporting to {output_dir}/...")

        # Build manifest
        first_model = list(self.models.values())[0]
        manifest = {
            "corpus": self.corpus_file,
            "smoothing": self.smoothing_method,
            "vocab_size": len(first_model.grams[0]),
            "models": [],
        }

        # Export each model
        for order in sorted(self.models.keys()):
            model = self.models[order]

            # Determine filename
            arpa_file = "uniform.arpa" if order == 0 else f"{order}gram.arpa"

            arpa_path = os.path.join(output_dir, arpa_file)

            # Write ARPA file
            if self.verbose:
                print(f"  Writing {arpa_file}...")

            with open(arpa_path, "w") as f:
                model.write(f)

            # Get file size
            size_mb = os.path.getsize(arpa_path) / (1024 * 1024)

            model_info = {
                "order": order,
                "file": arpa_file,
                "size_mb": round(size_mb, 2),
                "smoothing": model.smoothing_method,
            }

            # Add evaluation metrics if available
            if order in self.evaluations:
                eval_data = self.evaluations[order]
                model_info["perplexity"] = round(eval_data["perplexity"], 2)
                model_info["cross_entropy"] = round(eval_data["cross_entropy"], 2)
                model_info["oov_rate"] = round(eval_data["oov_rate"], 4)
                if "overall_backoff_rate" in eval_data:
                    model_info["backoff_rate"] = round(eval_data["overall_backoff_rate"], 4)

            # Convert to binary if requested
            if convert_to_binary:
                bin_file = "uniform.lm.bin" if order == 0 else f"{order}gram.lm.bin"

                try:
                    if self.verbose:
                        print(f"    Converting to binary: {bin_file}")
                    to_pocketsphinx_binary(arpa_path, verbose=False)
                    model_info["binary"] = bin_file
                except Exception as e:
                    if self.verbose:
                        print(f"    Binary conversion failed: {e}")

            manifest["models"].append(model_info)

        # Write manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        if self.verbose:
            print(f"Wrote manifest to {manifest_path}")

        return manifest_path

    def print_comparison(self) -> None:
        """
        Print formatted comparison table of all evaluated models.

        Raises:
            ValueError: If evaluate() hasn't been called yet

        Example:
            comparison.evaluate(test_file="test.txt")
            comparison.print_comparison()
        """
        if not self.evaluations:
            raise ValueError("Must call evaluate() before print_comparison()")

        print("\nModel Comparison")
        print("=" * 70)
        print(f"Corpus: {self.corpus_file}")
        print()
        print(f"{'Model':<10} {'PPL':>8} {'Entropy':>8} {'OOV%':>6} {'Backoff':>8}")
        print("-" * 70)

        for order in sorted(self.evaluations.keys()):
            eval_data = self.evaluations[order]

            name = "uniform" if order == 0 else f"{order}-gram"

            ppl = eval_data["perplexity"]
            entropy = eval_data["cross_entropy"]
            oov_pct = eval_data["oov_rate"] * 100
            backoff_pct = eval_data.get("overall_backoff_rate", 0.0) * 100

            print(f"{name:<10} {ppl:>8.1f} {entropy:>8.2f} {oov_pct:>6.1f} {backoff_pct:>7.1f}%")

    def print_recommendation(self, goal: str = "first-pass", max_perplexity_increase: float = 0.05) -> None:
        """
        Print recommendation with explanation.

        Args:
            goal: Optimization goal ("first-pass" or "rescoring")
            max_perplexity_increase: PPL tolerance for first-pass

        Example:
            comparison.print_recommendation(goal="first-pass")
        """
        best = self.recommend(goal=goal, max_perplexity_increase=max_perplexity_increase)
        eval_data = self.evaluations[best]

        print(f"\nRecommendation ({goal}): {best}-gram")
        print("-" * 50)
        print(f"  Perplexity:    {eval_data['perplexity']:>8.1f}")
        print(f"  Cross-entropy: {eval_data['cross_entropy']:>8.2f} bits/word")
        print(f"  OOV rate:      {eval_data['oov_rate'] * 100:>8.1f}%")

        if "overall_backoff_rate" in eval_data:
            print(f"  Backoff rate:  {eval_data['overall_backoff_rate'] * 100:>8.1f}%")

        # Add rationale
        if goal == "first-pass":
            best_ppl = min(e["perplexity"] for k, e in self.evaluations.items() if k > 0)
            ppl_increase = (eval_data["perplexity"] / best_ppl - 1) * 100

            if ppl_increase < MARGINAL_IMPROVEMENT_THRESHOLD * 100:
                print("\n  → Best perplexity among all models")
            else:
                print(f"\n  → {ppl_increase:.1f}% higher PPL than best, but simpler model")

            if "overall_backoff_rate" in eval_data:
                backoff = eval_data["overall_backoff_rate"]
                if backoff > 0.4:
                    print(f"  → Good backoff rate ({backoff * 100:.1f}%) for diverse N-best lists")
                elif backoff > 0.2:
                    print(f"  → Moderate backoff rate ({backoff * 100:.1f}%)")
                else:
                    print(f"  → Low backoff rate ({backoff * 100:.1f}%) - sharp predictions")

        elif goal == "rescoring":
            print("\n  → Lowest perplexity for best discrimination")

    def get_model(self, order: int) -> ArpaBoLM:
        """
        Get trained model by order.

        Args:
            order: N-gram order (0 for uniform baseline)

        Returns:
            Trained ArpaBoLM model

        Raises:
            KeyError: If order hasn't been trained
        """
        if order not in self.models:
            raise KeyError(f"Order {order} not found. Available: {sorted(self.models.keys())}")
        return self.models[order]

    def get_evaluation(self, order: int) -> dict:
        """
        Get evaluation results for specific order.

        Args:
            order: N-gram order (0 for uniform baseline)

        Returns:
            Evaluation dictionary

        Raises:
            KeyError: If order hasn't been evaluated
        """
        if order not in self.evaluations:
            raise KeyError(f"Order {order} not evaluated. Call evaluate() first.")
        return self.evaluations[order]

    def list_models(self) -> list[int]:
        """
        List available model orders.

        Returns:
            Sorted list of orders (0 = uniform baseline)
        """
        return sorted(self.models.keys())

    def summary(self) -> dict:
        """
        Get summary of all models and evaluations.

        Returns:
            Dictionary with corpus info, models, and evaluations

        Example:
            summary = comparison.summary()
            print(json.dumps(summary, indent=2))
        """
        return {
            "corpus_file": self.corpus_file,
            "smoothing_method": self.smoothing_method,
            "num_models": len(self.models),
            "orders": self.list_models(),
            "evaluations": self.evaluations if self.evaluations else None,
        }


def compare_smoothing_methods(
    corpus_file: str, test_file: str, methods: Optional[list[str]] = None, max_order: int = 3, verbose: bool = False
) -> dict[str, dict]:
    """
    Compare different smoothing methods on the same corpus.

    Useful for determining which smoothing method works best for your domain
    and corpus size.

    Args:
        corpus_file: Path to training corpus
        test_file: Path to test corpus for evaluation
        methods: List of smoothing methods to compare (None = all available)
        max_order: N-gram order to use for comparison (default: 3)
        verbose: Enable verbose output

    Returns:
        Dictionary mapping method_name → evaluation_metrics
        {
            method: {
                "perplexity": float,
                "cross_entropy": float,
                "num_words": int,
                "oov_rate": float,
                "training_time_seconds": float
            }
        }

    Example:
        results = compare_smoothing_methods(
            corpus_file="train.txt",
            test_file="test.txt",
            methods=["good_turing", "kneser_ney"],
            max_order=3
        )

        for method, metrics in results.items():
            print(f"{method}: PPL={metrics['perplexity']:.1f}")
    """
    if methods is None:
        methods = ["good_turing", "kneser_ney", "auto"]  # auto = Katz backoff with auto-tuning

    if verbose:
        print(f"Comparing smoothing methods: {methods}")
        print(f"Order: {max_order}")
        print()

    results = {}

    for method in methods:
        if verbose:
            print(f"Training {method}...", end=" ", flush=True)

        start_time = time.time()

        # Train model
        lm = ArpaBoLM(max_order=max_order, smoothing_method=method, verbose=False)

        with open(corpus_file) as f:
            lm.read_corpus(f)

        lm.compute()

        # Evaluate
        with open(test_file) as f:
            eval_results = lm.perplexity(f)

        elapsed = time.time() - start_time

        results[method] = {**eval_results, "training_time_seconds": elapsed}

        if verbose:
            print(f"done ({elapsed:.2f}s)")

    return results


def print_smoothing_comparison(results: dict[str, dict], test_file: str = "test data", max_order: int = 3) -> None:
    """
    Print formatted comparison of smoothing methods.

    Args:
        results: Results from compare_smoothing_methods()
        test_file: Test file name for display
        max_order: N-gram order used

    Example:
        results = compare_smoothing_methods("train.txt", "test.txt")
        print_smoothing_comparison(results, test_file="test.txt")
    """
    print("\nSmoothing Method Comparison")
    print("=" * 80)
    print(f"Order: {max_order}")
    print(f"Test file: {test_file}")
    print()
    print(f"{'Method':<15} {'Perplexity':>12} {'Entropy':>10} {'Time(s)':>10} {'Recommendation':<20}")
    print("-" * 80)

    # Sort by perplexity (best first)
    sorted_methods = sorted(results.items(), key=lambda x: x[1]["perplexity"])

    best_method = sorted_methods[0][0]

    for method, metrics in sorted_methods:
        ppl = metrics["perplexity"]
        entropy = metrics["cross_entropy"]
        time_sec = metrics["training_time_seconds"]

        # Recommendation text
        if method == best_method:
            rec = "← Best"
        elif method == "good_turing":
            rec = "Good for sparse data"
        elif method == "kneser_ney":
            rec = "Good for general use"
        elif method == "auto":
            rec = "Katz (auto-tuned)"
        elif method == "fixed":
            rec = "Katz (fixed)"
        elif method == "mle":
            rec = "No smoothing"
        else:
            rec = ""

        print(f"{method:<15} {ppl:>12.1f} {entropy:>10.2f} {time_sec:>10.2f} {rec:<20}")

    print()
    print(f"Best method: {best_method}")
    print(f"  Perplexity: {results[best_method]['perplexity']:.1f}")
    print(f"  Cross-entropy: {results[best_method]['cross_entropy']:.2f} bits/word")


def optimize_hyperparameters(
    corpus_file: str,
    orders: Optional[list[int]] = None,
    smoothing_methods: Optional[list[str]] = None,
    evaluation_mode: str = "holdout",
    holdout_ratio: float = 0.1,
    test_file: Optional[str] = None,
    goal: str = "perplexity",
    k_folds: Optional[int] = None,
    discount_masses: Optional[list[float]] = None,
    include_uniform: bool = False,
    verbose: bool = True,
    show_comparisons: bool = True,
    export_results: Optional[str] = None,
) -> dict:
    """
    Find optimal language model configuration (order, smoothing method, parameters).

    Performs grid search over n-gram orders, smoothing methods, and method-specific
    parameters to find the best configuration. Supports multiple evaluation strategies:
    source data with cross-validation, held-out validation set, or external test set.

    Args:
        corpus_file: Path to training corpus
        orders: N-gram orders to try (default: [1, 2, 3] - includes unigram baseline)
        smoothing_methods: Smoothing methods to try (default: ["good_turing", "kneser_ney"])
        evaluation_mode: How to evaluate models:
            - "source": Use k-fold cross-validation on source data
            - "holdout": Split corpus into train/dev sets
            - "external": Use separate test file
        holdout_ratio: Fraction of corpus to hold out for validation (default: 0.1)
        test_file: Path to external test file (required if evaluation_mode="external")
        goal: Optimization objective:
            - "perplexity": Minimize perplexity (best accuracy)
            - "first-pass": Balance perplexity with backoff for diversity
        k_folds: Number of folds for cross-validation (default: 5, only for "source" mode)
        discount_masses: Discount values to try for methods that support it (default: [0.5, 0.7, 0.9])
        include_uniform: Include uniform (0-gram) baseline for lower bound comparison (default: False)
        verbose: Enable detailed progress output
        show_comparisons: Display comparison tables during optimization (default: True)
        export_results: Optional path to export detailed results as JSON

    Returns:
        Dictionary with optimization results:
        {
            "best_config": {
                "order": int,
                "smoothing_method": str,
                "discount_mass": float or None,
                "perplexity": float,
                "cross_entropy": float
            },
            "all_results": [
                {
                    "order": int,
                    "smoothing_method": str,
                    "discount_mass": float or None,
                    "perplexity": float,
                    "cross_entropy": float,
                    "training_time": float
                },
                ...
            ],
            "evaluation_mode": str,
            "corpus_file": str,
            "test_file": str or None
        }

    Raises:
        ValueError: If invalid parameters or missing required files

    Examples:
        # Quick optimization with defaults (1-3 grams provide baseline and practical options)
        result = optimize_hyperparameters(
            corpus_file="train.txt"
        )

        # Just practical orders for ASR (skip unigram baseline)
        result = optimize_hyperparameters(
            corpus_file="train.txt",
            orders=[2, 3],  # Bigram and trigram only
            smoothing_methods=["kneser_ney"],  # Single best method
            evaluation_mode="holdout"
        )

        # Comprehensive search including higher orders
        result = optimize_hyperparameters(
            corpus_file="train.txt",
            orders=[1, 2, 3, 4, 5],  # Up to 5-gram
            smoothing_methods=["good_turing", "kneser_ney", "auto"],
            discount_masses=[0.3, 0.5, 0.7, 0.9],  # Finer grid
            evaluation_mode="external",
            test_file="test.txt"
        )

        # Minimal search (fastest)
        result = optimize_hyperparameters(
            corpus_file="train.txt",
            orders=[3],  # Only trigram
            smoothing_methods=["kneser_ney"],
            verbose=False,
            show_comparisons=False
        )

        # Cross-validation with custom folds
        result = optimize_hyperparameters(
            corpus_file="corpus.txt",
            evaluation_mode="source",
            k_folds=10,  # More folds = more robust
            orders=[2, 3, 4]
        )

    Note:
        - Grid search can be time-consuming for large parameter spaces
        - For "source" mode, uses k-fold cross-validation (slower but more robust)
        - For "holdout" mode, randomly splits corpus (faster but single evaluation)
        - For "external" mode, trains on full corpus and evaluates on test file
    """
    # Set defaults (conservative for quick optimization, fully configurable)
    if orders is None:
        # Include unigram for baseline (fast, provides lower bound)
        # Bigram and trigram are most practical for ASR
        orders = [1, 2, 3]

    if smoothing_methods is None:
        smoothing_methods = ["good_turing", "kneser_ney"]  # Most common methods

    if discount_masses is None:
        # For methods that use discount_mass (auto, fixed)
        # Grid search increments of 0.2 from 0.5 to 0.9
        discount_masses = [0.5, 0.7, 0.9]

    if k_folds is None:
        k_folds = 5  # Standard CV fold count

    # Validate evaluation mode
    if evaluation_mode not in ["source", "holdout", "external"]:
        raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}. Use 'source', 'holdout', or 'external'")

    if evaluation_mode == "external" and test_file is None:
        raise ValueError("test_file is required when evaluation_mode='external'")

    if evaluation_mode == "source" and k_folds < 2:
        raise ValueError(f"k_folds must be >= 2 for source evaluation, got {k_folds}")

    if verbose:
        print("Hyperparameter Optimization")
        print("=" * 70)
        print(f"Corpus: {corpus_file}")
        print(f"Evaluation mode: {evaluation_mode}")
        if evaluation_mode == "external":
            print(f"Test file: {test_file}")
        elif evaluation_mode == "holdout":
            print(f"Holdout ratio: {holdout_ratio:.1%}")
        elif evaluation_mode == "source":
            print(f"Cross-validation: {k_folds} folds")
        print(f"Orders: {orders}")
        print(f"Smoothing methods: {smoothing_methods}")
        print(f"Goal: {goal}")
        print()

    # Prepare data based on evaluation mode
    if evaluation_mode == "holdout":
        # Split corpus into train and dev
        if verbose:
            print("Splitting corpus into train/dev sets...")

        with open(corpus_file) as f:
            lines = [line for line in f if line.strip()]

        random.shuffle(lines)
        split_idx = int(len(lines) * (1 - holdout_ratio))

        train_lines = lines[:split_idx]
        dev_lines = lines[split_idx:]

        if verbose:
            print(f"  Train: {len(train_lines)} sentences")
            print(f"  Dev:   {len(dev_lines)} sentences")
            print()

        # Write temporary files
        import tempfile

        train_fd, train_path = tempfile.mkstemp(suffix=".txt", text=True)
        dev_fd, dev_path = tempfile.mkstemp(suffix=".txt", text=True)

        try:
            with os.fdopen(train_fd, "w") as f:
                f.writelines(train_lines)

            with os.fdopen(dev_fd, "w") as f:
                f.writelines(dev_lines)

            # Use train/dev split
            train_corpus_file = train_path
            eval_file = dev_path

        except Exception:
            os.close(train_fd)
            os.close(dev_fd)
            raise

    elif evaluation_mode == "external":
        # Use full corpus for training
        train_corpus_file = corpus_file
        eval_file = test_file

    else:  # source mode with cross-validation
        train_corpus_file = corpus_file
        eval_file = None

    # Grid search over all configurations
    all_results = []
    best_config = None
    best_score = float("inf")

    total_configs = 0
    for _order in orders:
        for method in smoothing_methods:
            # Determine if method uses discount_mass
            total_configs += len(discount_masses) if method in ["auto", "fixed"] else 1

    # Add uniform baseline if requested
    if include_uniform:
        total_configs += 1

    config_idx = 0
    results_by_order = {order: [] for order in orders}
    results_by_method = {method: [] for method in smoothing_methods}

    # Add order 0 for uniform if requested
    if include_uniform:
        results_by_order[0] = []

    try:
        # Evaluate uniform baseline first if requested (order 0)
        if include_uniform:
            config_idx += 1

            if verbose:
                print(f"\n[{config_idx}/{total_configs}] Testing uniform (0-gram) baseline...", flush=True)

            start_time = time.time()

            try:
                # Create uniform model
                # First, get vocabulary from a quick pass through training data
                vocab_set = set()
                with open(train_corpus_file) as f:
                    for line in f:
                        words = line.strip().split()
                        vocab_set.update(words)

                # Add sentence markers
                vocab_set.add("<s>")
                vocab_set.add("</s>")
                vocab = sorted(vocab_set)

                uniform_lm = ArpaBoLM.create_uniform(vocab, add_start=True)

                # Evaluate uniform model
                if evaluation_mode == "source":
                    # For cross-validation, need to evaluate on each fold
                    with open(train_corpus_file) as f:
                        lines = [line for line in f if line.strip()]

                    fold_size = len(lines) // k_folds
                    fold_perplexities = []

                    for fold_idx in range(k_folds):
                        test_start = fold_idx * fold_size
                        test_end = test_start + fold_size if fold_idx < k_folds - 1 else len(lines)
                        test_lines = lines[test_start:test_end]

                        test_corpus = StringIO("\n".join(test_lines))
                        fold_result = uniform_lm.perplexity(test_corpus)
                        fold_perplexities.append(fold_result["perplexity"])

                    mean_ppl = sum(fold_perplexities) / len(fold_perplexities)
                    # Calculate std dev
                    mean = mean_ppl
                    variance = (
                        sum((x - mean) ** 2 for x in fold_perplexities) / (len(fold_perplexities) - 1)
                        if len(fold_perplexities) > 1
                        else 0.0
                    )
                    std_ppl = math.sqrt(variance)

                    eval_metrics = {
                        "perplexity": mean_ppl,
                        "cross_entropy": math.log2(mean_ppl),  # Approximation
                        "std_perplexity": std_ppl,
                    }
                    eval_score = mean_ppl

                else:
                    # Holdout or external
                    with open(eval_file) as f:
                        result = uniform_lm.perplexity(f)

                    eval_score = result["perplexity"]
                    eval_metrics = {
                        "perplexity": result["perplexity"],
                        "cross_entropy": result["cross_entropy"],
                        "oov_rate": result["oov_rate"],
                    }

                elapsed = time.time() - start_time

                # Record uniform result
                uniform_result = {
                    "order": 0,
                    "smoothing_method": "uniform",
                    "discount_mass": None,
                    "training_time": elapsed,
                    **eval_metrics,
                }

                all_results.append(uniform_result)
                results_by_order[0].append(uniform_result)

                if verbose:
                    print(f"  → Perplexity: {eval_metrics['perplexity']:.1f} (time: {elapsed:.1f}s)")
                    print("  → Uniform baseline (lower bound)")

                # Track as best if first config
                if eval_score < best_score:
                    best_score = eval_score
                    best_config = uniform_result.copy()

            except Exception as e:
                if verbose:
                    print(f"  → Failed: {e}")

        for order in orders:
            if verbose and show_comparisons and len(all_results) > 0:
                print()
                print(f"\n{'=' * 70}")
                print(f"Completed order {order - 1}, starting order {order}")
                print(f"{'=' * 70}")
                _print_intermediate_comparison(all_results, results_by_order, order - 1)

            for method in smoothing_methods:
                # Determine parameter ranges
                param_values = discount_masses if method in ["auto", "fixed"] else [None]

                method_results = []

                for discount_mass in param_values:
                    config_idx += 1

                    if verbose:
                        method_desc = method
                        if discount_mass is not None:
                            method_desc += f" (d={discount_mass})"
                        print(f"\n[{config_idx}/{total_configs}] Testing {order}-gram {method_desc}...", flush=True)

                    start_time = time.time()

                    # Evaluate configuration
                    try:
                        if evaluation_mode == "source":
                            # Cross-validation
                            result = _evaluate_config_cv(
                                corpus_file=train_corpus_file,
                                order=order,
                                smoothing_method=method,
                                discount_mass=discount_mass,
                                k_folds=k_folds,
                                verbose=False,
                            )
                            eval_score = result["mean_perplexity"]
                            eval_metrics = {
                                "perplexity": result["mean_perplexity"],
                                "cross_entropy": result["mean_cross_entropy"],
                                "std_perplexity": result["std_perplexity"],
                            }

                        else:
                            # Holdout or external
                            result, trained_lm = _evaluate_config(
                                train_file=train_corpus_file,
                                test_file=eval_file,
                                order=order,
                                smoothing_method=method,
                                discount_mass=discount_mass,
                                verbose=False,
                            )
                            eval_score = result["perplexity"]
                            eval_metrics = {
                                "perplexity": result["perplexity"],
                                "cross_entropy": result["cross_entropy"],
                                "oov_rate": result["oov_rate"],
                            }

                        elapsed = time.time() - start_time

                        # Record result
                        config_result = {
                            "order": order,
                            "smoothing_method": method,
                            "discount_mass": discount_mass,
                            "training_time": elapsed,
                            **eval_metrics,
                        }

                        # For auto method, capture the optimized discount_mass
                        if method == "auto" and evaluation_mode != "source" and hasattr(trained_lm, "discount_mass"):
                            config_result["optimized_discount_mass"] = trained_lm.discount_mass

                        all_results.append(config_result)
                        results_by_order[order].append(config_result)
                        results_by_method[method].append(config_result)
                        method_results.append(config_result)

                        if verbose:
                            print(f"  → Perplexity: {eval_metrics['perplexity']:.1f} (time: {elapsed:.1f}s)")

                            # Show comparison vs best so far
                            if best_config:
                                improvement = (
                                    (best_config["perplexity"] - eval_metrics["perplexity"]) / best_config["perplexity"]
                                ) * 100
                                if improvement > 0:
                                    print(f"  * New best! {improvement:.1f}% better than previous best")
                                else:
                                    print(f"  → {abs(improvement):.1f}% worse than current best")

                        # Track best configuration
                        if eval_score < best_score:
                            best_score = eval_score
                            best_config = config_result.copy()

                    except Exception as e:
                        if verbose:
                            print(f"  → Failed: {e}")
                        continue

                # Show comparison for this method across parameters
                if verbose and show_comparisons and len(method_results) > 1:
                    _print_method_parameter_comparison(method_results, method)

    finally:
        # Clean up temporary files if using holdout mode
        if evaluation_mode == "holdout":
            try:
                os.unlink(train_path)
                os.unlink(dev_path)
            except Exception:
                pass

    if best_config is None:
        raise ValueError("No valid configurations found. All evaluations failed.")

    # Build comprehensive results
    optimization_results = {
        "best_config": best_config,
        "all_results": all_results,
        "results_by_order": results_by_order,
        "results_by_method": results_by_method,
        "evaluation_mode": evaluation_mode,
        "corpus_file": corpus_file,
        "test_file": test_file if evaluation_mode == "external" else None,
        "search_space": {
            "orders": orders,
            "smoothing_methods": smoothing_methods,
            "discount_masses": discount_masses if any(r["discount_mass"] is not None for r in all_results) else None,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Export results if requested
    if export_results:
        with open(export_results, "w") as f:
            json.dump(optimization_results, f, indent=2)
        if verbose:
            print(f"\nExported results to {export_results}")

    # Print summary
    if verbose:
        print()
        print("=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print()
        print("Best Configuration:")
        print("-" * 70)
        print(f"  Order:          {best_config['order']}")
        print(f"  Smoothing:      {best_config['smoothing_method']}")
        if best_config["discount_mass"] is not None:
            print(f"  Discount mass:  {best_config['discount_mass']}")
        print(f"  Perplexity:     {best_config['perplexity']:.1f}")
        print(f"  Cross-entropy:  {best_config['cross_entropy']:.2f} bits/word")
        if "std_perplexity" in best_config:
            print(f"  Std perplexity: ±{best_config['std_perplexity']:.1f}")
        if "oov_rate" in best_config:
            print(f"  OOV rate:       {best_config['oov_rate'] * 100:.2f}%")
        print(f"  Training time:  {best_config['training_time']:.1f}s")

        if show_comparisons:
            print()
            _print_final_summary(all_results, results_by_order, results_by_method, best_config)
            print()
            plot_optimization_results(optimization_results)

    return optimization_results


def _print_intermediate_comparison(all_results: list[dict], results_by_order: dict, completed_order: int) -> None:
    """Print comparison of results for a completed order."""
    if completed_order not in results_by_order or not results_by_order[completed_order]:
        return

    order_results = results_by_order[completed_order]
    print(f"\nSummary for {completed_order}-gram models:")
    print(f"{'Method':<20} {'Discount':<10} {'PPL':>10} {'Entropy':>10} {'Time(s)':>10}")
    print("-" * 70)

    for result in sorted(order_results, key=lambda x: x["perplexity"]):
        discount_str = f"{result['discount_mass']:.1f}" if result["discount_mass"] else "N/A"
        ppl = result["perplexity"]
        entropy = result["cross_entropy"]
        time_sec = result["training_time"]
        print(f"{result['smoothing_method']:<20} {discount_str:<10} {ppl:>10.1f} {entropy:>10.2f} {time_sec:>10.1f}")

    # Show best for this order
    best_for_order = min(order_results, key=lambda x: x["perplexity"])
    print(
        f"\n→ Best {completed_order}-gram: {best_for_order['smoothing_method']} "
        f"(PPL={best_for_order['perplexity']:.1f})"
    )


def _print_method_parameter_comparison(method_results: list[dict], method: str) -> None:
    """Print comparison of different parameters for the same method."""
    if len(method_results) <= 1:
        return

    print(f"\n  Parameter comparison for {method}:")
    print(f"  {'Discount':<12} {'PPL':>10} {'Entropy':>10}")
    print(f"  {'-' * 34}")

    for result in sorted(method_results, key=lambda x: x["perplexity"]):
        discount_str = f"{result['discount_mass']:.2f}" if result["discount_mass"] else "N/A"
        print(f"  {discount_str:<12} {result['perplexity']:>10.1f} {result['cross_entropy']:>10.2f}")


def _print_final_summary(
    all_results: list[dict], results_by_order: dict, results_by_method: dict, best_config: dict
) -> None:
    """Print comprehensive final summary with multiple comparison views."""
    print()
    print("=" * 70)
    print("DETAILED COMPARISON ANALYSIS")
    print("=" * 70)

    # Top configurations
    print()
    print("Top 10 Configurations (by perplexity):")
    print("-" * 70)
    print(f"{'Rank':<6} {'Order':<7} {'Method':<15} {'Discount':<10} {'PPL':>10} {'Entropy':>10}")
    print("-" * 70)

    sorted_results = sorted(all_results, key=lambda x: x["perplexity"])
    for rank, result in enumerate(sorted_results[:10], 1):
        discount_str = f"{result['discount_mass']:.1f}" if result["discount_mass"] else "N/A"
        marker = " *" if result == best_config else ""
        print(
            f"{rank:<6} {result['order']:<7} {result['smoothing_method']:<15} "
            f"{discount_str:<10} {result['perplexity']:>10.1f} {result['cross_entropy']:>10.2f}{marker}"
        )

    # Comparison by order
    print()
    print("=" * 70)
    print("Comparison by N-gram Order:")
    print("-" * 70)
    print(f"{'Order':<8} {'Best Method':<20} {'Best PPL':>12} {'Avg PPL':>12} {'# Configs':>12}")
    print("-" * 70)

    for order in sorted(results_by_order.keys()):
        if not results_by_order[order]:
            continue

        order_results = results_by_order[order]
        best = min(order_results, key=lambda x: x["perplexity"])
        avg_ppl = sum(r["perplexity"] for r in order_results) / len(order_results)

        print(
            f"{order:<8} {best['smoothing_method']:<20} {best['perplexity']:>12.1f} "
            f"{avg_ppl:>12.1f} {len(order_results):>12}"
        )

    # Comparison by smoothing method
    print()
    print("=" * 70)
    print("Comparison by Smoothing Method:")
    print("-" * 70)
    print(f"{'Method':<20} {'Best Order':<12} {'Best PPL':>12} {'Avg PPL':>12} {'# Configs':>12}")
    print("-" * 70)

    for method in sorted(results_by_method.keys()):
        if not results_by_method[method]:
            continue

        method_results = results_by_method[method]
        best = min(method_results, key=lambda x: x["perplexity"])
        avg_ppl = sum(r["perplexity"] for r in method_results) / len(method_results)

        print(
            f"{method:<20} {best['order']:<12} {best['perplexity']:>12.1f} {avg_ppl:>12.1f} {len(method_results):>12}"
        )

    # Perplexity vs Training Time trade-off
    print()
    print("=" * 70)
    print("Perplexity vs Training Time Trade-offs:")
    print("-" * 70)
    print(f"{'Config':<35} {'PPL':>10} {'Time(s)':>10} {'PPL/sec':>12}")
    print("-" * 70)

    # Show configurations sorted by efficiency (PPL per second of training)
    for result in sorted(all_results, key=lambda x: x["perplexity"] / max(x["training_time"], 0.1))[:5]:
        config_name = f"{result['order']}-gram {result['smoothing_method']}"
        if result["discount_mass"]:
            config_name += f" (d={result['discount_mass']:.1f})"
        efficiency = result["perplexity"] / max(result["training_time"], 0.1)
        print(f"{config_name:<35} {result['perplexity']:>10.1f} {result['training_time']:>10.1f} {efficiency:>12.1f}")

    # Analysis insights
    print()
    print("=" * 70)
    print("Key Insights:")
    print("-" * 70)

    # Order analysis
    order_winners = {}
    for order, order_results in results_by_order.items():
        if order_results:
            best = min(order_results, key=lambda x: x["perplexity"])
            order_winners[order] = best

    if len(order_winners) > 1:
        orders_sorted = sorted(order_winners.items(), key=lambda x: x[1]["perplexity"])
        best_order_config = orders_sorted[0][1]
        print(f"• Best order: {best_order_config['order']}-gram (PPL={best_order_config['perplexity']:.1f})")

        if len(orders_sorted) > 1:
            second_best = orders_sorted[1][1]
            improvement = (
                (second_best["perplexity"] - best_order_config["perplexity"]) / second_best["perplexity"]
            ) * 100
            print(f"  {improvement:.1f}% better than {second_best['order']}-gram")

    # Method analysis
    method_winners = {}
    for method, method_results in results_by_method.items():
        if method_results:
            best = min(method_results, key=lambda x: x["perplexity"])
            method_winners[method] = best

    if len(method_winners) > 1:
        methods_sorted = sorted(method_winners.items(), key=lambda x: x[1]["perplexity"])
        best_method_config = methods_sorted[0][1]
        print(
            f"\n• Best smoothing: {best_method_config['smoothing_method']} (PPL={best_method_config['perplexity']:.1f})"
        )

        if len(methods_sorted) > 1:
            second_best = methods_sorted[1][1]
            improvement = (
                (second_best["perplexity"] - best_method_config["perplexity"]) / second_best["perplexity"]
            ) * 100
            print(f"  {improvement:.1f}% better than {second_best['smoothing_method']}")

    # Time analysis
    fastest = min(all_results, key=lambda x: x["training_time"])
    slowest = max(all_results, key=lambda x: x["training_time"])
    print(f"\n• Training time range: {fastest['training_time']:.1f}s - {slowest['training_time']:.1f}s")
    print(f"  Fastest: {fastest['order']}-gram {fastest['smoothing_method']} (PPL={fastest['perplexity']:.1f})")
    print(f"  Slowest: {slowest['order']}-gram {slowest['smoothing_method']} (PPL={slowest['perplexity']:.1f})")


def _evaluate_config(
    train_file: str, test_file: str, order: int, smoothing_method: str, discount_mass: Optional[float], verbose: bool
) -> tuple[dict, "ArpaBoLM"]:
    """
    Evaluate a single configuration with train/test split.

    Args:
        train_file: Training corpus
        test_file: Test corpus
        order: N-gram order
        smoothing_method: Smoothing method
        discount_mass: Discount parameter (if applicable)
        verbose: Verbose output

    Returns:
        Tuple of (evaluation metrics dictionary, trained model)
    """
    # Build kwargs for ArpaBoLM
    kwargs = {"max_order": order, "smoothing_method": smoothing_method, "verbose": verbose}

    if discount_mass is not None:
        kwargs["discount_mass"] = discount_mass

    # Train model
    lm = ArpaBoLM(**kwargs)

    with open(train_file) as f:
        lm.read_corpus(f)

    lm.compute()

    # Evaluate
    with open(test_file) as f:
        results = lm.perplexity(f)

    return results, lm


def _evaluate_config_cv(
    corpus_file: str,
    order: int,
    smoothing_method: str,
    discount_mass: Optional[float],
    k_folds: int,
    verbose: bool,
) -> dict:
    """
    Evaluate a single configuration using k-fold cross-validation.

    Args:
        corpus_file: Corpus file
        order: N-gram order
        smoothing_method: Smoothing method
        discount_mass: Discount parameter (if applicable)
        k_folds: Number of folds
        verbose: Verbose output

    Returns:
        Dictionary with mean and std metrics
    """
    # Read corpus
    with open(corpus_file) as f:
        lines = [line for line in f if line.strip()]

    if len(lines) < k_folds:
        raise ValueError(f"Corpus has {len(lines)} lines but k_folds={k_folds}. Need more data.")

    fold_size = len(lines) // k_folds

    fold_results = []

    # Perform k-fold CV
    for fold_idx in range(k_folds):
        # Split data
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < k_folds - 1 else len(lines)

        test_lines = lines[test_start:test_end]
        train_lines = lines[:test_start] + lines[test_end:]

        # Build kwargs
        kwargs = {"max_order": order, "smoothing_method": smoothing_method, "verbose": False}

        if discount_mass is not None:
            kwargs["discount_mass"] = discount_mass

        # Train model
        lm = ArpaBoLM(**kwargs)

        for line in train_lines:
            lm.read_corpus_line(line)

        lm.compute()

        # Evaluate
        test_corpus = StringIO("\n".join(test_lines))
        result = lm.perplexity(test_corpus)

        fold_results.append(
            {
                "perplexity": result["perplexity"],
                "cross_entropy": result["cross_entropy"],
                "oov_rate": result["oov_rate"],
            }
        )

    # Aggregate results
    perplexities = [fr["perplexity"] for fr in fold_results]
    entropies = [fr["cross_entropy"] for fr in fold_results]
    oov_rates = [fr["oov_rate"] for fr in fold_results]

    mean_ppl = sum(perplexities) / len(perplexities)
    mean_entropy = sum(entropies) / len(entropies)
    mean_oov = sum(oov_rates) / len(oov_rates)

    # Calculate standard deviation
    def std_dev(values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    return {
        "mean_perplexity": mean_ppl,
        "std_perplexity": std_dev(perplexities),
        "mean_cross_entropy": mean_entropy,
        "std_cross_entropy": std_dev(entropies),
        "mean_oov_rate": mean_oov,
        "fold_results": fold_results,
    }


def get_optimization_preset(preset: str) -> dict:
    """
    Get predefined search space configurations for common use cases.

    Args:
        preset: Preset name:
            - "quick": Fast search (2-3 grams, 2 methods) - ~4 configs
            - "standard": Balanced search (1-4 grams, 2 methods) - ~8 configs
            - "thorough": Comprehensive search (1-5 grams, 3 methods) - ~15 configs
            - "asr": ASR-focused (2-3 grams, best methods) - ~4 configs
            - "minimal": Single best guess (3-gram, kneser_ney) - 1 config

    Returns:
        Dictionary with orders, smoothing_methods, and optionally discount_masses

    Raises:
        ValueError: If preset name is unknown

    Examples:
        # Use preset
        config = get_optimization_preset("quick")
        results = optimize_hyperparameters(
            corpus_file="train.txt",
            **config
        )

        # Override preset
        config = get_optimization_preset("standard")
        config["orders"] = [2, 3, 4]  # Skip unigram
        results = optimize_hyperparameters(
            corpus_file="train.txt",
            **config
        )
    """
    presets = {
        "quick": {
            "orders": [1, 2, 3],
            "smoothing_methods": ["good_turing", "kneser_ney"],
            "description": "Fast search with baseline (1-gram) and practical options (2-3 grams)",
        },
        "standard": {
            "orders": [1, 2, 3, 4],
            "smoothing_methods": ["good_turing", "kneser_ney"],
            "description": "Balanced search from baseline to 4-gram",
        },
        "thorough": {
            "orders": [1, 2, 3, 4, 5],
            "smoothing_methods": ["good_turing", "kneser_ney", "auto"],
            "discount_masses": [0.5, 0.7, 0.9],
            "description": "Comprehensive search including higher orders and parameter tuning",
        },
        "asr": {
            "orders": [2, 3],
            "smoothing_methods": ["kneser_ney", "good_turing"],
            "description": "ASR-focused (bigram/trigram only, proven methods)",
        },
        "minimal": {
            "orders": [3],
            "smoothing_methods": ["kneser_ney"],
            "description": "Single best configuration (trigram Kneser-Ney)",
        },
    }

    if preset not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(f"Unknown preset: '{preset}'. Available: {available}")

    config = presets[preset].copy()
    # Remove description from returned config
    config.pop("description", None)

    return config


def plot_optimization_results(results: dict, use_matplotlib: bool = True, output_file: Optional[str] = None) -> None:
    """
    Visualize optimization results with optional matplotlib plots.

    Shows text summary and optionally creates matplotlib plots if available.
    Matplotlib is not required - falls back to text-only display.

    Args:
        results: Results dictionary from optimize_hyperparameters()
        use_matplotlib: Try to use matplotlib if installed (default: True)
        output_file: Optional path to save matplotlib figure (e.g., "results.png")

    Example:
        results = optimize_hyperparameters("corpus.txt")

        # Text summary only (no matplotlib needed)
        plot_optimization_results(results, use_matplotlib=False)

        # Save to file (doesn't block)
        plot_optimization_results(results, output_file="optimization.png")

        # To display interactively, save then show manually:
        plot_optimization_results(results, output_file="plot.png")
        import matplotlib.pyplot as plt
        plt.show()  # This will block until window is closed
    """
    all_results = results["all_results"]
    results_by_order = results["results_by_order"]
    results_by_method = results["results_by_method"]

    if not all_results:
        print("No results to plot")
        return

    best = results["best_config"]
    worst = max(all_results, key=lambda x: x["perplexity"])

    # Text Summary (always shown)
    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 70)
    print()
    print("Best Configuration:")
    print(f"  {best['order']}-gram {best['smoothing_method']}")
    print(f"  Perplexity: {best['perplexity']:.1f}")
    print(f"  Cross-entropy: {best['cross_entropy']:.2f} bits/word")
    if "std_perplexity" in best:
        print(f"  Std dev: ±{best['std_perplexity']:.1f}")

    improvement = ((worst["perplexity"] - best["perplexity"]) / worst["perplexity"]) * 100
    print()
    print("Performance:")
    print(f"  Best perplexity:  {best['perplexity']:.1f}")
    print(f"  Worst perplexity: {worst['perplexity']:.1f}")
    print(f"  Improvement:      {improvement:.1f}%")
    print(f"  Configs tried:    {len(all_results)}")

    # Best by order
    print()
    print("Best by N-gram Order:")
    for order in sorted(results_by_order.keys()):
        if results_by_order[order]:
            order_best = min(results_by_order[order], key=lambda x: x["perplexity"])
            order_name = "uniform" if order == 0 else f"{order}-gram"
            marker = " *" if order_best == best else ""
            print(
                f"  {order_name:<10} {order_best['perplexity']:>8.1f} PPL  ({order_best['smoothing_method']}){marker}"
            )

    # Best by method
    print()
    print("Best by Smoothing Method:")
    for method in sorted(results_by_method.keys()):
        if results_by_method[method]:
            method_best = min(results_by_method[method], key=lambda x: x["perplexity"])
            marker = " *" if method_best == best else ""
            print(f"  {method:<15} {method_best['perplexity']:>8.1f} PPL  ({method_best['order']}-gram){marker}")

    # Try to create matplotlib plots if requested
    if use_matplotlib:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch

            # Create comprehensive figure with 2 rows
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # Plot 1: All results ranked by perplexity
            sorted_results = sorted(all_results, key=lambda x: x["perplexity"])

            labels = []
            ppls = []
            colors = []

            method_colors = {
                "uniform": "lightgray",
                "good_turing": "steelblue",
                "kneser_ney": "green",
                "auto": "coral",
                "fixed": "purple",
            }

            for r in sorted_results:
                method = r["smoothing_method"]
                order = r["order"]
                discount = r.get("discount_mass")

                if order == 0:
                    label = "U"
                elif discount is None:
                    label = f"{order}{method[0].upper()}"
                else:
                    label = f"{order}{method[0].upper()}{discount:.1f}"

                labels.append(label)
                ppls.append(r["perplexity"])
                colors.append(method_colors.get(method, "gray"))

            ax1.bar(range(len(labels)), ppls, color=colors, alpha=0.8)
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=90, fontsize=6)
            ax1.set_ylabel("Perplexity (lower is better)", fontsize=10)
            ax1.set_title("All Configurations Ranked by Perplexity", fontsize=12, fontweight="bold")
            ax1.grid(axis="y", alpha=0.3)

            # Mark best
            ax1.axhline(y=ppls[0], color="red", linestyle="--", linewidth=1, alpha=0.5)

            # Legend
            legend_elements = [
                Patch(facecolor=color, label=method)
                for method, color in method_colors.items()
                if any(r["smoothing_method"] == method for r in all_results)
            ]
            ax1.legend(handles=legend_elements, loc="upper left", fontsize=8)

            # Plot 2: Best by order
            order_best_ppl = {}
            for order, order_results in results_by_order.items():
                if order_results:
                    order_best = min(order_results, key=lambda x: x["perplexity"])
                    order_best_ppl[order] = order_best["perplexity"]

            if order_best_ppl:
                orders = sorted(order_best_ppl.keys())
                ppls_by_order = [order_best_ppl[o] for o in orders]
                labels_order = ["uniform" if o == 0 else f"{o}-gram" for o in orders]

                ax2.bar(labels_order, ppls_by_order, color=["lightgray" if o == 0 else "steelblue" for o in orders])
                ax2.set_xlabel("N-gram Order", fontsize=10)
                ax2.set_ylabel("Best Perplexity", fontsize=10)
                ax2.set_title("Best Perplexity by N-gram Order", fontsize=12, fontweight="bold")
                ax2.grid(axis="y", alpha=0.3)

            # Plot 3: Best by method
            method_best_ppl = {}
            for method, method_results in results_by_method.items():
                if method_results:
                    method_best = min(method_results, key=lambda x: x["perplexity"])
                    method_best_ppl[method] = method_best["perplexity"]

            if method_best_ppl:
                methods = sorted(method_best_ppl.keys())
                ppls_by_method = [method_best_ppl[m] for m in methods]

                ax3.bar(methods, ppls_by_method, color="coral")
                ax3.set_xlabel("Smoothing Method", fontsize=10)
                ax3.set_ylabel("Best Perplexity", fontsize=10)
                ax3.set_title("Best Perplexity by Smoothing Method", fontsize=12, fontweight="bold")
                ax3.grid(axis="y", alpha=0.3)
                ax3.tick_params(axis="x", rotation=45)

            # Plot 4: Discount mass sensitivity (if applicable)
            auto_results = [
                (r["order"], r.get("discount_mass"), r["perplexity"])
                for r in all_results
                if r["smoothing_method"] == "auto" and r.get("discount_mass") is not None
            ]
            fixed_results = [
                (r["order"], r.get("discount_mass"), r["perplexity"])
                for r in all_results
                if r["smoothing_method"] == "fixed" and r.get("discount_mass") is not None
            ]

            if auto_results or fixed_results:
                has_plots = False

                # Group by order for auto
                if auto_results:
                    for order in sorted({o for o, _, _ in auto_results}):
                        order_auto = sorted([(d, p) for o, d, p in auto_results if o == order])
                        if order_auto:
                            discounts, perps = zip(*order_auto)
                            ax4.plot(
                                discounts, perps, marker="o", label=f"auto {order}-gram", linewidth=2, markersize=6
                            )
                            has_plots = True

                # Group by order for fixed
                if fixed_results:
                    for order in sorted({o for o, _, _ in fixed_results}):
                        order_fixed = sorted([(d, p) for o, d, p in fixed_results if o == order])
                        if order_fixed:
                            discounts, perps = zip(*order_fixed)
                            ax4.plot(
                                discounts,
                                perps,
                                marker="s",
                                linestyle="--",
                                label=f"fixed {order}-gram",
                                linewidth=2,
                                alpha=0.7,
                                markersize=6,
                            )
                            has_plots = True

                if has_plots:
                    ax4.set_xlabel("Discount Mass", fontsize=10)
                    ax4.set_ylabel("Perplexity", fontsize=10)
                    ax4.set_title("Discount Mass Parameter Sensitivity (Katz Backoff)", fontsize=12, fontweight="bold")
                    ax4.legend(fontsize=8, loc="best")
                    ax4.grid(alpha=0.3)
                else:
                    ax4.text(
                        0.5,
                        0.5,
                        "No parameter tuning\nin search",
                        ha="center",
                        va="center",
                        fontsize=12,
                        transform=ax4.transAxes,
                    )
                    ax4.set_title("Discount Mass Parameter Sensitivity", fontsize=12, fontweight="bold")
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "No parameter tuning\nin search",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=ax4.transAxes,
                )
                ax4.set_title("Discount Mass Parameter Sensitivity", fontsize=12, fontweight="bold")
                ax4.set_xlabel("Discount Mass", fontsize=10)
                ax4.set_ylabel("Perplexity", fontsize=10)

            plt.tight_layout()

            # Save to file if requested
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches="tight")
                print()
                print(f"Plot saved to: {output_file}")
            # Note: We don't show() automatically to avoid blocking execution
            # Users can call plt.show() manually if they want interactive display

        except ImportError:
            if output_file or use_matplotlib:
                print()
                print("Note: matplotlib not installed. Install with:")
                print("  pip install matplotlib")
                print("Showing text summary only.")
        except Exception as e:
            print()
            print(f"Warning: Could not create matplotlib plots: {e}")
            print("Showing text summary only.")


def print_optimization_results(results: dict, detailed: bool = True) -> None:
    """
    Print formatted optimization results from optimize_hyperparameters().

    Args:
        results: Results dictionary from optimize_hyperparameters()
        detailed: Whether to show detailed comparison tables (default: True)

    Example:
        results = optimize_hyperparameters("corpus.txt", verbose=False)
        print_optimization_results(results)
    """
    best = results["best_config"]

    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Evaluation mode: {results['evaluation_mode']}")
    print(f"Corpus: {results['corpus_file']}")
    if results.get("test_file"):
        print(f"Test file: {results['test_file']}")
    print()

    print("Best Configuration:")
    print("-" * 70)
    print(f"  Order:          {best['order']}")
    print(f"  Smoothing:      {best['smoothing_method']}")
    if best["discount_mass"] is not None:
        print(f"  Discount mass:  {best['discount_mass']}")
    print(f"  Perplexity:     {best['perplexity']:.1f}")
    print(f"  Cross-entropy:  {best['cross_entropy']:.2f} bits/word")
    if "std_perplexity" in best:
        print(f"  Std perplexity: ±{best['std_perplexity']:.1f}")
    if "oov_rate" in best:
        print(f"  OOV rate:       {best['oov_rate'] * 100:.2f}%")
    print(f"  Training time:  {best['training_time']:.1f}s")

    if detailed:
        print()
        _print_final_summary(results["all_results"], results["results_by_order"], results["results_by_method"], best)
