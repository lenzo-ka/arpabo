"""Model comparison and optimization utilities."""

import json
import os
import time
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
        print(f"  OOV rate:      {eval_data['oov_rate']*100:>8.1f}%")

        if "overall_backoff_rate" in eval_data:
            print(f"  Backoff rate:  {eval_data['overall_backoff_rate']*100:>8.1f}%")

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
                    print(f"  → Good backoff rate ({backoff*100:.1f}%) for diverse N-best lists")
                elif backoff > 0.2:
                    print(f"  → Moderate backoff rate ({backoff*100:.1f}%)")
                else:
                    print(f"  → Low backoff rate ({backoff*100:.1f}%) - sharp predictions")

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
