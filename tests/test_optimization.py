"""Tests for hyperparameter optimization functionality."""

import json
import os
import tempfile

import pytest

from arpabo.comparison import optimize_hyperparameters, print_optimization_results


@pytest.fixture
def sample_corpus():
    """Create a small sample corpus for testing."""
    corpus = """the cat sat on the mat
the dog sat on the floor
the bird flew over the tree
a cat and a dog played together
the mat was on the floor
"""
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(corpus)
        yield path
    finally:
        os.unlink(path)


@pytest.fixture
def test_corpus():
    """Create a small test corpus."""
    corpus = """the cat sat on the mat
the dog played on the floor
"""
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(corpus)
        yield path
    finally:
        os.unlink(path)


def test_optimize_hyperparameters_holdout(sample_corpus):
    """Test hyperparameter optimization with holdout validation."""
    results = optimize_hyperparameters(
        corpus_file=sample_corpus,
        orders=[1, 2],
        smoothing_methods=["good_turing", "kneser_ney"],
        evaluation_mode="holdout",
        holdout_ratio=0.2,
        verbose=False,
        show_comparisons=False,
    )

    # Check result structure
    assert "best_config" in results
    assert "all_results" in results
    assert "results_by_order" in results
    assert "results_by_method" in results
    assert "evaluation_mode" in results

    # Check best config
    best = results["best_config"]
    assert "order" in best
    assert "smoothing_method" in best
    assert "perplexity" in best
    assert "cross_entropy" in best
    assert "training_time" in best

    # Check that we have results for all configurations
    assert len(results["all_results"]) == 4  # 2 orders × 2 methods

    # Check results by order
    assert 1 in results["results_by_order"]
    assert 2 in results["results_by_order"]

    # Check results by method
    assert "good_turing" in results["results_by_method"]
    assert "kneser_ney" in results["results_by_method"]


def test_optimize_hyperparameters_external(sample_corpus, test_corpus):
    """Test hyperparameter optimization with external test set."""
    results = optimize_hyperparameters(
        corpus_file=sample_corpus,
        orders=[1, 2],
        smoothing_methods=["good_turing"],
        evaluation_mode="external",
        test_file=test_corpus,
        verbose=False,
        show_comparisons=False,
    )

    assert "best_config" in results
    assert results["evaluation_mode"] == "external"
    assert results["test_file"] == test_corpus

    # Check OOV rate is present for external evaluation
    assert "oov_rate" in results["best_config"]


def test_optimize_hyperparameters_source(sample_corpus):
    """Test hyperparameter optimization with cross-validation."""
    results = optimize_hyperparameters(
        corpus_file=sample_corpus,
        orders=[1, 2],
        smoothing_methods=["good_turing"],
        evaluation_mode="source",
        k_folds=2,  # Small number for speed
        verbose=False,
        show_comparisons=False,
    )

    assert "best_config" in results
    assert results["evaluation_mode"] == "source"

    # Check that std_perplexity is present for cross-validation
    assert "std_perplexity" in results["best_config"]


def test_optimize_with_discount_masses(sample_corpus):
    """Test optimization with discount mass parameters."""
    results = optimize_hyperparameters(
        corpus_file=sample_corpus,
        orders=[2],
        smoothing_methods=["auto"],
        evaluation_mode="holdout",
        discount_masses=[0.5, 0.7],
        verbose=False,
        show_comparisons=False,
    )

    # Should have results for each discount mass
    assert len(results["all_results"]) == 2

    # Check that discount masses are recorded
    discount_masses = [r["discount_mass"] for r in results["all_results"]]
    assert 0.5 in discount_masses
    assert 0.7 in discount_masses


def test_export_results(sample_corpus):
    """Test exporting optimization results to JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        export_path = f.name

    try:
        optimize_hyperparameters(
            corpus_file=sample_corpus,
            orders=[1, 2],
            smoothing_methods=["good_turing"],
            evaluation_mode="holdout",
            verbose=False,
            show_comparisons=False,
            export_results=export_path,
        )

        # Check that file was created
        assert os.path.exists(export_path)

        # Check that JSON is valid and contains expected data
        with open(export_path) as f:
            exported = json.load(f)

        assert "best_config" in exported
        assert "all_results" in exported
        assert "timestamp" in exported
        assert "search_space" in exported

    finally:
        if os.path.exists(export_path):
            os.unlink(export_path)


def test_print_optimization_results(sample_corpus, capsys):
    """Test printing optimization results."""
    results = optimize_hyperparameters(
        corpus_file=sample_corpus,
        orders=[1, 2],
        smoothing_methods=["good_turing"],
        evaluation_mode="holdout",
        verbose=False,
        show_comparisons=False,
    )

    # Print with detailed comparison
    print_optimization_results(results, detailed=True)
    captured = capsys.readouterr()

    assert "OPTIMIZATION RESULTS" in captured.out
    assert "Best Configuration:" in captured.out
    assert "Perplexity:" in captured.out

    # Print without detailed comparison
    print_optimization_results(results, detailed=False)
    captured = capsys.readouterr()

    assert "OPTIMIZATION RESULTS" in captured.out
    assert "Best Configuration:" in captured.out


def test_optimization_verbose_output(sample_corpus, capsys):
    """Test that verbose mode produces expected output."""
    optimize_hyperparameters(
        corpus_file=sample_corpus,
        orders=[1, 2],
        smoothing_methods=["good_turing"],
        evaluation_mode="holdout",
        verbose=True,
        show_comparisons=True,
    )

    captured = capsys.readouterr()

    # Check for key output sections
    assert "Hyperparameter Optimization" in captured.out
    assert "Testing" in captured.out
    assert "OPTIMIZATION COMPLETE" in captured.out
    assert "Best Configuration:" in captured.out


def test_optimization_invalid_mode():
    """Test that invalid evaluation mode raises error."""
    with pytest.raises(ValueError, match="Invalid evaluation_mode"):
        optimize_hyperparameters(
            corpus_file="dummy.txt",
            evaluation_mode="invalid",
            verbose=False,
        )


def test_optimization_missing_test_file():
    """Test that missing test file for external mode raises error."""
    with pytest.raises(ValueError, match="test_file is required"):
        optimize_hyperparameters(
            corpus_file="dummy.txt",
            evaluation_mode="external",
            test_file=None,
            verbose=False,
        )


def test_optimization_results_ordering(sample_corpus):
    """Test that results are properly ordered by perplexity."""
    results = optimize_hyperparameters(
        corpus_file=sample_corpus,
        orders=[1, 2, 3],
        smoothing_methods=["good_turing"],
        evaluation_mode="holdout",
        verbose=False,
        show_comparisons=False,
    )

    # Check that best config has lowest perplexity
    best_ppl = results["best_config"]["perplexity"]

    for result in results["all_results"]:
        assert result["perplexity"] >= best_ppl


def test_optimization_with_alice_corpus():
    """Test optimization on real Alice corpus (integration test)."""
    import os

    alice_path = "src/arpabo/data/alice.txt"

    if not os.path.exists(alice_path):
        pytest.skip("Alice corpus not found")

    results = optimize_hyperparameters(
        corpus_file=alice_path,
        orders=[1, 2, 3],
        smoothing_methods=["good_turing", "kneser_ney"],
        evaluation_mode="holdout",
        holdout_ratio=0.15,
        include_uniform=True,
        verbose=False,
        show_comparisons=False,
    )

    # Check basic structure
    assert "best_config" in results
    assert "all_results" in results

    # Should have 7 results: uniform + (1,2,3) × (good_turing, kneser_ney)
    assert len(results["all_results"]) == 7

    # Check uniform baseline is present
    uniform_results = [r for r in results["all_results"] if r["order"] == 0]
    assert len(uniform_results) == 1
    assert uniform_results[0]["smoothing_method"] == "uniform"

    # Uniform should have worst perplexity (highest PPL)
    uniform_ppl = uniform_results[0]["perplexity"]
    best_ppl = results["best_config"]["perplexity"]
    assert uniform_ppl > best_ppl

    # Best should be one of the actual models (not uniform)
    assert results["best_config"]["order"] > 0

    # Perplexity should improve with order (generally)
    # Unigram should be worse than best 2 or 3-gram
    unigram_ppls = [r["perplexity"] for r in results["all_results"] if r["order"] == 1]
    higher_order_ppls = [r["perplexity"] for r in results["all_results"] if r["order"] >= 2]

    assert min(unigram_ppls) > min(higher_order_ppls)

    # Check that results_by_order includes all orders including 0
    assert 0 in results["results_by_order"]
    assert 1 in results["results_by_order"]
    assert 2 in results["results_by_order"]
    assert 3 in results["results_by_order"]


def test_optimization_with_uniform_baseline(sample_corpus):
    """Test that uniform baseline is correctly included and evaluated."""
    results = optimize_hyperparameters(
        corpus_file=sample_corpus,
        orders=[2],
        smoothing_methods=["good_turing"],
        evaluation_mode="holdout",
        include_uniform=True,
        verbose=False,
        show_comparisons=False,
    )

    # Should have 2 results: uniform + 2-gram good_turing
    assert len(results["all_results"]) == 2

    # Check uniform is present
    uniform_results = [r for r in results["all_results"] if r["order"] == 0]
    assert len(uniform_results) == 1
    assert uniform_results[0]["smoothing_method"] == "uniform"

    # Uniform should typically have worse perplexity than trained model
    # (This can vary with small/random data splits, so we just check it exists)
    uniform_ppl = uniform_results[0]["perplexity"]
    other_ppls = [r["perplexity"] for r in results["all_results"] if r["order"] > 0]
    # Just verify we have both uniform and trained results
    assert len(other_ppls) > 0
    assert uniform_ppl > 0


def test_plot_optimization_results(sample_corpus, capsys):
    """Test ASCII visualization of optimization results."""
    from arpabo import plot_optimization_results

    results = optimize_hyperparameters(
        corpus_file=sample_corpus,
        orders=[1, 2],
        smoothing_methods=["good_turing"],
        evaluation_mode="holdout",
        verbose=False,
        show_comparisons=False,
    )

    # Plot results
    plot_optimization_results(results, use_matplotlib=False)
    captured = capsys.readouterr()

    # Check for key sections in text summary
    assert "OPTIMIZATION RESULTS SUMMARY" in captured.out
    assert "Best Configuration:" in captured.out
    assert "Performance:" in captured.out
    assert "Best by N-gram Order:" in captured.out
    assert "Best by Smoothing Method:" in captured.out
