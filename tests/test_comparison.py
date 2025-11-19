"""Tests for ModelComparison class."""

import json
import os

import pytest

from arpabo.comparison import ModelComparison


class TestModelComparisonInit:
    """Test ModelComparison initialization."""

    def test_init(self, tmp_path):
        """Test basic initialization."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("the quick brown fox\nthe lazy dog")

        comparison = ModelComparison(corpus_file=str(corpus_file))

        assert comparison.corpus_file == str(corpus_file)
        assert comparison.smoothing_method == "kneser_ney"
        assert comparison.models == {}
        assert comparison.evaluations == {}

    def test_init_with_options(self, tmp_path):
        """Test initialization with custom options."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("test")

        comparison = ModelComparison(corpus_file=str(corpus_file), smoothing_method="good_turing", verbose=True)

        assert comparison.smoothing_method == "good_turing"
        assert comparison.verbose is True


class TestTrainOrders:
    """Test training multiple orders."""

    @pytest.fixture
    def corpus_file(self, tmp_path):
        """Create corpus file for testing."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            """the quick brown fox jumps over the lazy dog
the lazy dog sleeps under the brown tree
the quick fox runs through the green field"""
        )
        return corpus

    def test_train_orders_basic(self, corpus_file):
        """Test training multiple orders."""
        comparison = ModelComparison(corpus_file=str(corpus_file))
        models = comparison.train_orders([1, 2, 3])

        assert len(models) == 3
        assert 1 in models
        assert 2 in models
        assert 3 in models

        assert models[1].max_order == 1
        assert models[2].max_order == 2
        assert models[3].max_order == 3

    def test_train_orders_stores_models(self, corpus_file):
        """Test that trained models are stored."""
        comparison = ModelComparison(corpus_file=str(corpus_file))
        comparison.train_orders([2, 3])

        assert 2 in comparison.models
        assert 3 in comparison.models

    def test_train_non_sequential(self, corpus_file):
        """Test training non-sequential orders."""
        comparison = ModelComparison(corpus_file=str(corpus_file))
        models = comparison.train_orders([1, 3, 5])

        assert len(models) == 3
        assert all(order in models for order in [1, 3, 5])


class TestAddUniformBaseline:
    """Test adding uniform baseline."""

    @pytest.fixture
    def trained_comparison(self, tmp_path):
        """Create comparison with trained models."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the cat sat on the mat")

        comparison = ModelComparison(corpus_file=str(corpus))
        comparison.train_orders([1, 2, 3])
        return comparison

    def test_add_uniform(self, trained_comparison):
        """Test adding uniform baseline."""
        uniform = trained_comparison.add_uniform_baseline()

        assert 0 in trained_comparison.models
        assert trained_comparison.models[0] == uniform
        assert uniform.smoothing_method == "uniform"

    def test_uniform_without_training(self, tmp_path):
        """Test that uniform baseline requires prior training."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("test")

        comparison = ModelComparison(corpus_file=str(corpus))

        with pytest.raises(ValueError, match="Must call train_orders"):
            comparison.add_uniform_baseline()


class TestEvaluate:
    """Test model evaluation."""

    @pytest.fixture
    def comparison_with_models(self, tmp_path):
        """Create comparison with trained models."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the quick brown fox\nthe lazy dog")

        test = tmp_path / "test.txt"
        test.write_text("the quick fox")

        comparison = ModelComparison(corpus_file=str(corpus))
        comparison.train_orders([1, 2, 3])

        return comparison, test

    def test_evaluate_basic(self, comparison_with_models):
        """Test basic evaluation."""
        comparison, test_file = comparison_with_models
        results = comparison.evaluate(test_file=str(test_file))

        assert len(results) == 3
        assert 1 in results
        assert 2 in results
        assert 3 in results

        # Check that results have expected keys
        for _order, eval_data in results.items():
            assert "perplexity" in eval_data
            assert "cross_entropy" in eval_data
            assert "num_words" in eval_data

    def test_evaluate_includes_backoff(self, comparison_with_models):
        """Test that evaluation includes backoff by default."""
        comparison, test_file = comparison_with_models
        results = comparison.evaluate(test_file=str(test_file))

        # Should have backoff data
        for order in [2, 3]:  # Skip unigram
            assert "overall_backoff_rate" in results[order]
            assert "order_usage" in results[order]

    def test_evaluate_without_backoff(self, comparison_with_models):
        """Test evaluation without backoff analysis."""
        comparison, test_file = comparison_with_models
        results = comparison.evaluate(test_file=str(test_file), include_backoff=False)

        # Should not have backoff data
        for order in [1, 2, 3]:
            assert "perplexity" in results[order]
            # Backoff keys should not be present (except for uniform)

    def test_evaluate_with_uniform(self, comparison_with_models):
        """Test evaluation with uniform baseline."""
        comparison, test_file = comparison_with_models
        comparison.add_uniform_baseline()

        results = comparison.evaluate(test_file=str(test_file))

        assert 0 in results  # Uniform baseline
        assert results[0]["overall_backoff_rate"] == 0.0  # Unigram has no backoff

    def test_evaluate_without_models(self, tmp_path):
        """Test that evaluation fails without trained models."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("test")

        comparison = ModelComparison(corpus_file=str(corpus))

        with pytest.raises(ValueError, match="Must call train_orders"):
            comparison.evaluate(test_file="test.txt")


class TestRecommend:
    """Test recommendation logic."""

    @pytest.fixture
    def evaluated_comparison(self, tmp_path):
        """Create comparison with trained and evaluated models."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            """the quick brown fox jumps over the lazy dog
the lazy dog sleeps under the brown tree
the quick fox runs through the green field"""
        )

        test = tmp_path / "test.txt"
        test.write_text("the quick brown fox")

        comparison = ModelComparison(corpus_file=str(corpus))
        comparison.train_orders([1, 2, 3, 4])
        comparison.evaluate(test_file=str(test))

        return comparison

    def test_recommend_rescoring(self, evaluated_comparison):
        """Test recommendation for rescoring goal."""
        best = evaluated_comparison.recommend(goal="rescoring")

        # Should be the order with best (lowest) perplexity
        perplexities = {
            order: eval_data["perplexity"] for order, eval_data in evaluated_comparison.evaluations.items() if order > 0
        }
        expected_best = min(perplexities.keys(), key=lambda o: perplexities[o])

        assert best == expected_best

    def test_recommend_first_pass(self, evaluated_comparison):
        """Test recommendation for first-pass goal."""
        best = evaluated_comparison.recommend(goal="first-pass")

        # Should be valid order
        assert best in evaluated_comparison.models
        assert best > 0  # Should not recommend uniform

    def test_recommend_with_uniform(self, evaluated_comparison):
        """Test that uniform baseline is excluded from recommendations."""
        evaluated_comparison.add_uniform_baseline()
        evaluated_comparison.evaluate(test_file=evaluated_comparison.corpus_file)

        best = evaluated_comparison.recommend(goal="first-pass")

        # Should not recommend order 0 (uniform)
        assert best > 0

    def test_recommend_without_evaluation(self, tmp_path):
        """Test that recommendation fails without evaluation."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("test")

        comparison = ModelComparison(corpus_file=str(corpus))
        comparison.train_orders([1, 2])

        with pytest.raises(ValueError, match="Must call evaluate"):
            comparison.recommend()

    def test_recommend_invalid_goal(self, evaluated_comparison):
        """Test that invalid goal raises error."""
        with pytest.raises(ValueError, match="Unknown goal"):
            evaluated_comparison.recommend(goal="invalid")


class TestExportForOptimization:
    """Test export functionality."""

    @pytest.fixture
    def comparison_ready_for_export(self, tmp_path):
        """Create comparison ready for export."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the quick brown fox\nthe lazy dog")

        test = tmp_path / "test.txt"
        test.write_text("the fox")

        comparison = ModelComparison(corpus_file=str(corpus))
        comparison.train_orders([1, 2, 3])
        comparison.evaluate(test_file=str(test))

        return comparison, tmp_path

    def test_export_creates_files(self, comparison_ready_for_export):
        """Test that export creates ARPA files."""
        comparison, tmp_path = comparison_ready_for_export
        output_dir = tmp_path / "output"

        comparison.export_for_optimization(output_dir=str(output_dir), convert_to_binary=False)

        # Check ARPA files
        for order in [1, 2, 3]:
            arpa_file = output_dir / f"{order}gram.arpa"
            assert arpa_file.exists()

    def test_export_creates_manifest(self, comparison_ready_for_export):
        """Test that export creates manifest.json."""
        comparison, tmp_path = comparison_ready_for_export
        output_dir = tmp_path / "output"

        manifest_path = comparison.export_for_optimization(output_dir=str(output_dir), convert_to_binary=False)

        assert os.path.exists(manifest_path)

        # Load and verify manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "corpus" in manifest
        assert "smoothing" in manifest
        assert "vocab_size" in manifest
        assert "models" in manifest
        assert len(manifest["models"]) == 3

    def test_manifest_includes_metrics(self, comparison_ready_for_export):
        """Test that manifest includes evaluation metrics."""
        comparison, tmp_path = comparison_ready_for_export
        output_dir = tmp_path / "output"

        manifest_path = comparison.export_for_optimization(output_dir=str(output_dir), convert_to_binary=False)

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Check first model has metrics
        model_info = manifest["models"][0]
        assert "perplexity" in model_info
        assert "cross_entropy" in model_info
        assert "oov_rate" in model_info
        assert "size_mb" in model_info

    def test_export_with_uniform(self, comparison_ready_for_export):
        """Test export with uniform baseline."""
        comparison, tmp_path = comparison_ready_for_export
        comparison.add_uniform_baseline()
        comparison.evaluate(test_file=str(tmp_path / "test.txt"))

        output_dir = tmp_path / "output"
        comparison.export_for_optimization(output_dir=str(output_dir), convert_to_binary=False)

        # Should have uniform.arpa
        assert (output_dir / "uniform.arpa").exists()

    def test_export_without_models(self, tmp_path):
        """Test that export fails without trained models."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("test")

        comparison = ModelComparison(corpus_file=str(corpus))

        with pytest.raises(ValueError, match="Must call train_orders"):
            comparison.export_for_optimization(output_dir="output")


class TestPrintComparison:
    """Test print_comparison method."""

    @pytest.fixture
    def evaluated_comparison(self, tmp_path):
        """Create evaluated comparison."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the quick brown fox\nthe lazy dog")

        test = tmp_path / "test.txt"
        test.write_text("the fox")

        comparison = ModelComparison(corpus_file=str(corpus))
        comparison.train_orders([1, 2, 3])
        comparison.evaluate(test_file=str(test))

        return comparison

    def test_print_comparison(self, evaluated_comparison, capsys):
        """Test print_comparison output."""
        evaluated_comparison.print_comparison()

        captured = capsys.readouterr()
        assert "Model Comparison" in captured.out
        assert "PPL" in captured.out
        assert "Entropy" in captured.out
        assert "Backoff" in captured.out
        assert "1-gram" in captured.out
        assert "2-gram" in captured.out
        assert "3-gram" in captured.out

    def test_print_comparison_without_evaluation(self, tmp_path):
        """Test that print_comparison fails without evaluation."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("test")

        comparison = ModelComparison(corpus_file=str(corpus))
        comparison.train_orders([1, 2])

        with pytest.raises(ValueError, match="Must call evaluate"):
            comparison.print_comparison()


class TestPrintRecommendation:
    """Test print_recommendation method."""

    @pytest.fixture
    def evaluated_comparison(self, tmp_path):
        """Create evaluated comparison."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the quick brown fox\nthe lazy dog")

        test = tmp_path / "test.txt"
        test.write_text("the fox")

        comparison = ModelComparison(corpus_file=str(corpus))
        comparison.train_orders([1, 2, 3])
        comparison.evaluate(test_file=str(test))

        return comparison

    def test_print_recommendation(self, evaluated_comparison, capsys):
        """Test print_recommendation output."""
        evaluated_comparison.print_recommendation(goal="first-pass")

        captured = capsys.readouterr()
        assert "Recommendation" in captured.out
        assert "Perplexity:" in captured.out
        assert "gram" in captured.out


class TestHelperMethods:
    """Test helper methods."""

    @pytest.fixture
    def trained_comparison(self, tmp_path):
        """Create comparison with trained models."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the cat sat")

        comparison = ModelComparison(corpus_file=str(corpus))
        comparison.train_orders([1, 2])
        return comparison

    def test_get_model(self, trained_comparison):
        """Test get_model method."""
        model = trained_comparison.get_model(1)
        assert model.max_order == 1

        model = trained_comparison.get_model(2)
        assert model.max_order == 2

    def test_get_model_invalid(self, trained_comparison):
        """Test get_model with invalid order."""
        with pytest.raises(KeyError, match="Order 5 not found"):
            trained_comparison.get_model(5)

    def test_list_models(self, trained_comparison):
        """Test list_models method."""
        orders = trained_comparison.list_models()
        assert orders == [1, 2]

    def test_list_models_with_uniform(self, trained_comparison):
        """Test list_models with uniform baseline."""
        trained_comparison.add_uniform_baseline()
        orders = trained_comparison.list_models()
        assert orders == [0, 1, 2]

    def test_summary(self, trained_comparison):
        """Test summary method."""
        summary = trained_comparison.summary()

        assert "corpus_file" in summary
        assert "smoothing_method" in summary
        assert "num_models" in summary
        assert "orders" in summary
        assert summary["num_models"] == 2
        assert summary["orders"] == [1, 2]


class TestCompleteWorkflow:
    """Test complete workflows."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow from training to export."""
        # Create data
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            """the quick brown fox jumps over the lazy dog
the lazy dog sleeps under the brown tree"""
        )

        test = tmp_path / "test.txt"
        test.write_text("the quick fox")

        # Complete workflow
        comparison = ModelComparison(corpus_file=str(corpus))

        # Train
        comparison.train_orders([1, 2, 3])
        assert len(comparison.models) == 3

        # Add baseline
        comparison.add_uniform_baseline()
        assert 0 in comparison.models

        # Evaluate
        comparison.evaluate(test_file=str(test))
        assert len(comparison.evaluations) == 4

        # Recommend
        best = comparison.recommend(goal="first-pass")
        assert best > 0

        # Export
        output_dir = tmp_path / "output"
        manifest = comparison.export_for_optimization(output_dir=str(output_dir), convert_to_binary=False)

        assert os.path.exists(manifest)
        assert (output_dir / "1gram.arpa").exists()
        assert (output_dir / "2gram.arpa").exists()
        assert (output_dir / "3gram.arpa").exists()
        assert (output_dir / "uniform.arpa").exists()

    def test_workflow_different_smoothing(self, tmp_path):
        """Test workflow with different smoothing method."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the cat sat on the mat")

        test = tmp_path / "test.txt"
        test.write_text("the cat")

        comparison = ModelComparison(corpus_file=str(corpus), smoothing_method="good_turing")

        comparison.train_orders([1, 2])
        comparison.evaluate(test_file=str(test))

        # Check smoothing method
        for model in comparison.models.values():
            if model.smoothing_method != "uniform":
                assert model.smoothing_method == "good_turing"

    def test_incremental_workflow(self, tmp_path):
        """Test that workflow can be done incrementally."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the quick brown fox")

        test = tmp_path / "test.txt"
        test.write_text("the fox")

        comparison = ModelComparison(corpus_file=str(corpus))

        # Step 1: Train some models
        comparison.train_orders([1, 2])

        # Step 2: Evaluate
        comparison.evaluate(test_file=str(test))

        # Step 3: Get results
        summary = comparison.summary()
        assert summary["num_models"] == 2

        # Step 4: Add more context
        comparison.add_uniform_baseline()
        comparison.evaluate(test_file=str(test))

        assert len(comparison.models) == 3
        assert 0 in comparison.evaluations
