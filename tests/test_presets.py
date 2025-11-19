"""Tests for preset configurations."""

from io import StringIO

import pytest

from arpabo import get_preset, list_presets
from arpabo.lm import ArpaBoLM


class TestPresetFunctions:
    """Test preset utility functions."""

    def test_list_presets(self):
        """Test list_presets returns all presets."""
        presets = list_presets()

        assert isinstance(presets, list)
        assert len(presets) > 0
        assert "first-pass" in presets
        assert "rescoring" in presets
        assert "balanced" in presets

    def test_get_preset_valid(self):
        """Test get_preset with valid preset name."""
        config = get_preset("first-pass")

        assert isinstance(config, dict)
        assert "description" in config
        assert "smoothing_method" in config
        assert "recommended_order" in config

    def test_get_preset_invalid(self):
        """Test get_preset with invalid preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("invalid_preset")

    def test_all_presets_have_required_keys(self):
        """Test that all presets have required configuration keys."""
        required_keys = {"description", "smoothing_method", "recommended_order", "rationale", "use_case"}

        for preset_name in list_presets():
            config = get_preset(preset_name)

            for key in required_keys:
                assert key in config, f"Preset '{preset_name}' missing key '{key}'"


class TestFromPreset:
    """Test ArpaBoLM.from_preset() method."""

    def test_from_preset_first_pass(self):
        """Test creating model from first-pass preset."""
        lm = ArpaBoLM.from_preset("first-pass")

        assert lm.max_order == 2
        assert lm.smoothing_method == "good_turing"

    def test_from_preset_rescoring(self):
        """Test creating model from rescoring preset."""
        lm = ArpaBoLM.from_preset("rescoring")

        assert lm.max_order == 4
        assert lm.smoothing_method == "kneser_ney"

    def test_from_preset_balanced(self):
        """Test creating model from balanced preset."""
        lm = ArpaBoLM.from_preset("balanced")

        assert lm.max_order == 3
        assert lm.smoothing_method == "kneser_ney"

    def test_from_preset_fast(self):
        """Test creating model from fast preset."""
        lm = ArpaBoLM.from_preset("fast")

        assert lm.max_order == 2
        assert lm.smoothing_method == "good_turing"

    def test_from_preset_accurate(self):
        """Test creating model from accurate preset."""
        lm = ArpaBoLM.from_preset("accurate")

        assert lm.max_order == 5
        assert lm.smoothing_method == "kneser_ney"

    def test_from_preset_with_override_order(self):
        """Test overriding preset order."""
        lm = ArpaBoLM.from_preset("first-pass", max_order=4)

        assert lm.max_order == 4  # Overridden
        assert lm.smoothing_method == "good_turing"  # From preset

    def test_from_preset_with_override_smoothing(self):
        """Test overriding preset smoothing."""
        lm = ArpaBoLM.from_preset("rescoring", smoothing_method="good_turing")

        assert lm.max_order == 4  # From preset
        assert lm.smoothing_method == "good_turing"  # Overridden

    def test_from_preset_with_multiple_overrides(self):
        """Test overriding multiple preset parameters."""
        lm = ArpaBoLM.from_preset("balanced", max_order=5, verbose=True, case="lower")

        assert lm.max_order == 5
        assert lm.smoothing_method == "kneser_ney"  # From preset
        assert lm.verbose is True
        assert lm.case == "lower"

    def test_from_preset_invalid(self):
        """Test that invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            ArpaBoLM.from_preset("invalid")


class TestPresetUsage:
    """Test using preset models in real workflows."""

    @pytest.fixture
    def corpus(self):
        """Sample corpus for testing."""
        return StringIO("the quick brown fox jumps over the lazy dog")

    def test_preset_model_can_train(self, corpus):
        """Test that preset model can train on corpus."""
        lm = ArpaBoLM.from_preset("first-pass")
        lm.read_corpus(corpus)
        lm.compute()

        assert lm.max_order == 2
        assert len(lm.probs[0]) > 0

    def test_preset_model_can_write(self, corpus, tmp_path):
        """Test that preset model can write ARPA file."""
        lm = ArpaBoLM.from_preset("balanced")
        lm.read_corpus(corpus)
        lm.compute()

        output_file = tmp_path / "preset.arpa"
        lm.write_file(str(output_file))

        assert output_file.exists()

    def test_all_presets_work(self, corpus):
        """Test that all presets can train successfully."""
        for preset_name in list_presets():
            lm = ArpaBoLM.from_preset(preset_name)
            lm.read_corpus(StringIO(corpus.getvalue()))
            lm.compute()

            # Should have computed probabilities
            assert len(lm.probs[0]) > 0

    def test_preset_comparison(self, corpus, tmp_path):
        """Test comparing different presets."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("the quick fox")

        results = {}

        for preset_name in ["first-pass", "balanced", "rescoring"]:
            lm = ArpaBoLM.from_preset(preset_name)
            lm.read_corpus(StringIO(corpus.getvalue()))
            lm.compute()

            with open(test_file) as f:
                ppl = lm.perplexity(f)

            results[preset_name] = ppl["perplexity"]

        # All should have valid perplexities
        for _preset_name, ppl_value in results.items():
            assert ppl_value > 0


class TestPresetDocumentation:
    """Test preset documentation and descriptions."""

    def test_all_presets_documented(self):
        """Test that all presets have complete documentation."""
        for preset_name in list_presets():
            config = get_preset(preset_name)

            # Check documentation fields
            assert config["description"]
            assert config["rationale"]
            assert config["use_case"]

            # Check configuration fields
            assert config["smoothing_method"] in ["good_turing", "kneser_ney", "katz", "auto", "fixed"]
            assert config["recommended_order"] >= 1
            assert config["recommended_order"] <= 10  # Sanity check

    def test_preset_orders_reasonable(self):
        """Test that preset orders are in reasonable range."""
        for preset_name in list_presets():
            config = get_preset(preset_name)
            order = config["recommended_order"]

            # Orders should be reasonable (1-5 for most cases)
            assert 1 <= order <= 10


class TestPresetPrintOutput:
    """Test print_presets output."""

    def test_print_presets(self, capsys):
        """Test that print_presets outputs formatted information."""
        from arpabo.presets import print_presets

        print_presets()

        captured = capsys.readouterr()
        assert "Available Presets" in captured.out

        # Should mention all presets
        for preset_name in list_presets():
            assert preset_name in captured.out
