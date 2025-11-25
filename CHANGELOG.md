# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Hyperparameter Optimization Framework**
  - `optimize_hyperparameters()` function for automated model selection via grid search
  - Three evaluation modes:
    - `"holdout"`: Fast train/dev split (default, recommended)
    - `"external"`: Separate test file evaluation
    - `"source"`: K-fold cross-validation for robust estimates
  - Configurable search space with practical defaults:
    - Orders: [1, 2, 3] - includes unigram baseline
    - Methods: ["good_turing", "kneser_ney"] - most common
    - Discount masses: [0.5, 0.7, 0.9] - for Katz backoff variants
  - Support for uniform (0-gram) baseline as lower bound (`include_uniform=True`)
  - Captures optimized parameters from `auto` method

- **Visualization & Results Display**
  - `plot_optimization_results()` with dual output modes:
    - Clean text summary (always available, zero dependencies)
    - Comprehensive matplotlib plots (optional, 4-panel visualization)
  - Real-time progress tracking with comparison vs current best
  - Intermediate summaries after completing each n-gram order
  - Parameter comparison tables for tunable methods
  - Final analysis with multiple perspectives:
    - All configurations ranked by perplexity
    - Best by n-gram order and smoothing method
    - Discount mass parameter sensitivity curves
    - Key insights and improvement percentages

- **Optimization Presets**
  - `get_optimization_preset()` for common search configurations:
    - `"quick"`: 1-3 grams, 2 methods (~6 configs)
    - `"standard"`: 1-4 grams, 2 methods (~8 configs)
    - `"thorough"`: 1-5 grams, 3 methods + parameter tuning (~15+ configs)
    - `"asr"`: 2-3 grams optimized for ASR (~4 configs)
    - `"minimal"`: Single best configuration (3-gram Kneser-Ney)

- **Result Export & Analysis**
  - Export complete results to JSON with all metadata
  - `print_optimization_results()` for loading and re-displaying results
  - Structured output includes search space, timestamps, all metrics
  - Helper functions: `_evaluate_config()`, `_evaluate_config_cv()`

- **Documentation & Testing**
  - Complete user guide: `docs/hyperparameter_optimization.md`
  - 6 working examples in `examples/hyperparameter_optimization_example.py`
  - 13 comprehensive tests with 100% pass rate
  - Integration test using Alice corpus demonstrating real-world usage

### Changed
- `_evaluate_config()` now returns tuple of (metrics, model) to capture optimized parameters
- Default search space optimized for practical use cases (1-3 grams vs 1-4)
- Removed emoji from comparison output (replaced with ASCII markers)

### Dependencies
- Added optional `[viz]` extra for matplotlib visualization
- Core functionality remains zero-dependency (pure stdlib)

## [0.2.0] - 2025-11-19

Major release adding comprehensive n-gram optimization toolkit for first-pass ASR decoding.

### Added

**Phase 1: Core Evaluation Framework**
- Multi-order training with flexible syntax (`--orders 1-4`, `1,3,5`, `1-3,5,7-10`)
- `compute_multiple_orders()` method for efficient training
- Perplexity evaluation with `perplexity()` method
- `--eval` and `--eval-only` CLI flags for model evaluation
- OOV handling strategies (unk, skip, error)
- Model statistics with `get_statistics()` method
- Backoff rate analysis with `backoff_rate()` method
- `--stats` and `--backoff` CLI flags
- Uniform language model with `create_uniform()` classmethod
- `--uniform` CLI flag for maximum entropy baselines

**Phase 2: Usability & Integration**
- `ModelComparison` class for high-level optimization workflows
- Five preset configurations: first-pass, rescoring, balanced, fast, accurate
- `from_preset()` classmethod for easy model creation
- `--preset` and `--list-presets` CLI flags
- Smoothing method comparison with `compare_smoothing_methods()`
- `--compare-smoothing` CLI flag for automatic method selection

**Phase 3: Advanced Features**
- `InterpolatedModel` class for probability mixing
- `tune_interpolation_weights()` for EM-based weight optimization
- Vocabulary pruning with `prune_vocabulary()` method
- `--prune-vocab` CLI flag (frequency and top-k methods)
- K-fold cross-validation with `cross_validate()` function
- Statistical analysis with mean and standard deviation

**Testing & Documentation**
- 205 new tests added (301 total)
- 68% code coverage (up from 15%)
- 21 documentation files (~7,700 lines)
- 3 comprehensive user guides
- Working example scripts
- 10 detailed feature summaries

### Changed
- Improved code quality: eliminated local imports, magic numbers, code duplication
- Enhanced `ArpaBoLM` with 9 new methods
- Extended CLI with 10+ new flags
- Refactored probability lookup for better maintainability

### Fixed
- Python 3.9 compatibility with Union type hints
- All pre-commit hook issues
- Code formatting consistency

### Documentation
- Updated README with new features
- Added docs/multi_order_training.md
- Added docs/perplexity_evaluation.md
- Added docs/model_comparison.md
- Added examples/model_comparison_example.py

## [0.1.1] - 2025-11-18

### Fixed
- Python 3.9 compatibility (use Optional instead of | syntax)
- Exception chaining in convert.py
- Context managers in cli_normalize.py

## [0.1.0] - 2025-11-18

First public release of arpabo.

### Added
- ARPA format language model builder
- Multiple smoothing methods: Good-Turing, Kneser-Ney, Katz backoff
- Support for arbitrary n-gram orders
- Binary format conversion (PocketSphinx, Kaldi)
- Corpus normalization tool (`arpalm-normalize`)
- Interactive debug mode
- Built-in example corpus (Alice in Wonderland)
- Zero runtime dependencies (pure Python stdlib)
- Comprehensive test suite
- GitHub Actions CI/CD
