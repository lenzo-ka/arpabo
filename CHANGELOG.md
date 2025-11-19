# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
