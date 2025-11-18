# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
