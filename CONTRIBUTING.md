# Contributing to ArpaLM

Thank you for your interest in contributing to ArpaLM!

## Development Setup

```bash
git clone https://github.com/lenzo-ka/arpabo.git
cd arpabo
make venv
source venv/bin/activate
```

## Running Tests

```bash
# All tests
make test

# Fast tests only
make test-fast

# With coverage
pytest --cov=arpabo

# Specific test
pytest tests/test_smoothing.py -v
```

## Code Quality

Pre-commit hooks are installed automatically with `make venv`:

```bash
# Run manually
make pre-commit

# Lint
make lint

# Format
make format
```

## Project Structure

```
src/arpabo/
├── lm.py              # Core n-gram management
├── smoothing/         # Smoothing algorithms
│   ├── base.py
│   ├── good_turing.py
│   ├── kneser_ney.py
│   └── katz_backoff.py
├── arpa_io.py         # ARPA format I/O
├── normalize.py       # Text normalization
├── convert.py         # Binary conversion
├── debug.py           # Debug tools
├── data.py            # Example corpus
├── cli.py             # Main CLI
└── cli_normalize.py   # Normalization CLI
```

## Adding a New Smoothing Method

1. Create `src/arpabo/smoothing/my_method.py`:

```python
from arpabo.smoothing.base import SmoothingMethod
from arpabo.smoothing.utils import set_ngram_prob

class MyMethodSmoother(SmoothingMethod):
    def needs_backoff_weights(self) -> bool:
        return False

    def compute_probabilities(self, grams, sum_1, probs, alphas):
        # Your implementation
        pass
```

2. Add to `src/arpabo/smoothing/__init__.py`
3. Add to factory in `create_smoother()`
4. Add tests in `tests/test_smoothing.py`

## Coding Standards

- Python 3.9+ required
- Use modern type hints: `list[T]`, `dict[K,V]`, `T | None`
- Follow PEP 8 (enforced by ruff)
- No external dependencies in core library
- Alphabetically sort imports
- Use double quotes for strings

## Testing Requirements

- All tests must pass
- Add tests for new features
- Maintain >60% coverage
- Test with demo corpus
- Verify PocketSphinx compatibility if touching I/O

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Run linter: `make lint`
6. Commit with clear message
7. Push and create PR

## Pull Request Guidelines

- Use the PR template
- Reference related issues
- Include tests
- Update documentation
- Pass all CI checks

## Release Process

For maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions will auto-publish to PyPI

## Questions?

Open an issue or discussion on GitHub.
EOF
cat CONTRIBUTING.md
