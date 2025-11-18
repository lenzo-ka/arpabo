# ArpaLM Test Suite

## Running Tests

### All tests

```bash
make test
```

Or with pytest directly:

```bash
pytest
```

### Fast tests only (skip slow tests)

```bash
make test-fast
```

### Verbose output

```bash
make test-verbose
```

### Specific test file

```bash
pytest tests/test_smoothing.py
pytest tests/test_normalize.py -v
```

### Specific test

```bash
pytest tests/test_smoothing.py::TestGoodTuringSmoother::test_basic_functionality
```

## Test Coverage

Current coverage: **68%**

View HTML coverage report:

```bash
open htmlcov/index.html
```

## Test Structure

```
tests/
├── test_smoothing.py      # Smoothing methods (32 tests)
├── test_normalize.py      # Text normalization (18 tests)
├── test_lm.py            # Core LM functionality (18 tests)
├── test_arpa_io.py       # ARPA format I/O (7 tests)
├── test_data.py          # Example data (5 tests)
├── test_cli.py           # CLI interface (6 tests)
└── test_integration.py   # End-to-end workflows (6 tests)

Total: 78 tests
```

## Test Categories

### Unit Tests

- **test_smoothing.py**: Factory, Good-Turing, Kneser-Ney, Katz backoff
- **test_normalize.py**: Unicode, token, case, line normalization
- **test_lm.py**: N-gram counting, corpus reading, compute
- **test_arpa_io.py**: Read/write ARPA format
- **test_data.py**: Example corpus management

### Integration Tests

- **test_integration.py**: End-to-end workflows, Alice corpus
- **test_cli.py**: Command-line interface

### Compatibility Tests

- **test_pocketsphinx_compat.py**: PocketSphinx binary conversion

## What's Tested

### Smoothing Methods PASS
- Good-Turing smoothing
- Kneser-Ney smoothing
- Katz backoff (fixed/auto/MLE)
- Smoother factory
- Different n-gram orders

### Text Normalization PASS
- Unicode NFC normalization
- Token normalization (punctuation stripping)
- Case normalization (upper/lower)
- Line processing with markers
- Sphinx format detection

### Core LM PASS
- N-gram counting (unigram, bigram, trigram)
- Corpus reading (plain text, with markers)
- Multiple inputs
- Word file reading
- Probability computation

### ARPA Format PASS
- Writing ARPA files
- Reading ARPA files
- Round-trip (write → read)
- Format compliance
- Load and update models

### CLI PASS
- Help text
- Demo mode
- Text input
- All smoothing methods
- Different n-gram orders
- Error handling

### Integration PASS
- End-to-end workflows
- Alice corpus processing
- All smoothing methods on real data
- PocketSphinx compatibility

## Coverage Gaps

Areas with lower coverage (can improve later):
- CLI (0%) - subprocess tests don't count as coverage
- Debug tools (4%) - interactive features
- Some ARPA parsing edge cases

These are acceptable for initial release.

## Continuous Testing

Tests run automatically with pre-commit hooks before each commit.

To run tests manually before committing:

```bash
make test
```
