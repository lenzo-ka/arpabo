# Example Data

This directory contains example corpora for demonstration and testing.

## Files

- **alice.txt** - Alice's Adventures in Wonderland by Lewis Carroll (275 lines)
  - Classic public domain text
  - Good for demonstrating language model building
  - Sufficient size for meaningful n-gram statistics

## Usage

### From Python

```python
from arpalm import ArpaBoLM, get_example_corpus

# Get path to example corpus
corpus_path = get_example_corpus("alice.txt")

# Build a model
lm = ArpaBoLM()
with open(corpus_path) as f:
    lm.read_corpus(f)
lm.compute()
lm.write_file("alice.arpa")
```

### From CLI

```bash
# Use the demo corpus
arpalm --demo -o alice.arpa

# With custom options
arpalm --demo -o alice.arpa -m 4 -s kneser_ney -v
```

## Adding New Example Corpora

To add a new example corpus:

1. Add a `.txt` file to this directory
2. The file will automatically be included in the package
3. Access it via `get_example_corpus("filename.txt")`
