"""ArpaLM - Build ARPA format statistical language models

This package provides tools for building statistical language models in ARPA format.

Library Usage:
    from arpalm import ArpaBoLM

    lm = ArpaBoLM(max_order=3, smoothing_method="good_turing")
    with open("corpus.txt") as f:
        lm.read_corpus(f)
    lm.compute()
    lm.write_file("model.arpa")

Command Line Usage:
    arpalm corpus.txt -o model.arpa
    arpalm corpus.txt -o model.arpa -m 4 -s kneser_ney
    arpalm --demo -o model.arpa  # Use example corpus
"""

from arpabo.convert import (
    ConversionError,
    check_conversion_tools,
    to_kaldi_fst,
    to_pocketsphinx_binary,
)
from arpabo.data import get_example_corpus, list_example_corpora
from arpabo.lm import ArpaBoLM
from arpabo.normalize import (
    clean_text,
    normalize_case,
    normalize_line,
    normalize_token,
    normalize_unicode,
)
from arpabo.smoothing import (
    GoodTuringSmoother,
    KatzBackoffSmoother,
    KneserNeySmoother,
    SmoothingMethod,
    create_smoother,
)

__version__ = "0.1.1"
__all__ = [
    "ArpaBoLM",
    "ConversionError",
    "GoodTuringSmoother",
    "KatzBackoffSmoother",
    "KneserNeySmoother",
    "SmoothingMethod",
    "check_conversion_tools",
    "clean_text",
    "create_smoother",
    "get_example_corpus",
    "list_example_corpora",
    "normalize_case",
    "normalize_line",
    "normalize_token",
    "normalize_unicode",
    "to_kaldi_fst",
    "to_pocketsphinx_binary",
]
