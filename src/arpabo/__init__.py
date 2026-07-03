"""arpabo - Build ARPA format statistical language models

This package provides tools for building statistical language models in ARPA format.

Library Usage:
    from arpabo import ArpaBoLM

    lm = ArpaBoLM(max_order=3, smoothing_method="good_turing")
    with open("corpus.txt") as f:
        lm.read_corpus(f)
    lm.compute()
    lm.write_file("model.arpa")

Command Line Usage:
    arpabo corpus.txt -o model.arpa
    arpabo corpus.txt -o model.arpa -m 4 -s kneser_ney
    arpabo --demo -o model.arpa  # Use example corpus
"""

from arpabo.comparison import (
    ModelComparison,
    compare_smoothing_methods,
    optimize_hyperparameters,
    plot_optimization_results,
    print_optimization_results,
    print_smoothing_comparison,
)
from arpabo.convert import (
    ConversionError,
    check_conversion_tools,
    from_pocketsphinx_binary,
    to_kaldi_fst,
    to_pocketsphinx_binary,
    to_pocketsphinx_binary_native,
)
from arpabo.crossval import cross_validate, print_cv_results
from arpabo.data import get_example_corpus, list_example_corpora
from arpabo.interpolation import InterpolatedModel, tune_interpolation_weights
from arpabo.kenlm_bin import read_kenlm_bin
from arpabo.lm import ArpaBoLM
from arpabo.normalize import (
    clean_text,
    normalize_case,
    normalize_line,
    normalize_token,
    normalize_unicode,
)
from arpabo.presets import get_preset, list_presets, print_presets
from arpabo.smoothing import (
    GoodTuringSmoother,
    KatzBackoffSmoother,
    KneserNeySmoother,
    SmoothingMethod,
    create_smoother,
)
from arpabo.utils import parse_order_spec, parse_range_spec

__version__ = "0.4.0"
__all__ = [
    "ArpaBoLM",
    "ConversionError",
    "GoodTuringSmoother",
    "InterpolatedModel",
    "KatzBackoffSmoother",
    "KneserNeySmoother",
    "ModelComparison",
    "SmoothingMethod",
    "check_conversion_tools",
    "clean_text",
    "compare_smoothing_methods",
    "create_smoother",
    "cross_validate",
    "from_pocketsphinx_binary",
    "get_example_corpus",
    "get_preset",
    "list_example_corpora",
    "list_presets",
    "normalize_case",
    "normalize_line",
    "normalize_token",
    "normalize_unicode",
    "optimize_hyperparameters",
    "parse_order_spec",
    "parse_range_spec",
    "plot_optimization_results",
    "print_cv_results",
    "print_optimization_results",
    "print_presets",
    "print_smoothing_comparison",
    "read_kenlm_bin",
    "to_kaldi_fst",
    "to_pocketsphinx_binary",
    "to_pocketsphinx_binary_native",
    "tune_interpolation_weights",
]
