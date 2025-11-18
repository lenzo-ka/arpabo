#!/usr/bin/env python

import sys
from collections import defaultdict
from datetime import datetime
from io import StringIO
from math import log
from typing import Any, Optional, TextIO

from arpalm.arpa_io import load_arpa_file, write_arpa, write_arpa_file
from arpalm.debug import debug_sentence, interactive_debug, print_stats
from arpalm.normalize import normalize_line, normalize_token, normalize_unicode
from arpalm.smoothing import create_smoother

# Author: Kevin Lenzo


class ArpaBoLM:
    """
    Builds statistical language models that can predict the next word in a sequence.

    A language model learns patterns from text to predict what word comes next.
    For example, after seeing "the quick brown", it might predict "fox" is likely.

    This class supports multiple smoothing methods (Good-Turing, Katz backoff, Kneser-Ney)
    to handle unseen word combinations. Methods like Katz backoff use "backing off" to
    shorter sequences when an n-gram hasn't been seen before.

    Key features:
    - Supports arbitrary n-gram orders (default: trigrams for better context)
    - Multiple smoothing methods (Good-Turing, Katz backoff, Kneser-Ney)
    - Good-Turing smoothing by default (best for typical sparse data)
    - Outputs standard ARPA format language models

    Smoothing methods:
    - "good_turing": Good-Turing smoothing (default, best for sparse data)
    - "auto": Optimizes Katz backoff discount mass automatically
    - "fixed": Uses fixed discount mass for Katz backoff (use -d 0.0 for MLE)
    - "kneser_ney": Kneser-Ney smoothing with continuation counts

    Example:
        lm = ArpaBoLM()
        lm.read_corpus(open("corpus.txt"))
        lm.compute()
        lm.write_file("model.arpa")

    Normalization:
        - unicode_norm (default: True): Applies NFC normalization to entire lines
        - token_norm (default: False): Strips punctuation/symbols from tokens

    Binary Format:
        This tool outputs ARPA text format. To convert to binary format for
        use with PocketSphinx, use the pocketsphinx_lm_convert tool:

        $ pocketsphinx_lm_convert -i model.arpa -o model.lm.bin
    """

    # Mathematical constants
    LOG10 = log(10.0)  # Used for converting probabilities to log base 10

    @staticmethod
    def _make_nested_defaultdict(depth: int) -> Any:
        """Create a nested defaultdict with the specified depth."""
        if depth == 0:
            return defaultdict(int)
        return defaultdict(lambda: ArpaBoLM._make_nested_defaultdict(depth - 1))

    def __init__(
        self,
        add_start: bool = True,
        word_file: Optional[str] = None,
        word_file_count: int = 1,
        discount_mass: Optional[float] = None,
        discount_step: float = 0.05,
        case: Optional[str] = None,
        unicode_norm: bool = True,
        token_norm: bool = False,
        verbose: bool = False,
        max_order: int = 3,
        smoothing_method: str = "good_turing",
    ):
        self.add_start = add_start
        self.word_file = word_file
        self.word_file_count = word_file_count
        self.discount_mass = discount_mass
        self.case = case
        self.unicode_norm = unicode_norm
        self.token_norm = token_norm
        self.verbose = verbose
        self.max_order = max_order
        self.smoothing_method = smoothing_method

        if not 0.0 < discount_step < 1.0:
            raise AttributeError(f"Discount step {discount_step} out of range (0.0, 1.0)")
        self.discount_step = discount_step

        self.logfile = sys.stderr
        self.start_time = datetime.now()

        if self.verbose:
            print("Started", self.start_time.strftime("%Y-%m-%d %H:%M:%S"), file=self.logfile)

        # Create smoother instance
        self.smoother = create_smoother(
            method=smoothing_method,
            max_order=max_order,
            verbose=verbose,
            discount_mass=discount_mass,
            discount_step=discount_step,
        )

        # Extract discount_mass back (needed for ARPA output comments)
        if hasattr(self.smoother, "discount_mass"):
            self.discount_mass = self.smoother.discount_mass

        self.sent_count = 0
        self.grams: list[Any] = []
        self.counts: list[int] = []
        self.probs: list[Any] = []
        self.alphas: list[Any] = []

        for order in range(self.max_order):
            if order == 0:
                self.grams.append(defaultdict(int))
                self.probs.append({})
                self.alphas.append({})
            else:
                self.grams.append(self._make_nested_defaultdict(order))
                self.probs.append(self._make_nested_defaultdict(order))
                self.alphas.append(self._make_nested_defaultdict(order))
            self.counts.append(0)

        self.sum_1: int = 0

    # ========================
    # Input/Corpus Reading
    # ========================

    def _read_input_data(
        self, files: list[str], text_strings: Optional[list[str]] = None, word_file: Optional[str] = None
    ) -> None:
        """Helper to read input files, text strings, and word file"""
        for filepath in files:
            with open(filepath) as f:
                self.read_corpus(f)

        if text_strings:
            for text in text_strings:
                self.read_corpus(StringIO(text))

        if word_file:
            self.read_word_file(word_file)

    def read_word_file(self, path: str, count: Optional[int] = None) -> None:
        """
        Read in a file of words to add to the model,
        if not present, with the given count (default 1)
        """
        if self.verbose:
            print("Reading word file:", path, file=self.logfile)

        if count is None:
            count = self.word_file_count

        new_word_count = token_count = 0
        with open(path) as words_file:
            for token in words_file:
                token = token.strip()
                if not token:
                    continue
                if self.unicode_norm:
                    token = normalize_unicode(token)
                if self.case == "lower":
                    token = token.lower()
                elif self.case == "upper":
                    token = token.upper()
                if self.token_norm:
                    token = normalize_token(token)
                token_count += 1
                if token not in self.grams[0]:
                    self.grams[0][token] = count
                    new_word_count += 1

        if self.verbose:
            print(
                f"{new_word_count} new unique words",
                f"from {token_count} tokens,",
                f"each with count {count}",
                file=self.logfile,
            )

    def _process_corpus_line(self, line: str, add_markers: bool) -> Optional[list[str]]:
        """Process a single line from corpus: normalize, clean, add markers.

        Returns:
            List of words, or None if line should be skipped
        """
        return normalize_line(
            line, unicode_norm=self.unicode_norm, case=self.case, token_norm=self.token_norm, add_markers=add_markers
        )

    def read_corpus(self, infile):
        """
        Read in a text training corpus from a file handle.
        Handles three formats automatically per line:
        - Sphinx sentfile: <s> .* </s> (filename)
        - Text with markers: <s> .* </s>
        - Plain text without markers
        """
        if self.verbose:
            print("Reading corpus file, breaking per newline.", file=self.logfile)

        sent_count = 0
        for line in infile:
            words = self._process_corpus_line(line, self.add_start)
            if words is None:
                continue

            sent_count += 1
            wc = len(words)

            for j in range(wc):
                for order in range(min(self.max_order, wc - j)):
                    if order == 0:
                        self.grams[0][words[j]] += 1
                    else:
                        ngram_words = words[j : j + order + 1]
                        self._increment_ngram(self.grams[order], ngram_words)

        self.sent_count += sent_count
        if self.verbose:
            print(f"{sent_count} sentences", file=self.logfile)

    def _traverse_ngrams(self, gram_dict: Any, order: int, callback: Any) -> None:
        """Generic helper to traverse nested n-gram structure and apply a callback.

        Args:
            gram_dict: The nested defaultdict to traverse
            order: Current order (depth) to traverse
            callback: Function called with (gram_dict, parent_words, current_order)
                     at each level
        """

        def traverse(d: Any, parent_words: list[str], current_order: int) -> None:
            if current_order == 0:
                callback(d, parent_words, 0)
            else:
                for word, value in sorted(d.items()) if isinstance(d, dict) else d.items():
                    traverse(value, parent_words + [word], current_order - 1)

        traverse(gram_dict, [], order)

    def _increment_ngram(self, gram_dict: Any, ngram_words: list[str]) -> None:
        """Helper method to increment n-gram counts in nested defaultdict structure"""
        current = gram_dict
        for word in ngram_words[:-1]:
            current = current[word]
        current[ngram_words[-1]] += 1

    def compute(self) -> None:
        """
        Compute language model probabilities using the configured smoothing method.

        This method builds a statistical language model that can predict the next word
        in a sequence. It delegates probability computation to the smoothing method.

        How it works:
        1. Count n-grams of all orders (unigrams, bigrams, trigrams, etc.)
        2. Delegate to smoothing method to compute probabilities
        3. Smoothing handles unseen n-grams and backoff weights

        The smoothing method handles:
        - Probability estimation for all n-gram orders
        - Backoff weights (alphas) if needed
        - Optimization (for auto method)
        """
        if not self.grams[0]:
            sys.exit("No input?")

        self.sum_1 = sum(self.grams[0].values())

        for order in range(self.max_order):
            self.counts[order] = self._count_ngrams(self.grams[order], order)

        # Delegate to smoother
        self.smoother.compute_probabilities(grams=self.grams, sum_1=self.sum_1, probs=self.probs, alphas=self.alphas)

        # Update discount_mass if smoother optimized it
        if hasattr(self.smoother, "discount_mass"):
            self.discount_mass = self.smoother.discount_mass

    def _count_ngrams(self, gram_dict: Any, order: int) -> int:
        """Count how many different n-grams we have of a given order"""
        if order == 0:
            return len(gram_dict)

        count = 0
        for _key, value in gram_dict.items():
            if isinstance(value, dict):
                count += self._count_ngrams(value, order - 1)
            else:
                count += 1
        return count

    def _get_ngram_count(self, gram_dict: Any, ngram_words: list[str]) -> int:
        """Get the count of a specific n-gram from our nested dictionary structure"""
        current = gram_dict
        for word in ngram_words[:-1]:
            if word not in current:
                return 0
            current = current[word]
        return current.get(ngram_words[-1], 0)

    def _set_ngram_prob(self, prob_dict: Any, ngram_words: list[str], prob: float) -> None:
        """Store a probability value in our nested dictionary structure"""
        current = prob_dict
        for word in ngram_words[:-1]:
            current = current[word]
        current[ngram_words[-1]] = prob

    # ========================
    # ARPA Format I/O
    # ========================

    def write_file(self, out_path: str) -> bool:
        """Write language model to ARPA format file."""
        return write_arpa_file(self, out_path)

    def write(self, outfile: TextIO) -> None:
        """Write language model to ARPA format."""
        write_arpa(self, outfile)

    @classmethod
    def from_arpa_file(cls, arpa_file: str, verbose: bool = False) -> "ArpaBoLM":
        """Load an existing ARPA format language model from a file."""
        return load_arpa_file(arpa_file, cls, verbose)

    # ========================
    # Debug & Interactive Tools
    # ========================

    def debug_sentence(self, sentence: str) -> None:
        """Debug a sentence, showing probabilities at each step."""
        debug_sentence(self, sentence)

    def interactive_debug(self) -> None:
        """Start an interactive debug session."""
        interactive_debug(self)

    def print_stats(self) -> None:
        """Print model statistics to stdout."""
        print_stats(self)
