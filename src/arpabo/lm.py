#!/usr/bin/env python

import math
import sys
from collections import defaultdict
from datetime import datetime
from io import StringIO
from math import log
from typing import Any, Optional, TextIO, Union

from arpabo.arpa_io import get_ngram_prob, load_arpa_file, write_arpa, write_arpa_file
from arpabo.debug import debug_sentence, interactive_debug, print_stats
from arpabo.normalize import normalize_line, normalize_token, normalize_unicode
from arpabo.smoothing import create_smoother

# Author: Kevin Lenzo

# Constants
OOV_PROBABILITY = 1e-10  # Very small probability for out-of-vocabulary words
DEFAULT_MAX_PERPLEXITY_INCREASE = 0.05  # 5% tolerance for first-pass recommendations


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
            if self.read_corpus_line(line):
                sent_count += 1

        self.sent_count += sent_count
        if self.verbose:
            print(f"{sent_count} sentences", file=self.logfile)

    def read_corpus_line(self, line: str) -> bool:
        """
        Read a single line from corpus and update n-gram counts.

        Args:
            line: Single line of text to process

        Returns:
            True if line was processed, False if skipped

        Example:
            lm = ArpaBoLM(max_order=3)
            for line in open("corpus.txt"):
                lm.read_corpus_line(line)
            lm.compute()
        """
        words = self._process_corpus_line(line, self.add_start)
        if words is None:
            return False

        wc = len(words)

        for j in range(wc):
            for order in range(min(self.max_order, wc - j)):
                if order == 0:
                    self.grams[0][words[j]] += 1
                else:
                    ngram_words = words[j : j + order + 1]
                    self._increment_ngram(self.grams[order], ngram_words)

        return True

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

    def compute_multiple_orders(self, orders: list[int]) -> dict[int, "ArpaBoLM"]:
        """
        Train multiple n-gram orders from the same corpus.

        This is more efficient than training each order separately because
        the corpus only needs to be read and counted once. The n-gram counts
        are shared across all models.

        Args:
            orders: List of n-gram orders to train (e.g., [1, 2, 3, 4])

        Returns:
            Dictionary mapping order -> trained ArpaBoLM model

        Example:
            lm = ArpaBoLM()
            with open("corpus.txt") as f:
                lm.read_corpus(f)
            models = lm.compute_multiple_orders([1, 2, 3, 4])
            for order, model in models.items():
                model.write_file(f"{order}gram.arpa")
        """
        if not self.grams[0]:
            sys.exit("No input?")

        # Validate orders
        max_order_needed = max(orders)
        if max_order_needed > self.max_order:
            raise ValueError(
                f"Requested order {max_order_needed} exceeds corpus max_order {self.max_order}. "
                f"Create ArpaBoLM with max_order={max_order_needed} or higher."
            )

        models = {}

        for order in sorted(orders):
            # Create new model instance with same configuration
            lm = ArpaBoLM(
                add_start=self.add_start,
                word_file=self.word_file,
                word_file_count=self.word_file_count,
                discount_mass=self.discount_mass,
                discount_step=self.discount_step,
                case=self.case,
                unicode_norm=self.unicode_norm,
                token_norm=self.token_norm,
                verbose=self.verbose,
                max_order=order,
                smoothing_method=self.smoothing_method,
            )

            # Share corpus statistics
            lm.sent_count = self.sent_count

            # Copy grams up to the requested order
            for o in range(order):
                lm.grams[o] = self.grams[o]

            # Compute probabilities for this order
            lm.compute()

            models[order] = lm

        return models

    def perplexity(self, test_corpus: TextIO, oov_handling: str = "unk") -> dict:
        """
        Calculate perplexity on held-out test corpus.

        Perplexity measures how well the model predicts the test data.
        Lower perplexity indicates better prediction. A perplexity of N means
        the model is as uncertain as if it had to choose uniformly from N words.

        Args:
            test_corpus: File handle to test data (one sentence per line)
            oov_handling: How to handle out-of-vocabulary words:
                - "unk": Treat OOV words as <unk> token (default)
                - "skip": Skip OOV words in calculation
                - "error": Raise ValueError on OOV words

        Returns:
            Dictionary with evaluation metrics:
            {
                "perplexity": float,         # exp(-avg_log_prob)
                "cross_entropy": float,      # bits per word
                "num_sentences": int,
                "num_words": int,
                "num_oov": int,
                "oov_rate": float            # fraction of OOV words
            }

        Example:
            with open("test.txt") as f:
                results = lm.perplexity(f)
            print(f"Perplexity: {results['perplexity']:.1f}")
        """
        if not self.probs[0]:
            raise ValueError("Model has not been computed yet. Call compute() first.")

        total_log_prob = 0.0
        total_words = 0
        oov_count = 0
        sentence_count = 0

        for line in test_corpus:
            if not line.strip():
                continue

            sentence_count += 1

            # Normalize the line the same way as training
            words = normalize_line(
                line,
                unicode_norm=self.unicode_norm,
                case=self.case,
                token_norm=self.token_norm,
                add_markers=self.add_start,
            )

            for i, word in enumerate(words):
                # Check if word is in vocabulary
                if word not in self.probs[0]:
                    oov_count += 1

                    if oov_handling == "error":
                        raise ValueError(f"OOV word: '{word}'")
                    elif oov_handling == "skip":
                        continue
                    # else: handle as unk below

                # Get probability using helper
                prob, _ = self._get_ngram_probability(word, words, i)

                # Convert to log10 if not already
                prob_log10 = prob if prob < 0 else math.log10(prob)

                total_log_prob += prob_log10
                total_words += 1

        if total_words == 0:
            raise ValueError("No words found in test corpus")

        # Calculate metrics
        avg_log_prob = total_log_prob / total_words
        perplexity = math.pow(10, -avg_log_prob)

        # Cross-entropy in bits per word
        cross_entropy = -avg_log_prob / math.log10(2)

        return {
            "perplexity": perplexity,
            "cross_entropy": cross_entropy,
            "num_sentences": sentence_count,
            "num_words": total_words,
            "num_oov": oov_count,
            "oov_rate": oov_count / total_words if total_words > 0 else 0.0,
        }

    def print_perplexity_results(self, results: dict, test_file: str = "test data") -> None:
        """
        Print perplexity evaluation results in a formatted way.

        Args:
            results: Dictionary returned from perplexity()
            test_file: Name of test file for display
        """
        print("\nPerplexity Evaluation")
        print("=" * 50)
        print(f"Model: {self.max_order}-gram {self.smoothing_method}")
        print(f"Test file: {test_file}")
        print()
        print("Evaluation Results:")
        print(f"  Sentences:      {results['num_sentences']:>8,}")
        print(f"  Words:          {results['num_words']:>8,}")
        print(f"  OOV words:      {results['num_oov']:>8,} ({results['oov_rate'] * 100:.1f}%)")
        print()
        print(f"  Perplexity:     {results['perplexity']:>8.1f}")
        print(f"  Cross-entropy:  {results['cross_entropy']:>8.2f} bits/word")

    def get_statistics(self) -> dict:
        """
        Get comprehensive model statistics.

        Returns detailed information about the model including vocabulary size,
        n-gram counts, training corpus statistics, and model configuration.

        Returns:
            Dictionary with model statistics:
            {
                "order": int,
                "smoothing": str,
                "vocab_size": int,
                "ngram_counts": {order: count, ...},
                "training_corpus": {
                    "sentences": int,
                    "tokens": int
                }
            }

        Example:
            stats = lm.get_statistics()
            print(f"Vocabulary: {stats['vocab_size']} words")
            print(f"Trigrams: {stats['ngram_counts'][3]}")
        """
        if not self.probs[0]:
            raise ValueError("Model has not been computed yet. Call compute() first.")

        stats = {
            "order": self.max_order,
            "smoothing": self.smoothing_method,
            "vocab_size": len(self.probs[0]),
            "ngram_counts": {},
            "training_corpus": {"sentences": self.sent_count, "tokens": self.sum_1},
        }

        # Count n-grams at each order
        for order in range(self.max_order):
            stats["ngram_counts"][order + 1] = self.counts[order]

        return stats

    def backoff_rate(self, test_corpus: TextIO) -> dict:
        """
        Analyze backoff behavior on test data.

        Higher backoff rate indicates the model falls back to lower-order
        n-grams more often, which can indicate more diversity in first-pass
        decoding. Lower backoff rate means sharper, more confident predictions.

        Args:
            test_corpus: File handle to test data (one sentence per line)

        Returns:
            Dictionary with backoff statistics:
            {
                "overall_backoff_rate": float,     # Fraction that backed off
                "order_usage": {order: fraction},  # Usage of each order
                "total_queries": int               # Total n-gram lookups
            }

        Example:
            with open("test.txt") as f:
                backoff = lm.backoff_rate(f)
            print(f"Backoff rate: {backoff['overall_backoff_rate']*100:.1f}%")
        """
        if not self.probs[0]:
            raise ValueError("Model has not been computed yet. Call compute() first.")

        total_queries = 0
        order_hits = dict.fromkeys(range(1, self.max_order + 1), 0)

        for line in test_corpus:
            if not line.strip():
                continue

            # Normalize the line same as training
            words = normalize_line(
                line,
                unicode_norm=self.unicode_norm,
                case=self.case,
                token_norm=self.token_norm,
                add_markers=self.add_start,
            )

            for i, word in enumerate(words):
                # Skip OOV words
                if word not in self.probs[0]:
                    continue

                # Find which order was used
                order_used = self._get_order_used(word, words, i)
                order_hits[order_used] += 1
                total_queries += 1

        if total_queries == 0:
            return {
                "overall_backoff_rate": 0.0,
                "order_usage": dict.fromkeys(range(1, self.max_order + 1), 0.0),
                "total_queries": 0,
            }

        # Calculate usage fractions
        order_usage = {order: count / total_queries for order, count in order_hits.items()}

        # Backoff rate = fraction that didn't use full order
        full_order_rate = order_hits[self.max_order] / total_queries
        backoff_rate = 1.0 - full_order_rate

        return {"overall_backoff_rate": backoff_rate, "order_usage": order_usage, "total_queries": total_queries}

    def _get_ngram_probability(self, word: str, words: list[str], position: int) -> tuple[float, int]:
        """
        Get probability for word at position using best available n-gram order.

        Args:
            word: The word to look up
            words: Full list of words in sentence
            position: Index of word in sentence

        Returns:
            Tuple of (probability, order_used)
            - probability: Raw probability (not log), or OOV_PROBABILITY if not found
            - order_used: Which n-gram order was actually used (1 to max_order)
        """
        # Try from highest order down
        for order in range(min(position, self.max_order - 1), -1, -1):
            if order == 0:
                # Unigram
                if word in self.probs[0]:
                    return self.probs[0][word], 1
                return OOV_PROBABILITY, 1
            else:
                # Higher-order n-gram
                context_words = words[max(0, position - order) : position]
                ngram_words = context_words + [word]

                # Query probability
                prob_result = get_ngram_prob(self.probs[order], ngram_words)

                if not isinstance(prob_result, dict) and prob_result is not None and prob_result > 0:
                    return prob_result, order + 1

        # Fallback to OOV probability
        return OOV_PROBABILITY, 1

    def _get_order_used(self, word: str, words: list[str], position: int) -> int:
        """
        Determine which n-gram order was used for a word lookup.

        Args:
            word: The word being looked up
            words: Full list of words in sentence
            position: Index of word in sentence

        Returns:
            Order (1 to max_order) that was actually used
        """
        _, order_used = self._get_ngram_probability(word, words, position)
        return order_used

    def print_statistics(self, test_file: Optional[str] = None) -> None:
        """
        Print comprehensive model statistics in formatted output.

        Args:
            test_file: Optional test file for backoff analysis
        """
        stats = self.get_statistics()

        print("\nModel Statistics")
        print("=" * 50)
        print(f"Order:      {stats['order']}")
        print(f"Smoothing:  {stats['smoothing']}")
        print(f"Vocabulary: {stats['vocab_size']:,} words")
        print()
        print("N-gram counts:")
        for order in sorted(stats["ngram_counts"].keys()):
            count = stats["ngram_counts"][order]
            print(f"  {order}-grams: {count:>12,}")
        print()
        print("Training corpus:")
        print(f"  Sentences: {stats['training_corpus']['sentences']:>10,}")
        print(f"  Tokens:    {stats['training_corpus']['tokens']:>10,}")

        # Add backoff analysis if test file provided
        if test_file:
            print()
            print(f"Backoff analysis (on {test_file}):")
            with open(test_file) as f:
                backoff = self.backoff_rate(f)

            print(f"  Overall backoff rate: {backoff['overall_backoff_rate'] * 100:.1f}%")
            print()
            print("  Query resolution:")
            for order in sorted(backoff["order_usage"].keys(), reverse=True):
                usage = backoff["order_usage"][order]
                print(f"    {order}-gram hits: {usage * 100:>5.1f}%")

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

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "ArpaBoLM":
        """
        Create ArpaBoLM instance from preset configuration.

        Presets provide sensible defaults for common use cases like first-pass
        decoding, rescoring, fast decoding, etc.

        Args:
            preset_name: Name of preset ("first-pass", "rescoring", "balanced", "fast", "accurate")
            **overrides: Override any preset parameters (e.g., max_order=4, verbose=True)

        Returns:
            ArpaBoLM instance configured with preset + overrides

        Raises:
            ValueError: If preset_name is unknown

        Examples:
            # Use first-pass preset with default order (2)
            lm = ArpaBoLM.from_preset("first-pass")

            # Override order
            lm = ArpaBoLM.from_preset("first-pass", max_order=3)

            # Override smoothing
            lm = ArpaBoLM.from_preset("rescoring", smoothing_method="good_turing")

            # Then use normally
            lm.read_corpus(open("corpus.txt"))
            lm.compute()
            lm.write_file("model.arpa")

        Available presets:
            - first-pass: Good-Turing bigram, optimized for N-best diversity
            - rescoring: Kneser-Ney 4-gram, optimized for accuracy
            - balanced: Kneser-Ney 3-gram, general-purpose
            - fast: Good-Turing bigram, minimal memory
            - accurate: Kneser-Ney 5-gram, maximum accuracy
        """
        from arpabo.presets import get_preset

        # Get preset config
        config = get_preset(preset_name)

        # Extract relevant parameters for ArpaBoLM
        params = {"smoothing_method": config["smoothing_method"], "max_order": config["recommended_order"]}

        # Apply user overrides
        params.update(overrides)

        return cls(**params)

    @classmethod
    def create_uniform(cls, vocab: Union[list[str], str], add_start: bool = True) -> "ArpaBoLM":
        """
        Create uniform (maximum entropy) language model.

        All words have equal probability, useful as a baseline for comparison
        or for maximum acoustic-driven decoding in first-pass ASR.

        Args:
            vocab: List of words or path to vocab file (one word per line)
            add_start: Whether to include sentence markers <s> and </s>

        Returns:
            ArpaBoLM with uniform unigram distribution

        Example:
            # From word list
            lm = ArpaBoLM.create_uniform(["the", "cat", "sat"])

            # From vocabulary file
            lm = ArpaBoLM.create_uniform("vocab.txt")

            # Write to ARPA format
            lm.write_file("uniform.arpa")

        Note:
            - Creates unigram-only model (max_order=1)
            - All words get log10(1/N) probability
            - No backoff weights needed
            - Perplexity equals vocabulary size
        """
        # Load vocabulary
        if isinstance(vocab, str):
            with open(vocab) as f:
                words = [line.strip() for line in f if line.strip()]
        else:
            words = list(vocab)

        if not words:
            raise ValueError("Vocabulary cannot be empty")

        # Add sentence markers if requested
        if add_start:
            if "<s>" not in words:
                words.insert(0, "<s>")
            if "</s>" not in words:
                words.append("</s>")

        # Create model with uniform "smoothing" (actually no smoothing)
        lm = cls(max_order=1, add_start=add_start, verbose=False)

        # Set up uniform probabilities
        vocab_size = len(words)
        uniform_prob = 1.0 / vocab_size  # Raw probability, not log

        # Populate grams and probs
        for word in words:
            lm.grams[0][word] = 1  # Count of 1 (doesn't matter for uniform)
            lm.probs[0][word] = uniform_prob  # Store as raw probability

        # Set corpus statistics
        lm.sent_count = 0  # No actual corpus
        lm.sum_1 = len(words)  # Vocab size
        lm.counts[0] = len(words)

        # Mark as computed
        lm.smoothing_method = "uniform"

        return lm

    def prune_vocabulary(
        self, method: str = "frequency", threshold: Union[int, float] = 100, keep_markers: bool = True
    ) -> "ArpaBoLM":
        """
        Create new model with pruned vocabulary to reduce size.

        Useful for mobile deployment or memory-constrained environments.
        Removes low-frequency words based on threshold.

        Args:
            method: Pruning method:
                - "frequency": Keep words with count >= threshold
                - "topk": Keep top K most frequent words
            threshold: Minimum count (for frequency) or K value (for topk)
            keep_markers: Always keep <s> and </s> markers (default: True)

        Returns:
            New ArpaBoLM with pruned vocabulary

        Raises:
            ValueError: If model hasn't been trained or invalid method

        Example:
            # Keep words with frequency >= 100
            pruned = lm.prune_vocabulary(method="frequency", threshold=100)

            # Keep top 10,000 words
            pruned = lm.prune_vocabulary(method="topk", threshold=10000)

            # Evaluate impact
            before = lm.get_statistics()
            after = pruned.get_statistics()
            print(f"Vocab: {before['vocab_size']} → {after['vocab_size']}")
        """
        if not self.grams[0]:
            raise ValueError("Model has not been trained. Call read_corpus() and compute() first.")

        # Determine which words to keep
        if method == "frequency":
            # Keep words with count >= threshold
            pruned_vocab = {word for word, count in self.grams[0].items() if count >= threshold}

        elif method == "topk":
            # Keep top K most frequent words
            sorted_words = sorted(self.grams[0].items(), key=lambda x: x[1], reverse=True)
            pruned_vocab = {word for word, _ in sorted_words[: int(threshold)]}

        else:
            raise ValueError(f"Unknown pruning method: '{method}'. Use 'frequency' or 'topk'")

        # Always keep sentence markers if requested
        if keep_markers:
            if "<s>" in self.grams[0]:
                pruned_vocab.add("<s>")
            if "</s>" in self.grams[0]:
                pruned_vocab.add("</s>")

        if not pruned_vocab:
            raise ValueError("Pruning removed all words. Try lower threshold.")

        # Create new model with pruned vocabulary
        pruned_lm = ArpaBoLM(
            add_start=self.add_start,
            word_file=None,
            word_file_count=1,
            discount_mass=self.discount_mass,
            discount_step=self.discount_step,
            case=self.case,
            unicode_norm=self.unicode_norm,
            token_norm=self.token_norm,
            verbose=False,
            max_order=self.max_order,
            smoothing_method=self.smoothing_method,
        )

        # Filter n-grams to only include pruned vocabulary
        # Unigrams
        pruned_lm.grams[0] = {word: count for word, count in self.grams[0].items() if word in pruned_vocab}

        # Higher-order n-grams: only keep if all words in vocabulary
        for order in range(1, self.max_order):
            pruned_lm.grams[order] = self._filter_ngrams(self.grams[order], pruned_vocab, order)

        # Set sentence count
        pruned_lm.sent_count = self.sent_count

        # Recompute probabilities with pruned n-grams
        pruned_lm.compute()

        return pruned_lm

    def _filter_ngrams(self, gram_dict, vocab, order):
        """Recursively filter n-grams to only include words in vocabulary."""
        if order == 0:
            # Base case: filter leaf values
            return {word: count for word, count in gram_dict.items() if word in vocab}

        # Recursive case: filter nested dicts
        filtered = {}
        for word, nested in gram_dict.items():
            if word in vocab:
                filtered_nested = self._filter_ngrams(nested, vocab, order - 1)
                if filtered_nested:  # Only include if has valid children
                    filtered[word] = filtered_nested

        return filtered

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
