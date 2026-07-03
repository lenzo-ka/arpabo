"""ARPA format I/O for language models"""

import sys
from collections import defaultdict
from datetime import datetime
from math import exp, log
from typing import Any, TextIO

from arpabo.backoff import NEG_LOG10, START


def write_arpa_file(lm, out_path: str) -> bool:
    """Write language model to ARPA format file.

    Args:
        lm: ArpaBoLM instance
        out_path: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(out_path, "w") as outfile:
            write_arpa(lm, outfile)
    except OSError:
        return False
    return True


def write_arpa(lm, outfile: TextIO) -> None:
    """Write language model to ARPA format.

    Args:
        lm: ArpaBoLM instance
        outfile: Output file handle
    """
    if lm.verbose:
        print("Writing output file", file=lm.logfile)

    # Write corpus statistics as comment
    corpus_stats = [
        f"{lm.sent_count} sentences;",
        f"{lm.sum_1} words,",
    ]

    for order in range(lm.max_order):
        if lm.counts[order] > 0:
            corpus_stats.append(f"{lm.counts[order]} {order + 1}-grams,")

    corpus_stats.extend(
        [
            f"with fixed discount mass {lm.discount_mass}",
            "with unicode normalization" if lm.unicode_norm else "",
            "with token normalization" if lm.token_norm else "",
        ]
    )

    print("Corpus:", " ".join(corpus_stats), file=outfile)

    # Write data section
    print(file=outfile)
    print("\\data\\", file=outfile)

    for order in range(lm.max_order):
        if lm.counts[order] > 0:
            print(f"ngram {order + 1}={lm.counts[order]}", file=outfile)
    print(file=outfile)

    # Write n-grams
    for order in range(lm.max_order):
        if lm.counts[order] > 0:
            print(f"\\{order + 1}-grams:", file=outfile)
            _write_order_ngrams(lm, outfile, order)
            print(file=outfile)

    print(file=outfile)
    print("\\end\\", file=outfile)

    if lm.verbose:
        end_time = datetime.now()
        elapsed = end_time - lm.start_time
        print(
            f"Finished {end_time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {elapsed.total_seconds():.2f}s)",
            file=lm.logfile,
        )


def _write_order_ngrams(lm, outfile: TextIO, order: int) -> None:
    """Write n-grams of given order to output file."""

    def write_ngrams(gram_dict: Any, parent_words: list[str], current_order: int) -> None:
        if current_order == 0:
            for word, prob in sorted(gram_dict.items()):
                ngram_words = parent_words + [word]

                # <s> is never a predictable event: emit the -99 sentinel.
                is_start_unigram = order == 0 and word == START
                log_prob = NEG_LOG10 if (is_start_unigram or prob <= 0) else log(prob) / lm.LOG10

                words_str = " ".join(ngram_words)
                if order < lm.max_order - 1:
                    alpha = _lookup_nested(lm.alphas[order], ngram_words)
                    if alpha is None:
                        log_alpha = 0.0  # absent context -> implicit backoff weight 1.0
                    elif alpha > 0:
                        log_alpha = log(alpha) / lm.LOG10
                    else:
                        log_alpha = NEG_LOG10  # saturated context: no backoff mass
                    print(f"{log_prob:6.4f} {words_str} {log_alpha:6.4f}", file=outfile)
                else:
                    print(f"{log_prob:6.4f} {words_str}", file=outfile)
        else:
            for word, value in sorted(gram_dict.items()):
                write_ngrams(value, parent_words + [word], current_order - 1)

    write_ngrams(lm.probs[order], [], order)


def get_ngram_prob(prob_dict: Any, ngram_words: list[str]) -> float:
    """Get probability for n-gram from nested structure."""
    current = prob_dict
    for word in ngram_words:
        if word not in current:
            return 0.0
        current = current[word]
    return current


def _lookup_nested(root: Any, words: list[str]):
    """Return the value stored at ``words``, or None if the path is absent.

    Distinguishes a missing key (None) from a stored 0.0, which get_ngram_prob
    cannot -- the writer needs that difference to tell an absent backoff weight
    (implicit 1.0) from a saturated context (stored 0.0)."""
    current = root
    for word in words:
        if not isinstance(current, dict) or word not in current:
            return None
        current = current[word]
    return current if not isinstance(current, dict) else None


def load_arpa_file(arpa_file: str, lm_class, verbose: bool = False):
    """Load an existing ARPA format language model from a file.

    Args:
        arpa_file: Path to ARPA file
        lm_class: ArpaBoLM class (to create instance)
        verbose: Verbose output

    Returns:
        Loaded ArpaBoLM instance
    """
    if verbose:
        print(f"Loading ARPA model from: {arpa_file}", file=sys.stderr)

    lm = lm_class(verbose=verbose)

    # Initialize data structures
    lm.max_order = 0
    lm.grams = []
    lm.probs = []
    lm.alphas = []
    lm.counts = []
    lm.sum_1 = 0

    with open(arpa_file) as f:
        lines = f.readlines()

    # Parse ARPA format
    current_order = -1
    reading_ngrams = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line == "\\data\\":
            reading_ngrams = False
            continue
        elif line.startswith("ngram "):
            parts = line.split("=")
            if len(parts) == 2:
                order = int(parts[0].split()[1]) - 1
                count = int(parts[1])

                while len(lm.counts) <= order:
                    lm.counts.append(0)
                    if order == 0:
                        lm.grams.append(defaultdict(int))
                        lm.probs.append({})
                        lm.alphas.append({})
                    else:
                        lm.grams.append(lm._make_nested_defaultdict(order))
                        lm.probs.append(lm._make_nested_defaultdict(order))
                        lm.alphas.append(lm._make_nested_defaultdict(order))

                lm.counts[order] = count
                lm.max_order = max(lm.max_order, order + 1)

        elif line.startswith("\\") and line.endswith("-grams:"):
            order = int(line.split("-")[0][1:]) - 1
            current_order = order
            reading_ngrams = True
            continue

        elif reading_ngrams and current_order >= 0:
            _parse_arpa_ngram_line(lm, line, current_order)

    if verbose:
        print(f"Loaded model with max order {lm.max_order}", file=sys.stderr)
        print(f"Vocabulary size: {len(lm.probs[0])}", file=sys.stderr)
        print(f"N-gram counts: {lm.counts}", file=sys.stderr)

    return lm


def _parse_arpa_ngram_line(lm, line: str, current_order: int) -> None:
    """Parse a single n-gram line from ARPA format and store it.

    Parses positionally by the declared section order: an (order+1)-gram line is
    ``logprob w1 .. w_{order+1} [backoff]``. This is robust to vocabulary tokens
    that look like numbers (e.g. years, house numbers), which a "sniff the last
    field as a float" parser silently corrupts. Backoff weights are converted to
    the linear domain and stored keyed by the full n-gram (the context they apply
    to), matching how the writer reads them back.
    """
    parts = line.split()
    n = current_order + 1
    if len(parts) < 1 + n:
        return

    log_prob = float(parts[0])
    prob = exp(log_prob * lm.LOG10)
    words = parts[1 : 1 + n]

    alpha = None
    if len(parts) > 1 + n and current_order < lm.max_order - 1:
        alpha = exp(float(parts[1 + n]) * lm.LOG10)

    if n == 1:
        word = words[0]
        lm.probs[0][word] = prob
        lm.grams[0][word] = int(prob * 1000)
        lm.sum_1 += int(prob * 1000)
        if alpha is not None:
            lm.alphas[0][word] = alpha
    else:
        lm._set_ngram_prob(lm.probs[current_order], words, prob)
        lm._set_ngram_prob(lm.grams[current_order], words, int(prob * 1000))
        if alpha is not None:
            lm._set_ngram_prob(lm.alphas[current_order], words, alpha)
