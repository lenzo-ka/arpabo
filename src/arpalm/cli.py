#!/usr/bin/env python

"""Command-line interface for ArpaLM"""

import argparse
import sys

from arpalm.data import get_example_corpus
from arpalm.lm import ArpaBoLM


def _create_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Create an ARPA language model with smoothing (Katz backoff, Good-Turing, Kneser-Ney)",
        epilog="Convert to binary: pocketsphinx_lm_convert -i model.arpa -o model.lm.bin",
    )
    parser.add_argument("files", nargs="*", help="input text files")
    parser.add_argument("--demo", action="store_true", help="use example corpus (Alice in Wonderland)")
    parser.add_argument("-t", "--text", type=str, action="append", help="input text string (repeatable)")
    parser.add_argument("-w", "--word-file", type=str, help="vocabulary file (count set by -C)")
    parser.add_argument("-C", "--word-file-count", type=int, default=1, help="count for word file entries (default: 1)")
    parser.add_argument("-d", "--discount-mass", type=float, help="discount mass [0.0, 1.0) for fixed method")
    parser.add_argument(
        "--discount-step", type=float, default=0.05, help="step size for auto optimization (default: 0.05)"
    )
    parser.add_argument("-c", "--case", type=str, help="case fold: lower or upper")
    parser.add_argument("--no-add-start", dest="add_start", action="store_false", help="skip sentence markers")
    parser.add_argument(
        "--no-unicode-norm", dest="unicode_norm", action="store_false", help="disable unicode normalization"
    )
    parser.add_argument("-n", "--token-norm", dest="token_norm", action="store_true", help="normalize tokens")
    parser.add_argument("-o", "--output", type=str, help="output file (default: stdout)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output to stderr")
    parser.add_argument("-m", "--max-order", type=int, default=3, help="n-gram order (default: 3)")
    parser.add_argument(
        "-s",
        "--smoothing-method",
        type=str,
        default="good_turing",
        choices=["auto", "fixed", "mle", "good_turing", "kneser_ney"],
        help="smoothing method (default: good_turing)",
    )
    parser.add_argument("--debug", action="store_true", help="interactive debug mode")
    parser.add_argument("-l", "--load-model", type=str, help="load and optionally update existing model")
    parser.add_argument("-S", "--stats", action="store_true", help="show statistics and exit")
    parser.add_argument("--to-bin", action="store_true", help="convert to PocketSphinx binary (.lm.bin)")
    parser.add_argument("--to-fst", action="store_true", help="convert to Kaldi FST (.fst)")
    return parser


def _load_or_create_model(args):
    """Load existing model or create new one based on arguments.

    Returns:
        Configured ArpaBoLM instance
    """
    if args.load_model:
        lm = ArpaBoLM.from_arpa_file(args.load_model, verbose=args.verbose)

        if args.files or args.text:
            if args.verbose:
                print("Updating loaded model with new training data...", file=sys.stderr)

            # Update model parameters
            if args.case:
                lm.case = args.case
            if args.add_start is not None:
                lm.add_start = args.add_start
            if args.unicode_norm is not None:
                lm.unicode_norm = args.unicode_norm
            if args.token_norm is not None:
                lm.token_norm = args.token_norm
            if args.discount_step:
                lm.discount_step = args.discount_step

            lm._read_input_data(args.files, args.text, args.word_file)
            lm.compute()
    else:
        lm = ArpaBoLM(
            word_file=args.word_file,
            word_file_count=args.word_file_count,
            discount_mass=args.discount_mass,
            discount_step=args.discount_step,
            case=args.case,
            add_start=args.add_start,
            unicode_norm=args.unicode_norm,
            token_norm=args.token_norm,
            verbose=args.verbose,
            max_order=args.max_order,
            smoothing_method=args.smoothing_method,
        )

        lm._read_input_data(args.files, args.text, args.word_file)
        lm.compute()

    return lm


def _handle_conversions(args):
    """Handle binary format conversions if requested."""
    if not (args.to_bin or args.to_fst):
        return

    from arpalm.convert import ConversionError, to_kaldi_fst, to_pocketsphinx_binary

    if args.to_bin:
        try:
            bin_path = to_pocketsphinx_binary(args.output, verbose=args.verbose)
            if args.verbose:
                print(f"Created binary: {bin_path}", file=sys.stderr)
        except (FileNotFoundError, ConversionError) as e:
            print(f"Binary conversion failed: {e}", file=sys.stderr)
            print("Install PocketSphinx to enable binary conversion", file=sys.stderr)

    if args.to_fst:
        try:
            fst_path = to_kaldi_fst(args.output, verbose=args.verbose)
            if args.verbose:
                print(f"Created FST: {fst_path}", file=sys.stderr)
        except (FileNotFoundError, ConversionError) as e:
            print(f"FST conversion failed: {e}", file=sys.stderr)
            print("Install Kaldi to enable FST conversion", file=sys.stderr)


def main() -> None:
    """Main entry point for the arpalm command-line tool"""
    parser = _create_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.case and args.case not in ["lower", "upper"]:
        parser.error("--case must be lower or upper (if given)")

    if args.demo:
        example_file = get_example_corpus()
        if args.verbose:
            print(f"Using example corpus: {example_file}", file=sys.stderr)
        args.files = [example_file] + (args.files or [])

    if not args.load_model and not args.files and args.text is None and not args.demo:
        parser.error("Input must be specified with input files, --text, or --demo")

    # Build or load model
    lm = _load_or_create_model(args)

    # Handle different output modes
    if args.stats:
        lm.print_stats()
        return

    if args.debug:
        lm.interactive_debug()
        return

    # Write output
    if args.output:
        with open(args.output, "w") as outfile:
            lm.write(outfile)
        _handle_conversions(args)
    else:
        lm.write(sys.stdout)


if __name__ == "__main__":
    main()
