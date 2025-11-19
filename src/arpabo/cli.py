#!/usr/bin/env python

"""Command-line interface for ArpaLM"""

import argparse
import os
import sys

from arpabo.comparison import compare_smoothing_methods, print_smoothing_comparison
from arpabo.convert import ConversionError, to_kaldi_fst, to_pocketsphinx_binary
from arpabo.data import get_example_corpus
from arpabo.lm import ArpaBoLM
from arpabo.presets import get_preset, print_presets
from arpabo.utils import parse_order_spec


def _create_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Create an ARPA language model with smoothing (Katz backoff, Good-Turing, Kneser-Ney)",
        epilog="Convert to binary: pocketsphinx_lm_convert -i model.arpa -o model.lm.bin",
    )
    parser.add_argument("files", nargs="*", help="input text files")
    parser.add_argument("--demo", action="store_true", help="use example corpus (Alice in Wonderland)")
    parser.add_argument(
        "--uniform",
        type=str,
        metavar="VOCAB_FILE",
        help="create uniform LM from vocabulary file or extract vocab from corpus",
    )
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
        "--orders",
        type=str,
        help="train multiple orders (e.g., '1,2,4' or '1-4' or '1-3,5,7-10'); requires -o to be a directory",
    )
    parser.add_argument(
        "-s",
        "--smoothing-method",
        type=str,
        default="good_turing",
        choices=["auto", "fixed", "mle", "good_turing", "kneser_ney"],
        help="smoothing method (default: good_turing)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["first-pass", "rescoring", "balanced", "fast", "accurate"],
        help="use preset configuration (overrides -m and -s)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="list available presets and exit",
    )
    parser.add_argument("--debug", action="store_true", help="interactive debug mode")
    parser.add_argument("-l", "--load-model", type=str, help="load and optionally update existing model")
    parser.add_argument("-S", "--stats", action="store_true", help="show statistics and exit")
    parser.add_argument("--eval", type=str, metavar="TEST_FILE", help="evaluate model on test file")
    parser.add_argument(
        "--eval-only",
        nargs=2,
        metavar=("MODEL", "TEST_FILE"),
        help="evaluate existing model without training",
    )
    parser.add_argument(
        "--oov-handling",
        type=str,
        default="unk",
        choices=["unk", "skip", "error"],
        help="OOV handling for evaluation: unk (default), skip, or error",
    )
    parser.add_argument(
        "--backoff",
        type=str,
        metavar="TEST_FILE",
        help="show backoff rate analysis on test file (use with --stats or --eval)",
    )
    parser.add_argument(
        "--compare-smoothing",
        type=str,
        nargs="?",
        const="all",
        metavar="METHODS",
        help="compare smoothing methods (comma-separated or 'all'); requires --eval",
    )
    parser.add_argument(
        "--prune-vocab",
        type=str,
        metavar="METHOD:THRESHOLD",
        help="prune vocabulary (e.g., 'frequency:100' or 'topk:10000')",
    )
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


def _handle_conversions(args, output_path=None):
    """Handle binary format conversions if requested."""
    if not (args.to_bin or args.to_fst):
        return

    # Use provided path or fall back to args.output
    target_path = output_path or args.output

    if args.to_bin:
        try:
            bin_path = to_pocketsphinx_binary(target_path, verbose=args.verbose)
            if args.verbose:
                print(f"Created binary: {bin_path}", file=sys.stderr)
        except (FileNotFoundError, ConversionError) as e:
            print(f"Binary conversion failed: {e}", file=sys.stderr)
            print("Install PocketSphinx to enable binary conversion", file=sys.stderr)

    if args.to_fst:
        try:
            fst_path = to_kaldi_fst(target_path, verbose=args.verbose)
            if args.verbose:
                print(f"Created FST: {fst_path}", file=sys.stderr)
        except (FileNotFoundError, ConversionError) as e:
            print(f"FST conversion failed: {e}", file=sys.stderr)
            print("Install Kaldi to enable FST conversion", file=sys.stderr)


def _handle_eval_only(args):
    """Handle --eval-only mode: evaluate existing model."""
    model_file, test_file = args.eval_only

    if args.verbose:
        print(f"Loading model from {model_file}...", file=sys.stderr)

    lm = ArpaBoLM.from_arpa_file(model_file, verbose=args.verbose)

    if args.verbose:
        print(f"Evaluating on {test_file}...", file=sys.stderr)

    with open(test_file) as f:
        results = lm.perplexity(f, oov_handling=args.oov_handling)

    lm.print_perplexity_results(results, test_file=test_file)


def _handle_multi_order_training(args):
    """Handle training multiple n-gram orders."""
    orders = parse_order_spec(args.orders)

    if args.verbose:
        print(f"Training orders: {orders}", file=sys.stderr)

    if not args.output:
        print("Error: --orders requires -o to specify output directory", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Determine max order needed for corpus counting
    max_order_needed = max(orders)

    # Create model with highest order to count all n-grams
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
        max_order=max_order_needed,
        smoothing_method=args.smoothing_method,
    )

    # Read corpus
    lm._read_input_data(args.files, args.text, args.word_file)

    if args.verbose:
        print("Computing models...", file=sys.stderr)

    # Train all orders
    models = lm.compute_multiple_orders(orders)

    # Write each model to file
    for order, model in models.items():
        output_file = os.path.join(args.output, f"{order}gram.arpa")
        if args.verbose:
            print(f"Writing {order}-gram model to {output_file}", file=sys.stderr)

        with open(output_file, "w") as f:
            model.write(f)

        # Handle conversions for this model
        _handle_conversions(args, output_path=output_file)

    if args.verbose:
        print(f"\nSuccessfully created {len(orders)} models in {args.output}/", file=sys.stderr)
        print(f"Files: {', '.join(f'{o}gram.arpa' for o in orders)}", file=sys.stderr)


def _handle_uniform(args):
    """Handle --uniform mode: create uniform language model."""
    if not args.output:
        print("Error: --uniform requires -o to specify output file", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print("Creating uniform language model...", file=sys.stderr)

    # Check if uniform arg is a vocabulary file or if we should extract from corpus
    if os.path.exists(args.uniform):
        # Use as vocabulary file
        if args.verbose:
            print(f"Loading vocabulary from {args.uniform}", file=sys.stderr)
        lm = ArpaBoLM.create_uniform(args.uniform, add_start=args.add_start)
    elif args.files:
        # Extract vocabulary from corpus files
        if args.verbose:
            print("Extracting vocabulary from corpus files...", file=sys.stderr)

        # Read corpus to get vocabulary
        temp_lm = ArpaBoLM(max_order=1, case=args.case, verbose=False)
        temp_lm._read_input_data(args.files, args.text, args.word_file)

        # Extract vocabulary
        vocab = list(temp_lm.grams[0].keys())

        if args.verbose:
            print(f"Extracted {len(vocab)} unique words", file=sys.stderr)

        # Create uniform model from extracted vocabulary
        lm = ArpaBoLM.create_uniform(vocab, add_start=args.add_start)
    else:
        print("Error: --uniform requires either a vocabulary file or corpus files", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        stats = lm.get_statistics()
        print(f"Created uniform model with {stats['vocab_size']} words", file=sys.stderr)

    # Write output
    with open(args.output, "w") as outfile:
        lm.write(outfile)

    if args.verbose:
        print(f"Wrote uniform model to {args.output}", file=sys.stderr)

    # Handle conversions
    _handle_conversions(args)


def main() -> None:
    """Main entry point for the arpalm command-line tool"""
    parser = _create_parser()
    args = parser.parse_args()

    # Handle list-presets mode first
    if args.list_presets:
        print_presets()
        return

    # Handle eval-only mode first (doesn't need other validation)
    if args.eval_only:
        _handle_eval_only(args)
        return

    # Apply preset if specified
    if args.preset:
        preset_config = get_preset(args.preset)

        # Override args with preset values (if not explicitly set by user)
        if args.max_order == 3:  # Default value, not explicitly set
            args.max_order = preset_config["recommended_order"]
        if args.smoothing_method == "good_turing":  # Default value
            args.smoothing_method = preset_config["smoothing_method"]

        if args.verbose:
            print(f"Using preset: {args.preset}", file=sys.stderr)
            print(f"  {preset_config['description']}", file=sys.stderr)
            print(f"  Order: {args.max_order}, Smoothing: {args.smoothing_method}", file=sys.stderr)

    # Validate arguments
    if args.case and args.case not in ["lower", "upper"]:
        parser.error("--case must be lower or upper (if given)")

    # Handle demo mode early (may add files needed by other modes)
    if args.demo:
        example_file = get_example_corpus()
        if args.verbose:
            print(f"Using example corpus: {example_file}", file=sys.stderr)
        args.files = [example_file] + (args.files or [])

        # Apply normalization for demo (unless user explicitly set options)
        if args.case is None:
            args.case = "lower"
            if args.verbose:
                print("Demo: Using lowercase normalization", file=sys.stderr)

    # Handle uniform mode
    if args.uniform:
        _handle_uniform(args)
        return

    if not args.load_model and not args.files and args.text is None and not args.demo:
        parser.error("Input must be specified with input files, --text, or --demo")

    # Handle smoothing comparison mode
    if args.compare_smoothing:
        if not args.eval:
            parser.error("--compare-smoothing requires --eval to specify test file")

        # Parse methods
        methods = None if args.compare_smoothing == "all" else [m.strip() for m in args.compare_smoothing.split(",")]

        if args.verbose:
            print(f"Comparing smoothing methods on corpus files: {args.files}", file=sys.stderr)

        # Use first file as corpus
        corpus_file = args.files[0] if args.files else None
        if not corpus_file:
            parser.error("--compare-smoothing requires input corpus file")

        results = compare_smoothing_methods(
            corpus_file=corpus_file,
            test_file=args.eval,
            methods=methods,
            max_order=args.max_order,
            verbose=args.verbose,
        )

        print_smoothing_comparison(results, test_file=args.eval, max_order=args.max_order)
        return

    # Handle multi-order training mode
    if args.orders:
        _handle_multi_order_training(args)
        return

    # Build or load model
    lm = _load_or_create_model(args)

    # Apply vocabulary pruning if requested
    if args.prune_vocab:
        if args.verbose:
            print("Pruning vocabulary...", file=sys.stderr)

        # Parse pruning specification
        try:
            method, threshold_str = args.prune_vocab.split(":")
            threshold = int(threshold_str) if method == "topk" else float(threshold_str)
        except ValueError:
            print(
                "Error: --prune-vocab format is METHOD:THRESHOLD (e.g., 'frequency:100' or 'topk:10000')",
                file=sys.stderr,
            )
            sys.exit(1)

        before_stats = lm.get_statistics()
        lm = lm.prune_vocabulary(method=method, threshold=threshold, keep_markers=True)
        after_stats = lm.get_statistics()

        if args.verbose:
            print(f"  Vocabulary: {before_stats['vocab_size']:,} → {after_stats['vocab_size']:,}", file=sys.stderr)
            reduction = (1 - after_stats["vocab_size"] / before_stats["vocab_size"]) * 100
            print(f"  Reduction: {reduction:.1f}%", file=sys.stderr)

    # Handle different output modes
    if args.stats:
        # Use new print_statistics with optional backoff analysis
        lm.print_statistics(test_file=args.backoff)
        return

    if args.debug:
        lm.interactive_debug()
        return

    # Evaluate if requested
    if args.eval:
        if args.verbose:
            print(f"\nEvaluating on {args.eval}...", file=sys.stderr)
        with open(args.eval) as f:
            results = lm.perplexity(f, oov_handling=args.oov_handling)
        lm.print_perplexity_results(results, test_file=args.eval)

        # Show backoff analysis if requested
        if args.backoff:
            print("\nBackoff Analysis")
            print("=" * 50)
            with open(args.backoff) as f:
                backoff = lm.backoff_rate(f)
            print(f"Overall backoff rate: {backoff['overall_backoff_rate']*100:.1f}%")
            print()
            print("Query resolution:")
            for order in sorted(backoff["order_usage"].keys(), reverse=True):
                usage = backoff["order_usage"][order]
                print(f"  {order}-gram hits: {usage*100:>5.1f}%")

    # Write output
    if args.output:
        with open(args.output, "w") as outfile:
            lm.write(outfile)
        _handle_conversions(args)
    else:
        lm.write(sys.stdout)


if __name__ == "__main__":
    main()
