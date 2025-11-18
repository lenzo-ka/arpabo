#!/usr/bin/env python

"""Command-line interface for corpus normalization"""

import argparse
import sys

from arpabo.normalize import normalize_line


def _process_file(infile, outfile, args):
    """Process a single input file.

    Args:
        infile: Input file handle
        outfile: Output file handle
        args: Parsed arguments

    Returns:
        Tuple of (lines_processed, lines_skipped)
    """
    line_count = 0
    skipped_count = 0

    for line in infile:
        words = normalize_line(
            line,
            unicode_norm=args.unicode_norm,
            case=args.case,
            token_norm=args.token_norm,
            add_markers=args.add_markers,
        )

        if words is None:
            skipped_count += 1
            continue

        print(" ".join(words), file=outfile)
        line_count += 1

    return line_count, skipped_count


def main() -> None:
    """Main entry point for the arpabo-normalize tool"""
    parser = argparse.ArgumentParser(
        description="Normalize text corpus for language model training",
        epilog="Output can be piped directly to arpabo for model building",
    )
    parser.add_argument("files", nargs="*", help="input text files (default: stdin)")
    parser.add_argument("-o", "--output", type=str, help="output file (default: stdout)")
    parser.add_argument("-c", "--case", type=str, choices=["lower", "upper"], help="case normalization")
    parser.add_argument(
        "--no-unicode-norm", dest="unicode_norm", action="store_false", help="disable unicode NFC normalization"
    )
    parser.add_argument("-n", "--token-norm", action="store_true", help="normalize tokens (strip punctuation/symbols)")
    parser.add_argument("--add-markers", action="store_true", help="add sentence markers <s> and </s>")
    parser.add_argument(
        "--no-markers", dest="add_markers", action="store_false", help="do not add sentence markers (default)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output to stderr")

    args = parser.parse_args()

    total_lines = 0
    total_skipped = 0

    # Determine input files
    input_files = args.files if args.files else ["-"]

    # Process with proper context managers
    if args.output:
        with open(args.output, "w") as outfile:
            for input_source in input_files:
                if input_source == "-":
                    if args.verbose:
                        print("Processing: stdin", file=sys.stderr)
                    lines, skipped = _process_file(sys.stdin, outfile, args)
                else:
                    if args.verbose:
                        print(f"Processing: {input_source}", file=sys.stderr)
                    with open(input_source) as infile:
                        lines, skipped = _process_file(infile, outfile, args)

                total_lines += lines
                total_skipped += skipped
    else:
        # Output to stdout
        for input_source in input_files:
            if input_source == "-":
                if args.verbose:
                    print("Processing: stdin", file=sys.stderr)
                lines, skipped = _process_file(sys.stdin, sys.stdout, args)
            else:
                if args.verbose:
                    print(f"Processing: {input_source}", file=sys.stderr)
                with open(input_source) as infile:
                    lines, skipped = _process_file(infile, sys.stdout, args)

            total_lines += lines
            total_skipped += skipped

    if args.verbose:
        print(f"Processed {total_lines} lines", file=sys.stderr)
        if total_skipped > 0:
            print(f"Skipped {total_skipped} empty lines", file=sys.stderr)


if __name__ == "__main__":
    main()
