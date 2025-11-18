#!/usr/bin/env python

"""Command-line interface for corpus normalization"""

import argparse
import sys

from arpabo.normalize import normalize_line


def main() -> None:
    """Main entry point for the arpalm-normalize tool"""
    parser = argparse.ArgumentParser(
        description="Normalize text corpus for language model training",
        epilog="Output can be piped directly to arpalm for model building",
    )
    parser.add_argument("files", nargs="*", help="input text files (default: stdin)")
    parser.add_argument("-o", "--output", type=str, help="output file (default: stdout)")
    parser.add_argument("-c", "--case", type=str, choices=["lower", "upper"], help="case normalization")
    parser.add_argument(
        "--no-unicode-norm",
        dest="unicode_norm",
        action="store_false",
        help="disable unicode NFC normalization (default: enabled)",
    )
    parser.add_argument(
        "-n",
        "--token-norm",
        action="store_true",
        help="normalize tokens (strip punctuation/symbols)",
    )
    parser.add_argument(
        "--add-markers",
        action="store_true",
        help="add sentence markers <s> and </s>",
    )
    parser.add_argument(
        "--no-markers",
        dest="add_markers",
        action="store_false",
        help="do not add sentence markers (default)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output to stderr")

    args = parser.parse_args()

    # Determine output
    outfile = open(args.output, "w") if args.output else sys.stdout

    # Determine input files
    input_files = args.files or [sys.stdin]

    line_count = 0
    skipped_count = 0

    try:
        for input_source in input_files:
            if isinstance(input_source, str):
                if args.verbose:
                    print(f"Processing: {input_source}", file=sys.stderr)
                infile = open(input_source)
            else:
                infile = input_source

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

            if isinstance(input_source, str):
                infile.close()

        if args.verbose:
            print(f"Processed {line_count} lines", file=sys.stderr)
            if skipped_count > 0:
                print(f"Skipped {skipped_count} empty lines", file=sys.stderr)

    finally:
        if args.output:
            outfile.close()


if __name__ == "__main__":
    main()
