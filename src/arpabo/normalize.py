"""Text normalization utilities for ArpaLM"""

import re
import unicodedata as ud
from typing import Optional

# Unicode categories to remove during text normalization (punctuation, symbols, etc.)
NORM_EXCLUDE_CATEGORIES = {"P", "S", "C", "M", "Z"}


def normalize_unicode(text: str) -> str:
    """Apply Unicode NFC normalization to text.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text in NFC form
    """
    return ud.normalize("NFC", text)


def normalize_token(token: str, exclude_categories: Optional[set[str]] = None) -> str:
    """Normalize a single token by stripping leading/trailing punctuation and symbols.

    Preserves special sentence markers <s> and </s>.

    Args:
        token: Input token to normalize
        exclude_categories: Unicode categories to strip (default: P, S, C, M, Z)

    Returns:
        Normalized token with punctuation/symbols stripped from edges
    """
    if exclude_categories is None:
        exclude_categories = NORM_EXCLUDE_CATEGORIES

    # Preserve special markers
    if token in ("<s>", "</s>"):
        return token

    # Strip leading excluded characters
    while token and ud.category(token[0])[0] in exclude_categories:
        token = token[1:]

    # Strip trailing excluded characters
    while token and ud.category(token[-1])[0] in exclude_categories:
        token = token[:-1]

    return token


def normalize_case(text: str, case: Optional[str] = None) -> str:
    """Normalize text case.

    Args:
        text: Input text
        case: 'lower' or 'upper' (None leaves unchanged)

    Returns:
        Case-normalized text
    """
    if case == "lower":
        return text.lower()
    elif case == "upper":
        return text.upper()
    return text


def normalize_line(
    line: str, unicode_norm: bool = True, case: Optional[str] = None, token_norm: bool = False, add_markers: bool = True
) -> Optional[list[str]]:
    """Normalize a line of text and return tokens.

    Handles three input formats automatically:
    - Sphinx sentfile: <s> .* </s> (filename)
    - Text with markers: <s> .* </s>
    - Plain text without markers

    Args:
        line: Input line to process
        unicode_norm: Apply NFC normalization to the line
        case: 'lower' or 'upper' for case normalization
        token_norm: Strip punctuation/symbols from token edges
        add_markers: Add <s> and </s> markers if not present

    Returns:
        List of normalized tokens, or None if line should be skipped
    """
    # Unicode normalize the whole line first
    if unicode_norm:
        line = normalize_unicode(line)

    # Case normalization
    if case:
        line = normalize_case(line, case)

    line = line.strip()

    # Determine if line has markers:
    # 1. Sphinx sentfile format: <s> .* </s> (filename) - strip parens, has markers
    # 2. Text with markers: <s> .* </s> - has markers
    # 3. Plain text without markers - needs markers added
    has_markers = False
    if re.search(r"^<s>.*</s>\s+\(.*\)$", line):
        # Sphinx sentfile format - strip the (filename) part
        line = re.sub(r"\s+\(.*\)$", "", line)
        has_markers = True
    elif line.startswith("<s>") and line.endswith("</s>"):
        # Text with markers (but not sentfile format)
        has_markers = True

    words = line.split()
    if not words:
        return None

    # Add markers if needed and requested
    if add_markers and not has_markers:
        words = [w for w in words if w not in ("<s>", "</s>")]
        if not words:
            return None
        words = ["<s>"] + words + ["</s>"]

    # Normalize individual tokens
    if token_norm:
        words = [normalize_token(w) for w in words]
        words = [w for w in words if len(w)]
        if not words:
            return None

    return words


def clean_text(text: str, unicode_norm: bool = True, case: Optional[str] = None, token_norm: bool = False) -> str:
    """Clean text by applying normalization without sentence markers.

    Useful for preprocessing text before feeding to language model.

    Args:
        text: Input text
        unicode_norm: Apply NFC normalization
        case: 'lower' or 'upper' for case normalization
        token_norm: Strip punctuation/symbols from token edges

    Returns:
        Cleaned text string
    """
    if unicode_norm:
        text = normalize_unicode(text)

    if case:
        text = normalize_case(text, case)

    if token_norm:
        tokens = text.split()
        tokens = [normalize_token(t) for t in tokens]
        text = " ".join(t for t in tokens if t)

    return text
