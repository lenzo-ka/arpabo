"""Utility functions for arpabo."""


def parse_range_spec(spec: str, min_value: int = 1) -> list[int]:
    """
    Parse range specification string into list of integers.

    General-purpose parser for integer specifications with ranges.
    Useful for any scalar dimension (n-gram orders, dimensions, indices, etc.).

    Supports:
    - Single value: "3" -> [3]
    - Comma-separated: "1,2,4" -> [1, 2, 4]
    - Ranges: "1-4" -> [1, 2, 3, 4]
    - Mixed: "1-3,5,7-10" -> [1, 2, 3, 5, 7, 8, 9, 10]

    Args:
        spec: Range specification string
        min_value: Minimum allowed value (default: 1)

    Returns:
        Sorted list of unique integers

    Raises:
        ValueError: If specification is invalid

    Examples:
        >>> parse_range_spec("3")
        [3]
        >>> parse_range_spec("1,2,4")
        [1, 2, 4]
        >>> parse_range_spec("1-4")
        [1, 2, 3, 4]
        >>> parse_range_spec("1-3,5,7-10")
        [1, 2, 3, 5, 7, 8, 9, 10]
        >>> parse_range_spec("5,10,15", min_value=0)
        [5, 10, 15]
    """
    if not spec or not spec.strip():
        raise ValueError("Range specification cannot be empty")

    orders = set()

    # Split by commas
    parts = spec.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if it's a range (contains dash but not at the start)
        if "-" in part and not part.startswith("-"):
            range_parts = part.split("-")

            # Filter out empty strings from leading/trailing dashes
            range_parts = [p for p in range_parts if p]

            if len(range_parts) != 2:
                raise ValueError(f"Invalid range specification: '{part}'")

            try:
                start = int(range_parts[0])
                end = int(range_parts[1])
            except ValueError as e:
                raise ValueError(f"Invalid range specification: '{part}' (must be integers)") from e

            if start > end:
                raise ValueError(f"Invalid range: {start}-{end} (start must be <= end)")

            if start < min_value:
                raise ValueError(f"Value must be >= {min_value}, got {start}")

            orders.update(range(start, end + 1))
        else:
            # Single number (including negative numbers which will be caught below)
            try:
                order = int(part)
            except ValueError as e:
                raise ValueError(f"Invalid specification: '{part}' (must be integer)") from e

            if order < min_value:
                raise ValueError(f"Value must be >= {min_value}, got {order}")

            orders.add(order)

    if not orders:
        raise ValueError("No valid values found in specification")

    return sorted(orders)


# Backwards compatibility alias for n-gram order parsing
def parse_order_spec(spec: str) -> list[int]:
    """
    Parse n-gram order specification string into list of integers.

    This is a convenience wrapper around parse_range_spec() with min_value=1.
    For general-purpose range parsing, use parse_range_spec() directly.

    Args:
        spec: Order specification string (e.g., "1-4", "1,3,5")

    Returns:
        Sorted list of unique integers >= 1

    Examples:
        >>> parse_order_spec("1-4")
        [1, 2, 3, 4]
        >>> parse_order_spec("1,3,5")
        [1, 3, 5]
    """
    return parse_range_spec(spec, min_value=1)
