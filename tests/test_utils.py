"""Tests for utility functions."""

import pytest

from arpabo.utils import parse_order_spec, parse_range_spec


class TestParseRangeSpec:
    """Test general range specification parsing."""

    def test_single_value(self):
        """Test parsing single value."""
        assert parse_range_spec("3") == [3]
        assert parse_range_spec("1") == [1]
        assert parse_range_spec("10") == [10]

    def test_comma_separated(self):
        """Test parsing comma-separated values."""
        assert parse_range_spec("1,2,4") == [1, 2, 4]
        assert parse_range_spec("3,1,2") == [1, 2, 3]  # Should be sorted
        assert parse_range_spec("5,3,1,7") == [1, 3, 5, 7]

    def test_range(self):
        """Test parsing ranges."""
        assert parse_range_spec("1-4") == [1, 2, 3, 4]
        assert parse_range_spec("2-5") == [2, 3, 4, 5]
        assert parse_range_spec("1-1") == [1]

    def test_mixed(self):
        """Test parsing mixed specifications."""
        assert parse_range_spec("1-3,5") == [1, 2, 3, 5]
        assert parse_range_spec("1,3-5,7") == [1, 3, 4, 5, 7]
        assert parse_range_spec("1-3,5,7-10") == [1, 2, 3, 5, 7, 8, 9, 10]

    def test_duplicates_removed(self):
        """Test that duplicates are removed."""
        assert parse_range_spec("1,2,2,3") == [1, 2, 3]
        assert parse_range_spec("1-3,2-4") == [1, 2, 3, 4]

    def test_whitespace_handling(self):
        """Test whitespace is handled correctly."""
        assert parse_range_spec(" 1, 2, 3 ") == [1, 2, 3]
        assert parse_range_spec("1 - 4") == [1, 2, 3, 4]
        assert parse_range_spec(" 1 - 3 , 5 ") == [1, 2, 3, 5]

    def test_empty_spec(self):
        """Test empty specification raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_range_spec("")
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_range_spec("   ")

    def test_invalid_number(self):
        """Test invalid numbers raise errors."""
        with pytest.raises(ValueError, match="must be integer"):
            parse_range_spec("1,2,abc")
        with pytest.raises(ValueError, match="must be integer"):
            parse_range_spec("1.5")

    def test_invalid_range(self):
        """Test invalid ranges raise errors."""
        with pytest.raises(ValueError, match="start must be <= end"):
            parse_range_spec("5-3")
        with pytest.raises(ValueError, match="Invalid range specification"):
            parse_range_spec("1-2-3")

    def test_negative_value(self):
        """Test negative values raise errors with default min_value."""
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_range_spec("-1")
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_range_spec("0")
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_range_spec("1,-3")

    def test_zero_value(self):
        """Test zero value raises error with default min_value."""
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_range_spec("0")
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_range_spec("0-3")

    def test_custom_min_value(self):
        """Test custom min_value parameter."""
        # Allow 0 and above
        assert parse_range_spec("0,1,2", min_value=0) == [0, 1, 2]
        assert parse_range_spec("0-3", min_value=0) == [0, 1, 2, 3]

        # Require >= 10
        assert parse_range_spec("10,15,20", min_value=10) == [10, 15, 20]
        assert parse_range_spec("10-12", min_value=10) == [10, 11, 12]

        # Should reject values below min
        with pytest.raises(ValueError, match="must be >= 10"):
            parse_range_spec("5", min_value=10)


class TestParseOrderSpec:
    """Test backwards-compatible parse_order_spec wrapper."""

    def test_same_as_range_spec(self):
        """Test that parse_order_spec behaves same as parse_range_spec with min_value=1."""
        assert parse_order_spec("1-4") == parse_range_spec("1-4", min_value=1)
        assert parse_order_spec("1,3,5") == parse_range_spec("1,3,5", min_value=1)
        assert parse_order_spec("2-4,7") == parse_range_spec("2-4,7", min_value=1)

    def test_rejects_zero_and_negative(self):
        """Test that parse_order_spec rejects zero and negative values."""
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_order_spec("0")
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_order_spec("-1")
