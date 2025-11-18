"""Tests for text normalization"""

from arpalm.normalize import (
    clean_text,
    normalize_case,
    normalize_line,
    normalize_token,
    normalize_unicode,
)


class TestNormalizeUnicode:
    """Test Unicode normalization"""

    def test_nfc_normalization(self):
        # Combining characters
        text = "café"  # Could be c+a+f+e+´ or c+a+f+é
        normalized = normalize_unicode(text)
        assert normalized == "café"
        assert len(normalized) <= len(text) + 1  # NFC should compose

    def test_preserve_regular_text(self):
        text = "hello world"
        assert normalize_unicode(text) == text


class TestNormalizeToken:
    """Test token normalization"""

    def test_strip_punctuation(self):
        assert normalize_token("hello!") == "hello"
        assert normalize_token("(hello)") == "hello"
        assert normalize_token("...hello...") == "hello"

    def test_preserve_sentence_markers(self):
        assert normalize_token("<s>") == "<s>"
        assert normalize_token("</s>") == "</s>"

    def test_all_punctuation(self):
        assert normalize_token("***") == ""
        assert normalize_token("...") == ""
        assert normalize_token("!!!") == ""

    def test_mixed_content(self):
        assert normalize_token("(hello-world)") == "hello-world"
        assert normalize_token("'word'") == "word"


class TestNormalizeCase:
    """Test case normalization"""

    def test_lowercase(self):
        assert normalize_case("HELLO WORLD", "lower") == "hello world"
        assert normalize_case("Hello World", "lower") == "hello world"

    def test_uppercase(self):
        assert normalize_case("hello world", "upper") == "HELLO WORLD"
        assert normalize_case("Hello World", "upper") == "HELLO WORLD"

    def test_no_change(self):
        text = "Hello World"
        assert normalize_case(text, None) == text


class TestNormalizeLine:
    """Test line normalization"""

    def test_plain_text_adds_markers(self):
        words = normalize_line("hello world", add_markers=True)
        assert words == ["<s>", "hello", "world", "</s>"]

    def test_existing_markers_preserved(self):
        words = normalize_line("<s> hello world </s>", add_markers=True)
        assert words == ["<s>", "hello", "world", "</s>"]

    def test_no_markers(self):
        words = normalize_line("hello world", add_markers=False)
        assert words == ["hello", "world"]

    def test_sphinx_format(self):
        words = normalize_line("<s> hello world </s> (filename)", add_markers=True)
        assert words == ["<s>", "hello", "world", "</s>"]

    def test_empty_line(self):
        assert normalize_line("", add_markers=True) is None
        assert normalize_line("   ", add_markers=True) is None

    def test_token_normalization(self):
        words = normalize_line("hello, world!", token_norm=True, add_markers=True)
        assert words == ["<s>", "hello", "world", "</s>"]

    def test_all_punctuation_line(self):
        words = normalize_line("* * * * *", token_norm=True, add_markers=True)
        # After stripping punctuation, only markers remain
        assert words == ["<s>", "</s>"]

    def test_case_normalization(self):
        words = normalize_line("HELLO WORLD", case="lower", add_markers=True)
        assert words == ["<s>", "hello", "world", "</s>"]


class TestCleanText:
    """Test clean_text utility"""

    def test_basic_cleaning(self):
        text = "Hello, World!"
        cleaned = clean_text(text, token_norm=True)
        assert cleaned == "Hello World"

    def test_lowercase_and_token_norm(self):
        text = "Hello, WORLD!"
        cleaned = clean_text(text, case="lower", token_norm=True)
        assert cleaned == "hello world"

    def test_preserve_without_norm(self):
        text = "Hello, World!"
        cleaned = clean_text(text, token_norm=False)
        assert cleaned == text
