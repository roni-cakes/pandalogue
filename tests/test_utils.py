"""Tests for the shared utility functions in pandalogue.

These tests verify number and percentage formatting, text joining,
empty-cell detection, number parsing, text-to-number conversion,
mixed-value detection, ID-column recognition, and infinite-value notes.

Run from the project folder with:
    python -m pytest tests/test_utils.py -v
"""

import numpy as np
import pandas as pd
import pytest

from pandalogue.utils import (
    count_empty,
    count_text,
    drop_empty,
    empty_mask,
    format_number,
    format_percent,
    infinite_note,
    join_with_and,
    looks_like_id,
    mixes_numbers_and_words,
    parse_number,
    text_to_numbers,
)


@pytest.mark.parametrize(
    "part, whole, expected",
    [
        (3, 6, "50%"),
        (1, 6, "16.7%"),
        (2, 3, "66.7%"),
        (6, 6, "100%"),
        (0, 6, "0%"),
        (1, 8, "12.5%"),
    ],
)
def test_format_percent(part, whole, expected):
    """Verify that percentages are formatted with at most one decimal place."""
    assert format_percent(part, whole) == expected


def test_format_percent_with_zero_whole_does_not_crash():
    """Verify that formatting a percentage with a zero total returns 0%."""
    assert format_percent(0, 0) == "0%"


def test_join_with_and_empty():
    """Verify that joining an empty list returns an empty string."""
    assert join_with_and([]) == ""


def test_join_with_and_one_item():
    """Verify that joining one item returns that item without extra words."""
    assert join_with_and(["A"]) == "A"


def test_join_with_and_two_items():
    """Verify that two items are joined using the word 'and'."""
    assert join_with_and(["A", "B"]) == "A and B"


def test_join_with_and_three_items():
    """Verify that three items are joined with commas and a final 'and'."""
    assert join_with_and(["A", "B", "C"]) == "A, B and C"


def test_join_with_and_turns_items_into_text():
    """Verify that non-string items are converted to strings before joining."""
    assert join_with_and([1, True, None]) == "1, True and None"


def test_join_with_and_limit_hides_the_rest():
    """Verify that items beyond the specified limit are counted as hidden."""
    assert join_with_and(["A", "B", "C", "D"], limit=2) == "A, B and 2 more"


def test_join_with_and_limit_not_reached_changes_nothing():
    """Verify that a limit larger than or equal to the item count changes nothing."""
    assert join_with_and(["A", "B"], limit=5) == "A and B"
    assert join_with_and(["A", "B", "C"], limit=3) == "A, B and C"


def test_count_text_singular():
    """Verify that a count of one uses the singular noun."""
    assert count_text(1, "empty cell") == "1 empty cell"


@pytest.mark.parametrize("count", [0, 2, 10])
def test_count_text_plural(count):
    """Verify that counts other than one use the plural noun."""
    assert count_text(count, "empty cell") == f"{count} empty cells"


# ---------- format_number ----------

def test_format_number_matches_number_describer_rules():
    """Verify that numbers follow the formatting rules used by NumberDescriber."""
    assert format_number(85.0) == "85"
    assert format_number(0.0002) == "0.0002"
    assert format_number(2000000) == "2,000,000"


# ---------- the shared "is this empty?" rule ----------

def test_empty_mask_text_column():
    """Verify that blank strings, spaces, tabs, and None are marked as empty."""
    series = pd.Series(["ok", "", "  ", None, "\t", "0"])
    assert empty_mask(series).tolist() == [False, True, True, True, True, False]


def test_empty_mask_number_column():
    """Verify that NaN is empty while ordinary numerical values are not."""
    series = pd.Series([1.0, np.nan, 0.0])
    assert empty_mask(series).tolist() == [False, True, False]


def test_empty_mask_date_column():
    """Verify that missing dates are marked as empty."""
    series = pd.to_datetime(pd.Series(["2026-01-01", None]))
    assert empty_mask(series).tolist() == [False, True]


def test_empty_mask_true_false_column():
    """Verify that Boolean values are not incorrectly treated as empty."""
    series = pd.Series([True, False])
    assert empty_mask(series).tolist() == [False, False]


def test_empty_mask_category_column():
    """Verify that blank and missing values are detected in categorical data."""
    series = pd.Series(["a", " ", None], dtype="category")
    assert empty_mask(series).tolist() == [False, True, True]


def test_empty_mask_keeps_the_index():
    """Verify that the empty-value mask preserves the original Series index."""
    series = pd.Series(["a", ""], index=[10, 20])
    assert empty_mask(series).index.tolist() == [10, 20]


def test_empty_mask_of_empty_column():
    """Verify that an empty Series produces an empty Boolean mask."""
    assert empty_mask(pd.Series([], dtype=object)).tolist() == []


def test_count_empty():
    """Verify that count_empty returns the number of empty cells."""
    assert count_empty(pd.Series(["a", "", None, " "])) == 3


def test_drop_empty():
    """Verify that drop_empty removes empty cells while keeping valid values."""
    cleaned = drop_empty(pd.Series(["a", "", None, "b"]))
    assert cleaned.tolist() == ["a", "b"]


# ---------- parse_number / text_to_numbers ----------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("85", 85.0),
        (" 90.5 ", 90.5),
        ("-3", -3.0),
        ("1e3", 1000.0),
        ("1,200", 1200.0),
        (".5", 0.5),
        ("0", 0.0),
        ("0.5", 0.5),
    ],
)
def test_parse_number_accepts_plain_numbers(text, expected):
    """Verify that valid numerical strings are converted into floating-point values."""
    assert parse_number(text) == expected


@pytest.mark.parametrize(
    "text",
    ["abc", "", "12abc", "007", "nan", "inf", "1,2", "$5", "50%", "1 000"],
)
def test_parse_number_rejects_everything_else(text):
    """Verify that invalid numbers, words, and ID-like codes are rejected."""
    assert parse_number(text) is None


def test_text_to_numbers_reads_a_text_column():
    """Verify that numerical strings are converted and empty cells become NaN."""
    result = text_to_numbers(pd.Series(["78", " 91", "", None]))
    assert result.iloc[:2].tolist() == [78.0, 91.0]
    assert result.isna().tolist() == [False, False, True, True]


def test_text_to_numbers_refuses_words():
    """Verify that a text column containing ordinary words is rejected."""
    assert text_to_numbers(pd.Series(["78", "high"])) is None


def test_text_to_numbers_refuses_codes_with_leading_zeros():
    """Verify that ID-like numerical strings with leading zeros are rejected."""
    assert text_to_numbers(pd.Series(["007", "042"])) is None


def test_text_to_numbers_refuses_columns_with_no_numbers():
    """Verify that a column without any numerical text values is rejected."""
    assert text_to_numbers(pd.Series(["", None])) is None


def test_text_to_numbers_ignores_real_number_columns():
    """Verify that an already numerical column is not converted as text."""
    assert text_to_numbers(pd.Series([1, 2, 3])) is None


def test_text_to_numbers_refuses_true_false_in_text_column():
    """Verify that Boolean values mixed with numerical text are rejected."""
    assert text_to_numbers(pd.Series(["1", True], dtype=object)) is None


def test_text_to_numbers_keeps_the_index():
    """Verify that converting text to numbers preserves the original index."""
    result = text_to_numbers(pd.Series(["1", "2"], index=[5, 6]))
    assert result.index.tolist() == [5, 6]


# ---------- mixes_numbers_and_words ----------

def test_mixed_column_is_detected():
    """Verify that a column containing numerical text and words is detected as mixed."""
    assert mixes_numbers_and_words(pd.Series(["1", "apple", "3"]))


def test_real_numbers_and_words_are_mixed():
    """Verify that actual numerical values mixed with words are detected."""
    assert mixes_numbers_and_words(pd.Series([1, "apple", 2.5], dtype=object))


def test_only_words_is_not_mixed():
    """Verify that a column containing only words is not classified as mixed."""
    assert not mixes_numbers_and_words(pd.Series(["pear", "apple"]))


def test_only_numbers_as_text_is_not_mixed():
    """Verify that a column containing only numerical strings is not mixed."""
    assert not mixes_numbers_and_words(pd.Series(["1", "2"]))


def test_mixed_ignores_empty_cells():
    """Verify that empty cells do not count as numbers or words."""
    assert not mixes_numbers_and_words(pd.Series(["pear", "", None]))


def test_number_column_is_not_mixed():
    """Verify that a purely numerical column is not classified as mixed."""
    assert not mixes_numbers_and_words(pd.Series([1, 2, 3]))


# ---------- looks_like_id ----------

@pytest.mark.parametrize(
    "name",
    [
        "id",
        "ID",
        "Id",
        "student_id",
        "StudentID",
        "studentId",
        "student id",
        "user-id",
        "id_number",
        "ID Number",
    ],
)
def test_id_names(name):
    """Verify that common ID column naming patterns are recognized."""
    assert looks_like_id(name)


@pytest.mark.parametrize(
    "name",
    ["paid", "grade", "idea", "valid", "identity", "hours_studied", "width", ""],
)
def test_names_that_are_not_ids(name):
    """Verify that ordinary column names are not incorrectly identified as IDs."""
    assert not looks_like_id(name)


def test_non_text_names_do_not_crash():
    """Verify that non-string column names are handled without raising errors."""
    assert not looks_like_id(0)
    assert not looks_like_id(None)


# ---------- infinite_note ----------

def test_infinite_note_singular():
    """Verify that one excluded infinite value uses the singular sentence."""
    assert infinite_note(1) == "1 infinite value was left out."


@pytest.mark.parametrize("count", [2, 5])
def test_infinite_note_plural(count):
    """Verify that multiple excluded infinite values use the plural sentence."""
    assert infinite_note(count) == f"{count} infinite values were left out."
