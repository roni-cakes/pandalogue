"""Tests for the RelationshipDescriber class.

These tests verify inheritance, column validation, correlation calculations,
relationship descriptions, handling of invalid and incomplete data,
and the behavior of the class's special methods.

Run from the project folder with:
    python -m pytest tests/test_relationship.py -v
"""

import numpy as np
import pandas as pd
import pytest

from pandalogue.base import BaseDescriber
from pandalogue.relationship import RelationshipDescriber


def test_is_a_base_describer():
    """Verify that RelationshipDescriber inherits from BaseDescriber."""
    assert isinstance(RelationshipDescriber("a", "b"), BaseDescriber)


def test_column_and_other_are_read_only():
    """Verify that both column names can be accessed but not reassigned."""
    describer = RelationshipDescriber("a", "b")
    assert describer.column == "a"
    assert describer.other == "b"
    with pytest.raises(AttributeError):
        describer.other = "c"


def test_strong_positive_link():
    """Verify that strongly increasing columns produce a positive-link description."""
    df = pd.DataFrame({
        "hours_studied": [2, 5, 8, 10, 7],
        "grade": [62, 78, 88, 98, 85]
    })
    assert RelationshipDescriber("hours_studied", "grade").describe(df) == [
        "hours_studied and grade tend to go up together (a strong link)."
    ]


def test_strong_negative_link():
    """Verify that strongly decreasing relationships produce a negative-link description."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [9, 7, 5, 3, 1]})
    assert RelationshipDescriber("a", "b").describe(df) == [
        "When a goes up, b tends to go down (a strong link)."
    ]


A8 = [1, 2, 3, 4, 5, 6, 7, 8]


def test_moderate_link():
    """Verify that a moderate positive correlation is classified correctly."""
    df = pd.DataFrame({"a": A8, "b": [1, 6, 4, 3, 2, 5, 8, 7]})
    assert RelationshipDescriber("a", "b").describe(df) == [
        "a and b tend to go up together (a moderate link)."
    ]


def test_weak_link():
    """Verify that a weak positive correlation is classified correctly."""
    df = pd.DataFrame({"a": A8, "b": [3, 5, 7, 2, 4, 1, 6, 8]})
    assert RelationshipDescriber("a", "b").describe(df) == [
        "a and b tend to go up together (a weak link)."
    ]


def test_weak_negative_link():
    """Verify that a weak negative correlation is described correctly."""
    df = pd.DataFrame({"a": A8, "b": [6, 4, 2, 7, 5, 8, 3, 1]})
    assert RelationshipDescriber("a", "b").describe(df) == [
        "When a goes up, b tends to go down (a weak link)."
    ]


def test_no_clear_link():
    """Verify that a very weak correlation produces a no-clear-link description."""
    df = pd.DataFrame({"a": A8, "b": [4, 6, 2, 8, 3, 1, 7, 5]})
    assert RelationshipDescriber("a", "b").describe(df) == [
        "a and b do not show a clear link."
    ]


@pytest.mark.parametrize(
    "correlation, word",
    [
        (0.9, "strong"),
        (-0.7, "strong"),
        (0.5, "moderate"),
        (-0.4, "moderate"),
        (0.25, "weak"),
        (-0.2, "weak"),
        (0.19, None),
        (0.0, None),
    ],
)
def test_strength_word(correlation, word):
    """Verify that correlation magnitudes receive the correct strength labels."""
    assert RelationshipDescriber.strength_word(correlation) == word


def test_never_says_one_thing_causes_the_other():
    """Verify that relationship descriptions do not claim causation."""
    for b in ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 1, 2, 5, 4]):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": b})
        text = " ".join(RelationshipDescriber("a", "b").describe(df)).lower()
        assert "cause" not in text
        assert "because" not in text


def test_a_huge_outlier_does_not_hide_the_link():
    """Verify that Spearman correlation can identify a link despite a huge outlier."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 10_000_000]
    })
    assert "(a strong link)" in RelationshipDescriber("a", "b").describe(df)[0]


def test_rows_with_empty_cells_are_skipped():
    """Verify that rows with missing values are excluded from the comparison."""
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5, 6, 7],
        "b": [2, 4, 6, None, 10, 12, 14]
    })
    assert "go up together" in RelationshipDescriber("a", "b").describe(df)[0]


def test_infinite_values_are_skipped():
    """Verify that rows containing infinite values are excluded from the comparison."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, np.inf],
        "b": [2, 4, 6, 8, 10, 12]
    })
    assert "go up together" in RelationshipDescriber("a", "b").describe(df)[0]


def test_too_few_rows():
    """Verify that insufficient usable rows prevent correlation calculation."""
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [3, 4, 5, 6]})
    assert RelationshipDescriber("a", "b").describe(df) == [
        "There are too few rows with both a and b filled in to compare them."
    ]
    assert RelationshipDescriber("a", "b").correlation(df) is None


def test_rows_with_a_gap_in_either_column_do_not_count():
    """Verify that a row is excluded when either of its two values is missing."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, None],
        "b": [3, 4, None, 5, 6, 7]
    })
    assert "too few rows" in RelationshipDescriber("a", "b").describe(df)[0]


def test_exactly_the_minimum_number_of_rows_is_enough():
    """Verify that exactly MIN_ROWS usable pairs are sufficient for comparison."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    assert RelationshipDescriber.MIN_ROWS == 5
    assert "go up together" in RelationshipDescriber("a", "b").describe(df)[0]


def test_a_column_that_never_changes():
    """Verify that a constant column cannot be used to calculate a correlation."""
    df = pd.DataFrame({"a": [5, 5, 5, 5, 5], "b": [1, 2, 3, 4, 5]})
    assert RelationshipDescriber("a", "b").describe(df) == [
        "a has the same value in every row, so it cannot be compared with b."
    ]
    assert RelationshipDescriber("b", "a").describe(df) == [
        "a has the same value in every row, so it cannot be compared with b."
    ]


def test_correlation_is_a_number_between_minus_one_and_one():
    """Verify that the calculated correlation has the expected numerical value."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    assert RelationshipDescriber("a", "b").correlation(df) == pytest.approx(-1.0)


def test_returns_list_of_strings():
    """Verify that describe returns a list containing exactly one string."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    result = RelationshipDescriber("a", "b").describe(df)
    assert isinstance(result, list) and len(result) == 1
    assert isinstance(result[0], str)


def test_first_column_missing_raises_key_error():
    """Verify that a missing first column raises a KeyError."""
    df = pd.DataFrame({"b": [1, 2, 3]})
    with pytest.raises(KeyError):
        RelationshipDescriber("a", "b").describe(df)


def test_second_column_missing_raises_key_error():
    """Verify that a missing second column raises a KeyError."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError):
        RelationshipDescriber("a", "b").describe(df)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
        pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3]}),
        pd.DataFrame({"a": [1, 2, 3], "b": [True, False, True]}),
    ],
)
def test_non_number_column_raises_type_error(df):
    """Verify that text and Boolean columns are rejected as non-numerical data."""
    with pytest.raises(TypeError):
        RelationshipDescriber("a", "b").describe(df)


@pytest.mark.parametrize("bad_other", [None, "", "   "])
def test_empty_second_name_is_rejected(bad_other):
    """Verify that an empty or missing second column name raises a ValueError."""
    with pytest.raises(ValueError):
        RelationshipDescriber("a", bad_other)


def test_empty_first_name_is_rejected():
    """Verify that an empty first column name raises a ValueError."""
    with pytest.raises(ValueError):
        RelationshipDescriber("", "b")


def test_same_column_twice_is_rejected():
    """Verify that using the same column name twice raises a ValueError."""
    with pytest.raises(ValueError):
        RelationshipDescriber("a", "a")


def test_repr_shows_both_columns():
    """Verify that repr displays the class name and both column names."""
    assert repr(RelationshipDescriber("a", "b")) == (
        "RelationshipDescriber(column='a', other='b')"
    )


def test_equal_when_same_two_columns_in_the_same_order():
    """Verify that equality depends on both column names and their order."""
    assert RelationshipDescriber("a", "b") == RelationshipDescriber("a", "b")
    assert RelationshipDescriber("a", "b") != RelationshipDescriber("b", "a")
    assert RelationshipDescriber("a", "b") != RelationshipDescriber("a", "c")


def test_not_equal_to_other_kinds():
    """Verify that a RelationshipDescriber is not equal to another describer type."""
    from pandalogue.number import NumberDescriber

    assert RelationshipDescriber("a", "b") != NumberDescriber("a")


def test_works_in_a_set():
    """Verify that equal describers are treated as one item in a set."""
    assert len({
        RelationshipDescriber("a", "b"),
        RelationshipDescriber("a", "b")
    }) == 1
