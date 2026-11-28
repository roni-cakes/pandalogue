
import pandas as pd

from pandalogue.utils import (
    format_percent,
    format_number,
    join_with_and,
    count_text,
    infinite_note,
    empty_mask,
    count_empty,
    drop_empty,
    parse_number,
    text_to_numbers,
    mixes_numbers_and_words,
    looks_like_id,
)


def test_format_percent():
    assert format_percent(5, 10) == "50%"
    assert format_percent(1, 3) == "33.3%"
    assert format_percent(5, 0) == "0%"


def test_format_number():
    assert format_number(200082.46) == "200,082.46"
    assert format_number(1234) == "1,234"
    assert format_number(0) == "0"
    assert format_number(0.0002) == "0.0002"


def test_join_with_and():
    assert join_with_and([]) == ""
    assert join_with_and(["Apple"]) == "Apple"
    assert join_with_and(["Apple", "Banana"]) == "Apple and Banana"
    assert join_with_and(
        ["Apple", "Banana", "Orange"]
    ) == "Apple, Banana and Orange"
    assert join_with_and(
        ["Apple", "Banana", "Orange"], limit=2
    ) == "Apple, Banana and 1 more"


def test_count_text():
    assert count_text(1, "cell") == "1 cell"
    assert count_text(3, "cell") == "3 cells"


def test_infinite_note():
    assert infinite_note(0) == ""
    assert infinite_note(1) == "1 infinite value was left out."
    assert infinite_note(2) == "2 infinite values were left out."


def test_empty_mask():
    series = pd.Series(["Apple", "", None, "Banana", "  "])
    assert empty_mask(series).tolist() == [
        False, True, True, False, True
    ]


def test_count_empty():
    series = pd.Series(["Apple", "", None, "Banana"])
    assert count_empty(series) == 2


def test_drop_empty():
    series = pd.Series(["Apple", "", None, "Banana"])
    assert drop_empty(series).tolist() == ["Apple", "Banana"]


def test_parse_number():
    assert parse_number("123") == 123.0
    assert parse_number("12.5") == 12.5
    assert parse_number("-10") == -10.0
    assert parse_number("abc") is None
    assert parse_number("") is None
    assert parse_number("00123") is None
    assert parse_number("inf") is None
    assert parse_number(123) is None


def test_text_to_numbers():
    result = text_to_numbers(pd.Series(["10", "20", "30"]))
    assert result.tolist() == [10.0, 20.0, 30.0]


def test_text_to_numbers_invalid():
    assert text_to_numbers(
        pd.Series(["10", "Apple", "30"])
    ) is None


def test_text_to_numbers_empty():
    assert text_to_numbers(
        pd.Series(["", None])
    ) is None


def test_mixes_numbers_and_words():
    assert mixes_numbers_and_words(
        pd.Series(["10", "Apple", "30"])
    ) is True

    assert mixes_numbers_and_words(
        pd.Series(["10", "20", "30"])
    ) is False

    assert mixes_numbers_and_words(
        pd.Series(["Apple", "Banana"])
    ) is False


def test_looks_like_id():
    assert looks_like_id("id") is True
    assert looks_like_id("student_id") is True
    assert looks_like_id("ID_number") is True
    assert looks_like_id("studentId") is True
    assert looks_like_id("student_name") is False
    assert looks_like_id(123) is False

