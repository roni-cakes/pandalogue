"""Provide shared helper functions for the pandalogue package.

This module contains reusable utilities for formatting numbers and text,
joining lists, identifying empty cells, and interpreting columns that
contain numbers stored as text.

These helpers provide consistent behavior across the describer classes.
In particular, empty_mask() establishes the shared rule for identifying
missing and blank values in a pandas Series.
"""

import math
import re

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def format_percent(part, whole):
    """Format a part of a whole as a readable percentage.

    The result contains at most one decimal place. Trailing zeroes and
    unnecessary decimal points are removed.

    Args:
        part (int or float): The portion or value being measured.
        whole (int or float): The total value. If it is zero, the result
            is returned as ``"0%"``.

    Returns:
        str: The formatted percentage, such as ``"50%"`` or ``"33.3%"``.

    Example:
        >>> format_percent(1, 6)
        '16.7%'
        >>> format_percent(3, 6)
        '50%'
    """
    if whole == 0:
        return "0%"
    text = f"{part / whole * 100:.1f}"
    return text.rstrip("0").rstrip(".") + "%"


def format_number(value):
    """Convert a number into readable text.

    Normal numbers are formatted with commas for thousands and at most
    two decimal places. Small numbers retain their first two meaningful
    digits so they are not incorrectly displayed as zero. Values smaller
    than 0.000001 use scientific notation.

    Args:
        value (int or float): The numerical value to format.

    Returns:
        str: The formatted number as text.

    Example:
        >>> format_number(85.0)
        '85'
        >>> format_number(200082.456)
        '200,082.46'
        >>> format_number(0.0002)
        '0.0002'
        >>> format_number(0.00023456)
        '0.00023'
    """
    value = float(value)
    if value == 0:
        return "0"
    if math.isnan(value) or math.isinf(value):
        return str(value)

    size = abs(value)
    if size >= 0.01:
        text = f"{value:,.2f}"
    elif size >= 1e-6:
        decimals = -math.floor(math.log10(size)) + 1
        text = f"{value:.{decimals}f}"
    else:
        return f"{value:.1e}"
    return text.rstrip("0").rstrip(".")


def join_with_and(items, limit=None):
    """Join a collection of items into a readable phrase.

    Each item is converted to text before joining. When a limit is
    provided, only the specified number of items are displayed.
    Remaining items are summarized using ``"and N more"``.

    Args:
        items (list): The items to join into a phrase.
        limit (int, optional): Maximum number of items to display.
            Any remaining items are counted instead.

    Returns:
        str: A readable phrase. Examples include ``"A"``, ``"A and B"``,
        and ``"A, B and C"``. An empty collection returns an empty string.

    Example:
        >>> join_with_and(["A", "B", "C"])
        'A, B and C'
        >>> join_with_and(["A", "B", "C", "D"], limit=2)
        'A, B and 2 more'
    """
    texts = [str(item) for item in items]
    hidden = 0
    if limit is not None and len(texts) > limit:
        hidden = len(texts) - limit
        texts = texts[:limit]
    if hidden:
        return ", ".join(texts) + f" and {hidden} more"
    if len(texts) <= 1:
        return "".join(texts)
    return ", ".join(texts[:-1]) + " and " + texts[-1]


def count_text(count, noun):
    """Format a numerical count with a singular or plural noun.

    An ``s`` is added to the noun whenever the count is not exactly one.

    Args:
        count (int): The number of items being counted.
        noun (str): The singular noun describing the items.

    Returns:
        str: The count followed by the appropriately formatted noun.

    Example:
        >>> count_text(1, "empty cell")
        '1 empty cell'
        >>> count_text(3, "empty cell")
        '3 empty cells'
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def infinite_note(count):
    """Create a sentence describing excluded infinite values.

    The verb is adjusted according to whether one or multiple infinite
    values were excluded.

    Args:
        count (int): The number of infinite values that were left out.

    Returns:
        str: A sentence describing the excluded values.

    Example:
        >>> infinite_note(1)
        '1 infinite value was left out.'
        >>> infinite_note(2)
        '2 infinite values were left out.'
    """
    verb = "was" if count == 1 else "were"
    return f"{count_text(count, 'infinite value')} {verb} left out."


# ---------------------------------------------------------------------------
# "Is this cell empty?" (one rule for everyone)
# ---------------------------------------------------------------------------

def _can_hold_text(series):
    """Check whether a pandas Series can contain text values.

    This helper identifies object, string, and categorical data types.
    These types may contain blank strings that should be treated as
    empty cells by empty_mask().

    Args:
        series (pandas.Series): The column whose data type is checked.

    Returns:
        bool: True if the Series can hold text values, otherwise False.
    """
    return (
        is_object_dtype(series)
        or is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def empty_mask(series):
    """Identify empty cells in a pandas Series.

    A cell is considered empty when it contains None, NaN, NaT, an
    empty string, or a string containing only spaces or tabs.

    The returned Boolean Series preserves the original index.

    Args:
        series (pandas.Series): The column whose empty cells are checked.

    Returns:
        pandas.Series: A Boolean mask containing True for empty cells
            and False for non-empty cells.

    Example:
        >>> import pandas as pd
        >>> empty_mask(pd.Series(["ok", "", "  ", None])).tolist()
        [False, True, True, True]
    """
    mask = series.isna()
    if _can_hold_text(series):
        blank = [isinstance(cell, str) and cell.strip() == "" for cell in series]
        mask = mask | pd.Series(blank, index=series.index, dtype=bool)
    return mask


def count_empty(series):
    """Count the number of empty cells in a pandas Series.

    Uses empty_mask() to apply the shared definition of an empty cell.

    Args:
        series (pandas.Series): The column whose empty cells are counted.

    Returns:
        int: The total number of empty cells.
    """
    return int(empty_mask(series).sum())


def drop_empty(series):
    """Return a pandas Series with all empty cells removed.

    Uses empty_mask() to identify missing values and blank strings.
    The original index of the remaining values is preserved.

    Args:
        series (pandas.Series): The column to clean.

    Returns:
        pandas.Series: A Series containing only non-empty values.
    """
    return series[~empty_mask(series)]


# ---------------------------------------------------------------------------
# Reading columns that are not what they look like
# ---------------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_THOUSANDS_PATTERN = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_LEADING_ZERO_PATTERN = re.compile(r"^[+-]?0\d")


def parse_number(text):
    """Convert a plain numerical string into a floating-point number.

    Accepts integers, decimals, scientific notation, surrounding
    whitespace, and correctly formatted thousands separators.

    Strings with leading-zero codes such as ``"007"`` are rejected
    because they may represent identifiers rather than quantities.
    Words such as ``"nan"`` and ``"inf"`` are also rejected.

    Args:
        text (str): The text containing a possible numerical value.

    Returns:
        float or None: The converted numerical value, or None if the
            text does not represent an accepted plain number.

    Example:
        >>> parse_number(" 84.5 ")
        84.5
        >>> parse_number("1,200")
        1200.0
        >>> parse_number("007") is None
        True
    """
    cleaned = text.strip()
    if _LEADING_ZERO_PATTERN.match(cleaned):
        return None
    if _THOUSANDS_PATTERN.match(cleaned):
        cleaned = cleaned.replace(",", "")
    if not _NUMBER_PATTERN.match(cleaned):
        return None
    return float(cleaned)


def text_to_numbers(series):
    """Convert a text-based numerical column into floating-point values.

    Every non-empty cell must contain a valid number. Empty cells are
    converted to NaN. The original Series index is preserved.

    The function returns None when the input is not a text-based Series,
    contains invalid values, or does not contain any numerical text.

    Args:
        series (pandas.Series): The text column to convert.

    Returns:
        pandas.Series or None: A floating-point Series with the same
            index, or None when conversion is not possible.

    Example:
        >>> import pandas as pd
        >>> text_to_numbers(pd.Series(["78", " 91", ""])).tolist()[:2]
        [78.0, 91.0]
        >>> text_to_numbers(pd.Series(["78", "high"])) is None
        True
    """
    if not (is_object_dtype(series) or is_string_dtype(series)):
        return None

    empty = empty_mask(series).to_numpy()
    numbers = np.full(len(series), np.nan)
    saw_text = False
    for position, cell in enumerate(series):
        if empty[position]:
            continue
        if isinstance(cell, str):
            number = parse_number(cell)
            if number is None:
                return None
            saw_text = True
        elif isinstance(cell, (int, float, np.integer, np.floating)) and not isinstance(
            cell, (bool, np.bool_)
        ):
            number = float(cell)
        else:
            return None
        numbers[position] = number

    if not saw_text:
        return None
    return pd.Series(numbers, index=series.index)


def mixes_numbers_and_words(series):
    """Check whether a column contains both numerical and word-like values.

    Numerical values include actual numbers and text that can be
    interpreted as numbers. Other non-empty values are treated as words.
    Empty cells are ignored during the check.

    Args:
        series (pandas.Series): The column to examine.

    Returns:
        bool: True when both numerical and word-like values are present;
            otherwise False.

    Example:
        >>> import pandas as pd
        >>> mixes_numbers_and_words(pd.Series(["7", "apple", None]))
        True
        >>> mixes_numbers_and_words(pd.Series(["pear", "apple"]))
        False
    """
    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False

    saw_number = saw_word = False
    for cell in drop_empty(series):
        if isinstance(cell, str):
            is_number = parse_number(cell) is not None
        else:
            is_number = isinstance(
                cell, (int, float, np.integer, np.floating)
            ) and not isinstance(cell, (bool, np.bool_))
        saw_number = saw_number or is_number
        saw_word = saw_word or not is_number
        if saw_number and saw_word:
            return True
    return False


def looks_like_id(name):
    """Determine whether a column name appears to represent an identifier.

    A name is considered an ID when ``id`` appears as its first or
    last complete word. Words are separated by underscores, spaces,
    hyphens, and capital letters.

    This recognizes names such as ``student_id``, ``StudentID``,
    ``id_number``, and ``student id``, but not names such as
    ``paid`` or ``identity``.

    Args:
        name (str): The column name to examine.

    Returns:
        bool: True if the name appears to represent an ID column;
            otherwise False.

    Example:
        >>> looks_like_id("student_id"), looks_like_id("StudentID")
        (True, True)
        >>> looks_like_id("paid")
        False
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    words = [word for word in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if word]
    return bool(words) and (words[0] == "id" or words[-1] == "id")

