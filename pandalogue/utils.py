
import re
import math


def format_percent(part, whole):
    if whole == 0:
        return "0%"

    percentage = (part / whole) * 100
    return f"{percentage:.1f}".rstrip("0").rstrip(".") + "%"


def format_number(value):
    number = float(value)

    if number == 0:
        return "0"

    if abs(number) < 0.01:
        decimals = max(2, 9 - math.floor(math.log10(abs(number))))
        return f"{number:,.{decimals}f}".rstrip("0").rstrip(".")

    return f"{number:,.2f}".rstrip("0").rstrip(".")


def join_with_and(items, limit=None):
    items = list(items)

    if not items:
        return ""

    if limit is not None and limit <= 0:
        return f"{len(items)} more"

    if limit is not None and len(items) > limit:
        remaining = len(items) - limit
        items = items[:limit]
        return ", ".join(items) + f" and {remaining} more"

    if len(items) == 1:
        return items[0]

    if len(items) == 2:
        return " and ".join(items)

    return ", ".join(items[:-1]) + " and " + items[-1]


def count_text(count, noun):
    if count == 1:
        return f"{count} {noun}"

    if (
        noun.endswith("y")
        and len(noun) > 1
        and noun[-2].lower() not in "aeiou"
    ):
        plural = noun[:-1] + "ies"

    elif noun.endswith(("s", "x", "z", "ch", "sh")):
        plural = noun + "es"

    else:
        plural = noun + "s"

    return f"{count} {plural}"


def infinite_note(count):
    if count == 0:
        return ""

    if count == 1:
        return "1 infinite value was left out."

    return f"{count} infinite values were left out."


def empty_mask(series):
    missing = series.isna()
    blank = series.astype("string").str.strip().eq("").fillna(False)

    return (missing | blank).astype(bool)


def count_empty(series):
    return int(empty_mask(series).sum())


def drop_empty(series):
    return series[~empty_mask(series)]


def parse_number(text):
    if not isinstance(text, str):
        return None

    text = text.strip()

    if not text:
        return None

    if re.fullmatch(r"[+-]?0\d+(?:\.\d+)?", text):
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def text_to_numbers(series):
    filled = drop_empty(series)

    if filled.empty:
        return None

    numbers = filled.map(parse_number)

    if numbers.isna().any():
        return None

    return numbers.astype(float)


def mixes_numbers_and_words(series):
    filled = drop_empty(series)

    has_number = filled.map(
        lambda value: parse_number(str(value)) is not None
    ).any()

    has_word = filled.map(
        lambda value: parse_number(str(value)) is None
    ).any()

    return bool(has_number and has_word)


def looks_like_id(name):
    if not isinstance(name, str):
        return False

    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    words = re.findall(r"[a-zA-Z0-9]+", name.lower())

    if not words:
        return False

    return words[0] == "id" or words[-1] == "id"
