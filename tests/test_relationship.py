import numpy as np
import pandas as pd
import pytest

from pandalogue.relationship import RelationshipDescriber


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "hours": [1, 2, 3, 4, 5],
        "grade": [60, 70, 80, 90, 100],
        "stress": [100, 80, 60, 40, 20]
    })


def test_positive_relationship(sample_df):
    obj = RelationshipDescriber("hours", "grade")

    assert obj.describe(sample_df) == [
        "hours and grade tend to go up together (a strong link)."
    ]

    assert np.isclose(obj.correlation(sample_df), 1.0)


def test_negative_relationship(sample_df):
    obj = RelationshipDescriber("hours", "stress")

    assert obj.describe(sample_df) == [
        "When hours goes up, stress tends to go down (a strong link)."
    ]

    assert np.isclose(obj.correlation(sample_df), -1.0)


def test_too_few_rows():
    df = pd.DataFrame({
        "hours": [1, 2, 3, 4],
        "grade": [60, 70, 80, 90]
    })

    obj = RelationshipDescriber("hours", "grade")

    assert obj.describe(df) == [
        "There are too few rows with both hours and grade filled in to compare them."
    ]

    assert obj.correlation(df) is None


def test_constant_first_column():
    df = pd.DataFrame({
        "hours": [5, 5, 5, 5, 5],
        "grade": [60, 70, 80, 90, 100]
    })

    obj = RelationshipDescriber("hours", "grade")

    assert obj.describe(df) == [
        "hours has the same value in every row, so it cannot be compared with grade."
    ]

    assert obj.correlation(df) is None


def test_constant_second_column():
    df = pd.DataFrame({
        "hours": [1, 2, 3, 4, 5],
        "grade": [80, 80, 80, 80, 80]
    })

    obj = RelationshipDescriber("hours", "grade")

    assert obj.describe(df) == [
        "grade has the same value in every row, so it cannot be compared with hours."
    ]

    assert obj.correlation(df) is None


def test_missing_values():
    df = pd.DataFrame({
        "hours": [1, 2, 3, 4, 5, np.nan],
        "grade": [60, 70, 80, 90, 100, 110]
    })

    obj = RelationshipDescriber("hours", "grade")

    assert np.isclose(obj.correlation(df), 1.0)


def test_infinite_values_are_excluded():
    df = pd.DataFrame({
        "hours": [1, 2, 3, 4, 5, np.inf, 7],
        "grade": [60, 70, 80, 90, 100, 110, -np.inf]
    })

    obj = RelationshipDescriber("hours", "grade")

    assert np.isclose(obj.correlation(df), 1.0)


def test_missing_first_column():
    df = pd.DataFrame({
        "grade": [60, 70, 80, 90, 100]
    })

    obj = RelationshipDescriber("hours", "grade")

    with pytest.raises(KeyError):
        obj.describe(df)


def test_missing_second_column():
    df = pd.DataFrame({
        "hours": [1, 2, 3, 4, 5]
    })

    obj = RelationshipDescriber("hours", "grade")

    with pytest.raises(KeyError):
        obj.describe(df)


def test_text_column():
    df = pd.DataFrame({
        "hours": ["one", "two", "three", "four", "five"],
        "grade": [60, 70, 80, 90, 100]
    })

    obj = RelationshipDescriber("hours", "grade")

    with pytest.raises(TypeError):
        obj.describe(df)


def test_boolean_column():
    df = pd.DataFrame({
        "hours": [True, False, True, False, True],
        "grade": [60, 70, 80, 90, 100]
    })

    obj = RelationshipDescriber("hours", "grade")

    with pytest.raises(TypeError):
        obj.describe(df)


def test_invalid_dataframe():
    obj = RelationshipDescriber("hours", "grade")

    with pytest.raises(TypeError):
        obj.describe("not a dataframe")


def test_invalid_constructor():
    with pytest.raises(ValueError):
        RelationshipDescriber("hours", "")

    with pytest.raises(ValueError):
        RelationshipDescriber("hours", None)

    with pytest.raises(ValueError):
        RelationshipDescriber("hours", "hours")


def test_strength_word():
    assert RelationshipDescriber.strength_word(0.8) == "strong"
    assert RelationshipDescriber.strength_word(0.5) == "moderate"
    assert RelationshipDescriber.strength_word(0.3) == "weak"
    assert RelationshipDescriber.strength_word(0.1) is None
    assert RelationshipDescriber.strength_word(None) is None


def test_equality_and_hashing():
    first = RelationshipDescriber("hours", "grade")
    second = RelationshipDescriber("hours", "grade")
    third = RelationshipDescriber("grade", "hours")

    assert first == second
    assert first != third
    assert hash(first) == hash(second)


def test_repr():
    obj = RelationshipDescriber("hours", "grade")

    assert repr(obj) == (
        "RelationshipDescriber(column='hours', other='grade')"
    )


def test_other_is_read_only():
    obj = RelationshipDescriber("hours", "grade")

    with pytest.raises(AttributeError):
        obj.other = "stress"
