
import numpy as np
import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_complex_dtype
)

from .base import BaseDescriber
from .utils import empty_mask


class RelationshipDescriber(BaseDescriber):

    MIN_ROWS = 5

    def __init__(self, column, other):
        super().__init__(column)

        if other is None or str(other).strip() == "":
            raise ValueError("other must be the name of a column, not empty.")

        if other == column:
            raise ValueError("column and other must be different.")

        self._other = other

    @property
    def other(self):
        return self._other

    def _get_paired_values(self, df):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        self._check_column(df)

        if self.other not in df.columns:
            raise KeyError(
                f"Column '{self.other}' was not found in the table."
            )

        first = df[self.column]
        second = df[self.other]

        for name, series in [
            (self.column, first),
            (self.other, second)
        ]:
            if (
                not is_numeric_dtype(series.dtype)
                or is_bool_dtype(series.dtype)
                or is_complex_dtype(series.dtype)
            ):
                raise TypeError(
                    f"Column '{name}' must contain real numbers."
                )

        valid = ~(empty_mask(first) | empty_mask(second))

        x = first[valid].astype(float).to_numpy()
        y = second[valid].astype(float).to_numpy()

        finite = np.isfinite(x) & np.isfinite(y)

        return x[finite], y[finite]

    def correlation(self, df):

        x, y = self._get_paired_values(df)

        if len(x) < self.MIN_ROWS:
            return None

        if np.all(x == x[0]) or np.all(y == y[0]):
            return None

        x_ranks = pd.Series(x).rank(method="average").to_numpy()
        y_ranks = pd.Series(y).rank(method="average").to_numpy()

        result = np.corrcoef(x_ranks, y_ranks)[0, 1]

        return float(result)

    @staticmethod
    def strength_word(correlation):

        if correlation is None:
            return None

        strength = abs(correlation)

        if strength >= 0.7:
            return "strong"
        elif strength >= 0.4:
            return "moderate"
        elif strength >= 0.2:
            return "weak"

        return None

    def describe(self, df):

        x, y = self._get_paired_values(df)

        a = self.column
        b = self.other

        if len(x) < self.MIN_ROWS:
            return [
                f"There are too few rows with both {a} and {b} "
                "filled in to compare them."
            ]

        if np.all(x == x[0]):
            return [
                f"{a} has the same value in every row, "
                f"so it cannot be compared with {b}."
            ]

        if np.all(y == y[0]):
            return [
                f"{b} has the same value in every row, "
                f"so it cannot be compared with {a}."
            ]

        x_ranks = pd.Series(x).rank(method="average").to_numpy()
        y_ranks = pd.Series(y).rank(method="average").to_numpy()

        correlation = float(
            np.corrcoef(x_ranks, y_ranks)[0, 1]
        )

        strength = self.strength_word(correlation)

        if strength is None:
            return [
                f"{a} and {b} do not show a clear link."
            ]

        if correlation > 0:
            return [
                f"{a} and {b} tend to go up together "
                f"(a {strength} link)."
            ]

        return [
            f"When {a} goes up, {b} tends to go down "
            f"(a {strength} link)."
        ]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(column={self.column!r}, other={self.other!r})"
        )

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented

        return (
            self.column == other.column
            and self.other == other.other
        )

    def __hash__(self):
        return hash((
            type(self).__name__,
            self.column,
            self.other
        ))

