"""Describe the relationship between two numerical columns.

This module provides RelationshipDescriber, which compares two numerical
columns in a pandas DataFrame using Spearman rank correlation.

It skips rows containing missing or infinite values and produces a
plain-English description of whether the columns tend to move together.
It describes association only and does not imply causation.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .base import BaseDescriber


class RelationshipDescriber(BaseDescriber):
    """Describe how two numerical columns are related.

    RelationshipDescriber inherits from BaseDescriber. The inherited
    ``column`` attribute identifies the first column, while ``other``
    identifies the second column.

    The class uses Spearman rank correlation to measure whether the
    two columns tend to increase or decrease together. This method
    compares the ranks of values rather than their original magnitudes.

    Missing and infinite values are excluded before the comparison.
    At least ``MIN_ROWS`` usable rows are required. If either column
    has only one unique value, a correlation cannot be calculated.

    The descriptions explain association between columns, not
    cause-and-effect relationships.

    Attributes:
        column (str): The name of the first column. Read only.
        other (str): The name of the second column. Read only.

    Example:
        >>> import pandas as pd
        >>> from pandalogue.relationship import RelationshipDescriber
        >>> df = pd.DataFrame({
        ...     "hours": [1, 2, 3, 4, 5],
        ...     "grade": [50, 60, 80, 90, 95]
        ... })
        >>> describer = RelationshipDescriber("hours", "grade")
        >>> describer.describe(df)
        ['hours and grade tend to go up together (a strong link).']
    """

    MIN_ROWS = 5
    """int: Minimum number of usable row pairs required for comparison."""

    def __init__(self, column, other):
        """Initialize a describer for two different columns.

        Args:
            column (str): Name of the first numerical column.
            other (str): Name of the second numerical column.

        Raises:
            ValueError: If either column name is empty or both names
                identify the same column.
        """
        super().__init__(column)
        if other is None or str(other).strip() == "":
            raise ValueError("other must be the name of a column, not empty.")
        if other == column:
            raise ValueError("A relationship needs two different columns.")
        self._other = other

    @property
    def other(self):
        """str: Return the name of the second column.

        This property is read only. The second column name is stored
        internally in ``_other``.
        """
        return self._other

    @staticmethod
    def strength_word(correlation):
        """Classify the strength of a correlation.

        The absolute value is used so positive and negative correlations
        with the same magnitude receive the same strength description.

        Args:
            correlation (float): Spearman correlation value, normally
                between -1 and 1.

        Returns:
            str or None: ``"strong"`` for an absolute value of at least
            0.7, ``"moderate"`` for at least 0.4, ``"weak"`` for at
            least 0.2, or None when the value is below 0.2.
        """
        size = abs(correlation)
        if size >= 0.7:
            return "strong"
        if size >= 0.4:
            return "moderate"
        if size >= 0.2:
            return "weak"
        return None

    def correlation(self, df):
        """Calculate the Spearman correlation between the two columns.

        The calculation uses only rows where both columns contain finite
        numerical values.

        Args:
            df (pandas.DataFrame): DataFrame containing both columns.

        Returns:
            float or None: Spearman correlation between -1 and 1.
            Returns None when there are too few usable rows or either
            column has no variation.

        Raises:
            KeyError: If either requested column is missing.
            TypeError: If either column contains non-numerical or Boolean
                data.
        """
        return self._measure(df)[0]

    def describe(self, df):
        """Generate a plain-English description of the relationship.

        The description identifies whether the columns tend to increase
        together, move in opposite directions, or show no clear link.
        It does not claim that one column causes changes in the other.

        Args:
            df (pandas.DataFrame): DataFrame containing both columns.

        Returns:
            list[str]: A list containing exactly one descriptive sentence.

        Raises:
            KeyError: If either requested column is missing.
            TypeError: If either column contains non-numerical or Boolean
                data.
        """
        correlation, problem = self._measure(df)
        if problem is not None:
            return [problem]

        first, second = self._column, self._other
        word = self.strength_word(correlation)
        if word is None:
            return [f"{first} and {second} do not show a clear link."]
        if correlation > 0:
            return [f"{first} and {second} tend to go up together (a {word} link)."]
        return [f"When {first} goes up, {second} tends to go down (a {word} link)."]

    def _measure(self, df):
        """Prepare the columns and calculate their Spearman correlation.

        The method checks that both columns exist and contain numerical,
        non-Boolean data. It then removes rows with missing or infinite
        values, checks the minimum row requirement, and verifies that
        both columns contain more than one unique value.

        Args:
            df (pandas.DataFrame): DataFrame containing both columns.

        Returns:
            tuple: A pair containing either:
                - The calculated correlation and None, or
                - None and a sentence explaining why comparison is not
                  possible.

        Raises:
            KeyError: If either requested column is missing.
            TypeError: If the input columns are not numerical or are
                Boolean.
        """
        self._check_column(df)
        if self._other not in df.columns:
            raise KeyError(f"Column '{self._other}' was not found in the table.")
        for name in (self._column, self._other):
            values = df[name]
            if is_bool_dtype(values) or not is_numeric_dtype(values):
                raise TypeError(f"Column '{name}' does not hold numbers.")

        x = df[self._column].astype(float).to_numpy()
        y = df[self._other].astype(float).to_numpy()
        usable = np.isfinite(x) & np.isfinite(y)
        x, y = x[usable], y[usable]
        first, second = self._column, self._other

        if len(x) < self.MIN_ROWS:
            return None, (
                f"There are too few rows with both {first} and {second} "
                "filled in to compare them."
            )
        for name, values in ((first, x), (second, y)):
            if len(set(values.tolist())) < 2:
                other = second if name == first else first
                return None, (
                    f"{name} has the same value in every row, "
                    f"so it cannot be compared with {other}."
                )

        correlation = pd.Series(x).corr(pd.Series(y), method="spearman")
        return float(correlation), None

    def __repr__(self):
        """Return an informative representation of the describer.

        Returns:
            str: A string showing the class name and both column names.

        Example:
            >>> repr(RelationshipDescriber("hours", "grade"))
            "RelationshipDescriber(column='hours', other='grade')"
        """
        return (
            f"{self.__class__.__name__}"
            f"(column={self._column!r}, other={self._other!r})"
        )

    def __eq__(self, other):
        """Compare two describers using their class and column names.

        Two RelationshipDescriber objects are equal when they refer
        to the same first and second columns in the same order.

        Args:
            other (object): The object to compare with this describer.

        Returns:
            bool or NotImplemented: True when both column names match
            in the same order, False when they do not, or NotImplemented
            for objects of another type.
        """
        if not isinstance(other, RelationshipDescriber):
            return NotImplemented
        return self._column == other._column and self._other == other._other

    def __hash__(self):
        """Return a hash based on the describer's class and column names.

        Returns:
            int: Hash value that allows instances to be used in sets
            and as dictionary keys.
        """
        return hash((type(self).__name__, self._column, self._other))
