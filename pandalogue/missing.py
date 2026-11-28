"""Describe missing and empty cells in a column in plain English.

This module provides MissingDescriber, which inspects any column in a
pandas DataFrame and reports how many cells are missing or blank.

It relies on the shared empty-cell definition in ``utils`` so that
``None``, ``NaN``, ``NaT``, and whitespace-only strings are counted
consistently across the package.
"""

from .base import BaseDescriber
from .utils import count_empty, count_text, format_percent


class MissingDescriber(BaseDescriber):
    """Describe how many cells in a column are empty.

    MissingDescriber inherits from BaseDescriber and works on any column
    data type. A cell is treated as empty when it contains a missing value
    (``None``, ``NaN``, ``NaT``) or blank text (an empty string or spaces).

    Attributes:
        column (str): The name of the column being described. Read only.

    Example:
        >>> import pandas as pd
        >>> from pandalogue.missing import MissingDescriber
        >>> df = pd.DataFrame({"grade": [62, None, 98, 85]})
        >>> MissingDescriber("grade").describe(df)
        ['The grade column has 1 empty cell out of 4 (25%).']
    """

    def describe(self, df):
        """Generate a plain-English sentence describing empty cells in the column.

        Args:
            df (pandas.DataFrame): DataFrame containing the column.

        Returns:
            list[str]: A list containing exactly one descriptive sentence.

        Raises:
            KeyError: If the requested column is missing from the DataFrame.
        """
        self._check_column(df)
        name = self._column
        total = len(df)
        if total == 0:
            return [f"The {name} column has no rows."]

        missing = count_empty(df[name])
        if missing == 0:
            return [f"No cells in {name} are empty."]
        if missing == total:
            return [f"Every cell in {name} is empty ({missing} of {total})."]

        return [
            f"The {name} column has {count_text(missing, 'empty cell')} "
            f"out of {total} ({format_percent(missing, total)})."
        ]
