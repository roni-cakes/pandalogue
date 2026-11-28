"""The main package for pandalogue.

Pandalogue turns tabular data into short, plain-English summaries.
It provides describer classes for numerical, categorical, date,
missing-value, relationship, and rule-based analysis.

The package also provides Story and DataStory for organizing
and presenting data descriptions.

Public classes and the package version are exposed here so users
can import them directly from pandalogue.
"""

from .base import BaseDescriber
from .category import CategoryDescriber
from .date import DateDescriber
from .missing import MissingDescriber
from .number import NumberDescriber
from .relationship import RelationshipDescriber
from .rule import RuleDescriber
from .story import DataStory, Story

__version__ = "0.1.0"

__all__ = [
    "BaseDescriber",
    "CategoryDescriber",
    "DataStory",
    "DateDescriber",
    "MissingDescriber",
    "NumberDescriber",
    "RelationshipDescriber",
    "RuleDescriber",
    "Story",
    "__version__",
]


