
"""Tests for the public interface and initialization of the pandalogue package.

These tests verify that the package exposes its intended classes,
that its public exports are valid, and that its version matches
the version declared in setup.py.
"""

import pandalogue


def test_everything_can_be_imported_from_the_package():
    """Verify that all public classes can be imported from pandalogue.

    Also checks that the describer classes inherit from BaseDescriber
    and that Story and DataStory are available.
    """
    from pandalogue import (
        BaseDescriber,
        CategoryDescriber,
        DataStory,
        DateDescriber,
        MissingDescriber,
        NumberDescriber,
        RelationshipDescriber,
        RuleDescriber,
        Story,
    )

    for describer in (
        CategoryDescriber,
        DateDescriber,
        MissingDescriber,
        NumberDescriber,
        RelationshipDescriber,
        RuleDescriber,
    ):
        assert issubclass(describer, BaseDescriber)
    assert DataStory and Story


def test_all_lists_only_real_names():
     """Verify that every name listed in __all__ exists in the package."""
    for name in pandalogue.__all__:
        assert hasattr(pandalogue, name)


def test_version_matches_setup_py():
    """Verify that the package version matches the version in setup.py."""
    from pathlib import Path
    import re

    setup_text = (Path(__file__).parent.parent / "setup.py").read_text(encoding="utf-8")
    version = re.search(r'version="([^"]+)"', setup_text).group(1)
    assert pandalogue.__version__ == version
