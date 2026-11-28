
import pandalogue


def test_version():
    assert pandalogue.__version__ == "0.1.0"


def test_public_classes():
    assert hasattr(pandalogue, "BaseDescriber")
    assert hasattr(pandalogue, "NumberDescriber")
    assert hasattr(pandalogue, "CategoryDescriber")
    assert hasattr(pandalogue, "MissingDescriber")
    assert hasattr(pandalogue, "DateDescriber")
    assert hasattr(pandalogue, "RelationshipDescriber")
    assert hasattr(pandalogue, "Story")
    assert hasattr(pandalogue, "DataStory")


def test_public_utility_functions():
    assert callable(pandalogue.format_percent)
    assert callable(pandalogue.format_number)
    assert callable(pandalogue.join_with_and)
    assert callable(pandalogue.count_text)
    assert callable(pandalogue.infinite_note)
    assert callable(pandalogue.empty_mask)
    assert callable(pandalogue.count_empty)
    assert callable(pandalogue.drop_empty)
    assert callable(pandalogue.parse_number)
    assert callable(pandalogue.text_to_numbers)
    assert callable(pandalogue.mixes_numbers_and_words)
    assert callable(pandalogue.looks_like_id)


def test_all_exports():
    expected = [
        "__version__",
        "BaseDescriber",
        "NumberDescriber",
        "CategoryDescriber",
        "MissingDescriber",
        "DateDescriber",
        "RelationshipDescriber",
        "Story",
        "DataStory",
        "format_percent",
        "format_number",
        "join_with_and",
        "count_text",
        "infinite_note",
        "empty_mask",
        "count_empty",
        "drop_empty",
        "parse_number",
        "text_to_numbers",
        "mixes_numbers_and_words",
        "looks_like_id"
    ]

    assert pandalogue.__all__ == expected

