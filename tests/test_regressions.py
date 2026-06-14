"""Regression tests, one per fixed GitHub issue.

Each fix lands with a test named `test_issue_<number>` that fails against the
unfixed code and passes after the fix.
"""

import yaml

from custom_components.chime_tts.helpers.services_helper import ChimeTTSServicesHelper


def test_issue_294_build_chime_options_coerces_values_to_str():
    """Numeric/boolean-looking chime names stay strings so services.yaml parses (#294)."""
    custom = [
        {"label": "060", "value": "/config/chimes/060.mp3"},
        {"label": 12, "value": 12},  # adversarial non-str entry
        {"label": "yes", "value": "yes"},
        {"value": "/missing/label.mp3"},  # malformed: no label, dropped
        {"label": "no-value", "value": None},  # None value, dropped (not "None")
        "not-a-dict",  # malformed, dropped
    ]
    options = ChimeTTSServicesHelper._build_chime_options(custom)

    assert options
    for option in options:
        assert isinstance(option["label"], str)
        assert isinstance(option["value"], str)

    values = [o["value"] for o in options]
    labels = [o["label"] for o in options]
    assert "12" in values, "int value should be coerced to str"
    assert "/missing/label.mp3" not in values, "entry without a label should be dropped"
    assert "no-value" not in labels, "entry with a None value should be dropped"
    assert "None" not in values, "None must not be coerced to the string 'None'"


def test_issue_294_round_trips_through_yaml_as_str():
    """Options survive a YAML dump/load with their values intact as str (#294)."""
    options = ChimeTTSServicesHelper._build_chime_options(
        [
            {"label": "060", "value": "/m/060.mp3"},
            {"label": "on", "value": "on"},
            {"label": "3.5", "value": "3.5"},
        ]
    )
    reloaded = yaml.safe_load(yaml.safe_dump({"options": options}))["options"]
    for option in reloaded:
        assert isinstance(option["label"], str)
        assert isinstance(option["value"], str)


def test_issue_294_stale_structure_returns_none_not_crash():
    """A services.yaml missing the expected nesting yields None rather than raising (#294)."""
    assert ChimeTTSServicesHelper._get_field_options({}, "say", "chime_path") is None
    assert (
        ChimeTTSServicesHelper._get_field_options(
            {"say": {"fields": {"chime_path": {"selector": {}}}}}, "say", "chime_path"
        )
        is None
    )
