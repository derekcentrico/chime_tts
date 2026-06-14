"""Regression tests, one per fixed GitHub issue.

Each fix lands with a test named `test_issue_<number>` that fails against the
unfixed code and passes after the fix.
"""

import yaml

from custom_components.chime_tts.helpers.helpers import ChimeTTSHelper
from custom_components.chime_tts.helpers.services_helper import ChimeTTSServicesHelper


class _FakeState:
    def __init__(self, entity_id):
        self.entity_id = entity_id


class _FakeStates:
    def __init__(self, entity_ids):
        self._entity_ids = entity_ids

    def async_all(self, *args):
        return [_FakeState(i) for i in self._entity_ids]


class _FakeServices:
    def __init__(self, services=()):
        self._services = set(services)

    def has_service(self, domain, service):
        return f"{domain}.{service}" in self._services


class _FakeHass:
    def __init__(self, entity_ids=(), services=()):
        self.states = _FakeStates(list(entity_ids))
        self.services = _FakeServices(services)
        self.data = {}


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


def test_issue_291_full_entity_id_matches_installed():
    """A full tts.* entity id selects that entity instead of being rejected (#291)."""
    helper = ChimeTTSHelper()
    hass = _FakeHass(entity_ids=["tts.piper", "tts.home_assistant_cloud"])
    assert helper.get_tts_platform(hass, tts_platform="tts.piper") == "tts.piper"
    # a bare provider name resolves to the matching entity too
    assert helper.get_tts_platform(hass, tts_platform="piper") == "tts.piper"


def test_issue_308_gemini_not_diverted_to_google_translate():
    """A Google Generative AI entity is selected, not swapped for Google Translate (#308)."""
    helper = ChimeTTSHelper()
    hass = _FakeHass(
        entity_ids=["tts.google_generative_ai_01jabc", "tts.google_translate_en_com"]
    )
    selected = helper.get_tts_platform(
        hass, tts_platform="tts.google_generative_ai_01jabc"
    )
    assert selected == "tts.google_generative_ai_01jabc"


def test_issue_241_installed_list_keeps_full_entity_ids():
    """Installed platforms keep full entity ids rather than first-token truncation (#241)."""
    helper = ChimeTTSHelper()
    hass = _FakeHass(entity_ids=["tts.google_generative_ai_x", "tts.microsoft_edge"])
    installed = helper.get_installed_tts_platforms(hass)
    assert "tts.google_generative_ai_x" in installed
    assert "google" not in installed


def test_google_translate_fallback_for_unmatched_google_request():
    """An unmatched Google request still falls back to an installed Google entity."""
    helper = ChimeTTSHelper()
    hass = _FakeHass(entity_ids=["tts.google_translate_en_com"])
    selected = helper.get_tts_platform(hass, tts_platform="tts.google_cloud")
    assert selected == "tts.google_translate_en_com"


def test_ambiguous_google_name_does_not_pick_generative_ai():
    """A bare 'google' must not silently resolve to a generative-AI entity."""
    helper = ChimeTTSHelper()
    hass = _FakeHass(
        entity_ids=["tts.google_generative_ai_01jabc", "tts.google_translate_en_com"]
    )
    # Bare "google" prefixes both providers, so it is ambiguous and falls through
    # to the Google Translate fallback rather than matching generative AI.
    assert (
        helper.get_tts_platform(hass, tts_platform="google")
        == "tts.google_translate_en_com"
    )
