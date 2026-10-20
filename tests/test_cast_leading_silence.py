"""Google Cast targets get leading silence so the receiver does not clip playback."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydub import AudioSegment

from custom_components.chime_tts.const import (
    AUDIO_DURATION_KEY,
    CAST_LEADING_SILENCE_KEY,
    CROSSFADE_KEY,
    DEFAULT_CAST_LEADING_SILENCE_MS,
    INITIAL_DELAY_KEY,
    OFFSET_KEY,
    TEMP_PATH_KEY,
    WWW_PATH_KEY,
)
from custom_components.chime_tts.helpers.helpers import ChimeTTSHelper

integration_module = importlib.import_module("custom_components.chime_tts.__init__")


class Hass:
    """Home Assistant stand-in that runs executor jobs inline."""

    def __init__(self, root: Path | None = None) -> None:
        """Expose config paths under root."""
        base = root or Path("/config")
        self.config = SimpleNamespace(
            path=lambda *parts: str(base.joinpath(*parts)), media_dirs={}
        )

    async def async_add_executor_job(self, func, *args):
        """Run executor jobs inline."""
        return func(*args)


class MediaPlayerHelper:
    """Resolve each target to a player on the given platform."""

    def __init__(self, platforms: dict[str, str]) -> None:
        """Map entity ids to their integration platform."""
        self.platforms = platforms

    def parse_entity_ids(self, data, hass) -> list[str]:
        """Return the configured targets."""
        del data, hass
        return list(self.platforms)

    async def async_initialize_media_players(self, hass, entity_ids, *args):
        """Return player doubles carrying a platform."""
        del hass, args
        return [
            SimpleNamespace(entity_id=entity_id, platform=self.platforms[entity_id])
            for entity_id in entity_ids
        ]


async def _parse(
    data: dict, platforms: dict[str, str], default_data: dict | None = None
) -> dict:
    params = await ChimeTTSHelper().async_parse_params(
        hass=Hass(),
        data={"message": "Hello", **data},
        is_say_url=False,
        media_player_helper=MediaPlayerHelper(platforms),
        default_data=default_data,
    )
    assert params is not None
    return params


@pytest.mark.asyncio
async def test_cast_target_gets_default_leading_silence() -> None:
    """A Cast target with no initial_delay gets the Cast default."""
    params = await _parse(
        {}, {"media_player.kitchen": "cast", "media_player.den": "squeezebox"}
    )

    assert params[CAST_LEADING_SILENCE_KEY] == DEFAULT_CAST_LEADING_SILENCE_MS
    assert params["initial_delay"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "default_data", "expected_initial_delay"),
    [
        ({INITIAL_DELAY_KEY: 0}, None, 0),
        ({INITIAL_DELAY_KEY: 300}, None, 300),
        ({}, {INITIAL_DELAY_KEY: 500}, 500),
    ],
)
async def test_initial_delay_replaces_the_cast_default(
    data: dict, default_data: dict | None, expected_initial_delay: float
) -> None:
    """An explicit or configured initial_delay keeps its own behavior on Cast."""
    params = await _parse(data, {"media_player.kitchen": "cast"}, default_data)

    assert params[CAST_LEADING_SILENCE_KEY] == 0
    assert params["initial_delay"] == expected_initial_delay


@pytest.mark.asyncio
async def test_non_cast_targets_get_no_leading_silence() -> None:
    """Calls without a Cast target are unchanged."""
    params = await _parse({}, {"media_player.den": "sonos"})

    assert params[CAST_LEADING_SILENCE_KEY] == 0


@pytest.mark.asyncio
async def test_say_url_gets_no_leading_silence() -> None:
    """say_url has no target players to pad for."""
    params = await ChimeTTSHelper().async_parse_params(
        hass=Hass(),
        data={"message": "Hello"},
        is_say_url=True,
        media_player_helper=MediaPlayerHelper({}),
    )

    assert params is not None
    assert params[CAST_LEADING_SILENCE_KEY] == 0


def test_cast_leading_silence_is_part_of_cache_key() -> None:
    """Padded Cast audio is cached apart from unpadded and initial_delay audio."""
    key = integration_module.get_filename_hash_from_service_data
    base = {"message": "hi", "repeat": 2}

    assert key({**base, CAST_LEADING_SILENCE_KEY: 2000}, {}) != key(base, {})
    assert key({**base, CAST_LEADING_SILENCE_KEY: 2000}, {}) != key(
        {**base, "initial_delay": 2000}, {}
    )
    # Non-Cast calls keep their existing cache keys.
    assert key({**base, CAST_LEADING_SILENCE_KEY: 0}, {}) == key(base, {})


@pytest.fixture
def generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Stub audio generation around a 100 ms segment and capture exports."""
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(
        integration_module,
        "_data",
        {
            OFFSET_KEY: 0,
            CROSSFADE_KEY: 0,
            TEMP_PATH_KEY: str(temp),
            WWW_PATH_KEY: str(tmp_path / "www"),
        },
    )
    segment = AudioSegment.silent(duration=100)
    fs = integration_module.filesystem_helper
    monkeypatch.setattr(
        fs, "async_get_chime_path_with_offset", AsyncMock(return_value=(None, None))
    )
    monkeypatch.setattr(
        fs, "async_save_audio_to_folder", AsyncMock(return_value=str(temp / "a.mp3"))
    )
    monkeypatch.setattr(fs, "async_load_audio", AsyncMock(return_value=segment))
    export = AsyncMock()
    monkeypatch.setattr(fs, "async_export_audio", export)
    helper = integration_module.media_player_helper
    monkeypatch.setattr(helper, "get_alexa_media_players_count", lambda: 0)
    monkeypatch.setattr(helper, "get_media_content_id", lambda *args, **kwargs: "id")
    monkeypatch.setattr(
        helper, "get_media_players_of_platform", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        integration_module, "async_get_audio_from_path", AsyncMock(return_value=segment)
    )
    monkeypatch.setattr(
        integration_module, "async_process_segments", AsyncMock(return_value=segment)
    )
    monkeypatch.setattr(
        integration_module, "validate_audio_dict", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(integration_module.tts_audio_helper, "_data", {})
    return export


def _playback_params(tmp_path: Path, **extra: object) -> dict:
    return {
        "hass": Hass(tmp_path),
        "message": "Hello",
        "chime_path": None,
        "end_chime_path": None,
        "cache": False,
        "entity_ids": ["media_player.kitchen"],
        **extra,
    }


@pytest.mark.asyncio
async def test_cast_leading_silence_is_added_once_after_repeat(
    tmp_path: Path, generation: AsyncMock
) -> None:
    """The pad precedes the first play only and counts toward playback time."""
    audio_dict = await integration_module.async_get_playback_audio_path(
        _playback_params(
            tmp_path, repeat=2, **{CAST_LEADING_SILENCE_KEY: 2000}
        ),
        {},
    )

    # 2000 ms pad + three 100 ms plays in the final written file.
    exported: AudioSegment = generation.await_args.args[0]
    assert len(exported) == 2300
    assert audio_dict[AUDIO_DURATION_KEY] == pytest.approx(2.3)


@pytest.mark.asyncio
async def test_non_cast_playback_audio_is_unchanged(
    tmp_path: Path, generation: AsyncMock
) -> None:
    """Without a Cast pad the audio is not re-exported or lengthened."""
    audio_dict = await integration_module.async_get_playback_audio_path(
        _playback_params(tmp_path, **{CAST_LEADING_SILENCE_KEY: 0}), {}
    )

    generation.assert_not_awaited()
    assert audio_dict[AUDIO_DURATION_KEY] == pytest.approx(0.1)
