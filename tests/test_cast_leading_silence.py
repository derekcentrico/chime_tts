"""Google Cast targets get leading silence so the receiver does not clip playback."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydub import AudioSegment

from custom_components.chime_tts.const import (
    ADD_COVER_ART_KEY,
    AUDIO_DURATION_KEY,
    CAST_LEADING_SILENCE_KEY,
    CROSSFADE_KEY,
    DEFAULT_CAST_LEADING_SILENCE_MS,
    INITIAL_DELAY_KEY,
    LOCAL_PATH_KEY,
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




class Generation:
    """Stubbed audio pipeline that records every file it writes."""

    def __init__(self, temp: Path) -> None:
        """Track saved, converted and loaded files in memory."""
        self.temp = temp
        self.files: dict[str, AudioSegment] = {}
        self.saved: list[tuple[str, AudioSegment]] = []
        self.converted: list[tuple[str, str]] = []
        self.export = AsyncMock()

    async def save(self, hass, audio: AudioSegment, folder: str) -> str:
        """Record the audio written to a folder."""
        del hass
        path = f"{folder}/{len(self.saved)}.mp3"
        self.files[path] = audio
        self.saved.append((folder, audio))
        return path

    async def load(self, path: str) -> AudioSegment:
        """Decode a previously written file."""
        return self.files[path]

    async def convert(self, hass, path: str, ffmpeg_args: str) -> str:
        """Pretend FFmpeg rewrote the file with new properties."""
        del hass
        new_path = f"{path}.converted.mp3"
        self.files[new_path] = self.files[path]
        self.converted.append((path, ffmpeg_args))
        return new_path


@pytest.fixture
def generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generation:
    """Stub audio generation around a 100 ms segment."""
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
    stub = Generation(temp)
    segment = AudioSegment.silent(duration=100)
    fs = integration_module.filesystem_helper
    monkeypatch.setattr(
        fs, "async_get_chime_path_with_offset", AsyncMock(return_value=(None, None))
    )
    monkeypatch.setattr(fs, "async_save_audio_to_folder", stub.save)
    monkeypatch.setattr(fs, "async_load_audio", stub.load)
    monkeypatch.setattr(fs, "async_export_audio", stub.export)
    monkeypatch.setattr(
        fs, "async_get_external_url", AsyncMock(side_effect=lambda hass, path: path)
    )
    monkeypatch.setattr(fs, "path_to_parent_folder", lambda folder: "/config")
    monkeypatch.setattr(fs, "path_exists", lambda path: True)
    monkeypatch.setattr(
        integration_module.helpers, "async_ffmpeg_convert_from_file", stub.convert
    )
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
    return stub


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
    tmp_path: Path, generation: Generation
) -> None:
    """The pad precedes the first play only and counts toward playback time."""
    audio_dict = await integration_module.async_get_playback_audio_path(
        _playback_params(tmp_path, repeat=2, **{CAST_LEADING_SILENCE_KEY: 2000}),
        {},
    )

    # 2000 ms pad + three 100 ms plays, written once and never re-exported.
    assert [len(audio) for _, audio in generation.saved] == [2300]
    generation.export.assert_not_awaited()
    assert audio_dict[AUDIO_DURATION_KEY] == pytest.approx(2.3)


@pytest.mark.asyncio
async def test_non_cast_playback_audio_is_unchanged(
    tmp_path: Path, generation: Generation
) -> None:
    """Without a Cast pad the audio is not lengthened."""
    audio_dict = await integration_module.async_get_playback_audio_path(
        _playback_params(tmp_path, **{CAST_LEADING_SILENCE_KEY: 0}), {}
    )

    assert [len(audio) for _, audio in generation.saved] == [100]
    generation.export.assert_not_awaited()
    assert audio_dict[AUDIO_DURATION_KEY] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_public_copy_for_alexa_carries_the_cast_pad(
    tmp_path: Path, generation: Generation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixed Cast and Alexa targets share one padded timeline."""
    monkeypatch.setattr(
        integration_module.media_player_helper, "get_alexa_media_players_count", lambda: 1
    )
    audio_dict = await integration_module.async_get_playback_audio_path(
        _playback_params(
            tmp_path,
            entity_ids=["media_player.kitchen", "media_player.echo"],
            repeat=1,
            **{CAST_LEADING_SILENCE_KEY: 2000},
        ),
        {},
    )

    folders = [folder for folder, _ in generation.saved]
    assert folders == [str(generation.temp), str(tmp_path / "www")]
    assert [len(audio) for _, audio in generation.saved] == [2200, 2200]
    assert audio_dict[AUDIO_DURATION_KEY] == pytest.approx(2.2)


@pytest.mark.asyncio
async def test_cast_pad_goes_through_cover_art_and_ffmpeg_conversion(
    tmp_path: Path, generation: Generation
) -> None:
    """Cover art and the configured codec apply to the padded file and stay in it."""
    integration_module._data[ADD_COVER_ART_KEY] = True
    audio_dict = await integration_module.async_get_playback_audio_path(
        _playback_params(
            tmp_path, ffmpeg_args="-b:a 48k", **{CAST_LEADING_SILENCE_KEY: 2000}
        ),
        {},
    )

    assert [len(audio) for _, audio in generation.saved] == [2100]
    assert len(generation.converted) == 2
    assert "cover_art.jpg" in generation.converted[0][1]
    assert generation.converted[1][1] == "-b:a 48k"
    # The converted file is what plays; nothing overwrites it afterwards.
    generation.export.assert_not_awaited()
    assert audio_dict[LOCAL_PATH_KEY].endswith(".converted.mp3.converted.mp3")
    assert audio_dict[AUDIO_DURATION_KEY] == pytest.approx(2.1)


@pytest.mark.parametrize(
    ("frame_rate", "channels", "sample_width"),
    [(48000, 2, 4), (8000, 1, 2)],
)
def test_leading_silence_keeps_the_audio_format(
    frame_rate: int, channels: int, sample_width: int
) -> None:
    """The pad matches the audio, so it is never resampled or remixed."""
    audio = (
        AudioSegment.silent(duration=100, frame_rate=frame_rate)
        .set_channels(channels)
        .set_sample_width(sample_width)
    )

    padded = integration_module._prepend_silence(audio, 2000)

    assert padded.frame_rate == frame_rate
    assert padded.channels == channels
    assert padded.sample_width == sample_width
    assert len(padded) == 2100
