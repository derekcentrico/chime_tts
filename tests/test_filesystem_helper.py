"""Tests for deterministic filesystem helper behaviour."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest
from pydub import AudioSegment

from custom_components.chime_tts.helpers.filesystem import FilesystemHelper


class Hass:
    """Minimal Home Assistant facade for filesystem helper tests."""

    def __init__(self, root: Path) -> None:
        """Expose config paths and execute executor work inline."""
        self.root = root
        self.config = SimpleNamespace(
            external_url="http://ha.local/",
            path=lambda *parts: str(root.joinpath(*parts)),
        )

    async def async_add_executor_job(self, func, *args):
        """Run executor jobs synchronously in the test process."""
        return func(*args)


def test_path_helpers_and_chime_discovery(tmp_path: Path) -> None:
    """Path resolution is case tolerant and custom audio choices are discovered."""
    helper = FilesystemHelper()
    folder = tmp_path / "Chimes"
    folder.mkdir()
    audio = folder / "Bell.mp3"
    audio.write_bytes(b"audio")
    uppercase_audio = folder / "Reminder.MP3"
    uppercase_audio.write_bytes(b"audio")
    (folder / "readme.txt").write_text("not audio")

    assert helper.path_exists(str(audio)) is True
    assert helper.path_exists(str(folder / "bell.mp3")) is True
    assert helper.path_exists("") is False
    assert helper.path_to_parent_folder("definitely-missing") is None
    assert (
        helper.get_downloaded_chime_path("/tmp", "https://host/a:b.mp3")
        == "/tmp/host_a_b.mp3"
    )
    assert helper.make_folder_path_safe("folder") == "/folder/"
    assert helper.make_folder_path_safe("/folder/") == "/folder/"
    assert helper.make_folder_path_safe("") == ""
    assert len(helper.get_hash_for_string("same")) == 64
    assert helper._get_chime_options_from_path(str(folder)) == [
        {"label": "Bell", "value": str(audio)},
        {"label": "Reminder", "value": str(uppercase_audio)},
    ]
    fingerprint = helper._get_chime_directory_fingerprint(str(folder))
    assert {entry[0] for entry in fingerprint} == {"Bell.mp3", "Reminder.MP3"}


@pytest.mark.asyncio
async def test_async_path_url_and_file_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async wrappers create, copy, validate, expose, and delete local files."""
    helper = FilesystemHelper()
    hass = Hass(tmp_path)
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")
    target_dir = tmp_path / "target"
    public_dir = tmp_path / "www"
    public_dir.mkdir()
    public_file = public_dir / "sound.mp3"
    public_file.write_bytes(b"audio")

    assert await helper.async_validate_path(hass, str(source)) == str(source)
    queried_paths: list[str] = []
    with monkeypatch.context() as context:
        context.setattr(
            helper, "path_exists", lambda path: queried_paths.append(path) or True
        )
        assert (
            await helper.async_validate_path(hass, "/config/www/sound.mp3")
            == "/config/www/sound.mp3"
        )
    assert queried_paths == ["/config/www/sound.mp3"]
    assert await helper.async_create_folder(hass, str(target_dir)) is True
    assert await helper.async_copy_file(hass, str(source), str(target_dir)) == str(
        target_dir / "source.mp3"
    )
    assert await helper.async_copy_file(hass, str(source), "") is None
    assert (
        await helper.async_file_exists_in_directory(str(public_file), str(public_dir))
        is True
    )
    assert await helper.async_get_external_url(hass, str(public_file)) == (
        f"http://ha.local/{public_file}".replace("www/", "local/").replace(
            "http://ha.local//", "http://ha.local/"
        )
    )
    assert helper.get_local_path(hass, str(source)) == str(source)
    assert await helper.async_get_local_path(hass, str(source)) == str(source)
    assert helper.filepath_exists_locally(hass, str(source)) is True
    helper.delete_file(hass, str(source))
    assert source.exists() is False

    monkeypatch.setattr(helper, "async_load_audio", AsyncMock(return_value=None))
    monkeypatch.setattr(helper, "_async_fetch_chime", AsyncMock(return_value=b"x"))
    assert (
        await helper.async_download_file(
            hass, "https://example.com/sound.mp3", str(target_dir)
        )
        is None
    )


@pytest.mark.asyncio
async def test_chime_path_resolution_for_custom_url_and_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Custom chimes, URL cache, and Random Chime Sets resolve without recursion."""
    helper = FilesystemHelper()
    hass = Hass(tmp_path)
    custom = tmp_path / "tone.mp3"
    custom.write_bytes(b"audio")
    monkeypatch.setattr(
        helper,
        "async_get_chime_options_from_path",
        AsyncMock(return_value=[{"label": "Tone", "value": str(custom)}]),
    )
    data = {"custom_chimes_path": str(tmp_path)}
    assert await helper.async_get_chime_path("tone.wav", False, data, hass) == str(
        custom
    )

    cached = tmp_path / "cached.mp3"
    cached.write_bytes(b"audio")
    monkeypatch.setattr(
        helper, "get_downloaded_chime_path", lambda **kwargs: str(cached)
    )
    assert await helper.async_get_chime_path(
        "https://example.com/a.mp3", True, data, hass
    ) == str(cached)

    set_data = {
        "chime_sets": [{"id": "set", "name": "Set", "chimes": ["tone"]}],
        "custom_chimes_path": str(tmp_path),
    }
    assert await helper.async_get_chime_path_with_offset(
        "Set", False, set_data, hass
    ) == (str(custom), None)


class RecordingHass(Hass):
    """Hass facade that records which callables were sent to the executor."""

    def __init__(self, root: Path) -> None:
        """Start with an empty executor log."""
        super().__init__(root)
        self.executor_calls: list = []

    async def async_add_executor_job(self, func, *args):
        """Record the callable, then run it inline."""
        self.executor_calls.append(func)
        return func(*args)


@pytest.mark.asyncio
async def test_folder_creation_and_copy_run_in_executor(tmp_path: Path) -> None:
    """Blocking mkdir, copy and remove calls go through the executor."""
    helper = FilesystemHelper()
    hass = RecordingHass(tmp_path)
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")
    target_dir = tmp_path / "www" / "chime-tts"

    copied = await helper.async_copy_file(hass, str(source), str(target_dir))

    assert copied == str(target_dir / "source.mp3")
    assert (target_dir / "source.mp3").read_bytes() == b"audio"
    assert os.makedirs in hass.executor_calls
    assert shutil.copy in hass.executor_calls

    existing = tmp_path / "existing.mp3"
    existing.write_bytes(b"old")
    helper.async_export_audio = AsyncMock()
    helper.get_downloaded_chime_path = lambda url, folder: str(existing)
    await helper.async_save_audio_to_folder(
        hass, None, str(tmp_path), file_name="https://example.com/existing.mp3"
    )

    assert os.remove in hass.executor_calls
    assert existing.exists() is False


_DOWNLOAD_LIMIT = 1024


async def _audio(request: web.Request) -> web.Response:
    """Serve a small body with the content type named in the query string."""
    content_type = request.query.get("type", "audio/mpeg")
    return web.Response(body=b"chime", headers={"Content-Type": content_type})


async def _large(request: web.Request) -> web.Response:
    """Serve a body over the limit with an accurate Content-Length."""
    return web.Response(body=b"x" * (_DOWNLOAD_LIMIT * 2), content_type="audio/mpeg")


async def _large_chunked(request: web.Request) -> web.StreamResponse:
    """Stream a body over the limit without a Content-Length."""
    response = web.StreamResponse(headers={"Content-Type": "audio/mpeg"})
    response.enable_chunked_encoding()
    await response.prepare(request)
    for _ in range(8):
        await response.write(b"x" * (_DOWNLOAD_LIMIT // 2))
    await response.write_eof()
    return response


async def _slow(request: web.Request) -> web.StreamResponse:
    """Trickle a body one byte at a time so only a total timeout can stop it."""
    response = web.StreamResponse(headers={"Content-Type": "audio/mpeg"})
    response.enable_chunked_encoding()
    await response.prepare(request)
    for _ in range(20):
        await response.write(b"x")
        await asyncio.sleep(0.1)
    await response.write_eof()
    return response


async def _redirect(request: web.Request) -> web.Response:
    """Redirect through a chain of hops before serving audio."""
    hops = int(request.match_info["hops"])
    if hops <= 0:
        raise web.HTTPFound("/audio")
    raise web.HTTPFound(f"/redirect/{hops - 1}")


async def _missing(request: web.Request) -> web.Response:
    """Return a 404."""
    raise web.HTTPNotFound


@pytest.fixture
async def chime_server(socket_enabled: None) -> AsyncIterator[TestServer]:
    """Run a local HTTP server that serves chime responses."""
    app = web.Application()
    app.router.add_get("/audio", _audio)
    app.router.add_get("/large", _large)
    app.router.add_get("/large-chunked", _large_chunked)
    app.router.add_get("/slow", _slow)
    app.router.add_get("/redirect/{hops}", _redirect)
    app.router.add_get("/missing", _missing)
    server = TestServer(app)
    await server.start_server()
    yield server
    await server.close()


class ThreadedHass(Hass):
    """Facade whose executor jobs run off the loop, as in Home Assistant."""

    async def async_add_executor_job(self, func, *args):
        """Run executor jobs in a worker thread."""
        return await asyncio.to_thread(func, *args)


@pytest.fixture
async def download_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[FilesystemHelper, Hass, list[bytes]]]:
    """Filesystem helper on a real aiohttp session with pydub stubbed out."""
    module = importlib.import_module("custom_components.chime_tts.helpers.filesystem")
    # The aiodns resolver leaves a pycares thread behind, which HA's cleanup check rejects
    session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
    )
    monkeypatch.setattr(
        module, "async_get_clientsession", lambda hass: session, raising=False
    )
    monkeypatch.setattr(
        module, "CHIME_DOWNLOAD_MAX_BYTES", _DOWNLOAD_LIMIT, raising=False
    )
    monkeypatch.setattr(module, "CHIME_DOWNLOAD_TIMEOUT", 0.5, raising=False)

    loaded: list[bytes] = []

    async def load_audio(file) -> AudioSegment:
        loaded.append(file.getvalue())
        return AudioSegment.silent(duration=500)

    helper = FilesystemHelper()
    monkeypatch.setattr(helper, "async_load_audio", load_audio)
    monkeypatch.setattr(
        helper,
        "async_save_audio_to_folder",
        AsyncMock(return_value=str(tmp_path / "chime.mp3")),
    )
    yield helper, ThreadedHass(tmp_path), loaded
    await session.close()


async def test_download_chime_success(
    chime_server: TestServer, download_helper, tmp_path: Path
) -> None:
    """An audio response is loaded, saved, and returned with its duration."""
    helper, hass, loaded = download_helper
    result = await helper.async_download_file(
        hass, str(chime_server.make_url("/audio")), str(tmp_path)
    )
    assert result == {
        "local_path": str(tmp_path / "chime.mp3"),
        "audio_duration": 0.5,
    }
    assert loaded == [b"chime"]


@pytest.mark.parametrize(
    "content_type",
    ["application/octet-stream", "AUDIO/MPEG", "Audio/Ogg; codecs=opus", "application/x-audio"],
)
async def test_download_chime_accepts_audio_content_types(
    chime_server: TestServer, download_helper, tmp_path: Path, content_type: str
) -> None:
    """octet-stream and audio types are accepted regardless of case."""
    helper, hass, loaded = download_helper
    url = chime_server.make_url("/audio").with_query(type=content_type)
    assert await helper.async_download_file(hass, str(url), str(tmp_path)) is not None
    assert loaded == [b"chime"]


async def test_download_chime_rejects_non_audio(
    chime_server: TestServer, download_helper, tmp_path: Path
) -> None:
    """A non-audio response is not decoded."""
    helper, hass, loaded = download_helper
    url = chime_server.make_url("/audio").with_query(type="text/html")
    assert await helper.async_download_file(hass, str(url), str(tmp_path)) is None
    assert loaded == []


async def test_download_chime_rejects_http_error(
    chime_server: TestServer, download_helper, tmp_path: Path
) -> None:
    """A non-2xx status is reported and nothing is decoded."""
    helper, hass, loaded = download_helper
    url = str(chime_server.make_url("/missing"))
    assert await helper.async_download_file(hass, url, str(tmp_path)) is None
    assert loaded == []


async def test_download_chime_rejects_oversize_content_length(
    chime_server: TestServer,
    download_helper,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A Content-Length over the cap is rejected before the body is read."""
    helper, hass, loaded = download_helper
    url = str(chime_server.make_url("/large"))
    with caplog.at_level(logging.WARNING):
        assert await helper.async_download_file(hass, url, str(tmp_path)) is None
    assert loaded == []
    assert f"is {_DOWNLOAD_LIMIT * 2} bytes" in caplog.text


async def test_download_chime_rejects_oversize_stream(
    chime_server: TestServer,
    download_helper,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A chunked body that grows past the cap is abandoned mid-stream."""
    helper, hass, loaded = download_helper
    url = str(chime_server.make_url("/large-chunked"))
    with caplog.at_level(logging.WARNING):
        assert await helper.async_download_file(hass, url, str(tmp_path)) is None
    assert loaded == []
    assert "exceeded the" in caplog.text


async def test_download_chime_total_timeout(
    chime_server: TestServer,
    download_helper,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A server that keeps trickling bytes is cut off by the total timeout."""
    helper, hass, loaded = download_helper
    url = str(chime_server.make_url("/slow"))
    loop = asyncio.get_running_loop()
    started = loop.time()
    with caplog.at_level(logging.WARNING):
        assert await helper.async_download_file(hass, url, str(tmp_path)) is None
    assert loop.time() - started < 1.5
    assert loaded == []
    assert "Timeout Error" in caplog.text


async def test_download_chime_follows_redirects(
    chime_server: TestServer, download_helper, tmp_path: Path
) -> None:
    """Redirects within the limit are followed to the audio."""
    helper, hass, loaded = download_helper
    url = str(chime_server.make_url("/redirect/2"))
    assert await helper.async_download_file(hass, url, str(tmp_path)) is not None
    assert loaded == [b"chime"]


async def test_download_chime_rejects_too_many_redirects(
    chime_server: TestServer, download_helper, tmp_path: Path
) -> None:
    """A redirect chain longer than the limit is abandoned."""
    helper, hass, loaded = download_helper
    url = str(chime_server.make_url("/redirect/6"))
    assert await helper.async_download_file(hass, url, str(tmp_path)) is None
    assert loaded == []

