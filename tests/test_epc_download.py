"""Offline tests for the migrated EPC download path.

The EPC service moved to GOV.UK. The old host 301-redirected to an HTML landing
page, so ``raise_for_status()`` saw a 200 and passed and ``.json()`` then died
with "Expecting value: line 1 column 1 (char 0)" — naming neither the redirect
nor the content type. These tests pin the behaviour that prevents a repeat:
check Content-Type before parsing, validate the transferred size, and never
cache a partial archive.

All tests use a fake requests layer. Nothing here touches the network.
"""

from __future__ import annotations

import zipfile

import pytest

from ew_housing_energy_impact.ingestion import epc_source


class FakeResponse:
    def __init__(
        self,
        *,
        status=200,
        content_type="application/json",
        payload=None,
        chunks=None,
        url=None,
        raise_http=False,
    ):
        self.status_code = status
        self.headers = {"Content-Type": content_type} if content_type else {}
        self._payload = payload
        self._chunks = chunks or []
        self.url = url or epc_source.DOMESTIC_CSV_INFO_ENDPOINT
        self._raise_http = raise_http

    def raise_for_status(self):
        if self._raise_http:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self._payload is None:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload

    def iter_content(self, chunk_size):
        del chunk_size
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _zip_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("certificates.csv", "LMK_KEY\n1\n")
    return buffer.getvalue()


class TestJsonGuard:
    def test_html_response_names_the_content_type_not_a_decode_error(self, monkeypatch):
        """The original failure: a 200 carrying HTML."""
        monkeypatch.setattr(
            epc_source.requests,
            "get",
            lambda *a, **k: FakeResponse(content_type="text/html;charset=utf-8", payload=None),
        )

        with pytest.raises(RuntimeError) as exc:
            epc_source.fetch_download_info({"Authorization": "Bearer x"})

        message = str(exc.value)
        assert "text/html" in message, f"error must name the content type: {message}"
        assert "Expecting value" not in message, "must not surface as a raw decode error"

    def test_redirect_is_called_out_as_a_moved_endpoint(self, monkeypatch):
        """A cross-host redirect is the signal the endpoint moved."""
        monkeypatch.setattr(
            epc_source.requests,
            "get",
            lambda *a, **k: FakeResponse(
                content_type="text/html", url="https://somewhere-else.example/landing"
            ),
        )

        with pytest.raises(RuntimeError, match="redirected"):
            epc_source.fetch_download_info({})

    def test_valid_json_passes_through(self, monkeypatch):
        payload = {"data": {"fileSize": 2923946932, "lastUpdated": "2025-08-01T00:31:19.000+00:00"}}
        monkeypatch.setattr(
            epc_source.requests, "get", lambda *a, **k: FakeResponse(payload=payload)
        )

        info = epc_source.fetch_download_info({})
        assert info["fileSize"] == 2923946932


class TestDownloadFullLoad:
    def test_writes_the_archive_and_reports_bytes(self, tmp_path, monkeypatch):
        body = _zip_bytes()
        monkeypatch.setattr(
            epc_source.requests,
            "get",
            lambda *a, **k: FakeResponse(content_type="application/zip", chunks=[body]),
        )
        dest = tmp_path / "archive.zip"

        message = epc_source.download_full_load({}, expected_bytes=len(body), dest=dest)

        assert dest.exists()
        assert dest.read_bytes() == body
        assert str(len(body)) in message.replace(",", "")

    def test_truncated_transfer_is_rejected_and_not_cached(self, tmp_path, monkeypatch):
        """A short read must not leave a file a later run treats as a valid cache."""
        body = _zip_bytes()
        monkeypatch.setattr(
            epc_source.requests,
            "get",
            lambda *a, **k: FakeResponse(content_type="application/zip", chunks=[body]),
        )
        dest = tmp_path / "archive.zip"

        with pytest.raises(RuntimeError, match="truncated"):
            epc_source.download_full_load({}, expected_bytes=len(body) + 5000, dest=dest)

        assert not dest.exists(), "a truncated download must not be cached"
        assert not dest.with_suffix(".zip.part").exists()

    def test_partial_file_is_cleaned_up_on_network_failure(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise OSError("connection reset")

        monkeypatch.setattr(epc_source.requests, "get", boom)
        dest = tmp_path / "archive.zip"

        with pytest.raises(OSError):
            epc_source.download_full_load({}, dest=dest)

        assert not dest.exists()
        assert list(tmp_path.glob("*.part")) == [], "temp file left behind"

    def test_valid_cache_is_reused_without_a_request(self, tmp_path, monkeypatch):
        dest = tmp_path / "archive.zip"
        dest.write_bytes(_zip_bytes())

        def should_not_be_called(*a, **k):
            raise AssertionError("network was called despite a valid cache")

        monkeypatch.setattr(epc_source.requests, "get", should_not_be_called)

        message = epc_source.download_full_load({}, dest=dest)
        assert "Skipped" in message

    def test_corrupt_cache_is_replaced(self, tmp_path, monkeypatch):
        dest = tmp_path / "archive.zip"
        dest.write_bytes(b"not a zip")
        body = _zip_bytes()
        monkeypatch.setattr(
            epc_source.requests,
            "get",
            lambda *a, **k: FakeResponse(content_type="application/zip", chunks=[body]),
        )

        epc_source.download_full_load({}, dest=dest)
        assert dest.read_bytes() == body


class TestRetiredPaths:
    def test_per_file_download_points_at_the_replacement(self):
        with pytest.raises(NotImplementedError, match="download_full_load"):
            epc_source.download_one_file(("anything", {}))

    def test_parallel_driver_points_at_the_replacement(self):
        with pytest.raises(NotImplementedError, match="download_full_load"):
            epc_source.download_all({}, ["a", "b"])


class TestEndpoints:
    def test_endpoints_target_the_new_service(self):
        for url in (epc_source.DOMESTIC_CSV_ENDPOINT, epc_source.DOMESTIC_CSV_INFO_ENDPOINT):
            assert "api.get-energy-performance-data.communities.gov.uk" in url
            assert "opendatacommunities" not in url
