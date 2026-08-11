"""Offline tests for the S3 uploader using a fake client."""

from __future__ import annotations

from pathlib import Path

import pytest

from ew_housing_energy_impact.ingestion import s3_upload
from ew_housing_energy_impact.ingestion.epc_source import ExtractedSource


class FakeClientError(Exception):
    """Stands in for botocore.exceptions.ClientError."""


class FakeS3:
    """Minimal in-memory S3 double."""

    def __init__(self, existing: dict[str, str] | None = None):
        self.existing = existing or {}
        self.uploads: list[tuple[str, str]] = []

    def head_object(self, Bucket: str, Key: str):
        if Key not in self.existing:
            raise FakeClientError()
        return {"Metadata": {"content-md5": self.existing[Key]}}

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        self.uploads.append((Bucket, Key))
        self.existing[Key] = ExtraArgs["Metadata"]["content-md5"]


@pytest.fixture(autouse=True)
def _patch_client_error(monkeypatch):
    monkeypatch.setattr(s3_upload, "ClientError", FakeClientError)


def _source(tmp_path: Path, kind: str = "certificates") -> ExtractedSource:
    p = tmp_path / f"{kind}.csv"
    p.write_text("LMK_KEY\n1\n")
    return ExtractedSource("2025-09", kind, p)


def test_s3_key_is_period_keyed_with_no_timestamp():
    assert s3_upload.s3_key("certificates", "2025-09") == (
        "raw/certificates/2025-09/certificates.csv"
    )
    assert s3_upload.s3_key("recommendations", "2024") == (
        "raw/recommendations/2024/recommendations.csv"
    )


def test_uploads_when_object_absent(tmp_path):
    client = FakeS3()
    assert s3_upload.upload_source(client, "b", _source(tmp_path)) is True
    assert client.uploads == [("b", "raw/certificates/2025-09/certificates.csv")]


def test_skips_when_hash_matches(tmp_path):
    src = _source(tmp_path)
    key = s3_upload.s3_key(src.kind, src.period)
    client = FakeS3(existing={key: s3_upload.local_md5(src.path)})

    assert s3_upload.upload_source(client, "b", src) is False
    assert client.uploads == []


def test_reuploads_when_content_changed(tmp_path):
    src = _source(tmp_path)
    key = s3_upload.s3_key(src.kind, src.period)
    client = FakeS3(existing={key: "stale-hash"})

    assert s3_upload.upload_source(client, "b", src) is True
    assert client.uploads == [("b", key)]


def test_upload_all_counts_both_outcomes(tmp_path):
    a = _source(tmp_path, "certificates")
    b = _source(tmp_path, "recommendations")
    key_a = s3_upload.s3_key(a.kind, a.period)
    client = FakeS3(existing={key_a: s3_upload.local_md5(a.path)})

    assert s3_upload.upload_all(client, "b", [a, b]) == {"uploaded": 1, "skipped": 1}
