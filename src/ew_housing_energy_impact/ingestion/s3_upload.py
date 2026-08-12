"""Upload extracted EPC sources to S3 at stable, period-keyed paths."""

from __future__ import annotations

import hashlib
from pathlib import Path

from botocore.exceptions import ClientError

from ew_housing_energy_impact.ingestion.epc_source import ExtractedSource

_READ_BLOCK = 8 << 20  # 8 MB
_MD5_METADATA_KEY = "content-md5"


def s3_key(kind: str, period: str) -> str:
    """Stable key for one period's source file.

    The period is part of the data's identity; a run timestamp is not. Keys
    therefore carry no timestamp, so re-running overwrites in place rather than
    accumulating one copy per run.
    """
    return f"raw/{kind}/{period}/{kind}.csv"


def local_md5(path: Path) -> str:
    """MD5 of a file, read in blocks so a 122 MB certificate file is not loaded whole."""
    digest = hashlib.md5()  # noqa: S324 - content addressing, not security
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


_ABSENT_CODES = {"404", "NoSuchKey"}


def _stored_md5(client, bucket: str, key: str) -> str | None:
    """Stored content-md5 metadata for `key`, or None if the object does not exist.

    Only a genuinely-absent object should read as "needs upload". A permission
    denial, throttle, or any other ClientError must be re-raised rather than
    treated the same as "absent" — silently swallowing it would degrade into
    re-uploading every file on every run, defeating the point of comparing
    hashes in the first place. head_object reports absence as "404", but the
    string form varies by botocore version, so "NoSuchKey" is accepted too.
    """
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in _ABSENT_CODES:
            return None
        raise
    return head.get("Metadata", {}).get(_MD5_METADATA_KEY)


def upload_source(client, bucket: str, source: ExtractedSource) -> bool:
    """Upload one source unless S3 already holds identical content.

    Hashes the file once and reuses the digest for both the comparison and the
    upload's metadata, rather than reading a ~122 MB certificate file twice.
    """
    key = s3_key(source.kind, source.period)
    digest = local_md5(source.path)
    if _stored_md5(client, bucket, key) == digest:
        return False
    client.upload_file(
        Filename=str(source.path),
        Bucket=bucket,
        Key=key,
        ExtraArgs={"Metadata": {_MD5_METADATA_KEY: digest}},
    )
    return True


def upload_all(client, bucket: str, sources: list[ExtractedSource]) -> dict[str, int]:
    """Upload every source, returning counts of uploaded and skipped files."""
    result = {"uploaded": 0, "skipped": 0}
    for source in sources:
        result["uploaded" if upload_source(client, bucket, source) else "skipped"] += 1
    return result
