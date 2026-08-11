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


def needs_upload(client, bucket: str, key: str, path: Path) -> bool:
    """True when the object is absent or its stored hash differs from the file."""
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return True
    return head.get("Metadata", {}).get(_MD5_METADATA_KEY) != local_md5(path)


def upload_source(client, bucket: str, source: ExtractedSource) -> bool:
    """Upload one source unless S3 already holds identical content."""
    key = s3_key(source.kind, source.period)
    if not needs_upload(client, bucket, key, source.path):
        return False
    client.upload_file(
        Filename=str(source.path),
        Bucket=bucket,
        Key=key,
        ExtraArgs={"Metadata": {_MD5_METADATA_KEY: local_md5(source.path)}},
    )
    return True


def upload_all(client, bucket: str, sources: list[ExtractedSource]) -> dict[str, int]:
    """Upload every source, returning counts of uploaded and skipped files."""
    result = {"uploaded": 0, "skipped": 0}
    for source in sources:
        result["uploaded" if upload_source(client, bucket, source) else "skipped"] += 1
    return result
