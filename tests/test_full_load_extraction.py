"""Tests for normalising the national archive's layout.

The full-load archive downloaded from the live API is flat, with per-year
filenames — ``certificates-2008.csv`` through ``certificates-2026.csv``, plus the
same for recommendations, 38 entries in total. The rest of the pipeline expects
``domestic-<period>/<kind>.csv``, which is the shape the monthly archives have.

Verified against the real 8.26 GB archive. Without normalisation,
``period_from_name("domestic-full-load")`` returns None, so
``iter_extracted_sources`` skips the folder entirely and the pipeline silently
finds nothing.
"""

from __future__ import annotations

import zipfile

import pytest

from ew_housing_energy_impact.ingestion import epc_source


def _archive(path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, body in members.items():
            zf.writestr(name, body)


HEADER = "LMK_KEY,POSTCODE\n"


class TestLayoutNormalisation:
    def test_flat_per_year_members_become_period_folders(self, tmp_path):
        archive = tmp_path / "full.zip"
        _archive(
            archive,
            {
                "certificates-2008.csv": HEADER + "a,AA1\n",
                "certificates-2009.csv": HEADER + "b,BB2\n",
            },
        )
        dest = tmp_path / "extracted"

        found = epc_source.extract_full_load(archive=archive, dest_root=dest)

        assert (dest / "domestic-2008" / "certificates.csv").exists()
        assert (dest / "domestic-2009" / "certificates.csv").exists()
        assert [s.period for s in found] == ["2008", "2009"]
        assert {s.kind for s in found} == {"certificates"}

    def test_normalised_output_is_discoverable_by_the_existing_enumerator(self, tmp_path):
        """The whole point: downstream code must need no change."""
        archive = tmp_path / "full.zip"
        _archive(archive, {"certificates-2015.csv": HEADER + "x,XX1\n"})
        dest = tmp_path / "extracted"

        epc_source.extract_full_load(archive=archive, dest_root=dest)
        sources = epc_source.iter_extracted_sources(dest)

        assert [(s.period, s.kind) for s in sources] == [("2015", "certificates")]

    def test_recommendations_are_skipped_by_default(self, tmp_path):
        """69.7 GB against 27.4 GB, and nothing consumes them yet."""
        archive = tmp_path / "full.zip"
        _archive(
            archive,
            {
                "certificates-2020.csv": HEADER + "a,AA1\n",
                "recommendations-2020.csv": "LMK_KEY,IMPROVEMENT_ID\na,5\n",
            },
        )
        dest = tmp_path / "extracted"

        found = epc_source.extract_full_load(archive=archive, dest_root=dest)

        assert {s.kind for s in found} == {"certificates"}
        assert not (dest / "domestic-2020" / "recommendations.csv").exists()

    def test_recommendations_extracted_when_asked(self, tmp_path):
        archive = tmp_path / "full.zip"
        _archive(
            archive,
            {
                "certificates-2020.csv": HEADER + "a,AA1\n",
                "recommendations-2020.csv": "LMK_KEY,IMPROVEMENT_ID\na,5\n",
            },
        )
        dest = tmp_path / "extracted"

        found = epc_source.extract_full_load(
            archive=archive, kinds=("certificates", "recommendations"), dest_root=dest
        )

        assert {s.kind for s in found} == {"certificates", "recommendations"}
        assert (dest / "domestic-2020" / "recommendations.csv").exists()

    def test_unrecognised_members_are_ignored(self, tmp_path):
        archive = tmp_path / "full.zip"
        _archive(
            archive,
            {
                "certificates-2020.csv": HEADER + "a,AA1\n",
                "LICENCE.txt": "terms",
                "columns.csv": "x",
                "notes/readme.md": "hello",
            },
        )
        dest = tmp_path / "extracted"

        found = epc_source.extract_full_load(archive=archive, dest_root=dest)

        assert [s.period for s in found] == ["2020"]
        assert not (dest / "LICENCE.txt").exists()


class TestResumeAndFailure:
    def test_already_extracted_members_are_not_rewritten(self, tmp_path):
        """Re-running must be cheap; these members reach 6 GB each."""
        archive = tmp_path / "full.zip"
        _archive(archive, {"certificates-2020.csv": HEADER + "fresh,AA1\n"})
        dest = tmp_path / "extracted"

        epc_source.extract_full_load(archive=archive, dest_root=dest)
        target = dest / "domestic-2020" / "certificates.csv"
        target.write_text("SENTINEL — must not be overwritten\n")

        found = epc_source.extract_full_load(archive=archive, dest_root=dest)

        assert target.read_text().startswith("SENTINEL")
        assert [s.period for s in found] == ["2020"]

    def test_missing_archive_names_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope.zip"):
            epc_source.extract_full_load(archive=tmp_path / "nope.zip", dest_root=tmp_path)

    def test_no_part_files_survive_a_successful_run(self, tmp_path):
        archive = tmp_path / "full.zip"
        _archive(archive, {"certificates-2020.csv": HEADER + "a,AA1\n"})
        dest = tmp_path / "extracted"

        epc_source.extract_full_load(archive=archive, dest_root=dest)

        assert list(dest.rglob("*.part")) == []
