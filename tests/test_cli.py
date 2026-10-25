"""Offline tests for CLI-level helpers that do not belong to one pipeline module."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_require_snowflake_credentials_raises_when_missing(monkeypatch):
    from ew_housing_energy_impact import cli

    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    monkeypatch.delenv("SNOWFLAKE_USER", raising=False)
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "hunter2")

    with pytest.raises(ValueError, match="SNOWFLAKE_ACCOUNT.*SNOWFLAKE_USER"):
        cli._require_snowflake_credentials()


def test_require_snowflake_credentials_returns_connect_kwargs(monkeypatch):
    from ew_housing_energy_impact import cli

    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct123")
    monkeypatch.setenv("SNOWFLAKE_USER", "alice")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "hunter2")

    assert cli._require_snowflake_credentials() == {
        "account": "acct123",
        "user": "alice",
        "password": "hunter2",
    }


def test_train_routes_to_property_benchmark(monkeypatch, tmp_path):
    from ew_housing_energy_impact import cli
    from scripts.pipeline import benchmark_release

    calls = []
    monkeypatch.setattr(benchmark_release, "run_benchmark", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        "sys.argv",
        [
            "ewhei",
            "train",
            "--release",
            "data/releases/example",
            "--output",
            str(tmp_path / "new-run"),
            "--strategy",
            "chronological",
        ],
    )
    cli.main()
    assert calls == [
        {
            "release": Path("data/releases/example"),
            "output": tmp_path / "new-run",
            "strategy": "chronological",
            "train_cap": 200000,
            "validation_cap": 50000,
            "bootstrap": 500,
        }
    ]


def test_train_requires_explicit_versioned_inputs(monkeypatch):
    from ew_housing_energy_impact import cli

    monkeypatch.setattr("sys.argv", ["ewhei", "train"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
