"""Offline tests for CLI-level helpers that do not belong to one pipeline module."""

from __future__ import annotations

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
