"""Fixtures shared by the integration tests."""

import sys
import uuid

import pytest


@pytest.fixture
def lithops_localhost(tmp_path, monkeypatch):
    """Lithops on this machine, with a bucket of its own that is removed after."""
    bucket = f"test-{uuid.uuid4().hex[:8]}"
    config = tmp_path / "lithops.yaml"
    config.write_text(
        "lithops:\n  backend: localhost\n  storage: localhost\n  log_level: WARNING\n"
        f"localhost:\n  runtime: {sys.executable}\n  storage_bucket: {bucket}\n"
    )
    monkeypatch.setenv("LITHOPS_CONFIG_FILE", str(config))
    lithops = pytest.importorskip("lithops")
    yield bucket
    storage = lithops.Storage()
    for key in storage.list_keys(bucket, ""):
        storage.delete_object(bucket, key)
