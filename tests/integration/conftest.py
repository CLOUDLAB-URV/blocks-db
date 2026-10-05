"""Fixtures shared by the integration tests."""

import sys
import uuid

import pytest


def _write_config(path, bucket, lithops=None, localhost=None):
    sections = {
        "lithops": {"backend": "localhost", "storage": "localhost", "log_level": "WARNING", **(lithops or {})},
        "localhost": {"runtime": sys.executable, "storage_bucket": bucket, **(localhost or {})},
    }
    path.write_text("".join(
        f"{name}:\n" + "".join(f"  {key}: {value}\n" for key, value in settings.items())
        for name, settings in sections.items()
    ))


@pytest.fixture
def lithops_localhost(tmp_path, monkeypatch):
    """Lithops on this machine, with a bucket of its own that is removed after."""
    bucket = f"test-{uuid.uuid4().hex[:8]}"
    _write_config(tmp_path / "lithops.yaml", bucket)
    monkeypatch.setenv("LITHOPS_CONFIG_FILE", str(tmp_path / "lithops.yaml"))
    lithops = pytest.importorskip("lithops")
    yield bucket
    storage = lithops.Storage()
    for key in storage.list_keys(bucket, ""):
        storage.delete_object(bucket, key)


@pytest.fixture
def configure_lithops(lithops_localhost, tmp_path):
    """Rewrite the configuration of ``lithops_localhost`` with other settings
    of its ``lithops`` and ``localhost`` sections, such as an execution
    timeout or a runtime."""
    def configure(lithops=None, localhost=None):
        _write_config(tmp_path / "lithops.yaml", lithops_localhost, lithops, localhost)

    return configure
