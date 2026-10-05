"""Fixtures shared by every test. This conftest's directory also joins
sys.path, so both suites can import tests/helpers.py without touching
sys.path themselves."""

import pytest


@pytest.fixture(autouse=True)
def no_credential_lookup(monkeypatch):
    # boto3 resolves credentials when it builds a client; with none it asks
    # the EC2 metadata endpoint, for seconds per client. No test calls AWS
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
