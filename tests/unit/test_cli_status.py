"""What status prints: the counts of a parquet build come from its saved
configuration, and a CSV index prints what it always did."""

from types import SimpleNamespace

import pytest

from vectordb import cli


PARQUET_CONFIG = {
    "source_format": "parquet", "num_vectors": 15, "rejected": 1, "num_index": 2,
    "source_keys": ["s3://bucket/a.parquet", "s3://bucket/b.parquet"],
}


class FakeClient:
    """A dataset with one index, a pending file, and an id counter seeded
    with 16 source rows."""

    parquet = None  # the saved configuration of a parquet-built index

    def __init__(self, **kwargs):
        self.tracker = SimpleNamespace(get_pending_files=lambda name: ["pending/ds/1.csv"])

    def parquet_config(self, name, indexes=None):
        return self.parquet

    def list_indexes(self, name):
        return [("blocks", 2)]

    def has_pending_vectors(self, name):
        return True

    def get_indexed_count(self, name):
        return 16

    def get_indexed_ids(self, name):
        return set(range(16))


@pytest.fixture
def status(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "backend_config.json")
    monkeypatch.setattr(cli, "VectorDBClient", FakeClient)

    def status(*argv):
        monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "bucket", "status", "ds", *argv])
        cli.main()
        return capsys.readouterr().out

    return status


def test_a_parquet_index_reports_what_the_build_kept_and_rejected(status, monkeypatch):
    monkeypatch.setattr(FakeClient, "parquet", PARQUET_CONFIG)
    out = status()
    assert "  Source format: parquet\n" in out
    assert "  Indexed vectors: 15\n" in out and "  Rejected rows: 1\n" in out
    # neither the seeded counter nor a pending section, which it cannot have
    assert "16" not in out and "Pending" not in out


def test_verbose_adds_the_blocks_and_the_source_files_of_a_parquet_index(status, monkeypatch):
    monkeypatch.setattr(FakeClient, "parquet", PARQUET_CONFIG)
    out = status("-v")
    assert "  Blocks: 2\n" in out and "  Source files: 2\n" in out
    assert "Indexed IDs sample" not in out


def test_a_csv_index_prints_what_it_always_did(status):
    assert status() == (
        "\n=== Status for 'ds' ===\n\n"
        "Dataset: ds\n"
        "  Indexed: YES\n"
        "  Pending vectors: YES\n"
        "  Indexed vectors: 16\n"
        "  Pending files: 1\n"
    )
