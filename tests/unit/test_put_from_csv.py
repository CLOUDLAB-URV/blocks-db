"""put from a CSV file: a file of one row, one put per row, and per-row tags."""

import csv
import io
import json

import pytest

from vectordb import cli
from vectordb.utils import vector_tracking
from vectordb.utils.vector_tracking import VectorIndexTracker


class FakeS3:
    """Keeps the body of every pending file put writes."""

    def __init__(self):
        self.bodies = []

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.bodies.append(Body.decode())


class FakeTable:
    def put_item(self, Item):
        pass


@pytest.fixture
def put(tmp_path, monkeypatch):
    """Runs ``blocks-db put`` on a CSV file with the given text, through the
    real tracker over a fake bucket and table. Returns what it wrote, one
    list of rows per write, without the header."""
    storage = FakeS3()
    monkeypatch.setattr(vector_tracking, "s3", storage)

    class Client:
        def __init__(self, **kwargs):
            self.tracker = VectorIndexTracker("bucket", "us-east-1")
            self.tracker.table = FakeTable()

        def refuse_put_on_parquet(self, name):
            pass

    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "backend_config.json")
    monkeypatch.setattr(cli, "VectorDBClient", Client)

    def put(text, *options):
        source = tmp_path / "vectors.csv"
        source.write_text(text)
        monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "bucket", "put", "ds", str(source), *options])
        cli.main()
        return [list(csv.reader(io.StringIO(body)))[1:] for body in storage.bodies]

    return put


class TestAFileOfOneRow:
    def test_its_vector_is_put(self, put):
        assert put("7,0.1 0.2\n") == [[["7", "0.1 0.2"]]]

    def test_its_tags_are_kept(self, put):
        ((row,),) = put('7,0.1 0.2,{"source":"web"}\n')
        assert row[:2] == ["7", "0.1 0.2"]
        assert json.loads(row[2]) == {"source": "web"}


class TestSingle:
    def test_every_row_is_put_on_its_own(self, put):
        writes = put('1,0.1 0.2\n2,0.3 0.4,{"source":"api"}\n', "--single")
        assert [len(rows) for rows in writes] == [1, 1]
        assert writes[0][0] == ["1", "0.1 0.2"]
        assert writes[1][0][:2] == ["2", "0.3 0.4"]
        assert json.loads(writes[1][0][2]) == {"source": "api"}

    def test_tags_that_are_not_an_object_are_dropped(self, put):
        # as the loader behind the other put paths does
        assert put("1,0.1 0.2,[1]\n", "--single") == [[["1", "0.1 0.2"]]]
