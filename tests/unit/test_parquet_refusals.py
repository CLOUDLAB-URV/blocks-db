"""An index built from parquet refuses the CSV-path features it cannot
serve, and says why, instead of answering with nothing or failing in S3."""

from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from vectordb import cli
from vectordb import client as client_module
from vectordb.client import NotAvailableOnParquet, VectorDBClient
from vectordb.utils import dataset_ops, index_ops


def client_for(monkeypatch, source_format):
    """A client whose dataset ``ds`` has one index built from
    ``source_format``: "parquet", "csv" (a config from before the parquet
    path, without the key), or None for no index at all."""
    client = object.__new__(VectorDBClient)
    client.bucket = "bucket"
    client.wait_timeout = None
    client.tracker = SimpleNamespace(
        table_name="table",
        dynamodb=SimpleNamespace(meta=SimpleNamespace(client=SimpleNamespace(meta=SimpleNamespace(region_name="region")))),
    )
    indexes = [] if source_format is None else [("blocks", 4)]
    config = {"num_index": 4} if source_format == "csv" else {"num_index": 4, "source_format": source_format}
    monkeypatch.setattr(client, "list_indexes", lambda name: indexes)
    monkeypatch.setattr(client_module, "load_index_config", lambda *args: dict(config))
    return client


class TestClient:
    @pytest.mark.parametrize("call", [
        lambda c: c.get_vectors("ds", [1]),
        lambda c: c.list_vectors("ds", 10),
        lambda c: c.list_vectors_paginated("ds", 0, 10),
        lambda c: c.get_vector_ids_by_tags("ds", {"lang": "es"}),
    ], ids=["get_vectors", "list_vectors", "list_vectors_paginated", "get_vector_ids_by_tags"])
    def test_csv_path_reads_are_refused_by_name(self, monkeypatch, call):
        client = client_for(monkeypatch, "parquet")
        with pytest.raises(NotAvailableOnParquet, match="'ds' was built from parquet"):
            call(client)

    @pytest.mark.parametrize("refuse, reason", [
        (lambda c: c.refuse_put_on_parquet("ds"), "immutable"),
        (lambda c: c.refuse_tags_on_parquet("ds"), "tags"),
    ], ids=["put", "tags"])
    def test_the_refusals_the_command_line_asks_for_say_the_same_as_the_client_calls(self, monkeypatch, refuse, reason):
        with pytest.raises(NotAvailableOnParquet, match=f"'ds' was built from parquet: .*{reason}"):
            refuse(client_for(monkeypatch, "parquet"))
        refuse(client_for(monkeypatch, "csv"))  # nothing to refuse

    def test_reindexing_is_refused_before_anything_is_deleted(self, monkeypatch):
        # it would delete every block and the id map, then fail on source.csv
        client = client_for(monkeypatch, "parquet")
        deleted = []
        monkeypatch.setattr(client_module, "delete_indexes", lambda *args: deleted.append(args))
        with pytest.raises(NotAvailableOnParquet, match="initialize-database --format parquet"):
            client.reindex_pending("ds")
        assert deleted == []

    def test_a_csv_build_is_refused_before_it_overwrites_the_blocks(self, monkeypatch):
        # indexing writes centroid_i.ann over the parquet blocks and leaves
        # idmap/ behind, so the ids in the index no longer have provenance
        client = client_for(monkeypatch, "parquet")
        monkeypatch.setattr(client_module, "ServerlessVectorDB", lambda **config: pytest.fail("the build started"))
        with pytest.raises(NotAvailableOnParquet, match="delete the dataset first"):
            client.index_dataset("ds", {"implementation": "blocks", "num_index": 4})

    def test_a_csv_source_is_refused_before_it_is_uploaded(self, monkeypatch, tmp_path):
        client = client_for(monkeypatch, "parquet")
        monkeypatch.setattr(client_module, "upload_dataset", lambda *args: pytest.fail("the upload started"))
        source = tmp_path / "vectors.csv"
        source.write_text("1,0.0 1.0\n")
        with pytest.raises(NotAvailableOnParquet, match="another name"):
            client.create_dataset("ds", str(source))

    @pytest.mark.parametrize("source_format", ["csv", None])
    def test_a_csv_build_still_runs_where_no_parquet_index_stands(self, monkeypatch, source_format):
        client = client_for(monkeypatch, source_format)
        uploaded = []
        monkeypatch.setattr(client_module, "upload_dataset", lambda bucket, name, path: uploaded.append(name))
        client.create_dataset("ds", __file__)
        assert uploaded == ["ds"]

    def test_a_tag_filter_is_refused_not_answered_with_nothing(self, monkeypatch):
        # the query path turns a ValueError into empty results; this must get through
        client = client_for(monkeypatch, "parquet")
        with pytest.raises(NotAvailableOnParquet, match="tags"):
            client.query_batch("ds", [[0.0, 1.0]], k=1, filter_tags={"lang": "es"})

    def test_without_a_filter_a_parquet_index_is_still_queried(self, monkeypatch):
        client = client_for(monkeypatch, "parquet")
        monkeypatch.setattr(client_module, "ServerlessVectorDB", lambda **config: config)
        assert client._load_default_index("ds")["source_format"] == "parquet"

    def test_vectors_cannot_be_added_to_a_parquet_index(self, monkeypatch):
        # their ids would collide with the positional ids and have no provenance
        client = client_for(monkeypatch, "parquet")
        written = []
        client.tracker.put_vectors = lambda *args, **kwargs: written.append(args)
        with pytest.raises(NotAvailableOnParquet, match="immutable"):
            client.put_vector("ds", 7, [0.0, 1.0])
        assert written == []

    def test_vectors_are_still_added_to_a_csv_index(self, monkeypatch):
        client = client_for(monkeypatch, "csv")
        client.tracker.put_vectors = lambda *args, **kwargs: "pending/ds/1.csv"
        assert client.put_vectors("ds", [(7, [0.0, 1.0])]) == 1

    @pytest.mark.parametrize("source_format", ["csv", None])
    def test_csv_indexes_and_datasets_without_an_index_are_untouched(self, monkeypatch, source_format):
        client = client_for(monkeypatch, source_format)
        monkeypatch.setattr(client_module, "get_vectors_by_id", lambda *args: {"read": True})
        assert client.get_vectors("ds", [1]) == {"read": True}


class RefusingClient:
    """What the CLI sees for a parquet dataset."""

    tracker = None  # reaching it means the refusal came too late

    def __init__(self, **kwargs):
        pass

    def refuse_put_on_parquet(self, name):
        raise NotAvailableOnParquet(f"'{name}' was built from parquet: immutable")

    def refuse_tags_on_parquet(self, name):
        raise NotAvailableOnParquet(f"'{name}' was built from parquet: no tags")

    def get_vectors(self, name, ids):
        raise NotAvailableOnParquet(f"'{name}' was built from parquet: no source.csv")


class TestCli:
    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "backend_config.json")
        monkeypatch.setattr(cli, "VectorDBClient", RefusingClient)

        def untouched(*args, **kwargs):
            raise AssertionError("S3 was reached")

        monkeypatch.setattr(cli.boto3, "client", untouched)

        def run(*argv):
            monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "bucket", *argv])
            with pytest.raises(SystemExit) as stopped:
                cli.main()
            return str(stopped.value.code)

        return run

    def test_get_by_id_ends_with_the_reason_not_a_traceback(self, run):
        assert run("get", "ds", "1").startswith("Error: 'ds' was built from parquet")

    def test_get_by_tags_stops_before_scanning_the_bucket(self, run):
        assert run("get-by-tags", "ds", "--filter", '{"lang": "es"}').startswith("Error: 'ds' was built from parquet")

    def test_put_stops_before_anything_is_written(self, run, tmp_path):
        vectors = tmp_path / "vectors.csv"
        vectors.write_text("7,0.0 1.0\n")
        assert run("put", "ds", str(vectors)).startswith("Error: 'ds' was built from parquet")


class FakeBucket:
    """The listing and deletes delete_dataset makes, over a set of keys.
    A listing under ``refused`` fails the way S3 reports a denied one."""

    def __init__(self, keys, refused=None):
        self.keys = set(keys)
        self.refused = refused

    def delete_object(self, Bucket, Key):
        self.keys.discard(Key)

    def get_paginator(self, _name):
        keys, refused = self.keys, self.refused

        class Paginator:
            def paginate(self, Bucket, Prefix):
                if refused and Prefix.startswith(refused):
                    raise ClientError({"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "ListObjectsV2")
                yield {"Contents": [{"Key": key} for key in sorted(keys) if key.startswith(Prefix)]}

        return Paginator()

    def delete_objects(self, Bucket, Delete):
        self.keys -= {item["Key"] for item in Delete["Objects"]}


def test_deleting_a_dataset_removes_the_parquet_copies_it_uploaded(monkeypatch):
    bucket = FakeBucket([
        "datasets/ds/source/year=2026/language=spa/metadata_0_embeddings.parquet",
        "datasets/ds/source/year=2026/language=deu/metadata_0_embeddings.parquet",
        "datasets/ds-2/source/metadata_0_embeddings.parquet",  # another dataset
        "owi/spa/metadata_0_embeddings.parquet",  # a source read in place
    ])
    monkeypatch.setattr(dataset_ops, "s3", bucket)
    monkeypatch.setattr(index_ops, "delete_indexes", lambda *args: None)
    monkeypatch.setattr(index_ops, "delete_index_configs", lambda *args: None)
    dataset_ops.delete_dataset("bucket", "ds")
    assert bucket.keys == {
        "datasets/ds-2/source/metadata_0_embeddings.parquet",
        "owi/spa/metadata_0_embeddings.parquet",
    }


def test_copies_that_cannot_be_deleted_are_named_and_the_rest_still_goes(monkeypatch, capsys):
    bucket = FakeBucket(
        ["processed/ds/0.csv", "datasets/ds/source/metadata_0_embeddings.parquet"],
        refused="datasets/ds/source/",
    )
    monkeypatch.setattr(dataset_ops, "s3", bucket)
    monkeypatch.setattr(index_ops, "delete_indexes", lambda *args: None)
    monkeypatch.setattr(index_ops, "delete_index_configs", lambda *args: None)
    dataset_ops.delete_dataset("bucket", "ds")
    assert bucket.keys == {"datasets/ds/source/metadata_0_embeddings.parquet"}
    assert "Could not delete datasets/ds/source/: An error occurred (AccessDenied)" in capsys.readouterr().out


def test_an_interrupted_delete_stops_instead_of_going_on(monkeypatch):
    class InterruptedBucket(FakeBucket):
        def get_paginator(self, _name):
            raise KeyboardInterrupt()

    monkeypatch.setattr(dataset_ops, "s3", InterruptedBucket([]))
    monkeypatch.setattr(index_ops, "delete_indexes", lambda *args: pytest.fail("the indexes were deleted after the interrupt"))
    with pytest.raises(KeyboardInterrupt):
        dataset_ops.delete_dataset("bucket", "ds")
