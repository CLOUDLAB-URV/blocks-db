"""The client's parquet build: what it uploads, seals and cleans up."""

import pytest

from helpers import write_owi
from vectordb.client import IndexExists, VectorDBClient
from vectordb.indexing.planner import PlanError


class FakeS3:
    def __init__(self):
        self.uploads: dict[str, str] = {}
        self.deleted: list[str] = []
        self.objects: dict[str, bytes] = {}

    def upload_file(self, local, bucket, key):
        self.uploads[key] = local

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body

    def get_paginator(self, _name):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": key} for key in objects if key.startswith(Prefix)]}

        return Paginator()

    def delete_objects(self, Bucket, Delete):
        self.deleted += [item["Key"] for item in Delete["Objects"]]


class FakeTracker:
    def __init__(self):
        self.seeded = None
        self.strict = None
        self.fail = False

    def initialize_next_id(self, dataset_name, next_id, strict=False):
        if self.fail:
            raise RuntimeError(f"cannot seed the id counter of '{dataset_name}'")
        self.seeded = (dataset_name, next_id)
        self.strict = strict


def client_with(fake_s3=None) -> VectorDBClient:
    client = object.__new__(VectorDBClient)
    client.bucket = "bucket"
    client.wait_timeout = None
    client.s3 = fake_s3 or FakeS3()
    client.tracker = FakeTracker()
    return client


class StubDB:
    """Stands in for ServerlessVectorDB, which would build a Lithops executor."""

    built = None
    fail = False
    half_build = None  # an objects dict: a failing build first writes one block there

    def __init__(self, **params):
        StubDB.params = params

    def indexing_from_plan(self, plan):
        if StubDB.fail:
            if StubDB.half_build is not None:
                StubDB.half_build["indexes/ds/blocks/centroid_0.ann"] = b"half a build"
            raise StubDB.fail if isinstance(StubDB.fail, BaseException) else RuntimeError("a block failed")
        StubDB.built = plan
        return {"rows": plan.total_vectors - 1, "rejected": 1, "blocks": [], "total_indexing_blocks": 0.1}


@pytest.fixture
def partitioned_corpus(tmp_path):
    """Two partitions whose files share a name, as the publisher writes them."""
    for language in ("spa", "eng"):
        directory = tmp_path / "day" / f"language={language}"
        directory.mkdir(parents=True)
        write_owi(directory / "metadata_0_embeddings.parquet", rows=8, dimension=4, row_group_size=4)
    return tmp_path / "day"


@pytest.fixture
def stub_db(monkeypatch):
    from vectordb import client as client_module

    StubDB.built = None
    StubDB.fail = False
    monkeypatch.setattr(client_module, "ServerlessVectorDB", StubDB)
    return StubDB


def config(**overrides):
    values = {"implementation": "blocks", "num_index": 2, "k": 1, "features": 4}
    values.update(overrides)
    return values


class TestIndexParquetDataset:
    def test_upload_keys_keep_the_partition_path(self, partitioned_corpus, stub_db):
        from vectordb.indexing.prepare import expand_sources

        client = client_with()
        sources = expand_sources(str(partitioned_corpus))
        assert len(sources) == 2  # both partitions found, same file name

        client.index_parquet_dataset("ds", sources, config())

        keys = sorted(client.s3.uploads)
        assert keys == [
            "datasets/ds/source/language=eng/metadata_0_embeddings.parquet",
            "datasets/ds/source/language=spa/metadata_0_embeddings.parquet",
        ]
        # the plan the workers receive points at what was uploaded
        uris = {part.uri for block in stub_db.built.blocks for part in block.ranges}
        assert all(uri.startswith("s3://bucket/datasets/ds/source/") for uri in uris)
        assert len(uris) == 2

    def test_the_build_is_opened_with_the_wait_timeout_of_the_client(self, partitioned_corpus, stub_db):
        from vectordb.indexing.prepare import expand_sources

        client = client_with()
        client.wait_timeout = 300
        client.index_parquet_dataset("ds", expand_sources(str(partitioned_corpus)), config())
        assert stub_db.params["wait_timeout"] == 300

    def test_the_sealed_config_records_what_was_indexed(self, partitioned_corpus, stub_db):
        import json

        client = client_with()
        from vectordb.indexing.prepare import expand_sources

        client.index_parquet_dataset("ds", expand_sources(str(partitioned_corpus)), config())
        sealed = json.loads(client.s3.objects["indexes/ds/blocks/config.json"])
        assert sealed["source_format"] == "parquet"
        assert sealed["source_rows"] == 16 and sealed["num_vectors"] == 15 and sealed["rejected"] == 1
        assert sealed["block_ranges"] == [[0, 0, 7], [1, 8, 15]]
        assert sealed["features"] == 4
        # the id counter is seeded above the footer total, so the gap the
        # rejected row left is never handed to a later put
        assert client.tracker.seeded == ("ds", 16)
        assert client.tracker.strict is True

    def test_a_refused_plan_uploads_nothing(self, partitioned_corpus, stub_db):
        client = client_with()
        from vectordb.indexing.prepare import expand_sources

        sources = expand_sources(str(partitioned_corpus))
        with pytest.raises(PlanError, match="k .* must be declared"):
            client.index_parquet_dataset("ds", sources, {"implementation": "blocks", "num_index": 2, "features": 4})
        assert client.s3.uploads == {}
        assert stub_db.built is None

    def test_a_counter_that_cannot_be_seeded_stops_the_build_before_it_starts(self, partitioned_corpus, stub_db):
        client = client_with()
        from vectordb.indexing.prepare import expand_sources

        client.tracker.fail = True
        with pytest.raises(RuntimeError, match="cannot seed the id counter"):
            client.index_parquet_dataset("ds", expand_sources(str(partitioned_corpus)), config())
        assert client.s3.uploads == {}
        assert stub_db.built is None
        assert client.s3.objects == {}

    def test_without_saving_the_config_the_counter_is_not_touched(self, partitioned_corpus, stub_db):
        client = client_with()
        from vectordb.indexing.prepare import expand_sources

        client.tracker.fail = True  # would raise if it were called
        client.index_parquet_dataset("ds", expand_sources(str(partitioned_corpus)), config(), save_config=False)
        assert stub_db.built is not None

    def test_a_failed_build_leaves_no_blocks_and_no_config(self, partitioned_corpus, stub_db):
        client = client_with()
        from vectordb.indexing.prepare import expand_sources

        stub_db.half_build = client.s3.objects
        stub_db.fail = True
        with pytest.raises(RuntimeError, match="a block failed"):
            client.index_parquet_dataset("ds", expand_sources(str(partitioned_corpus)), config())
        assert client.s3.deleted == ["indexes/ds/blocks/centroid_0.ann"]
        assert "indexes/ds/blocks/config.json" not in client.s3.objects


    def test_a_second_build_with_the_same_name_is_refused_before_anything_happens(self, partitioned_corpus, stub_db):
        # an index is immutable: replacing it is a decision, never a side effect
        client = client_with()
        from vectordb.indexing.prepare import expand_sources

        client.s3.objects["indexes/ds/blocks/centroid_0.ann"] = b"an index"
        with pytest.raises(IndexExists, match="replace"):
            client.index_parquet_dataset("ds", expand_sources(str(partitioned_corpus)), config())
        assert client.s3.uploads == {} and client.s3.deleted == []
        assert client.tracker.seeded is None and stub_db.built is None

    def test_replace_deletes_the_previous_index_and_builds(self, partitioned_corpus, stub_db, capsys):
        client = client_with()
        from vectordb.indexing.prepare import expand_sources

        client.s3.objects["indexes/ds/blocks/centroid_0.ann"] = b"an index"
        client.s3.objects["indexes/ds/blocks/config.json"] = b"{}"
        client.index_parquet_dataset("ds", expand_sources(str(partitioned_corpus)), config(), replace=True)
        assert sorted(client.s3.deleted) == ["indexes/ds/blocks/centroid_0.ann", "indexes/ds/blocks/config.json"]
        assert stub_db.built is not None and "indexes/ds/blocks/config.json" in client.s3.objects
        # what was removed was the index being replaced, not a failed build
        assert "Removed 2 objects of the previous index under indexes/ds/blocks/" in capsys.readouterr().out

    def test_a_counter_that_cannot_be_seeded_leaves_the_previous_index_intact(self, partitioned_corpus, stub_db):
        # the seed is the first write of a build; with replace it must also
        # come before the old index is deleted, or a refused write leaves
        # nothing to search
        client = client_with()
        from vectordb.indexing.prepare import expand_sources

        client.s3.objects["indexes/ds/blocks/centroid_0.ann"] = b"an index"
        client.s3.objects["indexes/ds/blocks/config.json"] = b"{}"
        client.tracker.fail = True
        with pytest.raises(RuntimeError, match="cannot seed the id counter"):
            client.index_parquet_dataset("ds", expand_sources(str(partitioned_corpus)), config(), replace=True)
        assert client.s3.deleted == [] and client.s3.uploads == {}
        assert client.s3.objects["indexes/ds/blocks/centroid_0.ann"] == b"an index"
        assert stub_db.built is None

    def test_an_interrupted_build_is_cleaned_up_like_a_failed_one(self, partitioned_corpus, stub_db):
        # Ctrl-C raises KeyboardInterrupt, which is not an Exception: the
        # cleanup must not depend on the kind of failure
        client = client_with()
        from vectordb.indexing.prepare import expand_sources

        stub_db.half_build = client.s3.objects
        stub_db.fail = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            client.index_parquet_dataset("ds", expand_sources(str(partitioned_corpus)), config())
        assert client.s3.deleted == ["indexes/ds/blocks/centroid_0.ann"]
        assert "indexes/ds/blocks/config.json" not in client.s3.objects


class TestProvenance:
    def test_only_the_parts_covering_the_ids_are_fetched(self, monkeypatch):
        import io

        import pyarrow as pa
        import pyarrow.parquet as pq

        from vectordb import client as client_module

        def part(ids):
            sink = io.BytesIO()
            pq.write_table(
                pa.table({
                    "id": pa.array(ids, pa.int64()),
                    "record_id": [f"doc-{i}" for i in ids],
                    "chunk_idx": pa.array([0] * len(ids), pa.int64()),
                }),
                sink,
            )
            return sink.getvalue()

        fake = FakeS3()
        fake.objects["indexes/ds/blocks/idmap/block_0.parquet"] = part([0, 1])
        fake.objects["indexes/ds/blocks/idmap/block_1.parquet"] = part([2, 3])
        client = client_with(fake)
        fetched = []

        def get_object(Bucket, Key):
            fetched.append(Key)
            return {"Body": io.BytesIO(fake.objects[Key])}

        fake.get_object = get_object
        monkeypatch.setattr(
            client_module, "load_index_config",
            lambda bucket, dataset, implementation, num_index: {"block_ranges": [[0, 0, 1], [1, 2, 3]]},
        )

        assert client.provenance("ds", [3]) == {3: ("doc-3", 0)}
        assert fetched == ["indexes/ds/blocks/idmap/block_1.parquet"]
