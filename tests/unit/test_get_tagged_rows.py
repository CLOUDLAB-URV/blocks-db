"""get reads the vector of a row that carries tags in a third column."""

import json

from vectordb.utils import query_ops

# the three forms File Formats documents: one tag, several tags quoted as a
# CSV field, and no tags
ROWS = (
    '1,0.1 0.2,{"source":"web"}\n'
    '2,0.3 0.4,"{""source"":""api"",""priority"":""low""}"\n'
    '3,0.5 0.6\n'
)
VECTORS = {1: [0.1, 0.2], 2: [0.3, 0.4], 3: [0.5, 0.6]}
SOURCE = "datasets/ds/source.csv"


class Body:
    def __init__(self, data):
        self.data = data

    def iter_lines(self):
        return iter(self.data.splitlines())

    def read(self):
        return self.data


class FakeS3:
    """A bucket of text objects; get_object honors a byte range."""

    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self, objects):
        self.objects = {key: text.encode() for key, text in objects.items()}

    def get_object(self, Bucket, Key, Range=None):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        data = self.objects[Key]
        if Range:
            first, last = Range[len("bytes="):].split("-")
            data = data[int(first):int(last) + 1]
        return {"Body": Body(data)}

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [{"Key": key} for key in sorted(self.objects) if key.startswith(Prefix)]}


def bucket(monkeypatch, objects):
    monkeypatch.setattr(query_ops, "s3", FakeS3(objects))


class TestGetById:
    def test_from_the_source_file(self, monkeypatch):
        bucket(monkeypatch, {SOURCE: ROWS})
        found = query_ops.get_vectors_by_id("bucket", "ds", [1, 2, 3])
        assert {vid: entry["vector"] for vid, entry in found.items()} == VECTORS
        assert all(entry["source"] == "indexed" for entry in found.values())

    def test_from_a_block_of_the_source_file(self, monkeypatch):
        # with the block table, get reads only the byte range of the block
        blocks = [{"start_id": 1, "end_id": 3, "offset": 0, "size": len(ROWS.encode())}]
        bucket(monkeypatch, {SOURCE: ROWS, "tracking/csv_blocks_ds.json": json.dumps(blocks)})
        found = query_ops.get_vectors_by_id("bucket", "ds", [2])
        assert found == {2: {"vector": VECTORS[2], "source": "indexed"}}

    def test_from_a_pending_file(self, monkeypatch):
        # put writes its files with a header and the tags as a third column
        key = "pending/ds/batch.csv"
        bucket(monkeypatch, {SOURCE: "9,0.9 0.9\n", key: "id,vector,tags\r\n" + ROWS.replace("\n", "\r\n")})
        found = query_ops.get_vectors_by_id("bucket", "ds", [1, 2])
        assert found == {
            1: [{"vector": VECTORS[1], "source": "pending", "file": key}],
            2: [{"vector": VECTORS[2], "source": "pending", "file": key}],
        }


class TestList:
    def test_the_first_rows(self, monkeypatch):
        bucket(monkeypatch, {SOURCE: ROWS})
        assert query_ops.list_vectors("bucket", "ds", limit=2) == {1: VECTORS[1], 2: VECTORS[2]}

    def test_a_page(self, monkeypatch):
        bucket(monkeypatch, {SOURCE: ROWS})
        assert query_ops.list_vectors_paginated("bucket", "ds", start=1, limit=2) == {2: VECTORS[2], 3: VECTORS[3]}
