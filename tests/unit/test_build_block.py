"""The parquet map function: one block, its provenance map, its counts."""

import shutil
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq

import pytest

from helpers import write_canonical, write_owi
from vectordb.implementations.blocks.initialize import BlockTooSmall, build_block_from_parquet
from vectordb.indexing.planner import plan
from vectordb.utils.parquet import inspect


class FakeStorage:
    """upload_file, the call the map function makes, on a local directory."""

    def __init__(self, root: Path):
        self.root = root

    def upload_file(self, local_path, bucket, key):
        target = self.root / bucket / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, target)


class Params:
    features = 4
    k = 1
    n_probe = 1
    dataset = "ds"
    implementation = "blocks"
    storage_bucket = "bucket"


def test_owi_block_is_built_with_positional_ids_and_provenance(tmp_path):
    source = tmp_path / "metadata_0_embeddings.parquet"
    vectors = write_owi(source, rows=12, row_group_size=4, bad_rows={5})
    result = plan([inspect(str(source))], num_index=2, k=1)
    storage = FakeStorage(tmp_path / "store")

    # row groups of 4, 4, 4 rows and two blocks: the plan cuts on a row-group
    # edge, and the test follows whatever cut it chose; row 5 is rejected
    reports = [build_block_from_parquet(block, Params(), storage) for block in result.blocks]

    assert sum(r["rows"] for r in reports) == 11 and sum(r["rejected"] for r in reports) == 1
    for block, report in zip(result.blocks, reports):
        index = faiss.read_index(str(tmp_path / "store/bucket/indexes/ds/blocks" / f"centroid_{block.block}.ann"))
        assert index.ntotal == report["rows"]
        idmap = pq.read_table(tmp_path / "store/bucket/indexes/ds/blocks/idmap" / f"block_{block.block}.parquet").to_pydict()
        assert len(idmap["id"]) == report["rows"]
        expected_ids = [i for part in block.ranges for i in range(part.id_offset, part.id_offset + part.rows) if i != 5]
        assert idmap["id"] == expected_ids
        assert idmap["record_id"] == [f"doc-{i // 2}" for i in expected_ids]
        assert idmap["chunk_idx"] == [i % 2 for i in expected_ids]
        # a stored vector is found by its own id at distance 0
        probe = np.asarray([vectors[expected_ids[0]]], dtype=np.float32)
        distances, found = index.search(probe, 1)
        assert found[0][0] == expected_ids[0] and distances[0][0] < 1e-3


def test_canonical_block_keeps_the_file_ids_as_provenance_text(tmp_path):
    source = tmp_path / "vectors.parquet"
    write_canonical(source, rows=6, dimension=4, row_group_size=6)
    result = plan([inspect(str(source))], num_index=1, k=1)
    storage = FakeStorage(tmp_path / "store")

    report = build_block_from_parquet(result.blocks[0], Params(), storage)

    assert report == {"block": 0, "rows": 6, "rejected": 0, "seconds": report["seconds"]}
    idmap = pq.read_table(tmp_path / "store/bucket/indexes/ds/blocks/idmap/block_0.parquet").to_pydict()
    assert idmap["id"] == list(range(6))  # positional, dense
    assert idmap["record_id"] == [str(i) for i in range(100, 106)]
    assert idmap["chunk_idx"] == [0] * 6


def test_a_canonical_block_without_ids_keeps_the_index_id_as_provenance(tmp_path):
    source = tmp_path / "vectors.parquet"
    write_canonical(source, rows=6, dimension=4, row_group_size=3, with_ids=False)
    result = plan([inspect(str(source))], num_index=2, k=1)
    storage = FakeStorage(tmp_path / "store")

    reports = [build_block_from_parquet(block, Params(), storage) for block in result.blocks]

    assert [r["rows"] for r in reports] == [3, 3]
    for block in result.blocks:
        idmap = pq.read_table(tmp_path / "store/bucket/indexes/ds/blocks/idmap" / f"block_{block.block}.parquet").to_pydict()
        assert idmap["id"] == list(range(block.first_id, block.last_id + 1))
        assert idmap["record_id"] == [str(i) for i in idmap["id"]]
        assert idmap["chunk_idx"] == [0] * 3


def test_a_block_left_below_k_by_rejections_is_refused_by_name(tmp_path):
    # the planner can only bound k from the footers; rejected rows are
    # only known here, so the worker is the last line of defence
    source = tmp_path / "bad.parquet"
    write_owi(source, rows=8, row_group_size=4, bad_rows=set(range(4)))
    result = plan([inspect(str(source))], num_index=2, k=1)
    storage = FakeStorage(tmp_path / "store")

    class P(Params):
        k = 2

    with pytest.raises(BlockTooSmall, match=r"block \d: 0 usable rows out of 4 planned \(4 rejected"):
        build_block_from_parquet(result.blocks[0], P(), storage)
    assert not (tmp_path / "store").exists()  # nothing uploaded for that block


def test_a_sliced_block_and_a_two_file_block_build_the_rows_they_planned(tmp_path):
    # one row group, three blocks: the planner slices the row group, which
    # the tests above do not
    one = tmp_path / "one.parquet"
    write_owi(one, rows=9, row_group_size=9)
    sliced = plan([inspect(str(one))], num_index=3, k=1)
    storage = FakeStorage(tmp_path / "store")
    reports = [build_block_from_parquet(b, Params(), storage) for b in sliced.blocks]
    assert sum(r["rows"] for r in reports) == 9
    ids = []
    for block in sliced.blocks:
        table = pq.read_table(tmp_path / "store/bucket/indexes/ds/blocks/idmap" / f"block_{block.block}.parquet")
        ids += table.column("id").to_pylist()
    assert sorted(ids) == list(range(9))

    # two files in one block: _by_file must read each of them
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_owi(a, rows=4, row_group_size=2)
    write_owi(b, rows=4, row_group_size=2, seed=7)
    pair = plan([inspect(str(a)), inspect(str(b))], num_index=1, k=1)
    storage2 = FakeStorage(tmp_path / "store2")
    report = build_block_from_parquet(pair.blocks[0], Params(), storage2)
    assert report["rows"] == 8
    table = pq.read_table(tmp_path / "store2/bucket/indexes/ds/blocks/idmap/block_0.parquet")
    assert table.column("id").to_pylist() == list(range(8))
    assert len({uri for part in pair.blocks[0].ranges for uri in [part.uri]}) == 2
