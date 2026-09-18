"""The indexed search: n_probe governs it, each task owns its files, no sentinels."""

import shutil
from pathlib import Path

import faiss
import numpy as np
import orjson
import pytest

from vectordb.config import SvlessVectorDBParams
from vectordb.implementations.blocks import querying
from vectordb.implementations.blocks.indexing import FaissIVFIndex


class FakeStorage:
    """The three calls the search makes, over a local directory, recording
    where each block was downloaded to."""

    def __init__(self, root: Path):
        self.root = root
        self.downloads: list[str] = []

    def put_object(self, bucket, key, body):
        target = self.root / bucket / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)

    def get_object(self, bucket, key):
        return (self.root / bucket / key).read_bytes()

    def download_file(self, bucket, key, local_path):
        self.downloads.append(local_path)
        shutil.copy(self.root / bucket / key, local_path)

    def list_keys(self, bucket, prefix):
        base = self.root / bucket
        return [str(p.relative_to(base)) for p in base.rglob("*") if p.is_file() and str(p.relative_to(base)).startswith(prefix)]


def params(**overrides):
    values = dict(
        dataset="ds", implementation="blocks", storage_bucket="bucket",
        features=4, num_index=1, k=2, n_probe=1, k_search=10, k_result=10,
        query_batch_size=1,
    )
    values.update(overrides)
    return SvlessVectorDBParams(**values)


@pytest.fixture
def two_cluster_block(tmp_path):
    """One block of two well-separated IVF lists, six vectors each."""
    rng = np.random.default_rng(0)
    near = rng.normal(0.0, 0.01, (6, 4)).astype(np.float32)
    far = (rng.normal(0.0, 0.01, (6, 4)) + 10.0).astype(np.float32)
    vectors = np.concatenate([near, far])
    index = FaissIVFIndex(params()).build(np.arange(12), vectors)
    storage = FakeStorage(tmp_path / "store")
    local = tmp_path / "centroid_0.ann"
    faiss.write_index(index, str(local))
    storage.put_object("bucket", "indexes/ds/blocks/centroid_0.ann", local.read_bytes())
    storage.put_object("bucket", "q.json", orjson.dumps([near[0].tolist()]))
    return storage, vectors


def search(storage, config, blocks=(0,)):
    return querying._search_indexed(("q.json", list(blocks)), config.k_search, storage, config, 0.0)


class TestNProbeGovernsTheSearch:
    def test_widening_n_probe_reaches_the_second_list(self, two_cluster_block):
        # the whole point of applying config.n_probe after read_index: with
        # one list probed the far cluster is unreachable, with two it is
        storage, _ = two_cluster_block
        narrow = search(storage, params(n_probe=1))[0]
        wide = search(storage, params(n_probe=2))[0]
        assert len(narrow) == 6, narrow
        assert len(wide) == 10
        assert {row[0] for row in narrow} <= set(range(6))
        assert {row[0] for row in wide} & set(range(6, 12))

    def test_no_sentinel_id_is_returned_when_a_list_is_short(self, two_cluster_block):
        storage, _ = two_cluster_block
        hits = search(storage, params(n_probe=1))[0]
        assert all(row[0] >= 0 for row in hits)
        assert all(row[1] < 1e30 for row in hits)


class TestEachTaskOwnsItsFiles:
    def test_two_searches_never_share_a_local_path(self, two_cluster_block):
        # two map tasks on one filesystem used to overwrite and delete
        # /tmp/index_0.ann under each other
        storage, _ = two_cluster_block
        search(storage, params())
        search(storage, params())
        assert len(storage.downloads) == 2
        assert len(set(storage.downloads)) == 2
        assert not any(path.startswith("/tmp/index_") for path in storage.downloads)
