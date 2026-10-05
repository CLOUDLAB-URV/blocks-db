"""Build from parquet and query, end to end, on the Lithops localhost backend.

No AWS: the executor runs functions as local processes and the storage
is a local directory. Run it before any build in a cloud account.
"""

import io

import numpy as np
import pyarrow.parquet as pq
import pytest

from helpers import write_owi
from vectordb.indexing.planner import PlanError
from vectordb.indexing.prepare import expand_sources, prepare_build
from vectordb.utils.idmap import idmap_prefix, select


pytestmark = pytest.mark.integration


def test_build_and_query_a_parquet_source(tmp_path, lithops_localhost):
    import lithops

    from vectordb.serverless_vectordb import ServerlessVectorDB

    bucket = lithops_localhost
    source = tmp_path / "metadata_0_embeddings.parquet"
    vectors = write_owi(source, rows=40, dimension=8, row_group_size=5)

    build_plan, sealed = prepare_build(
        [str(source)],
        {
            "implementation": "blocks",
            "features": 8,
            "num_index": 2,
            "k": 1,
            "n_probe": 1,
            "k_search": 3,
            "k_result": 3,
            "query_batch_size": 1,
            "index_mem": 512,
            "search_map_mem": 512,
            "search_reduce_mem": 512,
        },
    )
    sealed["dataset"] = "ds"
    sealed["storage_bucket"] = bucket
    assert sealed["source_format"] == "parquet" and sealed["total_vectors"] == 40

    db = ServerlessVectorDB(**sealed)
    # the payload check reads the limit of the executor that runs the build
    db.indexing_executor.config["lithops"]["data_limit"] = 0.0005
    with pytest.raises(PlanError, match="Raise data_limit"):
        db.check_plan(build_plan)
    del db.indexing_executor.config["lithops"]["data_limit"]
    db.check_plan(build_plan)
    times = db.indexing_from_plan(build_plan)
    assert times["rows"] == 40 and times["rejected"] == 0
    assert [r["block"] for r in times["blocks"]] == [0, 1]

    storage = lithops.Storage()
    prefix = "indexes/ds/blocks/"
    keys = set(storage.list_keys(bucket, prefix))
    assert {k for k in keys if k.endswith(".ann")} == {f"{prefix}centroid_0.ann", f"{prefix}centroid_1.ann"}
    assert {k for k in keys if k.endswith(".parquet")} == {f"{prefix}idmap/block_0.parquet", f"{prefix}idmap/block_1.parquet"}

    # every id of the source appears exactly once across the idmap parts
    ids = []
    for key in sorted(k for k in keys if k.endswith(".parquet")):
        ids += pq.read_table(io.BytesIO(storage.get_object(bucket, key))).column("id").to_pylist()
    assert sorted(ids) == list(range(sealed["total_vectors"]))

    # self-recovery: a stored vector is its own nearest neighbor, whatever
    # block it landed in, and comes back with its positional id
    probes = np.asarray([vectors[3], vectors[37]], dtype=np.float32)
    results, timers = db.search(0, probes)
    assert [hits[0][0] for hits in results] == [3, 37]
    assert all(hits[0][1] < 1e-3 for hits in results)
    assert all(len(hits) == 3 for hits in results)

    # provenance: the ids resolve to the publisher's record and chunk
    parts = (storage.get_object(bucket, key) for key in sorted(storage.list_keys(bucket, idmap_prefix("ds", "blocks"))))
    assert select(parts, [3, 37]) == {3: ("doc-1", 1), 37: ("doc-18", 1)}


def test_a_block_refused_inside_a_function_reaches_the_caller_by_name(tmp_path, lithops_localhost):
    from vectordb.implementations.blocks.initialize import BlockTooSmall
    from vectordb.serverless_vectordb import ServerlessVectorDB

    source = tmp_path / "day"
    source.mkdir()
    write_owi(source / "a_embeddings.parquet", rows=40, dimension=8, row_group_size=40)
    # 38 of the 40 rows of the second file hold a vector of the wrong length
    write_owi(source / "b_embeddings.parquet", rows=40, dimension=8, row_group_size=40, seed=2, bad_rows=set(range(2, 40)))
    build_plan, sealed = prepare_build(
        expand_sources(str(source)),
        {"implementation": "blocks", "features": 8, "num_index": 2, "k": 5, "n_probe": 1, "index_mem": 512},
    )
    sealed["dataset"] = "ds"
    sealed["storage_bucket"] = lithops_localhost
    with pytest.raises(BlockTooSmall, match="2 usable rows out of 40 planned"):
        ServerlessVectorDB(**sealed).indexing_from_plan(build_plan)
