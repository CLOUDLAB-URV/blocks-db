"""What each build task of the parquet path carries, and how much all of
them may weigh before Lithops refuses to send them."""

import pytest

from helpers import write_owi
from vectordb.config import SvlessVectorDBParams
from vectordb.indexing import indexator
from vectordb.indexing.indexator import check_payload, initialize_from_plan
from vectordb.indexing.planner import PlanError, plan
from vectordb.utils.parquet import inspect


class FakeExecutor:
    """Records the map and answers it with one report per block."""

    def __init__(self, config=None):
        self.config = config or {"lithops": {"backend": "localhost"}}
        self.map_calls = []

    def map(self, function, iterdata, extra_args=None, runtime_memory=None):
        self.map_calls.append((function, iterdata, extra_args))
        return [FakeFuture(block.block) for block in iterdata]

    def get_result(self, futures):
        return [future.report for future in futures]


class FakeFuture:
    def __init__(self, block):
        self.report = {"block": block, "rows": 4, "rejected": 0, "seconds": 0.1}
        self.stats = {"worker_func_start_tstamp": 1.0, "host_job_create_tstamp": 0.0}


def params_for(built, **overrides):
    values = dict(
        dataset="ds", storage_bucket="bucket", implementation="blocks", features=4, k=1,
        num_index=built.num_index, source_format="parquet", source_keys=list(built.sources),
        block_ranges=[[b.block, b.first_id, b.last_id] for b in built.blocks],
    )
    values.update(overrides)
    return SvlessVectorDBParams(**values)


@pytest.fixture
def two_blocks(tmp_path):
    source = tmp_path / "metadata_0_embeddings.parquet"
    write_owi(source, rows=8, row_group_size=4)
    return plan([inspect(str(source))], num_index=2, k=1)


def test_a_task_carries_neither_the_source_list_nor_the_block_ranges(two_blocks):
    # the plan names the file and rows of every block; the two lists are
    # for config.json, and repeated in every task they outweigh the plan
    fexec = FakeExecutor()
    params = params_for(two_blocks)

    result = initialize_from_plan(two_blocks, params, fexec)

    (function, iterdata, extra_args), = fexec.map_calls
    assert function.__name__ == "build_block_from_parquet"
    assert iterdata == list(two_blocks.blocks)
    (carried,) = extra_args
    assert carried.source_keys is None and carried.block_ranges is None
    assert (carried.dataset, carried.features, carried.k, carried.storage_bucket) == ("ds", 4, 1, "bucket")
    assert params.source_keys == list(two_blocks.sources)  # the caller's copy is untouched
    assert result["rows"] == 8 and [r["block"] for r in result["blocks"]] == [0, 1]


class TestCheckPayload:
    def test_a_plan_over_the_data_limit_is_refused_with_the_numbers(self, two_blocks):
        fexec = FakeExecutor({"lithops": {"data_limit": 0.0005}})  # 512 bytes
        with pytest.raises(PlanError, match=r"2 build tasks weigh 0\.\d+ MiB \(1 source files, 2 row ranges\)"):
            check_payload(two_blocks, params_for(two_blocks), fexec)

    def test_the_limit_is_four_mebibytes_unless_the_configuration_says_otherwise(self, two_blocks, monkeypatch):
        assert check_payload(two_blocks, params_for(two_blocks), FakeExecutor()) is None
        # the default is the constant Lithops itself falls back on
        monkeypatch.setattr(indexator, "MAX_AGG_DATA_SIZE", 0.0005)
        with pytest.raises(PlanError, match="more than the 0.0005 MiB"):
            check_payload(two_blocks, params_for(two_blocks), FakeExecutor())

    def test_a_false_limit_disables_the_check_as_it_does_in_lithops(self, two_blocks):
        for limit in (0, None, False):
            assert check_payload(two_blocks, params_for(two_blocks), FakeExecutor({"lithops": {"data_limit": limit}})) is None

    def test_the_check_weighs_what_the_tasks_carry(self, two_blocks):
        # a source list of thousands of files weighs far more than the
        # limit here, and does not count: the tasks do not carry it
        heavy = params_for(two_blocks, source_keys=[f"s3://bucket/day/metadata_{i}_embeddings.parquet" for i in range(5000)])
        assert check_payload(two_blocks, heavy, FakeExecutor({"lithops": {"data_limit": 0.05}})) is None
