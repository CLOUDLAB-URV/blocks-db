"""The largest block of a parquet plan must fit the disk of a function,
whose size is read from the Lithops configuration of the executor."""

import pytest

from vectordb import serverless_vectordb
from vectordb.indexing.planner import BlockPlan, Plan, PlanError, Range
from vectordb.serverless_vectordb import ServerlessVectorDB


class FakeExecutor:
    """What check_plan reads of a FunctionExecutor: the backend name and
    its loaded configuration."""

    def __init__(self, backend, **settings):
        self.backend = backend
        self.config = {"lithops": {"backend": backend}, backend: settings}


def plan_of(rows, dimension):
    """One block of ``rows`` vectors, as the planner lays it out."""
    source = "s3://bucket/x.parquet"
    block = BlockPlan(block=0, ranges=(Range(uri=source, row_group=0, start=0, end=rows, id_offset=0),))
    return Plan(blocks=(block,), total_vectors=rows, dimension=dimension, dialect="owi-v2", sources=(source,))


@pytest.fixture
def db_on(monkeypatch):
    """A ServerlessVectorDB whose executor runs on the given backend."""
    monkeypatch.setattr(serverless_vectordb, "Orchestrator", lambda params, wait_timeout=None: None)

    def db_on(backend, **settings):
        monkeypatch.setattr(serverless_vectordb, "FunctionExecutor", lambda: FakeExecutor(backend, **settings))
        return ServerlessVectorDB(implementation="blocks")

    return db_on


def test_a_block_too_large_for_the_lambda_disk_is_refused(db_on):
    # 200 million vectors of 1024 values: about 780 GB on disk
    with pytest.raises(PlanError, match="ephemeral storage of 10240 MB.*aws_lambda.ephemeral_storage"):
        db_on("aws_lambda", ephemeral_storage=10240).check_plan(plan_of(200_000_000, 1024))


def test_a_block_that_fits_the_lambda_disk_passes(db_on):
    db_on("aws_lambda", ephemeral_storage=512).check_plan(plan_of(1000, 8))


def test_a_backend_without_the_setting_has_no_disk_limit(db_on):
    db_on("localhost", runtime="python3").check_plan(plan_of(200_000_000, 1024))
