"""The wait for the functions gives up when none finishes for longer than a
function may run, and keeps going while they finish however slowly."""

import pytest

from vectordb.utils.waiting import FunctionsTimedOut, collect, inactivity_window


class Future:
    def __init__(self, finishes_at):
        self.finishes_at = finishes_at
        self.done = False


class FakeExecutor:
    """A clock, the futures it marks done as the clock passes their time,
    and the two Lithops calls the wait makes."""

    def __init__(self, config, finishes_at=()):
        self.config = config
        self.futures = [Future(at) for at in finishes_at]
        self.now = 0.0
        self.calls = []

    def wait(self, futures, return_when, show_progressbar):
        self.calls.append("wait")
        for future in futures:
            if future.finishes_at is not None and future.finishes_at <= self.now:
                future.done = True

    def get_result(self, futures):
        self.calls.append("get_result")
        return [future.finishes_at for future in futures]

    def sleep(self, seconds):
        self.now += seconds


def lambda_config(runtime_timeout=900):
    return {"lithops": {"backend": "aws_lambda"}, "aws_lambda": {"runtime_timeout": runtime_timeout}}


def run(executor, window, poll=1.0):
    return collect(executor, executor.futures, window, poll_sec=poll, clock=lambda: executor.now, sleep=executor.sleep)


class TestTheWindow:
    def test_it_follows_the_function_timeout_of_the_backend(self):
        assert inactivity_window(FakeExecutor(lambda_config(900))) == 960

    def test_it_falls_back_on_the_general_execution_timeout(self):
        # localhost declares no runtime_timeout; Lithops gives it an execution_timeout
        config = {"lithops": {"backend": "localhost", "execution_timeout": 3600}, "localhost": {}}
        assert inactivity_window(FakeExecutor(config)) == 3660

    def test_a_configuration_without_any_timeout_gives_no_window(self):
        assert inactivity_window(FakeExecutor({"lithops": {"backend": "localhost"}, "localhost": {}})) is None


class TestCollect:
    def test_the_results_come_back_once_every_function_has_finished(self):
        executor = FakeExecutor(lambda_config(), finishes_at=[2, 5])
        assert run(executor, window=10) == [2, 5]
        assert executor.calls[-1] == "get_result"

    def test_the_wait_ends_when_nothing_finishes_for_a_window(self):
        executor = FakeExecutor(lambda_config(), finishes_at=[None, None, 1])
        with pytest.raises(FunctionsTimedOut, match="2 of 3 functions did not finish, and none finished in the last 10 s"):
            run(executor, window=10)
        assert executor.now == pytest.approx(11, abs=1)

    def test_functions_that_keep_finishing_keep_the_wait_alive(self):
        # throttled functions finish one after another: the total exceeds the
        # window but no single gap does, so nothing is wrong
        executor = FakeExecutor(lambda_config(), finishes_at=[8, 16, 24, 32])
        assert run(executor, window=10) == [8, 16, 24, 32]

    def test_a_zero_window_waits_as_get_result_does(self):
        executor = FakeExecutor(lambda_config(), finishes_at=[None])
        assert run(executor, window=0) == [None]
        assert executor.calls == ["get_result"]

    def test_without_a_declared_timeout_the_wait_is_unbounded(self):
        executor = FakeExecutor({"lithops": {"backend": "localhost"}, "localhost": {}}, finishes_at=[None])
        assert run(executor, window=None) == [None]
        assert executor.calls == ["get_result"]


class TestTheCallers:
    """A query and a build wait through collect, with the window they were given."""

    STATS = {"worker_func_start_tstamp": 1.0, "host_job_create_tstamp": 0.0}

    def test_a_query_waits_for_its_map_and_its_reduce(self, monkeypatch):
        from types import SimpleNamespace

        import numpy as np

        from vectordb.orchestration import orchestrator

        waited = []
        monkeypatch.setattr(orchestrator, "collect", lambda fexec, futures, window=None: waited.append(window) or [])
        search = object.__new__(orchestrator.Orchestrator)
        search.wait_timeout = 300
        search.config = SimpleNamespace(
            dataset="ds", num_index=4, storage_bucket="bucket", implementation="blocks",
            search_map_mem=1024, search_reduce_mem=1024, k_search=1, k_result=1,
        )
        search.function_executor = SimpleNamespace(
            config={"lithops": {"backend": "localhost"}},
            storage=SimpleNamespace(put_object=lambda **kwargs: None, delete_object=None),
            map=lambda *args, **kwargs: [],
        )
        search.query_strategy = SimpleNamespace(create_map_tasks=lambda *args, **kwargs: [])
        search.map_fn = search.reduce_fn = None
        search.pool = SimpleNamespace(submit=lambda *args: None)
        monkeypatch.setattr(search, "create_reduce_iterdata", lambda payload, k, per_task: ([], 0.0))
        search.search("q", np.zeros((1, 2)))
        assert waited == [300, 300]

    def test_a_build_waits_for_its_blocks(self, monkeypatch):
        from types import SimpleNamespace

        from vectordb.indexing import indexator

        waited = []
        monkeypatch.setattr(indexator, "collect", lambda fexec, futures, window=None: waited.append(window) or [1.0])
        executor = SimpleNamespace(map=lambda *args, **kwargs: [SimpleNamespace(stats=self.STATS)])
        params = SimpleNamespace(skip_kmeans=True, implementation="blocks", storage_bucket="bucket", num_index=4, index_mem=1024)
        indexator.initialize_database("datasets/ds/source.csv", params, executor, 2, 300)
        assert waited == [300]

    def test_the_facade_hands_its_window_to_the_build_and_to_the_query(self, monkeypatch):
        from vectordb import serverless_vectordb

        handed = {}
        monkeypatch.setattr(serverless_vectordb, "FunctionExecutor", lambda: "executor")
        monkeypatch.setattr(
            serverless_vectordb, "Orchestrator",
            lambda params, wait_timeout=None: handed.update(query=wait_timeout),
        )
        monkeypatch.setattr(
            serverless_vectordb, "initialize_database",
            lambda filename, params, fexec, workers, wait_timeout=None: handed.update(build=wait_timeout),
        )
        db = serverless_vectordb.ServerlessVectorDB(wait_timeout=300, dataset="ds", storage_bucket="bucket")
        db.indexing("datasets/ds/source.csv", 2)
        assert handed == {"query": 300, "build": 300}
