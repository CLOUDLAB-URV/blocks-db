"""The wait for the functions gives up when some never start and none starts
or finishes within the window, and hands over to get_result as soon as every
function has started."""

import pytest

from vectordb.utils.waiting import FunctionsTimedOut, collect, inactivity_window


class Future:
    """The public states the job monitor sets on a Lithops future, here
    following a clock: running from ``starts_at``, ready from ``finishes_at``
    (None: never)."""

    success = done = False

    def __init__(self, executor, starts_at, finishes_at):
        self.executor, self.starts_at, self.finishes_at = executor, starts_at, finishes_at

    def _passed(self, moment):
        return moment is not None and moment <= self.executor.now

    @property
    def ready(self):
        return self._passed(self.finishes_at)

    @property
    def running(self):
        return self._passed(self.starts_at) and not self.ready


class FakeExecutor:
    """A clock, the futures of a map, and the one Lithops call the wait makes."""

    def __init__(self, config, functions=()):
        self.config = config
        self.now = 0.0
        self.futures = [Future(self, starts_at, finishes_at) for starts_at, finishes_at in functions]
        self.handed_over_at = None

    def get_result(self, futures):
        self.handed_over_at = self.now
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
    def test_get_result_takes_over_as_soon_as_every_function_has_started(self):
        executor = FakeExecutor(lambda_config(), functions=[(0, 2), (1, 5)])
        assert run(executor, window=10) == [2, 5]
        assert executor.handed_over_at == 1

    def test_a_started_function_may_outlast_the_window(self):
        # Lithops' own timeout watches a function that has started
        executor = FakeExecutor(lambda_config(), functions=[(0, 100)])
        assert run(executor, window=10) == [100]
        assert executor.handed_over_at == 0

    def test_the_wait_ends_when_functions_never_start(self):
        executor = FakeExecutor(lambda_config(), functions=[(0, 1), (None, None), (None, None)])
        with pytest.raises(FunctionsTimedOut, match="2 of 3 functions never started, and no function started or finished in the last 10 s"):
            run(executor, window=10)
        assert executor.now == pytest.approx(12, abs=1)
        assert executor.handed_over_at is None

    def test_starts_and_finishes_both_keep_the_wait_alive(self):
        # throttled maps, whose functions start late: the whole wait exceeds
        # the window, but no gap without a start or a finish does
        starting = FakeExecutor(lambda_config(), functions=[(0, 30), (8, 30), (16, 30)])
        assert run(starting, window=10) == [30, 30, 30]
        assert starting.handed_over_at == 16
        finishing = FakeExecutor(lambda_config(), functions=[(0, 9), (18, 20)])
        assert run(finishing, window=10) == [9, 20]
        assert finishing.handed_over_at == 18

    def test_a_running_function_alone_is_no_progress(self):
        executor = FakeExecutor(lambda_config(), functions=[(0, None), (18, 20)])
        with pytest.raises(FunctionsTimedOut, match="1 of 2 functions never started"):
            run(executor, window=10)

    def test_a_zero_window_waits_as_get_result_does(self):
        executor = FakeExecutor(lambda_config(), functions=[(None, None)])
        assert run(executor, window=0) == [None]
        assert executor.handed_over_at == 0

    def test_without_a_declared_timeout_the_wait_is_unbounded(self):
        executor = FakeExecutor({"lithops": {"backend": "localhost"}, "localhost": {}}, functions=[(None, None)])
        assert run(executor, window=None) == [None]
        assert executor.handed_over_at == 0


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
