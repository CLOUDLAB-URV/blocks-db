"""The wait for the functions, against Lithops itself: it returns the results
and logs as get_result does, leaves a started function to Lithops' own
timeout, and ends, from any thread, when functions never start."""

import logging
import os
import shlex
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from vectordb.utils.waiting import FunctionsTimedOut, collect

pytestmark = pytest.mark.integration


def nap(seconds):
    time.sleep(seconds)
    return seconds


def fail_or_nap(seconds):
    time.sleep(seconds)
    if seconds < 1:
        raise ValueError("the function failed")
    return seconds


def blob(size):
    return b"x" * size


def lose_the_worker(test_process):
    worker = os.getppid()  # the process that would report the end
    if worker != test_process:
        os.kill(worker, signal.SIGKILL)


def never_starting_runtime(directory):
    """A runtime that answers as the Python interpreter, except that every
    task exits before it starts, so no function reports a start."""
    runtime = directory / "python-without-tasks"
    runtime.write_text(f'#!/bin/sh\n[ "$2" = run_job ] && exit 1\nexec {shlex.quote(sys.executable)} "$@"\n')
    runtime.chmod(0o755)
    return str(runtime)


def test_the_results_come_back_when_the_functions_finish(lithops_localhost):
    import lithops

    fexec = lithops.FunctionExecutor()
    futures = fexec.map(nap, [0.2, 0.4])
    assert collect(fexec, futures, window=10) == [0.2, 0.4]


def test_the_wait_logs_once_as_get_result_does(lithops_localhost, caplog, monkeypatch):
    # Lithops logs at INFO by default; the wait must not add a line of its own
    import lithops

    fexec = lithops.FunctionExecutor()
    monkeypatch.setattr(logging.getLogger("lithops"), "propagate", True)
    caplog.set_level(logging.INFO, logger="lithops")
    futures = fexec.map(nap, [0.5, 1])
    collect(fexec, futures, window=10)
    assert sum("Waiting for" in record.getMessage() for record in caplog.records) == 1


def test_a_result_kept_out_of_the_status_comes_back_too(lithops_localhost):
    # Lithops stores a result of 8 KB or more apart from the status of its function
    import lithops

    fexec = lithops.FunctionExecutor()
    futures = fexec.map(blob, [20_000])
    assert collect(fexec, futures, window=3) == [b"x" * 20_000]


def test_a_failure_is_raised_without_waiting_for_the_other_functions(lithops_localhost):
    import lithops

    fexec = lithops.FunctionExecutor()
    futures = fexec.map(fail_or_nap, [0.5, 10])
    started = time.monotonic()
    with pytest.raises(ValueError, match="the function failed"):
        collect(fexec, futures, window=30)
    assert time.monotonic() - started < 5
    assert all(future.done for future in futures)  # Lithops' cleanup closed them


def test_a_started_function_may_outlast_the_window(lithops_localhost):
    import lithops

    fexec = lithops.FunctionExecutor()
    futures = fexec.map(nap, [3])
    assert collect(fexec, futures, window=2) == [3]


def test_lithops_ends_a_started_function_that_hangs(configure_lithops):
    import lithops

    configure_lithops(lithops={"execution_timeout": 3})
    fexec = lithops.FunctionExecutor()
    futures = fexec.map(nap, [30])
    started = time.monotonic()
    with pytest.raises(TimeoutError) as raised:
        collect(fexec, futures, window=60)
    assert not isinstance(raised.value, FunctionsTimedOut)
    assert time.monotonic() - started < 10


def test_lithops_ends_a_started_function_whose_worker_dies(configure_lithops):
    # no end status ever arrives: the job monitor ends the function on its own
    import lithops

    configure_lithops(lithops={"execution_timeout": 3})
    test_process = os.getpid()
    outcome = {}

    def run():
        fexec = lithops.FunctionExecutor()
        futures = fexec.map(lose_the_worker, [test_process])
        try:
            collect(fexec, futures, window=60)
        except TimeoutError as error:
            outcome["error"] = error

    thread = threading.Thread(target=run, daemon=True)  # a wait that never ends fails the test, not the run
    thread.start()
    thread.join(timeout=30)
    assert "exceeded the execution timeout" in str(outcome.get("error"))
    assert not isinstance(outcome["error"], FunctionsTimedOut)


def test_the_wait_ends_when_the_functions_never_start(configure_lithops, tmp_path):
    import lithops

    configure_lithops(localhost={"runtime": never_starting_runtime(tmp_path)})
    fexec = lithops.FunctionExecutor()
    futures = fexec.map(nap, [0.2, 0.4])
    started = time.monotonic()
    with pytest.raises(FunctionsTimedOut, match="2 of 2 functions never started"):
        collect(fexec, futures, window=2)
    assert 2 < time.monotonic() - started < 5


def test_the_wait_ends_from_a_thread_too(configure_lithops, tmp_path):
    # the timeout of Lithops' own get_result relies on SIGALRM, which only
    # works in the main thread; the wait does not
    import lithops

    configure_lithops(localhost={"runtime": never_starting_runtime(tmp_path)})
    outcome = {}

    def run():
        fexec = lithops.FunctionExecutor()
        futures = fexec.map(nap, [0.2])
        try:
            collect(fexec, futures, window=2)
        except FunctionsTimedOut as error:
            outcome["error"] = error

    thread = threading.Thread(target=run, daemon=True)  # a wait that never ends fails the test, not the run
    thread.start()
    thread.join(timeout=30)
    assert isinstance(outcome.get("error"), FunctionsTimedOut)


def test_without_a_window_functions_that_never_start_are_awaited_forever(configure_lithops, tmp_path):
    # the failure the window exists for: get_result alone never returns
    configure_lithops(localhost={"runtime": never_starting_runtime(tmp_path)})
    script = textwrap.dedent("""
        import lithops
        from vectordb.utils.waiting import collect

        def nap(seconds):
            return seconds

        fexec = lithops.FunctionExecutor()
        futures = fexec.map(nap, [0.2])
        print("the wait starts", flush=True)
        collect(fexec, futures, window=0)
    """)
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))}
    with pytest.raises(subprocess.TimeoutExpired) as stopped:
        subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, timeout=6)
    assert b"the wait starts" in stopped.value.stdout
