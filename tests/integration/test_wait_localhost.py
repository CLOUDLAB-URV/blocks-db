"""The wait for the functions, against Lithops itself: it ends when nothing
finishes, from any thread, and returns the results otherwise."""

import threading
import time

import pytest

from vectordb.utils.waiting import FunctionsTimedOut, collect

pytestmark = pytest.mark.integration


def nap(seconds):
    time.sleep(seconds)
    return seconds


def test_the_results_come_back_when_the_functions_finish(lithops_localhost):
    import lithops

    fexec = lithops.FunctionExecutor()
    futures = fexec.map(nap, [0.2, 0.4])
    assert collect(fexec, futures, window=10, poll_sec=0.05) == [0.2, 0.4]


def test_the_wait_ends_when_nothing_finishes_within_the_window(lithops_localhost):
    import lithops

    fexec = lithops.FunctionExecutor()
    futures = fexec.map(nap, [3])
    started = time.monotonic()
    with pytest.raises(FunctionsTimedOut, match="1 of 1 functions did not finish"):
        collect(fexec, futures, window=1, poll_sec=0.05)
    assert time.monotonic() - started < 3
    fexec.get_result(futures)  # let the function end before the test does


def test_the_wait_works_from_a_thread_too(lithops_localhost):
    # the timeout of Lithops' own get_result relies on SIGALRM, which only
    # works in the main thread; the polling does not
    import lithops

    outcome = {}

    def run():
        fexec = lithops.FunctionExecutor()
        futures = fexec.map(nap, [3])
        try:
            collect(fexec, futures, window=1, poll_sec=0.05)
        except FunctionsTimedOut as error:
            outcome["error"] = error
        fexec.get_result(futures)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=30)
    assert isinstance(outcome.get("error"), FunctionsTimedOut)
