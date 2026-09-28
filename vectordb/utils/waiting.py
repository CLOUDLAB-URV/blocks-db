"""Wait for the functions of a map without waiting for ever.

The backend kills a function that outruns its timeout and Lithops reports
the error. A function that dies before it reports a start is never watched,
and the wait for it would go on for ever. ``collect`` polls the futures
instead and gives up when no function has finished for longer than a
function is allowed to run: while functions keep finishing, however slowly,
the wait continues.
"""

import time

from lithops.wait import ALWAYS

WAIT_MARGIN_SEC = 60
POLL_SEC = 0.1


class FunctionsTimedOut(TimeoutError):
    """No function finished for longer than a function is allowed to run."""


def inactivity_window(fexec, margin_sec=WAIT_MARGIN_SEC):
    """Seconds without any function finishing after which ``collect`` gives
    up: the backend's function timeout plus a margin, or None when the
    configuration declares no timeout at all."""
    backend = fexec.config["lithops"]["backend"]
    settings = fexec.config.get(backend) or {}
    limit = settings.get("runtime_timeout") or fexec.config["lithops"].get("execution_timeout")
    return int(limit) + margin_sec if limit else None


def collect(fexec, futures, window=None, poll_sec=POLL_SEC, clock=time.monotonic, sleep=time.sleep):
    """Return the results of ``futures``.

    ``window`` is the number of seconds without progress after which the
    wait ends with ``FunctionsTimedOut``: None derives it from the backend,
    0 waits for ever as ``get_result`` does. The polling uses Lithops' own
    status check and no signals, so it works from any thread.
    """
    if window is None:
        window = inactivity_window(fexec)
    if not window:
        return fexec.get_result(futures)

    done = sum(1 for future in futures if future.done)
    last_progress = clock()
    while done < len(futures):
        fexec.wait(futures, return_when=ALWAYS, show_progressbar=False)
        now_done = sum(1 for future in futures if future.done)
        if now_done > done:
            done, last_progress = now_done, clock()
        elif clock() - last_progress > window:
            raise FunctionsTimedOut(
                f"{len(futures) - done} of {len(futures)} functions did not finish, and none"
                f" finished in the last {window} s, longer than a function may run."
                " A function that dies before it reports a start is not seen by the"
                " backend's timeout; look for it in the backend's logs."
            )
        else:
            sleep(poll_sec)
    return fexec.get_result(futures)
