"""
timeout_utils.py

signal.alarm()-based timeouts only work reliably when the target code is
pure Python running between bytecode instructions. If the hang is inside a
broad `except Exception` retry loop somewhere in the call stack (which can
silently swallow our alarm-raised exception and just retry), or inside a
C extension that doesn't yield back to the interpreter, the alarm can be
delayed indefinitely or never actually stop anything.

This provides a hard, OS-level timeout instead: run the risky call in a
separate process via `multiprocessing`, and forcibly terminate that process
if it doesn't finish in time. This works regardless of what the target code
is doing internally, since termination happens at the OS process level, not
via a Python exception the target code could catch.
"""
import multiprocessing as mp


def _worker(fn, args, kwargs, queue):
    try:
        result = fn(*args, **kwargs)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("error", e))


def run_with_hard_timeout(fn, args=(), kwargs=None, timeout=60):
    """
    Runs fn(*args, **kwargs) in a subprocess. Returns (True, result) on
    success, or (False, None) if it didn't finish within `timeout` seconds
    (the subprocess is forcibly killed in that case).
    """
    kwargs = kwargs or {}
    queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(fn, args, kwargs, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()  # SIGKILL if terminate() didn't work
            proc.join()
        return False, None

    if not queue.empty():
        status, payload = queue.get()
        if status == "ok":
            return True, payload
        else:
            # the function itself raised a real (non-timeout) exception;
            # re-raise so existing except IndexError / RateLimitError
            # handling in eval_dataset still works as before
            raise payload
    return False, None
