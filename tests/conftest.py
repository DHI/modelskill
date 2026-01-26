import gc
import sys
import pytest


@pytest.fixture(autouse=True)
def gc_after_test():
    """Run Python's garbage collection after each test on Windows.

    The test suite currently opens many file handles that are not closed until garbage collection
    runs. This results in flaky test failures on Windows which has a limit of 512 open file handles
    per process. It could be possible to remove this in the future if the problem is solved upstream
    (most likely in MIKE IO). See issue below for details:

    https://github.com/DHI/mikecore-python/issues/41
    """
    if sys.platform.startswith("win"):
        yield
        gc.collect()
    else:
        yield
