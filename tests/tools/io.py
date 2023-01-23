import os
import tempfile
from contextlib import contextmanager


@contextmanager
def file_location(file_name):
    """
    Make a file in a temp directory.
    Attempts to remove the file
    at that location following the end of the context.
    """
    tempdir = tempfile.gettempdir()
    f = os.path.join(tempdir, file_name)
    yield f
    try:
        os.remove(f)
    except OSError:
        pass
