import scippnexus as snx


def load(filepath, *paths: str):
    with snx.File(filepath, 'r') as f:
        for path in paths:
            yield f[path][()]
