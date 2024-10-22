import importlib.metadata

from .detector import providers

try:
    __version__ = importlib.metadata.version("essspectroscopy")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib


__all__ = ['providers']
