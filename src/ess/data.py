# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Dict

__all__ = ['Registry']


class Registry:
    def __init__(self, instrument: str, files: Dict[str, str], version: str = '1'):
        import pooch

        self._registry = pooch.create(
            path=pooch.os_cache(f'ess/{instrument}'),
            env=f'ESS_{instrument.upper()}_DATA_DIR',
            base_url=f'https://public.esss.dk/groups/scipp/ess/{instrument}/'
            + '{version}/',
            version=version,
            registry=files,
        )

    def __getitem__(self, name: str) -> str:
        """
        Get the path to a file in the registry.

        Parameters
        ----------
        name:
            Name of the file to get the path for.
        """
        return self._registry.fetch(name)

    def get_path(self, name: str) -> str:
        """
        Get the path to a file in the registry.
        This is the deprecated way of getting a file path, and is mostly there for
        backwards compatibility.
        Use ``__getitem__`` instead.

        Parameters
        ----------
        name:
            Name of the file to get the path for.
        """
        return self[name]
