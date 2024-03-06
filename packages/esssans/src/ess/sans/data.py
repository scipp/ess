# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Dict


class Registry:
    def __init__(self, instrument: str, files: Dict[str, str], version: str):
        import pooch

        self._registry = pooch.create(
            path=pooch.os_cache(f'ess/{instrument}'),
            env=f'ESS_{instrument.upper()}_DATA_DIR',
            base_url=f'https://public.esss.dk/groups/scipp/ess/{instrument}/'
            + '{version}/',
            version=version,
            registry=files,
        )

    def __contains__(self, key):
        return key in self._registry.registry

    def get_path(self, name: str, unzip: bool = False) -> str:
        """
        Get the path to a file in the registry.

        Parameters
        ----------
        name:
            Name of the file to get the path for.
        unzip:
            If `True`, unzip the file before returning the path.
        """
        import pooch

        return self._registry.fetch(name, processor=pooch.Unzip() if unzip else None)


__all__ = ['Registry']
