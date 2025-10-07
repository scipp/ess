# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib


class Registry:
    def __init__(
        self,
        instrument: str,
        files: dict[str, str],
        version: str,
        retry_if_failed: int = 3,
    ):
        import pooch

        self._registry = pooch.create(
            path=pooch.os_cache(f'ess/{instrument}'),
            env=f'ESS_{instrument.upper()}_DATA_DIR',
            base_url=f'https://public.esss.dk/groups/scipp/ess/{instrument}/{version}/',
            version=version,
            retry_if_failed=retry_if_failed,
            registry=files,
        )

    def __contains__(self, key):
        return key in self._registry.registry

    def __call__(self, name: str, unzip: bool = False) -> pathlib.Path:
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

        if unzip:
            path = self._registry.fetch(name, processor=pooch.Unzip())[0]
        else:
            path = self._registry.fetch(name)
        return pathlib.Path(path)


_registry = Registry(
    instrument='imaging',
    files={
        'siemens_star.tiff': 'md5:0ba27c2daf745338959f5156a3b0a2c0',
        'resolving_power_test_target.tiff': 'md5:a5d414603797f4cc02fe7b2ae4d7aa48',
        # Measurements that Søren Schmidt (imaging IDS 2025) made at J-PARC.
        "siemens-star-measured.h5": "md5:8e333d36c7c102f474b2b66cb785f5e8",
        "siemens-star-openbeam.h5": "md5:ee429b2c247aeaafb0ef3ca4171f2e6a",
    },
    version="1",
)


def siemens_star_path() -> pathlib.Path:
    """
    Return the path to the Siemens star test image.
    """

    return _registry('siemens_star.tiff')


def resolving_power_test_target_path() -> pathlib.Path:
    """
    Return the path to the resolving power test target image.
    """

    return _registry('resolving_power_test_target.tiff')


def jparc_siemens_star_measured_path() -> pathlib.Path:
    """
    Return the path to the Siemens star test image measured at J-PARC.
    """

    return _registry('siemens-star-measured.h5')


def jparc_siemens_star_openbeam_path() -> pathlib.Path:
    """
    Return the path to the Siemens star open beam image measured at J-PARC.
    """

    return _registry('siemens-star-openbeam.h5')
