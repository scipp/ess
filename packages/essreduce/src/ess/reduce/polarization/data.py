# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
_version = "1"


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/polarization"),
        env="ESS_AMOR_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/polarization/{version}/",
        version=_version,
        registry={
            "f_drabkin_reb.csv": "md5:cf20d53ae4af7b337d06fb84ac353994",
        },
    )


_pooch = _make_pooch()


def example_polarization_efficiency_table() -> str:
    return _pooch.fetch("f_drabkin_reb.csv")


__all__ = ["example_polarization_efficiency_table"]
