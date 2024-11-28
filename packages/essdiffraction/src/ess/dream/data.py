# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with DREAM."""

_version = "1"

__all__ = ["get_path"]


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/dream"),
        env="ESS_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/dream/{version}/",
        version=_version,
        registry={
            "data_dream_with_sectors.csv.zip": "md5:52ae6eb3705e5e54306a001bc0ae85d8",
            "data_dream0_new_hkl_Si_pwd.csv.zip": "md5:d0ae518dc1b943bb817b3d18c354ed01",  # noqa: E501
            "DREAM_nexus_sorted-2023-12-07.nxs": "md5:22824e14f6eb950d24a720b2a0e2cb66",
            "DREAM_simple_pwd_workflow/data_dream_diamond_vana_container_sample_union.csv.zip": "md5:33302d0506b36aab74003b8aed4664cc",  # noqa: E501
            "DREAM_simple_pwd_workflow/data_dream_diamond_vana_container_sample_union_run2.csv.zip": "md5:c7758682f978d162dcb91e47c79abb83",  # noqa: E501
            "DREAM_simple_pwd_workflow/data_dream_vana_container_sample_union.csv.zip": "md5:1e22917b2bb68b5cacfb506b72700a4d",  # noqa: E501
            "DREAM_simple_pwd_workflow/data_dream_vanadium.csv.zip": "md5:e5addfc06768140c76533946433fa2ec",  # noqa: E501
            "DREAM_simple_pwd_workflow/data_dream_vanadium_inc_coh.csv.zip": "md5:39d1a44e248b12966b26f7c2f6c602a2",  # noqa: E501
            "DREAM_simple_pwd_workflow/Cave_TOF_Monitor_diam_in_can.dat": "md5:ef24f4a4186c628574046e6629e31611",  # noqa: E501
            "DREAM_simple_pwd_workflow/Cave_TOF_Monitor_van_can.dat": "md5:e63456c347fb36a362a0b5ae2556b3cf",  # noqa: E501
            "DREAM_simple_pwd_workflow/Cave_TOF_Monitor_vana_inc_coh.dat": "md5:701d66792f20eb283a4ce76bae0c8f8f",  # noqa: E501
        },
    )


_pooch = _make_pooch()


def get_path(name: str, unzip: bool = False) -> str:
    """
    Return the path to a data file bundled with ess.dream.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    import pooch

    return _pooch.fetch(name, processor=pooch.Unzip() if unzip else None)


def simulated_diamond_sample() -> str:
    """Path to a GEANT4 CSV file for a diamond sample.

    **Sample**:

    - Diamond powder
    - radius=0.0059 m
    - yheight=0.05-1e-9 m
    - material
        reflections of powder from "C_Diamond.laz" (McStas file)
        incoherent process with sigma=0.001 barns, packing_factor=1,
        unit cell volume=45.39 angstrom^3
        absorption of 0.062 1/m

    **Container**:

    - radius=0.006 m
    - yheight=0.05 m
    - material
        reflections of powder from "V.laz" (McStas file)
        incoherent process with sigma=4.935 barns, packing factor=1,
        unit cell volume=27.66 angstrom^3
        absorption of 36.73 1/m
    """
    return get_path(
        "DREAM_simple_pwd_workflow/"
        "data_dream_diamond_vana_container_sample_union.csv.zip"
    )


def simulated_vanadium_sample() -> str:
    """Path to a GEANT4 CSV file for a vanadium sample.

    Contains both coherent and incoherent scattering.

    **Sample**:

    - radius=0.006 m
    - yheight=0.01 m
    - material
        reflections of powder from "V.laz" (McStas file)
        incoherent process with sigma=4.935 barns, packing factor=1,
        unit cell volume=27.66 angstrom^3
        absorption of 36.73 1/m
    """
    return get_path("DREAM_simple_pwd_workflow/data_dream_vanadium.csv.zip")


def simulated_vanadium_sample_incoherent() -> str:
    """Path to a GEANT4 CSV file for a vanadium sample with only incoherent scattering.

    **Sample**:

    - outer radius of sample in (x,z) plane=0.006 m
    - vertical dimension of sample (along y)=0.01 m
    - packing factor=1
    """
    return get_path("DREAM_simple_pwd_workflow/data_dream_vanadium.csv.zip")


def simulated_empty_can() -> str:
    """Path to a GEANT4 CSV file for an empty can measurement.

    **Container**:

    - radius=0.006 m
    - yheight=0.05 m
    - material
        reflections of powder from "V.laz" (McStas file)
        incoherent process with sigma=4.935 barns, packing factor=1,
        unit cell volume=27.66 angstrom^3
        absorption of 36.73 1/m
    """
    return get_path(
        "DREAM_simple_pwd_workflow/data_dream_vana_container_sample_union.csv.zip"
    )


def simulated_monitor_diamond_sample() -> str:
    """Path to a Mcstas file for a monitor for the diamond measurement.

    This is the DREAM cave monitor for ``simulated_diamond_sample``.
    """
    return get_path("DREAM_simple_pwd_workflow/Cave_TOF_Monitor_diam_in_can.dat")


def simulated_monitor_vanadium_sample() -> str:
    """Path to a Mcstas file for a monitor for the vanadium measurement.

    This is the DREAM cave monitor for ``simulated_vanadium_sample``.
    """
    return get_path("DREAM_simple_pwd_workflow/Cave_TOF_Monitor_vana_inc_coh.dat")


def simulated_monitor_empty_can() -> str:
    """Path to a Mcstas file for a monitor for the empty can measurement.

    This is the DREAM cave monitor for ``simulated_empty_can``.
    """
    return get_path("DREAM_simple_pwd_workflow/Cave_TOF_Monitor_van_can.dat")
