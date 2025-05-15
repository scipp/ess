# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with DREAM."""

_version = "2"

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
            "DREAM_simple_pwd_workflow/data_dream_vana_container_sample_union.csv.zip": "md5:b8bb7c4bdf74ad5f19342bced8670915",  # noqa: E501
            "DREAM_simple_pwd_workflow/data_dream_vanadium.csv.zip": "md5:e5addfc06768140c76533946433fa2ec",  # noqa: E501
            "DREAM_simple_pwd_workflow/data_dream_vanadium_inc_coh.csv.zip": "md5:39d1a44e248b12966b26f7c2f6c602a2",  # noqa: E501
            "DREAM_simple_pwd_workflow/Cave_TOF_Monitor_diam_in_can.dat": "md5:ef24f4a4186c628574046e6629e31611",  # noqa: E501
            "DREAM_simple_pwd_workflow/Cave_TOF_Monitor_van_can.dat": "md5:2cdef7ad9912652149b7e687381d2e99",  # noqa: E501
            "DREAM_simple_pwd_workflow/Cave_TOF_Monitor_vana_inc_coh.dat": "md5:701d66792f20eb283a4ce76bae0c8f8f",  # noqa: E501
            "DREAM-high-flux-tof-lookup-table.h5": "md5:1b95a359fa7b0d8b4277806ece9bf279",  # noqa: E501
            # Smaller files for unit tests
            "DREAM_simple_pwd_workflow/TEST_data_dream_diamond_vana_container_sample_union.csv.zip": "md5:018a87e0934c1dd0f07a708e9d497891",  # noqa: E501
            "DREAM_simple_pwd_workflow/TEST_data_dream_vana_container_sample_union.csv.zip": "md5:d244126cd8012f9ed186f4e08a19f88d",  # noqa: E501
            "DREAM_simple_pwd_workflow/TEST_data_dream_vanadium.csv.zip": "md5:178f9bea9f35dbdef693e38ff893c258",  # noqa: E501
            "TEST_data_dream0_new_hkl_Si_pwd.csv.zip": "md5:df6c41f4b7b21e129915808f625828f6",  # noqa: E501
            "TEST_data_dream_with_sectors.csv.zip": "md5:2a6b5e40e6b67f6c71b25373bf4b11a1",  # noqa: E501
            # The TEST_DREAM_nexus_sorted-2023-12-07.nxs file was created using the
            # `shrink_nexus.py` script in the `tools` folder at the top level of the
            # `essdiffraction` repository.
            "TEST_DREAM_nexus_sorted-2023-12-07.nxs": "md5:599b426a93c46a7b4b09a874bf288c53",  # noqa: E501
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


def simulated_diamond_sample(*, small: bool = False) -> str:
    """Path to a GEANT4 CSV file for a diamond sample.

    SciCat:

    - PID: ``20.500.12269/d17b7169-b963-4b23-9455-d7332b992184``
    - URL: https://staging.scicat.ess.eu/datasets/20.500.12269%2Fd17b7169-b963-4b23-9455-d7332b992184

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

    Parameters
    ----------
    small:
        If True, return a smaller version of the data file, with randomly selected rows.
        The small version of the file was created using the following code:

        ```python
        import numpy as np
        import pandas as pd

        fname = dream.data.simulated_diamond_sample()
        df = pd.read_csv(fname, sep='\t')
        inds = np.sort(np.random.choice(len(df), 10_000, replace=False))
        df.iloc[inds].to_csv('TEST_' + fname.split('/')[-1], sep='\t', index=False)
        ```

    """
    prefix = "TEST_" if small else ""
    return get_path(
        "DREAM_simple_pwd_workflow/"
        f"{prefix}data_dream_diamond_vana_container_sample_union.csv.zip"
    )


def simulated_vanadium_sample(*, small: bool = False) -> str:
    """Path to a GEANT4 CSV file for a vanadium sample.

    SciCat:

    - PID: ``20.500.12269/9e43a311-dc17-4f0e-a4d4-2341186b5512``
    - URL: https://staging.scicat.ess.eu/datasets/20.500.12269%2F9e43a311-dc17-4f0e-a4d4-2341186b5512

    Contains both coherent and incoherent scattering.

    **Sample**:

    - radius=0.006 m
    - yheight=0.01 m
    - material
        reflections of powder from "V.laz" (McStas file)
        incoherent process with sigma=4.935 barns, packing factor=1,
        unit cell volume=27.66 angstrom^3
        absorption of 36.73 1/m

    Parameters
    ----------
    small:
        If True, return a smaller version of the data file, with randomly selected rows.
        The small version of the file was created using the following code:

        ```python
        import numpy as np
        import pandas as pd

        fname = dream.data.simulated_vanadium_sample()
        df = pd.read_csv(fname, sep='\t')
        inds = np.sort(np.random.choice(len(df), 10_000, replace=False))
        df.iloc[inds].to_csv('TEST_' + fname.split('/')[-1], sep='\t', index=False)
        ```
    """
    prefix = "TEST_" if small else ""
    return get_path(f"DREAM_simple_pwd_workflow/{prefix}data_dream_vanadium.csv.zip")


def simulated_vanadium_sample_incoherent(*, small: bool = False) -> str:
    """Path to a GEANT4 CSV file for a vanadium sample with only incoherent scattering.

    SciCat:

    - PID: ``20.500.12269/b170848e-455e-4c43-8666-68e20f922326``
    - URL: https://staging.scicat.ess.eu/datasets/20.500.12269%2Fb170848e-455e-4c43-8666-68e20f922326

    **Sample**:

    - outer radius of sample in (x,z) plane=0.006 m
    - vertical dimension of sample (along y)=0.01 m
    - packing factor=1

    Parameters
    ----------
    small:
        If True, return a smaller version of the data file, with randomly selected rows.
        The small version of the file was created using the following code:

        ```python
        import numpy as np
        import pandas as pd

        fname = dream.data.simulated_vanadium_sample_incoherent()
        df = pd.read_csv(fname, sep='\t')
        inds = np.sort(np.random.choice(len(df), 10_000, replace=False))
        df.iloc[inds].to_csv('TEST_' + fname.split('/')[-1], sep='\t', index=False)
        ```
    """
    prefix = "TEST_" if small else ""
    return get_path(f"DREAM_simple_pwd_workflow/{prefix}data_dream_vanadium.csv.zip")


def simulated_empty_can(*, small: bool = False) -> str:
    """Path to a GEANT4 CSV file for an empty can measurement.

    SciCat:

    - PID: ``20.500.12269/1a280698-aa5a-4cfb-bc4f-68fcd40462cc``
    - URL: https://staging.scicat.ess.eu/datasets/20.500.12269%2F1a280698-aa5a-4cfb-bc4f-68fcd40462cc

    **Container**:

    - radius=0.006 m
    - yheight=0.05 m
    - material
        reflections of powder from "V.laz" (McStas file)
        incoherent process with sigma=4.935 barns, packing factor=1,
        unit cell volume=27.66 angstrom^3
        absorption of 36.73 1/m

    Parameters
    ----------
    small:
        If True, return a smaller version of the data file, with randomly selected rows.
        The small version of the file was created using the following code:

        ```python
        import numpy as np
        import pandas as pd

        fname = dream.data.simulated_empty_can()
        df = pd.read_csv(fname, sep='\t')
        inds = np.sort(np.random.choice(len(df), 10_000, replace=False))
        df.iloc[inds].to_csv('TEST_' + fname.split('/')[-1], sep='\t', index=False)
        ```
    """
    prefix = "TEST_" if small else ""
    return get_path(
        "DREAM_simple_pwd_workflow/"
        f"{prefix}data_dream_vana_container_sample_union.csv.zip"
    )


def simulated_monitor_diamond_sample() -> str:
    """Path to a Mcstas file for a monitor for the diamond measurement.

    SciCat:

    - PID: ``20.500.12269/d17b7169-b963-4b23-9455-d7332b992184``
    - URL: https://staging.scicat.ess.eu/datasets/20.500.12269%2Fd17b7169-b963-4b23-9455-d7332b992184

    This is the DREAM cave monitor for ``simulated_diamond_sample``.
    """
    return get_path("DREAM_simple_pwd_workflow/Cave_TOF_Monitor_diam_in_can.dat")


def simulated_monitor_vanadium_sample() -> str:
    """Path to a Mcstas file for a monitor for the vanadium measurement.

    SciCat:

    - PID: ``20.500.12269/b170848e-455e-4c43-8666-68e20f922326``
    - URL: https://staging.scicat.ess.eu/datasets/20.500.12269%2Fb170848e-455e-4c43-8666-68e20f922326

    This is the DREAM cave monitor for ``simulated_vanadium_sample``.
    """
    return get_path("DREAM_simple_pwd_workflow/Cave_TOF_Monitor_vana_inc_coh.dat")


def simulated_monitor_empty_can() -> str:
    """Path to a Mcstas file for a monitor for the empty can measurement.

    SciCat:

    - PID: ``20.500.12269/1a280698-aa5a-4cfb-bc4f-68fcd40462cc``
    - URL: https://staging.scicat.ess.eu/datasets/20.500.12269%2F1a280698-aa5a-4cfb-bc4f-68fcd40462cc

    This is the DREAM cave monitor for ``simulated_empty_can``.
    """
    return get_path("DREAM_simple_pwd_workflow/Cave_TOF_Monitor_van_can.dat")


def tof_lookup_table_high_flux() -> str:
    """Path to a HDF5 file containing a lookup table for high-flux ToF.

    The table was created using the ``tof`` package and the chopper settings for the
    DREAM instrument in high-resolution mode.
    Note that the phase of the band-control chopper (BCC) was set to 240 degrees to
    match that of the simulated data (this has since been found to be non-optimal as it
    leads to time overlap between the two frames).

    The notebook that was used to create the table can be found at
    https://github.com/scipp/essdiffraction/blob/main/tools/dream-make-tof-lookup-table.ipynb
    """
    return get_path("DREAM-high-flux-tof-lookup-table.h5")
