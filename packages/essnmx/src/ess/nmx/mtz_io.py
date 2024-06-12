# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib
from typing import NewType

import gemmi
import numpy as np
import pandas as pd
import scipp as sc

# Index types for param table.
MTZFileIndex = NewType("MTZFileIndex", int)
"""The index of the mtz file when iterating over multiple mtz files."""

# User defined or configurable types
MTZFilePath = NewType("MTZFilePath", pathlib.Path)
"""Path to the mtz file"""
SpaceGroupDesc = NewType("SpaceGroupDesc", str)
"""The space group description. e.g. 'P 21 21 21'"""
DEFAULT_SPACE_GROUP_DESC = SpaceGroupDesc("P 21 21 21")
"""The default space group description to use if not found in the mtz files."""

# Custom column names
WavelengthColumnName = NewType("WavelengthColumnName", str)
"""The name of the wavelength column in the mtz file."""
DEFAULT_WAVELENGTH_COLUMN_NAME = WavelengthColumnName("LAMBDA")

IntensityColumnName = NewType("IntensityColumnName", str)
"""The name of the intensity column in the mtz file."""
DEFAULT_INTENSITY_COLUMN_NAME = IntensityColumnName("I")

StdDevColumnName = NewType("StdDevColumnName", str)
"""The name of the standard uncertainty of intensity column in the mtz file."""
DEFAULT_STD_DEV_COLUMN_NAME = StdDevColumnName("SIGI")

# Computed types
MtzDataFrame = NewType("MtzDataFrame", pd.DataFrame)
"""The raw mtz dataframe."""
NMXMtzDataFrame = NewType("NMXMtzDataFrame", pd.DataFrame)
"""The processed mtz dataframe with derived columns."""
NMXMtzDataArray = NewType("NMXMtzDataArray", sc.DataArray)


def read_mtz_file(file_path: MTZFilePath) -> gemmi.Mtz:
    """read mtz file"""

    return gemmi.read_mtz_file(file_path.as_posix())


def mtz_to_pandas(mtz: gemmi.Mtz) -> pd.DataFrame:
    """Converts the mtz file to a pandas dataframe.

    It is equivalent to the following code:

    .. code-block:: python

        import numpy as np
        import pandas as pd

        data = np.array(mtz, copy=False)
        columns = mtz.column_labels()
        return pd.DataFrame(data, columns=columns)

    It is recommended in the gemmi documentation.

    """

    return pd.DataFrame(  # Recommended in the gemmi documentation.
        data=np.array(mtz, copy=False), columns=mtz.column_labels()
    )


def process_single_mtz_to_dataframe(
    mtz: gemmi.Mtz,
    wavelength_column_name: WavelengthColumnName = DEFAULT_WAVELENGTH_COLUMN_NAME,
    intensity_column_name: IntensityColumnName = DEFAULT_INTENSITY_COLUMN_NAME,
    intensity_sig_col_name: StdDevColumnName = DEFAULT_STD_DEV_COLUMN_NAME,
) -> MtzDataFrame:
    """Select and derive columns from the original ``MtzDataFrame``.

    Parameters
    ----------
    mtz:
        The raw mtz dataset.

    wavelength_column_name:
        The name of the wavelength column in the mtz file.

    intensity_column_name:
        The name of the intensity column in the mtz file.

    intensity_sig_col_name:
        The name of the standard uncertainty of intensity column in the mtz file.

    Returns
    -------
    :
        The new mtz dataframe with derived and renamed columns.

        The derived columns are:

        - ``hkl``: The miller indices as a list of integers.
        - ``d``: The d-spacing calculated from the miller indices.
                 :math:``\\dfrac{2}{d^{2}} = \\dfrac{\\sin^2(\\theta)}{\\lambda^2}``
        - ``resolution``: The resolution calculated from the d-spacing.

        For consistent names of columns/coordinates, the following columns are renamed:

        - ``wavelength_column_name`` -> ``'wavelength'``
        - ``intensity_column_name`` -> ``'I'``
        - ``intensity_sig_col_name`` -> ``'SIGI'``

        Other columns are kept as they are.

    Notes
    -----
    :class:`pandas.DataFrame` is used from loading to merging,
    but :class:`gemmi.Mtz` has :func:`gemmi.Mtz:calculate_d`
    that can derive the ``d`` using ``HKL``.
    This part of the method must be called on each mtz file separately.

    """
    from .mtz_io import mtz_to_pandas

    orig_df = mtz_to_pandas(mtz)
    mtz_df = pd.DataFrame()

    # HKL should always be integer.
    mtz_df[["H", "K", "L"]] = orig_df[["H", "K", "L"]].astype(int)
    mtz_df["hkl"] = mtz_df[["H", "K", "L"]].values.tolist()

    def _calculate_d(row: pd.Series) -> float:
        return mtz.get_cell().calculate_d(row["hkl"])

    mtz_df["d"] = mtz_df.apply(_calculate_d, axis=1)
    mtz_df["resolution"] = (1 / mtz_df["d"]) ** 2 / 4
    mtz_df["wavelength"] = orig_df[wavelength_column_name]
    mtz_df[DEFAULT_INTENSITY_COLUMN_NAME] = orig_df[intensity_column_name]
    mtz_df[DEFAULT_STD_DEV_COLUMN_NAME] = orig_df[intensity_sig_col_name]
    # Keep other columns
    for column in [col for col in orig_df.columns if col not in mtz_df]:
        mtz_df[column] = orig_df[column]

    return MtzDataFrame(mtz_df)


def get_space_group_from_description(desc: SpaceGroupDesc) -> gemmi.SpaceGroup:
    """Retrieves spacegroup from parameter.

    Parameters
    ----------
    desc:
        The space group description to use if not found in the mtz files.

    Returns
    -------
    :
        The space group.
    """
    return gemmi.SpaceGroup(desc)


def get_space_group_from_mtz(mtz: gemmi.Mtz) -> gemmi.SpaceGroup | None:
    """Retrieves spacegroup from file.

    Spacegroup is always expected in any MTZ files, but it may be missing.

    Parameters
    ----------
    mtz:
        Raw mtz dataset.

    Returns
    -------
    :
        The space group, or None if not found.
    """
    return mtz.spacegroup


def get_unique_space_group(*spacegroups: gemmi.SpaceGroup | None) -> gemmi.SpaceGroup:
    """Retrieves the unique space group from multiple space groups.

    Parameters
    ----------
    spacegroups:
        The space groups to check.

    Returns
    -------
    :
        The unique space group.

    Raises
    ------
    ValueError:
        If there are multiple space groups.
    """
    spacegroups = [sgrp for sgrp in spacegroups if sgrp is not None]
    if len(spacegroups) == 0:
        raise ValueError("No space group found.")
    first = spacegroups[0]
    if all(sgrp == first for sgrp in spacegroups):
        return first
    raise ValueError(f"Multiple space groups found: {spacegroups}")


def get_reciprocal_asu(spacegroup: gemmi.SpaceGroup) -> gemmi.ReciprocalAsu:
    """Returns the reciprocal asymmetric unit from the space group."""

    return gemmi.ReciprocalAsu(spacegroup)


def merge_mtz_dataframes(*mtz_dfs: MtzDataFrame) -> MtzDataFrame:
    """Merge multiple mtz dataframes into one."""

    return MtzDataFrame(pd.concat(mtz_dfs, ignore_index=True))


def process_mtz_dataframe(
    *,
    mtz_df: MtzDataFrame,
    reciprocal_asu: gemmi.ReciprocalAsu,
    sg: gemmi.SpaceGroup,
) -> NMXMtzDataFrame:
    """Modify/Add columns of the shallow copy of a mtz dataframe.

    This method must be called after merging multiple mtz dataframe.
    """
    df = mtz_df.copy(deep=False)

    def _reciprocal_asu(row: pd.Series) -> list[int]:
        """Converts miller indices(HKL) to ASU indices."""

        return reciprocal_asu.to_asu(row["hkl"], sg.operations())[0]

    df["hkl_asu"] = df.apply(_reciprocal_asu, axis=1)
    # Unpack the indices for later.
    df[["H_ASU", "K_ASU", "L_ASU"]] = pd.DataFrame(
        df["hkl_asu"].to_list(), index=df.index
    )

    return NMXMtzDataFrame(df)


def nmx_mtz_dataframe_to_scipp_dataarray(
    nmx_mtz_df: NMXMtzDataFrame,
) -> NMXMtzDataArray:
    """Converts the processed mtz dataframe to a scipp dataarray.

    The intensity, with column name :attr:`~DEFAULT_INTENSITY_COLUMN_NAME`
    becomes the data and the standard uncertainty of intensity,
    with column name :attr:`~DEFAULT_SIGMA_INTENSITY_COLUMN_NAME`
    becomes the variances of the data.

    Parameters
    ----------
    nmx_mtz_df:
        The merged and processed mtz dataframe.

    Returns
    -------
    :
        The scipp dataarray with the intensity and variances.
        The ``I`` column becomes the data and the
        squared ``SIGI`` column becomes the variances.
        Therefore they are not in the coordinates.

        Following coordinates are modified:

        - ``hkl``: The miller indices as a string.
                   It is modified to have a string dtype
                   since is no dtype that can represent this in scipp.

        - ``hkl_asu``: The asymmetric unit of miller indices as a string.
                       This coordinate will be used to derive estimated scale factors.
                       It is modified to have a string dtype
                       as the same reason as why ``hkl`` coordinate is modified.

        Zero or negative intensities are removed from the dataarray.
        It can happen due to the post-processing of the data,
        e.g. background subtraction.

    """
    from scipp.compat.pandas_compat import from_pandas_dataframe, parse_bracket_header

    to_scipp = nmx_mtz_df.copy(deep=False)
    # Convert to scipp Dataset
    nmx_mtz_ds = from_pandas_dataframe(
        to_scipp,
        data_columns=[
            DEFAULT_INTENSITY_COLUMN_NAME,
            DEFAULT_STD_DEV_COLUMN_NAME,
        ],
        header_parser=parse_bracket_header,
    )
    # Pop the indices columns.
    # TODO: We can put them back once we support tuple[int] dtype.
    # See https://github.com/scipp/scipp/issues/3046 for more details.
    # Temporarily, we will manually convert them to a string.
    # It is done on the scipp variable instead of the dataframe
    # since columns with string dtype are converted to PyObject dtype
    # instead of string by `from_pandas_dataframe`.
    for indices_name in ("hkl", "hkl_asu"):
        nmx_mtz_ds.coords[indices_name] = sc.array(
            dims=nmx_mtz_ds.coords[indices_name].dims,
            values=nmx_mtz_df[indices_name].astype(str).tolist(),
            # `astype`` is not enough to convert the dtype to string.
            # The result of `astype` will have `PyObject` as a dtype.
        )
    # Add units
    nmx_mtz_ds.coords["wavelength"].unit = sc.units.angstrom
    for key in nmx_mtz_ds.keys():
        nmx_mtz_ds[key].unit = sc.units.dimensionless

    # Add variances
    nmx_mtz_da = nmx_mtz_ds[DEFAULT_INTENSITY_COLUMN_NAME].copy(deep=False)
    nmx_mtz_da.variances = (nmx_mtz_ds[DEFAULT_STD_DEV_COLUMN_NAME].data ** 2).values

    # Return DataArray without negative intensities
    return NMXMtzDataArray(nmx_mtz_da[nmx_mtz_da.data > 0])


providers = (
    read_mtz_file,
    process_single_mtz_to_dataframe,
    # get_space_group_from_description,
    get_space_group_from_mtz,
    get_reciprocal_asu,
    process_mtz_dataframe,
    nmx_mtz_dataframe_to_scipp_dataarray,
)
"""The providers related to the MTZ IO."""

default_parameters = {
    WavelengthColumnName: DEFAULT_WAVELENGTH_COLUMN_NAME,
    IntensityColumnName: DEFAULT_INTENSITY_COLUMN_NAME,
    StdDevColumnName: DEFAULT_STD_DEV_COLUMN_NAME,
}
"""The parameters related to the MTZ IO."""
