# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib
from typing import NewType, Optional

import gemmi
import numpy as np
import pandas as pd
import sciline as sl
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
DEFAULT_WAVELENGTH_COORD_NAME = "wavelength"

IntensityColumnName = NewType("IntensityColumnName", str)
"""The name of the intensity column in the mtz file."""
DEFAULT_INTENSITY_COLUMN_NAME = IntensityColumnName("I")

StdDevColumnName = NewType("StdDevColumnName", str)
"""The name of the standard uncertainty of intensity column in the mtz file."""
DEFAULT_STD_DEV_COLUMN_NAME = StdDevColumnName("SIGI")

# Computed types
RawMtz = NewType("RawMtz", gemmi.Mtz)
"""The mtz file as a gemmi object"""
RawMtzDataFrame = NewType("RawMtzDataFrame", pd.DataFrame)
"""The raw mtz dataframe."""
SpaceGroup = NewType("SpaceGroup", gemmi.SpaceGroup)
"""The space group."""
ReciprocalAsymmetricUnit = NewType("ReciprocalAsymmetricUnit", gemmi.ReciprocalAsu)
"""The reciprocal asymmetric unit."""
MergedMtzDataFrame = NewType("MergedMtzDataFrame", pd.DataFrame)
"""The merged mtz dataframe with derived columns."""
NMXMtzDataFrame = NewType("NMXMtzDataFrame", pd.DataFrame)
"""The processed mtz dataframe with derived columns."""
NMXMtzDataArray = NewType("NMXMtzDataArray", sc.DataArray)


def read_mtz_file(file_path: MTZFilePath) -> RawMtz:
    """read mtz file"""

    return RawMtz(gemmi.read_mtz_file(file_path.as_posix()))


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

    return RawMtzDataFrame(
        pd.DataFrame(  # Recommended in the gemmi documentation.
            data=np.array(mtz, copy=False), columns=mtz.column_labels()
        )
    )


def process_single_mtz_to_dataframe(
    mtz: RawMtz,
    wavelength_column_name: WavelengthColumnName = DEFAULT_WAVELENGTH_COLUMN_NAME,
    intensity_column_name: IntensityColumnName = DEFAULT_INTENSITY_COLUMN_NAME,
    intensity_sig_col_name: StdDevColumnName = DEFAULT_STD_DEV_COLUMN_NAME,
) -> RawMtzDataFrame:
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
    mtz_df[DEFAULT_WAVELENGTH_COORD_NAME] = orig_df[wavelength_column_name]
    mtz_df[DEFAULT_INTENSITY_COLUMN_NAME] = orig_df[intensity_column_name]
    mtz_df[DEFAULT_STD_DEV_COLUMN_NAME] = orig_df[intensity_sig_col_name]
    # Keep other columns
    for column in [col for col in orig_df.columns if col not in mtz_df]:
        mtz_df[column] = orig_df[column]

    return RawMtzDataFrame(mtz_df)


def get_space_group(
    mtzs: sl.Series[MTZFileIndex, RawMtz],
    spacegroup_desc: Optional[SpaceGroupDesc] = None,
) -> SpaceGroup:
    """Retrieves spacegroup from file or uses parameter.

    Manually provided space group description is prioritized over
    space group descriptions found in the mtz files.
    Spacegroup is always expected in any MTZ files, but it may be missing.

    Parameters
    ----------
    mtzs:
        A series of raw mtz datasets.

    spacegroup_desc:
        The space group description to use if not found in the mtz files.
        If None, :attr:`~DEFAULT_SPACE_GROUP_DESC` is used.

    Returns
    -------
    SpaceGroup
        The space group.

    Raises
    ------
    ValueError
        If multiple or no space groups are found
        but space group description is not provided.

    """
    space_groups = {
        sgrp.short_name(): sgrp
        for mtz in mtzs.values()
        if (sgrp := mtz.spacegroup) is not None
    }
    if spacegroup_desc is not None:  # Use the provided space group description
        return SpaceGroup(gemmi.SpaceGroup(spacegroup_desc))
    elif len(space_groups) > 1:
        raise ValueError(f"Multiple space groups found: {space_groups}")
    elif len(space_groups) == 1:
        return SpaceGroup(space_groups.popitem()[1])
    else:
        raise ValueError(
            "No space group found and no space group description provided."
        )


def get_reciprocal_asu(spacegroup: SpaceGroup) -> ReciprocalAsymmetricUnit:
    """Returns the reciprocal asymmetric unit from the space group."""

    return ReciprocalAsymmetricUnit(gemmi.ReciprocalAsu(spacegroup))


def merge_mtz_dataframes(
    mtz_dfs: sl.Series[MTZFileIndex, RawMtzDataFrame],
) -> MergedMtzDataFrame:
    """Merge multiple mtz dataframes into one."""

    return MergedMtzDataFrame(pd.concat(mtz_dfs.values(), ignore_index=True))


def process_merged_mtz_dataframe(
    *,
    merged_mtz_df: MergedMtzDataFrame,
    reciprocal_asu: ReciprocalAsymmetricUnit,
    sg: SpaceGroup,
) -> NMXMtzDataFrame:
    """Modify/Add columns of the shallow copy of a merged mtz dataframes.

    This method must be called after merging multiple mtz dataframes.
    """
    merged_df = merged_mtz_df.copy(deep=False)

    def _reciprocal_asu(row: pd.Series) -> list[int]:
        """Converts miller indices(HKL) to ASU indices."""

        return reciprocal_asu.to_asu(row["hkl"], sg.operations())[0]

    merged_df["hkl_asu"] = merged_df.apply(_reciprocal_asu, axis=1)
    # Unpack the indices for later.
    merged_df[["H_ASU", "K_ASU", "L_ASU"]] = pd.DataFrame(
        merged_df["hkl_asu"].to_list(), index=merged_df.index
    )

    return NMXMtzDataFrame(merged_df)


def _join_variables(*vars: sc.Variable, splitter: str = " ") -> sc.Variable:
    """Joins multiple integer dtype variables into a single string dtype variable.

    Parameters
    ----------
    vars:
        The integer dtype variables to join with same dimensions and length.

    splitter:
        The string to join the variables.

    Returns
    -------
    :
        The joined variable. It keeps the dimensions of the input variables.
        But it drops the units since the output is a string.

    Raises
    ------
    ValueError
        If the input variables have different dimensions or lengths.

    """
    # Check if all variables are integer
    if not all(var.dtype == int for var in vars):
        raise ValueError("All variables must be integer type.")
    # Check if all variables have the same dimensions
    dims = set(var.dim for var in vars)
    if len(dims) != 1:
        raise ValueError("All variables must have the same dimensions.")
    # Check if all variables have the same length
    lengths = set(len(var.values) for var in vars)
    if len(lengths) != 1:
        raise ValueError("All variables must have the same length.")

    return sc.array(
        dims=dims,
        values=[
            splitter.join(str(val) for val in row)
            for row in zip(*(var.values for var in vars))
        ],
    )


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

        Following coordinates are dropped from the dataframe:

        - ``hkl``: The miller indices as a list of integers.
                   There is no dtype that can represent this in scipp.

        Following coordinates are modified:

        - ``hkl_asu``: The miller indices as a string.
                       This coordinate will be used to derive estimated scale factors.
                       It is modified to have a string dtype
                       as the same reason as why ``hkl`` coordinate is dropped.

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
    # Temporarily, we will join them into a single string.
    # It is done on the scipp variable instead of the dataframe
    # since columns with string dtype are converted to PyObject dtype
    # instead of string by `from_pandas_dataframe`.
    nmx_mtz_ds = nmx_mtz_ds.drop_coords(["hkl", "hkl_asu"])
    nmx_mtz_ds.coords["hkl_asu"] = _join_variables(
        *(nmx_mtz_ds.coords[f"{idx_desc}_ASU"] for idx_desc in "HKL")
    )
    # Add units
    nmx_mtz_ds.coords[DEFAULT_WAVELENGTH_COORD_NAME].unit = sc.units.angstrom
    for key in nmx_mtz_ds.keys():
        nmx_mtz_ds[key].unit = sc.units.dimensionless

    # Add variances
    nmx_mtz_da = nmx_mtz_ds[DEFAULT_INTENSITY_COLUMN_NAME].copy(deep=False)
    nmx_mtz_da.variances = nmx_mtz_ds[DEFAULT_STD_DEV_COLUMN_NAME].data ** 2

    # Return DataArray
    return NMXMtzDataArray(nmx_mtz_da)


mtz_io_providers = (
    read_mtz_file,
    process_single_mtz_to_dataframe,
    get_space_group,
    get_reciprocal_asu,
    merge_mtz_dataframes,
    process_merged_mtz_dataframe,
    nmx_mtz_dataframe_to_scipp_dataarray,
)
"""The providers related to the MTZ IO."""

mtz_io_params = {
    WavelengthColumnName: DEFAULT_WAVELENGTH_COLUMN_NAME,
    IntensityColumnName: DEFAULT_INTENSITY_COLUMN_NAME,
    StdDevColumnName: DEFAULT_STD_DEV_COLUMN_NAME,
}
"""The parameters related to the MTZ IO."""
