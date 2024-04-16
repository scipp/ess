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


# Computed types
RawMtz = NewType("RawMtz", gemmi.Mtz)
"""The mtz file as a gemmi object"""
RawMtzDataFrame = NewType("RawMtzDataFrame", pd.DataFrame)
"""The raw mtz dataframe."""
SpaceGroup = NewType("SpaceGroup", gemmi.SpaceGroup)
"""The space group."""
RapioAsu = NewType("RapioAsu", gemmi.ReciprocalAsu)
"""The reciprocal asymmetric unit."""
MergedMtzDataFrame = NewType("MergedMtzDataFrame", pd.DataFrame)
"""The merged mtz dataframe with derived columns."""
NMXMtzDataFrame = NewType("NMXMtzDataFrame", pd.DataFrame)
"""The reduced mtz dataframe with derived columns."""
NMXMtzDataArray = NewType("NMXMtzDataArray", sc.DataArray)


def read_mtz_file(file_path: MTZFilePath) -> RawMtz:
    """read mtz file"""

    return RawMtz(gemmi.read_mtz_file(file_path.as_posix()))


def mtz_to_pandas(mtz: gemmi.Mtz) -> pd.DataFrame:
    """Converts the mtz file to a pandas dataframe.

    It is equivalent to the following code:
    ```python
    import numpy as np
    import pandas as pd

    data = np.array(mtz, copy=False)
    columns = mtz.column_labels()
    return pd.DataFrame(data, columns=columns)
    ```
    It is recommended in the gemmi documentation.

    """

    return RawMtzDataFrame(
        pd.DataFrame(  # Recommended in the gemmi documentation.
            data=np.array(mtz, copy=False), columns=mtz.column_labels()
        )
    )


def reduce_single_mtz(
    mtz: RawMtz,
    lambda_column_name: WavelengthColumnName = DEFAULT_WAVELENGTH_COLUMN_NAME,
) -> RawMtzDataFrame:
    """Select and derive columns from the original ``MtzDataFrame``.

    Parameters
    ----------
    mtz:
        The raw mtz dataset.

    lambda_column_name:
        The name of the wavelength column in the mtz file.

    Returns
    -------
    :
        The new mtz dataframe with derived columns.
        The derived columns are:

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
    # (2d)^{-2} = \sin^2(\theta)/\lambda^2
    mtz_df["resolution"] = (1 / mtz_df["d"]) ** 2 / 4
    mtz_df["I_div_SIGI"] = orig_df["I"] / orig_df["SIGI"]
    mtz_df[DEFAULT_WAVELENGTH_COLUMN_NAME] = orig_df[lambda_column_name]
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


def get_reciprocal_asu(spacegroup: SpaceGroup) -> RapioAsu:
    """Returns the reciprocal asymmetric unit from the space group."""

    return RapioAsu(gemmi.ReciprocalAsu(spacegroup))


def merge_mtz_dataframes(
    mtz_dfs: sl.Series[MTZFileIndex, RawMtzDataFrame],
) -> MergedMtzDataFrame:
    """Merge multiple mtz dataframes into one."""

    return MergedMtzDataFrame(pd.concat(mtz_dfs.values(), ignore_index=True))


def reduce_merged_mtz_dataframe(
    *,
    merged_mtz_df: MergedMtzDataFrame,
    rapio_asu: RapioAsu,
    sg: SpaceGroup,
) -> NMXMtzDataFrame:
    """Reduces the shallow copy of a merged mtz dataframes.

    This method must be called after merging multiple mtz dataframes.
    """
    merged_df = merged_mtz_df.copy(deep=False)

    def _rapio_asu_to_asu(row: pd.Series) -> list[int]:
        return rapio_asu.to_asu(row["hkl"], sg.operations())[0]

    merged_df["hkl_eq"] = merged_df.apply(_rapio_asu_to_asu, axis=1)

    def unpack_vector(row: pd.Series, *new_names) -> pd.DataFrame:
        return pd.DataFrame(
            {name: [val[i] for val in row] for i, name in enumerate(new_names)}
        )

    # Unpack HKL EQ
    merged_df[["H_EQ", "K_EQ", "L_EQ"]] = unpack_vector(
        merged_df["hkl_eq"], "H_EQ", "K_EQ", "L_EQ"
    )

    return NMXMtzDataFrame(merged_df)


def nmx_mtz_dataframe_to_scipp_dataarray(
    nmx_mtz_df: NMXMtzDataFrame,
) -> NMXMtzDataArray:
    """Converts the reduced mtz dataframe to a scipp dataarray."""
    from scipp.compat.pandas_compat import from_pandas_dataframe, parse_bracket_header

    # Add unit to the name
    to_scipp = nmx_mtz_df.copy(deep=False)
    to_scipp[DEFAULT_WAVELENGTH_COLUMN_NAME + " [Å]"] = to_scipp[
        DEFAULT_WAVELENGTH_COLUMN_NAME
    ]
    # Add dummy data column
    dummy_data_column_name = "DUMMY_DATA"
    to_scipp[dummy_data_column_name] = np.ones(len(to_scipp))
    # Pop the vector columns for later
    vector_columns = ("hkl", "hkl_eq")
    vector_coords = {col: to_scipp.pop(col) for col in vector_columns}
    # Add units
    to_scipp.rename(columns={"I": "I [dimensionless]"}, inplace=True)
    to_scipp.rename(columns={"SIGI": "SIGI [dimensionless]"}, inplace=True)
    # Convert to scipp Dataset
    nmx_mtz_ds = from_pandas_dataframe(
        to_scipp,
        data_columns=dummy_data_column_name,
        header_parser=parse_bracket_header,
    )
    # Add back the vector columns
    for col, values in vector_coords.items():
        nmx_mtz_ds.coords[col] = sc.vectors(
            dims=nmx_mtz_ds.dims, values=[val for val in values]
        )
    # Add HKL EQ hash coordinate for grouping
    nmx_mtz_ds.coords["hkl_eq_hash"] = sc.Variable(
        dims=nmx_mtz_ds.dims,
        values=[hash(tuple(val)) for val in nmx_mtz_ds.coords["hkl_eq"].values],
    )
    # Return DataArray
    return NMXMtzDataArray(nmx_mtz_ds[dummy_data_column_name].copy(deep=False))


mtz_io_providers = (
    read_mtz_file,
    reduce_single_mtz,
    get_space_group,
    get_reciprocal_asu,
    merge_mtz_dataframes,
    reduce_merged_mtz_dataframe,
    nmx_mtz_dataframe_to_scipp_dataarray,
)
"""The providers related to the MTZ IO."""
mtz_io_params = {
    WavelengthColumnName: DEFAULT_WAVELENGTH_COLUMN_NAME,
}
"""The parameters related to the MTZ IO."""
