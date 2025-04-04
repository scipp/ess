# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import json
import pathlib
from typing import NewType

import gemmi
import numpy as np
import pandas as pd
import scipp as sc

from .dials_reflection_io import load

# User defined or configurable types
DialsReflectionFilePath = NewType("DialsReflectionFilePath", pathlib.Path)
"""Path to the dials reflection file"""
DialsReflectionFile = NewType("DialsReflectionFile", dict)
"""The raw DIALS reflection file, read in as a dict"""
DialsExperimentFilePath = NewType("DialsExperimentFilePath", pathlib.Path)
"""Path to the dials experiment file"""
DialsExperiment = NewType("DialsExperiment", dict)
"""Experiment details from DIALS .expt file (JSON format)"""
SpaceGroupDesc = NewType("SpaceGroupDesc", str)
"""The space group description. e.g. 'P 21 21 21'"""
DEFAULT_SPACE_GROUP_DESC = SpaceGroupDesc("P 1")
"""The default space group description to use if not found in the input files."""
UnitCell = NewType("UnitCell", tuple[float])
"""The unit cell a, b, c in Angstrom, alpha, beta, gamma in degrees"""

# Custom column names
WavelengthColumnName = NewType("WavelengthColumnName", str)
"""The name of the wavelength column in the DIALS reflection file."""
DEFAULT_WAVELENGTH_COLUMN_NAME = WavelengthColumnName("LAMBDA")

IntensityColumnName = NewType("IntensityColumnName", str)
"""The name of the intensity column in the DIALS reflection file."""
DEFAULT_INTENSITY_COLUMN_NAME = IntensityColumnName("I")

VarianceColumnName = NewType("VarianceColumnName", str)
"""The name of the variance (stdev(I)**2) of intensity column in the DIALS reflection
file."""
DEFAULT_VARIANCE_COLUMN_NAME = VarianceColumnName("VARI")

StdDevColumnName = NewType("StdDevColumnName", str)
"""The name of the standard deviation of intensity column in the DIALS reflection
file."""
DEFAULT_STDEV_COLUMN_NAME = VarianceColumnName("SIGI")


# Computed types
DialsDataFrame = NewType("DialsDataFrame", pd.DataFrame)
"""The raw mtz dataframe."""
NMXDialsDataFrame = NewType("NMXDialsDataFrame", pd.DataFrame)
"""The processed mtz dataframe with derived columns."""
NMXDialsDataArray = NewType("NMXDialsDataArray", sc.DataArray)


def read_dials_reflection_file(
    file_path: DialsReflectionFilePath,
) -> DialsReflectionFile:
    """read dials reflection file"""

    return DialsReflectionFile(load(file_path.as_posix(), copy=True))


def read_dials_experiment_file(file_path: DialsExperimentFilePath) -> DialsExperiment:
    """Read Dials Experiment .expt file"""

    return DialsExperiment(json.load(open(file_path)))


def get_unit_cell(dials_expt: DialsExperiment) -> UnitCell:
    """
    Get the unit cell from the expt file.
    It is saved as real-space vectors so the unit cell has to be
    calculated from them.
    """
    crystal = dials_expt['crystal'][0]
    ra, rb, rc = tuple([crystal[f'real_space_{x}'] for x in 'abc'])
    a = np.linalg.norm(ra)
    b = np.linalg.norm(rb)
    c = np.linalg.norm(rc)
    al = np.rad2deg(np.arccos(np.dot(rb, rc) / (b * c)))
    be = np.rad2deg(np.arccos(np.dot(ra, rc) / (a * c)))
    ga = np.rad2deg(np.arccos(np.dot(ra, rb) / (a * b)))

    return UnitCell(a, b, c, al, be, ga)


def get_unique_space_group(dials_expt: DialsExperiment) -> gemmi.SpaceGroup:
    """
    Get space group from Dials expt file.
    For cctbx/disambiguation reasons it is saved as the Hall symbol, but
    the H-M notation can be back-determined with gemmi.
    """
    crystal = dials_expt['crystal'][0]
    sg_hall = crystal['space_group_hall_symbol']

    return gemmi.find_spacegroup_by_ops(gemmi.symops_from_hall(sg_hall))


def get_reciprocal_asu(spacegroup: gemmi.SpaceGroup) -> gemmi.ReciprocalAsu:
    """Returns the reciprocal asymmetric unit from the space group."""

    return gemmi.ReciprocalAsu(spacegroup)


def dials_refl_to_pandas(refls: dict) -> pd.DataFrame:
    """Converts the loaded DIALS reflection file to a pandas dataframe.

    It is equivalent to the following code:

    .. code-block:: python

        import numpy as np
        import pandas as pd

        data = np.array(mtz, copy=False)
        columns = mtz.column_labels()
        return pd.DataFrame(data, columns=columns)

    It is recommended in the gemmi documentation.

    """
    if refls.get('experiment_identifier'):  # this has no relevant information
        del refls['experiment_identifier']  # and it complicates loading as a df
    return pd.DataFrame(
        {
            key: list(val) if isinstance(val, np.ndarray) and val.ndim > 1 else val
            for key, val in refls.items()
        }
    )


def process_dials_refl_list_to_dataframe(
    refls: dict,
) -> DialsDataFrame:
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

        - ``SIGI``: The uncertainty of the intensity value, defined as the square root
        of the measured variance.

        For consistent names of columns/coordinates, the following columns are renamed:

        - ``wavelength_column_name`` -> ``'wavelength'``
        - ``intensity_column_name`` -> ``'I'``
        - ``intensity_sig_col_name`` -> ``'SIGI'``

        Other columns are kept as they are.

    """
    orig_df = dials_refl_to_pandas(refls)
    new_df = pd.DataFrame()

    new_df['H'] = orig_df['miller_index'].map(lambda x: x[0]).astype(int)
    new_df['K'] = orig_df['miller_index'].map(lambda x: x[1]).astype(int)
    new_df['L'] = orig_df['miller_index'].map(lambda x: x[2]).astype(int)

    new_df['hkl'] = orig_df['miller_index']
    new_df["d"] = orig_df['d']
    new_df['wavelength'] = orig_df['wavelength_cal']

    new_df[DEFAULT_INTENSITY_COLUMN_NAME] = orig_df['intensity.sum.value']
    new_df[DEFAULT_VARIANCE_COLUMN_NAME] = orig_df['intensity.sum.variance']
    new_df[DEFAULT_STDEV_COLUMN_NAME] = np.sqrt(orig_df['intensity.sum.variance'])

    for column in [col for col in orig_df.columns if col not in new_df]:
        new_df[column] = orig_df[column]

    return DialsDataFrame(new_df)


def process_dials_dataframe(
    *,
    dials_df: DialsDataFrame,
    reciprocal_asu: gemmi.ReciprocalAsu,
    sg: gemmi.SpaceGroup,
) -> NMXDialsDataFrame:
    """Modify/Add columns of the shallow copy of a dials dataframe.

    This method must be called after merging multiple mtz dataframe.
    """

    df = dials_df.copy(deep=False)

    def _reciprocal_asu(row: pd.Series) -> list[int]:
        """Converts miller indices(HKL) to ASU indices."""

        return reciprocal_asu.to_asu(row["hkl"], sg.operations())[0]

    df["hkl_asu"] = df.apply(_reciprocal_asu, axis=1)
    # Unpack the indices for later.
    df[["H_ASU", "K_ASU", "L_ASU"]] = pd.DataFrame(
        df["hkl_asu"].to_list(), index=df.index
    )

    return NMXDialsDataFrame(df)


def nmx_dials_dataframe_to_scipp_dataarray(
    nmx_mtz_df: NMXDialsDataFrame,
) -> NMXDialsDataArray:
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
            DEFAULT_STDEV_COLUMN_NAME,
            DEFAULT_VARIANCE_COLUMN_NAME,
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
    nmx_mtz_da.variances = nmx_mtz_ds[DEFAULT_VARIANCE_COLUMN_NAME].values

    # Return DataArray without negative intensities
    return NMXDialsDataArray(nmx_mtz_da[nmx_mtz_da.data > 0])
