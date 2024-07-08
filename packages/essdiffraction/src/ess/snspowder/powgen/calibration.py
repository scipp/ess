# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Components for non-ESS diffraction experiments.

WARNING This package will be removed in the future!
It only serves as helpers to develop workflows until it is determined,
which mechanisms and interfaces will be used at ESS.
"""

from pathlib import Path

import numpy as np
import scipp as sc
import scippneutron as scn

from ...powder.logging import get_logger


def _as_boolean_mask(var: sc.Variable) -> sc.Variable:
    if var.dtype in ('float32', 'float64'):
        if sc.any(var != var.to(dtype='int64')).value:
            raise ValueError(
                'Cannot construct boolean mask, the input mask has fractional values.'
            )
    return var.to(dtype=bool)


def _parse_calibration_instrument_args(
    filename: str | Path,
    *,
    instrument_filename: str | None = None,
    instrument_name: str | None = None,
) -> dict[str, str]:
    if instrument_filename is not None:
        if instrument_name is not None:
            raise ValueError(
                'Only one argument of `instrument_name` and '
                '`instrument_filename` is allowed, got both.'
            )
        instrument_arg = {'InstrumentFilename': instrument_filename}
        instrument_message = f'with instrument file {instrument_filename}'
    else:
        if instrument_name is None:
            raise ValueError(
                'Need one argument of `instrument_name` and '
                '`instrument_filename` is allowed, got neither.'
            )
        instrument_arg = {'InstrumentName': instrument_name}
        instrument_message = f'with instrument {instrument_name}'

    get_logger().info(
        'Loading calibration from file %s %s', filename, instrument_message
    )
    return instrument_arg


def load_calibration(
    filename: str | Path,
    *,
    instrument_filename: str | None = None,
    instrument_name: str | None = None,
    mantid_args: dict | None = None,
) -> sc.Dataset:
    """
    Load and return calibration data.

    Warning
    -------
    This function is designed to work with calibration files used by Mantid,
    specifically for POWGEN. It does not represent the interface used at ESS
    and will be removed in the future.

    Uses the Mantid algorithm `LoadDiffCal
    <https://docs.mantidproject.org/nightly/algorithms/LoadDiffCal-v1.html>`_
    and stores the data in a :class:`scipp.Dataset`

    Note that this function requires mantid to be installed and available in
    the same Python environment as ess.

    Parameters
    ----------
    filename:
        The name of the calibration file to load.
    instrument_filename:
        Instrument definition file.
    instrument_name:
        Name of the instrument.
    mantid_args:
        Dictionary with additional arguments for the
        `LoadDiffCal` Mantid algorithm.

    Returns
    -------
    :
        A Dataset containing the calibration data and masking.
    """

    mantid_args = {} if mantid_args is None else mantid_args
    mantid_args.update(
        _parse_calibration_instrument_args(
            filename,
            instrument_filename=instrument_filename,
            instrument_name=instrument_name,
        )
    )

    with scn.mantid.run_mantid_alg(
        'LoadDiffCal',
        Filename=str(filename),
        MakeGroupingWorkspace=False,
        **mantid_args,
    ) as ws:
        ds = scn.from_mantid(ws.OutputCalWorkspace)
        mask_ws = ws.OutputMaskWorkspace
        rows = mask_ws.getNumberHistograms()
        mask = sc.array(
            dims=['row'],
            values=np.fromiter(
                (mask_ws.readY(i)[0] for i in range(rows)), count=rows, dtype=np.bool_
            ),
            unit=None,
        )
    # This is deliberately not stored as a mask since that would make
    # subsequent handling, e.g., with groupby, more complicated. The mask
    # is conceptually not masking rows in this table, i.e., it is not
    # marking invalid rows, but rather describes masking for other data.
    ds["mask"] = _as_boolean_mask(mask)

    # The file does not define units
    # TODO why those units? Can we verify?
    ds["difc"].unit = 'us / angstrom'
    ds["difa"].unit = 'us / angstrom**2'
    ds["tzero"].unit = 'us'

    ds = ds.rename_dims({'row': 'detector'})
    ds.coords['detector'] = ds['detid'].data
    ds.coords['detector'].unit = None
    del ds['detid']

    return ds
