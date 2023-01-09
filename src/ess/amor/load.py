# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any, Optional
import warnings
from datetime import datetime
import scipp as sc
import scippneutron as scn
from .beamline import make_beamline
from ..logging import get_logger


def _tof_correction(data: sc.DataArray, dim: str = 'tof') -> sc.DataArray:
    """
    A correction for the presense of the chopper with respect to the "true" ToF.
    Also fold the two pulses.
    TODO: generalise mechanism to fold any number of pulses.

    Parameters
    ----------
    data:
        Input data array to correct.
    dim:
        Name of the time of flight dimension.

    Returns
    -------
    :
        ToF corrected data array.
    """
    if 'orso' in data.attrs:
        data.attrs['orso'].value.reduction.corrections += ['chopper ToF correction']
    tau = sc.to_unit(1 / (2 * data.coords['source_chopper_2'].value['frequency'].data),
                     data.coords[dim].unit)
    chopper_phase = data.coords['source_chopper_2'].value['phase'].data
    tof_offset = tau * chopper_phase / (180.0 * sc.units.deg)
    # Make 2 bins, one for each pulse
    edges = sc.concat([-tof_offset, tau - tof_offset, 2 * tau - tof_offset], dim)
    data = data.bin({dim: sc.to_unit(edges, data.coords[dim].unit)})
    # Make one offset for each bin
    offset = sc.concat([tof_offset, tof_offset - tau], dim)
    # Apply the offset on both bins
    data.bins.coords[dim] += offset
    # Rebin to exclude second (empty) pulse range
    return data.bin({dim: sc.concat([0. * sc.units.us, tau], dim)})


def load(filename,
         orso: Optional[Any] = None,
         beamline: Optional[dict] = None,
         disable_warnings: Optional[bool] = True) -> sc.DataArray:
    """
    Loader for a single Amor data file.

    Parameters
    ----------
    filename:
        Path of the file to load.
    orso:
        The orso object to be populated by additional information from the loaded file.
    beamline:
        A dict defining the beamline parameters.
    disable_warnings:
        Do not show warnings from file loading if `True`. Default is `True`.

    Returns
    -------
    :
        Data array object for Amor dataset.
    """
    get_logger('amor').info(
        "Loading '%s' as an Amor NeXus file",
        filename.filename if hasattr(filename, 'filename') else filename)
    if disable_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            data = scn.load_nexus(filename)
    else:
        data = scn.load_nexus(filename)

    # Convert tof nanoseconds to microseconds for convenience
    # TODO: is it safe to assume that the dtype of the binned wrapper coordinate is
    # the same as the dtype of the underlying event coordinate?
    data.bins.coords['tof'] = data.bins.coords['tof'].astype('float64', copy=False)
    data.coords['tof'] = data.coords['tof'].astype('float64', copy=False)
    data.bins.coords['tof'] = sc.to_unit(data.bins.coords['tof'], 'us', copy=False)
    data.coords['tof'] = sc.to_unit(data.coords['tof'], 'us', copy=False)

    # Add beamline parameters
    beamline = make_beamline() if beamline is None else beamline
    for key, value in beamline.items():
        data.coords[key] = value

    if orso is not None:
        populate_orso(orso=orso, data=data, filename=filename)
        data.attrs['orso'] = sc.scalar(orso)

    # Perform tof correction and fold two pulses
    return _tof_correction(data)


def populate_orso(orso: Any, data: sc.DataArray, filename: str) -> Any:
    """
    Populate the Orso object, by calling the :code:`base_orso` and adding data from the
    file.

    Parameters
    ----------
    orso:
        The orso object to be populated by additional information from the loaded file.
    data:
        Data array to source information from.
    filename:
        Path of the file to load.
    """
    orso.data_source.experiment.title = data.attrs['experiment_title'].value
    orso.data_source.experiment.instrument = data.attrs['instrument_name'].value
    orso.data_source.experiment.start_date = datetime.strftime(
        datetime.strptime(data.attrs['start_time'].value[:-3], '%Y-%m-%dT%H:%M:%S.%f'),
        '%Y-%m-%d')
    orso.data_source.measurement.data_files = [filename]
