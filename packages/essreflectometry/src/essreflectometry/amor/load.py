# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from pathlib import Path
from typing import Union

import scipp as sc
import scippnexus as snx

from ..logging import get_logger
from ..types import ChopperCorrectedTofEvents, Filename, RawData, RawEvents, Run
from .data import get_path
from .types import BeamlineParams


def chopper_tof_correction(data: RawEvents[Run]) -> ChopperCorrectedTofEvents[Run]:
    """
    A correction for the presence of the chopper with respect to the "true" ToF.
    Also fold the two pulses.
    TODO: generalize mechanism to fold any number of pulses.

    Parameters
    ----------
    data:
        Input data array to correct.

    Returns
    -------
    :
        ToF corrected data array.
    """
    dim = 'tof'
    tof_unit = data.bins.coords[dim].bins.unit
    tau = sc.to_unit(
        1 / (2 * data.coords['source_chopper_2'].value['frequency'].data),
        tof_unit,
    )
    chopper_phase = data.coords['source_chopper_2'].value['phase'].data
    tof_offset = tau * chopper_phase / (180.0 * sc.units.deg)
    # Make 2 bins, one for each pulse
    edges = sc.concat([-tof_offset, tau - tof_offset, 2 * tau - tof_offset], dim)
    data = data.bin({dim: sc.to_unit(edges, tof_unit)})
    # Make one offset for each bin
    offset = sc.concat([tof_offset, tof_offset - tau], dim)
    # Apply the offset on both bins
    data.bins.coords[dim] += offset
    # Rebin to exclude second (empty) pulse range
    data = data.bin({dim: sc.concat([0.0 * sc.units.us, tau], dim)})

    # Ad-hoc correction described in
    # https://scipp.github.io/ess/instruments/amor/amor_reduction.html
    data.coords['position'].fields.y += data.coords['position'].fields.z * sc.tan(
        2.0 * data.coords['sample_rotation'] - (0.955 * sc.units.deg)
    )

    return ChopperCorrectedTofEvents[Run](data)


def _assemble_event_data(dg: sc.DataGroup) -> sc.DataArray:
    """Extract the events as a data array with all required coords.

    Parameters
    ----------
    dg:
        A data group with the structure of an Amor NeXus file.

    Returns
    -------
    :
        A data array with the events extracted from ``dg``.
    """
    events = dg['instrument']['multiblade_detector']['data'].copy(deep=False)
    events.bins.coords['tof'] = events.bins.coords.pop('event_time_offset')
    events.coords['position'] = sc.spatial.as_vectors(
        events.coords.pop('x_pixel_offset'),
        events.coords.pop('y_pixel_offset'),
        events.coords.pop('z_pixel_offset'),
    )
    events.coords['sample_position'] = sc.vector([0, 0, 0], unit='m')
    return events


def _load_nexus_entry(filename: Union[str, Path]) -> sc.DataGroup:
    """Load the single entry of a nexus file."""
    with snx.File(filename, 'r') as f:
        if len(f.keys()) != 1:
            raise snx.NexusStructureError(
                f"Expected a single entry in file {filename}, got {len(f.keys())}"
            )
        return f['entry'][()]


def load_raw_nexus(filename: Filename[Run]) -> RawData[Run]:
    """Load unprocessed data and metadata from an Amor NeXus file.

    Parameters
    ----------
    filename:
        Filename of the NeXus file.

    Returns
    -------
    :
        Data and metadata.
    """
    filename = get_path(filename)
    get_logger('amor').info(
        "Loading '%s' as an Amor NeXus file",
        filename.filename if hasattr(filename, 'filename') else filename,
    )
    return RawData(_load_nexus_entry(filename))


def extract_events(
    raw_data: RawData[Run], beamline: BeamlineParams[Run]
) -> RawEvents[Run]:
    """Extract the events from unprocessed NeXus data.

    Parameters
    ----------
    raw_data:
        Data in a form representing an Amor NeXus file.
    beamline:
        A dict defining the beamline parameters.

    Returns
    -------
    :
        Data array object for Amor dataset.
    """
    data = _assemble_event_data(raw_data)

    # Recent versions of scippnexus no longer add variances for events by default, so
    # we add them here if they are missing.
    if data.bins.constituents['data'].data.variances is None:
        data.bins.constituents['data'].data.variances = data.bins.constituents[
            'data'
        ].data.values

    # Convert tof nanoseconds to microseconds for convenience
    data.bins.coords['tof'] = data.bins.coords['tof'].to(
        unit='us', dtype='float64', copy=False
    )

    # Add beamline parameters
    for key, value in beamline.items():
        data.coords[key] = value

    return RawEvents[Run](data)


providers = (extract_events, load_raw_nexus, chopper_tof_correction)
