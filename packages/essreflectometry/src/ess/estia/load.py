# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import h5py
import scipp as sc

from ..reflectometry.types import (
    DetectorData,
    Filename,
    RunType,
    SampleRotationOffset,
)
from .mcstas import parse_events_ascii, parse_events_h5


def load_mcstas_events(
    filename: Filename[RunType],
    sample_rotation_offset: SampleRotationOffset[RunType],
) -> DetectorData[RunType]:
    """
    Load event data from a McStas run and reshape it
    to look like what we would expect if
    we loaded data from a real experiment file.
    """
    if h5py.is_hdf5(filename):
        with h5py.File(filename) as f:
            da = parse_events_h5(f)
    else:
        with open(filename) as f:
            da = parse_events_ascii(f)

    da.coords['sample_rotation'] = sc.scalar(
        float(da.coords['omegaa'].value), unit='deg'
    )
    da.coords['detector_rotation'] = 2 * da.coords['sample_rotation'] + sc.scalar(
        1.65, unit='deg'
    )
    da.coords['sample_rotation'] += sample_rotation_offset.to(
        unit=da.coords['sample_rotation'].unit
    )

    xbins = sc.linspace('x', -0.25, 0.25, 14 * 32 + 1)
    ybins = sc.linspace('y', -0.25, 0.25, 65)
    da = da.bin(x=xbins, y=ybins).rename_dims({'y': 'stripe'})
    da.coords['stripe'] = sc.arange('stripe', 0, 64)
    da.coords['z_index'] = sc.arange('x', 14 * 32 - 1, -1, -1)

    # Information is not available in the mcstas output files, therefore it's hardcoded
    da.coords['sample_position'] = sc.vector([0.264298, -0.427595, 35.0512], unit='m')
    da.coords['source_position'] = sc.vector([0, 0, 0.0], unit='m')
    da.coords['detector_position'] = sc.vector(
        tuple(map(float, da.coords['position'].value.split(' '))), unit='m'
    )

    rotation_by_detector_rotation = sc.spatial.rotation(
        value=[
            sc.scalar(0.0),
            sc.sin(da.coords['detector_rotation'].to(unit='rad')),
            sc.scalar(0.0),
            sc.cos(da.coords['detector_rotation'].to(unit='rad')),
        ]
    )

    position = sc.spatial.as_vectors(
        x=sc.midpoints(da.coords['x']) * sc.scalar(1.0, unit='m'),
        y=sc.midpoints(da.coords['y']) * sc.scalar(1.0, unit='m'),
        z=sc.scalar(0.0, unit='m'),
    )
    da.coords['position'] = (
        da.coords['detector_position'] + rotation_by_detector_rotation * position
    )

    da.bins.coords['event_time_zero'] = (
        sc.scalar(0, unit='s') * da.bins.coords['t']
    ).to(unit='ns')
    da.bins.coords['event_time_offset'] = (
        sc.scalar(1, unit='s') * da.bins.coords['t']
    ).to(unit='ns') % sc.scalar(1 / 14, unit='s').to(unit='ns')
    da.bins.coords['wavelength'] = sc.scalar(1, unit='angstrom') * da.bins.coords['L']

    da.coords["sample_size"] = sc.scalar(1.0, unit='m') * float(
        da.coords['sample_length'].value
    )
    da.coords["beam_size"] = sc.scalar(2.0, unit='mm')

    da = da.fold('x', sizes={'blade': 14, 'wire': 32})
    da.bins.coords.pop('L')
    da.bins.coords.pop('t')
    return DetectorData[RunType](da)


providers = ()
