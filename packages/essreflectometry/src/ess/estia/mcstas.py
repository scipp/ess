# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pathlib

import h5py
import numpy as np
import pandas as pd
import scipp as sc
from scippnexus import NXsample, NXsource

from ess.reduce.nexus.types import DetectorBankSizes, Position

from ..reflectometry.load import load_h5
from ..reflectometry.types import (
    CoordTransformationGraph,
    DetectorLtotal,
    DetectorRotation,
    Filename,
    RawDetector,
    RawSampleRotation,
    RunType,
    SampleRotation,
    SampleRotationOffset,
)
from .beamline import DETECTOR_BANK_SIZES
from .conversions import coordinate_transformation_graph


def parse_metadata_ascii(lines):
    data = {}
    section = None
    for line in lines:
        if line.startswith('begin'):
            _, _, name = line.partition(' ')
            section = {}
        elif line.startswith('end'):
            data.setdefault(name.strip(), []).append(section)
            section = None
        else:
            if section is not None:
                key, _, value = line.partition(': ')
                section[key.strip()] = value.strip()
    return data


def parse_events_ascii(lines):
    meta = {}
    data = []
    for line in lines:
        if line.startswith('#'):
            key, _, value = line[2:].partition(': ')
            if '=' in value:
                key, _, value = value.partition('=')
            meta[key] = value
        else:
            break

    data = pd.read_csv(lines, comment='#', header=None, delimiter=' ')

    if 'ylabel' in meta:
        labels = meta['ylabel'].strip().split(' ')
        if labels[0] == 'p':
            da = sc.DataArray(
                # The squares on the variances is the correct way
                # to load weighted event data.
                # Consult the McStas documentation
                # (section 2.2.1) https://www.mcstas.org/documentation/manual/
                # for more information.
                sc.array(
                    dims=['events'],
                    values=data[0].to_numpy(),
                    variances=data[0].to_numpy() ** 2,
                ),
                coords={
                    label: sc.array(dims=['events'], values=data[i].to_numpy())
                    for i, label in enumerate(labels)
                    if i != 0
                },
            )
            for k, v in meta.items():
                da.coords[k] = sc.scalar(v)
            return da
    raise ValueError('Could not parse the file as a list of events.')


def parse_events_h5(f, events_to_sample_per_unit_weight=None):
    if isinstance(f, str):
        with h5py.File(f) as ff:
            return parse_events_h5(ff)

    data, events, params = load_h5(
        f,
        'NXentry/NXdetector/NXdata',
        'NXentry/NXdetector/NXdata/events',
        'NXentry/simulation/Param',
    )
    events = events[()]
    if events_to_sample_per_unit_weight is None:
        da = sc.DataArray(
            # The squares on the variances is the correct way to load
            # weighted event data.
            # Consult the McStas documentation
            # (section 2.2.1) https://www.mcstas.org/documentation/manual/
            # for more information.
            sc.array(dims=['events'], values=events[:, 0], variances=events[:, 0] ** 2),
        )
        for i, label in enumerate(data.attrs["ylabel"].decode().strip().split(' ')):
            if label == 'p':
                continue
            da.coords[label] = sc.array(dims=['events'], values=events[:, i])
    else:
        weights = events[:, 0]
        total_weight = weights.sum()
        nevents_to_sample = round(events_to_sample_per_unit_weight * total_weight)
        inds = np.random.choice(
            np.arange(len(weights)),
            nevents_to_sample,
            p=weights / total_weight,
        )
        da = sc.DataArray(
            sc.ones(dims=['events'], shape=(nevents_to_sample,), with_variances=True),
        )
        for i, label in enumerate(data.attrs["ylabel"].decode().strip().split(' ')):
            if label == 'p':
                continue
            da.coords[label] = sc.array(dims=['events'], values=events[inds, i])

    for name, value in data.attrs.items():
        da.coords[name] = sc.scalar(value.decode())

    for k, v in params.items():
        v = v[0]
        if isinstance(v, bytes):
            v = v.decode()
        da.coords[k] = sc.scalar(v)
    return da


def load_mcstas_provider(
    filename: Filename[RunType],
    sample_rotation_offset: SampleRotationOffset[RunType],
) -> RawDetector[RunType]:
    """See :py:`load_mcstas`."""
    da = load_mcstas(filename)
    da.coords['sample_rotation'] += sample_rotation_offset.to(
        unit=da.coords['sample_rotation'].unit
    )
    return RawDetector[RunType](da)


def load_mcstas(
    filename: str | pathlib.Path | sc.DataArray,
) -> sc.DataArray:
    """
    Load event data from a McStas run and reshape it
    to look like what we would expect if
    we loaded data from a real experiment file.
    """
    if isinstance(filename, sc.DataArray):
        da = filename
    elif h5py.is_hdf5(filename):
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

    nblades = DETECTOR_BANK_SIZES['multiblade_detector']['blade']
    nwires = DETECTOR_BANK_SIZES['multiblade_detector']['wire']
    nstrips = DETECTOR_BANK_SIZES['multiblade_detector']['strip']
    xbins = sc.linspace('x', -0.25, 0.25, nblades * nwires + 1)
    ybins = sc.linspace('y', -0.13, 0.13, nstrips + 1)
    da = da.bin(y=ybins, x=xbins).rename_dims({'y': 'strip'})
    da.coords['strip'] = sc.arange('strip', 0, nstrips)
    da.coords['z_index'] = sc.arange('x', nblades * nwires - 1, -1, -1)

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
    ).transpose(da.dims)
    da.coords['position'] = (
        da.coords['detector_position'] + rotation_by_detector_rotation * position
    )

    da.bins.coords['event_time_zero'] = (
        sc.scalar(0, unit='s') * da.bins.coords['t']
    ).to(unit='ns')
    da.bins.coords['event_time_offset'] = (
        sc.scalar(1, unit='s') * da.bins.coords['t']
    ).to(unit='ns') % sc.scalar(1 / 14, unit='s').to(unit='ns')
    da.bins.coords['wavelength_from_mcstas'] = (
        sc.scalar(1.0, unit='angstrom') * da.bins.coords['L']
    )

    da.coords["sample_size"] = sc.scalar(1.0, unit='m') * float(
        da.coords['sample_length'].value
    )
    da.coords["beam_size"] = sc.scalar(2.0, unit='mm')

    da = da.fold(
        'x',
        sizes={
            k: v
            for k, v in DETECTOR_BANK_SIZES['multiblade_detector'].items()
            if k in ('blade', 'wire')
        },
    )
    da.bins.coords.pop('L')
    da.bins.coords.pop('t')
    return da


def load_sample_rotation(da: RawDetector[RunType]) -> RawSampleRotation[RunType]:
    return da.coords['sample_rotation']


def load_detector_rotation(
    da: RawDetector[RunType],
) -> DetectorRotation[RunType]:
    return da.coords['detector_rotation']


def load_source_position(
    da: RawDetector[RunType],
) -> Position[NXsource, RunType]:
    return da.coords['source_position']


def load_sample_position(
    da: RawDetector[RunType],
) -> Position[NXsample, RunType]:
    return da.coords['sample_position']


def detector_ltotal_from_raw(
    da: RawDetector[RunType], graph: CoordTransformationGraph[RunType]
) -> DetectorLtotal[RunType]:
    return da.transform_coords(
        ['Ltotal'],
        graph=graph,
    ).coords['Ltotal']


def mcstas_wavelength_coordinate_transformation_graph(
    source_position: Position[NXsource, RunType],
    sample_position: Position[NXsample, RunType],
    sample_rotation: SampleRotation[RunType],
    detector_rotation: DetectorRotation[RunType],
    detector_bank_sizes: DetectorBankSizes,
) -> CoordTransformationGraph[RunType]:
    return {
        **coordinate_transformation_graph(
            source_position,
            sample_position,
            sample_rotation,
            detector_rotation,
            detector_bank_sizes,
        ),
        "wavelength": lambda wavelength_from_mcstas: wavelength_from_mcstas,
    }


providers = (
    load_mcstas_provider,
    load_sample_position,
    load_source_position,
    load_detector_rotation,
    load_sample_rotation,
    detector_ltotal_from_raw,
)
