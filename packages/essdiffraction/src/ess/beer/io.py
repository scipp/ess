# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import os
import re
from pathlib import Path

import h5py
import numpy as np
import scipp as sc
import scipp.constants
from ess.powder.types import CaveMonitor, RunType, WavelengthMonitor

from ess.reduce.nexus.types import DetectorBankSizes

from .types import (
    DetectorBank,
    Filename,
    ModulationPeriod,
    RawDetector,
    SampleRun,
    WavelengthDefinitionChopperDelay,
)


def _rotation_from_y_rotation_matrix(rot):
    '''Assuming the rotation is around the y-axis
    this function creates a rotation operator from the rotation matrix.'''
    angle = np.atan2(rot[2, 0], rot[0, 0])
    return sc.spatial.rotation(
        value=[
            0.0,
            np.sin(angle / 2),
            0.0,
            np.cos(angle / 2),
        ]
    )


def _find_all_h5(group: h5py.Group, matches, nxclass=None):
    name = group.name
    for p in group.keys():
        if re.match(matches, os.path.join(name, p)) and (
            nxclass is None or nxclass.encode() == group[p].attrs.get('NX_class')
        ):
            yield group[p]
    # Breadth first
    for p in group.keys():
        if isinstance(group[p], h5py.Group):
            yield from _find_all_h5(group[p], matches, nxclass)


def _find_h5(group: h5py.Group, matches):
    for p in group.keys():
        if re.match(matches, p):
            return group[p]
    else:
        raise ValueError(f'Could not find "{matches}" in {group}.')


def _load_h5(group: h5py.Group | str | Path, *paths: str):
    if isinstance(group, str | Path):
        with h5py.File(group) as group:
            yield from _load_h5(group, *paths)
        return
    for path in paths:
        g = group
        for p in path.strip('/').split('/'):
            g = (
                _unique_child_group_h5(g, p)
                if p.startswith('NX')
                else g.get(p)
                if p in g
                else _find_h5(g, p)
            )
        yield g


def _unique_child_group_h5(
    group: h5py.Group,
    nx_class: str,
) -> h5py.Group | None:
    out = None
    for v in group.values():
        if v.attrs.get("NX_class") == nx_class.encode():
            if out is None:
                out = v
            else:
                raise ValueError(
                    f'Expected exactly one {nx_class} group, but found more'
                )
    return out


def _center_wavelength_from_mode(mode, value_in_file):
    if value_in_file != '0':
        return sc.scalar(float(value_in_file), unit='angstrom')
    if mode in [
        '0',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '10',
        '15',
        '16',
    ]:
        return sc.scalar(2.1, unit='angstrom')
    elif mode in ['1', '2']:
        return sc.scalar(3.1, unit='angstrom')
    elif mode == '11':
        return sc.scalar(3.0, unit='angstrom')
    elif mode == '12':
        return sc.scalar(3.5, unit='angstrom')
    elif mode == '13':
        return sc.scalar(6.0, unit='angstrom')
    elif mode == '14':
        return sc.scalar(4.0, unit='angstrom')
    else:
        raise ValueError(f'Unkonwn chopper mode {mode}.')


def _effective_chopper_position_from_mode(
    mode,
    *,
    psc1_pos,
    psc2_pos,
    psc3_pos,
    mca_pos,
    mcb_pos,
    mcc_pos,
):
    if mode in ['0', '1', '2', '11']:
        return sc.vector([0.0, 0.0, 0.0], unit='m')
    elif mode in ['3', '4', '12', '13', '15']:
        return sc.vector(0.5 * (psc1_pos + psc3_pos), unit='m')
    elif mode in ['5', '6']:
        return sc.vector(0.5 * (psc1_pos + psc2_pos), unit='m')
    elif mode in ['7', '8', '9', '10']:
        return sc.vector(mca_pos, unit='m')
    elif mode == '14':
        return sc.vector(mcc_pos, unit='m')
    elif mode == '16':
        return sc.vector(0.5 * (mca_pos + mcb_pos), unit='m')
    else:
        raise ValueError(f'Unkonwn chopper mode {mode}.')


def _load_beer_mcstas(f, *, north_or_south=None, number=None, detector_sizes):
    # Allow the keys of detector_sizes to be both the
    # NeXus detector name or DetectorBank enum.
    # It makes the code a bit more complex here, but I think mixing them up
    # is an easy mistake to make and it makes sense
    # to handle that gracefully.
    normalized_detector_sizes = {
        (
            'south'
            if key in (DetectorBank.south, 'south_detector')
            else 'north'
            if key in (DetectorBank.north, 'north_detector')
            else key
        ): value
        for key, value in detector_sizes.items()
    }
    if north_or_south is not None:
        # In newer files the bank is determined by a name (north or south).
        sizes = normalized_detector_sizes[north_or_south]
    else:
        # In older files the detector bank is determined by an index.
        if number == 1:
            sizes = normalized_detector_sizes['south']
        elif number == 2:
            sizes = normalized_detector_sizes['north']
        else:
            raise ValueError(
                'Could not determine detector sizes from provided '
                f'`DetectorBankSizes`: {detector_sizes}. '
                'Expected keys: "south_detector" and "north_detector".'
            )

    positions = {
        name: f'/entry1/instrument/components/{key}/Position'
        for key in f['/entry1/instrument/components']
        for name in ['sampleMantid', 'PSC1', 'PSC2', 'PSC3', 'MCA', 'MCB', 'MCC']
        if name in key
    }
    (
        data_dir,
        params,
        sample_pos,
        psc1_pos,
        psc2_pos,
        psc3_pos,
        mca_pos,
        mcb_pos,
        mcc_pos,
    ) = _load_h5(
        f,
        'NXentry/NXdetector',
        'NXentry/simulation/Param',
        positions['sampleMantid'],
        positions['PSC1'],
        positions['PSC2'],
        positions['PSC3'],
        positions['MCA'],
        positions['MCB'],
        positions['MCC'],
    )
    data = (
        next(_find_all_h5(data_dir, f'.*{north_or_south}{number}'))
        if north_or_south is not None and number is not None
        else next(_find_all_h5(data_dir, f'.*{north_or_south}'))
        if north_or_south is not None
        else next(_find_all_h5(data_dir, f'/entry1.*bank.*{number}'))
    )
    events = data['events']

    beam_rotation = _find_h5(f['/entry1/instrument/components'], '.*sourceMantid.*')[
        'Rotation'
    ]
    detector_rotation = _find_h5(
        f['/entry1/instrument/components'],
        f'.*nD_Mantid_?{north_or_south}_{number}.*'
        if north_or_south is not None and number is not None
        else f'.*nD_Mantid_?{north_or_south}.*'
        if north_or_south is not None
        else f'.*nD_Mantid_?{number}.*',
    )['Rotation']

    events = events[()]
    da = sc.DataArray(
        sc.array(dims=['events'], values=events[:, 0], variances=events[:, 0] ** 2),
    )
    for name, value in data.attrs.items():
        if name in ('position',):
            da.coords[name] = sc.scalar(value.decode())

    for i, label in enumerate(data.attrs["ylabel"].decode().strip().split(' ')):
        if label == 'p':
            continue
        da.coords[label] = sc.array(dims=['events'], values=events[:, i])

    for k, v in params.items():
        v = v[0]
        if isinstance(v, bytes):
            v = v.decode()
        if k in ('mode', 'sample_filename', 'lambda'):
            da.coords[k] = sc.scalar(v)

    da.coords['wavelength_estimate'] = _center_wavelength_from_mode(
        da.coords['mode'].value,
        da.coords.pop('lambda').value if 'lambda' in da.coords else '0',
    )

    # Depending on the mode the effective chopper position is either the position
    # of one chopper or the midpoint between two choppers.
    # A neutron with "center wavelength" (called "lambda" in the McStas files)
    # emitted from the middle of the pulse reaches the effective chopper position
    # at the moment that is the center of the effective chopper opening window.
    # That means we can use the effective chopper position and the center wavelength
    # to obtain `t0`, the time when the neutron can be said to have passed the chopper.
    #
    # In real files we might not have this "lambda" parameter.
    # But we will have chopper settings that can be used to get the same information.
    da.coords['chopper_position'] = _effective_chopper_position_from_mode(
        da.coords['mode'].value,
        psc1_pos=psc1_pos[:],
        psc2_pos=psc2_pos[:],
        psc3_pos=psc3_pos[:],
        mca_pos=mca_pos[:],
        mcb_pos=mcb_pos[:],
        mcc_pos=mcc_pos[:],
    )

    da.coords['sample_position'] = sc.vector(sample_pos[:], unit='m')
    da.coords['detector_position'] = sc.vector(
        list(map(float, da.coords.pop('position').value.split(' '))), unit='m'
    )

    da.coords.pop('n')
    da.coords['x'].unit = 'm'
    da.coords['y'].unit = 'm'
    da.coords['t'].unit = 's'

    # Bin detector panel into rectangular "pixels"
    # similar in size to the physical detector pixels.
    da = da.bin(
        y=sc.linspace('y', -0.5, 0.5, sizes['y'] + 1, unit='m'),
        x=sc.linspace('x', -0.5, 0.5, sizes['x'] + 1, unit='m'),
    )

    # Compute the position of each pixel in the global coordinate system.
    # The detector local coordinate system is rotatated by the detector rotation,
    # and translated to the location of the detector in the global coordinate system.
    da.coords['position'] = (
        da.coords['detector_position']
        + _rotation_from_y_rotation_matrix(detector_rotation)
        * sc.spatial.as_vectors(
            sc.midpoints(da.coords['x']),
            sc.midpoints(da.coords['y']),
            sc.scalar(0.0, unit='m'),
        )
        # We need the dimension order of the positions to be the same
        # as the dimension order of the binned data array.
    ).transpose(da.dims)

    L1 = sc.norm(da.coords['sample_position'] - da.coords['chopper_position'])
    L2 = sc.norm(da.coords['position'] - da.coords['sample_position'])

    # Define the incident beam by rotating the z-axis by
    # the rotation of the "source" in McStas.
    incident_beam = L1 * (
        _rotation_from_y_rotation_matrix(beam_rotation) * sc.vector([0, 0, 1.0])
    )
    # Create a source position that gives us the incident beam
    # direction and length that we want.
    # In practice this should be hardcoded or determined from
    # some entry in the Nexus file.
    da.coords['source_position'] = da.coords['sample_position'] - incident_beam

    da.coords['moderator_to_detector_distance'] = (
        L1 + L2 + sc.norm(da.coords['chopper_position'])
    )
    da.coords['source_to_wavelength_definition_chopper_distance'] = sc.norm(
        da.coords['chopper_position']
    )

    t = da.bins.coords['t']
    da.bins.coords['event_time_offset'] = t % sc.scalar(1 / 14, unit='s').to(
        unit=t.unit
    )
    # Estimate of the time the neutron passed the virtual source chopper.
    # Used in pulse shaping mode to determine the wavelength.
    # Used in modulation mode automatic-peak-finding reduction to estimate d.
    # In practice this will probably be replaced by the regular tof workflow.
    # But I'm not 100% sure.
    da.coords["frame_cutoff_time"] = (
        sc.constants.m_n
        / sc.constants.h
        * da.coords['wavelength_estimate']
        * da.coords['moderator_to_detector_distance'].min().to(unit='angstrom')
    ).to(unit='s') - sc.scalar(1 / 14, unit='s') / 2

    del da.coords['x']
    del da.coords['y']
    # The binned t coordinate is kept because it can be useful
    # to understand resolution and to debug tof estimation.
    return da


def load_beer_mcstas(
    f: str | Path | h5py.File, bank: DetectorBank, detector_sizes: DetectorBankSizes
) -> sc.DataArray:
    '''Load beer McStas data from a file to a data array.'''
    if not isinstance(bank, DetectorBank):
        raise ValueError(
            '"bank" must be either ``DetectorBank.north`` or ``DetectorBank.south``'
        )

    if isinstance(f, str | Path):
        with h5py.File(f) as ff:
            return load_beer_mcstas(ff, bank=bank, detector_sizes=detector_sizes)

    if len(list(_find_all_h5(f['/entry1/data'], '.*bank', nxclass='NXdata'))) < 8:
        # If we don't find ~13 detectors, then assume the detectors are 2D
        if len(list(_find_all_h5(f['/entry1/data'], '.*south', nxclass='NXdata'))) > 0:
            return _load_beer_mcstas(
                f, north_or_south=bank.name, detector_sizes=detector_sizes
            )

        return _load_beer_mcstas(
            f,
            north_or_south=None,
            number=1 if bank == DetectorBank.south else 2,
            detector_sizes=detector_sizes,
        )

    return sc.concat(
        [
            _load_beer_mcstas(
                f,
                north_or_south=bank.name,
                number=number,
                detector_sizes=detector_sizes,
            )
            for number in range(1, 13)
        ],
        dim='panel',
    )


def _to_edges(centers: sc.Variable) -> sc.Variable:
    interior_edges = sc.midpoints(centers)
    return sc.concat(
        [
            2 * centers[0] - interior_edges[0],
            interior_edges,
            2 * centers[-1] - interior_edges[-1],
        ],
        dim=centers.dim,
    )


def load_beer_mcstas_monitor(f: str | Path | h5py.File):
    if isinstance(f, str | Path):
        with h5py.File(f) as ff:
            return load_beer_mcstas_monitor(ff)
    (
        monitor,
        wavelengths,
        data,
        errors,
        ncount,
    ) = _load_h5(
        f,
        'NXentry/NXdetector/Lmon_hereon_dat',
        'NXentry/NXdetector/Lmon_hereon_dat/Wavelength__AA_',
        'NXentry/NXdetector/Lmon_hereon_dat/data',
        'NXentry/NXdetector/Lmon_hereon_dat/errors',
        'NXentry/NXdetector/Lmon_hereon_dat/ncount',
    )
    da = sc.DataArray(
        sc.array(
            dims=['wavelength'], values=data[:], variances=errors[:], unit='counts'
        ),
        coords={
            'wavelength': _to_edges(
                sc.array(dims=['wavelength'], values=wavelengths[:], unit='angstrom')
            ),
            'ncount': sc.array(dims=['wavelength'], values=ncount[:], unit='counts'),
        },
    )
    for name, value in monitor.attrs.items():
        if name in ('position',):
            da.coords[name] = sc.scalar(value.decode())

    da.coords['position'] = sc.vector(
        list(map(float, da.coords.pop('position').value.split(' '))), unit='m'
    )
    return da


def load_beer_mcstas_provider(
    fname: Filename[RunType], bank: DetectorBank, detector_bank_sizes: DetectorBankSizes
) -> RawDetector[RunType]:
    return load_beer_mcstas(fname, bank, detector_bank_sizes)


def load_beer_mcstas_monitor_provider(
    fname: Filename[RunType],
) -> WavelengthMonitor[RunType, CaveMonitor]:
    return load_beer_mcstas_monitor(fname)


def mcstas_chopper_delay_from_mode(
    da: RawDetector[SampleRun],
) -> WavelengthDefinitionChopperDelay:
    '''These settings are good for the set of McStas runs that we
    use in the docs currently.
    Eventually we will want to determine this from the chopper information
    in the files, but that information is not in the simulation output.'''
    mode = da.coords['mode'].value
    if mode in ('7', '8', '9', '10'):
        return sc.scalar(0.0024730158730158727, unit='s')
    if mode == '16':
        return sc.scalar(0.000876984126984127, unit='s')
    raise ValueError(f'Mode {mode} is not known.')


def mcstas_chopper_delay_from_mode_new_simulations(
    da: RawDetector[SampleRun],
) -> WavelengthDefinitionChopperDelay:
    '''Celine has a new simulation with some changes to the chopper placement(?).
    For those simulations we need to adapt the chopper delay values.'''
    mode = da.coords['mode'].value
    if mode == '7':
        return sc.scalar(0.001370158730158727, unit='s')
    if mode == '8':
        return sc.scalar(0.001370158730158727, unit='s')
    if mode == '9':
        return sc.scalar(0.0022630158730158727, unit='s')
    if mode == '10':
        return sc.scalar(0.0022630158730158727, unit='s')
    if mode == '16':
        return sc.scalar(0.000476984126984127, unit='s')
    raise ValueError(f'Mode {mode} is not known.')


def mcstas_modulation_period_from_mode(da: RawDetector[SampleRun]) -> ModulationPeriod:
    mode = da.coords['mode'].value
    if mode in ('7', '8'):
        return sc.scalar(1.0 / (8 * 70), unit='s')
    if mode == '9':
        return sc.scalar(1.0 / (8 * 140), unit='s')
    if mode == '10':
        return sc.scalar(1.0 / (8 * 280), unit='s')
    if mode == '16':
        return sc.scalar(1.0 / (4 * 280), unit='s')
    raise ValueError(f'Mode {mode} is not known.')


mcstas_providers = (
    load_beer_mcstas_provider,
    load_beer_mcstas_monitor_provider,
    mcstas_chopper_delay_from_mode,
)
