# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from pathlib import Path

import h5py
import scipp as sc

from .types import (
    DetectorData,
    Filename,
    ModulationPeriod,
    SampleRun,
    TwoThetaMaskFunction,
    WavelengthDefinitionChopperDelay,
)


def _load_h5(group: h5py.Group | str, *paths: str):
    if isinstance(group, str):
        with h5py.File(group) as group:
            yield from _load_h5(group, *paths)
        return
    for path in paths:
        g = group
        for p in path.strip('/').split('/'):
            g = _unique_child_group_h5(g, p) if p.startswith('NX') else g.get(p)
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


def _load_beer_mcstas(f, bank=1):
    for key in f['/entry1/instrument/components']:
        if 'sampleMantid' in key:
            sample_position_path = f'/entry1/instrument/components/{key}/Position'
            break
    else:
        raise ValueError('Sample position entry not found in file.')
    data, events, params, sample_pos, chopper_pos = _load_h5(
        f,
        f'NXentry/NXdetector/bank{bank:02}_events_dat_list_p_x_y_n_id_t',
        f'NXentry/NXdetector/bank{bank:02}_events_dat_list_p_x_y_n_id_t/events',
        'NXentry/simulation/Param',
        sample_position_path,
        '/entry1/instrument/components/0017_cMCA/Position',
    )
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
        if k in ('mode', 'sample_filename'):
            da.coords[k] = sc.scalar(v)

    da.coords['sample_position'] = sc.vector(sample_pos[:], unit='m')
    da.coords['detector_position'] = sc.vector(
        list(map(float, da.coords.pop('position').value.split(' '))), unit='m'
    )
    da.coords['chopper_position'] = sc.vector(chopper_pos[:], unit='m')
    da.coords['x'].unit = 'm'
    da.coords['y'].unit = 'm'
    da.coords['t'].unit = 's'

    z = sc.norm(da.coords['detector_position'] - da.coords['sample_position'])
    L1 = sc.norm(da.coords['sample_position'] - da.coords['chopper_position'])
    L2 = sc.sqrt(da.coords['x'] ** 2 + da.coords['y'] ** 2 + z**2)
    # Source is assumed to be at the origin
    da.coords['L0'] = L1 + L2 + sc.norm(da.coords['chopper_position'])
    da.coords['Ltotal'] = L1 + L2
    da.coords['two_theta'] = sc.acos(
        (-da.coords['x'] if bank == 1 else da.coords['x']) / L2
    )

    # Save some space
    da.coords.pop('x')
    da.coords.pop('y')
    da.coords.pop('n')

    da.coords['event_time_offset'] = da.coords.pop('t')
    return da


def load_beer_mcstas(f: str | Path | h5py.File) -> sc.DataGroup:
    '''Load beer McStas data from a file to a
    data group with one data array for each bank.
    '''
    if isinstance(f, str | Path):
        with h5py.File(f) as ff:
            return load_beer_mcstas(ff)

    return sc.DataGroup(
        {
            'bank1': _load_beer_mcstas(f, bank=1),
            'bank2': _load_beer_mcstas(f, bank=2),
        }
    )


def load_beer_mcstas_provider(
    fname: Filename[SampleRun], two_theta_mask: TwoThetaMaskFunction
) -> DetectorData[SampleRun]:
    da = load_beer_mcstas(fname)
    da = (
        sc.DataGroup(
            {
                k: v.assign_masks(two_theta=two_theta_mask(v.coords['two_theta']))
                for k, v in da.items()
            }
        )
        if isinstance(da, sc.DataGroup)
        else da.assign_masks(two_theta=two_theta_mask(da.coords['two_theta']))
    )
    return DetectorData[SampleRun](da)


def mcstas_chopper_delay_from_mode(
    da: DetectorData[SampleRun],
) -> WavelengthDefinitionChopperDelay:
    mode = next(iter(d.coords['mode'] for d in da.values())).value
    if mode in ('7', '8'):
        return sc.scalar(0.00245635, unit='s')
    if mode in ('9', '10'):
        return sc.scalar(0.0033730158730158727, unit='s')
    if mode == '16':
        return sc.scalar(0.000876984126984127, unit='s')
    raise ValueError(f'Mode {mode} is not known.')


def mcstas_modulation_period_from_mode(da: DetectorData[SampleRun]) -> ModulationPeriod:
    mode = next(iter(d.coords['mode'] for d in da.values())).value
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
    mcstas_chopper_delay_from_mode,
    mcstas_modulation_period_from_mode,
)
