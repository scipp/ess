# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from pathlib import Path

import h5py
import scipp as sc
import scipp.constants

from .types import (
    Filename,
    ModulationPeriod,
    RawDetector,
    SampleRun,
    TwoThetaLimits,
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


def _load_beer_mcstas(f, bank=1):
    positions = {
        name: f'/entry1/instrument/components/{key}/Position'
        for key in f['/entry1/instrument/components']
        for name in ['sampleMantid', 'PSC1', 'PSC2', 'PSC3', 'MCA', 'MCB', 'MCC']
        if name in key
    }
    (
        data,
        events,
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
        f'NXentry/NXdetector/bank{bank:02}_events_dat_list_p_x_y_n_id_t',
        f'NXentry/NXdetector/bank{bank:02}_events_dat_list_p_x_y_n_id_t/events',
        'NXentry/simulation/Param',
        positions['sampleMantid'],
        positions['PSC1'],
        positions['PSC2'],
        positions['PSC3'],
        positions['MCA'],
        positions['MCB'],
        positions['MCC'],
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

    t = da.coords.pop('t')
    da.coords['event_time_offset'] = t % sc.scalar(1 / 14, unit='s').to(unit=t.unit)
    da.coords["tc"] = (
        sc.constants.m_n
        / sc.constants.h
        * da.coords['wavelength_estimate']
        * da.coords['L0'].min().to(unit='angstrom')
    ).to(unit='s') - sc.scalar(1 / 14, unit='s') / 2

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


def _not_between(x, a, b):
    return (x < a) | (b < x)


def load_beer_mcstas_provider(
    fname: Filename[SampleRun], two_theta_limits: TwoThetaLimits
) -> RawDetector[SampleRun]:
    da = load_beer_mcstas(fname)
    da = (
        sc.DataGroup(
            {
                k: v.assign_masks(
                    two_theta=_not_between(v.coords['two_theta'], *two_theta_limits)
                )
                for k, v in da.items()
            }
        )
        if isinstance(da, sc.DataGroup)
        else da.assign_masks(
            two_theta=_not_between(da.coords['two_theta'], *two_theta_limits)
        )
    )
    return RawDetector[SampleRun](da)


def mcstas_chopper_delay_from_mode(
    da: RawDetector[SampleRun],
) -> WavelengthDefinitionChopperDelay:
    '''These settings are good for the set of McStas runs that we
    use in the docs currently.
    Eventually we will want to determine this from the chopper information
    in the files, but that information is not in the simulation output.'''
    mode = next(iter(d.coords['mode'] for d in da.values())).value
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
    mode = next(iter(d.coords['mode'] for d in da.values())).value
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
)
