# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Read weighted events and detector geometry from MAGIC McStas simulations."""

from pathlib import Path

import mcstastox
import numpy as np
import scipp as sc

from ess.reduce.unwrap import PulsePeriod

from .types import DetectorBank, EmptyDetector, Filename, RawMcStasEvents, RunType

__all__ = ['load_mcstas_events', 'mcstas_detector_geometry']

_LOGGERS = {
    DetectorBank.detector_a: 'A_abs_logger',
    DetectorBank.detector_b: 'B_abs_logger',
}

_COORDINATE_UNITS = {
    'x': 'm',
    'y': 'm',
    'z': 'm',
    'vx': 'm/s',
    'vy': 'm/s',
    'vz': 'm/s',
    't': 's',
}


def load_mcstas_events(
    filename: Filename[RunType], bank: DetectorBank, pulse_period: PulsePeriod
) -> RawMcStasEvents[RunType]:
    """Read all rows of the selected bank's McStas absorption logger.

    The data contains the unnormalized simulation weights ``p``, using ``counts``
    as a weighted-count convention, not an absolute count rate. Variances are
    ``p**2``, as in the MAGIC Mantid converter and other ESS McStas readers.
    This assumes independent rows; correlations from McStas splitting are not
    represented.

    Coordinates ``x``, ``y``, ``z`` (m) and ``vx``, ``vy``, ``vz`` (m/s) are in
    the logger's local frame. ``t`` (s) is the original McStas arrival time,
    retained for diagnostics. Pulse-relative offsets are in ``event_time_offset``
    (s); ``event_time_zero`` uses the Unix epoch as the simulation origin, with
    pulse timestamps rounded to nanoseconds. Row order and zero weights are retained.
    No detector acceptance cuts, pixel assignment, or normalization are applied.

    The selected logger is loaded in full into memory.
    """
    period = pulse_period.to(unit='s', dtype='float64')
    if period.ndim or not np.isfinite(period.value) or period.value <= 0:
        raise ValueError('PulsePeriod must be a finite positive scalar duration.')
    component = _LOGGERS[bank]
    filename = Path(filename)
    with mcstastox.Read(filename.parent, filename.name) as reader:
        columns = reader.get_component_variables(component).split()
        expected = ['p', *_COORDINATE_UNITS]
        if len(columns) != len(expected) or set(columns) != set(expected):
            raise ValueError(f'{component} columns must be {expected}; got {columns}')
        values = reader.get_event_data(
            variables=expected, component_name=component, filter_zeros=False
        )

    weights = values.pop('p')
    data = sc.array(dims=['event'], values=weights, variances=weights**2, unit='counts')
    del weights
    coords = {
        name: sc.array(dims=['event'], values=values.pop(name), unit=unit)
        for name, unit in _COORDINATE_UNITS.items()
    }
    time = coords['t']
    if not sc.isfinite(time).all().value:
        raise ValueError('McStas arrival times must be finite.')
    offset = time % period
    # Keep round-off at frame boundaries inside the half-open pulse interval.
    offset = sc.where(offset < sc.scalar(0.0, unit='s'), offset + period, offset)
    offset = sc.where(offset >= period, offset - period, offset)
    coords['event_time_offset'] = offset
    coords['event_time_zero'] = sc.datetime(0, unit='ns') + sc.round(
        (time - offset).to(unit='ns')
    ).to(dtype='int64')
    return RawMcStasEvents[RunType](sc.DataArray(data, coords=coords))


def _geometry_parameter(parameters, name: str) -> float:
    try:
        value = float(np.asarray(parameters[name][()]).item())
    except (TypeError, ValueError) as error:
        raise ValueError(f'McStas parameter {name} must be a scalar number.') from error
    if not np.isfinite(value):
        raise ValueError(f'McStas parameter {name} must be finite.')
    return value


def mcstas_detector_geometry(
    filename: Filename[RunType], bank: DetectorBank
) -> EmptyDetector[RunType]:
    """Construct pixel IDs and positions without reading detector events.

    Uses the MAGIC cassette model from ``magic-notebooks``' ``voxelization.py``:
    both counters in a cassette share the central absorption plane. IDs increase
    with counter first, then anode, then cathode. McStasToX applies the logger's
    placement to put positions in the common simulation frame, in metres.

    Bank A reads ``A_r1`` (m), ``A_section_length`` (m), ``A_casette_gamma``
    (deg) and ``A_casette_omega`` (deg). Bank B uses the reference model's
    1 m radius, 0.53 m length and -0.94 degree cassette spacing, and requires
    ``B_casette_omega`` (deg). Channel counts and trapezoid heights are fixed
    by the reference model.
    """
    component = _LOGGERS[bank]
    filename = Path(filename)
    with mcstastox.Read(filename.parent, filename.name) as reader:
        parameters = reader.file['entry1/simulation/Param']
        if bank is DetectorBank.detector_a:
            cassettes, anodes, first_id = 60, 128, 1
            radius, length, spacing, tilt = (
                _geometry_parameter(parameters, name)
                for name in (
                    'A_r1',
                    'A_section_length',
                    'A_casette_gamma',
                    'A_casette_omega',
                )
            )
            top, bottom = (0.252863, 0.416545), (-0.722903, -1.077999)
        else:
            cassettes, anodes, first_id = 128, 16, 491521
            radius, length, spacing = 1.0, 0.53, -0.94
            tilt = _geometry_parameter(parameters, 'B_casette_omega')
            top, bottom = (0.051016, 0.077887), (-0.051016, -0.077887)
        if radius <= 0 or length <= 0 or spacing == 0 or abs(tilt) >= 90:
            raise ValueError(
                'Detector geometry requires positive radius and length, nonzero '
                'cassette spacing, and a cassette tilt strictly between '
                '-90 and 90 degrees.'
            )

        cathodes = 32
        index = np.arange(2 * cassettes * anodes * cathodes)
        counter = index % (2 * cassettes)
        anode = (index // (2 * cassettes)) % anodes
        cathode = index // (2 * cassettes * anodes)
        fraction = (cathode + 0.5) / cathodes
        depth = length * fraction
        gamma = (counter // 2) * np.deg2rad(spacing)
        tilt = np.deg2rad(tilt)
        upper = top[0] + (top[1] - top[0]) * fraction
        lower = bottom[0] + (bottom[1] - bottom[0]) * fraction
        local = np.column_stack(
            (
                radius * np.sin(gamma) + depth * np.sin(gamma + tilt),
                upper + (lower - upper) * (anode + 0.5) / anodes,
                radius * np.cos(gamma) + depth * np.cos(gamma + tilt),
            )
        )
        # Reuse McStasToX's row-vector placement convention, as FREIA does.
        position = reader.transform(local, component)

    pixel_id = sc.arange(
        'pixel_id', first_id, first_id + len(index), unit=None, dtype='int32'
    )
    return EmptyDetector[RunType](
        sc.DataArray(
            pixel_id,
            coords={
                'pixel_id': pixel_id,
                'position': sc.vectors(dims=['pixel_id'], values=position, unit='m'),
            },
        )
    )
