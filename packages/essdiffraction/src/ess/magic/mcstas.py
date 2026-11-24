# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Read weighted events and detector geometry from MAGIC McStas simulations."""

from dataclasses import dataclass
from pathlib import Path

import mcstastox
import numpy as np
import scipp as sc

from ess.reduce.unwrap import PulsePeriod

from .types import DetectorBank, EmptyDetector, Filename, RawDetector, RunType

__all__ = ['load_mcstas_events', 'mcstas_detector_geometry']

# Bound arithmetic temporaries without limiting the number of loaded events.
_ASSIGNMENT_CHUNK_SIZE = 2_000_000

_LOGGERS = {
    DetectorBank.detector_a: 'A_abs_logger',
    DetectorBank.detector_b: 'B_abs_logger',
}

_COORDINATE_UNITS = {
    'x': 'm',
    'y': 'm',
    'z': 'm',
    't': 's',
}


def _load_events(reader: mcstastox.Read, component: str) -> sc.DataArray:
    """Read all weighted logger rows in their local frame, before pixel assignment."""
    columns = reader.get_component_variables(component).split()
    expected = ['p', 'x', 'y', 'z', 'vx', 'vy', 'vz', 't']
    if len(columns) != len(expected) or set(columns) != set(expected):
        raise ValueError(f'{component} columns must be {expected}; got {columns}')
    values = reader.get_event_data(
        variables=['p', *_COORDINATE_UNITS],
        component_name=component,
        filter_zeros=False,
    )
    weights = values.pop('p')
    data = sc.array(dims=['event'], values=weights, variances=weights**2, unit='counts')
    del weights
    return sc.DataArray(
        data,
        coords={
            name: sc.array(dims=['event'], values=values.pop(name), unit=unit)
            for name, unit in _COORDINATE_UNITS.items()
        },
    )


def _geometry_parameter(parameters, name: str) -> float:
    try:
        value = float(np.asarray(parameters[name][()]).item())
    except (TypeError, ValueError) as error:
        raise ValueError(f'McStas parameter {name} must be a scalar number.') from error
    if not np.isfinite(value):
        raise ValueError(f'McStas parameter {name} must be finite.')
    return value


@dataclass(frozen=True)
class _DetectorModel:
    """Shared cassette parameters for pixel centres and their inverse mapping."""

    cassettes: int
    anodes: int
    first_id: int
    radius: float
    length: float
    spacing: float  # radians
    tilt: float  # radians
    top: tuple[float, float]
    bottom: tuple[float, float]
    cathodes: int = 32


def _detector_model(reader: mcstastox.Read, bank: DetectorBank) -> _DetectorModel:
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
    elif bank is DetectorBank.detector_b:
        cassettes, anodes, first_id = 128, 16, 491521
        radius, length, spacing = 1.0, 0.53, -0.94
        tilt = _geometry_parameter(parameters, 'B_casette_omega')
        top, bottom = (0.051016, 0.077887), (-0.051016, -0.077887)
    else:
        raise ValueError(f'Unknown MAGIC detector bank: {bank!r}')
    if radius <= 0 or length <= 0 or spacing == 0 or abs(tilt) >= 90:
        raise ValueError(
            'Detector geometry requires positive radius and length, nonzero '
            'cassette spacing, and a cassette tilt strictly between '
            '-90 and 90 degrees.'
        )
    return _DetectorModel(
        cassettes,
        anodes,
        first_id,
        radius,
        length,
        np.deg2rad(spacing),
        np.deg2rad(tilt),
        top,
        bottom,
    )


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
        model = _detector_model(reader, bank)
        index = np.arange(2 * model.cassettes * model.anodes * model.cathodes)
        counter = index % (2 * model.cassettes)
        anode = (index // (2 * model.cassettes)) % model.anodes
        cathode = index // (2 * model.cassettes * model.anodes)
        fraction = (cathode + 0.5) / model.cathodes
        depth = model.length * fraction
        gamma = (counter // 2) * model.spacing
        upper = model.top[0] + (model.top[1] - model.top[0]) * fraction
        lower = model.bottom[0] + (model.bottom[1] - model.bottom[0]) * fraction
        local = np.column_stack(
            (
                model.radius * np.sin(gamma) + depth * np.sin(gamma + model.tilt),
                upper + (lower - upper) * (anode + 0.5) / model.anodes,
                model.radius * np.cos(gamma) + depth * np.cos(gamma + model.tilt),
            )
        )
        # Reuse McStasToX's row-vector placement convention, as FREIA does.
        position = reader.transform(local, component)

    pixel_id = sc.arange(
        'pixel_id',
        model.first_id,
        model.first_id + len(index),
        unit=None,
        dtype='int32',
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


def _assign_pixel_ids(events: sc.DataArray, model: _DetectorModel) -> np.ndarray:
    # Keep the reference voxelization.py operation order at pixel boundaries.
    x, y, z = (
        events.coords[c].to(unit='m', dtype='float64', copy=False) for c in 'xyz'
    )
    radius_squared = x * x + z * z
    depth = (
        sc.sqrt(
            radius_squared
            - sc.scalar((model.radius * np.sin(model.tilt)) ** 2, unit='m^2')
        )
        - sc.scalar(model.radius * np.cos(model.tilt), unit='m')
    ) / sc.scalar(model.length, unit='m')
    cathode = sc.floor(model.cathodes * depth).values
    top, bottom = (
        sc.scalar(front, unit='m') + sc.scalar(back - front, unit='m') * depth
        for front, back in (model.top, model.bottom)
    )
    anode = sc.floor(model.anodes * ((y - top) / (bottom - top))).values
    correction = sc.asin(
        np.sin(model.tilt)
        * depth
        * sc.scalar(model.length, unit='m')
        / sc.sqrt(radius_squared)
    )
    fraction = (
        sc.atan2(y=x, x=z) - correction + sc.scalar(model.spacing * 0.5, unit='rad')
    ) / sc.scalar(model.cassettes * model.spacing, unit='rad')
    counter = sc.floor(2 * model.cassettes * fraction).values
    accepted = (
        (cathode >= 0)
        & (cathode < model.cathodes)
        & (anode >= 0)
        & (anode < model.anodes)
        & (counter >= 0)
        & (counter < 2 * model.cassettes)
    )
    pixel = np.full(events.sizes['event'], -1, dtype=np.int32)
    # Accepted indices are integral floats; IDs <= 622592 are exact in float64.
    pixel[accepted] = (
        counter[accepted]
        + 2 * model.cassettes * (anode[accepted] + model.anodes * cathode[accepted])
        + model.first_id
    ).astype(np.int32)
    return pixel


def load_mcstas_events(
    filename: Filename[RunType],
    bank: DetectorBank,
    detector: EmptyDetector[RunType],
    pulse_period: PulsePeriod,
) -> RawDetector[RunType]:
    """Load weighted events grouped by the selected bank's detector pixels.

    Pixel assignment uses logger-local positions and the MAGIC cassette model.
    Outside or invalid positions are excluded. Empty pixels, zero weights and
    detector masks are retained. Weights are unnormalized ``p`` with variances
    ``p**2``. Original arrival times remain in ``t`` (s). Pulse-relative offsets
    are in ``event_time_offset`` (s); ``event_time_zero`` uses the Unix epoch as
    the simulation origin, with pulse timestamps rounded to nanoseconds.
    """
    period = pulse_period.to(unit='s', dtype='float64')
    if period.ndim or not np.isfinite(period.value) or period.value <= 0:
        raise ValueError('PulsePeriod must be a finite positive scalar duration.')
    filename = Path(filename)
    with mcstastox.Read(filename.parent, filename.name) as reader:
        model = _detector_model(reader, bank)
        span = np.arctan2(
            model.length * np.sin(model.tilt),
            model.radius + model.length * np.cos(model.tilt),
        )
        ends = (-0.5 * model.spacing, (model.cassettes - 0.5) * model.spacing)
        if min(ends) + min(0.0, span) <= -np.pi or max(ends) + max(0.0, span) >= np.pi:
            raise ValueError('Detector angular footprint must stay within (-pi, pi).')
        raw = _load_events(reader, _LOGGERS[bank])

    pixel_id = sc.empty(sizes=raw.sizes, dtype='int32', unit=None)
    for start in range(0, raw.sizes['event'], _ASSIGNMENT_CHUNK_SIZE):
        stop = start + _ASSIGNMENT_CHUNK_SIZE
        pixel_id.values[start:stop] = _assign_pixel_ids(raw['event', start:stop], model)
    # Group only weights and times, avoiding copies of the absorption positions.
    events = sc.DataArray(raw.data, coords={'pixel_id': pixel_id, 't': raw.coords['t']})
    del raw
    grouped = events.group(detector.coords['pixel_id'])
    if grouped.bins.size().sum().value != np.count_nonzero(pixel_id.values >= 0):
        raise ValueError('Detector geometry does not contain all accepted pixel IDs.')
    time = grouped.bins.coords['t']
    if not sc.isfinite(time.bins.constituents['data']).all().value:
        raise ValueError('Accepted McStas arrival times must be finite.')
    offset = time % period
    # Keep round-off at frame boundaries inside the half-open pulse interval.
    offset = sc.where(offset < sc.scalar(0.0, unit='s'), offset + period, offset)
    offset = sc.where(offset >= period, offset - period, offset)
    grouped.bins.coords['event_time_offset'] = offset
    grouped.bins.coords['event_time_zero'] = sc.datetime(0, unit='ns') + sc.round(
        (time - offset).to(unit='ns')
    ).to(dtype='int64')
    return RawDetector[RunType](
        grouped.assign_coords(detector.coords).assign_masks(detector.masks)
    )
