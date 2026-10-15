# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Read weighted events from MAGIC McStas absorption loggers."""

from pathlib import Path

import mcstastox
import scipp as sc

from .types import DetectorBank, Filename, RawMcStasEvents, RunType

__all__ = ['load_mcstas_events']

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
    filename: Filename[RunType], bank: DetectorBank
) -> RawMcStasEvents[RunType]:
    """Read all rows of the selected bank's McStas absorption logger.

    The data contains the unnormalized simulation weights ``p``, using ``counts``
    as a weighted-count convention, not an absolute count rate. Variances are
    ``p**2``, as in the MAGIC Mantid converter and other ESS McStas readers.
    This assumes independent rows; correlations from McStas splitting are not
    represented.

    Coordinates ``x``, ``y``, ``z`` (m) and ``vx``, ``vy``, ``vz`` (m/s) are in
    the logger's local frame. ``t`` (s) is the original McStas arrival time,
    without an offset or pulse folding. Row order and zero weights are retained.
    No detector acceptance cuts, pixel assignment, or normalization are applied.

    The selected logger is loaded in full into memory.
    """
    component = {
        DetectorBank.detector_a: 'A_abs_logger',
        DetectorBank.detector_b: 'B_abs_logger',
    }[bank]
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
    return RawMcStasEvents[RunType](
        sc.DataArray(
            data,
            coords={
                name: sc.array(dims=['event'], values=values.pop(name), unit=unit)
                for name, unit in _COORDINATE_UNITS.items()
            },
        )
    )
