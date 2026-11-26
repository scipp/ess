# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Synthetic files in the layout written by Amor since 2026."""

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

from ess.amor.geometry import Detector

CHOPPER_SPEED_RPM = 450.0
# One revolution of the chopper
PULSE_PERIOD_NS = round(60e9 / CHOPPER_SPEED_RPM)
PROTON_CURRENT_UA = 1500.0
_START_NS = 1_792_000_000_000_000_000
_EPOCH = '1970-01-01T00:00:00Z'


def _group(parent: h5py.Group, name: str, nx_class: str) -> h5py.Group:
    group = parent.create_group(name)
    group.attrs['NX_class'] = nx_class
    return group


def _dataset(parent: h5py.Group, name: str, value, units: str | None = None, **attrs):
    dataset = parent.create_dataset(name, data=value)
    if units is not None:
        dataset.attrs['units'] = units
    dataset.attrs.update(attrs)
    return dataset


def write_amor_file(
    path: Path,
    *,
    mu: float,
    nu: float,
    blades: tuple[int, int],
    n_events: int,
    n_pulses: int,
    low_current_pulses: int = 0,
    seed: int = 0,
) -> Path:
    """Write a file with events spread evenly over time and over some blades.

    Parameters
    ----------
    path:
        Name of the file.
    mu:
        Angle of the sample surface, in degrees.
    nu:
        Angle of the detector, in degrees.
    blades:
        First blade and one past the last blade that the events fall on.
    n_events:
        Number of events.
    n_pulses:
        Number of chopper revolutions.
    low_current_pulses:
        Number of revolutions at the end of the run where the proton current
        is far below the nominal current.
    seed:
        Seed of the random number generator.
    """
    rng = np.random.default_rng(seed)
    n_wires, n_strips = Detector.nWires.value, Detector.nStrips.value
    pulse_times = _START_NS + PULSE_PERIOD_NS * np.arange(n_pulses, dtype=np.uint64)

    pulse = np.sort(rng.integers(0, n_pulses, n_events))
    event_id = (
        1
        + rng.integers(*blades, n_events) * n_wires * n_strips
        + rng.integers(0, n_wires, n_events) * n_strips
        + rng.integers(0, n_strips, n_events)
    )

    with h5py.File(path, 'w') as f:
        entry = _group(f, 'entry1', 'NXentry')
        _dataset(entry, 'title', 'Synthetic Amor run')
        _dataset(entry, 'start_time', '2026-10-15T15:38:22Z')
        _dataset(entry, 'end_time', '2026-10-15T15:49:31Z')

        user = _group(entry, 'user', 'NXuser')
        _dataset(user, 'name', 'Max Mustermann')
        _dataset(user, 'email', 'max.mustermann@ess.eu')
        _dataset(user, 'affiliation', 'European Spallation Source ERIC')

        sample = _group(entry, 'sample', 'NXsample')
        _dataset(sample, 'name', 'synthetic')
        _dataset(sample, 'model', 'air | Si')

        instrument = _group(entry, 'Amor', 'NXinstrument')
        _dataset(instrument, 'name', 'Amor')
        source = _group(instrument, 'source', 'NXsource')
        _dataset(source, 'name', 'SINQ')
        _dataset(source, 'probe', 'neutron')
        chopper = _group(instrument, 'chopper', 'NXdisk_chopper')
        _dataset(chopper, 'distance', -14842.0)
        _dataset(chopper, 'pair_separation', np.float32(1000.0))
        _dataset(chopper, 'phase', 0.0, 'deg')
        _dataset(chopper, 'rotation_speed', CHOPPER_SPEED_RPM, 'rpm')

        configuration = _group(entry, 'measurement_configuration', 'NXcollection')
        _dataset(configuration, 'mu', mu, 'deg')
        _dataset(configuration, 'nu', nu, 'deg')

        detector = _group(entry, 'detector', 'NXdata')
        events = _group(detector, 'data', 'NXevent_data')
        _dataset(events, 'event_time_zero', pulse_times, 'ns', start=_EPOCH)
        _dataset(events, 'event_index', np.searchsorted(pulse, np.arange(n_pulses)))
        _dataset(
            events,
            'event_time_offset',
            rng.integers(0, PULSE_PERIOD_NS, n_events).astype(np.int32),
            'ns',
        )
        _dataset(events, 'event_id', event_id.astype(np.int32))

        # The current is logged at times unrelated to the chopper revolutions
        n_good = n_pulses - low_current_pulses
        log_times = [_START_NS - 10**9, _START_NS + PULSE_PERIOD_NS // 3]
        log_values = [PROTON_CURRENT_UA, PROTON_CURRENT_UA]
        if low_current_pulses > 0:
            log_times.append(int(pulse_times[n_good]) - PULSE_PERIOD_NS // 3)
            log_values.append(10.0)
        proton_current = _group(detector, 'proton_current', 'NXlog')
        _dataset(
            proton_current, 'time', np.array(log_times, np.uint64), 'ns', start=_EPOCH
        )
        _dataset(proton_current, 'value', np.array(log_values), 'uA')

        monitor = _group(entry, 'beam_monitor', 'NXdata')
        _dataset(monitor, 'time', pulse_times, 'ns', start=_EPOCH)
    return path


@pytest.fixture(scope='session')
def make_amor_file(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[..., Path]:
    """Factory for synthetic Amor files, see :func:`write_amor_file`."""
    folder = tmp_path_factory.mktemp('amor')
    count = iter(range(1_000_000))

    def make(**kwargs) -> Path:
        return write_amor_file(folder / f'amor2026n{next(count):06d}.hdf', **kwargs)

    return make
