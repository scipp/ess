# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from ess import polarization as pol


def dummy_analyzer_spin() -> pol.CellSpin[pol.Analyzer]:
    time = sc.array(dims=['time'], values=[0, 500, 1000], unit='s')
    spin = sc.array(dims=['time'], values=[-1, 1], unit=None)
    return pol.CellSpin[pol.Analyzer](sc.DataArray(spin, coords={'time': time}))


def dummy_polarizer_spin() -> pol.CellSpin[pol.Polarizer]:
    time = sc.array(dims=['time'], values=[0, 250, 750, 1000], unit='s')
    spin = sc.array(dims=['time'], values=[-1, 1, -1], unit=None)
    return pol.CellSpin[pol.Polarizer](sc.DataArray(spin, coords={'time': time}))


# TODO we might not want this combine RunSection, keep inout states separate
def dummy_run_section() -> pol.RunSection:
    time = sc.array(dims=['time'], values=[0, 250, 500, 750], unit='s')
    subtime = sc.array(dims=['subtime'], values=[0, 20, 60, 100], unit='s')
    time = (time + subtime).flatten(to='time')
    time = sc.concat([time, sc.scalar(1000, unit='s')], 'time')
    # 0 = direct beam no cell
    # 1 = direct beam with polarizer
    # 2 = direct beam with analyzer
    # 3 = sample run
    section = sc.array(dims=['time'], values=[0, 1, 2, 3], unit=None)
    section = sc.concat([section] * 4, 'time')
    return pol.RunSection(sc.DataArray(section, coords={'time': time}))


def make_events(size: int = 1000) -> sc.DataArray:
    rng = np.random.default_rng()
    time = sc.array(dims=['event'], values=rng.integers(0, 1000, size), unit='s')
    values = sc.array(dims=['event'], values=rng.uniform(0.0, 1.0, size))
    return sc.DataArray(values, coords={'time': time})


def test_raw_data_by_run_section() -> None:
    events = make_events()
    section = dummy_run_section()
    result = pol.raw_data_by_run_section(events, section)
    result = result[result.coords['run_section'] == sc.index(0)]
    print(result)
    assert False


def make_iofq_numerator(size: int = 1000) -> sc.DataArray:
    pass


def make_IofQ(size: int = 1000) -> sc.DataArray:
    rng = np.random.default_rng()
    wavelength = sc.array(
        dims=['event'], values=rng.uniform(0.5, 5.0, size), unit='angstrom'
    )
    q = sc.array(dims=['event'], values=rng.uniform(0.0, 3.0, size), unit='1/angstrom')
    weights = sc.array(dims=['event'], values=rng.uniform(0.0, 1.0, size))
    # There are different DB runs at different times, we assume in `direct_beam` this
    # has been grouped by time already.
    time = sc.array(dims=['event'], values=rng.integers(0, 10, size))
    events = sc.DataArray(
        weights,
        coords={
            'wavelength': wavelength,
            'Q': q,
            'time': time,
        },
    )
    return events.group('time')


def test_run_section() -> None:
    section = dummy_run_section()
    print(section.coords['time'].values)
    print(section.values)
    assert False


def test_direct_beam_returns_expected_dims() -> None:
    data = make_IofQ()
    wavelength = sc.linspace(
        dim='wavelength', start=0.5, stop=5.0, num=100, unit='angstrom'
    )
    q_range = sc.array(dims=['Q'], values=[0.0, 1.0], unit='1/angstrom')
    background_q_range = sc.array(dims=['Q'], values=[1.0, 2.0], unit='1/angstrom')

    db = pol.direct_beam(
        event_data=data,
        wavelength=wavelength,
        q_range=q_range,
        background_q_range=background_q_range,
    )
    assert db.bins is None
    assert db.dims == ('time', 'wavelength')
