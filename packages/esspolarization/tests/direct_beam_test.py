# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from scipp.testing import assert_identical

from ess import polarization as pol

# Setup logs for four sections of length 250:
# - 10 s direct beam no cell (only in beginning)
# - For each section:
#   - 25 s direct beam with polarizer
#   - 25 s direct beam with analyzer
#   - 200 s sample run (4 states)
no_cell_start = sc.scalar(-10.0, unit='s')
section_length = sc.scalar(250.0, unit='s')


def _offset_time(da: sc.DataArray, offset: sc.Variable) -> sc.DataArray:
    da = da.copy()
    da.coords['time'] += offset
    return da


def dummy_analyzer_spin() -> pol.CellSpinLog[pol.Analyzer]:
    time = sc.array(dims=['time'], values=[0.0, 150], unit='s')
    spin = sc.array(dims=['time'], values=[1, -1], unit=None)
    da = sc.DataArray(spin, coords={'time': time})
    da = sc.concat(
        [
            _offset_time(da, 0 * section_length),
            _offset_time(da, 1 * section_length),
            _offset_time(da, 2 * section_length),
            _offset_time(da, 3 * section_length),
        ],
        'time',
    )
    da.coords['time'][0] = no_cell_start
    return pol.CellSpinLog[pol.Analyzer](da)


def dummy_polarizer_spin() -> pol.CellSpinLog[pol.Polarizer]:
    time = sc.array(dims=['time'], values=[100.0, 200], unit='s')
    spin = sc.array(dims=['time'], values=[-1, 1], unit=None)
    da = sc.DataArray(spin, coords={'time': time})
    start = da[1].copy()
    start.coords['time'] = no_cell_start
    da = sc.concat(
        [
            start,
            _offset_time(da, 0 * section_length),
            _offset_time(da, 1 * section_length),
            _offset_time(da, 2 * section_length),
            _offset_time(da, 3 * section_length),
        ],
        'time',
    )
    return pol.CellSpinLog[pol.Polarizer](da)


def dummy_sample_in_beam() -> pol.SampleInBeamLog:
    time = sc.array(
        dims=['time'],
        values=[no_cell_start.value, 50, 250, 300, 500, 550, 750, 800],
        unit='s',
    )
    in_beam = sc.array(
        dims=['time'], values=[False, True, False, True, False, True, False, True]
    )
    return pol.SampleInBeamLog(sc.DataArray(in_beam, coords={'time': time}))


def dummy_polarizer_in_beam() -> pol.CellInBeamLog[pol.Polarizer]:
    time = sc.array(dims=['time'], values=[25.0, 50], unit='s')
    time = sc.concat(
        [
            no_cell_start,
            sc.scalar(0.0, unit='s'),
            time + 0 * section_length,
            time + 1 * section_length,
            time + 2 * section_length,
            time + 3 * section_length,
        ],
        'time',
    )
    in_beam = sc.array(dims=['time'], values=[False, True], unit=None)
    in_beam = sc.concat([in_beam] * 5, 'time')
    return pol.CellInBeamLog[pol.Polarizer](
        sc.DataArray(in_beam, coords={'time': time})
    )


def dummy_analyzer_in_beam() -> pol.CellInBeamLog[pol.Analyzer]:
    time = sc.array(dims=['time'], values=[0.0, 25], unit='s')
    time = sc.concat(
        [
            time + 0 * section_length,
            time + 1 * section_length,
            time + 2 * section_length,
            time + 3 * section_length,
        ],
        'time',
    )
    time[0] = no_cell_start
    in_beam = sc.array(dims=['time'], values=[False, True], unit=None)
    in_beam = sc.concat([in_beam] * 4, 'time')
    return pol.CellInBeamLog[pol.Analyzer](sc.DataArray(in_beam, coords={'time': time}))


def make_events(size: int = 1000) -> sc.DataArray:
    rng = np.random.default_rng()
    time = sc.array(dims=['event'], values=rng.integers(0, 1000, size), unit='s')
    values = sc.array(dims=['event'], values=rng.uniform(0.0, 1.0, size))
    return sc.DataArray(values, coords={'time': time})


def test_determine_run_section() -> None:
    analyzer_spin = dummy_analyzer_spin()
    polarizer_spin = dummy_polarizer_spin()
    sample_in_beam = dummy_sample_in_beam()
    analyzer_in_beam = dummy_analyzer_in_beam()
    polarizer_in_beam = dummy_polarizer_in_beam()
    result = pol.determine_run_section(
        sample_in_beam=sample_in_beam,
        analyzer_in_beam=analyzer_in_beam,
        polarizer_in_beam=polarizer_in_beam,
        analyzer_spin=analyzer_spin,
        polarizer_spin=polarizer_spin,
    )
    # 1 no-cell + 4 sections of (2 single-cell direct-beam + 4 sample runs)
    assert result.sizes == {'time': 1 + 4 * (2 + 4)}
    expected_time = sc.array(
        dims=['time'],
        values=[
            -10.0,
            0.0,
            25.0,
            50.0,
            100.0,
            150.0,
            200.0,
            250.0,
            275.0,
            300.0,
            350.0,
            400.0,
            450.0,
            500.0,
            525.0,
            550.0,
            600.0,
            650.0,
            700.0,
            750.0,
            775.0,
            800.0,
            850.0,
            900.0,
            950.0,
        ],
        unit='s',
    )
    for value in result.values():
        assert_identical(value.coords['time'], expected_time)


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


def test_direct_beam_returns_expected_dims() -> None:
    data = make_IofQ()
    wavelength = sc.linspace(
        dim='wavelength', start=0.5, stop=5.0, num=100, unit='angstrom'
    )
    q_range = sc.array(dims=['Q'], values=[0.0, 1.0], unit='1/angstrom')
    background_q_range = sc.array(dims=['Q'], values=[1.0, 2.0], unit='1/angstrom')

    db = pol.direct_beam(
        data=data.bin(wavelength=wavelength),
        q_range=q_range,
        background_q_range=background_q_range,
    )
    assert db.bins is None
    assert db.dims == ('time', 'wavelength')
