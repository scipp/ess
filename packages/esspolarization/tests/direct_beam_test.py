# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose, assert_identical

from ess import polarization as pol
from ess.polarization import base

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


def dummy_analyzer_spin() -> base.CellSpinLog[pol.Analyzer]:
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
    return base.CellSpinLog[pol.Analyzer](da)


def dummy_polarizer_spin() -> base.CellSpinLog[pol.Polarizer]:
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
    return base.CellSpinLog[pol.Polarizer](da)


def dummy_sample_in_beam() -> base.SampleInBeamLog:
    time = sc.array(
        dims=['time'],
        values=[no_cell_start.value, 50, 250, 300, 500, 550, 750, 800],
        unit='s',
    )
    in_beam = sc.array(
        dims=['time'], values=[False, True, False, True, False, True, False, True]
    )
    return base.SampleInBeamLog(sc.DataArray(in_beam, coords={'time': time}))


def dummy_polarizer_in_beam() -> base.CellInBeamLog[pol.Polarizer]:
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
    return base.CellInBeamLog[pol.Polarizer](
        sc.DataArray(in_beam, coords={'time': time})
    )


def dummy_analyzer_in_beam() -> base.CellInBeamLog[pol.Analyzer]:
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
    return base.CellInBeamLog[pol.Analyzer](
        sc.DataArray(in_beam, coords={'time': time})
    )


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
    result = base.determine_run_section(
        sample_in_beam=sample_in_beam,
        analyzer_in_beam=analyzer_in_beam,
        polarizer_in_beam=polarizer_in_beam,
        analyzer_spin=analyzer_spin,
        polarizer_spin=polarizer_spin,
    )
    # 1 no-cell + 4 sections of (2 single-cell direct-beam + 4 sample runs)
    assert result.sizes == {'time': 1 + 4 * (2 + 4)}

    # Each section is: DB_pol, DB_ana, ++, -+, --, +-

    section = sc.array(
        dims=['time'], values=[0.0, 25.0, 50.0, 100.0, 150.0, 200.0], unit='s'
    )
    expected_time = sc.concat(
        [
            no_cell_start,
            section + 0 * section_length,
            section + 1 * section_length,
            section + 2 * section_length,
            section + 3 * section_length,
        ],
        'time',
    )
    for value in result.values():
        assert_identical(value.coords['time'], expected_time)

    section = [False, False, True, True, True, True]
    expected_sample_in_beam = sc.array(dims=['time'], values=[False] + 4 * section)
    assert_identical(result['sample_in_beam'].data, expected_sample_in_beam)

    section = [True, False, True, True, True, True]
    expected_polarizer_in_beam = sc.array(dims=['time'], values=[False] + 4 * section)
    assert_identical(result['polarizer_in_beam'].data, expected_polarizer_in_beam)

    section = [False, True, True, True, True, True]
    expected_analyzer_in_beam = sc.array(dims=['time'], values=[False] + 4 * section)
    assert_identical(result['analyzer_in_beam'].data, expected_analyzer_in_beam)

    section = [1, 1, 1, -1, -1, 1]
    expected_polarizer_spin = sc.array(
        dims=['time'], values=[1] + 4 * section, unit=None
    )
    assert_identical(result['polarizer_spin'].data, expected_polarizer_spin)

    section = [1, 1, 1, 1, -1, -1]
    expected_analyzer_spin = sc.array(
        dims=['time'], values=[1] + 4 * section, unit=None
    )
    assert_identical(result['analyzer_spin'].data, expected_analyzer_spin)


def make_IofQ(size: int = 1000) -> sc.DataArray:
    rng = np.random.default_rng()
    wavelength = sc.array(
        dims=['event'], values=rng.uniform(0.5, 5.0, size), unit='angstrom'
    )
    qx = sc.array(
        dims=['event'], values=rng.uniform(-3.1, 3.0, size), unit='1/angstrom'
    )
    qy = sc.array(
        dims=['event'], values=rng.uniform(-2.6, 2.5, size), unit='1/angstrom'
    )
    weights = sc.array(dims=['event'], values=rng.uniform(0.0, 1.0, size))
    # There are different DB runs at different times, we assume in `direct_beam` this
    # has been grouped by time already.
    time = sc.array(dims=['event'], values=rng.integers(0, 10, size))
    events = sc.DataArray(
        weights,
        coords={
            'wavelength': wavelength,
            'Qx': qx,
            'Qy': qy,
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

    db = pol.he3.direct_beam(
        data=data.bin(wavelength=wavelength),
        q_range=q_range,
        background_q_range=background_q_range,
    )
    assert db.bins is None
    assert db.dims == ('time', 'wavelength')


def test_direct_beam_raises_if_q_ranges_overlap() -> None:
    data = make_IofQ()
    wavelength = sc.linspace(
        dim='wavelength', start=0.5, stop=5.0, num=100, unit='angstrom'
    )
    q_range = sc.array(dims=['Q'], values=[0.0, 1.0], unit='1/angstrom')
    background_q_range = sc.array(dims=['Q'], values=[0.5, 2.0], unit='1/angstrom')

    with pytest.raises(
        ValueError, match='Background range must be after direct beam range'
    ):
        pol.he3.direct_beam(
            data=data.bin(wavelength=wavelength),
            q_range=q_range,
            background_q_range=background_q_range,
        )


def test_direct_beam_raises_if_q_range_not_positive() -> None:
    data = make_IofQ()
    wavelength = sc.linspace(
        dim='wavelength', start=0.5, stop=5.0, num=100, unit='angstrom'
    )
    q_range = sc.array(dims=['Q'], values=[-2.0, 2.0], unit='1/angstrom')
    background_q_range = sc.array(dims=['Q'], values=[3.0, 4.0], unit='1/angstrom')

    with pytest.raises(ValueError, match='Q-range must be positive'):
        pol.he3.direct_beam(
            data=data.bin(wavelength=wavelength),
            q_range=q_range,
            background_q_range=background_q_range,
        )


def test_direct_beam_operates_on_normalized_data() -> None:
    data = make_IofQ(size=int(1e6))
    data2 = data.bins.concatenate(data)
    wavelength = sc.linspace(
        dim='wavelength', start=0.5, stop=5.0, num=100, unit='angstrom'
    )
    q_range = sc.array(dims=['Q'], values=[0.0, 1.0], unit='1/angstrom')
    background_q_range = sc.array(dims=['Q'], values=[1.0, 2.0], unit='1/angstrom')

    db = pol.he3.direct_beam(
        data=data.bin(wavelength=wavelength),
        q_range=q_range,
        background_q_range=background_q_range,
    )
    db2 = pol.he3.direct_beam(
        data=data2.bin(wavelength=wavelength),
        q_range=q_range,
        background_q_range=background_q_range,
    )
    assert_allclose(db, db2)


def test_direct_beam_raises_if_input_is_counts() -> None:
    data = make_IofQ()
    wavelength = sc.linspace(
        dim='wavelength', start=0.5, stop=5.0, num=100, unit='angstrom'
    )
    q_range = sc.array(dims=['Q'], values=[0.0, 1.0], unit='1/angstrom')
    background_q_range = sc.array(dims=['Q'], values=[1.0, 2.0], unit='1/angstrom')

    data.bins.unit = 'counts'
    with pytest.raises(ValueError, match='Input data must be normalized'):
        pol.he3.direct_beam(
            data=data.bin(wavelength=wavelength),
            q_range=q_range,
            background_q_range=background_q_range,
        )
