# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from ess import polarization as pol


def dummy_analyzer_spin() -> pol.CellSpinLog[pol.Analyzer]:
    time = sc.array(dims=['time'], values=[0, 500], unit='s')
    spin = sc.array(dims=['time'], values=[-1, 1], unit=None)
    return pol.CellSpinLog[pol.Analyzer](sc.DataArray(spin, coords={'time': time}))


def dummy_polarizer_spin() -> pol.CellSpinLog[pol.Polarizer]:
    time = sc.array(dims=['time'], values=[0, 250, 750], unit='s')
    spin = sc.array(dims=['time'], values=[-1, 1, -1], unit=None)
    return pol.CellSpinLog[pol.Polarizer](sc.DataArray(spin, coords={'time': time}))


# Setup logs for for sections of length 250:
# - 10 s direct beam no cell
# - 20 s direct beam with polarizer
# - 20 s direct beam with analyzer
# - 200 s sample run
section_length = sc.scalar(250.0, unit='s')


def dummy_sample_in_beam() -> pol.SampleInBeamLog:
    time = sc.array(
        dims=['time'], values=[0, 50, 250, 300, 500, 550, 750, 800], unit='s'
    )
    in_beam = sc.array(
        dims=['time'], values=[False, True, False, True, False, True, False, True]
    )
    return pol.SampleInBeamLog(sc.DataArray(in_beam, coords={'time': time}))


def dummy_polarizer_in_beam() -> pol.CellInBeamLog[pol.Polarizer]:
    time = sc.array(dims=['time'], values=[0, 10, 30, 50], unit='s')
    time = sc.concat(
        [
            time + 0 * section_length,
            time + 1 * section_length,
            time + 2 * section_length,
            time + 3 * section_length,
        ],
        'time',
    )
    in_beam = sc.array(dims=['time'], values=[False, True, False, True], unit=None)
    in_beam = sc.concat([in_beam] * 4, 'time')
    return pol.CellInBeamLog[pol.Polarizer](
        sc.DataArray(in_beam, coords={'time': time})
    )


def dummy_analyzer_in_beam() -> pol.CellInBeamLog[pol.Analyzer]:
    time = sc.array(dims=['time'], values=[0, 30], unit='s')
    time = sc.concat(
        [
            time + 0 * section_length,
            time + 1 * section_length,
            time + 2 * section_length,
            time + 3 * section_length,
        ],
        'time',
    )
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
    assert result.sizes == {'time': 16}


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
