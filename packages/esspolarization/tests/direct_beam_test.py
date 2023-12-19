# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from ess import polarization as pol


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
        event_data=data,
        wavelength=wavelength,
        q_range=q_range,
        background_q_range=background_q_range,
    )
    assert db.bins is None
    assert db.dims == ('time', 'wavelength')
