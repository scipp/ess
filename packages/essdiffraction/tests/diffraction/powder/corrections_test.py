# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
import scipp.testing

from ess.diffraction.powder import merge_calibration
from ess.diffraction.powder.correction import lorentz_factor


@pytest.fixture
def calibration():
    rng = np.random.default_rng(789236)
    n = 30
    ds = sc.Dataset(
        data={
            'difa': sc.array(
                dims=['spectrum'],
                values=rng.uniform(1.0e2, 1.0e3, n),
                unit='us / angstrom**2',
            ),
            'difc': sc.array(
                dims=['spectrum'],
                values=rng.uniform(1.0e3, 1.0e4, n),
                unit='us / angstrom',
            ),
            'tzero': sc.array(
                dims=['spectrum'], values=rng.uniform(-1e2, 1e2, n), unit='us'
            ),
            'mask': sc.full(dims=['spectrum'], shape=[n], value=False, unit=None),
        },
        coords={'spectrum': sc.arange('spectrum', n, unit=None)},
    )
    return ds


def test_merge_calibration_add_all_parameters(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.arange('spectrum', calibration.sizes['spectrum'], unit=None)
        },
    )
    with_cal = merge_calibration(into=da, calibration=calibration)

    assert sc.identical(with_cal.coords['difa'], calibration['difa'].data)
    assert sc.identical(with_cal.coords['difc'], calibration['difc'].data)
    assert sc.identical(with_cal.coords['tzero'], calibration['tzero'].data)
    assert sc.identical(with_cal.masks['calibration'], calibration['mask'].data)


def test_merge_calibration_raises_if_spectrum_mismatch(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.zeros(
                sizes={'spectrum': calibration.sizes['spectrum']}, unit=None
            )
        },
    )
    with pytest.raises(ValueError):
        merge_calibration(into=da, calibration=calibration)


def test_merge_calibration_raises_if_difa_exists(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.arange('spectrum', calibration.sizes['spectrum'], unit=None),
            'difa': sc.ones(sizes={'spectrum': calibration.sizes['spectrum']}),
        },
    )
    with pytest.raises(ValueError):
        merge_calibration(into=da, calibration=calibration)


def test_merge_calibration_raises_if_difc_exists(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.arange('spectrum', calibration.sizes['spectrum'], unit=None),
            'difc': sc.ones(sizes={'spectrum': calibration.sizes['spectrum']}),
        },
    )
    with pytest.raises(ValueError):
        merge_calibration(into=da, calibration=calibration)


def test_merge_calibration_raises_if_tzero_exists(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.arange('spectrum', calibration.sizes['spectrum'], unit=None),
            'tzero': sc.ones(sizes={'spectrum': calibration.sizes['spectrum']}),
        },
    )
    with pytest.raises(ValueError):
        merge_calibration(into=da, calibration=calibration)


def test_merge_calibration_raises_if_mask_exists(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.arange('spectrum', calibration.sizes['spectrum'], unit=None)
        },
        masks={
            'calibration': sc.ones(sizes={'spectrum': calibration.sizes['spectrum']})
        },
    )
    with pytest.raises(ValueError):
        merge_calibration(into=da, calibration=calibration)


def test_lorentz_factor_dense_1d_coords():
    da = sc.DataArray(
        sc.ones(sizes={'detector_number': 3, 'dspacing': 4}),
        coords={
            'dspacing': sc.array(
                dims=['dspacing'], values=[0.1, 0.4, 0.7, 1.1], unit='angstrom'
            ),
            'two_theta': sc.array(
                dims=['detector_number'], values=[0.8, 0.9, 1.3], unit='rad'
            ),
            'detector_number': sc.array(
                dims=['detector_number'], values=[0, 1, 2], unit=None
            ),
        },
    )
    factor = lorentz_factor(da)

    assert factor.sizes == {'detector_number': 3, 'dspacing': 4}
    assert factor.unit == 'angstrom**4'
    assert factor.dtype == da.dtype
    d = da.coords['dspacing'].broadcast(sizes=factor.sizes).values
    two_theta = da.coords['two_theta'].broadcast(sizes=factor.sizes).values
    np.testing.assert_allclose(factor.data.values, d**4 * np.sin(two_theta / 2))

    assert set(factor.coords.keys()) == {'two_theta', 'dspacing'}
    for key in factor.coords:
        sc.testing.assert_identical(factor.coords[key], da.coords[key])


def test_lorentz_factor_dense_2d_coord():
    da = sc.DataArray(
        sc.ones(sizes={'detector_number': 3, 'dspacing': 4}),
        coords={
            'dspacing': sc.array(
                dims=['dspacing'], values=[0.1, 0.4, 0.7, 1.1], unit='angstrom'
            ).broadcast(sizes={'detector_number': 3, 'dspacing': 4}),
            'two_theta': sc.array(
                dims=['detector_number'], values=[0.8, 0.9, 1.3], unit='rad'
            ),
            'detector_number': sc.array(
                dims=['detector_number'], values=[0, 1, 2], unit=None
            ),
        },
    )
    factor = lorentz_factor(da)

    assert factor.sizes == {'detector_number': 3, 'dspacing': 4}
    assert factor.unit == 'angstrom**4'
    assert factor.dtype == da.dtype
    d = da.coords['dspacing'].values
    two_theta = da.coords['two_theta'].broadcast(sizes=factor.sizes).values
    np.testing.assert_allclose(factor.data.values, d**4 * np.sin(two_theta / 2))

    assert set(factor.coords.keys()) == {'two_theta', 'dspacing'}
    for key in factor.coords:
        sc.testing.assert_identical(factor.coords[key], da.coords[key])


def test_lorentz_factor_event_coords():
    buffer = sc.DataArray(
        sc.ones(sizes={'event': 6}),
        coords={
            'detector_number': sc.array(dims=['event'], values=[0, 3, 2, 2, 0, 4]),
            'dspacing': sc.array(
                dims=['event'], values=[0.1, 0.4, 0.2, 1.0, 1.3, 0.7], unit='angstrom'
            ),
            'two_theta': sc.array(
                dims=['event'], values=[0.8, 1.2, 1.0, 1.0, 0.8, 0.7], unit='rad'
            ),
        },
    )
    da = buffer.group('detector_number').bin(dspacing=2)
    factor = lorentz_factor(da)

    assert factor.sizes == {'detector_number': 4, 'dspacing': 2}
    assert factor.bins.unit == 'angstrom**4'
    assert factor.bins.dtype == da.bins.dtype
    d = buffer.coords['dspacing']
    two_theta = buffer.coords['two_theta']
    expected_buffer = sc.DataArray(d**4 * sc.sin(two_theta / 2), coords=buffer.coords)
    expected = expected_buffer.group('detector_number').bin(dspacing=2)
    np.testing.assert_allclose(
        factor.bins.concat().value.values, expected.bins.concat().value.values
    )


def test_lorentz_factor_needs_coords():
    da = sc.DataArray(
        sc.ones(sizes={'detector_number': 3, 'dspacing': 4}),
        coords={
            'detector_number': sc.array(
                dims=['detector_number'], values=[0, 1, 2], unit=None
            )
        },
    )
    with pytest.raises(KeyError):
        lorentz_factor(da)
