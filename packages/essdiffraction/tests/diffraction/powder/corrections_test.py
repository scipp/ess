# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

from ess.diffraction.powder import merge_calibration


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
