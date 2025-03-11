# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

from ess.reflectometry.normalization import reduce_sample_over_zw


@pytest.fixture
def sample(request):
    n = 50
    da = sc.DataArray(
        data=sc.ones(dims=('events',), shape=(n,)),
        coords={
            'wavelength': sc.linspace('events', 1, 5, n),
            'wire': sc.array(dims=('events',), values=np.random.randint(0, 5, n)),
            'stripe': sc.array(dims=('events',), values=np.random.randint(0, 10, n)),
        },
    )
    return da.group('wire', 'stripe')


@pytest.fixture
def reference(request):
    n = 50
    da = sc.DataArray(
        data=sc.ones(dims=('events',), shape=(n,)),
        coords={
            'wavelength': sc.linspace('events', 1, 5, n),
            'wire': sc.array(dims=('events',), values=np.random.randint(0, 5, n)),
            'stripe': sc.array(dims=('events',), values=np.random.randint(0, 10, n)),
        },
    )
    return da.group('wire').bin(wavelength=2).bins.sum()


def test_reduce_sample_over_zw_when_data_not_dimensionless(sample, reference):
    sample = sample.copy(deep=True)
    sample.bins.unit = '1/s'
    reduce_sample_over_zw(
        sample,
        reference,
        reference.coords['wavelength'],
    )


def test_reduce_sample_over_zw_has_expected_coords(sample, reference):
    r = reduce_sample_over_zw(
        sample,
        reference,
        reference.coords['wavelength'],
    )
    assert r.dims == reference.dims
    assert r.shape == reference.shape
    assert not (sample.bins is None) ^ (r.bins is None)
