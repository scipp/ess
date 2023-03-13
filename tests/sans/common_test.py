# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc

from ess.sans.common import mask_range


def test_mask_range_dense_data():
    x = sc.arange('x', 5., unit='m')
    da = sc.DataArray(data=x, coords={'x': x})
    masked = mask_range(da,
                        edges=sc.array(dims=['x'], values=[1.9, 3.001], unit='m'),
                        mask=sc.array(dims=['x'], values=[True]),
                        name='mymask')
    assert masked.masks['mymask'].values.tolist() == [False, False, True, True, False]


def test_mask_range_dense_data_bin_edges():
    x = sc.arange('x', 6., unit='m')
    da = sc.DataArray(data=x[:-1], coords={'x': x})
    masked = mask_range(da,
                        edges=sc.array(dims=['x'], values=[1.9, 3.001], unit='m'),
                        mask=sc.array(dims=['x'], values=[True]),
                        name='mymask')
    assert masked.masks['mymask'].values.tolist() == [False, True, True, True, False]


def test_mask_range_dense_data_two_ranges():
    x = sc.arange('x', 11., unit='m')
    da = sc.DataArray(data=x[:-1], coords={'x': x})
    masked = mask_range(da,
                        edges=sc.array(dims=['x'],
                                       values=[1.9, 3.001, 6.5, 8.8],
                                       unit='m'),
                        mask=sc.array(dims=['x'], values=[True, False, True]),
                        name='mymask')
    assert masked.masks['mymask'].values.tolist() == [
        False, True, True, True, False, False, True, True, True, False
    ]


def test_mask_range_binned_data_no_prior_binning():
    x = sc.arange('x', 5., unit='m')
    da = sc.DataArray(data=x, coords={'x': x, 'y': x + x.max()})
    binned = da.bin(y=2)
    edges = sc.array(dims=['x'], values=[1.9, 3.001], unit='m')
    masked = mask_range(binned,
                        edges=edges,
                        mask=sc.array(dims=['x'], values=[True]),
                        name='mymask')
    assert masked.masks['mymask'].values.tolist() == [True]
    assert sc.identical(masked.coords['x'], edges)


def test_mask_range_binned_data_has_prior_single_bin():
    x = sc.arange('x', 5., unit='m')
    da = sc.DataArray(data=x, coords={'x': x, 'y': x + x.max()})
    binned = da.bin(x=1)
    edges = sc.array(dims=['x'], values=[1.5, 3.001], unit='m')
    masked = mask_range(binned,
                        edges=edges,
                        mask=sc.array(dims=['x'], values=[True]),
                        name='mymask')
    assert masked.masks['mymask'].values.tolist() == [False, True, False]
    assert np.allclose(masked.coords['x'].values, [0., 1.5, 3.001, 4.])


def test_mask_range_binned_data_has_prior_multiple_bins():
    x = sc.arange('x', 5., unit='m')
    da = sc.DataArray(data=x, coords={'x': x, 'y': x + x.max()})
    binned = da.bin(x=2)
    edges = sc.array(dims=['x'], values=[1.5, 3.001], unit='m')
    masked = mask_range(binned,
                        edges=edges,
                        mask=sc.array(dims=['x'], values=[True]),
                        name='mymask')
    assert masked.masks['mymask'].values.tolist() == [False, True, True, False]
    assert np.allclose(masked.coords['x'].values, [0., 1.5, 2., 3.001, 4.])


def test_mask_range_binned_data_has_already_same_edge_as_mask():
    x = sc.arange('x', 5., unit='m')
    da = sc.DataArray(data=x, coords={'x': x, 'y': x + x.max()})
    binned = da.bin(x=2)
    edges = sc.array(dims=['x'], values=[binned.coords['x'][1].value, 3.001], unit='m')
    masked = mask_range(binned,
                        edges=edges,
                        mask=sc.array(dims=['x'], values=[True]),
                        name='mymask')
    assert masked.masks['mymask'].values.tolist() == [False, True, False]
    assert np.allclose(masked.coords['x'].values, [0., 2., 3.001, 4.])
