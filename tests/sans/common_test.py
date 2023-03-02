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
