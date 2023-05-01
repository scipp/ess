# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import tempfile
import scipp as sc
import scippnexus.v2 as snx
from scipp.testing import assert_identical

from ess.masking import save_detector_masks


def roundtrip(obj):
    with tempfile.TemporaryDirectory() as path:
        name = f'{path}/test.hdf5'
        save_detector_masks(name, obj)
        with snx.File(name) as f:
            return f[()]


def test_save_detector_masks_rountrip():
    dg = sc.DataGroup()
    dg['det1'] = sc.DataArray(
        data=sc.ones(dims=['x', 'y'], shape=[2, 2]),
        masks={
            'x': sc.array(dims=['x'], values=[True, False]),
            'x2': sc.array(dims=['x'], values=[True, True]),
            'y': sc.array(dims=['y'], values=[False, True]),
        },
    )
    dg['det2'] = sc.DataArray(
        data=sc.ones(dims=['x', 'y'], shape=[2, 2]),
        masks={'yx': sc.array(dims=['y', 'x'], values=[[True, False], [False, True]])},
    )
    dg['det3'] = sc.DataArray(data=sc.ones(dims=['x', 'y'], shape=[2, 2]))
    result = roundtrip(dg)
    for name in dg:
        assert_identical(
            result['entry']['instrument'][name], sc.DataGroup(dg[name].masks)
        )


def test_save_detector_masks_does_not_overwrite():
    dg = sc.DataGroup()
    dg['det1'] = sc.DataArray(
        data=sc.ones(dims=['x', 'y'], shape=[2, 2]),
        masks={'x': sc.array(dims=['x'], values=[True, False])},
    )
    with tempfile.TemporaryDirectory() as path:
        name = f'{path}/test.hdf5'
        save_detector_masks(name, dg)
        with pytest.raises(FileExistsError):
            save_detector_masks(name, dg)
