# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp import constants as cst
from scipp.testing import assert_allclose

from ess.reflectometry import corrections


def test_footprint_correction():
    data = sc.DataArray(
        data=sc.array(dims=['row'], values=[1.0], unit='counts'),
        coords={
            "theta": sc.array(dims=['row'], values=[cst.pi.value / 6], unit='rad'),
            "px": sc.array(dims=['row'], values=[1], unit=None),
        },
    ).group('px')
    out = corrections.footprint_correction(
        data,
        beam_size=sc.scalar(5, unit='mm'),
        sample_size=sc.scalar(10, unit='mm'),
    )
    expected = sc.scalar(1, unit='counts') / sc.erf(
        1 / (sc.scalar(2) * sc.sqrt(sc.scalar(2.0) * sc.log(sc.scalar(2.0))))
    )
    assert_allclose(
        out.data.values[0].data[0],
        expected,
    )
