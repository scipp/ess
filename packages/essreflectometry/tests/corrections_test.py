# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ess.reflectometry import corrections


def test_footprint_correction():
    data = sc.DataArray(
        data=sc.array(['row'], values=[1.0], unit='counts'),
        coords=dict(
            theta=sc.array(['row'], values=[sc.pi / 6], unit='rad'),
            px=sc.array(['row'], values=[1], unit=None),
        ),
    ).bin(px=1)
    out = corrections.footrint_correction(
        data,
        beam_size=sc.scalar(5, unit='mm'),
        sample_size=sc.scalar(10, unit='mm'),
    )
    sc.testing.assert_allclose(
        out.data.value[0],
        1
        / sc.erf(1 / (sc.scalar(2) * sc.sqrt(sc.scalar(2.0) * sc.log(sc.scalar(2.0))))),
    )
