# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import pi
from scipp.testing import assert_allclose

from ess.reflectometry import corrections


def test_footprint_on_sample():
    footprint = corrections.footprint_on_sample(
        pi / 6 * sc.scalar(1, unit='rad'),
        beam_size=sc.scalar(5, unit='mm'),
        sample_size=sc.scalar(10, unit='mm'),
    )
    expected = sc.erf(
        1 / (sc.scalar(2) * sc.sqrt(sc.scalar(2.0) * sc.log(sc.scalar(2.0))))
    )
    assert_allclose(
        footprint,
        expected,
    )
