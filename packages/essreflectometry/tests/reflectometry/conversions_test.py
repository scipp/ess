# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.testing import assert_allclose

from ess.reflectometry.conversions import (
    reflectometry_q,
    reflectometry_q_x,
    reflectometry_q_z,
)


def test_offspecular_q_reduces_to_specular_q_when_angles_match():
    wavelength = sc.array(dims=['event'], values=[2.0, 4.0, 8.0], unit='angstrom')
    angle = sc.array(dims=['event'], values=[0.3, 1.0, 3.5], unit='deg')

    assert_allclose(
        reflectometry_q_x(wavelength, angle, angle),
        sc.zeros(dims=['event'], shape=[3], unit='1/angstrom'),
    )
    assert_allclose(
        reflectometry_q_z(wavelength, angle, angle),
        reflectometry_q(wavelength, angle),
    )
