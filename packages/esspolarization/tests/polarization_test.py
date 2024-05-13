# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ess import polarization as pol


def test_he3_polarization() -> None:
    time = sc.linspace('time', 0.0, 10000.0, num=1000, unit='s')
    wavelength = sc.linspace('wavelength', 0.5, 5.0, num=40, unit='angstrom')
    C = sc.scalar(0.0)
    T1 = sc.scalar(1000.0, unit='s')
    opacity0 = sc.scalar(17.0, unit='1/angstrom')
    polarization = pol.base.polarization_function(time=time, C=C, T1=T1)
    opacity = pol.base.OpacityFunction(opacity0)(wavelength)
    transmission = pol.base.transmission_function(
        opacity=opacity,
        polarization=polarization,
        transmission_empty_glass=sc.scalar(0.9),
    )
    assert transmission == 1
