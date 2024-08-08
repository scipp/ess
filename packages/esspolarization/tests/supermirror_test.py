# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

import ess.polarization as pol


def test_SecondDegreePolynomialEfficiency_raises_if_units_incompatible():
    wav = sc.scalar(1.0, unit='m')
    eff = pol.SecondDegreePolynomialEfficiency(
        a=sc.scalar(1.0, unit='1/angstrom'),
        b=sc.scalar(1.0, unit='1/angstrom'),
        c=sc.scalar(1.0),
    )
    with pytest.raises(sc.UnitError, match=" to `dimensionless` is not valid"):
        eff(wavelength=wav)

    eff = pol.SecondDegreePolynomialEfficiency(
        a=sc.scalar(1.0, unit='1/angstrom**2'),
        b=sc.scalar(1.0, unit='1/angstrom**2'),
        c=sc.scalar(1.0),
    )
    with pytest.raises(sc.UnitError, match=" to `dimensionless` is not valid"):
        eff(wavelength=wav)

    eff = pol.SecondDegreePolynomialEfficiency(
        a=sc.scalar(1.0, unit='1/angstrom**2'),
        b=sc.scalar(1.0, unit='1/angstrom'),
        c=sc.scalar(1.0, unit='1/angstrom'),
    )
    with pytest.raises(sc.UnitError, match=" to `dimensionless` is not valid"):
        eff(wavelength=wav)

    eff = pol.SecondDegreePolynomialEfficiency(
        a=sc.scalar(1.0, unit='1/angstrom**2'),
        b=sc.scalar(1.0, unit='1/angstrom'),
        c=sc.scalar(1.0),
    )
    with pytest.raises(sc.UnitError, match=" to `dimensionless` is not valid"):
        eff(wavelength=wav / sc.scalar(1.0, unit='s'))


def test_SecondDegreePolynomialEfficiency_produces_correct_values():
    a = sc.scalar(1.0, unit='1/angstrom**2')
    b = sc.scalar(2.0, unit='1/angstrom')
    c = sc.scalar(3.0)
    f = pol.SecondDegreePolynomialEfficiency(a=a, b=b, c=c)
    assert f(wavelength=sc.scalar(0.0, unit='angstrom')) == 3.0
    assert f(wavelength=sc.scalar(1.0, unit='angstrom')) == 6.0
    assert f(wavelength=sc.scalar(2.0, unit='angstrom')) == 11.0


def test_SecondDegreePolynomialEfficiency_converts_units():
    a = sc.scalar(1.0, unit='1/angstrom**2')
    b = sc.scalar(20.0, unit='1/nm')
    c = sc.scalar(3.0)
    f = pol.SecondDegreePolynomialEfficiency(a=a, b=b, c=c)
    assert f(wavelength=sc.scalar(0.0, unit='angstrom')) == 3.0
    assert f(wavelength=sc.scalar(1.0, unit='angstrom')) == 6.0
    assert f(wavelength=sc.scalar(2.0, unit='angstrom')) == 11.0
    assert f(wavelength=sc.scalar(0.0, unit='nm')) == 3.0
    assert f(wavelength=sc.scalar(0.1, unit='nm')) == 6.0
    assert f(wavelength=sc.scalar(0.2, unit='nm')) == 11.0
