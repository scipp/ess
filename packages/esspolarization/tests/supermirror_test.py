# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

import ess.polarization as pol


def test_SecondOrderPolynomialEfficiency_raises_if_bad_coeff_units():
    with pytest.raises(sc.UnitError, match="a must have unit"):
        pol.SecondOrderPolynomialEfficiency(
            a=sc.scalar(1.0, unit='1/angstrom'),
            b=sc.scalar(1.0, unit='1/angstrom'),
            c=sc.scalar(1.0),
        )
    with pytest.raises(sc.UnitError, match="b must have unit"):
        pol.SecondOrderPolynomialEfficiency(
            a=sc.scalar(1.0, unit='1/angstrom**2'),
            b=sc.scalar(1.0, unit='1/angstrom**2'),
            c=sc.scalar(1.0),
        )
    with pytest.raises(sc.UnitError, match="c must be dimensionless"):
        pol.SecondOrderPolynomialEfficiency(
            a=sc.scalar(1.0, unit='1/angstrom**2'),
            b=sc.scalar(1.0, unit='1/angstrom'),
            c=sc.scalar(1.0, unit='1/angstrom'),
        )


def test_SecondOrderPolynomialEfficiencyF_raises_if_bad_units():
    func = pol.SecondOrderPolynomialEfficiency(
        a=sc.scalar(1.0, unit='1/angstrom**2'),
        b=sc.scalar(1.0, unit='1/angstrom'),
        c=sc.scalar(1.0),
    )
    wav = sc.scalar(1.0, unit='m')
    with pytest.raises(sc.UnitError, match="wavelength must have unit"):
        func(wavelength=wav)


def test_SecondOrderPolynomialEfficiency_produces_correct_values():
    a = sc.scalar(1.0, unit='1/angstrom**2')
    b = sc.scalar(2.0, unit='1/angstrom')
    c = sc.scalar(3.0)
    f = pol.SecondOrderPolynomialEfficiency(a=a, b=b, c=c)
    assert f(wavelength=sc.scalar(0.0, unit='angstrom')) == 3.0
    assert f(wavelength=sc.scalar(1.0, unit='angstrom')) == 6.0
    assert f(wavelength=sc.scalar(2.0, unit='angstrom')) == 11.0
