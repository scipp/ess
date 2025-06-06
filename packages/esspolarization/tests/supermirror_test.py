# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import io

import pytest
import scipp as sc
from scipp.testing import assert_allclose

import ess.polarization as pol
from ess.polarization.data import example_polarization_efficiency_table


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


def test_EfficiencyLookupTable_returns_expected_result():
    data = sc.midpoints(sc.linspace('wavelength', 0, 1, 10))
    data.variances = data.values
    tab = pol.EfficiencyLookupTable(
        sc.DataArray(
            data,
            coords={'wavelength': sc.linspace('wavelength', 0, 1, 10, unit='angstrom')},
        )
    )
    x = sc.midpoints(sc.linspace('wavelength', 0, 1, 10, unit='angstrom'))
    assert_allclose(
        tab(wavelength=x),
        sc.DataArray(sc.values(tab.table.data), coords={'wavelength': x}),
    )


def test_EfficiencyLookupTable_load_from_file():
    fname = io.StringIO('a,b,c\n1,2,3\n4,5,6')
    elt = pol.EfficiencyLookupTable.from_file(
        fname, wavelength_colname='a', efficiency_colname='b'
    )
    assert_allclose(
        sc.DataArray(
            sc.array(dims=('wavelength',), values=[2, 5]),
            coords={
                'wavelength': sc.array(
                    dims=('wavelength',), values=[1, 4], unit='angstrom'
                )
            },
        ),
        elt.table,
    )


def test_EfficiencyLookupTable_load_from_example_file():
    fname = example_polarization_efficiency_table()
    elt = pol.EfficiencyLookupTable.from_file(
        fname, wavelength_colname='# X ', efficiency_colname=' Y '
    )
    assert elt.table.size == 120
    assert elt.table.coords['wavelength'].min() == sc.scalar(3.05, unit='angstrom')
    assert elt.table.coords['wavelength'].max() == sc.scalar(14.95, unit='angstrom')
