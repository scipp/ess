# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess import polarization as pol


def test_opacity_from_cell_params() -> None:
    # In practice, the cell pressure, length, and temperature are constant, this is
    # just for testing.
    pressure = sc.array(dims=['pressure'], values=[1.0, 2.0], unit='bar')
    length = sc.array(dims=['cell_length'], values=[1.0, 2.0], unit='m')
    temperature = sc.array(dims=['temperature'], values=[200.0, 400.0], unit='K')
    wavelength = sc.array(dims=['wavelength'], values=[1.0, 2.0], unit='nm')
    opacity_function = pol.he3_opacity_from_cell_params(
        pressure=pressure, length=length, temperature=temperature
    )
    opacity = opacity_function(wavelength)
    assert_identical(2 * opacity['pressure', 0], opacity['pressure', 1])
    assert_identical(2 * opacity['cell_length', 0], opacity['cell_length', 1])
    assert_identical(2 * opacity['wavelength', 0], opacity['wavelength', 1])
    assert_identical(opacity['temperature', 0], 2 * opacity['temperature', 1])
    assert opacity.unit == ''


def test_opacity_from_cell_params_raises_with_temperature_in_degree_celsius() -> None:
    pressure = sc.scalar(1.0, unit='bar')
    length = sc.scalar(1.0, unit='m')
    temperature = sc.scalar(200.0, unit='degC')
    with pytest.raises(sc.UnitError):
        pol.he3_opacity_from_cell_params(
            pressure=pressure, length=length, temperature=temperature
        )


def test_opacity_from_beam_data() -> None:
    wavelength = sc.array(dims=['wavelength'], values=[1.0, 2.0], unit='nm')
    transmission_empty_glass = sc.scalar(0.5)
    direct_beam = sc.array(dims=['wavelength'], values=[2.0, 3.0], unit='counts')
    direct_beam = sc.DataArray(direct_beam, coords={'wavelength': wavelength})
    # Pretend known opacity0 for testing
    opacity0 = sc.scalar(0.3, unit='1/nm')
    ratio = transmission_empty_glass * sc.exp(-opacity0 * wavelength)
    direct_beam_cell = ratio * direct_beam
    opacity_function = pol.he3_opacity_from_beam_data(
        transmission_empty_glass=transmission_empty_glass,
        direct_beam=direct_beam,
        direct_beam_cell=direct_beam_cell,
    )
    opacity = opacity_function(wavelength)
    assert_identical(2 * opacity['wavelength', 0], opacity['wavelength', 1])
    assert sc.isclose(
        opacity_function.opacity0, opacity0.to(unit=opacity_function.opacity0.unit)
    )
