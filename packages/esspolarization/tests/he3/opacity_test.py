# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.polarization import he3


def test_opacity_from_cell_params() -> None:
    # In practice, the cell pressure, length, and temperature are constant, this is
    # just for testing.
    pressure = sc.array(dims=['pressure'], values=[1.0, 2.0], unit='bar')
    length = sc.array(dims=['cell_length'], values=[1.0, 2.0], unit='m')
    temperature = sc.array(dims=['temperature'], values=[200.0, 400.0], unit='K')
    wavelength = sc.array(dims=['wavelength'], values=[1.0, 2.0], unit='nm')
    opacity0 = he3.he3_opacity_from_cell_params(
        pressure=pressure, length=length, temperature=temperature
    )
    opacity_function = he3.he3_opacity_function_from_cell_opacity(opacity0)
    opacity = opacity_function(wavelength).data
    assert_identical(2 * opacity['pressure', 0], opacity['pressure', 1])
    assert_identical(2 * opacity['cell_length', 0], opacity['cell_length', 1])
    assert_identical(2 * opacity['wavelength', 0], opacity['wavelength', 1])
    assert_identical(opacity['temperature', 0], 2 * opacity['temperature', 1])
    assert opacity.unit == ''


def test_opacity_from_cell_params_reproduces_literature_value() -> None:
    # From Lee, Wai et al. (2023). Polarisation Development at the European Spallation
    # Source. EPJ Web of Conferences. 286. 10.1051/epjconf/202328603004.
    # At T = 20 deg C, p = 1 bar, lambda = 1 Angstrom, l = 1 cm we should get 0.0733.
    pressure = sc.scalar(1.0, unit='bar')
    length = sc.scalar(0.01, unit='m')
    temperature = sc.scalar(293.15, unit='K')
    wavelength = sc.scalar(1.0, unit='angstrom')
    opacity0 = he3.he3_opacity_from_cell_params(
        pressure=pressure, length=length, temperature=temperature
    )
    opacity_function = he3.he3_opacity_function_from_cell_opacity(opacity0)
    opacity = opacity_function(wavelength).data
    assert sc.isclose(opacity, sc.scalar(0.0733, unit=''), rtol=sc.scalar(1e-3))


def test_opacity_from_cell_params_raises_with_temperature_in_degree_celsius() -> None:
    pressure = sc.scalar(1.0, unit='bar')
    length = sc.scalar(1.0, unit='m')
    temperature = sc.scalar(200.0, unit='degC')
    with pytest.raises(sc.UnitError):
        he3.he3_opacity_from_cell_params(
            pressure=pressure, length=length, temperature=temperature
        )


def test_opacity_from_beam_data() -> None:
    wavelength = sc.array(dims=['wavelength'], values=[1.0, 2.0], unit='nm')
    transmission_empty_glass = sc.scalar(0.5)
    # Pretend known opacity0 for testing
    opacity0 = sc.scalar(0.3, unit='1/nm')
    ratio = transmission_empty_glass * sc.exp(-opacity0 * wavelength)
    transmission = sc.DataArray(ratio, coords={'wavelength': wavelength})
    opacity_function = he3.he3_opacity_function_from_beam_data(
        transmission_empty_glass=transmission_empty_glass,
        transmission_fraction=transmission,
        opacity0_initial_guess=opacity0 * 1.23,  # starting guess imperfect
    )
    opacity = opacity_function(wavelength).data
    assert sc.isclose(
        opacity_function.opacity0, opacity0.to(unit=opacity_function.opacity0.unit)
    )
    assert_identical(2 * opacity['wavelength', 0], opacity['wavelength', 1])
