# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

from ess.polarization import he3


def test_incoming_unpolarized_reproduces_input_params_within_errors() -> None:
    time = sc.linspace('time', 0.0, 1000000.0, num=1000, unit='s')
    wavelength = sc.linspace('wavelength', 0.5, 5.0, num=40, unit='angstrom')
    C = sc.scalar(1.3)
    T1 = sc.scalar(123456.0, unit='s')
    opacity0 = sc.scalar(0.88, unit='1/angstrom')
    polarization_function = he3.He3PolarizationFunction(C=C, T1=T1)
    opacity_function = he3.He3OpacityFunction(opacity0)
    transmission_empty_glass = sc.scalar(0.9)
    opacity = opacity_function(wavelength)
    polarization = polarization_function(time)
    transmission = he3.transmission_incoming_unpolarized(
        transmission_empty_glass=transmission_empty_glass,
        opacity=opacity,
        polarization=polarization,
    )

    result = he3.get_he3_transmission_incoming_unpolarized_from_fit_to_direct_beam(
        transmission_fraction=transmission,
        opacity_function=opacity_function,
        transmission_empty_glass=transmission_empty_glass,
    )
    polarization_function = result.polarization_function
    assert isinstance(polarization_function, he3.He3PolarizationFunction)

    # No noise, very close or exact match.
    assert sc.isclose(polarization_function.C, C)
    assert sc.isclose(polarization_function.T1, T1)

    rng = np.random.default_rng(seed=1234)
    transmission_noisy = transmission.copy()
    transmission_noisy.values += rng.normal(0.0, 0.01, transmission_noisy.shape)

    result = he3.get_he3_transmission_incoming_unpolarized_from_fit_to_direct_beam(
        transmission_fraction=transmission_noisy,
        opacity_function=opacity_function,
        transmission_empty_glass=transmission_empty_glass,
    )
    polarization_function = result.polarization_function

    # With noise, within 1% of the input values.
    assert sc.isclose(polarization_function.C, C, rtol=sc.scalar(1e-2))
    assert sc.isclose(polarization_function.T1, T1, rtol=sc.scalar(1e-2))


def test_incoming_polarized_reproduces_input_params_within_errors() -> None:
    time = sc.linspace('time', 0.0, 1000000.0, num=1000, unit='s')
    wavelength = sc.linspace('wavelength', 0.5, 5.0, num=40, unit='angstrom')
    C = sc.scalar(1.3)
    T1 = sc.scalar(123456.0, unit='s')
    opacity0 = sc.scalar(0.88, unit='1/angstrom')
    polarization_function = he3.He3PolarizationFunction(C=C, T1=T1)
    opacity_function = he3.He3OpacityFunction(opacity0)
    transmission_empty_glass = sc.scalar(0.9)
    transmission_function = he3.He3TransmissionFunction(
        transmission_empty_glass=transmission_empty_glass,
        opacity_function=opacity_function,
        polarization_function=polarization_function,
    )
    # State switch at 456th time point, cut further below into 4 channels total.
    plus = transmission_function(
        time=time[:456], wavelength=wavelength, plus_minus='plus'
    )
    minus = transmission_function(
        time=time[456:], wavelength=wavelength, plus_minus='minus'
    )

    result = he3.get_he3_transmission_incoming_polarized_from_fit_to_direct_beam(
        plus=plus,
        minus=minus,
        opacity_function=opacity_function,
        transmission_empty_glass=transmission_empty_glass,
    )
    polarization_function = result.polarization_function
    assert isinstance(polarization_function, he3.He3PolarizationFunction)

    # No noise, very close or exact match.
    assert sc.isclose(polarization_function.C, C)
    assert sc.isclose(polarization_function.T1, T1)

    rng = np.random.default_rng(seed=1234)
    plus_noisy = plus.copy()
    minus_noisy = minus.copy()
    plus_noisy.values += rng.normal(0.0, 0.01, plus_noisy.shape)
    minus_noisy.values += rng.normal(0.0, 0.01, minus_noisy.shape)

    result = he3.get_he3_transmission_incoming_polarized_from_fit_to_direct_beam(
        plus=plus_noisy,
        minus=minus_noisy,
        opacity_function=opacity_function,
        transmission_empty_glass=transmission_empty_glass,
    )
    polarization_function = result.polarization_function

    # With noise, within 1% of the input values.
    assert sc.isclose(polarization_function.C, C, rtol=sc.scalar(1e-2))
    assert sc.isclose(polarization_function.T1, T1, rtol=sc.scalar(1e-2))


def test_incoming_polarized_raises_if_plus_minus_coord_is_bad() -> None:
    time = sc.linspace('time', 0.0, 1000000.0, num=1000, unit='s')
    wavelength = sc.linspace('wavelength', 0.5, 5.0, num=40, unit='angstrom')
    C = sc.scalar(1.3)
    T1 = sc.scalar(123456.0, unit='s')
    opacity0 = sc.scalar(0.88, unit='1/angstrom')
    polarization_function = he3.He3PolarizationFunction(C=C, T1=T1)
    opacity_function = he3.He3OpacityFunction(opacity0)
    transmission_empty_glass = sc.scalar(0.9)
    transmission_function = he3.He3TransmissionFunction(
        transmission_empty_glass=transmission_empty_glass,
        opacity_function=opacity_function,
        polarization_function=polarization_function,
    )
    # State switch at 456th time point, cut further below into 4 channels total.
    plus = transmission_function(
        time=time[:456], wavelength=wavelength, plus_minus='plus'
    )
    minus = transmission_function(
        time=time[456:], wavelength=wavelength, plus_minus='minus'
    )

    with pytest.raises(ValueError, match='plus-minus coordinate of plus channel'):
        he3.get_he3_transmission_incoming_polarized_from_fit_to_direct_beam(
            plus=plus.assign_coords({'plus_minus': sc.scalar(-1)}),
            minus=minus,
            opacity_function=opacity_function,
            transmission_empty_glass=transmission_empty_glass,
        )
    with pytest.raises(ValueError, match='plus-minus coordinate of minus channel'):
        he3.get_he3_transmission_incoming_polarized_from_fit_to_direct_beam(
            plus=plus,
            minus=minus.assign_coords({'plus_minus': sc.scalar(1)}),
            opacity_function=opacity_function,
            transmission_empty_glass=transmission_empty_glass,
        )
