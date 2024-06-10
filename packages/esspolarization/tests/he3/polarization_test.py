# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from ess.polarization import he3


def test_he3_polarization_reproduces_input_params_within_errors() -> None:
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

    result = he3.get_he3_transmission_from_fit_to_direct_beam(
        transmission_fraction=transmission,
        opacity_function=opacity_function,
        transmission_empty_glass=transmission_empty_glass,
        incoming_polarized=False,
    )
    polarization_function = result.polarization_function
    assert isinstance(polarization_function, he3.He3PolarizationFunction)

    # No noise, very close or exact match.
    assert sc.isclose(polarization_function.C, C)
    assert sc.isclose(polarization_function.T1, T1)

    rng = np.random.default_rng(seed=1234)
    transmission_noisy = transmission.copy()
    transmission_noisy.values += rng.normal(0.0, 0.01, transmission_noisy.shape)

    result = he3.get_he3_transmission_from_fit_to_direct_beam(
        transmission_fraction=transmission_noisy,
        opacity_function=opacity_function,
        transmission_empty_glass=transmission_empty_glass,
        incoming_polarized=False,
    )
    polarization_function = result.polarization_function

    # With noise, within 1% of the input values.
    assert sc.isclose(polarization_function.C, C, rtol=sc.scalar(1e-2))
    assert sc.isclose(polarization_function.T1, T1, rtol=sc.scalar(1e-2))
