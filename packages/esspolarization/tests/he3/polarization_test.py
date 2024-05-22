# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

from ess.polarization import he3


@pytest.mark.parametrize(
    'split_ratio',
    [
        sc.scalar(0.5),
        sc.linspace('time', 1.2, 1.4, num=1000),
        sc.linspace('wavelength', 0.5, 1.5, num=40),
    ],
)
def test_he3_polarization_reproduces_input_params_within_errors(
    split_ratio: sc.Variable,
) -> None:
    time = sc.linspace('time', 0.0, 10000.0, num=1000, unit='s')
    wavelength = sc.linspace('wavelength', 0.5, 5.0, num=40, unit='angstrom')
    C = sc.scalar(1.3)
    T1 = sc.scalar(1234.0, unit='s')
    opacity0 = sc.scalar(0.6, unit='1/angstrom')
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
    direct_beam_polarized = sc.DataArray(
        transmission, coords={'time': time, 'wavelength': wavelength}
    )

    result = he3.get_he3_transmission_from_fit_to_direct_beam(
        direct_beam_no_cell=sc.ones_like(direct_beam_polarized) * split_ratio,
        direct_beam_polarized=direct_beam_polarized * split_ratio,
        opacity_function=opacity_function,
        transmission_empty_glass=transmission_empty_glass,
    )
    polarization_function = result.polarization_function
    assert isinstance(polarization_function, he3.He3PolarizationFunction)

    # No noise, very close or exact match.
    assert sc.isclose(polarization_function.C, C)
    assert sc.isclose(polarization_function.T1, T1)

    rng = np.random.default_rng(seed=1234)
    direct_beam_polarized_noisy = direct_beam_polarized.copy()
    direct_beam_polarized_noisy.values += rng.normal(
        0.0, 0.01, direct_beam_polarized_noisy.shape
    )

    result = he3.get_he3_transmission_from_fit_to_direct_beam(
        direct_beam_no_cell=sc.ones_like(direct_beam_polarized_noisy) * split_ratio,
        direct_beam_polarized=direct_beam_polarized * split_ratio,
        opacity_function=opacity_function,
        transmission_empty_glass=transmission_empty_glass,
    )
    polarization_function = result.polarization_function

    # With noise, within 1% of the input values.
    assert sc.isclose(polarization_function.C, C, rtol=sc.scalar(1e-2))
    assert sc.isclose(polarization_function.T1, T1, rtol=sc.scalar(1e-2))
