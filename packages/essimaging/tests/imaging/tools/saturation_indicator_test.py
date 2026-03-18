import numpy as np
import pytest
import scipp as sc

from ess.imaging.tools import saturation_indicator


@pytest.fixture
def intensity_as_function_of_gain_and_wavelength(seed, noise_floor):
    np.random.seed(seed)
    saturation_intensity = 1000
    wavelength = sc.linspace('wavelength', 2, 8.0, 1000, unit='angstrom')
    gains = sc.linspace('gain', 10, 200, 20)

    # Some maximum signal - could be anything
    maximum_intensity = (saturation_intensity / 10) * sc.sin(
        wavelength * sc.scalar(3.14, unit='rad/angstrom')
    ) + saturation_intensity

    def true_intensity(gain):
        return sc.scalar(noise_floor) + gain * sc.scalar(1.0, unit='1/angstrom^2') * (
            wavelength - wavelength.min()
        ) * (wavelength.max() - wavelength)

    def measured_intensity(gain):
        tI = true_intensity(gain)
        return sc.where(tI < maximum_intensity, tI, maximum_intensity) + sc.array(
            dims=tI.dims, values=noise_floor * np.random.randn(*tI.shape)
        )

    return sc.DataArray(
        sc.concat([measured_intensity(gain) for gain in gains], 'gain'),
        coords={'gain': gains, 'wavelength': wavelength},
    )


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('noise_floor', [0, 1, 10, 100])
def test_saturation_indicator(intensity_as_function_of_gain_and_wavelength):
    intensity = intensity_as_function_of_gain_and_wavelength
    indicator, gain = saturation_indicator(intensity, threshold=0.9)
    assert gain < 100
    assert gain > 80
    assert 'gain' in indicator.coords
