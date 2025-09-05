import numpy as np
import scipp as sc


def saturation_indicator(
    intensity: sc.DataArray,
    threshold: sc.Variable,
) -> sc.Variable:
    """

    Parameters
    -------------
    intensity:
        The intensity as a function of wavelength and gain.
    threshold:
        A threshold value determining the acceptable slope
        in 'wavelength' relative to the amplitude, of
        the ratio between intensities at subsequent 'gains'.

    Returns
    -------------
        The 'gain' value where the saturation occurs.
    """
    if intensity.dims != ('gain', 'wavelength'):
        raise ValueError(
            'Expected two dimensional input, with dimensions "gain" and "wavelength".'
        )

    wavelength = intensity.coords['wavelength']
    wavelength = (
        wavelength
        if len(wavelength) == intensity.shape[1]
        else sc.midpoints(wavelength)
    )
    _wavelength = wavelength.values
    _intensity = intensity.data.values

    p = [
        np.polyfit(_wavelength, _intensity[i + 1] / _intensity[i], 1)
        for i in range(_intensity.shape[0] - 1)
    ]
    slope_to_amplitude_ratio = sc.DataArray(
        sc.array(
            dims=['gain'],
            values=[abs(x[0] / x[1]) for x in p],
            unit=sc.Unit('dimensionless') / wavelength.unit,
        )
    )
    # Find index of the first entry where all subsequent slope_to_amplitude_ratios
    # are below the provided threshold.
    elbow_index = np.argmax(
        np.cumprod((slope_to_amplitude_ratio < threshold).values[::-1])[::-1]
    )
    return intensity.coords['gain']['gain', elbow_index]
