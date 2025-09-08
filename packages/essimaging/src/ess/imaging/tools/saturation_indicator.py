import numpy as np
import scipp as sc


def saturation_indicator(
    intensity: sc.DataArray,
    threshold: float = 0.9,
) -> tuple[sc.DataArray, sc.Variable]:
    """

    Parameters
    -------------
    intensity:
        The intensity as a function of gain and wavelength.
    threshold:
        Safety factor to avoid saturation region.
        Must be between 0 and 1.
        Corresponds to the acceptable reduction in intensity
        to create margin to the saturation region.

    Returns
    -------------
        The saturation indicator value as a function of 'gain',
        and the maximum gain value acceptable according to the
        provided threshold.
        Note that the maximum of the saturation indicator occurs
        at the gain where saturation kicks in.
    """
    if intensity.dims != ('gain', 'wavelength'):
        raise ValueError(
            'Expected two dimensional input, with dimensions "gain" and "wavelength".'
        )

    def indicator(i):
        # Assuming all intensity curves
        # in the non-saturated region are
        # proportional means the ratio between the largest
        # and second largest singular value is large.
        s = np.linalg.svd(i)[1]
        return s[0] / s[1]

    ind = np.array([indicator(intensity.values[:i]) for i in range(2, len(intensity))])
    max_gain_index = np.argmax(ind > threshold * ind.max())
    gain = intensity.coords['gain']['gain', 1:]
    return sc.DataArray(
        sc.array(dims=['gain'], values=ind), coords={'gain': gain}
    ), gain['gain', max_gain_index]
