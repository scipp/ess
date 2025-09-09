import numpy as np
import scipp as sc


def saturation_indicator(
    intensity: sc.DataArray,
    threshold: float = 0.9,
) -> tuple[sc.DataArray, sc.Variable]:
    """
    The intensity is supposed to be a signal that scales with the gain
    of the image intensifier.
    Near saturation that is not the case anymore.
    The indicator tracks the second component of the singular value decomposition
    of the intensity signal, at the gain when the amplitude of the second component
    changes quickly that indicates saturation.
    The indicator is scaled to have a maximum at the gain where saturation kicks in.

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
    """
    if intensity.dims != ('gain', 'wavelength'):
        raise ValueError(
            'Expected two dimensional input, with dimensions "gain" and "wavelength".'
        )

    gain = intensity.coords['gain']
    # The change in the amplitude of the second component indicates
    # a new component is present in the signal.
    indicator = np.linalg.svd(intensity.values, full_matrices=False)[0][:, 1]
    # The sign of indicator is arbitrary, to make sure the extremum is a maximum
    # multiply by the sign of the second derivative.
    indicator *= -np.sign(np.polyfit(gain.values, indicator, 2)[0])

    return sc.DataArray(
        sc.array(dims=['gain'], values=indicator), coords={'gain': gain}
    ), threshold * gain['gain', np.argmax(indicator)]
