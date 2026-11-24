from itertools import zip_longest
from math import isqrt

import numpy as np
import scipp as sc
from numpy.typing import NDArray


def _all_divisors(n: int) -> list[int]:
    return sorted(
        {d for i in range(1, isqrt(n) + 1) if n % i == 0 for d in (i, n // i)}
    )


def maximum_resolution_achievable(
    events: sc.DataArray,
    image_dims: tuple[str, str],
) -> sc.DataArray:
    """Coarsen an image until every unmasked bin contains at least one event.

    Parameters
    ----------
    events:
        DataArray binned in at least the ``image_dims`` dimensions and optionally more.
        No spatial coordinates are required. Other dimensions, such as wavelength
        or time-of-flight, retain their original binning. Event weights do not
        affect the choice of resolution.
    image_dims:
        Two distinct, non-empty dimensions defining the image grid.

    Returns
    -------
    :
        Events concatenated within rectangular blocks of neighboring pixels.
        Masked pixels are excluded. Coordinates depending on the image dimensions
        are dropped by the concatenation. The input is not modified.

    Raises
    ------
    ValueError
        If the input is not binned, the image dimensions are invalid, or even
        combining all image pixels leaves an empty unmasked bin.

    Notes
    -----
    Block sizes follow the ascending divisors of each image dimension, paired
    in order. Once one dimension is fully combined, only the other is coarsened.
    The first valid grid in this sequence is returned, not necessarily the finest
    among all possible rectangular grids. Powers of two give successive factors
    of two; other sizes can give larger jumps.

    The search folds and sums bin counts. Event contents are concatenated only
    once, after choosing the resolution, without re-binning individual events.
    """
    if events.bins is None:
        raise ValueError('Input data must be binned.')

    if len(image_dims) != 2 or len(set(image_dims)) != 2:
        raise ValueError('image_dims must contain two distinct dimensions.')
    if any(dim not in events.sizes or events.sizes[dim] == 0 for dim in image_dims):
        raise ValueError('Image dimensions must be present and non-empty.')
    if events.size == 0:
        raise ValueError('Input data must contain at least one bin.')

    x, y = image_dims

    xdivisors = _all_divisors(events.sizes[x])
    ydivisors = _all_divisors(events.sizes[y])
    # Avoid collisions with both outer dimensions and the event dimension.
    used_dims = set(events.dims) | {events.bins.constituents['dim']}
    x_aux, y_aux = '_x_aux', '_y_aux'
    while x_aux in used_dims:
        x_aux += '_'
    while y_aux in used_dims:
        y_aux += '_'
    counts = events.bins.size()

    for i, j in zip_longest(xdivisors, ydivisors):
        i = xdivisors[-1] if i is None else i
        j = ydivisors[-1] if j is None else j
        x_sizes, y_sizes = {x: -1, x_aux: i}, {y: -1, y_aux: j}
        # Keep masks on the counts so sum excludes the same pixels as concat.
        coarse_counts = (
            counts.fold(x, sizes=x_sizes).fold(y, sizes=y_sizes).sum([x_aux, y_aux])
        )
        if coarse_counts.min().value > 0:
            return (
                events.fold(x, sizes=x_sizes)
                .fold(y, sizes=y_sizes)
                .bins.concat([x_aux, y_aux])
            )

    raise ValueError(
        'Even at the coarsest pixel binning there are some pixels that have no events.'
        ' The binning in the remaining dimensions may be too fine.'
    )


def _radial_profile(data: NDArray) -> tuple[NDArray, NDArray]:
    """Return frequencies and radial means of an fftshifted spectrum."""
    fy = np.fft.fftshift(np.fft.fftfreq(data.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(data.shape[1]))
    radius = np.hypot(fy[:, None], fx[None, :])
    # Use the finest FFT spacing, so the zero-frequency bin contains only
    # zero-frequency component.
    # The tolerance keeps frequencies on a bin boundary in that bin despite rounding.
    bins = np.floor(radius * max(data.shape) + 1e-12).astype(np.intp)
    count = np.bincount(bins.ravel())
    populated = count > 0
    frequency = (
        np.bincount(bins.ravel(), weights=radius.ravel())[populated] / count[populated]
    )
    profile = (
        np.bincount(bins.ravel(), weights=data.ravel())[populated] / count[populated]
    )
    return frequency, profile


def normalized_radial_spectrum(image: sc.DataArray) -> sc.DataArray:
    """Return the normalized, radially averaged Fourier-amplitude spectrum.

    Parameters
    ----------
    image:
        A two-dimensional image sampled on equally spaced square pixels. Pixel
        coordinates are not used; the frequency is expressed per image pixel.

    Returns
    -------
    :
        Fourier-amplitude spectrum with a dimensionless ``frequency`` coordinate
        in cycles (line pairs) per pixel. Each coordinate is the mean frequency
        magnitude in its radial bin. Only bins up to 0.5 cycles per pixel are
        included, inside the circle bounded by the axial Nyquist frequencies.
        The zero-frequency amplitude is normalized to one.

    Notes
    -----
    This is an amplitude spectrum, not a power spectrum. It can be used to inspect
    the measured and reference spectra underlying :func:`modulation_transfer_function`.
    """
    values = image.values / image.values.sum()
    amplitude = np.abs(np.fft.fftshift(np.fft.fft2(values)))
    frequency, profile = _radial_profile(amplitude)
    keep = frequency <= 0.5
    return sc.DataArray(
        sc.array(dims=['frequency'], values=profile[keep]),
        coords={'frequency': sc.array(dims=['frequency'], values=frequency[keep])},
    )


def modulation_transfer_function(
    measured_image: sc.DataArray,
    open_beam_image: sc.DataArray,
    target: sc.DataArray,
) -> sc.DataArray:
    '''
    Computes the modulation transfer function (MTF) of
    the camera given a measured image and the
    ideal image that would have been captured if
    the instrument had infinite resolution.

    Parameters
    ------------
    measured_image:
        The image of the sample captured by the camera.
    open_beam_image:
        The image without the sample captured by the camera.
    target:
        A perfect image of the sample
        on the same grid as `measured_image`.

    Returns
    ------------
    :
        The modulation transfer function as a function
        of "frequency" representing "line pairs" per pixel.

    Notes
    -----------

    Computing modulation transfer function (MTF)
    ============================================

    The definition of the MTF is

    .. math::

        \\mathrm{MTF}(f) = |\\mathcal{F}(P)|

    where :math:`\\mathcal{F}(P)` is the Fourier transform of the point spread function :math:`P`.

    The Fourier transform of the point spread function is really a function of two variables, but it is assumed that the MTF does not vary depending on the direction of change, so here it's denoted as a function of the frequency independent of direction:

    .. math::

        \\mathrm{MTF}(\\|(f_x, f_y)\\|) = |\\mathcal{F}(P)|(f_x, f_y)

    Model for images in detector
    ----------------------------

    The intensity distribution in the detector (the "image") :math:`I` is modeled as

    .. math::

        I = I_0 S \\star P

    where :math:`I_0` is the intensity distribution at the sample, :math:`S` is the transmission function of the sample, and :math:`P` is the point-spread function.

    For the open beam we don't have any sample and the intensity distribution in the detector is modeled as

    .. math::

        I_{ob} = I_0 \\star P

    Approximation
    -------------

    Assuming :math:`I_0` is more or less uniform, and :math:`P` is relatively localized, we can approximate

    .. math::

        I_0 \\star P \\approx I_0

    Making this assumption we can substitute :math:`I_0` for :math:`I_{ob}` in the model for the image:

    .. math::

        I = I_{ob} S \\star P

    Applying the Fourier transform on both sides we have

    .. math::

        \\mathcal{F}(I) = \\mathcal{F}(I_{ob} S)\\, \\mathcal{F}(P)

    which implies

    .. math::

        |\\mathcal{F}(P)| = \\left| \\frac{\\mathcal{F}(I)}{\\mathcal{F}(I_{ob} S)} \\right|

    and therefore

    .. math::

        \\mathrm{MTF}(\\|(f_x, f_y)\\|) =
        \\frac{|\\mathcal{F}(I)|(f_x, f_y)}{|\\mathcal{F}(I_{ob} S)|(f_x, f_y)}

    Finally, integrating over constant frequency magnitude:

    .. math::

        \\mathrm{MTF}(f) =
        \\frac{\\int_{\\|(f_x, f_y)\\| = f} |\\mathcal{F}(I)|(f_x, f_y)\\, df_x\\, df_y}
             {\\int_{\\|(f_x, f_y)\\| = f} |\\mathcal{F}(I_{ob} S)|(f_x, f_y)\\, df_x\\, df_y}

    Conclusion
    ----------

    The modulation transfer function at frequency :math:`f` can be estimated as the ratio of the Fourier transform of the image (integrated over constant frequency magnitude) to the Fourier transform of the open beam image multiplied by the sample mask (also integrated over constant frequency magnitude).
    '''  # noqa: E501
    reference = (open_beam_image * target).to(unit=measured_image.unit)
    return normalized_radial_spectrum(measured_image) / normalized_radial_spectrum(
        reference
    )


def estimate_cut_off_frequency(mtf: sc.DataArray) -> sc.Variable:
    '''Estimates the cut off frequency of
    the modulation transfer function (mtf).

    Parameters
    -------------
    mtf:
        A (potentially noisy) modulation transfer function curve
        having a coordinate named "frequency".

    Returns
    -------------
    :
        An estimate of the frequency where the modulation
        transfer function goes to zero, the "cut off frequency".
    '''
    _freq = np.concat([[0.0], mtf.coords['frequency'].values])
    _mtf = np.concat([[1.0], mtf.values])
    # The line should go through (0, 1), so give it a big weight.
    # 10 x total_weight was determined good enough by trial and error.
    w = np.concat([[10 * len(mtf)], np.ones(len(mtf))])
    m = np.ones(len(_freq), dtype='bool')
    fc = np.nan
    maxiters = 100
    for _ in range(maxiters):
        p = np.polyfit(_freq[m], _mtf[m], 1, w=w[m])
        # 1e-4 is used as a threshold because the method is not
        # accurate to less than 1e-4 anyway so we can just as well stop there.
        if abs(-p[1] / p[0] - fc) < 1e-4:
            break
        fc = -p[1] / p[0]
        m = np.polyval(p, _freq) >= 0
    # Correction factor 9/8 is the ratio between where a linear approximation
    # of the MTF of a circular apparture crosses 0 and where the actual cutoff frequency
    # of the same circular apparture is.
    # For reference:
    # import sympy as sp
    # x, f, a = sp.symbols('x, f, a', positive=True)
    # sp.solve(sp.integrate(sp.diff((1 - a * x - 2 / sp.pi * (sp.acos(x/f) - x/f * sp.sqrt(1 - x**2/f**2)))**2, a), (x, 0, f)), f)  # noqa: E501
    return 9 / 8 * sc.scalar(-p[1] / p[0], unit=mtf.coords['frequency'].unit)


def mtf_less_than(mtf: sc.DataArray, limit: sc.Variable) -> sc.Variable:
    '''Computes the frequency where the
    modulation transfer function goes below ``limit``.

    Parameters
    --------------
    mtf:
        A (potentially noisy) modulation transfer function curve
        having a coordinate named "frequency".

    limit:
        The modulation transfer function value at the returned frequency.

    Returns
    -----------
    :
        The frequency where the modulation transfer function goes below "limit".
    '''
    return mtf.coords['frequency'][mtf.data <= limit].min()
