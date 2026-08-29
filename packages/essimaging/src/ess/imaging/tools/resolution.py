from itertools import zip_longest

import numpy as np
import scipp as sc
from numpy.typing import NDArray


def _all_divisors(n):
    for i in range(1, n + 1):
        if n % i == 0:
            yield i


def maximum_resolution_achievable(
    events: sc.DataArray,
    image_dims: tuple[str, str],
):
    """
    Estimates the maximum resolution achievable
    given a desired binning.
    The maximum achievable resolution is defined
    as the resolution in ``xy`` such that
    there is at least one event in every ``xy...`` pixel,
    where ``...`` represents the rest of the binning dimenensions
    of the event data, typically wavelength or time-of-flight.

    Parameters
    -------------
    events:
        DataArray binned in at least the ``image_dims`` dimensions and optionally more.
        The method works best when the number of bins in each ``image_dims`` dimension
        is a power of ``2``.

    Returns
    -------------
        The event data array folded to the finest resolution where there is at least
        one event in every pixel.
    """
    if events.bins is None:
        raise ValueError(
            'Input data must be binned.'
            ' For best result, number of bins in each image dimension should'
            ' be a power of 2.'
        )

    x, y = image_dims

    xdivisors = list(_all_divisors(events.sizes[x]))
    ydivisors = list(_all_divisors(events.sizes[y]))

    for i, j in zip_longest(xdivisors, ydivisors):
        i = xdivisors[-1] if i is None else i
        j = ydivisors[-1] if j is None else j
        tmp_events = (
            events.fold(x, sizes={x: -1, '_x_aux': i})
            .fold(y, sizes={y: -1, '_y_aux': j})
            .bins.concat(['_x_aux', '_y_aux'])
        )
        min_counts_per_pixel = tmp_events.bins.size().min()
        if min_counts_per_pixel.value > 0:
            return tmp_events

    raise ValueError(
        'Even at the coarsest pixel binning there are some pixels that have no events.'
        ' Probably because the wavelength/time-of-arrival binning is too fine,'
        ' or because number of bins in each image dimension is not a power of 2.'
    )


def _radial_profile(data: NDArray) -> NDArray:
    '''Integrate ellipses around center of image.'''
    y, x = np.indices(data.shape)
    cy, cx = np.array(data.shape) / 2.0
    r = np.hypot((cx * cy) ** 0.5 * (x - cx) / cx, (cx * cy) ** 0.5 * (y - cy) / cy)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    return tbin / (nr + 1e-15)


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
    _measured = measured_image.values
    # Can't do inplace because dtype of sum might be different from dtype of input
    _measured = _measured / _measured.sum()
    _reference = (open_beam_image * target).to(unit=measured_image.unit).values
    _reference = _reference / _reference.sum()
    f_ideal = np.abs(np.fft.fftshift(np.fft.fft2(_reference)))
    f_measured = np.abs(np.fft.fftshift(np.fft.fft2(_measured)))
    _mtf = _radial_profile(f_measured) / _radial_profile(f_ideal)
    return sc.DataArray(
        sc.array(dims=['frequency'], values=_mtf),
        # Unit of frequency is line_pairs / pixel but since both of those are
        # a kind of counts I think in our unit system that is best
        # represented as 'dimensionless'.
        # The largest frequency magnitude in 2d fft is sqrt(1/2).
        coords={'frequency': sc.linspace('frequency', 0, (1 / 2) ** 0.5, len(_mtf))},
        # We're only interested in frequencies below 0.5 oscillations per pixel
        # because those above are unphysical.
    )['frequency', : sc.scalar(0.5)]


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
