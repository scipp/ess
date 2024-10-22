# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import warnings

import scipp as sc
from scipy.optimize import OptimizeWarning

from .types import (
    FootprintCorrectedData,
    IdealReferenceIntensity,
    NormalizationFactor,
    QBins,
    QResolution,
    ReflectivityData,
    ReflectivityOverQ,
    SampleRun,
    WavelengthBins,
)


def normalization_factor(
    da: FootprintCorrectedData[SampleRun],
    corr: IdealReferenceIntensity,
    wbins: WavelengthBins,
) -> NormalizationFactor:
    """The correction matrix gives us the expected intensity at each
    (z_index, wavelength) bin assuming the reflectivity is one.
    To normalize the sample measurement we need to integrate the total
    expected intensity in every Q-bin.
    Note that Q refers to the 'sample-Q', different from the 'reference-Q'.

    The 'sample-Q' is computed taking the mean of the sample measurement Q
    value in every (z_index, wavelength) bin.
    One complication however is that some bins have zero intensity from the
    sample measurement, so we are unable to assign a 'sample-Q' value to those bins.
    Therefore we estimate the intensity in the missing bins by fitting the
    'sample-q' as a function of z_index and wavelength.

    Steps:
        Approximate 'sample-q' in every (z_index, wavelength) bin
        Fit 'sample-q'.
        Compute 'sample-q' in all bins using the fit.
        Return the reference intensity with the 'sample-q' as a coordinate.

    """
    sample_q = (
        da.bin(wavelength=wbins, dim=set(da.dims) - set(da.coords["z_index"].dims))
        .bins.coords["Q"]
        .bins.mean()
    )

    wm = sc.midpoints(corr.coords["wavelength"])

    def q_of_z_wavelength(wavelength, a, b):
        return a + b / wavelength

    with warnings.catch_warnings():
        # `curve_fit` raises a warning if it fails to estimate variances.
        # We don't care here because we only need the parameter values and anyway
        # assume that the fit worked.
        # The warning can be caused by there being too few points to estimate
        # uncertainties because of masks.
        warnings.filterwarnings(
            "ignore",
            message="Covariance of the parameters could not be estimated",
            category=OptimizeWarning,
        )
        p, _ = sc.curve_fit(
            ["wavelength"],
            q_of_z_wavelength,
            sc.DataArray(
                data=sample_q,
                coords={"wavelength": wm},
                masks={
                    **corr.masks,
                    "_sample_q_isnan": sc.isnan(sample_q),
                },
            ),
            p0={"a": sc.scalar(-1e-3, unit="1/angstrom")},
        )
    return sc.DataArray(
        data=corr.data,
        coords={
            "Q": q_of_z_wavelength(
                wm,
                sc.values(p["a"]),
                sc.values(p["b"]),
            ).data,
        },
        masks=corr.masks,
    )


def reflectivity_over_q(
    da: FootprintCorrectedData[SampleRun],
    n: NormalizationFactor,
    qbins: QBins,
    qres: QResolution,
) -> ReflectivityOverQ:
    """
    Normalize the sample measurement by the (ideally calibrated) supermirror.

    Parameters
    ----------
    sample:
        Sample measurement with coord 'Q'
    supermirror:
        Supermirror measurement with coord of 'Q' representing the sample 'Q'

    Returns
    -------
    :
        Reflectivity as a function of Q
    """
    reflectivity = da.bin(Q=qbins, dim=da.dims) / sc.values(n.hist(Q=qbins, dim=n.dims))
    reflectivity.coords['Q_resolution'] = qres.data
    for coord, value in da.coords.items():
        if (
            not isinstance(value, sc.Variable)
            or len(set(value.dims) - set(reflectivity.dims)) == 0
        ):
            reflectivity.coords[coord] = value
    return ReflectivityOverQ(reflectivity)


def reflectivity_per_event(
    da: FootprintCorrectedData[SampleRun],
    n: IdealReferenceIntensity,
    wbins: WavelengthBins,
) -> ReflectivityData:
    """
    Weight the sample measurement by the (ideally calibrated) supermirror.

    Returns:
        reflectivity "per event"
    """
    reflectivity = da.bin(wavelength=wbins, dim=set(da.dims) - set(n.dims)) / sc.values(
        n
    )
    for coord, value in da.coords.items():
        if (
            not isinstance(value, sc.Variable)
            or len(set(value.dims) - set(reflectivity.dims)) == 0
        ):
            reflectivity.coords[coord] = value
    return ReflectivityData(reflectivity)


providers = (reflectivity_over_q, normalization_factor, reflectivity_per_event)
