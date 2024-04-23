# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import List

import numpy as np
import scipp as sc
from sciline import Pipeline

from .i_of_q import resample_direct_beam
from .types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    Denominator,
    FinalSummedQ,
    Numerator,
    ProcessedWavelengthBands,
    SampleRun,
    WavelengthBands,
    WavelengthBins,
)


def _compute_efficiency_correction(
    iofq_full: sc.DataArray,
    iofq_bands: sc.DataArray,
    wavelength_band_dim: str,
    I0: sc.Variable,
) -> sc.DataArray:
    """
    Compute the factor by which to multiply the direct beam function inside each
    wavelength band so that the $I(Q)$ curves for the full wavelength range and inside
    the bands overlap.

    Parameters
    ----------
    iofq_full:
        The $I(Q)$ for the full wavelength range.
    iofq_bands:
        The $I(Q)$ for the wavelength bands.
    wavelength_band_dim:
        The name of the wavelength band dimension.
    I0:
        The intensity of the I(Q) for the known sample at the lowest Q value.
    """
    invalid = (iofq_bands.data <= sc.scalar(0.0)) | ~sc.isfinite(iofq_bands.data)
    data = np.where(invalid.values, np.nan, (iofq_bands.data / iofq_full.data).values)
    eff = np.nanmedian(data, axis=iofq_bands.dims.index('Q'))

    scaling = sc.values(iofq_full['Q', 0].data) / I0
    # Note: do not use a `set` here because the order of dimensions is important
    dims = [dim for dim in iofq_bands.dims if dim != 'Q']
    out = sc.array(dims=dims, values=eff) * scaling
    return out.rename_dims({wavelength_band_dim: 'wavelength'})


def direct_beam(pipeline: Pipeline, I0: sc.Variable, niter: int = 5) -> List[dict]:
    """
    Compute the direct beam function.

    Procedure:

    The idea behind the direct beam iterations is to determine an efficiency of the
    detectors as a function of wavelength.
    To calculate this, it is possible to compute $I(Q)$ for the full wavelength range,
    and for individual slices (bands) of the wavelength range.
    If the direct beam function used in the $I(Q)$ computation is correct, then $I(Q)$
    curves for the full wavelength range and inside the bands should overlap.

    We require two pipelines, one for the full wavelength range and one for the bands.

    The steps are as follows:

     1. Create a flat direct beam function, as a function of wavelength, with
        wavelength bins corresponding to the wavelength bands
     2. Calculate inside each band by how much one would have to multiply the final
        $I(Q)$ so that the curve would overlap with the full range curve
     3. Multiply the direct beam values inside each wavelength band by this factor
     4. Compare the full-range $I(Q)$ to a theoretical reference and add the
        corresponding additional scaling to the direct beam function
     5. Iterate a given number of times (typically less than 10) so as to gradually
        converge on a direct beam function

    TODO: For simplicity and robustness, we currently specify the number of times to
    iterate. We could imagine in the future having a convergence criterion instead to
    determine when to stop iterating.

    Parameters
    ----------
    pipeline:
        The pipeline to compute the differential scattering cross section I(Q).
    I0:
        The intensity of the I(Q) for the known sample at the lowest Q value.
    niter:
        The number of iterations to perform.
    """

    direct_beam_function = None
    bands = pipeline.compute(ProcessedWavelengthBands)
    band_dim = (set(bands.dims) - {'wavelength'}).pop()
    full_wavelength_range = sc.concat([bands.min(), bands.max()], dim='wavelength')

    pipeline = pipeline.copy()

    wavelength_bins = pipeline.compute(WavelengthBins)
    parts = (
        FinalSummedQ[SampleRun, Numerator],
        FinalSummedQ[SampleRun, Denominator],
        FinalSummedQ[BackgroundRun, Numerator],
        FinalSummedQ[BackgroundRun, Denominator],
    )
    parts = pipeline.compute(parts)
    # Convert events to histograms to make normalization (in every iteration) cheap
    for key in [
        FinalSummedQ[SampleRun, Numerator],
        FinalSummedQ[BackgroundRun, Numerator],
    ]:
        parts[key] = parts[key].hist(wavelength=wavelength_bins)

    # For now we simply strip all uncertainties, since direct beam function is
    # computed without variances anyway.
    parts = {key: sc.values(result) for key, result in parts.items()}
    for key, part in parts.items():
        pipeline[key] = part
    sample0 = parts[FinalSummedQ[SampleRun, Denominator]]
    background0 = parts[FinalSummedQ[BackgroundRun, Denominator]]

    results = []

    for _it in range(niter):
        # The first time we compute I(Q), the direct beam function is not in the
        # parameters, nor given by any providers, so it will be considered flat.
        # TODO: Should we have a check that DirectBeam cannot be computed from the
        # pipeline?
        pipeline[WavelengthBands] = full_wavelength_range
        iofq_full = pipeline.compute(BackgroundSubtractedIofQ)
        pipeline[WavelengthBands] = bands
        iofq_bands = pipeline.compute(BackgroundSubtractedIofQ)

        if direct_beam_function is None:
            # Make a flat direct beam
            dims = [dim for dim in iofq_bands.dims if dim != 'Q']
            direct_beam_function = sc.DataArray(
                data=sc.ones(sizes={dim: iofq_bands.sizes[dim] for dim in dims}),
                coords={band_dim: sc.midpoints(bands, dim='wavelength').squeeze()},
            ).rename({band_dim: 'wavelength'})

        direct_beam_function *= _compute_efficiency_correction(
            iofq_full=iofq_full,
            iofq_bands=iofq_bands,
            wavelength_band_dim=band_dim,
            I0=I0,
        )

        # Scale denominator terms that were initially computed without direct beam
        # with the current direct beam function.
        db = resample_direct_beam(
            direct_beam=direct_beam_function,
            wavelength_bins=wavelength_bins,
        )
        db.coords['wavelength'] = sc.midpoints(
            db.coords['wavelength'], dim='wavelength'
        )
        pipeline[FinalSummedQ[SampleRun, Denominator]] = sample0 * db
        pipeline[FinalSummedQ[BackgroundRun, Denominator]] = background0 * db

        results.append(
            {
                'iofq_full': iofq_full,
                'iofq_bands': iofq_bands,
                'direct_beam': direct_beam_function,
            }
        )
    return results
